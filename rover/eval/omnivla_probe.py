#!/usr/bin/env python3
"""Offline probe: OmniVLA-edge on recorded rover episodes.

Reference check before adding a goal-pose channel to SmolVLA. Runs the released
OmniVLA-edge checkpoint (ViNT-based, ~108M, MIT) on frames this project already
recorded, under three goal modalities:

    pose      (id 4)  goal (x, y) from episode.json, in the frame's body frame
    language  (id 7)  the episode instruction, no goal pose
    both      (id 8)  goal pose + instruction

The pose run is the validity check on this shim. OmniVLA-edge trained on fisheye
and 360 imagery while our camera is a 100 deg pinhole, so if pose-conditioned
driving fails there is a domain or convention error and the language number
carries no information.

Conventions read off OmniVLA/inference/run_omnivla_edge.py:

    goal_pose = [front/S, left/S, cos(dyaw), sin(dyaw)],  S = 0.1 m
    actions   = 8 x (front, left, cos, sin); actions[:, :2] * S -> metres
    images    = 6-frame context (5 history + current) at 96x96, plus a
                224x224 CLIP view of the current frame

Body frame is ROS REP-103 (x front, y left) on both sides -- the same convention
gt_pose and datagen/relabel.py already use -- so no axis remap is needed.

Usage:
    python rover/eval/omnivla_probe.py --episode rover/data/raw/open_ground_seed01000
"""

import argparse
import bisect
import json
import math
import os
import sys

import numpy as np
import torch
from PIL import Image

S = 0.1          # metric_waypoint_spacing
CONTEXT = 5      # 5 history frames + current = 6
IMGSIZE = (96, 96)
IMGSIZE_CLIP = (224, 224)

MODALITY = {'pose': 4, 'language': 7, 'both': 8}

MODEL_PARAMS = {
    'model_type': 'omnivla-edge',
    'len_traj_pred': 8,
    'learn_angle': True,
    'context_size': CONTEXT,
    'obs_encoder': 'efficientnet-b0',
    'encoding_size': 256,
    'obs_encoding_size': 1024,
    'goal_encoding_size': 1024,
    'late_fusion': False,
    'mha_num_attention_heads': 4,
    'mha_num_attention_layers': 4,
    'mha_ff_dim_factor': 4,
    'clip_type': 'ViT-B/32',
}


def read_jsonl(path):
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


class PoseTrack:
    """Same interpolation as datagen/relabel.py, kept standalone so the probe
    does not need the datagen package on sys.path."""

    def __init__(self, rows):
        self.t = [r['t'] for r in rows]
        self.rows = rows

    def at(self, t):
        i = bisect.bisect_left(self.t, t)
        if i <= 0:
            r = self.rows[0]
            return r['x'], r['y'], r['yaw']
        if i >= len(self.t):
            r = self.rows[-1]
            return r['x'], r['y'], r['yaw']
        r0, r1 = self.rows[i - 1], self.rows[i]
        a = (t - r0['t']) / (r1['t'] - r0['t']) if r1['t'] > r0['t'] else 0.0
        dyaw = r1['yaw'] - r0['yaw']
        while dyaw > math.pi:
            dyaw -= 2 * math.pi
        while dyaw < -math.pi:
            dyaw += 2 * math.pi
        return (r0['x'] + a * (r1['x'] - r0['x']),
                r0['y'] + a * (r1['y'] - r0['y']),
                r0['yaw'] + a * dyaw)


def to_body(px, py, pose):
    """World point -> (front, left) in the body frame at pose. Mirrors
    relabel.waypoint_chunk's transform."""
    x0, y0, yaw0 = pose
    dx, dy = px - x0, py - y0
    c, s = math.cos(-yaw0), math.sin(-yaw0)
    return c * dx - s * dy, s * dx + c * dy


def build_batch(frames_dir, idx, frame_rows, goal_front, goal_left,
                instruction, device, text_encoder, mask96, mask224,
                transform_mask, transform_map):
    """Assemble the 7 tensors OmniVLA-edge's forward() expects."""
    # 6-frame context: idx-5 .. idx, clamped at the episode start.
    ctx_ids = [max(0, idx - (CONTEXT - i)) for i in range(CONTEXT)] + [idx]
    ctx = []
    for j in ctx_ids:
        p = os.path.join(frames_dir, f'{frame_rows[j]["i"]:06d}.jpg')
        ctx.append(Image.open(p).convert('RGB').resize(IMGSIZE))
    cur_path = os.path.join(frames_dir, f'{frame_rows[idx]["i"]:06d}.jpg')
    cur_224 = Image.open(cur_path).convert('RGB').resize(IMGSIZE_CLIP)

    obs = transform_mask(ctx, mask96).to(device)
    obs_cur = torch.split(obs, 3, dim=1)[-1]
    cur_large = transform_mask(cur_224, mask224).to(device)

    # Satellite modality is unused; the released code feeds black images.
    black = Image.new('RGB', (352, 352), color=(0, 0, 0))
    map_images = torch.cat(
        (transform_map(black).to(device), transform_map(black).to(device), obs_cur), axis=1)

    dyaw = math.atan2(goal_left, goal_front)   # face the goal
    goal_pose = torch.tensor(
        [[goal_front / S, goal_left / S, math.cos(dyaw), math.sin(dyaw)]],
        dtype=torch.float32, device=device)

    # Goal image is unused in pose/language modalities but forward() needs the
    # tensor (and derives the batch size from it).
    goal_image = transform_mask(
        Image.new('RGB', IMGSIZE, color=(0, 0, 0)), mask96).to(device)

    import clip
    with torch.no_grad():
        feat_text = text_encoder.encode_text(clip.tokenize(instruction, truncate=True).to(device))

    return {
        'obs_images': obs,
        'goal_pose_torch': goal_pose,
        'map_images': map_images,
        'goal_image': goal_image,
        'cur_large_img': cur_large,
        'feat_text': feat_text.float(),
    }


def bearing_deg(front, left):
    return math.degrees(math.atan2(left, front))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--episode', required=True, help='rover/data/raw/<scene>_seed<N>')
    ap.add_argument('--frame', type=int, default=0, help='frame index within the episode')
    ap.add_argument('--omnivla-root', default='/ref/OmniVLA')
    ap.add_argument('--ckpt', default=None, help='defaults to <root>/omnivla-edge/omnivla-edge.pth')
    ap.add_argument('--modality', default='all', choices=['pose', 'language', 'both', 'all'])
    ap.add_argument('--instruction', default=None, help='override the episode instruction')
    args = ap.parse_args()

    sys.path.insert(0, os.path.join(args.omnivla_root, 'inference'))
    from utils_policy import load_model, transform_images_PIL_mask, transform_images_map

    ep = args.episode
    cfg = json.load(open(f'{ep}/episode.json'))['config']
    props = cfg['props']
    goal = props[cfg['goal_index']]
    instruction = args.instruction or cfg['instruction']

    frame_rows = read_jsonl(f'{ep}/frames.jsonl')
    track = PoseTrack(read_jsonl(f'{ep}/gt_pose.jsonl'))
    idx = args.frame
    pose = track.at(frame_rows[idx]['t'])

    # Ground truth geometry at this frame.
    gf, gl = to_body(goal['x'], goal['y'], pose)
    print(f'episode      {os.path.basename(ep)}  frame {idx} (t={frame_rows[idx]["t"]:.2f})')
    print(f'instruction  "{instruction}"')
    print(f'pose         x={pose[0]:.2f} y={pose[1]:.2f} yaw={math.degrees(pose[2]):.1f} deg')
    print(f'\nprops in body frame (front, left) -- goal marked:')
    prop_b = []
    for i, p in enumerate(props):
        f, l = to_body(p['x'], p['y'], pose)
        prop_b.append((f, l, p))
        mark = '  <-- GOAL' if i == cfg['goal_index'] else ''
        print(f'  {i} {p["color"]:6s} {p["shape"]:6s} front={f:6.2f} left={l:6.2f} '
              f'range={math.hypot(f, l):5.2f} bearing={bearing_deg(f, l):+7.1f}{mark}')

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    ckpt = args.ckpt or os.path.join(args.omnivla_root, 'omnivla-edge', 'omnivla-edge.pth')
    model, text_encoder, _ = load_model(ckpt, MODEL_PARAMS, device)
    model = model.to(device).eval()
    text_encoder = text_encoder.to(device).eval()

    mask96 = np.ones((96, 96, 3), dtype=np.float32)
    mask224 = np.ones((224, 224, 3), dtype=np.float32)

    batch = build_batch(f'{ep}/frames', idx, frame_rows, gf, gl, instruction, device,
                        text_encoder, mask96, mask224,
                        transform_images_PIL_mask, transform_images_map)

    modes = ['pose', 'language', 'both'] if args.modality == 'all' else [args.modality]
    print(f'\n{"modality":10s} {"endpoint (m)":>20s} {"bearing":>9s}   heads toward')
    print('-' * 72)
    for m in modes:
        mid = torch.tensor([MODALITY[m]], device=device)
        with torch.no_grad():
            actions, _, _ = model(batch['obs_images'], batch['goal_pose_torch'],
                                  batch['map_images'], batch['goal_image'],
                                  mid, batch['feat_text'], batch['cur_large_img'])
        wp = actions[0].float().cpu().numpy()
        end_f, end_l = wp[-1, 0] * S, wp[-1, 1] * S
        b = bearing_deg(end_f, end_l)
        # Which prop's bearing does the predicted heading match best?
        best = min(prop_b, key=lambda pb: abs(bearing_deg(pb[0], pb[1]) - b))
        tag = f'{best[2]["color"]} {best[2]["shape"]}'
        hit = 'GOAL' if best[2] is goal else 'not goal'
        print(f'{m:10s} {end_f:8.2f}, {end_l:8.2f} {b:+8.1f}deg   {tag:14s} ({hit})')


if __name__ == '__main__':
    main()
