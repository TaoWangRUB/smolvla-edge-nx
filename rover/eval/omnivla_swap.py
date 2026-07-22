#!/usr/bin/env python3
"""Swap test for OmniVLA-edge on recorded rover frames.

The colour-grounding measurement this project has never been able to make
cleanly. For each scene we pick a same-shape/different-colour pair that is
(a) both in front of the rover, (b) on opposite sides, and (c) at comparable
range -- so neither proximity nor saliency can substitute for reading the
colour word. Then we run the identical frame twice:

    run A   instruction names the goal
    run B   instruction names the hard negative

and ask whether the predicted trajectory follows the commanded object both
times. A policy that steers to a fixed salient object scores 0.

Reported per scene:
    bearingA/B   predicted heading under each instruction (deg, + = left)
    dBrg         |bearingA - bearingB|, directly comparable to the SmolVLA
                 probe numbers in eval_results/grounding_diagnosis.md
                 (stage1_v3 15.3 deg, stage1d_deeplm 22.3 deg)
    pass         both runs on the commanded object's side

The pose-conditioned run is the validity check: it should score ~1.0. If it
does not, the shim or the sim/real domain gap is broken and the language
numbers carry no information.

Usage:
    python rover/eval/omnivla_swap.py --top 12
"""

import argparse
import glob
import json
import math
import os
import sys

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from omnivla_probe import (MODALITY, MODEL_PARAMS, S, PoseTrack, bearing_deg,
                           build_batch, read_jsonl, to_body)


def find_pairs(raw_root, min_sep=45.0, max_range_ratio=1.6):
    """Scenes with a same-shape/different-colour pair on opposite sides at
    comparable range. Returns the most discriminative first."""
    out = []
    for ep in sorted(glob.glob(os.path.join(raw_root, '*'))):
        try:
            cfg = json.load(open(f'{ep}/episode.json'))['config']
            p0 = json.loads(open(f'{ep}/gt_pose.jsonl').readline())
        except Exception:
            continue
        props, gi = cfg['props'], cfg['goal_index']
        g = props[gi]
        gf, gl = to_body(g['x'], g['y'], (p0['x'], p0['y'], p0['yaw']))
        gb, gr = bearing_deg(gf, gl), math.hypot(gf, gl)
        for j, p in enumerate(props):
            if j == gi or p['shape'] != g['shape'] or p['color'] == g['color']:
                continue
            nf, nl = to_body(p['x'], p['y'], (p0['x'], p0['y'], p0['yaw']))
            nb, nr = bearing_deg(nf, nl), math.hypot(nf, nl)
            if gf < 0.5 or nf < 0.5:
                continue
            if gb * nb >= 0:                       # must be on opposite sides
                continue
            if abs(gb - nb) < min_sep:
                continue
            if max(gr, nr) / min(gr, nr) > max_range_ratio:
                continue
            out.append({'ep': ep, 'goal': g, 'neg': p, 'gb': gb, 'nb': nb,
                        'gr': gr, 'nr': nr, 'sep': abs(gb - nb)})
    out.sort(key=lambda r: -r['sep'])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--raw-root', default='rover/data/raw')
    ap.add_argument('--omnivla-root', default='/ref/OmniVLA')
    ap.add_argument('--ckpt', default=None)
    ap.add_argument('--top', type=int, default=12)
    ap.add_argument('--modality', default='language', choices=['language', 'pose', 'both'])
    args = ap.parse_args()

    sys.path.insert(0, os.path.join(args.omnivla_root, 'inference'))
    from utils_policy import load_model, transform_images_PIL_mask, transform_images_map

    pairs = find_pairs(args.raw_root)[:args.top]
    if not pairs:
        print('no discriminative pairs found')
        return
    print(f'{len(pairs)} scenes; modality={args.modality}\n')

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    ckpt = args.ckpt or os.path.join(args.omnivla_root, 'omnivla-edge', 'omnivla-edge.pth')
    model, text_encoder, _ = load_model(ckpt, MODEL_PARAMS, device)
    model, text_encoder = model.to(device).eval(), text_encoder.to(device).eval()
    mask96 = np.ones((96, 96, 3), dtype=np.float32)
    mask224 = np.ones((224, 224, 3), dtype=np.float32)
    mid = torch.tensor([MODALITY[args.modality]], device=device)

    def predict(ep, idx, frame_rows, front, left, instruction):
        b = build_batch(f'{ep}/frames', idx, frame_rows, front, left, instruction,
                        device, text_encoder, mask96, mask224,
                        transform_images_PIL_mask, transform_images_map)
        with torch.no_grad():
            a, _, _ = model(b['obs_images'], b['goal_pose_torch'], b['map_images'],
                            b['goal_image'], mid, b['feat_text'], b['cur_large_img'])
        wp = a[0].float().cpu().numpy()
        return bearing_deg(wp[-1, 0] * S, wp[-1, 1] * S)

    hdr = f'{"episode":24} {"pair":22} {"cmdA":>6} {"brgA":>7} {"cmdB":>6} {"brgB":>7} {"dBrg":>6}  pass'
    print(hdr); print('-' * len(hdr))
    npass, dsum = 0, 0.0
    for r in pairs:
        ep = r['ep']
        frame_rows = read_jsonl(f'{ep}/frames.jsonl')
        track = PoseTrack(read_jsonl(f'{ep}/gt_pose.jsonl'))
        pose = track.at(frame_rows[0]['t'])
        g, n = r['goal'], r['neg']
        gf, gl = to_body(g['x'], g['y'], pose)
        nf, nl = to_body(n['x'], n['y'], pose)
        insA = f'drive to the {g["color"]} {g["shape"]}'
        insB = f'drive to the {n["color"]} {n["shape"]}'
        # Each run is conditioned on its own commanded object (matters for
        # pose/both; for language-only the pose channel is masked out).
        bA = predict(ep, 0, frame_rows, gf, gl, insA)
        bB = predict(ep, 0, frame_rows, nf, nl, insB)
        ok = (bA * r['gb'] > 0) and (bB * r['nb'] > 0)
        npass += ok
        dsum += abs(bA - bB)
        pair = f'{g["color"]} vs {n["color"]} {g["shape"]}'
        print(f'{os.path.basename(ep):24} {pair:22} {r["gb"]:+6.0f} {bA:+7.1f} '
              f'{r["nb"]:+6.0f} {bB:+7.1f} {abs(bA-bB):6.1f}  {"PASS" if ok else "."}')

    n = len(pairs)
    print(f'\nswap {npass}/{n} = {npass/n:.2f}   mean |dBearing| = {dsum/n:.1f} deg')
    print('SmolVLA reference (grounding_diagnosis.md): stage1_v3 15.3 deg, '
          'stage1d_deeplm 22.3 deg, swap ~chance')


if __name__ == '__main__':
    main()
