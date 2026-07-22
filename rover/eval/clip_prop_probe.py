#!/usr/bin/env python3
"""Can CLIP see our props at all?

OmniVLA-edge encodes the instruction with CLIP ViT-B/32 and scores it against
CLIP visual features. Its language conditioning does nothing on our frames
(swap 2/12, mean bearing change 1.5 deg) while its pose conditioning is perfect
(12/12) -- so the failure is in the language/vision pathway. This isolates
which half.

For each same-shape/different-colour pair we project both props into the image
using the known intrinsics and extrinsics, crop them, and ask CLIP to match
each crop to its colour. If CLIP is at chance here, no CLIP-conditioned policy
can ground colour on this imagery and the problem is our renderer, not the
model -- which would also affect the planned NanoOWL/OWL-ViT detector route.

Usage:
    python rover/eval/clip_prop_probe.py --top 12
"""

import argparse
import json
import math
import os
import sys

import torch
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from omnivla_probe import PoseTrack, read_jsonl, to_body
from omnivla_swap import find_pairs

CAM_W, CAM_H = 1280, 800
FX = FY = 537.0237350463867
CX, CY = 640.0, 400.0
CAM_HEIGHT = 0.20
CAM_X_OFFSET = 0.187
PROP_CENTER_Z = 0.25      # approximate centroid height of the props


def project(front, left, z=PROP_CENTER_Z):
    """Body frame (x front, y left, z up) -> pixel. Camera optical frame is
    z forward, x right, y down; pitch is 0 so no rotation is needed."""
    fz = front - CAM_X_OFFSET
    if fz <= 0.1:
        return None
    u = CX - FX * left / fz
    v = CY - FY * (z - CAM_HEIGHT) / fz
    return u, v, fz


def crop(img, front, left, radius=0.18):
    p = project(front, left)
    if p is None:
        return None
    u, v, fz = p
    half = max(40.0, 2.2 * FX * radius / fz)
    box = (u - half, v - half, u + half, v + half)
    if u < -half or u > CAM_W + half:
        return None
    return img.crop(tuple(int(round(b)) for b in box)).resize((224, 224))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--raw-root', default='rover/data/raw')
    ap.add_argument('--top', type=int, default=12)
    ap.add_argument('--save-crops', default=None, help='dir to dump crops for eyeballing')
    args = ap.parse_args()

    import clip
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model, preprocess = clip.load('ViT-B/32', device=device)
    model = model.eval()

    pairs = find_pairs(args.raw_root)[:args.top]
    print(f'{len(pairs)} same-shape/different-colour pairs\n')
    hdr = f'{"episode":24} {"shape":8} {"goal":7} {"neg":7}  {"P(goal|goalCrop)":>16} {"P(neg|negCrop)":>15}  both'
    print(hdr); print('-' * len(hdr))

    nboth = ncrop = 0
    for r in pairs:
        ep, g, n = r['ep'], r['goal'], r['neg']
        frame_rows = read_jsonl(f'{ep}/frames.jsonl')
        track = PoseTrack(read_jsonl(f'{ep}/gt_pose.jsonl'))
        pose = track.at(frame_rows[0]['t'])
        img = Image.open(f'{ep}/frames/{frame_rows[0]["i"]:06d}.jpg').convert('RGB')

        gf, gl = to_body(g['x'], g['y'], pose)
        nf, nl = to_body(n['x'], n['y'], pose)
        cg, cn = crop(img, gf, gl), crop(img, nf, nl)
        if cg is None or cn is None:
            print(f'{os.path.basename(ep):24} {g["shape"]:8} -- prop outside frame, skipped')
            continue
        ncrop += 1

        texts = [f'a {g["color"]} {g["shape"]}', f'a {n["color"]} {n["shape"]}']
        tok = clip.tokenize(texts).to(device)
        ims = torch.cat([preprocess(cg).unsqueeze(0), preprocess(cn).unsqueeze(0)]).to(device)
        with torch.no_grad():
            logits, _ = model(ims, tok)
            probs = logits.softmax(dim=-1).cpu().numpy()
        p_goal, p_neg = probs[0][0], probs[1][1]
        ok = p_goal > 0.5 and p_neg > 0.5
        nboth += ok
        print(f'{os.path.basename(ep):24} {g["shape"]:8} {g["color"]:7} {n["color"]:7} '
              f'{p_goal:16.2f} {p_neg:15.2f}  {"YES" if ok else "."}')

        if args.save_crops:
            os.makedirs(args.save_crops, exist_ok=True)
            b = os.path.basename(ep)
            cg.save(f'{args.save_crops}/{b}_goal_{g["color"]}_{g["shape"]}.png')
            cn.save(f'{args.save_crops}/{b}_neg_{n["color"]}_{n["shape"]}.png')

    if ncrop:
        print(f'\nboth crops correctly matched: {nboth}/{ncrop} = {nboth/ncrop:.2f}  (chance 0.25)')


if __name__ == '__main__':
    main()
