#!/usr/bin/env python3
"""End-to-end acquisition test on RECORDED frames (design D9, step c preview).

Runs the full acquisition path offline — no sim, no ROS:

    instruction ("drive to the red crate")
      -> open-vocabulary detector (OWL-ViT)  -> bbox
      -> ground-plane projection             -> body-frame (x, y)
      -> compared against GROUND TRUTH from scene_config.json

Ground truth: the prop's world position and the rover's spawn pose are both in
the episode config, so the true body-frame goal is exact. This measures whether
the acquisition path can replace the policy's (failed) goal selection.

  python rover/eval_results/test_acquisition_offline.py --raw-root rover/data/raw_v4 --n 20
"""

import argparse
import glob
import json
import math
import os
import sys

sys.path.insert(0, 'rover/ros2/src/rover_runtime/scripts')
from goal_projection import (  # noqa: E402
    CAM_HEIGHT, OBJECT_RADIUS, bbox_to_body_groundplane)


def tiles_of(img, spec):
    """Overlapping tiles as (crop, (x_offset, y_offset)). '1x1' = whole image.

    Small distant props survive the detector's square resize poorly; cropping
    raises their effective resolution. 50% overlap so a prop straddling a tile
    boundary still lands whole in a neighbour.
    """
    cols, rows = (int(v) for v in spec.lower().split('x'))
    if cols == 1 and rows == 1:
        return [(img, (0, 0))]
    W, H = img.size
    tw, th = W // cols, H // rows
    sx, sy = tw // 2, th // 2                      # 50% overlap
    out = []
    for r in range(rows * 2 - 1):
        for c in range(cols * 2 - 1):
            x, y = c * sx, r * sy
            if x + tw > W or y + th > H:
                continue
            out.append((img.crop((x, y, x + tw, y + th)), (x, y)))
    return out


def true_body_goal(cfg, idx=None):
    """Commanded prop in the body frame at spawn (exact, from the config)."""
    idx = cfg['goal_index'] if idx is None else idx
    p = cfg['props'][idx]
    sx, sy, syaw = cfg['spawn']
    dx, dy = p['x'] - sx, p['y'] - sy
    c, s = math.cos(-syaw), math.sin(-syaw)
    return c * dx - s * dy, s * dx + c * dy


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--raw-root', default='rover/data/raw_v4')
    ap.add_argument('--n', type=int, default=20)
    ap.add_argument('--model', default='google/owlv2-base-patch16-ensemble')
    ap.add_argument('--thresh', type=float, default=0.08)
    ap.add_argument('--tiles', default='1x1',
                    help='CxR overlapping tiles, e.g. 3x2. Acquisition runs at 0.1-1 Hz\n'
                         '(design D3), so extra forward passes are affordable.')
    ap.add_argument('--multi-query', dest='single_query', action='store_false',
                    help='batch all prop phrases in one pass (suppresses targets)')
    ap.add_argument('--template', default='{}',
                    help="prompt template, e.g. 'a photo of a {}'")
    ap.add_argument('--frames', type=int, default=1,
                    help='try the first K frames; goal memory needs only ONE hit')
    ap.add_argument('--calibrate', action='store_true',
                    help='fit camera height against ground truth and report it')
    args = ap.parse_args()

    import torch
    from PIL import Image
    from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection

    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    proc = AutoProcessor.from_pretrained(args.model)
    model = AutoModelForZeroShotObjectDetection.from_pretrained(args.model).eval().to(dev)
    print(f'detector: {args.model} on {dev}')

    eps = [d for d in sorted(glob.glob(os.path.join(args.raw_root, '*')))
           if os.path.exists(os.path.join(d, 'scene_config.json'))][:args.n]

    ok_detect = 0
    samples = []
    total = 0

    for d in eps:
        cfg = json.load(open(os.path.join(d, 'scene_config.json')))
        fps = [os.path.join(d, 'frames', f'{i:06d}.jpg') for i in range(args.frames)]
        fps = [f for f in fps if os.path.exists(f)]
        if not fps:
            continue
        total += 1
        goal = cfg['props'][cfg['goal_index']]
        # query every prop description present, so we can check the detector
        # picks the COMMANDED one rather than just "an object"
        queries = [args.template.format(f"{p['color']} {p['shape']}") for p in cfg['props']]
        uniq = list(dict.fromkeys(queries))
        target_q = args.template.format(f"{goal['color']} {goal['shape']}")

        # ONE query per forward pass. Batching every prop phrase together lets
        # OWL-ViT's queries suppress each other -- measured: 13/39 targets
        # scored EXACTLY 0.0 in multi-query mode and 0.10-0.29 alone.
        qlist = [target_q] if args.single_query else uniq
        ti = qlist.index(target_q)
        best, best_score = None, -1
        for fp in fps:                     # first confident hit wins
            img = Image.open(fp).convert('RGB')
            for crop, (ox, oy) in tiles_of(img, args.tiles):
                inputs = proc(text=[qlist], images=crop, return_tensors='pt').to(dev)
                with torch.no_grad():
                    out = model(**inputs)
                res = proc.post_process_grounded_object_detection(
                    outputs=out, target_sizes=torch.tensor([crop.size[::-1]]).to(dev),
                    threshold=args.thresh)[0]
                for score, label, box in zip(res['scores'], res['labels'], res['boxes']):
                    if int(label) == ti and float(score) > best_score:
                        b = [float(v) for v in box]
                        # tile coords -> full-image coords
                        best = [b[0] + ox, b[1] + oy, b[2] + ox, b[3] + oy]
                        best_score = float(score)
            if best is not None:
                break
        if best is None:
            continue
        ok_detect += 1

        samples.append((tuple(best), OBJECT_RADIUS.get(goal['shape'], 0.0), cfg))

    def evaluate(h):
        errs, bears, right = [], [], 0
        for bbox, r, cfg in samples:
            proj = bbox_to_body_groundplane(bbox, cam_height=h, object_radius=r)
            if proj is None:
                continue
            tx, ty = true_body_goal(cfg)
            errs.append(math.hypot(proj[0] - tx, proj[1] - ty))
            bears.append(abs(math.atan2(proj[1], proj[0]) - math.atan2(ty, tx)))
            # nearest-prop check: does this acquisition send us to the right one?
            cand = [(math.hypot(proj[0] - bx, proj[1] - by), i) for i, (bx, by)
                    in enumerate(true_body_goal(cfg, i) for i in range(len(cfg['props'])))]
            right += (min(cand)[1] == cfg['goal_index'])
        return errs, bears, right

    print(f'\nepisodes: {total} | commanded phrase detected: {ok_detect}/{total}')

    if args.calibrate and samples:
        best_h, best_med = None, 1e9
        for i in range(10, 41):
            h = i / 100.0
            e, _, _ = evaluate(h)
            med = sorted(e)[len(e) // 2] if e else 1e9
            if med < best_med:
                best_h, best_med = h, med
        print(f'CALIBRATION: best cam_height = {best_h:.2f} m '
              f'(median err {best_med:.2f} m); module default = {CAM_HEIGHT:.2f} m')

    errs, bearing_errs, picked_right = evaluate(CAM_HEIGHT)
    n = len(errs)
    if n:
        errs_s = sorted(errs)
        print(f'position error  median {errs_s[n//2]:.2f} m   mean {sum(errs)/n:.2f} m'
              f'   p90 {errs_s[int(0.9*(n-1))]:.2f} m')
        print(f'bearing error   median {math.degrees(sorted(bearing_errs)[n//2]):.1f} deg')
        print(f'acquisition selects the COMMANDED prop: {picked_right}/{n} '
              f'({100*picked_right/n:.0f}%)   <-- the number that matters')
        print(f'within 0.6 m goal ring: {sum(e < 0.6 for e in errs)}/{n}')
        print(f'(policy baseline for comparison: swap-flip ~= chance, 2-3/10 success)')


if __name__ == '__main__':
    main()
