#!/usr/bin/env python3
"""Offline grounding probe (cheap diagnostic, no sim/driving).

For recorded episodes, feed the trained policy the SAME spawn frame under
different instructions and check whether the predicted waypoint chunk turns
toward the commanded object. This isolates model grounding from closed-loop
execution: if the chunk direction tracks the instruction, the policy reads
language; if it's instruction-invariant, it doesn't.

Metrics:
  - directional accuracy: sign(pred lateral y) == sign(commanded prop's
    body-frame lateral) — does it steer to the correct side?
  - swap flip rate: for goal vs same-shape/diff-color hard negative placed on
    OPPOSITE lateral sides, does the predicted side flip when the word flips?
    (the offline analogue of the closed-loop swap test)

Run in smolvla-edge:sim with PYTHONPATH=/work/src.
  python rover/eval_results/grounding_probe.py --checkpoint <ckpt> [--n 40]
"""

import argparse
import glob
import json
import math
import os

import cv2
import numpy as np
import torch


def body_bearing(prop, spawn):
    sx, sy, syaw = spawn
    ang = math.atan2(prop['y'] - sy, prop['x'] - sx) - syaw
    while ang > math.pi:
        ang -= 2 * math.pi
    while ang < -math.pi:
        ang += 2 * math.pi
    return ang  # >0 = left, <0 = right (REP-103 body frame)


def pred_bearing(chunk):
    """Direction of the predicted chunk (mean of waypoints, body frame)."""
    xs = [c[0] for c in chunk]
    ys = [c[1] for c in chunk]
    return math.atan2(sum(ys) / len(ys), sum(xs) / len(xs))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--checkpoint', required=True)
    ap.add_argument('--raw-root', default='rover/data/raw')
    ap.add_argument('--n', type=int, default=40)
    ap.add_argument('--device', default='cuda')
    args = ap.parse_args()

    from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
    from smolvla_edge.common import make_language_tokenizer

    policy = SmolVLAPolicy.from_pretrained(args.checkpoint).eval().to(args.device)
    tok = make_language_tokenizer(policy, args.device)

    eps = sorted(glob.glob(os.path.join(args.raw_root, '*')))
    eps = [e for e in eps if os.path.exists(os.path.join(e, 'scene_config.json'))][:args.n]

    dir_hits = dir_tot = 0
    flip_hits = flip_tot = 0
    per_prop_delta = []  # |pred bearing difference| across instructions in a scene

    for ep in eps:
        cfg = json.load(open(os.path.join(ep, 'scene_config.json')))
        fp = os.path.join(ep, 'frames', '000000.jpg')
        if not os.path.exists(fp):
            continue
        img = cv2.cvtColor(cv2.imread(fp), cv2.COLOR_BGR2RGB)
        im = torch.from_numpy(img).permute(2, 0, 1).float().div_(255.0)
        im = im.unsqueeze(0).to(args.device)
        state = torch.zeros(1, 3, device=args.device)
        spawn = tuple(cfg['spawn'])

        preds = {}  # prop name -> pred bearing
        for prop in cfg['props']:
            task = f"drive to the {prop['color']} {prop['shape']}"
            batch = {'observation.images.camera1': im, 'observation.state': state}
            batch.update(tok(task))
            with torch.no_grad():
                policy.reset()
                a = policy.select_action(batch)
            chunk = a.squeeze().reshape(10, 3).cpu().tolist()
            pb = pred_bearing(chunk)
            preds[prop['name']] = pb
            # directional accuracy vs this prop's true bearing (only count props
            # meaningfully off-axis so left/right is well-defined)
            tb = body_bearing(prop, spawn)
            if abs(tb) > math.radians(8):
                dir_tot += 1
                dir_hits += (pb > 0) == (tb > 0)

        # swap flip: goal (idx 0) vs same-shape/diff-color negative (idx 2)
        props = cfg['props']
        if len(props) > 2:
            g, neg = props[0], props[2]
            gb, nb = body_bearing(g, spawn), body_bearing(neg, spawn)
            if (gb > 0) != (nb > 0):  # opposite sides -> flip is well-defined
                flip_tot += 1
                pg, pn = preds[g['name']], preds[neg['name']]
                flip_hits += ((pg > 0) == (gb > 0)) and ((pn > 0) == (nb > 0))
                per_prop_delta.append(abs(pg - pn))

    print(f'episodes probed: {len(eps)}')
    print(f'directional accuracy (steer to correct side): '
          f'{dir_hits}/{dir_tot} = {dir_hits/max(1,dir_tot):.2f}  (chance 0.50)')
    print(f'offline swap flip rate (opposite-side goal vs hard-neg): '
          f'{flip_hits}/{flip_tot} = {flip_hits/max(1,flip_tot):.2f}  (chance 0.25)')
    if per_prop_delta:
        print(f'mean |pred-bearing change| between goal/neg instructions: '
              f'{np.degrees(np.mean(per_prop_delta)):.1f} deg '
              f'(0 = instruction ignored)')


if __name__ == '__main__':
    main()
