#!/usr/bin/env python3
"""Verify every recorded episode matches the CURRENT sampler (seed-determinism).

`scene_manager.sample()` is a pure function of (scene, seed) plus the
ROVER_DIST_*/ROVER_FILLERS/bearing env knobs. So re-sampling a recorded seed and
diffing prop positions detects any episode generated under different code or
different settings.

This is not hypothetical: `colcon build --symlink-install` makes scene_manager.py
a symlink into the source tree, so editing the sampler changes the NEXT episode
of an already-running batch. On 2026-07-21 that silently split rover_vla_v4 into
323 old-distribution and 95 wide-bearing episodes. Run this before every
conversion/training run — a mixed dataset is invisible in the loss curve and
confounds whatever the dataset was built to test.

  python rover/datagen/verify_dataset.py --raw-root rover/data/raw_v4 \
      --dist-min 2.0 --dist-max 3.5 --fillers 1
  python rover/datagen/verify_dataset.py --raw-root ... --write-regen-list out.txt

Exit code 0 = every episode matches; 1 = mismatches (listed).
"""

import argparse
import glob
import importlib.util
import json
import os
import re
import sys

SM_PATH = 'rover/ros2/src/rover_sim/scripts/scene_manager.py'


def load_sampler(path=SM_PATH):
    spec = importlib.util.spec_from_file_location('scene_manager', path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)       # reads the ROVER_* env set by the caller
    return mod


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--raw-root', default='rover/data/raw_v4')
    ap.add_argument('--dist-min', default='2.0')
    ap.add_argument('--dist-max', default='3.5')
    ap.add_argument('--fillers', default='1')
    ap.add_argument('--bearing-outer', default=None,
                    help='only if the sampler has the v5 wide-bearing patch')
    ap.add_argument('--write-regen-list', default='',
                    help='write mismatching "<scene> <seed>" lines here')
    ap.add_argument('--tol', type=float, default=1e-6)
    args = ap.parse_args()

    # Env must be set BEFORE the module executes -- the knobs are read at import.
    os.environ['ROVER_DIST_MIN'] = args.dist_min
    os.environ['ROVER_DIST_MAX'] = args.dist_max
    os.environ['ROVER_FILLERS'] = args.fillers
    if args.bearing_outer is not None:
        os.environ['ROVER_BEARING_OUTER'] = args.bearing_outer
    sm = load_sampler()

    ok, bad, errs = 0, [], []
    for f in sorted(glob.glob(os.path.join(args.raw_root, '*', 'scene_config.json'))):
        cfg = json.load(open(f))
        try:
            ref = sm.sample(cfg['scene'], cfg['seed'])
        except Exception as e:                       # placement failure etc.
            errs.append((os.path.basename(os.path.dirname(f)), repr(e)))
            continue
        same = len(ref.props) == len(cfg['props']) and all(
            abs(rp.x - cp['x']) < args.tol and abs(rp.y - cp['y']) < args.tol
            and rp.shape == cp['shape'] and rp.color == cp['color']
            for rp, cp in zip(ref.props, cfg['props']))
        same = same and ref.goal_index == cfg['goal_index']
        if same:
            ok += 1
        else:
            bad.append((cfg['scene'], os.path.basename(os.path.dirname(f))))

    print(f'root: {args.raw_root}')
    print(f'  sampler env: dist {args.dist_min}-{args.dist_max}, '
          f'fillers {args.fillers}'
          + (f', bearing_outer {args.bearing_outer}' if args.bearing_outer else ''))
    print(f'  match:    {ok}')
    print(f'  MISMATCH: {len(bad)}')
    if errs:
        print(f'  regen errors: {len(errs)} (e.g. {errs[0]})')
    if bad:
        by_scene = {}
        for scene, _ in bad:
            by_scene[scene] = by_scene.get(scene, 0) + 1
        print(f'  by scene: {by_scene}')
        for _, name in bad[:5]:
            print(f'    e.g. {name}')
        if args.write_regen_list:
            lines = sorted(f"{s} {re.sub(r'.*_seed', '', n)}" for s, n in bad)
            open(args.write_regen_list, 'w').write('\n'.join(lines) + '\n')
            print(f'  regen list -> {args.write_regen_list} ({len(lines)} seeds)')
    return 1 if bad else 0


if __name__ == '__main__':
    sys.exit(main())
