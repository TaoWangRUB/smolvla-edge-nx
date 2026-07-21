#!/usr/bin/env python3
"""Episode orchestrator (task 1.7): scene -> recorder -> expert -> verdict.

One call produces one raw episode directory (see episode_recorder.py) plus
episode.json = {scene config, expert verdict, success/collision flags,
frame count, wall/sim bookkeeping}. This is the datagen entry point that
2.3 will loop over seeds.

Usage (inside the sim container, workspace sourced, sim already running):
  ros2 run rover_expert run_episode.py --scene open_ground --seed 2 \
      --out-root /vla/rover/data/raw
"""

import argparse
import json
import os
import signal
import subprocess
import sys
import time


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--scene', required=True)
    ap.add_argument('--seed', type=int, required=True)
    ap.add_argument('--out-root', required=True)
    ap.add_argument('--cruise', type=float, default=0.5)
    args = ap.parse_args()

    ep_dir = os.path.join(args.out_root, f'{args.scene}_seed{args.seed:05d}')
    os.makedirs(ep_dir, exist_ok=True)
    cfg_path = os.path.join(ep_dir, 'scene_config.json')

    r = subprocess.run(['ros2', 'run', 'rover_sim', 'scene_manager.py', 'apply',
                        '--scene', args.scene, '--seed', str(args.seed),
                        '--out', cfg_path],
                       capture_output=True, text=True)
    if r.returncode != 0:
        print(f'scene apply failed: {r.stderr.strip()}', file=sys.stderr)
        sys.exit(2)
    time.sleep(2.0)  # let spawned entities render before frames are recorded

    # Own process group so the stop signal reaches the recorder itself, not
    # just the ros2-run wrapper (measured: wrapper swallowed SIGINT and the
    # recorder kept logging across episodes).
    rec = subprocess.Popen(['ros2', 'run', 'rover_expert', 'episode_recorder.py',
                            '--out', ep_dir],
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                           preexec_fn=os.setsid)
    time.sleep(1.0)

    t0 = time.time()
    exp = subprocess.run(['ros2', 'run', 'rover_expert', 'expert_driver.py',
                          '--config', cfg_path, '--cruise', str(args.cruise)],
                         capture_output=True, text=True, timeout=180)
    wall_s = time.time() - t0

    time.sleep(0.5)
    os.killpg(os.getpgid(rec.pid), signal.SIGINT)
    try:
        rec.wait(timeout=10)
    except subprocess.TimeoutExpired:
        os.killpg(os.getpgid(rec.pid), signal.SIGKILL)

    # Scan for the verdict rather than taking the LAST line: `ros2 run` appends
    # "[ros2run]: Process exited with failure 1" to stdout whenever the child
    # exits non-zero, which the expert does on every unsuccessful episode.
    # Taking the last line therefore replaced every real verdict with
    # "expert crashed: " (stderr is empty), hiding the actual reason -- e.g.
    # "A* found no path" -- from both the episode.json and the batch log.
    verdict = None
    for line in reversed(exp.stdout.strip().splitlines()):
        line = line.strip()
        if line.startswith('{') and line.endswith('}'):
            try:
                verdict = json.loads(line)
                break
            except json.JSONDecodeError:
                continue
    if verdict is None:
        verdict = {'success': False,
                   'reason': f'no verdict; rc={exp.returncode} '
                             f'stderr={exp.stderr[-200:]} '
                             f'stdout={exp.stdout[-200:]}'}

    n_frames = len([f for f in os.listdir(os.path.join(ep_dir, 'frames'))
                    if f.endswith('.jpg')]) if os.path.isdir(
                        os.path.join(ep_dir, 'frames')) else 0
    with open(cfg_path) as f:
        cfg = json.load(f)
    episode = {
        'config': cfg,
        'verdict': verdict,
        'success': bool(verdict.get('success')),
        'collision': bool(verdict.get('min_clearance_m', 1.0) is not None
                          and verdict.get('min_clearance_m', 1.0) <= 0.0),
        'frames': n_frames,
        'wall_time_s': round(wall_s, 2),
    }
    with open(os.path.join(ep_dir, 'episode.json'), 'w') as f:
        json.dump(episode, f, indent=2)
    print(json.dumps({'dir': ep_dir, 'success': episode['success'],
                      'frames': n_frames,
                      'time_s': verdict.get('time_s')}))
    sys.exit(0 if episode['success'] else 1)


if __name__ == '__main__':
    main()
