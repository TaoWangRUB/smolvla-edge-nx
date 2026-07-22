#!/usr/bin/env python3
"""Convert raw rover episodes (episode_recorder.py layout) to a LeRobotDataset.

Runs in the lerobot environment (smolvla-edge:sim image), NOT in the ROS
container. Frame-aligned features at the camera rate (15 Hz):

  observation.image    RGB video (native 1280x800)
  observation.state    [speed, yaw_rate, steering]     (interpolated to frame t)
  observation.gt_pose  [x, y, yaw]                     (interpolated to frame t)
  action               [v, w] expert command           (zero-order hold)
  task                 the episode instruction

The action field is provisional: task 2.1's hindsight relabeler rebuilds the
dataset with K x (x, y, v) waypoint chunks from the raw 50 Hz gt_pose stream
(which stays in the raw episode dirs — this converter is not lossy for it).

Usage:
  python to_lerobot.py --raw-root rover/data/raw --out rover/data/lerobot \
      --repo-id local/rover_sim_v0 [--include-failures]
"""

import argparse
import bisect
import json
import math
import pathlib
import shutil

import numpy as np


def read_jsonl(path):
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


class Series:
    """Time-indexed interpolation over a jsonl stream."""

    def __init__(self, rows, fields):
        self.t = [r['t'] for r in rows]
        self.vals = [[r[f] for f in fields] for r in rows]

    def lerp(self, t):
        i = bisect.bisect_left(self.t, t)
        if i <= 0:
            return list(self.vals[0])
        if i >= len(self.t):
            return list(self.vals[-1])
        t0, t1 = self.t[i - 1], self.t[i]
        a = 0.0 if t1 == t0 else (t - t0) / (t1 - t0)
        return [v0 + a * (v1 - v0)
                for v0, v1 in zip(self.vals[i - 1], self.vals[i])]

    def hold(self, t):
        i = bisect.bisect_right(self.t, t)
        return list(self.vals[max(0, i - 1)]) if self.t else None


def lerp_yaw(pose_rows, t):
    """Pose interpolation with yaw wrap handling."""
    ts = [r['t'] for r in pose_rows]
    i = bisect.bisect_left(ts, t)
    if i <= 0:
        r = pose_rows[0]
        return [r['x'], r['y'], r['yaw']]
    if i >= len(ts):
        r = pose_rows[-1]
        return [r['x'], r['y'], r['yaw']]
    r0, r1 = pose_rows[i - 1], pose_rows[i]
    a = 0.0 if r1['t'] == r0['t'] else (t - r0['t']) / (r1['t'] - r0['t'])
    dyaw = r1['yaw'] - r0['yaw']
    while dyaw > math.pi:
        dyaw -= 2 * math.pi
    while dyaw < -math.pi:
        dyaw += 2 * math.pi
    return [r0['x'] + a * (r1['x'] - r0['x']),
            r0['y'] + a * (r1['y'] - r0['y']),
            r0['yaw'] + a * dyaw]


def convert(raw_root, out_root, repo_id, fps, include_failures,
            chunk_k=0, chunk_dt=0.25, use_paraphrase=False, scenes=None,
            goal_state_p_drop=None):
    import cv2
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    if chunk_k or use_paraphrase or goal_state_p_drop is not None:
        import sys as _sys
        _sys.path.insert(0, str(pathlib.Path(__file__).parent))
    if chunk_k or goal_state_p_drop is not None:
        from relabel import PoseTrack, waypoint_chunk
    if goal_state_p_drop is not None:
        from relabel import goal_state
    if use_paraphrase:
        from instructions import paraphrase

    # Goal-channel noise (task 2.11 / design D10): per-episode bias models the
    # acquisition path's systematic error (one detection, held by goal memory),
    # per-frame jitter the odom/projection noise. Scales approximate the
    # measured 5-16 cm ranging error across 2.0-3.5 m (D9 validation).
    GOAL_BIAS_SIGMA = 0.05    # m, drawn once per episode
    GOAL_JITTER_SIGMA = 0.03  # m, drawn per frame

    out = pathlib.Path(out_root) / repo_id.split('/')[-1]
    if out.exists():
        shutil.rmtree(out)

    if chunk_k:
        action_feature = {
            'dtype': 'float32', 'shape': (3 * chunk_k,),
            'names': [f'{f}{i}' for i in range(chunk_k)
                      for f in ('x', 'y', 'v')],
        }
    else:
        action_feature = {'dtype': 'float32', 'shape': (2,),
                          'names': ['v', 'w']}
    features = {
        'observation.image': {'dtype': 'video', 'shape': (800, 1280, 3),
                              'names': ['height', 'width', 'channels']},
        'observation.state': (
            {'dtype': 'float32', 'shape': (7,),
             'names': ['speed', 'yaw_rate', 'steering',
                       'goal_x', 'goal_y', 'goal_cos', 'goal_sin']}
            if goal_state_p_drop is not None else
            {'dtype': 'float32', 'shape': (3,),
             'names': ['speed', 'yaw_rate', 'steering']}),
        'observation.gt_pose': {'dtype': 'float32', 'shape': (3,),
                                'names': ['x', 'y', 'yaw']},
        'action': action_feature,
    }
    ds = LeRobotDataset.create(repo_id=repo_id, fps=fps, root=out,
                               features=features, robot_type='ackermann_1_16',
                               use_videos=True)

    eps = sorted(p for p in pathlib.Path(raw_root).iterdir()
                 if (p / 'episode.json').exists())
    if scenes:
        # Scene families are not always interchangeable: v4's 2.0-3.5 m range
        # makes ~half of `corridor` impassable (1.2 m hallway vs a 0.57 m
        # inflated crate), so its survivors are a selection-biased sample of
        # only-passable layouts. Filtering here keeps that judgement in the
        # conversion command rather than in a hand-curated directory.
        keep = set(scenes)
        eps = [e for e in eps if e.name.rsplit('_seed', 1)[0] in keep]
        print(f'scene filter {sorted(keep)}: {len(eps)} episodes')
    n_done = 0
    for ep in eps:
        meta = json.load(open(ep / 'episode.json'))
        if not meta['success'] and not include_failures:
            print(f'skip (failed): {ep.name}')
            continue
        frames = read_jsonl(ep / 'frames.jsonl')
        pose_rows = read_jsonl(ep / 'gt_pose.jsonl')
        state = Series(read_jsonl(ep / 'state.jsonl'),
                       ['speed', 'yaw_rate', 'steering'])
        cmd = Series(read_jsonl(ep / 'cmd.jsonl'), ['v', 'w'])
        cfg = meta['config']
        task = cfg['instruction']
        if use_paraphrase:
            goal = cfg['props'][cfg['goal_index']]
            task = paraphrase(goal['color'], goal['shape'], cfg['seed'])
        track = (PoseTrack(pose_rows)
                 if (chunk_k or goal_state_p_drop is not None) else None)

        goal_cfg = None
        if goal_state_p_drop is not None:
            # Seed-deterministic per episode: bias, jitter and the dropout
            # draw all reproduce on reconversion (same convention as the
            # paraphrase draw).
            g = cfg['props'][cfg['goal_index']]
            rng = np.random.default_rng(cfg['seed'] * 9973 + 411)
            dropped = rng.random() < goal_state_p_drop
            bias = rng.normal(0.0, GOAL_BIAS_SIGMA, size=2)
            goal_cfg = (g, rng, dropped, bias)

        for fr in frames:
            img = cv2.imread(str(ep / 'frames' / f"{fr['i']:06d}.jpg"))
            if chunk_k:
                ch = waypoint_chunk(track, fr['t'], k=chunk_k, dt=chunk_dt)
                action = np.asarray([c for wp in ch for c in wp],
                                    dtype=np.float32)
            else:
                action = np.asarray(cmd.hold(fr['t']) or [0.0, 0.0],
                                    dtype=np.float32)
            obs_state = np.asarray(state.lerp(fr['t']), dtype=np.float32)
            if goal_cfg is not None:
                g, rng, dropped, bias = goal_cfg
                if dropped:
                    gs = [0.0, 0.0, 0.0, 0.0]   # reserved no-goal value
                else:
                    gs = goal_state(track, fr['t'], (g['x'], g['y']),
                                    bias=bias,
                                    jitter=rng.normal(0.0, GOAL_JITTER_SIGMA,
                                                      size=2))
                obs_state = np.concatenate(
                    [obs_state, np.asarray(gs, dtype=np.float32)])
            frame = {
                'observation.image': cv2.cvtColor(img, cv2.COLOR_BGR2RGB),
                'observation.state': obs_state,
                'observation.gt_pose': np.asarray(lerp_yaw(pose_rows, fr['t']),
                                                  dtype=np.float32),
                'action': action,
            }
            try:
                ds.add_frame({**frame, 'task': task})
            except (TypeError, ValueError):
                ds.add_frame(frame, task=task)
        ds.save_episode()
        n_done += 1
        print(f'converted: {ep.name} ({len(frames)} frames, task={task!r})')
    print(f'done: {n_done} episodes -> {out}')
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--raw-root', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--repo-id', default='local/rover_sim_v0')
    ap.add_argument('--fps', type=int, default=15)
    ap.add_argument('--include-failures', action='store_true')
    ap.add_argument('--chunk-k', type=int, default=0,
                    help='emit K x (x,y,v) hindsight chunks as the action '
                         '(flat 3K dims); 0 = provisional [v, w]')
    ap.add_argument('--chunk-dt', type=float, default=0.25)
    ap.add_argument('--scenes', default='',
                    help='comma-separated scene families to include '
                         '(default: all present)')
    ap.add_argument('--paraphrase', action='store_true',
                    help='replace canonical instructions with train-pool '
                         'paraphrases (heldout pool stays reserved)')
    ap.add_argument('--goal-state', type=float, default=None, metavar='P_DROP',
                    help='append the 4-dim goal channel [gx, gy, cos, sin] to '
                         'observation.state (task 2.11 / design D10); the value '
                         'is the per-episode goal-dropout probability (e.g. 0.3;'
                         ' dropped episodes carry the all-zero no-goal value so '
                         'language stays load-bearing)')
    args = ap.parse_args()
    convert(args.raw_root, args.out, args.repo_id, args.fps,
            args.include_failures, args.chunk_k, args.chunk_dt,
            args.paraphrase,
            scenes=[s for s in args.scenes.split(',') if s] or None,
            goal_state_p_drop=args.goal_state)


if __name__ == '__main__':
    main()
