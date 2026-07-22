#!/usr/bin/env python3
"""Hindsight waypoint relabeler (task 1.8 exit / task 2.1 core).

For a frame at time t0: the future GT poses at t0 + k*dt (k=1..K) are
transformed into the body frame at t0, each with the speed at that future
time, giving the K x (x, y, v) waypoint chunk the policy learns to emit
(design D2). Works identically for clean and noisy-recovery data because it
reads only logged poses — no planner internals.

Episode end handling: futures past the last pose clamp to the final pose
with v = 0 (the "stop" label).

Pure stdlib — runs in both the ROS container and the lerobot image.

CLI demo (prints chunks at 10/50/90% of an episode + sanity checks):
  python relabel.py --episode-dir rover/data/raw/open_ground_seed00002
"""

import argparse
import bisect
import json
import math

K_DEFAULT = 12
DT_DEFAULT = 0.25


def read_jsonl(path):
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


class PoseTrack:
    """Interpolates (x, y, yaw, speed) from the 50 Hz gt_pose stream."""

    def __init__(self, rows):
        self.t = [r['t'] for r in rows]
        self.rows = rows

    def at(self, t):
        i = bisect.bisect_left(self.t, t)
        if i <= 0:
            r = self.rows[0]
            return r['x'], r['y'], r['yaw'], math.hypot(r['vx'], r['vy'])
        if i >= len(self.t):
            r = self.rows[-1]
            return r['x'], r['y'], r['yaw'], 0.0   # past the end: stopped
        r0, r1 = self.rows[i - 1], self.rows[i]
        a = (t - r0['t']) / (r1['t'] - r0['t']) if r1['t'] > r0['t'] else 0.0
        dyaw = r1['yaw'] - r0['yaw']
        while dyaw > math.pi:
            dyaw -= 2 * math.pi
        while dyaw < -math.pi:
            dyaw += 2 * math.pi
        return (r0['x'] + a * (r1['x'] - r0['x']),
                r0['y'] + a * (r1['y'] - r0['y']),
                r0['yaw'] + a * dyaw,
                math.hypot(r0['vx'] + a * (r1['vx'] - r0['vx']),
                           r0['vy'] + a * (r1['vy'] - r0['vy'])))


def waypoint_chunk(track, t0, k=K_DEFAULT, dt=DT_DEFAULT):
    """K x (x, y, v) in the body frame of the pose at t0."""
    x0, y0, yaw0, _ = track.at(t0)
    c, s = math.cos(-yaw0), math.sin(-yaw0)
    chunk = []
    for i in range(1, k + 1):
        xw, yw, _, v = track.at(t0 + i * dt)
        dx, dy = xw - x0, yw - y0
        chunk.append((c * dx - s * dy, s * dx + c * dy, v))
    return chunk


def waypoint_chunk_arclen(track, t0, k=K_DEFAULT, ds=0.125, cruise=0.5):
    """K x (x, y, v) at fixed ds ARC-LENGTH spacing along the remaining path.

    The early-stop fix (tasks 2.11 follow-up): time-parameterized labels
    shrink toward a dot as the expert decelerates at the goal ring, teaching
    the policy a stop it then fires ~30 cm early; closed-loop the tracker's
    at_end latch turns that into a permanent park (frozen frame -> same
    prediction). Arc-length labels keep full geometric extent until the path
    physically ends, so there is no early stop to learn -- speed becomes the
    executor's job. v is pinned at cruise; only waypoints clamped at the
    path's end (the expert's in-ring stop point) carry v=0.
    """
    x0, y0, yaw0, _ = track.at(t0)
    c, s = math.cos(-yaw0), math.sin(-yaw0)
    pts = [(0.0, 0.0)]
    cum = [0.0]
    i = bisect.bisect_left(track.t, t0)
    for r in track.rows[i:]:
        dx, dy = r['x'] - x0, r['y'] - y0
        bx, by = c * dx - s * dy, s * dx + c * dy
        d = math.hypot(bx - pts[-1][0], by - pts[-1][1])
        if d < 1e-3:            # skip stationary/jitter samples
            continue
        pts.append((bx, by))
        cum.append(cum[-1] + d)
    out = []
    for j in range(1, k + 1):
        sj = j * ds
        if sj >= cum[-1]:       # past the path end: expert's stop point, v=0
            out.append((pts[-1][0], pts[-1][1], 0.0))
            continue
        idx = bisect.bisect_right(cum, sj) - 1
        a = (sj - cum[idx]) / (cum[idx + 1] - cum[idx])
        out.append((pts[idx][0] + a * (pts[idx + 1][0] - pts[idx][0]),
                    pts[idx][1] + a * (pts[idx + 1][1] - pts[idx][1]),
                    cruise))
    return out


def goal_body(track, t0, goal_xy):
    """Commanded goal's world (x, y) -> (front, left) in the body frame at t0.

    The static-point counterpart of waypoint_chunk's transform: this is what
    goal_memory_node computes online from the odom-frame goal, produced here
    in hindsight for the goal-state training channel (task 2.11 / design D10).
    """
    x0, y0, yaw0, _ = track.at(t0)
    dx, dy = goal_xy[0] - x0, goal_xy[1] - y0
    c, s = math.cos(-yaw0), math.sin(-yaw0)
    return c * dx - s * dy, s * dx + c * dy


def goal_state(track, t0, goal_xy, bias=(0.0, 0.0), jitter=(0.0, 0.0)):
    """4-dim goal channel [gx, gy, cos psi, sin psi], psi = bearing to goal.

    bias is a per-episode offset (detector/projection systematic, held fixed
    while the rover moves -- goal memory stores one acquisition), jitter is
    per-frame noise. All-zeros is the reserved "no goal" value ((0,0) for
    cos/sin is off the unit circle, so it never occurs naturally): emitted by
    the converter's goal-dropout episodes so the policy keeps a language-only
    operating mode (OmniVLA's modality-dropout recipe).
    """
    gf, gl = goal_body(track, t0, goal_xy)
    gf += bias[0] + jitter[0]
    gl += bias[1] + jitter[1]
    psi = math.atan2(gl, gf)
    return [gf, gl, math.cos(psi), math.sin(psi)]


def episode_chunks(episode_dir, k=K_DEFAULT, dt=DT_DEFAULT):
    """(frame_t, chunk) for every recorded frame."""
    track = PoseTrack(read_jsonl(f'{episode_dir}/gt_pose.jsonl'))
    frames = read_jsonl(f'{episode_dir}/frames.jsonl')
    return [(fr['t'], waypoint_chunk(track, fr['t'], k, dt)) for fr in frames]


def sanity_check(chunks, cruise=0.55):
    """Structural invariants; returns list of violation strings."""
    bad = []
    for t, ch in chunks:
        if math.hypot(ch[0][0], ch[0][1]) > cruise * DT_DEFAULT * 2.5:
            bad.append(f't={t:.2f}: first waypoint too far {ch[0][:2]}')
        arc = [math.hypot(x, y) for x, y, _ in ch]
        if any(b < a - 0.02 for a, b in zip(arc, arc[1:])):
            # Radial distance may shrink on tight arcs, but never sharply
            # for a forward-only expert.
            bad.append(f't={t:.2f}: radial distance collapses')
        if any(v < -1e-6 or v > cruise + 0.1 for _, _, v in ch):
            bad.append(f't={t:.2f}: speed out of range')
    return bad


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--episode-dir', required=True)
    ap.add_argument('--k', type=int, default=K_DEFAULT)
    ap.add_argument('--dt', type=float, default=DT_DEFAULT)
    args = ap.parse_args()

    chunks = episode_chunks(args.episode_dir, args.k, args.dt)
    n = len(chunks)
    print(f'{n} frames relabeled (K={args.k}, dt={args.dt})')
    for frac in (0.1, 0.5, 0.9):
        t, ch = chunks[int(n * frac)]
        head = '  '.join(f'({x:+.2f},{y:+.2f},{v:.2f})' for x, y, v in ch[:4])
        print(f'  t={t:7.2f}  wp[0:4]: {head}  ...  wp[-1]: '
              f'({ch[-1][0]:+.2f},{ch[-1][1]:+.2f},{ch[-1][2]:.2f})')
    bad = sanity_check(chunks)
    print(f'sanity: {len(bad)} violations' + (f'; first: {bad[0]}' if bad else ''))
    return 0 if not bad else 1


if __name__ == '__main__':
    raise SystemExit(main())
