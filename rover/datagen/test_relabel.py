#!/usr/bin/env python3
"""Unit tests for the hindsight relabeler (task 2.1): straight/turn/stop.

Synthetic 50 Hz pose tracks with known geometry; run directly:
  python rover/datagen/test_relabel.py
"""

import math

from relabel import PoseTrack, waypoint_chunk

PASS = []
FAIL = []


def check(name, cond, detail=''):
    (PASS if cond else FAIL).append(name)
    if not cond:
        print(f'FAIL {name}: {detail}')


def make_rows(fn, T=10.0, hz=50):
    rows = []
    for i in range(int(T * hz) + 1):
        t = i / hz
        x, y, yaw, vx, vy = fn(t)
        rows.append({'t': t, 'x': x, 'y': y, 'yaw': yaw, 'vx': vx, 'vy': vy})
    return rows


def test_straight():
    v = 0.5
    track = PoseTrack(make_rows(lambda t: (v * t, 0.0, 0.0, v, 0.0)))
    ch = waypoint_chunk(track, 2.0, k=10, dt=0.25)
    for i, (x, y, s) in enumerate(ch, 1):
        check('straight.x', abs(x - v * 0.25 * i) < 1e-6, f'wp{i} x={x}')
        check('straight.y', abs(y) < 1e-9, f'wp{i} y={y}')
        check('straight.v', abs(s - v) < 1e-6, f'wp{i} v={s}')


def test_straight_rotated():
    """Same straight line but heading 90 deg: body frame must still see +x."""
    v = 0.5
    track = PoseTrack(make_rows(
        lambda t: (0.0, v * t, math.pi / 2, 0.0, v)))
    ch = waypoint_chunk(track, 2.0, k=10, dt=0.25)
    for i, (x, y, s) in enumerate(ch, 1):
        check('rot.x', abs(x - v * 0.25 * i) < 1e-6, f'wp{i} x={x}')
        check('rot.y', abs(y) < 1e-6, f'wp{i} y={y}')


def test_turn():
    """Constant left turn radius R at speed v: waypoints on the circle."""
    R, v = 1.0, 0.5
    w = v / R
    track = PoseTrack(make_rows(
        lambda t: (R * math.sin(w * t), R * (1 - math.cos(w * t)),
                   w * t, v * math.cos(w * t), v * math.sin(w * t))))
    ch = waypoint_chunk(track, 4.0, k=10, dt=0.25)
    # 1e-4 tolerance: linear interpolation between 50 Hz samples chords the
    # arc by ~3e-6 m; anything below 0.1 mm is irrelevant at rover scale.
    for i, (x, y, s) in enumerate(ch, 1):
        a = w * 0.25 * i
        ex, ey = R * math.sin(a), R * (1 - math.cos(a))
        check('turn.xy', math.hypot(x - ex, y - ey) < 1e-4,
              f'wp{i} ({x:.4f},{y:.4f}) vs ({ex:.4f},{ey:.4f})')
        check('turn.v', abs(s - v) < 1e-4, f'wp{i} v={s}')
        check('turn.left', y > -1e-9, f'wp{i} y={y}')


def test_stop():
    """Track ends at t=5: waypoints past the end clamp to final pose, v=0."""
    v = 0.5
    track = PoseTrack(make_rows(lambda t: (v * t, 0.0, 0.0, v, 0.0), T=5.0))
    ch = waypoint_chunk(track, 4.5, k=10, dt=0.25)
    end_x = v * 5.0 - v * 4.5
    for i, (x, y, s) in enumerate(ch, 1):
        t_i = 4.5 + 0.25 * i
        if t_i <= 5.0:
            check('stop.pre', abs(x - v * 0.25 * i) < 1e-6, f'wp{i} x={x}')
        else:
            check('stop.clamp_x', abs(x - end_x) < 1e-6, f'wp{i} x={x} vs {end_x}')
            check('stop.v0', abs(s) < 1e-9, f'wp{i} v={s}')


if __name__ == '__main__':
    test_straight()
    test_straight_rotated()
    test_turn()
    test_stop()
    print(f'{len(PASS)} checks passed, {len(FAIL)} failed')
    raise SystemExit(1 if FAIL else 0)
