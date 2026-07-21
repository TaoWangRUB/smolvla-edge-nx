#!/usr/bin/env python3
"""Offline tests for the goal-memory transforms (task 5.1).

The claim under test: once a goal is stored in the odom frame, its body-frame
bearing/range stays correct as the rover drives and turns — including when the
goal falls outside the camera FOV. That is the property the memoryless policy
lacks. Pure math, no ROS.

  python rover/eval_results/test_goal_memory.py
"""

import math
import sys

sys.path.insert(0, 'rover/ros2/src/rover_runtime/scripts')
from goal_memory_node import to_body, to_odom, straight_chunk  # noqa: E402

HFOV_HALF = math.radians(50.0)   # the rover's ~100 deg camera
PASS, FAIL = [], []


def check(name, cond, detail=''):
    (PASS if cond else FAIL).append(name)
    if not cond:
        print(f'FAIL {name}: {detail}')


def test_roundtrip():
    for pose in [(0, 0, 0), (2.5, -1.0, 0.7), (-3, 4, -2.2)]:
        for g in [(5, 1), (0.2, -3), (-2, -2)]:
            b = to_body(*g, *pose)
            back = to_odom(*b, *pose)
            check('roundtrip', math.dist(back, g) < 1e-9,
                  f'{g} -> {b} -> {back}')


def test_bearing_sign():
    # goal directly left of a rover facing +x  ->  +y body, bearing +90 deg
    bx, by = to_body(0.0, 2.0, 0.0, 0.0, 0.0)
    check('left_is_positive_y', by > 0 and abs(bx) < 1e-9, f'({bx},{by})')
    # rover rotated +90 deg: the same world point is now straight ahead
    bx, by = to_body(0.0, 2.0, 0.0, 0.0, math.pi / 2)
    check('rotate_puts_goal_ahead', bx > 1.9 and abs(by) < 1e-9, f'({bx},{by})')


def test_persists_when_out_of_view():
    """Drive past a goal and turn away; memory must still point back to it."""
    goal = (3.0, 1.5)
    # rover drives to (4,0) then faces +x: goal is now BEHIND-left
    pose = (4.0, 0.0, 0.0)
    bx, by = to_body(*goal, *pose)
    bearing = math.atan2(by, bx)
    in_view = abs(bearing) <= HFOV_HALF
    check('goal_left_the_fov', not in_view,
          f'bearing {math.degrees(bearing):.0f} deg should exceed {math.degrees(HFOV_HALF):.0f}')
    # ...yet range/bearing remain exact
    check('range_still_exact', abs(math.hypot(bx, by) - math.dist(goal, pose[:2])) < 1e-9)
    check('points_backwards', bx < 0, f'bx={bx:.2f} should be negative (behind)')


def test_chunk_shape_and_stop():
    ch = straight_chunk(3.0, 0.0)
    check('chunk_len', len(ch) == 10, str(len(ch)))
    xs = [c[0] for c in ch]
    check('chunk_monotonic', all(b >= a - 1e-9 for a, b in zip(xs, xs[1:])), str(xs))
    check('chunk_spacing', abs(xs[0] - 0.5 * 0.25) < 1e-6, f'{xs[0]}')
    # a goal already reached -> zero-velocity (parked) chunk
    ch = straight_chunk(0.2, 0.0)
    check('parks_at_goal', all(abs(c[2]) < 1e-9 for c in ch),
          str([c[2] for c in ch]))


def test_chunk_direction():
    ch = straight_chunk(0.0, 2.0)          # goal straight left
    check('chunk_turns_left', all(c[1] >= -1e-9 for c in ch) and ch[-1][1] > 0,
          str(ch[-1]))


if __name__ == '__main__':
    test_roundtrip(); test_bearing_sign(); test_persists_when_out_of_view()
    test_chunk_shape_and_stop(); test_chunk_direction()
    print(f'{len(PASS)} checks passed, {len(FAIL)} failed')
    raise SystemExit(1 if FAIL else 0)
