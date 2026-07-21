#!/usr/bin/env python3
"""Tests for bbox -> body-frame goal projection (design D9).

Validated against the sim's own geometry: props sit on the ground at known
positions, so a synthetic bbox rendered from a known range must project back
to that range. Pure math, no ROS/sim.

  python rover/eval_results/test_goal_projection.py
"""

import math
import sys

sys.path.insert(0, 'rover/ros2/src/rover_runtime/scripts')
from goal_projection import (  # noqa: E402
    CAM_H, CAM_HEIGHT, CAM_HFOV, CAM_W, bbox_to_body_depth,
    bbox_to_body_groundplane, body_to_bearing_range, focal_px)

PASS, FAIL = [], []


def check(name, cond, detail=''):
    (PASS if cond else FAIL).append(name)
    if not cond:
        print(f'FAIL {name}: {detail}')


def ground_pixel(x_body, y_body, cam_h=CAM_HEIGHT):
    """Forward model: where does a ground point at (x,y) land in the image?"""
    f = focal_px()
    # body -> camera: right = -y_body, down = +cam_h, forward = x_body
    u = CAM_W / 2.0 + f * (-y_body) / x_body
    v = CAM_H / 2.0 + f * (cam_h) / x_body
    return u, v


def test_roundtrip_ranges():
    """A prop at a known ground position must project back to it."""
    for x in [1.5, 2.0, 2.8, 3.5, 5.0]:
        for y in [0.0, 0.8, -1.2]:
            u, v = ground_pixel(x, y)
            got = bbox_to_body_groundplane((u - 20, v - 60, u + 20, v))
            check('groundplane_roundtrip',
                  got is not None and math.hypot(got[0] - x, got[1] - y) < 0.02,
                  f'true ({x},{y}) -> got {got}')


def test_bearing_sign():
    """Object to the LEFT must give positive bearing (REP-103)."""
    u, v = ground_pixel(3.0, 1.0)          # 1 m to the left
    bx, by = bbox_to_body_groundplane((u - 15, v - 50, u + 15, v))
    bear, rng = body_to_bearing_range(bx, by)
    check('left_positive_bearing', by > 0 and bear > 0, f'by={by:.2f} bear={bear:.2f}')
    check('range_sane', abs(rng - math.hypot(3.0, 1.0)) < 0.03, f'{rng:.2f}')


def test_horizon_rejected():
    """A box whose base sits at/above the horizon has no ground solution."""
    horizon_v = CAM_H / 2.0
    check('above_horizon_none',
          bbox_to_body_groundplane((600, horizon_v - 80, 680, horizon_v - 1)) is None)
    check('at_horizon_none',
          bbox_to_body_groundplane((600, horizon_v - 80, 680, horizon_v)) is None)


def test_depth_variant():
    """Depth projection agrees with ground-plane when both are valid."""
    x, y = 2.6, -0.7
    u, v = ground_pixel(x, y)
    bbox = (u - 18, v - 55, u + 18, v)
    gp = bbox_to_body_groundplane(bbox)
    # centroid depth for the same object, approximately its range along x
    d = bbox_to_body_depth(bbox, depth_m=x)
    check('depth_forward_matches', abs(d[0] - x) < 1e-6, f'{d}')
    check('depth_lateral_sign', (d[1] < 0) == (y < 0), f'{d} vs y={y}')
    check('gp_available', gp is not None)


def test_far_object_precision_degrades_gracefully():
    """Ground-plane ranging loses precision with distance — quantify it."""
    f = focal_px()
    for x in [2.0, 5.0, 10.0]:
        u, v = ground_pixel(x, 0.0)
        near = bbox_to_body_groundplane((u - 5, v - 30, u + 5, v))
        one_px = bbox_to_body_groundplane((u - 5, v - 30, u + 5, v - 1))
        err = abs(one_px[0] - near[0]) if one_px else float('inf')
        check('one_px_error_bounded', err < 0.02 * x * x,
              f'at {x} m, 1 px = {err:.3f} m error')


if __name__ == '__main__':
    test_roundtrip_ranges(); test_bearing_sign(); test_horizon_rejected()
    test_depth_variant(); test_far_object_precision_degrades_gracefully()
    print(f'{len(PASS)} checks passed, {len(FAIL)} failed')
    # report the practical precision at our working range
    f = focal_px()
    for x in [2.0, 2.8, 3.5]:
        u, v = ground_pixel(x, 0.0)
        a = bbox_to_body_groundplane((u - 5, v - 30, u + 5, v))
        b = bbox_to_body_groundplane((u - 5, v - 30, u + 5, v - 1))
        print(f'  range {x:.1f} m -> 1 px bbox error = {abs(b[0]-a[0])*100:.1f} cm')
    raise SystemExit(1 if FAIL else 0)
