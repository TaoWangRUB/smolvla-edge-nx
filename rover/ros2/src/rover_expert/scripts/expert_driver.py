#!/usr/bin/env python3
"""Privileged expert (task 1.6): A* on the episode map + Pure Pursuit.

Input: the episode config JSON written by scene_manager.py (privileged map:
prop poses + per-scene static structures). Plans start->goal on an inflated
occupancy grid, shortcut-smooths the path, tracks it with Pure Pursuit at
50 Hz through /cmd_vel (TwistStamped), respecting the measured feasibility
limit |w| <= v / R_MIN (R_MIN = 0.341 m, inner-wheel steering limit).

Exits with a one-line JSON verdict on stdout:
  {"success": ..., "time_s": ..., "path_len_m": ..., "final_goal_dist_m": ...,
   "min_clearance_m": ..., "instruction": ...}

Verdict rules: success = reached the goal-approach distance without the
clearance monitor ever seeing prop/static penetration, within the timeout.
"""

import argparse
import heapq
import json
import math
import sys

import rclpy
from rclpy.node import Node
from rclpy.parameter import Parameter
from geometry_msgs.msg import TwistStamped
from nav_msgs.msg import Odometry

R_MIN = 0.341                 # measured min feasible turn radius (tasks.md 1.3)
ROVER_RADIUS = 0.18           # circumscribed footprint approximation
INFLATE = ROVER_RADIUS + 0.10
GRID_RES = 0.05
CRUISE = 0.5
GOAL_REACH = 0.60             # stop distance from goal prop center
TIMEOUT_S = 60.0

# Footprint radii of spawnable props (matches scene_manager SHAPES).
PROP_RADIUS = {'barrel': 0.15, 'pillar': 0.08, 'crate': 0.29, 'ball': 0.15}

# Static structures per scene family: axis-aligned boxes (cx, cy, sx, sy).
# Mirrors the world SDFs; keep in sync until worlds are generated (M1).
SCENE_STATICS = {
    'open_ground': [],
    'props_ground': [],
    'corridor': [(3.0, 0.9, 10.0, 0.1), (3.0, -0.9, 10.0, 0.1)],
    'parking_lot': [(4.0, -0.6, 0.55, 0.25), (4.0, 0.6, 0.55, 0.25)],
}
# Planning area per scene (xmin, xmax, ymin, ymax), padded around action.
SCENE_AREA = {
    'open_ground': (-1.5, 9.0, -4.0, 4.0),
    'props_ground': (-1.5, 9.0, -4.0, 4.0),
    'corridor': (-1.5, 9.0, -1.5, 1.5),
    'parking_lot': (-1.5, 9.0, -3.0, 3.0),
}


class Grid:
    def __init__(self, cfg):
        self.x0, self.x1, self.y0, self.y1 = SCENE_AREA[cfg['scene']]
        self.nx = int((self.x1 - self.x0) / GRID_RES) + 1
        self.ny = int((self.y1 - self.y0) / GRID_RES) + 1
        self.blocked = bytearray(self.nx * self.ny)
        goal = cfg['props'][cfg['goal_index']]
        for i, p in enumerate(cfg['props']):
            r = PROP_RADIUS[p['shape']] + INFLATE
            # The goal prop is a *destination*: inflate it less so the
            # approach cell ring stays reachable.
            if i == cfg['goal_index']:
                r = PROP_RADIUS[p['shape']] + ROVER_RADIUS
            self.block_circle(p['x'], p['y'], r)
        for cx, cy, sx, sy in SCENE_STATICS[cfg['scene']]:
            self.block_rect(cx, cy, sx / 2 + INFLATE, sy / 2 + INFLATE)

    def idx(self, ix, iy):
        return iy * self.nx + ix

    def to_cell(self, x, y):
        return (int(round((x - self.x0) / GRID_RES)),
                int(round((y - self.y0) / GRID_RES)))

    def to_xy(self, ix, iy):
        return self.x0 + ix * GRID_RES, self.y0 + iy * GRID_RES

    def in_bounds(self, ix, iy):
        return 0 <= ix < self.nx and 0 <= iy < self.ny

    def block_circle(self, x, y, r):
        ix0, iy0 = self.to_cell(x - r, y - r)
        ix1, iy1 = self.to_cell(x + r, y + r)
        for iy in range(max(0, iy0), min(self.ny, iy1 + 1)):
            for ix in range(max(0, ix0), min(self.nx, ix1 + 1)):
                px, py = self.to_xy(ix, iy)
                if math.hypot(px - x, py - y) <= r:
                    self.blocked[self.idx(ix, iy)] = 1

    def block_rect(self, cx, cy, hx, hy):
        ix0, iy0 = self.to_cell(cx - hx, cy - hy)
        ix1, iy1 = self.to_cell(cx + hx, cy + hy)
        for iy in range(max(0, iy0), min(self.ny, iy1 + 1)):
            for ix in range(max(0, ix0), min(self.nx, ix1 + 1)):
                self.blocked[self.idx(ix, iy)] = 1

    def free(self, ix, iy):
        return self.in_bounds(ix, iy) and not self.blocked[self.idx(ix, iy)]

    def los(self, a, b):
        """Line of sight between metric points, sampled at half resolution."""
        d = math.hypot(b[0] - a[0], b[1] - a[1])
        n = max(2, int(d / (GRID_RES / 2)))
        for i in range(n + 1):
            t = i / n
            ix, iy = self.to_cell(a[0] + t * (b[0] - a[0]),
                                  a[1] + t * (b[1] - a[1]))
            if not self.free(ix, iy):
                return False
        return True


def astar(grid, start_xy, goal_xy):
    start = grid.to_cell(*start_xy)
    goal = grid.to_cell(*goal_xy)
    if not grid.free(*start):
        # Spawn cell may fall inside inflation of nearby clutter; nudge out.
        start = nearest_free(grid, start)
    if not grid.free(*goal):
        goal = nearest_free(grid, goal)
    if start is None or goal is None:
        return None

    def h(c):
        dx, dy = abs(c[0] - goal[0]), abs(c[1] - goal[1])
        return max(dx, dy) + 0.41421 * min(dx, dy)

    openq = [(h(start), 0.0, start)]
    came, gsc = {start: None}, {start: 0.0}
    while openq:
        _, g, cur = heapq.heappop(openq)
        if cur == goal:
            path = []
            while cur is not None:
                path.append(grid.to_xy(*cur))
                cur = came[cur]
            return path[::-1]
        if g > gsc.get(cur, 1e18):
            continue
        cx, cy = cur
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                if dx == 0 and dy == 0:
                    continue
                nx, ny = cx + dx, cy + dy
                if not grid.free(nx, ny):
                    continue
                if dx and dy and not (grid.free(cx + dx, cy) and
                                      grid.free(cx, cy + dy)):
                    continue  # no corner cutting
                ng = g + (1.41421 if dx and dy else 1.0)
                nc = (nx, ny)
                if ng < gsc.get(nc, 1e18):
                    gsc[nc] = ng
                    came[nc] = cur
                    heapq.heappush(openq, (ng + h(nc), ng, nc))
    return None


def nearest_free(grid, cell):
    for r in range(1, 30):
        for dx in range(-r, r + 1):
            for dy in range(-r, r + 1):
                if max(abs(dx), abs(dy)) == r and grid.free(cell[0] + dx,
                                                            cell[1] + dy):
                    return (cell[0] + dx, cell[1] + dy)
    return None


def smooth(grid, path):
    """Line-of-sight shortcutting, then densify to ~5 cm spacing."""
    if not path:
        return path
    out = [path[0]]
    i = 0
    while i < len(path) - 1:
        j = len(path) - 1
        while j > i + 1 and not grid.los(path[i], path[j]):
            j -= 1
        out.append(path[j])
        i = j
    dense = []
    for a, b in zip(out, out[1:]):
        d = math.hypot(b[0] - a[0], b[1] - a[1])
        n = max(1, int(d / 0.05))
        for k in range(n):
            t = k / n
            dense.append((a[0] + t * (b[0] - a[0]), a[1] + t * (b[1] - a[1])))
    dense.append(out[-1])
    return dense


def quat_yaw(q):
    return math.atan2(2.0 * (q.w * q.z + q.x * q.y),
                      1.0 - 2.0 * (q.y * q.y + q.z * q.z))


class Expert(Node):
    def __init__(self, cfg, cruise):
        super().__init__('expert_driver', parameter_overrides=[
            Parameter('use_sim_time', Parameter.Type.BOOL, True)])
        self.cfg = cfg
        self.cruise = cruise
        self.goal = cfg['props'][cfg['goal_index']]
        self.grid = Grid(cfg)
        self.pose = None
        self.t0 = None
        self.wp_i = 0
        self.min_clearance = 1e9
        self.traveled = 0.0
        self.last_xy = None
        self.result = None
        self.path = None
        self.pub = self.create_publisher(TwistStamped, '/cmd_vel', 10)
        self.create_subscription(Odometry, '/ackermann/gt_odom', self.on_odom, 50)
        self.create_timer(0.02, self.tick)

    def plan(self):
        sx, sy = self.pose[0], self.pose[1]
        gx, gy = self.goal['x'], self.goal['y']
        # Goal-visible-at-start validity check (sampler guarantees it; verify).
        bearing = abs(self._norm(math.atan2(gy - sy, gx - sx) - self.pose[2]))
        if bearing > math.radians(50):
            return f'goal not visible at start (bearing {math.degrees(bearing):.0f} deg)'
        raw = astar(self.grid, (sx, sy), (gx, gy))
        if raw is None:
            return 'A* found no path'
        path = smooth(self.grid, raw)
        # Trim the tail that enters the goal-reach circle.
        self.path = [p for p in path
                     if math.hypot(p[0] - gx, p[1] - gy) > GOAL_REACH * 0.8]
        if not self.path:
            return 'path fully inside goal circle'
        return None

    @staticmethod
    def _norm(a):
        while a > math.pi:
            a -= 2 * math.pi
        while a < -math.pi:
            a += 2 * math.pi
        return a

    def on_odom(self, m):
        p = m.pose.pose.position
        self.pose = (p.x, p.y, quat_yaw(m.pose.pose.orientation))
        if self.last_xy is not None:
            self.traveled += math.hypot(p.x - self.last_xy[0],
                                        p.y - self.last_xy[1])
        self.last_xy = (p.x, p.y)
        # Clearance monitor: rover circle vs prop footprints and statics.
        for prop in self.cfg['props']:
            c = (math.hypot(p.x - prop['x'], p.y - prop['y'])
                 - PROP_RADIUS[prop['shape']] - ROVER_RADIUS)
            self.min_clearance = min(self.min_clearance, c)
        for cx, cy, sx, sy in SCENE_STATICS[self.cfg['scene']]:
            dx = max(abs(p.x - cx) - sx / 2, 0.0)
            dy = max(abs(p.y - cy) - sy / 2, 0.0)
            self.min_clearance = min(self.min_clearance,
                                     math.hypot(dx, dy) - ROVER_RADIUS)

    def tick(self):
        if self.pose is None or self.result is not None:
            return
        now = self.get_clock().now().nanoseconds * 1e-9
        if now <= 0:
            return
        if self.t0 is None:
            self.t0 = now
            err = self.plan()
            if err:
                self.finish(False, now, err)
                return
        t = now - self.t0
        x, y, yaw = self.pose
        gdist = math.hypot(x - self.goal['x'], y - self.goal['y'])
        if gdist <= GOAL_REACH:
            self.stop()
            self.finish(True, now)
            return
        if t > TIMEOUT_S:
            self.stop()
            self.finish(False, now, 'timeout')
            return

        # Pure Pursuit: first path point beyond lookahead, monotonic index.
        lookahead = max(0.35, min(1.0, 1.2 * self.cruise))
        while (self.wp_i < len(self.path) - 1 and
               math.hypot(self.path[self.wp_i][0] - x,
                          self.path[self.wp_i][1] - y) < lookahead):
            self.wp_i += 1
        tx, ty = self.path[self.wp_i]
        alpha = self._norm(math.atan2(ty - y, tx - x) - yaw)
        ld = max(0.2, math.hypot(tx - x, ty - y))

        v = self.cruise
        if abs(alpha) > 0.8:            # sharp correction: slow down
            v = 0.2
        v = min(v, max(0.15, 0.8 * gdist))
        w = 2.0 * v * math.sin(alpha) / ld
        wmax = v / R_MIN                # measured feasibility clamp
        w = max(-wmax, min(wmax, w))

        msg = TwistStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.twist.linear.x = v
        msg.twist.angular.z = w
        self.pub.publish(msg)

    def stop(self):
        msg = TwistStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        self.pub.publish(msg)

    def finish(self, success, now, reason=None):
        gdist = (math.hypot(self.pose[0] - self.goal['x'],
                            self.pose[1] - self.goal['y'])
                 if self.pose else None)
        self.result = {
            'success': bool(success and self.min_clearance > 0.0),
            'reached': bool(success),
            'reason': reason,
            'time_s': round(now - (self.t0 or now), 2),
            'path_len_m': round(self.traveled, 3),
            'final_goal_dist_m': round(gdist, 3) if gdist is not None else None,
            'min_clearance_m': round(self.min_clearance, 3),
            'instruction': self.cfg['instruction'],
        }


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--config', required=True, help='episode config JSON')
    ap.add_argument('--cruise', type=float, default=CRUISE)
    args = ap.parse_args()
    with open(args.config) as f:
        cfg = json.load(f)

    rclpy.init()
    node = Expert(cfg, args.cruise)
    while rclpy.ok() and node.result is None:
        rclpy.spin_once(node, timeout_sec=0.5)
    # Let the final stop command flush.
    for _ in range(5):
        node.stop()
        rclpy.spin_once(node, timeout_sec=0.05)
    print(json.dumps(node.result))
    sys.exit(0 if node.result.get('success') else 1)


if __name__ == '__main__':
    main()
