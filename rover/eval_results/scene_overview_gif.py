#!/usr/bin/env python3
"""Top-down world-frame replay of a recorded episode -> animated GIF.

Draws, from a raw episode dir (scene_config.json + gt_pose.jsonl):
  - props at their world positions (color = prop color, marker = shape),
    the goal ringed;
  - the privileged global A* path (recomputed from the config, world frame);
  - the ego rover (triangle at the logged pose) + its traveled path;
  - the local body-frame waypoint chunk (K x (x,y,v) hindsight, drawn ahead of
    the ego in the world) + a body-frame inset showing the same chunk with the
    rover at the origin (x forward, y left).

Self-contained (planner + relabel logic inlined) — pure Python + cv2 + PIL, no
ROS. Run e.g. in a throwaway smolvla-edge:sim container (CPU only):
  python rover/eval_results/scene_overview_gif.py \
      --episode-dir rover/data/raw_v3/open_ground_seed01006 --out rover/overview.gif
"""

import argparse
import bisect
import heapq
import json
import math
import os

import cv2
import numpy as np
from PIL import Image

# ---- planner constants (mirror rover_expert/expert_driver.py) --------------
ROVER_RADIUS = 0.18
INFLATE = ROVER_RADIUS + 0.10
GRID_RES = 0.05
PROP_RADIUS = {'barrel': 0.15, 'pillar': 0.08, 'crate': 0.29, 'ball': 0.15}
DRAW_RADIUS = {'barrel': 0.15, 'pillar': 0.08, 'ball': 0.15}   # crate drawn as box
SCENE_STATICS = {
    'open_ground': [], 'props_ground': [],
    'corridor': [(3.0, 0.9, 10.0, 0.1), (3.0, -0.9, 10.0, 0.1)],
    'parking_lot': [(4.0, -0.6, 0.55, 0.25), (4.0, 0.6, 0.55, 0.25)],
}
SCENE_AREA = {
    'open_ground': (-1.5, 9.0, -4.0, 4.0), 'props_ground': (-1.5, 9.0, -4.0, 4.0),
    'corridor': (-1.5, 9.0, -1.5, 1.5), 'parking_lot': (-1.5, 9.0, -3.0, 3.0),
}
GOAL_REACH = 0.60
K, DT = 10, 0.25
# color name -> BGR
COL = {'red': (40, 40, 210), 'blue': (210, 90, 40),
       'green': (60, 170, 60), 'yellow': (40, 200, 220)}


# ---- A* on the inflated occupancy grid (inlined) ---------------------------
class Grid:
    def __init__(self, cfg):
        self.x0, self.x1, self.y0, self.y1 = SCENE_AREA[cfg['scene']]
        self.nx = int((self.x1 - self.x0) / GRID_RES) + 1
        self.ny = int((self.y1 - self.y0) / GRID_RES) + 1
        self.blocked = bytearray(self.nx * self.ny)
        for i, p in enumerate(cfg['props']):
            r = (PROP_RADIUS[p['shape']] + ROVER_RADIUS if i == cfg['goal_index']
                 else PROP_RADIUS[p['shape']] + INFLATE)
            self._circle(p['x'], p['y'], r)
        for cx, cy, sx, sy in SCENE_STATICS[cfg['scene']]:
            self._rect(cx, cy, sx / 2 + INFLATE, sy / 2 + INFLATE)

    def _i(self, ix, iy):
        return iy * self.nx + ix

    def cell(self, x, y):
        return (int(round((x - self.x0) / GRID_RES)),
                int(round((y - self.y0) / GRID_RES)))

    def xy(self, ix, iy):
        return self.x0 + ix * GRID_RES, self.y0 + iy * GRID_RES

    def _circle(self, x, y, r):
        for iy in range(self.ny):
            for ix in range(self.nx):
                px, py = self.xy(ix, iy)
                if math.hypot(px - x, py - y) <= r:
                    self.blocked[self._i(ix, iy)] = 1

    def _rect(self, cx, cy, hx, hy):
        a = self.cell(cx - hx, cy - hy)
        b = self.cell(cx + hx, cy + hy)
        for iy in range(max(0, a[1]), min(self.ny, b[1] + 1)):
            for ix in range(max(0, a[0]), min(self.nx, b[0] + 1)):
                self.blocked[self._i(ix, iy)] = 1

    def free(self, ix, iy):
        return 0 <= ix < self.nx and 0 <= iy < self.ny and not self.blocked[self._i(ix, iy)]

    def los(self, a, b):
        d = math.hypot(b[0] - a[0], b[1] - a[1])
        n = max(2, int(d / (GRID_RES / 2)))
        for i in range(n + 1):
            t = i / n
            if not self.free(*self.cell(a[0] + t * (b[0] - a[0]),
                                        a[1] + t * (b[1] - a[1]))):
                return False
        return True


def nearest_free(g, c):
    for r in range(1, 30):
        for dx in range(-r, r + 1):
            for dy in range(-r, r + 1):
                if max(abs(dx), abs(dy)) == r and g.free(c[0] + dx, c[1] + dy):
                    return (c[0] + dx, c[1] + dy)
    return None


def astar(g, s_xy, go_xy):
    s, go = g.cell(*s_xy), g.cell(*go_xy)
    if not g.free(*s):
        s = nearest_free(g, s)
    if not g.free(*go):
        go = nearest_free(g, go)
    if not s or not go:
        return None

    def h(c):
        dx, dy = abs(c[0] - go[0]), abs(c[1] - go[1])
        return max(dx, dy) + 0.41421 * min(dx, dy)

    oq = [(h(s), 0.0, s)]
    came, gsc = {s: None}, {s: 0.0}
    while oq:
        _, gg, cur = heapq.heappop(oq)
        if cur == go:
            path = []
            while cur is not None:
                path.append(g.xy(*cur))
                cur = came[cur]
            return path[::-1]
        if gg > gsc.get(cur, 1e18):
            continue
        cx, cy = cur
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                if dx == 0 and dy == 0:
                    continue
                nx, ny = cx + dx, cy + dy
                if not g.free(nx, ny):
                    continue
                if dx and dy and not (g.free(cx + dx, cy) and g.free(cx, cy + dy)):
                    continue
                ng = gg + (1.41421 if dx and dy else 1.0)
                if ng < gsc.get((nx, ny), 1e18):
                    gsc[(nx, ny)] = ng
                    came[(nx, ny)] = cur
                    heapq.heappush(oq, (ng + h((nx, ny)), ng, (nx, ny)))
    return None


def smooth(g, path):
    if not path:
        return path
    out = [path[0]]
    i = 0
    while i < len(path) - 1:
        j = len(path) - 1
        while j > i + 1 and not g.los(path[i], path[j]):
            j -= 1
        out.append(path[j])
        i = j
    dense = []
    for a, b in zip(out, out[1:]):
        n = max(1, int(math.hypot(b[0] - a[0], b[1] - a[1]) / 0.05))
        for kk in range(n):
            t = kk / n
            dense.append((a[0] + t * (b[0] - a[0]), a[1] + t * (b[1] - a[1])))
    dense.append(out[-1])
    return dense


# ---- pose track + hindsight chunk (inlined from relabel.py) ----------------
def load_jsonl(p):
    return [json.loads(l) for l in open(p) if l.strip()]


class Pose:
    def __init__(self, rows):
        self.t = [r['t'] for r in rows]
        self.rows = rows

    def at(self, t):
        i = bisect.bisect_left(self.t, t)
        if i <= 0:
            r = self.rows[0]
            return r['x'], r['y'], r['yaw']
        if i >= len(self.t):
            r = self.rows[-1]
            return r['x'], r['y'], r['yaw']
        r0, r1 = self.rows[i - 1], self.rows[i]
        a = (t - r0['t']) / (r1['t'] - r0['t']) if r1['t'] > r0['t'] else 0.0
        dy = (r1['yaw'] - r0['yaw'] + math.pi) % (2 * math.pi) - math.pi
        return (r0['x'] + a * (r1['x'] - r0['x']),
                r0['y'] + a * (r1['y'] - r0['y']), r0['yaw'] + a * dy)


def body_chunk(track, t0):
    x0, y0, yaw0 = track.at(t0)
    c, s = math.cos(-yaw0), math.sin(-yaw0)
    out = []
    for i in range(1, K + 1):
        xw, yw, _ = track.at(t0 + i * DT)
        dx, dy = xw - x0, yw - y0
        out.append((c * dx - s * dy, s * dx + c * dy))
    return out


# ---- rendering -------------------------------------------------------------
def render(ep_dir, out_path, fps=10):
    cfg = json.load(open(os.path.join(ep_dir, 'scene_config.json')))
    poses = load_jsonl(os.path.join(ep_dir, 'gt_pose.jsonl'))
    track = Pose(poses)
    goal = cfg['props'][cfg['goal_index']]

    raw = astar(Grid(cfg), (poses[0]['x'], poses[0]['y']), (goal['x'], goal['y']))
    gpath = smooth(Grid(cfg), raw) if raw else []

    # world bounds from props + full trajectory
    xs = [p['x'] for p in cfg['props']] + [r['x'] for r in poses]
    ys = [p['y'] for p in cfg['props']] + [r['y'] for r in poses]
    m = 0.8
    wx0, wx1 = min(xs) - m, max(xs) + m
    wy0, wy1 = min(ys) - m, max(ys) + m
    W, H, pad = 760, 560, 20
    sc = min((W - 2 * pad) / (wx1 - wx0), (H - 2 * pad) / (wy1 - wy0))

    def px(x, y):  # world -> image (y-up flipped)
        return (int(pad + (x - wx0) * sc), int(H - pad - (y - wy0) * sc))

    def base_frame():
        img = np.full((H, W, 3), 28, np.uint8)
        # faint 1 m grid
        gx = math.floor(wx0)
        while gx <= wx1:
            cv2.line(img, px(gx, wy0), px(gx, wy1), (44, 44, 44), 1)
            gx += 1
        gy = math.floor(wy0)
        while gy <= wy1:
            cv2.line(img, px(wx0, gy), px(wx1, gy), (44, 44, 44), 1)
            gy += 1
        # global A* path (amber)
        for a, b in zip(gpath, gpath[1:]):
            cv2.line(img, px(*a), px(*b), (40, 165, 220), 2)
        # props
        for i, p in enumerate(cfg['props']):
            c = COL.get(p['color'], (200, 200, 200))
            if p['shape'] == 'crate':
                h = 0.2
                cv2.rectangle(img, px(p['x'] - h, p['y'] + h),
                              px(p['x'] + h, p['y'] - h), c, -1)
            else:
                cv2.circle(img, px(p['x'], p['y']),
                           max(4, int(DRAW_RADIUS[p['shape']] * sc)), c, -1)
            if i == cfg['goal_index']:
                cv2.circle(img, px(p['x'], p['y']),
                           int(GOAL_REACH * sc), (255, 255, 255), 2)
                cv2.putText(img, 'GOAL', (px(p['x'], p['y'])[0] - 16,
                            px(p['x'], p['y'])[1] - int(GOAL_REACH * sc) - 6),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
        return img

    base = base_frame()
    step = max(1, int(round(50 / fps)))   # gt_pose is ~50 Hz
    idx = list(range(0, len(poses), step))
    frames = []
    for k in idx:
        t = poses[k]['t']
        x, y, yaw = track.at(t)
        img = base.copy()

        # traveled path (blue)
        trav = [px(poses[j]['x'], poses[j]['y']) for j in range(0, k + 1, 2)]
        for a, b in zip(trav, trav[1:]):
            cv2.line(img, a, b, (235, 150, 70), 3)

        # local body-frame chunk -> world (the short-horizon intent)
        chunk = body_chunk(track, t)
        cw, sw = math.cos(yaw), math.sin(yaw)
        pts = [px(x + cw * bx - sw * by, y + sw * bx + cw * by) for bx, by in chunk]
        for a, b in zip([px(x, y)] + pts, pts):
            cv2.line(img, a, b, (120, 240, 120), 2)
        for pt in pts:
            cv2.circle(img, pt, 3, (120, 240, 120), -1)

        # ego triangle
        L, wq = 0.34, 0.18
        tri = np.array([px(x + cw * L, y + sw * L),
                        px(x - cw * 0.12 - sw * wq, y - sw * 0.12 + cw * wq),
                        px(x - cw * 0.12 + sw * wq, y - sw * 0.12 - cw * wq)], np.int32)
        cv2.fillPoly(img, [tri], (255, 255, 255))
        cv2.polylines(img, [tri], True, (30, 30, 30), 1)

        # body-frame inset (rover at origin, x forward=up, y left=left)
        iw, ih = 150, 150
        ox, oy = W - iw - 10, 10
        cv2.rectangle(img, (ox, oy), (ox + iw, oy + ih), (18, 18, 18), -1)
        cv2.rectangle(img, (ox, oy), (ox + iw, oy + ih), (80, 80, 80), 1)
        cx, cy, bs = ox + iw // 2, oy + ih - 24, 26.0  # px per metre
        cv2.putText(img, 'body frame', (ox + 6, oy + 14),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, (170, 170, 170), 1, cv2.LINE_AA)
        prev = (cx, cy)
        for bx, by in chunk:
            q = (int(cx - by * bs), int(cy - bx * bs))  # x fwd=up, y left=left
            cv2.line(img, prev, q, (120, 240, 120), 2)
            cv2.circle(img, q, 2, (120, 240, 120), -1)
            prev = q
        cv2.circle(img, (cx, cy), 4, (255, 255, 255), -1)

        # title
        cv2.putText(img, f"world-frame overview  |  \"{cfg['instruction']}\"  |  t={t - poses[0]['t']:5.1f}s",
                    (12, H - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (230, 230, 230), 1, cv2.LINE_AA)
        cv2.putText(img, "global A* (amber)    local chunk (green)    traveled (blue)",
                    (12, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (200, 200, 200), 1, cv2.LINE_AA)

        frames.append(Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)))

    frames[0].save(out_path, save_all=True, append_images=frames[1:],
                   duration=int(1000 / fps), loop=0, optimize=True)
    print(f'wrote {out_path}: {len(frames)} frames, {os.path.getsize(out_path)//1024} KiB')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--episode-dir', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--fps', type=int, default=10)
    a = ap.parse_args()
    render(a.episode_dir, a.out, a.fps)


if __name__ == '__main__':
    main()
