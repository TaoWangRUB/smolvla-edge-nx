#!/usr/bin/env python3
"""Closed-loop policy evaluation (task 2.7): success rate + swap test.

Prerequisites: sim running (matching world), policy server running
(rover/runtime/policy_server.py on the Titan X). Per episode this script
resets the scene (fresh eval seed), starts tracker_node + chunk_client_node
with the episode's instruction, and watches GT odometry until the rover
reaches the commanded target's 0.6 m ring (success), collides (clearance
<= 0), or times out. The rover never sees privileged info — only this
referee does.

Swap test (--swap): each scene is run twice from identical layout — once
commanding the goal, once commanding the same-shape/different-color hard
negative (skipped if that combo is not unique in the scene). The pair
passes when the rover approaches the *commanded* prop both times.

Verdicts stream as JSON lines; a summary prints at the end.
"""

import argparse
import json
import os
import math
import signal
import subprocess
import sys
import time

import rclpy
from rclpy.node import Node
from rclpy.parameter import Parameter
from nav_msgs.msg import Odometry

# Keep in sync with expert_driver.py (same privileged geometry).
PROP_RADIUS = {'barrel': 0.15, 'pillar': 0.08, 'crate': 0.29, 'ball': 0.15}
SCENE_STATICS = {
    'open_ground': [],
    'props_ground': [],
    'corridor': [(3.0, 0.9, 10.0, 0.1), (3.0, -0.9, 10.0, 0.1)],
    'parking_lot': [(4.0, -0.6, 0.55, 0.25), (4.0, 0.6, 0.55, 0.25)],
}
ROVER_RADIUS = 0.18
REACH = 0.60
TIMEOUT_S = 40.0


class Referee(Node):
    """Watches GT odometry; declares reached / collision / timeout."""

    def __init__(self, cfg, target_idx):
        super().__init__('eval_referee', parameter_overrides=[
            Parameter('use_sim_time', Parameter.Type.BOOL, True)])
        self.cfg = cfg
        self.target = cfg['props'][target_idx]
        self.result = None
        self.t0 = None
        self.min_clear = 1e9
        self.pose = None
        self.trace = []          # (t, x, y, yaw) for post-hoc visualisation
        self.create_subscription(Odometry, '/ackermann/gt_odom', self.on_odom, 50)
        self.create_timer(0.05, self.tick)

    def on_odom(self, m):
        p = m.pose.pose.position
        self.pose = (p.x, p.y)
        q = m.pose.pose.orientation
        self.trace.append({
            't': m.header.stamp.sec + m.header.stamp.nanosec * 1e-9,
            'x': round(p.x, 5), 'y': round(p.y, 5),
            'yaw': round(math.atan2(2.0 * (q.w * q.z + q.x * q.y),
                                    1.0 - 2.0 * (q.y * q.y + q.z * q.z)), 5),
            'vx': round(m.twist.twist.linear.x, 5),
            'vy': round(m.twist.twist.linear.y, 5),
        })
        for prop in self.cfg['props']:
            c = (math.hypot(p.x - prop['x'], p.y - prop['y'])
                 - PROP_RADIUS[prop['shape']] - ROVER_RADIUS)
            self.min_clear = min(self.min_clear, c)
        for cx, cy, sx, sy in SCENE_STATICS[self.cfg['scene']]:
            dx = max(abs(p.x - cx) - sx / 2, 0.0)
            dy = max(abs(p.y - cy) - sy / 2, 0.0)
            self.min_clear = min(self.min_clear, math.hypot(dx, dy) - ROVER_RADIUS)

    def tick(self):
        if self.result is not None or self.pose is None:
            return
        now = self.get_clock().now().nanoseconds * 1e-9
        if now <= 0:
            return
        if self.t0 is None:
            self.t0 = now
        d = math.hypot(self.pose[0] - self.target['x'],
                       self.pose[1] - self.target['y'])
        if d <= REACH:
            self.finish('reached', now)
        elif self.min_clear <= 0.0:
            self.finish('collision', now)
        elif now - self.t0 > TIMEOUT_S:
            self.finish('timeout', now)

    def finish(self, outcome, now):
        x, y = self.pose
        dists = {p['name']: round(math.hypot(x - p['x'], y - p['y']), 3)
                 for p in self.cfg['props']}
        nearest = min(dists, key=dists.get)
        self.result = {
            'outcome': outcome,
            'success': outcome == 'reached' and self.min_clear > 0.0,
            'time_s': round(now - self.t0, 2),
            'min_clearance_m': round(self.min_clear, 3),
            'nearest_prop': nearest,
            'dist_to_target': dists[self.target['name']],
        }


def start_node(pkg, script, params):
    args = ['ros2', 'run', pkg, script, '--ros-args']
    for k, v in params.items():
        args += ['-p', f'{k}:={v}']
    import os
    return subprocess.Popen(args, stdout=subprocess.DEVNULL,
                            stderr=subprocess.DEVNULL, preexec_fn=os.setsid)


def stop_node(proc):
    import os
    try:
        os.killpg(os.getpgid(proc.pid), signal.SIGINT)
        proc.wait(timeout=8)
    except Exception:
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
        except Exception:
            pass


def instruction_for(prop):
    return f"drive to the {prop['color']} {prop['shape']}"


def run_one(scene, seed, server_host, server_port, target_idx=0,
            instruction=None, reset_scene=True, trace_dir=None,
            send_goal=False):
    cfg_path = f'/tmp/eval_cfg_{scene}_{seed}.json'
    if reset_scene:
        r = subprocess.run(['ros2', 'run', 'rover_sim', 'scene_manager.py',
                            'apply', '--scene', scene, '--seed', str(seed),
                            '--out', cfg_path],
                           capture_output=True, text=True)
        if r.returncode != 0:
            return {'outcome': 'scene_failed', 'success': False}, None
        time.sleep(2.0)
    else:
        subprocess.run(['ros2', 'run', 'rover_sim', 'scene_manager.py',
                        'apply', '--scene', scene, '--seed', str(seed),
                        '--out', cfg_path], capture_output=True, text=True)
        time.sleep(2.0)
    cfg = json.load(open(cfg_path))
    task = instruction or instruction_for(cfg['props'][target_idx])

    tracker = start_node('rover_runtime', 'tracker_node.py', {})
    client_params = {
        'server_host': server_host, 'server_port': server_port,
        'instruction': task,
    }
    if send_goal:
        # PRIVILEGED: hands the commanded prop's world position to the policy,
        # for measuring a pose-conditioned upper bound. Never set for a
        # language-grounding evaluation.
        g = cfg['props'][target_idx]
        client_params['goal_xy'] = f'{g["x"]},{g["y"]}'
    client = start_node('rover_runtime', 'chunk_client_node.py', client_params)
    time.sleep(1.0)

    rclpy.init()
    ref = Referee(cfg, target_idx)
    # Wall-clock backstop. The Referee's own 40 s cutoff is SIM-time and fires
    # from its /clock + /gt_odom callbacks; if those stall (leaked nodes
    # starving the executor, a wedged sim clock) the loop below never exits,
    # run_one never returns, and stop_node() never runs -- so the tracker and
    # client leak, load climbs, and the next run's teardown lags too. That
    # runaway is what turned a 13 min eval into 90 min on one seed. A generous
    # wall cap (sim TIMEOUT_S / lowest expected RTF, + startup slack) guarantees
    # every run_one returns and tears its nodes down.
    wall_deadline = time.time() + 90.0
    while rclpy.ok() and ref.result is None:
        rclpy.spin_once(ref, timeout_sec=0.5)
        if time.time() > wall_deadline:
            ref.result = {'outcome': 'wall_timeout', 'success': False,
                          'time_s': None, 'min_clearance_m': None,
                          'nearest_prop': None, 'dist_to_target': None}
            break
    rclpy.shutdown()

    stop_node(client)
    stop_node(tracker)
    res = dict(ref.result)
    res.update({'scene': scene, 'seed': seed, 'instruction': task,
                'target': cfg['props'][target_idx]['name']})

    # Optional trace dump: same layout a raw episode uses, so
    # rover/eval_results/scene_overview_gif.py can render it directly.
    if trace_dir:
        import shutil
        d = os.path.join(trace_dir, f'{scene}_seed{seed:05d}_t{target_idx}')
        os.makedirs(d, exist_ok=True)
        shutil.copy(cfg_path, os.path.join(d, 'scene_config.json'))
        with open(os.path.join(d, 'gt_pose.jsonl'), 'w') as f:
            for row in ref.trace:
                f.write(json.dumps(row) + '\n')
        with open(os.path.join(d, 'episode.json'), 'w') as f:
            json.dump({'config': cfg, 'verdict': res,
                       'success': res['success']}, f, indent=2)
    return res, cfg


def swap_target_index(cfg):
    """Index of the same-shape/diff-color hard negative if uniquely
    describable in this scene, else None. Sampler places it at index 2."""
    props = cfg['props']
    cand = props[2]
    combo = (cand['shape'], cand['color'])
    if sum(1 for p in props if (p['shape'], p['color']) == combo) != 1:
        return None
    return 2


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--scene', required=True)
    ap.add_argument('--seed0', type=int, default=9000)
    ap.add_argument('--episodes', type=int, default=10)
    ap.add_argument('--swap', action='store_true')
    ap.add_argument('--server-host', default='127.0.0.1')
    ap.add_argument('--server-port', type=int, default=8790)
    ap.add_argument('--trace-dir', help='dump per-episode pose traces for GIF rendering')
    ap.add_argument('--send-goal', action='store_true',
                    help='PRIVILEGED: send the commanded goal pose to the policy '
                         '(pose-conditioned upper bound, not a grounding test)')
    args = ap.parse_args()

    n_ok = 0
    swap_pairs = swap_ok = 0
    for i in range(args.episodes):
        seed = args.seed0 + i
        res, cfg = run_one(args.scene, seed, args.server_host, args.server_port,
                           trace_dir=args.trace_dir, send_goal=args.send_goal)
        print(json.dumps(res), flush=True)
        n_ok += bool(res['success'])
        if args.swap:
            alt = swap_target_index(cfg)
            if alt is None:
                continue
            res2, _ = run_one(args.scene, seed, args.server_host,
                              args.server_port, target_idx=alt,
                              trace_dir=args.trace_dir, send_goal=args.send_goal)
            print(json.dumps(res2), flush=True)
            swap_pairs += 1
            a_ok = res['success'] and res['nearest_prop'] == res['target']
            b_ok = res2['success'] and res2['nearest_prop'] == res2['target']
            swap_ok += bool(a_ok and b_ok)

    print(f'SUMMARY scene={args.scene} success={n_ok}/{args.episodes}'
          + (f' swap={swap_ok}/{swap_pairs}' if args.swap else ''))


if __name__ == '__main__':
    main()
