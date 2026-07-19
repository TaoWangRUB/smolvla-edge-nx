#!/usr/bin/env python3
"""M0 exit gate part A (task 1.8): logged actions re-drive the sim.

Resets the scene to the episode's (scene, seed) — the sampler is
seed-deterministic — then publishes the logged /cmd_vel stream at its
original sim-time cadence and compares the resulting GT trajectory to the
logged one. Open-loop physics replay is not bitwise; the gate checks the
log is complete enough to reproduce the drive:

  PASS if final-position error < 0.3 m and max path deviation < 0.4 m.

Usage (sim running, workspace sourced):
  ros2 run rover_expert replay_episode.py --episode-dir <raw episode dir>
"""

import argparse
import json
import math
import subprocess
import sys
import time

import rclpy
from rclpy.node import Node
from rclpy.parameter import Parameter
from geometry_msgs.msg import TwistStamped
from nav_msgs.msg import Odometry

FINAL_ERR_M = 0.3
MAX_DEV_M = 0.4


def read_jsonl(path):
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


class Replay(Node):
    def __init__(self, cmds, logged):
        super().__init__('replay_episode', parameter_overrides=[
            Parameter('use_sim_time', Parameter.Type.BOOL, True)])
        self.cmds = cmds
        self.logged = logged          # [(t, x, y)]
        self.t_first = cmds[0]['t']
        self.t_last = cmds[-1]['t']
        self.i = 0
        self.t0 = None
        self.live = []                # [(t_rel, x, y)]
        self.result = None
        self.pose = None
        self.pub = self.create_publisher(TwistStamped, '/cmd_vel', 10)
        self.create_subscription(Odometry, '/ackermann/gt_odom', self.on_odom, 50)
        self.create_timer(0.005, self.tick)

    def on_odom(self, m):
        p = m.pose.pose.position
        t = m.header.stamp.sec + m.header.stamp.nanosec * 1e-9
        self.pose = (t, p.x, p.y)

    def tick(self):
        if self.result is not None or self.pose is None:
            return
        now = self.get_clock().now().nanoseconds * 1e-9
        if now <= 0:
            return
        if self.t0 is None:
            self.t0 = now
        rel = now - self.t0
        if self.pose is not None:
            self.live.append((rel, self.pose[1], self.pose[2]))
        while (self.i < len(self.cmds) and
               self.cmds[self.i]['t'] - self.t_first <= rel):
            c = self.cmds[self.i]
            msg = TwistStamped()
            msg.header.stamp = self.get_clock().now().to_msg()
            msg.twist.linear.x = c['v']
            msg.twist.angular.z = c['w']
            self.pub.publish(msg)
            self.i += 1
        if rel > (self.t_last - self.t_first) + 1.0:
            self.finish()

    def finish(self):
        # Logged trajectory, time-relative to its first command.
        ref = [(r['t'] - self.t_first, r['x'], r['y']) for r in self.logged]
        lx, ly = self.live[-1][1], self.live[-1][2]
        rx, ry = ref[-1][1], ref[-1][2]
        final_err = math.hypot(lx - rx, ly - ry)
        # Max deviation: for each live sample, distance to time-matched ref.
        max_dev = 0.0
        j = 0
        for t, x, y in self.live:
            while j < len(ref) - 1 and ref[j][0] < t:
                j += 1
            max_dev = max(max_dev, math.hypot(x - ref[j][1], y - ref[j][2]))
        self.result = {
            'replay_pass': final_err < FINAL_ERR_M and max_dev < MAX_DEV_M,
            'final_err_m': round(final_err, 3),
            'max_dev_m': round(max_dev, 3),
            'cmds_replayed': self.i,
            'duration_s': round(self.t_last - self.t_first, 2),
        }


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--episode-dir', required=True)
    args = ap.parse_args()
    ep = json.load(open(f'{args.episode_dir}/episode.json'))
    cfg = ep['config']
    cmds = read_jsonl(f'{args.episode_dir}/cmd.jsonl')
    logged = read_jsonl(f'{args.episode_dir}/gt_pose.jsonl')

    r = subprocess.run(['ros2', 'run', 'rover_sim', 'scene_manager.py', 'apply',
                        '--scene', cfg['scene'], '--seed', str(cfg['seed'])],
                       capture_output=True, text=True)
    if r.returncode != 0:
        print(f'scene reset failed: {r.stderr.strip()[-200:]}', file=sys.stderr)
        sys.exit(2)
    time.sleep(2.0)

    rclpy.init()
    node = Replay(cmds, logged)
    while rclpy.ok() and node.result is None:
        rclpy.spin_once(node, timeout_sec=0.5)
    print(json.dumps(node.result))
    sys.exit(0 if node.result['replay_pass'] else 1)


if __name__ == '__main__':
    main()
