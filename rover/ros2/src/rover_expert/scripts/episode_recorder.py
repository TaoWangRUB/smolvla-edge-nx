#!/usr/bin/env python3
"""Raw episode recorder (task 1.7, sim side — no lerobot dependency).

Writes one episode directory:
  frames/%06d.jpg      RGB at the camera rate (15 Hz), native 1280x800, q90
  frames.jsonl         {"i": idx, "t": sim time of the image header}
  gt_pose.jsonl        {"t", "x", "y", "yaw", "vx", "vy"}   50 Hz GT odom
  state.jsonl          {"t", "speed", "yaw_rate", "steering"}  50 Hz
  cmd.jsonl            {"t", "v", "w"}   every expert command
  camera_info.json     K matrix + resolution (locked OV9782 nominals)

Timestamps are sim-time header stamps throughout (clock discipline).
episode.json (config + verdict + flags) is written by run_episode.py.
Stop with SIGINT/SIGTERM; JSONL streams are line-buffered.
"""

import argparse
import json
import math
import os
import signal
import sys

import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.parameter import Parameter
from geometry_msgs.msg import TwistStamped
from nav_msgs.msg import Odometry
from sensor_msgs.msg import CameraInfo, Image
from std_msgs.msg import Float32MultiArray


def stamp_s(header):
    return header.stamp.sec + header.stamp.nanosec * 1e-9


class Recorder(Node):
    def __init__(self, out):
        super().__init__('episode_recorder', parameter_overrides=[
            Parameter('use_sim_time', Parameter.Type.BOOL, True)])
        self.out = out
        os.makedirs(f'{out}/frames', exist_ok=True)
        self.f_frames = open(f'{out}/frames.jsonl', 'w', buffering=1)
        self.f_pose = open(f'{out}/gt_pose.jsonl', 'w', buffering=1)
        self.f_state = open(f'{out}/state.jsonl', 'w', buffering=1)
        self.f_cmd = open(f'{out}/cmd.jsonl', 'w', buffering=1)
        self.i = 0
        self.have_info = False
        self.last_state_t = None

        self.create_subscription(Image, '/vla_camera/image', self.on_image, 10)
        self.create_subscription(CameraInfo, '/vla_camera/camera_info',
                                 self.on_info, 1)
        self.create_subscription(Odometry, '/ackermann/gt_odom', self.on_odom, 50)
        self.create_subscription(Float32MultiArray, '/observation/state',
                                 self.on_state, 50)
        self.create_subscription(TwistStamped, '/cmd_vel', self.on_cmd, 50)

    def on_image(self, m):
        img = np.frombuffer(bytes(m.data), np.uint8).reshape(m.height, m.width, 3)
        cv2.imwrite(f'{self.out}/frames/{self.i:06d}.jpg',
                    cv2.cvtColor(img, cv2.COLOR_RGB2BGR),
                    [cv2.IMWRITE_JPEG_QUALITY, 90])
        self.f_frames.write(json.dumps({'i': self.i, 't': stamp_s(m.header)}) + '\n')
        self.i += 1

    def on_info(self, m):
        if self.have_info:
            return
        self.have_info = True
        with open(f'{self.out}/camera_info.json', 'w') as f:
            json.dump({'width': m.width, 'height': m.height,
                       'k': list(m.k), 'd': list(m.d),
                       'frame_id': m.header.frame_id}, f, indent=2)

    def on_odom(self, m):
        q = m.pose.pose.orientation
        yaw = math.atan2(2.0 * (q.w * q.z + q.x * q.y),
                         1.0 - 2.0 * (q.y * q.y + q.z * q.z))
        p = m.pose.pose.position
        v = m.twist.twist.linear
        self.f_pose.write(json.dumps({
            't': stamp_s(m.header), 'x': round(p.x, 5), 'y': round(p.y, 5),
            'yaw': round(yaw, 5), 'vx': round(v.x, 5), 'vy': round(v.y, 5),
        }) + '\n')

    def on_state(self, m):
        # Float32MultiArray has no header; stamp with node sim clock.
        t = self.get_clock().now().nanoseconds * 1e-9
        if self.last_state_t == t:
            return
        self.last_state_t = t
        s = list(m.data)
        self.f_state.write(json.dumps({
            't': round(t, 5), 'speed': round(s[0], 5),
            'yaw_rate': round(s[1], 5), 'steering': round(s[2], 5),
        }) + '\n')

    def on_cmd(self, m):
        self.f_cmd.write(json.dumps({
            't': stamp_s(m.header), 'v': round(m.twist.linear.x, 5),
            'w': round(m.twist.angular.z, 5),
        }) + '\n')


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--out', required=True)
    args = ap.parse_args()

    rclpy.init()
    node = Recorder(args.out)
    stop = {'flag': False}

    def handler(signum, frame):
        stop['flag'] = True

    signal.signal(signal.SIGINT, handler)
    signal.signal(signal.SIGTERM, handler)
    while rclpy.ok() and not stop['flag']:
        rclpy.spin_once(node, timeout_sec=0.2)
    print(f'recorded {node.i} frames', file=sys.stderr)


if __name__ == '__main__':
    main()
