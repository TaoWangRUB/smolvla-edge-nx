#!/usr/bin/env python3
"""Async policy-loop client (task 2.6): observation -> policy server -> chunk.

Runs in the sim (Jazzy) container. Keeps the latest camera frame + state,
and in a request loop (sequential, one in flight — AsyncRunner semantics):
JPEG-encodes the frame, sends {state, instruction, capture_t} + JPEG over a
stdlib TCP socket to the policy server (rover/runtime/policy_server.py,
torch side), and publishes the returned chunk on /waypoint_chunk with the
ORIGINAL capture_t so the tracker's latency compensation is exact.

The transport is the client/server lineage (D8); on the NX deployment the
chunk source becomes the all-ROS2 policy node, topics unchanged.
"""

import json
import socket
import struct
import threading

import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.parameter import Parameter
from sensor_msgs.msg import Image
from std_msgs.msg import Float32MultiArray


def request(host, port, header: dict, blob: bytes, timeout=10.0):
    h = json.dumps(header).encode()
    with socket.create_connection((host, port), timeout=timeout) as s:
        s.sendall(struct.pack('!II', len(h), len(blob)) + h + blob)
        n = struct.unpack('!I', recv_exact(s, 4))[0]
        return json.loads(recv_exact(s, n).decode())


def recv_exact(s, n):
    buf = b''
    while len(buf) < n:
        d = s.recv(n - len(buf))
        if not d:
            raise ConnectionError('short read')
        buf += d
    return buf


class ChunkClient(Node):
    def __init__(self):
        super().__init__('chunk_client', parameter_overrides=[
            Parameter('use_sim_time', Parameter.Type.BOOL, True)])
        self.declare_parameter('server_host', '127.0.0.1')
        self.declare_parameter('server_port', 8790)
        self.declare_parameter('instruction', 'drive to the red barrel')
        self.host = self.get_parameter('server_host').value
        self.port = int(self.get_parameter('server_port').value)

        self.lock = threading.Lock()
        self.frame = None            # (capture_t, jpeg bytes)
        self.state = [0.0, 0.0, 0.0]
        self.pub = self.create_publisher(Float32MultiArray, '/waypoint_chunk', 10)
        self.create_subscription(Image, '/vla_camera/image', self.on_image, 1)
        self.create_subscription(Float32MultiArray, '/observation/state',
                                 self.on_state, 10)
        threading.Thread(target=self.loop, daemon=True).start()

    def on_image(self, m):
        t = m.header.stamp.sec + m.header.stamp.nanosec * 1e-9
        img = np.frombuffer(bytes(m.data), np.uint8).reshape(m.height, m.width, 3)
        ok, jpg = cv2.imencode('.jpg', cv2.cvtColor(img, cv2.COLOR_RGB2BGR),
                               [cv2.IMWRITE_JPEG_QUALITY, 92])
        if ok:
            with self.lock:
                self.frame = (t, jpg.tobytes())

    def on_state(self, m):
        with self.lock:
            self.state = list(m.data)

    def loop(self):
        import time
        while rclpy.ok():
            with self.lock:
                frame = self.frame
                state = list(self.state)
            if frame is None:
                time.sleep(0.1)
                continue
            capture_t, jpg = frame
            try:
                rep = request(self.host, self.port, {
                    'state': state,
                    'instruction': self.get_parameter('instruction').value,
                    'capture_t': capture_t,
                }, jpg)
            except (OSError, ConnectionError) as e:
                self.get_logger().warn(f'policy server: {e}', throttle_duration_sec=5.0)
                time.sleep(0.5)
                continue
            chunk = rep['chunk']
            out = Float32MultiArray()
            out.data = [float(capture_t), float(len(chunk))] + [
                float(v) for wp in chunk for v in wp]
            self.pub.publish(out)


def main():
    rclpy.init()
    rclpy.spin(ChunkClient())


if __name__ == '__main__':
    main()
