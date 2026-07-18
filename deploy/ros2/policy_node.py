"""All-ROS2 policy server node — runs ON the Xavier NX (smolvla-jetson:humble image).

The DDS replacement for the gRPC PolicyServer: subscribes PolicyRequest, runs the
fp16-graph chunk predictor (lazy CUDA-graph capture on the first request), publishes
PolicyChunk. One predictor per task string, exactly like deploy/client_server/server.py.

    source /opt/ros/humble/setup.bash && source /opt/msgs_ws/setup.bash
    SRC=/repo/src CKPT=/repo/data/smolvla_transfer_cube_020000 \
        python3 /repo/deploy/ros2/policy_node.py

Parameters (ROS): precision (default fp16-graph), flow_steps (default 3), device (cuda).
"""
from __future__ import annotations

import os
import sys
import time

sys.path.insert(0, os.environ.get("SRC", "/repo/src"))

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy

from smolvla_msgs.msg import PolicyChunk, PolicyRequest

CKPT = os.environ.get("CKPT", "/repo/data/smolvla_transfer_cube_020000")


class PolicyNode(Node):
    def __init__(self):
        super().__init__("smolvla_policy")
        self.declare_parameter("precision", "fp16-graph")
        self.declare_parameter("flow_steps", 3)
        self.declare_parameter("device", "cuda")
        precision = self.get_parameter("precision").value
        flow_steps = int(self.get_parameter("flow_steps").value)
        device = self.get_parameter("device").value

        from smolvla_edge.common import load_policy

        self.policy, self.device = load_policy(CKPT, device)
        if precision.startswith("fp16"):
            self.policy = self.policy.half()
        self.policy.config.num_steps = flow_steps
        self.policy.reset()
        self.precision = precision
        self._predictors: dict = {}

        qos = QoSProfile(depth=10, reliability=ReliabilityPolicy.RELIABLE)
        self.pub = self.create_publisher(PolicyChunk, "/policy/chunk", qos)
        self.sub = self.create_subscription(PolicyRequest, "/policy/request",
                                            self.on_request, qos)
        self.get_logger().info(
            f"policy ready: {CKPT} on {self.device} ({precision}, fs{flow_steps})")

    def _predictor(self, task: str):
        if task not in self._predictors:
            from smolvla_edge.async_infer import make_chunk_predictor

            self._predictors[task] = make_chunk_predictor(
                self.policy, CKPT, self.device, task or None, precision=self.precision)
        return self._predictors[task]

    def on_request(self, req: PolicyRequest):
        recv = time.time()
        img = np.frombuffer(bytes(req.image_top.data), dtype=np.uint8).reshape(
            req.image_top.height, req.image_top.width, 3)
        obs = {"pixels": {"top": img},
               "agent_pos": np.asarray(req.agent_pos, dtype=np.float32)}
        chunk = np.asarray(self._predictor(req.task)(obs), dtype=np.float32)
        out = PolicyChunk()
        out.request_id = req.request_id
        out.rows, out.cols = int(chunk.shape[0]), int(chunk.shape[1])
        out.data = chunk.reshape(-1).tolist()
        out.server_recv_ts = recv
        out.server_send_ts = time.time()
        self.pub.publish(out)


def main():
    rclpy.init()
    node = PolicyNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
