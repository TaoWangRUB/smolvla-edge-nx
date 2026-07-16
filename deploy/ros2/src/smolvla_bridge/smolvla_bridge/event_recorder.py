"""Dump the async client's /events stream to JSONL for trace comparison against Python
AsyncRunner traces (task 3.6)."""

from __future__ import annotations

import json
from pathlib import Path

import rclpy
from rclpy.node import Node

from smolvla_msgs.msg import TickEvent


class EventRecorder(Node):
    def __init__(self) -> None:
        super().__init__("event_recorder")
        self.declare_parameter("output", "/workspace/benchmarks/results/ros2/events.jsonl")
        path = Path(self.get_parameter("output").value)
        path.parent.mkdir(parents=True, exist_ok=True)
        self.fh = path.open("w")
        self.create_subscription(TickEvent, "/events", self.on_event, 100)
        self.get_logger().info(f"recording /events -> {path}")

    def on_event(self, ev: TickEvent) -> None:
        self.fh.write(json.dumps({
            "episode": ev.episode, "tick": ev.tick, "queue": ev.queue_depth,
            "queue_after": ev.queue_after, "sent": ev.sent, "filtered": ev.filtered,
            "merged": ev.merged, "idle": ev.idle,
            "proc_ms": round(ev.proc_ms, 4),
            **({"latency_s": round(ev.rtt_s, 4)} if ev.rtt_s else {}),
        }) + "\n")
        self.fh.flush()


def main() -> None:
    rclpy.init()
    node = EventRecorder()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, rclpy.executors.ExternalShutdownException):
        pass
    finally:
        node.fh.close()


if __name__ == "__main__":
    main()
