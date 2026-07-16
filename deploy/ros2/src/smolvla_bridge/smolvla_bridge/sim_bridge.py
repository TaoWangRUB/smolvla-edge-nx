"""Driver-style sim bridge (openspec: ros2-cpp-async-deployment, design D3).

Maps the SimEnv gRPC service (gym-aloha, sim container) onto ROS2 topics and OWNS the timing
contract: one control tick per 50 Hz wall-timer callback, one `/observation` published per
tick, one action consumed per tick. Semantics mirror `smolvla_edge.eval.eval_sim_async`:

- After a reset the sim does NOT advance until the episode's first action arrives (the
  in-process harness blocks on the first chunk in `start_episode`; not advancing is the
  distributed equivalent).
- A tick with no fresh action on `/action` steps the sim with the previous action — an idle
  tick, the sim analogue of the robot holding pose.
- Success protocol identical to eval.py: success == max episode reward >= 4.

Publishes `smolvla_msgs/SimObservation` (image + joint state + tick in ONE message, so the
client never time-syncs topics), subscribes `std_msgs/Float32MultiArray` on `/action`, and
writes per-episode results JSON compatible with the Python harness numbers.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy

from sensor_msgs.msg import Image
from std_msgs.msg import Float32MultiArray
from smolvla_msgs.msg import SimObservation

import grpc

# generated stubs live in the repo, mounted at /workspace (compose sets PYTHONPATH, and the
# launch file passes the same default); keep an explicit fallback for bare `ros2 run`
sys.path.insert(0, "/workspace/deploy/client_server")
import policy_pb2  # noqa: E402
import policy_pb2_grpc  # noqa: E402


class SimBridge(Node):
    def __init__(self) -> None:
        super().__init__("sim_bridge")
        self.declare_parameter("server", "sim-server:50052")
        self.declare_parameter("fps", 50.0)
        self.declare_parameter("episodes", 5)
        self.declare_parameter("start_seed", 0)
        self.declare_parameter("max_steps", 400)
        self.declare_parameter("results_path", "")
        # per-episode GIF of the rendered top camera (empty = off). Frames come free: the
        # bridge already forwards the rendered RGB frame in every /observation.
        self.declare_parameter("gif_dir", "")

        self.server = self.get_parameter("server").value
        self.fps = float(self.get_parameter("fps").value)
        self.n_episodes = int(self.get_parameter("episodes").value)
        self.start_seed = int(self.get_parameter("start_seed").value)
        self.max_steps = int(self.get_parameter("max_steps").value)
        self.results_path = self.get_parameter("results_path").value
        self.gif_dir = self.get_parameter("gif_dir").value

        channel = grpc.insecure_channel(self.server)
        grpc.channel_ready_future(channel).result(timeout=30)
        self.stub = policy_pb2_grpc.SimEnvStub(channel)
        self.spec = self.stub.Spec(policy_pb2.SimSpecRequest())
        self.get_logger().info(
            f"SimEnv at {self.server}: {self.spec.env_id} action_dim={self.spec.action_dim} "
            f"max_steps={self.spec.max_steps}; running {self.n_episodes} episodes at "
            f"{self.fps} Hz")

        qos = QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE)
        self.obs_pub = self.create_publisher(SimObservation, "/observation", qos)
        self.action_sub = self.create_subscription(
            Float32MultiArray, "/action", self._on_action, qos)

        self.episode = -1
        self.tick = 0
        self.last_action: np.ndarray | None = None
        self.fresh_action: np.ndarray | None = None
        self.ep_max_r = 0.0
        self.ep_idle = 0
        self.ep_start_wall = 0.0
        self.tick_wall_times: list[float] = []
        # timing decomposition (ros2-cpp-async-deployment, 6.6 Hz root-cause):
        #   step_ms  = gRPC Step round trip to sim-server (sim physics+render+pack+network)
        #   dds_rt_ms = obs publish -> matching /action received (DDS obs out + client node
        #               processing + DDS action back) — the transport leg profile_tick can't see
        self.step_ms: list[float] = []
        self.dds_rt_ms: list[float] = []
        self.obs_pub_time = 0.0
        self.ep_frames: list[np.ndarray] = []  # rendered top-cam frames for the episode GIF
        self.results: list[dict] = []
        self.done = False

        self._reset_episode()
        self.timer = self.create_timer(1.0 / self.fps, self._on_tick)

    # -- helpers ---------------------------------------------------------------------------

    def _on_action(self, msg: Float32MultiArray) -> None:
        # DDS round trip: this action is the client's answer to the obs we last published.
        if self.obs_pub_time:
            self.dds_rt_ms.append((time.monotonic() - self.obs_pub_time) * 1e3)
        self.fresh_action = np.asarray(msg.data, dtype=np.float32)

    def _publish_obs(self, obs_msg: policy_pb2.Observation, tick: int) -> None:
        img_pb = obs_msg.images[0]
        h, w, c = img_pb.shape
        img = Image()
        img.header.stamp = self.get_clock().now().to_msg()
        img.height, img.width = h, w
        img.encoding = "rgb8"
        img.step = w * c
        img.data = img_pb.data
        out = SimObservation()
        out.tick = tick
        out.episode = self.episode
        out.image_top = img
        out.agent_pos = list(obs_msg.tensors[0].data)
        self.obs_pub.publish(out)
        self.obs_pub_time = time.monotonic()

    def _record_frame(self, obs_msg: policy_pb2.Observation) -> None:
        if not self.gif_dir:
            return
        img = obs_msg.images[0]
        h, w, c = img.shape
        self.ep_frames.append(
            np.frombuffer(img.data, dtype=np.uint8).reshape(h, w, c).copy())

    def _write_gif(self, seed: int, success: bool) -> str | None:
        if not self.gif_dir or not self.ep_frames:
            return None
        from PIL import Image as PILImage

        out_dir = Path(self.gif_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        tag = "success" if success else "fail"
        path = out_dir / f"ep{self.episode:02d}_seed{seed:02d}_{tag}.gif"
        # stride to ~12.5 fps and cap the long side at 360 px so 50 gifs stay small
        stride = 2
        frames = self.ep_frames[::stride]
        pil = []
        for f in frames:
            im = PILImage.fromarray(f)
            if max(im.size) > 360:
                s = 360 / max(im.size)
                im = im.resize((int(im.width * s), int(im.height * s)))
            pil.append(im.convert("P", palette=PILImage.ADAPTIVE))
        # ~40 ms measured tick cadence * stride -> roughly real-time playback
        pil[0].save(path, save_all=True, append_images=pil[1:],
                    duration=stride * 40, loop=0, optimize=True)
        return str(path)

    def _reset_episode(self) -> None:
        self.episode += 1
        seed = self.start_seed + self.episode
        rep = self.stub.SimReset(policy_pb2.SimResetRequest(seed=seed))
        self.tick = 0
        self.last_action = None
        self.fresh_action = None
        self.boot_obs = rep.observation  # re-published each tick until the first action
        self.ep_max_r = 0.0
        self.ep_idle = 0
        self.ep_start_wall = time.monotonic()
        self.tick_wall_times = []
        self.ep_frames = []
        self._record_frame(rep.observation)
        self._publish_obs(rep.observation, rep.tick)
        self.get_logger().info(f"episode {self.episode + 1}/{self.n_episodes} (seed={seed})")

    def _finish_episode(self, success: bool) -> None:
        wall = time.monotonic() - self.ep_start_wall
        jitter = np.diff(self.tick_wall_times) if len(self.tick_wall_times) > 2 else np.array([0.0])
        seed = self.start_seed + self.episode
        gif = self._write_gif(seed, success)
        result = {
            "episode": self.episode, "seed": seed,
            "success": bool(success), "max_reward": float(self.ep_max_r),
            "ticks": int(self.tick), "idle_ticks": int(self.ep_idle),
            "wall_s": round(wall, 3),
            "tick_ms_p50": round(float(np.percentile(jitter, 50)) * 1e3, 3),
            "tick_ms_p95": round(float(np.percentile(jitter, 95)) * 1e3, 3),
        }
        if gif:
            result["gif"] = gif
        self.results.append(result)
        self.get_logger().info(
            f"episode {self.episode + 1}: success={success} max_reward={self.ep_max_r:.1f} "
            f"ticks={self.tick} idle={self.ep_idle} "
            f"tick p50/p95 {result['tick_ms_p50']}/{result['tick_ms_p95']} ms")
        if self.episode + 1 >= self.n_episodes:
            self._write_results()
            self.done = True
            self.timer.cancel()
            self.get_logger().info("all episodes done")
            rclpy.shutdown()
        else:
            self._reset_episode()

    def _write_results(self) -> None:
        n = len(self.results)
        succ = sum(r["success"] for r in self.results)

        def _stats(xs: list[float]) -> dict:
            if not xs:
                return {"n": 0}
            a = np.asarray(xs)
            return {"n": int(a.size), "p50": round(float(np.percentile(a, 50)), 3),
                    "p95": round(float(np.percentile(a, 95)), 3),
                    "mean": round(float(a.mean()), 3)}

        summary = {
            "mode": "ros2-stage",
            "env_id": self.spec.env_id,
            "episodes": n, "successes": succ,
            "success_rate": succ / n if n else float("nan"),
            "mean_idle_ticks": float(np.mean([r["idle_ticks"] for r in self.results])),
            "fps": self.fps,
            # timing decomposition of the control tick (all ms):
            "timing_ms": {
                "grpc_step": _stats(self.step_ms),   # sim-server round trip (physics+render+net)
                "dds_roundtrip": _stats(self.dds_rt_ms),  # obs->action DDS + client node
            },
            "per_episode": self.results,
            "note": "ROS2 bridge closed-loop; success = gym-aloha reward>=4",
        }
        print(json.dumps({k: v for k, v in summary.items() if k != "per_episode"}, indent=2))
        if self.results_path:
            p = Path(self.results_path)
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(json.dumps(summary, indent=2))
            self.get_logger().info(f"results -> {self.results_path}")

    # -- one control tick (design D3) --------------------------------------------------------

    def _on_tick(self) -> None:
        if self.done:
            return
        # boot phase: no action ever arrived this episode -> the sim does not advance (the
        # in-process harness blocks on the first chunk; same thing), but the observation is
        # RE-PUBLISHED so the client gets a tick to notice the landed chunk and act — without
        # this, client (acts on obs) and bridge (steps on action) deadlock at tick 0
        if self.fresh_action is None and self.last_action is None:
            self._publish_obs(self.boot_obs, 0)
            return

        if self.fresh_action is None:
            # Grace window before declaring idle: the client answers an observation in ~2 ms
            # when its queue is non-empty, but this (possibly overdue) timer callback can fire
            # before that answer arrives. Idle must mean "the client had nothing" (AsyncRunner
            # queue-empty semantics), not "we outraced the client" — so give the current
            # observation one tick period to be answered before stepping with a held pose.
            if time.monotonic() - self.obs_pub_time < (1.0 / self.fps):
                return
            action = self.last_action  # true idle tick: hold previous commanded pose
            self.ep_idle += 1
        else:
            action, self.fresh_action = self.fresh_action, None
        self.last_action = action
        self.tick_wall_times.append(time.monotonic())

        _t_step = time.monotonic()
        rep = self.stub.Step(policy_pb2.SimStepRequest(action=action.ravel().tolist()))
        self.step_ms.append((time.monotonic() - _t_step) * 1e3)
        self.tick = rep.tick
        self.ep_max_r = max(self.ep_max_r, rep.reward)
        self._record_frame(rep.observation)
        if rep.terminated or rep.truncated or self.tick >= self.max_steps:
            self._finish_episode(self.ep_max_r >= 4.0)
        else:
            self._publish_obs(rep.observation, rep.tick)


def main() -> None:
    rclpy.init()
    node = SimBridge()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, rclpy.executors.ExternalShutdownException):
        pass


if __name__ == "__main__":
    main()
