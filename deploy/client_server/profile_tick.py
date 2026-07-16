"""Break the control tick into its cost segments (ros2-cpp-async-deployment, 6.6 Hz analysis).

Runs in two places:
  - sim container:  in-process env costs + local gRPC round trip
        docker compose run --rm shell python deploy/client_server/profile_tick.py --local
  - ros2 container: cross-container gRPC round trip (adds the docker network hop)
        docker compose run --rm ros2 python3 deploy/client_server/profile_tick.py --server sim-server:50052
"""

from __future__ import annotations

import argparse
import statistics
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import policy_pb2
import policy_pb2_grpc


def pct(xs, p):
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(len(xs) * p / 100))]


def report(name, xs):
    print(f"{name:38s} p50={statistics.median(xs)*1e3:7.1f} ms  p95={pct(xs, 95)*1e3:7.1f} ms")


def profile_local(steps: int) -> None:
    import gymnasium as gym
    import gym_aloha  # noqa: F401
    from sim_server import _pack_observation

    env = gym.make("gym_aloha/AlohaTransferCube-v0", obs_type="pixels_agent_pos",
                   render_mode="rgb_array", max_episode_steps=10_000)
    obs, _ = env.reset(seed=0)
    action = env.action_space.sample() * 0
    t_step, t_pack = [], []
    for _ in range(steps):
        t0 = time.perf_counter()
        obs, *_ = env.step(action)
        t1 = time.perf_counter()
        msg = _pack_observation(obs)
        t2 = time.perf_counter()
        t_step.append(t1 - t0)
        t_pack.append(t2 - t1)
    env.close()
    report("env.step (mujoco + EGL render)", t_step)
    report("proto pack (tobytes + message)", t_pack)
    print(f"observation payload: {len(msg.images[0].data) / 1024:.0f} KiB")


def profile_rpc(server: str, steps: int) -> None:
    import grpc

    channel = grpc.insecure_channel(server)
    grpc.channel_ready_future(channel).result(timeout=15)
    stub = policy_pb2_grpc.SimEnvStub(channel)
    spec = stub.Spec(policy_pb2.SimSpecRequest())
    stub.SimReset(policy_pb2.SimResetRequest(seed=0))
    action = [0.0] * spec.action_dim
    t_rpc = []
    for _ in range(steps):
        t0 = time.perf_counter()
        stub.Step(policy_pb2.SimStepRequest(action=action))
        t_rpc.append(time.perf_counter() - t0)
    report(f"SimEnv.Step round trip ({server})", t_rpc)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--local", action="store_true", help="in-process env + pack timings")
    ap.add_argument("--server", default=None, help="also time Step RPC against this address")
    ap.add_argument("--steps", type=int, default=200)
    args = ap.parse_args()
    if args.local:
        profile_local(args.steps)
    if args.server:
        profile_rpc(args.server, args.steps)


if __name__ == "__main__":
    main()
