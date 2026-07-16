"""gym-aloha behind the SimEnv gRPC service (openspec: ros2-cpp-async-deployment, design D2).

Runs inside the smolvla-edge:sim container — the only place the mujoco-2.3.7-matched env
exists — and makes it reachable from the ROS2 container. The env protocol mirrors
`smolvla_edge.eval.eval_sim*` exactly: `gym.make(env_id, obs_type=..., render_mode="rgb_array",
max_episode_steps=max_steps)`, `reset(seed=...)`, success == reward >= 4. Observation packing
follows remote.py's conventions (Image "pixels.top" raw_uint8 HWC, Tensor "agent_pos") so any
client of the Policy service can feed chunks straight from SimEnv replies.

Single-env, single-episode-at-a-time by design: a lock serializes Reset/Step, matching the
one-action-per-tick discretization the bridge enforces (design D3).

    python deploy/client_server/sim_server.py --env-id gym_aloha/AlohaTransferCube-v0 --port 50052
"""

from __future__ import annotations

import argparse
import sys
import threading
from concurrent import futures
from pathlib import Path

import grpc
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))  # policy_pb2* live next to this file
import policy_pb2
import policy_pb2_grpc


def _pack_observation(obs: dict) -> policy_pb2.Observation:
    top = np.ascontiguousarray(obs["pixels"]["top"], dtype=np.uint8)
    state = np.asarray(obs["agent_pos"], dtype=np.float32)
    return policy_pb2.Observation(
        images=[policy_pb2.Image(key="pixels.top", data=top.tobytes(),
                                 shape=list(top.shape), encoding="raw_uint8")],
        tensors=[policy_pb2.Tensor(key="agent_pos", data=state.ravel().tolist(),
                                   shape=list(state.shape))],
    )


class SimEnvServicer(policy_pb2_grpc.SimEnvServicer):
    def __init__(self, env_id: str, obs_type: str, max_steps: int):
        import gymnasium as gym
        import gym_aloha  # noqa: F401  # registers the gym_aloha/* env ids

        self.env_id, self.obs_type, self.max_steps = env_id, obs_type, max_steps
        # override the registered 400-step TimeLimit, same as eval_sim_async: under the
        # fixed-time protocol idle ticks eat into the budget, so the cap must be ours
        self.env = gym.make(env_id, obs_type=obs_type, render_mode="rgb_array",
                            max_episode_steps=max_steps)
        self.action_dim = int(np.prod(self.env.action_space.shape))
        self.tick = 0
        self.lock = threading.Lock()
        print(f"[sim_server] {env_id} (obs_type={obs_type}, action_dim={self.action_dim}, "
              f"max_steps={max_steps})")

    def Spec(self, request, context):
        return policy_pb2.SimSpecReply(env_id=self.env_id, obs_type=self.obs_type,
                                       action_dim=self.action_dim, max_steps=self.max_steps)

    def SimReset(self, request, context):
        with self.lock:
            seed = request.seed if request.HasField("seed") else None
            obs, _ = self.env.reset(seed=seed)
            self.tick = 0
            return policy_pb2.SimResetReply(observation=_pack_observation(obs), tick=self.tick)

    def Step(self, request, context):
        with self.lock:
            action = np.asarray(request.action, dtype=np.float32)
            if action.size != self.action_dim:
                context.abort(grpc.StatusCode.INVALID_ARGUMENT,
                              f"expected action of size {self.action_dim}, got {action.size}")
            obs, reward, terminated, truncated, _ = self.env.step(action)
            self.tick += 1
            return policy_pb2.SimStepReply(
                observation=_pack_observation(obs), reward=float(reward),
                terminated=bool(terminated), truncated=bool(truncated), tick=self.tick,
            )


def main() -> None:
    ap = argparse.ArgumentParser(description="gym-aloha behind the SimEnv gRPC service.")
    ap.add_argument("--env-id", default="gym_aloha/AlohaTransferCube-v0")
    ap.add_argument("--obs-type", default="pixels_agent_pos")
    ap.add_argument("--max-steps", type=int, default=400)
    ap.add_argument("--port", type=int, default=50052)
    args = ap.parse_args()

    server = grpc.server(futures.ThreadPoolExecutor(max_workers=4))
    policy_pb2_grpc.add_SimEnvServicer_to_server(
        SimEnvServicer(args.env_id, args.obs_type, args.max_steps), server)
    server.add_insecure_port(f"[::]:{args.port}")
    server.start()
    print(f"[sim_server] listening on :{args.port}")
    server.wait_for_termination()


if __name__ == "__main__":
    main()
