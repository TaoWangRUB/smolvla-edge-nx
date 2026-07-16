"""Validate the SimEnv shim (openspec: ros2-cpp-async-deployment, task 2.3).

Two gates, strongest first:

1. **Transparency**: a local in-process env and the remote shim env, reset with the same seed
   and fed the same fixed action sequence, must produce byte-identical observations and equal
   rewards/flags at every step. If this holds the shim is a transparent transport and any
   client behaves identically through it.
2. **Spec scenario** (needs --policy-path): the same closed-loop protocol as
   `smolvla_edge.eval.eval_sim` — same stepper code, same seeds, success == reward >= 4 —
   with env.step() replaced by SimEnv.Step. Reports success rate for comparison against the
   in-process eval baseline.

Run inside the sim container against a running sim-server:

    docker compose run --rm shell python deploy/client_server/sim_env_check.py \
        --server sim-server:50052 --policy-path outputs/train/smolvla_transfer_cube/checkpoints/020000 \
        --episodes 5 --task "Pick up the cube with the right arm and transfer it to the left arm."
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import grpc
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import policy_pb2
import policy_pb2_grpc


def _unpack_observation(msg: policy_pb2.Observation) -> dict:
    img = msg.images[0]
    assert img.key == "pixels.top" and img.encoding == "raw_uint8", (img.key, img.encoding)
    top = np.frombuffer(img.data, dtype=np.uint8).reshape(tuple(img.shape))
    state = np.asarray(msg.tensors[0].data, dtype=np.float32).reshape(tuple(msg.tensors[0].shape))
    return {"pixels": {"top": top}, "agent_pos": state}


def check_transparency(stub, spec, steps: int, seed: int) -> None:
    import gymnasium as gym
    import gym_aloha  # noqa: F401

    env = gym.make(spec.env_id, obs_type=spec.obs_type, render_mode="rgb_array",
                   max_episode_steps=spec.max_steps)
    local_obs, _ = env.reset(seed=seed)
    remote_obs = _unpack_observation(
        stub.SimReset(policy_pb2.SimResetRequest(seed=seed)).observation)

    assert np.array_equal(local_obs["pixels"]["top"], remote_obs["pixels"]["top"]), \
        "reset: camera frames differ"
    assert np.allclose(local_obs["agent_pos"], remote_obs["agent_pos"], atol=0), \
        "reset: agent_pos differs"

    # deterministic, non-trivial action sequence spanning the joint range
    rng = np.random.default_rng(seed)
    mid = (env.action_space.high + env.action_space.low) / 2
    amp = (env.action_space.high - env.action_space.low) / 8
    for t in range(steps):
        action = (mid + amp * np.sin(0.05 * t + rng.uniform(0, np.pi, mid.shape))).astype(
            np.float32)
        l_obs, l_rew, l_term, l_trunc, _ = env.step(action)
        rep = stub.Step(policy_pb2.SimStepRequest(action=action.ravel().tolist()))
        r_obs = _unpack_observation(rep.observation)
        assert np.array_equal(l_obs["pixels"]["top"], r_obs["pixels"]["top"]), \
            f"step {t}: camera frames differ"
        assert np.allclose(l_obs["agent_pos"], r_obs["agent_pos"], atol=0), \
            f"step {t}: agent_pos differs"
        assert l_rew == rep.reward and l_term == rep.terminated and l_trunc == rep.truncated, \
            f"step {t}: reward/flags differ ({l_rew} vs {rep.reward})"
    env.close()
    print(f"[check] transparency: PASS ({steps} steps byte-identical, seed={seed})")


def check_policy_rollout(stub, spec, policy_path: str, task: str, episodes: int) -> None:
    from smolvla_edge.common import load_policy
    from smolvla_edge.eval import make_sim_stepper

    policy, device = load_policy(policy_path, "auto")
    stepper = make_sim_stepper(policy, policy_path, device, task or None)
    successes = 0
    for ep in range(episodes):
        obs = _unpack_observation(
            stub.SimReset(policy_pb2.SimResetRequest(seed=ep)).observation)
        policy.reset()
        ep_max_r = 0.0
        for _ in range(spec.max_steps):
            act = np.asarray(stepper(obs), dtype=np.float32)
            rep = stub.Step(policy_pb2.SimStepRequest(action=act.ravel().tolist()))
            obs = _unpack_observation(rep.observation)
            ep_max_r = max(ep_max_r, float(rep.reward))
            if rep.terminated or rep.truncated:
                break
        successes += int(ep_max_r >= 4.0)
        print(f"[check] episode {ep + 1}/{episodes}: max_reward={ep_max_r:.1f} "
              f"success={ep_max_r >= 4.0}")
    print(f"[check] policy-through-shim success rate: {successes}/{episodes}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Validate the SimEnv gRPC shim.")
    ap.add_argument("--server", default="sim-server:50052")
    ap.add_argument("--transparency-steps", type=int, default=100)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--policy-path", default=None,
                    help="run the closed-loop spec check with this checkpoint")
    ap.add_argument("--episodes", type=int, default=5)
    ap.add_argument("--task", default="")
    args = ap.parse_args()

    channel = grpc.insecure_channel(args.server)
    stub = policy_pb2_grpc.SimEnvStub(channel)
    spec = stub.Spec(policy_pb2.SimSpecRequest())
    print(f"[check] SimEnv spec: {spec.env_id} obs_type={spec.obs_type} "
          f"action_dim={spec.action_dim} max_steps={spec.max_steps}")

    check_transparency(stub, spec, args.transparency_steps, args.seed)
    if args.policy_path:
        check_policy_rollout(stub, spec, args.policy_path, args.task, args.episodes)


if __name__ == "__main__":
    main()
