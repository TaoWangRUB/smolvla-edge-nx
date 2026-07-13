"""Phase 1 deliverable: evaluate a checkpoint and report a success-rate number.

Two eval styles are supported, in order of fidelity:

1. **Sim rollout** (default; the number you quote): the policy drives a gym-aloha MuJoCo env and
   success is the env's own success flag (reward >= 4). No robot required. Needs `gym-aloha`.
2. **Open-loop replay** (fallback, no sim install): step through held-out episodes and measure
   action-prediction agreement (MSE / threshold accuracy) against logged actions. Cheap, but a
   proxy — label it as such in the writeup.

    # closed-loop success rate in simulation (the deliverable):
    python -m smolvla_edge.eval \
        --policy-path outputs/train/smolvla_aloha/checkpoints/last \
        --mode sim --env-id gym_aloha/AlohaInsertion-v0 --episodes 20

    # open-loop proxy (no sim):
    python -m smolvla_edge.eval --policy-path <ckpt> --mode replay \
        --dataset-repo-id lerobot/aloha_sim_insertion_human
"""

from __future__ import annotations

import argparse

from .common import load_dataset, load_policy


def eval_replay(policy, device, ds, threshold: float, max_frames: int) -> dict:
    """Open-loop replay proxy: compare predicted vs. logged actions on held-out frames."""
    import torch

    abs_errors: list[float] = []
    within: list[float] = []
    policy.reset()
    n = min(len(ds), max_frames)
    for i in range(n):
        frame = ds[i]
        batch = {
            k: (v.to(device).unsqueeze(0) if isinstance(v, torch.Tensor) else v)
            for k, v in frame.items()
        }
        gt = frame.get("action")
        if gt is None:
            continue
        with torch.no_grad():
            pred = policy.select_action(batch).squeeze(0).cpu()
        err = (pred - gt.cpu()).abs()
        abs_errors.append(err.mean().item())
        within.append((err < threshold).float().mean().item())

    mae = sum(abs_errors) / len(abs_errors) if abs_errors else float("nan")
    acc = sum(within) / len(within) if within else float("nan")
    return {
        "mode": "replay",
        "frames": len(abs_errors),
        "action_mae": mae,
        f"within_{threshold}_acc": acc,
        "note": "open-loop proxy, NOT a true task success rate",
    }


# Observation-key mapping from gym-aloha -> the SmolVLA policy batch.
#
# VERIFY THESE against your checkpoint on the dev box: the exact keys must match the
# checkpoint's `policy.config.input_features`. For LeRobot ALOHA sim the convention is a top
# camera at `observation.images.top` and the 14-D joint state at `observation.state`. gym-aloha
# returns cameras under `pixels` (dict) or `images`, and state under `agent_pos` (obs_type
# "pixels_agent_pos") or `qpos`. This helper is defensive about both.
IMAGE_KEY = "observation.images.top"
STATE_KEY = "observation.state"


def _aloha_obs_to_batch(obs: dict, task: str, device: str) -> dict:
    """Map one gym-aloha observation dict to a batched SmolVLA input dict."""
    import numpy as np
    import torch

    cams = obs.get("pixels", obs.get("images"))
    top = cams.get("top") if isinstance(cams, dict) else cams  # HxWx3 uint8
    if isinstance(cams, dict) and "top" not in cams:
        top = next(iter(cams.values()))
    img = torch.from_numpy(np.asarray(top)).permute(2, 0, 1).float() / 255.0  # CHW in [0,1]

    state_np = obs.get("agent_pos", obs.get("qpos"))
    state = torch.from_numpy(np.asarray(state_np, dtype=np.float32))

    return {
        IMAGE_KEY: img.unsqueeze(0).to(device),
        STATE_KEY: state.unsqueeze(0).to(device),
        "task": [task],  # SmolVLA is language-conditioned; one instruction per batch element
    }


def eval_sim(
    policy,
    device,
    env_id: str,
    task: str,
    n_episodes: int,
    max_steps: int,
    obs_type: str = "pixels_agent_pos",
) -> dict:
    """Closed-loop rollout in gym-aloha; the env's success flag is the quoted success rate.

    gym-aloha gives a max reward of 4 on task success (peg insertion / cube transfer), so an
    episode is a success iff it reaches reward >= 4. This is the true (closed-loop) success rate,
    unlike the open-loop `eval_replay` proxy.

    Requires `pip install gym-aloha`. The obs->batch key mapping (see `_aloha_obs_to_batch`) is a
    best-effort default; verify it against `policy.config.input_features` on the dev box.
    """
    import gymnasium as gym
    import gym_aloha  # noqa: F401  # registers the gym_aloha/* env ids
    import torch

    env = gym.make(env_id, obs_type=obs_type, render_mode="rgb_array")
    successes = 0
    max_rewards: list[float] = []
    try:
        for ep in range(n_episodes):
            obs, _ = env.reset(seed=ep)
            policy.reset()
            ep_max_r = 0.0
            for _ in range(max_steps):
                batch = _aloha_obs_to_batch(obs, task, device)
                with torch.no_grad():
                    action = policy.select_action(batch)
                act_np = action.squeeze(0).float().cpu().numpy()
                obs, reward, terminated, truncated, _ = env.step(act_np)
                ep_max_r = max(ep_max_r, float(reward))
                if terminated or truncated:
                    break
            successes += int(ep_max_r >= 4.0)
            max_rewards.append(ep_max_r)
            print(f"[eval] episode {ep + 1}/{n_episodes}: max_reward={ep_max_r:.1f} "
                  f"success={ep_max_r >= 4.0}")
    finally:
        env.close()

    return {
        "mode": "sim",
        "env_id": env_id,
        "episodes": n_episodes,
        "successes": successes,
        "success_rate": successes / n_episodes if n_episodes else float("nan"),
        "mean_max_reward": sum(max_rewards) / len(max_rewards) if max_rewards else float("nan"),
        "note": "closed-loop gym-aloha success rate (reward>=4)",
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Evaluate a SmolVLA checkpoint.")
    ap.add_argument("--policy-path", required=True)
    ap.add_argument("--dataset-repo-id", default="lerobot/aloha_sim_insertion_human")
    ap.add_argument("--mode", choices=["sim", "replay"], default="sim")
    ap.add_argument("--episodes", type=int, default=10)
    ap.add_argument("--max-frames", type=int, default=2000, help="(replay) frame cap")
    ap.add_argument("--max-steps", type=int, default=400, help="(sim) steps per episode")
    ap.add_argument("--threshold", type=float, default=0.05, help="(replay) per-dim tol")
    ap.add_argument("--env-id", default="gym_aloha/AlohaInsertion-v0", help="(sim) gym env id")
    ap.add_argument("--task", default="insert the peg into the socket", help="(sim) instruction")
    ap.add_argument("--obs-type", default="pixels_agent_pos", help="(sim) gym-aloha obs_type")
    ap.add_argument("--device", default="auto")
    args = ap.parse_args()

    policy, device = load_policy(args.policy_path, args.device)

    if args.mode == "sim":
        result = eval_sim(
            policy, device, args.env_id, args.task, args.episodes, args.max_steps, args.obs_type
        )
    else:
        ds = load_dataset(args.dataset_repo_id, episodes=list(range(args.episodes)))
        result = eval_replay(policy, device, ds, args.threshold, args.max_frames)

    print("[eval] result:")
    for k, v in result.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
