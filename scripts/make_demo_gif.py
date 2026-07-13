"""Assemble a demo GIF — either a live policy ROLLOUT in gym-aloha, or a dataset replay.

**rollout** (default; the real demo): run a policy closed-loop in the sim via the same
machinery as `smolvla_edge.eval`, render every step, and keep the first SUCCESSFUL episode
(retrying seeds). This shows the policy actually doing the task, not a recorded human demo.

    python scripts/make_demo_gif.py --mode rollout \
        --policy-path lerobot/act_aloha_sim_transfer_cube_human \
        --env-id gym_aloha/AlohaTransferCube-v0 --out benchmarks/results/demo.gif

**replay** (fallback; no policy needed): read frames straight out of a LeRobot dataset.
Use --max-frames large enough to cover the whole episode (ALOHA sim episodes are ~500
frames @ 50 fps), and --stride to keep the GIF small and ~real-time.

    python scripts/make_demo_gif.py --mode replay \
        --dataset-repo-id lerobot/aloha_sim_insertion_human --max-frames 500 --stride 3
"""

from __future__ import annotations

import argparse
from pathlib import Path


def find_image_key(frame: dict) -> str:
    for k, v in frame.items():
        if "image" in k.lower() and hasattr(v, "ndim") and v.ndim == 3:
            return k
    raise SystemExit(f"No image-like key found in frame keys: {list(frame)}")


def to_uint8_hwc(t):
    import numpy as np

    arr = t.detach().cpu().numpy() if hasattr(t, "detach") else np.asarray(t)
    if arr.shape[0] in (1, 3) and arr.ndim == 3:  # CHW -> HWC
        arr = arr.transpose(1, 2, 0)
    if arr.dtype != "uint8":
        arr = (arr.clip(0, 1) * 255).astype("uint8") if arr.max() <= 1.0 else arr.astype("uint8")
    return arr


def frames_replay(args) -> list:
    """Read frames straight from the dataset (human demo replay)."""
    from smolvla_edge.common import load_dataset

    ds = load_dataset(args.dataset_repo_id, episodes=list(range(args.episodes)))
    img_key = find_image_key(ds[0])
    n = min(len(ds), args.max_frames)
    print(f"[gif] replay: {n}/{len(ds)} frames from {args.dataset_repo_id} (key={img_key})")
    return [to_uint8_hwc(ds[i][img_key]) for i in range(0, n, args.stride)]


def frames_rollout(args) -> list:
    """Run the policy closed-loop in gym-aloha and render; keep the first successful episode."""
    import gymnasium as gym
    import gym_aloha  # noqa: F401
    import torch

    from smolvla_edge.common import load_policy
    from smolvla_edge.eval import _aloha_obs_to_batch, _load_normalizers

    policy, device = load_policy(args.policy_path, args.device)
    normalize_obs, unnormalize_action = _load_normalizers(policy, args.policy_path, device)
    env = gym.make(args.env_id, obs_type="pixels_agent_pos", render_mode="rgb_array")

    best_frames: list = []
    best_reward = -1.0
    try:
        for seed in range(args.max_tries):
            obs, _ = env.reset(seed=seed)
            policy.reset()
            frames, ep_max_r = [], 0.0
            for _ in range(args.max_steps):
                frames.append(env.render())
                batch = normalize_obs(_aloha_obs_to_batch(obs, device, args.task or None))
                with torch.no_grad():
                    action = unnormalize_action(policy.select_action(batch))
                obs, reward, terminated, truncated, _ = env.step(
                    action.squeeze(0).float().cpu().numpy()
                )
                ep_max_r = max(ep_max_r, float(reward))
                if terminated or truncated:
                    break
            print(f"[gif] rollout seed {seed}: max_reward={ep_max_r} "
                  f"({'SUCCESS' if ep_max_r >= 4.0 else 'no success'})")
            if ep_max_r > best_reward:
                best_reward, best_frames = ep_max_r, frames
            if ep_max_r >= 4.0:
                break
    finally:
        env.close()

    if best_reward < 4.0:
        print(f"[gif] WARNING: no successful episode in {args.max_tries} tries; "
              f"using the best one (max_reward={best_reward})")
    return [to_uint8_hwc(f) for f in best_frames[:: args.stride]]


def main() -> None:
    ap = argparse.ArgumentParser(description="Build a demo GIF (policy rollout or dataset replay).")
    ap.add_argument("--mode", choices=["rollout", "replay"], default="rollout")
    # rollout args
    ap.add_argument("--policy-path", default="lerobot/act_aloha_sim_transfer_cube_human")
    ap.add_argument("--env-id", default="gym_aloha/AlohaTransferCube-v0")
    ap.add_argument("--task", default="", help="language instruction (SmolVLA); empty for ACT")
    ap.add_argument("--max-steps", type=int, default=400, help="(rollout) steps per episode")
    ap.add_argument("--max-tries", type=int, default=5, help="(rollout) seeds to try for a success")
    ap.add_argument("--device", default="auto")
    # replay args
    ap.add_argument("--dataset-repo-id", default="lerobot/aloha_sim_insertion_human")
    ap.add_argument("--episodes", type=int, default=1)
    ap.add_argument("--max-frames", type=int, default=500)
    # shared
    ap.add_argument("--stride", type=int, default=3, help="keep every Nth frame (size/speed)")
    ap.add_argument("--fps", type=int, default=17, help="GIF playback fps (50/stride ~= real-time)")
    ap.add_argument("--out", default="benchmarks/results/demo.gif")
    args = ap.parse_args()

    import imageio.v2 as imageio

    frames = frames_rollout(args) if args.mode == "rollout" else frames_replay(args)
    if not frames:
        raise SystemExit("[gif] no frames produced")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    imageio.mimsave(out, frames, fps=args.fps, loop=0)
    print(f"[gif] wrote {out} ({len(frames)} frames @ {args.fps}fps, mode={args.mode})")


if __name__ == "__main__":
    main()
