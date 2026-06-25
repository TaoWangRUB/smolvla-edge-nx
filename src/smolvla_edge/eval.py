"""Phase 1 deliverable: evaluate a checkpoint and report a success-rate number.

Two eval styles are supported, in order of fidelity:

1. **Sim rollout** (preferred when a matching gym env is available): the policy drives the
   environment and success is the env's own success flag. This is the number you quote.
2. **Open-loop replay** (always available, no robot/sim): step through held-out episodes and
   measure action-prediction agreement (MSE / threshold accuracy) against logged actions.
   This validates the checkpoint cheaply but is a proxy, not a true success rate — label it
   as such in the writeup.

    python -m smolvla_edge.eval \
        --policy-path outputs/train/smolvla_so101/checkpoints/last \
        --dataset-repo-id lerobot/svla_so101_pickplace \
        --mode replay
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


def eval_sim(policy, device, env_id: str, n_episodes: int, max_steps: int) -> dict:
    """Closed-loop rollout in a gym env; quote env success flag as the success rate.

    Left as a thin stub: wire to the gym env that matches your dataset's embodiment
    (e.g. a LeRobot/gym-aloha-style env). Kept separate so the replay path always works.
    """
    raise NotImplementedError(
        "Sim rollout eval not wired yet — install the matching gym env and implement the "
        "step loop here, then report env success over n_episodes. Use --mode replay until "
        "then."
    )


def main() -> None:
    ap = argparse.ArgumentParser(description="Evaluate a SmolVLA checkpoint.")
    ap.add_argument("--policy-path", required=True)
    ap.add_argument("--dataset-repo-id", default="lerobot/svla_so101_pickplace")
    ap.add_argument("--mode", choices=["replay", "sim"], default="replay")
    ap.add_argument("--episodes", type=int, default=10)
    ap.add_argument("--max-frames", type=int, default=2000, help="(replay) frame cap")
    ap.add_argument("--max-steps", type=int, default=400, help="(sim) steps per episode")
    ap.add_argument("--threshold", type=float, default=0.05, help="(replay) per-dim tol")
    ap.add_argument("--env-id", default=None, help="(sim) gym env id")
    ap.add_argument("--device", default="auto")
    args = ap.parse_args()

    policy, device = load_policy(args.policy_path, args.device)

    if args.mode == "sim":
        if not args.env_id:
            ap.error("--mode sim requires --env-id")
        result = eval_sim(policy, device, args.env_id, args.episodes, args.max_steps)
    else:
        ds = load_dataset(args.dataset_repo_id, episodes=list(range(args.episodes)))
        result = eval_replay(policy, device, ds, args.threshold, args.max_frames)

    print("[eval] result:")
    for k, v in result.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
