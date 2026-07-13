"""Phase 1 smoke-test: load a policy + dataset and run inference end-to-end.

Run this FIRST to validate the whole stack (model download, dataset decode, action prediction)
before spending GPU-hours on training.

Pair the policy with a dataset of the SAME embodiment: `lerobot/smolvla_base` is SO-101 (single
arm), so smoke-test it against an SO-101 dataset. A fine-tuned ALOHA checkpoint pairs with the
ALOHA dataset. Mismatched pairs fail on the input-feature dims. (To verify a policy *in the ALOHA
sim env* instead of a dataset, use `smolvla_edge.eval --mode sim`.)

    python -m smolvla_edge.infer \
        --policy-path lerobot/smolvla_base \
        --dataset-repo-id lerobot/svla_so101_pickplace \
        --episodes 2
"""

from __future__ import annotations

import argparse

from .common import Timer, load_dataset, load_policy


def main() -> None:
    ap = argparse.ArgumentParser(description="SmolVLA inference smoke-test.")
    ap.add_argument("--policy-path", default="lerobot/smolvla_base")
    # Default matches the base policy's SO-101 embodiment (see module docstring).
    ap.add_argument("--dataset-repo-id", default="lerobot/svla_so101_pickplace")
    ap.add_argument("--episodes", type=int, default=2, help="How many episodes to stream.")
    ap.add_argument("--max-frames", type=int, default=50, help="Frames per episode cap.")
    ap.add_argument("--device", default="auto")
    args = ap.parse_args()

    import torch

    print(f"[infer] loading policy: {args.policy_path}")
    policy, device = load_policy(args.policy_path, args.device)
    print(f"[infer] device: {device}")

    print(f"[infer] loading dataset: {args.dataset_repo_id}")
    ds = load_dataset(args.dataset_repo_id, episodes=list(range(args.episodes)))

    timer = Timer(device=device)
    n = 0
    policy.reset()
    for i in range(min(len(ds), args.episodes * args.max_frames)):
        frame = ds[i]
        # Move tensors to the policy device; pass through non-tensors untouched.
        batch = {
            k: (v.to(device).unsqueeze(0) if isinstance(v, torch.Tensor) else v)
            for k, v in frame.items()
        }
        with torch.no_grad(), timer.section("select_action"):
            action = policy.select_action(batch)
        n += 1
        if n <= 3:
            shape = tuple(action.shape) if hasattr(action, "shape") else type(action)
            print(f"[infer] step {n}: action shape={shape}")

    print(f"[infer] OK — ran {n} inference steps.")
    for name, stats in timer.summary().items():
        print(
            f"[infer] {name}: mean={stats['mean_ms']:.1f}ms "
            f"p50={stats['p50_ms']:.1f}ms p95={stats['p95_ms']:.1f}ms "
            f"(n={stats['count']})"
        )


if __name__ == "__main__":
    main()
