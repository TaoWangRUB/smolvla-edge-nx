"""Assemble a short demo GIF from replayed/held-out episode frames.

No physical robot or live sim required — this reads images straight out of a LeRobot dataset
(optionally overlaying the policy's predicted action) and writes a GIF for the README.

    python scripts/make_demo_gif.py \
        --dataset-repo-id lerobot/svla_so101_pickplace \
        --episodes 1 --out docs/assets/demo.gif
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


def main() -> None:
    ap = argparse.ArgumentParser(description="Build a demo GIF from dataset frames.")
    ap.add_argument("--dataset-repo-id", default="lerobot/svla_so101_pickplace")
    ap.add_argument("--episodes", type=int, default=1)
    ap.add_argument("--max-frames", type=int, default=120)
    ap.add_argument("--fps", type=int, default=15)
    ap.add_argument("--out", default="docs/assets/demo.gif")
    args = ap.parse_args()

    import imageio.v2 as imageio

    from smolvla_edge.common import load_dataset

    ds = load_dataset(args.dataset_repo_id, episodes=list(range(args.episodes)))
    img_key = find_image_key(ds[0])
    n = min(len(ds), args.max_frames)
    frames = [to_uint8_hwc(ds[i][img_key]) for i in range(n)]

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    imageio.mimsave(out, frames, fps=args.fps)
    print(f"[gif] wrote {out} ({len(frames)} frames @ {args.fps}fps, key={img_key})")


if __name__ == "__main__":
    main()
