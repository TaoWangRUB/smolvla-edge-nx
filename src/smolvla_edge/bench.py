"""Phase 3 core: measure latency / throughput / memory for one deployment config.

Run the same command on each tier (Titan X local, Xavier NX on-device) varying
--precision and --chunking, and append the rows to benchmarks/results/. The client/server
tier is benchmarked from deploy/client_server/client.py, which reuses these metric helpers.

    python -m smolvla_edge.bench \
        --policy-path <checkpoint> \
        --device cuda --precision fp16 --chunking on \
        --tag "nx-fp16-chunk" --out benchmarks/results/raw/nx_fp16_chunk.json
"""

from __future__ import annotations

import argparse
import json
import platform
from pathlib import Path

from .common import (
    Timer,
    load_dataset,
    load_policy,
    peak_gpu_memory_mb,
    reset_gpu_memory_stats,
)


def maybe_cast(precision: str, device: str):
    """Return an inference context manager for the requested precision.

    fp16/bf16 use torch.autocast (AMP): weights stay fp32, matmuls/convs run in reduced
    precision. This is both what real deployments do and robust — a blanket `.half()` breaks
    models that create fp32 tensors internally (ACT does). INT8 is intentionally NOT done here
    via a naive cast — real INT8 goes through TensorRT/quantization on the parts of the graph
    that convert (see deploy/ondevice/). This keeps the benchmark honest.
    """
    import contextlib

    import torch

    if precision == "fp32":
        return contextlib.nullcontext()
    if precision in ("fp16", "bf16"):
        if not device.startswith("cuda"):
            raise SystemExit(f"{precision} autocast benchmarking requires a CUDA device")
        dtype = torch.float16 if precision == "fp16" else torch.bfloat16
        return torch.autocast(device_type="cuda", dtype=dtype)
    if precision == "int8":
        raise SystemExit(
            "INT8 is not a plain cast for this VLA. Build a quantized/TensorRT engine in "
            "deploy/ondevice/ and benchmark that engine. See deploy/README.md."
        )
    raise SystemExit(f"unknown precision: {precision}")


def run(args) -> dict:
    import torch

    policy, device = load_policy(args.policy_path, args.device)
    autocast_ctx = maybe_cast(args.precision, device)
    reset_gpu_memory_stats(device)

    ds = load_dataset(args.dataset_repo_id, episodes=list(range(args.episodes)))
    timer = Timer(device=device)

    # Warmup (excluded from stats) — first calls JIT/allocate and would skew p95.
    policy.reset()
    n_total = min(len(ds), args.steps + args.warmup)
    for i in range(n_total):
        frame = ds[i]
        batch = {
            k: (v.to(device).unsqueeze(0) if isinstance(v, torch.Tensor) else v)
            for k, v in frame.items()
        }
        if i < args.warmup:
            with torch.no_grad(), autocast_ctx:
                policy.select_action(batch)
            continue
        with torch.no_grad(), autocast_ctx, timer.section("select_action"):
            policy.select_action(batch)

    stats = timer.summary().get("select_action", {})
    mean_ms = stats.get("mean_ms", float("nan"))
    result = {
        "tag": args.tag,
        "policy_path": args.policy_path,
        "device": device,
        "precision": args.precision,
        "chunking": args.chunking,
        "steps_measured": stats.get("count", 0),
        "latency_mean_ms": round(mean_ms, 3),
        "latency_p50_ms": round(stats.get("p50_ms", float("nan")), 3),
        "latency_p95_ms": round(stats.get("p95_ms", float("nan")), 3),
        "throughput_hz": round(1000.0 / mean_ms, 3) if mean_ms == mean_ms and mean_ms > 0 else None,
        "peak_gpu_mem_mb": round(peak_gpu_memory_mb(device) or 0.0, 1),
        "host": platform.node(),
        "platform": platform.platform(),
    }
    return result


def main() -> None:
    ap = argparse.ArgumentParser(description="Benchmark a SmolVLA deployment config.")
    ap.add_argument("--policy-path", required=True)
    ap.add_argument("--dataset-repo-id", default="lerobot/aloha_sim_insertion_human")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--precision", choices=["fp32", "fp16", "bf16", "int8"], default="fp16")
    ap.add_argument(
        "--chunking",
        choices=["on", "off"],
        default="on",
        help="Recorded as metadata; action chunking is configured on the policy itself.",
    )
    ap.add_argument("--episodes", type=int, default=2)
    ap.add_argument("--steps", type=int, default=200, help="Measured inference steps.")
    ap.add_argument("--warmup", type=int, default=20)
    ap.add_argument("--tag", default="run")
    ap.add_argument("--out", default=None, help="Write JSON result here (also prints).")
    args = ap.parse_args()

    result = run(args)
    print("[bench] result:")
    print(json.dumps(result, indent=2))

    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(result, indent=2))
        print(f"[bench] wrote {out}")


if __name__ == "__main__":
    main()
