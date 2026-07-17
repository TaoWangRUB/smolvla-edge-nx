"""On-device latency benchmark for the exported SmolVLA graph under onnxruntime-gpu.

The exported graph bakes in the instruction tokens + all normalization (see
models/onnx/*.meta.json), so this benchmark needs ONLY numpy + onnxruntime — no torch, no
lerobot, no tokenizer. It feeds correctly-shaped random inputs (image in [0,1], plus the
state and flow-matching noise) and reports per-inference latency. onnxruntime's run() is
synchronous w.r.t. the CUDA/TensorRT EP, so wall time is the true device latency.

For a real observation stream, replace _random_feed() with replayed dataset frames — the
shapes are identical (image[1,3,512,512], state[1,14], noise[1,50,32] -> action[1,50,14]).

    python3 deploy/onnx/bench_ort.py --model models/onnx/smolvla_transfer_cube.onnx \
        --provider cuda --iters 50
"""

from __future__ import annotations

import argparse
import statistics
import time

import numpy as np
import onnxruntime as ort

PROVIDERS = {
    "cuda": "CUDAExecutionProvider",
    "tensorrt": "TensorrtExecutionProvider",
    "cpu": "CPUExecutionProvider",
}


def _random_feed(sess: ort.InferenceSession, seed: int = 0) -> dict:
    rng = np.random.default_rng(seed)
    feed = {}
    for i in sess.get_inputs():
        # All three inputs are statically shaped in the export; guard dynamic dims anyway.
        shape = [d if isinstance(d, int) and d > 0 else 1 for d in i.shape]
        feed[i.name] = rng.random(size=shape, dtype=np.float32)
    return feed


def main() -> int:
    ap = argparse.ArgumentParser(description="onnxruntime-gpu latency benchmark for SmolVLA.")
    ap.add_argument("--model", required=True)
    ap.add_argument("--provider", default="cuda", choices=list(PROVIDERS))
    ap.add_argument("--iters", type=int, default=50)
    ap.add_argument("--warmup", type=int, default=5)
    args = ap.parse_args()

    ep = PROVIDERS[args.provider]
    available = ort.get_available_providers()
    if ep not in available:
        raise SystemExit(f"[bench] {ep} not available; onnxruntime sees {available}")

    # Always keep CPUExecutionProvider as the tail fallback: ORT 1.15's CUDA/TRT EPs lack a
    # few kernels this graph uses (e.g. ArgMin), and without CPU in the list those nodes raise
    # NOT_IMPLEMENTED instead of falling back. The heavy compute still runs on the requested EP.
    provider_list = [ep] if ep == "CPUExecutionProvider" else [ep, "CPUExecutionProvider"]
    sess = ort.InferenceSession(args.model, providers=provider_list)
    print(f"[bench] model={args.model}")
    print(f"[bench] requested={ep} active={sess.get_providers()}")
    for i in sess.get_inputs():
        print(f"[bench]   input {i.name}: {i.shape} {i.type}")

    feed = _random_feed(sess)
    for _ in range(args.warmup):
        sess.run(None, feed)

    times_ms = []
    for _ in range(args.iters):
        t0 = time.perf_counter()
        sess.run(None, feed)
        times_ms.append((time.perf_counter() - t0) * 1e3)

    times_ms.sort()
    p = lambda q: times_ms[min(int(len(times_ms) * q), len(times_ms) - 1)]
    print(
        f"[bench] {ep} | {args.iters} iters (ms): "
        f"mean={statistics.mean(times_ms):.1f} "
        f"p50={p(0.50):.1f} p90={p(0.90):.1f} "
        f"min={times_ms[0]:.1f} max={times_ms[-1]:.1f}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
