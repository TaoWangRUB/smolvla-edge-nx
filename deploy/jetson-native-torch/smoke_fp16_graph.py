"""Smoke test for make_chunk_predictor(precision="fp16-graph") — the lazy CUDA-graph mode.

Runs anywhere with CUDA (dev box or NX container). Verifies: capture succeeds via the
lazy path, actions stay valid/input-sensitive, and reports eager-vs-graphed latency.
"""
import os, sys, time
sys.path.insert(0, os.environ.get("SRC", "/workspace/src"))
import torch
if not hasattr(torch, "xpu"):
    torch.xpu = type("xpu", (), {"is_available": staticmethod(lambda: False)})()
import numpy as np
from smolvla_edge.common import load_policy
from smolvla_edge.async_infer import make_chunk_predictor

CKPT = os.environ.get("CKPT", "data/smolvla_transfer_cube_020000")
TASK = "Pick up the cube with the right arm and transfer it to the left arm."

policy, _ = load_policy(CKPT, "cuda")
policy.config.num_steps = 3

# eager fp16 baseline first (separate predictor, before the graph patch)
policy.eval().half()
pred_eager = make_chunk_predictor(policy, CKPT, "cuda", TASK, precision="fp16")
rng = np.random.default_rng(0)
def obs():
    return {"pixels": {"top": rng.integers(0, 256, (480, 640, 3), dtype=np.uint8)},
            "agent_pos": rng.standard_normal(14).astype(np.float32)}

def bench(pred, name, warm=2, it=8):
    for _ in range(warm):
        pred(obs())
    torch.cuda.synchronize()
    ts = []
    for _ in range(it):
        t = time.perf_counter(); pred(obs()); torch.cuda.synchronize()
        ts.append((time.perf_counter() - t) * 1e3)
    ts.sort(); print(f"{name}: p50={ts[len(ts)//2]:.0f}ms min={ts[0]:.0f}ms", flush=True)
    return ts[len(ts)//2]

e = bench(pred_eager, "eager fp16 e2e")

pred_graph = make_chunk_predictor(policy, CKPT, "cuda", TASK, precision="fp16-graph")
t = time.perf_counter()
a0 = pred_graph(obs())  # first call: eager + lazy capture
print(f"first call (incl. capture): {time.perf_counter()-t:.1f}s", flush=True)
g = bench(pred_graph, "fp16-graph e2e")

chunks = np.stack([pred_graph(obs()) for _ in range(4)])
ok = (not np.isnan(chunks).any()) and chunks.std(0).mean() > 1e-4
print(f"chunks valid={not np.isnan(chunks).any()} input-sensitive std={chunks.std(0).mean():.4f}")
print(f"SUMMARY: eager={e:.0f}ms -> fp16-graph={g:.0f}ms  {'PASS' if ok else 'FAIL'}", flush=True)
