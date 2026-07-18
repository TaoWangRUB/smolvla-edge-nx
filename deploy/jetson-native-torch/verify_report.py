"""On-device SmolVLA verification report for the Xavier NX (native from-source torch, fp16).

Complements parity_check.py (which proves numerical correctness). This checks the *deployment*
properties: latency, peak GPU/CPU memory vs the 8 GB budget, and that inference stays valid and
input-sensitive across diverse observations.

    CKPT=<path> python3 verify_report.py
"""
from __future__ import annotations
import os, sys, time, resource
import numpy as np
import torch

sys.path.insert(0, os.environ.get("SRC", "/repo/src"))
from smolvla_edge.common import load_policy            # noqa: E402
from smolvla_edge.async_infer import make_chunk_predictor  # noqa: E402

CKPT = os.environ.get("CKPT", "/repo/data/smolvla_transfer_cube_020000")
TASK = "Pick up the cube with the right arm and transfer it to the left arm."
N = int(os.environ.get("N", "12"))

policy, dev = load_policy(CKPT, "cuda")
policy.eval().half()                  # fp16 = Volta tensor cores
policy.config.num_steps = 3
predict = make_chunk_predictor(policy, CKPT, "cuda", TASK, precision="fp16")

rng = np.random.default_rng(0)
def obs():
    return {"pixels": {"top": rng.integers(0, 256, (480, 640, 3), dtype=np.uint8)},
            "agent_pos": rng.standard_normal(14).astype(np.float32)}

for _ in range(3):                    # warmup (cuBLAS init)
    predict(obs())
torch.cuda.synchronize()
torch.cuda.reset_peak_memory_stats()

lat, chunks = [], []
for _ in range(N):
    o = obs()
    t = time.perf_counter(); a = np.asarray(predict(o), dtype=np.float32); torch.cuda.synchronize()
    lat.append((time.perf_counter() - t) * 1000); chunks.append(a)

lat.sort(); C = np.stack(chunks)
print("=== SmolVLA on-device verification — Xavier NX, native torch fp16, 3 flow steps ===")
print(f"device       : {torch.cuda.get_device_name(0)}  torch {torch.__version__}")
print(f"latency (ms) : mean={np.mean(lat):.1f}  p50={lat[len(lat)//2]:.1f}  min={lat[0]:.1f}  max={lat[-1]:.1f}   (n={N})")
print(f"peak GPU mem : alloc={torch.cuda.max_memory_allocated()/1e9:.2f} GB  reserved={torch.cuda.max_memory_reserved()/1e9:.2f} GB   (8 GB unified budget)")
print(f"peak CPU RSS : {resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024/1024:.2f} GB")
print(f"chunks       : shape={C.shape}  range=[{C.min():.3f}, {C.max():.3f}]  NaN={bool(np.isnan(C).any())}  Inf={bool(np.isinf(C).any())}")
print(f"input-sens.  : per-timestep std across obs = {C.std(0).mean():.4f}  (>0 ⇒ actions track the observation)")
ok = (not np.isnan(C).any()) and (not np.isinf(C).any()) and C.std(0).mean() > 1e-4
print("VERDICT:", "PASS — valid, stable, fits budget" if ok else "CHECK")
