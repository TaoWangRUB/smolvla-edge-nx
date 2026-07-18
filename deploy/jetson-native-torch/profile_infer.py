"""Profile one SmolVLA chunk inference (fp16, 3 flow steps) — CPU-dispatch vs GPU-compute split.

Run the same script on the Xavier NX (native from-source torch) and the host A2000 (official
torch) to see where each spends its time. Reads CKPT/SRC from env.

    CKPT=<path> python3 profile_infer.py
"""
import os, sys, time
sys.path.insert(0, os.environ.get("SRC", "/repo/src"))
import numpy as np, torch
from smolvla_edge.common import load_policy
from smolvla_edge.async_infer import make_chunk_predictor
from torch.profiler import profile, ProfilerActivity

CKPT = os.environ.get("CKPT", "/repo/data/smolvla_transfer_cube_020000")
TASK = "Pick up the cube with the right arm and transfer it to the left arm."

policy, _ = load_policy(CKPT, "cuda")
policy.eval().half()
policy.config.num_steps = 3
predict = make_chunk_predictor(policy, CKPT, "cuda", TASK, precision="fp16")

rng = np.random.default_rng(0)
obs = {"pixels": {"top": rng.integers(0, 256, (480, 640, 3), dtype=np.uint8)},
       "agent_pos": rng.standard_normal(14).astype(np.float32)}

for _ in range(3):
    predict(obs)
torch.cuda.synchronize()

ts = []
for _ in range(10):
    t = time.perf_counter(); predict(obs); torch.cuda.synchronize()
    ts.append((time.perf_counter() - t) * 1000)
ts.sort()
print(f"device={torch.cuda.get_device_name(0)} torch={torch.__version__}")
print(f"wall latency ms: mean={sum(ts)/len(ts):.1f} p50={ts[len(ts)//2]:.1f} min={ts[0]:.1f}")

with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
    for _ in range(3):
        predict(obs)
    torch.cuda.synchronize()
print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=12))
ka = prof.key_averages()
_cu = sum(getattr(e, "self_device_time_total", getattr(e, "self_cuda_time_total", 0)) for e in ka) / 1000 / 3
_cp = sum(e.self_cpu_time_total for e in ka) / 1000 / 3
print(f"per-inference: self-CUDA={_cu:.1f}ms  self-CPU={_cp:.1f}ms")
