import os, sys, time, copy
sys.path.insert(0, "/repo/src")
import torch
if not hasattr(torch, "xpu"):  # lerobot 0.4.4 expects torch>=2.4
    torch.xpu = type("xpu", (), {"is_available": staticmethod(lambda: False)})()
import numpy as np
from smolvla_edge.common import load_policy, resolve_policy_path
from smolvla_edge.eval import _batchify_obs
from lerobot.envs.utils import preprocess_observation
from lerobot.policies.factory import make_pre_post_processors
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
CKPT="/repo/data/smolvla_transfer_cube_020000"; TASK="Pick up the cube with the right arm and transfer it to the left arm."
policy450,_=load_policy(CKPT,"cuda")
pre,post=make_pre_post_processors(policy_cfg=policy450.config, pretrained_path=resolve_policy_path(CKPT), preprocessor_overrides={"device_processor":{"device":"cuda"}})
rng=np.random.default_rng(0)
obs={"pixels":{"top":rng.integers(0,256,(480,640,3),dtype=np.uint8)},"agent_pos":rng.standard_normal(14).astype(np.float32)}
def make_pred(p):
    def f():
        o=preprocess_observation(_batchify_obs(obs)); o["task"]=[TASK]; o=pre(o)
        with torch.no_grad(), torch.autocast("cuda",dtype=torch.float16): return p.predict_action_chunk(o)
    return f
def bench(p,name):
    f=make_pred(p)
    for _ in range(3): f(); torch.cuda.synchronize()
    ts=[]
    for _ in range(10):
        t=time.perf_counter(); f(); torch.cuda.synchronize(); ts.append((time.perf_counter()-t)*1000)
    ts.sort(); print(f"{name}: p50={ts[len(ts)//2]:.1f}ms min={ts[0]:.1f}ms  mem={torch.cuda.max_memory_allocated()/1e9:.2f}GB")
policy450.eval().half(); policy450.config.num_steps=3
bench(policy450,"450M (SmolVLM2-500M, 16 layers)")
print("=== building 256M variant ===")
cfg=copy.deepcopy(policy450.config)
cfg.vlm_model_name="HuggingFaceTB/SmolVLM2-256M-Video-Instruct"
try:
    p256=SmolVLAPolicy(cfg).eval().half().to("cuda"); p256.config.num_steps=3
    torch.cuda.reset_peak_memory_stats()
    bench(p256,"256M (SmolVLM2-256M)")
except Exception as e:
    print(f"256M build FAILED: {type(e).__name__}: {str(e)[:200]}")
