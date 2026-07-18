import sys, time, copy
sys.path.insert(0, "/repo/src")
import torch
if not hasattr(torch,"xpu"): torch.xpu=type("xpu",(),{"is_available":staticmethod(lambda:False)})()
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
def bench(p,name):
    def f():
        o=preprocess_observation(_batchify_obs(obs)); o["task"]=[TASK]; o=pre(o)
        with torch.no_grad(), torch.autocast("cuda",dtype=torch.float16): return p.predict_action_chunk(o)
    for _ in range(3): f(); torch.cuda.synchronize()
    ts=[]
    for _ in range(10):
        t=time.perf_counter(); f(); torch.cuda.synchronize(); ts.append((time.perf_counter()-t)*1000)
    ts.sort(); print(f"{name}: p50={ts[len(ts)//2]:.0f}ms")
policy450.eval().half(); policy450.config.num_steps=3
bench(policy450,"500M backbone, 16 layers (baseline)")
for N in (8,4):
    cfg=copy.deepcopy(policy450.config); cfg.num_vlm_layers=N
    try:
        p=SmolVLAPolicy(cfg).eval().half().to("cuda"); p.config.num_steps=3
        bench(p,f"500M backbone, {N} layers")
        del p; torch.cuda.empty_cache()
    except Exception as e: print(f"{N} layers FAILED: {type(e).__name__}: {str(e)[:150]}")
