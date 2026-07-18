import os, sys, time
sys.path.insert(0, "/repo/src")
import numpy as np, torch
from smolvla_edge.common import load_policy, resolve_policy_path
from smolvla_edge.eval import _batchify_obs
from lerobot.envs.utils import preprocess_observation
from lerobot.policies.factory import make_pre_post_processors
from torch.profiler import profile, ProfilerActivity

CKPT="/repo/data/smolvla_transfer_cube_020000"; TASK="Pick up the cube with the right arm and transfer it to the left arm."
policy,_=load_policy(CKPT,"cuda"); policy.eval().half(); policy.config.num_steps=3
pp=resolve_policy_path(CKPT)
pre,post=make_pre_post_processors(policy_cfg=policy.config, pretrained_path=pp, preprocessor_overrides={"device_processor":{"device":"cuda"}})
rng=np.random.default_rng(0)
obs={"pixels":{"top":rng.integers(0,256,(480,640,3),dtype=np.uint8)},"agent_pos":rng.standard_normal(14).astype(np.float32)}
def prep():
    o=preprocess_observation(_batchify_obs(obs)); o["task"]=[TASK]; return pre(o)
def fwd_autocast(o):
    with torch.no_grad(), torch.autocast("cuda",dtype=torch.float16):
        return policy.predict_action_chunk(o)
def fwd_pure(o):
    o={k:(v.half() if torch.is_tensor(v) and v.is_floating_point() else v) for k,v in o.items()}
    with torch.inference_mode():
        return policy.predict_action_chunk(o)
def bench(fwd,name):
    try:
        for _ in range(3): post(fwd(prep()).float()); torch.cuda.synchronize()
    except Exception as e:
        print(f"{name}: FAILED {type(e).__name__}: {str(e)[:120]}"); return
    ts=[]
    for _ in range(8):
        o=prep(); t=time.perf_counter(); fwd(o); torch.cuda.synchronize(); ts.append((time.perf_counter()-t)*1000)
    ts.sort()
    with profile(activities=[ProfilerActivity.CPU,ProfilerActivity.CUDA]) as p:
        fwd(prep()); torch.cuda.synchronize()
    nc=sum(e.count for e in p.key_averages() if e.key in ("aten::to","aten::_to_copy","aten::copy_"))
    print(f"{name}: wall p50={ts[len(ts)//2]:.1f}ms  copy/to ops/inf={nc}")
bench(fwd_autocast,"half + autocast (current)")
bench(fwd_pure,"pure fp16 (no autocast, inference_mode)")
