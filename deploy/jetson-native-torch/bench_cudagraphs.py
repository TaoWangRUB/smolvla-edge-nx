import sys, time
sys.path.insert(0, "/repo/src")
import torch
if not hasattr(torch,"xpu"): torch.xpu=type("xpu",(),{"is_available":staticmethod(lambda:False)})()
import numpy as np
from smolvla_edge.common import load_policy, resolve_policy_path
from smolvla_edge.eval import _batchify_obs
from lerobot.envs.utils import preprocess_observation
from lerobot.policies.factory import make_pre_post_processors
CKPT="/repo/data/smolvla_transfer_cube_020000"; TASK="Pick up the cube with the right arm and transfer it to the left arm."
try: print("backends:", [b for b in torch._dynamo.list_backends() if "graph" in b or "eager" in b])
except Exception as e: print("list_backends:", e)
policy,_=load_policy(CKPT,"cuda"); policy.eval().half(); policy.config.num_steps=3
pre,post=make_pre_post_processors(policy_cfg=policy.config, pretrained_path=resolve_policy_path(CKPT), preprocessor_overrides={"device_processor":{"device":"cuda"}})
rng=np.random.default_rng(0)
obs={"pixels":{"top":rng.integers(0,256,(480,640,3),dtype=np.uint8)},"agent_pos":rng.standard_normal(14).astype(np.float32)}
def f():
    o=preprocess_observation(_batchify_obs(obs)); o["task"]=[TASK]; o=pre(o)
    with torch.no_grad(), torch.autocast("cuda",dtype=torch.float16): return policy.predict_action_chunk(o)
def bench(name,warm=3,it=10):
    for _ in range(warm):
        try: f(); 
        except Exception as e: print(f"{name}: RUN FAILED {type(e).__name__}: {str(e)[:160]}"); return
    torch.cuda.synchronize()
    ts=[]
    for _ in range(it):
        t=time.perf_counter(); f(); torch.cuda.synchronize(); ts.append((time.perf_counter()-t)*1000)
    ts.sort(); print(f"{name}: p50={ts[len(ts)//2]:.0f}ms")
bench("eager fp16")
print("=== compiling model with cudagraphs backend (first call slow) ===")
try:
    policy.model=torch.compile(policy.model, backend="cudagraphs")
    bench("torch.compile(cudagraphs)", warm=5)
except Exception as e:
    print(f"cudagraphs compile FAILED: {type(e).__name__}: {str(e)[:200]}")
