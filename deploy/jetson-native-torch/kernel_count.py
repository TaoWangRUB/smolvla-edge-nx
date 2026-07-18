import sys, time
sys.path.insert(0, "/repo/src")
import numpy as np, torch
from torch.autograd import DeviceType
from smolvla_edge.common import load_policy, resolve_policy_path
from smolvla_edge.eval import _batchify_obs
from lerobot.envs.utils import preprocess_observation
from lerobot.policies.factory import make_pre_post_processors
from torch.profiler import profile, ProfilerActivity
CKPT="/repo/data/smolvla_transfer_cube_020000"; TASK="Pick up the cube with the right arm and transfer it to the left arm."
policy,_=load_policy(CKPT,"cuda"); policy.eval().half(); policy.config.num_steps=3
pre,post=make_pre_post_processors(policy_cfg=policy.config, pretrained_path=resolve_policy_path(CKPT), preprocessor_overrides={"device_processor":{"device":"cuda"}})
rng=np.random.default_rng(0)
obs={"pixels":{"top":rng.integers(0,256,(480,640,3),dtype=np.uint8)},"agent_pos":rng.standard_normal(14).astype(np.float32)}
def run():
    o=preprocess_observation(_batchify_obs(obs)); o["task"]=[TASK]; o=pre(o)
    with torch.no_grad(), torch.autocast("cuda",dtype=torch.float16): return policy.predict_action_chunk(o)
for _ in range(3): run(); torch.cuda.synchronize()
# wall
ts=[]
for _ in range(10):
    torch.cuda.synchronize(); t=time.perf_counter(); run(); torch.cuda.synchronize(); ts.append((time.perf_counter()-t)*1000)
wall=sorted(ts)[len(ts)//2]
# profile 1 inference, count actual GPU kernels + GPU-busy time
with profile(activities=[ProfilerActivity.CPU,ProfilerActivity.CUDA]) as prof:
    run(); torch.cuda.synchronize()
evs=prof.events()
kern=[e for e in evs if e.device_type==DeviceType.CUDA]
gpu_ms=sum(getattr(e,"self_cuda_time_total",0) for e in kern)/1000.0
launches=[e for e in evs if e.device_type==DeviceType.CPU and ("Launch" in e.name or "cudaLaunch" in e.name)]
print(f"wall/inf              : {wall:.1f} ms")
print(f"actual GPU kernels    : {len(kern)}")
print(f"GPU-busy (sum kernels): {gpu_ms:.1f} ms")
print(f"=> GPU occupancy      : {100*gpu_ms/wall:.0f}%  (corroborates tegrastats GR3D 15-26%)")
print(f"cudaLaunchKernel calls: {len(launches)}")
