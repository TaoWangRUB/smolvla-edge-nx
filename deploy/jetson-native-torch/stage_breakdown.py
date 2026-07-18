import sys, time, statistics as st
sys.path.insert(0, "/repo/src")
import numpy as np, torch
from smolvla_edge.common import load_policy, resolve_policy_path
from smolvla_edge.eval import _batchify_obs
from lerobot.envs.utils import preprocess_observation
from lerobot.policies.factory import make_pre_post_processors
CKPT="/repo/data/smolvla_transfer_cube_020000"; TASK="Pick up the cube with the right arm and transfer it to the left arm."
policy,_=load_policy(CKPT,"cuda"); policy.eval().half(); policy.config.num_steps=3
pre,post=make_pre_post_processors(policy_cfg=policy.config, pretrained_path=resolve_policy_path(CKPT), preprocessor_overrides={"device_processor":{"device":"cuda"}})
rng=np.random.default_rng(0)
obs={"pixels":{"top":rng.integers(0,256,(480,640,3),dtype=np.uint8)},"agent_pos":rng.standard_normal(14).astype(np.float32)}
proc=[];model=[];pst=[]
for _ in range(3):  # warmup
    o=preprocess_observation(_batchify_obs(obs)); o["task"]=[TASK]; o=pre(o)
    with torch.no_grad(), torch.autocast("cuda",dtype=torch.float16): c=policy.predict_action_chunk(o)
    post(c.float()); torch.cuda.synchronize()
print("RUNNING_INFERENCE_LOOP")  # marker for tegrastats window
for _ in range(20):
    torch.cuda.synchronize(); t0=time.perf_counter()
    o=preprocess_observation(_batchify_obs(obs)); o["task"]=[TASK]; o=pre(o)
    torch.cuda.synchronize(); t1=time.perf_counter()
    with torch.no_grad(), torch.autocast("cuda",dtype=torch.float16): c=policy.predict_action_chunk(o)
    torch.cuda.synchronize(); t2=time.perf_counter()
    r=post(c.float()).squeeze(0).float().cpu().numpy(); t3=time.perf_counter()
    proc.append((t1-t0)*1000); model.append((t2-t1)*1000); pst.append((t3-t2)*1000)
print(f"STAGE breakdown (median of 20): processor={st.median(proc):.1f}ms  MODEL_FWD={st.median(model):.1f}ms  post={st.median(pst):.1f}ms  total={st.median(proc)+st.median(model)+st.median(pst):.1f}ms")
