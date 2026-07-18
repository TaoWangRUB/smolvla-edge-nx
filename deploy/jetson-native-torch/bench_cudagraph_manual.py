"""Manual CUDA Graph capture of the FULL SmolVLA sample_actions forward (prefix + 3 denoise steps).

torch.compile(backend="cudagraphs") only gave ~4% because dynamo breaks on the flow-matching
loop. Manual torch.cuda.CUDAGraph capture records the raw kernel stream from the UNMODIFIED
eager code — no tracing. The workload is static-shape (fixed task string / image size / 3
steps), noise is injectable, and the inference path has no CPU syncs. The only capture-illegal
ops are `torch.tensor(...)` constant constructions (pageable H2D) inside
embed_prefix/embed_suffix/sample_actions — solved by value-keyed constant folding during
warmup+capture only.
"""
import sys, time
sys.path.insert(0, "/repo/src")
import torch
if not hasattr(torch, "xpu"):
    torch.xpu = type("xpu", (), {"is_available": staticmethod(lambda: False)})()
import numpy as np
from smolvla_edge.common import load_policy, resolve_policy_path
from smolvla_edge.eval import _batchify_obs
from lerobot.envs.utils import preprocess_observation
from lerobot.policies.factory import make_pre_post_processors

CKPT = "/repo/data/smolvla_transfer_cube_020000"
TASK = "Pick up the cube with the right arm and transfer it to the left arm."

policy, _ = load_policy(CKPT, "cuda")
policy.eval().half()
policy.config.num_steps = 3
model = policy.model
pre, post = make_pre_post_processors(
    policy_cfg=policy.config, pretrained_path=resolve_policy_path(CKPT),
    preprocessor_overrides={"device_processor": {"device": "cuda"}})

rng = np.random.default_rng(0)
O = {"pixels": {"top": rng.integers(0, 256, (480, 640, 3), dtype=np.uint8)},
     "agent_pos": rng.standard_normal(14).astype(np.float32)}

def predict():
    b = preprocess_observation(_batchify_obs(O)); b["task"] = [TASK]; b = pre(b)
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
        return policy.predict_action_chunk(b)

def bench(name, fn, warm=3, it=12):
    for _ in range(warm):
        fn()
    torch.cuda.synchronize()
    ts = []
    for _ in range(it):
        t = time.perf_counter(); fn(); torch.cuda.synchronize()
        ts.append((time.perf_counter() - t) * 1e3)
    ts.sort(); print(f"{name}: p50={ts[len(ts)//2]:.0f}ms min={ts[0]:.0f}ms", flush=True)
    return ts[len(ts)//2]

# ---- 1) eager baseline; spy one real call to harvest static input templates
p50_eager = bench("eager fp16 e2e", predict)

orig_sample = model.sample_actions
spied = {}
def spy(images, img_masks, lang_tokens, lang_masks, state, noise=None, **kw):
    if noise is None:
        noise = model.sample_noise(
            (state.shape[0], policy.config.chunk_size, policy.config.max_action_dim), state.device)
    spied["args"] = ([i.clone() for i in images], [m.clone() for m in img_masks],
                     lang_tokens.clone(), lang_masks.clone(), state.clone(), noise.clone())
    return orig_sample(images, img_masks, lang_tokens, lang_masks, state, noise=noise, **kw)
model.sample_actions = spy
predict()
model.sample_actions = orig_sample
s_imgs, s_imasks, s_lang, s_lmask, s_state, s_noise = spied["args"]
print("static shapes:", [tuple(i.shape) for i in s_imgs], "lang", tuple(s_lang.shape),
      "state", tuple(s_state.shape), "noise", tuple(s_noise.shape), flush=True)

# ---- capture-safe vision tower: the HF SmolVLM vision forward creates its patch mask on CPU
# (pageable H2D) and the NaViT embeddings loop syncs — both constant for our fixed 512x512
# full-mask input, so precompute and inject.
vlm = model.vlm_with_expert
vm = vlm.get_vlm_model().vision_model
ps = vm.patch_size
PATCH_MASK = torch.ones(
    (1, s_imgs[0].shape[-2] // ps, s_imgs[0].shape[-1] // ps),
    dtype=torch.bool, device=s_imgs[0].device)

emb_mod = vm.embeddings
rec = {}
_orig_pe_fwd = emb_mod.position_embedding.forward
def _pe_spy(ids):
    rec["ids"] = ids
    return _orig_pe_fwd(ids)
emb_mod.position_embedding.forward = _pe_spy
with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
    ref_img_emb = vlm.embed_image(s_imgs[0]).clone()
emb_mod.position_embedding.forward = _orig_pe_fwd
POS_IDS = rec["ids"].clone()

_orig_emb_fwd = emb_mod.forward
def _emb_fwd_safe(pixel_values, patch_attention_mask=None, **kw):
    pe = emb_mod.patch_embedding(pixel_values)
    return pe.flatten(2).transpose(1, 2) + emb_mod.position_embedding(POS_IDS)
emb_mod.forward = _emb_fwd_safe

_orig_vm_fwd = vm.forward
def _vm_fwd_safe(pixel_values, patch_attention_mask=None, **kw):
    return _orig_vm_fwd(
        pixel_values,
        patch_attention_mask=PATCH_MASK if patch_attention_mask is None else patch_attention_mask,
        **kw)
vm.forward = _vm_fwd_safe

with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
    new_img_emb = vlm.embed_image(s_imgs[0])
dv = (new_img_emb.float() - ref_img_emb.float()).abs().max().item()
print(f"vision-patch parity: max-abs-diff={dv:.2e}", flush=True)
assert dv < 1e-3, "vision patch changed numerics"

def graph_fn():
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16, cache_enabled=False):
        return orig_sample(s_imgs, s_imasks, s_lang, s_lmask, s_state, noise=s_noise)

ref = graph_fn().clone()  # eager reference on the SAME static inputs (parity target)
torch.cuda.synchronize()

# ---- 2) constant-fold torch.tensor (and warm the fold) so capture sees no pageable H2D
_tt = torch.tensor; _cache = {}
def _tt_cached(data, *a, **kw):
    try:
        key = (repr(data), repr(a), repr(sorted(kw.items(), key=lambda x: x[0])))
    except Exception:
        key = None
    if key is not None and key in _cache:
        return _cache[key]
    t = _tt(data, *a, **kw)
    if key is not None:
        _cache[key] = t
    return t

torch.tensor = _tt_cached
try:
    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        for _ in range(3):
            graph_fn()
    torch.cuda.current_stream().wait_stream(side)
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        static_out = graph_fn()
    print(f"CAPTURE OK (torch.tensor constants folded: {len(_cache)})", flush=True)
except Exception as e:
    print(f"CAPTURE FAILED: {type(e).__name__}: {e}", flush=True)
    raise
finally:
    torch.tensor = _tt

# ---- 3) parity: replay must reproduce eager on identical inputs
g.replay(); torch.cuda.synchronize()
d = (static_out.float() - ref.float()).abs().max().item()
cos = torch.nn.functional.cosine_similarity(
    static_out.float().flatten(), ref.float().flatten(), dim=0).item()
print(f"parity replay-vs-eager: max-abs-diff={d:.2e} cosine={cos:.6f}", flush=True)

# ---- 4) pure replay floor (model forward with zero CPU dispatch)
p50_replay = bench("graph replay only", lambda: g.replay())

# ---- 5) honest e2e: preproc + buffer copies + replay + postproc via the normal predict path
def graphed_sample(images, img_masks, lang_tokens, lang_masks, state, noise=None, **kw):
    for b, n in zip(s_imgs, images):
        b.copy_(n, non_blocking=True)
    for b, n in zip(s_imasks, img_masks):
        b.copy_(n, non_blocking=True)
    s_lang.copy_(lang_tokens, non_blocking=True)
    s_lmask.copy_(lang_masks, non_blocking=True)
    s_state.copy_(state, non_blocking=True)
    if noise is not None:
        s_noise.copy_(noise)
    else:
        s_noise.normal_()  # fresh flow-matching noise, generated outside the graph
    g.replay()
    return static_out.clone()

model.sample_actions = graphed_sample
p50_graphed = bench("graphed e2e (preproc+copies+replay+post)", predict)

# ---- 6) varying-input parity: replay must track a NEW observation through the buffer copies
O2 = {"pixels": {"top": rng.integers(0, 256, (480, 640, 3), dtype=np.uint8)},
      "agent_pos": rng.standard_normal(14).astype(np.float32)}
fixed_noise = torch.randn(1, policy.config.chunk_size, policy.config.max_action_dim,
                          generator=torch.Generator().manual_seed(7)).cuda().float()
def predict_obs(o):
    b = preprocess_observation(_batchify_obs(o)); b["task"] = [TASK]; b = pre(b)
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
        return policy.predict_action_chunk(b)
model.sample_actions = (lambda images, img_masks, lt, lm, st, noise=None, **k:
                        graphed_sample(images, img_masks, lt, lm, st, noise=fixed_noise, **k))
a_graph = predict_obs(O2).clone()
model.sample_actions = (lambda images, img_masks, lt, lm, st, noise=None, **k:
                        orig_sample(images, img_masks, lt, lm, st, noise=fixed_noise, **k))
a_eager = predict_obs(O2).clone()
d2 = (a_graph.float() - a_eager.float()).abs().max().item()
cos2 = torch.nn.functional.cosine_similarity(
    a_graph.float().flatten(), a_eager.float().flatten(), dim=0).item()
print(f"varying-input parity (new obs, same noise): max-abs-diff={d2:.2e} cosine={cos2:.6f}", flush=True)

print(f"peak GPU mem: alloc={torch.cuda.max_memory_allocated()/1e9:.2f} GB "
      f"reserved={torch.cuda.max_memory_reserved()/1e9:.2f} GB", flush=True)
print(f"SUMMARY: eager={p50_eager:.0f}ms -> graphed={p50_graphed:.0f}ms "
      f"(pure-replay floor {p50_replay:.0f}ms)", flush=True)
