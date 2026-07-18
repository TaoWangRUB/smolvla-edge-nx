"""450M vs 256M backbone under manual CUDA Graph capture on the Xavier NX.

Eager, the 256M backbone did NOT help (591 vs ~600 ms) because execution was launch-bound —
latency tracked op count, not FLOPs. Graph replay removes all CPU dispatch, so latency should
now track real GPU compute; this measures whether width finally matters once the dispatch
bottleneck is gone. The 256M variant runs with random expert weights (latency-only; no 256M
fine-tune exists).
"""
import copy, gc, sys, time
sys.path.insert(0, "/repo/src")
import torch
if not hasattr(torch, "xpu"):
    torch.xpu = type("xpu", (), {"is_available": staticmethod(lambda: False)})()
import numpy as np
from smolvla_edge.common import load_policy, resolve_policy_path
from smolvla_edge.eval import _batchify_obs
from lerobot.envs.utils import preprocess_observation
from lerobot.policies.factory import make_pre_post_processors
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy

CKPT = "/repo/data/smolvla_transfer_cube_020000"
TASK = "Pick up the cube with the right arm and transfer it to the left arm."

policy450, _ = load_policy(CKPT, "cuda")
pre, _post = make_pre_post_processors(
    policy_cfg=policy450.config, pretrained_path=resolve_policy_path(CKPT),
    preprocessor_overrides={"device_processor": {"device": "cuda"}})
cfg256 = copy.deepcopy(policy450.config)
rng = np.random.default_rng(0)
O = {"pixels": {"top": rng.integers(0, 256, (480, 640, 3), dtype=np.uint8)},
     "agent_pos": rng.standard_normal(14).astype(np.float32)}

def bench(name, fn, warm=3, it=12):
    for _ in range(warm):
        fn()
    torch.cuda.synchronize()
    ts = []
    for _ in range(it):
        t = time.perf_counter(); fn(); torch.cuda.synchronize()
        ts.append((time.perf_counter() - t) * 1e3)
    ts.sort(); print(f"  {name}: p50={ts[len(ts)//2]:.0f}ms min={ts[0]:.0f}ms", flush=True)
    return ts[len(ts)//2]

def capture_flow(policy, label):
    """Full manual-capture flow (same recipe as bench_cudagraph_manual.py) for one policy."""
    print(f"=== {label} ===", flush=True)
    policy.eval().half(); policy.config.num_steps = 3
    model = policy.model

    def predict():
        b = preprocess_observation(_batchify_obs(O)); b["task"] = [TASK]; b = pre(b)
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
            return policy.predict_action_chunk(b)

    e = bench("eager fp16 e2e", predict)

    orig_sample = model.sample_actions
    spied = {}
    def spy(images, img_masks, lang_tokens, lang_masks, state, noise=None, **kw):
        if noise is None:
            noise = model.sample_noise((state.shape[0], policy.config.chunk_size,
                                        policy.config.max_action_dim), state.device)
        spied["args"] = ([i.clone() for i in images], [m.clone() for m in img_masks],
                         lang_tokens.clone(), lang_masks.clone(), state.clone(), noise.clone())
        return orig_sample(images, img_masks, lang_tokens, lang_masks, state, noise=noise, **kw)
    model.sample_actions = spy
    predict()
    model.sample_actions = orig_sample
    s_imgs, s_imasks, s_lang, s_lmask, s_state, s_noise = spied["args"]

    # capture-safe vision tower: inject constant patch mask + precomputed position ids
    vm = model.vlm_with_expert.get_vlm_model().vision_model
    ps = vm.patch_size
    patch_mask = torch.ones((1, s_imgs[0].shape[-2] // ps, s_imgs[0].shape[-1] // ps),
                            dtype=torch.bool, device=s_imgs[0].device)
    emb_mod = vm.embeddings
    rec = {}
    _orig_pe = emb_mod.position_embedding.forward
    emb_mod.position_embedding.forward = \
        lambda ids: (rec.__setitem__("ids", ids), _orig_pe(ids))[1]
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
        ref_img = model.vlm_with_expert.embed_image(s_imgs[0]).clone()
    emb_mod.position_embedding.forward = _orig_pe
    pos_ids = rec["ids"].clone()
    def _emb_safe(pixel_values, patch_attention_mask=None, **kw):
        pe2 = emb_mod.patch_embedding(pixel_values)
        return pe2.flatten(2).transpose(1, 2) + emb_mod.position_embedding(pos_ids)
    emb_mod.forward = _emb_safe
    _orig_vm = vm.forward
    vm.forward = (lambda pixel_values, patch_attention_mask=None, **kw:
                  _orig_vm(pixel_values,
                           patch_attention_mask=patch_mask if patch_attention_mask is None
                           else patch_attention_mask, **kw))
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
        new_img = model.vlm_with_expert.embed_image(s_imgs[0])
    dv = (new_img.float() - ref_img.float()).abs().max().item()
    assert dv < 1e-3, f"vision patch changed numerics: {dv}"

    def graph_fn():
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16, cache_enabled=False):
            return orig_sample(s_imgs, s_imasks, s_lang, s_lmask, s_state, noise=s_noise)

    ref = graph_fn().clone(); torch.cuda.synchronize()

    # constant-fold torch.tensor during warmup+capture (kills pageable-H2D of constants)
    _tt = torch.tensor; _cache = {}
    def _tt_c(data, *a, **kw):
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
    torch.tensor = _tt_c
    try:
        side = torch.cuda.Stream(); side.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side):
            for _ in range(3):
                graph_fn()
        torch.cuda.current_stream().wait_stream(side); torch.cuda.synchronize()
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            static_out = graph_fn()
    finally:
        torch.tensor = _tt
    g.replay(); torch.cuda.synchronize()
    d = (static_out.float() - ref.float()).abs().max().item()
    print(f"  capture OK, replay-vs-eager parity max-abs-diff={d:.2e}", flush=True)

    r = bench("graph replay only", lambda: g.replay())

    def graphed_sample(images, img_masks, lang_tokens, lang_masks, state, noise=None, **kw):
        for b2, n in zip(s_imgs, images):
            b2.copy_(n, non_blocking=True)
        for b2, n in zip(s_imasks, img_masks):
            b2.copy_(n, non_blocking=True)
        s_lang.copy_(lang_tokens, non_blocking=True)
        s_lmask.copy_(lang_masks, non_blocking=True)
        s_state.copy_(state, non_blocking=True)
        if noise is not None:
            s_noise.copy_(noise)
        else:
            s_noise.normal_()
        g.replay()
        return static_out.clone()
    model.sample_actions = graphed_sample
    ge = bench("graphed e2e", predict)
    model.sample_actions = orig_sample
    vm.forward = _orig_vm
    emb_mod.forward = type(emb_mod).forward.__get__(emb_mod)
    print(f"  peak GPU mem: alloc={torch.cuda.max_memory_allocated()/1e9:.2f} GB", flush=True)
    return e, r, ge

r450 = capture_flow(policy450, "450M fine-tuned (SmolVLM2-500M backbone, 16 VLM layers)")
del policy450
gc.collect(); torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()

print("=== building 256M variant (SmolVLM2-256M backbone, random expert weights) ===", flush=True)
cfg256.vlm_model_name = "HuggingFaceTB/SmolVLM2-256M-Video-Instruct"
p256 = SmolVLAPolicy(cfg256).eval().half().to("cuda")
r256 = capture_flow(p256, "256M (SmolVLM2-256M backbone, 16 VLM layers)")

print("\nCOMPARISON (Xavier NX, fp16, 3 flow steps):", flush=True)
print(f"  450M: eager={r450[0]:.0f}  replay={r450[1]:.0f}  graphed-e2e={r450[2]:.0f} ms", flush=True)
print(f"  256M: eager={r256[0]:.0f}  replay={r256[1]:.0f}  graphed-e2e={r256[2]:.0f} ms", flush=True)
