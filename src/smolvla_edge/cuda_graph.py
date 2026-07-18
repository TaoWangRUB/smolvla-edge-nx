"""Lazy CUDA-Graph capture of SmolVLA's full ``sample_actions`` forward.

The fix for launch-bound eager execution on weak-CPU edge boards (Jetson Xavier NX:
608 -> 233 ms/chunk, bitwise-identical actions — see deploy/jetson-native-torch/README.md,
"The fix"). The first call runs eager and harvests static input templates; capture then
happens inline (one-time, a few seconds) and every later call copies the inputs into
static buffers and replays the recorded kernel stream with a single launch.

Shapes are baked at capture: image resolution, padded task-token length, flow steps,
chunk size. Task *content* and observations may vary call-to-call; a shape mismatch
falls back to eager for that call. Any failure during capture restores eager permanently.

Capture-blockers neutralized (all constant H2D copies, illegal during stream capture):
  1. ``torch.tensor(list/scalar)`` constants in embed_prefix/embed_suffix/the flow loop
     -> value-keyed constant folding of ``torch.tensor`` during warmup+capture only.
  2. The HF SmolVLM vision tower's ``patch_attention_mask`` (built on CPU per call)
     -> inject a precomputed all-ones bool GPU mask.
  3. The NaViT position-ids loop in ``SmolVLMVisionEmbeddings`` (``.cpu()`` syncs)
     -> precompute position ids, swap in a capture-safe embeddings forward
     (equivalence asserted before use).
"""

from __future__ import annotations

import threading


def enable_lazy_graph(policy, verbose: bool = True) -> None:
    """Patch ``policy.model.sample_actions`` to capture-and-replay lazily.

    The policy must already be ``.eval().half()`` on CUDA (the fp16 tensor-core config
    the capture was validated against). Safe to call once per policy instance.
    """
    import torch

    model = policy.model
    orig_sample = model.sample_actions
    lock = threading.Lock()
    st: dict = {"phase": "pending"}  # pending -> ready | failed
    log = print if verbose else (lambda *a, **k: None)

    def _fold_constants():
        """Context patch: constant-fold torch.tensor during warmup+capture."""
        _tt = torch.tensor
        cache: dict = {}

        def cached(data, *a, **kw):
            try:
                key = (repr(data), repr(a), repr(sorted(kw.items(), key=lambda x: x[0])))
            except Exception:
                key = None
            if key is not None and key in cache:
                return cache[key]
            t = _tt(data, *a, **kw)
            if key is not None:
                cache[key] = t
            return t

        return _tt, cached

    def _patch_vision(s_img):
        """Make the HF vision tower capture-safe; returns an undo callable."""
        vm = model.vlm_with_expert.get_vlm_model().vision_model
        ps = vm.patch_size
        patch_mask = torch.ones((1, s_img.shape[-2] // ps, s_img.shape[-1] // ps),
                                dtype=torch.bool, device=s_img.device)
        emb_mod = vm.embeddings

        rec: dict = {}
        _orig_pe = emb_mod.position_embedding.forward
        emb_mod.position_embedding.forward = \
            lambda ids: (rec.__setitem__("ids", ids), _orig_pe(ids))[1]
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
            ref = model.vlm_with_expert.embed_image(s_img).clone()
        emb_mod.position_embedding.forward = _orig_pe
        pos_ids = rec["ids"].clone()

        def _emb_safe(pixel_values, patch_attention_mask=None, **kw):
            pe = emb_mod.patch_embedding(pixel_values)
            return pe.flatten(2).transpose(1, 2) + emb_mod.position_embedding(pos_ids)

        _orig_emb_fwd = emb_mod.__dict__.get("forward")
        emb_mod.forward = _emb_safe
        _orig_vm_fwd = vm.forward
        vm.forward = (lambda pixel_values, patch_attention_mask=None, **kw:
                      _orig_vm_fwd(pixel_values,
                                   patch_attention_mask=patch_mask
                                   if patch_attention_mask is None else patch_attention_mask,
                                   **kw))

        def undo():
            vm.forward = _orig_vm_fwd
            if _orig_emb_fwd is None:
                emb_mod.__dict__.pop("forward", None)
            else:
                emb_mod.forward = _orig_emb_fwd

        with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
            new = model.vlm_with_expert.embed_image(s_img)
        dv = (new.float() - ref.float()).abs().max().item()
        if dv > 1e-3:
            undo()
            raise RuntimeError(f"vision capture-patch changed numerics: {dv:.2e}")
        return undo

    def _try_capture(s_args, ref_out):
        s_imgs, s_imasks, s_lang, s_lmask, s_state, s_noise = s_args
        undo_vision = _patch_vision(s_imgs[0])

        def graph_fn():
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16,
                                                 cache_enabled=False):
                return orig_sample(s_imgs, s_imasks, s_lang, s_lmask, s_state,
                                   noise=s_noise)

        _tt, cached = _fold_constants()
        torch.tensor = cached
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
        except Exception:
            undo_vision()
            raise
        finally:
            torch.tensor = _tt

        g.replay()
        torch.cuda.synchronize()
        d = (static_out.float() - ref_out.float()).abs().max().item()
        if d > 1e-2:
            undo_vision()
            raise RuntimeError(f"graph replay parity failed: max-abs-diff={d:.2e}")
        log(f"[cuda-graph] captured full sample_actions; replay parity diff={d:.2e}")

        shapes = ([tuple(i.shape) for i in s_imgs], tuple(s_lang.shape),
                  tuple(s_state.shape))

        def graphed(images, img_masks, lang_tokens, lang_masks, state, noise=None, **kw):
            if ([tuple(i.shape) for i in images], tuple(lang_tokens.shape),
                    tuple(state.shape)) != shapes:
                log("[cuda-graph] input shapes changed — eager fallback for this call")
                return orig_sample(images, img_masks, lang_tokens, lang_masks, state,
                                   noise=noise, **kw)
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
                s_noise.normal_()  # fresh flow noise, generated outside the graph
            g.replay()
            return static_out.clone()

        return graphed

    def lazy(images, img_masks, lang_tokens, lang_masks, state, noise=None, **kw):
        with lock:
            if st["phase"] == "ready":
                return st["graphed"](images, img_masks, lang_tokens, lang_masks, state,
                                     noise=noise, **kw)
            if st["phase"] == "failed":
                return orig_sample(images, img_masks, lang_tokens, lang_masks, state,
                                   noise=noise, **kw)
            # first call: run eager (correct result now), then capture from its inputs
            if noise is None:
                noise = model.sample_noise(
                    (state.shape[0], policy.config.chunk_size,
                     policy.config.max_action_dim), state.device)
            out = orig_sample(images, img_masks, lang_tokens, lang_masks, state,
                              noise=noise, **kw)
            try:
                torch.cuda.synchronize()
                s_args = ([i.clone() for i in images], [m.clone() for m in img_masks],
                          lang_tokens.clone(), lang_masks.clone(), state.clone(),
                          noise.clone())
                st["graphed"] = _try_capture(s_args, out.clone())
                st["phase"] = "ready"
                log("[cuda-graph] lazy capture complete — replay path active")
            except Exception as e:
                st["phase"] = "failed"
                log(f"[cuda-graph] capture failed ({type(e).__name__}: {str(e)[:200]}) "
                    "— staying eager")
            return out

    model.sample_actions = lazy
