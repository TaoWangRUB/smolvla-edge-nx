"""Monolithic ONNX export of the fine-tuned SmolVLA checkpoint (ros2-cpp-async-deployment, D5).

Runs inside the smolvla-edge:sim container (lerobot 0.4.4 = checkpoint era, so preprocessing and
normalization match server.py exactly). Produces ONE graph covering vision + the fixed-task
language prefix + the action expert with all 10 flow-matching Euler steps unrolled.

The graph bakes in everything the serving side would otherwise need Python/lerobot for:
  - the task instruction, as fixed token ids (no C++ tokenizer)                [D5]
  - SigLIP image normalization ([0,1] -> [-1,1]; the processor's VISUAL norm is IDENTITY)
  - proprioceptive state normalization (MEAN_STD) + pad 14 -> max_state_dim
  - action un-normalization (MEAN_STD) + unpad max_action_dim -> action_dim

so the server feeds only: a resized [0,1] image, the raw joint state, and the denoising noise
(explicit input, so parity tests are deterministic under a fixed seed).

    python deploy/onnx/export_smolvla.py \
        --checkpoint outputs/train/smolvla_transfer_cube/checkpoints/020000 \
        --task "Pick up the cube with the right arm and transfer it to the left arm." \
        --out models/onnx/smolvla_transfer_cube.onnx

Why dynamo=True: the legacy TorchScript exporter chokes on the SmolVLM2 rotary embedding
(ScalarType ComplexDouble); the TorchDynamo exporter traces the whole VLM cleanly. RoPE still
leaves a few float64 Cos/Sin nodes that ONNX Runtime's CPU kernels reject, so we down-cast every
float64 tensor to float32 after export (the whole model is fp32; the doubles are spurious
high-precision intermediates).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import warnings
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))


# --------------------------------------------------------------------------------------------
# the exported graph: raw resized image + raw state + noise -> unnormalized action chunk
# --------------------------------------------------------------------------------------------
class SmolVLAOnnxWrapper(nn.Module):
    def __init__(self, model, tokens, masks, img_mask, state_mean, state_std,
                 action_mean, action_std, max_state_dim, action_dim):
        super().__init__()
        self.model = model
        self.action_dim = int(action_dim)
        self.max_state_dim = int(max_state_dim)
        self.register_buffer("tokens", tokens)          # (1, L) fixed task token ids
        self.register_buffer("masks", masks)            # (1, L) fixed attention mask
        self.register_buffer("img_mask", img_mask)      # (1,) single present camera
        self.register_buffer("state_mean", state_mean)  # (state_dim,)
        self.register_buffer("state_std", state_std)
        self.register_buffer("action_mean", action_mean)  # (action_dim,)
        self.register_buffer("action_std", action_std)

    def forward(self, image: torch.Tensor, state: torch.Tensor, noise: torch.Tensor):
        # image: (1,3,H,W) in [0,1] (already resized serving-side) -> SigLIP [-1,1]
        img = image * 2.0 - 1.0
        # state: (1, action_dim) raw -> MEAN_STD normalize -> pad to max_state_dim with zeros
        st = (state - self.state_mean) / self.state_std
        st = F.pad(st, (0, self.max_state_dim - st.shape[1]))
        actions = self.model.sample_actions(
            [img], [self.img_mask], self.tokens, self.masks, st, noise=noise)
        a = actions[:, :, : self.action_dim]              # unpad
        a = a * self.action_std + self.action_mean        # un-normalize (MEAN_STD)
        return a


def _bake_task_tokens(pre_processor, task: str, device):
    """Run the real preprocessing pipeline on one dummy observation so the baked tokens/masks
    are byte-identical to what server.py feeds (includes the SmolVLA newline step)."""
    dummy = {
        "observation.state": torch.zeros(14),
        "observation.images.top": torch.zeros(3, 224, 224),
        "task": task,
    }
    out = pre_processor(dummy)
    return (out["observation.language.tokens"].to(device),
            out["observation.language.attention_mask"].to(device))


def _stats(norm_step, key):
    ts = norm_step._tensor_stats[key]
    return ts["mean"].float().clone(), ts["std"].float().clone()


def _strip_layernorm_extra_outputs(model_proto) -> int:
    """TensorRT rejects LayerNormalization nodes with >1 output; the TorchDynamo export emits
    all 25 with (Y, Mean, InvStdDev) but the Mean/InvStdDev are unused. Drop them to 1 output so
    the TRT EP can compile the graph (CUDA/CPU EPs are unaffected)."""
    n = 0
    for node in model_proto.graph.node:
        if node.op_type == "LayerNormalization" and len(node.output) > 1:
            del node.output[1:]
            n += 1
    return n


def _double_to_float(model_proto):
    """Down-cast every float64 tensor in the graph to float32 (ORT has no fp64 Cos/Sin CPU
    kernels; the model is fp32, so these are spurious RoPE-precision intermediates)."""
    import onnx
    from onnx import TensorProto, numpy_helper

    D, Fl = TensorProto.DOUBLE, TensorProto.FLOAT
    n = 0

    def fix_tensor(t):
        nonlocal n
        if t.data_type == D:
            arr = numpy_helper.to_array(t).astype(np.float32)
            t.CopyFrom(numpy_helper.from_array(arr, t.name))
            n += 1

    g = model_proto.graph
    for init in g.initializer:
        fix_tensor(init)
    for vi in list(g.input) + list(g.output) + list(g.value_info):
        if vi.type.tensor_type.elem_type == D:
            vi.type.tensor_type.elem_type = Fl
            n += 1
    for node in g.node:
        for attr in node.attribute:
            if attr.name == "to" and attr.i == D:
                attr.i = Fl
                n += 1
            if attr.type == onnx.AttributeProto.TENSOR:
                fix_tensor(attr.t)
            for t in attr.tensors:
                fix_tensor(t)
    return n


def main() -> None:
    ap = argparse.ArgumentParser(description="Monolithic ONNX export of a SmolVLA checkpoint.")
    ap.add_argument("--checkpoint", default="outputs/train/smolvla_transfer_cube/checkpoints/020000")
    ap.add_argument("--task", required=True, help="fixed task instruction (baked into the graph)")
    ap.add_argument("--out", default="models/onnx/smolvla.onnx")
    ap.add_argument("--device", default="cpu", help="export device (cpu is enough; fp32)")
    ap.add_argument("--opset", type=int, default=17)
    ap.add_argument("--flow-steps", type=int, default=None,
                    help="override flow-matching Euler steps to UNROLL (match deployment; the "
                         "Stage-1 server runs fs3 — a 10-step graph is ~3x slower and drains the "
                         "async queue under GPU contention)")
    args = ap.parse_args()

    from lerobot.policies.factory import make_pre_post_processors
    from smolvla_edge.common import load_policy, resolve_policy_path

    from onnx_patches import patch_rope_no_sequence_ops
    patch_rope_no_sequence_ops()  # all-CUDA graph (no SplitToSequence/ScatterND) -> CUDA Graphs

    policy, dev = load_policy(args.checkpoint, args.device)
    policy.eval().float()  # force fp32 (SmolVLM2 backbone ships bf16; ORT rejects bf16 Conv)
    if args.flow_steps is not None:
        policy.config.num_steps = args.flow_steps  # unroll this many Euler steps (match server)
    pp = resolve_policy_path(args.checkpoint)
    pre, post = make_pre_post_processors(policy_cfg=policy.config, pretrained_path=pp)
    norm = next(s for s in pre.steps if type(s).__name__ == "NormalizerProcessorStep")

    state_mean, state_std = _stats(norm, "observation.state")
    action_mean, action_std = _stats(norm, "action")
    tokens, masks = _bake_task_tokens(pre, args.task, dev)
    img_mask = torch.ones(1, dtype=torch.bool, device=dev)

    cfg = policy.config
    action_dim = cfg.action_feature.shape[0]
    wrapper = SmolVLAOnnxWrapper(
        policy.model, tokens, masks, img_mask, state_mean, state_std,
        action_mean, action_std, cfg.max_state_dim, action_dim).eval().to(dev)

    # example inputs (batch 1) — resize is serving-side, so the graph image is already 512x512
    H = W = cfg.resize_imgs_with_padding[0]
    ex_image = torch.rand(1, 3, H, W, device=dev)
    ex_state = torch.zeros(1, action_dim, device=dev)
    ex_noise = torch.randn(1, cfg.chunk_size, cfg.max_action_dim, device=dev)

    with torch.no_grad():
        ref = wrapper(ex_image, ex_state, ex_noise)
    print(f"[export] eager forward OK -> action chunk {tuple(ref.shape)}")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    print("[export] tracing with the TorchDynamo ONNX exporter (this takes a few minutes)...")
    program = torch.onnx.export(
        wrapper, (ex_image, ex_state, ex_noise), dynamo=True,
        input_names=["image", "state", "noise"], output_names=["action_chunk"],
        opset_version=args.opset)
    tmp = out_path.with_suffix(".raw.onnx")
    program.save(str(tmp))

    import onnx

    mp = onnx.load(str(tmp))
    fixed = _double_to_float(mp)
    print(f"[export] down-cast {fixed} float64 tensors -> float32")
    stripped = _strip_layernorm_extra_outputs(mp)
    print(f"[export] stripped extra outputs from {stripped} LayerNormalization nodes (TRT compat)")
    onnx.checker.check_model(mp) if mp.ByteSize() < 2 * 1024**3 else None
    big = mp.ByteSize() >= 2 * 1024**3
    onnx.save(mp, str(out_path), save_as_external_data=big,
              all_tensors_to_one_file=True, location=out_path.name + ".data" if big else None)
    tmp.unlink(missing_ok=True)
    tmp.with_suffix(".onnx.data").unlink(missing_ok=True)

    sha = hashlib.sha256(out_path.read_bytes()).hexdigest()[:16] if not big else "external"
    meta = {
        "checkpoint": args.checkpoint,
        "task": args.task,
        "token_ids": tokens[0].tolist(),
        "attention_mask": masks[0].int().tolist(),
        "inputs": {"image": [1, 3, H, W], "state": [1, action_dim],
                   "noise": [1, cfg.chunk_size, cfg.max_action_dim]},
        "output": {"action_chunk": [1, cfg.chunk_size, action_dim]},
        "precision": "fp32", "opset": args.opset,
        "num_flow_steps": cfg.num_steps, "external_weights": big, "sha256_16": sha,
        "normalization": "baked: image [0,1]->[-1,1], state MEAN_STD+pad, action MEAN_STD unpad",
    }
    meta_path = out_path.with_suffix(".meta.json")
    meta_path.write_text(json.dumps(meta, indent=2))
    print(f"[export] wrote {out_path} ({out_path.stat().st_size/1e6:.0f} MB) + {meta_path.name}")


if __name__ == "__main__":
    main()
