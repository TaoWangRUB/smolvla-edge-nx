"""Deterministic SmolVLA inference for cross-device parity of the from-source Jetson torch.

Runs the fine-tuned policy on ONE fixed synthetic observation with PINNED flow-matching noise
(same trick as deploy/onnx/parity.py), so the action chunk is fully deterministic. Run it on the
reference (dev-box) torch and on the from-source Xavier torch; the two chunks must match
(cosine ~1.0, small max-abs-diff) for the hand-built Jetson PyTorch to be numerically correct.

    CKPT=<path> OUT=/out/chunk.npy python3 parity_check.py
"""
from __future__ import annotations
import hashlib, os, sys
import numpy as np
import torch

sys.path.insert(0, os.environ.get("SRC", "/repo/src"))
from smolvla_edge.common import load_policy            # noqa: E402
from smolvla_edge.async_infer import make_chunk_predictor  # noqa: E402

CKPT = os.environ.get("CKPT", "/repo/data/smolvla_transfer_cube_020000")
OUT = os.environ.get("OUT", "/out/chunk.npy")
TASK = "Pick up the cube with the right arm and transfer it to the left arm."

policy, dev = load_policy(CKPT, "cuda")
policy.eval().float()                 # fp32 both sides for a fair numeric compare
policy.config.num_steps = 3           # match the ONNX export (3 flow steps)
cfg = policy.config
chunk, madim = cfg.chunk_size, cfg.max_action_dim

# pin the flow-matching noise so sampling is deterministic (parity.py's approach)
fixed_noise = torch.randn(1, chunk, madim, generator=torch.Generator().manual_seed(0)).to(dev).float()
_orig = policy.model.sample_actions
def _spy(images, img_masks, lang, lang_mask, state, noise=None, **kw):
    return _orig(images, img_masks, lang, lang_mask, state, noise=fixed_noise, **kw)
policy.model.sample_actions = _spy

predict = make_chunk_predictor(policy, CKPT, "cuda", TASK, precision="fp32")

rng = np.random.default_rng(42)       # identical obs on every device
obs = {"pixels": {"top": rng.integers(0, 256, (480, 640, 3), dtype=np.uint8)},
       "agent_pos": rng.standard_normal(14).astype(np.float32)}

a = np.asarray(predict(obs), dtype=np.float32)
os.makedirs(os.path.dirname(OUT), exist_ok=True)
np.save(OUT, a)
print(f"device={torch.cuda.get_device_name(0)} torch={torch.__version__}")
print(f"chunk shape={a.shape} mean={a.mean():.6f} std={a.std():.6f} sum={a.sum():.6f}")
print(f"sha16={hashlib.sha256(a.tobytes()).hexdigest()[:16]}  -> saved {OUT}")
