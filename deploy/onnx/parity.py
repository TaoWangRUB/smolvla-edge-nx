"""Enforced numerical parity gate for the ONNX export (ros2-cpp-async-deployment, D5 gate).

Compares the exported graph against the PyTorch policy on >= N held-out observations under a
FIXED noise seed, and EXITS NON-ZERO unless, on every observation,
    max-abs-diff <= 1e-4  AND  cosine >= 0.9999   (FP32 baseline).
A computed-but-unenforced PASS/FAIL is a spec violation (tether's E7 cautionary tale), so the
gate is the process exit code, not a printed number.

    python deploy/onnx/parity.py \
        --checkpoint outputs/train/smolvla_transfer_cube/checkpoints/020000 \
        --onnx models/onnx/smolvla_transfer_cube.onnx \
        --task "Pick up the cube with the right arm and transfer it to the left arm." \
        --observations 100

Reference = the exact server code path (`make_chunk_predictor`, same as server.py), so
preprocessing / normalization can never silently drift. A spy on `sample_actions` (a) pins the
denoising noise so both sides start from identical latents and (b) captures the [-1,1] image the
model actually consumed, which we feed back to the graph as [0,1] — the graph re-applies the
baked SigLIP + state normalization, so those are under test too. Rollout seed is disjoint from
the Stage-1 eval seeds (default 900).
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

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

MAX_ABS_DIFF = 1e-4
MIN_COSINE = 0.9999


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    a, b = a.ravel(), b.ravel()
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))


def main() -> int:
    ap = argparse.ArgumentParser(description="Enforced ONNX parity gate for SmolVLA.")
    ap.add_argument("--checkpoint", default="outputs/train/smolvla_transfer_cube/checkpoints/020000")
    ap.add_argument("--onnx", default="models/onnx/smolvla_transfer_cube.onnx")
    ap.add_argument("--task", required=True)
    ap.add_argument("--observations", type=int, default=100)
    ap.add_argument("--seed", type=int, default=900, help="rollout seed (disjoint from eval 0-49)")
    ap.add_argument("--env-id", default="gym_aloha/AlohaTransferCube-v0")
    ap.add_argument("--flow-steps", type=int, default=None, help="match the exported graph's steps")
    ap.add_argument("--report", default="benchmarks/results/onnx_parity.json")
    args = ap.parse_args()

    import gymnasium as gym
    import gym_aloha  # noqa: F401
    import onnxruntime as ort

    from smolvla_edge.async_infer import make_chunk_predictor
    from smolvla_edge.common import load_policy

    policy, _ = load_policy(args.checkpoint, "cpu")
    policy.eval().float()
    cfg = policy.config
    if args.flow_steps is not None:
        cfg.num_steps = args.flow_steps  # match the exported graph
    action_dim = cfg.action_feature.shape[0]

    # server-exact reference path; a spy pins the noise and grabs the consumed image
    fixed_noise = torch.randn(1, cfg.chunk_size, cfg.max_action_dim,
                              generator=torch.Generator().manual_seed(0))
    cap: dict = {}
    orig_sample = policy.model.sample_actions

    def spy(images, img_masks, lang, lang_mask, state, noise=None, **kw):
        cap["image01"] = ((images[0] + 1.0) / 2.0).detach().cpu().numpy().astype(np.float32)
        return orig_sample(images, img_masks, lang, lang_mask, state, noise=fixed_noise, **kw)

    policy.model.sample_actions = spy
    predict = make_chunk_predictor(policy, args.checkpoint, "cpu", args.task, precision="fp32")

    sess = ort.InferenceSession(args.onnx, providers=["CPUExecutionProvider"])
    in_names = [i.name for i in sess.get_inputs()]

    env = gym.make(args.env_id, obs_type="pixels_agent_pos", render_mode="rgb_array",
                   max_episode_steps=10_000)
    obs, _ = env.reset(seed=args.seed)
    hold = env.action_space.sample() * 0

    rows, worst_diff, worst_cos = [], 0.0, 1.0
    for i in range(args.observations):
        ref = np.asarray(predict(obs), dtype=np.float32)          # (chunk, action_dim), server-exact
        image01 = cap["image01"]
        raw_state = np.asarray(obs["agent_pos"], dtype=np.float32)[None, :action_dim]
        feed = dict(zip(in_names, [image01, raw_state, fixed_noise.numpy()]))
        out = sess.run(None, feed)[0][0]                          # drop batch dim -> (chunk, action_dim)

        d = float(np.abs(out - ref).max())
        c = _cosine(out, ref)
        worst_diff, worst_cos = max(worst_diff, d), min(worst_cos, c)
        rows.append({"obs": i, "max_abs_diff": d, "cosine": c,
                     "pass": d <= MAX_ABS_DIFF and c >= MIN_COSINE})

        obs, _, term, trunc, _ = env.step(hold)
        if term or trunc:
            obs, _ = env.reset(seed=args.seed + i + 1)
    env.close()

    failures = [r for r in rows if not r["pass"]]
    passed = not failures
    big = Path(args.onnx).stat().st_size >= 2 * 1024**3
    report = {
        "onnx": args.onnx,
        "onnx_sha256_16": "external" if big
        else hashlib.sha256(Path(args.onnx).read_bytes()).hexdigest()[:16],
        "checkpoint": args.checkpoint, "task": args.task,
        "observations": args.observations,
        "tolerances": {"max_abs_diff": MAX_ABS_DIFF, "min_cosine": MIN_COSINE},
        "worst_max_abs_diff": worst_diff, "worst_cosine": worst_cos,
        "passed": passed, "failures": failures[:10], "per_obs": rows,
    }
    rp = Path(args.report)
    rp.parent.mkdir(parents=True, exist_ok=True)
    rp.write_text(json.dumps(report, indent=2))

    print(f"[parity] {args.observations} obs | worst max-abs-diff={worst_diff:.3e} "
          f"(<= {MAX_ABS_DIFF}) | worst cosine={worst_cos:.8f} (>= {MIN_COSINE})")
    if passed:
        print(f"[parity] PASS -> {rp}")
        return 0
    print(f"[parity] FAIL on {len(failures)} obs, e.g. {failures[0]} -> {rp}")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
