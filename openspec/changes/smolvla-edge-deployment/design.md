## Context

SmolVLA is a SmolVLM-2 backbone plus a flow-matching action expert, pretrained on SO-100/SO-101
manipulation data. The deliverable is a demo: a correct checkpoint plus a credible edge-deployment
and latency story on a Jetson Xavier NX (8 GB) that is already on hand. No robot arm and no new
hardware are required — fine-tuning uses public Hugging Face datasets, eval runs on replayed /
held-out episodes, and deployment targets the NX.

Hardware in play:
- **Dev + inference**: Titan X (Maxwell, 12 GB, pre-Tensor-core).
- **Training run**: rent an A100/H100 for a few hours (~20k steps ≈ 4 h on a single A100). The
  Titan X can train at batch 16 (~6 GB) but is much slower, so it stays a dev box.
- **Edge target**: Jetson Xavier NX, 8 GB.
- **LeRobot**: pinned to `v0.5.0`.

The Jetson is its own world — aarch64 wheels, NVIDIA-provided torch, TensorRT baked into JetPack,
and power modes that materially change benchmark numbers. `pip install torch` from PyPI must not be
run there.

## Goals / Non-Goals

**Goals:**
- A *correct* fine-tuned SO-101 pick-and-place checkpoint with a held-out success-rate number,
  reached fast so effort can shift to edge work.
- The policy running entirely on the Xavier NX under real-time constraints, with an honest
  accounting of what converted to a faster engine and what didn't.
- A reproducible cross-tier latency/throughput/memory table plus a short demo GIF.
- A second deployment point (workstation gRPC server + NX client) that mirrors how real robots
  offload heavy inference.

**Non-Goals:**
- **Mobile-rover embodiment.** The on-hand rover is an Ackermann/mobile base, not an arm. SmolVLA's
  pretraining prior is for manipulation, so adapting it means new observation/action spaces, a new
  action head, and more aggressive re-training — a research project, not a two-week demo. The
  Xavier NX latency work carries over unchanged, which is exactly why manipulation is done first.
  Framing for the writeup: "manipulation validates the VLA + edge pipeline; the rover is the
  natural next embodiment, and the edge-deployment work carries over unchanged."
- A clean end-to-end TensorRT engine for the whole VLA (see Risks).
- Training a policy from scratch — this is fine-tuning from `lerobot/smolvla_base`.

## Decisions

**Fine-tune, don't train from scratch.** SmolVLA is pretrained on SO-100/SO-101, so a public
SO-101 dataset (`lerobot/svla_so101_pickplace`) is the turnkey path to a correct checkpoint.
Alternative (rover / from-scratch) rejected as out of scope above.

**Smoke-test before training anything.** Run `smolvla_edge.infer` on `lerobot/smolvla_base`
against the dataset first, so stack/version problems surface before burning rented-GPU hours.

**Split training off the dev box.** Rent an A100/H100 for the ~20k-step run; keep the Titan X for
dev and inference. Local training is possible at batch 16 (~6 GB) but too slow to iterate on.

**On-device optimization order (by payoff):**
1. *Action chunking + decoupled execution* — SmolVLA separates action prediction from execution,
   cutting task time ~30% on average; predict a chunk, execute it open-loop while the next
   prediction computes.
2. *Low-Hz VLM stage* — the SmolVLM-2 backbone is the expensive part; run it at a lower rate than
   the action expert / control loop.
3. *Precision* — FP16 first (near-free, big memory win on 8 GB). INT8 only via real
   quantization/TensorRT on subgraphs that convert, never a naive cast.

**Client/server over gRPC.** A `policy.proto` defines the RPC; the workstation serves the policy,
the NX client replays held-out frames as observations and reports round-trip latency split into
server-compute vs. network overhead. Chosen to mirror real offload deployments and to add a second
benchmark tier cheaply.

**Benchmark harness owns honesty.** `smolvla_edge.bench` drops per-run JSON into
`benchmarks/results/raw/`; `benchmarks/collate.py` regenerates `summary.csv` and the markdown table.
The harness refuses naive INT8 casts so the INT8 row can only be filled by a real engine.

## Xavier NX (JetPack) setup

Operational reference for standing up the edge target (migrated from `docs/setup-jetson.md`):

1. **Base.** Flash a recent JetPack and record the exact version — it pins the CUDA/cuDNN/TensorRT
   triple. Confirm with `nvcc --version` and `dpkg -l | grep -i tensorrt`. Max out clocks for
   benchmarking and record the mode: `sudo nvpmodel -m 0` (record `nvpmodel -q`) then
   `sudo jetson_clocks`.
2. **PyTorch.** Install the NVIDIA aarch64 torch wheel matching the JetPack (from the NVIDIA Jetson
   PyTorch index/forum), then torchvision built against it. Verify with
   `python -c "import torch; print(torch.__version__, torch.cuda.is_available())"`.
3. **LeRobot + this repo.** `pip install -r requirements.txt` (torch is already installed; pip must
   not replace it) then `pip install -e .`. If `lerobot[smolvla]` tries to pull a PyPI torch,
   install LeRobot with `--no-deps` and add its remaining deps manually so the NVIDIA torch stays.
4. **Memory headroom on 8 GB.** Prefer FP16; mount swap/zram if you hit OOM during model load.
   Close the desktop GUI (`sudo systemctl isolate multi-user.target`) to reclaim VRAM. Watch live
   with `tegrastats` (or `jtop` from `jetson-stats`).
5. **Sanity check.** `python -m smolvla_edge.infer --policy-path <checkpoint> --episodes 1
   --max-frames 10`, then proceed to on-device deployment and the benchmark commands.

## Risks / Trade-offs

- **Full TensorRT of a SmolVLM-2 + flow-matching VLA is non-trivial** → Don't claim a clean
  end-to-end TRT engine. Deliver "what converts" (vision encoder / static subgraphs → engine +
  speedup), "what doesn't" (dynamic control flow, the flow-matching sampler loop → why + fallback),
  and the latency budget hit anyway with a per-stage breakdown. Capture attempts in
  `deploy/ondevice/conversion_notes.md`.
- **8 GB OOM on model load** → FP16, swap/zram, and closing the desktop GUI; measure peak memory in
  the benchmark harness.
- **JetPack drift changes numbers** → record JetPack version and power mode with every benchmark run
  so results stay comparable.
- **Dependency conflict: `lerobot[smolvla]` clobbering NVIDIA torch** → `--no-deps` install path on
  the NX, verified via the torch/CUDA import check.

## Migration Plan

- Move planning docs into this change and delete `docs/`; update `README.md` links from `docs/…`
  to the corresponding `openspec/changes/smolvla-edge-deployment/…` artifacts (and `openspec/specs/`
  once archived). Demo media that lived under `docs/assets/` moves alongside the benchmark outputs.
