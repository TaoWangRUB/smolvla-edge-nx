## Why

Almost every SmolVLA example stops at a workstation or a physical arm; very few tell the
**8 GB Jetson Xavier NX optimization story**. Fine-tuning a flow-matching VLA is table-stakes —
the portfolio-worthy engineering is getting a SmolVLM-2 + flow-matching-expert policy to run
under real-time, on-device edge constraints and being honest about what converts, what doesn't,
and the latency budget you hit anyway. This change captures the full project: get a correct
checkpoint fast, then spend the real effort on edge deployment and latency engineering.

## What Changes

- Add a fine-tuning + validation path: smoke-test the stack on `lerobot/smolvla_base`, fine-tune
  on a public SO-101 pick-and-place dataset, and report a held-out success rate.
- Add **on-device** Xavier NX deployment: action chunking / decoupled execution, a low-Hz VLM
  stage, FP16, and INT8 only via real quantization/TensorRT on subgraphs that convert (naive
  INT8 casts are rejected on purpose).
- Add a **client/server** deployment mode: policy on the Titan X workstation, NX as a thin gRPC
  control client, giving a second benchmark point that includes network overhead.
- Add a cross-tier **latency benchmark harness** producing a results table (Titan X local; NX
  on-device FP16/INT8 ±chunking; NX-client/WS-server) plus a demo GIF of replayed episodes.
- Migrate the forward-looking planning docs (`docs/roadmap.md`, `docs/setup-jetson.md`,
  `docs/future-work-rover.md`) into this OpenSpec change; the mobile-rover embodiment is recorded
  as an explicit non-goal.

## Capabilities

### New Capabilities
- `smolvla-finetuning`: validate the LeRobot/SmolVLA stack and fine-tune a correct SO-101
  pick-and-place checkpoint with a held-out success-rate metric.
- `edge-ondevice-inference`: run the fine-tuned policy entirely on a Jetson Xavier NX (8 GB)
  under real-time constraints using chunking, a low-Hz VLM stage, and FP16/INT8 precision.
- `edge-client-server`: offload inference to a workstation over gRPC with the NX as a thin
  control client that replays held-out frames and reports split latency.
- `latency-benchmarking`: run the same workload across deployment tiers and collate the results
  into a reproducible latency/throughput/memory table plus demo media.

### Modified Capabilities
<!-- None: openspec/specs/ is currently empty; all capabilities are new. -->

## Impact

- **Code**: `src/smolvla_edge/{infer,eval,bench,common}.py`, `scripts/train.sh`,
  `scripts/make_demo_gif.py`, `configs/train.so101_pickplace.yaml`,
  `deploy/ondevice/`, `deploy/client_server/` (proto + server + client), `benchmarks/collate.py`.
- **Dependencies**: LeRobot pinned to `v0.5.0` (`lerobot[smolvla]`), ffmpeg/torchcodec on the
  dev box; NVIDIA aarch64 torch + JetPack CUDA/cuDNN/TensorRT on the Xavier NX.
- **Docs**: `docs/` is removed; its content now lives in this change's proposal/design/tasks.
  `README.md` links are updated to point at `openspec/` instead of `docs/`.
