## Why

Almost every SmolVLA example stops at a workstation or a physical arm; very few tell the
**8 GB Jetson Xavier NX optimization story**. Fine-tuning a flow-matching VLA is table-stakes —
the portfolio-worthy engineering is getting a SmolVLM-2 + flow-matching-expert policy to run
under real-time, on-device edge constraints and being honest about what converts, what doesn't,
and the latency budget you hit anyway. This change captures the full project: get a correct
checkpoint fast, then spend the real effort on edge deployment and latency engineering.

## What Changes

- Add a **simulation** fine-tuning + validation path (no robot required): smoke-test on
  `lerobot/smolvla_base`, fine-tune on a LeRobot-native ALOHA sim dataset
  (`lerobot/aloha_sim_insertion_human`), and report a **closed-loop success rate** by rolling out
  in the matching gym-aloha MuJoCo env (`gym_aloha/AlohaInsertion-v0`). An open-loop replay proxy
  remains as a no-sim fallback. (Real SO-101 on hardware is deferred — no robot on hand.)
- Add **on-device** Xavier NX deployment (**optional, only with a Jetson on hand**): action
  chunking / decoupled execution, a low-Hz VLM stage, FP16, and INT8 only via real
  quantization/TensorRT on subgraphs that convert (naive INT8 casts are rejected on purpose).
- Add a **client/server** deployment mode (optional): policy on a workstation, a second host as a
  thin gRPC control client, giving a benchmark point that includes network overhead.
- Add a cross-tier **latency benchmark harness** producing a results table (local GPU always;
  NX on-device FP16/INT8 ±chunking and client/server when a Jetson is available) plus a demo GIF
  built from ALOHA sim frames.
- Migrate the forward-looking planning docs (`docs/roadmap.md`, `docs/setup-jetson.md`,
  `docs/future-work-rover.md`) into this OpenSpec change; the mobile-rover embodiment is recorded
  as an explicit non-goal.

## Capabilities

### New Capabilities
- `smolvla-finetuning`: validate the LeRobot/SmolVLA stack and fine-tune a correct ALOHA-sim
  checkpoint, reporting a closed-loop success rate from gym-aloha rollouts (no robot).
- `async-sim-inference`: reproduce the SmolVLA paper's asynchronous inference stack (§3.3,
  Algorithm 1) fully in simulation — action queue + threshold `g`, joint-space similarity
  filter, chunk aggregation on overlap, paced control loop — and compare sync vs async on
  success rate and time-to-completion (no hardware required; the on-device phase reuses it).
- `edge-ondevice-inference`: run the fine-tuned policy entirely on a Jetson Xavier NX (8 GB)
  under real-time constraints using chunking, a low-Hz VLM stage, and FP16/INT8 precision.
- `edge-client-server`: offload inference to a workstation over gRPC with the NX as a thin
  control client that replays held-out frames and reports split latency.
- `latency-benchmarking`: run the same workload across deployment tiers and collate the results
  into a reproducible latency/throughput/memory table plus demo media.

### Modified Capabilities
<!-- None: openspec/specs/ is currently empty; all capabilities are new. -->

## Impact

- **Code**: `src/smolvla_edge/{infer,eval,bench,common}.py` (`eval.py` implements the gym-aloha
  closed-loop rollout), `scripts/train.sh`, `scripts/make_demo_gif.py`,
  `configs/train.aloha_sim.yaml` (+ `configs/train.so101_pickplace.yaml` kept for the future
  real-robot path), `deploy/ondevice/`, `deploy/client_server/`, `benchmarks/collate.py`.
- **Dependencies**: LeRobot pinned to `v0.5.0` (`lerobot[smolvla]`), `gym-aloha` + MuJoCo
  (`MUJOCO_GL=egl` on headless boxes), ffmpeg/torchcodec on the dev box; NVIDIA aarch64 torch +
  JetPack CUDA/cuDNN/TensorRT on the Xavier NX (only if the optional edge phase is done).
- **Docs**: `docs/` is removed; its content now lives in this change's proposal/design/tasks.
  `README.md` links are updated to point at `openspec/` instead of `docs/`.
