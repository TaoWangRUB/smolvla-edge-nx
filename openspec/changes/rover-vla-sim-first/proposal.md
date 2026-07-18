# Proposal — Ackermann Rover VLA, Simulation-First

## Why

The ALOHA-sim → Xavier NX project proved the full pipeline this repo exists for: fine-tune a
flow-matching VLA, reproduce asynchronous chunked execution, and get it running on-device at
laptop-class latency (233 ms/chunk, bitwise-exact CUDA-graph capture) behind a ROS 2 interface.
The natural next embodiment is the one the hardware on the shelf was bought for: a **1/16
Ackermann rover** driven by a language-conditioned navigation policy — "drive to the red cone" →
safe, smooth vehicle motion. Navigation is also the embodiment where this repo's proven
machinery (chunked trajectories, latency-compensated async execution, grounding evaluation,
edge serving) transfers essentially verbatim, and it is deliberately structured as a miniature
of the consensus full-scale AD architecture (DriveVLM/EMMA dual-system lineage), so everything
built here has a documented upgrade path.

Three principles shape every decision. **Sim-first**: no hardware is purchased until the policy
works in simulation (single exception: the real camera model is chosen early so the simulated
camera can match it). **Staged autonomy**: the learned model predicts *intent* (a short local
trajectory); classical components handle *execution* (tracking, stabilization) and *safety*
(collision veto). **Trust allocation follows sensor allocation**: each subsystem gets exactly
the sensors it needs at the rate it needs them, and the safety-critical path never depends on
the neural network.

## What Changes

- Add a **simulation environment**: Isaac Sim + Isaac Lab Ackermann rover (URDF/USD matched to
  the real vehicle geometry), multi-environment scenes with per-episode domain randomization,
  and a simulated camera locked to the real camera's resolution/intrinsics. **Refinement over
  the draft plan**: M0 opens with an explicit *simulator feasibility gate* — Isaac Sim's RTX
  rendering needs ≳8–12 GB VRAM and the current dev GPU (RTX 2000 Ada, 4 GB) cannot run it, so
  week 1 decides between a bigger GPU / cloud instance / promoting the documented Gazebo(+heavy
  augmentation) fallback, instead of discovering this mid-M0.
- Add an **automated expert data pipeline**: privileged-map A* + Pure Pursuit/MPC expert,
  DART-style noise injection with expert relabeling from the first large dataset, a
  DAgger pass in M2, templated + LLM-paraphrased instructions, hindsight body-frame waypoint
  relabeling, all recorded in **LeRobot format** (drops into the existing training loop).
- Add the **waypoint policy**: SmolVLA with the action space redefined to K × (x, y, v)
  body-frame waypoints (chunk_size = K, action_dim = 3 — the existing
  `make_chunk_predictor`/normalization path is unchanged). Two-stage training (frozen backbone
  → LoRA). Contingencies pre-committed: (κ, v) curvature parameterization if tracking
  oscillates, action-expert re-init if manipulation pretraining hurts, Qwen2.5-VL + diffusion
  head as the architecture fallback — never a plain MSE regression head (mode-averaging cuts
  through obstacles).
- Add the **three-loop runtime**: 50–100 Hz tracker + EKF + geometric safety veto; 4–10 Hz
  async policy loop (this repo's Algorithm-1 stack, latency-compensated by odometry delta);
  optional 0.1–1 Hz mission layer (M4). **Refinement**: the runtime is the already-proven
  all-ROS2 shape — the policy served as a DDS node (`policy_node.py` pattern, pre-warmed
  capture at startup), the C++ `async_client` lineage as the chunk manager feeding the tracker.
- Add the **grounding evaluation protocol** before any data scaling: closed-loop success on
  held-out scenes, collision/intervention rates, the **swap test** with attribute-sharing hard
  negatives, held-out paraphrases, path quality, and failure slicing by logged randomization
  metadata. Scale data/unfreeze the model only against a metric-identified deficiency.
- Add the **sim-to-real deployment stage** behind a hardware purchase gate (M3).
  **Refinement over the draft plan**: the deployment baseline is the **Xavier NX already on
  hand with the measured fp16-graph path** — 233 ms/chunk = 4.3 chunks/s against a 2–3 s chunk
  horizon is ~10× replan overlap, comfortably inside the async operating envelope this repo
  derived and measured; the rover interface (fixed camera res, padded instruction, K×3 chunk)
  is *more* capture-friendly than ALOHA's. Jetson Orin becomes the upgrade if 5–10 Hz is ever
  metric-justified, not a prerequisite. Hardware-level timestamping (global-shutter camera,
  EKF pose interpolated at shutter time, blinking-LED latency bench) is required before any
  latency-compensation value is trusted.

## Capabilities

### New Capabilities

- `rover-sim-environment`: Isaac Sim/Isaac Lab Ackermann rover, randomized multi-target scenes
  where language is *necessary*, camera matched to the selected real sensor; Gazebo fallback
  behind an explicit M0 feasibility gate.
- `rover-expert-datagen`: privileged expert driving, recovery-data generation (DART noise +
  DAgger relabeling), templated+paraphrased instructions, hindsight waypoint labels, LeRobot
  episodes with full randomization metadata.
- `rover-waypoint-policy`: SmolVLA fine-tuned to emit K × (x, y, v) body-frame waypoint chunks;
  two-stage training on a single consumer GPU; pre-committed contingency ladder.
- `rover-async-control`: three-loop runtime — Pure Pursuit/MPC tracker + EKF + network-independent
  safety veto at 50–100 Hz, latency-compensated async chunk execution at 4–10 Hz, chunk-staleness
  watchdog with speed ramp-down; all-ROS2 serving shape.
- `rover-grounding-eval`: closed-loop evaluation with the swap test (hard negatives), paraphrase
  holdout, safety metrics, and metadata-sliced failure analysis as the scaling decision rule.
- `rover-sim2real-deployment`: M3 hardware gate — driver bring-up, hardware timestamping
  discipline, fp16-graph deployment on the Xavier NX, controlled-space transfer-gap
  quantification against the sim baseline.

## Non-Goals (recorded)

- No hardware purchase before the M2 exit criteria pass (camera model *selection* is allowed
  and required early; purchase of the rover/Jetson/sensors is the M3 gate).
- No fisheye/stereo camera input to the policy; no raw IMU or lidar into the policy — depth/lidar
  feed only the geometric safety monitor.
- No mission layer (SLAM + global planning) before M4; phases 1–2 run entirely without it.
- No plain-MSE waypoint regression head under any fallback.
- Full-scale AD (surround BEV, navigation router, RSS-formalized envelope) is the documented
  upgrade path in design.md §10, not part of this change.
