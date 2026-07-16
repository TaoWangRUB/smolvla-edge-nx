# Design — ROS2 C++ Async Deployment

## Context

The repo has a working async inference stack, entirely in Python and entirely inside one
container:

- `src/smolvla_edge/async_infer.py` — `AsyncRunner`: one action popped per 20 ms tick, chunk
  re-trigger when `len(queue)/n < g` (checked after the pop, Algorithm 1 line 6), an `epsilon`
  joint-space similarity filter, and `aggregate_chunks` (`new_wins` = linear 0.5→1.0 blend on
  the overlap). Sync inference is the `g = 0` limit of the same loop.
- `deploy/client_server/server.py` — Python gRPC server (`PredictChunk` / `Reset` / `Health`)
  wrapping the fine-tuned checkpoint on GPU.
- `docker/Dockerfile` → `smolvla-edge:sim` — **Python 3.11** locked by gym-aloha's mujoco 2.3.7
  pin, lerobot 0.4.4 (checkpoint-era). `docker-compose.yml` mounts the repo at `/workspace`,
  `runtime: nvidia`, `MUJOCO_GL=egl`.

Hard environment facts that shape everything below:

1. **The sim cannot leave its container** (py3.11 vs host py3.12), and **ROS2 Jazzy cannot enter
   it** (Jazzy = Ubuntu 24.04 / py3.12). Two containers minimum; Docker is a design input, not
   packaging.
2. Host: WSL2, RTX 2000 Ada 8 GB (sm_89). CUDA EP and TensorRT 10 both supported. VRAM is shared
   by EGL rendering + policy inference. WSL2 latency drifts with host load — all timing claims
   need measured provenance (established practice in this repo's README).
3. **The ROS2 Docker environment already exists.** `ackermann_rover_x86_64_jazzy` (built from
   `~/workspace/ackermann_rover_humble/docker/Dockerfile` with `ROS_DISTRO=jazzy`) is a working
   Jazzy desktop dev image on this host: Ubuntu 24.04, ros-jazzy-desktop, Gazebo Harmonic,
   colcon/vcstool/rosdep, OpenCV/Eigen, protoc, non-root `taowang` (uid 1000) user, entrypoint
   that sources ROS. Verified gaps for this project: no `grpc_cpp_plugin`/`libgrpc++-dev`, no
   Python `grpc` module, and its entrypoint runs `rosdep install` against a mounted rover
   workspace. The host `/opt/ros/jazzy` install is for introspection/debugging only.
4. No official SmolVLA ONNX export exists (lerobot #1899/#3146 declined/open; optimum lacks
   SmolVLM2). FastCrest/tether (BSL 1.1) proves a monolithic export with the 10-step Euler loop
   unrolled reaches cos ≈ 1.0 parity — so the export is feasible, but we own the code.

## Goals / Non-Goals

**Goals:**

- ROS2 Jazzy C++ control node behaviorally equivalent to `AsyncRunner` (same g / epsilon /
  aggregation semantics, same tick model), validated against the Python stack.
- gym-aloha reachable from ROS2 across the container boundary without touching the sim image's
  Python stack.
- Fine-tuned checkpoint exported to ONNX with verified numerical parity; C++ gRPC server on
  ONNX Runtime CUDA EP behind the unchanged `Policy` proto.
- Every stage independently runnable and comparable to the previous stage (success rate +
  action-chunk diff), so a regression is attributable to one layer.

**Non-Goals:**

- Isaac Sim / Isaac Lab-Arena (needs more VRAM than 8 GB comfortably allows and is unsupported
  on WSL2; the Cache-SCA/CoRL2026-CSI IsaacLab-SO101 checkpoints are noted for a future change).
- Real hardware drivers, Xavier NX ROS2 build (JetPack 5 = Ubuntu 20.04; future work).
- Cross-machine DDS; replacing gRPC as the inference transport.
- Beating the Python server's latency. The C++ server's win is deployability (no Python runtime),
  not GPU-bound inference speed; benchmarks report honestly whichever way they land.

## Decisions

### D1. Container topology: three services, gRPC-only boundaries, DDS confined to one container

```
docker-compose network: smolvla
┌─────────────────────────────┐   ┌──────────────────────────────┐
│ smolvla-edge:sim (py3.11)   │   │ smolvla-edge:ros2 (Jazzy)    │
│                             │   │                              │
│  sim-server  (SimEnv gRPC)◄─┼───┼─ sim_bridge (rclpy)          │
│                             │   │     │ topics: /observation/* │
│  policy-server (Policy gRPC)│   │     ▼         /action        │
│      ▲ (Stage 1: unchanged) │   │  async_client (rclcpp, C++)  │
└──────┼──────────────────────┘   └──────┼───────────────────────┘
       └────────── PredictChunk ─────────┘
Stage 2 adds: cpp-server (Policy gRPC, ONNX Runtime CUDA EP) — same proto, compose profile swap.
```

- **All ROS2 nodes live in one container.** DDS discovery across Docker bridge networks on WSL2
  (multicast) is notoriously unreliable; confining DDS to a single container's localhost makes it
  a non-issue. Alternative — host networking + cross-container DDS — rejected: fragile on WSL2
  and buys nothing, since the pieces that must cross containers already speak gRPC.
- **gRPC is the only inter-container protocol** (`SimEnv` sim shim + existing `Policy`). This
  mirrors the target production shape: robot controller ↔ inference server over a socket.
- The ROS2 image gets `runtime: nvidia` for Stage 2 (cpp-server can live there or in its own
  slim service); Stage 1 ROS2 nodes need no GPU.

### D1a. ROS2 image: thin overlay on `ackermann_rover_x86_64_jazzy`, not a from-scratch build

`docker/ros2.Dockerfile` is `FROM ackermann_rover_x86_64_jazzy:latest` and adds only:
`libgrpc++-dev protobuf-compiler-grpc` (apt, Ubuntu 24.04), `grpcio`/`grpcio-tools` for the
rclpy bridge (`pip --break-system-packages` or `python3-grpcio`), and a project entrypoint that
skips the rover image's startup `rosdep install` (it targets a rover workspace that won't be
mounted here). Rationale: the base is already built, known-good on this host (WSL2), and carries
the full Jazzy/colcon toolchain — rebuilding it from `ros:jazzy` buys nothing and loses the
proven environment. Trade-off accepted: ~11.8 GB of rover extras (Gazebo Harmonic, rtabmap,
turtlebot3) ride along unused; disk is already spent since the base exists. Alternative
(`FROM ros:jazzy` minimal image, ~3 GB) is recorded as the portable fallback if this ever needs
to build on a machine without the rover image — the overlay's apt/pip layer is identical either
way, so switching bases later is a one-line change.

### D2. Sim shim: a `SimEnv` gRPC service inside the existing sim container

`deploy/client_server/sim_server.py` wraps the gym-aloha env the same way `eval.py` does
(`Reset() → obs`, `Step(action) → obs, reward, done`, JPEG-or-raw camera frames + qpos, reusing
the proto's `Image`/`Tensor` messages). Added to `policy.proto` as a new service; the `Policy`
service is untouched, so existing Python clients keep working.

Alternatives rejected:
- rclpy inside the sim container — impossible (py3.11 vs Jazzy's py3.12).
- Building ROS2 from source against py3.11 — days of toolchain work for zero portfolio value.
- Shared-memory / volume-file exchange — reinvents an RPC badly.

### D3. Timing contract owned by the bridge node, not the sim

gym-aloha steps as fast as CPU allows; the *bridge* enforces "one `/action` per 20 ms tick, hold
pose on idle ticks" — the same discretization `async_infer.py` documents. The rclpy bridge runs a
50 Hz wall-clock timer: publish latest observation, wait ≤ one period for an `/action`, step the
sim (hold previous pose if none arrived — idle tick), repeat. This keeps sim-time == tick-count
semantics identical to the Python harness so success rates are comparable.

### D4. C++ `async_client` node: a faithful port, not a redesign

`rclcpp` node, structure mirrors `AsyncRunner` member-for-member:

- **Observation-driven ticks** (refined during implementation): the bridge owns the 50 Hz
  clock (D3), so the client executes one Algorithm-1 tick per received `/observation` rather
  than free-running its own wall timer — two unsynchronized 50 Hz timers would drift in phase
  and deliver 0-or-2 pops per sim tick. One `SimObservation` == one tick: pop → publish
  `/action`; after the pop, if `queue.size()/n < g`, snapshot the observation and hand it to
  the gRPC worker.
- **Real time replaces virtual time**: `AsyncRunner` delays chunk visibility by `ceil(L/dt)`
  ticks to emulate latency inside a faster-than-realtime sim. The bridge paces the sim at
  wall-clock 50 Hz, so a chunk becomes visible when its gRPC reply actually lands — same
  algorithm, the latency is real instead of emulated. (Same reasoning for the cold start:
  `start_episode` blocks in Python; here the bridge holds the sim until the first action.)
- One worker thread owns the gRPC `PredictChunk` stub (async completion queue not needed — one
  in-flight request max, exactly like the Python stack's single non-blocking trigger).
- `epsilon` similarity filter on joint state and `new_wins` overlap aggregation ported with the
  same formulas (`w = linspace(0.5, 1.0, m)`); parameters (`g`, `epsilon`, `dt`, aggregator,
  server address) as ROS parameters in the launch file.
- Publishes the same per-tick event fields (`sent`, `filtered`, `merged`, `idle`) on a
  diagnostics topic so runs can be compared event-by-event against Python runs.

Sync mode is `g = 0`, same as Python — no separate node.

### D5. ONNX export: monolithic graph, unrolled denoise loop, fixed task tokens — own the script

Export script runs **inside the sim container** (lerobot 0.4.4 = checkpoint era, so preprocessing
/ normalization match `server.py` exactly). Approach follows what tether validated: trace the
whole `select_action`-equivalent as one graph — vision encoder + language prefix + action expert
with the 10 Euler steps unrolled (fixed step count makes the loop trivially traceable). The task
instruction is fixed per deployment, so token ids are baked in at export time — no C++ tokenizer.
Noise input becomes an explicit graph input (deterministic parity tests feed a fixed seed).

- Why not multi-graph + C++ loop: more moving parts, more C++ surface to get wrong, and the
  monolithic route is empirically proven; revisit only if the single graph fights `torch.onnx`.
- Why not depend on tether: BSL 1.1 is fine for this use, but the export is the intellectual
  core of the project — we write it, and keep tether as a cross-check reference if stuck.
- FP32 first (bit-parity target vs PyTorch FP32), FP16/TRT after parity is locked in.
- Parity harness: N held-out observations → PyTorch chunk vs ORT chunk; gate at
  max-abs-diff ≤ 1e-4 and cosine ≥ 0.9999 (tether's E7 issue is a cautionary tale: compute the
  gate *and enforce it*).

### D6. C++ inference server: ONNX Runtime CUDA EP first, TensorRT EP as a flag

`deploy/cpp_server/` — gRPC C++ server implementing `Policy` (`PredictChunk`/`Reset`/`Health`)
over `onnxruntime-gpu`'s C++ API. TensorRT EP is a runtime option once CUDA EP parity is proven
(engine cache on a volume; first-build latency is minutes). Compose profile
(`--profile cpp` vs `--profile py`) swaps which server backs the same port, so A/B is one flag.

### D7. Validation: the Python stack is the oracle at every stage

- **Stage 1 gate:** same checkpoint, same env, same seeds — ROS2/C++ client + Python server must
  match the pure-Python async runner on success rate over ≥ 50 episodes (within binomial noise)
  and produce equivalent per-tick event traces.
- **Stage 2 gate:** offline chunk parity (D5) first, then closed-loop success with cpp-server
  swapped in. A regression isolates to exactly one layer by construction.
- Latency rows (client-observed `PredictChunk` round trip, per-tick jitter) join the existing
  benchmark table with measured provenance, per repo convention.

### D8. Deployment pipeline: one gated script, not a CI platform

`scripts/deploy_pipeline.sh` (runs the stages via `docker compose run`) chains the existing
gates in dependency order: **export → parity (D5 gate) → cpp-server image build → closed-loop
sim regression (D7 Stage 2 gate) → benchmark rows appended with provenance**. Non-zero exit at
the first failed gate; each stage writes its artifact (ONNX + parity report, image tag, episode
results, benchmark rows) under a run-stamped directory in `benchmarks/results/pipeline/<ts>/`
with checkpoint/export/image hashes for lineage. Rationale: the industry pattern is the *gated
chain* (GR00T's build→verify→benchmark), not any particular orchestrator — a shell entrypoint
over compose services demonstrates the automation without dragging in Airflow/Kubeflow, and
GitHub Actions can call the same entrypoint later. Alternative (Makefile targets) rejected only
because stage ordering + fail-fast + run-stamped artifact dirs read clearer in one script.

## Risks / Trade-offs

- **[ONNX export of the SmolVLM2 backbone fails to trace]** → Follow tether's known-good
  monolithic recipe; fall back to multi-graph split (vision+prefix / one Euler step looped in
  C++); last resort, wrap tether itself for the export while keeping our parity harness.
- **[lerobot 0.4.4 checkpoint pre/post-processing drifts from what C++ feeds the graph]** → Bake
  normalization into the exported graph (export from the same wrapper `server.py` calls), so the
  C++ server sends raw resized images + qpos only.
- **[8 GB VRAM shared: EGL rendering + Python server + cpp-server during A/B]** → Never run both
  servers simultaneously (compose profiles enforce this); `MUJOCO_GL=osmesa` fallback documented;
  measure VRAM per service into the benchmark table.
- **[50 Hz tick jitter in containers on WSL2 invalidates timing claims]** → Report jitter
  percentiles alongside means; steady-state measurement windows; note host-load sensitivity
  (established repo practice). The tick *count* model (D3) keeps success rates valid even when
  wall-clock jitters.
- **[DDS image transport overhead inside the ROS2 container]** → **MEASURED, and it binds the
  loop**: the sim→bridge→client tick (mujoco step + EGL render + 921 KB raw frame over gRPC,
  then again over DDS) costs 100–150 ms on this WSL2 box — the stack sustains ~6.6 Hz, not
  50 Hz. Consequence for validation: chunk latency in *ticks* is ~6× smaller than the Python
  oracle's at dt=20 ms, so success rates are only comparable at matched dt (gate v2 analysis).
  The 50 Hz claim needs transport work (JPEG on the wire, shrinking the double image hop) —
  recorded as follow-up; until then, benchmark rows report the measured effective rate.
- **[Scope creep: ROS2 ecosystem (tf, URDF, RViz) invites gold-plating]** → The node set is
  frozen at bridge + client (+ diagnostics topic); visualization is explicitly out of scope.

## Migration Plan

1. **Stage 0**: `docker/ros2.Dockerfile` + compose services; `ros2 doctor` and a talker/listener
   smoke test pass in the container; sim image rebuilt only to add grpc reflection (if needed).
2. **Stage 1a**: `sim_server.py` + `SimEnv` proto; validated by a Python client replaying the
   existing `eval.py` rollout through the shim (success rate matches in-process eval).
3. **Stage 1b**: rclpy bridge + C++ async client vs unchanged Python `policy-server`; Stage 1
   gate (D7). **Rollback:** the Python stack is never removed — every compose service remains
   runnable throughout.
4. **Stage 2a**: export + parity harness (D5 gate).
5. **Stage 2b**: cpp-server behind `--profile cpp`; Stage 2 gate; TensorRT EP flag last.
6. README + benchmark table updates land with each gate, not at the end.

## Open Questions

- FP16 export: quality loss on flow-matching action heads is usually negligible but must be
  measured — gate on closed-loop success, not just tensor diffs.
- TensorRT EP vs native TRT engine + C++ runner: EP is less code; revisit only if EP performance
  disappoints.
- Whether `SimEnv` should also serve rendered frames for the demo GIF pipeline or keep that in
  the existing `eval.py` path (leaning: keep in `eval.py`, out of scope here).
