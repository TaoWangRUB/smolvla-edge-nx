# Tasks — ROS2 C++ Async Deployment

## 1. Stage 0 — Docker environment (ros2-docker-env)

- [x] 1.1 Write `docker/ros2.Dockerfile` as a thin overlay `FROM ackermann_rover_x86_64_jazzy`
      (design D1a): add `libgrpc++-dev protobuf-compiler-grpc` + Python `grpcio`/`grpcio-tools`,
      override the rover entrypoint (skip its `rosdep install`); build as `smolvla-edge:ros2`
      and verify `grpc_cpp_plugin` + `python3 -c "import grpc"`
- [x] 1.2 Add `sim-server`, `policy-server`, `ros2` services to `docker-compose.yml` on a shared
      network (nvidia runtime only on `sim-server`/`policy-server`); keep all existing services
      working
- [x] 1.3 Smoke-test: `ros2 doctor` + talker/listener inside the `ros2` container; gRPC
      `Health` of `policy-server` reachable from the `ros2` container by service name
- [x] 1.4 Verify sim image stack unchanged (mujoco 2.3.7 / lerobot 0.4.4; `eval` service still
      passes its known-good baseline)

## 2. Stage 1a — Sim shim (ros2-sim-bridge, sim side)

- [x] 2.1 Extend `deploy/client_server/proto/policy.proto` with the `SimEnv` service
      (Reset/Step/observation reuse of `Image`/`Tensor`); regenerate Python stubs via
      `gen_proto.sh`; confirm existing Policy client/server unaffected
- [x] 2.2 Implement `deploy/client_server/sim_server.py` wrapping the gym-aloha env the same
      way `smolvla_edge.eval` does (seeding included); runs in the sim container
- [x] 2.3 Reference-client validation (`sim_env_check.py`): **transparency PASS** (100 steps,
      camera/state/reward byte-identical local vs shim, seed 0) and closed-loop through-shim
      4/5 vs in-process 3/5 on seeds 0-4 (unseeded flow noise; both within binomial noise of
      the 14/20 = 70% baseline)

## 3. Stage 1b — ROS2 nodes (ros2-sim-bridge bridge side, ros2-async-control)

- [x] 3.1 Scaffold the `deploy/ros2/` colcon workspace (`smolvla_bridge` Python pkg,
      `smolvla_client` C++ pkg, `smolvla_msgs`); builds clean in the ros2 container
- [x] 3.2 Implement the rclpy sim-bridge node: SimEnv gRPC client ↔ topics (`/observation`
      single message with image+state+tick, `/action`), tick owner with cold-start
      re-publication (deadlock fix) and a one-tick grace window before declaring idle
      (race fix — bridge and client idle counts now agree at 0)
- [x] 3.3 Generate C++ gRPC stubs from `policy.proto` in the CMake build of `smolvla_client`
- [x] 3.4 Implement the `async_client` rclcpp node: observation-driven ticks (bridge owns the
      50 Hz clock), queue pop/publish, threshold `g` after pop, single in-flight PredictChunk
      worker thread, ROS parameters (`g`, `epsilon`, aggregator, server address)
- [x] 3.5 Port `epsilon` similarity filter and `new_wins`/`blend` aggregation + ramp_in;
      gtest vs fixtures exported from `async_infer.aggregate_chunks`: 20 cases, 2/2 pass
- [x] 3.6 Per-tick diagnostics on `/events` (sent/filtered/merged/idle, RTT) + `event_recorder`
      JSONL dump comparable with Python `AsyncRunner` traces
- [x] 3.7 `stage1.launch.py` (bridge + client + recorder, Shutdown on bridge exit); e2e vs the
      unchanged Python `policy-server`: 2/2 success, idle=0 both sides, ~32 s/episode,
      tick p50 100-120 ms (image-transport-bound; honest number for the benchmark table).
      Env fixes en route: policy-server pinned fp32 (fp16+PredictChunk dtype crash),
      ROS_LOCALHOST_ONLY on the ros2 service (concurrent containers cross-wired DDS)
- [x] 3.8 **Stage 1 gate: PASS** (see design "DDS image transport" finding for the caveat):
      - v1 (fs10, g=0.7): ROS2 34/50 = 68% vs oracle 0/50 — invalid comparison: ℓ≈1.0 s is
        outside the 50 Hz envelope (README's own finding), and the ROS2 tick measured
        151.6 ms (6.6 Hz effective), so its latency-in-ticks is ~6× smaller
      - v2 (matched config: same fs3 server, g=0.5, ramp_in=5, seeds 0-49): ROS2
        **37/50 = 74%, idle 0.0, RTT p50 0.48 s** vs oracle at dt=20 ms 21/50 = 42%
        (75 idle ticks/ep) — gap fully attributed to tick duration
      - **final, matched dt (oracle --fps 6.62, same server/seeds): ROS2 37/50 = 74% vs
        oracle 39/50 = 78%, z = 0.47 → within binomial noise; both idle=0, sends/ep
        12.9 vs ~11, latency 0.48 s both** — algorithm layer equivalent; the 50 Hz
        transport gap is the recorded follow-up (JPEG on the wire)
      - artifacts: `benchmarks/results/ros2/stage1_gate{,_fs3}.json` + events JSONL

- [x] 3.9 **Native-Linux re-verification + 6.6 Hz root cause** (RTX A2000 Laptop, FastDDS
      `rmw_fastrtps_cpp`): rebuilt both images on the new host, regenerated the gitignored gRPC
      Python stubs, added in-loop timing (`sim_bridge` `grpc_step`/`dds_roundtrip`, C++
      `TickEvent.proc_ms`) and per-episode GIF recording (`gif_dir:=`).
      - **The 6.6 Hz did not reproduce — same stack now runs 40.0 ms/tick = 25 Hz.** The old
        151 ms was host-specific: `env.step` EGL render 70.5 ms → **7.3 ms** here, and the DDS
        leg ~80 ms → **~12 ms** here (both ~10× faster on a real GPU + fast intra-host DDS).
      - Measured tick decomposition (~15 250 ticks): gRPC `Step` 16.0 ms + DDS 900 KiB round trip
        11.6 ms + C++ node 0.06 ms = ~28 ms real work, rounded to 40 ms by the fixed 50 Hz
        bridge timer. **Not DDS-bound; the timer quantization is the biggest lever** (event-driven
        step → ~35 Hz). PredictChunk RTT 229 ms (SmolVLA fs3) stays off the critical path.
      - **Async confirmed working as designed:** idle = 0 on every real tick — the 229 ms fs3
        server latency is fully hidden behind queue execution (Algorithm 1 intent).
      - 50-episode closed-loop re-run (seeds 0–49, `g=0.5`, ramp-in 5, per-episode GIF) with the
        **fine-tuned SmolVLA deliverable** (fs3, fp32): **40/50 = 80 %, idle 0.04, 12.5 sends/ep**
        — inside the original gate's binomial band (ROS2 74 %, oracle 78 %), now at native 25 Hz.
        Artifacts: `benchmarks/results/ros2/stage1_50ep.json`, `.../gifs_50/`,
        `.../timing_breakdown.json`.
      - **Fresh native-Linux head-to-head vs the Python `AsyncRunner` oracle** (same checkpoint,
        server, seeds; oracle at matched dt `--fps 25`): ROS2 **40/50 = 80 %** vs Python
        **33/50 = 66 %** (`benchmarks/results/ros2/python_oracle_50ep.json`). Failures are nested —
        the 10 ROS2 fails are a strict subset of the 17 Python fails, zero ROS2-only. Not an
        algorithmic difference (unit-tested-equal port): the oracle's virtual-time model rounds
        latency up to whole ticks (`ceil(L/dt)`, async_infer.py:187) — 205 ms → 6 ticks — while the
        C++ client uses real wall-clock chunk arrival (async_client.cpp:9-11), ~1 tick less stale.
        The real-time deployment pays **no** accuracy penalty vs the idealized model.
      - Follow-up unchanged: 900 KiB raw frame → JPEG on the wire trims the 12 ms DDS leg; and an
        event-driven bridge step removes the 10 ms timer-quantization tax.

## 4. Stage 2a — ONNX export + parity (policy-onnx-export)

- [ ] 4.1 Export script `deploy/onnx/export_smolvla.py` (runs in sim container): monolithic
      graph, 10 Euler steps unrolled, fixed task tokens baked in, normalization inside the
      graph, noise as explicit input; task string recorded in graph metadata
- [ ] 4.2 Load-check under ONNX Runtime (Python) with expected I/O shapes; external-weights
      layout if > 2 GB
- [ ] 4.3 Parity harness `deploy/onnx/parity.py`: ≥ 100 held-out observations, fixed noise
      seed; ENFORCED gate max-abs-diff ≤ 1e-4 AND cosine ≥ 0.9999 (non-zero exit on fail);
      report artifact with hashes into `benchmarks/results/`
- [ ] 4.4 (Optional after FP32 parity) FP16 variant; gate on closed-loop success, not tensor
      diffs

## 5. Stage 2b — C++ inference server (cpp-inference-server)

- [ ] 5.1 Scaffold `deploy/cpp_server/` (CMake, gRPC C++ service impl of Policy, ONNX Runtime
      GPU C++ API, CUDA EP); containerize (extend `smolvla-edge:ros2` or a slim runtime image)
- [ ] 5.2 Implement PredictChunk (raw image/state ingest → graph inputs → chunk reply with
      timestamps), Reset, Health (device/precision/model path/provider)
- [ ] 5.3 Compose profiles `py` / `cpp` backing the same server address; verify only one server
      in the GPU process list at a time
- [ ] 5.4 Wire-compat check: Python reference client and the ROS2 client both run unmodified
      against the C++ server
- [ ] 5.5 **Stage 2 gate:** closed-loop batch (same seeds, ≥ 50 episodes) vs Python-server
      baseline within binomial noise
- [ ] 5.6 TensorRT EP flag + persistent engine cache volume; Health reports active provider;
      measure first-build vs cached startup

## 6. Pipeline automation (deployment-pipeline)

- [ ] 6.1 `scripts/deploy_pipeline.sh <checkpoint>`: chain export → parity → cpp-server image
      build → closed-loop regression → benchmark rows via `docker compose run`, fail-fast with
      the failing gate named
- [ ] 6.2 Run-stamped artifact dirs under `benchmarks/results/pipeline/<ts>/` with a manifest
      (checkpoint/export/image hashes, per-gate status)
- [ ] 6.3 Green-run the pipeline on the fine-tuned checkpoint; force a parity failure to
      demonstrate fail-fast; commit both manifests as provenance examples

## 7. Benchmarks + docs

- [ ] 7.1 Add ROS2+py-server and ROS2+cpp-server (CUDA EP, and TRT EP if done) rows to the
      benchmark table: PredictChunk RTT, per-tick jitter percentiles, VRAM per service — with
      measured provenance per repo convention
- [ ] 7.2 README: ROS2 quickstart (compose commands, profiles), pipeline usage, architecture
      diagram, honest Python-vs-C++ result discussion
- [ ] 7.3 Record deferred items (Xavier NX ROS2 build, Isaac Lab-Arena option with the
      IsaacLab-SO101 community checkpoints) in the change's design Open Questions / future work
