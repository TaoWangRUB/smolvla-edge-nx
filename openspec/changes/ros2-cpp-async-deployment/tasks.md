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

- [x] 4.1 Export script `deploy/onnx/export_smolvla.py` (runs in sim container): monolithic
      graph, 10 Euler steps unrolled, fixed task tokens baked in, normalization inside the
      graph (image [0,1]→[-1,1], state MEAN_STD+pad, action MEAN_STD unpad), noise as explicit
      input; task + token ids + shapes + hash recorded in `<out>.meta.json`.
      Key findings: the **legacy TorchScript exporter fails** (`ScalarType ComplexDouble` from the
      SmolVLM2 rotary embedding) — the **TorchDynamo exporter** (`dynamo=True`, needs `onnxscript`)
      traces the whole VLM cleanly. Must force the model to **fp32** (SmolVLM2 ships bf16 → ORT
      rejects bf16 Conv) and **down-cast 330 float64 tensors → float32** post-export (ORT has no
      fp64 Cos/Sin CPU kernels — spurious RoPE-precision intermediates). Output 1871 MB single file.
- [x] 4.2 Load-check under ONNX Runtime (`InferenceSession`, CPU EP) with expected I/O shapes
      (`image[1,3,512,512]`, `state[1,14]`, `noise[1,50,32]` → `action_chunk[1,50,14]`); < 2 GB so
      single-file (external-weights branch wired for > 2 GB).
- [x] 4.3 Parity harness `deploy/onnx/parity.py`: 100 held-out observations (rollout seed 900,
      disjoint from eval 0-49), fixed noise seed, reference via the exact `make_chunk_predictor`
      server path (spy pins noise + captures the consumed image). **ENFORCED gate is the process
      exit code** (max-abs-diff ≤ 1e-4 AND cosine ≥ 0.9999, non-zero on fail). **PASS: worst
      max-abs-diff 4.29e-6, worst cosine 0.99999988** over 100 obs →
      `benchmarks/results/onnx_parity.json` (with checkpoint + onnx sha).
- [x] 4.4 (Optional after FP32 parity) FP16 ONNX variant — **closed as superseded**: the native
      fp16 + manual CUDA-Graph path (`precision="fp16-graph"`, smolvla-edge-deployment §6.2b) is
      2.6× faster than the fp32 ONNX baseline with **bitwise-identical** actions and passed the
      closed-loop gate (see 5.5 rerun below); an ONNX-fp16 re-export can no longer beat it on
      either speed or accuracy risk.

## 5. Stage 2b — C++ inference server (cpp-inference-server)

- [x] 5.1 Scaffolded `deploy/cpp_server/` (CMake + `src/server.cpp`, gRPC C++ Policy impl over
      the ONNX Runtime C++ CUDA EP). Containerized as `docker/cpp_server.Dockerfile` FROM
      `smolvla-edge:sim` (already carries CUDA 12.4 + cuDNN 9 + cuBLAS via the pytorch base's pip
      nvidia wheels — cheaper than adding CUDA to the Jazzy image), adding gRPC/protobuf/OpenCV
      C++ + the ORT GPU C++ release, with `LD_LIBRARY_PATH` -> the nvidia wheel lib dirs (the
      rover's pattern; setting it inside the process is too late for the loader). CUDA EP loads;
      fp32 graph peaks at **2.3 GB VRAM → fits the 4 GB GPU** (fp16 not needed).
- [x] 5.2 PredictChunk (serving-side `resize_with_pad` in OpenCV: 480x640 raw uint8 -> 384x512
      bilinear -> pad 128 top -> 512x512 [0,1] CHW; state passthrough; noise sampled here) +
      Reset (stateless graph) + Health (device/precision/model/provider). Dims read from the
      graph at load, so one binary serves any exported variant.
- [x] 5.3 Compose `cpp-server` service (profile `cpp`, nvidia runtime, same `:50051` as the
      Python server) + `CPP_PROVIDER`/`ONNX_MODEL` env; A/B is which server you start.
- [x] 5.4 **Wire-compat PASS:** the Python reference client and the **ROS2 `async_client` run
      unmodified** against the C++ server (valid [50,14] chunks, Health cuda/fp32).
- [x] 5.5 **Stage 2 gate: PASSED via the fp16-graph server** (the C++ ORT attempt below remains
      the recorded negative result). Rerun of the identical 50-ep protocol (seeds 0-49, g=0.5,
      ramp_in 5, fs3) against `policy-server --precision fp16-graph` (CUDA-graph capture, 53 ms
      exclusive on the A2000): **36/50 = 72%, idle 0.0** — inside the binomial band of the 74-80%
      references and far above the ORT-CUDA server's 22%. Artifacts:
      `benchmarks/results/ros2/stage2_fp16graph_50ep{.json,_events.jsonl}`. Original ORT finding
      (kept for the writeup): **FAILS on this GPU, for a characterized latency reason (not a
      correctness bug).** ROS2 stack -> C++ server, 50 ep: **11/50 = 22 %, idle 64.7/ep** vs the
      80 % Python-server baseline. Root cause: ORT CUDA EP inference of the 500 M-param SmolVLM2
      VLM is **~1.0 s/chunk** on the RTX A2000 Laptop (GPU pinned ~1 GHz/20 W at 100 % util under
      sustained load) vs PyTorch's **229 ms** — the cost is the VLM *prefix* forward, not the
      Euler steps (fs3 export did not help), so at g=0.5 the ~1 s async buffer starves -> idle ->
      failure. The graph itself is correct (parity 4e-6) and wire-compatible; the gate would pass
      on a workstation-class GPU (design's Titan X tier). Artifact:
      `benchmarks/results/ros2/stage2_cpp_50ep.json`. Remedy under evaluation: **5.6 TensorRT EP**.
- [x] 5.6 **Latency root-cause investigated (the 5.5 wall) — closed: superseded by manual CUDA-graph
      capture in native torch** (smolvla-edge-deployment 6.2b), which attacks the same
      launch/fusion economics from the torch side and won (bitwise-exact, 2.6×; the ORT graph
      remains kernel-execution-bound and TRT still can't parse it — no longer worth pursuing).
      Original investigation record: ORT node profiling: 97% CUDA, the
      TorchDynamo graph is 30852 tiny nodes. Tried, in order: ORT graph-opt (30852→4920, no RTT
      change — ORT does it online); `transformers.optimizer` (544 ms, −22%); a **RoPE patch**
      (`deploy/onnx/onnx_patches.py`: `apply_rope` split→slice + in-place→cat) that removes ALL
      `SplitToSequence`/`SequenceAt`/`ScatterND` → **graph is now 100% CUDA-partitioned** (parity
      still 4e-6), which **unblocks CUDA Graphs**. But CUDA Graphs gave **no speedup** (618 vs
      621 ms) → the server is **kernel-execution-bound, not launch-overhead-bound**: ORT's unfused
      small kernels execute slower than PyTorch's fused ones (cuBLAS/flash-attn/fused-LN), and the
      RTX A2000 Laptop sits at ~1 GHz/20 W under load. **TensorRT EP** (wired via
      `CPP_PROVIDER=tensorrt`). **TRT attempted**: baked `tensorrt-cu12==10.16` into the image +
      LD_LIBRARY_PATH; the EP loads, but TRT's ONNX parser can't compile the dynamo graph — chain
      of op-compat errors (LayerNormalization >1 output → fixed with `_strip_layernorm_extra_outputs`
      in the export, 25 nodes; then ReduceMean; then dynamic-axis "Axis must be an initializer"; …),
      so ORT fragments it into CUDA-fallback islands with no fusion benefit. Conclusion: closing the
      fusion gap needs either a **workstation GPU** (recommended — ORT-CUDA ~200-250 ms, gate passes
      as-is) or a **TRT-clean re-export** (open-ended op surgery). CUDA Graphs is a dead end (the
      graph is kernel-execution-bound, not launch-overhead-bound).

- [x] 5.7 **Same graph, real edge target, pure-Python ORT-GPU** (cross-ref: smolvla-edge-deployment
      §6). The Stage-2a monolithic ONNX now runs on a **Jetson Xavier NX (8 GB)** under the ORT
      CUDA EP without the C++ server: **~610 ms mean/chunk** (fp32, JetPack 5.1 / CUDA 11.4,
      20 W/6-core; `docker/jetson_infer.Dockerfile` + `docker-compose.jetson.yml` +
      `deploy/onnx/bench_ort.py`). Confirms the §5.5/5.6 read that the wall is GPU-fusion/throughput,
      not a correctness bug — the identical graph is portable to the edge; latency just scales with
      the device. JP5's newest cp38/CUDA-11.4 wheel is `onnxruntime-gpu 1.15.1`, so the graph needs
      `deploy/onnx/patch_for_ort115.py` (IR 10→9 + int64→int32 ArgMin casts) to load.

## 6. Pipeline automation (deployment-pipeline)

- [x] 6.1 `scripts/deploy_pipeline.sh <checkpoint>` — **done**, gates: export (REUSE_ONNX=1 skips a
      deterministic re-export) → enforced parity → **fp16-graph capture smoke** (the shipped serving
      path; replaces the superseded cpp-server build gate) → ROS2 closed-loop regression (≥60%
      success) → collate; fail-fast with the failing gate named
- [x] 6.2 Run-stamped artifact dirs under `benchmarks/results/pipeline/<ts>/` with `manifest.json`
      (checkpoint + onnx sha16, per-gate status) + per-gate logs — done
- [x] 6.3 **Green run PASS** (20260718_215428: parity PASS, graph-smoke PASS, closed-loop 4/5 = 80%,
      collate PASS) and **forced failure demonstrated** (20260718_215843: `PARITY_FLOW_STEPS=10`
      breaks the torch reference vs the fs3 graph → parity FAIL → pipeline stops, manifest records
      the failing gate). Both manifests committed as provenance examples

## 7. Benchmarks + docs

- [x] 7.1 Benchmark rows — **done, in `summary.csv` (16 tiers)**: ROS2+py-server fp32 (80%, RTT
      229 ms, 25 Hz tick), ROS2+cpp-server ORT-CUDA (22%, ~1.0 s — recorded negative),
      ROS2+py-server **fp16-graph** (72% Stage-2 gate), and the **all-ROS2 NX tier** (below)
- [x] 7.2 README ROS2 chapter — **done**: quickstart/compose commands + tick decomposition were
      already in; added Stage-2b honest C++ ORT negative + how the gate passed (fp16-graph),
      pipeline usage, and the **all-ROS2 section** (see 7.4)
- [x] 7.3 Deferred items recorded in design.md ("Deferred / future work"): resolved open
      questions (fp16 export, TRT — both superseded), JPEG-on-the-wire, event-driven bridge
      step, Isaac Lab-Arena option
- [x] 7.4 **All-ROS2 policy hop (client AND server on ROS 2) — built and measured.**
      `docker/jetson_ros2_humble.Dockerfile` overlays ROS 2 Humble on the JP5 torch image
      (22.04/py3.10 = Humble's tier); `deploy/ros2/policy_node.py` serves the fp16-graph
      predictor as an rclpy node ON the Xavier NX (`/policy/request`→`/policy/chunk`);
      `async_client` grew `transport:=ros2` (gRPC default unchanged); new msgs
      PolicyRequest/PolicyChunk built on both distros. Cross-host cross-distro DDS
      (Humble↔Jazzy) discovered out of the box. **Closed loop 10 eps: 8/10 = 80%**, ep0 idle
      414 = one-time lazy CUDA-graph capture (pre-warm to remove), steady-state idle 9–17/ep.
      Artifacts: `benchmarks/results/ros2/allros2_nx_10ep{.json,_events.jsonl}`
