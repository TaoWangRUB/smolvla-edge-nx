# ROS2 C++ Async Deployment

## Why

The async inference stack (SmolVLA §3.3, Algorithm 1) is proven in pure Python, but it stops
short of how a real robot would consume it: a Python control loop inside a research container is
neither deterministic at 50 Hz nor deployable to a robot controller. The meaningful next step is a
**ROS2 (Jazzy) C++ control node** driving the same client/server architecture, with the policy
exported out of PyTorch into **ONNX for C++ GPU inference** — staged so every step is validated
against the existing Python stack as a regression oracle.

**Docker is a first-class constraint, not an afterthought.** The sim stack is locked inside the
`smolvla-edge:sim` container (Python 3.11 for the mujoco 2.3.7 / gym-aloha match; lerobot 0.4.4 —
see `docker/Dockerfile`), while ROS2 Jazzy is an Ubuntu 24.04 / Python 3.12 world. The two can
never share one image, so the architecture must be a **multi-container docker-compose topology**
where gRPC is the only thing that crosses container boundaries — which conveniently is exactly
the shape of a real robot deployment (controller box ↔ inference workstation).

## What Changes

- **ROS2 Jazzy container as a thin overlay on an image that already exists on this host**:
  `docker/ros2.Dockerfile` builds `smolvla-edge:ros2` `FROM ackermann_rover_x86_64_jazzy`
  (the user's rover dev image — ROS2 Jazzy desktop, Gazebo Harmonic, colcon/rosdep, OpenCV,
  built from `~/workspace/ackermann_rover_humble/docker/`), adding only the gRPC C++/Python
  toolchain it lacks and a project entrypoint. Added to `docker-compose.yml` alongside the
  existing `smolvla-edge:sim` services. All ROS2 nodes live here; the existing sim image is
  **not modified structurally** (only gains a small sim-server entrypoint).
- **Stage 1 — ROS2 control plane, Python policy server unchanged:**
  - A **sim gRPC shim** inside the existing sim container exposing gym-aloha as a service
    (`Reset` / `Step` / observations with cameras + qpos), so the py3.11-locked sim is reachable
    across the container boundary.
  - A Python `rclpy` **sim-bridge node** (ROS2 container) that maps the sim service onto ROS
    topics (`/observation/*`, `/joint_states`, `/action`) and enforces the one-action-per-20 ms
    tick timing model from `src/smolvla_edge/async_infer.py`.
  - A C++ `rclcpp` **async client node** porting the Algorithm-1 client: action queue, threshold
    `g`, joint-space similarity filter, chunk aggregation on overlap, 50 Hz timer. It speaks the
    existing `PredictChunk` gRPC (proto reused verbatim via C++ codegen) to the **unchanged
    Python policy server**.
- **Stage 2 — policy out of Python:**
  - **ONNX export** of the fine-tuned SmolVLA checkpoint as a monolithic graph with the 10-step
    flow-matching Euler loop unrolled (the approach FastCrest/tether validated at cos ≈ 1.0),
    plus a parity harness comparing action chunks against PyTorch on identical observations.
  - A **C++ gRPC inference server** (ONNX Runtime CUDA EP first, TensorRT EP as a follow-up)
    behind the same proto, swappable with the Python server via a compose profile.
- **Regression oracle throughout:** identical observations → action-chunk max-abs-diff, and
  closed-loop gym-aloha success rate ROS2/C++ stack vs Python stack.
- **Pipeline automation (the industry-shaped deliverable):** one gated command that takes a
  checkpoint and produces the deployable artifact chain — ONNX + enforced parity report →
  container build → closed-loop sim regression gate → benchmark rows — failing loudly at the
  first gate that misses. Mirrors the GR00T build→verify→benchmark deployment pattern.
- Non-goals: Isaac Sim/Isaac Lab (8 GB VRAM + WSL2 make it impractical now; revisit later),
  real-robot hardware drivers, Jetson Xavier NX build of the ROS2 stack (JetPack 5 is Ubuntu
  20.04 — recorded as future work), replacing gRPC with DDS across machines.

## Capabilities

### New Capabilities
- `ros2-docker-env`: reproducible ROS2 Jazzy + CUDA container and docker-compose topology in
  which every other capability runs; GPU passthrough (WSL2 / nvidia runtime) and inter-container
  networking for gRPC and DDS.
- `ros2-sim-bridge`: gym-aloha exposed across the container boundary (sim-side gRPC shim) and
  mapped onto ROS2 topics by a driver-style bridge node that owns the 50 Hz / 20 ms-per-action
  timing contract.
- `ros2-async-control`: the Algorithm-1 client side as a C++ `rclcpp` node — action queue,
  threshold `g`, similarity filter, chunk aggregation, paced tick — behaviorally equivalent to
  `async_infer.py`.
- `policy-onnx-export`: fine-tuned SmolVLA checkpoint exported to ONNX (monolithic, denoise loop
  unrolled) with numerical parity verified against the PyTorch policy.
- `cpp-inference-server`: C++ gRPC server implementing the existing `Policy` proto over ONNX
  Runtime CUDA EP (TensorRT EP optional), drop-in replacement for the Python server.
- `deployment-pipeline`: single-command, gate-ordered automation checkpoint → export+parity →
  package → sim regression → benchmark rows, with versioned artifacts and non-zero exit on the
  first failed gate.

### Modified Capabilities
<!-- openspec/specs/ is still empty (smolvla-edge-deployment is not yet archived), so there are
     no established specs to modify. This change builds on that change's async-sim-inference and
     edge-client-server capabilities but introduces only new specs. -->

## Impact

- **Code**: new `deploy/ros2/` colcon workspace (C++ client node, rclpy bridge node, launch
  files), new `deploy/client_server/sim_server.py` (sim shim, runs in the sim container), new
  `deploy/onnx/` (export + parity scripts), new `deploy/cpp_server/` (ONNX Runtime C++ server);
  `deploy/client_server/proto/policy.proto` extended with a `SimEnv` service (existing `Policy`
  service untouched).
- **Docker**: new `docker/ros2.Dockerfile`; `docker-compose.yml` gains `sim-server`,
  `policy-server` (existing image), `ros2` (+ later `cpp-server`) services on a shared network.
- **Dependencies**: ROS2 Jazzy (containerized; host install at `/opt/ros/jazzy` used only for
  introspection), gRPC C++ + protobuf toolchain, ONNX Runtime GPU C++ package, `onnx`/`onnxsim`
  in the export environment. Host GPU: RTX 2000 Ada 8 GB (sm_89) — supports CUDA EP and
  TensorRT 10.
- **Docs**: README gains a ROS2 quickstart section; benchmark table gains ROS2/C++ tiers.
