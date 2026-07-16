## ADDED Requirements

### Requirement: Containerized ROS2 Jazzy runtime

The system SHALL provide a ROS2 Jazzy image (`smolvla-edge:ros2`, built from
`docker/ros2.Dockerfile`) that carries the colcon-built `deploy/ros2/` workspace toolchain
(rclcpp, rclpy, gRPC C++, protobuf) and runs all ROS2 nodes of this change. The existing
`smolvla-edge:sim` image SHALL NOT change its Python stack (py3.11 / mujoco 2.3.7 /
lerobot 0.4.4); it may only gain new entrypoints.

#### Scenario: ROS2 container smoke test

- **WHEN** an operator runs `docker compose run --rm ros2 ros2 doctor`
- **THEN** the check completes without errors inside the container, with no ROS2 dependency on
  the host install

#### Scenario: Sim image stack unchanged

- **WHEN** the sim image is rebuilt after this change
- **THEN** `python -c "import mujoco, lerobot"` inside it reports mujoco 2.3.7 and
  lerobot 0.4.4, and existing compose services (`eval`, `train`, `bench`) run as before

### Requirement: Compose topology with gRPC-only container boundaries

`docker-compose.yml` SHALL define the services `sim-server`, `policy-server` and `ros2` on a
shared network, where the only inter-container protocols are the `SimEnv` and `Policy` gRPC
services. DDS traffic SHALL remain inside the `ros2` container.

#### Scenario: Services reachable by name

- **WHEN** the `ros2` container resolves `sim-server:<port>` and `policy-server:<port>`
- **THEN** both gRPC `Health`/readiness probes succeed over the compose network

#### Scenario: No cross-container DDS

- **WHEN** ROS2 nodes are running in the `ros2` service
- **THEN** `ros2 node list` executed in any other container discovers no nodes (DDS is confined
  to the `ros2` container)

### Requirement: GPU access where and only where needed

Services performing inference or EGL rendering SHALL request the NVIDIA runtime; the Stage 1
ROS2 control nodes SHALL run without GPU access.

#### Scenario: Stage 1 runs GPU-free in the ROS2 container

- **WHEN** the Stage 1 stack runs (`sim-server` + `policy-server` + `ros2`)
- **THEN** only `sim-server` (EGL) and `policy-server` (CUDA) hold GPU resources, verified via
  `nvidia-smi` process listing
