## ADDED Requirements

### Requirement: Three-loop runtime with rate separation

The runtime SHALL run three nested loops: a 50–100 Hz control loop (Pure Pursuit initially,
kinematic MPC later) tracking the newest chunk on EKF relative odometry (IMU + wheel encoders)
and enforcing hard limits (max speed, steering rate, min turn radius); a 4–10 Hz asynchronous
policy loop; and an optional 0.1–1 Hz mission loop (M4) that communicates exclusively through
the instruction interface and never touches the control path.

#### Scenario: No global localization dependency

- **WHEN** the tracker follows a chunk over its 2–3 s lifetime
- **THEN** only relative odometry is consumed; no SLAM or global pose is required in phases 1–2

### Requirement: Latency-compensated asynchronous chunk execution

The policy loop SHALL execute asynchronously in the SmolVLA style: the tracker continues on the
current chunk while the next computes; each arriving chunk SHALL replace the queue after being
transformed by the odometry delta accumulated since its trigger image's capture time.

#### Scenario: Chunk arrival mid-execution

- **WHEN** a new chunk arrives while the tracker is mid-way through the current chunk
- **THEN** the new chunk is shifted into the current body frame using the odometry delta since
  image capture and replaces the queue without stopping the vehicle

### Requirement: Staleness watchdog with speed ramp

A watchdog SHALL monitor the age of the newest chunk and ramp vehicle speed to zero when the
age exceeds approximately 1 s.

#### Scenario: Inference stall

- **WHEN** chunk prediction stalls or the policy node dies
- **THEN** the vehicle ramps to a stop within the watchdog window without safety-monitor
  involvement

### Requirement: Network-independent geometric safety veto

Depth or 2D lidar SHALL feed only a hand-written geometric safety monitor that caps speed or
stops the vehicle when an obstacle enters the drive corridor. The veto SHALL be independent of
all neural-network outputs, and raw IMU SHALL never reach the policy (EKF-fused state only).

#### Scenario: Obstacle inside the corridor

- **WHEN** the depth/lidar monitor detects an obstacle within the corridor ahead
- **THEN** speed is capped or the vehicle stops regardless of the current chunk's content

### Requirement: All-ROS2 serving shape with pre-warmed capture

The policy SHALL be served as a ROS 2 node over DDS (request/reply topics with correlation, in
the pattern of `deploy/ros2/policy_node.py`), with the CUDA-graph capture pre-warmed at node
startup so the first mission chunk pays no capture latency, and with a compressed image topic
when the loop crosses hosts.

#### Scenario: First chunk at full speed

- **WHEN** the policy node reports ready and the first observation arrives
- **THEN** the first chunk returns at steady-state latency (no lazy-capture stall)
