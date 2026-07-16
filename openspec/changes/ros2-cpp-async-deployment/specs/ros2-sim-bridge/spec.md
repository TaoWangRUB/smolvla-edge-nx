## ADDED Requirements

### Requirement: SimEnv gRPC service inside the sim container

The system SHALL expose the gym-aloha environment through a `SimEnv` gRPC service
(`deploy/client_server/sim_server.py`, running in the `smolvla-edge:sim` container) with
`Reset`, `Step(action)` and observation retrieval, reusing the proto's `Image`/`Tensor`
messages for camera frames and joint state. The existing `Policy` service definition SHALL
remain unchanged.

#### Scenario: Rollout through the shim matches in-process eval

- **WHEN** a Python reference client replays the existing eval rollout through `SimEnv`
  (same checkpoint, env, seeds) instead of stepping the env in-process
- **THEN** the closed-loop success rate over the episode batch matches the in-process
  `smolvla_edge.eval` result

#### Scenario: Existing clients unaffected

- **WHEN** the proto gains the `SimEnv` service and stubs are regenerated
- **THEN** the existing Python `Policy` client/server pair runs unmodified

### Requirement: Driver-style ROS2 bridge node

A Python `rclpy` node in the `ros2` container SHALL map `SimEnv` onto ROS2 topics — publishing
camera images and joint state, subscribing to the action topic — acting as the stand-in for a
robot driver node.

#### Scenario: Observation topics live

- **WHEN** the bridge is launched against a running `sim-server`
- **THEN** camera image topics and `/joint_states` publish at the control rate, and each
  message carries the sim tick index for traceability

### Requirement: Bridge owns the 20 ms tick timing contract

The bridge SHALL enforce the discretization of the existing sim harness: one action per 20 ms
control tick (50 Hz); if no action arrives within a tick, the previous pose is held and the
tick is recorded as an idle tick. Sim time SHALL therefore advance by tick count, keeping
success-rate comparisons valid regardless of wall-clock jitter.

#### Scenario: Idle tick on late action

- **WHEN** the action topic delivers nothing within a control period
- **THEN** the bridge steps the sim holding the previous pose and increments the idle-tick
  counter exposed on its diagnostics topic

#### Scenario: Tick accounting matches the Python harness

- **WHEN** an async episode completes through the bridge
- **THEN** reported total ticks and idle ticks are consistent with the
  `async_infer.AsyncRunner` event model for the same trace
