## ADDED Requirements

### Requirement: C++ Policy server over ONNX Runtime CUDA EP

The system SHALL provide a C++ gRPC server implementing the existing `Policy` service
(`PredictChunk`, `Reset`, `Health`) that runs the exported ONNX graph on the ONNX Runtime CUDA
execution provider. It SHALL be wire-compatible with the Python server: any existing client
works against either server unmodified. `Health` SHALL report device, precision and the loaded
model path, and a TensorRT EP mode SHALL be selectable by flag with a persistent engine cache.

#### Scenario: Drop-in wire compatibility

- **WHEN** the Stage 1 ROS2 client (or the Python reference client) points at the C++ server
  instead of the Python server
- **THEN** `PredictChunk` returns action chunks of identical shape and dtype semantics with no
  client changes

#### Scenario: TensorRT EP opt-in

- **WHEN** the server starts with the TensorRT EP flag and a cached engine exists
- **THEN** it serves without engine rebuild, and `Health` reports the active provider

### Requirement: Compose-profile server swap for A/B evaluation

`docker-compose.yml` SHALL expose the Python and C++ servers as mutually exclusive profiles
backing the same address, so the serving backend is selected by one flag and the two servers
never contend for the 8 GB GPU simultaneously.

#### Scenario: One-flag backend swap

- **WHEN** the operator switches between the Python and C++ server profiles
- **THEN** the ROS2 stack runs unchanged against either backend, and only one server process
  appears in the GPU process list

### Requirement: Closed-loop equivalence and benchmarked latency

With the parity-gated export, the C++ server SHALL match the Python server's closed-loop
gym-aloha success rate (same seeds, ≥ 50 episodes, within binomial noise), and its
client-observed `PredictChunk` round-trip latency and VRAM footprint SHALL be added to the
benchmark table with measured provenance, reported honestly whichever direction they land.

#### Scenario: Stage 2 closed-loop gate

- **WHEN** the full ROS2 stack runs the episode batch against the C++ server
- **THEN** success rate matches the Python-server baseline within binomial noise, and the
  benchmark table gains ROS2+py-server and ROS2+cpp-server rows
