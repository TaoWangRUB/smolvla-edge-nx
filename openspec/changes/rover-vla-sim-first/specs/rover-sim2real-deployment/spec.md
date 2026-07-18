## ADDED Requirements

### Requirement: Hardware purchase gated on M2 exit criteria

No rover, Jetson, IMU, or depth-sensor purchase SHALL occur before the M2 exit criteria pass
(success and collision metrics on held-out scenes, failure modes characterized). Camera model
*selection* is exempt and required in M0 so the simulated camera can match it.

#### Scenario: Gate enforced

- **WHEN** M2 evaluation has not yet met its targets
- **THEN** M3 procurement does not begin, and effort goes to the metric-identified deficiency

### Requirement: Deployment baseline is the measured Xavier NX fp16-graph path

The Jetson deployment SHALL baseline on the on-hand Xavier NX using
`make_chunk_predictor(precision="fp16-graph")` (measured in this repo: 233 ms/chunk
bitwise-exact ⇒ 4.3 chunks/s against a 2–3 s chunk horizon), with capture pre-warmed at node
startup. A Jetson Orin SHALL be adopted only if evaluation demonstrates a need the NX cannot
meet (e.g., ≥5–10 Hz replanning or a larger backbone).

#### Scenario: Chunk rate verified on device

- **WHEN** the rover checkpoint is deployed on the NX
- **THEN** measured chunk rate is ≥ 4 Hz with numerical parity against the training checkpoint
  before any real-world driving

### Requirement: Hardware timestamping before trusted latency compensation

The real stack SHALL use hardware-level timestamps: camera shutter time from the sensor (or an
external trigger synced to the IMU clock), and the EKF pose interpolated from its state history
at shutter time. A bench measurement of true capture-to-userspace latency (blinking-LED test)
SHALL pass before any latency-compensation value is trusted.

#### Scenario: Userspace timestamps rejected

- **WHEN** only frame-arrival (userspace) timestamps are available for the camera
- **THEN** latency compensation is not enabled (a 50 ms timestamp error at 3 m/s misplaces the
  body frame by 15 cm) and the timestamping task blocks real-world runs

### Requirement: Controlled-space transfer-gap quantification

First real-world runs SHALL occur in a controlled space at reduced speed caps with the
depth/lidar safety monitor and a physical e-stop active, and SHALL compare real closed-loop
metrics against the simulation baseline to quantify the transfer gap before the operating
envelope widens.

#### Scenario: Transfer gap documented

- **WHEN** the first controlled-space evaluation completes
- **THEN** a real-vs-sim metric comparison (success, collision, interventions, path quality)
  is recorded, and envelope expansion is justified against it
