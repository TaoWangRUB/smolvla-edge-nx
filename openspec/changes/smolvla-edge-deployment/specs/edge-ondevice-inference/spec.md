## ADDED Requirements

### Requirement: Run the policy on-device within 8 GB

The system SHALL run the fine-tuned policy entirely on a Jetson Xavier NX (8 GB) without
exhausting memory, using FP16 as the default precision.

#### Scenario: On-device sanity check

- **WHEN** an operator runs `python -m smolvla_edge.infer --policy-path <checkpoint> --episodes 1 --max-frames 10` on the NX
- **THEN** the policy loads and executes the frames on the NX GPU
- **AND** peak memory stays within the 8 GB budget (with FP16, and swap/zram as fallback)

### Requirement: Action chunking with decoupled execution

The system SHALL support action chunking that decouples prediction from execution, predicting a
chunk of actions and executing it open-loop while the next chunk is computed.

#### Scenario: Enable chunking

- **WHEN** an operator runs a benchmark with `--chunking on`
- **THEN** the policy predicts action chunks and executes them without blocking on each step's
  prediction
- **AND** measured task time is lower than the equivalent `--chunking off` run

### Requirement: Low-Hz VLM stage

The system SHALL allow the SmolVLM-2 backbone to run at a lower rate than the action expert /
control loop, so the expensive vision stage does not gate the control frequency.

#### Scenario: VLM runs below control rate

- **WHEN** on-device inference is configured with a reduced VLM rate
- **THEN** the action expert / control loop runs at its full rate between VLM updates

### Requirement: INT8 only via a real engine

The system SHALL reject naive INT8 casts and permit INT8 only through real quantization or a
TensorRT engine on the subgraphs that convert.

#### Scenario: Naive INT8 is refused

- **WHEN** an operator requests INT8 without a real quantized/TensorRT engine
- **THEN** the benchmark harness refuses the run rather than reporting a naive-cast number

### Requirement: Honest conversion notes

The system SHALL record which subgraphs converted to a faster engine, which did not and why, and
the per-stage latency budget, in `deploy/ondevice/conversion_notes.md`.

#### Scenario: Document a conversion attempt

- **WHEN** a TensorRT/quantization conversion is attempted on part of the graph
- **THEN** the outcome (converted + speedup, or not-converted + reason + fallback) and the
  per-stage latency breakdown are recorded in the conversion notes
