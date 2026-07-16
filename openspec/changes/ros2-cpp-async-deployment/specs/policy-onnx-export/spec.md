## ADDED Requirements

### Requirement: Monolithic ONNX export of the fine-tuned checkpoint

The system SHALL export the fine-tuned SmolVLA checkpoint to a single ONNX graph covering
vision encoding, the fixed-task language prefix (token ids baked in at export time) and the
action expert with its 10 flow-matching Euler steps unrolled. The export SHALL run inside the
`smolvla-edge:sim` container (lerobot 0.4.4, checkpoint-era preprocessing) and SHALL bake input
normalization into the graph so the serving side feeds raw resized images and joint state only.
The denoising noise SHALL be an explicit graph input.

#### Scenario: Export produces a loadable graph

- **WHEN** the export script runs against the fine-tuned checkpoint
- **THEN** it writes `models/onnx/<checkpoint>.onnx` (with external weights if > 2 GB) that
  loads under ONNX Runtime and reports the expected input/output shapes
  (`[horizon, action_dim]` chunk output)

#### Scenario: Fixed task baked in

- **WHEN** the exported graph is invoked
- **THEN** no tokenizer or text input is required at inference time; the task instruction used
  at export is recorded in the graph metadata

### Requirement: Enforced numerical parity gate

A parity harness SHALL compare exported-graph outputs against the PyTorch policy on ≥ 100
held-out observations with a fixed noise seed, and SHALL FAIL (non-zero exit) unless
max-abs-diff ≤ 1e-4 AND cosine similarity ≥ 0.9999 on every action chunk (FP32 baseline). FP16
variants SHALL additionally be gated on closed-loop success rate, not tensor diffs alone.

#### Scenario: Parity gate blocks a bad export

- **WHEN** an export exceeds the tolerance on any observation
- **THEN** the harness exits non-zero and names the failing observation and metric (a computed
  but unenforced PASS/FAIL is a spec violation)

#### Scenario: Parity report artifact

- **WHEN** the harness passes
- **THEN** it writes a report (tolerances, per-observation max diff, checkpoint and export
  hashes) under `benchmarks/results/` for provenance
