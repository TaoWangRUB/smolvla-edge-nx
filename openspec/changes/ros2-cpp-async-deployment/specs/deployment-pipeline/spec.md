## ADDED Requirements

### Requirement: Single-command gated deployment chain

The system SHALL provide one entrypoint (`scripts/deploy_pipeline.sh <checkpoint>`) that runs
the deployment chain in dependency order — ONNX export, parity gate, C++ server image build,
closed-loop sim regression gate, benchmark-row generation — and SHALL exit non-zero at the
first failed gate without executing later stages. No stage SHALL require human editing between
checkpoint input and benchmark output.

#### Scenario: Green run end-to-end

- **WHEN** the pipeline runs against a known-good checkpoint
- **THEN** it completes all stages and emits the ONNX model, parity report, server image tag,
  regression result and benchmark rows in one invocation

#### Scenario: Fail-fast on a gate

- **WHEN** the parity gate exceeds tolerance
- **THEN** the pipeline exits non-zero, reports which gate failed, and the sim-regression and
  benchmark stages do not run

### Requirement: Versioned run artifacts with lineage

Each pipeline run SHALL write its artifacts under a run-stamped directory in
`benchmarks/results/pipeline/`, recording checkpoint hash, export hash, image tag and gate
outcomes, so any deployed artifact is traceable to the exact checkpoint and code that produced
it.

#### Scenario: Lineage recorded

- **WHEN** a pipeline run finishes (pass or fail)
- **THEN** its run directory contains a manifest with checkpoint/export/image identifiers and
  per-gate pass/fail status
