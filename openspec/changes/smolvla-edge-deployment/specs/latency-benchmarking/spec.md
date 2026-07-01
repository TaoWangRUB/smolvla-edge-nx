## ADDED Requirements

### Requirement: Uniform benchmark workload across tiers

The system SHALL run the same workload across every deployment tier — Titan X local, NX
on-device (precision × chunking sweep), and NX-client / workstation-server — via
`smolvla_edge.bench` and the gRPC client, dropping per-run JSON into `benchmarks/results/raw/`.

#### Scenario: Produce an on-device benchmark row

- **WHEN** an operator runs `python -m smolvla_edge.bench --policy-path <ckpt> --device cuda --precision fp16 --chunking on --tag nx-fp16-chunk --out benchmarks/results/raw/nx_fp16_chunk.json`
- **THEN** a JSON result file is written with the run's metrics and tag

### Requirement: Record end-to-end latency, throughput, and memory

Each benchmark run SHALL record end-to-end latency (mean and p95), action-chunk frequency,
throughput, and peak memory, and SHALL record the JetPack version and power mode on the NX so
results stay comparable across runs.

#### Scenario: Metrics captured per run

- **WHEN** a benchmark run completes
- **THEN** its JSON contains latency mean, p95, throughput, and peak memory
- **AND** NX runs additionally record the JetPack version and power mode

### Requirement: Collate results into a reproducible table

The system SHALL collate the raw JSON into `results/summary.csv` and a regenerated markdown
results table via `benchmarks/collate.py`.

#### Scenario: Regenerate the table

- **WHEN** an operator runs `python benchmarks/collate.py`
- **THEN** `results/summary.csv` and the markdown results table are regenerated from the raw JSON

### Requirement: Demo media of replayed episodes

The system SHALL produce a short GIF/video of the policy executing replayed episodes (no physical
robot) via `scripts/make_demo_gif.py`, for embedding in the top-level README.

#### Scenario: Build the demo GIF

- **WHEN** an operator runs `scripts/make_demo_gif.py` on replayed episodes
- **THEN** a GIF/video artifact is produced for the writeup
