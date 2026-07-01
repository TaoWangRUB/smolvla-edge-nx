## ADDED Requirements

### Requirement: Stack smoke-test on the base model

The system SHALL provide an inference entrypoint that runs `lerobot/smolvla_base` against a
public SO-101 dataset without any training, so stack and version problems surface before any
GPU time is spent on fine-tuning.

#### Scenario: Smoke-test the base model

- **WHEN** an operator runs `python -m smolvla_edge.infer --policy-path lerobot/smolvla_base --dataset-repo-id lerobot/svla_so101_pickplace --episodes 2`
- **THEN** the policy loads and produces actions for the replayed episodes without error
- **AND** no fine-tuning or checkpoint writing occurs

### Requirement: Fine-tune an SO-101 pick-and-place checkpoint

The system SHALL fine-tune SmolVLA from `lerobot/smolvla_base` on the SO-101 pick-and-place
dataset via a thin wrapper over `lerobot-train`, driven by a versioned training config.

#### Scenario: Run the fine-tune

- **WHEN** an operator runs `bash scripts/train.sh` using `configs/train.so101_pickplace.yaml`
- **THEN** training runs for the configured step budget (~20k steps)
- **AND** checkpoints are written under `outputs/train/smolvla_so101/checkpoints/`

### Requirement: Report a held-out success rate

The system SHALL evaluate a checkpoint on held-out episodes and report a success-rate metric,
which is the correctness deliverable of the fine-tuning phase.

#### Scenario: Evaluate a checkpoint

- **WHEN** an operator runs `python -m smolvla_edge.eval --policy-path <checkpoint> --dataset-repo-id lerobot/svla_so101_pickplace`
- **THEN** the system evaluates on held-out episodes
- **AND** prints a success-rate number for the checkpoint
