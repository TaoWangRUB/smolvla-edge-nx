## ADDED Requirements

### Requirement: Stack smoke-test on the base model

The system SHALL provide an inference entrypoint that runs `lerobot/smolvla_base` against a
public dataset without any training, so stack and version problems surface before any GPU time is
spent on fine-tuning.

#### Scenario: Smoke-test the base model

- **WHEN** an operator runs `python -m smolvla_edge.infer --policy-path lerobot/smolvla_base --dataset-repo-id lerobot/aloha_sim_insertion_human --episodes 2`
- **THEN** the policy loads and produces actions for the replayed episodes without error
- **AND** no fine-tuning or checkpoint writing occurs

### Requirement: Fine-tune an ALOHA-sim checkpoint

The system SHALL fine-tune SmolVLA from `lerobot/smolvla_base` on a LeRobot-native ALOHA
simulation dataset via a thin wrapper over `lerobot-train`, driven by a versioned training config,
so the correctness loop requires no physical robot.

#### Scenario: Run the fine-tune

- **WHEN** an operator runs `bash scripts/train.sh` using `configs/train.aloha_sim.yaml`
- **THEN** training runs for the configured step budget (~20k steps)
- **AND** checkpoints are written under `outputs/train/smolvla_aloha/checkpoints/`

### Requirement: Report a closed-loop success rate in simulation

The system SHALL evaluate a checkpoint by running closed-loop rollouts in the matching gym-aloha
MuJoCo environment and report the environment's success rate (reward >= 4), which is the
correctness deliverable of the fine-tuning phase. An open-loop replay proxy SHALL remain available
as a fallback when the sim is not installed, clearly labelled as a proxy rather than a true
success rate.

#### Scenario: Evaluate a checkpoint closed-loop in sim

- **WHEN** an operator runs `python -m smolvla_edge.eval --policy-path <checkpoint> --mode sim --env-id gym_aloha/AlohaInsertion-v0 --episodes 20`
- **THEN** the system runs the policy in the MuJoCo env for each episode
- **AND** counts an episode as a success when the env reward reaches 4
- **AND** prints a success-rate number over the evaluated episodes

#### Scenario: Open-loop replay fallback (no sim)

- **WHEN** an operator runs `python -m smolvla_edge.eval --policy-path <checkpoint> --mode replay --dataset-repo-id lerobot/aloha_sim_insertion_human`
- **THEN** the system compares predicted vs. logged actions on held-out frames
- **AND** prints action-agreement metrics labelled as an open-loop proxy, not a task success rate
