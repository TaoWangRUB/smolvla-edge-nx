## ADDED Requirements

### Requirement: Waypoint action space on the unchanged SmolVLA stack

The policy SHALL be SmolVLA with the action space redefined to a chunk of K × (x, y, v)
body-frame waypoints (LeRobot `chunk_size = K`, `action_dim = 3`) with a 2–3 s horizon,
consuming a front RGB frame, the tokenized instruction, and a three-element state vector
(speed, yaw rate, steering angle). Steering angle and heading SHALL NOT be predicted along the
trajectory (both are functions of the path under Ackermann kinematics).

#### Scenario: Existing serving path unchanged

- **WHEN** the trained rover checkpoint is loaded through `make_chunk_predictor`
- **THEN** chunk prediction, normalization, and async execution work without code changes
  beyond the checkpoint's own action statistics

### Requirement: Two-stage training on a single consumer GPU

Training SHALL proceed in two stages: (1) frozen vision-language backbone with only the action
expert/head trained on the initial dataset; (2) LoRA adapters on the backbone with the larger
dataset. Full fine-tuning SHALL require an evaluation-backed justification.

#### Scenario: Pipeline verified before capacity is unlocked

- **WHEN** stage one completes on the ~500–1,000-episode dataset
- **THEN** end-to-end closed-loop rollouts run before any backbone parameters are unfrozen

### Requirement: Pre-committed contingency ladder

The project SHALL apply contingencies in this order, without improvisation: vision-encoder LoRA
pulled into M1 if the swap test fails under a frozen backbone; action-expert re-initialization
(pretrained VLM kept) if the manipulation-pretrained flow head converges poorly; a
curvature-velocity (κ, v) output parameterization if M1 rollouts show tracking oscillation;
Qwen2.5-VL plus a diffusion-style action head as the architecture fallback. A plain MSE
regression head SHALL NOT be used under any fallback.

#### Scenario: Multimodal maneuvers preserved

- **WHEN** an obstacle admits both a left and a right pass in evaluation
- **THEN** the deployed head (flow-matching or diffusion) commits to one mode rather than
  averaging through the obstacle
