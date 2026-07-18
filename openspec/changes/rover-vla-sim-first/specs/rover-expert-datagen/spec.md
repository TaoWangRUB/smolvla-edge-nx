## ADDED Requirements

### Requirement: Automated privileged expert with goal-visibility validity

An automated expert SHALL drive every training episode — an A-star route on the privileged sim
map executed by Pure Pursuit or MPC through the Ackermann model — and an episode SHALL be valid
only if the goal is visible from the onboard camera at episode start.

#### Scenario: Invisible goal rejected

- **WHEN** a sampled start/goal pair leaves the goal outside the camera's view at episode start
- **THEN** the episode is discarded or resampled and never enters the training set

### Requirement: Recovery data from the first large dataset

The pipeline SHALL inject DART-style noise into the expert's commands while recording the
expert's corrective actions from the first large dataset onward, and SHALL support a
DAgger-style pass (roll out the trained policy, relabel visited states with expert actions)
in M2.

#### Scenario: Recovery states present in training data

- **WHEN** the first large dataset is generated
- **THEN** a configured fraction of episodes contains injected command noise with the expert's
  corrections as labels, so the policy observes off-nominal states with recovery actions

### Requirement: Hindsight body-frame waypoint labels

For each recorded frame, waypoint targets SHALL be produced by transforming the expert's future
poses over the horizon into that frame's body frame, sampled at fixed Δt (0.2–0.25 s) for
K = 8–16 points with per-point target speed, normalized to approximately [−1, 1] from dataset
statistics. The same relabeling SHALL apply unchanged to noisy-recovery episodes.

#### Scenario: Labels independent of planner internals

- **WHEN** waypoint labels are generated for any episode, clean or noise-injected
- **THEN** they derive solely from logged ground-truth poses, with no dependency on the
  expert planner's internal state

### Requirement: Templated instructions with paraphrase holdout

Instructions SHALL be generated from templates tied to the sampled goal and scene, paraphrased
offline by an LLM into varied phrasings, with a held-out phrasing set reserved exclusively for
evaluation.

#### Scenario: Held-out phrasings never trained on

- **WHEN** the training and evaluation splits are constructed
- **THEN** no evaluation paraphrase appears in any training episode's instruction

### Requirement: LeRobot episode format with full metadata

Episodes SHALL be stored in LeRobot format containing: front RGB at 10–15 Hz at the real
camera's resolution, ground-truth pose at 50 Hz, (speed, yaw rate, steering angle), expert
commands, optional simulated depth (debug/auxiliary only — never a policy input), and per-episode
randomization configuration, camera intrinsics/extrinsics, collision flags, and success flags.

#### Scenario: Episodes drop into the existing training loop

- **WHEN** a generated dataset is pointed at the SmolVLA/LeRobot training entrypoint
- **THEN** training runs without format adaptation beyond the action-space definition
