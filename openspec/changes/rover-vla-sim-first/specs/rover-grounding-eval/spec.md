## ADDED Requirements

### Requirement: Closed-loop evaluation on held-out environments, defined before scaling

The evaluation protocol SHALL be defined before any data scaling and SHALL run closed-loop in
held-out simulation environments never seen in training, measuring: task success rate,
collision rate, safety-monitor intervention rate, instruction-following accuracy, robustness to
held-out paraphrases, and path quality (smoothness, tracking error, time-to-goal).

#### Scenario: Scaling requires a metric-identified deficiency

- **WHEN** a proposal is made to generate more data or unfreeze more of the model
- **THEN** it cites a specific deficiency identified by these metrics (which environment
  factor, which grounding failure), not a general desire for more

### Requirement: Swap test with attribute-sharing hard negatives

Instruction-following SHALL be measured by the swap test: in a scene with two candidate
targets, swapping the instruction must swap the behavior. Test scenes SHALL include hard
negatives sharing attributes with the goal (same color, different shape; same shape, different
color) so that single-attribute grounding fails the test.

#### Scenario: Language-grounding collapse detected

- **WHEN** a policy navigates to the same target regardless of the instruction in swap-test
  scenes
- **THEN** the swap-test score is at chance and the failure is classified as grounding
  collapse (triggering the pre-committed vision-encoder LoRA contingency in M1)

### Requirement: Failure slicing by randomization metadata

Evaluation tooling SHALL join failures against the logged per-episode randomization
configuration and report failure rates per environment factor (lighting, scene family, ground
material, distractor density, etc.).

#### Scenario: Factor-level failure report

- **WHEN** an evaluation sweep completes
- **THEN** the report identifies which environment factors correlate with failure, in the
  style of this repo's collated benchmark discipline (seeded, budgeted, honest accounting)
