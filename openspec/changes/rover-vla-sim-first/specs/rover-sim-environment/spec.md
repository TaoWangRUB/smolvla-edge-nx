## ADDED Requirements

### Requirement: Simulator feasibility gate before environment work

The project SHALL verify, before any scene or rover modeling begins, that Isaac Sim with RTX
rendering runs on an available GPU, and SHALL record the go/no-go decision (local GPU, cloud,
or promotion of the Gazebo+augmentation fallback) with measured FPS and VRAM.

#### Scenario: Insufficient GPU detected

- **WHEN** Isaac Sim fails to run with RTX rendering on the available GPU (e.g., a 4 GB card)
- **THEN** the M0 gate records the failure and the chosen remedy (GPU upgrade / cloud / Gazebo
  fallback) before any milestone work proceeds, and only one simulator is maintained thereafter

### Requirement: Rover and camera fidelity to the selected real hardware

The simulated rover SHALL match the real 1/16 Ackermann vehicle's wheelbase, track width,
steering limits, and camera mount geometry; the simulated camera SHALL be locked to the
resolution and intrinsics of the real camera model selected in M0 (global-shutter pinhole,
~100–110° HFOV), with randomization applied around those nominal values.

#### Scenario: Camera selected before scenes are populated

- **WHEN** the M0 camera-selection task completes
- **THEN** the simulated camera configuration equals the selected sensor's resolution and
  intrinsics, and all subsequent recordings use that configuration

### Requirement: Scenes where language is necessary

Training and evaluation scenes SHALL include multiple candidate targets with attribute-sharing
hard negatives, such that a policy that ignores the instruction, or grounds only one attribute,
cannot reach above-chance task success.

#### Scenario: Hard negatives present

- **WHEN** an episode is generated with the goal "drive to the red cone"
- **THEN** the scene contains at least one distractor sharing color (red barrel) or shape
  (blue cone) with the goal object

### Requirement: Per-episode domain randomization with logged configuration

Each episode SHALL randomize lighting, time of day, weather, ground materials, object
placement, camera exposure and small extrinsic jitter, sensor noise, spawn pose, and goal
location, and SHALL log the full randomization configuration in the episode metadata.

#### Scenario: Failure attribution possible

- **WHEN** an evaluation failure is analyzed
- **THEN** the episode's logged randomization configuration identifies the environment factors
  present, enabling failure slicing by factor
