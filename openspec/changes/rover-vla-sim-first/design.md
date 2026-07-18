# Design — Ackermann Rover VLA, Simulation-First

Decisions carry D-numbers; contingencies are pre-committed so mid-milestone surprises have a
scripted response instead of a redesign. Refinements over the original draft plan are marked
**[refined]** and are grounded in measured results from the ALOHA→NX project in this repo.

## D1 — Simulator: Isaac Sim + Isaac Lab, behind an explicit feasibility gate [refined]

For a vision policy the sim-to-real gap lives almost entirely in the pixels: RTX rendering,
Replicator domain randomization, native Ackermann support, and the ROS 2 bridge outweigh
Gazebo's lighter footprint. The practical costs are (a) GPU contention — Isaac Sim and training
compete for VRAM, so data generation and training are time-shared with headless overnight
generation as the default; and (b) **Isaac Sim will not run on the current 4 GB dev GPU** —
RTX rendering realistically wants ≳8–12 GB. M0 therefore *opens* with a feasibility gate:
secure a capable GPU (local or cloud) or promote the documented fallback — Gazebo + aggressive
image augmentation, accepting a larger transfer gap. The project never maintains both
simulators in parallel (double cost of rover model, controllers, recording stack).

The rover is a 1/16 Ackermann vehicle in USD/URDF with the real wheelbase, track width,
steering limits, and camera mount geometry. The simulated camera is locked to the *selected*
real camera's resolution and intrinsics, and randomization is applied around those nominals.

**Camera [decided early, purchased late]**: single forward pinhole, ~100–110° horizontal FOV,
**global shutter** with hardware timestamping or external trigger (OV9281-class,
Jetson-friendly). No fisheye (out of distribution for pretrained encoders), no stereo (a
learned policy cannot consume the second RGB view directly; monocular cues suffice at rover
speeds). At 1/16 scale, 2–3 m/s is highway-equivalent optical flow — rolling shutter would add
tearing that sim cannot cheaply reproduce. The known residual gap (exposure motion blur, absent
from Isaac's crisp renders) is closed by a **velocity-scaled directional blur augmentation** at
training time (kernel from each frame's logged speed and yaw rate) — nearly free, tunable
without regenerating data.

Scenes: parking lots, dirt/grass, forest trails, warehouses, corridors, narrow passages;
props: cones, barrels, gates, trees, signage, pedestrians. Per-episode randomization: lighting,
time of day, weather, ground materials, object placement, exposure, small extrinsic jitter,
sensor noise, spawn pose, goal location. Scenes are constructed so **language is necessary**:
multiple candidate targets per scene with attribute-sharing hard negatives (red cone beside a
red barrel → shape must ground; beside a blue cone → color must ground).

## D2 — Action space: K × (x, y, v) body-frame waypoints; no heading channel

The policy emits a chunk of K = 8–16 waypoints at fixed Δt (0.2–0.25 s), each (x, y, v) in the
body frame at prediction time, 2–3 s horizon. In LeRobot terms: `chunk_size = K`,
`action_dim = 3` — the flow-matching expert, `make_chunk_predictor`, and normalization
machinery are unchanged; only the meaning and statistics of the chunk change.

Steering angle and heading are deliberately **not** predicted: for Ackermann kinematics both
are functions of the path (δ = atan(L·κ), θ = path tangent); a separate channel would be
redundant and could contradict the (x, y) sequence. Non-holonomic feasibility is handled by
construction — every training trajectory comes from an Ackermann expert, so infeasible lateral
motion is never observed — and the tracker projects residual infeasibility onto the constraint
set. **Contingency (pre-committed)**: if M1 rollouts show tracking oscillation, switch the
output parameterization to a **curvature-velocity sequence (κ, v)** integrated through the
kinematic model (feasible by construction) — do not append a heading channel. Heading/signed
velocity enters only when reversing/parking is in scope (path tangent directionally ambiguous);
that is also when a rear camera becomes worth logging (a forward-only policy cannot see where
it backs up).

## D3 — Runtime: three nested loops; trust allocation follows sensor allocation

- **Control loop, 50–100 Hz**: Pure Pursuit initially, kinematic MPC later, tracking the newest
  chunk on **relative odometry** from an EKF (IMU + wheel encoders). A chunk lives 2–3 s, so
  odometry drift over its lifetime is negligible — no global localization. This layer enforces
  hard limits (max speed, steering rate, min turn radius) and accepts the safety veto.
- **Policy loop, 4–10 Hz**: async in the SmolVLA style — the tracker continues on the current
  chunk while the next computes; each new chunk **replaces** the queue after latency
  compensation by the odometry delta since image capture. Only the first 100–300 ms of a chunk
  is typically executed; the tail is a jitter buffer. Watchdog: chunk age > ~1 s ⇒ ramp speed
  to zero (extends this repo's chunk-staleness accounting).
- **Mission loop, 0.1–1 Hz, M4-optional**: SLAM + global planner emitting sub-goals through the
  instruction interface only; it never touches the control path.

Sensor asymmetry is deliberate: camera → policy; IMU → EKF only (never the policy); depth/2D
lidar → **safety monitor only** — a hand-written geometric corridor check that caps speed or
stops, independent of anything the network outputs.

**[refined] The runtime is the proven all-ROS2 shape from this repo**: the policy runs as a DDS
node (the `policy_node.py` pattern — request/reply topics, request-id correlation, reliable
QoS), the C++ `async_client` lineage becomes the chunk manager feeding the tracker, and two
measured lessons carry over verbatim: **pre-warm the CUDA-graph capture at node startup** (the
all-ROS2 run paid 414 idle ticks in episode 0 for lazy capture) and **compress the image topic**
(900 KiB raw frames over reliable DDS were the steady-state latency lever).

## D4 — Data: automated expert + recovery data from the start

A privileged expert drives every episode: A* on the sim map → Pure Pursuit/MPC through the
Ackermann model. Episode validity requires the goal visible from the onboard camera at start —
the policy is never trained on tasks its observation cannot solve.

Clean demonstrations are insufficient (a cloned policy has never seen a recovery state):
**DART-style noise injection** on the expert's commands while recording the expert's corrective
actions ships with the first large dataset; a **DAgger pass** (roll out the trained policy,
relabel its states with expert actions) lands in M2. This is expected to matter more for
closed-loop success than an order of magnitude more clean data.

Waypoint targets by **hindsight relabeling**: each frame's future expert poses transformed into
that frame's body frame, sampled at Δt, per-point target speed attached — no planner internals,
identical for noisy-recovery data. Instructions: templates tied to sampled goal/scene, LLM
paraphrasing offline, a held-out phrasing set reserved for evaluation.

Logged per timestep: front RGB (10–15 Hz, real camera resolution), ground-truth pose (50 Hz —
the source of all labels), speed/yaw-rate/steering, expert commands, optional depth (debugging
and auxiliary losses only). Per episode: full randomization config, intrinsics/extrinsics,
collision and success flags — the metadata that later answers "which factors break the policy."
Budget in episodes, not hours: verify on ~500–1,000 episodes (30 s–2 min each, ~10–20 h), scale
only against a demonstrated deficiency (D6).

## D5 — Model: SmolVLA with pre-committed contingency ladder

Two-stage training on a single consumer GPU (4090-class sufficient; stage one runs on less):
(1) freeze the VLM backbone, train the action expert on the initial dataset — verifies the
pipeline end-to-end; (2) LoRA on the backbone with the larger dataset. Full fine-tune only with
an evaluation-backed reason.

Contingencies, in escalation order:
1. **Vision-encoder LoRA pulled forward into M1** if the swap test fails under a fully frozen
   backbone (plausible — the encoder has never seen Isaac-rendered textures); language model
   stays frozen.
2. **Re-initialize the action expert from scratch** (keep the pretrained VLM) if the
   manipulation-pretrained flow head converges poorly — removes the unlearning burn-in at
   negligible cost.
3. **(κ, v) output parameterization** per D2 if tracking oscillates.
4. **Qwen2.5-VL + diffusion-style action head** as the architecture fallback. Waypoint
   distributions are genuinely multimodal (pass left vs right) — an argument *for*
   flow-matching/diffusion heads and a hard rule against plain MSE regression (mode-averaging
   cuts through the obstacle). A regression head is never an allowed fallback.

## D6 — Evaluation before scaling: the swap test is the centerpiece

All evaluation closed-loop in held-out environments: task success; collision and
safety-intervention rates; **swap test** (two candidate targets in scene — swapping the
instruction must swap the behavior; hard negatives share attributes so single-attribute
grounding fails); held-out paraphrase robustness; path quality (smoothness, tracking error,
time-to-goal); failure slicing by randomization metadata. Decision rule: scale data or unfreeze
model capacity only when a metric identifies the specific deficiency. This is the grounding
analogue of the repo's existing eval discipline (protocolized seeds, fixed budgets, honest idle
accounting), and reuses its harness patterns.

## D7 — Deployment: measured Xavier NX baseline; Orin as upgrade, not prerequisite [refined]

The draft plan targeted Orin. This repo has since **measured** the on-hand Xavier NX:
fp16 + full-forward CUDA-graph capture = 233 ms/chunk (bitwise-exact), 4.3 chunks/s — against a
2–3 s chunk horizon that is ~10× replan overlap, comfortably inside the async operating
envelope (ℓ/Δt ≪ g·n) derived and verified in the ALOHA project. The rover policy is *more*
capture-friendly than ALOHA's (fixed camera resolution, padded instruction tokens, K×3 chunk —
all static shapes). Deployment therefore baselines on the NX with `precision="fp16-graph"`;
Orin NX/AGX is the upgrade if evaluation ever demands 5–10 Hz replanning or a larger backbone.
The 256M-backbone result (196 ms graphed) is the pre-measured speed lever if a smaller VLM
proves sufficient for navigation.

**Timestamping discipline (hard requirement before trusting latency compensation)**: userspace
arrival time is not capture time — 50 ms of timestamp error at 3 m/s misplaces the body frame
by 15 cm. Real stack: camera shutter time from the sensor (or external trigger synced to the
IMU clock), EKF pose **interpolated from state history at shutter time**, and an M3 bench
measurement of true capture-to-userspace latency (blinking-LED test) before any compensation
value is trusted. First real runs: controlled space, reduced speed caps, depth/lidar monitor +
physical e-stop active, real-vs-sim metric comparison to quantify the transfer gap before
widening the envelope.

## D8 — Reuse inventory from this repo [refined]

| Asset | Reuse |
|---|---|
| LeRobot dataset format + training loop | rover episodes drop in unchanged (D2 action redefinition only) |
| `async_infer.AsyncRunner` + Algorithm-1 semantics | policy-loop chunk management; staleness watchdog basis |
| `cuda_graph.py` (`fp16-graph`) | NX serving at 233 ms; pre-warm at node startup |
| all-ROS2 `policy_node.py` + `async_client transport:=ros2` | the rover runtime's serving shape |
| `scripts/deploy_pipeline.sh` gate pattern | rover pipeline: datagen sanity → train → closed-loop gate → collate |
| eval protocol discipline (seeded, budgeted, metadata-sliced) | grounding eval (D6) |
| `smolvla-jetson:{jp5-cu118,humble}` images | rover Jetson runtime base |

## §10 — Upgrade path toward full-scale AD (recorded, out of scope)

The stack is a miniature of the consensus dual-system AD architecture; scaling is three
subsystem upgrades around a nearly invariant policy contract:

- **Perception**: one camera → surround **BEV fusion** (Lift-Splat or BEVFormer-style queries →
  ego-centric metric grid, ~200×200 over ±50 m, tokenized into the VLM à la OmniDrive's
  Q-Former). BEV preserves the ego-centric metric frame the waypoint chunks already live in,
  accumulates temporally (occlusion, implicit velocity), and is the natural lidar late-fusion
  point (BEVFusion). New cost: multi-camera calibration + online miscalibration detection.
- **Instruction source**: templates → **navigation router** (graph search over a road network,
  GNSS-conditioned), emitting sub-goals as command enums, BEV-projected route polylines, or
  free-form text (EMMA-class; where VLM language pretraining pays off). The swap test scales
  into **route-compliance evaluation** — executing the commanded turn, not the salient one.
- **Safety**: single-sensor veto → **formally specified envelope** (RSS / Safety Force Field):
  radar+lidar with different physics from cameras, provably-safe acceleration sets clamping
  every trajectory, a graduated degradation ladder ending in a maintained minimal-risk
  maneuver, on separate compute (ASIL-D under ISO 26262; learned stack under ISO 21448/SOTIF).
  This is the industrial form of "trust allocation follows sensor allocation."
- **Invariant contract**: RGB→BEV tokens (still visual tokens); template→router text (still
  instruction tokens); 3 state scalars→kinematic-history token; K×(x,y,v) body-frame chunk→
  BEV-frame trajectory with confidences (still short-horizon spatial intent, receding-horizon,
  executed by a controller that owns feasibility and safety). Lidar/radar stay outside the
  policy by default (direct BEV tokenization noted as a research frontier that retains the
  independent safety path).

## Open Questions

- Isaac Sim compute: which GPU (local purchase vs cloud) if the M0 gate rejects the current
  4 GB card — decided at M0, not before.
- K and Δt final values (start K=12, Δt=0.25 s → 3 s horizon; tune against tracker behavior
  in M1).
- Whether the 256M backbone suffices for navigation (would ship 196 ms on the NX) — decide on
  M2 evaluation, not speculation.
- Rear camera + reversing scope: log rear RGB in sim from M2 if parking tasks are wanted in
  M4; otherwise omit.
