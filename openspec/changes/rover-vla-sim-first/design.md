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

**Gate outcome (2026-07-18): Gazebo fallback promoted.** No available GPU can run Isaac Sim
(A2000 4 GB below the VRAM floor; Titan X 12 GB lacks RT cores; cloud RTX rejected on recurring
cost). Gazebo Harmonic measured on the 4 GB card: RTF ≈ 1.0 with the locked 1280×800 camera
rendering at 15 Hz, ~250 MiB VRAM. The simulator is the simplified `rover_sim` package in this
repo (extracted from the `ackermann_rover_humble` digital twin, PX4/ArduPilot coupling
removed); the "+ aggressive image augmentation" half of the fallback becomes a training-time
obligation (M1's velocity-scaled blur, plus stronger appearance randomization to compensate the
plainer renderer).

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

**On-hand hardware [confirmed 2026-07-20].** The `ackermann_rover_humble` platform is already
built and bring-up-complete: 1/16 Ackermann chassis with PX4 actuation (steering servo + ESC),
Xavier NX, D435i (color RGB + IR-stereo depth + IMU), T265 (mono global-shutter fisheye + IMU),
cuVSLAM/VINS odometry, EKF, RPLidar, CubePilot. So M3 is *not* a hardware buy-in — the depth,
IMU, lidar and odometry the safety monitor and EKF need all exist. The one policy-facing gap is
the forward RGB camera: the policy needs **color + global-shutter + rectilinear + ~100° FOV**,
and no on-hand camera has all four (D435i color is rolling-shutter and ~69° HFOV; T265 is mono
fisheye — ruled out by D1). The OV9782-class module (task 1.2) fills exactly that gap for ~$40.

**Camera bridge (zero purchase, degraded).** For a first real-world transfer test before the
OV9782 arrives, the D435i color stream (`/d435i/color/image_raw`) can feed the policy — but the
sim/real match then breaks (sim camera is locked to OV9782's 100° HFOV / fx=537). Re-lock the
simulated camera to D435i-color intrinsics for that specific run and treat the rolling-shutter
motion artifacts as an extra transfer-gap term; keep the OV9782 as the final-config sensor.

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

- ~~Isaac Sim compute: which GPU (local purchase vs cloud) if the M0 gate rejects the current
  4 GB card — decided at M0, not before.~~ **Resolved 2026-07-18**: gate rejected all options;
  Gazebo fallback promoted (see D1 gate outcome).
- K and Δt final values (start K=12, Δt=0.25 s → 3 s horizon; tune against tracker behavior
  in M1).
- Whether the 256M backbone suffices for navigation (would ship 196 ms on the NX) — decide on
  M2 evaluation, not speculation.
- Rear camera + reversing scope: log rear RGB in sim from M2 if parking tasks are wanted in
  M4; otherwise omit.

## D9 — Goal memory + acquisition: split grounding from control [added 2026-07-21]

**Context (what M1 measured).** Five interventions failed to lift the swap test above chance —
data-confound fix, frozen backbone, full vision unfreeze, constrained vision adaptation, and an
un-truncated 32-layer LM. Trace analysis then showed *why the metric could not move*: the policy
approaches **some** prop precisely (0.32–0.60 m in 8/10 runs) but **never heads toward the
commanded one**, and when the target is far it goes to the **nearest** prop in 4/5 cases. The
expert scores 10/10 on the same scenes. The earlier "colour binding is broken" reading was
therefore **under-determined** — a proximity-driven policy and a colour-blind one produce
identical swap scores.

**Root cause.** SmolVLA here is a *local, memoryless* controller doing *long-horizon goal
selection*: `n_obs_steps=1` (and SmolVLA's modeling code never reads that field at all, so frame
history is unavailable without model surgery), chunk horizon = K·Δt = 2.5 s ≈ **1.25 m**, against
goals **2–7 m** away. Once the goal leaves the ~100° FOV **the input contains no information
about it**. D3 already assigned goal persistence to the mission loop; deferring that loop to
M4-optional left the policy doing the mission layer's job.

**Decision.** Persist the goal in **map/state, not policy weights**, and split the loops as D3
intended:

1. **Acquisition (slow, ~0.1–1 Hz)** — an **open-vocabulary detector** grounds the instruction to
   a bounding box: OWL-ViT via **NanoOWL** (TensorRT-optimised for Jetson; lowest measured edge
   latency) as the deployment target, Grounding DINO for accuracy or YOLO-World for throughput as
   alternatives. Zero-shot, no training, and it emits *spatial* output rather than text to parse.
2. **Geometry** — bbox → body-frame position. On hardware the **D435i depth** (already owned)
   gives the point directly. In sim there is no depth on the VLA camera, so **ground-plane
   projection** is used instead: camera height 0.15 m, pitch 0, HFOV 100°, 1280×800, props resting
   on the ground ⇒ ray-cast the bbox *bottom edge* to the ground plane. Accurate enough at 2–3 m.
3. **Memory** — `rover_runtime/goal_memory_node.py` stores the goal in the **odom frame** and
   republishes body-frame range/bearing as the rover moves, so it stays correct out of view. It
   emits the policy's own `/waypoint_chunk` format so the existing tracker consumes it unchanged.
4. **Control (fast, 50 Hz)** — the existing Pure-Pursuit tracker, unchanged.

**Rejected, with reasons.**
- *Qwen2.5-VL + diffusion head as the policy backbone (D5 contingency 4)* — breaks deployment.
  Published Jetson Xavier deployment of Qwen2.5-VL-3B: **12.1 GB peak, 28.5 s inference**; the
  Xavier NX has 8 GB. Even AGX Orin users report very low token rates. A grounding model belongs
  in the *slow* loop, where NanoOWL fits and a 3B VLM does not.
- *Frame history (`n_obs_steps>1`)* — unavailable; SmolVLA ignores the field.
- *Semantic-map methods (VLMaps, VLFM)* — they solve a **harder** problem (goal not visible,
  frontier exploration). D4 guarantees the goal is visible at episode start, so one-shot grounding
  suffices. Deferred to the M4 beyond-line-of-sight case.

**Deployment check (measured, so the decision stays honest).** Un-truncating the LM costs
450M → **706M** params and 220 → **383 ms** on the Titan X (fp32), extrapolating to ~**405 ms**
(2.5 chunks/s) on the NX versus the measured 233 ms (4.3 chunks/s) baseline — degraded but still
~6× replan overlap against the 2.5 s chunk, and ~1.4 GB at fp16 on an 8 GB board. So the deep-LM
variant remains deployable; the 3B VLM does not.

**Consequence.** The VLA's role narrows to **local reactive control**; goal selection moves to the
mission layer. This is the dual-system split design §10 already anticipated (DriveVLM/EMMA
lineage), arrived at empirically rather than by analogy.

**Validation order.** (a) ground-plane projection + unit tests; (b) **privileged-goal integration
test** (`/goal_memory/set_odom`) — isolates memory+tracker before adding detector error, and is
the "just tell it the goal position" fallback; (c) swap in the detector for real acquisition.

### D9 validation — acquisition path measured offline (2026-07-21)

Step (a) of the D9 validation order is **done, and the result is decisive**. The full
acquisition path (open-vocabulary detector → ground-plane projection → body-frame goal) was
run on 59 recorded `rover_vla_v4` episodes and scored against ground truth taken from
`scene_config.json` (prop world position + spawn pose ⇒ exact body-frame goal). No sim, no
ROS — so this ran while datagen held the simulator.

| metric | acquisition path | SmolVLA policy (M1) |
|---|---|---|
| commanded phrase detected | **59/59** | — |
| **selects the commanded prop** | **58/59 (98%)** | ≈ chance (swap-flip 0.05–0.27) |
| position error (median / p90) | **0.13 m / 0.35 m** | ends 0.32–0.60 m from *some* prop |
| bearing error (median) | **0.1°** | 22.3° change under swap |
| inside the 0.6 m goal ring | **58/59** | 2–3/10 |

Detector: `google/owlv2-base-patch16-ensemble`, threshold 0.05.

**Two geometry terms were missing on the first attempt** and together caused a systematic
~25% under-range (median error 0.63 m — right at the goal ring, i.e. it would have looked
"almost working" while being wrong):

1. `vla_cam_z = 0.15` in the URDF is the camera height above **base_link**, not above the
   ground. True height ≈ **0.20 m**. A least-squares sweep over recorded episodes converged
   independently on 0.20 m, matching the URDF derivation — a genuine cross-check, so the
   number is trusted rather than fitted.
2. The camera sits **forward** of base_link (`wheel_base/2 + 0.10 = 0.187 m`). Projection
   yields camera-frame range; labels and tracker use the body frame.

Plus a perception term: the bbox bottom edge is the object's **near face**, not its centre,
so the class radius (`OBJECT_RADIUS`) is added back along the ray.

**Gotcha for the runtime detector node — query one phrase per forward pass.** Batching every
prop phrase into a single OWL-ViT call lets the queries suppress one another: 13/39 targets
scored **exactly 0.0000** in multi-query mode and **0.10–0.29** when queried alone. This was
invisible to a threshold sweep (0.015→0.08 gave bit-identical results, because the scores were
not low, they were zero), and it looked exactly like a detector capability limit. Recall went
**26/39 → 59/59** on this one change. Querying alone did *not* cause confident locks onto the
wrong object — selection accuracy stayed at 98%.

**What this settles.** Goal *selection* — the thing SmolVLA never learned — is solved to 98%
by a component that needs no training. The remaining risk in D9 is not acquisition; it is
whether memory + tracker hold the goal once it leaves view. That is validation step (b).

### D9 risk — the deployable detector cannot ground these phrases (2026-07-21)

The 98% acquisition result above used `owlv2-base-patch16-ensemble`. **NanoOWL — the TensorRT
path that made an on-NX detector look cheap — is built on `owlvit-base-patch32`, and that model
does not work for this task.** Measured on the same 59 episodes, single-query:

| model | recall | correct-prop | params | input | latency (A2000) |
|---|---|---|---|---|---|
| `owlvit-base-patch32` | **13/59 (22%)** | 13/13 | 153M | 768² | **271 ms** |
| `owlvit-base-patch32`, 3×2 tiles | 23/59 (39%) | 20/22 (91%) | — | — | ~6× above |
| `owlvit-base-patch16` | 6/59 (10%) | 6/6 | — | — | — |
| `owlv2-base-patch16-ensemble` | **59/59 (100%)** | **58/59 (98%)** | 155M | 960² | **3190 ms** |

Precision is perfect whenever patch32 fires — it is purely a **recall** failure.

Two hypotheses were tested and both rejected:

- *Object scale.* Tiling to raise effective resolution moved recall only 22% → 39%.
- *Range.* Recall vs. true range is **flat**: 29% at 0.5 m, 28% at 2.0 m, 19% at 3.0 m, 0% at
  3.5 m. At half a metre the prop dominates the frame and it still misses ~70%. So an
  approach-and-lock search behaviour does **not** recover acquisition.

The gap is model quality (OWLv2's web-scale self-training), not resolution or distance. Same
parameter count; the cost difference is tokens — 960²/patch16 = 3600 vs 768²/patch32 = 576.

**Consequence.** D9's acquisition stage requires an OWLv2-class detector, which is ~12× the
compute NanoOWL promised. This is **not yet fatal**, because per D3 acquisition is a mission-loop
operation at 0.1–1 Hz, not a control-loop one: the goal enters odom-frame memory and is held, so
a multi-second one-shot acquisition is tolerable where a multi-second control step would not be.
But the NanoOWL "~95 ms on Orin" figure must be struck from the deployment plan.

**Open, and must be measured on real hardware before M4 commits:** OWLv2 latency and memory on
the Xavier NX under TensorRT/fp16. A2000-fp32 → NX extrapolation spans ~1–10 s, which is too wide
to plan against. Until that number exists, treat on-NX acquisition as **unproven**.

Fallbacks if OWLv2 does not fit, in preference order: (1) distil OWLv2 into a smaller detector on
our own recorded episodes (`scene_config.json` gives exact supervision); (2) run acquisition once
off-board at mission start; (3) closed-vocabulary colour+shape segmentation — trivially fast and
near-perfect in sim, but abandons open-vocabulary generality and would not survive sim2real.

### D9 validation — corrected across scene families (2026-07-21)

The 98% acquisition figure above was measured on `open_ground` **only** (the first 59 recorded
episodes are all one family). Re-run once `corridor` episodes existed:

| scene family | detected | **selects commanded prop** | position err (median / p90) | inside 0.6 m ring |
|---|---|---|---|---|
| `open_ground` | 59/59 | 58/59 (98%) | 0.13 m / 0.35 m | 58/59 |
| `corridor` | 55/60 | 49/55 (**89%**) | 0.22 m / 0.45 m | 51/55 |
| **combined** | 114/119 | **107/114 (94%)** | — | 109/114 |

**The honest headline is 94%, not 98%.** Corridor is harder on both axes: walls and clutter give
the detector more to confuse the target with, and median position error nearly doubles.

Failure decomposition for corridor: 55 detected, 51 inside the goal ring, 49 nearest-to-correct.
So ~4 are projection/range error and ~6 are the detector locking onto the *wrong object* — the
dominant failure is grounding, not geometry. That matters for where effort goes: improving the
ground-plane model would buy little; disambiguating among similar props would buy more.

Calibration re-fit on corridor prefers `cam_height` **0.18 m** vs open_ground's 0.20 m. The
module keeps 0.20 (it matches the URDF derivation, and the 0.02 m difference is well inside the
goal ring); the discrepancy is logged as a small unexplained bias, not tuned away per-scene.

`parking_lot` is still ungenerated and **unvalidated** — it has bay lines and the ground-slab
randomisation is skipped there, so it is the most likely family to break the flat-ground
assumption. Re-run `test_acquisition_offline.py --scene parking_lot` when datagen reaches it.

Even at 94%, this remains far above the policy it replaces (≈ chance goal selection, 2-3/10
success), so the D9 decision stands.

### v4 datagen — corridor is geometrically incompatible with the short horizon (2026-07-21)

v4 finished 437/570: open_ground 87%, parking_lot 88%, **corridor 53%**. The corridor deficit is
not flakiness and not fixable by retry — roughly half of v4's corridor scenes have **no path at
all**.

Mechanism. `corridor` is 1.2 m wide (`SCENE_BOUNDS y = -0.6..0.6`). A crate's blocking radius is
`PROP_RADIUS 0.29 + INFLATE 0.28 = 0.57 m`. Compressing the goal range to 2.0-3.5 m (the whole
point of v4) forces every prop into a 1.5 m stretch of that hallway, so two crates routinely span
it end to end — e.g. seed 2002: crate at y=+0.21 blocks [-0.36, 0.78], crate at y=-0.40 blocks
[-0.97, 0.17]. Union covers the corridor; A* correctly reports no path.

Verified by holding everything else fixed and varying only the sampler env on one seed:

| sampler | prop spread | expert |
|---|---|---|
| v3 (2.0-7.0 m, 3 fillers) | 2.9-6.4 m | success, path 2.81 m |
| v4 (2.0-3.5 m, 1 filler) | 2.0-3.3 m | **A\* found no path** |

**Decision: corridor is dropped from v4.** The horizon hypothesis is testable on open_ground +
parking_lot, and keeping corridor would train on a *selection-biased* subset — only the layouts
that happen not to block. Corridor returns in v5 with a scene-specific distance range.

**Two logging bugs found here, both of which hid the diagnosis for hours and are now fixed:**

1. `run_episode.py` took `exp.stdout.strip().splitlines()[-1]` as the verdict. `ros2 run` appends
   `[ros2run]: Process exited with failure 1` to **stdout** whenever the child exits non-zero,
   which the expert does on every unsuccessful episode. So every real verdict was overwritten with
   `expert crashed: ` (stderr being empty), and 133 genuine `success:false` results were
   indistinguishable from crashes. Now scans for the last JSON-shaped line.
2. `batch_datagen.sh` / `regen_seeds.sh` piped `| tail -1`, reproducing the same error in the batch
   log. Now `| grep -oE "\{.*\}" | tail -1`.

Consequence for process: **a verdict field that can be silently replaced by a wrapper's error text
is worse than no verdict**, because it looks like data. The first cleanup pass classified these 133
as "transient, retry will help" on the strength of that fake signature, and the retry produced 1/26
before the contradiction surfaced.
