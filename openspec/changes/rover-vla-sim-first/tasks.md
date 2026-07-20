# Tasks — Ackermann Rover VLA, Simulation-First

Milestones M0–M4 map to sections 1–5. Every milestone has an explicit exit criterion; hardware
purchase is gated behind M2 (except camera *selection*, which is an M0 task by design).

## 1. M0 — Foundations (2–3 weeks)

- [x] 1.1 **Simulator feasibility gate (week 1, blocking)**: verify Isaac Sim + Isaac Lab runs
      with RTX rendering on an available GPU (current dev GPU is 4 GB — expected FAIL). Decide:
      local GPU upgrade / cloud instance / promote the Gazebo+augmentation fallback. Record the
      decision and the measured FPS/VRAM in this file.
      **DECIDED 2026-07-18 — Gazebo fallback promoted.** Isaac Sim cannot run on any available
      GPU: dev laptop RTX A2000 4 GB is below the ≥8 GB floor; the Titan X 12 GB has no RT
      cores (hard requirement — even A100/H100 are excluded); NVIDIA cloud rejected (software
      free, but RTX instances $0.5–1.5/h recurring). Gazebo Harmonic (the existing
      `ackermann_rover_x86_64_jazzy` image) measured headless on the A2000: **RTF ≈ 1.00 with
      the 1280×800 camera rendering at a full 15 Hz, 212–311 MiB VRAM** — coexists with
      training on the same 4 GB card. Per D1, Gazebo is now the *only* maintained simulator.
      Simplified standalone sim lives in `rover/ros2/src/rover_sim` (this repo); the
      `ackermann_rover_humble` workspace keeps its PX4/ArduPilot stack untouched (its
      Dockerfile gained the missing `ros-$ROS_DISTRO-gz-ros2-control`). Known transport note:
      raw 1280×800 RGB bridges at only ~4.3 Hz over reliable DDS (render side is 15 Hz) —
      compress or downscale at the recorder (same lever as the ALOHA all-ROS2 result).
- [x] 1.2 Select the real camera model (global shutter, ~100–110° HFOV pinhole, hardware
      timestamp or external trigger; OV9281-class candidate). Record resolution + intrinsics —
      these become the simulated camera's locked configuration. (Selection only; no purchase.)
      **SELECTED 2026-07-18 — OV9782-based module** (the draft's OV9281 is *monochrome*; color
      is required for attribute grounding, and OV9782 is its color sibling: 1MP 1/4" global
      shutter, 1280×800, MIPI CSI-2 and UVC module options, Jetson-supported). Locked sim
      configuration: **1280×800 @ 15 Hz, HFOV 100° → fx = fy = 537.0, cx = 640, cy = 400**
      (verified live in the sim `camera_info`). Exact module form (Arducam UVC B0223-class vs
      Jetvariety MIPI) and the ~100° low-distortion M12 lens are finalized at M3 purchase —
      stock UVC lens is 70°, so the M12 lens swap is part of the purchase decision.
- [x] 1.3 Model the 1/16 Ackermann rover in USD/URDF: real wheelbase, track width, steering
      limits, camera mount geometry; verify kinematics (min turn radius) in sim.
      **DONE 2026-07-18** — `rover/ros2/src/rover_sim/urdf/rover_vla.urdf.xacro`, a stripped
      copy of the real vehicle's digital twin (`description_robot`): wheelbase 0.174 m, track
      0.174 m, wheel Ø 77 mm, steering ±0.6 rad, same inertials/friction; PX4/RealSense/T265/
      lidar branches and the 42 MB mesh dropped. Verified closed-loop via
      `ackermann_steering_controller`: straight run 0.000 m lateral drift; commanded R = 0.357 m
      → measured 0.355 m; pose-derived yaw rate +1.400 rad/s = command. **Min feasible turn
      radius ≈ 0.341 m** (inner wheel hits the 0.6 rad limit first: R = L/tan(0.6) + track/2);
      commanding beyond it (e.g. ω=2.0 @ 0.5 m/s) breaks the geometry and scrubs wheels — the
      tracker's hard-limit clamp (task 2.5) is mandatory, not optional. Two recorder notes:
      derive yaw rate from pose history (the OdometryPublisher `twist` field is unreliable at
      high curvature), and the camera mount pose is provisional until M3 locks the real mount.
- [x] 1.4 Build 2–3 initial scenes (parking lot, corridor, open ground with props) with the
      randomization hooks (lighting, materials, placement, spawn/goal) wired but minimal.
      **DONE 2026-07-19** — three structural worlds (`open_ground`, `corridor`, `parking_lot`)
      + `scene_manager.py`: seed-deterministic EpisodeConfig applied to the *running* world via
      gz services — **~6 s episode reset, no Gazebo restart**. Wired: sun direction/intensity
      (via spawning an `ep_light` entity — `/world/*/light_config` acks but never reaches the
      sensors render scene, measured), ground-color slab (skipped for parking_lot so bay lines
      stay visible), prop placement with **guaranteed hard negatives** (same-color + same-shape
      distractors always present; exact goal duplicates excluded so the instruction stays
      unique), spawn-pose teleport (verified err 0.000 m), goal + instruction string; config
      JSON = episode metadata. Camera-verified frames for all three families. Deferred to M1:
      weather, exposure/extrinsic jitter, sensor noise, texture-level materials, sky.
      Verification lesson recorded in README: **never run two gz servers** — services
      round-robin between them and episodes half-apply.
- [x] 1.5 ROS 2 bridge up: /observation (camera + state) out, /cmd (steer, throttle) in;
      clock discipline documented (sim time).
      **DONE 2026-07-19** — `/observation` ≜ {`/vla_camera/image`, `/vla_camera/camera_info`,
      `/observation/state`}; state = 50 Hz `[speed, yaw_rate, steering]` (yaw rate pose-derived
      per the 1.3 finding; steering = equivalent bicycle angle from both knuckles). Command
      interface decision: **(v, ω) `TwistStamped` on `/cmd_vel`**, not raw steer/throttle —
      identical to the real rover's PX4 `rover_speed_steering` cmd_vel mapping, so the sim2real
      interface is unchanged; steer/throttle conversion is the controller's job on both sides.
      Closed-loop verified: state reads v 0.500 / yaw 1.000 / steer 0.335 against a
      (0.5 m/s, 1.0 rad/s) command. Clock discipline documented in rover/README.md (single gz
      `/clock`, `use_sim_time` everywhere, header-stamp arithmetic only, `ros2 topic pub`
      wall-stamp warning).
- [x] 1.6 Expert stack: privileged A* on the sim map + Pure Pursuit through the Ackermann
      model; drives sampled start→goal routes with goal-visible-at-start validity check.
      **DONE 2026-07-19** — `rover/ros2/src/rover_expert/scripts/expert_driver.py`: occupancy
      grid built from the episode config (prop footprints + per-scene statics, inflated by
      rover radius + 10 cm; goal prop inflated less so the approach ring stays reachable),
      8-connected A* (no corner cutting), line-of-sight shortcut smoothing densified to 5 cm,
      Pure Pursuit at 50 Hz with the measured feasibility clamp |ω| ≤ v/0.341 and near-goal
      speed ramp; goal-visible-at-start re-checked; per-run clearance monitor; one-line JSON
      verdict. **Verified 10/10 sampled episodes across all three scene families** (5×
      open_ground, 3× corridor, 2× parking_lot): all reach the 0.6 m goal ring (final dist
      0.591–0.597 m), min clearance always positive (tightest 0.094 m beyond the 0.18 m
      circumscribed rover circle, weaving past parked boxes), routes 1.7–6.4 m in 3.7–14 s at
      0.5 m/s cruise. Arrival frame confirms the goal prop fills the camera. Known M0
      limitation (recorded): SCENE_STATICS duplicates the world SDF geometry — unify when
      worlds become generated (M1).
- [x] 1.7 Recorder → **LeRobot format**: RGB 10–15 Hz, GT pose 50 Hz, (speed, yaw rate,
      steering), expert commands, per-episode randomization config + intrinsics + collision/
      success flags. Optional sim depth channel (debug only).
      **DONE 2026-07-19** — two-stage pipeline. (1) Sim side (`rover_expert`):
      `episode_recorder.py` logs frames (15.15 Hz measured, native 1280×800 JPEG q90) +
      `gt_pose`/`state` at exactly 50 Hz + every expert command + camera intrinsics, all
      sim-time stamped; `run_episode.py` orchestrates scene→record→expert→verdict and writes
      `episode.json` (config + verdict + success/collision flags) — the 2.3 datagen entry
      point. (2) `rover/datagen/to_lerobot.py` (runs in the `smolvla-edge:sim` image) converts
      raw episodes → LeRobotDataset v0.4.4: `observation.image` (video), `observation.state`,
      `observation.gt_pose`, provisional `action`=[v,ω] (task 2.1 rebuilds actions as K×(x,y,v)
      chunks from the raw 50 Hz poses, which stay in the episode dirs), task = instruction.
      Verified loading with `video_backend='pyav'` (repo convention): 2 episodes / 370 frames /
      fps 15, correct shapes and instruction strings. Corrections/notes: the earlier "4.3 Hz
      bridge cap" was a `ros2 topic hz` CLI artifact — a real subscriber receives the full
      15 Hz, so no compression needed at M0 scale; recorder orchestration must signal the
      recorder's *process group* (`ros2 run` wrapper swallows SIGINT); on hosts without a
      default route set `GZ_IP=127.0.0.1` or gz-transport discovery fails. Sim depth channel:
      omitted (optional; add at M2 with the safety-monitor work). Raw episodes + converted
      dataset live under `rover/data/` (gitignored).
- [x] 1.8 **Exit**: one scripted episode replays end-to-end from logged data (actions re-drive
      the sim; waypoint labels reconstruct from logged poses).
      **PASSED 2026-07-19 — M0 COMPLETE.** (a) `replay_episode.py`: scene reset from the
      logged (scene, seed), 524 logged commands re-published at their sim-time cadence —
      trajectory reproduces with **final error 0.020 m, max deviation 0.026 m** over the 5.1 m
      route (physics effectively deterministic under identical command streams). (b)
      `rover/datagen/relabel.py`: K=12 @ Δt=0.25 s body-frame (x, y, v) chunks from the raw
      50 Hz poses — straight segments spaced exactly v·Δt, turning frames show lateral
      progression, episode end clamps to the stop point with v→0; 0 sanity violations across
      all 181 frames. Hardware note: the Titan X (Maxwell, 12 GB, sm_52) is now attached via
      eGPU and torch 2.6+cu124 in `smolvla-edge:sim` retains sm_50 kernels — verified matmul;
      it is the designated *training* GPU (it still cannot render for Isaac Sim).

## 2. M1 — Pipeline proof (2–4 weeks)

- [ ] 2.1 Hindsight waypoint relabeler: per-frame future poses → body frame, K=12 @ Δt=0.25 s,
      per-point speed; unit test on straight/turn/stop episodes; normalization stats to
      ~[−1, 1].
- [ ] 2.2 Instruction generation: templates over sampled goal/scene ("drive to the {color}
      {object}", "turn {direction} after the {landmark}"), offline LLM paraphrasing, held-out
      phrasing split reserved.
- [ ] 2.3 Generate ~500–1,000 episodes across the M0 scenes with multi-target layouts and
      attribute-sharing hard negatives (red cone + red barrel; red cone + blue cone).
      *Progress 2026-07-19*: batch machinery proven — 36-episode pilot (12 seeds × 3 scenes,
      32/36 expert success; failures logged with reasons and auto-skipped) → 34-episode
      `local/rover_vla_v1` (4,812 frames). Full-scale run is a parameter change on the same
      script (`/tmp/batch.sh` pattern → promote into `rover/datagen/` when scaling).
- [ ] 2.4 Stage-one training: SmolVLA, frozen VLM backbone, action expert on K×(x,y,v)
      (chunk_size=K, action_dim=3); velocity-scaled directional blur augmentation active.
      *Progress 2026-07-19 — **process verified on small data (user-directed gate)***:
      stage-one smoke on the **Titan X** (Maxwell 12 GB, fp32 — lerobot's hardcoded bf16 VLM
      load patched, see `rover/train_smoke.sh`): 300 steps @ batch 8, 1.1 step/s, **loss
      2.94 → 0.38**, checkpoint saved, reloaded, and produced correct-shape 10×(x,y,v) chunks
      on a dataset frame with proper language tokenization (`make_language_tokenizer`).
      **Action-space refinement (recorded)**: with stock `lerobot-train`, hindsight chunks are
      stored FLAT (K=10 → 30 dims ≤ max_action_dim=32, `chunk_size=1`) because lerobot's
      `delta_timestamps` chunking would gather future frames' own-body-frame actions — wrong
      frame semantics for D2's single-frame chunk. Remaining for the real 2.4: full-scale
      data, blur augmentation, and the D2-native (chunk_size=K, action_dim=3) shape if a
      custom collate is adopted. Ops notes: `--shm-size=8g` mandatory (64 MB default kills
      dataloader workers); Maxwell cannot bf16 (CUBLAS_STATUS_NOT_SUPPORTED).
- [x] 2.5 Tracker node: Pure Pursuit at 50–100 Hz on EKF odometry (sim: GT + noise), hard
      limits (max speed, steering rate, min turn radius) enforced.
      **DONE 2026-07-19** — `rover_runtime/tracker_node.py`: 50 Hz PP on waypoint chunks,
      hard limits v ≤ 0.8 and |ω| ≤ v/0.341 (the measured min-turn-radius clamp), chunk
      REPLACES path. Unit-verified in an isolated ROS domain: straight chunk → v = 0.500,
      ω = 0 exactly; watchdog below. *Noted gap: odometry noise injection (GT is used raw in
      sim for now) — add with the M2 robustness work.*
- [x] 2.6 Async policy loop wired (AsyncRunner semantics): chunk replaces queue after
      odometry-delta latency compensation; staleness watchdog (age > 1 s ⇒ speed ramp to 0).
      **DONE 2026-07-19** — `chunk_client_node.py` (sequential request loop, one in flight;
      latest-frame JPEG + state + instruction over stdlib TCP; publishes `/waypoint_chunk`
      carrying the original capture_t) + `rover/runtime/policy_server.py` (SmolVLA checkpoint
      on the Titan X). Latency compensation = tracker transforms the chunk into world frame
      using pose history at capture_t. Verified: watchdog ramps v to 0 at 2.2 s staleness;
      server round-trip **~280 ms/inference** (3.6 chunks/s vs 2.5 s horizon ≈ 9× replan
      overlap — inside D3's envelope). Transport is the D8 client/server lineage; on the NX
      the chunk source becomes the all-ROS2 policy node with identical topics.
- [x] 2.7 Closed-loop rollouts in training-like scenes; measure success rate + swap test.
      **MEASURED 2026-07-19 (stage1_v2, frozen backbone)** — open_ground, unseen seeds
      9000–9009, full async stack (policy server on Titan X ~280 ms/chunk → chunk client →
      tracker): **success 5/10, swap 2/8**. Failure anatomy: (a) *saliency capture* — in 6/8
      failed swap pairs the rover approached the SAME prop under both instructions (goes to a
      preferred prop, ignores language); notably both swap PASSES were same-shape crate pairs
      differing only in color, so color grounding partially works; (b) *razor-thin margins* —
      5 "collisions" were −0.000…−0.008 m grazes against the circumscribed-circle model
      (imitating the expert's own thin clearances, minus precision); (c) 2 timeouts. Eval log:
      `rover/outputs/eval_stage1_v2_open_ground.log`.
- [ ] 2.8 **Exit**: policy reaches visible goals above threshold in training-like scenes; swap
      test above chance. **Escape valve (pre-committed)**: if the swap test fails under the
      fully frozen backbone, pull vision-encoder LoRA forward into M1 (language model stays
      frozen).
      **GATE NOT PASSED at stage1_v2 (swap 2/8) → escape valve invoked 2026-07-19.**
      Implementation: lerobot has no "vision-trains/LM-frozen" combo
      (`train_expert_only=false` unfreezes the LM too), so the runtime patch that already
      fixes bf16 also freezes `text_model` in the unfrozen branch → stage1b trains vision
      encoder + connector + expert with the language tower frozen, warm-started from the
      stage1_v2 checkpoint. The saliency-capture failure mode is exactly the predicted
      frozen-SigLIP-on-synthetic-textures deficiency (D5 contingency 1).
- [ ] 2.9 Contingency check: if tracking oscillates on policy chunks, switch output to (κ, v)
      per design D2 before touching model capacity.

## 3. M2 — Robust policy (4–6 weeks)

- [ ] 3.1 DART-style noise injection into the expert with corrective-action recording; regenerate
      the large dataset with recovery states included.
- [ ] 3.2 DAgger pass: roll out the trained policy, relabel visited states with expert actions,
      retrain; measure closed-loop delta attributable to recovery data.
- [ ] 3.3 Full randomization sweep across all scene families (parking, dirt/grass, trail,
      warehouse, corridor, narrow passage) with pedestrians/dynamic props.
- [ ] 3.4 Stage-two training: LoRA on the backbone with the full dataset.
- [ ] 3.5 Full evaluation protocol on held-out environments: success, collision +
      safety-intervention rates, swap test with hard negatives, held-out paraphrases, path
      quality (smoothness, tracking error, time-to-goal).
- [ ] 3.6 Failure slicing by logged randomization metadata → written failure-mode
      characterization per environment factor.
- [ ] 3.7 Safety monitor node (sim): geometric corridor check on sim depth, speed cap / stop
      veto, measured intervention rate — network-independent by construction.
- [ ] 3.8 **Exit (camera-purchase / real-run gate)**: success + collision metrics meet targets
      on held-out scenes; failure modes characterized; scaling decisions justified by metrics
      only. (Rescoped 2026-07-20: the rover/Jetson/sensors are already on hand — see 4.1 — so
      this gate now authorizes the single OV9782 camera purchase and real-world runs, not a
      hardware buy-in.)

## 4. M3 — Real rover (hardware on hand; only the policy camera is a purchase)

- [ ] 4.1 **Hardware inventory (rescoped 2026-07-20 — most hardware already owned)**: the
      `ackermann_rover_humble` platform exists and is bring-up-complete — 1/16 Ackermann
      chassis with PX4 actuation (steering servo + ESC), **Xavier NX** (the design-D7
      baseline), **D435i** (color RGB `/d435i/color/image_raw` + IR-stereo depth + IMU) and
      **T265** (mono global-shutter fisheye stereo + IMU), cuVSLAM/VINS odometry, EKF, RPLidar,
      CubePilot IMU/GPS. **The only genuine gap for the policy is a color, global-shutter,
      rectilinear, ~100° RGB camera** — none of the on-hand cameras hit all four (D435i color
      is rolling-shutter and ~69° HFOV; T265 is mono + fisheye, ruled out by D1). Purchase is
      therefore a single ~$40 OV9782-class module (task 1.2), not a rover/Jetson/sensor buy.
      **Zero-purchase bridge**: the D435i color stream can drive a first transfer test before
      the OV9782 arrives (accepting rolling shutter + narrower FOV; re-lock the sim camera to
      D435i intrinsics for that run — see design D7).
- [ ] 4.2 Bring-up: sensor drivers, EKF (robot_localization), tracker, safety monitor as the
      same ROS 2 nodes as sim; actuator interface (steering servo PWM + ESC).
- [ ] 4.3 **Timestamping bench (blocking)**: hardware shutter timestamps wired; blinking-LED
      capture-to-userspace latency measurement; EKF pose interpolation at shutter time —
      no latency-compensation value trusted before this passes.
- [ ] 4.4 Deploy the policy on the Jetson via `precision="fp16-graph"` (pre-warm capture at
      node startup); verify chunk rate ≥ 4 Hz and parity vs the training checkpoint.
- [ ] 4.5 Controlled-space runs at reduced speed caps with depth/lidar monitor + physical
      e-stop; real closed-loop metrics vs the sim baseline → quantified transfer gap.
- [ ] 4.6 **Exit**: reliable visible-goal navigation on real hardware within the safety
      envelope; transfer gap documented before the operating envelope widens.

## 5. M4 — Extensions (optional)

- [ ] 5.1 Mission layer: SLAM + global planner emitting sub-goals through the instruction
      interface only (0.1–1 Hz); beyond-line-of-sight goals.
- [ ] 5.2 Rear camera logging + reversing/parking tasks (adds heading/signed-velocity channel
      per design D2's scope rule).
- [ ] 5.3 Baseline comparison: end-to-end control-output policy vs the waypoint architecture
      on the same evaluation protocol.
- [ ] 5.4 Backbone study if metrics justify: 256M (pre-measured 196 ms on NX) vs 450M vs
      larger; Qwen2.5-VL + diffusion head as the recorded architecture fallback.
- [ ] 5.5 Record upgrade-path notes (design §10) against actual findings: BEV front-end,
      navigation router, formalized safety envelope.
