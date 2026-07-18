# Tasks — Ackermann Rover VLA, Simulation-First

Milestones M0–M4 map to sections 1–5. Every milestone has an explicit exit criterion; hardware
purchase is gated behind M2 (except camera *selection*, which is an M0 task by design).

## 1. M0 — Foundations (2–3 weeks)

- [ ] 1.1 **Simulator feasibility gate (week 1, blocking)**: verify Isaac Sim + Isaac Lab runs
      with RTX rendering on an available GPU (current dev GPU is 4 GB — expected FAIL). Decide:
      local GPU upgrade / cloud instance / promote the Gazebo+augmentation fallback. Record the
      decision and the measured FPS/VRAM in this file.
- [ ] 1.2 Select the real camera model (global shutter, ~100–110° HFOV pinhole, hardware
      timestamp or external trigger; OV9281-class candidate). Record resolution + intrinsics —
      these become the simulated camera's locked configuration. (Selection only; no purchase.)
- [ ] 1.3 Model the 1/16 Ackermann rover in USD/URDF: real wheelbase, track width, steering
      limits, camera mount geometry; verify kinematics (min turn radius) in sim.
- [ ] 1.4 Build 2–3 initial scenes (parking lot, corridor, open ground with props) with the
      randomization hooks (lighting, materials, placement, spawn/goal) wired but minimal.
- [ ] 1.5 ROS 2 bridge up: /observation (camera + state) out, /cmd (steer, throttle) in;
      clock discipline documented (sim time).
- [ ] 1.6 Expert stack: privileged A* on the sim map + Pure Pursuit through the Ackermann
      model; drives sampled start→goal routes with goal-visible-at-start validity check.
- [ ] 1.7 Recorder → **LeRobot format**: RGB 10–15 Hz, GT pose 50 Hz, (speed, yaw rate,
      steering), expert commands, per-episode randomization config + intrinsics + collision/
      success flags. Optional sim depth channel (debug only).
- [ ] 1.8 **Exit**: one scripted episode replays end-to-end from logged data (actions re-drive
      the sim; waypoint labels reconstruct from logged poses).

## 2. M1 — Pipeline proof (2–4 weeks)

- [ ] 2.1 Hindsight waypoint relabeler: per-frame future poses → body frame, K=12 @ Δt=0.25 s,
      per-point speed; unit test on straight/turn/stop episodes; normalization stats to
      ~[−1, 1].
- [ ] 2.2 Instruction generation: templates over sampled goal/scene ("drive to the {color}
      {object}", "turn {direction} after the {landmark}"), offline LLM paraphrasing, held-out
      phrasing split reserved.
- [ ] 2.3 Generate ~500–1,000 episodes across the M0 scenes with multi-target layouts and
      attribute-sharing hard negatives (red cone + red barrel; red cone + blue cone).
- [ ] 2.4 Stage-one training: SmolVLA, frozen VLM backbone, action expert on K×(x,y,v)
      (chunk_size=K, action_dim=3); velocity-scaled directional blur augmentation active.
- [ ] 2.5 Tracker node: Pure Pursuit at 50–100 Hz on EKF odometry (sim: GT + noise), hard
      limits (max speed, steering rate, min turn radius) enforced.
- [ ] 2.6 Async policy loop wired (AsyncRunner semantics): chunk replaces queue after
      odometry-delta latency compensation; staleness watchdog (age > 1 s ⇒ speed ramp to 0).
- [ ] 2.7 Closed-loop rollouts in training-like scenes; measure success rate + swap test.
- [ ] 2.8 **Exit**: policy reaches visible goals above threshold in training-like scenes; swap
      test above chance. **Escape valve (pre-committed)**: if the swap test fails under the
      fully frozen backbone, pull vision-encoder LoRA forward into M1 (language model stays
      frozen).
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
- [ ] 3.8 **Exit (hardware purchase gate)**: success + collision metrics meet targets on
      held-out scenes; failure modes characterized; scaling decisions justified by metrics only.

## 4. M3 — Real rover (hardware purchase gate opens here)

- [ ] 4.1 Purchase: rover chassis, Jetson (Xavier NX on hand is the baseline per design D7;
      Orin only if M2 metrics demand it), the selected camera, IMU, depth sensor, e-stop.
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
