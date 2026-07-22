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
      *Full-scale run 2026-07-19/20*: `local/rover_vla_v2` (568 ep / 78k frames) → stage1_v2,
      10k steps, loss 2.94 → 0.108, wandb project `rover-vla`. Evaluated (see 2.7/2.8): the
      grounding gate failed on a *data* confound, fixed in the sampler; the retrain on the
      `local/rover_vla_v3` data is the next run.
      **Training host options**: (a) **Titan X** local (Maxwell, fp32, batch 8, ~2.5 h) via
      `rover/train_smoke.sh` — set `WANDB=true WANDB_API_KEY=…`; (b) **Google Colab** via
      `rover/train_colab.ipynb` — A100/L4 give bf16 + batch 32–64 (~10× faster), T4/V100 auto-
      fall back to the fp32 patch; datagen and closed-loop eval stay on the local rover host,
      Colab does training only (upload the dataset tarball to Drive, download the checkpoint).
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
      `rover/eval_results/eval_stage1_v2_open_ground.log`.
- [ ] 2.8 **Exit**: policy reaches visible goals above threshold in training-like scenes; swap
      test above chance. **Escape valve (pre-committed)**: if the swap test fails under the
      fully frozen backbone, pull vision-encoder LoRA forward into M1 (language model stays
      frozen).
      > **STATUS 2026-07-21 (REFRAMED) — GATE OPEN. Likely a horizon/memory problem, not colour.**
      > Probe swap-flip (chance 0.25): stage1_v2 0.27 → stage1_v3 0.18 → stage1c_v3 0.05 →
      > stage1d_deeplm (32 LM layers) **0.27** with instruction-sensitivity up 15°→**22.3°** —
      > best yet but still ≈ chance, so **five interventions have failed to lift colour grounding**
      > (data-confound fix, frozen backbone, full vision unfreeze, constrained vision adaptation,
      > deeper LM).
      > **Correction:** the earlier "colour binding is broken" conclusion is **under-determined** —
      > a proximity/saliency-driven policy yields the *same* swap score as a colour-blind one.
      > Trace analysis: the rover ends **0.32–0.60 m from *some* prop in 8/10** (approach works)
      > but **never heads toward the commanded target**, and when that target is far (rank 4–6) it
      > goes to the **nearest** prop in **4/5** cases. Expert = **10/10** on the same seeds.
      > **Structural root cause:** SmolVLA is a *local, memoryless* policy (`n_obs_steps=1`, chunk
      > = 2.5 s ≈ **1.25 m**) asked to do long-horizon goal selection to targets **2–7 m** away —
      > and once the goal leaves the ~100° FOV the input contains *nothing* about it. Design D3's
      > **mission loop was deferred (M4-optional), so the policy was doing the mission layer's job**.
      > **In flight:** `rover_vla_v4` with goals at **2.0–3.5 m** (range-equalised) — keeps the goal
      > in view and removes the distance confound, making the swap test a clean colour measure.
      > **Long-term:** goal memory — `n_obs_steps>1`, or (better) the D3 mission layer holding the
      > goal in the odom frame. Full analysis: `rover/eval_results/grounding_diagnosis.md`.
      **GATE NOT PASSED at stage1_v2 (swap 2/8) → escape valve invoked 2026-07-19.**
      Implementation: lerobot has no "vision-trains/LM-frozen" combo
      (`train_expert_only=false` unfreezes the LM too), so the runtime patch that already
      fixes bf16 also freezes `text_model` in the unfrozen branch → stage1b trains vision
      encoder + connector + expert with the language tower frozen, warm-started from the
      stage1_v2 checkpoint. The saliency-capture failure mode is exactly the predicted
      frozen-SigLIP-on-synthetic-textures deficiency (D5 contingency 1).
      **RESULT 2026-07-20 — escape valve REGRESSED, did not fix grounding.** stage1b (10k
      steps, wandb run `les64mau`, loss → ~0.12) re-evaluated on the identical seeds 9000–9009:
      **success 2/10 (was 5/10), swap 0/8 (was 2/8)** — worse on both axes. Symptom shift:
      more timeouts and wandering (one run drove 11 m from every prop), i.e. the *navigation*
      degraded, not just grounding. Diagnosis: stage1b was a **FULL vision-encoder unfreeze**,
      but D5 contingency 1 specifies **vision-encoder LoRA** — the full unfreeze of a ~400 M
      SigLIP on ~1 epoch of 568 synthetic episodes (fp32 batch 8) is under-constrained; it
      drifted the pretrained features and pulled them out from under the expert head that had
      converged on them in stage1. Evidence reading: the frozen features were *not* the sole
      bottleneck (unconstrained adaptation hurt), so the next move is either the *constrained*
      LoRA the design actually prescribed, or the data-side rungs (harder per-scene negatives /
      more instruction diversity / more episodes). Eval log:
      `rover/eval_results/eval_stage1b_v2_open_ground.log`. **Gate still open; decision pending.**
      **ROOT CAUSE FOUND 2026-07-20 (offline grounding probe).**
      `rover/eval_results/grounding_probe.py` feeds the trained policy the same frame under
      different instructions and measures whether the predicted chunk turns toward the named
      object. stage1_v2: **directional accuracy 0.71** (reads shape/side) but **swap-flip 0.10 —
      below chance 0.25** (ignores color among same-shape objects; mean bearing change 12.7°).
      The policy grounds *shape* but not *color*. Cause is a **data confound**, not model
      capacity: `scene_manager.sample()` placed the goal in the forward camera cone while all
      distractors scattered uniformly, so the goal was always the most salient object → the
      policy learned "drive to the object ahead" and never needed to read color; the swap test
      (which commands the peripheral hard negative) exposed it. Web context: SigLIP *does*
      encode color, but frozen-VLM SmolVLA + tiny expert is a documented weak grounder — so the
      data must force color to matter. **Fix committed**: sampler now places all props in-cone
      with comparable centrality (goal not privileged; same-shape/different-color twin equally
      visible → color is the only disambiguator). Full writeup:
      `rover/eval_results/grounding_diagnosis.md`. **Next: regenerate → `local/rover_vla_v3` →
      retrain frozen backbone (on Colab, see 2.4) → re-run swap.** If swap lifts, the frozen
      backbone binds color once data forces it (no LoRA needed); if flat, LoRA (properly, not a
      full unfreeze) is the next rung.
      **RESULT 2026-07-21 — sampler fix did NOT restore color grounding (frozen-backbone limit).**
      `local/rover_vla_v3` (520 ep, confound-fixed sampler) → stage1_v3 (frozen backbone, 10k,
      loss ~0.16, wandb `bya6kgwd`). Closed-loop open_ground seeds 9000–9009: **success 3/10,
      swap 0/9** — *but not comparable to the v2 baseline*: the v3 eval scenes use the same
      fixed sampler, so props cluster in the cone and navigation is genuinely harder (many
      grazes/timeouts; e.g. 9008 swap headed to the correct prop but timed out 1.05 m short).
      **Controlled offline probe (both checkpoints on identical raw_v3 frames)** decouples
      grounding from that difficulty: stage1_v3 directional **0.71** vs stage1_v2 **0.62** (modest
      shape/side gain), but color swap-flip **0.18 (v3) vs 0.27 (v2)** — *both at chance (0.25)*.
      Verdict: removing the saliency shortcut was a real data improvement and modestly helped
      shape/side grounding, but **color-word binding is unchanged and at chance** — the data
      confound was not the cause of the color failure. Combined with the full-unfreeze
      regression, this isolates it to **model capacity**: frozen SmolVLA (single SigLIP + tiny
      action expert) does not bind color attributes at this scale. Next rung per D5: **vision-
      encoder LoRA done correctly** (low-rank, LM + base-SigLIP frozen — *not* the full unfreeze
      that regressed); if LoRA also flat, escalate to color-stressed data / richer color cues, or
      the D5-contingency-4 architecture (Qwen2.5-VL + diffusion head). Logs:
      `rover/eval_results/eval_stage1_v3_open_ground.log`. Methodology note: keep a fixed eval
      scene set across model versions, or closed-loop success conflates policy quality with scene
      difficulty (observed here).
      **ESCAPE-VALVE RESULT 2026-07-21 — constrained vision adaptation also failed; bottleneck
      is the truncated LM, not vision.** `stage1c_v3` (warm-start stage1_v3, unfreeze top-2 of
      16 vision layers, LM + lower vision frozen, 126M trainable verified pre-launch, 10k steps,
      loss 0.117; the robust stand-in for LoRA — lerobot/peft was blocked by a `diffusers`
      peft>=0.17 pin and a factory that reads `--policy.path=<ckpt> --use_peft` as *load an
      existing adapter*). Probe on identical frames: **directional 0.75 (best of all models) but
      colour swap-flip 0.05 — below chance 0.25** (stage1_v3 was 0.18, stage1_v2 0.27). Vision
      adaptation made it a *better navigator and a worse colour-grounder*: it steers to a fixed
      salient object more confidently, still ignoring the colour word (~15° bearing change).
      **Four interventions have now failed to move colour grounding** (data-confound fix, frozen
      backbone, full vision unfreeze, constrained vision adaptation) — so the vision encoder is
      not the bottleneck. SigLIP encodes colour; the missing step is *binding* the colour word to
      an object, which happens in the LM — and SmolVLA truncates it to the first **16 of 32**
      layers (`num_vlm_layers`). **Next (priority): raise `num_vlm_layers` 16→32, from base**
      (`MODE='C'` in `rover/train_colab.ipynb`; needs A100 VRAM since the expert resizes to
      match). If that is also flat, escalate to D5 contingency 4 (Qwen2.5-VL + diffusion head).
      Hardware note: the Titan X eGPU dropped off the Thunderbolt bus mid-run (`nv_pci_remove`,
      required a reboot), costing one run — checkpoints are now `save_freq=500` and long runs
      should prefer Colab.
- [ ] 2.9 Contingency check: if tracking oscillates on policy chunks, switch output to (κ, v)
      per design D2 before touching model capacity.
- [x] 2.10 **DONE 2026-07-22 — OmniVLA-edge reference evaluation: the failure is missing
      inputs, not capacity.** Released navigation checkpoint (`NHirose/omnivla-edge`, ~108M
      ViNT-lineage, MIT; EfficientNet-B0 × 6-frame history + CLIP-text + goal-pose channel with
      modality dropout, L1 head) evaluated zero-shot on our frames and seeds via a drop-in
      server speaking the policy_server wire protocol (`rover/runtime/omnivla_server.py`).
      **Offline** (12 range-equalised same-shape/diff-colour pairs, `rover/eval/omnivla_swap.py`):
      pose-conditioned swap **12/12**, mean bearing change 29.0°; language-conditioned **2/12**,
      1.5° — language conditioning does nothing on our imagery even in a nav-pretrained model.
      CLIP itself reads our prop colours fine (**9/11**, chance 0.25, `clip_prop_probe.py`) —
      so colour *perception* is intact and the broken step is language→spatial-action binding,
      in ours and in theirs. **Closed-loop** (eval seeds 9000–9009, same tracker/referee;
      `--send-goal` privileged pose via new optional `goal_xy` client param):
      | run | success |
      |---|---|
      | expert | 10/10 |
      | SmolVLA stage1c_v3 (trained, 520 ep) | 3/10 |
      | OmniVLA-edge language (zero-shot) | 4/10 |
      | OmniVLA-edge **pose** (zero-shot, fixed executor) | **7/10** |
      Executor bugs found closed-loop (all in our shim, model exonerated by replay — it emits
      ~1 m paths even for a goal at the origin): (1) stop-intent paths → all-zero chunk →
      tracker `at_end` latch = **permanent park** (seeds 9001/9007/9009 froze; 9007 froze at
      its closest approach, 5 cm from a blocking prop); (2) server exceptions closed the socket
      with no reply → silent tracker starvation; (3) arrival stop-point 0.55 m + tracker
      GOAL_TOL 0.15 = park at 0.70 m, 10 cm *outside* the 0.6 m ring (8× timeout at 0.69–0.70).
      Fixes: Ackermann-feasible recovery arc (R ≥ 0.36 m), in-ring approach stop (0.40 m),
      always-reply error path; 32 unit checks. Remaining failures are model-level: 2 transit
      grazes (−0.005/−0.017 m), 1 goal-blocked-by-prop hold. Warm latency 56–80 ms (vs
      SmolVLA 280 ms) on the Titan X. Artifacts: `rover/gifs/compare4_seed900*.gif` (expert | SmolVLA |
      OmniVLA-lang | OmniVLA-pose), traces in `eval_traces_omnivla_{lang,pose}_v2/`.
      **Decisions:** (a) **v4 short-horizon dataset deprioritized** — it tests the visibility
      confound, but the pose-vs-language contrast (7/10 vs 4/10 in one model on identical
      frames) already isolates the failure to goal conditioning; (b) the scaling-law reading of
      stage1d (2.8: "capacity") is **revised**: a 9× smaller navigation policy with a goal
      channel beats trained SmolVLA zero-shot, so the binding constraint is the input/objective
      design, not parameters; (c) language's job moves to *selection* (detector), geometry to
      *steering* (goal channel) — the D9/D3 decomposition, now measured stage-by-stage.
- [x] 2.11 **DONE 2026-07-22 — SmolVLA goal-channel adaptation: 7/10, parity with the
      nav-pretrained reference; the goal channel was the whole fix.** Extended
      `observation.state` [speed, yaw_rate, steering] → + [goal_x, goal_y, cos ψ, sin ψ] (SmolVLA
      pads to `max_state_dim=32`, so a data change — no model surgery). `relabel.py goal_state()`
      + `to_lerobot.py --goal-state 0.3`: goal from `episode.json` props + `gt_pose`, per-episode
      bias σ=5 cm + per-frame jitter σ=3 cm (D9's measured 5–16 cm), 0.33 goal-dropout so
      language stays load-bearing. Frozen-backbone stage-1 on `rover_vla_v3g`, 10k steps.

      **Result (seeds 9000–9009, same tracker/referee):**
      | checkpoint / serving | success |
      |---|---|
      | SmolVLA language-only (goal zeroed) | 1/10 |
      | SmolVLA + goal, **no arrival assist** | 0/10 (drives to the RIGHT object, parks 0.78–1.00 m — just outside the 0.6 m ring) |
      | **SmolVLA + goal + arrival assist** | **7/10** |
      | OmniVLA-edge pose (reference) | 7/10 |

      **The goal channel solved the hard part** (drives to the commanded object — the failure of
      the whole M1). The residual is the *last-meter* problem, not grounding: the policy stops
      where the expert demos stop (~0.56 m ring edge), the tracker parks ~0.15 m short of any
      chunk end, so it lands ~0.70–0.90 m out. The arrival assist (executor drives the final
      ~0.15 m into the ring; `approach_chunk` targets 0.40 m so realised stop 0.40–0.55 m) closes
      it — the SAME executor treatment OmniVLA-edge got, so 7-vs-7 is symmetric. Six clean reaches
      at 0.59–0.60 m; the three misses (9000 wander, 9001/9007 grazes −6/−7 mm) are the clutter
      class, shared with OmniVLA-edge. GIFs: `rover/gifs/v3g2_900*.gif`, four-panel `rover/gifs/full4_900*.gif`.

      **Two serving bugs found and fixed (they masqueraded as model failures — the model was fine
      from stage1_v3g on):** (1) lerobot 0.4.4 keeps normalization in a *separate* processor file
      not loaded by the raw server → the goal channel was fed un-normalized and appeared dead
      (offline probe was flat until `make_pre_post_processors` was wired in); this also means ALL
      historical SmolVLA closed-loop numbers (stage1c_v3's 3/10) carried a raw-serving over-scaling
      artifact — re-served honestly, stage1c_v3 is 0/10, so language-conditioned SmolVLA was never
      above ~1/10. (2) `--arrival-assist` behind a flag dropped by nested-docker quoting → a run
      that was really 7/10 read as 0/10; the assist is now default-on (`--no-arrival-assist` to
      disable), since goal-mode arrival structurally needs it.

      **NEGATIVE result — arc-length labels (rover_vla_v3g2) were null.** Hypothesised the early
      stop came from time-parameterized labels shrinking near the goal; built `waypoint_chunk_arclen`
      (fixed 0.125 m arc spacing, v pinned at cruise) + 1/range dim + approach-band oversampling,
      retrained (stage1_v3g2, 8-dim). Offline probe on identical frames: v3g2 and v3g emit
      **near-identical chunks** (extent 1.25 m far, collapsing to ~0.04 m at 0.68 m in BOTH). The
      collapse is not a label artifact — it is the model faithfully reproducing that the EXPERT
      stops at the ring. Arc-length changed nothing; the 7/10 is the goal channel + assist, not the
      labels. Lesson: probe offline before spending GPU-hours on a label hypothesis.
      Success criterion (≥ OmniVLA-edge 7/10): **met.**
      **Action head stays flow matching (decided 2026-07-22):** navigation is *more*
      action-multimodal than manipulation (left/right obstacle splits — the reason NoMaD put a
      diffusion head on ViNT), FM is the cheaper successor of diffusion (~10 steps vs 50–100,
      already inside the measured 233 ms NX envelope), and the L1-regression baseline's two
      grazing collisions are the signature failure of unimodal heads on binary route choices.
      Also methodological: 2.11 tests the input hypothesis — one variable at a time; revisit
      the head only if the bake-off traces show mode-averaging or chunk-to-chunk dithering
      (known mitigation: warm-started sampling / RTC-style chunk consistency).
      **Temporal context stays out (decided 2026-07-22):** `n_obs_steps` is dead code in
      SmolVLA (config default only, never read by modeling code — verified in lerobot 0.4.4),
      and the reference experiment shows history isn't the signal: OmniVLA-edge *has* 6-frame
      history yet its language mode scored 2/12, while its history-independent goal channel
      scored 12/12. Goal memory supplies the temporal information exactly (odom-frame
      persistence) rather than asking the policy to learn object permanence from pixels — D3's
      memory-in-state principle. Contingency if dynamic obstacles (M3 pedestrians) demand
      motion cues: feed the previous frame as a second image key — SmolVLA accepts multiple
      cameras and SmolVLM2 is video-pretrained, so two-frame context needs fine-tuning only,
      no `n_obs_steps` surgery (+~64 prefix tokens on the NX budget).
- [ ] 2.12 **Policy bake-off gate (pick the M2 backbone on numbers).** Same seeds, same
      referee: (a) SmolVLA+goal (2.11) vs (b) OmniVLA-edge fine-tuned on our expert episodes
      (pose+language dropout; ~108M trains fp32 on the Titan X). Report success, swap,
      NX-projected latency. The winner becomes the M2 training target; the loser is retired.
      Either way the runtime keeps the async chunk contract (tracker/client unchanged).

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
      **PULLED FORWARD 2026-07-21 — this is the fix for the M1 failure, not an M4 extension.**
      M1 diagnosis (§2.8) found the policy is *memoryless*: `n_obs_steps=1`, and SmolVLA's
      modeling code **never reads that field at all** (verified — it appears only as a config
      default), so frame history is unavailable without model surgery. Its whole input is one
      frame, so **a goal outside the ~100° FOV is unrecoverable** — exactly the
      "beyond-line-of-sight goals" this task names. Goal persistence belongs in map/state, per
      D3, not in policy weights.
      *Scaffold done*: `rover_runtime/goal_memory_node.py` stores the goal in the **odom frame**
      and republishes its body-frame range/bearing as the rover moves, so it stays correct when
      the goal leaves view. It also emits the **policy's own `/waypoint_chunk` format**, so the
      existing tracker drives to a remembered goal with no other changes — demonstrating the
      full architecture (ground once → remember in odom → chunk → tracker) and letting the
      mission layer stand in for the policy on the "go to a known goal" leg. Grounding modules
      speak body frame (`/goal_memory/set_relative`, e.g. "red crate 3 m ahead, 20° left");
      the node converts once and remembers. Transforms unit-tested offline
      (`rover/eval_results/test_goal_memory.py`, 19 checks) — including the decisive case:
      goal behind the rover, outside the FOV, range still exact and bearing pointing back.
      *Remaining*: the acquisition path (what supplies the goal — a detector, a slow VLM, or the
      policy at first sight) and an in-sim integration test once the v4 datagen releases the sim.
- [ ] 5.2 Rear camera logging + reversing/parking tasks (adds heading/signed-velocity channel
      per design D2's scope rule).
- [ ] 5.3 Baseline comparison: end-to-end control-output policy vs the waypoint architecture
      on the same evaluation protocol.
- [ ] 5.4 Backbone study if metrics justify: 256M (pre-measured 196 ms on NX) vs 450M vs
      larger; Qwen2.5-VL + diffusion head as the recorded architecture fallback.
- [ ] 5.5 Record upgrade-path notes (design §10) against actual findings: BEV front-end,
      navigation router, formalized safety envelope.

- [x] 5.2 Acquisition geometry + offline validation — DONE 2026-07-21
      `rover_runtime/goal_projection.py` (bbox → body frame; ground-plane for the sim's
      RGB-only camera, depth variant for the D435i) + `eval_results/test_goal_projection.py`
      (30 checks) + `eval_results/test_acquisition_offline.py` (end-to-end vs ground truth).
      Measured on 59 recorded v4 episodes: detected 59/59, **selects the commanded prop
      58/59 (98%)**, position error median 0.13 m / p90 0.35 m, bearing 0.1°, 58/59 inside
      the 0.6 m ring. Calibrated camera height 0.20 m (data fit and URDF derivation agree).
      Key gotcha recorded in design.md: query ONE phrase per detector pass — multi-query
      suppression zeroed 13/39 targets and mimicked a capability limit.
- [ ] 5.3 Privileged-goal integration test (D9 step b) — needs the sim; blocked on v4 datagen.
      Publish the true goal to `/goal_memory/set_odom`, drive with the tracker, confirm
      out-of-view goals are still reached. This is the user's stated fallback ("just tell it
      the goal position") and isolates memory+tracker before detector error is added.
- [~] 5.4 Swap the detector in for real acquisition (D9 step c) - NODE WRITTEN 2026-07-21,
      awaiting sim. `rover_runtime/goal_acquisition_node.py`: instruction -> target phrase ->
      OWLv2 (one query per pass) -> ground-plane projection -> `/goal_memory/set_relative`,
      published ONCE then latched (goal_memory holds it in odom). Inference runs off the
      executor thread - it is seconds long and would otherwise stall /clock and the 50 Hz
      tracker. `eval_results/test_goal_acquisition.py`: 304 checks, all 18 datagen templates
      x 16 colour/shape pairs incl. held-out phrasings, plus a guard that the node's
      vocabulary has not drifted from scene_manager's.
      **NanoOWL is ruled out** — its `owlvit-base-patch32` backbone gets 22% recall here
      (39% tiled) vs OWLv2's 100%, and recall is FLAT vs range (29% at 0.5 m), so neither
      tiling nor approach-and-lock recovers it. Needs an OWLv2-class detector.
- [ ] 5.5 **Measure OWLv2 on the Xavier NX** (TensorRT/fp16 latency + VRAM). Blocking risk
      for D9: A2000-fp32 is 3190 ms and the NX extrapolation spans ~1-10 s. Acquisition is
      0.1-1 Hz (D3) so seconds are tolerable, but the number must be real before M4.

- [x] 2.10 **v4 horizon test** — DONE 2026-07-22, outcome = row 3 (NEGATIVE for grounding).
      (Numbered 2.10: an unrelated contingency item above already holds 2.9.)

      **Result** (identical scenes, seeds 9000-9009, goals 2.0-3.5 m, closed-loop + swap):

      | metric | stage1_v4 | stage1_v3 CONTROL on same scenes |
      |---|---|---|
      | reached the commanded prop | **4/10** | 1/10 |
      | reached the WRONG prop | 2 | 0 |
      | closed-loop swap (both of pair correct) | **1/9** | **1/9** |

      Success rose; swap did not move. Per the pre-registered table below this is the
      saliency-shortcut row: short-horizon training improves drive-to-nearby-object
      competence, but **keeping the goal in view for the entire approach is not sufficient
      for SmolVLA to bind the instruction to the target** — the 2 wrong-prop reaches are
      that shortcut observed directly. The horizon hypothesis is refuted for grounding;
      grounding stays with the D9 acquisition path (measured 94% correct-prop offline).
      Training: 539 eps / 54,194 frames (1.5 epochs vs v3's 1.1), loss 3.43 -> 0.52,
      v3's exact recipe. Logs: eval_results/eval_{v4,v3ctl}_open_ground_v4scenes.log.
      parking_lot eval for both checkpoints: optional follow-up (needs world swap); the
      open_ground pair already answers the hypothesis.

      Ops incidents recorded during the run, both fixed in-tree: (a) run_eval's referee
      loop could hang forever when a DDS-wedged sim stalled /clock (leaked tracker/client
      nodes, load runaway, 91 min on one seed) -> 90 s wall-clock backstop guarantees
      teardown, wall_timeout in the log is now the wedge signal, recovery needs a sim
      restart; (b) grep|tee block-buffering hid eval output until pipe close -> stdbuf -oL.

      --- original pre-registered design below ---

      Question: does shortening the goal range restore language grounding, or was
      M1's failure caused by something else?

      **Held fixed** (matching `stage1_v3` exactly): 10000 steps, batch 8, lr 1e-4,
      `num_vlm_layers=16`, `freeze_vision_encoder=true`, `chunk_size=1`, frozen backbone.
      **Varied**: goal range only (2.0-7.0 m -> 2.0-3.5 m).

      Two confounds, both fixed BEFORE the run rather than caveated after:

      1. *Dataset size.* Dropping corridor plus shorter episodes left v4 at ~374 episodes /
         ~37k frames vs v3's 520 / 71.5k - half the data, and 2.2 epochs vs 1.1 at the same
         step count. A weak v4 would then be unattributable (horizon vs. less data).
         Fixed by 166 extension seeds -> ~520 episodes.
      2. *Eval distribution.* v4 must be evaluated on v4-style scenes (2.0-3.5 m), but then
         a better v4 score could just mean short scenes are easier to evaluate on.
         **Control: run the existing `stage1_v3` checkpoint on the SAME v4-style eval
         scenes.** Costs one extra eval run and is the only thing that separates
         "short-horizon training helps" from "short-horizon eval is easier".

      | outcome | reading |
      |---|---|
      | v4 swap-flip >> v3-on-short-eval | horizon was the cause; hypothesis confirmed |
      | v4 ~= v3-on-short-eval, both high | the eval got easier; training changed nothing |
      | v4 success up, swap-flip ~ chance (0.25) | the saliency shortcut at close range - a NEGATIVE result despite the better success number |
      | v4 swap-flip ~ chance, success flat | horizon was not the cause; grounding fails for another reason |

      Baseline to beat: `stage1_v3` swap-flip 0.18, directional 0.71, 2-3/10 closed-loop.
