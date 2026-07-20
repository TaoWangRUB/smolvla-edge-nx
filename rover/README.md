# Rover VLA — simulation-first workspace

Language-conditioned Ackermann navigation ("drive to the red barrel" → safe vehicle
motion): SmolVLA fine-tuned to emit body-frame waypoint chunks, developed
simulation-first in **Gazebo Harmonic**. The M0 feasibility gate rejected Isaac Sim on
available hardware (4 GB A2000; Titan X has no RT cores) — decision record in tasks.md §1.1.

**OpenSpec** (the authoritative plan; every decision and measurement is recorded there):

- [proposal.md](../openspec/changes/rover-vla-sim-first/proposal.md) — why, what changes, capabilities
- [design.md](../openspec/changes/rover-vla-sim-first/design.md) — decisions D1–D8 (simulator, action space, three-loop runtime, data, model, eval, deployment, reuse) + the AD upgrade path
- [tasks.md](../openspec/changes/rover-vla-sim-first/tasks.md) — M0–M4 milestones with per-task DONE records and measured numbers
- specs: [sim-environment](../openspec/changes/rover-vla-sim-first/specs/rover-sim-environment/spec.md) · [expert-datagen](../openspec/changes/rover-vla-sim-first/specs/rover-expert-datagen/spec.md) · [waypoint-policy](../openspec/changes/rover-vla-sim-first/specs/rover-waypoint-policy/spec.md) · [async-control](../openspec/changes/rover-vla-sim-first/specs/rover-async-control/spec.md) · [grounding-eval](../openspec/changes/rover-vla-sim-first/specs/rover-grounding-eval/spec.md) · [sim2real](../openspec/changes/rover-vla-sim-first/specs/rover-sim2real-deployment/spec.md)

## Layout

| Path | Contents |
|---|---|
| `ros2/src/rover_sim` | Vehicle + world: 1/16 Ackermann URDF (real dynamics, PX4/RealSense coupling stripped), OV9782-locked camera (1280×800 @ 15 Hz, HFOV 100°, fx=fy=537.0), 3 scene-family worlds, `scene_manager.py` randomizer, `state_publisher.py`, headless bringup launch |
| `ros2/src/rover_expert` | Privileged tooling: A*+Pure-Pursuit expert, episode recorder, `run_episode.py` orchestrator, `replay_episode.py` (M0 gate), `run_eval.py` (closed-loop eval + swap test) |
| `ros2/src/rover_runtime` | Policy runtime: `tracker_node.py` (50 Hz PP on chunks), `chunk_client_node.py` (async policy loop) |
| `datagen/` | `batch_datagen.sh`, `relabel.py` (+ unit tests), `instructions.py` (paraphrase pools), `to_lerobot.py` converter |
| `runtime/policy_server.py` | Torch-side chunk server (Titan X) |
| `train_smoke.sh` | Stage-one training entry (env-var scalable to full runs) |
| `data/`, `outputs/` | Generated episodes / checkpoints (gitignored) |

## Gazebo sim procedure

### Containers

Two images, two roles:

- **Sim (ROS 2 Jazzy + Gazebo Harmonic)**: `ackermann_rover_x86_64_jazzy` — or the locally
  committed `ackermann_rover_x86_64_jazzy:vla` which already has `ros-jazzy-gz-ros2-control`
  baked in (the base Dockerfile in `ackermann_rover_humble` also has the fix for future builds).
- **Torch (training / policy server / conversion)**: `smolvla-edge:sim` (lerobot 0.4.4,
  torch 2.6+cu124 — retains `sm_50`, so the Maxwell Titan X works).

```bash
docker run -d --name vla_sim --runtime nvidia \
  -e NVIDIA_VISIBLE_DEVICES=all -e NVIDIA_DRIVER_CAPABILITIES=all \
  -v "$PWD":/vla -w /vla --network host \
  ackermann_rover_x86_64_jazzy:vla sleep infinity

docker exec vla_sim bash -c 'source /opt/ros/jazzy/setup.bash && cd /vla/rover/ros2 && colcon build --symlink-install'
docker exec -d vla_sim bash -c 'source /opt/ros/jazzy/setup.bash && source /vla/rover/ros2/install/setup.bash && \
  ros2 launch rover_sim sim_bringup.launch.py world:=/vla/rover/ros2/install/rover_sim/share/rover_sim/worlds/open_ground.sdf'
```

The launch stands up: gz server (headless; `gui:=true` for the GUI, needs `xhost +local:`),
rover spawn, `ros_gz` bridge, `robot_state_publisher`, `state_publisher`, and the
`ros2_control` controllers. It contains **no trajectory logic** — trajectory authors
(expert or policy runtime) are separate per-episode processes entering via `/cmd_vel`.

### Observation / command contract ([design D3](../openspec/changes/rover-vla-sim-first/design.md))

All nodes run `use_sim_time`; `/observation` ≜ {image, camera_info, state}.

| Topic | Type | Dir | Notes |
|---|---|---|---|
| `/cmd_vel` | `geometry_msgs/TwistStamped` | in | (v, ω) controller reference; identical to the real rover's PX4 `rover_speed_steering` mapping |
| `/waypoint_chunk` | `std_msgs/Float32MultiArray` | in (tracker) | `[capture_t, K, x0,y0,v0, …]`, body frame at capture_t |
| `/observation/state` | `std_msgs/Float32MultiArray` | out | 50 Hz `[speed, yaw_rate, steering]`; yaw rate pose-derived, steering = bicycle-equivalent |
| `/ackermann/gt_odom` | `nav_msgs/Odometry` | out | GT pose 50 Hz — label source; derive yaw rate from poses, not its `twist` |
| `/vla_camera/image` (+`camera_info`) | `sensor_msgs/*` | out | 1280×800 RGB @ 15 Hz (`ros2 topic hz` under-reports; real subscribers get full rate) |

Measured vehicle limits: steering ±0.6 rad → **min feasible turn radius 0.341 m**
(inner-wheel limit). Anything commanding ω must clamp |ω| ≤ v/0.341 — over-commanding
breaks the Ackermann geometry and scrubs wheels (measured in M0).

### Clock discipline

Gazebo is the single time authority (`/clock` bridged). All stamps are sim-time
`header.stamp`; latency compensation counts sim-time deltas between image capture and
chunk application. Never `ros2 topic pub` against the controller (wall-clock stamps).

### Per-episode scene randomization (in-place, ~6 s, no restart)

```bash
ros2 run rover_sim scene_manager.py apply --scene corridor --seed 42 --out ep42.json
ros2 run rover_sim scene_manager.py clear --scene corridor
```

Seed-deterministic per episode (full config JSON-logged → failure slicing per
[grounding-eval spec](../openspec/changes/rover-vla-sim-first/specs/rover-grounding-eval/spec.md)):
sun (spawned as an `ep_light` entity — runtime `light_config` never reaches the sensors
render scene), ground-color slab (skipped in `parking_lot` to keep bay lines), prop
layout, spawn pose, goal + canonical instruction. Deferred to M1+: weather, exposure,
extrinsic jitter, sensor noise, texture-level materials.

### Ops gotchas (all discovered the hard way)

- **One gz server only** — two servers on the same world name merge gz-transport and
  requests round-robin between them; episodes half-apply. `pgrep -a ruby` before launching.
- **No default route** → `export GZ_IP=127.0.0.1` everywhere, or discovery fails.
- gz service CLI calls intermittently exceed 3 s under batch load — `scene_manager`
  retries ×3 (removals excepted: absent-entity negatives return fast and must not retry).
- The batch scripts kill sim processes between scene families; inside the *sim container*
  that's safe (own PID namespace). The `ackermann_rover_humble` compose containers use
  `pid: host` — never pattern-`pkill` from `docker exec` there.

## Training sampling strategy

**Seed ranges are the sampling design** (seed → deterministic episode):

| Range | Purpose |
|---|---|
| 1000–1189 / 2000–2189 / 3000–3189 | Training: open_ground / corridor / parking_lot (190 each) |
| 100–111 / 200–211 / 300–311 (+2, 5, 42–44) | Pilot/smoke episodes (kept in the dataset) |
| **9000+** | **Evaluation only — never recorded, never trained on** |

- **Language necessity is guaranteed per scene** ([sim-environment spec](../openspec/changes/rover-vla-sim-first/specs/rover-sim-environment/spec.md)):
  every episode contains a same-color/different-shape distractor AND a
  same-shape/different-color distractor beside the goal; exact goal (shape,color)
  duplicates are excluded so the instruction identifies a unique target.
- **Only successful expert episodes train** (goal ring reached AND min clearance > 0 —
  verdict in each `episode.json`). Failures are kept as verdicts for slicing, not as demos.
- **Retry pass**: failed (scene, seed) pairs are re-run once after a batch — seed
  determinism regenerates the identical episode, recovering infra-caused failures while
  genuinely blocked layouts fail again and stay excluded.
- **Instructions**: canonical template at record time; at dataset build each episode gets a
  seed-deterministic paraphrase from the 12-template train pool
  (`datagen/instructions.py`). The 6 held-out templates are reserved for evaluation
  (paraphrase-robustness metric, [design D6](../openspec/changes/rover-vla-sim-first/design.md)).
- **Action labels**: hindsight waypoint chunks — future GT poses at Δt=0.25 s transformed
  into the frame's body frame with per-point speed (`datagen/relabel.py`, unit-tested;
  episode end clamps to the stop point with v→0). Normalization to ~[−1,1] comes from
  LeRobot dataset stats computed at conversion.

### Batch generation + conversion

```bash
docker exec -d vla_sim bash /vla/rover/datagen/batch_datagen.sh          # SEEDS_PER_SCENE=190
# after completion: retry pass over failed seeds, then:
docker run --rm -v "$PWD":/work -w /work -e HF_HOME=/work/.hf_cache smolvla-edge:sim \
  python rover/datagen/to_lerobot.py --raw-root rover/data/raw --out rover/data/lerobot \
  --repo-id local/rover_vla_v2 --chunk-k 10 --paraphrase
```

Raw episode dirs keep the 50 Hz pose stream (relabeling stays lossless); the LeRobot
dataset carries `observation.image` (video), `observation.state`, `observation.gt_pose`,
`action` (flat 30-dim chunk), `task` (paraphrased instruction). Load with
`video_backend='pyav'`.

## Training pipeline ([design D5](../openspec/changes/rover-vla-sim-first/design.md))

Stage one: frozen VLM backbone, action expert trained — stock `lerobot-train` +
`lerobot/smolvla_base`:

```bash
REPO_ID=local/rover_vla_v2 ROOT=rover/data/lerobot/rover_vla_v2 \
STEPS=10000 BATCH_SIZE=8 OUTPUT_DIR=rover/outputs/train/stage1_v2 GPU=1 \
bash rover/train_smoke.sh
```

**Action-space decision (recorded in tasks.md §2.4)**: the K=10 × (x,y,v) chunk is stored
FLAT (30 dims ≤ SmolVLA's `max_action_dim=32`) with `--policy.chunk_size=1`, because
lerobot's `delta_timestamps` chunking would gather future frames' *own-body-frame* actions —
wrong frame semantics for [design D2](../openspec/changes/rover-vla-sim-first/design.md)'s
single-frame chunk. Revisit (custom collate → chunk_size=K, action_dim=3) if quality demands.

Set `WANDB=true WANDB_API_KEY=…` to stream metrics to the `rover-vla` project. Resume by
pointing `POLICY_PATH` at `…/checkpoints/last/pretrained_model` instead of the base.

### Training hosts

**Local Titan X (Maxwell, 12 GB)** — baked into `train_smoke.sh`:
- `--shm-size=8g` (docker's 64 MB `/dev/shm` default silently kills dataloader workers);
- lerobot's hardcoded `torch_dtype="bfloat16"` VLM load patched to `float32`
  (Maxwell: `CUBLAS_STATUS_NOT_SUPPORTED` on bf16 GEMM);
- fp32 @ batch 8 ≈ 0.86 s/step; a 10k-step run ≈ 2.5 h.

**Google Colab** — [`rover/train_colab.ipynb`](train_colab.ipynb) (open via
`colab.research.google.com/github/<owner>/smolvla-edge-nx/blob/feature/rover-vla-sim/rover/train_colab.ipynb`):
- A100/L4 give bf16 + batch 32–64 (~10× faster than the Titan X); T4/V100 auto-fall back to
  the fp32 patch at batch 16 (compute-capability detected in the notebook);
- **datagen and closed-loop eval stay local** — Colab does training only. Flow: regenerate the
  dataset locally → `tar czf` it → upload to Drive → run the notebook → download the checkpoint
  tarball → eval locally with `policy_server.py` + `run_eval.py`;
- streams to the same `rover-vla` wandb project (key via a Colab secret).

## Eval pipeline ([design D6](../openspec/changes/rover-vla-sim-first/design.md), tasks 2.7–2.8)

Three processes: sim (Jazzy container) + policy server (Titan X) + per-episode runtime trio.

```bash
# 1. policy server (torch side; command in rover/runtime/policy_server.py header)
# 2. closed-loop eval, eval-only seeds, in the sim container:
ros2 run rover_expert run_eval.py --scene open_ground --seed0 9000 --episodes 20 --swap
```

Per episode `run_eval.py`: resets the scene → starts `tracker_node` +
`chunk_client_node` (instruction as ROS param) → a privileged **referee** watches GT
odometry for reached (0.6 m ring) / collision (clearance ≤ 0) / timeout (40 s). The
policy sees only camera + state; privileged geometry stays in the referee.

- **Success rate**: `reached AND clearance > 0`, per scene family.
- **Swap test** (the grounding centerpiece): identical layout run twice — instruction on
  the goal, then on the same-shape/different-color hard negative (skipped when that
  combo isn't unique). Pair passes only if the rover approaches the *commanded* prop
  both times. Above-chance swap ⇒ the policy reads language, not saliency.
- Measured async envelope: ~280 ms/chunk on the Titan X ≈ 3.6 chunks/s vs the 2.5 s
  chunk horizon (≈9× replan overlap); tracker watchdog ramps to stop past 1 s staleness.

M1 exit ([tasks.md §2.8](../openspec/changes/rover-vla-sim-first/tasks.md)): above-threshold
success in training-like scenes + above-chance swap.

### M1 results so far + the grounding diagnosis

| Model | success | swap | note |
|---|---|---|---|
| stage1_v2 (frozen backbone) | 5/10 | 2/8 | best; swap ≈ chance |
| stage1b_v2 (full vision unfreeze) | 2/10 | 0/8 | **regressed** — full unfreeze ≠ design's LoRA |

The gate failed on grounding (swap ≈ chance). An **offline grounding probe**
([`rover/eval_results/grounding_probe.py`](eval_results/grounding_probe.py)) localized it:
the policy grounds **shape** (directional accuracy 0.71) but **not color** (swap-flip 0.10,
below chance) — it steers to a fixed salient object regardless of the color word. Root cause
was a **data confound**, not model capacity: the sampler placed the goal in the camera cone
while distractors scattered uniformly, so "drive to the object ahead" was a winning shortcut.
Fixed in `scene_manager.sample()` (all props in-cone, equal centrality → color is required).
Full writeup: [`rover/eval_results/grounding_diagnosis.md`](eval_results/grounding_diagnosis.md).
**Next:** regenerate → `local/rover_vla_v3` → retrain frozen backbone (Colab) → re-run swap.
If swap stays flat, the pre-committed escape valve is vision-encoder **LoRA** — low-rank and
constrained (the full unfreeze that regressed was *not* that; LoRA keeps the language model and
the base SigLIP weights frozen).

## Demos

`demo_open_ground.gif` / `demo_corridor.gif` (untracked): onboard-camera recordings of
expert episodes with instruction overlay — regenerate via the frame recorder pattern in
the session logs, or `run_episode.py` + ffmpeg over the saved frames.
