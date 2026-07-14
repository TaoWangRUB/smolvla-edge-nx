# SmolVLA on the Edge — Deploying a Flow-Matching VLA on 8 GB Jetson Xavier NX

Fine-tune [SmolVLA](https://huggingface.co/lerobot/smolvla_base) on a public SO-101
manipulation dataset, then **deploy and benchmark it on a Jetson Xavier NX (8 GB)** — the
part most tutorials skip.

> **The thesis of this repo:** fine-tuning a VLA is table-stakes. The interesting,
> portfolio-worthy engineering is getting a SmolVLM-2 + flow-matching-expert policy to run
> under real-time, on-device, 8 GB edge constraints — and being honest about *what converts,
> what doesn't, and the latency budget you hit anyway.*

No robot arm required: SmolVLA fine-tunes from public Hugging Face datasets, and evaluation
runs **closed-loop in the gym-aloha MuJoCo simulator** (plus open-loop replay as a fallback).
The Xavier NX edge phase is fully specced and kicks in whenever a Jetson is on hand.

---

## Demo — the fine-tuned SmolVLA doing the task, with its latency on screen

![This repo's fine-tuned SmolVLA running the bimanual cube transfer closed-loop, with sim-time, control rate, per-step policy latency and task reward overlaid](benchmarks/results/demo_smolvla.gif)

**What you're watching.** This repo's **fine-tuned SmolVLA** completing the bimanual cube
transfer **closed-loop** in `gym_aloha/AlohaTransferCube-v0`, rendered by the eval harness
(`scripts/make_demo_gif.py --mode rollout`). Not a replayed demonstration — every action is
produced by the network from the current camera image + joint state.

- **Model:** SmolVLA, 450 M params (SmolVLM-2 backbone + flow-matching action expert),
  language-conditioned — instruction: *"Pick up the cube with the right arm and transfer it
  to the left arm."*
- **Fine-tune:** single **Colab A100**, **20 k steps**, batch 64, **~2.5 h**
  (`notebooks/colab_train_smolvla_aloha.ipynb`)
- **Scene:** two ViperX arms (14 DoF total), one 480×640 top camera
- **Result:** **70 % success** over 20 episodes — vs 65 % for the ACT baseline (see Results)

**Timing, plainly.** The robot runs a **50 Hz control loop**: it needs one action every
**20 ms**. SmolVLA does not run its network every step — one full inference produces a
**chunk of 50 actions**, which are then executed one per control step. So each control step
is one of two kinds (all times measured on the RTX 2000 Ada by the CUDA-synced per-step
timer shown in the GIF header):

- **Replay step (49 out of every 50): ~5–7 ms.** The robot executes a precomputed action
  from the chunk — **no neural network runs**. The 5–7 ms is purely our harness preparing
  the *next* observation (camera image → tensor → GPU, tokenize the instruction, normalize).
  On these steps that preparation is even discarded — a known optimization to reclaim.
- **Inference step (1 out of every 50): ~300 ms.** The full model runs once: the SmolVLM-2
  backbone encodes image + instruction + state, then the flow-matching action expert
  integrates 10 steps to produce the next 50 actions.

The arithmetic that follows from this:
- 50 actions ÷ 50 Hz → **one chunk covers exactly 1 s of robot motion**, so the full
  network runs **once per second** — that (and nothing else) is the "VLM at 1 Hz" claim.
- 300 ms ÷ 50 actions = **6 ms of compute per executed action**, under the 20 ms budget —
  the GPU keeps up on average.
- But at each inference step the robot would stand still for **300 ms = 15 missed control
  ticks** if prediction and execution were serialized. Decoupling them (compute the next
  chunk while the current one is still executing) hides the stall — doing that within 8 GB
  is Phase 2's job.

For comparison, the **ACT baseline** (~52 M task-specific specialist) through the identical
harness — far cheaper per step, but no language conditioning and a lower success rate (see
Results):

![The pretrained ACT baseline running the same task through the same harness](benchmarks/results/demo.gif)

**The header, field by field:**

| field | meaning |
|---|---|
| `sim t` | simulated time (steps × 20 ms). On a physical robot this trajectory would take the same wall-clock time — the whole handover is ~6.5 s |
| `50 Hz` | the control loop: one action consumed every 20 ms of sim time |
| `policy X ms` | wall time to obtain **that step's** action (CUDA-synced, RTX 2000 Ada). SmolVLA: ~5–7 ms queue pops, **~300 ms** chunk-boundary refills. ACT baseline: ~1 ms pops, ~14 ms refills. That pop-vs-refill rhythm *is* action chunking |
| `reward N/4` | gym-aloha's contact-based progress ladder: 1 = right gripper touches the cube, 2 = lifted off the table, 3 = left gripper touches it, 4 = left arm holds it alone → **SUCCESS**. An episode counts as a success iff it reaches 4 |

After success the sim runs ~1.5 s longer (so the GIF doesn't cut at the handover instant) and
holds the final frame. Regenerate with any checkpoint:
`python scripts/make_demo_gif.py --mode rollout --policy-path <ckpt> --task "<instruction>"`.

---

## Why this project

The role this targets emphasizes *real-time / on-device / edge constraints* and *strong
latency on real robots*. Almost every SmolVLA example deploys to a workstation or a physical
arm; very few tell the **8 GB Jetson optimization story**. This repo owns exactly that gap.

Effort is weighted accordingly: get a *correct* checkpoint fast, then spend the real time on
edge deployment and latency engineering.

---

## Track scope

- **In scope — manipulation in simulation (ALOHA sim).** With no robot on hand, the correctness
  loop runs entirely in the LeRobot-native gym-aloha MuJoCo env: fine-tune on
  `lerobot/aloha_sim_insertion_human`, evaluate **closed-loop** with the env's own success flag.
  The real SO-101 path (`configs/train.so101_pickplace.yaml`) is kept for when hardware exists.
- **Optional — Jetson Xavier NX edge deployment.** Kept fully specced (Phase 2) but parked until
  a Jetson is on hand.
- **Future work — mobile rover.** A rover is a different embodiment (mobile base, not an arm).
  Adapting SmolVLA to it is a research project, not a two-week demo. See the *Non-Goals* in
  [the change design](openspec/changes/smolvla-edge-deployment/design.md).

---

## Roadmap

Progress: **18 / 27 tasks** — details in
[the change tasks](openspec/changes/smolvla-edge-deployment/tasks.md).

**Headline result — the head-to-head is in: the fine-tuned SmolVLA wins.** On
`AlohaTransferCube-v0`, identical 20-episode protocol, matched simulator:
**SmolVLA 14/20 = 70 %** vs **ACT baseline 13/20 = 65 %** — the language-conditioned
generalist beats the task-specific specialist, at ~40× the inference compute. The SmolVLA
checkpoint was fine-tuned on a **single Colab A100** (20 k steps, batch 64, ~2.5 h); all
evaluation and latency numbers below come from the local RTX 2000 Ada container. That compute
gap (36 Hz ceiling vs the 50 Hz control loop) is exactly what the edge phase exists to close.

| Phase | What | Status | Notes |
|-------|------|--------|-------|
| 0 | **Scaffold + environment** — repo, pins, host env, **Docker env** | ✅ 6/6 | matched-mujoco container built & verified |
| 1 | **Correctness (sim)** — verify-first, fine-tune SmolVLA, closed-loop eval | ✅ 6/6 | **Deliverable: fine-tuned SmolVLA 70 % success** (transfer cube, 20 eps, matched mujoco) vs official ACT baseline **65 %** on identical seeds; trained 20k steps on Colab A100 |
| 2 | **Edge deployment** (optional) — Xavier NX on-device + client/server | ⏸ 0/7 | parked until a Jetson NX is on hand; chunking, low-Hz VLM, INT8-where-it-converts |
| 3 | **Benchmarks + writeup** — latency table + demo GIF + narrative | 🔄 3/5 | ✅ demo GIFs (fine-tuned SmolVLA + ACT baseline, latency overlays), collate, narrative through Phase 1; NX benchmark tiers pending hardware |

**Measured so far** (evaluated on RTX 2000 Ada, matched-mujoco container; SmolVLA trained on a Colab A100):

| | ACT (80 M specialist, official pretrained) | SmolVLA (450 M generalist, fine-tuned on A100) |
|---|---|---|
| transfer-cube success (20 eps) | 65 % | **70 %** |
| select_action mean / throughput | 0.68 ms / 1474 Hz | 27.7 ms / 36 Hz (chunk-boundary VLM prefill dominates) |
| peak GPU memory | 266 MB | 927 MB |

**Failure modes** (for the writeup): SmolVLA's 6 failures were mostly post-grasp stalls
(reward 1–2); ACT's included one complete miss (reward 0). Neither drops the cube post-transfer.
Two transferable findings: (1) **the simulator version is part of the eval** — the same ACT
checkpoint scores 60 % under mujoco 3.10 vs 80 % under the matched 2.3.7 container; (2)
**verify-first pays** — one pretrained-policy rollout caught a normalization bug that silently
zeroed success rates before any GPU-hours were spent.

Training pipeline hardening from the Colab sessions (HF Xet downloads unreliable from Colab →
datasets/models staged from Drive tarballs; full findings in
[the change design](openspec/changes/smolvla-edge-deployment/design.md)).

---

## Environment

### Docker (preferred) — matched-simulator container

The recommended way to run everything (sim eval, inference, fine-tuning, benchmarks) is the
Docker environment: [docker/Dockerfile](docker/Dockerfile) + [docker-compose.yml](docker-compose.yml),
following the same conventions as the BEV_Jetson / rover projects (nvidia runtime, repo mounted
at `/workspace`, per-purpose compose services).

**Why a container is not just convenience here — it fixes a real version conflict:**

- `lerobot >= 0.5.0` requires **Python ≥ 3.12**
- gym-aloha's pinned `mujoco 2.3.7` (the version the ALOHA sim datasets/checkpoints were
  generated with) only has wheels for **Python ≤ 3.11**

These are mutually exclusive in one native env. The container runs **py3.11 + lerobot 0.4.4 +
the matched mujoco 2.3.7 / dm_control 1.0.14 pair** — and the match is measurable: the pretrained
ACT transfer-cube checkpoint scores **80 % success in-container vs 60 % on a host mujoco 3.x**
stack. Details in [the change design](openspec/changes/smolvla-edge-deployment/design.md)
(*"Simulation setup — verified findings"*).

Prerequisites: Docker with the NVIDIA container runtime (`docker info | grep -i nvidia`).
Compose v2 (`docker compose`) or legacy v1 (`docker-compose`) both work.

```bash
docker compose build                    # build the smolvla-edge:sim image (once, ~8 GB)

docker compose run --rm verify          # known-good baseline: pretrained ACT on transfer cube
                                        #   -> expect ~80% success (4/5 episodes)

docker compose run --rm shell           # interactive shell inside the container

# generic eval — pass any smolvla_edge.eval flags via EVAL_ARGS:
EVAL_ARGS="--mode sim --policy-path lerobot/act_aloha_sim_insertion_human \
           --env-id gym_aloha/AlohaInsertion-v0 --episodes 10 --task ''" \
  docker compose run --rm eval

docker compose run --rm infer           # smoke-test smolvla_base on its SO-101 dataset
docker compose run --rm train           # fine-tune via scripts/train.sh (needs a big GPU)
BENCH_ARGS="--policy-path <ckpt> --precision fp16" docker compose run --rm bench
```

Notes:
- The repo root is mounted at `/workspace`; edits on the host are live in the container.
- Model/dataset downloads persist across runs in the `hf-cache` / `torch-cache` volumes.
- Headless rendering uses `MUJOCO_GL=egl` (GPU). If EGL is unavailable:
  `MUJOCO_GL=osmesa docker compose run --rm verify` (CPU rendering, slower).
- Set `HF_TOKEN=...` in the environment for authenticated/faster HF downloads.

### Native (host) install — alternative

A host install works too, but which mujoco you get depends on the Python version, and
**mujoco 3.x will under-score checkpoints/datasets generated under 2.x** (see above):

```bash
# Python 3.10/3.11: requirements.txt works as-is (matched mujoco 2.3.7)
pip install -r requirements.txt && sudo apt-get install -y ffmpeg

# Python 3.12: gym-aloha's mujoco pin has no wheel — use the verified workaround
bash scripts/setup_sim.sh               # lerobot 0.5.0 + mujoco 3.x + gym-aloha --no-deps
```

### Hardware

- **Dev + inference:** any CUDA GPU (verified on an RTX 2000 Ada laptop GPU, WSL2).
- **Training run:** rent an A100/H100 for a few hours (≈20k steps ≈ 4 h on a single A100).
- **Edge target (optional):** Jetson Xavier NX, 8 GB — see the *Xavier NX (JetPack) setup*
  section of [the change design](openspec/changes/smolvla-edge-deployment/design.md); the Jetson
  is its own world (aarch64 wheels, TensorRT, power modes) and does not use this image.

---

## Quickstart (Docker)

```bash
# 0. Build the image, then prove the whole sim/eval pipeline with a pretrained policy
#    BEFORE training anything (verify-first): env, rollout, normalization, success metric.
docker compose build
docker compose run --rm verify          # pretrained ACT, transfer cube -> ~80% success

# 1. Smoke-test the SmolVLA base model (pairs with its SO-101 embodiment dataset).
docker compose run --rm infer

# 2. Fine-tune SmolVLA on the ALOHA sim dataset (run on a big GPU; see configs/train.aloha_sim.yaml).
#    Local smoke run: BATCH_SIZE=4 STEPS=1000 docker compose run --rm train
#    Full 20k-step run on Colab: notebooks/colab_train_smolvla_aloha.ipynb (same lerobot
#    version as the container -> the checkpoint drops straight into eval below)
docker compose run --rm train

# 3. Evaluate YOUR checkpoint closed-loop in sim -> the success-rate deliverable.
EVAL_ARGS="--mode sim --policy-path outputs/train/smolvla_aloha/checkpoints/last \
           --env-id gym_aloha/AlohaInsertion-v0 --episodes 20" \
  docker compose run --rm eval

# 4a. Latency benchmark for one deployment config.
BENCH_ARGS="--policy-path <checkpoint> --device cuda --precision fp16 --chunking on \
            --tag local-fp16 --out benchmarks/results/raw/local_fp16.json" \
  docker compose run --rm bench

# 4b. (optional, with a Jetson) client/server: policy on the workstation, NX as control client.
python deploy/client_server/server.py --policy-path <checkpoint>   # on the workstation
python deploy/client_server/client.py --server <host:port>         # on the Xavier NX
```

Every step also runs natively (same commands without the compose wrapper, e.g.
`python -m smolvla_edge.eval ...`) if you set up a host env per the *Environment* section.

---

## Deployment modes (Phase 2)

**On-device.** SmolVLA decouples action *prediction* from *execution*, cutting task time
~30% on average — lean on it. Run the VLM stage at low Hz, use action chunking, and apply
INT8/quantization where the graph converts. Honest framing: full TensorRT of a SmolVLM-2 +
flow-matching-expert VLA is non-trivial; the credible deliverable is the latency budget plus a
clear "what converted / what didn't" table.

**Client–server.** Policy on the Titan X workstation, NX as a thin control client over gRPC.
This mirrors how real customer robots offload inference and gives the second benchmark point.

See [deploy/README.md](deploy/README.md).

---

## Benchmarks (Phase 3 — the centerpiece)

The headline artifact is a results table across deployment tiers plus a short GIF of the
policy executing replayed episodes (no physical robot needed). Metrics: end-to-end latency,
action-chunk frequency, throughput, peak memory.

Results live in [benchmarks/results/](benchmarks/results/) and are summarized in
[benchmarks/README.md](benchmarks/README.md).

---

## Repo layout

```
smolvla-edge-nx/
├── src/smolvla_edge/      # infer / eval / bench entrypoints + shared utils
│                          #   eval.py: closed-loop gym-aloha rollouts (make_sim_stepper
│                          #   handles old- and new-format checkpoints transparently)
├── scripts/               # train.sh, setup_sim.sh, make_demo_gif.py (rollout GIFs)
├── notebooks/             # 01/02: Transformer->SmolVLA from-scratch tutorials;
│                          #   colab_train_smolvla_aloha.ipynb: the Colab fine-tune (T4/A100)
├── configs/               # training configs (aloha_sim primary, so101 kept for later)
├── docker/ + docker-compose.yml   # the preferred env: matched mujoco 2.3.7 container;
│                          #   services: verify / eval / infer / train / bench / shell
├── data/                  # dataset tarballs for Drive/Colab staging   (gitignored)
├── models/                # pretrained-model cache tarball for Colab   (gitignored)
├── outputs/               # local training checkpoints                 (gitignored)
├── deploy/
│   ├── ondevice/          # Xavier NX on-device notes, quantization/TRT attempts
│   └── client_server/     # gRPC server (workstation) + client (NX), proto
└── benchmarks/            # bench harness + results (summary.csv, demo.gif)
```

Project plans, design, specs, and the phased task list live in
[openspec/changes/smolvla-edge-deployment/](openspec/changes/smolvla-edge-deployment/).

## License

MIT — see [LICENSE](LICENSE).

## Acknowledgements

Built on [LeRobot](https://github.com/huggingface/lerobot) and the SmolVLA base model by the
Hugging Face robotics team.
