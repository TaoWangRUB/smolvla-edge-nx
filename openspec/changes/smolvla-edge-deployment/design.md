## Context

SmolVLA is a SmolVLM-2 backbone plus a flow-matching action expert, pretrained on SO-100/SO-101
manipulation data. The deliverable is a demo: a correct checkpoint (fine-tuned + evaluated
**entirely in simulation**, since no robot is on hand) plus — optionally, if a Jetson is available —
a credible edge-deployment and latency story on a Xavier NX (8 GB). No robot arm and no new hardware
are required for the correctness half: fine-tuning uses a public Hugging Face ALOHA sim dataset, and
eval runs **closed-loop in the gym-aloha MuJoCo env** (its success flag is the quoted number).

Hardware in play:
- **Dev + inference**: Titan X (Maxwell, 12 GB, pre-Tensor-core).
- **Training run**: rent an A100/H100 for a few hours (~20k steps ≈ 4 h on a single A100). The
  Titan X can train at batch 16 (~6 GB) but is much slower, so it stays a dev box.
- **Edge target**: Jetson Xavier NX, 8 GB.
- **LeRobot**: pinned to `v0.5.0`.

The Jetson is its own world — aarch64 wheels, NVIDIA-provided torch, TensorRT baked into JetPack,
and power modes that materially change benchmark numbers. `pip install torch` from PyPI must not be
run there.

## Goals / Non-Goals

**Goals:**
- A *correct* fine-tuned ALOHA-sim checkpoint with a **closed-loop** success-rate number from
  gym-aloha rollouts, reached with **no robot** so the correctness half always ships.
- *(Optional, if a Jetson NX is on hand)* the policy running entirely on the Xavier NX under
  real-time constraints, with an honest accounting of what converted to a faster engine and what
  didn't.
- A reproducible latency/throughput/memory table (local GPU always; NX tiers when available) plus
  a short demo GIF built from ALOHA sim frames.
- *(Optional)* a second deployment point (workstation gRPC server + thin control client) that
  mirrors how real robots offload heavy inference.

**Non-Goals:**
- **Mobile-rover embodiment.** The on-hand rover is an Ackermann/mobile base, not an arm. SmolVLA's
  pretraining prior is for manipulation, so adapting it means new observation/action spaces, a new
  action head, and more aggressive re-training — a research project, not a two-week demo. The
  Xavier NX latency work carries over unchanged, which is exactly why manipulation is done first.
  Framing for the writeup: "manipulation validates the VLA + edge pipeline; the rover is the
  natural next embodiment, and the edge-deployment work carries over unchanged."
- A clean end-to-end TensorRT engine for the whole VLA (see Risks).
- Training a policy from scratch — this is fine-tuning from `lerobot/smolvla_base`.

## Decisions

**Evaluate in simulation (gym-aloha), not on a robot or in Isaac Sim / Gazebo.** With no arm on
hand, the correctness loop needs a simulator that plugs into LeRobot's dataset/policy API. The
turnkey choice is **gym-aloha**, a LeRobot-native MuJoCo env with ready datasets
(`lerobot/aloha_sim_insertion_human`) and a first-class success flag — it wires straight into
`smolvla_edge.eval --mode sim`. *Isaac Sim / Gazebo were rejected:* neither is integrated with
LeRobot's observation/action schema, so they'd require hand-building the obs bridge, camera
rendering, and a matching dataset (weeks of plumbing, plus Omniverse/ROS overhead) for zero
leverage on a train→eval→demo loop. *LIBERO / Meta-World* (the SmolVLA paper's sim benchmarks) are
a stretch goal — more impressive but not turnkey in LeRobot v0.5.0 (undocumented wiring). Trade-off
accepted: ALOHA is bimanual (14-D action) rather than the SO-101 embodiment SmolVLA was pretrained
on, so the fine-tune adapts the state/action projectors; the obs→policy key mapping in
`eval._aloha_obs_to_batch` must be verified against `policy.config.input_features` on the dev box.

**Fine-tune, don't train from scratch.** SmolVLA is pretrained on manipulation data, so fine-tuning
from `lerobot/smolvla_base` on the ALOHA sim dataset (`lerobot/aloha_sim_insertion_human`) is the
turnkey path to a correct checkpoint. The real SO-101 path (`configs/train.so101_pickplace.yaml`)
is kept for when a robot is available. Alternative (rover / from-scratch) rejected as out of scope.

**Smoke-test before training anything.** Run `smolvla_edge.infer` on `lerobot/smolvla_base`
against the dataset first, so stack/version problems surface before burning rented-GPU hours.

**Split training off the dev box.** Rent an A100/H100 for the ~20k-step run; keep the Titan X for
dev and inference. Local training is possible at batch 16 (~6 GB) but too slow to iterate on.

**On-device optimization order (by payoff):**
1. *Action chunking + decoupled execution* — SmolVLA separates action prediction from execution,
   cutting task time ~30% on average; predict a chunk, execute it open-loop while the next
   prediction computes.
2. *Low-Hz VLM stage* — the SmolVLM-2 backbone is the expensive part; run it at a lower rate than
   the action expert / control loop.
3. *Precision* — FP16 first (near-free, big memory win on 8 GB). INT8 only via real
   quantization/TensorRT on subgraphs that convert, never a naive cast.

**Client/server over gRPC.** A `policy.proto` defines the RPC; the workstation serves the policy,
the NX client replays held-out frames as observations and reports round-trip latency split into
server-compute vs. network overhead. Chosen to mirror real offload deployments and to add a second
benchmark tier cheaply.

**Benchmark harness owns honesty.** `smolvla_edge.bench` drops per-run JSON into
`benchmarks/results/raw/`; `benchmarks/collate.py` regenerates `summary.csv` and the markdown table.
The harness refuses naive INT8 casts so the INT8 row can only be filled by a real engine.

## Xavier NX (JetPack) setup

Operational reference for standing up the edge target (migrated from `docs/setup-jetson.md`):

1. **Base.** Flash a recent JetPack and record the exact version — it pins the CUDA/cuDNN/TensorRT
   triple. Confirm with `nvcc --version` and `dpkg -l | grep -i tensorrt`. Max out clocks for
   benchmarking and record the mode: `sudo nvpmodel -m 0` (record `nvpmodel -q`) then
   `sudo jetson_clocks`.
2. **PyTorch.** Install the NVIDIA aarch64 torch wheel matching the JetPack (from the NVIDIA Jetson
   PyTorch index/forum), then torchvision built against it. Verify with
   `python -c "import torch; print(torch.__version__, torch.cuda.is_available())"`.
3. **LeRobot + this repo.** `pip install -r requirements.txt` (torch is already installed; pip must
   not replace it) then `pip install -e .`. If `lerobot[smolvla]` tries to pull a PyPI torch,
   install LeRobot with `--no-deps` and add its remaining deps manually so the NVIDIA torch stays.
4. **Memory headroom on 8 GB.** Prefer FP16; mount swap/zram if you hit OOM during model load.
   Close the desktop GUI (`sudo systemctl isolate multi-user.target`) to reclaim VRAM. Watch live
   with `tegrastats` (or `jtop` from `jetson-stats`).
5. **Sanity check.** `python -m smolvla_edge.infer --policy-path <checkpoint> --episodes 1
   --max-frames 10`, then proceed to on-device deployment and the benchmark commands.

## Simulation setup — verified findings

Standing up gym-aloha and running a pretrained policy end-to-end surfaced three things worth
recording (all confirmed on Python 3.12 / torch 2.10 / mujoco 3.10 / lerobot 0.5.0, WSL2 + RTX 2000
Ada, headless EGL):

1. **mujoco 2.3.7 (gym-aloha's pin) has no py3.12 wheel.** Install a modern mujoco 3.x + matching
   dm_control and add gym-aloha with `--no-deps` (see `scripts/setup_sim.sh`). gym-aloha is a thin
   dm_control wrapper, so mujoco 3.x works at runtime; the dm_control↔mujoco pair MUST match
   (a skew shows up as `AttributeError: 'MjModel' object has no attribute 'flex_interp'`).
2. **LeRobot 0.5.0 externalized normalization.** Input/output normalization moved out of the policy
   model into a separate processor pipeline (`make_pre_post_processors`), so calling
   `select_action` on raw inputs yields garbage (success 0, reward 0). `eval_sim` now applies the
   processor pipeline for 0.5.0 checkpoints and falls back to stats baked into an old checkpoint's
   `model.safetensors`. Old pretrained checkpoints (e.g. `lerobot/act_aloha_sim_insertion_human`)
   are old-format (no processor json) → the baked-in path is used.
3. **Sim-version gap — and the Docker fix.** The HF ALOHA datasets and old checkpoints were
   generated with mujoco 2.x; evaluating in mujoco 3.10 transfers coarse behavior but degrades
   fine manipulation. Measured with pretrained ACT checkpoints:
   - insertion (precision): **0/3**, stuck at grasp (reward 2/4);
   - transfer cube (coarser): **3/5 = 60%** success — the harness's verified positive baseline.
   The fix is the containerized environment (`docker/Dockerfile` + `docker-compose.yml`,
   conventions mirrored from BEV_Jetson / ackermann_rover_humble): a py3.11 image where
   `pip install gym-aloha` resolves the **matched mujoco 2.3.7 + dm_control 1.0.14** pair the
   checkpoints were trained under — CUDA torch base, repo mounted at `/workspace`,
   `runtime: nvidia` + EGL for headless rendering, HF cache in a named volume, per-purpose
   services (`verify`, `eval`, `infer`, `train`, `bench`, `shell`). Train and evaluate inside the
   container so data generation and eval share one mujoco.

## Colab training — verified findings

Standing up the 20k-step fine-tune on Colab (notebook `notebooks/colab_train_smolvla_aloha.ipynb`)
surfaced its own set of hard-won facts:

1. **HF Xet downloads are unreliable from Colab.** Xet-backed files (dataset parquets, model
   safetensors) throttle to ~KB/s unauthenticated, 403 via the xet-bridge with lerobot's pinned
   hub 0.35 ("invalid key pair id"), and stall mid-file even authenticated with a current client;
   plain-CDN files (MP4s) download fine — hence wildly confusing mixed symptoms. Resolution:
   **stage everything from Drive tarballs** packaged from the local (working) machine — dataset
   (~90 MB) + model caches (~1.6 GB) — and verify tarball completeness by loading the FULL dataset
   through LeRobotDataset before packaging (a partial bench-download tarball caused a mid-run fetch).
2. **lerobot-train refuses any pre-existing output dir** without `--resume` — even an empty one
   just created for a tee log. Console logs must be written NEXT TO the output dir; the notebook
   wipes checkpoint-less leftover dirs and protects dirs holding real checkpoints.
3. **Version triangle**: lerobot 0.5.x needs py>=3.12; Colab is py3.12, so the notebook uses
   lerobot 0.4.4 (same as the Docker image) making the checkpoint drop-in compatible with local
   eval. huggingface_hub must stay <0.36 (a `-U` to hub 1.x breaks lerobot + transformers).
4. **Warmup auto-scales with run length** (1000 steps → 6 for a 200-step dry run, → 666 for 20k),
   so early-step losses are not comparable across run lengths.
5. **Hardware economics measured**: T4 5.35 s/step @ batch 32 (~30 h for 20k) vs A100 0.45 s/step
   @ batch 64 (~2.5 h). Dry-run on T4, full run on A100.

## Risks / Trade-offs

- **Full TensorRT of a SmolVLM-2 + flow-matching VLA is non-trivial** → Don't claim a clean
  end-to-end TRT engine. Deliver "what converts" (vision encoder / static subgraphs → engine +
  speedup), "what doesn't" (dynamic control flow, the flow-matching sampler loop → why + fallback),
  and the latency budget hit anyway with a per-stage breakdown. Capture attempts in
  `deploy/ondevice/conversion_notes.md`.
- **8 GB OOM on model load** → FP16, swap/zram, and closing the desktop GUI; measure peak memory in
  the benchmark harness.
- **JetPack drift changes numbers** → record JetPack version and power mode with every benchmark run
  so results stay comparable.
- **Dependency conflict: `lerobot[smolvla]` clobbering NVIDIA torch** → `--no-deps` install path on
  the NX, verified via the torch/CUDA import check.

## Migration Plan

- Move planning docs into this change and delete `docs/`; update `README.md` links from `docs/…`
  to the corresponding `openspec/changes/smolvla-edge-deployment/…` artifacts (and `openspec/specs/`
  once archived). Demo media that lived under `docs/assets/` moves alongside the benchmark outputs.
