# SmolVLA on the Edge — Flow-Matching VLA on an 8 GB Jetson Xavier NX

Deploy a language-conditioned flow-matching VLA under real-time, on-device, 8 GB edge
constraints — and be honest about *what converts, what doesn't, and the latency budget you
hit anyway*. Fine-tuning is table-stakes; the engineering is everything after it.

The project is sim-first (no robot arm required) and proceeds in phases, each gated on
measured numbers. Manipulation phases 0–3 are **complete**; the current phase adapts the
same stack to a **1/16 Ackermann rover** (language-conditioned navigation, Gazebo
sim-first) — see [rover/README.md](rover/README.md).

---

## Project arc

| Phase | What | Status | Headline number |
|---|---|---|---|
| 1 | **Correctness (sim)** — fine-tune SmolVLA on ALOHA transfer-cube, closed-loop eval | ✅ | **70 %** success (20 eps) vs 65 % ACT baseline — the 450M generalist beats the 80M specialist |
| 2 | **Edge deployment** — Xavier NX on-device | ✅ | **233 ms/chunk** on the NX: from-source torch 2.2.2 (JP5), fp16 + manual full-forward CUDA-Graph capture, bitwise-identical actions (eager was 608 ms — launch-bound, not compute-bound) |
| 2b | **Async inference** — the SmolVLA paper's Algorithm 1, reproduced honestly | ✅ | success parity with sync, **19–21 % faster time-to-success, zero idle ticks** with the NX serving (ℓ_S ≈ 0.25 s) |
| 3 | **Production shape** — ROS 2 C++ async client; all-DDS policy hop | ✅ | C++ stack **80 %** @ native 25 Hz; all-ROS2 (Humble policy node on the NX ↔ Jazzy client over DDS) **80 %** — the rover's end-state architecture |
| 4 | **Rover VLA (current)** — Ackermann waypoint policy, Gazebo sim-first | 🚧 M1 | grounding diagnosed → goal-conditioned pivot (D9/D10); reference model measured: pose-conditioned **7/10** zero-shot vs language 4/10 vs trained SmolVLA 3/10 |

Authoritative plans, designs, and per-task measurement records live in OpenSpec:
[smolvla-edge-deployment](openspec/changes/smolvla-edge-deployment/) (38/38) ·
[ros2-cpp-async-deployment](openspec/changes/ros2-cpp-async-deployment/) (34/34) ·
[rover-vla-sim-first](openspec/changes/rover-vla-sim-first/) (active).

---

## Demo — the fine-tuned SmolVLA, latency on screen

![Fine-tuned SmolVLA running bimanual cube transfer closed-loop with per-step latency overlaid](benchmarks/results/demo_smolvla.gif)

Closed-loop in `gym_aloha/AlohaTransferCube-v0` — every action from the network, not a
replay. The rhythm visible in the header *is* action chunking:

- **50 Hz control loop**, one action per 20 ms tick; one inference yields a **50-action chunk**.
- **Replay steps** (49 of 50): ~5–7 ms — no network runs. **Inference steps** (1 of 50):
  ~300 ms — SmolVLM-2 prefill + 10 flow-matching steps.
- 300 ms ÷ 50 actions = 6 ms/action: the GPU keeps up *on average*, but serialized it would
  stall 15 ticks at every chunk boundary — hiding that stall is what the async phase solves.

Regenerate: `python scripts/make_demo_gif.py --mode rollout --policy-path <ckpt> --task "<instruction>"`.

---

## What was measured (the findings that transfer)

**Edge latency is a dispatch problem before it is a compute problem.** Eager fp16 on the NX
(608 ms) ran the GPU ~20 % busy — the Carmel CPU couldn't dispatch kernels fast enough.
Manual `torch.cuda.CUDAGraph` capture of the entire forward (static shapes: fixed camera,
padded tokens, 3 flow steps, injectable noise) gave **233 ms end-to-end, bitwise-identical
actions**. Runtime-level fusion lost: the ONNX/TorchDynamo export is 30 852 unfused nodes
(~1.0 s in C++ ORT), TensorRT rejects the graph — recording the framework's own kernel
stream beat every export path. Recipe productionized as `precision="fp16-graph"`
(`src/smolvla_edge/cuda_graph.py`); details in
[deploy/jetson-native-torch/README.md](deploy/jetson-native-torch/README.md) and
[deploy/README.md](deploy/README.md).

**Async has a hard operating envelope.** No-starvation and no-perpetual-replanning together
require in-loop chunk latency **ℓ < n·Δt/2** (500 ms at n=50, 50 Hz) — outside it no
threshold `g` works (measured: 45 % vs sync's 65 % at ℓ ≈ 0.9 s). And deep chunk splices
need seam smoothing: without `--ramp-in`, the executed-vs-new seam is a torque spike that
cost 30 points of success. Both findings, with queue traces matching the paper's Fig. 3,
are recorded in the [edge-deployment design](openspec/changes/smolvla-edge-deployment/design.md).

**The simulator version is part of the eval.** The same ACT checkpoint scores 80 % under the
matched mujoco 2.3.7 container vs 60 % under host mujoco 3.x — hence the pinned
py3.11 + lerobot 0.4.4 + mujoco 2.3.7 image everything runs in.

| Tier (fine-tuned SmolVLA unless noted) | Number |
|---|---|
| RTX 2000 Ada, PyTorch fp32 | ~300 ms/chunk |
| RTX A2000, fp16-graph | 53 ms/chunk |
| Xavier NX, ORT CUDA EP fp32 (torch-free fallback) | ~610 ms/chunk |
| **Xavier NX, native torch fp16-graph** | **233 ms/chunk** |
| NX-served closed loop, async g=0.5 | 70 % success, 0 idle ticks |
| ROS2 C++ client, native Linux 25 Hz, 50 eps | 80 % success |
| All-ROS2 (Humble node on NX ↔ Jazzy over DDS), 10 eps | 80 % success |

Full tier table: [benchmarks/README.md](benchmarks/README.md) ·
[benchmarks/results/](benchmarks/results/).

---

## Current phase — rover VLA, simulation-first

The manipulation stack's end-state (on-device policy node, DDS policy hop, async chunks)
is the rover's starting architecture. New problem: *"drive to the red barrel"* → safe
Ackermann motion — body-frame waypoint chunks from a camera + instruction, developed in
Gazebo Harmonic against a privileged A* expert, with a swap test (same scene, instruction
on goal vs hard negative) as the grounding centerpiece.

Status (M1): navigation works, **end-to-end language grounding does not — in any model
tested**. Six interventions on SmolVLA left the swap test at chance; the released
OmniVLA-edge navigation specialist (9× smaller) scored 12/12 offline / 7/10 closed-loop
when given a **goal pose** but 2/12 / 4/10 on language, on identical frames — while CLIP
reads the prop colours fine (9/11). The failure is isolated to binding words to spatial
targets inside the policy; the measured escape is the mission-layer split: an
open-vocabulary detector *selects* (94 % offline), goal memory persists the target in the
odom frame, and the policy *steers* on a goal channel (design D9/D10, tasks 2.10–2.12).

Everything — vehicle sim, expert datagen, training recipes, eval harness, the grounding
diagnosis, and the per-decision record — lives in [rover/README.md](rover/README.md) and
[openspec/changes/rover-vla-sim-first/](openspec/changes/rover-vla-sim-first/).

---

## Quickstart

Docker with the NVIDIA runtime; the container fixes a real conflict (lerobot ≥ 0.5 wants
py ≥ 3.12; the matched mujoco 2.3.7 has wheels only ≤ 3.11).

```bash
docker compose build                    # smolvla-edge:sim (~8 GB, once)
docker compose run --rm verify          # pretrained ACT baseline -> ~80% (verify-first)
docker compose run --rm infer           # smoke-test smolvla_base
docker compose run --rm train           # fine-tune (big GPU; Colab notebook provided)

# closed-loop eval of your checkpoint
EVAL_ARGS="--mode sim --policy-path outputs/train/smolvla_aloha/checkpoints/last \
           --env-id gym_aloha/AlohaTransferCube-v0 --episodes 20" \
  docker compose run --rm eval

# async client/server (two shells) and the ROS2 stack: see deploy/README.md
# Jetson on-device: docker compose -f docker-compose.jetson.yml run --rm bench
# rover sim procedure: rover/README.md
```

---

## Repo layout

```
smolvla-edge-nx/
├── src/smolvla_edge/      # eval / bench entrypoints; AsyncRunner (paper Alg. 1);
│                          #   cuda_graph.py (fp16-graph capture)
├── scripts/               # train.sh, make_demo_gif.py, deploy_pipeline.sh
├── notebooks/             # from-scratch tutorials + the Colab fine-tune notebook
├── configs/               # training configs
├── docker/                # sim container + ROS2 overlay + Jetson images
├── deploy/
│   ├── ondevice/          # NX on-device notes, ONNX export + parity gate
│   ├── client_server/     # gRPC policy/sim servers, tick profiler
│   ├── ros2/              # colcon ws: C++ async client, bridge, NX policy node
│   └── jetson-native-torch/  # the JP5 from-source torch + CUDA-graph recipe
├── benchmarks/            # harness + results (summary.csv, GIFs, ros2/)
├── rover/                 # ← current phase: Ackermann rover VLA, Gazebo sim-first
│   ├── ros2/              #   rover_sim / rover_expert / rover_runtime packages
│   ├── datagen/           #   expert episodes -> LeRobot datasets (hindsight chunks)
│   ├── runtime/           #   policy servers (SmolVLA + OmniVLA-edge reference)
│   └── eval*/             #   probes, closed-loop results, grounding diagnosis
└── openspec/              # proposals, designs (D-numbered decisions), task records
```

## License

MIT — see [LICENSE](LICENSE). Built on [LeRobot](https://github.com/huggingface/lerobot)
and the SmolVLA base model by the Hugging Face robotics team.
