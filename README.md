# SmolVLA on the Edge — Deploying a Flow-Matching VLA on 8 GB Jetson Xavier NX

Fine-tune [SmolVLA](https://huggingface.co/lerobot/smolvla_base) on a public SO-101
manipulation dataset, then **deploy and benchmark it on a Jetson Xavier NX (8 GB)** — the
part most tutorials skip.

> **The thesis of this repo:** fine-tuning a VLA is table-stakes. The interesting,
> portfolio-worthy engineering is getting a SmolVLM-2 + flow-matching-expert policy to run
> under real-time, on-device, 8 GB edge constraints — and being honest about *what converts,
> what doesn't, and the latency budget you hit anyway.*

No robot arm and no new hardware required: SmolVLA fine-tunes from public Hugging Face
datasets, eval runs on replayed/held-out episodes, and deployment targets a Xavier NX that's
already on hand.

---

## Why this project

The role this targets emphasizes *real-time / on-device / edge constraints* and *strong
latency on real robots*. Almost every SmolVLA example deploys to a workstation or a physical
arm; very few tell the **8 GB Jetson optimization story**. This repo owns exactly that gap.

Effort is weighted accordingly: get a *correct* checkpoint fast, then spend the real time on
edge deployment and latency engineering.

---

## Track scope

- **In scope — manipulation (SO-101).** SmolVLA is pretrained on SO-100/SO-101 manipulation
  data, so a public SO-101 dataset is the turnkey path to a correct checkpoint.
- **Future work — mobile rover.** A rover is a different embodiment (mobile base, not an arm).
  Adapting SmolVLA to it is a research project, not a two-week demo. See the *Non-Goals* in
  [the change design](openspec/changes/smolvla-edge-deployment/design.md).

---

## Roadmap

| Phase | What | Status | Notes |
|-------|------|--------|-------|
| 0 | **Scaffold** — repo, MIT license, pinned LeRobot, env on Titan X | ☐ | `pip install "lerobot[smolvla]"` + ffmpeg/torchcodec |
| 1 | **Correctness** — validate stack with base-model inference, then fine-tune | ☐ | `lerobot/smolvla_base` → `lerobot/svla_so101_pickplace`, 20k steps |
| 2 | **Edge deployment** (the differentiator) — on-device + client/server | ☐ | Xavier NX 8 GB; chunking, low-Hz VLM, INT8 where the graph converts |
| 3 | **Benchmarks + writeup** (the centerpiece) — latency table + video | ☐ | Titan X vs NX (FP16/INT8, ±chunking) vs NX-client/workstation-server |

Track progress in [the change tasks](openspec/changes/smolvla-edge-deployment/tasks.md).

---

## Environment

- **Dev + inference:** Titan X (Maxwell, 12 GB — pre-Tensor-core).
- **Training run:** rent an A100/H100 for a few hours (≈20k steps ≈ 4 h on a single A100).
  The Titan X *can* train at batch 16 (~6 GB) but will be much slower; keep it for dev.
- **Edge target:** Jetson Xavier NX, 8 GB.
- **LeRobot:** pinned to `v0.5.0` (see [requirements.txt](requirements.txt)).

```bash
# On the Titan X dev box (and the training box)
python -m venv .venv && source .venv/bin/activate
pip install -U pip
pip install -r requirements.txt          # pins lerobot[smolvla]==0.5.0
# System deps for video decoding:
#   sudo apt-get install -y ffmpeg
#   (torchcodec is pulled in by lerobot[smolvla])
```

See the *Xavier NX (JetPack) setup* section of
[the change design](openspec/changes/smolvla-edge-deployment/design.md) for the edge setup, which
is its own world (aarch64 wheels, TensorRT, power modes).

---

## Quickstart

```bash
# 0. Smoke-test the stack on the base model BEFORE training anything.
python -m smolvla_edge.infer \
  --policy-path lerobot/smolvla_base \
  --dataset-repo-id lerobot/svla_so101_pickplace \
  --episodes 2

# 1. Fine-tune (run this on the rented A100/H100).
bash scripts/train.sh         # wraps lerobot-train, see configs/train.so101_pickplace.yaml

# 2. Evaluate the checkpoint on held-out episodes -> success rate.
python -m smolvla_edge.eval \
  --policy-path outputs/train/smolvla_so101/checkpoints/last \
  --dataset-repo-id lerobot/svla_so101_pickplace

# 3a. On-device benchmark (run on the Xavier NX).
python -m smolvla_edge.bench \
  --policy-path <checkpoint> --device cuda --precision fp16 --chunking on

# 3b. Client/server: policy on the workstation, NX as the control client.
python deploy/client_server/server.py --policy-path <checkpoint>   # on Titan X box
python deploy/client_server/client.py --server <host:port>         # on Xavier NX
```

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
├── scripts/               # train.sh and other thin CLI wrappers
├── configs/               # training + eval YAML configs
├── deploy/
│   ├── ondevice/          # Xavier NX on-device notes, quantization/TRT attempts
│   └── client_server/     # gRPC server (workstation) + client (NX), proto
└── benchmarks/            # benchmark harness + results table
```

Project plans, design, specs, and the phased task list live in
[openspec/changes/smolvla-edge-deployment/](openspec/changes/smolvla-edge-deployment/).

## License

MIT — see [LICENSE](LICENSE).

## Acknowledgements

Built on [LeRobot](https://github.com/huggingface/lerobot) and the SmolVLA base model by the
Hugging Face robotics team.
