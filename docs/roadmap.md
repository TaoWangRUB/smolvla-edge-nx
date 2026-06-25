# Roadmap & checklist

Demo first — it's the gating artifact for the application. Strategic frame: fine-tuning is
table-stakes; the **Xavier NX edge deployment + latency engineering** is the differentiator.
Get a correct checkpoint fast, then spend the real time on edge.

## Phase 0 — Scaffold (½ day)
- [x] Public repo + MIT license
- [x] README skeleton with the narrative
- [x] Pin LeRobot v0.5.0 (requirements.txt)
- [ ] Create env on Titan X: `pip install -r requirements.txt` (+ system ffmpeg)
- [ ] Decide the one showcase task (default: SO-101 pick-and-place)

## Phase 1 — Correctness: fine-tune (a few days)
- [ ] Smoke-test the stack: `python -m smolvla_edge.infer` on `lerobot/smolvla_base`
- [ ] Fine-tune on rented A100/H100: `bash scripts/train.sh` (20k steps ≈ 4 h on one A100)
- [ ] Keep Titan X for dev/inference (batch 16 ≈ 6 GB if training locally — much slower)
- [ ] **Deliverable:** checkpoint + success-rate number from held-out eval

## Phase 2 — Differentiator: Xavier NX deployment (bulk of the work)
- [ ] Get checkpoint running on the NX (see docs/setup-jetson.md)
- [ ] On-device: action chunking, low-Hz VLM, FP16, INT8 where the graph converts
- [ ] Log conversions honestly in deploy/ondevice/conversion_notes.md
- [ ] Client/server: policy on Titan X, NX as gRPC control client

## Phase 3 — Benchmarks + writeup (centerpiece)
- [ ] Results table: Titan X local; NX on-device (FP16 vs INT8, ±chunking); NX-client/WS-server
- [ ] Metrics: end-to-end latency, action-chunk freq, throughput, memory
- [ ] Short GIF/video of policy executing replayed episodes (no robot needed)
- [ ] README narrative: "Deploying a flow-matching VLA on 8 GB edge hardware"

## Out of scope (future work)
- [ ] Mobile rover embodiment — see docs/future-work-rover.md
