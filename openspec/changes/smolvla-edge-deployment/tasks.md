## 1. Scaffold (Phase 0)

- [x] 1.1 Public repo + MIT license
- [x] 1.2 README skeleton with the narrative
- [x] 1.3 Pin LeRobot v0.5.0 in `requirements.txt`
- [ ] 1.4 Create env on the Titan X dev box: `pip install -r requirements.txt` (+ system ffmpeg)
- [ ] 1.5 Confirm the showcase task (default: SO-101 pick-and-place)

## 2. Correctness — fine-tune (Phase 1)

- [ ] 2.1 Smoke-test the stack: `python -m smolvla_edge.infer` on `lerobot/smolvla_base`
- [ ] 2.2 Fine-tune on a rented A100/H100: `bash scripts/train.sh` (~20k steps ≈ 4 h on one A100)
- [ ] 2.3 Keep the Titan X for dev/inference (batch 16 ≈ 6 GB locally, much slower)
- [ ] 2.4 Evaluate the checkpoint on held-out episodes → success-rate number (deliverable)

## 3. Edge deployment — Xavier NX (Phase 2)

- [ ] 3.1 Stand up the NX environment per the design's JetPack setup (record JetPack version + power mode)
- [ ] 3.2 Get the checkpoint running on the NX (FP16, within the 8 GB budget)
- [ ] 3.3 On-device: implement action chunking + decoupled execution
- [ ] 3.4 On-device: run the VLM stage at low Hz relative to the control loop
- [ ] 3.5 On-device: INT8 only via a real quantized/TensorRT engine on subgraphs that convert
- [ ] 3.6 Log conversions honestly in `deploy/ondevice/conversion_notes.md` (what converted / what didn't / per-stage budget)
- [ ] 3.7 Client/server: policy on the Titan X gRPC server, NX as the control client (split latency)

## 4. Benchmarks + writeup (Phase 3)

- [ ] 4.1 Produce the results rows across tiers (Titan X local; NX FP16/INT8 ±chunking; NX-client/WS-server)
- [ ] 4.2 Capture metrics: end-to-end latency, action-chunk frequency, throughput, peak memory
- [ ] 4.3 Collate raw JSON → `results/summary.csv` + markdown table via `benchmarks/collate.py`
- [ ] 4.4 Build the demo GIF/video of replayed episodes via `scripts/make_demo_gif.py`
- [ ] 4.5 Write the README narrative: "Deploying a flow-matching VLA on 8 GB edge hardware"

## 5. Docs migration

- [x] 5.1 Remove `docs/` (roadmap, setup-jetson, future-work-rover) now captured in this change
- [x] 5.2 Update `README.md` links from `docs/…` to the OpenSpec change/specs
- [x] 5.3 Relocate demo media output from `docs/assets/` to `benchmarks/results/`
