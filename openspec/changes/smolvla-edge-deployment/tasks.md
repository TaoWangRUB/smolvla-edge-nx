## 1. Scaffold (Phase 0)

- [x] 1.1 Public repo + MIT license
- [x] 1.2 README skeleton with the narrative
- [x] 1.3 Pin LeRobot v0.5.0 in `requirements.txt`
- [x] 1.4 Create env on the dev box: LeRobot 0.5.0 + mujoco 3.10 + dm_control 1.0.43 + gym-aloha installed & verified (py3.12 needs the workaround in `scripts/setup_sim.sh`; headless via `MUJOCO_GL=egl`)
- [x] 1.6 Dockerized environment (preferred path): `docker/Dockerfile` + `docker-compose.yml` — CUDA torch base (py3.11 → matched mujoco 2.3.7), repo at `/workspace`, nvidia runtime + EGL, HF-cache volume, services `verify`/`eval`/`infer`/`train`/`bench`/`shell`
- [x] 1.5 Confirm the showcase task — **ALOHA sim insertion** (`gym_aloha/AlohaInsertion-v0`, dataset `lerobot/aloha_sim_insertion_human`). Chosen so the correctness loop runs with **no robot**; real SO-101 is out of scope for now (no hardware). See `configs/train.aloha_sim.yaml`.

## 2. Correctness — verify in sim, then fine-tune (Phase 1)

- [ ] 2.1 Smoke-test the stack: `python -m smolvla_edge.infer --policy-path lerobot/smolvla_base --dataset-repo-id lerobot/svla_so101_pickplace` (base is SO-101 embodiment — pair accordingly)
- [x] 2.2 **Verify-first (no fine-tune):** pretrained checkpoints run through the harness end-to-end. Transfer cube: **3/5 = 60% success** (`lerobot/act_aloha_sim_transfer_cube_human`, the positive baseline). Insertion: 0/3, grasps only (reward 2/4) — mujoco 2.x→3.10 sim-version gap (see design "Simulation setup"). Reproduce: `docker compose run --rm verify`
- [x] 2.2b Re-ran verify inside the Docker image (matched mujoco 2.3.7): transfer cube **4/5 = 80%** (vs 60% on host mujoco 3.10 — the sim-version gap measured directly). Insertion number pending more episodes.
- [ ] 2.3 Fine-tune SmolVLA on the ALOHA dataset: `bash scripts/train.sh` (uses `configs/train.aloha_sim.yaml`; A100/H100, or Titan X slower)
- [ ] 2.4 Verify the obs→policy key mapping in `smolvla_edge.eval._aloha_obs_to_batch` against your checkpoint's `policy.config.input_features` (largely settled once 2.2 passes)
- [ ] 2.5 Evaluate the fine-tuned SmolVLA closed-loop → success-rate number (deliverable): `python -m smolvla_edge.eval --mode sim --policy-path outputs/train/smolvla_aloha/checkpoints/last --env-id gym_aloha/AlohaInsertion-v0 --episodes 20`

## 3. Edge deployment — Xavier NX (Phase 2, OPTIONAL — only if a Jetson NX is on hand)

- [ ] 3.1 Stand up the NX environment per the design's JetPack setup (record JetPack version + power mode)
- [ ] 3.2 Get the checkpoint running on the NX (FP16, within the 8 GB budget)
- [ ] 3.3 On-device: implement action chunking + decoupled execution
- [ ] 3.4 On-device: run the VLM stage at low Hz relative to the control loop
- [ ] 3.5 On-device: INT8 only via a real quantized/TensorRT engine on subgraphs that convert
- [ ] 3.6 Log conversions honestly in `deploy/ondevice/conversion_notes.md` (what converted / what didn't / per-stage budget)
- [ ] 3.7 Client/server: policy on a workstation gRPC server, NX (or any second host) as the control client (split latency)

## 4. Benchmarks + writeup (Phase 3)

- [ ] 4.1 Produce the results rows across available tiers (local GPU always; NX FP16/INT8 ±chunking and client/server only if a Jetson is on hand)
- [ ] 4.2 Capture metrics: end-to-end latency, action-chunk frequency, throughput, peak memory
- [ ] 4.3 Collate raw JSON → `results/summary.csv` + markdown table via `benchmarks/collate.py`
- [ ] 4.4 Build the demo GIF from the ALOHA sim dataset frames via `scripts/make_demo_gif.py`
- [ ] 4.5 Write the README narrative: "Fine-tuning SmolVLA in sim, and deploying a flow-matching VLA on 8 GB edge hardware"

## 5. Docs migration

- [x] 5.1 Remove `docs/` (roadmap, setup-jetson, future-work-rover) now captured in this change
- [x] 5.2 Update `README.md` links from `docs/…` to the OpenSpec change/specs
- [x] 5.3 Relocate demo media output from `docs/assets/` to `benchmarks/results/`
