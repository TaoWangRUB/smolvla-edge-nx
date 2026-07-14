## 1. Scaffold (Phase 0)

- [x] 1.1 Public repo + MIT license
- [x] 1.2 README skeleton with the narrative
- [x] 1.3 Pin LeRobot v0.5.0 in `requirements.txt`
- [x] 1.4 Create env on the dev box: LeRobot 0.5.0 + mujoco 3.10 + dm_control 1.0.43 + gym-aloha installed & verified (py3.12 needs the workaround in `scripts/setup_sim.sh`; headless via `MUJOCO_GL=egl`)
- [x] 1.6 Dockerized environment (preferred path): `docker/Dockerfile` + `docker-compose.yml` — CUDA torch base (py3.11 → matched mujoco 2.3.7), repo at `/workspace`, nvidia runtime + EGL, HF-cache volume, services `verify`/`eval`/`infer`/`train`/`bench`/`shell`
- [x] 1.5 Confirm the showcase task — **ALOHA sim insertion** (`gym_aloha/AlohaInsertion-v0`, dataset `lerobot/aloha_sim_insertion_human`). Chosen so the correctness loop runs with **no robot**; real SO-101 is out of scope for now (no hardware). See `configs/train.aloha_sim.yaml`.

## 2. Correctness — verify in sim, then fine-tune (Phase 1)

- [x] 2.1 Smoke-test the stack: `docker compose run --rm infer` — smolvla_base ran 100 steps on the SO-101 dataset (6-D actions; select_action p50 2.7 ms / mean 12.6 ms incl. VLM prefill at chunk boundaries). Needed: dataset→policy image-key remap + external language tokenization (lerobot 0.4.x has no processor pipeline) — now built into `infer.py`/`common.py`
- [x] 2.2 **Verify-first (no fine-tune):** pretrained checkpoints run through the harness end-to-end. Transfer cube: **3/5 = 60% success** (`lerobot/act_aloha_sim_transfer_cube_human`, the positive baseline). Insertion: 0/3, grasps only (reward 2/4) — mujoco 2.x→3.10 sim-version gap (see design "Simulation setup"). Reproduce: `docker compose run --rm verify`
- [x] 2.2b Re-ran verify inside the Docker image (matched mujoco 2.3.7): transfer cube **4/5 = 80%** (vs 60% on host mujoco 3.10 — the sim-version gap measured directly). Insertion: still 0/5 in-container (mean max reward 2.2, one episode reached 3/4) — matched mujoco helps but doesn't recover the published ~50% at 5 episodes; likely needs more episodes and/or ACT temporal-ensembling settings.
- [ ] 2.3 Fine-tune SmolVLA — **full 20k-step run IN FLIGHT on Colab A100** (2026-07-14, `notebooks/colab_train_smolvla_aloha.ipynb`): dataset switched to **transfer cube** for the head-to-head; batch 64, 2.2 steps/s, ~2.5 h ETA; loss 0.75→0.050 by step 1.2k (warmup 666 steps), wandb run `5uykcoqr`. Earlier validation: local smoke (1k steps, RTX 2000 Ada, checkpoint evals closed-loop) + T4 dry run (200 steps, loss 0.183)
- [x] 2.4 Obs→policy mapping verified with a real fine-tuned SmolVLA checkpoint: fresh checkpoints carry a saved processor pipeline (rename top→camera1, task tokenization, normalization) which `eval.make_sim_stepper` now uses; old hub checkpoints fall back to the manual mapping (ACT regression 2/2 success)
- [ ] 2.5 **Head-to-head (transfer cube, 20-episode protocol):** evaluate the fine-tuned SmolVLA and compare against the ACT baseline. **Baseline measured: ACT 13/20 = 65%** (mean max reward 3.15; failures mostly stall post-grasp), in-container matched mujoco. Latency rows already in summary.csv: ACT 0.68 ms/1474 Hz/266 MB vs SmolVLA-arch 27.7 ms/36 Hz/927 MB. Your model: `docker compose run --rm shell python -m smolvla_edge.eval --mode sim --policy-path outputs/train/smolvla_transfer_cube/checkpoints/last --env-id gym_aloha/AlohaTransferCube-v0 --episodes 20 --task "Pick up the cube with the right arm and transfer it to the left arm."`

## 3. Edge deployment — Xavier NX (Phase 2, OPTIONAL — only if a Jetson NX is on hand)

- [ ] 3.1 Stand up the NX environment per the design's JetPack setup (record JetPack version + power mode)
- [ ] 3.2 Get the checkpoint running on the NX (FP16, within the 8 GB budget)
- [ ] 3.3 On-device: implement action chunking + decoupled execution
- [ ] 3.4 On-device: run the VLM stage at low Hz relative to the control loop
- [ ] 3.5 On-device: INT8 only via a real quantized/TensorRT engine on subgraphs that convert
- [ ] 3.6 Log conversions honestly in `deploy/ondevice/conversion_notes.md` (what converted / what didn't / per-stage budget)
- [ ] 3.7 Client/server: policy on a workstation gRPC server, NX (or any second host) as the control client (split latency)

## 4. Benchmarks + writeup (Phase 3)

- [ ] 4.1 Produce the results rows across available tiers — **local-GPU tier done** (RTX 2000 Ada, pretrained ACT: fp32 0.68 ms/1474 Hz/266 MB; fp16-autocast 0.69 ms/1444 Hz — parity: tiny model is overhead-bound). NX tiers pending hardware; SmolVLA rows pending the fine-tune
- [ ] 4.2 Capture metrics: end-to-end latency, throughput, peak memory ✔ (in bench.py rows); action-chunk frequency still to add
- [x] 4.3 Collate raw JSON → `results/summary.csv` + markdown table via `benchmarks/collate.py` (verified in-container, 2 rows)
- [x] 4.4 Demo GIF via `scripts/make_demo_gif.py` → `benchmarks/results/demo.gif`. Now a **closed-loop policy rollout** (pretrained ACT succeeding at cube transfer in gym-aloha, reward 4/4, ~real-time), not a dataset replay; `--mode replay` kept as the no-policy fallback
- [ ] 4.5 Write the README narrative: "Fine-tuning SmolVLA in sim, and deploying a flow-matching VLA on 8 GB edge hardware"

## 5. Docs migration

- [x] 5.1 Remove `docs/` (roadmap, setup-jetson, future-work-rover) now captured in this change
- [x] 5.2 Update `README.md` links from `docs/…` to the OpenSpec change/specs
- [x] 5.3 Relocate demo media output from `docs/assets/` to `benchmarks/results/`
