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
- [x] 2.3 Fine-tune SmolVLA — **done**: full 20k-step batch-64 run on Colab A100 (transfer cube, `notebooks/colab_train_smolvla_aloha.ipynb`); checkpoint staged at `outputs/train/smolvla_transfer_cube/checkpoints/020000`. Earlier validation: local smoke (1k steps) + T4 dry run
- [x] 2.4 Obs→policy mapping verified with a real fine-tuned SmolVLA checkpoint: fresh checkpoints carry a saved processor pipeline (rename top→camera1, task tokenization, normalization) which `eval.make_sim_stepper` now uses; old hub checkpoints fall back to the manual mapping (ACT regression 2/2 success)
- [x] 2.5 **Deliverable — head-to-head done (transfer cube, 20 eps, identical seeds, matched mujoco):** fine-tuned **SmolVLA 14/20 = 70%** (mean max reward 3.20) vs **ACT baseline 13/20 = 65%** (3.15). Latency context from summary.csv: ACT 0.68 ms/1474 Hz/266 MB vs SmolVLA-arch 27.7 ms/36 Hz/927 MB — SmolVLA wins on success rate at ~40x the compute, the exact trade the edge phase (chunking, low-Hz VLM) is designed to exploit. Reproduce: `docker compose run --rm shell python -m smolvla_edge.eval --mode sim --policy-path outputs/train/smolvla_transfer_cube/checkpoints/020000 --env-id gym_aloha/AlohaTransferCube-v0 --episodes 20 --task "Pick up the cube with the right arm and transfer it to the left arm."`

## 3. Asynchronous inference in sim (Phase 2 — CURRENT FOCUS)

Reproduce the SmolVLA paper's async inference stack (§3.3, Algorithm 1, Figures 2/3/5)
entirely in simulation: a `RobotClient` consumes an action queue while a `PolicyServer`
computes the next chunk in the background; queue threshold `g`, joint-space similarity
filter `ε`, and chunk aggregation `f` on overlapping timesteps. Sync mode (drain the whole
chunk, then block on inference) is the `g = 0` sequential limit of the same loop.

- [x] 3.1 Async runtime core in `src/smolvla_edge/async_infer.py` — Algorithm 1 loop (pop per tick; non-blocking chunk prediction on a single worker when `|A_t|/n < g`; queue carried over while inference runs). Selftest (`python -m smolvla_edge.async_infer`, fake 40 ms predictor): g=0.5 sawtooth with 0 idle ticks; g=0 costs exactly ceil(ℓ/Δt)=3 idle ticks/cycle. **Correctness anchor:** async `--g 0 --idle freeze` reproduces the legacy sync eval exactly (ACT transfer cube, 2/2 same seeds, same success ticks)
- [x] 3.2 Chunk aggregation `f(A_t, Ã_t+1)` on overlap — `new_wins` default + `blend` (linear 0.5→1.0 ramp toward the fresher chunk), pluggable. **Merge alignment is pop-based, not tick-based**: skip only timesteps covered by actions actually executed since the trigger observation; idle (hold) ticks don't advance the trajectory — the tick-based version silently discarded ~46/50 of every chunk under 0.9 s latency
- [x] 3.3 Joint-space similarity filter (`ε` L2 in joint space, 0=off); empty queue always forces processing of the latest observation
- [x] 3.4 Virtual-time pacing: 1 env.step = 1 tick of Δt=1/fps (50 Hz ALOHA); a chunk that took L s lands ceil(L/Δt) ticks after trigger; the sim clock blocks if ticks outrun the wall clock, so idle ticks are honest regardless of env speed. Idle ticks execute a hold action (`--idle freeze` = legacy frozen-env for comparability)
- [x] 3.5 Wired into the harness: `python -m smolvla_edge.eval --mode sim --inference async --g --epsilon --fps --aggregate --idle --out --save-traces`; per-episode stats (obs sent/filtered, merges, idle ticks, latency, success tick) + per-tick queue traces in the JSON
- [x] 3.5b **Measured reality on the RTX 2000 Ada tier (5-ep matrix, 700-tick fixed-time budget, honest idle):** g=0 **4/5**, 466 ticks-to-success; g=0.7 **3/5**, 420; g=1.0 **3/5**, 434; g=0.7 + `--flow-steps 5` **2/5**, 350 (fastest, but 5 denoising steps degrade the actions). Async is ~10% faster to success, muted because in-loop ℓ_S ≈ 0.9–1.0 s vs **333 ms exclusive warm** — in-process inference contends (GIL/CPU) with mujoco stepping+rendering. Decomposition: flow expert ~29 ms × 10 steps dominates; VLM prefill only ~43 ms; **fp16 autocast is slower** (390 vs 333 ms, launch-bound). ACT (no temporal ensembling) degrades under mid-chunk replanning (2/2 sync → 0/2 at g=0.7) — consistent with the paper pairing async with flow matching. Consequence: task 3.8 (remote PolicyServer, separate process/host) is the real de-contention fix, exactly the paper's deployment shape
- [x] 3.6 Reproduce Figure 3 — `benchmarks/plot_async_queue.py` → `benchmarks/results/async_queue_trace.png` (g ∈ {0, 0.7, 1.0}, ep 0): g=0 full drains with ~25-tick dead zones; g>0 partial refills (fresh-minus-overlap) with shorter starvation dwells. Similarity-filter (ε>0) variant still to plot — at ℓ_S ≈ n·Δt every trigger fires with the queue already low, so the filter never engages; revisit after 3.8 cuts latency
- [x] 3.7 **Deliverable — sync vs async head-to-head: DONE.** Final protocol: fs3 remote server (in-loop ℓ_S ≈ 0.45 s), 20 eps, seeds 0-19, 700-tick fixed-time budget, `--idle hold`. **Idealized (frozen env): 16/20 = 80% @ 272 ticks. Sync (g=0): 13/20 = 65% @ 407 ticks (idle 178). Async (g=0.5, ramp-in 5): 14/20 = 70% @ 329 ticks (idle 78) — success parity (+5pp) and 19% faster time-to-success**, the paper's Figure-5 result reproduced in sim. Measured Fig-3 plot regenerated from these runs (`benchmarks/results/async_queue_trace.png`): g=0 zero-dwells vs g=0.5 zigzag that stays off the floor. Both stack additions beyond the paper were required to get here: honest tick accounting + `--ramp-in` seam smoothing.
  **Attempt 1 (fp32, 10 flow steps, remote, 20 eps) — negative result, kept for the writeup:** ℓ_S drifted to ~0.9 s (> the 700 ms g=0.7 buffer; host desktop shares this WSL2 GPU) → async 9/20 = 45% @ 442 ticks vs sync 13/20 = 65% @ 514 — above the paper's operating condition ℓ_S < g·n·Δt async *costs* success instead of matching it. Confirms latency-first ordering. **Fix chosen:** `--flow-steps 5` → 183 ms exclusive (sweep: 333/183/136 ms at 10/5/3 steps; `half()` needs input casts, dropped — flow steps suffice).
  **fs5 matrix (20 eps each, remote, in-loop ℓ ≈ 500–570 ms):** quality floor (g=0 freeze) **16/20 = 80%** @ 270 ticks — 5 flow steps do NOT hurt the policy; sync (g=0 hold) **15/20 = 75%** @ 422 (idle 188 ≈ theory); async g=0.7 **9/20 = 45%** @ 316; async g=0.7 `blend` **8/20 = 40%** @ 273; async g=0.5 **11/20 = 55%** @ 312. Success degrades monotonically with replan count (5.5 → 14 → 18 → 20/ep), NOT with total idle — deep chunk splices (ℓ ≈ 26–28 ticks ⇒ merge executes `chunk[27:]`, a trajectory whose prefix was never followed; flow-matching multimodality makes consecutive chunks disagree) break motion coherence, and blending ≈ replacement at this depth (lerobot's own `weighted_average = 0.3·old + 0.7·new` is the same family).
  **The empty-window theorem for this box:** no-starvation needs g ≥ (ℓ/Δt)/n; sparse-replanning needs g ≤ 1 − (ℓ/Δt)/n; both ⇒ ℓ/Δt < n/2 = 500 ms in-loop. At ℓ ≈ 560 ms the window is EMPTY — no g works. `--flow-steps 3` (fs3): quality floor **16/20 = 80%** (3 steps lose nothing), in-loop ℓ ≈ 0.37–0.49 s = window open at g=0.5.
  **fs3 elimination chain (20 eps each):** g=0.5 plain **8/20 = 40%** (in-window, idle 42 — starvation ruled out); `--noise-seed 0` **7/20 = 35%** (flow multimodality ruled out); **`--ramp-in 5` → 14/20 = 70% @ 329 ticks — RECOVERED.** Mechanism proven: splice discontinuities in absolute joint targets (chunks disagree by degrees at the seam; ~14-19 steps/ep = torque spikes = dropped cube). 5-tick linear easing from the last executed action fixes it. Note: lerobot's `weighted_average` smooths chunk-vs-chunk overlap but NOT the executed-trajectory-vs-new-chunk seam — invisible at paper-regime splice depths, fatal at ours. Remaining for the final table: fs3 sync g=0 hold baseline (running) → like-for-like speedup number
- [x] 3.8 Remote variant — **done, localhost-verified**: `PredictChunk` RPC in `policy.proto`; server routes through `make_chunk_predictor` (checkpoint-correct normalization); `smolvla_edge.remote` predictor drops into `AsyncRunner` unchanged; `eval --server host:port` = thin client (no local policy load). **Measured (g=0.7, 2 eps): 2/2 success, 293 ticks-to-success = 37% faster than the g=0 sequential limit (466) — the paper's ~30% Figure-5 result reproduced in sim.** Split: ℓ_S 667 ms server-compute, 2.3 ms network. Process separation halved in-loop latency (1.0 s → 0.67 s < the 700 ms g=0.7 buffer); residual vs. 333 ms exclusive is same-box CPU sharing — a true second host closes it (§6.7). Gotcha: launch the server with `python -u` (block-buffered stdout hides the "listening" line from readiness greps)
- [ ] 3.9 Collate async rows into `benchmarks/results/summary.csv` (still open) — **README chapter done**: "Asynchronous inference — the paper's Algorithm 1, reproduced in sim" with the headline 3-row table, measured + synthetic Fig-3 plots, mermaid architecture diagram (client eval loop / AsyncRunner / worker thread / PolicyServer), lerobot-comparison table, the two beyond-the-paper findings (operating-envelope inequality, ramp-in seam smoothing + elimination chain), and reproduce commands

## 4. Benchmarks + writeup (Phase 3)

- [ ] 4.1 Produce the results rows across available tiers — **local-GPU tier done** (RTX 2000 Ada, pretrained ACT: fp32 0.68 ms/1474 Hz/266 MB; fp16-autocast 0.69 ms/1444 Hz — parity: tiny model is overhead-bound). NX tiers pending hardware; SmolVLA rows pending the fine-tune
- [ ] 4.2 Capture metrics: end-to-end latency, throughput, peak memory ✔ (in bench.py rows); action-chunk frequency still to add
- [x] 4.3 Collate raw JSON → `results/summary.csv` + markdown table via `benchmarks/collate.py` (verified in-container, 2 rows)
- [x] 4.4 Demo GIF via `scripts/make_demo_gif.py` → `benchmarks/results/demo.gif`. Now a **closed-loop policy rollout** (pretrained ACT succeeding at cube transfer in gym-aloha, reward 4/4, ~real-time), not a dataset replay; `--mode replay` kept as the no-policy fallback
- [x] 4.5 README narrative written through Phase 1: SmolVLA hero demo GIF + ACT baseline GIF, headline head-to-head (70 % vs 65 %), measured table, failure modes, findings. Edge-hardware chapters get appended when Phase 2 runs on a Jetson

## 5. Docs migration

- [x] 5.1 Remove `docs/` (roadmap, setup-jetson, future-work-rover) now captured in this change
- [x] 5.2 Update `README.md` links from `docs/…` to the OpenSpec change/specs
- [x] 5.3 Relocate demo media output from `docs/assets/` to `benchmarks/results/`

## 6. Edge hardware verification — Xavier NX (a Jetson NX is now on hand)

Everything above runs with no hardware. A **Jetson Xavier NX (8 GB) Developer Kit** is now
connected (`xaiver-eth`=10.42.0.2 / `xaiver-wlan`, user `nvidia`, repo at
`~/workspace/smolvla-edge-nx`). First on-device milestone chose the **pure-Python ONNX Runtime
GPU** path (serve the Stage-2a monolithic ONNX graph) over the torch-checkpoint + §3 async
stack: the graph bakes in the instruction + normalization, so on-device inference needs only
`numpy` + `onnxruntime-gpu` — no torch/lerobot/tokenizer, which fits the 8 GB budget cleanly.

- [x] 6.1 **NX environment stood up.** Xavier NX 8 GB Dev Kit, **JetPack 5.1 / L4T R35.4.1**,
      CUDA 11.4, TensorRT 8.5.2, cuDNN 8, Python 3.8, 6.7 GiB unified RAM + 3.3 GiB swap, power
      mode **MODE_20W_6CORE**. Container `smolvla-edge:jetson` (`docker/jetson_infer.Dockerfile`,
      `docker-compose.jetson.yml`): lean `nvcr.io/nvidia/l4t-base:r35.2.1` base (l4t-jetpack:r35.4.1
      is 5.3 GB — too big for ~14 GB free disk) + vendored `onnxruntime-gpu 1.15.1` cp38 wheel
      (`deploy/onnx/wheels/`; the only CUDA-11.4/cp38 build for JP5, jetson-ai-lab index has flaky
      DNS). NOT reused the rover image (Ubuntu 24.04 / py3.12 → no cp38 ORT wheel). Build with
      `DOCKER_BUILDKIT=0 docker build …` (Jetson's compose lacks buildx ≥0.17).
- [~] 6.2 **Checkpoint running on the NX — fp32 via ORT-GPU (partial; FP16 pending).** The
      exported graph runs on the NX GPU under the ORT CUDA EP: **~610 ms mean/chunk** (p50 610,
      p90 620, 20-iter), vs ~300 ms PyTorch on the RTX 2000 Ada dev box — ~2× as expected at the
      edge, still under the 1 s a 50-action chunk buys at 50 Hz. Fits 6.7 GiB unified RAM (fp32
      graph ~1.8 GB). Bench: `docker compose -f docker-compose.jetson.yml run --rm bench`
      (`deploy/onnx/bench_ort.py`). Two shims to load under ORT 1.15.1
      (`deploy/onnx/patch_for_ort115.py`, run in the dev sim container — 1.8 GB load/save OOMs the
      NX): IR 10→9 downgrade + 24 int64→int32 Casts before ArgMin/ArgMax (ORT 1.15 lacks int64
      ArgMin kernels). CUDA/cuDNN/TRT bind-mounted from the host (`l4t-base` ships no toolkit) via
      the rover's `/usr/local/cuda` + `LD_LIBRARY_PATH` pattern.
- [x] 6.2b **On-device SOLVED — 233 ms/chunk via native torch + manual CUDA Graph capture.**
      Built the "impossible" JP5 native stack from source (torch 2.2.2 cp310/CUDA 11.8/sm_72 via
      `cuda-compat`, torchvision, lerobot 0.4.4 — image `wtlove876/smolvla-jetson:jp5-cu118`,
      parity 2.8e-6 vs reference). Profiled eager fp16 (~630 ms): **launch-bound**, GPU ~80% idle
      (GR3D 15–26%, EMC 12%, preproc 7 ms) — so per-op-compute levers do nothing (256M: 591 ms;
      INT8 same category; full-TRT: 8 GB builder OOM; `torch.compile(cudagraphs)`: 622 ms, dynamo
      breaks on the flow loop). The cure: **manual `torch.cuda.CUDAGraph` capture of the whole
      forward** (static shapes + injectable noise; 3 constant-H2D blockers folded — torch.tensor
      constants, HF vision patch mask, NaViT pos-ids): **eager 608 → 233 ms e2e, replay floor
      211 ms, bitwise-identical actions (0.00 diff, also on new obs through the buffer copies)**,
      +0.11 GB. Harness + writeup: `deploy/jetson-native-torch/` (`bench_cudagraph_manual.py`,
      README "The fix"). Depth lever if <150 ms ever needed: 16/8/4 layers = 613/441/287 ms (retrain).
- [ ] 6.3 On-device: run the §3 async stack (action queue + decoupled execution) on the NX
- [ ] 6.4 On-device: run the VLM stage at low Hz relative to the control loop
- [x] 6.5 On-device INT8/TensorRT — **closed by measurement, not needed**: the workload is
      launch-bound (see 6.2b), so INT8/quantization attacks per-op compute and cannot help;
      full-model TRT OOMs the 8 GB builder and the TRT parser rejects the flow-matching graph
      (ros2-cpp §5.6 wall confirmed). CUDA Graph capture supersedes both.
- [ ] 6.6 Log conversions honestly in `deploy/ondevice/conversion_notes.md` (what converted / what didn't / per-stage budget)
- [ ] 6.7 Client/server split-latency tier measured from the NX as the client (the remote variant of task 3.8, re-run on real edge hardware)
