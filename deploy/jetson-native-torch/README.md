# Native PyTorch + LeRobot + SmolVLA on Jetson Xavier NX (JetPack 5)

Running the **native PyTorch/LeRobot SmolVLA stack on the Xavier NX GPU** — the path everyone
says is impossible on JetPack 5 — built from source and verified. Complements the pure-Python
ONNX-Runtime deployment in [../README.md](../README.md); this is the "torch actually runs on the
device" alternative.

## Deployment progress (timeline)

Two on-device paths were brought up on the Xavier NX, in order:

1. **Pure-Python ONNX-Runtime GPU** (`../README.md`, `../onnx/`) — export → `onnxruntime-gpu 1.15.1`
   (cp38/CUDA-11.4) with the IR-9 / int64-ArgMin patch: **~610 ms/chunk, fp32**. No torch needed.
2. **Native from-source PyTorch + LeRobot** (this dir) — the milestones:
   - Verified **CUDA 11.8 runs on the R35 (11.4) driver** via `cuda-compat` (sees `Xavier sm_72`).
   - Built **PyTorch 2.2.2** from source (cp310, CUDA 11.8, sm_72) — ~5 h compile.
   - Built torchvision 0.17.2; installed LeRobot 0.4.4 on the custom torch.
   - **Fine-tuned SmolVLA checkpoint runs on the GPU**: fp16 **627 ms/chunk**.
   - **Parity-verified** vs the reference torch (2.8e-6, cosine 1.0) + deployment report (fits 8 GB).
   - Published self-contained image: **`wtlove876/smolvla-jetson:jp5-cu118`** (Docker Hub, public).
   - **Manual CUDA Graph capture of the full forward: 608 → 233 ms/chunk** (bitwise-identical
     actions) — laptop-class latency on the Xavier, no retrain, no quantization.

Build-infra note: the from-source build had to run on the **internal NVMe**, not the USB SSD — a
Ugreen/RTL9210 USB enclosure **dropped off the bus three times** under sustained build I/O
(`error -71`, fs remounts RO). To free NVMe space the rover Docker image was backed up to the USB
(`docker save`) and removed (restore: `docker load -i`; it's also published as
`wtlove876/ackermann-rover:jazzy-aarch64`).

## Why it's "impossible" (and the bridge)

Xavier is stuck on **JetPack 5** (JetPack 6 is Orin-only) ⇒ Python 3.8 + CUDA 11.4, and:
- LeRobot requires **Python ≥ 3.10**; SmolVLA requires **torch ≥ 2.2**.
- NVIDIA / dusty-nv JetPack-5 torch wheels are **cp38** (Python 3.8); the Jetson AI Lab pip index
  **dropped JetPack 5** (it now only serves JP6). So no prebuilt torch+lerobot exists for this board.

The bridge: build everything from source for **Python 3.10 + CUDA 11.8 + sm_72**, running CUDA 11.8
on the fixed R35 (CUDA-11.4) driver via **`cuda-compat`**.

## Result — verified

**Numerical parity** (`parity_check.py`, same checkpoint + pinned flow-matching noise):

| | device | torch | action-chunk sha16 |
|---|---|---|---|
| reference | dev GPU | 2.6.0 (official) | 84916da9… |
| **Xavier NX** | Xavier sm_72 | **2.2.2 (from source)** | d813ad6c… |

`max-abs-diff = 2.8e-6`, `cosine = 1.0000` → the hand-built Jetson torch produces **numerically
identical** actions to the reference (consistent with the ONNX parity gate's 4.3e-6). Since the
actions match the reference exactly, closed-loop task success on the NX equals the dev-box result
(70%) — no separate sim run needed.

**Deployment report** (`verify_report.py`, fp16, 12 diverse observations, Xavier NX):
`peak GPU mem alloc **1.01 GB** / reserved 1.07 GB` (of the 8 GB unified budget — huge headroom),
peak CPU RSS 3.3 GB, actions valid (no NaN/Inf) and input-sensitive, latency ~0.63–0.74 s. PASS.

**Latency** (SmolVLA transfer-cube checkpoint, 3 flow steps, Xavier NX):

| precision | native torch | note |
|---|---|---|
| bf16 | 1795 ms | ❌ **Volta (sm_72) has no hardware bf16** — emulated, ~3× slower |
| fp32 | 817 ms | no tensor cores |
| **fp16** | **627 ms** | ✅ Volta fp16 tensor cores |
| **fp16 + CUDA Graph replay** | **233 ms** | ✅✅ full-forward manual capture — see below |
| ONNX-ORT (fp32) | ~610 ms | reference |

Takeaway: on Volta, native torch eager **matches** ONNX but doesn't beat it (no flash-attention
on sm_72; this lean build also drops mem-efficient attention). **Always `.half()` on Volta.**
With manual CUDA Graph capture (below) native torch then beats everything else on the board by
~2.5× — landing at the laptop-server-class ~230 ms originally targeted.

## The build recipe

1. **CUDA 11.8 for Jetson** — `cuda-tegra-repo-ubuntu2004-11-8-local_11.8.0-1_arm64.deb`,
   `dpkg -x`-extracted to a prefix (no host install). `compat/libcuda.so` runs 11.8 on the R35 driver.
2. **Base** — `ubuntu:22.04` container (native Python 3.10) + host L4T driver via `runtime: nvidia`.
3. **PyTorch 2.2.2** (`build_torch_nvme.sh`): `TORCH_CUDA_ARCH_LIST=7.2`, `CUDAARCHS=72`
   (CMake CUDA-probe), `LIBRARY_PATH=<cuda>/lib64` (nvcc device-link needs `libcudart_static.a`),
   `USE_CUDNN=0 USE_FLASH_ATTENTION=0 USE_MEM_EFF_ATTENTION=0 USE_FBGEMM=0 USE_XNNPACK=0`,
   `MAX_JOBS=3` + swap (the `libtorch_cuda.so` link OOMs 8 GB otherwise).
4. **torchvision 0.17.2** (`build_vision.sh`, `FORCE_CUDA=0` — LeRobot only needs the transforms).
5. **LeRobot 0.4.4** (`install_lerobot.sh`): `--no-deps` + concrete deps, but **drop
   rerun-sdk/wandb** (force numpy≥2; torch 2.2 needs numpy<2) and pin
   `opencv-python-headless==4.9.0.80`; **keep pyserial/pynput** (LeRobot eagerly imports
   `lerobot.motors`→`serial`).

## Performance analysis & debugging (why ~630 ms, and what it isn't)

We profiled the fp16 chunk latency (~630 ms) against an A2000 laptop's 92 ms at the *same*
fp16 / 3-step config, and traced the bottleneck. The precision trap comes first: **Volta
(sm_72) has no hardware bf16**, so bf16 is emulated (1795 ms) — always `.half()` → fp16.

**What it is NOT** (each ruled out by measurement, not assumption):

| suspected cause | test | result |
|---|---|---|
| GPU clock | `jetson_clocks` pins GPU 114→1109 MHz | **no change** ⇒ not clock-bound |
| preprocessing / H2D copies | CUDA-synced stage timing (`stage_breakdown.py`) | processor **7 ms**, model_fwd **632 ms (99%)**, post 0.8 ms |
| memory bandwidth | `tegrastats EMC_FREQ` during inference | **11–14%** ⇒ not memory-bound |
| harness copy churn | profiler + code read (`copy_experiment.py`) | the ~4,400 `to`/`copy_` ops are **inside** the transformer (autocast fp16↔fp32 + SmolVLM2 fp32 upcasts of RMSNorm/softmax/RoPE), not the loop. Pure fp16 (no autocast) **crashes** ("Float and Half") — the model holds fp32 tensors, so autocast is required |

**What it IS:** `tegrastats GR3D_FREQ` (GPU utilization) sits at **15–26%** during inference — the
GPU is **idle ~80%** of wall time. With EMC at 12% (not memory-bound) and preprocessing at 7 ms,
this is **launch-bound / framework-overhead-bound with low GPU occupancy** — a 450 M transformer
executed as many small ops on a weak-CPU (Carmel) + Volta board. (The definitive per-kernel
launch→gap timeline would need Nsight Systems / `nsys`, not installed on this box; the *hardware*
GPU-occupancy already establishes the low-occupancy conclusion.)

**Head-to-head vs the A2000 laptop** (same `profile_infer.py`, fp16, 3 steps):

| per inference | A2000 laptop | Xavier NX | ratio |
|---|---|---|---|
| wall | 92 ms | 738 ms | 8× |
| self-CPU (op dispatch) | 102 ms | 808 ms | ~8× |
| self-CUDA (GPU compute) | 86 ms | 883 ms | ~10× |

~8× slower, split roughly evenly across CPU dispatch and GPU compute — a fundamental hardware gap
(faster host CPU + Ampere flash-attention GEMMs), not a fixable misconfig. (A commonly cited
"230 ms" laptop figure is a heavier fp32/server config; apples-to-apples fp16/3-step is 92 ms.)

**Does ~630 ms block a mobile rover? No** — this is the key deployment point. Action chunking
decouples inference from the control rate: one inference yields **50 actions**, executed with no
network in between. At a rover's ~10 Hz base-control rate a 50-action chunk = **5 s of motion**,
so the net runs once per ~5 s and the async runner hides it entirely. **On-device solely is
viable** — offloading to a host (impractical on a moving robot) is unnecessary.

## The fix: manual CUDA Graph capture (608 → 233 ms, 2.6×)

Launch-bound means the classical cure is **record-once-replay**: capture the entire kernel
stream and replay it with a single launch, eliminating all per-op CPU dispatch.
`torch.compile(backend="cudagraphs")` gave only ~4% — **dynamo** breaks on the flow-matching
loop — but that is a *tracing* failure, not a capture failure. Manual `torch.cuda.CUDAGraph`
capture records the raw stream from the **unmodified eager code**, and this workload is secretly
static-shape: fixed camera resolution (512×512), fixed 48-token padded task, fixed 3 flow steps,
fixed 50×32 chunk, and injectable noise (`sample_actions(noise=…)`).

`bench_cudagraph_manual.py` captures the **full forward** (SigLIP encoder + 16-layer VLM prefix
+ all 3 denoise steps) as ONE graph. Three capture-blockers had to be neutralized, all constant
H2D copies of constants:
1. `torch.tensor(list/scalar)` constants inside `embed_prefix`/`embed_suffix`/the flow loop
   (attention-mask lists, the √dim scale, the 3 time scalars) — value-keyed constant folding of
   `torch.tensor` during warmup+capture (7 constants folded).
2. The HF vision tower's `patch_attention_mask` (created on CPU, `.to(cuda)` per call) —
   inject a precomputed all-ones bool GPU mask.
3. The NaViT position-ids loop in `SmolVLMVisionEmbeddings` (`.cpu()` syncs) — precompute
   position ids once and swap in a capture-safe embeddings forward (verified `max-abs-diff = 0`).

Per inference, only the observation/state/lang/noise buffers are `copy_`'d and the graph
replayed. **Results (Xavier NX, fp16, 3 steps, single run):**

| | latency |
|---|---|
| eager fp16 e2e | 608 ms |
| **graph replay only (model fwd, zero CPU dispatch)** | **211 ms** |
| **graphed e2e (preproc + copies + replay + post)** | **233 ms** |

Parity: replay vs eager on identical inputs **max-abs-diff = 0.00, cosine = 1.000000** (bitwise —
same kernels, same order), and replay vs eager on a *new* observation through the buffer copies is
also 0.00. Memory cost: +0.11 GB reserved (1.18 total). This also **confirms the diagnosis**: the
GPU's real work is 211 ms; the other ~400 ms of eager wall time was pure CPU dispatch overhead.

**Once dispatch is gone, the bottleneck moves back to GPU compute — and width finally pays**
(`bench_cudagraph_256m.py`, same run protocol):

| backbone | eager | graphed e2e | replay floor | peak mem |
|---|---|---|---|---|
| 450M (SmolVLM2-500M) | 581 ms | 237 ms | 215 ms | 1.02 GB |
| 256M (SmolVLM2-256M) | 577 ms | **196 ms** | 176 ms | 0.61 GB |

Eager, the 256M backbone bought nothing (launch-bound ⇒ latency tracks op count). Graphed, it's
~17% faster — proof the workload became compute-bound. A 256M *fine-tune* would ship at ~196 ms.

**Productionized** as `make_chunk_predictor(..., precision="fp16-graph")`
(`src/smolvla_edge/cuda_graph.py`): the first call runs eager and lazily captures; any capture
failure falls back to eager fp16; a shape change (camera resolution / token padding) falls back
per-call. The gRPC `PolicyServer` exposes it as `--precision fp16-graph`. Also verified on the
dev box (RTX A2000): eager 97 ms → **53 ms** — the laptop was partly launch-bound too.

Constraint: shapes are baked at capture (camera resolution, tokenized-task padding, flow steps,
chunk size). Task *content* may change (tokens flow through the input buffers); a resolution or
config change needs a one-off re-capture (~30 s at startup).

**Optimization levers, assessed (final):**
1. **Manual CUDA Graph capture — DONE, 2.6×** (608 → 233 ms). The winning lever; everything
   below is now moot or secondary on this board.
2. **TensorRT** — right idea (fusion = fewer launches) but full-stack TRT OOMs the 8 GB builder
   and can't parse the flow-matching graph (confirmed); encoder-only TRT (~15–25%) superseded.
3. **INT8 / narrower (256M) / precision tricks** — attack per-op *compute*, not op count: proven
   no help (256M random-weight: 591 ms) on a launch-bound workload.
4. **Fewer VLM layers** — linear (16/8/4 layers = 613/441/287 ms) but needs retrain; composes
   with graph capture if <150 ms is ever needed.
5. **`torch.compile` inductor** — dead (no triton wheel for CUDA-11.8 aarch64);
   `backend="cudagraphs"` ~4% (dynamo graph breaks). Superseded by manual capture.
6. **Orin NX** — no longer required to hit ~230 ms.

**Debug toolkit** (this directory, all runnable in the image): `parity_check.py` (cross-device
numeric parity), `verify_report.py` (memory/validity/latency), `profile_infer.py` (CPU-vs-GPU op
profile), `stage_breakdown.py` (processor/model/post timing), `copy_experiment.py` (autocast vs
pure-fp16), `kernel_count.py` (occupancy), `bench_cudagraph_manual.py` (CUDA Graph capture +
parity + bench), `bench_cudagraph_256m.py` (450M-vs-256M under capture),
`smoke_fp16_graph.py` (the integrated `precision="fp16-graph"` mode). Run any via
`docker run --runtime nvidia … wtlove876/smolvla-jetson:jp5-cu118 python3 /repo/deploy/jetson-native-torch/<script>.py`.

## The published image

Self-contained (CUDA baked in), runs on any JetPack-5 Jetson with just `--runtime nvidia`:

```bash
docker pull wtlove876/smolvla-jetson:jp5-cu118

docker run --rm --runtime nvidia -e NVIDIA_VISIBLE_DEVICES=all -e NVIDIA_DRIVER_CAPABILITIES=all \
  -v $PWD:/repo -e SRC=/repo/src -e CKPT=/repo/data/<checkpoint> \
  wtlove876/smolvla-jetson:jp5-cu118 python3 /repo/deploy/jetson-native-torch/parity_check.py
```

Wheels (`torch-2.2.2-cp310`, `torchvision-0.17.2-cp310` for linux_aarch64) are kept out of git;
grab them from the image (`/usr/local/lib/python3.10/dist-packages`) or the Jetson build host.

## Notes
- The build must run on the **internal NVMe**, not a USB SSD — a USB SSD (Ugreen RTL9210) dropped
  off the bus repeatedly under sustained build I/O.
- `torchvision.io` warns about libjpeg/libpng (built without them); harmless — SmolVLA doesn't use it.
