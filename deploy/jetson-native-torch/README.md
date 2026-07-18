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
| ONNX-ORT (fp32) | ~610 ms | reference |

Takeaway: on Volta, native torch **matches** ONNX but doesn't beat it (no flash-attention on
sm_72; this lean build also drops mem-efficient attention). **Always `.half()` on Volta.**

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
decouples inference from the control rate: one ~700 ms inference yields **50 actions**, executed
with no network in between. At a rover's ~10 Hz base-control rate a 50-action chunk = **5 s of
motion**, so the net runs once per ~5 s and the async runner hides it entirely. **On-device solely
is viable** — offloading to a host (impractical on a moving robot) is unnecessary.

**Optimization levers, assessed (ranked by realism on this board):**
1. **TensorRT / ORT-TRT-EP on the SigLIP vision encoder only** — the one worthwhile lever (fuses
   the encoder's GEMMs/LayerNorms → fewer launches, attacking the launch-bound half). Full-stack
   TRT OOMs the 8 GB builder and can't parse the flow-matching dynamo graph (both confirmed
   earlier). Realistic **~15–25% end-to-end** (~630 → ~500 ms); can't reach laptop numbers.
2. **INT8** — accuracy-gated; TRT-INT8 calibration is the heavy path.
3. **`torch.compile` / CUDA graphs** — dead here (no triton wheel for CUDA-11.8 aarch64).
4. **Newer hardware (Orin NX)** — highest ROI if <300 ms is truly required; no software knob
   closes the Volta gap.

**Debug toolkit** (this directory, all runnable in the image): `parity_check.py` (cross-device
numeric parity), `verify_report.py` (memory/validity/latency), `profile_infer.py` (CPU-vs-GPU op
profile), `stage_breakdown.py` (processor/model/post timing), `copy_experiment.py` (autocast vs
pure-fp16), `kernel_count.py` (occupancy). Run any via
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
