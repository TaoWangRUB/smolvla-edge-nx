# Native PyTorch + LeRobot + SmolVLA on Jetson Xavier NX (JetPack 5)

Running the **native PyTorch/LeRobot SmolVLA stack on the Xavier NX GPU** — the path everyone
says is impossible on JetPack 5 — built from source and verified. Complements the pure-Python
ONNX-Runtime deployment in [../README.md](../README.md); this is the "torch actually runs on the
device" alternative.

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
identical** actions to the reference (consistent with the ONNX parity gate's 4.3e-6).

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
