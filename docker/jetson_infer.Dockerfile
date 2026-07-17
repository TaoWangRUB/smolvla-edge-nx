# smolvla-edge — Jetson Xavier NX on-device ONNX inference image (Phase 2, Mode A).
#
# Why NOT layer on the rover image (ackermann_rover_aarch64_jazzy) — the tempting reuse:
#   - Jazzy => Ubuntu 24.04 => Python 3.12. The only prebuilt onnxruntime-gpu wheels for
#     JetPack 5 (L4T R35, CUDA 11.4) are cp38 (Python 3.8). A py3.12 + CUDA-11.4 ORT wheel
#     does not exist, so that base forces a multi-hour build-from-source.
#   - Ubuntu 24.04 (glibc 2.39) is off the supported L4T R35 path: the host is Ubuntu 20.04
#     (glibc 2.31) and the CUDA/cuDNN/TensorRT libraries the nvidia container runtime injects
#     are built against 20.04. 24.04 "usually" runs them, but it's fragile and unsupported.
#   - It carries 11.6 GB of ROS payload we don't use, and this Jetson has ~14 GB free.
#   - Its `ros2` compose service is deliberately CPU-only ("no GPU in Stage 1"), so GPU access
#     isn't even wired there.
#
# So we MATCH THE HOST instead: an L4T r35.4.1 base (Ubuntu 20.04, Python 3.8); CUDA 11.4 +
# cuDNN + TensorRT are injected at runtime by the nvidia container runtime (compose sets
# runtime: nvidia). This is the natively supported Jetson path and stays tiny on disk.
#
# The exported graph bakes in the instruction tokens AND all normalization
# (see models/onnx/*.meta.json: image[0,1]->[-1,1], state MEAN_STD, action MEAN_STD), so
# on-device inference needs ONLY numpy + onnxruntime-gpu — no torch, no lerobot, no tokenizer.

# l4t-base has no r35.4.1 tag (deprecated at r35.2.1); l4t-jetpack:r35.4.1 exists but bundles
# CUDA/cuDNN/TRT (~several GB, and this Jetson has ~14 GB free). We use the lean l4t-base:r35.2.1
# instead: it's Ubuntu 20.04 + Python 3.8, and CUDA 11.4 + cuDNN + TensorRT are injected from
# the r35.4.1 HOST at runtime by the nvidia container runtime — so the r35.2.1↔r35.4.1 minor
# gap is immaterial (the container ships no CUDA of its own). l4t-base is public on nvcr.io.
FROM nvcr.io/nvidia/l4t-base:r35.2.1

ENV DEBIAN_FRONTEND=noninteractive
SHELL ["/bin/bash", "-c"]

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3-pip python3-dev libopenblas-dev \
    && rm -rf /var/lib/apt/lists/*

RUN python3 -m pip install --no-cache-dir --upgrade "pip<25"

# onnxruntime-gpu for JetPack 5 / CUDA 11.4 / cp38, VENDORED as a wheel (this build has no
# reliable route to the Jetson AI Lab pip index — its DNS is flaky from both this Jetson and
# the dev box). The wheel is the Jetson Zoo build (cp38 / linux_aarch64) with the CUDA and
# TensorRT execution providers baked in; re-fetch it from https://elinux.org/Jetson_Zoo#ONNX_Runtime
# if you ever need to. Its pure-python deps (numpy/protobuf/sympy/...) still come from PyPI,
# which IS reachable. numpy<2 to satisfy the py3.8 / ORT-1.15 ABI.
# pip needs the PEP 427 filename intact, so COPY it verbatim (don't rename to *.whl).
COPY deploy/onnx/wheels/onnxruntime_gpu-1.15.1-cp38-cp38-linux_aarch64.whl /tmp/ort/
RUN python3 -m pip install --no-cache-dir "numpy>=1.24.4,<2" \
    && python3 -m pip install --no-cache-dir /tmp/ort/onnxruntime_gpu-1.15.1-cp38-cp38-linux_aarch64.whl \
    && rm -rf /tmp/ort

# Only needed if you serve the graph over the Policy gRPC proto (client/server point); the
# pure benchmark path (deploy/onnx/bench_ort.py) does not import these.
RUN python3 -m pip install --no-cache-dir "grpcio==1.66.2" "protobuf>=5.27,<6"

ENV PYTHONPATH=/workspace/src
WORKDIR /workspace
CMD ["bash"]
