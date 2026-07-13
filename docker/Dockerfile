# smolvla-edge sim/dev image — CUDA torch + LeRobot 0.4.4 + gym-aloha (mujoco 2.x, matched)
#
# Why a container instead of the host python: the host is Python 3.12, where gym-aloha's pinned
# mujoco 2.3.7 has no wheel — forcing mujoco 3.x and a sim-version gap vs. the mujoco-2.x-generated
# ALOHA datasets/checkpoints (fine insertion fails, see design.md "Simulation setup"). This image
# uses the pytorch/py3.11 base so `pip install gym-aloha` gets the *matched* mujoco 2.3.7 +
# dm_control 1.0.14 pair the checkpoints were trained under.
#
# ⚠️ lerobot is 0.4.4 in here, NOT the repo's 0.5.0 pin: lerobot 0.5.x requires Python >=3.12,
# which is mutually exclusive with mujoco 2.3.7 (<=3.11). 0.4.4 is the same era as the pretrained
# ALOHA checkpoints — policies still normalize internally, so old-format checkpoints load with
# their stats intact (no processor pipeline needed). The host py3.12 env stays on lerobot 0.5.0.
#
# Source is NOT baked in — the repo root is mounted at /workspace by docker-compose.yml.

FROM pytorch/pytorch:2.6.0-cuda12.4-cudnn9-runtime

ENV DEBIAN_FRONTEND=noninteractive
SHELL ["/bin/bash", "-c"]

# System deps:
#  - ffmpeg: LeRobot video decode (torchcodec)
#  - libegl1/libglvnd0/libgles2: headless GPU rendering (MUJOCO_GL=egl) via the NVIDIA runtime
#  - libosmesa6: CPU fallback renderer (MUJOCO_GL=osmesa)
#  - libglfw3: dm_control's default GL loader
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg git build-essential \
    libegl1 libglvnd0 libgles2 libgl1 libosmesa6 libglfw3 \
    && rm -rf /var/lib/apt/lists/*
# build-essential: some lerobot deps (evdev) ship no wheel and compile at install time

# NVIDIA runtime injects the vendor EGL stack when graphics capability is requested;
# this ICD file makes GLVND find it.
RUN mkdir -p /usr/share/glvnd/egl_vendor.d && printf '%s\n' \
    '{ "file_format_version" : "1.0.0", "ICD" : { "library_path" : "libEGL_nvidia.so.0" } }' \
    > /usr/share/glvnd/egl_vendor.d/10_nvidia.json

# Python stack. Torch/torchvision are pinned to the base image's versions so pip cannot
# "upgrade" them to a CPU build. gym-aloha resolves to mujoco 2.3.7 + dm_control 1.0.14 on
# py3.11 — the matched pair the ALOHA sim datasets/checkpoints were generated with.
RUN pip install --no-cache-dir \
    "torch==2.6.0" "torchvision==0.21.0" \
    "lerobot[smolvla]==0.4.4" \
    "gym-aloha==0.1.1" \
    "grpcio==1.66.2" "grpcio-tools==1.66.2" "protobuf>=5.27,<6" \
    "huggingface_hub" numpy pandas matplotlib "imageio[ffmpeg]"

ENV MUJOCO_GL=egl \
    PYTHONPATH=/workspace/src \
    HF_HOME=/root/.cache/huggingface

WORKDIR /workspace
CMD ["bash"]
