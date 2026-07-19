#!/usr/bin/env bash
# Stage-one smoke training on the Titan X (task 2.4 process verification).
#
# Small data, few steps: the goal is to verify the pipeline end-to-end
# (dataset loads, SmolVLA adapts to the 30-dim flat waypoint chunk action,
# loss decreases, checkpoint saves) BEFORE any large datagen or rented-GPU
# run. Maxwell has no bf16/tensor cores: fp32, small batch, expect slow
# steps — that is fine for a smoke run.
#
# Action space note (D2 refinement for stock lerobot-train): the dataset
# stores the K=10 x (x,y,v) hindsight chunk FLAT (30 dims <= SmolVLA's
# max_action_dim=32) with policy chunk_size=1, because lerobot's
# delta_timestamps chunking would gather future frames' own-body-frame
# actions (wrong frame semantics for D2). The flow expert still predicts
# the full 2.5 s spatial intent per frame.
set -euo pipefail

REPO_ID="${REPO_ID:-local/rover_vla_v1}"
ROOT="${ROOT:-rover/data/lerobot/rover_vla_v1}"
BATCH_SIZE="${BATCH_SIZE:-8}"
STEPS="${STEPS:-300}"
OUTPUT_DIR="${OUTPUT_DIR:-rover/outputs/train/smoke_v1}"
GPU="${GPU:-1}"   # 1 = Titan X (12 GB), 0 = A2000 (4 GB)

docker run --rm --runtime nvidia \
  -e NVIDIA_VISIBLE_DEVICES=all -e CUDA_VISIBLE_DEVICES="${GPU}" \
  -e HF_HOME=/work/.hf_cache \
  -v "$PWD":/work -w /work \
  smolvla-edge:sim \
  lerobot-train \
    --policy.path=lerobot/smolvla_base \
    --policy.push_to_hub=false \
    --policy.chunk_size=1 \
    --policy.n_action_steps=1 \
    --dataset.repo_id="${REPO_ID}" \
    --dataset.root="${ROOT}" \
    --dataset.video_backend=pyav \
    --rename_map='{"observation.image": "observation.images.camera1"}' \
    --batch_size="${BATCH_SIZE}" \
    --steps="${STEPS}" \
    --save_freq="${STEPS}" \
    --log_freq=10 \
    --output_dir="${OUTPUT_DIR}"
