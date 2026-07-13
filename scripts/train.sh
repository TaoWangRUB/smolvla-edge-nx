#!/usr/bin/env bash
# Phase 1 fine-tune. Run this on the rented A100/H100 (not the Titan X).
#
# Defaults match configs/train.aloha_sim.yaml (the no-robot sim showcase). Override via env vars:
#   DATASET_REPO_ID=lerobot/svla_so101_pickplace OUTPUT_DIR=outputs/train/smolvla_so101 bash scripts/train.sh
#   BATCH_SIZE=16 STEPS=20000 bash scripts/train.sh
#
# Smaller GPU? SmolVLA uses ~11.5 GB at batch 44; batch 16 fits ~6 GB.
set -euo pipefail

POLICY_PATH="${POLICY_PATH:-lerobot/smolvla_base}"
DATASET_REPO_ID="${DATASET_REPO_ID:-lerobot/aloha_sim_insertion_human}"
BATCH_SIZE="${BATCH_SIZE:-64}"
STEPS="${STEPS:-20000}"
SAVE_FREQ="${SAVE_FREQ:-2000}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/train/smolvla_aloha}"

echo "[train] policy=${POLICY_PATH} dataset=${DATASET_REPO_ID}"
echo "[train] batch_size=${BATCH_SIZE} steps=${STEPS} -> ${OUTPUT_DIR}"

# 20k steps ~= 4h on a single A100. Resume from a checkpoint by pointing
# --policy.path at an existing checkpoints/last dir instead of the base model.
lerobot-train \
  --policy.path="${POLICY_PATH}" \
  --dataset.repo_id="${DATASET_REPO_ID}" \
  --batch_size="${BATCH_SIZE}" \
  --steps="${STEPS}" \
  --save_freq="${SAVE_FREQ}" \
  --output_dir="${OUTPUT_DIR}"

echo "[train] done. Checkpoints in ${OUTPUT_DIR}/checkpoints/"
echo "[train] next: python -m smolvla_edge.eval --mode sim --policy-path ${OUTPUT_DIR}/checkpoints/last \\"
echo "                 --env-id gym_aloha/AlohaInsertion-v0 --episodes 20"
