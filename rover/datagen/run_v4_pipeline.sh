#!/bin/bash
# v4 horizon test: verify -> convert -> train, unattended (openspec tasks 2.9).
# Runs on the HOST (drives docker), not inside the sim container.
#
# Gate order matters. Verification runs FIRST and hard-stops the pipeline,
# because --symlink-install let a sampler edit go live mid-batch once already:
# a mixed dataset is invisible in the loss curve and would silently confound
# the very hypothesis v4 exists to test.
set -uo pipefail
cd "$(dirname "$0")/../.."

RAW=rover/data/raw_v4
# to_lerobot appends repo_id's basename: out = OUT_PARENT/<repo basename>.
# Passing the full dataset path creates rover_vla_v4/rover_vla_v4 and the
# existence check below then "fails" on a conversion that actually succeeded.
OUT_PARENT=rover/data/lerobot
OUT="$OUT_PARENT/rover_vla_v4"
REPO=local/rover_vla_v4
SCENES="${SCENES:-open_ground,parking_lot}"     # corridor: no path at 2.0-3.5 m
LOG=rover/data/pipeline_v4.log

say () { echo "[$(date +%T)] $*" | tee -a "$LOG"; }
say "PIPELINE_START scenes=$SCENES"

# --- 1. verify seed-determinism -------------------------------------------
say "verify: re-sampling every recorded seed against the current sampler"
docker run --rm -v "$PWD":/vla -w /vla smolvla-edge:sim \
  python /vla/rover/datagen/verify_dataset.py --raw-root "$RAW" \
  --write-regen-list /vla/rover/datagen/v4_postverify_regen.txt >> "$LOG" 2>&1
VERIFY_RC=$?
# Only open_ground/parking_lot are converted, so corridor mismatches are moot.
BAD=$(grep -cE '^(open_ground|parking_lot) ' rover/datagen/v4_postverify_regen.txt 2>/dev/null || echo 0)
say "verify: rc=$VERIFY_RC, og/pl mismatches=$BAD"
if [ "$BAD" -gt 0 ]; then
  say "ABORT: $BAD converted-scene episodes disagree with the sampler."
  say "       Fix with regen_seeds.sh before training - do NOT train on a mixed set."
  exit 1
fi

# --- 2. convert ------------------------------------------------------------
if [ -d "$OUT" ]; then
  say "convert: $OUT exists, moving aside"
  mv "$OUT" "${OUT}.bak.$(date +%s)"
fi
say "convert: -> $OUT (chunk_k=10 x (x,y,v) = 30-dim flat action, fps 15)"
docker run --rm --shm-size=8g -v "$PWD":/vla -w /vla -e HF_HOME=/vla/.hf_cache \
  smolvla-edge:sim \
  python /vla/rover/datagen/to_lerobot.py \
    --raw-root "$RAW" --out "$OUT_PARENT" --repo-id "$REPO" \
    --scenes "$SCENES" --chunk-k 10 --chunk-dt 0.25 --fps 15 >> "$LOG" 2>&1
if [ ! -f "$OUT/meta/info.json" ]; then
  say "ABORT: conversion produced no dataset"
  exit 1
fi
python3 - "$OUT" >> "$LOG" 2>&1 <<'PY'
import json, sys
m = json.load(open(sys.argv[1] + '/meta/info.json'))
print(f"  converted: {m['total_episodes']} episodes, {m['total_frames']} frames")
print(f"  action dim {m['features']['action']['shape']}")
PY
say "$(grep -E 'converted:|action dim' "$LOG" | tail -2 | tr '\n' ' ')"

# --- 3. train (v3's exact recipe; only the goal range differs) -------------
say "train: 10000 steps, batch 8, lr 1e-4, 16 VLM layers, frozen vision"
REPO_ID="$REPO" ROOT="$OUT" STEPS=10000 BATCH_SIZE=8 SAVE_FREQ=500 \
  OUTPUT_DIR=rover/outputs/train/stage1_v4 GPU=1 \
  bash rover/train_smoke.sh >> "$LOG" 2>&1
say "train: exit=$?"

CKPT=$(ls -d rover/outputs/train/stage1_v4/checkpoints/*/ 2>/dev/null | sort | tail -1)
say "PIPELINE_DONE last_checkpoint=${CKPT:-NONE}"
