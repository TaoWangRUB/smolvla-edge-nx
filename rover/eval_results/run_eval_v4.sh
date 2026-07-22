#!/bin/bash
# Closed-loop + swap eval for one checkpoint on v4-style scenes (openspec 2.9).
# Runs on the HOST. Orchestrates two containers:
#   torch  (smolvla-edge:sim, GPU 1)  policy_server.py  :8790
#   sim    (vla_sim, ROS+gz)          run_eval.py       --swap
#
#   bash rover/eval_results/run_eval_v4.sh <ckpt_tag> <checkpoint_dir> [scenes...]
#
# Scenes are applied at v4 range (2.0-3.5 m) via ROVER_DIST_*, so this is a
# fair test of what the policy trained on. The SAME command with the v3
# checkpoint is the control that separates "short-horizon training helped"
# from "short-horizon eval is easier" (task 2.9).
set -uo pipefail
cd "$(dirname "$0")/../.."

TAG="${1:?tag}"; CKPT="${2:?checkpoint dir}"; shift 2
SCENES=("${@:-open_ground parking_lot}")
PORT=8790
SRVC="eval_server_${TAG}"
SIMC=vla_sim
say () { echo "[$(date +%T)] $*"; }

# v4 world spawns props at 2.0-3.5 m; every scene_manager apply inherits this.
export ROVER_DIST_MIN=2.0 ROVER_DIST_MAX=3.5 ROVER_FILLERS=1

say "eval $TAG  ckpt=$CKPT  scenes=${SCENES[*]}"

# --- policy server: dedicated HOST-network torch container on the Titan X --
# A bridge-network container cannot be reached at 127.0.0.1:$PORT from the
# host-network sim container, so the server gets its own --network host
# container rather than reusing a stray one.
docker rm -f "$SRVC" >/dev/null 2>&1
# Maxwell (Titan X, sm_52) has no bf16: lerobot loads the VLM as bfloat16 and
# cuBLAS fails with CUBLAS_STATUS_NOT_SUPPORTED on the first GEMM. Same sed
# patch train_smoke.sh applies, needed again here because the server container
# loads the checkpoint fresh.
SED="sed -i 's/torch_dtype=\"bfloat16\"/torch_dtype=\"float32\"/' \
  /opt/conda/lib/python3.11/site-packages/lerobot/policies/smolvla/smolvlm_with_expert.py"
docker run -d --name "$SRVC" --runtime nvidia --network host \
  -e NVIDIA_VISIBLE_DEVICES=all -e CUDA_VISIBLE_DEVICES=1 -e HF_HOME=/work/.hf_cache \
  -e PYTHONPATH=/work/src \
  -v "$PWD":/work -w /work smolvla-edge:sim \
  bash -lc "$SED && python rover/runtime/policy_server.py \
    --checkpoint $CKPT/checkpoints/last/pretrained_model --port $PORT" >/dev/null
# wait for "policy server on :PORT (device=cuda)"
for i in $(seq 1 60); do
  docker logs "$SRVC" 2>&1 | grep -qa "policy server on" && break
  sleep 5
done
say "server: $(docker logs "$SRVC" 2>&1 | tail -1)"

# --- eval each scene (swap on) --------------------------------------------
for scene in ${SCENES[*]}; do
  # the sim world must match the scene family
  docker exec "$SIMC" bash -lc "P1='gz''[ ]sim'; ps -o cmd -C ruby 2>/dev/null | grep -q ${scene}.sdf" \
    || say "WARN: sim world may not be $scene (start ${scene}.sdf first)"
  OUT=rover/eval_results/eval_${TAG}_${scene}_v4scenes.log
  say "eval $scene -> $OUT"
  docker exec "$SIMC" bash -lc "
    export GZ_IP=127.0.0.1 ROVER_DIST_MIN=2.0 ROVER_DIST_MAX=3.5 ROVER_FILLERS=1
    source /opt/ros/jazzy/setup.bash && source /vla/rover/ros2/install/setup.bash
    cd /vla && ros2 run rover_expert run_eval.py \
      --scene $scene --seed0 9000 --episodes 10 --swap --server-port $PORT" \
    2>&1 | stdbuf -oL grep -E '"outcome"|SWAP|swap|success|SUMMARY' | tee "$OUT"
done

docker rm -f "$SRVC" >/dev/null 2>&1
say "eval $TAG done"
