#!/bin/bash
# Targeted re-generation of specific (scene, seed) pairs. Runs INSIDE the sim
# container, same as batch_datagen.sh:
#   docker exec -d vla_sim bash /vla/rover/datagen/regen_seeds.sh
#
# Why this exists: `colcon build --symlink-install` makes scene_manager.py a
# SYMLINK to the source tree, so editing the sampler takes effect on the NEXT
# episode of an already-running batch. On 2026-07-21 a bearing-distribution
# change went live mid-run and 95 episodes were generated under it. Because the
# sampler is seed-deterministic, the affected episodes can be found exactly
# (re-sample each seed, diff prop positions) and re-run — no full restart.
#
# Input: LIST, lines of "<scene> <seed>" (see datagen/v4_regen_list.txt).
# Groups by scene so each world is brought up ONCE, not per episode.
#
# Env: LIST, OUT_ROOT, LOG.
export GZ_IP=127.0.0.1
source /opt/ros/jazzy/setup.bash
source /vla/rover/ros2/install/setup.bash

LIST="${LIST:-/vla/rover/datagen/v4_regen_list.txt}"
OUT_ROOT="${OUT_ROOT:-/vla/rover/data/raw_v4}"
LOG="${LOG:-/vla/rover/data/regen_v4.log}"

# Sampler settings MUST match the batch these episodes belong to, or the
# regenerated scenes differ from their neighbours in a second way.
export ROVER_DIST_MIN="${ROVER_DIST_MIN:-2.0}"
export ROVER_DIST_MAX="${ROVER_DIST_MAX:-3.5}"
export ROVER_FILLERS="${ROVER_FILLERS:-1}"

[ -f "$LIST" ] || { echo "no list at $LIST"; exit 1; }
echo "REGEN_START $(date +%T) list=$LIST n=$(wc -l < "$LIST")" > "$LOG"
echo "REGEN_ENV dist=$ROVER_DIST_MIN-$ROVER_DIST_MAX fillers=$ROVER_FILLERS" >> "$LOG"

# Safe here: the sim container has its OWN pid namespace. Never do this in the
# ackermann_rover_humble compose containers, which run `pid: host`.
kill_sim () {
  pkill -f "ros2 launch rover_sim" 2>/dev/null
  pkill -f "gz sim" 2>/dev/null
  pkill -f parameter_bridge 2>/dev/null
  pkill -f state_publisher 2>/dev/null
  pkill -f robot_state_publisher 2>/dev/null
  sleep 3
}

for scene in $(awk '{print $1}' "$LIST" | sort -u); do
  kill_sim
  ros2 launch rover_sim sim_bringup.launch.py \
    world:=/vla/rover/ros2/install/rover_sim/share/rover_sim/worlds/$scene.sdf \
    > /tmp/regen_$scene.log 2>&1 &
  for i in $(seq 1 80); do
    grep -qaE "activated ackermann_steering" /tmp/regen_$scene.log && break
    sleep 3
  done
  if ! grep -qaE "activated ackermann_steering" /tmp/regen_$scene.log; then
    echo "ERROR sim $scene failed to start" >> "$LOG"
    continue
  fi
  echo "SCENE_UP $scene" >> "$LOG"

  for seed in $(awk -v s="$scene" '$1==s {print $2}' "$LIST"); do
    # Drop the stale episode first: run_episode writes into a seeded dir and a
    # leftover frames/ would mix old and new renders.
    rm -rf "$OUT_ROOT/${scene}_seed${seed}"
    out=$(timeout 240 ros2 run rover_expert run_episode.py \
          --scene "$scene" --seed "$((10#$seed))" --out-root "$OUT_ROOT" 2>/dev/null | tail -1)
    echo "EPISODE_DONE $scene $seed $out" >> "$LOG"
  done
done

kill_sim
echo "REGEN_DONE $(date +%T)" >> "$LOG"
