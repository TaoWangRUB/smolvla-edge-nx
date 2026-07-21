#!/bin/bash
# Batch episode generation (task 2.3). Runs INSIDE the sim container:
#   docker exec -d <c> bash /vla/rover/datagen/batch_datagen.sh
#
# One scene family at a time: launch its world, wait for controllers, loop
# run_episode over a seed range, then move on. Failures are recorded (with
# episode.json verdicts) and skipped at conversion. Progress goes to
# $LOG (one EPISODE_DONE line each; SCENE_UP / BATCH_DONE / ERROR markers).
#
# Env overrides: SEEDS_PER_SCENE (default 190), OUT_ROOT, LOG.
# (No `set -u`: ROS setup.bash trips over it with unset AMENT_* variables.)
export GZ_IP=127.0.0.1
source /opt/ros/jazzy/setup.bash
source /vla/rover/ros2/install/setup.bash

SEEDS_PER_SCENE="${SEEDS_PER_SCENE:-190}"
OUT_ROOT="${OUT_ROOT:-/vla/rover/data/raw}"
LOG="${LOG:-/vla/rover/data/batch_full.log}"

echo "BATCH_START $(date +%T) seeds_per_scene=$SEEDS_PER_SCENE" > "$LOG"

run_scene () {
  local scene=$1 seed0=$2
  pkill -f "ros2 launch rover_sim" 2>/dev/null
  pkill -f "gz sim" 2>/dev/null
  pkill -f parameter_bridge 2>/dev/null
  pkill -f state_publisher 2>/dev/null
  pkill -f robot_state_publisher 2>/dev/null
  sleep 3
  ros2 launch rover_sim sim_bringup.launch.py \
    world:=/vla/rover/ros2/install/rover_sim/share/rover_sim/worlds/$scene.sdf \
    > /tmp/sim_$scene.log 2>&1 &
  for i in $(seq 1 80); do
    grep -qaE "activated ackermann_steering" /tmp/sim_$scene.log && break
    sleep 3
  done
  if ! grep -qaE "activated ackermann_steering" /tmp/sim_$scene.log; then
    echo "ERROR sim $scene failed to start" >> "$LOG"
    return 1
  fi
  echo "SCENE_UP $scene" >> "$LOG"
  for k in $(seq 0 $((SEEDS_PER_SCENE - 1))); do
    local seed=$((seed0 + k))
    out=$(timeout 240 ros2 run rover_expert run_episode.py \
          --scene $scene --seed $seed --out-root "$OUT_ROOT" 2>/dev/null | grep -oE "\{.*\}" | tail -1)
    echo "EPISODE_DONE $scene $seed $out" >> "$LOG"
  done
}

run_scene open_ground 1000
run_scene corridor 2000
run_scene parking_lot 3000

pkill -f "ros2 launch rover_sim" 2>/dev/null
pkill -f "gz sim" 2>/dev/null
echo "BATCH_DONE $(date +%T)" >> "$LOG"
