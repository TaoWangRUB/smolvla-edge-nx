#!/bin/bash
# smolvla-edge ROS2 entrypoint — replaces the rover image's entrypoint, which expects a rover
# workspace (FastDDS eth profile probe + startup `rosdep install`). Here: source ROS, source
# the colcon overlay if it has been built, exec.
set -e

source "/opt/ros/${ROS_DISTRO:-jazzy}/setup.bash"

OVERLAY="/workspace/deploy/ros2/install/setup.bash"
if [[ -f "${OVERLAY}" ]]; then
  source "${OVERLAY}"
fi

exec "$@"
