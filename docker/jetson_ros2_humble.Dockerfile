# ROS 2 Humble overlay on the from-source JetPack-5 torch image (design: all-ROS2 policy
# serving on the Xavier NX). The base is Ubuntu 22.04 + Python 3.10 — exactly Humble's
# target tier — so rclpy coexists with the custom cp310 torch build.
#
#   docker build -f docker/jetson_ros2_humble.Dockerfile -t smolvla-jetson:humble .
#
# The policy node itself lives in the repo (mounted at /repo); smolvla_msgs is built once
# into /opt/msgs_ws (baked here so runs need no colcon at startup).
FROM wtlove876/smolvla-jetson:jp5-cu118

ENV DEBIAN_FRONTEND=noninteractive LANG=C.UTF-8 LC_ALL=C.UTF-8
RUN apt-get update && apt-get install -y --no-install-recommends \
        curl gnupg lsb-release && \
    curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key \
        -o /usr/share/keyrings/ros-archive-keyring.gpg && \
    echo "deb [arch=arm64 signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] \
        http://packages.ros.org/ros2/ubuntu jammy main" \
        > /etc/apt/sources.list.d/ros2.list && \
    apt-get update && apt-get install -y --no-install-recommends \
        ros-humble-ros-base ros-humble-sensor-msgs \
        python3-colcon-common-extensions ros-humble-rosidl-default-generators && \
    rm -rf /var/lib/apt/lists/*

# Bake smolvla_msgs (identical .msg sources as the Jazzy side — cross-distro DDS needs
# matching type definitions).
COPY deploy/ros2/src/smolvla_msgs /opt/msgs_src/smolvla_msgs
RUN bash -lc "source /opt/ros/humble/setup.bash && \
    colcon build --base-paths /opt/msgs_src --packages-select smolvla_msgs \
        --install-base /opt/msgs_ws --build-base /tmp/msgs_build && rm -rf /tmp/msgs_build"

# Entry: callers source /opt/ros/humble/setup.bash + /opt/msgs_ws/setup.bash themselves.
