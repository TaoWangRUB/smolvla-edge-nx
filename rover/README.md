# Rover VLA — simulation-first workspace

Phase workspace for the `rover-vla-sim-first` openspec change: a language-conditioned
Ackermann navigation policy, developed simulation-first in **Gazebo Harmonic** (the M0
feasibility gate rejected Isaac Sim on available hardware — see
`openspec/changes/rover-vla-sim-first/tasks.md` §1.1).

## Layout

- `ros2/src/rover_sim` — minimal sim package: the 1/16 Ackermann rover URDF (verbatim
  vehicle dynamics from the real rover's digital twin in `ackermann_rover_humble`,
  PX4/ArduPilot/RealSense coupling stripped), the OV9782-locked forward camera
  (1280×800 @ 15 Hz, HFOV 100°, fx=fy=537.0), a props smoke world, controllers, and a
  headless-by-default bringup launch.

## Running (uses the existing `ackermann_rover_x86_64_jazzy` image)

```bash
docker run -d --name vla_sim --runtime nvidia \
  -e NVIDIA_VISIBLE_DEVICES=all -e NVIDIA_DRIVER_CAPABILITIES=all \
  -v "$PWD":/vla -w /vla --network host \
  ackermann_rover_x86_64_jazzy sleep infinity

docker exec vla_sim bash -c 'sudo apt-get update -qq && sudo apt-get install -y ros-jazzy-gz-ros2-control'  # until the image is rebuilt
docker exec vla_sim bash -c 'source /opt/ros/jazzy/setup.bash && cd /vla/rover/ros2 && colcon build --symlink-install'
docker exec -d vla_sim bash -c 'source /opt/ros/jazzy/setup.bash && source /vla/rover/ros2/install/setup.bash && ros2 launch rover_sim sim_bringup.launch.py'
```

Interface (all `use_sim_time`):

| Topic | Type | Direction | Notes |
|---|---|---|---|
| `/cmd_vel` | `geometry_msgs/TwistStamped` | in | Ackermann controller reference |
| `/ackermann/gt_odom` | `nav_msgs/Odometry` | out | GT pose 50 Hz (label source; derive yaw rate from poses, not its twist) |
| `/vla_camera/image` | `sensor_msgs/Image` | out | 1280×800 RGB; raw DDS caps ~4.3 Hz — compress/downscale for recording |
| `/vla_camera/camera_info` | `sensor_msgs/CameraInfo` | out | fx=fy=537.0, cx=640, cy=400 |

Verified vehicle limits (M0): steering ±0.6 rad, **min feasible turn radius ≈ 0.341 m**
(inner-wheel limit) — the tracker must clamp ω to |ω| ≤ v / 0.341 before publishing.

GUI debugging: add `gui:=true` to the launch (needs X forwarding / `xhost +local:`).
