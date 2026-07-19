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

## Observation / command contract (task 1.5)

`/observation` is the topic set {`/vla_camera/image`, `/vla_camera/camera_info`,
`/observation/state`}; commands enter on `/cmd_vel`. All nodes run `use_sim_time`.

| Topic | Type | Direction | Notes |
|---|---|---|---|
| `/cmd_vel` | `geometry_msgs/TwistStamped` | in | Ackermann controller reference (v, ω). Matches the real rover's PX4 `rover_speed_steering` cmd_vel mapping. Clamp ω first (see limits) |
| `/observation/state` | `std_msgs/Float32MultiArray` | out | 50 Hz: `[speed m/s, yaw_rate rad/s, steering rad]` — yaw rate pose-derived, steering = equivalent bicycle angle |
| `/ackermann/gt_odom` | `nav_msgs/Odometry` | out | GT pose 50 Hz (label source; derive yaw rate from poses, not its twist) |
| `/vla_camera/image` | `sensor_msgs/Image` | out | 1280×800 RGB at 15 Hz (a real subscriber gets the full rate; `ros2 topic hz` under-reports ~4 Hz — CLI overhead on 3 MB msgs) |
| `/vla_camera/camera_info` | `sensor_msgs/CameraInfo` | out | fx=fy=537.0, cx=640, cy=400 |

Verified vehicle limits (M0): steering ±0.6 rad, **min feasible turn radius ≈ 0.341 m**
(inner-wheel limit) — the tracker must clamp ω to |ω| ≤ v / 0.341 before publishing.

### Clock discipline (sim)

- Gazebo is the single time authority: `/clock` is bridged and **every** node sets
  `use_sim_time` — a node on wall clock silently corrupts stamp arithmetic.
- All timestamps are `header.stamp` in sim time; nothing may read wall clock for
  data association. Latency compensation (async policy loop) counts the sim-time
  delta between the image `header.stamp` and chunk application.
- `ros2 topic pub` stamps with wall clock — never use it against the controller's
  `reference_timeout`; publish from a `use_sim_time` node (see the drive tests).
- The recorder stores sim-time stamps; episode replay (task 1.8) depends on them.

## Scenes + per-episode randomization (task 1.4)

Scene families (structure-only worlds): `open_ground.sdf`, `corridor.sdf`,
`parking_lot.sdf` (+ `props_ground.sdf` static smoke world). Per-episode
randomization is applied to the *running* world — no Gazebo restart:

```bash
ros2 launch rover_sim sim_bringup.launch.py world:=$(ros2 pkg prefix rover_sim)/share/rover_sim/worlds/corridor.sdf
ros2 run rover_sim scene_manager.py apply --scene corridor --seed 42 --out ep42.json
ros2 run rover_sim scene_manager.py clear --scene corridor
```

**One simulator instance only**: before (re)launching, make sure no stale `gz sim`
server is running (`pgrep -a ruby`). Two servers on the same world name merge
their gz-transport topics/services and requests round-robin between them —
episodes half-apply, teleports "fail", and debugging becomes archaeology.

Randomized per episode (seed-deterministic, JSON-logged for failure slicing):
sun elevation/azimuth/intensity (spawned as an `ep_light` entity — runtime
`light_config` never reaches the sensors render scene), ground color slab
(skipped in `parking_lot` so bay markings stay visible), prop set (shape ×
color with **guaranteed hard negatives** — a same-color and a same-shape
distractor always accompany the goal, and exact goal duplicates are excluded),
prop placement, rover spawn pose, goal + instruction string.
Exposure/extrinsic jitter/sensor noise are recorder-side hooks (M1).

GUI debugging: add `gui:=true` to the launch (needs X forwarding / `xhost +local:`).

**No default route?** If the host has no network (or gz prints `Exception sending a
multicast message`), export `GZ_IP=127.0.0.1` in every shell that talks to Gazebo —
gz-transport discovery needs a usable interface and falls over without one.

## Recording episodes (task 1.7)

With the sim running, one command produces one raw episode directory
(frames @15 Hz JPEG, 50 Hz `gt_pose`/`state` JSONL, expert commands,
intrinsics, `episode.json` with config + verdict + success/collision flags):

```bash
ros2 run rover_expert run_episode.py --scene open_ground --seed 2 --out-root /vla/rover/data/raw
```

Convert raw episodes to a LeRobotDataset (runs in the lerobot env, not ROS):

```bash
docker run --rm -v "$PWD":/work -w /work -e HF_HOME=/work/.hf_cache smolvla-edge:sim \
  python rover/datagen/to_lerobot.py --raw-root rover/data/raw --out rover/data/lerobot
```

Load with `video_backend='pyav'` (repo convention). The LeRobot `action` field
is the expert (v, ω) command for now — task 2.1's hindsight relabeler rebuilds
it as K×(x, y, v) waypoint chunks from the raw 50 Hz pose stream, which stays
in the episode dirs precisely for that purpose. `rover/data/` is gitignored.
