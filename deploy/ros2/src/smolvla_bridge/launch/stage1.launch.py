"""Stage 1 stack: sim bridge + C++ async client + event recorder (task 3.7).

    ros2 launch smolvla_bridge stage1.launch.py episodes:=5 g:=0.7 \
        task:="Pick up the cube with the right arm and transfer it to the left arm."

Assumes sim-server and policy-server compose services are up (gRPC, other container).
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, Shutdown
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

ARGS = [
    ("sim_server", "sim-server:50052"),
    ("policy_server", "policy-server:50051"),
    ("task", ""),
    ("g", "0.7"),
    ("epsilon", "0.0"),
    ("aggregate", "new_wins"),
    ("ramp_in", "0"),
    ("fps", "50.0"),
    ("episodes", "5"),
    ("start_seed", "0"),
    ("max_steps", "400"),
    ("results_path", "/workspace/benchmarks/results/ros2/stage1.json"),
    ("events_path", "/workspace/benchmarks/results/ros2/stage1_events.jsonl"),
    ("gif_dir", ""),
]


def generate_launch_description() -> LaunchDescription:
    decls = [DeclareLaunchArgument(n, default_value=v) for n, v in ARGS]
    cfg = {n: LaunchConfiguration(n) for n, _ in ARGS}
    return LaunchDescription(decls + [
        Node(
            package="smolvla_client",
            executable="async_client",
            output="screen",
            parameters=[{
                "server": cfg["policy_server"], "task": cfg["task"], "g": cfg["g"],
                "epsilon": cfg["epsilon"], "aggregate": cfg["aggregate"],
                "ramp_in": cfg["ramp_in"],
            }],
        ),
        Node(
            package="smolvla_bridge",
            executable="event_recorder",
            output="screen",
            parameters=[{"output": cfg["events_path"]}],
        ),
        Node(
            package="smolvla_bridge",
            executable="sim_bridge",
            output="screen",
            # the bridge is the run's lifecycle owner: when it finishes its episodes the
            # whole launch (client, recorder) must exit, or the container lives forever
            on_exit=Shutdown(),
            parameters=[{
                "server": cfg["sim_server"], "fps": cfg["fps"],
                "episodes": cfg["episodes"], "start_seed": cfg["start_seed"],
                "max_steps": cfg["max_steps"], "results_path": cfg["results_path"],
                "gif_dir": cfg["gif_dir"],
            }],
        ),
    ])
