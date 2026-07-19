#!/usr/bin/env python3
"""Headless-by-default Gazebo bringup for the VLA rover sim.

Starts gz sim (server only unless gui:=true), spawns the rover, bridges the
minimal topic set the VLA pipeline needs, and activates ros2_control:

  out: /clock, /ackermann/gt_odom (GT pose 50 Hz), /vla_camera/image (15 Hz),
       /vla_camera/camera_info, /ackermann_steering_controller/odometry
  in:  /cmd_vel (geometry_msgs/TwistStamped -> ackermann_steering_controller)
"""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    IncludeLaunchDescription,
    RegisterEventHandler,
)
from launch.conditions import IfCondition, UnlessCondition
from launch.event_handlers import OnProcessExit
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import Command, LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def generate_launch_description() -> LaunchDescription:
    pkg_share = get_package_share_directory('rover_sim')
    gz_share = get_package_share_directory('ros_gz_sim')
    default_world = os.path.join(pkg_share, 'worlds', 'props_ground.sdf')
    xacro_file = os.path.join(pkg_share, 'urdf', 'rover_vla.urdf.xacro')

    world = LaunchConfiguration('world')
    gui = LaunchConfiguration('gui')
    use_sim_time = {'use_sim_time': True}

    declare_world = DeclareLaunchArgument(
        'world', default_value=default_world, description='SDF world file.')
    declare_gui = DeclareLaunchArgument(
        'gui', default_value='false', description='Run the Gazebo GUI client.')
    declare_x = DeclareLaunchArgument('x', default_value='0.0')
    declare_y = DeclareLaunchArgument('y', default_value='0.0')
    declare_z = DeclareLaunchArgument('z', default_value='0.1')

    gazebo_server = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(gz_share, 'launch', 'gz_sim.launch.py')),
        launch_arguments={'gz_args': ['-s -r ', world]}.items(),
        condition=UnlessCondition(gui),
    )
    gazebo_full = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(gz_share, 'launch', 'gz_sim.launch.py')),
        launch_arguments={'gz_args': ['-r ', world]}.items(),
        condition=IfCondition(gui),
    )

    robot_state_publisher = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        output='screen',
        parameters=[{
            'robot_description': ParameterValue(
                Command(['xacro ', xacro_file]), value_type=str),
            **use_sim_time,
        }],
    )

    spawn_entity = Node(
        package='ros_gz_sim',
        executable='create',
        arguments=[
            '-topic', 'robot_description',
            '-name', 'ackermann',
            '-x', LaunchConfiguration('x'),
            '-y', LaunchConfiguration('y'),
            '-z', LaunchConfiguration('z'),
        ],
        output='screen',
    )

    bridge = Node(
        package='ros_gz_bridge',
        executable='parameter_bridge',
        arguments=[
            '/clock@rosgraph_msgs/msg/Clock[gz.msgs.Clock',
            '/ackermann/gt_odom@nav_msgs/msg/Odometry[gz.msgs.Odometry',
            '/vla_camera/image@sensor_msgs/msg/Image[gz.msgs.Image',
            '/vla_camera/camera_info@sensor_msgs/msg/CameraInfo[gz.msgs.CameraInfo',
        ],
        output='screen',
        parameters=[use_sim_time],
    )

    state_publisher = Node(
        package='rover_sim',
        executable='state_publisher.py',
        output='screen',
    )

    controller_spawner = Node(
        package='controller_manager',
        executable='spawner',
        arguments=[
            'joint_state_broadcaster',
            'ackermann_steering_controller',
            '--controller-manager-timeout', '120',
        ],
        output='screen',
    )
    controllers_after_spawn = RegisterEventHandler(
        event_handler=OnProcessExit(
            target_action=spawn_entity,
            on_exit=[controller_spawner],
        ),
    )

    return LaunchDescription([
        declare_world,
        declare_gui,
        declare_x,
        declare_y,
        declare_z,
        gazebo_server,
        gazebo_full,
        robot_state_publisher,
        bridge,
        state_publisher,
        spawn_entity,
        controllers_after_spawn,
    ])
