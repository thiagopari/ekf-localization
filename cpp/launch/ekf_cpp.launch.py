"""Run the C++ EKF/UKF node. Start the sim first: ros2 launch ekf_localization turtlebot3_headless.launch.py"""
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

# Parameter names and defaults are the ones ekf_node.py declares.
ARGS = {
    'use_sim_time': 'true',
    'gps_rate': '1.0',
    'add_odom_noise': 'true',
    'odom_noise_std': '0.02',
    'enable_gps_outage': 'false',
    'gps_outage_start': '30.0',
    'gps_outage_duration': '15.0',
    'enable_scan_matching': 'true',
    'scan_match_min_interval': '0.1',
}


def generate_launch_description():
    return LaunchDescription(
        [DeclareLaunchArgument(name, default_value=value) for name, value in ARGS.items()] + [
            Node(
                package='ekf_localization_cpp',
                executable='ekf_node',
                output='screen',
                parameters=[{name: LaunchConfiguration(name) for name in ARGS}],
            ),
        ])
