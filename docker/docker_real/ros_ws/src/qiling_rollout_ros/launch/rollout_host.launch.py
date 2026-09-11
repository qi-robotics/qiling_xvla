from pathlib import Path

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    package_share = Path(get_package_share_directory("qiling_rollout_ros"))
    default_config = package_share / "config" / "rollout_host.yaml"
    return LaunchDescription([
        DeclareLaunchArgument("config_file", default_value=str(default_config)),
        DeclareLaunchArgument(
            "execution_mode",
            default_value="",
            description="Empty uses YAML; allowed values: shadow, armed.",
        ),
        Node(
            package="qiling_rollout_ros",
            executable="rollout_ros_bridge",
            name="qiling_rollout_ros_bridge",
            output="screen",
            parameters=[{
                "config_file": LaunchConfiguration("config_file"),
                "execution_mode": LaunchConfiguration("execution_mode"),
            }],
        ),
    ])
