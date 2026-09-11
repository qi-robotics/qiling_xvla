from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument("output_rate_hz", default_value="15.0"),
        DeclareLaunchArgument("jpeg_quality", default_value="80"),
        Node(
            package="qiling_rollout_ros",
            executable="rollout_image_compressor",
            name="qiling_rollout_image_compressor",
            output="screen",
            parameters=[{
                "output_rate_hz": ParameterValue(
                    LaunchConfiguration("output_rate_hz"), value_type=float),
                "jpeg_quality": ParameterValue(
                    LaunchConfiguration("jpeg_quality"), value_type=int),
            }],
        ),
    ])
