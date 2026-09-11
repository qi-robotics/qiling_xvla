from setuptools import find_packages, setup

package_name = "qiling_rollout_ros"

setup(
    name=package_name,
    version="0.1.0",
    packages=find_packages(exclude=["test"]),
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
        ("share/" + package_name + "/launch", [
            "launch/rollout_host.launch.py",
            "launch/rollout_image_transport.launch.py",
        ]),
        ("share/" + package_name + "/config", ["config/rollout_host.yaml"]),
        ("share/" + package_name, ["README.md"]),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="qiling",
    maintainer_email="user@example.com",
    description="ROS 2 observation/action bridge for Qiling XVLA rollout.",
    license="Apache-2.0",
    entry_points={
        "console_scripts": [
            "rollout_ros_bridge = qiling_rollout_ros.rollout_ros_bridge:main",
            "rollout_image_compressor = qiling_rollout_ros.rollout_image_compressor:main",
        ],
    },
)
