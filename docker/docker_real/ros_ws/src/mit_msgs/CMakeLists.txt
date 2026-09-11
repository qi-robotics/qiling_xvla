cmake_minimum_required(VERSION 3.8)
project(mit_msgs)

# find dependencies
find_package(ament_cmake REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(rosidl_default_generators REQUIRED)

rosidl_generate_interfaces(
        ${PROJECT_NAME}
        "msg/MITJointCommand.msg"
        "msg/MITJointCommands.msg"
        "msg/MITLowState.msg"
        DEPENDENCIES builtin_interfaces std_msgs geometry_msgs sensor_msgs
)
ament_package()



