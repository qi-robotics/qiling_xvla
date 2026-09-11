#include "topic_convertor/topic_convertor_node.hpp"

#include <cmath>

namespace {

void rpy_to_quaternion(float roll, float pitch, float yaw, float & w, float & x, float & y, float & z)
{
    const float cr = std::cos(roll * 0.5f);
    const float sr = std::sin(roll * 0.5f);
    const float cp = std::cos(pitch * 0.5f);
    const float sp = std::sin(pitch * 0.5f);
    const float cy = std::cos(yaw * 0.5f);
    const float sy = std::sin(yaw * 0.5f);
    w = cr * cp * cy + sr * sp * sy;
    x = sr * cp * cy - cr * sp * sy;
    y = cr * sp * cy + sr * cp * sy;
    z = cr * cp * sy - sr * sp * cy;
}

}  // namespace

MitToQisnConverter::MitToQisnConverter() : Node("mit_to_qisn_converter")
{
    expected_motor_count_ = static_cast<size_t>(
        this->declare_parameter<int>("expected_motor_count", 26));
    strict_command_size_ = this->declare_parameter<bool>("strict_command_size", true);
    enable_state_bridge_ = this->declare_parameter<bool>("enable_state_bridge", true);
    // The real-robot workflow uses this node as a bidirectional bridge by
    // default: lowstate -> human_lower_state and human_lower_command -> lowcmd.
    enable_command_bridge_ = this->declare_parameter<bool>("enable_command_bridge", true);

    // Initialize subscribers and publishers
    if (enable_command_bridge_) {
        mit_cmd_sub_ = this->create_subscription<mit_msgs::msg::MITJointCommands>(
            "/human_lower_command", 10, std::bind(&MitToQisnConverter::MitJointCommandsCallback, this, std::placeholders::_1));
        qisn_cmd_pub_ = this->create_publisher<qi::msg::LowCmd>("lowcmd", 10);
    }
    if (enable_state_bridge_) {
        mit_state_pub_ = this->create_publisher<mit_msgs::msg::MITLowState>("/human_lower_state", 10);
        qisn_state_sub_ = this->create_subscription<qi::msg::LowState>(
            "lowstate", 10, std::bind(&MitToQisnConverter::QisnLowStateCallback, this, std::placeholders::_1));
    }

    RCLCPP_INFO(
        this->get_logger(),
        "Mit to qisn topic convertor running, expected_motor_count=%zu, strict_command_size=%s, "
        "enable_state_bridge=%s, enable_command_bridge=%s",
        expected_motor_count_,
        strict_command_size_ ? "true" : "false",
        enable_state_bridge_ ? "true" : "false",
        enable_command_bridge_ ? "true" : "false");
}

/**
 * @brief MIT关节命令转qisn关节命令
 * 
 * @param msg MIT关节命令
 */
void MitToQisnConverter::MitJointCommandsCallback(const mit_msgs::msg::MITJointCommands::SharedPtr msg)
{
    const int mit_cmd_len = static_cast<int>(msg->commands.size());
    size_t motor_count = static_cast<size_t>(mit_cmd_len);
    if (expected_motor_count_ > 0) {
        if (static_cast<size_t>(mit_cmd_len) != expected_motor_count_) {
            const char * message =
                "MIT command has %d motors, expected %zu motors for real body SDK "
                "(12 legs + 7 left arm + 7 right arm; dexterous hands are not in lowcmd)";
            if (strict_command_size_) {
                RCLCPP_ERROR_THROTTLE(
                    this->get_logger(), *this->get_clock(), 2000,
                    message,
                    mit_cmd_len, expected_motor_count_);
                return;
            }
            RCLCPP_WARN_THROTTLE(
                this->get_logger(), *this->get_clock(), 5000,
                "truncating MIT command from %d to %zu motors (strict_command_size=false)",
                mit_cmd_len, expected_motor_count_);
            motor_count = std::min(static_cast<size_t>(mit_cmd_len), expected_motor_count_);
        }
    }

    qi::msg::LowCmd qisn_cmd;
    qisn_cmd.motors.resize(motor_count);
    qisn_cmd.mode = 1;
    qisn_cmd.mode_ak = 1;
    qisn_cmd.mode_ctrl = 1;

    for (size_t i = 0; i < motor_count; i++) {
        qisn_cmd.motors[i].mode = 1;
        qisn_cmd.motors[i].q = msg->commands[i].pos;
        qisn_cmd.motors[i].dq = msg->commands[i].vel;
        qisn_cmd.motors[i].tau = msg->commands[i].eff;
        qisn_cmd.motors[i].kp = msg->commands[i].kp;
        qisn_cmd.motors[i].kd = msg->commands[i].kd;
    }
    if (qisn_cmd_pub_) {
        qisn_cmd_pub_->publish(qisn_cmd);
    }
}

/**
 * @brief qisn底层状态转MIT底层状态
 * 
 * @param msg qi server发来的底层状态
 */
void MitToQisnConverter::QisnLowStateCallback(const qi::msg::LowState::SharedPtr msg)
{
    mit_msgs::msg::MITLowState mit_state;
    const int qisn_state_len = static_cast<int>(msg->motors.size());
    if (expected_motor_count_ == 0 && qisn_state_len > 0) {
        expected_motor_count_ = static_cast<size_t>(qisn_state_len);
        RCLCPP_INFO(
            this->get_logger(), "body motor count from lowstate: %zu (dexterous hands excluded)",
            expected_motor_count_);
    } else if (
        expected_motor_count_ > 0 &&
        static_cast<size_t>(qisn_state_len) != expected_motor_count_)
    {
        RCLCPP_WARN_THROTTLE(
            this->get_logger(), *this->get_clock(), 5000,
            "lowstate has %d motors, expected %zu body motors",
            qisn_state_len, expected_motor_count_);
    }
    mit_state.joint_states.position.resize(qisn_state_len);
    mit_state.joint_states.velocity.resize(qisn_state_len);
    mit_state.joint_states.effort.resize(qisn_state_len);
    // 赋值关节状态
    for (int i = 0; i < qisn_state_len; i ++){
        mit_state.joint_states.position[i] = msg->motors[i].q;
        mit_state.joint_states.velocity[i] = msg->motors[i].dq;
        mit_state.joint_states.effort[i] = msg->motors[i].tau_est;
    }
    // 赋值imu状态
    if (msg->imus.empty()) {
        RCLCPP_WARN_THROTTLE(
            this->get_logger(), *this->get_clock(), 5000,
            "lowstate has no IMU entries; /human_lower_state.imu will be zero");
    } else {
        float qw = msg->imus[0].quaternion[0];
        float qx = msg->imus[0].quaternion[1];
        float qy = msg->imus[0].quaternion[2];
        float qz = msg->imus[0].quaternion[3];
        const float quat_norm = qw * qw + qx * qx + qy * qy + qz * qz;
        if (quat_norm < 1e-6f) {
            rpy_to_quaternion(
                msg->imus[0].rpy[0], msg->imus[0].rpy[1], msg->imus[0].rpy[2],
                qw, qx, qy, qz);
            RCLCPP_WARN_THROTTLE(
                this->get_logger(), *this->get_clock(), 5000,
                "IMU quaternion is zero; falling back to rpy -> quaternion");
        }
        mit_state.imu.orientation.w = qw;
        mit_state.imu.orientation.x = qx;
        mit_state.imu.orientation.y = qy;
        mit_state.imu.orientation.z = qz;

        mit_state.imu.angular_velocity.x = msg->imus[0].gyroscope[0];
        mit_state.imu.angular_velocity.y = msg->imus[0].gyroscope[1];
        mit_state.imu.angular_velocity.z = msg->imus[0].gyroscope[2];

        mit_state.imu.linear_acceleration.x = msg->imus[0].accelerometer[0];
        mit_state.imu.linear_acceleration.y = msg->imus[0].accelerometer[1];
        mit_state.imu.linear_acceleration.z = msg->imus[0].accelerometer[2];
    }
    // 发布mit消息
    mit_state_pub_->publish(mit_state);
}

int main(int argc, char ** argv)
{
// 初始化 ROS 2 系统
  rclcpp::init(argc, argv);

  // 创建一个 MitToQisnConverter 类型的节点对象
  auto mit_to_qisn_converter_node = std::make_shared<MitToQisnConverter>();

  // 保持节点运行
  rclcpp::spin(mit_to_qisn_converter_node);

  // 清理并关闭 ROS 2 系统
  rclcpp::shutdown();
  return 0;
}
