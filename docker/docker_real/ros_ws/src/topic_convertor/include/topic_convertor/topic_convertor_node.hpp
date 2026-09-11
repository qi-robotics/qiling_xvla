#include <rclcpp/rclcpp.hpp>
#include "mit_msgs/msg/mit_joint_commands.hpp"
#include "mit_msgs/msg/mit_low_state.hpp"
#include "qi/msg/low_cmd.hpp"
#include "qi/msg/low_state.hpp"

class MitToQisnConverter : public rclcpp::Node
{
public:
    MitToQisnConverter();

private:

    void MitJointCommandsCallback(const mit_msgs::msg::MITJointCommands::SharedPtr msg);
    void QisnLowStateCallback(const qi::msg::LowState::SharedPtr msg);

    rclcpp::Subscription<mit_msgs::msg::MITJointCommands>::SharedPtr mit_cmd_sub_;
    rclcpp::Publisher<mit_msgs::msg::MITLowState>::SharedPtr mit_state_pub_;

    rclcpp::Subscription<qi::msg::LowState>::SharedPtr qisn_state_sub_;
    rclcpp::Publisher<qi::msg::LowCmd>::SharedPtr qisn_cmd_pub_;

    size_t expected_motor_count_{0};
    bool strict_command_size_{true};
    bool enable_state_bridge_{true};
    bool enable_command_bridge_{true};
};
