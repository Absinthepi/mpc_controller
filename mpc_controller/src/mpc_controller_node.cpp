#include <rclcpp/rclcpp.hpp>
#include <ackermann_msgs/msg/ackermann_drive_stamped.hpp>
#include <nav_msgs/msg/odometry.hpp>
#include <nav_msgs/msg/path.hpp>
#include <tf2/LinearMath/Quaternion.h>
#include <tf2/LinearMath/Matrix3x3.h>
#include <Eigen/Dense>
#include <deque>
#include <mutex>

#include <cppad/cppad.hpp>
#include <cppad/ipopt/solve.hpp>

#include "mpc_controller/helpers.h"
#include "mpc_controller/MPC.h"

class MPCController : public rclcpp::Node {
public:
    MPCController() : Node("mpc_controller") {
        // 声明参数
        this->declare_parameter("latency_ms", 100.0);
        this->declare_parameter("ref_v", 40.0);
        this->declare_parameter("poly_order", 2);
        this->declare_parameter("w_cte", 2000.0);
        this->declare_parameter("w_epsi", 2000.0);
        this->declare_parameter("w_v", 1.0);
        this->declare_parameter("w_delta", 5.0);
        this->declare_parameter("w_a", 5.0);
        this->declare_parameter("w_ddelta", 200.0);
        this->declare_parameter("w_da", 10.0);

        // 订阅
        path_sub_ = this->create_subscription<nav_msgs::msg::Path>(
            "/reference_path", 10,
            std::bind(&MPCController::pathCallback, this, std::placeholders::_1));
        odom_sub_ = this->create_subscription<nav_msgs::msg::Odometry>(
            "/odom", 10,
            std::bind(&MPCController::odomCallback, this, std::placeholders::_1));

        // 发布
        cmd_pub_ = this->create_publisher<ackermann_msgs::msg::AckermannDriveStamped>(
            "/control_cmd", 10);

        mpc_ = std::make_unique<MPC>();

        RCLCPP_INFO(this->get_logger(), "MPC controller node started.");
    }

private:
    struct State {
        rclcpp::Time t;
        double x, y, yaw, v;
    };

    void pathCallback(const nav_msgs::msg::Path::SharedPtr msg) {
        std::lock_guard<std::mutex> lock(mtx_);
        global_path_x_.clear();
        global_path_y_.clear();
        for (const auto& pose : msg->poses) {
            global_path_x_.push_back(pose.pose.position.x);
            global_path_y_.push_back(pose.pose.position.y);
        }
        path_received_ = true;
    }

    void odomCallback(const nav_msgs::msg::Odometry::SharedPtr msg) {
        if (!path_received_) {
            RCLCPP_WARN_THROTTLE(this->get_logger(), *this->get_clock(), 2000,
                                 "No reference path received yet.");
            return;
        }

        double latency_ms = this->get_parameter("latency_ms").as_double();
        double ref_v = this->get_parameter("ref_v").as_double();
        int poly_order = this->get_parameter("poly_order").as_int();

        double px = msg->pose.pose.position.x;
        double py = msg->pose.pose.position.y;
        double v = msg->twist.twist.linear.x;

        tf2::Quaternion q(
            msg->pose.pose.orientation.x,
            msg->pose.pose.orientation.y,
            msg->pose.pose.orientation.z,
            msg->pose.pose.orientation.w);
        double roll, pitch, yaw;
        tf2::Matrix3x3(q).getRPY(roll, pitch, yaw);

        rclcpp::Time now = this->get_clock()->now();
        {
            std::lock_guard<std::mutex> lock(mtx_);
            state_history_.push_back({now, px, py, yaw, v});
            while (!state_history_.empty() &&
                   (now - state_history_.front().t).seconds() > 1.0) {
                state_history_.pop_front();
            }
        }

        rclcpp::Time target_time = now - rclcpp::Duration::from_seconds(latency_ms / 1000.0);
        State delayed_state;
        bool found = false;
        {
            std::lock_guard<std::mutex> lock(mtx_);
            for (auto it = state_history_.rbegin(); it != state_history_.rend(); ++it) {
                if (it->t <= target_time) {
                    delayed_state = *it;
                    found = true;
                    break;
                }
            }
            if (!found && !state_history_.empty()) {
                delayed_state = state_history_.front();
                found = true;
            }
        }

        if (!found) {
            RCLCPP_WARN(this->get_logger(), "No delayed state available.");
            return;
        }

        std::vector<double> path_x, path_y;
        {
            std::lock_guard<std::mutex> lock(mtx_);
            path_x = global_path_x_;
            path_y = global_path_y_;
        }

        Eigen::VectorXd x_vehicle(path_x.size());
        Eigen::VectorXd y_vehicle(path_x.size());
        for (size_t i = 0; i < path_x.size(); ++i) {
            auto local = globalToLocal(path_x[i], path_y[i],
                                       delayed_state.x, delayed_state.y, delayed_state.yaw);
            x_vehicle(i) = local.first;
            y_vehicle(i) = local.second;
        }

        auto coeffs = polyfit(x_vehicle, y_vehicle, poly_order);

        double cte = polyeval(coeffs, 0.0);
        double epsi = -std::atan(coeffs[1]);

        Eigen::VectorXd state(6);
        state << 0.0, 0.0, 0.0, delayed_state.v, cte, epsi;

        // 调用 MPC 求解器
        std::vector<double> solution = mpc_->Solve(state, coeffs);
        if (solution.empty()) {
            RCLCPP_WARN(this->get_logger(), "MPC solver failed.");
            return;
        }

        double steer = solution[0];
        double accel = solution[1];
        steer = std::max(-0.5, std::min(0.5, steer));
        accel = std::max(-1.0, std::min(1.0, accel));

        ackermann_msgs::msg::AckermannDriveStamped cmd;
        cmd.header.stamp = now;
        cmd.drive.steering_angle = steer;
        cmd.drive.speed = ref_v + accel * 0.1;
        cmd.drive.acceleration = accel;
        cmd_pub_->publish(cmd);
    }

    rclcpp::Subscription<nav_msgs::msg::Path>::SharedPtr path_sub_;
    rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr odom_sub_;
    rclcpp::Publisher<ackermann_msgs::msg::AckermannDriveStamped>::SharedPtr cmd_pub_;

    std::unique_ptr<MPC> mpc_;

    std::vector<double> global_path_x_, global_path_y_;
    bool path_received_ = false;
    std::deque<State> state_history_;
    std::mutex mtx_;
};

int main(int argc, char ** argv) {
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<MPCController>());
    rclcpp::shutdown();
    return 0;
}