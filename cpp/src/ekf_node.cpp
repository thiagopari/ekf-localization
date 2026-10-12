// rclcpp port of ekf_localization/ekf_node.py: same topics, parameters, rates and outputs.

#include <algorithm>
#include <cmath>
#include <memory>
#include <optional>
#include <random>
#include <string>
#include <vector>

#include "geometry_msgs/msg/pose_stamped.hpp"
#include "geometry_msgs/msg/transform_stamped.hpp"
#include "nav_msgs/msg/odometry.hpp"
#include "nav_msgs/msg/path.hpp"
#include "rclcpp/rclcpp.hpp"
#include "sensor_msgs/msg/imu.hpp"
#include "sensor_msgs/msg/laser_scan.hpp"
#include "std_msgs/msg/float64_multi_array.hpp"
#include "tf2_ros/transform_broadcaster.h"
#include "visualization_msgs/msg/marker.hpp"

#include "ekf_localization_cpp/ekf.hpp"
#include "ekf_localization_cpp/scan_matcher.hpp"
#include "ekf_localization_cpp/ukf.hpp"

using geometry_msgs::msg::PoseStamped;
using geometry_msgs::msg::TransformStamped;
using nav_msgs::msg::Odometry;
using nav_msgs::msg::Path;
using visualization_msgs::msg::Marker;

namespace
{
constexpr size_t kMaxPathPoses = 1000;
constexpr size_t kMaxErrors = 5000;

geometry_msgs::msg::Quaternion yaw_to_quaternion(double yaw)  // quaternion_from_euler(0, 0, yaw)
{
  geometry_msgs::msg::Quaternion q;
  q.z = std::sin(yaw / 2);
  q.w = std::cos(yaw / 2);
  return q;
}

double yaw_from_quaternion(const geometry_msgs::msg::Quaternion & q)  // euler_from_quaternion()[2]
{
  return std::atan2(2 * (q.w * q.z + q.x * q.y), q.w * q.w + q.x * q.x - q.y * q.y - q.z * q.z);
}

PoseStamped make_pose(const builtin_interfaces::msg::Time & stamp, double x, double y, double yaw)
{
  PoseStamped p;
  p.header.stamp = stamp;
  p.header.frame_id = "odom";
  p.pose.position.x = x;
  p.pose.position.y = y;
  p.pose.orientation = yaw_to_quaternion(yaw);
  return p;
}

template<typename T>
void push_capped(std::vector<T> & v, T item, size_t cap)
{
  v.push_back(std::move(item));
  if (v.size() > cap) {
    v.erase(v.begin(), v.end() - cap);
  }
}

double rms(const std::vector<double> & e)
{
  double sq = 0;
  for (const double x : e) {
    sq += x * x;
  }
  return std::sqrt(sq / e.size());
}
}  // namespace

class EKFNode : public rclcpp::Node
{
public:
  EKFNode()
  : Node("ekf_localization_node"),
    gps_rate_(declare_parameter("gps_rate", 1.0)),
    add_odom_noise_(declare_parameter("add_odom_noise", true)),
    odom_noise_std_(declare_parameter("odom_noise_std", 0.02)),
    gps_outage_start_(declare_parameter("gps_outage_start", 30.0)),
    gps_outage_duration_(declare_parameter("gps_outage_duration", 15.0)),
    enable_gps_outage_(declare_parameter("enable_gps_outage", false)),
    enable_scan_matching_(declare_parameter("enable_scan_matching", true)),
    scan_match_min_interval_(declare_parameter("scan_match_min_interval", 0.1)),
    gps_period_(1.0 / gps_rate_),
    rng_(std::random_device{}())
  {
    // ekf_node.py carries its own EKF copy with 0.3 m GPS noise (ekf.py uses 0.5 m) and a
    // 0 < dt <= 1 guard in predict (applied in odom_callback below).
    ekf_.R_gps = Eigen::Vector2d(0.3 * 0.3, 0.3 * 0.3).asDiagonal();

    tf_broadcaster_ = std::make_unique<tf2_ros::TransformBroadcaster>(*this);

    pose_pub_ = create_publisher<PoseStamped>("ekf_pose", 10);
    ekf_path_pub_ = create_publisher<Path>("ekf_path", 10);
    ukf_path_pub_ = create_publisher<Path>("ukf_path", 10);
    odom_path_pub_ = create_publisher<Path>("odom_path", 10);
    dr_path_pub_ = create_publisher<Path>("dr_path", 10);
    cov_marker_pub_ = create_publisher<Marker>("ekf_covariance", 10);
    scan_path_pub_ = create_publisher<Path>("scan_path", 10);
    metrics_pub_ = create_publisher<std_msgs::msg::Float64MultiArray>("ekf_metrics", 10);
    for (Path * p : {&ekf_path_, &ukf_path_, &odom_path_, &dr_path_, &scan_path_}) {
      p->header.frame_id = "odom";
    }

    metrics_timer_ = create_timer(std::chrono::seconds(1), [this] {publish_metrics();});
    odom_sub_ = create_subscription<Odometry>(
      "odom", 10, [this](const Odometry & msg) {odom_callback(msg);});
    imu_sub_ = create_subscription<sensor_msgs::msg::Imu>(
      "imu", 10, [this](const sensor_msgs::msg::Imu & msg) {
        const double yaw = yaw_from_quaternion(msg.orientation);
        ekf_.update_imu(yaw);
        ukf_.update_imu(yaw);
      });
    scan_sub_ = create_subscription<sensor_msgs::msg::LaserScan>(
      "scan", 10, [this](const sensor_msgs::msg::LaserScan & msg) {scan_callback(msg);});

    RCLCPP_INFO(get_logger(), "EKF Localization Node started (C++)");
    RCLCPP_INFO(get_logger(), "  GPS rate: %.1f Hz", gps_rate_);
    RCLCPP_INFO(get_logger(), "  Odom noise injection: %s", add_odom_noise_ ? "True" : "False");
    if (enable_gps_outage_) {
      RCLCPP_INFO(
        get_logger(), "  GPS outage enabled: starts at %.1fs, duration %.1fs",
        gps_outage_start_, gps_outage_duration_);
    }
    RCLCPP_INFO(
      get_logger(), "  Scan matching: %s", enable_scan_matching_ ? "enabled" : "disabled");
  }

private:
  void odom_callback(const Odometry & msg)
  {
    const double current_time = rclcpp::Time(msg.header.stamp).seconds();
    broadcast_odom_tf(msg);

    double v = msg.twist.twist.linear.x;
    double omega = msg.twist.twist.angular.z;
    if (add_odom_noise_) {
      v += odom_noise_std_ * normal_(rng_);
      omega += odom_noise_std_ * 0.5 * normal_(rng_);
    }

    if (last_odom_time_) {
      const double dt = current_time - *last_odom_time_;
      if (dt > 0 && dt <= 1.0) {
        ekf_.predict(v, omega, dt);
      }
      ukf_.predict(v, omega, dt);  // guards dt itself
      if (dt > 0 && dt < 1.0) {    // dead reckoning: same noisy velocities, no corrections
        const double theta_mid = dr_state_(2) + omega * dt / 2;
        dr_state_(0) += v * dt * std::cos(theta_mid);
        dr_state_(1) += v * dt * std::sin(theta_mid);
        dr_state_(2) += omega * dt;
      }
    }
    last_odom_time_ = current_time;

    if (!last_gps_time_) {
      last_gps_time_ = current_time;
    }
    if (!start_time_) {
      start_time_ = current_time;
    }
    const double elapsed = current_time - *start_time_;

    bool gps_available = true;
    if (enable_gps_outage_) {
      const double outage_end = gps_outage_start_ + gps_outage_duration_;
      if (gps_outage_start_ <= elapsed && elapsed < outage_end) {
        gps_available = false;
        if (!in_gps_outage_) {
          in_gps_outage_ = true;
          RCLCPP_WARN(get_logger(), "GPS OUTAGE STARTED at %.1fs - relying on IMU only", elapsed);
        }
      } else if (in_gps_outage_) {
        in_gps_outage_ = false;
        RCLCPP_WARN(get_logger(), "GPS RESTORED at %.1fs", elapsed);
      }
    }

    if (gps_available && current_time - *last_gps_time_ >= gps_period_) {
      // Simulated GPS: ground-truth odom position + 0.3 m noise
      const Eigen::Vector2d z(
        msg.pose.pose.position.x + 0.3 * normal_(rng_),
        msg.pose.pose.position.y + 0.3 * normal_(rng_));
      ekf_.update_gps(z);
      ukf_.update_gps(z);
      last_gps_time_ = current_time;
      RCLCPP_DEBUG(get_logger(), "GPS update: (%.2f, %.2f)", z.x(), z.y());
    }

    publish_estimate(msg.header.stamp);
    publish_dr_path(msg.header.stamp);
  }

  void publish_path(
    Path & path, const rclcpp::Publisher<Path>::SharedPtr & pub, PoseStamped pose,
    const builtin_interfaces::msg::Time & stamp)
  {
    push_capped(path.poses, std::move(pose), kMaxPathPoses);
    path.header.stamp = stamp;
    pub->publish(path);
  }

  double error_to_truth(double x, double y) const
  {
    return std::hypot(x - ground_truth_pos_.x(), y - ground_truth_pos_.y());
  }

  void scan_callback(const sensor_msgs::msg::LaserScan & msg)
  {
    if (!enable_scan_matching_) {
      return;
    }
    const double current_time = rclcpp::Time(msg.header.stamp).seconds();
    double dt = 0.1;  // default for the first scan
    if (last_scan_time_) {
      dt = current_time - *last_scan_time_;
      if (dt < scan_match_min_interval_) {
        return;
      }
    }

    const auto points = ekf_localization_cpp::ScanMatcher::laserscan_to_points(
      msg.ranges, msg.angle_min, msg.angle_max, msg.range_min, msg.range_max);
    if (points.cols() < 50) {
      return;
    }

    const auto r = scan_matcher_.match(points);
    ++scan_match_count_;
    if (r.success) {
      ++scan_match_success_count_;
      const Eigen::Vector3d & d = r.delta_pose;
      const std::string reason =
        scan_matcher_.validate_match(d, r.fitness, r.rmse, r.num_correspondences, dt);
      if (reason == "VALID") {
        RCLCPP_INFO(
          get_logger(), "VALID match: delta=[%.3f, %.3f, %.3f], fitness=%.2f, rmse=%.4f, corr=%d",
          d(0), d(1), d(2), r.fitness, r.rmse, r.num_correspondences);

        // Delta rotated by the scan-only heading, as in ekf_node.py.
        const double c = std::cos(scan_match_pose_(2)), s = std::sin(scan_match_pose_(2));
        const Eigen::Vector2d step(c * d(0) - s * d(1), s * d(0) + c * d(1));
        scan_match_pose_.head<2>() += step;
        scan_match_pose_(2) += d(2);
        RCLCPP_INFO(
          get_logger(), "Accumulated scan_pose: [%.3f, %.3f, %.3f]",
          scan_match_pose_(0), scan_match_pose_(1), scan_match_pose_(2));

        // Position measurement = current EKF position + that delta, with ICP-derived noise,
        // applied through the GPS update with R_gps swapped in and restored.
        const Eigen::Matrix3d R_scan =
          scan_matcher_.estimate_covariance(r.fitness, r.rmse, r.num_correspondences);
        const Eigen::Vector2d z = ekf_.x.head<2>() + step;
        const Eigen::Matrix2d saved_R_gps = ekf_.R_gps;
        ekf_.R_gps = ukf_.R_gps = R_scan.topLeftCorner<2, 2>();
        ekf_.update_gps(z);
        ukf_.update_gps(z);
        ekf_.R_gps = ukf_.R_gps = saved_R_gps;

        publish_path(
          scan_path_, scan_path_pub_,
          make_pose(msg.header.stamp, scan_match_pose_(0), scan_match_pose_(1), scan_match_pose_(2)),
          msg.header.stamp);
        push_capped(scan_errors_, error_to_truth(scan_match_pose_(0), scan_match_pose_(1)), kMaxErrors);
      } else {
        RCLCPP_WARN(
          get_logger(), "REJECTED match: %s, delta=[%.3f, %.3f, %.3f], fitness=%.2f",
          reason.c_str(), d(0), d(1), d(2), r.fitness);
      }
    }
    last_scan_time_ = current_time;
  }

  void broadcast_odom_tf(const Odometry & msg)
  {
    TransformStamped t;
    t.header.stamp = msg.header.stamp;
    t.header.frame_id = "odom";
    t.child_frame_id = "base_footprint";
    t.transform.translation.x = msg.pose.pose.position.x;
    t.transform.translation.y = msg.pose.pose.position.y;
    t.transform.translation.z = msg.pose.pose.position.z;
    t.transform.rotation = msg.pose.pose.orientation;
    tf_broadcaster_->sendTransform(t);

    ground_truth_pos_ = {msg.pose.pose.position.x, msg.pose.pose.position.y};

    PoseStamped odom_pose;
    odom_pose.header = msg.header;
    odom_pose.pose = msg.pose.pose;
    publish_path(odom_path_, odom_path_pub_, odom_pose, msg.header.stamp);
  }

  void publish_dr_path(const builtin_interfaces::msg::Time & stamp)
  {
    publish_path(
      dr_path_, dr_path_pub_, make_pose(stamp, dr_state_(0), dr_state_(1), dr_state_(2)), stamp);
    push_capped(dr_errors_, error_to_truth(dr_state_(0), dr_state_(1)), kMaxErrors);
  }

  void publish_estimate(const builtin_interfaces::msg::Time & stamp)
  {
    const Eigen::Vector3d s = ekf_.x;
    const PoseStamped pose = make_pose(stamp, s(0), s(1), s(2));
    pose_pub_->publish(pose);
    publish_path(ekf_path_, ekf_path_pub_, pose, stamp);
    push_capped(ekf_errors_, error_to_truth(s(0), s(1)), kMaxErrors);

    const Eigen::Vector3d u = ukf_.x;
    publish_path(ukf_path_, ukf_path_pub_, make_pose(stamp, u(0), u(1), u(2)), stamp);
    push_capped(ukf_errors_, error_to_truth(u(0), u(1)), kMaxErrors);

    publish_covariance_ellipse(stamp, s);

    TransformStamped t;
    t.header.stamp = stamp;
    t.header.frame_id = "odom";
    t.child_frame_id = "base_footprint_ekf";
    t.transform.translation.x = s(0);
    t.transform.translation.y = s(1);
    t.transform.rotation = pose.pose.orientation;
    tf_broadcaster_->sendTransform(t);
  }

  void publish_covariance_ellipse(
    const builtin_interfaces::msg::Time & stamp, const Eigen::Vector3d & state)
  {
    // 3-sigma ellipse of the 2x2 position covariance; eigenvalues come back ascending.
    const Eigen::SelfAdjointEigenSolver<Eigen::Matrix2d> eig(ekf_.P.topLeftCorner<2, 2>());
    const Eigen::Vector2d major = eig.eigenvectors().col(1);
    const double scale = 3.0;

    Marker m;
    m.header.stamp = stamp;
    m.header.frame_id = "odom";
    m.ns = "ekf_covariance";
    m.id = 0;
    m.type = Marker::CYLINDER;
    m.action = Marker::ADD;
    m.pose.position.x = state(0);
    m.pose.position.y = state(1);
    m.pose.position.z = 0.01;
    m.pose.orientation = yaw_to_quaternion(std::atan2(major.y(), major.x()));
    m.scale.x = 2 * scale * std::sqrt(std::abs(eig.eigenvalues()(1)));
    m.scale.y = 2 * scale * std::sqrt(std::abs(eig.eigenvalues()(0)));
    m.scale.z = 0.02;
    m.color.r = 0.0;
    m.color.g = 0.5;
    m.color.b = 1.0;
    m.color.a = 0.4;
    m.lifetime.sec = 0;
    m.lifetime.nanosec = 100000000;
    cov_marker_pub_->publish(m);
  }

  void publish_metrics()
  {
    if (ekf_errors_.size() < 10 || dr_errors_.size() < 10 || ukf_errors_.size() < 10) {
      return;
    }
    const double ekf_rms = rms(ekf_errors_), ukf_rms = rms(ukf_errors_), dr_rms = rms(dr_errors_);
    const double ekf_imp = dr_rms > 0 ? (1 - ekf_rms / dr_rms) * 100 : 0;
    const double ukf_imp = dr_rms > 0 ? (1 - ukf_rms / dr_rms) * 100 : 0;
    const auto max_of = [](const std::vector<double> & e) {
        return *std::max_element(e.begin(), e.end());
      };

    const double scan_rms = scan_errors_.size() > 10 ? rms(scan_errors_) : 0.0;
    const double scan_rate =
      scan_match_count_ > 0 ? 100.0 * scan_match_success_count_ / scan_match_count_ : 0.0;

    std_msgs::msg::Float64MultiArray msg;  // same 13-value layout as ekf_node.py
    msg.data = {
      ekf_rms, max_of(ekf_errors_), ekf_errors_.back(),
      ukf_rms, max_of(ukf_errors_), ukf_errors_.back(),
      dr_rms, max_of(dr_errors_), dr_errors_.back(),
      ekf_imp, ukf_imp, scan_rms, scan_rate};
    metrics_pub_->publish(msg);

    const std::string scan_status =
      enable_scan_matching_ ? "Scan: " + std::to_string(std::lround(scan_rate)) + "%" : "Scan: OFF";
    RCLCPP_INFO(
      get_logger(), "%s | %s | EKF: %.3fm | UKF: %.3fm | DR: %.3fm | Imp: %.1f%%",
      in_gps_outage_ ? "GPS: OUT" : "GPS: OK", scan_status.c_str(), ekf_rms, ukf_rms, dr_rms,
      ekf_imp);
  }

  const double gps_rate_;
  const bool add_odom_noise_;
  const double odom_noise_std_;
  const double gps_outage_start_;
  const double gps_outage_duration_;
  const bool enable_gps_outage_;
  const bool enable_scan_matching_;
  const double scan_match_min_interval_;
  const double gps_period_;

  ekf_localization_cpp::EKF ekf_;
  ekf_localization_cpp::UKF ukf_;
  ekf_localization_cpp::ScanMatcher scan_matcher_;  // Python's ctor args equal the defaults
  Eigen::Vector3d scan_match_pose_ = Eigen::Vector3d::Zero();
  std::optional<double> last_scan_time_;
  int scan_match_count_ = 0, scan_match_success_count_ = 0;
  Eigen::Vector3d dr_state_ = Eigen::Vector3d::Zero();
  Eigen::Vector2d ground_truth_pos_ = Eigen::Vector2d::Zero();

  std::optional<double> last_odom_time_, last_gps_time_, start_time_;
  bool in_gps_outage_ = false;

  std::mt19937 rng_;
  std::normal_distribution<double> normal_;

  Path ekf_path_, ukf_path_, odom_path_, dr_path_, scan_path_;
  std::vector<double> ekf_errors_, ukf_errors_, dr_errors_, scan_errors_;

  std::unique_ptr<tf2_ros::TransformBroadcaster> tf_broadcaster_;
  rclcpp::Publisher<PoseStamped>::SharedPtr pose_pub_;
  rclcpp::Publisher<Path>::SharedPtr ekf_path_pub_, ukf_path_pub_, odom_path_pub_, dr_path_pub_,
    scan_path_pub_;
  rclcpp::Publisher<Marker>::SharedPtr cov_marker_pub_;
  rclcpp::Publisher<std_msgs::msg::Float64MultiArray>::SharedPtr metrics_pub_;
  rclcpp::TimerBase::SharedPtr metrics_timer_;
  rclcpp::Subscription<Odometry>::SharedPtr odom_sub_;
  rclcpp::Subscription<sensor_msgs::msg::Imu>::SharedPtr imu_sub_;
  rclcpp::Subscription<sensor_msgs::msg::LaserScan>::SharedPtr scan_sub_;
};

int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<EKFNode>());
  rclcpp::shutdown();
}
