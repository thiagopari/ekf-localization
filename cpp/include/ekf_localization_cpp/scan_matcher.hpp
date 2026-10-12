#pragma once

#include <Eigen/Dense>

#include <optional>
#include <random>
#include <string>
#include <vector>

namespace ekf_localization_cpp
{

using Points = Eigen::Matrix2Xd;  // one 2D point per column

// Port of ekf_localization/scan_matcher.py::ScanMatcher: multi-resolution trimmed ICP,
// scan-to-previous-scan, with PCL's KdTreeFLANN in place of scipy's cKDTree.
// Not ported: match()'s initial_guess argument (ekf_node.py never passes it).
class ScanMatcher
{
public:
  struct Result
  {
    Eigen::Vector3d delta_pose = Eigen::Vector3d::Zero();  // [x, y, theta]
    double rmse = 0.0;
    double fitness = 0.0;
    int num_correspondences = 0;
    bool success = false;
  };

  int max_iterations = 50;
  double convergence_threshold = 1e-6;
  double max_correspondence_dist = 0.5;
  double trim_ratio = 0.8;
  int min_points = 50;
  std::vector<double> voxel_sizes{0.1, 0.05, 0.02};  // coarse to fine
  std::optional<Points> prev_scan;

  static Points laserscan_to_points(
    const std::vector<float> & ranges, double angle_min, double angle_max,
    double range_min = 0.1, double range_max = 30.0);

  Result match(const Points & current_scan);

  Eigen::Matrix3d estimate_covariance(
    double fitness, double rmse, int num_correspondences, double base_variance = 0.01) const;

  // Returns "VALID" or the rejection reason, as the Python tuple's second element.
  std::string validate_match(
    const Eigen::Vector3d & delta_pose, double fitness, double rmse, int num_correspondences,
    double dt, double max_velocity = 2.0, double max_angular_velocity = 2.0) const;

private:
  std::mt19937 rng_{std::random_device{}()};  // final-refinement subsample (np.random.choice)
};

}  // namespace ekf_localization_cpp
