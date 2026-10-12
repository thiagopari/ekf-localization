#pragma once

#include <Eigen/Dense>

namespace ekf_localization_cpp
{

// Port of ekf_localization/ukf.py::UKFLocalization. State [x, y, theta], 2n+1 = 7 sigma points.
struct UKF
{
  static constexpr int n = 3;
  static constexpr double alpha = 1e-3;
  static constexpr double beta = 2.0;
  static constexpr double kappa = 0.0;
  static constexpr double lambda = alpha * alpha * (n + kappa) - n;

  Eigen::Vector3d x = Eigen::Vector3d::Zero();
  Eigen::Matrix3d P = Eigen::Vector3d(0.1, 0.1, 0.1).asDiagonal();
  Eigen::Matrix3d Q = Eigen::Vector3d(0.02 * 0.02, 0.02 * 0.02, 0.01 * 0.01).asDiagonal();
  Eigen::Matrix2d R_gps = Eigen::Vector2d(0.3 * 0.3, 0.3 * 0.3).asDiagonal();  // ukf.py: 30 cm
  double R_imu = 0.02 * 0.02;

  Eigen::Matrix<double, 2 * n + 1, 1> Wm, Wc;

  UKF();
  // Skips the step unless 0 < dt <= 1, same as ukf.py.
  void predict(double v, double omega, double dt);
  void update_gps(const Eigen::Vector2d & z);
  void update_imu(double z);
};

}  // namespace ekf_localization_cpp
