#pragma once

#include <Eigen/Dense>

namespace ekf_localization_cpp
{

// (angle + pi) % (2 pi) - pi with Python's floored modulo, so results match ekf.py bit for bit.
double normalize_angle(double angle);

// Port of ekf_localization/ekf.py::EKFLocalization. State [x, y, theta].
// Members are public like the Python attributes (ekf_node.py swaps R_gps at runtime).
struct EKF
{
  Eigen::Vector3d x = Eigen::Vector3d::Zero();
  Eigen::Matrix3d P = Eigen::Vector3d(0.1, 0.1, 0.1).asDiagonal();
  Eigen::Matrix3d Q = Eigen::Vector3d(0.02 * 0.02, 0.02 * 0.02, 0.01 * 0.01).asDiagonal();
  Eigen::Matrix2d R_gps = Eigen::Vector2d(0.5 * 0.5, 0.5 * 0.5).asDiagonal();  // 50 cm
  double R_imu = 0.02 * 0.02;                                                    // ~1 degree

  // Differential-drive midpoint motion model. No dt guard, same as ekf.py.
  void predict(double v, double omega, double dt);
  void update_gps(const Eigen::Vector2d & z);
  void update_imu(double z);
};

}  // namespace ekf_localization_cpp
