#include "ekf_localization_cpp/ekf.hpp"

#include <cmath>

namespace ekf_localization_cpp
{

namespace
{
const Eigen::Matrix<double, 2, 3> H_gps = (Eigen::Matrix<double, 2, 3>() << 1, 0, 0, 0, 1, 0).finished();
const Eigen::RowVector3d H_imu(0, 0, 1);
}  // namespace

double normalize_angle(double angle)
{
  double r = std::fmod(angle + M_PI, 2 * M_PI);
  if (r < 0) {
    r += 2 * M_PI;
  }
  return r - M_PI;
}

void EKF::predict(double v, double omega, double dt)
{
  const double theta_mid = x(2) + omega * dt / 2;

  x(0) += v * dt * std::cos(theta_mid);
  x(1) += v * dt * std::sin(theta_mid);
  x(2) = normalize_angle(x(2) + omega * dt);

  Eigen::Matrix3d F;
  F << 1, 0, -v * dt * std::sin(theta_mid),
       0, 1, v * dt * std::cos(theta_mid),
       0, 0, 1;
  P = F * P * F.transpose() + Q;
}

void EKF::update_gps(const Eigen::Vector2d & z)
{
  const Eigen::Vector2d y = z - H_gps * x;
  const Eigen::Matrix2d S = H_gps * P * H_gps.transpose() + R_gps;
  const Eigen::Matrix<double, 3, 2> K = P * H_gps.transpose() * S.inverse();

  x += K * y;
  x(2) = normalize_angle(x(2));
  // Simple form (I - KH)P, as in ekf.py (its comment says Joseph form; it is not).
  P = (Eigen::Matrix3d::Identity() - K * H_gps) * P;
}

void EKF::update_imu(double z)
{
  const double y = normalize_angle(z - (H_imu * x).value());
  const double S = (H_imu * P * H_imu.transpose()).value() + R_imu;
  const Eigen::Vector3d K = P * H_imu.transpose() * (1.0 / S);  // np.linalg.inv of a 1x1

  x += K * y;
  x(2) = normalize_angle(x(2));
  P = (Eigen::Matrix3d::Identity() - K * H_imu) * P;
}

}  // namespace ekf_localization_cpp
