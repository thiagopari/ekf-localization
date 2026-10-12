#include "ekf_localization_cpp/ukf.hpp"

#include <cmath>

#include "ekf_localization_cpp/ekf.hpp"  // normalize_angle

// Loops and accumulation order follow ukf.py on purpose: W0 is about -1e6 (alpha = 1e-3),
// so reordering the weighted sums changes results far above 1 ulp.

namespace ekf_localization_cpp
{

namespace
{
constexpr int N = UKF::n;
constexpr int NS = 2 * N + 1;
using Sigma = Eigen::Matrix<double, NS, N>;

// np.linalg.inv of a 2x2 in reference-LAPACK dgesv order (LU, partial pivoting). Eigen's
// cofactor inverse differs in the last bit, which the -1e6 weights blow up to ~2.5e-9.
Eigen::Matrix2d inverse_2x2(const Eigen::Matrix2d & A)
{
  const int p = std::abs(A(1, 0)) > std::abs(A(0, 0)) ? 1 : 0;  // pivot row
  const int q = 1 - p;
  const double u11 = A(p, 0), u12 = A(p, 1);
  const double l = (1.0 / u11) * A(q, 0);
  const double u22 = A(q, 1) - u12 * l;
  Eigen::Matrix2d X;
  for (int j = 0; j < 2; ++j) {
    const double y1 = p == j ? 1.0 : 0.0;
    const double y2 = (q == j ? 1.0 : 0.0) - y1 * l;
    X(1, j) = y2 / u22;
    X(0, j) = (y1 - X(1, j) * u12) / u11;
  }
  return X;
}

// Columns of the square root S (S S^T = (n + lambda) P), for both the Cholesky and the
// eigendecomposition fallback, same as ukf.py.
Sigma sigma_points(const Eigen::Vector3d & x, const Eigen::Matrix3d & P)
{
  Eigen::Matrix3d sqrt_P;
  Eigen::LLT<Eigen::Matrix3d> llt((N + UKF::lambda) * P);
  if (llt.info() == Eigen::Success) {
    sqrt_P = llt.matrixL();
  } else {
    Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> eig(P);
    const Eigen::Vector3d s = ((N + UKF::lambda) * eig.eigenvalues().cwiseMax(1e-10)).cwiseSqrt();
    sqrt_P = eig.eigenvectors() * s.asDiagonal();
  }

  Sigma sp;
  sp.row(0) = x.transpose();
  for (int i = 0; i < N; ++i) {
    sp.row(i + 1) = (x + sqrt_P.col(i)).transpose();
    sp.row(N + i + 1) = (x - sqrt_P.col(i)).transpose();
  }
  return sp;
}

Eigen::Vector3d motion_model(const Eigen::Vector3d & x, double v, double omega, double dt)
{
  const double theta_mid = x(2) + omega * dt / 2;
  return {
    x(0) + v * dt * std::cos(theta_mid),
    x(1) + v * dt * std::sin(theta_mid),
    normalize_angle(x(2) + omega * dt)};
}
}  // namespace

UKF::UKF()
{
  Wm(0) = lambda / (n + lambda);
  Wc(0) = lambda / (n + lambda) + (1 - alpha * alpha + beta);
  for (int i = 1; i < NS; ++i) {
    Wm(i) = 1 / (2 * (n + lambda));
    Wc(i) = 1 / (2 * (n + lambda));
  }
}

void UKF::predict(double v, double omega, double dt)
{
  if (dt <= 0 || dt > 1.0) {
    return;
  }

  const Sigma sp = sigma_points(x, P);
  Sigma sp_pred;
  for (int i = 0; i < NS; ++i) {
    sp_pred.row(i) = motion_model(sp.row(i).transpose(), v, omega, dt).transpose();
  }

  Eigen::Vector3d x_pred = Eigen::Vector3d::Zero();
  for (int i = 0; i < NS; ++i) {
    x_pred += Wm(i) * sp_pred.row(i).transpose();
  }
  x_pred(2) = normalize_angle(x_pred(2));

  Eigen::Matrix3d P_pred = Eigen::Matrix3d::Zero();
  for (int i = 0; i < NS; ++i) {
    Eigen::Vector3d diff = sp_pred.row(i).transpose() - x_pred;
    diff(2) = normalize_angle(diff(2));
    const Eigen::Matrix3d outer = diff * diff.transpose();  // np.outer, then scale
    P_pred += Wc(i) * outer;
  }
  P_pred += Q;

  x = x_pred;
  P = P_pred;
}

void UKF::update_gps(const Eigen::Vector2d & z)
{
  const Sigma sp = sigma_points(x, P);
  const Eigen::Matrix<double, NS, 2> Z = sp.leftCols<2>();  // h(x) = [x, y]

  Eigen::Vector2d z_pred = Eigen::Vector2d::Zero();
  for (int i = 0; i < NS; ++i) {
    z_pred += Wm(i) * Z.row(i).transpose();
  }

  Eigen::Matrix2d Pzz = Eigen::Matrix2d::Zero();
  for (int i = 0; i < NS; ++i) {
    const Eigen::Vector2d diff = Z.row(i).transpose() - z_pred;
    const Eigen::Matrix2d outer = diff * diff.transpose();
    Pzz += Wc(i) * outer;
  }
  Pzz += R_gps;

  Eigen::Matrix<double, 3, 2> Pxz = Eigen::Matrix<double, 3, 2>::Zero();
  for (int i = 0; i < NS; ++i) {
    Eigen::Vector3d x_diff = sp.row(i).transpose() - x;
    x_diff(2) = normalize_angle(x_diff(2));
    const Eigen::Vector2d z_diff = Z.row(i).transpose() - z_pred;
    const Eigen::Matrix<double, 3, 2> outer = x_diff * z_diff.transpose();
    Pxz += Wc(i) * outer;
  }

  const Eigen::Matrix<double, 3, 2> K = Pxz * inverse_2x2(Pzz);
  x += K * (z - z_pred);
  x(2) = normalize_angle(x(2));
  const Eigen::Matrix<double, 3, 2> KPzz = K * Pzz;
  P -= KPzz * K.transpose();
}

void UKF::update_imu(double z)
{
  const Sigma sp = sigma_points(x, P);
  const Eigen::Matrix<double, NS, 1> Z = sp.col(2);  // h(x) = theta

  double z_pred = 0;
  for (int i = 0; i < NS; ++i) {
    z_pred += Wm(i) * Z(i);
  }
  z_pred = normalize_angle(z_pred);

  double Pzz = 0;
  for (int i = 0; i < NS; ++i) {
    const double diff = normalize_angle(Z(i) - z_pred);
    Pzz += Wc(i) * (diff * diff);
  }
  Pzz += R_imu;

  Eigen::Vector3d Pxz = Eigen::Vector3d::Zero();
  for (int i = 0; i < NS; ++i) {
    Eigen::Vector3d x_diff = sp.row(i).transpose() - x;
    x_diff(2) = normalize_angle(x_diff(2));
    const double z_diff = normalize_angle(Z(i) - z_pred);
    const Eigen::Vector3d outer = x_diff * z_diff;
    Pxz += Wc(i) * outer;
  }

  const Eigen::Vector3d K = Pxz * (1.0 / Pzz);  // np.linalg.inv of a 1x1
  x += K * normalize_angle(z - z_pred);
  x(2) = normalize_angle(x(2));
  const Eigen::Vector3d KPzz = K * Pzz;
  P -= KPzz * K.transpose();
}

}  // namespace ekf_localization_cpp
