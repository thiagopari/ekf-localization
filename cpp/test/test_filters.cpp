#include <gtest/gtest.h>

#include <cmath>
#include <random>

#include "ekf_localization_cpp/ekf.hpp"
#include "ekf_localization_cpp/ukf.hpp"

using ekf_localization_cpp::EKF;
using ekf_localization_cpp::normalize_angle;
using ekf_localization_cpp::UKF;

TEST(NormalizeAngle, MatchesPythonFlooredModulo) {
  EXPECT_DOUBLE_EQ(normalize_angle(0.0), 0.0);
  EXPECT_DOUBLE_EQ(normalize_angle(M_PI), -M_PI);  // Python: (pi + pi) % 2pi - pi == -pi
  EXPECT_NEAR(normalize_angle(3 * M_PI / 2), -M_PI / 2, 1e-15);
  EXPECT_NEAR(normalize_angle(-3 * M_PI / 2), M_PI / 2, 1e-15);
  EXPECT_NEAR(normalize_angle(10.0), 10.0 - 4 * M_PI, 1e-14);
}

TEST(EKF, StraightLineWithZeroOmega) {
  for (const double heading : {0.0, 0.7, -2.5}) {
    EKF ekf;
    ekf.x(2) = heading;
    for (int i = 0; i < 100; ++i) {
      ekf.predict(0.5, 0.0, 0.02);  // 1 m in 2 s
    }
    EXPECT_NEAR(ekf.x(0), std::cos(heading), 1e-12);
    EXPECT_NEAR(ekf.x(1), std::sin(heading), 1e-12);
    EXPECT_NEAR(ekf.x(2), heading, 1e-12);
  }
  // Heading uncertainty leaks into the lateral axis only: y variance grows past x variance.
  EKF ekf;
  for (int i = 0; i < 100; ++i) {
    ekf.predict(0.5, 0.0, 0.02);
  }
  EXPECT_GT(ekf.P(1, 1), ekf.P(0, 0));
}

template<typename Filter>
class FilterTest : public ::testing::Test {};
using Filters = ::testing::Types<EKF, UKF>;
TYPED_TEST_SUITE(FilterTest, Filters);

TYPED_TEST(FilterTest, CovarianceStaysSymmetricPSD) {
  TypeParam f;
  std::mt19937 rng(7);
  std::uniform_real_distribution<double> v(0.0, 0.5), w(-1.0, 1.0), noise(-0.3, 0.3);
  for (int k = 0; k < 3000; ++k) {
    f.predict(v(rng), w(rng), 0.02);
    f.update_imu(normalize_angle(f.x(2) + 0.1 * noise(rng)));
    if (k % 50 == 0) {
      f.update_gps(f.x.template head<2>() + Eigen::Vector2d(noise(rng), noise(rng)));
    }
    ASSERT_LT((f.P - f.P.transpose()).cwiseAbs().maxCoeff(), 1e-12) << "step " << k;
    const Eigen::Matrix3d sym = 0.5 * (f.P + f.P.transpose());
    ASSERT_GT(Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d>(sym).eigenvalues().minCoeff(), 0.0)
      << "step " << k;
  }
}

TYPED_TEST(FilterTest, UpdatesReduceTrace) {
  TypeParam f;
  for (int i = 0; i < 50; ++i) {
    f.predict(0.3, 0.2, 0.02);
  }
  double before = f.P.trace();
  f.update_gps(Eigen::Vector2d(0.3, 0.1));
  EXPECT_LT(f.P.trace(), before);

  before = f.P.trace();
  f.update_imu(0.2);
  EXPECT_LT(f.P.trace(), before);
}
