// Replays the sequence make_golden.py ran through the ORIGINAL ekf.py / ukf.py and checks
// that the C++ filters land on the same state and covariance at every step.

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <cstdio>

#include "ekf_localization_cpp/ekf.hpp"
#include "ekf_localization_cpp/ukf.hpp"
#include "golden.hpp"

using ekf_localization_cpp::EKF;
using ekf_localization_cpp::normalize_angle;
using ekf_localization_cpp::UKF;

constexpr double kTol = 1e-9;

template<typename Filter>
void expect_matches_python(int f, const char * name)
{
  const auto steps = load_golden(GOLDEN_CSV);
  ASSERT_GT(steps.size(), 1000u);
  Filter filt;
  double max_dx = 0, max_dP = 0;
  for (const auto & s : steps) {
    apply(filt, s);
    Eigen::Vector3d dx = filt.x - s.x[f];
    dx(2) = normalize_angle(dx(2));  // +pi vs -pi is the same heading
    max_dx = std::max(max_dx, dx.cwiseAbs().maxCoeff());
    max_dP = std::max(max_dP, (filt.P - s.P[f]).cwiseAbs().maxCoeff());
  }
  std::printf("[golden] %s: %zu steps, max|dx| = %.3g, max|dP| = %.3g\n",
    name, steps.size(), max_dx, max_dP);
  EXPECT_LT(max_dx, kTol);
  EXPECT_LT(max_dP, kTol);
}

TEST(Golden, EkfMatchesPython) {expect_matches_python<EKF>(0, "EKF");}
TEST(Golden, UkfMatchesPython) {expect_matches_python<UKF>(1, "UKF");}
