#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <vector>

#include "ekf_localization_cpp/scan_matcher.hpp"

using ekf_localization_cpp::ScanMatcher;

// 360-beam scan (angles linspace(-pi, pi)) of a 6 m x 4 m room centred on the origin.
std::vector<float> room_scan(double x, double y, double yaw)
{
  std::vector<float> ranges(360);
  for (int i = 0; i < 360; ++i) {
    const double a = yaw - M_PI + 2 * M_PI * i / 359;
    const double c = std::cos(a), s = std::sin(a);
    double best = 1e9;
    for (const double wx : {-3.0, 3.0}) {
      if (std::abs(c) > 1e-12 && (wx - x) / c > 0) {best = std::min(best, (wx - x) / c);}
    }
    for (const double wy : {-2.0, 2.0}) {
      if (std::abs(s) > 1e-12 && (wy - y) / s > 0) {best = std::min(best, (wy - y) / s);}
    }
    ranges[i] = best;
  }
  return ranges;
}

TEST(ScanMatcher, RecoversMotionBetweenRoomScans) {
  ScanMatcher m;
  const auto prev = ScanMatcher::laserscan_to_points(room_scan(0, 0, 0), -M_PI, M_PI);
  const auto cur = ScanMatcher::laserscan_to_points(room_scan(0.1, 0.05, 0.05), -M_PI, M_PI);
  ASSERT_EQ(prev.cols(), 360);

  EXPECT_FALSE(m.match(prev).success);  // first scan only primes prev_scan
  const auto r = m.match(cur);
  EXPECT_TRUE(r.success);
  // 1 cm / 10 mrad: the worst of 2000 runs was 4.7 mm (random 200-point final refinement)
  EXPECT_NEAR(r.delta_pose(0), 0.1, 1e-2);
  EXPECT_NEAR(r.delta_pose(1), 0.05, 1e-2);
  EXPECT_NEAR(r.delta_pose(2), 0.05, 1e-2);
  EXPECT_EQ(m.validate_match(r.delta_pose, r.fitness, r.rmse, r.num_correspondences, 0.1), "VALID");
  EXPECT_EQ(
    m.validate_match(Eigen::Vector3d(1, 0, 0), 0.9, 0.01, 100, 0.1),
    "IMPLAUSIBLE_LINEAR_VELOCITY");
}

TEST(ScanMatcher, FiltersInvalidRanges) {
  const std::vector<float> ranges{1.0f, NAN, 0.05f, 50.0f, INFINITY, 2.0f};
  EXPECT_EQ(ScanMatcher::laserscan_to_points(ranges, 0.0, 1.0).cols(), 2);
}
