// Per-step cost of predict + update_imu (+ update_gps at 1 Hz) on the golden sequence.
// Usage: bench_filters golden.csv   (compare with: make_golden.py golden.csv --bench)

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <vector>

#include "ekf_localization_cpp/ekf.hpp"
#include "ekf_localization_cpp/ukf.hpp"
#include "golden.hpp"

template<typename Filter>
double us_per_step(const std::vector<GoldenStep> & steps, double & sink, int reps = 200)
{
  std::vector<double> runs;
  for (int r = 0; r < reps; ++r) {
    Filter f;
    const auto t0 = std::chrono::steady_clock::now();
    for (const auto & s : steps) {
      apply(f, s);
    }
    const auto t1 = std::chrono::steady_clock::now();
    sink += f.x.sum() + f.P.trace();  // keep the work observable
    runs.push_back(std::chrono::duration<double, std::micro>(t1 - t0).count() / steps.size());
  }
  std::nth_element(runs.begin(), runs.begin() + reps / 2, runs.end());
  return runs[reps / 2];
}

int main(int argc, char ** argv)
{
  if (argc < 2) {
    std::fprintf(stderr, "usage: %s golden.csv\n", argv[0]);
    return 1;
  }
  const auto steps = load_golden(argv[1]);
  double sink = 0;
  std::printf("c++ EKF: %.3f us/step (median of 200 runs, %zu steps)\n",
    us_per_step<ekf_localization_cpp::EKF>(steps, sink), steps.size());
  std::printf("c++ UKF: %.3f us/step (median of 200 runs)\n",
    us_per_step<ekf_localization_cpp::UKF>(steps, sink));
  std::printf("(checksum %g)\n", sink);
}
