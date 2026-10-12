#pragma once
// Loader for the CSV written by make_golden.py (header row, then 31 numeric columns per step).

#include <Eigen/Dense>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

struct GoldenStep
{
  double v, omega, dt, z_imu;
  bool gps;
  Eigen::Vector2d z_gps;
  Eigen::Vector3d x[2];  // [0] = EKF, [1] = UKF, after this step
  Eigen::Matrix3d P[2];
};

inline std::vector<GoldenStep> load_golden(const std::string & path)
{
  std::ifstream in(path);
  if (!in) {
    throw std::runtime_error("cannot open " + path + " (run make_golden.py first)");
  }
  std::string line;
  std::getline(in, line);  // header
  std::vector<GoldenStep> steps;
  while (std::getline(in, line)) {
    std::vector<double> c;
    std::stringstream ss(line);
    for (std::string cell; std::getline(ss, cell, ',');) {
      c.push_back(std::stod(cell));
    }
    if (c.size() != 31) {
      throw std::runtime_error("bad row in " + path);
    }
    GoldenStep s{c[0], c[1], c[2], c[3], c[4] != 0, {c[5], c[6]}, {}, {}};
    for (int f = 0; f < 2; ++f) {
      const double * r = &c[7 + 12 * f];
      s.x[f] << r[0], r[1], r[2];
      s.P[f] = Eigen::Map<const Eigen::Matrix<double, 3, 3, Eigen::RowMajor>>(r + 3);
    }
    steps.push_back(s);
  }
  return steps;
}

template<typename Filter>
void apply(Filter & f, const GoldenStep & s)
{
  f.predict(s.v, s.omega, s.dt);
  f.update_imu(s.z_imu);
  if (s.gps) {
    f.update_gps(s.z_gps);
  }
}
