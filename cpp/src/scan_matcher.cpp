#include "ekf_localization_cpp/scan_matcher.hpp"

#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/point_types.h>

#include <algorithm>
#include <cmath>
#include <iterator>
#include <limits>
#include <map>
#include <numeric>
#include <utility>

namespace ekf_localization_cpp
{

namespace
{
constexpr double kInf = std::numeric_limits<double>::infinity();

struct Transform
{
  Eigen::Matrix2d R = Eigen::Matrix2d::Identity();
  Eigen::Vector2d t = Eigen::Vector2d::Zero();
};

Points apply(const Transform & T, const Points & p) {return (T.R * p).colwise() + T.t;}

// total <- step o total  (R_total = R @ R_total; t_total = R @ t_total + t)
void compose(Transform & total, const Transform & step)
{
  total.t = step.R * total.t + step.t;
  total.R = step.R * total.R;
}

// 2D nearest-neighbour index over `target` (stored as XYZ with z = 0; FLANN works in float,
// so distances are recomputed in double by the callers).
class Tree
{
public:
  explicit Tree(const Points & target)
  : cloud_(new pcl::PointCloud<pcl::PointXYZ>)
  {
    for (Eigen::Index i = 0; i < target.cols(); ++i) {
      cloud_->push_back(pcl::PointXYZ(target(0, i), target(1, i), 0.0f));
    }
    tree_.setInputCloud(cloud_);
  }

  int nearest(const Eigen::Vector2d & q) const
  {
    pcl::Indices idx(1);
    std::vector<float> sq_dist(1);
    tree_.nearestKSearch(pcl::PointXYZ(q.x(), q.y(), 0.0f), 1, idx, sq_dist);
    return idx[0];
  }

private:
  pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_;
  pcl::KdTreeFLANN<pcl::PointXYZ> tree_;
};

Points voxel_downsample(const Points & p, double voxel_size)
{
  // std::map keeps np.unique's lexicographic voxel order; sums run in input order like np.add.at.
  std::map<std::pair<long, long>, std::pair<Eigen::Vector2d, int>> cells;
  for (Eigen::Index i = 0; i < p.cols(); ++i) {
    const std::pair<long, long> key{
      static_cast<long>(std::floor(p(0, i) / voxel_size)),
      static_cast<long>(std::floor(p(1, i) / voxel_size))};
    auto & cell = cells.try_emplace(key, Eigen::Vector2d::Zero(), 0).first->second;
    cell.first += p.col(i);
    ++cell.second;
  }
  Points out(2, cells.size());
  Eigen::Index k = 0;
  for (const auto & [key, cell] : cells) {
    out.col(k++) = cell.first / cell.second;
  }
  return out;
}

Transform compute_transform_svd(const Points & src, const Points & tgt)
{
  const Eigen::Vector2d src_c = src.rowwise().mean();
  const Eigen::Vector2d tgt_c = tgt.rowwise().mean();
  const Eigen::Matrix2d H = (src.colwise() - src_c) * (tgt.colwise() - tgt_c).transpose();
  const Eigen::JacobiSVD<Eigen::Matrix2d> svd(H, Eigen::ComputeFullU | Eigen::ComputeFullV);
  Eigen::Matrix2d V = svd.matrixV();
  Transform T;
  T.R = V * svd.matrixU().transpose();
  if (T.R.determinant() < 0) {  // reflection
    V.col(1) *= -1;
    T.R = V * svd.matrixU().transpose();
  }
  T.t = tgt_c - T.R * src_c;
  return T;
}

struct Matches
{
  Points src, tgt;
  Eigen::VectorXd dist;
};

struct IcpResult
{
  Transform T;
  double rmse = kInf;
  int num_correspondences = 0;
};
}  // namespace

Points ScanMatcher::laserscan_to_points(
  const std::vector<float> & ranges, double angle_min, double angle_max,
  double range_min, double range_max)
{
  const size_t n = ranges.size();
  const double step = n > 1 ? (angle_max - angle_min) / (n - 1) : 0.0;  // np.linspace
  std::vector<Eigen::Vector2d> pts;
  for (size_t i = 0; i < n; ++i) {
    const double a = (n > 1 && i == n - 1) ? angle_max : i * step + angle_min;
    const double r = ranges[i];
    if (std::isfinite(r) && r >= range_min && r <= range_max && r > 0.0) {
      pts.emplace_back(r * std::cos(a), r * std::sin(a));
    }
  }
  Points out(2, pts.size());
  for (size_t i = 0; i < pts.size(); ++i) {
    out.col(i) = pts[i];
  }
  return out;
}

namespace
{
std::optional<Matches> find_correspondences(
  const ScanMatcher & m, const Points & src, const Points & target, const Tree & tree)
{
  std::vector<int> src_idx, tgt_idx;
  std::vector<double> dist;
  for (Eigen::Index i = 0; i < src.cols(); ++i) {
    const int j = tree.nearest(src.col(i));
    const double d = (src.col(i) - target.col(j)).norm();
    if (d < m.max_correspondence_dist) {
      src_idx.push_back(i);
      tgt_idx.push_back(j);
      dist.push_back(d);
    }
  }
  if (static_cast<int>(dist.size()) < m.min_points) {
    return std::nullopt;
  }

  // Trimmed ICP: keep the closest trim_ratio fraction.
  const int num_keep = std::max(static_cast<int>(dist.size() * m.trim_ratio), m.min_points);
  std::vector<int> order(dist.size());
  std::iota(order.begin(), order.end(), 0);
  std::partial_sort(
    order.begin(), order.begin() + num_keep, order.end(),
    [&](int a, int b) {return dist[a] < dist[b];});

  Matches out{Points(2, num_keep), Points(2, num_keep), Eigen::VectorXd(num_keep)};
  for (int k = 0; k < num_keep; ++k) {
    out.src.col(k) = src.col(src_idx[order[k]]);
    out.tgt.col(k) = target.col(tgt_idx[order[k]]);
    out.dist(k) = dist[order[k]];
  }
  return out;
}

IcpResult icp_single_scale(
  const ScanMatcher & m, const Points & source, const Points & target, int max_iter)
{
  const Tree tree(target);  // Python rebuilds the same tree every iteration
  Points src = source;
  IcpResult r;
  double prev_rmse = kInf;
  for (int it = 0; it < max_iter; ++it) {
    const auto matches = find_correspondences(m, src, target, tree);
    if (!matches) {
      return {r.T, kInf, 0};
    }
    const Transform step = compute_transform_svd(matches->src, matches->tgt);
    src = apply(step, src);
    compose(r.T, step);
    r.rmse = std::sqrt(matches->dist.squaredNorm() / matches->dist.size());
    r.num_correspondences = matches->dist.size();
    if (std::abs(prev_rmse - r.rmse) < m.convergence_threshold) {
      break;
    }
    prev_rmse = r.rmse;
  }
  return r;
}
}  // namespace

ScanMatcher::Result ScanMatcher::match(const Points & current_scan)
{
  if (!prev_scan || current_scan.cols() < min_points || prev_scan->cols() < min_points) {
    prev_scan = current_scan;
    return {};
  }

  // Multi-resolution alignment, coarse to fine.
  Points source = current_scan;
  Transform total;
  for (size_t i = 0; i < voxel_sizes.size(); ++i) {
    const IcpResult level = icp_single_scale(
      *this, voxel_downsample(source, voxel_sizes[i]), voxel_downsample(*prev_scan, voxel_sizes[i]),
      i == 0 ? max_iterations : max_iterations / 2);
    source = apply(level.T, source);
    compose(total, level.T);
  }

  // Final refinement at full resolution on at most 200 random points.
  Points sample = source;
  if (source.cols() > 200) {
    std::vector<int> all(source.cols()), pick;
    std::iota(all.begin(), all.end(), 0);
    std::sample(all.begin(), all.end(), std::back_inserter(pick), 200, rng_);
    sample = source(Eigen::all, pick);
  }
  const IcpResult fin = icp_single_scale(*this, sample, *prev_scan, 10);
  compose(total, fin.T);

  Result r;
  r.delta_pose << total.t, std::atan2(total.R(1, 0), total.R(0, 0));
  r.rmse = fin.rmse;
  r.num_correspondences = fin.num_correspondences;

  // Fitness uses `source` before the final refinement, as in Python.
  const Tree tree(*prev_scan);
  int inliers = 0;
  for (Eigen::Index i = 0; i < source.cols(); ++i) {
    const int j = tree.nearest(source.col(i));
    inliers += (source.col(i) - prev_scan->col(j)).norm() < max_correspondence_dist;
  }
  r.fitness = static_cast<double>(inliers) / source.cols();
  r.success = r.fitness > 0.3 && r.rmse < 0.1 && r.num_correspondences >= min_points;

  prev_scan = current_scan;
  return r;
}

Eigen::Matrix3d ScanMatcher::estimate_covariance(
  double fitness, double rmse, int num_correspondences, double base_variance) const
{
  double scale;
  if (fitness < 0.3) {
    scale = 100.0;
  } else if (fitness > 0.8) {
    scale = 0.5 + rmse * 5;
  } else {
    scale = 1.0 + (0.8 - fitness) * 20;
  }
  scale *= std::max(1.0, 100.0 / num_correspondences);
  scale *= 1.0 + rmse * 10;
  const double pos_var = base_variance * scale;
  return Eigen::Vector3d(pos_var, pos_var, base_variance * scale * 0.5).asDiagonal();
}

std::string ScanMatcher::validate_match(
  const Eigen::Vector3d & delta_pose, double fitness, double rmse, int num_correspondences,
  double dt, double max_velocity, double max_angular_velocity) const
{
  if (fitness < 0.3) {
    return "LOW_FITNESS";
  }
  if (rmse > 0.15) {
    return "HIGH_RMSE";
  }
  if (num_correspondences < min_points) {
    return "FEW_CORRESPONDENCES";
  }
  if (dt > 0) {
    if (delta_pose.head<2>().norm() / dt > max_velocity) {
      return "IMPLAUSIBLE_LINEAR_VELOCITY";
    }
    if (std::abs(delta_pose(2)) / dt > max_angular_velocity) {
      return "IMPLAUSIBLE_ANGULAR_VELOCITY";
    }
  }
  return "VALID";
}

}  // namespace ekf_localization_cpp
