"""
Robust 2D LiDAR Scan Matching Module

Implements trimmed ICP with:
- Point-to-point alignment using SVD
- Outlier rejection via trimming
- Multi-resolution coarse-to-fine alignment
- Covariance estimation from fitness
- Failure detection

For use with ROS2 sensor_msgs/LaserScan
"""

import numpy as np
from scipy.spatial import cKDTree
from typing import Tuple, Optional


class ScanMatcher:
    """
    Robust 2D scan matcher using trimmed ICP.
    """
    
    def __init__(self, 
                 max_iterations: int = 50,
                 convergence_threshold: float = 1e-6,
                 max_correspondence_dist: float = 0.5,
                 trim_ratio: float = 0.8,
                 min_points: int = 50):
        """
        Initialize scan matcher.
        
        Args:
            max_iterations: Maximum ICP iterations per resolution level
            convergence_threshold: Stop when RMSE change is below this
            max_correspondence_dist: Maximum distance for valid correspondences
            trim_ratio: Fraction of closest correspondences to keep (0.7-0.9)
            min_points: Minimum points required for valid matching
        """
        self.max_iterations = max_iterations
        self.convergence_threshold = convergence_threshold
        self.max_correspondence_dist = max_correspondence_dist
        self.trim_ratio = trim_ratio
        self.min_points = min_points
        
        # Previous scan storage
        self.prev_scan = None
        
        # Multi-resolution voxel sizes (coarse to fine)
        self.voxel_sizes = [0.1, 0.05, 0.02]
    
    def laserscan_to_points(self, ranges: np.ndarray, 
                            angle_min: float, 
                            angle_max: float,
                            range_min: float = 0.1,
                            range_max: float = 30.0) -> np.ndarray:
        """
        Convert laser scan ranges to 2D point array.
        
        Args:
            ranges: Array of range measurements
            angle_min: Start angle (radians)
            angle_max: End angle (radians)
            range_min: Minimum valid range
            range_max: Maximum valid range
            
        Returns:
            Nx2 numpy array of [x, y] points
        """
        angles = np.linspace(angle_min, angle_max, len(ranges))
        
        # Filter invalid readings
        valid = (np.isfinite(ranges) & 
                 (ranges >= range_min) & 
                 (ranges <= range_max) &
                 (ranges > 0.0))
        
        x = ranges[valid] * np.cos(angles[valid])
        y = ranges[valid] * np.sin(angles[valid])
        
        return np.column_stack([x, y])
    
    def voxel_downsample(self, points: np.ndarray, 
                         voxel_size: float) -> np.ndarray:
        """
        Downsample point cloud using voxel grid.
        
        Args:
            points: Nx2 point array
            voxel_size: Size of voxel grid cells
            
        Returns:
            Downsampled point array
        """
        if len(points) == 0:
            return points
            
        # Quantize points to voxel grid
        quantized = np.floor(points / voxel_size).astype(int)
        
        # Find unique voxels and compute centroids
        unique_voxels, inverse = np.unique(quantized, axis=0, return_inverse=True)
        
        # Compute mean point per voxel
        downsampled = np.zeros((len(unique_voxels), 2))
        counts = np.zeros(len(unique_voxels))
        
        np.add.at(downsampled, inverse, points)
        np.add.at(counts, inverse, 1)
        
        downsampled /= counts[:, np.newaxis]
        
        return downsampled
    
    def find_correspondences(self, source: np.ndarray, 
                            target: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Find nearest neighbor correspondences with trimming.
        
        Args:
            source: Source point cloud
            target: Target point cloud
            
        Returns:
            Tuple of (trimmed_source, trimmed_target, distances)
        """
        tree = cKDTree(target)
        distances, indices = tree.query(source, k=1)
        
        # Apply distance threshold
        valid = distances < self.max_correspondence_dist
        
        if np.sum(valid) < self.min_points:
            # Not enough valid correspondences
            return None, None, None
        
        distances = distances[valid]
        indices = indices[valid]
        source_valid = source[valid]
        
        # Trimmed ICP: keep only closest trim_ratio fraction
        num_keep = max(int(len(distances) * self.trim_ratio), self.min_points)
        sorted_idx = np.argsort(distances)[:num_keep]
        
        trimmed_source = source_valid[sorted_idx]
        trimmed_target = target[indices[sorted_idx]]
        trimmed_distances = distances[sorted_idx]
        
        return trimmed_source, trimmed_target, trimmed_distances
    
    def compute_transform_svd(self, source: np.ndarray, 
                              target: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute rigid transformation using SVD (Procrustes analysis).
        
        Args:
            source: Source points (matched)
            target: Target points (matched)
            
        Returns:
            Tuple of (2x2 rotation matrix, 2x1 translation vector)
        """
        # Compute centroids
        src_centroid = source.mean(axis=0)
        tgt_centroid = target.mean(axis=0)
        
        # Center the points
        src_centered = source - src_centroid
        tgt_centered = target - tgt_centroid
        
        # Compute cross-covariance matrix
        H = src_centered.T @ tgt_centered
        
        # SVD decomposition
        U, _, Vt = np.linalg.svd(H)
        
        # Compute rotation
        R = Vt.T @ U.T
        
        # Handle reflection case
        if np.linalg.det(R) < 0:
            Vt[-1, :] *= -1
            R = Vt.T @ U.T
        
        # Compute translation
        t = tgt_centroid - R @ src_centroid
        
        return R, t
    
    def apply_transform(self, points: np.ndarray, 
                        R: np.ndarray, 
                        t: np.ndarray) -> np.ndarray:
        """Apply rigid transformation to points."""
        return (R @ points.T).T + t
    
    def icp_single_scale(self, source: np.ndarray, 
                         target: np.ndarray,
                         max_iter: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray, float, int]:
        """
        Single-scale ICP alignment.
        
        Args:
            source: Source point cloud
            target: Target point cloud
            max_iter: Override for max iterations
            
        Returns:
            Tuple of (rotation, translation, rmse, num_correspondences)
        """
        if max_iter is None:
            max_iter = self.max_iterations
            
        src = source.copy()
        R_total = np.eye(2)
        t_total = np.zeros(2)
        prev_rmse = float('inf')
        
        for iteration in range(max_iter):
            # Find correspondences
            src_matched, tgt_matched, distances = self.find_correspondences(src, target)
            
            if src_matched is None:
                # Not enough correspondences
                return R_total, t_total, float('inf'), 0
            
            # Compute transformation
            R, t = self.compute_transform_svd(src_matched, tgt_matched)
            
            # Apply transformation
            src = self.apply_transform(src, R, t)
            
            # Accumulate transformation
            R_total = R @ R_total
            t_total = R @ t_total + t
            
            # Check convergence
            rmse = np.sqrt(np.mean(distances**2))
            if abs(prev_rmse - rmse) < self.convergence_threshold:
                break
            prev_rmse = rmse
        
        return R_total, t_total, rmse, len(src_matched) if src_matched is not None else 0
    
    def match(self, current_scan: np.ndarray, 
              initial_guess: Optional[np.ndarray] = None) -> Tuple[np.ndarray, float, float, int, bool]:
        """
        Match current scan against previous scan using multi-resolution ICP.
        
        Args:
            current_scan: Nx2 array of current scan points
            initial_guess: Optional [x, y, theta] initial transformation
            
        Returns:
            Tuple of (delta_pose [x, y, theta], rmse, fitness, num_correspondences, success)
        """
        if self.prev_scan is None or len(current_scan) < self.min_points:
            self.prev_scan = current_scan
            return np.zeros(3), 0.0, 0.0, 0, False
        
        if len(self.prev_scan) < self.min_points:
            self.prev_scan = current_scan
            return np.zeros(3), 0.0, 0.0, 0, False
        
        # Apply initial guess if provided
        source = current_scan.copy()
        if initial_guess is not None:
            c, s = np.cos(initial_guess[2]), np.sin(initial_guess[2])
            R_init = np.array([[c, -s], [s, c]])
            t_init = initial_guess[:2]
            source = self.apply_transform(source, R_init, t_init)
        
        # Multi-resolution alignment
        R_total = np.eye(2)
        t_total = np.zeros(2)
        
        for i, voxel_size in enumerate(self.voxel_sizes):
            # Downsample both clouds
            src_ds = self.voxel_downsample(source, voxel_size)
            tgt_ds = self.voxel_downsample(self.prev_scan, voxel_size)
            
            # More iterations for coarse level, fewer for fine
            max_iter = self.max_iterations if i == 0 else self.max_iterations // 2
            
            # Run ICP at this scale
            R, t, rmse, num_corr = self.icp_single_scale(src_ds, tgt_ds, max_iter)
            
            # Apply to full-resolution source
            source = self.apply_transform(source, R, t)
            
            # Accumulate transformation
            R_total = R @ R_total
            t_total = R @ t_total + t
        
        # Final refinement at full resolution (limited points for speed)
        if len(source) > 200:
            idx = np.random.choice(len(source), 200, replace=False)
            src_sample = source[idx]
        else:
            src_sample = source
            
        R_final, t_final, final_rmse, num_correspondences = self.icp_single_scale(
            src_sample, self.prev_scan, max_iter=10
        )
        
        R_total = R_final @ R_total
        t_total = R_final @ t_total + t_final
        
        # Include initial guess in total transformation
        if initial_guess is not None:
            c, s = np.cos(initial_guess[2]), np.sin(initial_guess[2])
            R_init = np.array([[c, -s], [s, c]])
            R_total = R_total @ R_init
            t_total = R_total @ initial_guess[:2] + t_total
        
        # Extract [x, y, theta] from transformation
        theta = np.arctan2(R_total[1, 0], R_total[0, 0])
        delta_pose = np.array([t_total[0], t_total[1], theta])
        
        # Compute fitness (fraction of points with good correspondences)
        tree = cKDTree(self.prev_scan)
        distances, _ = tree.query(source, k=1)
        fitness = np.mean(distances < self.max_correspondence_dist)
        
        # Determine success
        success = (fitness > 0.3 and 
                   final_rmse < 0.1 and 
                   num_correspondences >= self.min_points)
        
        # Update previous scan
        self.prev_scan = current_scan
        
        return delta_pose, final_rmse, fitness, num_correspondences, success
    
    def estimate_covariance(self, fitness: float, 
                           rmse: float, 
                           num_correspondences: int,
                           base_variance: float = 0.01) -> np.ndarray:
        """
        Estimate measurement covariance from ICP quality metrics.
        
        Args:
            fitness: Fraction of inlier correspondences
            rmse: Root mean square error of alignment
            num_correspondences: Number of correspondences used
            base_variance: Base variance for perfect match
            
        Returns:
            3x3 covariance matrix for [x, y, theta]
        """
        # Scale factor based on quality
        if fitness < 0.3:
            scale = 100.0  # Very uncertain
        elif fitness > 0.8:
            scale = 0.5 + rmse * 5  # High confidence
        else:
            # Linear interpolation
            scale = 1.0 + (0.8 - fitness) * 20
        
        # Reduce uncertainty with more correspondences
        corr_factor = max(1.0, 100.0 / num_correspondences)
        scale *= corr_factor
        
        # Add RMSE contribution
        scale *= (1.0 + rmse * 10)
        
        # Build covariance matrix
        pos_var = base_variance * scale
        theta_var = base_variance * scale * 0.5  # Angular usually more certain
        
        return np.diag([pos_var, pos_var, theta_var])
    
    def validate_match(self, delta_pose: np.ndarray,
                      fitness: float,
                      rmse: float,
                      num_correspondences: int,
                      dt: float,
                      max_velocity: float = 2.0,
                      max_angular_velocity: float = 2.0) -> Tuple[bool, str]:
        """
        Validate scan match result before using in filter.
        
        Args:
            delta_pose: [x, y, theta] transformation
            fitness: Match fitness score
            rmse: Alignment RMSE
            num_correspondences: Number of correspondences
            dt: Time since last scan
            max_velocity: Maximum plausible linear velocity (m/s)
            max_angular_velocity: Maximum plausible angular velocity (rad/s)
            
        Returns:
            Tuple of (is_valid, reason_string)
        """
        # Quality thresholds
        if fitness < 0.3:
            return False, "LOW_FITNESS"
        
        if rmse > 0.15:
            return False, "HIGH_RMSE"
        
        if num_correspondences < self.min_points:
            return False, "FEW_CORRESPONDENCES"
        
        # Physical plausibility
        if dt > 0:
            linear_vel = np.linalg.norm(delta_pose[:2]) / dt
            angular_vel = abs(delta_pose[2]) / dt
            
            if linear_vel > max_velocity:
                return False, "IMPLAUSIBLE_LINEAR_VELOCITY"
            
            if angular_vel > max_angular_velocity:
                return False, "IMPLAUSIBLE_ANGULAR_VELOCITY"
        
        return True, "VALID"
    
    def reset(self):
        """Reset the matcher state (clear previous scan)."""
        self.prev_scan = None
