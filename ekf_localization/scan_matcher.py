#!/usr/bin/env python3
"""
Point-to-Line ICP Scan Matcher for 2D LiDAR
Implements PL-ICP with quadratic convergence for robot localization
"""

import numpy as np
from scipy.spatial import KDTree
from typing import Tuple, Optional
from dataclasses import dataclass


@dataclass
class ScanMatchResult:
    """Result of scan matching operation"""
    success: bool
    transform: np.ndarray  # [dx, dy, dtheta]
    covariance: np.ndarray  # 3x3 covariance matrix
    fitness: float  # Fitness score (0-1, higher is better)
    inlier_rmse: float  # RMSE of inlier correspondences
    num_iterations: int
    
    
class ScanMatcher:
    """
    Point-to-Line ICP scan matcher for 2D LiDAR localization.
    
    Implements the Point-to-Line ICP variant from Censi (2008) which achieves
    quadratic convergence by projecting error onto surface normals.
    """
    
    def __init__(self,
                 max_iterations: int = 15,
                 convergence_threshold: float = 1e-6,
                 max_correspondence_distance: float = 0.3,
                 min_fitness: float = 0.3,
                 max_inlier_rmse: float = 0.02,
                 voxel_size: float = 0.02,
                 range_min: float = 0.12,
                 range_max: float = 2.8):
        """
        Initialize scan matcher.
        
        Args:
            max_iterations: Maximum ICP iterations
            convergence_threshold: Stop when transform change < threshold
            max_correspondence_distance: Maximum distance for point matching (m)
            min_fitness: Minimum fitness score to accept result (0-1)
            max_inlier_rmse: Maximum RMSE of inliers to accept result (m)
            voxel_size: Voxel size for downsampling (m)
            range_min: Minimum valid range reading (m)
            range_max: Maximum valid range reading (m)
        """
        self.max_iterations = max_iterations
        self.convergence_threshold = convergence_threshold
        self.max_correspondence_distance = max_correspondence_distance
        self.min_fitness = min_fitness
        self.max_inlier_rmse = max_inlier_rmse
        self.voxel_size = voxel_size
        self.range_min = range_min
        self.range_max = range_max
        
        self.prev_scan_points = None
        self.prev_normals = None
        
    def scan_to_points(self, ranges: np.ndarray, angles: np.ndarray) -> np.ndarray:
        """
        Convert LaserScan ranges to 2D point cloud.
        
        Args:
            ranges: Array of range measurements (m)
            angles: Array of angles for each range (rad)
            
        Returns:
            Nx2 array of [x, y] points in sensor frame
        """
        # Filter invalid readings
        valid_mask = (ranges > self.range_min) & (ranges < self.range_max) & np.isfinite(ranges)
        valid_ranges = ranges[valid_mask]
        valid_angles = angles[valid_mask]
        
        # Convert polar to Cartesian
        x = valid_ranges * np.cos(valid_angles)
        y = valid_ranges * np.sin(valid_angles)
        
        points = np.column_stack([x, y])
        return points
    
    def voxel_downsample(self, points: np.ndarray) -> np.ndarray:
        """
        Downsample point cloud using voxel grid.
        
        Args:
            points: Nx2 array of points
            
        Returns:
            Mx2 array of downsampled points (M <= N)
        """
        if len(points) == 0:
            return points
            
        # Quantize points to voxel grid
        voxel_indices = np.floor(points / self.voxel_size).astype(int)
        
        # Find unique voxels
        unique_voxels, inverse_indices = np.unique(voxel_indices, axis=0, return_inverse=True)
        
        # Compute centroid for each voxel
        downsampled = np.zeros((len(unique_voxels), 2))
        for i in range(len(unique_voxels)):
            mask = inverse_indices == i
            downsampled[i] = np.mean(points[mask], axis=0)
            
        return downsampled
    
    def estimate_normals(self, points: np.ndarray, k: int = 5) -> np.ndarray:
        """
        Estimate surface normals using local neighborhood.
        
        For 2D, the normal is perpendicular to the line fitted through k neighbors.
        
        Args:
            points: Nx2 array of points
            k: Number of neighbors to use
            
        Returns:
            Nx2 array of unit normals
        """
        if len(points) < k:
            # Not enough points, use pointing-inward normals
            normals = -points / (np.linalg.norm(points, axis=1, keepdims=True) + 1e-6)
            return normals
            
        # Build KD-tree for neighbor search
        tree = KDTree(points)
        normals = np.zeros_like(points)
        
        for i, point in enumerate(points):
            # Find k nearest neighbors
            _, indices = tree.query(point, k=min(k, len(points)))
            neighbors = points[indices]
            
            # Fit line using PCA: compute covariance and take smallest eigenvector
            centered = neighbors - neighbors.mean(axis=0)
            cov = centered.T @ centered
            eigenvalues, eigenvectors = np.linalg.eigh(cov)
            
            # Normal is the eigenvector with smallest eigenvalue (perpendicular to line)
            normal = eigenvectors[:, 0]
            
            # Orient normal toward sensor origin (assuming sensor at origin)
            if np.dot(normal, -point) < 0:
                normal = -normal
                
            normals[i] = normal / (np.linalg.norm(normal) + 1e-6)
            
        return normals
    
    def point_to_line_icp(self,
                          source: np.ndarray,
                          target: np.ndarray,
                          target_normals: np.ndarray,
                          initial_transform: Optional[np.ndarray] = None) -> ScanMatchResult:
        """
        Point-to-Line ICP alignment.
        
        Minimizes: Σᵢ (nᵢᵀ · (Rp_i + t - q_i))²
        where nᵢ is the normal at matched target point q_i
        
        Args:
            source: Mx2 source points
            target: Nx2 target points
            target_normals: Nx2 normals at target points
            initial_transform: Initial [dx, dy, dtheta] guess
            
        Returns:
            ScanMatchResult with transformation and diagnostics
        """
        if len(source) < 3 or len(target) < 3:
            return ScanMatchResult(
                success=False,
                transform=np.zeros(3),
                covariance=np.eye(3) * 1e6,
                fitness=0.0,
                inlier_rmse=float('inf'),
                num_iterations=0
            )
        
        # Build KD-tree for target
        target_tree = KDTree(target)
        
        # Initialize transformation
        if initial_transform is None:
            transform = np.zeros(3)  # [dx, dy, dtheta]
        else:
            transform = initial_transform.copy()
        
        source_transformed = source.copy()
        
        # ICP iterations
        for iteration in range(self.max_iterations):
            # Apply current transformation to source
            c, s = np.cos(transform[2]), np.sin(transform[2])
            R = np.array([[c, -s], [s, c]])
            source_transformed = (R @ source.T).T + transform[:2]
            
            # Find correspondences using nearest neighbor
            distances, indices = target_tree.query(source_transformed)
            
            # Filter correspondences by distance
            valid_mask = distances < self.max_correspondence_distance
            if np.sum(valid_mask) < 3:
                # Too few correspondences
                return ScanMatchResult(
                    success=False,
                    transform=transform,
                    covariance=np.eye(3) * 1e6,
                    fitness=0.0,
                    inlier_rmse=float('inf'),
                    num_iterations=iteration
                )
            
            source_matched = source_transformed[valid_mask]
            target_matched = target[indices[valid_mask]]
            normals_matched = target_normals[indices[valid_mask]]
            
            # Solve for incremental transformation using point-to-plane formulation
            # Build linear system: A @ [dx, dy, dtheta]ᵀ = b
            A = np.zeros((len(source_matched), 3))
            b = np.zeros(len(source_matched))
            
            for i in range(len(source_matched)):
                p = source_matched[i]
                q = target_matched[i]
                n = normals_matched[i]
                
                # Point-to-plane error: nᵀ(p - q)
                error = n @ (p - q)
                
                # Jacobian: ∂(Rp + t - q)/∂[dx, dy, dtheta]
                # For small dtheta: R ≈ [1, -dtheta; dtheta, 1]
                # So: ∂(Rp)/∂theta ≈ [-p_y, p_x]
                A[i, 0] = n[0]  # ∂/∂dx
                A[i, 1] = n[1]  # ∂/∂dy
                A[i, 2] = n @ np.array([-p[1], p[0]])  # ∂/∂dtheta
                
                b[i] = -error
            
            # Solve least squares: delta = (AᵀA)⁻¹Aᵀb
            try:
                delta = np.linalg.lstsq(A, b, rcond=None)[0]
            except np.linalg.LinAlgError:
                # Singular system
                return ScanMatchResult(
                    success=False,
                    transform=transform,
                    covariance=np.eye(3) * 1e6,
                    fitness=0.0,
                    inlier_rmse=float('inf'),
                    num_iterations=iteration
                )
            
            # Update transformation
            transform += delta
            
            # Normalize angle to [-π, π]
            transform[2] = np.arctan2(np.sin(transform[2]), np.cos(transform[2]))
            
            # Check convergence
            if np.linalg.norm(delta) < self.convergence_threshold:
                break
        
        # Compute final fitness and RMSE
        c, s = np.cos(transform[2]), np.sin(transform[2])
        R = np.array([[c, -s], [s, c]])
        source_final = (R @ source.T).T + transform[:2]
        distances, _ = target_tree.query(source_final)
        
        inlier_mask = distances < self.max_correspondence_distance
        fitness = np.sum(inlier_mask) / len(source)
        inlier_rmse = np.sqrt(np.mean(distances[inlier_mask]**2)) if np.any(inlier_mask) else float('inf')
        
        # Estimate covariance from Hessian
        covariance = self.estimate_covariance(A, inlier_rmse)
        
        # Validate result
        success = (fitness >= self.min_fitness and inlier_rmse <= self.max_inlier_rmse)
        
        return ScanMatchResult(
            success=success,
            transform=transform,
            covariance=covariance,
            fitness=fitness,
            inlier_rmse=inlier_rmse,
            num_iterations=iteration + 1
        )
    
    def estimate_covariance(self, A: np.ndarray, sigma: float) -> np.ndarray:
        """
        Estimate transformation covariance from Hessian.
        
        Cov ≈ σ² × (AᵀA)⁻¹
        
        Args:
            A: Jacobian matrix from last ICP iteration
            sigma: Measurement noise standard deviation (m)
            
        Returns:
            3x3 covariance matrix
        """
        H = A.T @ A  # Hessian approximation
        
        # Check conditioning
        condition_number = np.linalg.cond(H)
        if condition_number > 1000:
            # Degenerate geometry, inflate covariance
            return np.eye(3) * sigma**2 * condition_number
        
        try:
            H_inv = np.linalg.inv(H)
            covariance = sigma**2 * H_inv
            
            # Ensure positive definite
            eigenvalues = np.linalg.eigvalsh(covariance)
            if np.any(eigenvalues < 0):
                covariance = np.eye(3) * sigma**2
                
            return covariance
        except np.linalg.LinAlgError:
            # Singular Hessian
            return np.eye(3) * 1e6
    
    def process_scan(self,
                     ranges: np.ndarray,
                     angles: np.ndarray,
                     odom_delta: Optional[np.ndarray] = None) -> Optional[ScanMatchResult]:
        """
        Process new scan and return relative motion estimate.
        
        Args:
            ranges: Range measurements from LaserScan (m)
            angles: Angles for each range (rad)
            odom_delta: Odometry-based motion estimate [dx, dy, dtheta] as initial guess
            
        Returns:
            ScanMatchResult or None if first scan or failure
        """
        # Convert scan to points
        points = self.scan_to_points(ranges, angles)
        
        if len(points) < 10:
            # Too few valid points
            return None
        
        # Downsample
        points_down = self.voxel_downsample(points)
        
        if self.prev_scan_points is None:
            # First scan - initialize
            self.prev_scan_points = points_down
            self.prev_normals = self.estimate_normals(points_down)
            return None
        
        # Estimate normals for previous scan if not available
        if self.prev_normals is None:
            self.prev_normals = self.estimate_normals(self.prev_scan_points)
        
        # Run ICP: align current scan to previous scan
        result = self.point_to_line_icp(
            source=points_down,
            target=self.prev_scan_points,
            target_normals=self.prev_normals,
            initial_transform=odom_delta
        )
        
        # Update reference scan
        if result.success:
            self.prev_scan_points = points_down
            self.prev_normals = self.estimate_normals(points_down)
        
        return result
