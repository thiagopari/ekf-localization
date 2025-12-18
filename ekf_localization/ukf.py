"""
Unscented Kalman Filter for 2D Robot Localization

Unlike EKF which linearizes using Jacobians, UKF uses sigma points
to capture the probability distribution and propagates them through
the nonlinear functions directly.

State vector: [x, y, theta]
"""

import numpy as np


class UKFLocalization:
    """
    Unscented Kalman Filter implementation for 2D localization.
    """
    
    def __init__(self):
        # State dimension
        self.n = 3  # [x, y, theta]
        
        # State and covariance
        self.x = np.zeros(self.n)
        self.P = np.diag([0.1, 0.1, 0.1])
        
        # Process noise covariance
        self.Q = np.diag([0.02**2, 0.02**2, 0.01**2])
        
        # Measurement noise - GPS [x, y]
        self.R_gps = np.diag([0.3**2, 0.3**2])
        
        # Measurement noise - IMU heading
        self.R_imu = np.array([[0.02**2]])
        
        # UKF parameters
        self.alpha = 1e-3  # Spread of sigma points
        self.beta = 2      # Prior knowledge (2 is optimal for Gaussian)
        self.kappa = 0     # Secondary scaling parameter
        
        # Derived parameters
        self.lambda_ = self.alpha**2 * (self.n + self.kappa) - self.n
        self.gamma = np.sqrt(self.n + self.lambda_)
        
        # Weights for mean and covariance
        self.Wm = np.zeros(2 * self.n + 1)
        self.Wc = np.zeros(2 * self.n + 1)
        
        self.Wm[0] = self.lambda_ / (self.n + self.lambda_)
        self.Wc[0] = self.lambda_ / (self.n + self.lambda_) + (1 - self.alpha**2 + self.beta)
        
        for i in range(1, 2 * self.n + 1):
            self.Wm[i] = 1 / (2 * (self.n + self.lambda_))
            self.Wc[i] = 1 / (2 * (self.n + self.lambda_))
    
    def normalize_angle(self, angle):
        """Wrap angle to [-pi, pi]."""
        return (angle + np.pi) % (2 * np.pi) - np.pi
    
    def generate_sigma_points(self, x, P):
        """Generate 2n+1 sigma points around mean x with covariance P."""
        n = len(x)
        sigma_points = np.zeros((2 * n + 1, n))
        
        # First sigma point is the mean
        sigma_points[0] = x
        
        # Matrix square root of P
        try:
            sqrt_P = np.linalg.cholesky((n + self.lambda_) * P)
        except np.linalg.LinAlgError:
            # If Cholesky fails, use eigendecomposition
            eigvals, eigvecs = np.linalg.eigh(P)
            eigvals = np.maximum(eigvals, 1e-10)  # Ensure positive
            sqrt_P = eigvecs @ np.diag(np.sqrt((n + self.lambda_) * eigvals))
        
        # Generate remaining sigma points
        for i in range(n):
            sigma_points[i + 1] = x + sqrt_P[i]
            sigma_points[n + i + 1] = x - sqrt_P[i]
        
        return sigma_points
    
    def motion_model(self, x, v, omega, dt):
        """Propagate state through motion model."""
        theta = x[2]
        theta_mid = theta + omega * dt / 2
        
        x_new = np.zeros(3)
        x_new[0] = x[0] + v * dt * np.cos(theta_mid)
        x_new[1] = x[1] + v * dt * np.sin(theta_mid)
        x_new[2] = self.normalize_angle(x[2] + omega * dt)
        
        return x_new
    
    def predict(self, v, omega, dt):
        """
        UKF prediction step.
        
        Args:
            v: Linear velocity (m/s)
            omega: Angular velocity (rad/s)
            dt: Timestep (s)
        """
        if dt <= 0 or dt > 1.0:
            return
        
        # Generate sigma points
        sigma_points = self.generate_sigma_points(self.x, self.P)
        
        # Propagate sigma points through motion model
        sigma_points_pred = np.zeros_like(sigma_points)
        for i in range(2 * self.n + 1):
            sigma_points_pred[i] = self.motion_model(sigma_points[i], v, omega, dt)
        
        # Compute predicted mean
        x_pred = np.zeros(self.n)
        for i in range(2 * self.n + 1):
            x_pred += self.Wm[i] * sigma_points_pred[i]
        x_pred[2] = self.normalize_angle(x_pred[2])
        
        # Compute predicted covariance
        P_pred = np.zeros((self.n, self.n))
        for i in range(2 * self.n + 1):
            diff = sigma_points_pred[i] - x_pred
            diff[2] = self.normalize_angle(diff[2])
            P_pred += self.Wc[i] * np.outer(diff, diff)
        
        # Add process noise
        P_pred += self.Q
        
        self.x = x_pred
        self.P = P_pred
    
    def update_gps(self, z_gps):
        """
        UKF update step with GPS position measurement.
        
        Args:
            z_gps: GPS measurement [x, y]
        """
        z = np.array(z_gps)
        m = len(z)  # Measurement dimension
        
        # Generate sigma points
        sigma_points = self.generate_sigma_points(self.x, self.P)
        
        # Transform sigma points to measurement space (GPS measures x, y)
        Z_sigma = np.zeros((2 * self.n + 1, m))
        for i in range(2 * self.n + 1):
            Z_sigma[i] = sigma_points[i][:2]  # h(x) = [x, y]
        
        # Predicted measurement mean
        z_pred = np.zeros(m)
        for i in range(2 * self.n + 1):
            z_pred += self.Wm[i] * Z_sigma[i]
        
        # Measurement covariance
        Pzz = np.zeros((m, m))
        for i in range(2 * self.n + 1):
            diff = Z_sigma[i] - z_pred
            Pzz += self.Wc[i] * np.outer(diff, diff)
        Pzz += self.R_gps
        
        # Cross-covariance
        Pxz = np.zeros((self.n, m))
        for i in range(2 * self.n + 1):
            x_diff = sigma_points[i] - self.x
            x_diff[2] = self.normalize_angle(x_diff[2])
            z_diff = Z_sigma[i] - z_pred
            Pxz += self.Wc[i] * np.outer(x_diff, z_diff)
        
        # Kalman gain
        K = Pxz @ np.linalg.inv(Pzz)
        
        # Update state
        innovation = z - z_pred
        self.x = self.x + K @ innovation
        self.x[2] = self.normalize_angle(self.x[2])
        
        # Update covariance
        self.P = self.P - K @ Pzz @ K.T
    
    def update_imu(self, z_imu):
        """
        UKF update step with IMU heading measurement.
        
        Args:
            z_imu: IMU heading (rad)
        """
        z = np.array([z_imu])
        m = 1  # Measurement dimension
        
        # Generate sigma points
        sigma_points = self.generate_sigma_points(self.x, self.P)
        
        # Transform sigma points to measurement space (IMU measures theta)
        Z_sigma = np.zeros((2 * self.n + 1, m))
        for i in range(2 * self.n + 1):
            Z_sigma[i, 0] = sigma_points[i][2]  # h(x) = theta
        
        # Predicted measurement mean
        z_pred = np.zeros(m)
        for i in range(2 * self.n + 1):
            z_pred += self.Wm[i] * Z_sigma[i]
        z_pred[0] = self.normalize_angle(z_pred[0])
        
        # Measurement covariance
        Pzz = np.zeros((m, m))
        for i in range(2 * self.n + 1):
            diff = Z_sigma[i] - z_pred
            diff[0] = self.normalize_angle(diff[0])
            Pzz += self.Wc[i] * np.outer(diff, diff)
        Pzz += self.R_imu
        
        # Cross-covariance
        Pxz = np.zeros((self.n, m))
        for i in range(2 * self.n + 1):
            x_diff = sigma_points[i] - self.x
            x_diff[2] = self.normalize_angle(x_diff[2])
            z_diff = Z_sigma[i] - z_pred
            z_diff[0] = self.normalize_angle(z_diff[0])
            Pxz += self.Wc[i] * np.outer(x_diff, z_diff)
        
        # Kalman gain
        K = Pxz @ np.linalg.inv(Pzz)
        
        # Update state
        innovation = z - z_pred
        innovation[0] = self.normalize_angle(innovation[0])
        self.x = self.x + (K @ innovation).flatten()
        self.x[2] = self.normalize_angle(self.x[2])
        
        # Update covariance
        self.P = self.P - K @ Pzz @ K.T
    
    def get_state(self):
        """Return current state estimate."""
        return self.x.copy()
    
    def get_covariance(self):
        """Return current covariance matrix."""
        return self.P.copy()
