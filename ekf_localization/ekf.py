import numpy as np

class EKFLocalization:
    """
    Extended Kalman Filter for 2D robot localization.
    
    Sensor fusion architecture:
    - Prediction: Uses velocity commands (dead reckoning)
    - IMU update: Corrects heading (high rate, ~50Hz)
    - GPS update: Corrects position (low rate, ~1Hz)
    
    State vector: [x, y, theta]
    """
    
    def __init__(self):
        # State: [x, y, theta]
        self.x = np.zeros(3)
        
        # Covariance matrix
        self.P = np.diag([0.1, 0.1, 0.1])
        
        # Process noise covariance (tunable)
        self.Q = np.diag([0.02**2, 0.02**2, 0.01**2])
        
        # Measurement noise - GPS position [x, y]
        self.R_gps = np.diag([0.5**2, 0.5**2])  # 50cm uncertainty
        
        # Measurement noise - IMU heading
        self.R_imu = np.array([[0.02**2]])  # ~1 degree
        
        # Measurement matrices
        self.H_gps = np.array([
            [1, 0, 0],
            [0, 1, 0]
        ])
        self.H_imu = np.array([[0, 0, 1]])
    
    def normalize_angle(self, angle):
        """Wrap angle to [-pi, pi]."""
        return (angle + np.pi) % (2 * np.pi) - np.pi
    
    def predict(self, v, omega, dt):
        """
        Prediction step using differential drive motion model.
        
        Args:
            v: Linear velocity (m/s) - from wheel odometry
            omega: Angular velocity (rad/s) - from wheel odometry
            dt: Timestep (s)
        """
        theta = self.x[2]
        theta_mid = theta + omega * dt / 2
        
        # Motion model
        self.x[0] += v * dt * np.cos(theta_mid)
        self.x[1] += v * dt * np.sin(theta_mid)
        self.x[2] += omega * dt
        self.x[2] = self.normalize_angle(self.x[2])
        
        # Jacobian of motion model
        F = np.array([
            [1, 0, -v * dt * np.sin(theta_mid)],
            [0, 1,  v * dt * np.cos(theta_mid)],
            [0, 0, 1]
        ])
        
        # Covariance prediction
        self.P = F @ self.P @ F.T + self.Q
    
    def update_gps(self, z_gps):
        """
        Update step with GPS position measurement.
        
        Args:
            z_gps: GPS measurement [x, y]
        """
        z = np.array(z_gps)
        H = self.H_gps
        R = self.R_gps
        
        # Predicted measurement
        z_pred = H @ self.x
        
        # Innovation
        y = z - z_pred
        
        # Innovation covariance
        S = H @ self.P @ H.T + R
        
        # Kalman gain
        K = self.P @ H.T @ np.linalg.inv(S)
        
        # State update
        self.x = self.x + K @ y
        self.x[2] = self.normalize_angle(self.x[2])
        
        # Covariance update (Joseph form for numerical stability)
        I = np.eye(3)
        self.P = (I - K @ H) @ self.P
    
    def update_imu(self, z_imu):
        """
        Update step with IMU heading measurement.
        
        Args:
            z_imu: IMU heading (rad)
        """
        z = np.array([z_imu])
        H = self.H_imu
        R = self.R_imu
        
        # Innovation
        y = z - H @ self.x
        y[0] = self.normalize_angle(y[0])
        
        # Innovation covariance
        S = H @ self.P @ H.T + R
        
        # Kalman gain
        K = self.P @ H.T @ np.linalg.inv(S)
        
        # State update
        self.x = self.x + (K @ y).flatten()
        self.x[2] = self.normalize_angle(self.x[2])
        
        # Covariance update
        I = np.eye(3)
        self.P = (I - K @ H) @ self.P
    
    def get_state(self):
        """Return current state estimate."""
        return self.x.copy()
    
    def get_covariance(self):
        """Return current covariance matrix."""
        return self.P.copy()


# =============================================================================
# Improved test with realistic sensor fusion scenario
# =============================================================================

def generate_ground_truth(dt, duration):
    """Generate ground truth trajectory - figure-8 pattern."""
    t = np.arange(0, duration, dt)
    n = len(t)
    
    # Control inputs for interesting trajectory
    v = 0.3 * np.ones(n)  # 0.3 m/s forward
    omega = 0.3 * np.sin(0.2 * t)  # Smooth turning
    
    # Integrate to get ground truth
    x_true = np.zeros((n, 3))
    for i in range(1, n):
        theta = x_true[i-1, 2]
        theta_mid = theta + omega[i-1] * dt / 2
        x_true[i, 0] = x_true[i-1, 0] + v[i-1] * dt * np.cos(theta_mid)
        x_true[i, 1] = x_true[i-1, 1] + v[i-1] * dt * np.sin(theta_mid)
        x_true[i, 2] = x_true[i-1, 2] + omega[i-1] * dt
    
    return t, x_true, v, omega


def generate_sensor_data(t, x_true, v_true, omega_true, dt):
    """
    Generate realistic noisy sensor data.
    
    Returns:
        v_odom, omega_odom: Noisy velocity measurements (every timestep)
        z_imu: Noisy heading measurements (every timestep)
        z_gps: Noisy position measurements (1 Hz)
        gps_indices: Timestep indices where GPS is available
    """
    n = len(t)
    
    # Wheel odometry velocities - biased and noisy
    # Simulates wheel slip, calibration errors
    v_bias = 0.02  # 2 cm/s systematic bias
    omega_bias = 0.01  # Small heading rate bias
    v_odom = v_true + v_bias + np.random.normal(0, 0.02, n)
    omega_odom = omega_true + omega_bias + np.random.normal(0, 0.01, n)
    
    # IMU heading - accurate but with slow drift
    imu_drift_rate = 0.001  # rad/s drift
    imu_drift = np.cumsum(np.ones(n) * imu_drift_rate * dt)
    z_imu = x_true[:, 2] + imu_drift + np.random.normal(0, 0.02, n)
    
    # GPS position - available at 1 Hz, noisy but unbiased
    gps_period = int(1.0 / dt)  # Every 1 second
    gps_indices = np.arange(0, n, gps_period)
    z_gps = x_true[gps_indices, :2] + np.random.normal(0, 0.5, (len(gps_indices), 2))
    
    return v_odom, omega_odom, z_imu, z_gps, gps_indices


def run_dead_reckoning(v_odom, omega_odom, dt):
    """Pure dead reckoning using only odometry velocities."""
    n = len(v_odom)
    x_dr = np.zeros((n, 3))
    
    for i in range(1, n):
        theta = x_dr[i-1, 2]
        theta_mid = theta + omega_odom[i-1] * dt / 2
        x_dr[i, 0] = x_dr[i-1, 0] + v_odom[i-1] * dt * np.cos(theta_mid)
        x_dr[i, 1] = x_dr[i-1, 1] + v_odom[i-1] * dt * np.sin(theta_mid)
        x_dr[i, 2] = x_dr[i-1, 2] + omega_odom[i-1] * dt
    
    return x_dr


def run_test():
    """Run EKF test demonstrating sensor fusion value."""
    # Parameters
    dt = 0.02  # 50 Hz
    duration = 60.0  # 60 seconds - longer to show drift
    
    print("Generating trajectory and sensor data...")
    t, x_true, v_true, omega_true = generate_ground_truth(dt, duration)
    v_odom, omega_odom, z_imu, z_gps, gps_indices = generate_sensor_data(
        t, x_true, v_true, omega_true, dt
    )
    
    # Run pure dead reckoning (baseline)
    print("Running dead reckoning baseline...")
    x_dr = run_dead_reckoning(v_odom, omega_odom, dt)
    
    # Run EKF
    print("Running EKF with sensor fusion...")
    ekf = EKFLocalization()
    x_est = np.zeros_like(x_true)
    gps_idx = 0
    
    for i in range(len(t)):
        # Predict using odometry velocities
        if i > 0:
            ekf.predict(v_odom[i-1], omega_odom[i-1], dt)
        
        # IMU update every timestep (high rate)
        ekf.update_imu(z_imu[i])
        
        # GPS update at 1 Hz (low rate)
        if gps_idx < len(gps_indices) and i == gps_indices[gps_idx]:
            ekf.update_gps(z_gps[gps_idx])
            gps_idx += 1
        
        x_est[i] = ekf.get_state()
    
    # Compute errors
    ekf_pos_error = np.sqrt((x_est[:, 0] - x_true[:, 0])**2 + 
                            (x_est[:, 1] - x_true[:, 1])**2)
    dr_pos_error = np.sqrt((x_dr[:, 0] - x_true[:, 0])**2 + 
                           (x_dr[:, 1] - x_true[:, 1])**2)
    
    print("\n" + "="*50)
    print("RESULTS")
    print("="*50)
    print(f"Dead Reckoning RMS error:  {np.sqrt(np.mean(dr_pos_error**2)):.3f} m")
    print(f"EKF Fusion RMS error:      {np.sqrt(np.mean(ekf_pos_error**2)):.3f} m")
    print(f"Improvement:               {(1 - np.sqrt(np.mean(ekf_pos_error**2)) / np.sqrt(np.mean(dr_pos_error**2)))*100:.1f}%")
    print("-"*50)
    print(f"Dead Reckoning final error: {dr_pos_error[-1]:.3f} m")
    print(f"EKF Fusion final error:     {ekf_pos_error[-1]:.3f} m")
    print("="*50)
    
    return t, x_true, x_est, x_dr, z_gps, gps_indices, ekf_pos_error, dr_pos_error


if __name__ == "__main__":
    results = run_test()
    t, x_true, x_est, x_dr, z_gps, gps_indices, ekf_err, dr_err = results
    
    # Plot if matplotlib available
    try:
        import matplotlib.pyplot as plt
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # Trajectory comparison
        ax = axes[0, 0]
        ax.plot(x_true[:, 0], x_true[:, 1], 'g-', label='Ground Truth', linewidth=2)
        ax.plot(x_dr[:, 0], x_dr[:, 1], 'r--', alpha=0.7, label='Dead Reckoning')
        ax.plot(x_est[:, 0], x_est[:, 1], 'b-', label='EKF Estimate', linewidth=1.5)
        ax.scatter(z_gps[:, 0], z_gps[:, 1], c='orange', s=20, alpha=0.5, 
                   label='GPS Measurements', zorder=5)
        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        ax.set_title('Trajectory Comparison')
        ax.legend()
        ax.axis('equal')
        ax.grid(True, alpha=0.3)
        
        # Position error over time
        ax = axes[0, 1]
        ax.plot(t, dr_err, 'r-', alpha=0.7, label='Dead Reckoning Error')
        ax.plot(t, ekf_err, 'b-', label='EKF Error')
        ax.axhline(y=np.mean(ekf_err), color='b', linestyle=':', alpha=0.5)
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Position Error (m)')
        ax.set_title('Position Error Over Time')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Heading comparison
        ax = axes[1, 0]
        ax.plot(t, x_true[:, 2], 'g-', label='Ground Truth', linewidth=2)
        ax.plot(t, x_dr[:, 2], 'r--', alpha=0.7, label='Dead Reckoning')
        ax.plot(t, x_est[:, 2], 'b-', label='EKF Estimate')
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Heading (rad)')
        ax.set_title('Heading Comparison')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Error histogram
        ax = axes[1, 1]
        ax.hist(dr_err, bins=30, alpha=0.5, color='red', label='Dead Reckoning')
        ax.hist(ekf_err, bins=30, alpha=0.5, color='blue', label='EKF')
        ax.axvline(x=np.mean(dr_err), color='red', linestyle='--', 
                   label=f'DR Mean: {np.mean(dr_err):.2f}m')
        ax.axvline(x=np.mean(ekf_err), color='blue', linestyle='--',
                   label=f'EKF Mean: {np.mean(ekf_err):.2f}m')
        ax.set_xlabel('Position Error (m)')
        ax.set_ylabel('Count')
        ax.set_title('Error Distribution')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('ekf_fusion_results.png', dpi=150)
        plt.show()
        print("\nPlot saved to ekf_fusion_results.png")
        
    except ImportError:
        print("Matplotlib not available - skipping plots")
