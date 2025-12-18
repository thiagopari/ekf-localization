#!/usr/bin/env python3
"""
EKF Localization ROS2 Node

Fuses wheel odometry velocities, IMU heading, and simulated GPS
to produce a filtered pose estimate.

Subscriptions:
    /imu (sensor_msgs/Imu): IMU data with orientation
    /odom (nav_msgs/Odometry): Wheel odometry for velocities + simulated GPS

Publications:
    /ekf_pose (geometry_msgs/PoseStamped): Filtered pose estimate
    /tf (tf2_msgs/TFMessage): Transform from odom -> base_footprint_ekf
"""

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.time import Time
from sensor_msgs.msg import Imu
from nav_msgs.msg import Odometry
from geometry_msgs.msg import PoseStamped, TransformStamped
from nav_msgs.msg import Path
from tf2_ros import TransformBroadcaster
import tf_transformations


class EKFLocalization:
    """EKF implementation - same as standalone version."""
    
    def __init__(self):
        self.x = np.zeros(3)
        self.P = np.diag([0.1, 0.1, 0.1])
        self.Q = np.diag([0.02**2, 0.02**2, 0.01**2])
        self.R_gps = np.diag([0.3**2, 0.3**2])
        self.R_imu = np.array([[0.02**2]])
        self.H_gps = np.array([[1, 0, 0], [0, 1, 0]])
        self.H_imu = np.array([[0, 0, 1]])
    
    def normalize_angle(self, angle):
        return (angle + np.pi) % (2 * np.pi) - np.pi
    
    def predict(self, v, omega, dt):
        if dt <= 0 or dt > 1.0:
            return
        
        theta = self.x[2]
        theta_mid = theta + omega * dt / 2
        
        self.x[0] += v * dt * np.cos(theta_mid)
        self.x[1] += v * dt * np.sin(theta_mid)
        self.x[2] += omega * dt
        self.x[2] = self.normalize_angle(self.x[2])
        
        F = np.array([
            [1, 0, -v * dt * np.sin(theta_mid)],
            [0, 1,  v * dt * np.cos(theta_mid)],
            [0, 0, 1]
        ])
        
        self.P = F @ self.P @ F.T + self.Q
    
    def update_gps(self, z_gps):
        z = np.array(z_gps)
        H = self.H_gps
        R = self.R_gps
        
        y = z - H @ self.x
        S = H @ self.P @ H.T + R
        K = self.P @ H.T @ np.linalg.inv(S)
        
        self.x = self.x + K @ y
        self.x[2] = self.normalize_angle(self.x[2])
        self.P = (np.eye(3) - K @ H) @ self.P
    
    def update_imu(self, z_imu):
        z = np.array([z_imu])
        H = self.H_imu
        R = self.R_imu
        
        y = z - H @ self.x
        y[0] = self.normalize_angle(y[0])
        S = H @ self.P @ H.T + R
        K = self.P @ H.T @ np.linalg.inv(S)
        
        self.x = self.x + (K @ y).flatten()
        self.x[2] = self.normalize_angle(self.x[2])
        self.P = (np.eye(3) - K @ H) @ self.P
    
    def get_state(self):
        return self.x.copy()
    
    def get_covariance(self):
        return self.P.copy()


class EKFNode(Node):
    """ROS2 node wrapping the EKF."""
    
    def __init__(self):
        super().__init__('ekf_localization_node')
        
        # Declare parameters
        self.declare_parameter('gps_rate', 1.0)  # Hz
        self.declare_parameter('add_odom_noise', True)
        self.declare_parameter('odom_noise_std', 0.02)  # m/s
        
        # Get parameters
        self.gps_rate = self.get_parameter('gps_rate').value
        self.add_odom_noise = self.get_parameter('add_odom_noise').value
        self.odom_noise_std = self.get_parameter('odom_noise_std').value
        
        # Initialize EKF
        self.ekf = EKFLocalization()
        
        # Timing
        self.last_odom_time = None
        self.last_gps_time = None
        self.gps_period = 1.0 / self.gps_rate
        
        # TF broadcaster
        self.tf_broadcaster = TransformBroadcaster(self)
        
        # Publishers
        self.pose_pub = self.create_publisher(PoseStamped, 'ekf_pose', 10)
        self.ekf_path_pub = self.create_publisher(Path, 'ekf_path', 10)
        self.odom_path_pub = self.create_publisher(Path, 'odom_path', 10)
        
        # Path storage
        self.ekf_path = Path()
        self.ekf_path.header.frame_id = 'odom'
        self.odom_path = Path()
        self.odom_path.header.frame_id = 'odom'
        self.dr_path = Path()  # Dead reckoning path (noisy velocities, no GPS)
        self.dr_path.header.frame_id = 'odom'
        
        # Dead reckoning state (integrates noisy velocities without correction)
        self.dr_state = np.zeros(3)  # [x, y, theta]
        
        # Publisher for dead reckoning path
        self.dr_path_pub = self.create_publisher(Path, 'dr_path', 10)
        
        # Store latest odom for TF broadcasting
        self.latest_odom_pose = None
        
        # Subscribers
        self.odom_sub = self.create_subscription(
            Odometry, 'odom', self.odom_callback, 10)
        self.imu_sub = self.create_subscription(
            Imu, 'imu', self.imu_callback, 10)
        
        self.get_logger().info('EKF Localization Node started')
        self.get_logger().info(f'  GPS rate: {self.gps_rate} Hz')
        self.get_logger().info(f'  Odom noise injection: {self.add_odom_noise}')
    
    def get_yaw_from_quaternion(self, q):
        """Extract yaw from quaternion."""
        # quaternion format: [x, y, z, w]
        euler = tf_transformations.euler_from_quaternion([q.x, q.y, q.z, q.w])
        return euler[2]  # yaw
    
    def odom_callback(self, msg: Odometry):
        """Handle odometry messages - prediction step + simulated GPS."""
        current_time = Time.from_msg(msg.header.stamp).nanoseconds / 1e9
        
        # Broadcast odom -> base_footprint TF (missing from Gazebo)
        self.broadcast_odom_tf(msg)
        
        # Get velocities
        v = msg.twist.twist.linear.x
        omega = msg.twist.twist.angular.z
        
        # Add noise to simulate realistic wheel odometry
        if self.add_odom_noise:
            v_noisy = v + np.random.normal(0, self.odom_noise_std)
            omega_noisy = omega + np.random.normal(0, self.odom_noise_std * 0.5)
        else:
            v_noisy = v
            omega_noisy = omega
        
        # Compute dt
        if self.last_odom_time is not None:
            dt = current_time - self.last_odom_time
            
            # Prediction step (EKF uses noisy velocities)
            self.ekf.predict(v_noisy, omega_noisy, dt)
            
            # Update dead reckoning state (same noisy velocities, no corrections)
            if dt > 0 and dt < 1.0:
                theta = self.dr_state[2]
                theta_mid = theta + omega_noisy * dt / 2
                self.dr_state[0] += v_noisy * dt * np.cos(theta_mid)
                self.dr_state[1] += v_noisy * dt * np.sin(theta_mid)
                self.dr_state[2] += omega_noisy * dt
        
        self.last_odom_time = current_time
        
        # Simulated GPS update at specified rate
        if self.last_gps_time is None:
            self.last_gps_time = current_time
        
        if current_time - self.last_gps_time >= self.gps_period:
            # Use odometry position as "GPS" with added noise
            gps_x = msg.pose.pose.position.x
            gps_y = msg.pose.pose.position.y
            
            # Add GPS noise
            gps_x += np.random.normal(0, 0.3)
            gps_y += np.random.normal(0, 0.3)
            
            self.ekf.update_gps([gps_x, gps_y])
            self.last_gps_time = current_time
            self.get_logger().debug(f'GPS update: ({gps_x:.2f}, {gps_y:.2f})')
        
        # Publish current estimate
        self.publish_estimate(msg.header.stamp)
        
        # Publish dead reckoning path
        self.publish_dr_path(msg.header.stamp)
    
    def imu_callback(self, msg: Imu):
        """Handle IMU messages - heading update step."""
        yaw = self.get_yaw_from_quaternion(msg.orientation)
        self.ekf.update_imu(yaw)
    
    def publish_dr_path(self, stamp):
        """Publish dead reckoning path (noisy velocities, no corrections)."""
        dr_pose = PoseStamped()
        dr_pose.header.stamp = stamp
        dr_pose.header.frame_id = 'odom'
        dr_pose.pose.position.x = self.dr_state[0]
        dr_pose.pose.position.y = self.dr_state[1]
        dr_pose.pose.position.z = 0.0
        
        q = tf_transformations.quaternion_from_euler(0, 0, self.dr_state[2])
        dr_pose.pose.orientation.x = q[0]
        dr_pose.pose.orientation.y = q[1]
        dr_pose.pose.orientation.z = q[2]
        dr_pose.pose.orientation.w = q[3]
        
        self.dr_path.poses.append(dr_pose)
        if len(self.dr_path.poses) > 1000:
            self.dr_path.poses = self.dr_path.poses[-1000:]
        
        self.dr_path.header.stamp = stamp
        self.dr_path_pub.publish(self.dr_path)
    
    def broadcast_odom_tf(self, msg: Odometry):
        """Broadcast odom -> base_footprint transform (missing from Gazebo)."""
        t = TransformStamped()
        t.header.stamp = msg.header.stamp
        t.header.frame_id = 'odom'
        t.child_frame_id = 'base_footprint'
        
        t.transform.translation.x = msg.pose.pose.position.x
        t.transform.translation.y = msg.pose.pose.position.y
        t.transform.translation.z = msg.pose.pose.position.z
        t.transform.rotation = msg.pose.pose.orientation
        
        self.tf_broadcaster.sendTransform(t)
        
        # Store odometry path
        odom_pose = PoseStamped()
        odom_pose.header = msg.header
        odom_pose.pose = msg.pose.pose
        self.odom_path.poses.append(odom_pose)
        
        # Limit path length to avoid memory issues
        if len(self.odom_path.poses) > 1000:
            self.odom_path.poses = self.odom_path.poses[-1000:]
        
        # Publish odometry path
        self.odom_path.header.stamp = msg.header.stamp
        self.odom_path_pub.publish(self.odom_path)
    
    def publish_estimate(self, stamp):
        """Publish filtered pose estimate and TF."""
        state = self.ekf.get_state()
        
        # Publish PoseStamped
        pose_msg = PoseStamped()
        pose_msg.header.stamp = stamp
        pose_msg.header.frame_id = 'odom'
        
        pose_msg.pose.position.x = state[0]
        pose_msg.pose.position.y = state[1]
        pose_msg.pose.position.z = 0.0
        
        q = tf_transformations.quaternion_from_euler(0, 0, state[2])
        pose_msg.pose.orientation.x = q[0]
        pose_msg.pose.orientation.y = q[1]
        pose_msg.pose.orientation.z = q[2]
        pose_msg.pose.orientation.w = q[3]
        
        self.pose_pub.publish(pose_msg)
        
        # Add to EKF path and publish
        self.ekf_path.poses.append(pose_msg)
        if len(self.ekf_path.poses) > 1000:
            self.ekf_path.poses = self.ekf_path.poses[-1000:]
        self.ekf_path.header.stamp = stamp
        self.ekf_path_pub.publish(self.ekf_path)
        
        # Broadcast TF
        t = TransformStamped()
        t.header.stamp = stamp
        t.header.frame_id = 'odom'
        t.child_frame_id = 'base_footprint_ekf'
        
        t.transform.translation.x = state[0]
        t.transform.translation.y = state[1]
        t.transform.translation.z = 0.0
        t.transform.rotation = pose_msg.pose.orientation
        
        self.tf_broadcaster.sendTransform(t)


def main(args=None):
    rclpy.init(args=args)
    node = EKFNode()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
