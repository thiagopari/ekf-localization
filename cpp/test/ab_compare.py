#!/usr/bin/env python3
"""Node-level A/B: play one seeded synthetic bag into the Python and the C++ node at once.

    source ~/ekf_cpp_ws/install/setup.bash
    python3 ab_compare.py [--rate 1.0] [--duration 60]

The bag holds noisy /odom twist (wheel odometry), /odom pose = ground truth + 0.3 m noise
(the nodes sample it at 1 Hz as GPS) and a noisy /imu heading. Both nodes run with
add_odom_noise:=false and their own GPS noise zeroed (np.random.normal is patched in the
Python node, gps_noise_std:=0 in the C++ node), so every random draw comes from the bag.
The nodes simulate the 15 s GPS outage themselves (enable_gps_outage:=true, 30 s..45 s).
Reports: max /ekf_pose difference, RMS vs ground truth (EKF, UKF, dead reckoning), CPU %.
"""
import argparse
import math
import os
import signal
import subprocess
import sys
import tempfile
import time

import numpy as np
import psutil
import rclpy
from nav_msgs.msg import Odometry, Path as PathMsg
from geometry_msgs.msg import PoseStamped
from rclpy.qos import QoSProfile
from rclpy.serialization import serialize_message
from sensor_msgs.msg import Imu
import rosbag2_py
from ament_index_python.packages import get_package_prefix

DT = 0.02
T0_NS = 1000 * 10**9
NODE_PARAMS = ['-p', 'add_odom_noise:=false', '-p', 'enable_gps_outage:=true',
               '-p', 'gps_outage_start:=30.0', '-p', 'gps_outage_duration:=15.0']

# Stand-in for ros-jazzy-tf-transformations (not installed here): the two functions
# ekf_node.py uses, with transforms3d's formulas (quat2mat -> mat2euler 'sxyz').
TF_SHIM = '''
import math
def euler_from_quaternion(q, axes='sxyz'):
    x, y, z, w = q
    s = 2.0 / (w*w + x*x + y*y + z*z)
    X, Y, Z = x*s, y*s, z*s
    m00, m10 = 1.0 - (y*Y + z*Z), x*Y + w*Z
    m20, m21, m22 = x*Z - w*Y, y*Z + w*X, 1.0 - (x*X + y*Y)
    cy = math.sqrt(m00*m00 + m10*m10)
    return math.atan2(m21, m22), math.atan2(-m20, cy), math.atan2(m10, m00)
def quaternion_from_euler(ai, aj, ak, axes='sxyz'):
    assert ai == 0 and aj == 0
    return [0.0, 0.0, math.sin(ak / 2.0), math.cos(ak / 2.0)]
'''
PY_NODE = ('import numpy as np; np.random.normal = lambda loc=0.0, scale=1.0, size=None: loc; '
           'from ekf_localization.ekf_node import main; main()')


def make_truth_and_bag(path, duration, seed=7):
    rng = np.random.default_rng(seed)
    n = int(round(duration / DT))
    t = np.arange(n) * DT
    v = np.full(n, 0.3)
    omega = 0.3 * np.sin(0.2 * t)
    truth = np.zeros((n, 3))
    for k in range(1, n):  # midpoint model; odom message k reports the twist of step k
        mid = truth[k - 1, 2] + omega[k] * DT / 2
        truth[k] = truth[k - 1] + [v[k] * DT * math.cos(mid),
                                   v[k] * DT * math.sin(mid), omega[k] * DT]
    v_odom = v + 0.02 + rng.normal(0, 0.02, n)
    w_odom = omega + 0.01 + rng.normal(0, 0.01, n)
    yaw_imu = truth[:, 2] + 0.001 * t + rng.normal(0, 0.02, n)
    gps_xy = truth[:, :2] + rng.normal(0, 0.3, (n, 2))

    writer = rosbag2_py.SequentialWriter()
    writer.open(rosbag2_py.StorageOptions(uri=path, storage_id='mcap'),
                rosbag2_py.ConverterOptions('cdr', 'cdr'))
    for i, (name, typ) in enumerate((('/odom', 'nav_msgs/msg/Odometry'),
                                     ('/imu', 'sensor_msgs/msg/Imu'))):
        writer.create_topic(rosbag2_py.TopicMetadata(
            id=i, name=name, type=typ, serialization_format='cdr'))
    for k in range(n):
        stamp = T0_NS + k * int(DT * 1e9)
        o = Odometry()
        o.header.stamp.sec, o.header.stamp.nanosec = divmod(stamp, 10**9)
        o.header.frame_id = 'odom'
        o.pose.pose.position.x, o.pose.pose.position.y = gps_xy[k]
        o.pose.pose.orientation.z = math.sin(truth[k, 2] / 2)
        o.pose.pose.orientation.w = math.cos(truth[k, 2] / 2)
        o.twist.twist.linear.x, o.twist.twist.angular.z = v_odom[k], w_odom[k]
        writer.write('/odom', serialize_message(o), stamp)
        imu = Imu()  # 10 ms after odom so every node sees predict -> IMU update in order
        imu.header.stamp.sec, imu.header.stamp.nanosec = divmod(stamp + 10**7, 10**9)
        imu.orientation.z = math.sin(yaw_imu[k] / 2)
        imu.orientation.w = math.cos(yaw_imu[k] / 2)
        writer.write('/imu', serialize_message(imu), stamp + 10**7)
    del writer

    # Dead reckoning baseline, as ekf_node.py integrates it (same noisy velocities).
    dr = np.zeros((n, 3))
    for k in range(1, n):
        mid = dr[k - 1, 2] + w_odom[k] * DT / 2
        dr[k] = dr[k - 1] + [v_odom[k] * DT * math.cos(mid),
                             v_odom[k] * DT * math.sin(mid), w_odom[k] * DT]
    return truth, dr


def yaw(q):
    return math.atan2(2 * (q.w * q.z + q.x * q.y), 1 - 2 * (q.y * q.y + q.z * q.z))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--rate', type=float, default=1.0)
    ap.add_argument('--duration', type=float, default=60.0)
    args = ap.parse_args()

    # private discovery: any other ROS 2 process on this machine (domain 0) must not publish into the run
    os.environ['ROS_DOMAIN_ID'] = '77'
    os.environ['ROS_AUTOMATIC_DISCOVERY_RANGE'] = 'LOCALHOST'
    tmp = tempfile.mkdtemp(prefix='ekf_ab_')
    truth, dr = make_truth_and_bag(os.path.join(tmp, 'bag'), args.duration)
    with open(os.path.join(tmp, 'tf_transformations.py'), 'w') as f:
        f.write(TF_SHIM)
    env = dict(os.environ, PYTHONPATH=tmp + os.pathsep + os.environ.get('PYTHONPATH', ''))

    remap = ['-r', 'odom:=/odom', '-r', 'imu:=/imu', '-r', 'scan:=/scan']
    # the C++ binary itself, not `ros2 run`: terminating that wrapper left the node running (and holding pipes)
    cpp_bin = os.path.join(get_package_prefix('ekf_localization_cpp'), 'lib', 'ekf_localization_cpp', 'ekf_node')
    quiet = dict(stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, start_new_session=True)
    nodes = {
        'py': subprocess.Popen([sys.executable, '-c', PY_NODE, '--ros-args', '-r', '__ns:=/py',
                                *remap, *NODE_PARAMS], env=env, **quiet),
        'cpp': subprocess.Popen([cpp_bin, '--ros-args', '-r', '__ns:=/cpp', *remap, *NODE_PARAMS,
                                 '-p', 'gps_noise_std:=0.0'], **quiet),
    }
    try:
        run(args, nodes, truth, dr, tmp, quiet)
    finally:  # never leave nodes behind, whatever happened
        for p in nodes.values():
            if p.poll() is None:
                os.killpg(p.pid, signal.SIGTERM)
        for p in nodes.values():
            try:
                p.wait(timeout=5)
            except subprocess.TimeoutExpired:
                os.killpg(p.pid, signal.SIGKILL)


def run(args, nodes, truth, dr, tmp, quiet):

    rclpy.init()
    rec = rclpy.create_node('ab_recorder')
    got = {name: {'ekf': {}, 'ukf': {}} for name in nodes}
    qos = QoSProfile(depth=5000)

    def stamp_ns(h):
        return h.stamp.sec * 10**9 + h.stamp.nanosec

    def on_pose(out):
        return lambda m: out.update({stamp_ns(m.header): (
            m.pose.position.x, m.pose.position.y, yaw(m.pose.orientation))})

    def on_path(out):  # newest pose of the UKF path
        return lambda m: out.update({stamp_ns(m.header): (
            m.poses[-1].pose.position.x, m.poses[-1].pose.position.y)})

    for name in nodes:
        rec.create_subscription(PoseStamped, f'/{name}/ekf_pose', on_pose(got[name]['ekf']), qos)
        rec.create_subscription(PathMsg, f'/{name}/ukf_path', on_path(got[name]['ukf']), qos)

    deadline = time.time() + 5
    while time.time() < deadline:  # let discovery settle before playback
        rclpy.spin_once(rec, timeout_sec=0.1)
    procs = {n: psutil.Process(p.pid) for n, p in nodes.items()}
    cpu0 = {n: sum(p.cpu_times()[:2]) for n, p in procs.items()}
    t_start = time.time()
    # --delay: let both nodes discover the player's publishers so neither drops early messages
    play = subprocess.Popen(['ros2', 'bag', 'play', os.path.join(tmp, 'bag'), '--rate',
                             str(args.rate), '--delay', '2', '--disable-keyboard-controls'], **quiet)
    play_deadline = time.time() + args.duration / args.rate + 30
    while play.poll() is None:
        rclpy.spin_once(rec, timeout_sec=0.05)
        if time.time() > play_deadline:
            os.killpg(play.pid, signal.SIGKILL)
            raise RuntimeError('ros2 bag play did not finish')
    wall = time.time() - t_start
    cpu = {n: 100 * (sum(p.cpu_times()[:2]) - cpu0[n]) / wall for n, p in procs.items()}
    deadline = time.time() + 2
    while time.time() < deadline:
        rclpy.spin_once(rec, timeout_sec=0.05)

    idx = {T0_NS + k * int(DT * 1e9): k for k in range(len(truth))}
    common = sorted(set(got['py']['ekf']) & set(got['cpp']['ekf']))
    diff = np.array([np.subtract(got['py']['ekf'][s], got['cpp']['ekf'][s]) for s in common])
    diff[:, 2] = (diff[:, 2] + np.pi) % (2 * np.pi) - np.pi

    def rms(est):
        e = [np.hypot(*(np.array(xy[:2]) - truth[idx[s], :2])) for s, xy in est.items()]
        return math.sqrt(np.mean(np.square(e)))

    print(f'bag: {len(truth)} odom + {len(truth)} imu msgs, {args.duration:.0f} s, '
          f'played at {args.rate}x in {wall:.1f} s')
    print(f"/ekf_pose received: python {len(got['py']['ekf'])}, c++ {len(got['cpp']['ekf'])}, "
          f'common stamps {len(common)}')
    big = np.flatnonzero(np.abs(diff).max(axis=1) > 1e-9)
    print(f'max |python - c++| on /ekf_pose: x {np.abs(diff[:, 0]).max():.3g} m, '
          f'y {np.abs(diff[:, 1]).max():.3g} m, yaw {np.abs(diff[:, 2]).max():.3g} rad; '
          f'steps above 1e-9: {len(big)}' + (f' (first at step {big[0]})' if len(big) else ''))
    for name in nodes:
        print(f"{name:>6}: EKF RMS {rms(got[name]['ekf']):.3f} m, UKF RMS "
              f"{rms(got[name]['ukf']):.3f} m, CPU {cpu[name]:.1f} %")
    print(f'dead reckoning RMS {math.sqrt(np.mean(np.sum((dr[:, :2] - truth[:, :2])**2, 1))):.3f} m')
    rec.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
