#!/usr/bin/env python3
"""Run the ORIGINAL ekf.py / ukf.py on a seeded synthetic sequence and write a golden CSV.

    make_golden.py OUT.csv            # write the CSV the C++ golden test replays
    make_golden.py OUT.csv --bench    # also time the Python filters on the same sequence

Each row is one step: predict(v, omega, dt); update_imu(z_imu); update_gps(gx, gy) if gps,
followed by the EKF and UKF state and row-major covariance after that step.
"""
import os
import sys
import time
from pathlib import Path

os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')  # before numpy import; tiny matrices
import numpy as np  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # repo root
from ekf_localization.ekf import EKFLocalization  # noqa: E402
from ekf_localization.ukf import UKFLocalization  # noqa: E402

SEED = 42
DT = 0.02           # 50 Hz odom + IMU
DURATION = 60.0
OUTAGE = (30.0, 45.0)  # 15 s GPS outage, same as the node defaults


def make_sequence():
    rng = np.random.default_rng(SEED)
    n = int(round(DURATION / DT))
    dt = DT + rng.uniform(-0.002, 0.002, n)          # timestamp jitter
    t = np.cumsum(dt)
    v = np.full(n, 0.3)
    omega = 0.4 + 0.3 * np.sin(0.2 * t)               # net turn: heading wraps through +-pi

    truth = np.zeros((n, 3))
    s = np.zeros(3)
    for k in range(n):                                 # same midpoint model as ekf.py
        mid = s[2] + omega[k] * dt[k] / 2
        s = s + [v[k] * dt[k] * np.cos(mid), v[k] * dt[k] * np.sin(mid), omega[k] * dt[k]]
        truth[k] = s

    v_odom = v + 0.02 + rng.normal(0, 0.02, n)         # biased, noisy wheel odometry
    w_odom = omega + 0.01 + rng.normal(0, 0.01, n)
    z_imu = truth[:, 2] + 0.001 * t + rng.normal(0, 0.02, n)
    z_imu = (z_imu + np.pi) % (2 * np.pi) - np.pi      # IMU yaw arrives wrapped
    gps = (np.floor(t) > np.floor(t - dt)) & ~((t >= OUTAGE[0]) & (t < OUTAGE[1]))
    g = truth[:, :2] + rng.normal(0, 0.5, (n, 2))
    return np.column_stack([v_odom, w_odom, dt, z_imu, gps, g])


def run(filt, seq):
    out = np.empty((len(seq), 12))
    for k, (v, w, dt, zi, gps, gx, gy) in enumerate(seq):
        filt.predict(v, w, dt)
        filt.update_imu(zi)
        if gps:
            filt.update_gps([gx, gy])
        out[k, :3] = filt.x
        out[k, 3:] = filt.P.ravel()
    return out


def bench(make_filter, seq, reps=20):
    rows = seq.tolist()                                 # plain floats: no numpy indexing in the loop
    times = []
    for _ in range(reps):
        f = make_filter()
        t0 = time.perf_counter_ns()
        for v, w, dt, zi, gps, gx, gy in rows:
            f.predict(v, w, dt)
            f.update_imu(zi)
            if gps:
                f.update_gps([gx, gy])
        times.append(time.perf_counter_ns() - t0)
    return np.median(times) / len(rows) / 1e3


def main():
    out_csv = sys.argv[1]
    seq = make_sequence()
    cols = ['v', 'omega', 'dt', 'z_imu', 'gps', 'gx', 'gy']
    for name in ('ekf', 'ukf'):
        cols += [f'{name}_{c}' for c in ('x', 'y', 'th')]
        cols += [f'{name}_P{i}{j}' for i in range(3) for j in range(3)]
    data = np.hstack([seq, run(EKFLocalization(), seq), run(UKFLocalization(), seq)])
    np.savetxt(out_csv, data, delimiter=',', fmt='%.17g', header=','.join(cols), comments='')
    print(f'wrote {len(seq)} steps ({int(seq[:, 4].sum())} GPS fixes) to {out_csv}')

    if '--bench' in sys.argv:
        for name, cls in (('EKF', EKFLocalization), ('UKF', UKFLocalization)):
            print(f'python {name}: {bench(cls, seq):.2f} us/step (median of 20 runs)')


if __name__ == '__main__':
    main()
