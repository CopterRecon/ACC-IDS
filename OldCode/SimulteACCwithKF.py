import numpy as np
import matplotlib.pyplot as plt

def simulate_acc_kf():
    # =======================
    # Simulation parameters
    # =======================
    dt = 0.1          # time step [s]
    T = 40.0          # total time [s]
    N = int(T / dt)

    # -----------------------
    # True states (arrays)
    # -----------------------
    d   = np.zeros(N)  # gap distance: lead - host [m]
    v_l = np.zeros(N)  # lead vehicle speed [m/s]
    v_h = np.zeros(N)  # host vehicle speed [m/s]
    a_l = np.zeros(N)  # lead acceleration [m/s^2]
    a_h = np.zeros(N)  # host acceleration command [m/s^2]

    # Initial conditions
    d[0]   = 40.0      # initial gap [m]
    v_l[0] = 25.0      # lead speed ~90 km/h
    v_h[0] = 20.0      # host speed ~72 km/h

    # =======================
    # ACC controller params
    # =======================
    d0   = 5.0         # standstill distance [m]
    tau  = 1.5         # desired time headway [s]
    Kp   = 0.4         # distance error gain
    Kv   = 0.8         # relative speed gain
    a_min, a_max = -4.0, 2.0   # accel limits [m/s^2] (braking, throttle)

    # =======================
    # Kalman filter setup
    # State x = [ d, v_l, v_h ]^T
    # =======================
    A = np.array([
        [1.0, dt, -dt],
        [0.0, 1.0,  0.0],
        [0.0, 0.0,  1.0]
    ])

    # Measurements: z = [ d_meas, v_rel_meas, v_h_meas ]^T
    # where v_rel = v_l - v_h
    H = np.array([
        [1.0, 0.0,  0.0],   # d_meas
        [0.0, 1.0, -1.0],   # v_rel_meas
        [0.0, 0.0,  1.0]    # v_h_meas
    ])

    # Process noise covariance (tune as needed)
    Q = np.diag([0.5, 0.2, 0.2]) * 0.01

    # Measurement noise covariance (tune as needed)
    R = np.diag([
        4.0,   # distance noise variance (m^2)
        1.0,   # relative speed noise variance ((m/s)^2)
        0.5    # host speed noise variance ((m/s)^2)
    ])

    # Initial state estimate and covariance
    x_hat = np.array([d[0], v_l[0], v_h[0]], dtype=float)
    P = np.eye(3) * 10.0

    # History of estimates
    x_hat_hist = np.zeros((N, 3))
    x_hat_hist[0] = x_hat

    # Measurement noise std devs
    sigma_d    = np.sqrt(R[0, 0])
    sigma_vrel = np.sqrt(R[1, 1])
    sigma_vh   = np.sqrt(R[2, 2])

    # =======================
    # Main simulation loop
    # =======================
    for k in range(N - 1):
        t = k * dt

        # -----------------------
        # Lead vehicle accel profile
        # (brakes from t=10s to t=15s)
        # -----------------------
        if 10.0 <= t <= 15.0:
            a_l[k] = -2.0   # m/s^2 braking
        else:
            a_l[k] = 0.0

        # -----------------------
        # ACC control using KF estimates
        # -----------------------
        d_hat, v_l_hat, v_h_hat = x_hat  # estimated states

        # Desired gap: d_des = d0 + tau * v_h
        d_des = d0 + tau * v_h_hat
        e_d   = d_hat - d_des           # distance error
        v_rel_hat = v_l_hat - v_h_hat   # estimated relative speed

        # Simple PD-like ACC law
        a_cmd = Kp * e_d + Kv * v_rel_hat
        a_cmd = np.clip(a_cmd, a_min, a_max)
        a_h[k] = a_cmd

        # -----------------------
        # True vehicle dynamics
        # -----------------------
        # Lead
        v_l[k+1] = max(0.0, v_l[k] + a_l[k] * dt)

        # Host
        v_h[k+1] = max(0.0, v_h[k] + a_h[k] * dt)

        # Gap distance
        d[k+1] = max(0.0, d[k] + (v_l[k] - v_h[k]) * dt)

        # -----------------------
        # Measurements with noise
        # -----------------------
        v_rel_true = v_l[k] - v_h[k]

        z = np.array([
            d[k+1] + np.random.randn() * sigma_d,
            v_rel_true + np.random.randn() * sigma_vrel,
            v_h[k+1] + np.random.randn() * sigma_vh
        ])

        # -----------------------
        # Kalman Filter: predict
        # -----------------------
        x_hat = A @ x_hat
        P = A @ P @ A.T + Q

        # -----------------------
        # Kalman Filter: update
        # -----------------------
        S = H @ P @ H.T + R
        K_gain = P @ H.T @ np.linalg.inv(S)
        y = z - H @ x_hat          # innovation
        x_hat = x_hat + K_gain @ y
        P = (np.eye(3) - K_gain @ H) @ P

        x_hat_hist[k+1] = x_hat

    return dt, d, v_l, v_h, x_hat_hist, a_l, a_h

if __name__ == "__main__":
    dt, d, v_l, v_h, x_hat_hist, a_l, a_h = simulate_acc_kf()

    N = len(d)
    t = np.arange(N) * dt

    # =======================
    # Plot results
    # =======================
    plt.figure(figsize=(10, 6))
    plt.plot(t, d, label="Gap distance d (true)")
    plt.plot(t, x_hat_hist[:, 0], label="Gap distance d (estimated)", linestyle="--")
    plt.xlabel("Time [s]")
    plt.ylabel("Distance [m]")
    plt.title("True vs Estimated Gap Distance")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    plt.figure(figsize=(10, 6))
    plt.plot(t, v_l, label="Lead speed v_l (true)")
    plt.plot(t, x_hat_hist[:, 1], label="Lead speed v_l (estimated)", linestyle="--")
    plt.plot(t, v_h, label="Host speed v_h (true)")
    plt.plot(t, x_hat_hist[:, 2], label="Host speed v_h (estimated)", linestyle="--")
    plt.xlabel("Time [s]")
    plt.ylabel("Speed [m/s]")
    plt.title("True vs Estimated Speeds")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    plt.figure(figsize=(10, 6))
    plt.plot(t, a_l, label="Lead accel a_l (profile)")
    plt.plot(t, a_h, label="Host accel a_h (ACC command)")
    plt.xlabel("Time [s]")
    plt.ylabel("Acceleration [m/s²]")
    plt.title("Accelerations")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    plt.show()