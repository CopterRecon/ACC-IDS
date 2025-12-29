import numpy as np

# --- Parameters ---
dt = 0.1
A = np.array([[1.0]])      # state is v_h
H = np.array([[1.0]])      # we measure v_h directly
Q = np.array([[0.01]])     # process noise covariance
R = np.array([[0.5]])      # measurement noise covariance

# Initial state
x = np.array([[20.0]])     # initial estimate of v_h (m/s)
P = np.array([[1.0]])      # initial covariance

# --- Generate 10 random measurements around true speed of ~20 m/s ---
np.random.seed(0)
measured_speed = 20 + np.random.normal(0, 2, size=10)

print("Measured speeds:", measured_speed)

# --- Kalman filter loop ---
estimates = []
print("0- Q:", Q, "R:", R, "P:", P, "A:",A)

for z in measured_speed:

    print("0- Q:", Q, "R:", R, "P:", P, "A:",A, "z:", z)
    # Prediction
    x = A @ x
    P = A @ P @ A.T + Q

    # Update
    K = P @ H.T @ np.linalg.inv(H @ P @ H.T + R)
    x = x + K @ (z - H @ x)
    P = (np.eye(1) - K @ H) @ P

    print("0- Q:", Q, "R:", R, "P:", P, "K:", K, "A:",A, "z:", z)
    estimates.append(x[0, 0])

# Print results
print("Estimated v_h values:", estimates)
print("Q:", Q, "R:", R, "P:", P, "K:", K)