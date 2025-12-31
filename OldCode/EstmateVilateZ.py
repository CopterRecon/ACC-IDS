import numpy as np

def compute_v_thr(d_hat_k, v_l_hat_k, u, h, dt):
    """
    Compute v_thr at step k+1 given:
    - d_hat_k     : estimated gap d_k
    - v_l_hat_k   : estimated lead speed v_l,k
    - u           : deceleration parameter
    - h           : headway time
    - dt          : sampling time Δt
    """
    b = 0.278 * h + dt
    inside = b**2 + (0.156 / u) * (d_hat_k + v_l_hat_k * dt)
    v_thr_k1 = (u / 0.078) * (-b + np.sqrt(inside))
    return v_thr_k1


def compute_z_threshold(v_h_pred_k1, K_k1, d_hat_k, v_l_hat_k, u, h, dt):
    """
    Compute the threshold on the measurement z_{k+1} (measured host speed)
    such that the updated Kalman-filter estimate v_h(t+1|t+1) exceeds v_thr.

    Inputs:
    - v_h_pred_k1 : predicted host speed  v_h(t+1 | t)
    - K_k1        : Kalman gain at time k+1 (scalar, for speed update)
    - d_hat_k     : estimated gap d_k at time k
    - v_l_hat_k   : estimated lead speed v_l,k at time k
    - u           : deceleration parameter
    - h           : headway time
    - dt          : sampling time Δt

    Returns:
    - z_thr_k1    : minimum measurement z_{k+1} such that v_h(t+1|t+1) > v_thr
    """
    # 1) Compute threshold speed v_thr at k+1
    v_thr_k1 = compute_v_thr(d_hat_k, v_l_hat_k, u, h, dt)

    # 2) Compute measurement threshold z_{k+1}
    #    z_thr = [v_thr - (1 - K)*v_pred] / K
    z_thr_k1 = (v_thr_k1 - (1.0 - K_k1) * v_h_pred_k1) / K_k1

    return z_thr_k1


# Example usage:
if __name__ == "__main__":
    # Example parameters (replace with your KF values)
    u   = 3.4      # m/s^2
    h   = 2.0      # s
    dt  = 1.0      # s
    d_hat_k    = 170.0   # m, estimated gap at time k
    v_l_hat_k  = 30.0    # m/s, estimated lead speed
    v_h_pred_k1 = 25.0   # m/s, predicted host speed at time k+1 (v_h(t+1|t))
    K_k1 = 0.6           # Kalman gain at time k+1 (for speed)

    z_thr = compute_z_threshold(v_h_pred_k1, K_k1, d_hat_k, v_l_hat_k, u, h, dt)
    print("Measurement threshold z_{k+1}:", z_thr)
    # If your actual measured speed z_{k+1} > z_thr, then v_h(t+1|t+1) > v_thr.