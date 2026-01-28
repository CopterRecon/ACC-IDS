import numpy as np
from .constants import KMH_TO_MS
from typing import Tuple

''' 
     This method computes the safe distance for the host vehicle
'''
def safe_distance(v_host_kmh: float, h: float, u: float) -> float:
    """
    d = v*h + v^2/(2u) in meters (host-only)
    """
    v = v_host_kmh * KMH_TO_MS
    return (v * h) + (v * v) / (2.0 * u)

'''
     This computes the distance between the lead and host vehicle
'''
def gap_update(prev_gap_m: float, v_host_kmh: float, v_lead_kmh: float, dt: float) -> float:
    """
    d_{k+1} = d_k + (v_lead - v_host) * dt, in meters.
    """
    return prev_gap_m + ((v_lead_kmh - v_host_kmh) * KMH_TO_MS * dt)

'''
    This method computes the threshold speed for the host vehicle
'''
def compute_v_thr(gap_m, v_lead_kmh, u, h, dt) -> float:
    
    p = 0.039 / u
    b = KMH_TO_MS * (h + dt)
    c = -(gap_m + KMH_TO_MS * dt * v_lead_kmh)

    disc = b*b - 4.0*p*c
    disc = np.maximum(disc, 0.0)
    v_thr = (-b + np.sqrt(disc)) / (2.0 * p)
    return float(v_thr)


'''
'''
def required_gap_eq17(v_host_kmh, v_lead_kmh, u, h, dt) -> float:
    """
    Rearranged Eq(17) boundary -> required gap in meters, consistent with compute_v_thr.
    """
    return (KMH_TO_MS*(h+dt)*v_host_kmh) + ((KMH_TO_MS**2)/(2*u))*(v_host_kmh**2) - (KMH_TO_MS*dt*v_lead_kmh)


'''
'''
def g_eq17(v_host_kmh, v_lead_kmh, gap_m, u, h, dt):
    """
    g(v) = p v^2 + b v + c  where g(v)=0 at the threshold speed.
    Using exact unit conversions (km/h -> m/s).
    """
    p = (KMH_TO_MS**2) / (2.0 * u)
    b = KMH_TO_MS * (h + dt)
    c = -(gap_m + KMH_TO_MS * dt * v_lead_kmh)
    return p * (np.asarray(v_host_kmh) ** 2) + b * np.asarray(v_host_kmh) + c



'''
    Lemma 4.2: measurement threshold z_thr at k+1 such that the KF-updated
    host-speed estimate exceeds the threshold speed v_thr.
'''

def kalman_gain_scalar(P_pred: float, R: float) -> float:
    """
    Scalar Kalman gain for a 1D Kalman filter:
        K = P_pred / (P_pred + R)

    P_pred : predicted covariance (typically >= 0)
    R      : measurement noise variance (typically > 0)
    """
    denom = P_pred + R
    if denom <= 0:
        raise ValueError("P_pred + R must be > 0.")
    return float(P_pred / denom)

def lemma42_z_threshold(
    v_pred_k1: float,   # v_h(k+1|k) in km/h
    P_pred_k1: float,   # P(k+1|k)
    R: float,           # measurement noise variance
    v_thr: float,       # host speed threshold
) -> Tuple[float, float]:
    """
    Lemma 4.2 (paper):
      For a 1D KF update at k+1,
          v_hat(k+1|k+1) = (1 - K_{k+1}) * v_hat(k+1|k) + K_{k+1} * z(k+1)

      The updated estimate exceeds v_thr if:
          z(k+1) > z_thr(k+1)

      where:
          z_thr(k+1) = ( v_thr(k+1) - (1 - K_{k+1}) * v_hat(k+1|k) ) / K_{k+1}

    Inputs here use:
        v_pred_k1 = v_hat(k+1|k)   (km/h)
        P_pred_k1 = P(k+1|k)
        R         = measurement noise variance
        v_thr     = speed threshold (km/h)

    Returns:
        (z_thr_k1, K_k1)
    """
    K_k1 = kalman_gain_scalar(P_pred_k1, R)

    EPS_K = 1e-12
    if K_k1 <= EPS_K:
        return float("inf"), float(K_k1)

    z_thr_k1 = (v_thr - (1.0 - K_k1) * v_pred_k1) / K_k1
    return float(z_thr_k1), float(K_k1)


def lemma42_is_violation(
    z_meas_k1: float,
    v_pred_k1: float,
    P_pred_k1: float,
    R: float,
    v_thr: float,
) -> bool:
    """
    Convenience predicate for Lemma 4.2 inequality:
        True iff z_meas_k1 > z_thr_k1
    """
    z_thr_k1, _K = lemma42_z_threshold(v_pred_k1, P_pred_k1, R, v_thr)
    return bool(z_meas_k1 > z_thr_k1)

