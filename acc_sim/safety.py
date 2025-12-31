# acc_sim/safety.py

import numpy as np
from .constants import KMH_TO_MS

def safe_distance_simple(v_host_kmh: float, h: float, u: float) -> float:
    """
    d = v*h + v^2/(2u) in meters (host-only)
    """
    v = v_host_kmh * KMH_TO_MS
    return (v * h) + (v * v) / (2.0 * u)

def gap_update(prev_gap_m: float, v_host_kmh: float, v_lead_kmh: float, dt: float) -> float:
    """
    d_{k+1} = d_k + (v_lead - v_host) * dt, in meters.
    """
    return prev_gap_m + ((v_lead_kmh - v_host_kmh) * KMH_TO_MS * dt)

def compute_v_thr(gap_m, v_lead_kmh, u, h, dt) -> float:
    """
    Threshold host speed v_thr (km/h) from quadratic:
      (KMH_TO_MS^2/(2u)) v^2 + (KMH_TO_MS*(h+dt)) v - (gap + KMH_TO_MS*dt*v_lead) = 0
    """
    p = (KMH_TO_MS**2) / (2.0 * u)
    b = KMH_TO_MS * (h + dt)
    c = -(gap_m + KMH_TO_MS * dt * v_lead_kmh)

    disc = b*b - 4.0*p*c
    disc = np.maximum(disc, 0.0)
    v_thr = (-b + np.sqrt(disc)) / (2.0 * p)
    return float(v_thr)

def required_gap_eq17(v_host_kmh, v_lead_kmh, u, h, dt) -> float:
    """
    Rearranged Eq(17) boundary -> required gap in meters, consistent with compute_v_thr.
    """
    return (KMH_TO_MS*(h+dt)*v_host_kmh) + ((KMH_TO_MS**2)/(2*u))*(v_host_kmh**2) - (KMH_TO_MS*dt*v_lead_kmh)


def g_eq17(v_host_kmh, v_lead_kmh, gap_m, u, h, dt):
    """
    g(v) = p v^2 + b v + c  where g(v)=0 at the threshold speed.
    Using exact unit conversions (km/h -> m/s).
    """
    p = (KMH_TO_MS**2) / (2.0 * u)
    b = KMH_TO_MS * (h + dt)
    c = -(gap_m + KMH_TO_MS * dt * v_lead_kmh)
    return p * (np.asarray(v_host_kmh) ** 2) + b * np.asarray(v_host_kmh) + c

# acc_sim/safety.py

'''
    Lemma 4.2: measurement threshold z_thr at k+1 such that the KF-updated
    host-speed estimate exceeds the threshold speed v_thr.
'''

def lemma42_z_threshold(
    v_pred_k1: float,   # v_h(k+1|k) in km/h
    P_pred_k1: float,   # P(k+1|k)
    R: float,           # measurement noise variance
    v_thr: float,       # host speed threshold
):
    """
    Returns (z_thr_k1, v_thr_k1, K_k1)
    """
    if P_pred_k1 + R <= 0:
        raise ValueError("P_pred_k1 + R must be > 0.")
    K_k1 = P_pred_k1 / (P_pred_k1 + R)
    if K_k1 <= 0:
        raise ValueError("Kalman gain is 0; z_thr is not finite.")
    
    # Lemma 4.2: z_thr = (v_thr - (1-K)*v_pred)/K
    z_thr_k1 = (v_thr - (1.0 - K_k1) * v_pred_k1) / K_k1
    return float(z_thr_k1), float(K_k1)