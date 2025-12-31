
# Filename: acc_sim/plots.py
# Author: Lotfi ben Othmane <lotfi.benothmane@unt.edu> 
# Created: 2025-12-29
# Description: Implements the functions for plotting the vehicles speed metrics and distances 
# License: -

import matplotlib.pyplot as plt
import numpy as np
from .safety import *
from .constants import KMH_TO_MS


def plot_speed_threshold(
    lead_speeds_kmh=(20, 70, 110),
    d_min=0.0,
    d_max=80.0,
    n=500,
    u=3.4,
    h=2.0,
    dt=0.1,
):
    """
    Plot v_thr (km/h) vs gap distance d (m) for several lead speeds.
    """
    d_vals = np.linspace(d_min, d_max, int(n))
    
    plt.figure()
    for v_l in lead_speeds_kmh:
        thr_vals = [compute_v_thr(d, v_l, u=u, h=h, dt=dt) for d in d_vals]
        plt.plot(d_vals, thr_vals, label=f"Lead speed = {v_l} km/h")
        
    plt.xlabel("Gap distance d(t) (m)")
    plt.ylabel("Threshold host speed v_thr (km/h)")
    plt.grid(True)
    plt.legend()
    plt.show()
    
def plot_gap_vs_required(df):
    plt.figure()
    plt.plot(df["step"], df["gap_m"], label="Gap d (m)")
    plt.plot(df["step"], df["d_req_m"], label="Required gap d_req (m)")
    plt.xlabel("Time step")
    plt.ylabel("Distance (m)")
    plt.grid(True)
    plt.legend()
    plt.show()

def plot_host_vs_threshold(df):
    plt.figure()
    plt.plot(df["step"], df["v_host_kmh"], label="Host speed v_h (km/h)")
    plt.plot(df["step"], df["v_thr_kmh"], label="Threshold speed v_thr (km/h)")
    plt.xlabel("Time step")
    plt.ylabel("Speed (km/h)")
    plt.grid(True)
    plt.legend()
    plt.show()

def plot_Measuredhost_vs_threshold(df):
    plt.figure()
    plt.plot(df["step"], df["v_host_kmh"], marker ="+", markevery=20, label="Host speed v_h (km/h)")
    plt.plot(df["step"], df["v_thr_kmh"], label="Threshold speed v_thr (km/h)")
    plt.plot(df["step"], df["z_thr"], label="Measured speed z_thr (km/h)")
    plt.plot(df["step"], df["v_filtFault_kmh"], label="Host speed with KF v_h+(km/h)")
    plt.plot(df["step"], df["v_lead_kmh"], label="Lead speed v_l(km/h)")
    plt.xlabel("Time step")
    plt.ylabel("Speed (km/h)")
    plt.grid(True)
    plt.legend()
    plt.show()
    
def plot_Attackhost_vs_threshold(df):
    plt.figure()
    plt.plot(df["step"], df["v_host_kmh"], marker ="+", markevery=20, label="Host speed v_h (km/h)")
    plt.plot(df["step"], df["v_thr_kmh"], label="Threshold speed v_thr (km/h)")
    plt.plot(df["step"], df["v_filtAttack_kmh"], label="Filtered speed ")
    plt.plot(df["step"], df["z_attack_kmh"], label="Measured speed with attacks (km/h)")
    plt.plot(df["step"], df["v_lead_kmh"], label="Lead speed v_l(km/h)")
    plt.xlabel("Time step")
    plt.ylabel("Speed (km/h)")
    plt.grid(True)
    plt.legend()
    plt.show()
    
def plot_speeds(df):
    plt.figure()
    plt.plot(df["step"], df["v_host_kmh"], label="Host (km/h)")
    plt.plot(df["step"], df["v_lead_kmh"], label="Lead (km/h)")
    plt.xlabel("Time step")
    plt.ylabel("Speed (km/h)")
    plt.grid(True)
    plt.legend()
    plt.show()
    
def plot_distance_gap_vs_speed_threshold(
    v_lead_kmh: float,
    gap_m: float,
    u: float,
    h: float,
    dt: float,
    v_min_kmh: float = 0.0,
    v_max_kmh: float = 120.0,
    n: int = 800,
):
    """
    Plot g(v) (Eq. 17) vs host speed, and mark v_thr where g(v)=0.
    """
    v = np.linspace(v_min_kmh, v_max_kmh, int(n))
    
    # Threshold speed from the same model
    v_thr = compute_v_thr(gap_m, v_lead_kmh, u=u, h=h, dt=dt)
    
    y = g_eq17(v, v_lead_kmh, gap_m, u=u, h=h, dt=dt)
    
    plt.figure()
    plt.plot(v, y, label="g(v) (Eq. 17)")
    plt.axhline(0)
    plt.axvline(v_thr, linestyle="--", label=f"v_thr ≈ {v_thr:.2f} km/h")
    plt.xlabel("Host speed v (km/h)")
    plt.ylabel("g(v) = d_safe(v) - gap (m)")
    plt.grid(True)
    plt.legend()
    plt.show()