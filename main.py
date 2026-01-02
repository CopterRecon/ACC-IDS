# Filename: acc_sim/main.py
# Author: Lotfi ben Othmane <lotfi.benothmane@unt.edu> 
# Created: 2025-12-29
# Description: Implements the functions for plotting the vehicles speed metrics and distances 
# License: -

import random
import numpy as np
import datetime

from acc_sim.vehicle import VehicleParams, VehicleState, VehicleModel
from acc_sim.controllers import LeadCruiseController, LeadControllerParams, HostACCController, HostControllerParams
from acc_sim.safety import safe_distance
from acc_sim.simulator import TwoCarSimulator, SimConfig
from acc_sim.plots import plot_gap_vs_safedistance, plot_host_vs_threshold, plot_speeds, plot_speed_threshold, plot_distance_gap_vs_speed_threshold, plot_Measuredhost_vs_threshold,plot_Attackhost_vs_threshold
from acc_sim.filters import KalmanFilter
from acc_sim.constants import KMH_TO_MS, MS_TO_KMH


def main(scenario):
    random.seed(0)
    np.random.seed(0)

    # --- Config ---
    cfg = SimConfig(h=2.0, dt=0.1, steps=1000, stop_gap_m=2.0)

    # --- Vehicles ---
    host_params = VehicleParams(mass=1200.0, u_brake=3.4, a_max=1.5)
    lead_params = VehicleParams(mass=1200.0, u_brake=3.4, a_max=1.5)

    host = VehicleModel("Host", host_params, VehicleState(speed_kmh=25.0))
    lead = VehicleModel("Lead", lead_params, VehicleState(speed_kmh=30.0))

    # --- Controllers ---
    lead_ctrl = LeadCruiseController(LeadControllerParams(v_set_kmh=90.0, u=lead_params.u_brake))
    host_ctrl = HostACCController(HostControllerParams(cruise_kmh=120.0, u=host_params.u_brake))

    # --- Initial gap (based on simple safe distance) ---
    init_safe = safe_distance(host.s.speed_kmh, h=cfg.h, u=host_params.u_brake)
    init_gap = 1.1 * init_safe
    print(f"Initial safe distance: {init_safe:.3f} m, initial gap: {init_gap:.3f} m")

    sim = TwoCarSimulator(host, lead, host_ctrl, lead_ctrl, cfg, init_gap_m=init_gap)
    df = sim.run(scenario)

    print("Safe distance violations:", df.attrs.get("safe_distance_violations"))
    print("Threshold violations:", df.attrs.get("threshold_violations"))
    
    # Write the dataframe log to file for debugging
    df.to_csv('.\output\SimulationOutput'+ datetime.datetime.now().strftime("%Y-%m-%d %H-%M-%S") +'.txt', sep='\t', index=False, header=True)
    
    # --- Plots ---
    plot_speeds(df)
    plot_gap_vs_safedistance(df)
    plot_host_vs_threshold(df)
    #plot_speed_threshold(lead_speeds_kmh=(20, 70, 110),d_min=0,d_max=80,u=3.4,h=2.0,dt=0.1)
    #plot_distance_gap_vs_speed_threshold(v_lead_kmh=30.0,gap_m=41.0,u=host_params.u_brake,h=cfg.h,dt=cfg.dt,v_max_kmh=140)
    
    plot_Measuredhost_vs_threshold(df)
    
    # Plots speed attacks and relation to other data
    plot_Attackhost_vs_threshold(df)
    # Optional: save
    # df.to_csv("simulation.csv", index=False)

if __name__ == "__main__":
    
    # Simulating Scenario of no injection of faulty speed
    #main(1)
    
    # Simulating Scenario of random injection of faulty speed
    #main(2)
    
    # Simulating Scenario of attack injection of faulty speed
    main(3)