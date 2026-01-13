# Filename: acc_sim/main.py
# Author: Lotfi ben Othmane <lotfi.benothmane@unt.edu>
# Modified by: Naga Prudhvi Mareedu
# Description: ACC–IDS simulation with selectable scenarios (1–4)
# License: -

import random
import numpy as np
import datetime
import os

from acc_sim.vehicle import VehicleParams, VehicleState, VehicleModel
from acc_sim.controllers import (
    LeadCruiseController, LeadControllerParams,
    HostACCController, HostControllerParams
)
from acc_sim.safety import safe_distance
from acc_sim.simulator import TwoCarSimulator, SimConfig
from acc_sim.plots import (
    plot_gap_vs_safedistance,
    plot_host_vs_threshold,
    plot_speeds,
    plot_Measuredhost_vs_threshold,
    plot_Attackhost_vs_threshold
)

# ==========================================================
# USER SETTINGS (CHANGE ONLY THESE)
# ==========================================================
SCENARIO_TO_RUN = 4    # 1, 2, 3, or 4
IDS_ACCURACY = 0.90    # used only for scenario 4
N_RUNS = 100            # Monte-Carlo runs
# ==========================================================


def main(scenario, ids_accuracy=1.0):

    random.seed()
    np.random.seed()

    # --- Config ---
    cfg = SimConfig(h=2.0, dt=0.1, steps=1000, stop_gap_m=2.0)

    # --- Vehicles ---
    host_params = VehicleParams(mass=1200.0, u_brake=3.4, a_max=1.5)
    lead_params = VehicleParams(mass=1200.0, u_brake=3.4, a_max=1.5)

    host = VehicleModel("Host", host_params, VehicleState(speed_kmh=25.0))
    lead = VehicleModel("Lead", lead_params, VehicleState(speed_kmh=30.0))

    # --- Controllers ---
    lead_ctrl = LeadCruiseController(
        LeadControllerParams(v_set_kmh=90.0, u=lead_params.u_brake)
    )
    host_ctrl = HostACCController(
        HostControllerParams(cruise_kmh=120.0, u=host_params.u_brake)
    )

    # --- Initial gap ---
    init_safe = safe_distance(
        host.s.speed_kmh, h=cfg.h, u=host_params.u_brake
    )
    init_gap = 1.1 * init_safe

    sim = TwoCarSimulator(
        host, lead, host_ctrl, lead_ctrl,
        cfg, init_gap_m=init_gap,
        ids_accuracy=ids_accuracy
    )

    df, Ncrashes = sim.run(scenario)

    print(f"[Scenario {scenario}] IDS accuracy: {ids_accuracy}")
    print("Safe distance violations:", df.attrs.get("safe_distance_violations"))
    print("Speed threshold violations:", df.attrs.get("Speed threshold_violations"))
    print("Z threshold violations:", df.attrs.get("Z threshold_violations"))
    print("Crashes:", df.attrs.get("Crashes"))

    # --- Save output ---
    os.makedirs("output", exist_ok=True)
    filename = (
        f"output/Scenario{scenario}_"
        f"{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.csv"
    )
    df.to_csv(filename, index=False)

    # --- Plots ---
    if scenario == 1:
        plot_speeds(df)
        plot_gap_vs_safedistance(df)

    elif scenario == 2:
        plot_host_vs_threshold(df)
        plot_Measuredhost_vs_threshold(df)
        plot_gap_vs_safedistance(df)

    elif scenario >= 3:
        plot_Attackhost_vs_threshold(df)
        plot_gap_vs_safedistance(df)

    return Ncrashes


# ==========================================================
# MAIN EXECUTION
# ==========================================================
if __name__ == "__main__":

    total_crashes = 0

    for _ in range(N_RUNS):
        total_crashes += main(
            scenario=SCENARIO_TO_RUN,
            ids_accuracy=IDS_ACCURACY
        )

    print("===================================")
    print("Scenario:", SCENARIO_TO_RUN)
    print("IDS Accuracy (α):", IDS_ACCURACY)
    print("Total runs:", N_RUNS)
    print("Total crashes:", total_crashes)
    print("Crash ratio:", total_crashes / N_RUNS)
    print("===================================")
