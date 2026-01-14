# Filename: main.py

import numpy as np
import matplotlib.pyplot as plt

from acc_sim.vehicle import VehicleParams, VehicleState, VehicleModel
from acc_sim.controllers import (
    LeadCruiseController, LeadControllerParams,
    HostACCController, HostControllerParams
)
from acc_sim.safety import safe_distance
from acc_sim.simulator import TwoCarSimulator, SimConfig


# ================= SETTINGS =================
SCENARIO = 4                    # <<< CHANGE: scenario explicitly defined
N_RUNS = 20                     # <<< CHANGE: multiple runs per accuracy
ACCURACIES = np.linspace(0.1, 1.0, 10)  # <<< CHANGE: sweep accuracy
# ============================================


def run_single_simulation(ids_accuracy, scenario):
    """
    Runs ONE independent simulation.
    Returns:
        (accuracy, crash_time, scenario)
    """

    cfg = SimConfig()

    host_params = VehicleParams(1200, 3.4, 1.5)
    lead_params = VehicleParams(1200, 3.4, 1.5)

    host = VehicleModel("Host", host_params, VehicleState(25))
    lead = VehicleModel("Lead", lead_params, VehicleState(30))

    host_ctrl = HostACCController(
        HostControllerParams(120, host_params.u_brake)
    )
    lead_ctrl = LeadCruiseController(
        LeadControllerParams(90, lead_params.u_brake)
    )

    init_gap = 1.1 * safe_distance(
        host.s.speed_kmh, cfg.h, host_params.u_brake
    )

    # <<< CHANGE: same simulator class, new instance per run
    sim = TwoCarSimulator(
        host, lead,
        host_ctrl, lead_ctrl,
        cfg, init_gap,
        ids_accuracy
    )

    crash_time = sim.run(scenario)

    return ids_accuracy, crash_time, scenario   # <<< CHANGE: return triple


def accuracy_vs_time_to_crash():

    acc_vals = []       # <<< CHANGE: store accuracy
    crash_times = []    # <<< CHANGE: store crash time

    for acc in ACCURACIES:

        run_crashes = []

        for _ in range(N_RUNS):
            _, crash_t, _ = run_single_simulation(acc, SCENARIO)

            if crash_t != -1:
                run_crashes.append(crash_t)

        # <<< CHANGE: worst-case (earliest) crash
        if len(run_crashes) == 0:
            acc_vals.append(acc)
            crash_times.append(0)
        else:
            acc_vals.append(acc)
            crash_times.append(min(run_crashes))

        print(f"Accuracy={acc:.2f}, Crash time={crash_times[-1]}")

    # <<< CHANGE: new required plot
    plt.figure(figsize=(8, 5))
    plt.plot(acc_vals, crash_times, marker='o')
    plt.xlabel("IDS Accuracy")
    plt.ylabel("Time to Crash (steps)")
    plt.title("IDS Accuracy vs Time-to-Crash")
    plt.grid(True)
    plt.show()


if __name__ == "__main__":
    accuracy_vs_time_to_crash()
