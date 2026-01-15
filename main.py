import numpy as np
import matplotlib.pyplot as plt

from acc_sim.simulator import TwoCarSimulator, SimConfig
from acc_sim.vehicle import VehicleModel, VehicleParams, VehicleState
from acc_sim.controllers import (
    LeadCruiseController,
    HostACCController,
    LeadControllerParams,
    HostControllerParams
)


def accuracy_vs_time_to_crash():
    """
    Research-grade experiment:
    IDS accuracy vs time-to-crash.

    IMPORTANT:
    - crash_time == -1 means NO crash
    - mapped to full simulation horizon (cfg.steps)
    """

    cfg = SimConfig()
    scenario = 4  # IDS-enabled attack scenario

    acc_vals = np.arange(0.1, 1.01, 0.1)
    crash_times = []

    for acc in acc_vals:

        # ===== Vehicle parameters & states =====
        host_vehicle_params = VehicleParams()
        lead_vehicle_params = VehicleParams()

        host_state = VehicleState(speed_kmh=80.0)
        lead_state = VehicleState(speed_kmh=80.0)

        host = VehicleModel("host", host_vehicle_params, host_state)
        lead = VehicleModel("lead", lead_vehicle_params, lead_state)

        # ===== Controller parameters (THIS IS THE KEY FIX) =====
        lead_ctrl_params = LeadControllerParams(
            v_set_kmh=80.0
        )

        host_ctrl_params = HostControllerParams(
            cruise_kmh=100.0
        )

        host_ctrl = HostACCController(host_ctrl_params)
        lead_ctrl = LeadCruiseController(lead_ctrl_params)

        init_gap_m = 30.0

        sim = TwoCarSimulator(
            host=host,
            lead=lead,
            host_ctrl=host_ctrl,
            lead_ctrl=lead_ctrl,
            cfg=cfg,
            init_gap_m=init_gap_m,
            ids_accuracy=acc
        )

        crash_time = sim.run(scenario)

        # ===== RESEARCH-CORRECT HANDLING OF NO-CRASH =====
        if crash_time == -1:
            crash_time = cfg.steps

        crash_times.append(crash_time)

        print(f"Accuracy={acc:.2f}, Time-to-crash={crash_time}")

    # ===== Plot =====
    plt.figure(figsize=(8, 5))
    plt.plot(acc_vals, crash_times, marker='o')
    plt.xlabel("IDS Accuracy")
    plt.ylabel("Time to Crash (steps)")
    plt.title("IDS Accuracy vs Time-to-Crash")
    plt.grid(True)
    plt.show()


if __name__ == "__main__":
    accuracy_vs_time_to_crash()
