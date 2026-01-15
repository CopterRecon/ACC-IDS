# Filename: acc_sim/simulator.py

import random
import numpy as np
from dataclasses import dataclass
import pandas as pd

from .vehicle import VehicleModel
from .safety import (
    gap_update, required_gap_eq17,
    compute_v_thr, lemma42_z_threshold, safe_distance
)
from .controllers import LeadCruiseController, HostACCController
from .filters import KalmanFilter
from .attacks import SpeedFaultInjector, SpeedAttackConfig
from acc_sim.constants import R, Q, P0


@dataclass
class SimConfig:
    h: float = 2.0
    dt: float = 0.1
    steps: int = 1000
    stop_gap_m: float = 2.0


class TwoCarSimulator:

    def __init__(
        self,
        host,
        lead,
        host_ctrl,
        lead_ctrl,
        cfg,
        init_gap_m,
        ids_accuracy            # accuracy passed per run
    ):
        self.host = host
        self.lead = lead
        self.host_ctrl = host_ctrl
        self.lead_ctrl = lead_ctrl
        self.cfg = cfg
        self.gap_m = init_gap_m
        self.ids_accuracy = ids_accuracy
        self.records = []

        self.attack_cfg = SpeedAttackConfig(
            enabled=True,
            mode="ramp_bias",
            start_step=100,
            ramp_kmh_per_s=1.0,
            max_ramp_bias_kmh=40.0
        )

        self.attacker = SpeedFaultInjector(self.attack_cfg, cfg.dt)
        self.kf = KalmanFilter(
            x0=host.s.speed_kmh, P0=P0, Q=Q, R=R
        )

    def run(self, scenario):
        """
        Returns:
            crash_time (int): timestep where crash occurs
                              -1 if no crash
        """

        crash_time = -1

        for i in range(self.cfg.steps):

            # ===== Lead vehicle =====
            lead_th, lead_br, _ = self.lead_ctrl.act(
                self.lead.s.speed_kmh, dt=self.cfg.dt
            )

            # Noise added (as requested)
            lead_th = np.clip(lead_th + np.random.normal(0, 0.5), 0, 1)
            lead_br = np.clip(lead_br + np.random.normal(0, 0.5), 0, 1)

            vL, _ = self.lead.step(lead_th, lead_br, self.cfg.dt)

            # ===== Host vehicle =====
            host_th, host_br = self.host_ctrl.act(
                self.host.s.speed_kmh,
                vL,
                self.gap_m,
                h=self.cfg.h,
                dt=self.cfg.dt
            )

            vH_real, _ = self.host.step(host_th, host_br, self.cfg.dt)

            # ===== Kalman Filter =====
            self.kf.predict()

            z_attack = vH_real
            attack_active = False

            # Attack only in scenarios ≥ 3
            if scenario >= 3:
                z_attack, attack_active, _ = self.attacker.apply(vH_real, i)

                # <<< FIX: IDS ONLY rejects attack measurement (no braking)
                if scenario == 4 and attack_active:
                    if random.random() < self.ids_accuracy:
                        z_attack = vH_real   # IDS cleans measurement

            vH = self.kf.update(z_attack)
            self.host.s.speed_kmh = vH

            # ===== Gap update =====
            self.gap_m = gap_update(
                self.gap_m, vH_real, vL, self.cfg.dt
            )

            d_safe = safe_distance(
                vH_real, self.cfg.h, self.host.p.u_brake
            )

            # ===== Crash detection =====
            if self.gap_m < self.cfg.stop_gap_m:
                crash_time = i
                return crash_time

            # ===== Logging (unchanged for plots) =====
            self.records.append({
                "step": i,
                "gap_m": self.gap_m,
                "d_safe": d_safe,
                "v_host_kmh": vH,
                "v_lead_kmh": vL,
                "z_attack_kmh": z_attack,
                "IDS_accuracy": self.ids_accuracy
            })

        return crash_time
