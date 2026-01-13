# Filename: acc_sim/simulator.py
# Author: Lotfi ben Othmane <lotfi.benothmane@unt.edu>
# Modified by: Naga Prudhvi Mareedu
# Description: Simulator with IDS accuracy modeling (Section 6.5)
# License: -

import random
import numpy as np
from dataclasses import dataclass
import pandas as pd

from .vehicle import VehicleModel
from .safety import (
    gap_update, required_gap_eq17, compute_v_thr,
    lemma42_z_threshold, safe_distance
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
        host: VehicleModel,
        lead: VehicleModel,
        host_ctrl: HostACCController,
        lead_ctrl: LeadCruiseController,
        cfg: SimConfig,
        init_gap_m: float,
        ids_accuracy: float = 1.0
    ):
        self.host = host
        self.lead = lead
        self.host_ctrl = host_ctrl
        self.lead_ctrl = lead_ctrl
        self.cfg = cfg
        self.gap_m = float(init_gap_m)
        self.records = []

        # IDS accuracy α
        self.ids_accuracy = float(np.clip(ids_accuracy, 0.0, 1.0))

        # Attack configuration
        self.attack_cfg = SpeedAttackConfig(
            enabled=True,
            mode="ramp_bias",
            start_step=200,
            ramp_kmh_per_s=0.3,
            max_ramp_bias_kmh=10.0
        )

        self.speed_attacker = SpeedFaultInjector(
            self.attack_cfg, dt=self.cfg.dt
        )

        # Kalman Filter
        self.kf_host = KalmanFilter(
            x0=host.s.speed_kmh, P0=P0, Q=Q, R=R
        )

    def run(self, scenario) -> pd.DataFrame:

        ntimes = rtimes = ctimes = ztimes = 0
        ActivateIDS = 0

        for i in range(self.cfg.steps):

            # -------- Lead vehicle --------
            lead_th, lead_br, lead_event = self.lead_ctrl.act(
                self.lead.s.speed_kmh, dt=self.cfg.dt
            )

            lead_th = np.clip(lead_th + np.random.normal(0, 0.5), 0, 1)
            lead_br = np.clip(lead_br + np.random.normal(0, 0.5), 0, 1)

            vL, aL = self.lead.step(lead_th, lead_br, self.cfg.dt)

            # -------- Host control --------
            host_th, host_br = self.host_ctrl.act(
                self.host.s.speed_kmh, vL,
                self.gap_m, h=self.cfg.h, dt=self.cfg.dt
            )

            # IDS override (Eq. 31)
            if scenario == 4 and ActivateIDS == 1:
                host_br = 1.0
                host_th = 0.0
                ActivateIDS = 0

            vH_real, aH = self.host.step(
                host_th, host_br, self.cfg.dt
            )

            # -------- Kalman Filter --------
            self.kf_host.predict()
            v_pred_k1 = self.kf_host.x
            P_pred_k1 = self.kf_host.P

            Effective_vH = vH_real
            z_attack = vH_real
            attack_active = 0
            inj_delta = 0

            # -------- Attack + IDS --------
            if scenario >= 3:
                z_attack, attack_active, inj_delta = \
                    self.speed_attacker.apply(vH_real, i)

                if scenario == 4 and attack_active:
                    if random.random() < self.ids_accuracy:
                        ActivateIDS = 1
                        z_attack = vH_real

            vH = self.kf_host.update(z_attack)
            self.host.s.speed_kmh = vH

            # -------- Gap update --------
            self.gap_m = gap_update(
                self.gap_m, Effective_vH, vL, self.cfg.dt
            )

            # -------- Safety metrics --------
            u = self.host.p.u_brake
            d_safe = safe_distance(Effective_vH, self.cfg.h, u)
            d_req = required_gap_eq17(vH, vL, u=u, h=self.cfg.h, dt=self.cfg.dt)
            v_thr = compute_v_thr(self.gap_m, vL, u=u, h=self.cfg.h, dt=self.cfg.dt)

            z_thr, _ = lemma42_z_threshold(
                v_pred_k1, P_pred_k1, self.kf_host.R, v_thr
            )

            ntimes += int(self.gap_m < d_safe)
            rtimes += int(vH > v_thr)
            ztimes += int(vH > z_thr)

            # -------- RECORD (FIXED INDENTATION) --------
            self.records.append({
                "step": i,
                "v_host_kmh": vH,
                "Effective_vH": Effective_vH,
                "v_lead_kmh": vL,
                "gap_m": self.gap_m,
                "d_safe": d_safe,
                "d_req_m": d_req,
                "v_thr_kmh": v_thr,
                "z_thr": z_thr,
                "z_attack_kmh": z_attack,
                "host_throttle": host_th,
                "host_brake": host_br,
                "attack_active": attack_active,
                "inj_delta_kmh": inj_delta,
                "IDS_accuracy": self.ids_accuracy
            })

            if self.gap_m < self.cfg.stop_gap_m:
#need exact time 
                ctimes = 1
                break

        df = pd.DataFrame(self.records)
        df.attrs["safe_distance_violations"] = ntimes
        df.attrs["Speed threshold_violations"] = rtimes
        df.attrs["Z threshold_violations"] = ztimes
        df.attrs["Crashes"] = ctimes

        return df, ctimes
