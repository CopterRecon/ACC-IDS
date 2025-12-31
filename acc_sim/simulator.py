
# Filename: acc_sim/simulator.py
# Author: Lotfi ben Othmane <lotfi.benothmane@unt.edu> 
# Created: 2025-12-29
# Description: Implements the simulator 
# License: -

import random
import numpy as np

from dataclasses import dataclass
import pandas as pd
from .vehicle import VehicleModel
from .safety import gap_update, required_gap_eq17, compute_v_thr,lemma42_z_threshold
from .controllers import LeadCruiseController, HostACCController
from .filters import KalmanFilter
from .attacks import SpeedFaultInjector, SpeedAttackConfig


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
        kf_host: KalmanFilter
    ):
        self.host = host
        self.lead = lead
        self.host_ctrl = host_ctrl
        self.lead_ctrl = lead_ctrl
        self.cfg = cfg
        self.gap_m = float(init_gap_m)
        self.kf_host = kf_host
        self.records = []

        self.attack_cfg =SpeedAttackConfig(
            enabled=True,
            mode="ramp_bias",
            start_step=200,
            ramp_kmh_per_s=0.3,
            max_ramp_bias_kmh=20.0
        )
        self.speed_attacker = SpeedFaultInjector(self.attack_cfg, dt=self.cfg.dt)
        

    def run(self) -> pd.DataFrame:
        ntimes = 0
        rtimes = 0

        for k in range(self.cfg.steps):
            # Lead control + step
            lead_th, lead_br, lead_event = self.lead_ctrl.act(self.lead.s.speed_kmh, dt=self.cfg.dt)
            
            # small random disturbance
            lead_th += np.random.normal(0.0, 0.5)  # throttle noise
            lead_br += np.random.normal(0.0, 0.5)  # brake noise
            
            lead_th = float(np.clip(lead_th, 0, 1))
            lead_br = float(np.clip(lead_br, 0, 1))
            
            vL, aL = self.lead.step(lead_th, lead_br, self.cfg.dt)

            # Host control + step
            host_th, host_br, v_thr, v_tgt, d_req_dbg = self.host_ctrl.act(
                self.host.s.speed_kmh, vL, self.gap_m, h=self.cfg.h, dt=self.cfg.dt
            )
            
            vH, aH = self.host.step(host_th, host_br, self.cfg.dt)
            
            # Update gap (m)
            self.gap_m = gap_update(self.gap_m, vH, vL, self.cfg.dt)

            # Safety metrics (use the controller’s u via vehicle params)
            u = self.host.p.u_brake
            d_req = required_gap_eq17(vH, vL, u=u, h=self.cfg.h, dt=self.cfg.dt)
            v_thr_now = compute_v_thr(self.gap_m, vL, u=u, h=self.cfg.h, dt=self.cfg.dt)

            potential_crash = int(self.gap_m < d_req)
            speed_risk = int(vH > v_thr_now)
            ntimes += potential_crash
            rtimes += speed_risk
            
            #Add the Z
            # Suppose kf_host tracks host speed in km/h
            
            print (f"kf_host.P: {self.kf_host.P}")
            
            # --- KF predict ---
            self.kf_host.predict()
            v_pred_k1 = self.kf_host.x
            P_pred_k1 = self.kf_host.P
            
            
            #  This code simululates random faults processed by kalman filter 
            # --- Measurement (noisy) and KF update ---
            z_meas = vH + np.random.normal(0.0, np.sqrt(self.kf_host.R))
            v_filtFault = self.kf_host.update(z_meas)
            
            # --- Lemma 4.2: measurement threshold ---
            z_thr, K_k1 = lemma42_z_threshold(
                v_pred_k1=v_filtFault,
                P_pred_k1=P_pred_k1,
                R=self.kf_host.R,
                v_thr=v_thr_now,
            )
            
            #Simulate attack
            z_clean = vH + np.random.normal(0.0, np.sqrt(self.kf_host.R))  # sensor noise
            z_attack, attack_active, inj_delta = self.speed_attacker.apply(z_clean, k)
            # The injected data is fed to the kalman filter
            v_filtAttack = self.kf_host.update(z_attack)
            
            
            

            self.records.append({
                "step": k,
                "v_host_kmh": vH,
                "v_lead_kmh": vL,
                "gap_m": self.gap_m,
                "d_req_m": d_req,
                "v_thr_kmh": v_thr_now,
                "v_target_kmh": v_tgt,
                "z_thr":z_thr,
                "host_throttle": host_th,
                "host_brake": host_br,
                "lead_brake_event": int(lead_event),
                "lead_throttle": lead_th,
                "lead_brake": lead_br,
                "potential_crash": potential_crash,
                "speed_risk": speed_risk,
                "a_host_mps2": aH,
                "a_lead_mps2": aL,
                "z_meas_kmh": z_meas,
                "v_filtFault_kmh": v_filtFault,
                "K_k1": K_k1,
                "v_pred_kmh": v_pred_k1,
                "P_pred": P_pred_k1,
                "z_meas_kmh": z_clean,
                "z_attack_kmh": z_attack,
                "v_filtAttack_kmh": v_filtAttack,
                "attack_active": attack_active,
                "inj_delta_kmh": inj_delta,
                "meas_exceeds_z_thr": int(z_attack > z_thr),
            })

            if self.gap_m < self.cfg.stop_gap_m:
                break

        df = pd.DataFrame(self.records)
        df.attrs["safe_distance_violations"] = ntimes
        df.attrs["threshold_violations"] = rtimes
        return df