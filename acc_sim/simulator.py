# acc_sim/simulator.py
import random
import numpy as np

from dataclasses import dataclass
import pandas as pd
from .vehicle import VehicleModel
from .safety import gap_update, required_gap_eq17, compute_v_thr
from .controllers import LeadCruiseController, HostACCController

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
        init_gap_m: float
    ):
        self.host = host
        self.lead = lead
        self.host_ctrl = host_ctrl
        self.lead_ctrl = lead_ctrl
        self.cfg = cfg
        self.gap_m = float(init_gap_m)
        self.records = []

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

            self.records.append({
                "step": k,
                "v_host_kmh": vH,
                "v_lead_kmh": vL,
                "gap_m": self.gap_m,
                "d_req_m": d_req,
                "v_thr_kmh": v_thr_now,
                "v_target_kmh": v_tgt,
                "host_throttle": host_th,
                "host_brake": host_br,
                "lead_brake_event": int(lead_event),
                "lead_throttle": lead_th,
                "lead_brake": lead_br,
                "potential_crash": potential_crash,
                "speed_risk": speed_risk,
                "a_host_mps2": aH,
                "a_lead_mps2": aL,
            })

            if self.gap_m < self.cfg.stop_gap_m:
                break

        df = pd.DataFrame(self.records)
        df.attrs["safe_distance_violations"] = ntimes
        df.attrs["threshold_violations"] = rtimes
        return df