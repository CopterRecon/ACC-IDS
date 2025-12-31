
# Filename: acc_sim/controllers.py
# Author: Lotfi ben Othmane <lotfi.benothmane@unt.edu> 
# Created: 2025-12-29
# Description: Models vehicle controllers
# License: -

import numpy as np
from dataclasses import dataclass
from .constants import KMH_TO_MS
from .safety import compute_v_thr, required_gap_eq17

@dataclass
class LeadControllerParams:
    v_set_kmh: float = 90.0
    a_comfort: float = 1.2
    k_acc: float = 0.12
    u: float = 3.4

import random
import numpy as np
from .constants import KMH_TO_MS

class LeadCruiseController:
    """
    Cruise controller + random braking events.
    - Most of the time: track v_set_kmh.
    - Sometimes: enter a braking event (hold brake for some duration).
    """
    
    def __init__(
        self,
        p,
        seed: int=None,
        # Event frequency: expected events per minute (e.g., 2 => about once every 30s)
        events_per_min: float = 2.0,
        # Brake intensity range during event
        brake_min: float = 0.2,
        brake_max: float = 0.8,
        # Event duration range (seconds)
        dur_min_s: float = 0.5,
        dur_max_s: float = 3.0,
        # Minimum time between events (seconds)
        cooldown_s: float = 5.0,
    ):
        self.p = p
        self._rng = random.Random(seed)
        
        self.events_per_min = float(events_per_min)
        self.brake_min = float(brake_min)
        self.brake_max = float(brake_max)
        self.dur_min_s = float(dur_min_s)
        self.dur_max_s = float(dur_max_s)
        self.cooldown_s = float(cooldown_s)
        
        # internal state
        self._in_event = False
        self._event_time_left = 0.0
        self._event_brake = 0.0
        self._cooldown_left = 0.0
        
    def _maybe_start_event(self, dt: float):
        if self._in_event or self._cooldown_left > 0:
            return
        
        # Convert events/min into per-step probability
        # rate per second = events_per_min / 60
        rate_per_s = self.events_per_min / 60.0
        p_start = rate_per_s * dt  # small dt approximation
        
        if self._rng.random() < p_start:
            self._in_event = True
            self._event_time_left = self._rng.uniform(self.dur_min_s, self.dur_max_s)
            self._event_brake = self._rng.uniform(self.brake_min, self.brake_max)
            
    def act(self, v_lead_kmh: float, dt: float = 0.1):
        # update timers
        if self._cooldown_left > 0:
            self._cooldown_left = max(0.0, self._cooldown_left - dt)
            
        # maybe start an event
        self._maybe_start_event(dt)
        
        # If in braking event: override with brake, throttle=0
        if self._in_event:
            self._event_time_left -= dt
            throttle = 0.0
            brake = self._event_brake
            
            if self._event_time_left <= 0:
                self._in_event = False
                self._event_time_left = 0.0
                self._event_brake = 0.0
                self._cooldown_left = self.cooldown_s
                
            return throttle, brake, True  # True => braking event active
        
        # Otherwise: normal cruise control to v_set_kmh (your accel-based logic)
        err = self.p.v_set_kmh - v_lead_kmh
        a_des = self.p.k_acc * err * KMH_TO_MS
        a_des = max(-self.p.u, min(self.p.a_comfort, a_des))
        
        if a_des >= 0:
            throttle = a_des / self.p.a_comfort if self.p.a_comfort > 0 else 0.0
            brake = 0.0
        else:
            throttle = 0.0
            brake = (-a_des) / self.p.u if self.p.u > 0 else 0.0
            
        return float(np.clip(throttle, 0, 1)), float(np.clip(brake, 0, 1)), False

@dataclass
class HostControllerParams:
    cruise_kmh: float = 120.0
    a_comfort: float = 1.5
    k_acc: float = 0.15
    u: float = 3.4
    hard_brake: float = 0.8

class HostACCController:
    def __init__(self, p: HostControllerParams):
        self.p = p

    def act(self, v_host_kmh: float, v_lead_kmh: float, gap_m: float, h: float, dt: float):
        v_thr = compute_v_thr(gap_m, v_lead_kmh, u=self.p.u, h=h, dt=dt)
        v_target = min(self.p.cruise_kmh, v_lead_kmh, v_thr)

        err = v_target - v_host_kmh
        a_des = self.p.k_acc * err * KMH_TO_MS
        a_des = max(-self.p.u, min(self.p.a_comfort, a_des))

        if a_des >= 0:
            throttle = a_des / self.p.a_comfort if self.p.a_comfort > 0 else 0.0
            brake = 0.0
        else:
            throttle = 0.0
            brake = (-a_des) / self.p.u if self.p.u > 0 else 0.0

        # extra safety
        d_req = required_gap_eq17(v_host_kmh, v_lead_kmh, u=self.p.u, h=h, dt=dt)
            #if gap_m < d_req:
            #throttle = 0.0
            #brake = max(brake, self.p.hard_brake)

        return float(np.clip(throttle, 0, 1)), float(np.clip(brake, 0, 1)), v_thr, v_target, d_req