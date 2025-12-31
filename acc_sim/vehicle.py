# acc_sim/vehicle.py

from dataclasses import dataclass
from .constants import KMH_TO_MS, MS_TO_KMH, G

@dataclass
class VehicleParams:
    mass: float = 1200.0
    C_drag: float = 0.30
    A_frontal: float = 2.0
    rho_air: float = 1.225
    C_rolling: float = 0.010
    max_engine_force: float = 5000.0
    u_brake: float = 3.4      # max braking decel magnitude (m/s^2)
    a_max: float = 1.5        # comfort accel cap (m/s^2)

@dataclass
class VehicleState:
    speed_kmh: float

class VehicleModel:
    """
    Longitudinal dynamics model. Speeds in km/h externally, internal integration in m/s.
    """
    def __init__(self, name: str, params: VehicleParams, state: VehicleState):
        self.name = name
        self.p = params
        self.s = state

    def step(self, throttle: float, brake: float, dt: float):
        throttle = max(0.0, min(1.0, float(throttle)))
        brake = max(0.0, min(1.0, float(brake)))
        dt = max(0.0, float(dt))

        v_ms = max(0.0, self.s.speed_kmh) * KMH_TO_MS

        # Forces
        tractive = self.p.max_engine_force * throttle
        drag = 0.5 * self.p.rho_air * (v_ms ** 2) * self.p.C_drag * self.p.A_frontal
        rolling = self.p.mass * G * self.p.C_rolling

        max_brake_force = self.p.mass * self.p.u_brake
        brake_force = max_brake_force * brake

        net_force = tractive - drag - rolling - brake_force
        acc = net_force / self.p.mass  # m/s^2

        # Cap braking and acceleration
        if brake > 0:
            acc = max(acc, -(brake * self.p.u_brake))
        acc = min(acc, self.p.a_max)

        # Integrate
        v_new_ms = max(0.0, v_ms + acc * dt)
        if dt > 0:
            acc = (v_new_ms - v_ms) / dt

        self.s.speed_kmh = v_new_ms * MS_TO_KMH
        return self.s.speed_kmh, acc