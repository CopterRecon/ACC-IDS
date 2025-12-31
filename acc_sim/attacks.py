
# Filename: acc_sim/attacks.py
# Author: ChatGPT
# Created: 2025-12-30
# Description: Models speed injection attacks
# License: -

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Tuple
import numpy as np


@dataclass
class SpeedAttackConfig:
    enabled: bool = False

    # Attack scheduling
    start_step: int = 200
    end_step: Optional[int] = None  # None => until the end

    # Modes: "bias", "scale", "ramp_bias", "pulse", "replay"
    mode: str = "ramp_bias"

    # --- bias / ramp_bias ---
    bias_kmh: float = 8.0                 # for mode="bias"
    ramp_kmh_per_s: float = 0.4           # for mode="ramp_bias" (km/h per second)
    max_ramp_bias_kmh: float = 25.0       # clamp ramp bias

    # --- scale ---
    scale: float = 1.10                   # for mode="scale" (10% exaggeration)

    # --- pulse (random spike events) ---
    pulse_rate_per_s: float = 0.02        # expected pulse starts per second
    pulse_amp_range_kmh: Tuple[float, float] = (5.0, 20.0)
    pulse_dur_range_s: Tuple[float, float] = (0.5, 2.0)

    # --- replay ---
    replay_delay_steps: int = 20

    # Output limits (keep physically plausible)
    clip_min_kmh: float = 0.0
    clip_max_kmh: float = 220.0

    # RNG seed for reproducibility
    seed: int = 1234


class SpeedFaultInjector:
    """
    Applies fault injection to a speed measurement (km/h).
    Maintains internal state for ramp/pulse/replay attacks.
    """

    def __init__(self, cfg: SpeedAttackConfig, dt: float):
        self.cfg = cfg
        self.dt = float(dt)
        self.rng = np.random.default_rng(cfg.seed)

        # state for ramp bias
        self._ramp_bias = 0.0

        # state for pulse events
        self._pulse_active = False
        self._pulse_time_left = 0.0
        self._pulse_amp = 0.0

        # state for replay
        self._buffer: list[float] = []

    def _in_window(self, k: int) -> bool:
        if not self.cfg.enabled:
            return False
        if k < self.cfg.start_step:
            return False
        if self.cfg.end_step is not None and k > self.cfg.end_step:
            return False
        return True

    def apply(self, z_kmh: float, k: int) -> tuple[float, int, float]:
        """
        Inputs:
          z_kmh : original measurement (km/h)
          k     : step index

        Returns:
          z_attacked_kmh, attack_active(0/1), injected_delta_kmh
        """
        # keep replay buffer regardless (so attack can start later)
        self._buffer.append(float(z_kmh))

        if not self._in_window(k):
            z_out = float(np.clip(z_kmh, self.cfg.clip_min_kmh, self.cfg.clip_max_kmh))
            return z_out, 0, z_out - float(z_kmh)

        mode = self.cfg.mode.lower()
        z_att = float(z_kmh)

        if mode == "bias":
            z_att = z_kmh + self.cfg.bias_kmh

        elif mode == "scale":
            z_att = z_kmh * self.cfg.scale

        elif mode == "ramp_bias":
            # increase bias gradually over time (stealthy drift)
            self._ramp_bias += self.cfg.ramp_kmh_per_s * self.dt
            self._ramp_bias = float(np.clip(self._ramp_bias, -self.cfg.max_ramp_bias_kmh, self.cfg.max_ramp_bias_kmh))
            z_att = z_kmh + self._ramp_bias

        elif mode == "pulse":
            # occasionally start a pulse; while active, add constant offset
            if self._pulse_active:
                self._pulse_time_left -= self.dt
                z_att = z_kmh + self._pulse_amp
                if self._pulse_time_left <= 0:
                    self._pulse_active = False
                    self._pulse_time_left = 0.0
                    self._pulse_amp = 0.0
            else:
                p_start = self.cfg.pulse_rate_per_s * self.dt
                if self.rng.random() < p_start:
                    self._pulse_active = True
                    self._pulse_time_left = float(self.rng.uniform(*self.cfg.pulse_dur_range_s))
                    self._pulse_amp = float(self.rng.uniform(*self.cfg.pulse_amp_range_kmh))
                    z_att = z_kmh + self._pulse_amp

        elif mode == "replay":
            d = max(1, int(self.cfg.replay_delay_steps))
            if len(self._buffer) > d:
                z_att = self._buffer[-d]  # replay older measurement
            else:
                z_att = z_kmh

        else:
            # unknown mode => no attack
            z_att = z_kmh

        z_att = float(np.clip(z_att, self.cfg.clip_min_kmh, self.cfg.clip_max_kmh))
        return z_att, 1, z_att - float(z_kmh)