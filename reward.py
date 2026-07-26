"""
reward.py
=========
De reward function, apart gehouden zodat je hier kan spelen zonder aan de
env-machinerie te komen. Alle getallen staan in settings.py sectie 4.

Vorm van de snelheidsreward ("flat-top gaussian" rond 50 km/u):

      1.0 |        ______________
          |       /              \
          |      /                \
      0.0 |_____/                  \_________
          0    45   50   55            100 km/u
                 ^ SPEED_PEAK_BAND (+/- 5)

  * binnen +/- 5 km/u van het doel: volle reward 1.0
  * daarbuiten: gaussisch afvallend, met een APARTE sigma voor te traag en
    te snel. Zo wordt te hard rijden strenger bestraft dan te zacht rijden
    (SPEED_SIGMA_FAST < SPEED_SIGMA_SLOW), maar beide worden bestraft.

Totale reward per stap:
    r  =  SPEED_REWARD_WEIGHT * snelheidsterm
        + PROGRESS_WEIGHT     * bewegingsbonus (optical flow)
        + IDLE_PENALTY        als de auto stilstaat
        - STEER_JERK_WEIGHT   * |stuurverandering|
        + CRASH/OFFROAD/STUCK_PENALTY bij een terminaal event
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, Optional

import settings


# ---------------------------------------------------------------------------
# Losse termen
# ---------------------------------------------------------------------------


def speed_term(speed: Optional[float]) -> float:
    """
    Flat-top gaussian rond TARGET_SPEED.
    Bij onbekende snelheid (OCR faalde) geven we 0: neutraal, niet bestraffend.
    """
    if speed is None:
        return 0.0

    deviation = abs(speed - settings.TARGET_SPEED)
    if deviation <= settings.SPEED_PEAK_BAND:
        return 1.0

    excess = deviation - settings.SPEED_PEAK_BAND
    sigma = settings.SPEED_SIGMA_FAST if speed > settings.TARGET_SPEED else settings.SPEED_SIGMA_SLOW
    sigma = max(sigma, 1e-3)
    return math.exp(-(excess * excess) / (2.0 * sigma * sigma))


def progress_term(flow: float) -> float:
    """
    Bewegingsbonus op basis van optical flow. Voorkomt dat 'stilstaan' een
    lokaal optimum wordt: stilstaan is namelijk risicoloos (geen crash), dus
    zonder deze term kan de agent leren om gewoon niets te doen.
    Genormaliseerd op PROGRESS_FLOW_REF en afgekapt op 1.0.
    """
    ref = max(settings.PROGRESS_FLOW_REF, 1e-3)
    return max(0.0, min(flow / ref, 1.0))


def idle_term(speed: Optional[float], flow: float) -> float:
    """
    Straf bij stilstand. Als OCR de snelheid niet kent, vallen we terug op de
    optical flow: bijna geen beweging in beeld = we staan stil.
    """
    if speed is None:
        standing_still = flow < (0.15 * settings.PROGRESS_FLOW_REF)
    else:
        standing_still = speed < settings.IDLE_SPEED
    return settings.IDLE_PENALTY if standing_still else 0.0


def jerk_term(steer: float, prev_steer: float) -> float:
    """Kleine straf op abrupte stuurwissels -> vloeiender rijgedrag."""
    return -settings.STEER_JERK_WEIGHT * abs(steer - prev_steer)


def terminal_penalty(reason: str) -> float:
    return {
        "crash": settings.CRASH_PENALTY,
        "flip": settings.CRASH_PENALTY,
        "offroad": settings.OFFROAD_PENALTY,
        "stuck": settings.STUCK_PENALTY,
    }.get(reason, 0.0)


# ---------------------------------------------------------------------------
# Alles samen
# ---------------------------------------------------------------------------


@dataclass
class RewardBreakdown:
    """Opsplitsing per term - de viewer en de logs tonen dit."""

    total: float = 0.0
    speed: float = 0.0
    progress: float = 0.0
    idle: float = 0.0
    jerk: float = 0.0
    terminal: float = 0.0

    def as_dict(self) -> Dict[str, float]:
        return {
            "r_total": self.total,
            "r_speed": self.speed,
            "r_progress": self.progress,
            "r_idle": self.idle,
            "r_jerk": self.jerk,
            "r_terminal": self.terminal,
        }


def compute_reward(
    speed: Optional[float],
    flow: float,
    steer: float,
    prev_steer: float,
    terminal_reason: str = "",
) -> RewardBreakdown:
    """Bereken de reward voor één stap."""
    rb = RewardBreakdown()
    rb.speed = settings.SPEED_REWARD_WEIGHT * speed_term(speed)
    rb.progress = settings.PROGRESS_WEIGHT * progress_term(flow)
    rb.idle = idle_term(speed, flow)
    rb.jerk = jerk_term(steer, prev_steer)
    rb.terminal = terminal_penalty(terminal_reason)
    rb.total = rb.speed + rb.progress + rb.idle + rb.jerk + rb.terminal
    return rb
