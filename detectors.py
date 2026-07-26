"""
detectors.py
============
Puur VISUELE detectie van crashes, omslaan, van-de-weg-af en vastzitten.
Geen enkele game-state, alleen pixels + de OCR-snelheid (die zelf ook uit
pixels komt).

===========================================================================
AANPAK (bewust simpel gehouden zodat je hem zelf kan verfijnen)
===========================================================================

We combineren vijf goedkope, onafhankelijke signalen. Elk signaal alleen is
te fragiel; samen zijn ze bruikbaar. Alle drempels staan in settings.py
sectie 6.

(a) PLOTSE SNELHEIDSVAL  -> botsing
    We houden een ringbuffer van (tijd, snelheid) bij. Als de snelheid
    binnen CRASH_DROP_WINDOW_S (0.35 s) meer dan CRASH_SPEED_DROP (18 km/u)
    daalt terwijl je sneller reed dan CRASH_MIN_SPEED, dan ben je ergens
    tegenaan gereden. Remmen doet dit niet zo abrupt.
    -> Verfijnen: eis dat de rem NIET ingedrukt was (throttle/brake worden
       al doorgegeven aan update()).

(b) BEELDSCHOK  -> impact
    Gemiddeld absoluut frameverschil. Normaal rijden geeft een gelijkmatig
    laag verschil; een klap, een salto of een camerawissel geeft een korte
    piek boven IMPACT_DIFF_THRESHOLD. We eisen IMPACT_CONSECUTIVE frames op
    rij zodat een enkele HUD-flits niet meetelt.
    -> Verfijnen: normaliseer de drempel op de huidige snelheid (snel rijden
       geeft van nature meer beeldverandering).

(c) LUCHT ONDERAAN  -> omgeslagen
    Als de auto op zijn dak ligt, staat de lucht onderaan het beeld. We
    vergelijken de gemiddelde helderheid van de bovenste en onderste strook.
    Is de onderste FLIP_BRIGHTNESS_MARGIN helderder gedurende
    FLIP_CONSECUTIVE frames, dan liggen we op de kop.
    -> Verfijnen: ook de horizonlijn schatten met een Hough- of Sobel-fit en
       de hoek ervan gebruiken (werkt ook bij nacht/tunnels).

(d) WEGDEK-FRACTIE  -> van de weg af
    Tijdens de kalibratie sample je de kleur van het asfalt vlak voor de
    auto. Tijdens het rijden kijken we in ROAD_ROI (een vak onderaan het
    midden van het beeld, dus vlak voor de neus) hoeveel procent van de
    pixels binnen de tolerantieband rond die referentiekleur valt. Asfalt is
    grijs = lage saturatie; gras/zand/water zijn duidelijk kleuriger of veel
    donkerder/lichter. Zakt die fractie onder ROAD_MIN_FRACTION gedurende
    OFFROAD_CONSECUTIVE frames, dan rijden we naast de weg.
    -> Verfijnen: meerdere ROI's (links/midden/rechts) om ook "half op de
       berm" te vangen, of een echte lane-detectie met Canny + Hough.

(e) VASTZITTEN
    Snelheid onder STUCK_SPEED terwijl er meer dan STUCK_THROTTLE gas
    gegeven wordt, gedurende STUCK_SECONDS. Klassiek "neus tegen een boom".

Alle detectie staat uit tijdens de eerste WARMUP_STEPS van een episode,
zodat het inzakken van de vering na een recover geen valse crash geeft.
===========================================================================
"""

from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass
from typing import Deque, Optional, Tuple

import cv2
import numpy as np

import settings


# ---------------------------------------------------------------------------
# (d) Wegdetectie
# ---------------------------------------------------------------------------


class RoadDetector:
    """Meet welk deel van een ROI onderaan het beeld op het gekalibreerde
    wegdek lijkt."""

    def __init__(self, road_hsv: Optional[Tuple[int, int, int]]) -> None:
        self.road_hsv = tuple(int(v) for v in road_hsv) if road_hsv else None
        self.last_fraction: float = 1.0
        self.last_mask: Optional[np.ndarray] = None

    @property
    def enabled(self) -> bool:
        return self.road_hsv is not None

    @staticmethod
    def roi_pixels(frame: np.ndarray) -> Tuple[int, int, int, int]:
        """ROAD_ROI (fracties) -> pixelcoördinaten (x0, y0, x1, y1)."""
        h, w = frame.shape[:2]
        x0f, y0f, x1f, y1f = settings.ROAD_ROI
        x0, x1 = int(w * x0f), int(w * x1f)
        y0, y1 = int(h * y0f), int(h * y1f)
        return x0, y0, max(x1, x0 + 1), max(y1, y0 + 1)

    def sample_reference(self, frame_bgr: np.ndarray) -> Tuple[int, int, int]:
        """Mediane HSV in de ROI. Gebruikt door de kalibratie: parkeer op de
        weg en sample."""
        x0, y0, x1, y1 = self.roi_pixels(frame_bgr)
        roi = cv2.cvtColor(frame_bgr[y0:y1, x0:x1], cv2.COLOR_BGR2HSV)
        med = np.median(roi.reshape(-1, 3), axis=0)
        self.road_hsv = (int(med[0]), int(med[1]), int(med[2]))
        return self.road_hsv

    def update(self, frame_bgr: np.ndarray) -> float:
        """Geeft de fractie 'wegdek-achtige' pixels in de ROI (0..1)."""
        if not self.enabled:
            self.last_fraction = 1.0
            return 1.0

        x0, y0, x1, y1 = self.roi_pixels(frame_bgr)
        hsv = cv2.cvtColor(frame_bgr[y0:y1, x0:x1], cv2.COLOR_BGR2HSV)
        h, s, v = hsv[:, :, 0].astype(np.int16), hsv[:, :, 1].astype(np.int16), hsv[:, :, 2].astype(np.int16)
        rh, rs, rv = self.road_hsv

        # Saturatie- en helderheidsband gelden altijd.
        mask = (np.abs(s - rs) <= settings.ROAD_TOL_S) & (np.abs(v - rv) <= settings.ROAD_TOL_V)

        # Hue is bij grijs asfalt pure ruis: enkel meenemen als de referentie
        # écht kleur heeft. Hue is circulair (0..179), dus zo vergelijken:
        if rs > settings.ROAD_GREY_S_MAX:
            dh = np.abs(h - rh)
            dh = np.minimum(dh, 180 - dh)
            mask &= dh <= settings.ROAD_TOL_H

        self.last_mask = (mask.astype(np.uint8) * 255)
        self.last_fraction = float(mask.mean())
        return self.last_fraction


# ---------------------------------------------------------------------------
# Resultaat van één detectie-update
# ---------------------------------------------------------------------------


@dataclass
class DetectionResult:
    done: bool = False
    reason: str = ""            # "crash" | "flip" | "offroad" | "stuck" | ""
    road_fraction: float = 1.0
    frame_diff: float = 0.0
    speed_drop: float = 0.0
    flip_margin: float = 0.0
    offroad_streak: int = 0
    stuck_seconds: float = 0.0


# ---------------------------------------------------------------------------
# De gecombineerde detector
# ---------------------------------------------------------------------------


class EventDetector:
    def __init__(self, road_hsv: Optional[Tuple[int, int, int]]) -> None:
        self.road = RoadDetector(road_hsv)
        self._speed_hist: Deque[Tuple[float, float]] = deque(maxlen=64)
        self._impact_streak = 0
        self._flip_streak = 0
        self._offroad_streak = 0
        self._stuck_since: Optional[float] = None
        self.steps = 0
        self.last = DetectionResult()

    def reset(self) -> None:
        self._speed_hist.clear()
        self._impact_streak = 0
        self._flip_streak = 0
        self._offroad_streak = 0
        self._stuck_since = None
        self.steps = 0
        self.last = DetectionResult()

    # -- helpers -----------------------------------------------------------
    def _speed_drop(self, speed: Optional[float], now: float) -> float:
        """Grootste daling binnen het tijdsvenster."""
        if speed is None:
            return 0.0
        self._speed_hist.append((now, speed))
        window = now - settings.CRASH_DROP_WINDOW_S
        peak = None
        for t, v in self._speed_hist:
            if t >= window and (peak is None or v > peak):
                peak = v
        if peak is None or peak < settings.CRASH_MIN_SPEED:
            return 0.0
        return max(0.0, peak - speed)

    @staticmethod
    def _flip_margin(frame_bgr: np.ndarray) -> float:
        """Hoeveel helderder de onderste strook is dan de bovenste."""
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        h = gray.shape[0]
        band = max(1, h // 6)
        return float(np.mean(gray[-band:, :])) - float(np.mean(gray[:band, :]))

    # -- hoofdupdate -------------------------------------------------------
    def update(
        self,
        frame_bgr: np.ndarray,
        speed: Optional[float],
        frame_diff: float,
        throttle: float,
        brake: float,
        now: Optional[float] = None,
    ) -> DetectionResult:
        now = now if now is not None else time.perf_counter()
        self.steps += 1

        res = DetectionResult()
        res.road_fraction = self.road.update(frame_bgr)
        res.frame_diff = frame_diff
        res.speed_drop = self._speed_drop(speed, now)
        res.flip_margin = self._flip_margin(frame_bgr)

        # --- streaks bijhouden ---
        self._impact_streak = self._impact_streak + 1 if frame_diff > settings.IMPACT_DIFF_THRESHOLD else 0
        self._flip_streak = self._flip_streak + 1 if res.flip_margin > settings.FLIP_BRIGHTNESS_MARGIN else 0

        if self.road.enabled and res.road_fraction < settings.ROAD_MIN_FRACTION:
            self._offroad_streak += 1
        else:
            self._offroad_streak = 0

        if speed is not None and speed < settings.STUCK_SPEED and throttle > settings.STUCK_THROTTLE:
            if self._stuck_since is None:
                self._stuck_since = now
        else:
            self._stuck_since = None

        # --- warmup: kijken maar niet oordelen ---
        # De streaks worden hier bewust op 0 gehouden. Anders staat de teller
        # al vol zodra de warmup afloopt en eindigt de episode meteen op stap
        # WARMUP_STEPS+1 -> een oneindige reset-lus. Na de warmup moet elk
        # signaal zijn volledige streak opnieuw opbouwen.
        if self.steps <= settings.WARMUP_STEPS:
            self._impact_streak = 0
            self._flip_streak = 0
            self._offroad_streak = 0
            self._stuck_since = None
            self.last = res
            return res

        res.offroad_streak = self._offroad_streak
        res.stuck_seconds = 0.0 if self._stuck_since is None else (now - self._stuck_since)

        # --- beslissen (volgorde = prioriteit) ---
        # (a) + (b): plotse snelheidsval is het sterkste signaal, en telt
        # dubbel wanneer er ook een beeldschok bij komt. Remmen sluiten we uit.
        hard_drop = res.speed_drop >= settings.CRASH_SPEED_DROP and brake < 0.3
        impact = self._impact_streak >= settings.IMPACT_CONSECUTIVE
        if hard_drop or (impact and res.speed_drop >= settings.CRASH_SPEED_DROP * 0.5 and brake < 0.3):
            res.done, res.reason = True, "crash"
        # (c) omgeslagen
        elif self._flip_streak >= settings.FLIP_CONSECUTIVE:
            res.done, res.reason = True, "flip"
        # (d) van de weg af
        elif self._offroad_streak >= settings.OFFROAD_CONSECUTIVE:
            res.done, res.reason = True, "offroad"
        # (e) vastzitten
        elif res.stuck_seconds >= settings.STUCK_SECONDS:
            res.done, res.reason = True, "stuck"

        self.last = res
        return res

    def notify_gap(self) -> None:
        """
        Aanroepen na een onderbreking (PPO-update, pauze). De tijdreeksen
        kloppen dan niet meer: een 'gat' van 2 s zou anders als een plotse
        snelheidsval gelezen worden.
        """
        self._speed_hist.clear()
        self._impact_streak = 0
        self._stuck_since = None
