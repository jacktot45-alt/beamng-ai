"""
capture.py
==========
Screen capture (mss) + voorbewerking naar de observatie voor het CNN,
plus een simpele optical-flow schatter die als "vooruitgang"-proxy dient.

Belangrijk: een `mss.mss()` instantie is niet thread-safe. Elke thread die
grabt maakt zijn eigen instantie (de OCR-thread doet dat ook).
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import cv2
import numpy as np

try:
    import mss
except ImportError as exc:  # pragma: no cover
    raise ImportError("mss ontbreekt. Installeer met:  pip install mss") from exc

import settings


class ScreenCapture:
    """Grabt herhaaldelijk dezelfde schermregio als BGR-beeld."""

    def __init__(self, region: Dict[str, int]) -> None:
        self.region = dict(region)
        self._sct = mss.mss()

    def grab(self) -> np.ndarray:
        """Geeft een BGR uint8 array (h, w, 3)."""
        raw = self._sct.grab(self.region)
        frame = np.asarray(raw, dtype=np.uint8)  # BGRA
        return cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)

    def close(self) -> None:
        try:
            self._sct.close()
        except Exception:
            pass


def grab_monitor(monitor_index: int = 1) -> Tuple[np.ndarray, Dict[str, int]]:
    """Eenmalige screenshot van een volledige monitor (voor de kalibratie)."""
    with mss.mss() as sct:
        mon = sct.monitors[monitor_index]
        raw = sct.grab(mon)
        frame = cv2.cvtColor(np.asarray(raw, dtype=np.uint8), cv2.COLOR_BGRA2BGR)
        return frame, dict(mon)


def list_monitors() -> list:
    with mss.mss() as sct:
        return [dict(m) for m in sct.monitors]


# ---------------------------------------------------------------------------
# Voorbewerking naar de observatie
# ---------------------------------------------------------------------------


def crop_fractional(frame: np.ndarray, crop: Tuple[float, float, float, float]) -> np.ndarray:
    """Crop met fracties (top, bottom, left, right) van settings.OBS_CROP."""
    h, w = frame.shape[:2]
    top, bottom, left, right = crop
    y0, y1 = int(h * top), int(h * bottom)
    x0, x1 = int(w * left), int(w * right)
    # Nooit een lege crop teruggeven, ook niet bij rare instellingen.
    y1 = max(y1, y0 + 1)
    x1 = max(x1, x0 + 1)
    return frame[y0:y1, x0:x1]


def preprocess_observation(frame_bgr: np.ndarray) -> np.ndarray:
    """
    Game-frame -> observatie voor SB3.

    Stappen: fractionele crop (lucht/HUD weg) -> grayscale -> resize naar
    OBS_SIZE. Resultaat is (H, W, 1) uint8, wat SB3 als image space herkent.
    De framestacking van 4 frames doet VecFrameStack in train.py.
    """
    cropped = crop_fractional(frame_bgr, settings.OBS_CROP)
    gray = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)
    small = cv2.resize(gray, settings.OBS_SIZE, interpolation=cv2.INTER_AREA)
    return small[:, :, None]


# ---------------------------------------------------------------------------
# Bewegingsschatter (proxy voor "ik ga vooruit")
# ---------------------------------------------------------------------------


class MotionEstimator:
    """
    Schat hoeveel beeld er beweegt met Farneback optical flow op een sterk
    verkleind frame (goedkoop: enkele ms).

    We kijken enkel naar de onderste helft van het beeld: daar zit het wegdek
    dat langs de camera schuift. De lucht en verre bergen bewegen nauwelijks
    en zouden de schatting verwateren.

    De teruggegeven waarde is GEEN km/u, enkel een relatieve maat. Hij wordt
    gebruikt voor (a) de kleine vooruitgangsbonus in de reward en (b) als
    ruwe fallback wanneer OCR de snelheid niet kan lezen.
    """

    def __init__(self, size: Tuple[int, int] = (128, 72)) -> None:
        self.size = size
        self._prev: Optional[np.ndarray] = None
        self.last_flow: float = 0.0

    def reset(self) -> None:
        self._prev = None
        self.last_flow = 0.0

    def update(self, frame_bgr: np.ndarray) -> float:
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        small = cv2.resize(gray, self.size, interpolation=cv2.INTER_AREA)

        if self._prev is None:
            self._prev = small
            self.last_flow = 0.0
            return 0.0

        flow = cv2.calcOpticalFlowFarneback(
            self._prev, small, None,
            pyr_scale=0.5, levels=2, winsize=13,
            iterations=2, poly_n=5, poly_sigma=1.1, flags=0,
        )
        self._prev = small

        half = self.size[1] // 2
        mag = np.linalg.norm(flow[half:, :, :], axis=2)
        self.last_flow = float(np.mean(mag))
        return self.last_flow


class FrameDiffMeter:
    """
    Gemiddeld absoluut verschil tussen opeenvolgende frames (0-255).

    Een normale rit geeft een laag, gelijkmatig verschil. Een klap, een
    camera-shake of een rollover geeft een korte, forse piek. Zie detectors.py.
    """

    def __init__(self, size: Tuple[int, int] = (96, 54)) -> None:
        self.size = size
        self._prev: Optional[np.ndarray] = None
        self.last_diff: float = 0.0

    def reset(self) -> None:
        self._prev = None
        self.last_diff = 0.0

    def update(self, frame_bgr: np.ndarray) -> float:
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        small = cv2.resize(gray, self.size, interpolation=cv2.INTER_AREA)
        if self._prev is None:
            self._prev = small
            self.last_diff = 0.0
            return 0.0
        diff = cv2.absdiff(self._prev, small)
        self._prev = small
        self.last_diff = float(np.mean(diff))
        return self.last_diff
