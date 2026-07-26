"""
speed_ocr.py
============
Leest de snelheid van de BeamNG-HUD met OCR op een vaste crop van het scherm.

Ontwerpkeuze: OCR draait in een APARTE THREAD op zijn eigen tempo (OCR_HZ).
Reden: pytesseract kost 15-40 ms per call en easyocr nog meer. Als dat in de
control loop zou zitten, haal je de vaste timestep van 10 Hz niet meer. De
control loop leest gewoon de laatst bekende waarde uit (`reader.speed`).

Ruisfilter in drie lagen:
  1. whitelist op cijfers + plausibel bereik (0 .. SPEED_MAX_VALUE)
  2. snelheidslimiet op de verandering (SPEED_MAX_DELTA_PER_S) -> gooit
     leesfouten weg zoals "8" dat plots als "88" gelezen wordt
  3. houdbaarheidsdatum: een meting ouder dan SPEED_HOLD_S is "onbekend"

Backends: "pytesseract", "easyocr", of "none".
"""

from __future__ import annotations

import re
import threading
import time
from typing import Dict, Optional, Tuple

import cv2
import numpy as np

import settings
from config_store import OcrPrep

_DIGITS = re.compile(r"\d+")


# ---------------------------------------------------------------------------
# Voorbewerking van de HUD-crop
# ---------------------------------------------------------------------------


def preprocess_speed_crop(bgr: np.ndarray, prep: OcrPrep) -> np.ndarray:
    """
    Maakt van een klein HUD-cropje iets wat een OCR-engine graag leest:
    fors upscalen, grijswaarden, eventueel inverteren, drempelen en een
    witte rand eromheen (Tesseract houdt van marge).
    """
    img = bgr
    if prep.scale and prep.scale > 1:
        img = cv2.resize(img, None, fx=prep.scale, fy=prep.scale, interpolation=cv2.INTER_CUBIC)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # BeamNG-HUD is doorgaans lichte cijfers op een donkere achtergrond.
    # OCR-engines verwachten liever donkere tekst op wit -> standaard omkeren.
    if not prep.invert:
        gray = cv2.bitwise_not(gray)

    if prep.threshold == "otsu":
        _, gray = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    elif prep.threshold == "adaptive":
        gray = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 10
        )
    elif prep.threshold == "fixed":
        _, gray = cv2.threshold(gray, int(prep.thresh_value), 255, cv2.THRESH_BINARY)
    # "none" -> onbewerkt grijs doorgeven

    # Kleine gaten in de cijfers dichten (anti-aliasing van de HUD).
    gray = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, np.ones((2, 2), np.uint8))
    return cv2.copyMakeBorder(gray, 12, 12, 12, 12, cv2.BORDER_CONSTANT, value=255)


def parse_speed_text(text: str) -> Optional[float]:
    """
    Haal het meest waarschijnlijke getal uit een OCR-string.

    Twee lezingen worden geprobeerd, want OCR faalt op twee manieren:
      1. spaties weg -> vangt "1 2 0" op, één getal dat Tesseract opsplitst
         omdat de HUD-cijfers ver uit elkaar staan;
      2. spaties behouden -> vangt "1 120" op, waar die "1" ruis is (een
         randje van een icoon) en 120 de echte snelheid.
    Van alle plausibele kandidaten (0 .. SPEED_MAX_VALUE) wint de langste
    cijferreeks; die is bijna altijd de snelheid zelf.
    """
    if not text:
        return None

    candidates = []
    for variant in (text.replace(" ", ""), text):
        for token in _DIGITS.findall(variant):
            try:
                value = float(token)
            except ValueError:
                continue
            if 0 <= value <= settings.SPEED_MAX_VALUE:
                candidates.append((len(token), value))

    if not candidates:
        return None
    return max(candidates)[1]


# ---------------------------------------------------------------------------
# OCR-engines
# ---------------------------------------------------------------------------


class _PytesseractEngine:
    name = "pytesseract"

    def __init__(self) -> None:
        try:
            import pytesseract
        except ImportError as exc:
            raise ImportError("pip install pytesseract  (+ installeer Tesseract-OCR)") from exc
        self._pt = pytesseract
        if settings.TESSERACT_CMD:
            import os

            if os.path.exists(settings.TESSERACT_CMD):
                pytesseract.pytesseract.tesseract_cmd = settings.TESSERACT_CMD
        # psm 7 = één tekstregel, psm 8 = één woord. We proberen beide.
        self._configs = (
            "--oem 3 --psm 7 -c tessedit_char_whitelist=0123456789",
            "--oem 3 --psm 8 -c tessedit_char_whitelist=0123456789",
        )

    def read(self, img: np.ndarray) -> Optional[float]:
        for cfg in self._configs:
            try:
                txt = self._pt.image_to_string(img, config=cfg)
            except Exception:
                return None
            value = parse_speed_text(txt)
            if value is not None:
                return value
        return None


class _EasyOcrEngine:
    name = "easyocr"

    def __init__(self) -> None:
        try:
            import easyocr
        except ImportError as exc:
            raise ImportError("pip install easyocr") from exc
        print("[ocr] EasyOCR model laden (eerste keer duurt dit even)...")
        self._reader = easyocr.Reader(["en"], gpu=True, verbose=False)

    def read(self, img: np.ndarray) -> Optional[float]:
        try:
            parts = self._reader.readtext(img, allowlist="0123456789", detail=0, paragraph=False)
        except Exception:
            return None
        return parse_speed_text("".join(parts))


class _NullEngine:
    name = "none"

    def read(self, img: np.ndarray) -> Optional[float]:
        return None


def build_engine(backend: str):
    backend = (backend or "none").lower()
    if backend == "pytesseract":
        return _PytesseractEngine()
    if backend == "easyocr":
        return _EasyOcrEngine()
    return _NullEngine()


# ---------------------------------------------------------------------------
# De threaded reader
# ---------------------------------------------------------------------------


class SpeedReader:
    """
    Achtergrond-thread die de HUD-crop grabt en er een snelheid uit leest.

    Gebruik:
        reader = SpeedReader(region, prep)
        reader.start()
        ...
        v = reader.speed          # float km/u, of None als onbekend
        img = reader.debug_image  # laatste voorbewerkte crop (voor de viewer)
        reader.stop()
    """

    def __init__(self, region: Optional[Dict[str, int]], prep: OcrPrep,
                 backend: Optional[str] = None) -> None:
        self.region = dict(region) if region else None
        self.prep = prep
        self.backend_name = backend or settings.OCR_BACKEND
        self._engine = None

        self._lock = threading.Lock()
        self._thread: Optional[threading.Thread] = None
        self._running = False

        self._value: Optional[float] = None      # laatste GEACCEPTEERDE meting
        self._value_time: float = 0.0
        self._raw_value: Optional[float] = None  # laatste ruwe OCR-uitkomst
        self._debug_img: Optional[np.ndarray] = None
        self._reads = 0
        self._fails = 0

    # -- lifecycle ---------------------------------------------------------
    def start(self) -> None:
        if self.region is None or self.backend_name == "none":
            print("[ocr] Geen snelheidsregio of backend 'none': OCR staat uit.")
            return
        self._engine = build_engine(self.backend_name)
        self._running = True
        self._thread = threading.Thread(target=self._loop, name="SpeedOCR", daemon=True)
        self._thread.start()
        print(f"[ocr] Snelheidslezer gestart (backend={self._engine.name}, {settings.OCR_HZ:.0f} Hz)")

    def stop(self) -> None:
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None

    # -- publieke state ----------------------------------------------------
    @property
    def speed(self) -> Optional[float]:
        """Laatste geldige snelheid, of None als hij te oud/onbekend is."""
        with self._lock:
            if self._value is None:
                return None
            if time.perf_counter() - self._value_time > settings.SPEED_HOLD_S:
                return None
            return self._value

    @property
    def speed_or_zero(self) -> float:
        v = self.speed
        return 0.0 if v is None else v

    @property
    def debug_image(self) -> Optional[np.ndarray]:
        with self._lock:
            return None if self._debug_img is None else self._debug_img.copy()

    @property
    def stats(self) -> Tuple[int, int]:
        """(aantal pogingen, aantal mislukte lezingen)"""
        with self._lock:
            return self._reads, self._fails

    def reset_history(self) -> None:
        """Na een vehicle-reset: oude waarde vergeten zodat het deltafilter
        niet blijft hangen op de snelheid van vóór de crash."""
        with self._lock:
            self._value = None
            self._value_time = 0.0

    # -- interne loop ------------------------------------------------------
    def _loop(self) -> None:
        import mss  # eigen instantie per thread!

        period = 1.0 / max(settings.OCR_HZ, 0.5)
        with mss.mss() as sct:
            while self._running:
                t0 = time.perf_counter()
                try:
                    raw = np.asarray(sct.grab(self.region), dtype=np.uint8)
                    bgr = cv2.cvtColor(raw, cv2.COLOR_BGRA2BGR)
                    prepped = preprocess_speed_crop(bgr, self.prep)
                    value = self._engine.read(prepped)
                    self._accept(value, prepped)
                except Exception as exc:
                    # Nooit de trainingsloop laten sneuvelen op een OCR-hikje.
                    with self._lock:
                        self._fails += 1
                    if self._fails in (1, 50, 500):
                        print(f"[ocr] leesfout: {exc}")

                sleep = period - (time.perf_counter() - t0)
                if sleep > 0:
                    time.sleep(sleep)

    def _accept(self, value: Optional[float], debug_img: np.ndarray) -> None:
        """Plausibiliteitscheck + opslaan."""
        now = time.perf_counter()
        with self._lock:
            self._reads += 1
            self._debug_img = debug_img
            self._raw_value = value

            if value is None:
                self._fails += 1
                return

            if self._value is not None:
                dt = max(now - self._value_time, 1e-3)
                max_delta = settings.SPEED_MAX_DELTA_PER_S * dt
                # Ruime ondergrens zodat normale versnelling nooit geweigerd wordt.
                if abs(value - self._value) > max(max_delta, 8.0):
                    self._fails += 1
                    return

            self._value = value
            self._value_time = now
