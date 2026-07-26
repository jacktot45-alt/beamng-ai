"""
calibrate.py
============
Interactieve kalibratie. Één keer doen (per schermresolutie / UI-schaal);
het resultaat gaat naar config.json en wordt daarna automatisch hergebruikt.

Run met:  python train.py --calibrate     (of: python calibrate.py)

Stappen:
  1. Monitor kiezen
  2. Het BeamNG-venster aanduiden met de muis (sleep een kader)
  3. De cijfers van de snelheidsmeter aanduiden met de muis
  4. OCR live testen en de voorbewerking bijstellen tot het klopt
  5. De kleur van het wegdek samplen (auto op de weg parkeren)
  6. Reset-modus kiezen
"""

from __future__ import annotations

import time
from typing import Optional, Tuple

import cv2
import numpy as np

import settings
from capture import ScreenCapture, grab_monitor, list_monitors
from config_store import CONFIG_PATH, Config, OcrPrep, Region, load, save
from detectors import RoadDetector
from speed_ocr import build_engine, preprocess_speed_crop
from utils import enable_dpi_awareness

WIN = "Kalibratie - BeamNG RL"
MAX_PREVIEW_W = 1280   # schermafbeelding wordt hierop geschaald om te passen


# ---------------------------------------------------------------------------
# Hulpjes
# ---------------------------------------------------------------------------


def _fit(img: np.ndarray, max_w: int = MAX_PREVIEW_W) -> Tuple[np.ndarray, float]:
    """Schaal een beeld zodat het op het scherm past. Geeft (beeld, schaal)."""
    h, w = img.shape[:2]
    if w <= max_w:
        return img, 1.0
    scale = max_w / w
    return cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA), scale


def _banner(img: np.ndarray, lines) -> np.ndarray:
    """Plak een instructiebalk bovenaan een beeld."""
    out = img.copy()
    bar_h = 26 * len(lines) + 14
    cv2.rectangle(out, (0, 0), (out.shape[1], bar_h), (20, 20, 24), -1)
    for i, line in enumerate(lines):
        cv2.putText(out, line, (14, 24 + i * 26), cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (240, 240, 245), 1, cv2.LINE_AA)
    return out


def _select_roi(img: np.ndarray, lines) -> Optional[Tuple[int, int, int, int]]:
    """cv2.selectROI met instructiebalk. Geeft (x, y, w, h) in beeldcoördinaten."""
    shown = _banner(img, lines)
    cv2.namedWindow(WIN, cv2.WINDOW_AUTOSIZE)
    box = cv2.selectROI(WIN, shown, showCrosshair=True, fromCenter=False)
    cv2.destroyWindow(WIN)
    cv2.waitKey(1)
    x, y, w, h = (int(v) for v in box)
    if w <= 0 or h <= 0:
        return None
    return x, y, w, h


def _ask(prompt: str, default: str = "") -> str:
    suffix = f" [{default}]" if default else ""
    try:
        answer = input(f"{prompt}{suffix}: ").strip()
    except EOFError:
        answer = ""
    return answer or default


# ---------------------------------------------------------------------------
# Stap 1: monitor
# ---------------------------------------------------------------------------


def step_monitor(cfg: Config) -> None:
    mons = list_monitors()
    print("\n--- STAP 1/6: Welke monitor draait BeamNG? ---")
    for i, m in enumerate(mons):
        label = "alle monitors samen" if i == 0 else f"monitor {i}"
        print(f"  [{i}] {label}: {m['width']}x{m['height']} op ({m['left']}, {m['top']})")
    default = str(cfg.monitor_index if cfg.monitor_index < len(mons) else 1)
    while True:
        raw = _ask("Kies een nummer", default)
        try:
            idx = int(raw)
            if 0 <= idx < len(mons):
                cfg.monitor_index = idx
                return
        except ValueError:
            pass
        print("  Ongeldige keuze.")


# ---------------------------------------------------------------------------
# Stap 2: game-venster
# ---------------------------------------------------------------------------


def step_game_region(cfg: Config) -> None:
    print("\n--- STAP 2/6: Waar staat het BeamNG-venster? ---")
    print("  Zet BeamNG in beeld (windowed of borderless) en kom hier terug.")
    print("  Er wordt zo een screenshot genomen; sleep daarin een kader rond")
    print("  het SPELBEELD (zonder titelbalk/vensterrand).")
    input("  Druk ENTER om de screenshot te nemen... ")
    time.sleep(0.4)

    full, mon = grab_monitor(cfg.monitor_index)
    preview, scale = _fit(full)
    box = _select_roi(preview, [
        "Sleep een kader rond het BEAMNG-SPELBEELD (niet de vensterrand).",
        "ENTER of SPATIE = bevestigen   |   C = annuleren (hele monitor)",
    ])

    if box is None:
        print("  Geen selectie -> de hele monitor wordt gebruikt.")
        cfg.game_region = Region(mon["left"], mon["top"], mon["width"], mon["height"])
    else:
        x, y, w, h = box
        cfg.game_region = Region(
            left=mon["left"] + int(x / scale),
            top=mon["top"] + int(y / scale),
            width=int(w / scale),
            height=int(h / scale),
        )
    r = cfg.game_region
    print(f"  Game-regio: {r.width}x{r.height} op ({r.left}, {r.top})")


# ---------------------------------------------------------------------------
# Stap 3: snelheidsmeter
# ---------------------------------------------------------------------------


def step_speed_region(cfg: Config) -> None:
    print("\n--- STAP 3/6: Waar staan de CIJFERS van de snelheidsmeter? ---")
    print("  Zorg dat de HUD zichtbaar is en de auto ergens rijdt of stilstaat.")
    print("  Selecteer STRAK rond enkel de cijfers - dus zonder 'km/h', zonder")
    print("  de toerenteller en zonder versnellingsindicator. Hoe strakker,")
    print("  hoe betrouwbaarder de OCR.")
    input("  Druk ENTER om een screenshot van het spel te nemen... ")
    time.sleep(0.4)

    cap = ScreenCapture(cfg.game_region.as_mss())
    frame = cap.grab()
    cap.close()

    preview, scale = _fit(frame)
    box = _select_roi(preview, [
        "Sleep een strak kader rond ENKEL DE CIJFERS van de snelheidsmeter.",
        "ENTER of SPATIE = bevestigen   |   C = overslaan (geen OCR)",
    ])
    if box is None:
        print("  Overgeslagen: er wordt geen snelheid gelezen (reward valt terug op beweging).")
        cfg.speed_region = None
        return

    g = cfg.game_region
    x, y, w, h = box
    cfg.speed_region = Region(
        left=g.left + int(x / scale),
        top=g.top + int(y / scale),
        width=max(int(w / scale), 8),
        height=max(int(h / scale), 8),
    )
    r = cfg.speed_region
    print(f"  Snelheidsregio: {r.width}x{r.height} op ({r.left}, {r.top})")


# ---------------------------------------------------------------------------
# Stap 4: OCR afstellen
# ---------------------------------------------------------------------------


def step_ocr_tuning(cfg: Config) -> None:
    if cfg.speed_region is None or settings.OCR_BACKEND == "none":
        return

    print("\n--- STAP 4/6: OCR live testen ---")
    print("  Er opent een venster met de voorbewerkte crop en de gelezen waarde.")
    print("  Rijd wat rond (of laat de auto stilstaan) en kijk of het getal klopt.")
    print("  Toetsen:  i = inverteren   t = drempelmethode   +/- = upscale")
    print("            ENTER = opslaan en verder")

    try:
        engine = build_engine(settings.OCR_BACKEND)
    except ImportError as exc:
        print(f"  OCR-backend niet beschikbaar: {exc}")
        print("  Je kan later settings.OCR_BACKEND aanpassen. Stap overgeslagen.")
        return

    prep = cfg.ocr_prep
    thresholds = ["otsu", "adaptive", "fixed", "none"]
    cap = ScreenCapture(cfg.speed_region.as_mss())
    cv2.namedWindow(WIN, cv2.WINDOW_AUTOSIZE)

    last_value, last_ok, total = None, 0, 0
    try:
        while True:
            raw = cap.grab()
            prepped = preprocess_speed_crop(raw, prep)
            value = engine.read(prepped)
            total += 1
            if value is not None:
                last_value, last_ok = value, last_ok + 1

            vis = cv2.cvtColor(prepped, cv2.COLOR_GRAY2BGR) if prepped.ndim == 2 else prepped
            vis, _ = _fit(vis, 520)
            canvas = np.full((vis.shape[0] + 132, max(vis.shape[1], 520), 3), (24, 24, 28), np.uint8)
            canvas[110:110 + vis.shape[0], 0:vis.shape[1]] = vis

            colour = (90, 220, 120) if value is not None else (80, 90, 245)
            shown = f"{value:.0f} km/u" if value is not None else "-- geen cijfers gelezen --"
            cv2.putText(canvas, shown, (14, 44), cv2.FONT_HERSHEY_SIMPLEX, 1.0, colour, 2, cv2.LINE_AA)
            rate = 100.0 * last_ok / total if total else 0.0
            cv2.putText(canvas, f"leesbaar: {rate:.0f}%   laatste geldige: "
                                f"{'-' if last_value is None else int(last_value)}",
                        (14, 72), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 160), 1, cv2.LINE_AA)
            cv2.putText(canvas, f"invert={prep.invert}  threshold={prep.threshold}  scale={prep.scale}"
                                f"   [i/t/+/-]  ENTER=klaar",
                        (14, 96), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (200, 200, 210), 1, cv2.LINE_AA)

            cv2.imshow(WIN, canvas)
            key = cv2.waitKey(60) & 0xFF
            if key in (13, 10):      # ENTER
                break
            if key == ord("i"):
                prep.invert = not prep.invert
                last_ok, total = 0, 0
            elif key == ord("t"):
                prep.threshold = thresholds[(thresholds.index(prep.threshold) + 1) % len(thresholds)]
                last_ok, total = 0, 0
            elif key in (ord("+"), ord("=")):
                prep.scale = min(prep.scale + 1, 8)
                last_ok, total = 0, 0
            elif key in (ord("-"), ord("_")):
                prep.scale = max(prep.scale - 1, 1)
                last_ok, total = 0, 0
    finally:
        cap.close()
        cv2.destroyWindow(WIN)
        cv2.waitKey(1)

    cfg.ocr_prep = prep
    print(f"  Opgeslagen: invert={prep.invert}, threshold={prep.threshold}, scale={prep.scale}")


# ---------------------------------------------------------------------------
# Stap 5: wegkleur
# ---------------------------------------------------------------------------


def step_road_colour(cfg: Config) -> None:
    print("\n--- STAP 5/6: Kleur van het wegdek samplen ---")
    print("  Zet de auto MIDDEN OP DE WEG (stilstaand is prima), met normaal")
    print("  daglicht. Het groene kader in het venster is de zone die gesampled")
    print("  wordt; daar moet enkel asfalt in zitten - geen berm, geen wegmarkering.")
    print("  Toetsen:  SPATIE = samplen   ENTER = klaar   C = overslaan")

    cap = ScreenCapture(cfg.game_region.as_mss())
    detector = RoadDetector(cfg.road_hsv)
    cv2.namedWindow(WIN, cv2.WINDOW_AUTOSIZE)

    try:
        while True:
            frame = cap.grab()
            x0, y0, x1, y1 = RoadDetector.roi_pixels(frame)
            frac = detector.update(frame) if detector.enabled else 0.0

            vis = frame.copy()
            cv2.rectangle(vis, (x0, y0), (x1, y1), (90, 220, 120), 2)
            vis, _ = _fit(vis, 1000)
            lines = ["SPATIE = wegkleur samplen   |   ENTER = klaar   |   C = overslaan"]
            if detector.enabled:
                lines.append(f"HSV-referentie {detector.road_hsv}   ->   wegdek in kader: {frac * 100:.0f}%")
            else:
                lines.append("Nog niets gesampled.")
            vis = _banner(vis, lines)

            # Masker als klein voorbeeldje rechtsonder tonen.
            if detector.enabled and detector.last_mask is not None and detector.last_mask.size:
                m = cv2.resize(detector.last_mask, (160, 70), interpolation=cv2.INTER_NEAREST)
                vis[vis.shape[0] - 80:vis.shape[0] - 10, vis.shape[1] - 170:vis.shape[1] - 10] = \
                    cv2.cvtColor(m, cv2.COLOR_GRAY2BGR)

            cv2.imshow(WIN, vis)
            key = cv2.waitKey(40) & 0xFF
            if key == 32:  # SPATIE
                hsv = detector.sample_reference(frame)
                cfg.road_hsv = list(hsv)
                print(f"  Gesampled: HSV={hsv}")
            elif key in (13, 10):
                break
            elif key in (ord("c"), ord("C"), 27):
                print("  Overgeslagen: offroad-detectie staat uit "
                      "(crash-detectie op snelheid/impact blijft wel werken).")
                cfg.road_hsv = None
                break
    finally:
        cap.close()
        cv2.destroyWindow(WIN)
        cv2.waitKey(1)


# ---------------------------------------------------------------------------
# Stap 6: reset-modus
# ---------------------------------------------------------------------------


def step_reset_mode(cfg: Config) -> None:
    print("\n--- STAP 6/6: Hoe wordt de auto na een crash gereset? ---")
    print("  [1] gamepad  - de agent drukt op knop "
          f"{settings.RESET_GAMEPAD_BUTTON} van de virtuele controller.")
    print("                 Bind die knop in BeamNG onder")
    print("                 Options > Controls > zoek 'Recover Vehicle'.")
    print("                 Volledig automatisch: aanbevolen voor lange sessies.")
    print("  [2] manual   - jij zet de auto klaar en drukt op R. Geen setup nodig.")
    print("  [3] key      - er wordt een INSERT-toets gesimuleerd (BeamNG's")
    print("                 standaard recover-toets). Enkel voor resetten.")
    mapping = {"1": "gamepad", "2": "manual", "3": "key"}
    default = {"gamepad": "1", "manual": "2", "key": "3"}.get(cfg.reset_mode, "2")
    choice = _ask("Kies", default)
    cfg.reset_mode = mapping.get(choice, "manual")
    print(f"  Reset-modus: {cfg.reset_mode}")
    if cfg.reset_mode == "gamepad":
        print(f"  >> Vergeet niet knop {settings.RESET_GAMEPAD_BUTTON} in BeamNG te binden "
              f"aan 'Recover Vehicle'!")


# ---------------------------------------------------------------------------
# Hoofdroutine
# ---------------------------------------------------------------------------


def run_calibration(existing: Optional[Config] = None) -> Config:
    enable_dpi_awareness()
    cfg = existing or load() or Config()

    print("=" * 70)
    print("  KALIBRATIE BeamNG RL vision agent")
    print("=" * 70)
    print("  Zorg dat BeamNG.drive draait, in WINDOWED of BORDERLESS mode,")
    print("  met de HUD zichtbaar. Alt-tab gewoon heen en weer tussen dit")
    print("  venster en het spel.")

    step_monitor(cfg)
    step_game_region(cfg)
    step_speed_region(cfg)
    step_ocr_tuning(cfg)
    step_road_colour(cfg)
    step_reset_mode(cfg)

    save(cfg)
    print("\n" + "=" * 70)
    print(f"  Klaar. Alles staat in {CONFIG_PATH}")
    print("  Controleer je instellingen met:   python train.py --preview")
    print("  Start de training met:            python train.py")
    print("=" * 70)
    return cfg


if __name__ == "__main__":
    run_calibration()
