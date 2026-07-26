"""
viewer.py
=========
Het live OpenCV-venster dat je naast BeamNG zet terwijl de agent traint.

Wat je te zien krijgt:
  LINKS BOVEN   de observatie zoals het CNN hem ziet (84x84 grayscale,
                opgeschaald zodat je hem kan lezen) + de laatste frames uit
                de stack als strip eronder
  LINKS ONDER   de voorbewerkte HUD-crop die naar OCR gaat -> zo zie je
                meteen of je snelheidskalibratie deugt
  MIDDEN BOVEN  grote snelheidsweergave + doelsnelheid + reward van deze stap
  MIDDEN        balken voor stuur / gas / rem zoals ze naar de controller gaan
  RECHTS        detectie-status (wegdek-fractie, frame-diff, streaks)
  ONDER         reward-over-time: lijngrafiek van de laatste N stappen +
                balkjes met het totaal per afgelopen episode

Toetsen (met het viewer-venster actief):
  q  training netjes stoppen en het model opslaan
  p  pauzeren / hervatten (controls gaan neutraal)
  r  "auto staat klaar" tijdens een handmatige reset
  d  debug-overlay aan/uit (wegdek-masker)

LET OP: OpenCV-GUI wil alle calls vanuit dezelfde thread. Alles hier wordt
aangeroepen vanuit env.step()/env.reset(), dus vanuit de main thread.
"""

from __future__ import annotations

from collections import deque
from typing import Deque, Optional

import cv2
import numpy as np

import settings

# Kleuren (BGR)
BG = (24, 24, 28)
PANEL = (38, 38, 44)
FG = (235, 235, 240)
DIM = (150, 150, 160)
ACCENT = (90, 200, 255)
GOOD = (90, 220, 120)
WARN = (60, 190, 255)
BAD = (80, 90, 245)
GRID = (60, 60, 68)


def _text(img, txt, org, scale=0.5, color=FG, thick=1):
    cv2.putText(img, txt, org, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thick, cv2.LINE_AA)


def _panel(img, x, y, w, h, title=None):
    cv2.rectangle(img, (x, y), (x + w, y + h), PANEL, -1)
    cv2.rectangle(img, (x, y), (x + w, y + h), GRID, 1)
    if title:
        _text(img, title, (x + 8, y + 18), 0.44, DIM)


class LiveViewer:
    """Tekent één samengesteld dashboardvenster."""

    def __init__(self, enabled: bool = True) -> None:
        self.enabled = enabled
        self.width, self.height = settings.VIEWER_SIZE
        self.reward_hist: Deque[float] = deque(maxlen=settings.VIEWER_REWARD_HISTORY)
        self.episode_hist: Deque[float] = deque(maxlen=settings.VIEWER_EPISODE_HISTORY)
        self.frame_strip: Deque[np.ndarray] = deque(maxlen=settings.FRAME_STACK)
        self.show_debug = True
        self._window_ready = False
        self._last_key = -1

    # -- venster -----------------------------------------------------------
    def _ensure_window(self) -> None:
        if not self._window_ready:
            cv2.namedWindow(settings.VIEWER_WINDOW, cv2.WINDOW_AUTOSIZE)
            self._window_ready = True

    def close(self) -> None:
        if self._window_ready:
            try:
                cv2.destroyWindow(settings.VIEWER_WINDOW)
            except Exception:
                pass
            self._window_ready = False

    @property
    def last_key(self) -> int:
        return self._last_key

    # -- data voeden -------------------------------------------------------
    def push_reward(self, r: float) -> None:
        self.reward_hist.append(float(r))

    def push_episode(self, total: float) -> None:
        self.episode_hist.append(float(total))

    # -- hoofdrender -------------------------------------------------------
    def render(self, s: dict) -> int:
        """
        `s` is een dict met de huidige state (zie beamng_env._viewer_state).
        Geeft de ingedrukte toets terug (of -1).
        """
        if not self.enabled:
            return -1
        self._ensure_window()

        canvas = np.full((self.height, self.width, 3), BG, dtype=np.uint8)

        self._draw_observation(canvas, s)
        self._draw_ocr(canvas, s)
        self._draw_telemetry(canvas, s)
        self._draw_inputs(canvas, s)
        self._draw_detection(canvas, s)
        self._draw_graphs(canvas)
        self._draw_banner(canvas, s)

        cv2.imshow(settings.VIEWER_WINDOW, canvas)
        self._last_key = cv2.waitKey(1) & 0xFF
        if self._last_key == ord("d"):
            self.show_debug = not self.show_debug
        return self._last_key

    # -- onderdelen --------------------------------------------------------
    def _draw_observation(self, c, s):
        x, y, w, h = 10, 10, 250, 300
        _panel(c, x, y, w, h, "OBSERVATIE (input van het CNN)")

        obs = s.get("obs")
        if obs is not None:
            img = obs[:, :, 0] if obs.ndim == 3 else obs
            big = cv2.resize(img, (230, 230), interpolation=cv2.INTER_NEAREST)
            c[y + 26:y + 256, x + 10:x + 240] = cv2.cvtColor(big, cv2.COLOR_GRAY2BGR)
            self.frame_strip.append(cv2.resize(img, (52, 52), interpolation=cv2.INTER_AREA))

        # Strip met de laatste frames -> zo zie je de framestack "bewegen".
        _text(c, f"framestack ({settings.FRAME_STACK})", (x + 10, y + 274), 0.38, DIM)
        for i, f in enumerate(list(self.frame_strip)[-4:]):
            fx = x + 10 + i * 58
            c[y + 280:y + 296, fx:fx + 52] = cv2.cvtColor(
                cv2.resize(f, (52, 16), interpolation=cv2.INTER_AREA), cv2.COLOR_GRAY2BGR
            )

    def _draw_ocr(self, c, s):
        x, y, w, h = 10, 320, 250, 130
        _panel(c, x, y, w, h, "HUD-CROP NAAR OCR")
        dbg = s.get("ocr_image")
        if dbg is not None:
            img = dbg if dbg.ndim == 3 else cv2.cvtColor(dbg, cv2.COLOR_GRAY2BGR)
            th, tw = 70, 230
            scale = min(tw / img.shape[1], th / img.shape[0])
            nw, nh = max(1, int(img.shape[1] * scale)), max(1, int(img.shape[0] * scale))
            small = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA)
            c[y + 30:y + 30 + nh, x + 10:x + 10 + nw] = small
        else:
            _text(c, "OCR uit / geen beeld", (x + 12, y + 60), 0.42, DIM)

        reads, fails = s.get("ocr_stats", (0, 0))
        rate = 100.0 * (1.0 - fails / reads) if reads else 0.0
        _text(c, f"leesbaar: {rate:5.1f}%  ({reads} pogingen)", (x + 10, y + 108), 0.4, DIM)

    def _draw_telemetry(self, c, s):
        x, y, w, h = 270, 10, 330, 150
        _panel(c, x, y, w, h, "SNELHEID / REWARD")

        speed = s.get("speed")
        if speed is None:
            _text(c, "-- km/u", (x + 14, y + 68), 1.25, BAD, 2)
            _text(c, "OCR leest niets", (x + 16, y + 92), 0.42, BAD)
        else:
            dev = abs(speed - settings.TARGET_SPEED)
            col = GOOD if dev <= settings.SPEED_PEAK_BAND else (WARN if dev <= 20 else BAD)
            _text(c, f"{speed:5.0f}", (x + 14, y + 76), 1.9, col, 3)
            _text(c, "km/u", (x + 150, y + 76), 0.6, DIM)
            _text(c, f"doel {settings.TARGET_SPEED:.0f} (+/-{settings.SPEED_PEAK_BAND:.0f})",
                  (x + 16, y + 100), 0.44, DIM)

        # Doelsnelheidsbalk: markeer de piekband.
        bx, by, bw, bh = x + 14, y + 112, w - 28, 12
        cv2.rectangle(c, (bx, by), (bx + bw, by + bh), (52, 52, 60), -1)
        vmax = 120.0
        lo = int(bw * max(0.0, settings.TARGET_SPEED - settings.SPEED_PEAK_BAND) / vmax)
        hi = int(bw * min(vmax, settings.TARGET_SPEED + settings.SPEED_PEAK_BAND) / vmax)
        cv2.rectangle(c, (bx + lo, by), (bx + hi, by + bh), (60, 110, 70), -1)
        if speed is not None:
            px = bx + int(bw * min(max(speed, 0.0), vmax) / vmax)
            cv2.line(c, (px, by - 3), (px, by + bh + 3), ACCENT, 2)

        r = s.get("reward", 0.0)
        rcol = GOOD if r > 0.4 else (WARN if r > -0.2 else BAD)
        _text(c, f"reward stap: {r:+.3f}", (x + 14, y + 142), 0.52, rcol)

        # Opsplitsing van de reward-termen.
        x2, y2, w2, h2 = 610, 10, 340, 150
        _panel(c, x2, y2, w2, h2, "REWARD-OPSPLITSING")
        parts = s.get("reward_parts", {})
        rows = [("snelheid", "r_speed"), ("vooruitgang", "r_progress"),
                ("stilstand", "r_idle"), ("stuur-jerk", "r_jerk"),
                ("terminaal", "r_terminal")]
        for i, (label, key) in enumerate(rows):
            val = parts.get(key, 0.0)
            yy = y2 + 40 + i * 21
            _text(c, label, (x2 + 12, yy), 0.42, DIM)
            _text(c, f"{val:+.3f}", (x2 + 118, yy), 0.42,
                  GOOD if val > 0.001 else (BAD if val < -0.001 else DIM))
            # Mini-balkje rond een nulpunt.
            zx = x2 + 200
            span = 110
            cv2.line(c, (zx, yy - 4), (zx, yy + 2), GRID, 1)
            wpx = int(max(-1.0, min(1.0, val)) * span * 0.5)
            if wpx:
                cv2.rectangle(c, (zx, yy - 8), (zx + wpx, yy - 1),
                              GOOD if wpx > 0 else BAD, -1)

    def _draw_inputs(self, c, s):
        x, y, w, h = 270, 170, 330, 130
        _panel(c, x, y, w, h, "CONTROLLER-OUTPUT")

        steer = s.get("steer", 0.0)
        throttle = s.get("throttle", 0.0)
        brake = s.get("brake", 0.0)

        # Stuurbalk rond het midden.
        bx, by, bw, bh = x + 14, y + 40, w - 28, 16
        cv2.rectangle(c, (bx, by), (bx + bw, by + bh), (52, 52, 60), -1)
        mid = bx + bw // 2
        cv2.line(c, (mid, by - 3), (mid, by + bh + 3), GRID, 1)
        sw = int(steer * (bw // 2))
        if sw:
            cv2.rectangle(c, (mid, by + 2), (mid + sw, by + bh - 2), ACCENT, -1)
        _text(c, f"stuur  {steer:+.2f}", (bx, by - 6), 0.42, DIM)

        for i, (label, val, col) in enumerate((("gas (RT)", throttle, GOOD), ("rem (LT)", brake, BAD))):
            yy = y + 76 + i * 26
            cv2.rectangle(c, (bx, yy), (bx + bw, yy + 14), (52, 52, 60), -1)
            cv2.rectangle(c, (bx, yy), (bx + int(bw * val), yy + 14), col, -1)
            _text(c, f"{label} {val:.2f}", (bx + 4, yy + 11), 0.38, FG)

    def _draw_detection(self, c, s):
        x, y, w, h = 610, 170, 340, 130
        _panel(c, x, y, w, h, "CRASH- / OFFROAD-DETECTIE")

        det = s.get("detection", {})
        frac = det.get("road_fraction", 1.0)
        fcol = GOOD if frac >= settings.ROAD_MIN_FRACTION else BAD
        _text(c, f"wegdek in ROI : {frac * 100:5.1f}%  (min {settings.ROAD_MIN_FRACTION * 100:.0f}%)",
              (x + 12, y + 40), 0.42, fcol)
        _text(c, f"frame-diff    : {det.get('frame_diff', 0.0):5.1f}  (max {settings.IMPACT_DIFF_THRESHOLD:.0f})",
              (x + 12, y + 60), 0.42, DIM)
        _text(c, f"snelheidsval  : {det.get('speed_drop', 0.0):5.1f} km/u", (x + 12, y + 80), 0.42, DIM)
        _text(c, f"offroad-streak: {det.get('offroad_streak', 0)}/{settings.OFFROAD_CONSECUTIVE}"
                 f"   vast: {det.get('stuck_seconds', 0.0):.1f}s", (x + 12, y + 100), 0.42, DIM)

        # Wegdek-masker als klein debugbeeld rechtsonder in dit paneel.
        mask = s.get("road_mask")
        if self.show_debug and mask is not None and mask.size:
            m = cv2.resize(mask, (72, 30), interpolation=cv2.INTER_NEAREST)
            c[y + 88:y + 118, x + w - 84:x + w - 12] = cv2.cvtColor(m, cv2.COLOR_GRAY2BGR)

    def _draw_graphs(self, c):
        # (1) reward per stap, lijngrafiek (rechts naast de observatie)
        x, y, w, h = 270, 310, 680, 140
        _panel(c, x, y, w, h, f"REWARD PER STAP (laatste {settings.VIEWER_REWARD_HISTORY})")
        self._line_plot(c, x + 10, y + 26, w - 20, h - 40, list(self.reward_hist))

        # (2) totale reward per episode, balkjes (volle breedte onderaan)
        x, y, w, h = 10, 460, 940, 150
        _panel(c, x, y, w, h, f"TOTALE REWARD PER EPISODE (laatste {settings.VIEWER_EPISODE_HISTORY})")
        self._bar_plot(c, x + 10, y + 26, w - 20, h - 40, list(self.episode_hist))

    @staticmethod
    def _line_plot(c, x, y, w, h, data):
        cv2.rectangle(c, (x, y), (x + w, y + h), (30, 30, 36), -1)
        if len(data) < 2:
            _text(c, "wachten op data...", (x + 10, y + h // 2), 0.42, DIM)
            return
        lo, hi = min(data), max(data)
        if hi - lo < 1e-6:
            lo, hi = lo - 0.5, hi + 0.5
        pad = 0.08 * (hi - lo)
        lo, hi = lo - pad, hi + pad

        def ypix(v):
            return int(y + h - (v - lo) / (hi - lo) * h)

        # Nullijn
        if lo < 0 < hi:
            zy = ypix(0.0)
            cv2.line(c, (x, zy), (x + w, zy), GRID, 1)
            _text(c, "0", (x + 3, zy - 3), 0.34, GRID)

        step = w / max(len(data) - 1, 1)
        pts = [(int(x + i * step), ypix(v)) for i, v in enumerate(data)]
        cv2.polylines(c, [np.array(pts, np.int32)], False, ACCENT, 1, cv2.LINE_AA)

        # Voortschrijdend gemiddelde over 25 stappen.
        if len(data) >= 25:
            k = 25
            ma = np.convolve(np.array(data, np.float32), np.ones(k) / k, mode="valid")
            off = len(data) - len(ma)
            mpts = [(int(x + (i + off) * step), ypix(float(v))) for i, v in enumerate(ma)]
            cv2.polylines(c, [np.array(mpts, np.int32)], False, GOOD, 2, cv2.LINE_AA)

        _text(c, f"{hi:+.2f}", (x + w - 52, y + 12), 0.36, DIM)
        _text(c, f"{lo:+.2f}", (x + w - 52, y + h - 4), 0.36, DIM)

    @staticmethod
    def _bar_plot(c, x, y, w, h, data):
        cv2.rectangle(c, (x, y), (x + w, y + h), (30, 30, 36), -1)
        if not data:
            _text(c, "nog geen episode afgerond...", (x + 10, y + h // 2), 0.42, DIM)
            return
        lo, hi = min(min(data), 0.0), max(max(data), 0.0)
        if hi - lo < 1e-6:
            hi = lo + 1.0

        def ypix(v):
            return int(y + h - (v - lo) / (hi - lo) * h)

        zy = ypix(0.0)
        cv2.line(c, (x, zy), (x + w, zy), GRID, 1)
        bw = max(2, int(w / max(len(data), 1)) - 2)
        for i, v in enumerate(data):
            bx = int(x + i * (w / max(len(data), 1)))
            cv2.rectangle(c, (bx, ypix(v)), (bx + bw, zy), GOOD if v >= 0 else BAD, -1)

        _text(c, f"max {hi:+.1f}", (x + w - 90, y + 12), 0.36, DIM)
        _text(c, f"laatste {data[-1]:+.1f}", (x + w - 90, y + h - 4), 0.36, DIM)

    def _draw_banner(self, c, s):
        y = self.height - 12
        ep = s.get("episode", 0)
        step = s.get("step", 0)
        total = s.get("total_steps", 0)
        ep_r = s.get("episode_reward", 0.0)
        mean_v = s.get("mean_speed", 0.0)
        status = s.get("status", "")

        _text(c, f"episode {ep}  |  stap {step}  |  totaal {total}  |  "
                 f"ep-reward {ep_r:+.1f}  |  gem. snelheid {mean_v:.1f} km/u",
              (12, y - 18), 0.44, FG)
        _text(c, "q=stop en opslaan   p=pauze   r=auto klaar   d=debug",
              (12, y), 0.4, DIM)

        if status:
            col = BAD if any(k in status.lower() for k in ("crash", "offroad", "flip", "vast")) else WARN
            (tw, th), _ = cv2.getTextSize(status, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv2.rectangle(c, (self.width - tw - 26, y - 34), (self.width - 10, y - 34 + th + 14), col, -1)
            _text(c, status, (self.width - tw - 18, y - 34 + th + 4), 0.6, (20, 20, 24), 2)

    # -- speciale schermen -------------------------------------------------
    def show_message(self, title: str, lines, wait_ms: int = 30) -> int:
        """Groot bericht (gebruikt tijdens reset/pauze). Geeft de toets terug."""
        if not self.enabled:
            return -1
        self._ensure_window()
        c = np.full((self.height, self.width, 3), BG, dtype=np.uint8)
        _panel(c, 40, 40, self.width - 80, self.height - 80)
        _text(c, title, (70, 110), 1.0, ACCENT, 2)
        for i, line in enumerate(lines):
            _text(c, line, (70, 165 + i * 32), 0.6, FG)
        cv2.imshow(settings.VIEWER_WINDOW, c)
        self._last_key = cv2.waitKey(wait_ms) & 0xFF
        return self._last_key
