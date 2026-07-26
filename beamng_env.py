"""
beamng_env.py
=============
De gymnasium.Env die BeamNG.drive behandelt als een black box:
kijken via screen capture, sturen via een virtuele Xbox-controller.
Geen BeamNGpy, geen telemetry, geen memory reading, geen versnelde simulatie.

Observatie : (84, 84, 1) uint8 grayscale. De framestacking van 4 frames
             gebeurt buiten de env met SB3's VecFrameStack (zie train.py).
Actie      : Box(-1, 1, shape=(2,))
                action[0] = sturen (-1 links .. +1 rechts)
                action[1] = gas/rem (>0 gas op RT, <0 rem op LT)
Reward     : zie reward.py
Done       : crash / omgeslagen / offroad / vastzitten (terminated)
             of MAX_EPISODE_STEPS bereikt (truncated)

Realtime-eigenaardigheden waar rekening mee gehouden is:
  * step() houdt een VASTE timestep aan (STEP_DT). Loopt een stap uit omdat
    het spel hapert, dan wordt dat gelogd maar niet ingehaald.
  * Tijdens een PPO-update staat de env stil terwijl de auto doorrijdt.
    train.py roept daarom on_train_pause()/on_train_resume() aan: gas los,
    beetje rem, en daarna alle tijdreeksen wissen (anders leest de detector
    dat gat als een crash).
"""

from __future__ import annotations

import time
from typing import Any, Dict, Optional, Tuple

import cv2
import gymnasium as gym
import numpy as np
from gymnasium import spaces

import settings
from capture import FrameDiffMeter, MotionEstimator, ScreenCapture, preprocess_observation
from config_store import Config
from controller import DummyGamepad, VirtualGamepad, decode_action
from detectors import EventDetector
from reward import compute_reward
from speed_ocr import SpeedReader
from utils import TerminalKeys, send_key
from viewer import LiveViewer


class BeamNGVisionEnv(gym.Env):
    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        cfg: Config,
        use_viewer: bool = True,
        use_gamepad: bool = True,
        verbose: bool = True,
    ) -> None:
        super().__init__()
        self.cfg = cfg
        self.verbose = verbose

        # --- spaces -------------------------------------------------------
        w, h = settings.OBS_SIZE
        self.observation_space = spaces.Box(low=0, high=255, shape=(h, w, 1), dtype=np.uint8)
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)

        # --- I/O ----------------------------------------------------------
        if cfg.game_region is None:
            raise ValueError("Geen game_region gekalibreerd. Run: python train.py --calibrate")
        self.capture = ScreenCapture(cfg.game_region.as_mss())
        self.pad = VirtualGamepad() if use_gamepad else DummyGamepad()
        self.speed_reader = SpeedReader(
            cfg.speed_region.as_mss() if cfg.speed_region else None, cfg.ocr_prep
        )
        self.speed_reader.start()

        # --- verwerking ---------------------------------------------------
        self.motion = MotionEstimator()
        self.framediff = FrameDiffMeter()
        self.detector = EventDetector(cfg.road_hsv)
        self.viewer = LiveViewer(enabled=use_viewer and settings.VIEWER_ENABLED)
        self.keys = TerminalKeys()

        # --- episode-state ------------------------------------------------
        self.episode = 0
        self.step_idx = 0
        self.total_steps = 0
        self.episode_reward = 0.0
        self.prev_steer = 0.0
        self._speed_sum = 0.0
        self._speed_n = 0
        self._last_step_time: Optional[float] = None
        self._last_frame: Optional[np.ndarray] = None
        self._last_obs = np.zeros((h, w, 1), dtype=np.uint8)
        self._last_reward_parts: Dict[str, float] = {}
        self._last_reward = 0.0
        self._status = ""
        self._timing_gap = True   # eerste stap na een gat: timing niet vertrouwen

        # --- besturing van buitenaf ---------------------------------------
        self.stop_requested = False   # 'q' -> train.py stopt netjes
        self.paused = False

    # ======================================================================
    # RESET
    # ======================================================================
    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)

        self.pad.neutral()
        self.episode += 1
        self.step_idx = 0
        self.episode_reward = 0.0
        self.prev_steer = 0.0
        self._speed_sum = 0.0
        self._speed_n = 0
        self._status = ""

        self._recover_vehicle()

        # Alle tijdafhankelijke state wissen: de auto staat nu ergens anders.
        self.motion.reset()
        self.framediff.reset()
        self.detector.reset()
        self.speed_reader.reset_history()
        self._last_step_time = None
        self._timing_gap = True

        frame = self.capture.grab()
        self._last_frame = frame
        self.motion.update(frame)
        self.framediff.update(frame)
        obs = preprocess_observation(frame)
        self._last_obs = obs

        if self.verbose:
            print(f"[env] --- Episode {self.episode} gestart ---", flush=True)
        return obs, {}

    def _recover_vehicle(self) -> None:
        """
        Zet de auto terug op de weg. Drie modi, ingesteld in config.json
        (de kalibratie vraagt ernaar):

        "gamepad" : drukt op RESET_GAMEPAD_BUTTON. Bind die knop in BeamNG
                    onder Options > Controls > Vehicle > "Recover Vehicle".
                    Volledig automatisch en blijft binnen de "alleen
                    controller"-regel. Aanbevolen voor lange sessies.
        "key"     : simuleert INSERT (BeamNG's standaard recover-toets).
                    Alleen voor resetten, nooit voor het rijden.
        "manual"  : jij zet de auto zelf klaar en drukt op R (in de terminal
                    of in het viewer-venster).
        """
        mode = (self.cfg.reset_mode or "manual").lower()

        if mode in ("gamepad", "key"):
            for attempt in range(1, settings.RESET_MAX_RETRIES + 2):
                if mode == "gamepad":
                    ok = self.pad.tap_button(settings.RESET_GAMEPAD_BUTTON,
                                             settings.RESET_BUTTON_HOLD_S)
                    what = f"gamepad-knop {settings.RESET_GAMEPAD_BUTTON}"
                else:
                    ok = send_key("insert", 0.08)
                    what = "toets INSERT"

                if not ok and attempt == 1:
                    print(f"[env] Automatische reset via {what} lukte niet -> handmatig.")
                    mode = "manual"
                    break

                self._wait_with_viewer(settings.RESET_SETTLE_S,
                                       "AUTO WORDT GERECOVERD",
                                       [f"Recover via {what} (poging {attempt})",
                                        "Even wachten tot de auto stilstaat..."])

                # Controleren of we weer op de weg staan.
                frame = self.capture.grab()
                frac = self.detector.road.update(frame)
                if not self.detector.road.enabled or frac >= settings.ROAD_MIN_FRACTION:
                    self.pad.neutral()
                    return
                if attempt > settings.RESET_MAX_RETRIES:
                    print(f"[env] Na {attempt} pogingen nog steeds geen weg onder de wielen "
                          f"(wegdek-fractie {frac:.2f}). Even zelf ingrijpen.")
                    mode = "manual"
                    break

        if mode == "manual":
            self._wait_for_manual_ready()

        self.pad.neutral()

    def _wait_for_manual_ready(self) -> None:
        print("\n" + "=" * 66)
        print("  RESET: zet de auto terug op de weg.")
        print("    - In BeamNG:  INSERT = recover vehicle,  R = reset naar spawn")
        print("    - Druk daarna op  R  (in dit terminalvenster of in de viewer)")
        print("    - q = training stoppen en model opslaan")
        print("=" * 66, flush=True)

        self.keys.flush()
        while True:
            key = self.viewer.show_message(
                "WACHTEN OP RESET",
                [
                    "Zet de auto terug op de weg.",
                    "INSERT = recover vehicle in BeamNG.",
                    "",
                    "Druk R zodra de auto klaarstaat.",
                    "Druk Q om de training te stoppen en op te slaan.",
                ],
                wait_ms=30,
            )
            term = self.keys.poll()
            if key == ord("r") or term == "r":
                break
            if key == ord("q") or term == "q":
                self.stop_requested = True
                break
            if not self.viewer.enabled and not self.keys.available:
                # Geen viewer én geen msvcrt (niet-Windows): val terug op input().
                input("  Druk ENTER zodra de auto klaarstaat... ")
                break
            time.sleep(0.01)

    def _wait_with_viewer(self, seconds: float, title: str, lines) -> None:
        """Wacht terwijl het viewer-venster responsief blijft."""
        end = time.perf_counter() + seconds
        while time.perf_counter() < end:
            remaining = end - time.perf_counter()
            self.viewer.show_message(title, list(lines) + [f"", f"nog {remaining:.1f} s"], wait_ms=30)
            if not self.viewer.enabled:
                time.sleep(0.05)

    # ======================================================================
    # STEP
    # ======================================================================
    def step(self, action):
        # ------------------------------------------------------------------
        # 1) Actie -> controller
        # ------------------------------------------------------------------
        steer, throttle, brake, _ = decode_action(action, self.prev_steer)
        self.pad.apply(steer, throttle, brake)

        # ------------------------------------------------------------------
        # 2) Vaste timestep aanhouden
        # ------------------------------------------------------------------
        now = time.perf_counter()
        if self._last_step_time is not None and not self._timing_gap:
            sleep = settings.STEP_DT - (now - self._last_step_time)
            if sleep > 0:
                time.sleep(sleep)
        self._timing_gap = False
        step_start = time.perf_counter()
        dt = settings.STEP_DT if self._last_step_time is None else step_start - self._last_step_time
        self._last_step_time = step_start

        # ------------------------------------------------------------------
        # 3) Waarnemen
        # ------------------------------------------------------------------
        frame = self.capture.grab()
        self._last_frame = frame
        speed = self.speed_reader.speed
        flow = self.motion.update(frame)
        diff = self.framediff.update(frame)

        # ------------------------------------------------------------------
        # 4) Crash / offroad / vast?
        # ------------------------------------------------------------------
        det = self.detector.update(frame, speed, diff, throttle, brake, now=step_start)

        # ------------------------------------------------------------------
        # 5) Reward
        # ------------------------------------------------------------------
        rb = compute_reward(
            speed=speed, flow=flow, steer=steer,
            prev_steer=self.prev_steer, terminal_reason=det.reason,
        )
        reward = rb.total

        self.prev_steer = steer
        self.step_idx += 1
        self.total_steps += 1
        self.episode_reward += reward
        self._last_reward = reward
        self._last_reward_parts = rb.as_dict()
        if speed is not None:
            self._speed_sum += speed
            self._speed_n += 1

        # ------------------------------------------------------------------
        # 6) Done-condities
        # ------------------------------------------------------------------
        terminated = bool(det.done)
        # Nooit allebei tegelijk: bij een crash bootstrapt PPO de waarde niet,
        # bij een tijdslimiet wél. Die twee mogen elkaar niet overlappen.
        truncated = (not terminated) and self.step_idx >= settings.MAX_EPISODE_STEPS

        if terminated:
            self._status = {
                "crash": "CRASH", "flip": "OMGESLAGEN",
                "offroad": "OFFROAD", "stuck": "VAST",
            }.get(det.reason, det.reason.upper())
            self.pad.coast_to_stop(0.6)
        elif truncated:
            self._status = "TIJD OP"
        else:
            self._status = ""

        # ------------------------------------------------------------------
        # 7) Observatie + viewer
        # ------------------------------------------------------------------
        obs = preprocess_observation(frame)
        self._last_obs = obs

        if self.viewer.enabled and (self.total_steps % max(settings.VIEWER_EVERY_N_STEPS, 1) == 0):
            self.viewer.push_reward(reward)
            key = self.viewer.render(self._viewer_state(speed, det))
            self._handle_key(key)
        self._handle_key(self._poll_terminal())

        if self.paused:
            self._pause_loop()

        # ------------------------------------------------------------------
        # 8) Info voor de logs
        # ------------------------------------------------------------------
        mean_speed = self._speed_sum / self._speed_n if self._speed_n else 0.0
        info: Dict[str, Any] = {
            "speed": -1.0 if speed is None else float(speed),
            "mean_speed": float(mean_speed),
            "flow": float(flow),
            "road_fraction": float(det.road_fraction),
            "step_dt": float(dt),
            **self._last_reward_parts,
        }
        if terminated or truncated:
            info["end_reason"] = det.reason if terminated else "timeout"
            info["episode_mean_speed"] = float(mean_speed)
            self.viewer.push_episode(self.episode_reward)
            if self.verbose:
                print(
                    f"[env] Episode {self.episode} klaar: {self.step_idx} stappen, "
                    f"reward {self.episode_reward:+.1f}, gem. snelheid {mean_speed:.1f} km/u, "
                    f"reden: {info['end_reason']}",
                    flush=True,
                )

        return obs, float(reward), terminated, truncated, info

    # ======================================================================
    # Viewer-state + toetsen
    # ======================================================================
    def _viewer_state(self, speed, det) -> Dict[str, Any]:
        return {
            "obs": self._last_obs,
            "ocr_image": self.speed_reader.debug_image,
            "ocr_stats": self.speed_reader.stats,
            "speed": speed,
            "reward": self._last_reward,
            "reward_parts": self._last_reward_parts,
            "steer": self.pad.steer,
            "throttle": self.pad.throttle,
            "brake": self.pad.brake,
            "detection": {
                "road_fraction": det.road_fraction,
                "frame_diff": det.frame_diff,
                "speed_drop": det.speed_drop,
                "offroad_streak": det.offroad_streak,
                "stuck_seconds": det.stuck_seconds,
            },
            "road_mask": self.detector.road.last_mask,
            "episode": self.episode,
            "step": self.step_idx,
            "total_steps": self.total_steps,
            "episode_reward": self.episode_reward,
            "mean_speed": self._speed_sum / self._speed_n if self._speed_n else 0.0,
            "status": self._status,
        }

    def _poll_terminal(self) -> int:
        ch = self.keys.poll()
        return ord(ch) if ch else -1

    def _handle_key(self, key: int) -> None:
        if key == ord("q"):
            print("[env] Stop gevraagd (q). Training wordt afgerond en opgeslagen...")
            self.stop_requested = True
        elif key == ord("p"):
            self.paused = not self.paused
            print(f"[env] {'Gepauzeerd' if self.paused else 'Hervat'}.")

    def _pause_loop(self) -> None:
        """Blijf hangen tot 'p' opnieuw gedrukt wordt; controls neutraal."""
        self.pad.neutral()
        while self.paused and not self.stop_requested:
            key = self.viewer.show_message(
                "GEPAUZEERD",
                ["De controller staat neutraal.",
                 "", "p = hervatten", "q = stoppen en opslaan"],
                wait_ms=50,
            )
            term = self._poll_terminal()
            for k in (key, term):
                if k == ord("p"):
                    self.paused = False
                elif k == ord("q"):
                    self.paused = False
                    self.stop_requested = True
            if not self.viewer.enabled:
                time.sleep(0.05)
        # Timing en tijdreeksen kloppen niet meer na een pauze.
        self.on_train_resume()

    # ======================================================================
    # Haakjes voor train.py (PPO-updates)
    # ======================================================================
    def on_train_pause(self) -> None:
        """Aangeroepen vlak voor een PPO-update: de auto krijgt geen nieuwe
        commando's meer, dus zetten we gas los en remmen we zacht."""
        self.pad.coast_to_stop(0.45)

    def on_train_resume(self) -> None:
        """Na de update: tijdreeksen wissen zodat het gat geen valse crash geeft."""
        self.detector.notify_gap()
        self.motion.reset()
        self.framediff.reset()
        self.speed_reader.reset_history()
        self._last_step_time = None
        self._timing_gap = True

    # ======================================================================
    # Opruimen
    # ======================================================================
    def render(self):
        if self._last_frame is not None:
            self.viewer.render(self._viewer_state(self.speed_reader.speed, self.detector.last))

    def close(self):
        try:
            self.pad.neutral()
        finally:
            self.speed_reader.stop()
            self.capture.close()
            self.pad.close()
            self.viewer.close()
            cv2.destroyAllWindows()
