"""
controller.py
=============
Virtuele Xbox 360-controller via vgamepad (ViGEmBus).

Mapping (zoals gevraagd):
    linker analoge stick X  -> sturen
    rechter trigger (RT)    -> gas
    linker trigger (LT)     -> remmen

BeamNG ziet dit als een echte gamepad; er wordt geen enkele toets
gesimuleerd voor het rijden zelf.

Vereist: ViGEmBus-driver geïnstalleerd (zie README).
"""

from __future__ import annotations

import time
from typing import Optional

import settings
from utils import clamp

try:
    import vgamepad as vg
except ImportError:  # pragma: no cover - pas bij gebruik hard falen
    vg = None


# Knopnamen die je in config/settings kan gebruiken (bv. voor "Recover Vehicle").
def _button_map():
    if vg is None:
        return {}
    return {
        "A": vg.XUSB_BUTTON.XUSB_GAMEPAD_A,
        "B": vg.XUSB_BUTTON.XUSB_GAMEPAD_B,
        "X": vg.XUSB_BUTTON.XUSB_GAMEPAD_X,
        "Y": vg.XUSB_BUTTON.XUSB_GAMEPAD_Y,
        "LB": vg.XUSB_BUTTON.XUSB_GAMEPAD_LEFT_SHOULDER,
        "RB": vg.XUSB_BUTTON.XUSB_GAMEPAD_RIGHT_SHOULDER,
        "BACK": vg.XUSB_BUTTON.XUSB_GAMEPAD_BACK,
        "START": vg.XUSB_BUTTON.XUSB_GAMEPAD_START,
        "DPAD_UP": vg.XUSB_BUTTON.XUSB_GAMEPAD_DPAD_UP,
        "DPAD_DOWN": vg.XUSB_BUTTON.XUSB_GAMEPAD_DPAD_DOWN,
        "DPAD_LEFT": vg.XUSB_BUTTON.XUSB_GAMEPAD_DPAD_LEFT,
        "DPAD_RIGHT": vg.XUSB_BUTTON.XUSB_GAMEPAD_DPAD_RIGHT,
    }


class VirtualGamepad:
    """Dunne wrapper rond vgamepad.VX360Gamepad."""

    def __init__(self) -> None:
        if vg is None:
            raise ImportError(
                "vgamepad ontbreekt.\n"
                "  1) pip install vgamepad\n"
                "  2) Installeer de ViGEmBus-driver (de vgamepad-installer biedt dit aan,\n"
                "     of haal hem van https://github.com/nefarius/ViGEmBus/releases)\n"
                "  3) Herstart je PC als de driver net geïnstalleerd is."
            )
        try:
            self.pad = vg.VX360Gamepad()
        except Exception as exc:
            raise RuntimeError(
                "Kon geen virtuele controller aanmaken. Staat de ViGEmBus-driver "
                f"geïnstalleerd en draait de service?\nOnderliggende fout: {exc}"
            ) from exc

        self.buttons = _button_map()
        # Laatst verstuurde waarden (handig voor de viewer).
        self.steer = 0.0
        self.throttle = 0.0
        self.brake = 0.0
        self.neutral()

    # -- rijden ------------------------------------------------------------
    def apply(self, steer: float, throttle: float, brake: float) -> None:
        """
        steer:    -1.0 (links) .. +1.0 (rechts)
        throttle:  0.0 .. 1.0  -> rechter trigger
        brake:     0.0 .. 1.0  -> linker trigger
        """
        steer = clamp(float(steer), -1.0, 1.0)
        throttle = clamp(float(throttle), 0.0, 1.0)
        brake = clamp(float(brake), 0.0, 1.0)

        self.pad.left_joystick_float(x_value_float=steer, y_value_float=0.0)
        self.pad.right_trigger_float(value_float=throttle)
        self.pad.left_trigger_float(value_float=brake)
        self.pad.update()

        self.steer, self.throttle, self.brake = steer, throttle, brake

    def neutral(self) -> None:
        """Alles los: geen stuur, geen gas, geen rem."""
        self.apply(0.0, 0.0, 0.0)

    def coast_to_stop(self, brake: float = 0.5) -> None:
        """Recht vooruit met wat rem; gebruikt tijdens PPO-updates en resets."""
        self.apply(0.0, 0.0, brake)

    # -- knoppen -----------------------------------------------------------
    def tap_button(self, name: str, hold: float = 0.15) -> bool:
        """Druk een knop kort in (bv. de knop die jij aan Recover Vehicle bond)."""
        btn = self.buttons.get(name.upper())
        if btn is None:
            print(f"[controller] Onbekende knop '{name}'. Keuzes: {sorted(self.buttons)}")
            return False
        self.pad.press_button(button=btn)
        self.pad.update()
        time.sleep(hold)
        self.pad.release_button(button=btn)
        self.pad.update()
        return True

    # -- opruimen ----------------------------------------------------------
    def close(self) -> None:
        try:
            self.pad.reset()
            self.pad.update()
        except Exception:
            pass


class DummyGamepad:
    """
    No-op controller voor `--preview` of debuggen zonder ViGEmBus.
    Handig om je crops en OCR te testen zonder dat de auto beweegt.
    """

    def __init__(self) -> None:
        self.steer = self.throttle = self.brake = 0.0
        self.buttons = {}

    def apply(self, steer: float, throttle: float, brake: float) -> None:
        self.steer = clamp(float(steer), -1.0, 1.0)
        self.throttle = clamp(float(throttle), 0.0, 1.0)
        self.brake = clamp(float(brake), 0.0, 1.0)

    def neutral(self) -> None:
        self.apply(0.0, 0.0, 0.0)

    def coast_to_stop(self, brake: float = 0.5) -> None:
        self.apply(0.0, 0.0, brake)

    def tap_button(self, name: str, hold: float = 0.15) -> bool:
        return False

    def close(self) -> None:
        pass


def decode_action(action, prev_steer: float) -> tuple:
    """
    Zet de rauwe PPO-actie om naar (steer, throttle, brake, smoothed_steer).

    action[0] = sturen (-1..1), action[1] = gas/rem (>0 gas, <0 rem).
    Hier zitten de deadzones, de stuurlimiet en de smoothing.
    Wil je een discrete action space of aparte gas/rem-assen? Pas dit
    aan plus de action_space in beamng_env.py.
    """
    raw_steer = clamp(float(action[0]), -1.0, 1.0)
    raw_pedal = clamp(float(action[1]), -1.0, 1.0)

    if abs(raw_steer) < settings.STEER_DEADZONE:
        raw_steer = 0.0
    target_steer = raw_steer * settings.STEER_LIMIT

    # Exponentiële smoothing: de stick beweegt niet abrupt van links naar rechts.
    a = clamp(settings.STEER_SMOOTHING, 0.0, 0.95)
    steer = a * prev_steer + (1.0 - a) * target_steer

    throttle = 0.0
    brake = 0.0
    if raw_pedal > settings.THROTTLE_DEADZONE:
        throttle = (raw_pedal - settings.THROTTLE_DEADZONE) / (1.0 - settings.THROTTLE_DEADZONE)
        throttle *= settings.THROTTLE_SCALE
    elif raw_pedal < -settings.THROTTLE_DEADZONE:
        brake = (-raw_pedal - settings.THROTTLE_DEADZONE) / (1.0 - settings.THROTTLE_DEADZONE)
        brake *= settings.BRAKE_SCALE

    return steer, clamp(throttle, 0.0, 1.0), clamp(brake, 0.0, 1.0), steer
