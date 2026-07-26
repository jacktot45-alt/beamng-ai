"""
utils.py
========
Kleine Windows-/terminal-hulpjes die verder nergens thuishoren.
"""

from __future__ import annotations

import ctypes
import sys
import time
from typing import Optional

# ---------------------------------------------------------------------------
# DPI awareness
# ---------------------------------------------------------------------------
# Zonder dit liegt Windows over je schermcoördinaten zodra je Windows-schaling
# op iets anders dan 100 % staat: mss grabt dan de verkeerde regio en al je
# crops staan scheef. Moet aangeroepen worden VOOR de eerste screen capture.


def enable_dpi_awareness() -> None:
    if sys.platform != "win32":
        return
    try:
        # 2 = PROCESS_PER_MONITOR_DPI_AWARE
        ctypes.windll.shcore.SetProcessDpiAwareness(2)
    except Exception:
        try:
            ctypes.windll.user32.SetProcessDPIAware()
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Niet-blokkerende toetsaanslagen in de terminal
# ---------------------------------------------------------------------------
# Gebruikt voor "druk R zodra de auto klaarstaat" zonder dat de viewer bevriest.


class TerminalKeys:
    """Poll losse toetsen zonder te blokkeren (Windows: msvcrt)."""

    def __init__(self) -> None:
        self._msvcrt = None
        if sys.platform == "win32":
            try:
                import msvcrt  # type: ignore

                self._msvcrt = msvcrt
            except Exception:
                self._msvcrt = None

    @property
    def available(self) -> bool:
        return self._msvcrt is not None

    def poll(self) -> Optional[str]:
        """Geeft de ingedrukte toets terug als kleine letter, of None."""
        if self._msvcrt is None:
            return None
        if not self._msvcrt.kbhit():
            return None
        try:
            ch = self._msvcrt.getch()
        except Exception:
            return None
        if ch in (b"\x00", b"\xe0"):  # speciale toets: tweede byte opeten
            try:
                self._msvcrt.getch()
            except Exception:
                pass
            return None
        try:
            return ch.decode("utf-8", errors="ignore").lower()
        except Exception:
            return None

    def flush(self) -> None:
        while self.poll() is not None:
            pass


# ---------------------------------------------------------------------------
# Toetsaanslag simuleren (ENKEL voor het resetten van het voertuig)
# ---------------------------------------------------------------------------
# De agent zelf stuurt uitsluitend via de virtuele controller. Deze functie is
# een optioneel gemak voor reset_mode == "key" (bv. INSERT = Recover Vehicle in
# BeamNG). SendInput met scancodes werkt betrouwbaarder in games dan de
# virtual-key variant, want DirectInput leest scancodes.

_SCANCODES = {
    "insert": 0xD2,
    "home": 0xC7,
    "r": 0x13,
    "i": 0x17,
    "delete": 0xD3,
    "end": 0xCF,
}
_EXTENDED = {"insert", "home", "delete", "end"}


def send_key(name: str, hold: float = 0.08) -> bool:
    """Druk kort een toets in via SendInput. Geeft False op niet-Windows."""
    if sys.platform != "win32":
        return False
    name = name.lower()
    if name not in _SCANCODES:
        return False

    KEYEVENTF_SCANCODE = 0x0008
    KEYEVENTF_KEYUP = 0x0002
    KEYEVENTF_EXTENDEDKEY = 0x0001

    class KEYBDINPUT(ctypes.Structure):
        _fields_ = [
            ("wVk", ctypes.c_ushort),
            ("wScan", ctypes.c_ushort),
            ("dwFlags", ctypes.c_ulong),
            ("time", ctypes.c_ulong),
            ("dwExtraInfo", ctypes.POINTER(ctypes.c_ulong)),
        ]

    class _INPUTunion(ctypes.Union):
        _fields_ = [("ki", KEYBDINPUT), ("padding", ctypes.c_ubyte * 24)]

    class INPUT(ctypes.Structure):
        _fields_ = [("type", ctypes.c_ulong), ("union", _INPUTunion)]

    scan = _SCANCODES[name]
    flags = KEYEVENTF_SCANCODE | (KEYEVENTF_EXTENDEDKEY if name in _EXTENDED else 0)

    def _send(extra_flags: int) -> None:
        inp = INPUT(type=1)
        inp.union.ki = KEYBDINPUT(0, scan, flags | extra_flags, 0, None)
        ctypes.windll.user32.SendInput(1, ctypes.byref(inp), ctypes.sizeof(INPUT))

    _send(0)
    time.sleep(hold)
    _send(KEYEVENTF_KEYUP)
    return True


# ---------------------------------------------------------------------------
# Diversen
# ---------------------------------------------------------------------------


def clamp(value: float, low: float, high: float) -> float:
    return low if value < low else (high if value > high else value)


def countdown(seconds: int, message: str = "Start over") -> None:
    for i in range(seconds, 0, -1):
        print(f"  {message} {i}...", flush=True)
        time.sleep(1.0)
