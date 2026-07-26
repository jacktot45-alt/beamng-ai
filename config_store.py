"""
config_store.py
===============
Laden/opslaan van de kalibratie in `config.json`.

Alles wat per PC verschilt (schermresolutie, positie van het BeamNG-venster,
waar de snelheidsmeter staat, hoe het asfalt eruitziet) hoort hier thuis.
Gedragsparameters horen in settings.py.
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional

import settings

CONFIG_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "config.json")


@dataclass
class Region:
    """Een rechthoek in absolute schermcoördinaten (zoals mss ze wil)."""

    left: int
    top: int
    width: int
    height: int

    def as_mss(self) -> Dict[str, int]:
        return {"left": self.left, "top": self.top, "width": self.width, "height": self.height}

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "Region":
        return Region(int(d["left"]), int(d["top"]), int(d["width"]), int(d["height"]))


@dataclass
class OcrPrep:
    """Voorbewerkings-instellingen voor de HUD-crop, bepaald tijdens kalibratie."""

    scale: int = 4               # upscale-factor voor OCR (kleine cijfers -> groot)
    invert: bool = False         # True als de cijfers donker op een lichte achtergrond staan
    threshold: str = "otsu"      # "otsu" | "adaptive" | "fixed" | "none"
    thresh_value: int = 128      # enkel gebruikt bij "fixed"

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "OcrPrep":
        base = OcrPrep()
        return OcrPrep(
            scale=int(d.get("scale", base.scale)),
            invert=bool(d.get("invert", base.invert)),
            threshold=str(d.get("threshold", base.threshold)),
            thresh_value=int(d.get("thresh_value", base.thresh_value)),
        )


@dataclass
class Config:
    """Volledige kalibratie-state."""

    monitor_index: int = 1
    game_region: Optional[Region] = None      # het BeamNG-venster op je scherm
    speed_region: Optional[Region] = None     # crop rond de cijfers van de snelheidsmeter
    ocr_prep: OcrPrep = field(default_factory=OcrPrep)
    # Referentiekleur van het wegdek in HSV (OpenCV-schaal: H 0-179, S/V 0-255).
    road_hsv: Optional[List[int]] = None
    reset_mode: str = settings.DEFAULT_RESET_MODE
    version: int = 1

    # -- serialisatie ------------------------------------------------------
    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["game_region"] = asdict(self.game_region) if self.game_region else None
        d["speed_region"] = asdict(self.speed_region) if self.speed_region else None
        return d

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "Config":
        cfg = Config()
        cfg.monitor_index = int(d.get("monitor_index", 1))
        if d.get("game_region"):
            cfg.game_region = Region.from_dict(d["game_region"])
        if d.get("speed_region"):
            cfg.speed_region = Region.from_dict(d["speed_region"])
        cfg.ocr_prep = OcrPrep.from_dict(d.get("ocr_prep", {}))
        road = d.get("road_hsv")
        cfg.road_hsv = [int(v) for v in road] if road else None
        cfg.reset_mode = str(d.get("reset_mode", settings.DEFAULT_RESET_MODE))
        cfg.version = int(d.get("version", 1))
        return cfg

    # -- validatie ---------------------------------------------------------
    def is_complete(self) -> bool:
        """Minimaal nodig om te kunnen trainen: het game-venster."""
        return self.game_region is not None

    def missing(self) -> List[str]:
        out = []
        if self.game_region is None:
            out.append("game_region (het BeamNG-venster)")
        if self.speed_region is None and settings.OCR_BACKEND != "none":
            out.append("speed_region (de snelheidsmeter in de HUD)")
        if self.road_hsv is None:
            out.append("road_hsv (referentiekleur van het wegdek)")
        return out


def load(path: str = CONFIG_PATH) -> Optional[Config]:
    if not os.path.exists(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as fh:
            return Config.from_dict(json.load(fh))
    except Exception as exc:  # corrupte config mag de training niet laten crashen
        print(f"[config] Kon {path} niet lezen ({exc}). Kalibreer opnieuw.")
        return None


def save(cfg: Config, path: str = CONFIG_PATH) -> None:
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(cfg.to_dict(), fh, indent=2)
    print(f"[config] Kalibratie opgeslagen in {path}")
