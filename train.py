"""
train.py  -  BeamNG.drive RL vision agent
=========================================
Traint een PPO-agent die BeamNG.drive bestuurt op ENKEL beeldherkenning.
Geen BeamNGpy, geen telemetry, geen memory reading, geen simulatie-versnelling.
De agent kijkt naar je scherm en duwt op een virtuele Xbox 360-controller,
precies zoals een mens dat zou doen.

---------------------------------------------------------------------------
INSTALLATIE (Windows)
---------------------------------------------------------------------------
1) Python 3.10 of 3.11 (64-bit).

2) Packages:
       pip install -r requirements.txt
   of handmatig:
       pip install vgamepad mss opencv-python numpy gymnasium stable-baselines3 pytesseract
       pip install torch --index-url https://download.pytorch.org/whl/cu121   # met NVIDIA GPU
   LET OP: `opencv-python`, NIET `opencv-python-headless` (die heeft geen GUI
   en dan werkt de live viewer niet).

3) ViGEmBus-driver (nodig voor vgamepad):
   De installer van vgamepad biedt dit aan tijdens `pip install vgamepad`.
   Lukt dat niet, haal hem dan handmatig:
       https://github.com/nefarius/ViGEmBus/releases
   Herstart je PC na de installatie.

4) Tesseract-OCR (als je OCR_BACKEND = "pytesseract" gebruikt):
       https://github.com/UB-Mannheim/tesseract/wiki
   Zet het pad naar tesseract.exe in settings.TESSERACT_CMD.
   Alternatief: zet settings.OCR_BACKEND = "easyocr" (`pip install easyocr`),
   zwaarder maar geen aparte .exe nodig.

---------------------------------------------------------------------------
BEAMNG INSTELLEN
---------------------------------------------------------------------------
* Options > Graphics:  WINDOWED of BORDERLESS (NIET exclusive fullscreen,
  anders kan mss het beeld niet betrouwbaar grabben en flikkert alt-tab).
* Vaste resolutie; verander die niet meer na de kalibratie (anders kloppen
  je crops niet meer -> gewoon opnieuw kalibreren).
* HUD zichtbaar met de digitale snelheidsmeter in beeld.
* Options > Controls: de virtuele controller ("Xbox 360 Controller for
  Windows") moet als input toegelaten zijn. Test even of de auto reageert
  op de stick voor je begint te trainen.
* Kies een makkelijk circuit om mee te beginnen: West Coast USA of Italy
  hebben lange, brede stukken weg. Gridmap met een simpele rechte weg is
  nog makkelijker voor de eerste sessies.
* Kies een auto die niet te snel is (bv. de Ibishu Covet) - 50 km/u halen
  met een supercar is lastiger te doseren.
* Zet reset_mode op "gamepad" tijdens de kalibratie en bind knop Y in
  BeamNG aan "Recover Vehicle" als je onbewaakt wil trainen.

---------------------------------------------------------------------------
GEBRUIK
---------------------------------------------------------------------------
    python train.py --calibrate     # eenmalig: crops instellen
    python train.py --preview       # alles testen zonder te trainen/rijden
    python train.py                 # trainen
    python train.py --resume checkpoints/latest.zip
    python train.py --no-viewer     # zonder het dashboardvenster (iets sneller)

Tijdens de training (viewer-venster of terminal actief):
    q = stoppen en model opslaan     p = pauze     r = "auto staat klaar"
---------------------------------------------------------------------------
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
import time
from typing import Optional

import settings
from config_store import Config, load as load_config
from utils import countdown, enable_dpi_awareness


# ===========================================================================
# Logging-callback: reward + snelheid per episode naar CSV en TensorBoard
# ===========================================================================


def build_callbacks(env_ref):
    from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback

    class EpisodeLogger(BaseCallback):
        """
        Schrijft per afgeronde episode een regel naar logs/episodes.csv en
        stuurt dezelfde cijfers naar TensorBoard. Regelt ook het netjes
        stoppen bij 'q' en het neutraal zetten van de controller tijdens
        een PPO-update (anders rijdt de auto stuurloos door terwijl het
        netwerk aan het leren is).
        """

        def __init__(self, env, csv_path: str, verbose: int = 0):
            super().__init__(verbose)
            self.env = env
            self.csv_path = csv_path
            self.episode_index = 0
            self.start_time = time.time()
            os.makedirs(os.path.dirname(csv_path) or ".", exist_ok=True)
            if not os.path.exists(csv_path):
                with open(csv_path, "w", newline="", encoding="utf-8") as fh:
                    csv.writer(fh).writerow([
                        "episode", "timesteps", "wallclock_s", "episode_reward",
                        "episode_length", "mean_speed_kmh", "end_reason",
                    ])

        # -- per stap -----------------------------------------------------
        def _on_step(self) -> bool:
            for info in self.locals.get("infos", []):
                ep = info.get("episode")          # gezet door de Monitor-wrapper
                if ep is None:
                    continue
                self.episode_index += 1
                row = [
                    self.episode_index,
                    int(self.num_timesteps),
                    round(time.time() - self.start_time, 1),
                    round(float(ep["r"]), 3),
                    int(ep["l"]),
                    round(float(info.get("episode_mean_speed", 0.0)), 2),
                    info.get("end_reason", "?"),
                ]
                with open(self.csv_path, "a", newline="", encoding="utf-8") as fh:
                    csv.writer(fh).writerow(row)

                self.logger.record("episode/reward", float(ep["r"]))
                self.logger.record("episode/length", int(ep["l"]))
                self.logger.record("episode/mean_speed_kmh",
                                   float(info.get("episode_mean_speed", 0.0)))
                print(f"[log] ep {self.episode_index:4d} | {int(ep['l']):4d} stappen | "
                      f"reward {float(ep['r']):+8.1f} | "
                      f"gem. {float(info.get('episode_mean_speed', 0.0)):5.1f} km/u | "
                      f"{info.get('end_reason', '?')}", flush=True)

            # 'q' in de viewer/terminal -> learn() netjes afbreken.
            if getattr(self.env, "stop_requested", False):
                print("[train] Stop gevraagd -> rollout afronden en opslaan.")
                return False
            return True

        # -- rond een PPO-update ------------------------------------------
        def _on_rollout_end(self) -> None:
            # De env wordt nu enkele seconden niet gestept: gas los, zacht remmen.
            self.env.on_train_pause()

        def _on_rollout_start(self) -> None:
            # Tijdreeksen wissen zodat het gat niet als crash gelezen wordt.
            self.env.on_train_resume()

    checkpoint = CheckpointCallback(
        save_freq=settings.CHECKPOINT_EVERY,
        save_path=settings.CHECKPOINT_DIR,
        name_prefix="beamng_ppo",
        verbose=1,
    )
    return [EpisodeLogger(env_ref, settings.EPISODE_CSV), checkpoint]


# ===========================================================================
# Env opbouwen
# ===========================================================================


def make_env(cfg: Config, use_viewer: bool, use_gamepad: bool = True):
    """Maakt de env + Monitor + DummyVecEnv + VecFrameStack(4)."""
    from stable_baselines3.common.monitor import Monitor
    from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack

    from beamng_env import BeamNGVisionEnv

    raw_env = BeamNGVisionEnv(cfg, use_viewer=use_viewer, use_gamepad=use_gamepad)
    os.makedirs(settings.LOG_DIR, exist_ok=True)
    monitored = Monitor(raw_env, filename=os.path.join(settings.LOG_DIR, "monitor.csv"))
    venv = DummyVecEnv([lambda: monitored])
    # Framestacking: 4 grijze frames op elkaar -> het CNN kan beweging zien.
    venv = VecFrameStack(venv, n_stack=settings.FRAME_STACK)
    return raw_env, venv


# ===========================================================================
# Preview-modus: alles testen zonder RL
# ===========================================================================


def run_preview(cfg: Config) -> None:
    """
    Draait de volledige waarnemingsketen (capture, OCR, detectie, viewer)
    met een NUL-actie en een dummy-controller. De auto beweegt dus niet.
    Ideaal om je crops, OCR en wegkleur te controleren voor je gaat trainen.
    """
    import numpy as np

    from beamng_env import BeamNGVisionEnv

    print("\n[preview] Capture + OCR + detectie testen. De auto wordt NIET bestuurd.")
    print("[preview] Rijd gerust zelf rond met je eigen controller/toetsenbord")
    print("[preview] en kijk of snelheid, wegdek-fractie en crash-detectie kloppen.")
    print("[preview] q = stoppen\n")

    env = BeamNGVisionEnv(cfg, use_viewer=True, use_gamepad=False, verbose=True)
    zero = np.zeros(2, dtype=np.float32)
    try:
        env.reset()
        while not env.stop_requested:
            _, _, terminated, truncated, info = env.step(zero)
            if terminated or truncated:
                print(f"[preview] Detectie zou de episode nu beëindigen: "
                      f"{info.get('end_reason')} -> episode herstarten.")
                env.reset()
    except KeyboardInterrupt:
        print("\n[preview] Onderbroken.")
    finally:
        env.close()


# ===========================================================================
# Diagnose
# ===========================================================================


def run_check() -> int:
    print("\n--- Omgevingscheck ---")
    problems = 0

    def probe(label, fn):
        nonlocal problems
        try:
            detail = fn()
            print(f"  [ok]   {label}{(' - ' + detail) if detail else ''}")
        except Exception as exc:
            problems += 1
            print(f"  [FOUT] {label}: {exc}")

    probe("platform is Windows", lambda: sys.platform if sys.platform == "win32"
          else (_ for _ in ()).throw(RuntimeError(f"{sys.platform} - vgamepad werkt enkel op Windows")))
    probe("mss", lambda: __import__("mss").__version__ if hasattr(__import__("mss"), "__version__") else "")
    probe("opencv (met GUI)", lambda: __import__("cv2").__version__)
    probe("numpy", lambda: __import__("numpy").__version__)
    probe("gymnasium", lambda: __import__("gymnasium").__version__)
    probe("stable-baselines3", lambda: __import__("stable_baselines3").__version__)

    def _torch():
        import torch
        return f"{torch.__version__}, cuda={'ja' if torch.cuda.is_available() else 'nee (traag!)'}"
    probe("torch", _torch)

    def _vgamepad():
        from controller import VirtualGamepad
        pad = VirtualGamepad()
        pad.close()
        return "virtuele controller aangemaakt"
    probe("vgamepad + ViGEmBus", _vgamepad)

    def _ocr():
        from speed_ocr import build_engine
        return build_engine(settings.OCR_BACKEND).name
    probe(f"OCR-backend ({settings.OCR_BACKEND})", _ocr)

    cfg = load_config()
    if cfg is None:
        problems += 1
        print("  [FOUT] config.json ontbreekt - run: python train.py --calibrate")
    else:
        missing = cfg.missing()
        if missing:
            print(f"  [let op] kalibratie onvolledig: {', '.join(missing)}")
        else:
            print("  [ok]   config.json volledig")

    print(f"\n{'Alles in orde.' if problems == 0 else f'{problems} probleem(en) gevonden.'}\n")
    return problems


# ===========================================================================
# Trainen
# ===========================================================================


def run_training(cfg: Config, args) -> None:
    from stable_baselines3 import PPO

    os.makedirs(settings.CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(settings.TENSORBOARD_DIR, exist_ok=True)

    raw_env, venv = make_env(cfg, use_viewer=not args.no_viewer)

    # --- model laden of aanmaken -----------------------------------------
    if args.resume:
        if not os.path.exists(args.resume):
            print(f"[train] Checkpoint niet gevonden: {args.resume}")
            raw_env.close()
            return
        print(f"[train] Hervatten vanaf {args.resume}")
        model = PPO.load(args.resume, env=venv, tensorboard_log=settings.TENSORBOARD_DIR)
    else:
        print("[train] Nieuw PPO-model (CnnPolicy).")
        model = PPO(
            "CnnPolicy",
            venv,
            verbose=1,
            tensorboard_log=settings.TENSORBOARD_DIR,
            **settings.PPO_KWARGS,
        )

    print("\n" + "=" * 70)
    print("  KLAAR OM TE TRAINEN")
    print("  * Zet het BeamNG-venster zichtbaar (niet geminimaliseerd!).")
    print("  * Sleep het viewer-venster ernaast zodat je alles kan volgen.")
    print("  * q = stoppen en opslaan   |   p = pauze   |   r = auto klaar")
    print("=" * 70)
    countdown(settings.START_COUNTDOWN_S, "Training start over")

    callbacks = build_callbacks(raw_env)
    latest = os.path.join(settings.CHECKPOINT_DIR, "latest.zip")

    try:
        model.learn(
            total_timesteps=args.timesteps,
            callback=callbacks,
            reset_num_timesteps=not args.resume,
            tb_log_name="ppo_beamng",
            progress_bar=False,
        )
    except KeyboardInterrupt:
        print("\n[train] Ctrl+C -> model wordt nog opgeslagen...")
    finally:
        try:
            model.save(latest)
            print(f"[train] Model opgeslagen in {latest}")
            print(f"[train] Hervatten met: python train.py --resume {latest}")
        except Exception as exc:
            print(f"[train] Opslaan mislukt: {exc}")
        raw_env.close()
        print(f"[train] Episode-log: {settings.EPISODE_CSV}")
        print(f"[train] TensorBoard: tensorboard --logdir {settings.TENSORBOARD_DIR}")


# ===========================================================================
# CLI
# ===========================================================================


def main() -> None:
    parser = argparse.ArgumentParser(
        description="RL-agent die BeamNG.drive bestuurt op basis van beeldherkenning.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--calibrate", action="store_true",
                        help="(her)kalibreer schermregio's, OCR en wegkleur")
    parser.add_argument("--preview", action="store_true",
                        help="test capture/OCR/detectie zonder te trainen of te rijden")
    parser.add_argument("--check", action="store_true",
                        help="controleer packages, drivers en kalibratie")
    parser.add_argument("--resume", type=str, default=None,
                        help="hervat vanaf een checkpoint (.zip)")
    parser.add_argument("--timesteps", type=int, default=settings.TOTAL_TIMESTEPS,
                        help=f"aantal trainingsstappen (standaard {settings.TOTAL_TIMESTEPS})")
    parser.add_argument("--no-viewer", action="store_true",
                        help="draai zonder het live viewer-venster")
    args = parser.parse_args()

    # MOET voor de eerste screen capture, anders kloppen de coördinaten niet
    # bij Windows-schaling != 100 %.
    enable_dpi_awareness()

    if args.check:
        sys.exit(1 if run_check() else 0)

    from calibrate import run_calibration

    cfg: Optional[Config] = load_config()
    if args.calibrate or cfg is None or not cfg.is_complete():
        if cfg is None:
            print("[setup] Nog geen config.json gevonden -> kalibratie starten.")
        elif not args.calibrate:
            print(f"[setup] Kalibratie onvolledig ({', '.join(cfg.missing())}) -> opnieuw.")
        cfg = run_calibration(cfg)

    missing = cfg.missing()
    if missing:
        print(f"[setup] Let op, nog niet gekalibreerd: {', '.join(missing)}")
        print("[setup] Die onderdelen worden overgeslagen. "
              "Run `python train.py --calibrate` om ze alsnog in te stellen.")

    if args.preview:
        run_preview(cfg)
        return

    run_training(cfg, args)


if __name__ == "__main__":
    main()
