"""
settings.py
===========
ALLE tuning-knoppen op één plek. Je kan hier bijna alles aanpassen zonder
de rest van de architectuur te moeten begrijpen.

Wat NIET hier staat: de kalibratie (schermregio's, HUD-crop, wegkleur).
Die wordt interactief bepaald door `python train.py --calibrate` en
opgeslagen in `config.json`.

Vuistregel:
  * settings.py  -> "hoe gedraagt de agent zich" (reward, timing, PPO)
  * config.json  -> "waar staat wat op mijn scherm" (per PC verschillend)
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# 1. TIMING / CONTROL LOOP
# ---------------------------------------------------------------------------

# Vaste timestep van de agent. 0.10 s = 10 beslissingen per seconde.
# Lager (0.05) = reactiever maar zwaarder voor je CPU en meer stappen per
# seconde nodig om iets te leren. 0.08 - 0.12 is een prima werkgebied.
STEP_DT = 0.10

# Hoeveel stappen mag één episode maximaal duren voor hij afgekapt wordt
# (truncated). 3000 * 0.10 s = 5 minuten rijden.
MAX_EPISODE_STEPS = 3000

# Aantal stappen aan het begin van een episode waarin crash/offroad-detectie
# uitgeschakeld staat. Geeft de auto tijd om te "settelen" na een reset.
WARMUP_STEPS = 15


# ---------------------------------------------------------------------------
# 2. OBSERVATIE (wat het CNN ziet)
# ---------------------------------------------------------------------------

# Resolutie van het beeld dat naar het netwerk gaat. 84x84 is de klassieke
# Atari-maat en snel; 96x96 geeft iets meer detail maar is trager.
OBS_SIZE = (84, 84)          # (breedte, hoogte)

# Aantal frames dat gestapeld wordt (via SB3 VecFrameStack). Met 4 frames kan
# het netwerk snelheid/beweging afleiden uit puur beeld.
FRAME_STACK = 4

# Crop van het game-venster VOORDAT er gedownscaled wordt, als fracties.
# Doel: lucht en HUD wegknippen zodat er zoveel mogelijk "weg" in beeld zit.
#   (top, bottom, left, right) -> 0.0 = links/boven, 1.0 = rechts/onder
# Standaard: bovenste 32 % weg (lucht/spiegels), onderste 6 % weg (HUD).
# Tip: run `python train.py --preview` en kijk in de viewer of de weg mooi
# gevuld in beeld staat.
OBS_CROP = (0.32, 0.94, 0.02, 0.98)


# ---------------------------------------------------------------------------
# 3. ACTIES / CONTROLLER
# ---------------------------------------------------------------------------

# De action space is continu: Box(-1, 1, shape=(2,))
#   action[0] = sturen        -1.0 = vol links, +1.0 = vol rechts
#   action[1] = gas / rem     >0 = rechter trigger (gas), <0 = linker trigger (rem)

# Maximale stuuruitslag. 1.0 = volledige stick. Beperken tot ~0.6 maakt het
# leren rustiger want de auto slingert minder.
STEER_LIMIT = 0.65

# Exponentiële smoothing op de stuurhoek: 0.0 = geen smoothing (schokkerig),
# 0.9 = heel traag. 0.55 voelt als een menselijke pols.
STEER_SMOOTHING = 0.55

# Deadzone: kleine stuurcommando's worden 0. Voorkomt constant micro-trillen.
STEER_DEADZONE = 0.05

# Gas/rem mapping. Onder deze drempel doet de agent niets (coasting).
THROTTLE_DEADZONE = 0.05
# Schaal op de triggers, handig om de auto in het begin wat rustiger te maken.
THROTTLE_SCALE = 1.0
BRAKE_SCALE = 0.8


# ---------------------------------------------------------------------------
# 4. REWARD FUNCTION
# ---------------------------------------------------------------------------
# De volledige berekening staat in reward.py; hier staan enkel de getallen.

# --- snelheidsdoel ---
TARGET_SPEED = 50.0          # km/u waar de reward maximaal is
SPEED_PEAK_BAND = 5.0        # +/- deze band krijgt de volle reward (1.0)
SPEED_SIGMA_SLOW = 12.0      # hoe hard te traag rijden afgestraft wordt
SPEED_SIGMA_FAST = 9.0       # te hard rijden wordt iets strenger afgestraft
SPEED_REWARD_WEIGHT = 1.0    # gewicht van de snelheidsterm in de totale reward

# --- stilstand ---
IDLE_SPEED = 2.0             # onder deze km/u telt de auto als "stilstaand"
IDLE_PENALTY = -0.45         # straf per stap bij stilstand

# --- vooruitgangsbonus (optical-flow proxy) ---
# Kleine bonus voor beweging in beeld. Zonder dit kan de agent leren dat
# stilstaan "veilig" is (geen crash) en dus een lokaal optimum wordt.
PROGRESS_WEIGHT = 0.15
PROGRESS_FLOW_REF = 3.0      # flow-magnitude die als "volle bonus" telt

# --- stuurvlotheid ---
# Kleine straf op abrupte stuurwissels -> vloeiender rijgedrag.
STEER_JERK_WEIGHT = 0.06

# --- terminale events ---
CRASH_PENALTY = -12.0        # botsing / omgeslagen
OFFROAD_PENALTY = -8.0       # van de weg af
STUCK_PENALTY = -6.0         # vast tegen iets aan, gas open maar geen snelheid

# Wat te doen als OCR de snelheid even niet kan lezen: de snelheidsterm wordt
# dan 0 (neutraal) en enkel de progress-bonus telt nog mee.
# Zie ook SPEED_HOLD_S hieronder.


# ---------------------------------------------------------------------------
# 5. SNELHEIDSDETECTIE (OCR)
# ---------------------------------------------------------------------------

# "pytesseract" (licht, CPU, vereist Tesseract-installatie)
# "easyocr"     (zwaarder, GPU-vriendelijk, geen extra .exe nodig)
# "none"        (geen OCR; reward valt terug op enkel de progress-bonus)
OCR_BACKEND = "pytesseract"

# Pad naar tesseract.exe. Laat op None als het in je PATH staat.
TESSERACT_CMD = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

# OCR draait in een aparte thread op een eigen tempo, zodat de traagheid van
# OCR de control loop van 10 Hz niet blokkeert.
OCR_HZ = 8.0

# Hoe lang een oude meting nog geldig blijft als OCR even faalt (seconden).
# Daarna wordt de snelheid als "onbekend" gemarkeerd.
SPEED_HOLD_S = 0.6

# Plausibiliteitsfilter: metingen die sneller veranderen dan dit (km/u per
# seconde) worden verworpen als OCR-ruis. Een auto haalt geen 200 km/u/s.
SPEED_MAX_DELTA_PER_S = 120.0
SPEED_MAX_VALUE = 400.0      # alles hierboven is zeker een leesfout


# ---------------------------------------------------------------------------
# 6. CRASH- / OFFROAD-DETECTIE
# ---------------------------------------------------------------------------
# De volledige uitleg van de aanpak staat bovenaan detectors.py.

# --- (a) plotse snelheidsval = botsing ---
CRASH_SPEED_DROP = 18.0      # km/u verlies ...
CRASH_DROP_WINDOW_S = 0.35   # ... binnen dit tijdsvenster
CRASH_MIN_SPEED = 15.0       # enkel checken als je eerst sneller reed dan dit

# --- (b) beeldschok = impact ---
# Gemiddeld absoluut frameverschil (0-255) dat als "klap" telt.
IMPACT_DIFF_THRESHOLD = 46.0
IMPACT_CONSECUTIVE = 2       # aantal opeenvolgende frames boven de drempel

# --- (c) omgeslagen: lucht onderaan het beeld ---
FLIP_BRIGHTNESS_MARGIN = 38.0   # onderste strook zoveel helderder dan bovenste
FLIP_CONSECUTIVE = 6            # gedurende zoveel frames

# --- (d) van de weg af: wegkleur-fractie in een ROI onderaan het beeld ---
# ROI als fracties van het game-venster (x0, y0, x1, y1).
ROAD_ROI = (0.34, 0.70, 0.66, 0.90)
ROAD_MIN_FRACTION = 0.35        # minder dan 35 % "wegkleur" = verdacht
OFFROAD_CONSECUTIVE = 12        # ~1.2 s bij 10 Hz voordat we het aftellen
# Toleranties rond de gekalibreerde wegkleur (HSV, OpenCV-schaal H:0-179).
ROAD_TOL_H = 25
ROAD_TOL_S = 60
ROAD_TOL_V = 60
# Als de gekalibreerde saturatie hieronder ligt behandelen we asfalt als
# "grijs" en negeren we de hue (die is dan sowieso ruis).
ROAD_GREY_S_MAX = 70

# --- (e) vastzitten ---
STUCK_SPEED = 1.5            # km/u
STUCK_THROTTLE = 0.4         # met minstens zoveel gas
STUCK_SECONDS = 3.0          # gedurende zoveel seconden


# ---------------------------------------------------------------------------
# 7. RESET-GEDRAG
# ---------------------------------------------------------------------------

# Wordt overschreven door config.json (de kalibratie vraagt ernaar).
#   "gamepad" -> drukt op een controllerknop die jij in BeamNG bindt aan
#                "Recover Vehicle". Volledig automatisch: aanbevolen.
#   "manual"  -> vraagt je in de terminal / viewer om R te drukken zodra de
#                auto klaarstaat.
#   "key"     -> simuleert een toetsaanslag (standaard INSERT = recover).
#                Alleen voor het resetten, nooit voor het rijden zelf.
DEFAULT_RESET_MODE = "manual"

# Welke gamepad-knop bij reset_mode "gamepad". Zie controller.py BUTTONS.
RESET_GAMEPAD_BUTTON = "Y"
RESET_BUTTON_HOLD_S = 0.20

# Wachttijd na een recover voor de auto stilstaat en het beeld gesetteld is.
RESET_SETTLE_S = 3.0
# Hoeveel keer we automatisch opnieuw proberen te recoveren voor we de
# gebruiker om hulp vragen.
RESET_MAX_RETRIES = 2


# ---------------------------------------------------------------------------
# 8. LIVE VIEWER
# ---------------------------------------------------------------------------

VIEWER_ENABLED = True
VIEWER_WINDOW = "BeamNG RL - live viewer"
VIEWER_SIZE = (960, 660)     # (breedte, hoogte) van het viewer-venster
VIEWER_EVERY_N_STEPS = 1     # op 2 zetten = halve tekensnelheid, minder CPU
VIEWER_REWARD_HISTORY = 300  # aantal stappen in het reward-grafiekje
VIEWER_EPISODE_HISTORY = 40  # aantal episodes in het balkjes-grafiekje


# ---------------------------------------------------------------------------
# 9. PPO / TRAINING
# ---------------------------------------------------------------------------

TOTAL_TIMESTEPS = 300_000

PPO_KWARGS = dict(
    learning_rate=2.5e-4,
    # n_steps = hoeveel stappen verzameld worden voor er geleerd wordt.
    # LET OP: tijdens het leren rijdt de auto even door zonder aansturing.
    # Daarom zetten we de controls automatisch neutraal (zie train.py).
    # 512 * 0.1 s = ~51 s rijden per update.
    n_steps=512,
    batch_size=128,
    n_epochs=4,              # laag houden -> kortere pauze tijdens de update
    gamma=0.99,
    gae_lambda=0.95,
    clip_range=0.2,
    ent_coef=0.005,          # beetje exploratie
    vf_coef=0.5,
    max_grad_norm=0.5,
)

# Elke hoeveel stappen een checkpoint wegschrijven.
CHECKPOINT_EVERY = 5_000

CHECKPOINT_DIR = "checkpoints"
LOG_DIR = "logs"
TENSORBOARD_DIR = "logs/tb"
EPISODE_CSV = "logs/episodes.csv"

# Aantal seconden aftellen voor de training start, zodat je tijd hebt om het
# BeamNG-venster te focussen.
START_COUNTDOWN_S = 5
