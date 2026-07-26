# BeamNG.drive RL vision agent

Een reinforcement learning agent die leert rijden in **BeamNG.drive** puur op
basis van **beeldherkenning**. Geen BeamNGpy, geen telemetry, geen memory
reading, geen versnelde simulatie. De agent kijkt naar je scherm en duwt op een
**virtuele Xbox 360-controller** — precies zoals een mens dat zou doen, in
realtime.

**Doel van de agent:** ongeveer **50 km/u** rijden zonder te crashen of van de
weg af te gaan.

> ### 👉 Nog nooit met Python gewerkt?
> Begin dan niet hier, maar bij **[TUTORIAL.md](TUTORIAL.md)** — een
> klik-voor-klik handleiding die je in ~45 minuten van niks naar een
> trainende AI brengt. Dit bestand is de technische naslag.

---

## Bestanden

| Bestand | Wat het doet |
|---|---|
| `train.py` | Startpunt. CLI, PPO-training, checkpointing, logging. |
| `settings.py` | **Alle tuning-knoppen.** Reward, timing, detectiedrempels, PPO-hyperparameters. |
| `beamng_env.py` | De `gymnasium.Env`: `reset()`, `step()`, done-condities. |
| `reward.py` | De reward function, apart zodat je er makkelijk aan kan sleutelen. |
| `detectors.py` | Crash- / omslaan- / offroad- / vastzit-detectie (puur visueel). |
| `speed_ocr.py` | Snelheid lezen van de HUD via OCR, in een aparte thread. |
| `capture.py` | Screen capture, voorbewerking naar 84×84, optical flow. |
| `controller.py` | Virtuele Xbox-controller (vgamepad) + actie-decodering. |
| `viewer.py` | Het live OpenCV-dashboard. |
| `calibrate.py` | Interactieve kalibratie-wizard. |
| `config_store.py` | Laden/opslaan van `config.json`. |
| `utils.py` | DPI-awareness, toetsen pollen, losse hulpjes. |

---

## 1. Installatie (Windows)

Python 3.10 of 3.11, 64-bit.

```bat
pip install -r requirements.txt
```

Met een NVIDIA-GPU zet je torch daarna om naar de CUDA-build (fors sneller):

```bat
pip install torch --index-url https://download.pytorch.org/whl/cu121
```

> **Let op:** `opencv-python`, **niet** `opencv-python-headless`. De headless
> variant heeft geen GUI en dan werkt de live viewer niet.

### ViGEmBus-driver (verplicht voor `vgamepad`)

`vgamepad` emuleert een echte Xbox 360-controller via de ViGEmBus-kernel
driver. De installer van `vgamepad` biedt die meestal zelf aan tijdens
`pip install`. Gebeurt dat niet, installeer hem dan handmatig:

<https://github.com/nefarius/ViGEmBus/releases>

**Herstart je PC na de installatie.**

### Tesseract-OCR (voor de snelheidsdetectie)

<https://github.com/UB-Mannheim/tesseract/wiki>

Zet daarna het pad naar `tesseract.exe` in `settings.py`:

```python
TESSERACT_CMD = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
```

Geen zin in een aparte installatie? Zet dan `OCR_BACKEND = "easyocr"` in
`settings.py` en doe `pip install easyocr`. Zwaarder, maar draait op je GPU en
heeft geen losse `.exe` nodig.

Controleer alles in één keer:

```bat
python train.py --check
```

---

## 2. BeamNG.drive instellen

* **Graphics → Windowed of Borderless.** Géén exclusive fullscreen: `mss` kan
  dat niet betrouwbaar grabben en alt-tabben gaat flikkeren.
* **Vaste resolutie.** Verander die niet meer na de kalibratie, anders kloppen
  je crops niet meer (dan gewoon opnieuw kalibreren).
* **HUD zichtbaar**, met de digitale snelheidsmeter in beeld — daar leest de
  OCR van.
* **Controls:** de virtuele controller verschijnt als *"Xbox 360 Controller for
  Windows"*. Test eerst even of de auto reageert op de stick voordat je gaat
  trainen.
* **Auto:** neem iets traags en voorspelbaars, bijvoorbeeld de **Ibishu Covet**.
  Exact 50 km/u aanhouden met een supercar is onnodig lastig.
* **Circuit:** begin op **Gridmap** met een lange rechte weg, of West Coast USA /
  Italy. Bochtige bergwegen zijn veel te moeilijk voor de eerste sessies.
* **Wil je onbewaakt trainen?** Bind een controllerknop aan *Recover Vehicle*:
  `Options → Controls`, zoek op "Recover", en bind knop **Y** van de virtuele
  controller. Kies daarna reset-modus `gamepad` tijdens de kalibratie.

---

## 3. Kalibreren (één keer)

```bat
python train.py --calibrate
```

De wizard loopt door zes stappen:

1. **Monitor kiezen** — op welk scherm draait BeamNG.
2. **Game-venster aanduiden** — er wordt een screenshot genomen; sleep met de
   muis een kader rond het spelbeeld (zonder titelbalk).
3. **Snelheidsmeter aanduiden** — sleep een **strak** kader rond enkel de
   cijfers. Dus zonder "km/h", zonder toerenteller, zonder versnellingsbak.
   Hoe strakker, hoe betrouwbaarder de OCR.
4. **OCR live testen** — je ziet de voorbewerkte crop én het gelezen getal.
   Klopt het niet? Druk `i` (inverteren), `t` (drempelmethode) of `+`/`-`
   (upscale) tot het percentage "leesbaar" hoog staat. `ENTER` = opslaan.
5. **Wegkleur samplen** — parkeer de auto midden op de weg; het groene kader
   moet enkel asfalt bevatten. `SPATIE` = samplen, `ENTER` = klaar.
   Dit voedt de offroad-detectie. Overslaan met `C` mag; dan valt enkel de
   offroad-detectie weg.
6. **Reset-modus kiezen** — `gamepad` (automatisch, aanbevolen), `manual`
   (jij drukt op R), of `key` (INSERT-toets simuleren).

Alles gaat naar **`config.json`**. Die hoef je daarna nooit meer aan te raken —
tenzij je resolutie of UI-schaal verandert.

---

## 4. Eerst controleren, dan trainen

```bat
python train.py --preview
```

Dit draait de volledige waarnemingsketen (capture → OCR → detectie → viewer)
**zonder te trainen en zonder de auto te besturen**. Rijd zelf wat rond en kijk
of de snelheid klopt, of de wegdek-fractie hoog staat op de weg en laag in het
gras, en of een crash correct herkend wordt. Pas hier je drempels aan; dat
scheelt je later uren.

Daarna:

```bat
python train.py
```

Andere opties:

```bat
python train.py --resume checkpoints/latest.zip   # verder trainen
python train.py --timesteps 100000                # korter/langer
python train.py --no-viewer                       # zonder dashboard (sneller)
python train.py --check                           # diagnose
```

---

## 5. Wat je in de live viewer ziet

Sleep het viewer-venster naast je BeamNG-venster.

```
┌──────────────┬───────────────────────┬────────────────────────┐
│ OBSERVATIE   │  SNELHEID / REWARD    │  REWARD-OPSPLITSING    │
│ 84×84 grijs, │  grote km/u-weergave  │  snelheid  +0.98       │
│ zoals het    │  doelbalk met de      │  vooruitgang +0.14     │
│ CNN het ziet │  ±5 km/u piekband     │  stilstand  +0.00      │
│              │  reward van deze stap │  stuur-jerk -0.02      │
│ framestack   ├───────────────────────┼────────────────────────┤
│ (4 frames)   │  CONTROLLER-OUTPUT    │  CRASH-/OFFROAD-       │
├──────────────┤  stuur ───┼───        │  DETECTIE              │
│ HUD-CROP     │  gas  ████████        │  wegdek 87%            │
│ NAAR OCR     │  rem  ░░░░░░░░        │  frame-diff, streaks   │
│ leesbaar 94% ├───────────────────────┴────────────────────────┤
└──────────────┤  REWARD PER STAP (lijngrafiek + gemiddelde)    │
               ├────────────────────────────────────────────────┤
               │  TOTALE REWARD PER EPISODE (balkjes)           │
               └────────────────────────────────────────────────┘
   episode 12 | stap 340 | totaal 8420 | ep-reward +212.4 | gem. 47.3 km/u
```

Concreet:

* **Observatie** — exact het beeld dat naar het CNN gaat, opgeschaald zodat je
  het kan lezen. Zie je hier vooral lucht of HUD? Pas `OBS_CROP` aan.
* **Framestack** — de laatste 4 frames als strip; zo zie je de beweging waar
  het netwerk zijn snelheidsgevoel uit haalt.
* **HUD-crop naar OCR** — de voorbewerkte crop plus het leesbaarheidspercentage.
  Zakt dat onder ~80 %, herkalibreer dan stap 3/4.
* **Snelheid** — groot en gekleurd: groen binnen de piekband, oranje er net
  buiten, rood ver ervandaan. `--` betekent dat OCR niets kan lezen.
* **Reward-opsplitsing** — per term, met balkjes rond een nulpunt. Hier zie je
  meteen waarom een stap goed of slecht scoorde.
* **Controller-output** — wat er werkelijk naar de virtuele stick en triggers
  gaat, ná deadzone en smoothing.
* **Detectie** — wegdek-fractie, frame-diff en de streak-tellers. Handig om te
  zien of je vlak voor een valse crash-detectie zat.
* **Grafieken** — reward per stap (met voortschrijdend gemiddelde) en het
  totaal per afgelopen episode.

### Toetsen

| Toets | Effect |
|---|---|
| `q` | training stoppen en het model opslaan |
| `p` | pauzeren / hervatten (controller gaat neutraal) |
| `r` | "auto staat klaar" tijdens een handmatige reset |
| `d` | debug-overlay (wegdek-masker) aan/uit |

> De virtuele controller stuurt enkel naar het **gefocuste** venster, dus tijdens
> de training staat BeamNG vooraan. Wil je op `q`/`p`/`r` drukken, klik dan eerst
> op het viewer-venster of op de terminal. De auto rolt dan even stuurloos door —
> dat is normaal.

---

## 6. Hoe het werkt

### Observatie
Screen capture met `mss` → crop (lucht en HUD eraf) → grayscale → 84×84.
SB3's `VecFrameStack` stapelt 4 frames, zodat het netwerk beweging en dus
snelheid kan afleiden uit puur beeld.

### Acties
Continue `Box(-1, 1, shape=(2,))`:

* `action[0]` → sturen, linker analoge stick (met deadzone, limiet en smoothing)
* `action[1]` → `> 0` gas op de rechter trigger, `< 0` remmen op de linker trigger

### Reward

```
      1.0 |        ______________
          |       /              \
          |      /                \
      0.0 |_____/                  \_________
          0    45   50   55            100 km/u
```

* **Snelheid** — flat-top gaussian: volle reward binnen 50 ± 5 km/u, daarbuiten
  gaussisch afvallend. Te hard rijden wordt iets strenger afgestraft dan te
  traag (`SPEED_SIGMA_FAST < SPEED_SIGMA_SLOW`), maar **beide** kosten reward.
* **Stilstand** — `-0.45` per stap onder 2 km/u.
* **Vooruitgang** — kleine bonus op basis van optical flow. Zonder deze term is
  stilstaan een lokaal optimum: het levert namelijk nooit een crash op.
* **Stuur-jerk** — kleine straf op abrupte stuurwissels, geeft vloeiender rijden.
* **Terminaal** — `-12` bij crash/omslaan, `-8` offroad, `-6` vastzitten.

Alles staat in `settings.py` sectie 4, de berekening in `reward.py`.

### Crash- en offroad-detectie (puur visueel)

Vijf goedkope signalen, gecombineerd. De volledige uitleg staat bovenaan
`detectors.py`; kort samengevat:

1. **Plotse snelheidsval** — meer dan 18 km/u verlies binnen 0,35 s zonder dat
   de rem ingedrukt is → botsing.
2. **Beeldschok** — piek in het gemiddelde frameverschil → impact.
3. **Lucht onderaan** — onderste beeldstrook veel helderder dan de bovenste →
   op het dak.
4. **Wegdek-fractie** — percentage pixels in een ROI vlak voor de auto dat op
   het gekalibreerde asfalt lijkt. Te laag en te lang → van de weg af.
5. **Vastzitten** — gas open maar geen snelheid gedurende 3 s.

De eerste `WARMUP_STEPS` stappen van een episode wordt er niets afgekeurd, zodat
de auto rustig kan settelen na een recover.

### Reset
Na een crash zet `reset()` de auto terug op de weg:

* **`gamepad`** *(aanbevolen)* — drukt op knop `Y` van de virtuele controller.
  Bind die in BeamNG aan *Recover Vehicle*. Volledig automatisch en blijft
  binnen de "alleen controller"-regel. Er wordt daarna gecontroleerd of er weer
  wegdek onder de wielen ligt; zo niet volgen er nog twee pogingen en anders
  wordt jou om hulp gevraagd.
* **`manual`** — de terminal en de viewer vragen je de auto klaar te zetten;
  druk daarna op `R`.
* **`key`** — simuleert de `INSERT`-toets (BeamNG's recover). Uitsluitend voor
  het resetten; het rijden zelf gaat altijd via de controller.

### Realtime-details die makkelijk over het hoofd gezien worden

* **Vaste timestep.** `step()` slaapt tot `STEP_DT` vol is. Hapert het spel, dan
  wordt dat gelogd (`info["step_dt"]`) maar niet ingehaald.
* **PPO-updates.** Tijdens `model.train()` staat de env stil terwijl de auto
  doorrijdt. Daarom zet de callback vlak ervoor gas los met een beetje rem, en
  wist hij daarna alle tijdreeksen — anders leest de crash-detector dat gat van
  enkele seconden als een plotse snelheidsval.
* **OCR in een aparte thread.** Tesseract kost 15–40 ms per call; in de control
  loop zou je de 10 Hz nooit halen. De loop leest gewoon de laatst bekende
  waarde, met een houdbaarheidsdatum van 0,6 s.
* **DPI-awareness.** Wordt aangezet vóór de eerste capture. Zonder dat liegt
  Windows over je schermcoördinaten zodra je schaling niet op 100 % staat.

---

## 7. Logging en checkpoints

* `checkpoints/beamng_ppo_*.zip` — elke 5 000 stappen (`CHECKPOINT_EVERY`)
* `checkpoints/latest.zip` — bij afsluiten, ook na `Ctrl+C` of `q`
* `logs/episodes.csv` — per episode: reward, lengte, gemiddelde snelheid, reden
  van beëindiging
* `logs/monitor.csv` — de standaard SB3 Monitor-log
* TensorBoard:

```bat
tensorboard --logdir logs/tb
```

---

## 8. Zelf bijstellen

Bijna alles zit in `settings.py`, per thema gegroepeerd:

| Wil je... | Pas dit aan |
|---|---|
| een andere doelsnelheid | `TARGET_SPEED`, `SPEED_PEAK_BAND` |
| strenger op te hard rijden | `SPEED_SIGMA_FAST` omlaag |
| rustiger stuurgedrag | `STEER_LIMIT` omlaag, `STEER_SMOOTHING` omhoog |
| minder valse crashmeldingen | `CRASH_SPEED_DROP` en `IMPACT_DIFF_THRESHOLD` omhoog |
| minder valse offroad-meldingen | `ROAD_MIN_FRACTION` omlaag, `OFFROAD_CONSECUTIVE` omhoog |
| meer/minder beeld voor het CNN | `OBS_CROP`, `OBS_SIZE` |
| sneller reageren | `STEP_DT` omlaag (zwaarder voor je CPU) |
| andere PPO-instellingen | `PPO_KWARGS` |

De reward zelf herschrijf je in `reward.py`; de env hoef je daarvoor niet aan te
raken.

---

## 9. Als het misgaat

| Probleem | Oorzaak / oplossing |
|---|---|
| `vgamepad` fout bij het starten | ViGEmBus niet geïnstalleerd of PC niet herstart. Check met `python train.py --check`. |
| Auto reageert niet | BeamNG staat niet vooraan, of de controller is niet toegelaten in `Options → Controls`. |
| Snelheid blijft `--` | Crop te ruim of verkeerd. Herkalibreer stap 3/4 en let op het leesbaarheidspercentage. |
| Snelheid springt onzinnig | Crop bevat meer dan alleen de cijfers (toerenteller, versnelling). Strakker selecteren. |
| Episodes eindigen meteen | Wegkleur verkeerd gesampled, of `ROAD_MIN_FRACTION` te hoog. Check de wegdek-fractie in `--preview`. |
| Alles staat scheef in beeld | Windows-schaling veranderd, of BeamNG-venster verplaatst. Herkalibreer. |
| Zwart beeld in de capture | Exclusive fullscreen. Zet BeamNG op windowed/borderless. |
| Viewer opent niet | `opencv-python-headless` geïnstalleerd i.p.v. `opencv-python`. |
| Training is traag | Torch draait op CPU. Installeer de CUDA-build. |

---

## 10. Realistische verwachtingen

Dit leert in **realtime**, dus 10 stappen per seconde: 100 000 stappen is
ongeveer **3 uur effectief rijden**, plus de tijd die in resets gaat zitten.
Reken op meerdere sessies, en gebruik `--resume` om verder te gaan.

Wat je ongeveer mag verwachten:

* **0 – 5 k stappen:** willekeurig gedrag, veel crashes, korte episodes.
* **5 – 20 k stappen:** de agent ontdekt gas geven; episodes worden langer.
* **20 – 60 k stappen:** hij begint de weg te volgen en rond de doelsnelheid te
  hangen op rechte stukken.
* **60 k+:** redelijk stabiel op eenvoudige circuits. Bochten blijven het
  moeilijkst, want de agent heeft geen enkel besef van de weg vóór hem behalve
  wat er letterlijk in beeld staat.

Begin klein: een rechte weg op Gridmap, een trage auto, en `--preview` om te
controleren dat je waarnemingen kloppen voordat je uren aan training investeert.
