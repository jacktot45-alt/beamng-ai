# Super simpele handleiding

Van niks naar een trainende AI in ongeveer **45 minuten**. Je hoeft geen
Python te kennen. Volg gewoon de stappen van boven naar beneden.

Na elke stap staat er **"Wat je moet zien"**. Klopt dat niet? Stop daar en kijk
onderaan bij [Als er iets misgaat](#als-er-iets-misgaat). Niet doorgaan naar de
volgende stap — dan stapelen de problemen zich op.

> **Tip:** houd dit venster open naast je terminal. Alle commando's kan je
> kopiëren en plakken (rechtermuisknop in de terminal = plakken).

**Inhoud**

1. [Python installeren](#stap-1--python-installeren-5-min)
2. [De bestanden ophalen](#stap-2--de-bestanden-ophalen-3-min)
3. [Een terminal openen in de juiste map](#stap-3--een-terminal-openen-in-de-juiste-map-1-min)
4. [Packages installeren](#stap-4--packages-installeren-5-min)
5. [De controller-driver](#stap-5--de-controller-driver-vigembus-3-min)
6. [Tesseract voor het aflezen van de snelheid](#stap-6--tesseract-voor-het-aflezen-van-de-snelheid-3-min)
7. [Alles controleren](#stap-7--alles-controleren-1-min)
8. [BeamNG klaarzetten](#stap-8--beamng-klaarzetten-5-min)
9. [Kalibreren](#stap-9--kalibreren-5-min)
10. [Controleren of hij goed kijkt](#stap-10--controleren-of-hij-goed-kijkt-5-min)
11. [Trainen!](#stap-11--trainen)
12. [Stoppen en later verdergaan](#stap-12--stoppen-en-later-verdergaan)

---

## Stap 1 — Python installeren (5 min)

Heb je al Python 3.10 of 3.11? Sla deze stap over.

1. Ga naar <https://www.python.org/downloads/release/python-3119/>
2. Scroll naar beneden naar **Files**
3. Download **Windows installer (64-bit)**
4. Start de installer

> ### ⚠️ Het allerbelangrijkste van deze hele handleiding
>
> Zet in het eerste scherm van de installer een vinkje bij
> **"Add python.exe to PATH"** (onderaan het venster).
>
> Vergeet je dat, dan werkt geen enkel commando hieronder. Je kan de installer
> dan gewoon opnieuw draaien.

5. Klik **Install Now** en wacht tot hij klaar is.

**Wat je moet zien:** open het Startmenu, typ `cmd`, druk Enter. Typ in het
zwarte venster:

```bat
python --version
```

Er moet iets staan als `Python 3.11.9`. Krijg je een foutmelding of opent de
Microsoft Store? Dan is het vinkje niet gezet — installeer opnieuw.

---

## Stap 2 — De bestanden ophalen (3 min)

De makkelijkste manier, zonder git:

1. Ga naar <https://github.com/jacktot45-alt/beamng-ai>
2. Klik linksboven op het knopje waar **`main`** staat en kies de branch
   **`claude/beamng-rl-vision-agent-gs36oc`**
3. Klik rechts op de groene knop **Code** → **Download ZIP**
4. Pak de ZIP uit naar een map die je makkelijk terugvindt, bijvoorbeeld:

```
C:\beamng-ai
```

**Wat je moet zien:** in die map staan bestanden als `train.py`,
`settings.py`, `README.md`. Staat er nog een extra map tussen (zoals
`beamng-ai-claude-beamng-rl-vision-agent-gs36oc`)? Ga dan díé map in — daar
moet `train.py` direct in staan.

<details>
<summary>Liever met git? (klik open)</summary>

```bat
git clone https://github.com/jacktot45-alt/beamng-ai.git C:\beamng-ai
cd C:\beamng-ai
git checkout claude/beamng-rl-vision-agent-gs36oc
```
</details>

---

## Stap 3 — Een terminal openen in de juiste map (1 min)

Dit ga je elke keer doen, dus onthoud het:

1. Open de map `C:\beamng-ai` in Verkenner
2. Klik in de **adresbalk** bovenaan (waar het pad staat)
3. Typ `cmd` en druk Enter

Er opent een zwart venster dat al in de juiste map staat.

**Wat je moet zien:** de regel begint met `C:\beamng-ai>`. Test het even:

```bat
dir
```

Je moet `train.py` in de lijst zien staan.

---

## Stap 4 — Packages installeren (5 min)

Typ in datzelfde zwarte venster:

```bat
pip install -r requirements.txt
```

Nu wordt er van alles gedownload. **Dit duurt een paar minuten** (torch alleen
al is ruim 200 MB). Laat het rustig lopen. Gele waarschuwingen zijn prima; let
alleen op rode `ERROR`-regels.

**Wat je moet zien:** helemaal onderaan `Successfully installed ...` met een
lange lijst namen.

### Heb je een NVIDIA-videokaart?

Doe dan hierna nog dit — het maakt de training een stuk sneller:

```bat
pip install torch --index-url https://download.pytorch.org/whl/cu121
```

Weet je het niet zeker? Sla het over. Het werkt ook zonder, alleen trager.

---

## Stap 5 — De controller-driver (ViGEmBus) (3 min)

Dit is de driver die Windows laat denken dat er een echte Xbox-controller
aangesloten is. Zonder dit kan de AI niet rijden.

Meestal wordt hij automatisch mee geïnstalleerd in stap 4. Zo niet:

1. Ga naar <https://github.com/nefarius/ViGEmBus/releases>
2. Download het `.exe`-bestand van de bovenste (nieuwste) versie
3. Installeer het
4. **Herstart je PC** ← echt doen, anders werkt het niet

---

## Stap 6 — Tesseract (voor het aflezen van de snelheid) (3 min)

Dit programma leest de cijfers van je snelheidsmeter van het scherm.

1. Ga naar <https://github.com/UB-Mannheim/tesseract/wiki>
2. Download de **64-bit installer**
3. Installeer met **alle standaardinstellingen** — verander het installatiepad
   niet, dan klopt het automatisch

**Wat je moet zien:** de map `C:\Program Files\Tesseract-OCR` bestaat en er
staat een `tesseract.exe` in.

Heb je hem toch ergens anders geïnstalleerd? Open dan `settings.py` met
Kladblok en pas deze regel aan naar jouw pad:

```python
TESSERACT_CMD = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
```

---

## Stap 7 — Alles controleren (1 min)

Nu kijken we of alles er staat. Typ:

```bat
python train.py --check
```

**Wat je moet zien:**

```
--- Omgevingscheck ---
  [ok]   platform is Windows - win32
  [ok]   mss - 9.0.1
  [ok]   opencv (met GUI) - 4.9.0
  [ok]   numpy - 1.26.4
  [ok]   gymnasium - 0.29.1
  [ok]   stable-baselines3 - 2.2.1
  [ok]   torch - 2.2.0, cuda=ja
  [ok]   vgamepad + ViGEmBus - virtuele controller aangemaakt
  [ok]   OCR-backend (pytesseract) - pytesseract
  [FOUT] config.json ontbreekt - run: python train.py --calibrate
```

Alleen die laatste regel over `config.json` mag rood zijn — dat gaan we nu
oplossen. Staat er ergens anders `[FOUT]`? Kijk onderaan bij
[Als er iets misgaat](#als-er-iets-misgaat).

`cuda=nee` is geen fout, alleen trager.

---

## Stap 8 — BeamNG klaarzetten (5 min)

Start BeamNG.drive en zet dit goed:

| Instelling | Waarde | Waarom |
|---|---|---|
| **Graphics → Display mode** | **Windowed** of **Borderless** | In fullscreen kan het programma je scherm niet meelezen |
| **Resolutie** | maakt niet uit, maar **verander hem later niet meer** | anders klopt de kalibratie niet meer |
| **HUD** | aan, met de snelheidsmeter zichtbaar | daar wordt de snelheid van afgelezen |

Kies daarna:

* **Level:** **Gridmap** — daar heb je lange kaarsrechte wegen. Begin hier.
  Bochtige bergwegen zijn veel te moeilijk voor het begin.
* **Auto:** **Ibishu Covet** — traag en voorspelbaar. Met een supercar is
  precies 50 km/u aanhouden onnodig lastig.

### Werkt de controller straks wel?

De regel `[ok] vgamepad + ViGEmBus` uit stap 7 betekent dat de **driver** in
orde is. Of **BeamNG** hem ook accepteert merk je pas als de training loopt:
de virtuele controller bestaat alleen zolang het script draait.

Dus geen zorgen nu — je test dit vanzelf in stap 11. Beweegt de auto daar
binnen een paar seconden? Dan werkt het. Beweegt hij niet, laat de training
dan gewoon dóórlopen en kijk in BeamNG bij **Options → Controls**: daar moet
*"Xbox 360 Controller for Windows"* staan. Staat hij er niet, herstart BeamNG
terwijl het script blijft draaien.

### Optioneel maar handig: automatisch resetten

Wil je de AI uren laten trainen zonder er zelf bij te zitten?

1. Ga in BeamNG naar **Options → Controls**
2. Zoek op **"recover"**
3. Bind de actie *Recover Vehicle* aan **knop Y** van de controller

Doe je dit niet, dan is dat prima — je moet dan alleen zelf op `R` drukken na
elke crash. Dat kies je zo in stap 9.

---

## Stap 9 — Kalibreren (5 min)

Nu vertel je het programma waar het naar moet kijken. Dit doe je **één keer**.

Zorg dat BeamNG draait en zichtbaar is. Typ dan:

```bat
python train.py --calibrate
```

Je krijgt zes stappen. Je mag steeds heen en weer alt-tabben tussen de terminal
en het spel.

**1. Welke monitor?**
Typ het nummer van het scherm waar BeamNG op staat (meestal `1`).

**2. Waar staat het BeamNG-venster?**
Druk Enter → er wordt een screenshot gemaakt → **sleep met je muis een kader
rond het spelbeeld** (dus niet rond de titelbalk of de vensterrand) → druk
**Enter** om te bevestigen.

**3. Waar staat de snelheidsmeter?**
Nu zie je alleen het spelbeeld. **Sleep een strak kader rond alleen de
cijfers** van de snelheid.

> Dit is de stap waar het meestal misgaat. Selecteer **alleen de cijfers**.
> Dus niet de "km/h" ernaast, niet de toerenteller, niet het bakje eromheen.
> Liever iets te strak dan te ruim.

**4. OCR testen**
Je ziet nu live wat het programma leest. Bovenaan staat een getal en een
percentage "leesbaar".

* Klopt het getal met wat er in het spel staat, en is "leesbaar" hoog
  (80 % of meer)? Druk **Enter** en ga door.
* Staat er onzin? Druk op `i` (kleuren omdraaien), of op `t` (andere methode),
  of op `+` (groter maken). Blijf proberen tot het klopt, druk dan Enter.
* Lukt het echt niet? Ga terug naar stap 3 met een strakkere selectie.

**5. Kleur van de weg**
Zet de auto midden op de weg. Je ziet een groen kader vlak voor de auto — daar
mag **alleen asfalt** in zitten, geen gras en geen witte streep. Druk op
**spatie** om te meten, dan op **Enter**.

Je ziet nu een percentage. Op de weg moet dat hoog zijn (70 % of meer). Rijd
even het gras op om te testen: dan moet het laag zijn. Klopt dat? Enter.

**6. Hoe resetten na een crash?**
* Heb je in stap 8 knop Y gebonden? Typ `1` (automatisch, aanbevolen).
* Zo niet? Typ `2` (jij drukt zelf op R na elke crash).

**Wat je moet zien:** `Kalibratie opgeslagen in C:\beamng-ai\config.json`

Klaar. Dit hoef je nooit meer te doen — behalve als je je schermresolutie of
je Windows-schaling verandert.

---

## Stap 10 — Controleren of hij goed kijkt (5 min)

**Sla deze stap niet over.** Hij kost 5 minuten en kan je uren nutteloze
training besparen.

```bat
python train.py --preview
```

Nu opent het dashboard-venster. **De AI rijdt hier nog niet** — jij rijdt zelf,
met je eigen toetsenbord of controller, en het programma kijkt alleen mee.

Sleep het dashboard naast je BeamNG-venster zodat je allebei ziet.

Rijd wat rond en controleer deze vier dingen:

| Wat | Waar in het dashboard | Moet zijn |
|---|---|---|
| **Snelheid** | groot getal midden bovenin | hetzelfde als in het spel |
| **Leesbaar %** | linksonder onder de HUD-crop | 80 % of hoger |
| **Wegdek %** | rechts, bij detectie | hoog op de weg, laag in het gras |
| **Observatie** | linksboven, het grijze plaatje | vooral wég te zien, weinig lucht |

Rijd ook eens expres tegen iets aan. Er moet dan rood **CRASH** verschijnen.
En rijd het gras op: na een seconde of twee moet er **OFFROAD** staan.

> Daarna verschijnt er een scherm **"WACHTEN OP RESET"**. Dat hoort zo: in
> preview stuurt het programma bewust niks naar je auto, ook geen resetknop.
> Zet de auto zelf terug (in BeamNG is `Insert` = recover) en druk op `R` om
> door te gaan.

Werkt dat allemaal? Druk op `q` om te stoppen. **Je bent er klaar voor.**

Klopt er iets niet? Kijk in de tabel onderaan — bijna alles los je op door
stap 9 opnieuw te doen.

---

## Stap 11 — Trainen!

```bat
python train.py
```

Je krijgt 5 seconden aftellen. **Klik in die tijd op het BeamNG-venster**, want
de virtuele controller stuurt alleen naar het venster dat vooraan staat.

En dan rijdt hij. Slecht, in het begin. Heel slecht.

### Wat je ziet gebeuren

| Wanneer | Wat je ziet |
|---|---|
| **eerste 10 minuten** | volledig willekeurig, crasht constant, korte episodes |
| **na ~1 uur** | ontdekt gas geven, episodes worden langer |
| **na ~3 uur** | volgt de weg op rechte stukken, hangt rond 50 km/u |
| **na ~6 uur** | redelijk stabiel op makkelijke circuits |

Dit is normaal. De AI leert in **echte tijd**, dus 10 beslissingen per seconde —
niet versneld. Geduld hoort erbij. Laat het gerust een paar uur lopen.

> Vind je het in het begin te chaotisch? Open `settings.py` met Kladblok en zet
> `STEER_LIMIT = 0.65` naar `0.4`. Dan stuurt hij minder wild.

### De knoppen

Wil je op een knop drukken, **klik dan eerst op het dashboard-venster of op de
terminal**. De auto rolt dan even stuurloos door — dat hoort zo, want de
controller stuurt alleen naar het venster dat vooraan staat.

| Toets | Wat het doet |
|---|---|
| `q` | stoppen **en opslaan** |
| `p` | pauze |
| `r` | "de auto staat weer klaar" (als je resetmodus `manual` koos) |
| `d` | extra debug-info aan/uit |

---

## Stap 12 — Stoppen en later verdergaan

Druk op **`q`**, of op `Ctrl+C` in de terminal. Er wordt altijd opgeslagen.

Verdergaan waar je gebleven was:

```bat
python train.py --resume checkpoints/latest.zip
```

Wil je zien hoe het gaat? In `logs/episodes.csv` staat per episode de reward,
de gemiddelde snelheid en waarom de episode eindigde. Open dat gewoon met
Excel.

Liever grafieken?

```bat
tensorboard --logdir logs/tb
```

Open dan <http://localhost:6006> in je browser.

---

## Spiekbriefje

Alles wat je nodig hebt, op één plek:

```bat
python train.py --check                            :: werkt alles?
python train.py --calibrate                        :: opnieuw instellen
python train.py --preview                          :: kijken zonder trainen
python train.py                                    :: trainen
python train.py --resume checkpoints/latest.zip    :: verdergaan
```

Terminal openen in de juiste map: map openen in Verkenner → in de adresbalk
klikken → `cmd` typen → Enter.

---

## Als er iets misgaat

| Wat je ziet | Wat je doet |
|---|---|
| `'python' is not recognized` | Het vinkje "Add python.exe to PATH" is vergeten in stap 1. Installeer Python opnieuw. |
| `'pip' is not recognized` | Zelfde oorzaak. Of probeer `python -m pip` in plaats van `pip`. |
| `can't open file 'train.py'` | Je terminal staat in de verkeerde map. Doe stap 3 opnieuw. |
| `[FOUT] vgamepad + ViGEmBus` | Driver niet geïnstalleerd of PC niet herstart. Doe stap 5, herstart echt. |
| `[FOUT] OCR-backend` | Tesseract niet geïnstalleerd of op een ander pad. Doe stap 6. |
| Auto reageert nergens op | BeamNG staat niet vooraan (klik erop), of de controller staat uit in Options → Controls. |
| Snelheid blijft `--` | Crop klopt niet. Doe stap 9 opnieuw, selecteer **alleen** de cijfers. |
| Snelheid springt naar rare getallen | Er zit meer dan de cijfers in je selectie (toerenteller, versnelling). Strakker selecteren. |
| Episodes stoppen meteen | De wegkleur is verkeerd gemeten. Doe stap 9 opnieuw en zorg dat er alleen asfalt in het groene kader zit. Of zet `ROAD_MIN_FRACTION` in `settings.py` van `0.35` naar `0.20`. |
| Zwart beeld in het dashboard | BeamNG staat in fullscreen. Zet hem op Windowed of Borderless. |
| Dashboard opent niet | Verkeerde opencv. Doe: `pip uninstall opencv-python-headless` en dan `pip install opencv-python`. |
| Alles staat scheef in beeld | Je hebt het BeamNG-venster verplaatst of je Windows-schaling veranderd. Doe stap 9 opnieuw. |
| Training voelt heel traag | Torch draait op je processor in plaats van je videokaart. Zie het NVIDIA-stukje in stap 4. |
| Hij stuurt veel te wild | `settings.py` openen, `STEER_LIMIT` van `0.65` naar `0.4`. |
| Hij blijft stilstaan | `settings.py` openen, `IDLE_PENALTY` van `-0.45` naar `-0.8`. |

Bij twijfel: **doe stap 9 (kalibreren) opnieuw en controleer met stap 10
(preview).** Negen van de tien problemen komen daaruit voort.

---

## En daarna?

Als het draait en je wil zelf spelen: alles wat je kan bijstellen staat in
`settings.py`, met uitleg per regel. De belangrijkste:

```python
TARGET_SPEED = 50.0     # hoe snel hij moet willen rijden
STEER_LIMIT = 0.65      # hoe ver hij mag sturen (lager = rustiger)
CRASH_PENALTY = -12.0   # hoe erg een crash is
STEP_DT = 0.10          # 10 beslissingen per seconde
```

De volledige technische uitleg staat in [README.md](README.md).
