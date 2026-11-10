# Regard 👁️

**Savoir où se pose votre regard, application par application**, avec une simple webcam.
Et à terme, piloter son PC avec le visage (clic par clin d'œil, curseur qui suit le regard…).

Tout fonctionne **en local** : aucune image ne quitte votre PC et rien n'est enregistré
en dehors des coordonnées du regard (dans `~/.regard/regard.db`).

## Ce que ça fait

- **Calibration** (≈ 1 min) : vous fixez 15 points, puis 5 points **en bougeant doucement
  la tête** (une flèche indique le mouvement). Regard apprend ainsi la correspondance
  yeux → écran *et* à compenser les mouvements de tête, puis mesure sa précision.
- **Correction automatique par les clics** : quand vous cliquez, vous regardez le curseur.
  Chaque clic devient un point de calibration gratuit : Regard corrige sa dérive en direct
  et se ré-entraîne en arrière-plan toutes les 6 clics. La fenêtre de contrôle affiche
  l'amélioration (« erreur 140 → 60 px »).
- **3 clignements rapides → le regard pilote le curseur** (et 3 de plus pour arrêter).
- **Suivi en direct** : une petite fenêtre affiche l'application regardée (« chrome »,
  « Code », « Discord »…) et le titre de la fenêtre ; un point rouge montre le regard estimé.
- **Rapport** : temps passé à regarder chaque application, carte de chaleur sur l'écran
  entier et **à l'intérieur de chaque fenêtre**, titres les plus regardés.
- **Gestes du visage** (expérimental) : clin d'œil gauche/droit, yeux fermés longtemps,
  bouche ouverte, sourcils levés, sourire → clic, clic droit, défilement, curseur qui suit
  le regard. Désactivés par défaut.

## Installation (Windows)

1. Installez [Python](https://www.python.org/downloads/) 3.10 ou plus récent en cochant
   **« Add python.exe to PATH »**.
2. Double-cliquez sur **`installer.bat`** (une seule fois).
3. Double-cliquez sur **`Regard.bat`** et choisissez une option.

En ligne de commande (Windows, macOS, Linux) :

```bash
python -m venv .venv
```

```bash
.venv/bin/pip install -r requirements.txt   # Windows : .venv\Scripts\pip
```

```bash
.venv/bin/python regard.py track
```

## Commandes

| Commande | Rôle |
|---|---|
| `python regard.py preview` | Aperçu webcam : yeux, iris, pose de la tête, scores des gestes |
| `python regard.py calibrate` | Calibrer (option `--grid 4x6` pour plus de points) |
| `python regard.py track` | Suivre le regard (calibre d'abord si nécessaire) |
| `python regard.py track --recalibrate` | Refaire la calibration avant de suivre |
| `python regard.py track --actions` | Démarrer avec les actions par gestes activées |
| `python regard.py track --no-autocorrect` | Sans la correction par les clics |
| `python regard.py calibrate --no-head` | Calibration rapide, sans la phase « tête mobile » |
| `python regard.py report --days 7` | Rapport HTML des 7 derniers jours |

Options communes : `--camera 1` (autre webcam), `--res 1920x1080`.

## Précision : à quoi s'attendre

Une webcam n'est pas un eye-tracker infrarouge (Tobii…). Après une bonne calibration,
comptez une erreur de **2 à 4 cm** sur l'écran : largement assez pour savoir **quelle
application ou quelle zone** vous regardez, pas assez pour viser un petit bouton.

Pour de meilleurs résultats :
- visage bien éclairé de face (pas de fenêtre lumineuse derrière vous) ;
- webcam au-dessus de l'écran, à hauteur des yeux ;
- bien faire la phase « tête mobile » de la calibration (mouvements doux, ±10°) ;
- cliquer normalement : chaque clic affine le modèle. Si vous changez complètement de
  place, **recalibrez** (option 2 de `Regard.bat`) ;
- les lunettes fonctionnent, mais les reflets dégradent la précision.

## Gestes du visage et actions

**3 clignements rapides** (en moins de 1,5 s) : le curseur suit votre regard. Encore
3 clignements : il s'arrête. Toujours disponible pendant le suivi. Le curseur ne bouge pas
tant que le regard reste dans une petite zone (45 px) pour éviter les tremblements ;
réglable avec `follow_deadzone` et `follow_smoothing` dans `options`.

Les autres gestes déclenchent des actions **seulement si elles sont activées** (case
« Actions par gestes du visage », `--actions`, ou yeux fermés ~1 s) :

| Geste | Action |
|---|---|
| 3 clignements rapides | le regard pilote le curseur (on / off) |
| Clin d'œil gauche | clic gauche (à l'endroit du regard) |
| Clin d'œil droit | clic droit |
| Yeux fermés ~1 s | activer / désactiver les actions (interrupteur de sécurité) |
| Sourcils levés | défilement vers le haut |

Tout est réglable dans `~/.regard/gestures.json` (seuils, durées, actions :
`left_click`, `right_click`, `double_click`, `scroll_up`, `scroll_down`, `toggle_follow`,
`toggle_actions`, `none`). Utilisez `python regard.py preview` pour voir les scores en direct
et ajuster les seuils à votre visage. Selon la caméra, « gauche » et « droit » peuvent être
inversés : l'aperçu permet de vérifier.

## Comment ça marche

```
webcam ─► MediaPipe Face Landmarker ─► caractéristiques ─► modèle calibré ─► filtre 1€ ─► point (x, y)
           478 points + iris            iris dans l'œil,                                     │
           52 expressions               ouverture,                                           ▼
           pose de la tête              pose de la tête                     fenêtre sous le point (Win32)
                                                                            ─► SQLite ─► rapport HTML
```

- `regard/face.py` — webcam, MediaPipe, extraction des caractéristiques ;
- `regard/model.py` — régression ridge polynomiale (validation croisée), filtre 1€ ;
- `regard/adapt.py` — correction automatique par les clics (correction locale + ré-apprentissage) ;
- `regard/calibration.py` — écran de calibration et de vérification ;
- `regard/windows.py` — application sous un point : Windows (Win32, y compris applis UWP),
  macOS (Quartz) et Linux X11 ;
- `regard/tracker.py` — boucle de suivi, point du regard, fenêtre de contrôle ;
- `regard/gestures.py` — détection des gestes et actions souris ;
- `regard/store.py`, `regard/report.py` — enregistrement et rapport.

## Limites connues / pistes

- Calibration sur l'écran principal uniquement (pas encore de multi-écrans).
- Les changements de distance à l'écran (se pencher en avant) sont moins bien compensés
  que les rotations de tête ; la correction par les clics les rattrape au fil de l'eau.
- Les corrections apprises par les clics ne sont pas conservées d'une session à l'autre.
- macOS : autoriser la webcam et « Enregistrement de l'écran » (pour les titres de fenêtres).
- Linux : X11 uniquement (Wayland ne permet pas de savoir quelle fenêtre est sous un point).
