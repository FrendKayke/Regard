# Regard 👁️

**Savoir où se pose votre regard, application par application**, avec une simple webcam.
Et à terme, piloter son PC avec le visage (clic par clin d'œil, curseur qui suit le regard…).

Tout fonctionne **en local** : aucune image ne quitte votre PC et rien n'est enregistré
en dehors des coordonnées du regard (dans `~/.regard/regard.db`).

## Ce que ça fait

- **Calibration** (≈ 30 s) : vous fixez 15 points, Regard apprend la correspondance entre
  vos yeux et l'écran, puis mesure sa propre précision sur 5 points de vérification.
- **Suivi en direct** : une petite fenêtre affiche l'application regardée (« chrome »,
  « Code », « Discord »…) et le titre de la fenêtre ; un point rouge montre le regard estimé.
- **Rapport** : temps passé à regarder chaque application, carte de chaleur sur l'écran
  entier et **à l'intérieur de chaque fenêtre**, titres les plus regardés.
- **Gestes du visage** (expérimental) : clin d'œil gauche/droit, yeux fermés longtemps,
  bouche ouverte, sourcils levés, sourire → clic, clic droit, défilement, curseur qui suit
  le regard. Désactivés par défaut.

## Installation (Windows)

1. Installez [Python 3.12](https://www.python.org/downloads/) en cochant
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
| `python regard.py report --days 7` | Rapport HTML des 7 derniers jours |

Options communes : `--camera 1` (autre webcam), `--res 1920x1080`.

## Précision : à quoi s'attendre

Une webcam n'est pas un eye-tracker infrarouge (Tobii…). Après une bonne calibration,
comptez une erreur de **2 à 4 cm** sur l'écran : largement assez pour savoir **quelle
application ou quelle zone** vous regardez, pas assez pour viser un petit bouton.

Pour de meilleurs résultats :
- visage bien éclairé de face (pas de fenêtre lumineuse derrière vous) ;
- webcam au-dessus de l'écran, à hauteur des yeux ;
- garder à peu près la même position qu'à la calibration — si vous bougez beaucoup
  ou changez de place, **recalibrez** (option 2 de `Regard.bat`) ;
- les lunettes fonctionnent, mais les reflets dégradent la précision.

## Gestes du visage et actions

Activez-les avec la case « Actions par gestes du visage » ou `--actions`. Par défaut :

| Geste | Action |
|---|---|
| Clin d'œil gauche | clic gauche (à l'endroit du regard) |
| Clin d'œil droit | clic droit |
| Yeux fermés ~1 s | activer / désactiver les actions (interrupteur de sécurité) |
| Bouche ouverte | le curseur suit le regard (on / off) |
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
- `regard/calibration.py` — écran de calibration et de vérification ;
- `regard/windows.py` — application sous un point : Windows (Win32, y compris applis UWP),
  macOS (Quartz) et Linux X11 ;
- `regard/tracker.py` — boucle de suivi, point du regard, fenêtre de contrôle ;
- `regard/gestures.py` — détection des gestes et actions souris ;
- `regard/store.py`, `regard/report.py` — enregistrement et rapport.

## Limites connues / pistes

- Calibration sur l'écran principal uniquement (pas encore de multi-écrans).
- Les mouvements de tête importants dégradent la précision : une calibration « tête
  mobile » ou une correction continue (ex. : quand vous cliquez, vous regardez le curseur)
  amélioreraient beaucoup les choses.
- macOS : autoriser la webcam et « Enregistrement de l'écran » (pour les titres de fenêtres).
- Linux : X11 uniquement (Wayland ne permet pas de savoir quelle fenêtre est sous un point).
