#!/usr/bin/env python3
"""Regard — savoir où se pose votre regard, application par application.

Commandes :
  python regard.py preview      vérifier la webcam, les yeux et les gestes du visage
  python regard.py calibrate    calibrer le regard (≈ 30 s)
  python regard.py track        suivre le regard en direct et l'enregistrer
  python regard.py report       générer le rapport HTML
"""
from __future__ import annotations

import argparse
import sys
import time
import webbrowser
from pathlib import Path


def cmd_preview(args):
    import cv2

    from regard.face import EYES, FaceAnalyzer, open_camera
    from regard.gestures import GestureDetector, load_config

    analyzer = FaceAnalyzer()
    cap = open_camera(args.camera, *args.res)
    detector = GestureDetector(load_config()[0])
    events: list[tuple[float, str]] = []
    print("Aperçu : Q ou Échap pour quitter.")
    last, fps = time.monotonic(), 0.0
    while True:
        ok, bgr = cap.read()
        if not ok:
            print("Impossible de lire la webcam.")
            break
        t = time.monotonic()
        fps = 0.9 * fps + 0.1 / max(t - last, 1e-3)
        last = t
        face = analyzer.process(bgr, t)
        view = bgr.copy()
        lines = [f"{fps:4.1f} img/s"]
        if face is None:
            lines.append("aucun visage")
        else:
            for eye in EYES.values():
                for i in (eye["a"], eye["b"], eye["up"], eye["low"]):
                    cv2.circle(view, tuple(face.landmarks[i].astype(int)), 2, (0, 200, 255), -1)
                c = face.landmarks[eye["iris"]].mean(axis=0).astype(int)
                cv2.circle(view, tuple(c), 3, (0, 0, 255), -1)
            f = face.features
            lines += [f"iris x {f[0]:+.3f}  y {f[1]:+.3f}  ouverture {f[2]:.2f}",
                      f"tete lacet {f[3]:+.0f}  tangage {f[4]:+.0f}  roulis {f[5]:+.0f}  dist {abs(f[8]):.0f} cm"]
            keys = ["eyeBlinkLeft", "eyeBlinkRight", "jawOpen", "browInnerUp",
                    "mouthSmileLeft", "mouthSmileRight"]
            lines += [f"{k:16s} {'#' * int(face.blend.get(k, 0) * 20):20s} {face.blend.get(k, 0):.2f}"
                      for k in keys]
            for g in detector.update(face.blend, t):
                events.append((t, g))
        if face is None:
            detector.update(None, t)
        lines += [f"> {g}" for ts, g in events if t - ts < 3]
        view = cv2.flip(view, 1)       # effet miroir pour l'affichage seulement
        for i, s in enumerate(lines):
            cv2.putText(view, s, (12, 28 + 24 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 4)
            cv2.putText(view, s, (12, 28 + 24 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        cv2.imshow("Regard - apercu", view)
        if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
            break
    cap.release()
    cv2.destroyAllWindows()
    analyzer.close()


def _start_source(args):
    from regard.face import FaceSource
    source = FaceSource(args.camera, *args.res)
    source.start()
    source.ready.wait(30)
    if source.error:
        sys.exit(f"Erreur : {source.error}")
    return source


def _calibrate(source, args):
    from regard.calibration import calibrate
    rows, cols = args.grid
    model = calibrate(source, rows, cols, head=not args.no_head)
    if model is None:
        print("Calibration annulée.")
    else:
        err = f"{model.validation_error:.0f} px" if model.validation_error else "?"
        print(f"Calibration enregistrée (erreur de validation : {err}).")
    return model


def cmd_calibrate(args):
    source = _start_source(args)
    _calibrate(source, args)
    source.stop()


def cmd_track(args):
    import tkinter as tk

    from regard.calibration import calibration_path
    from regard.model import GazeModel
    from regard.tracker import GazeTracker, StatusWindow

    source = _start_source(args)
    path = calibration_path()
    model = None
    if path.exists() and not args.recalibrate:
        model = GazeModel.load(path)
        probe = tk.Tk()
        screen = (probe.winfo_screenwidth(), probe.winfo_screenheight())
        probe.destroy()
        if tuple(model.screen) != screen:
            print(f"L'écran a changé ({model.screen} → {screen}) : nouvelle calibration.")
            model = None
    if model is None:
        model = _calibrate(source, args)
        if model is None:
            source.stop()
            return
    tracker = GazeTracker(model, model.screen, use_actions=args.actions,
                          record=not args.no_record, autocorrect=not args.no_autocorrect)
    source.on_frame = tracker.on_frame
    print("Suivi en cours. Fermez la petite fenêtre « Regard » pour arrêter.")
    try:
        StatusWindow(source, tracker, show_dot=not args.no_dot).run()
    except KeyboardInterrupt:
        pass
    finally:
        source.on_frame = None
        source.stop()
        time.sleep(0.1)
        tracker.close()
    print("Données enregistrées. Rapport : python regard.py report")


def cmd_report(args):
    from regard.report import build_report
    out = build_report(args.days, Path(args.out))
    print(f"Rapport écrit : {out.resolve()}")
    if not args.no_open:
        webbrowser.open(out.resolve().as_uri())


def _pair(sep):
    def parse(s):
        a, b = s.lower().split(sep)
        return int(a), int(b)
    return parse


def main():
    from regard.windows import enable_dpi_awareness
    enable_dpi_awareness()

    cam = argparse.ArgumentParser(add_help=False)
    cam.add_argument("--camera", type=int, default=0, help="numéro de la webcam (défaut 0)")
    cam.add_argument("--res", type=_pair("x"), default=(1280, 720), help="résolution, ex. 1280x720")
    p = argparse.ArgumentParser(description="Regard — suivi du regard par application")
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("preview", parents=[cam], help="vérifier la webcam, les yeux et les gestes"
                   ).set_defaults(fn=cmd_preview)
    c = sub.add_parser("calibrate", parents=[cam], help="calibrer le regard")
    c.add_argument("--grid", type=_pair("x"), default=(3, 5), help="lignes x colonnes (défaut 3x5)")
    c.add_argument("--no-head", action="store_true", help="sans la phase « tête mobile »")
    c.set_defaults(fn=cmd_calibrate)
    t = sub.add_parser("track", parents=[cam], help="suivre le regard en direct")
    t.add_argument("--grid", type=_pair("x"), default=(3, 5), help=argparse.SUPPRESS)
    t.add_argument("--no-head", action="store_true", help=argparse.SUPPRESS)
    t.add_argument("--recalibrate", action="store_true", help="refaire la calibration avant")
    t.add_argument("--no-autocorrect", action="store_true",
                   help="ne pas corriger le regard à partir des clics")
    t.add_argument("--no-dot", action="store_true", help="ne pas afficher le point du regard")
    t.add_argument("--no-record", action="store_true", help="ne rien enregistrer")
    t.add_argument("--actions", action="store_true", help="activer les actions par gestes du visage")
    t.set_defaults(fn=cmd_track)
    r = sub.add_parser("report", help="générer le rapport HTML")
    r.add_argument("--days", type=float, default=1, help="période en jours (défaut 1)")
    r.add_argument("--out", default="rapport-regard.html")
    r.add_argument("--no-open", action="store_true", help="ne pas ouvrir le navigateur")
    r.set_defaults(fn=cmd_report)
    args = p.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
