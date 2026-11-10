"""Rapport HTML : temps de regard par application + cartes de chaleur."""
from __future__ import annotations

import html
import json
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

from .store import ABSENT, OFFSCREEN, connect

GRID_APP = (48, 30)
GRID_SCREEN = (64, 36)


def _blur(g: np.ndarray, passes: int = 2) -> np.ndarray:
    k = np.array([1, 4, 6, 4, 1], dtype=float) / 16
    for _ in range(passes):
        g = np.apply_along_axis(lambda r: np.convolve(r, k, mode="same"), 1, g)
        g = np.apply_along_axis(lambda c: np.convolve(c, k, mode="same"), 0, g)
    return g


def _heat(xs, ys, ws, grid) -> list:
    gx, gy = grid
    g = np.zeros((gy, gx))
    if len(xs):
        ix = np.clip((np.asarray(xs) * gx).astype(int), 0, gx - 1)
        iy = np.clip((np.asarray(ys) * gy).astype(int), 0, gy - 1)
        np.add.at(g, (iy, ix), ws)
        g = _blur(g)
        if g.max() > 0:
            g = g / g.max()
    return np.round(g, 3).tolist()


def fmt_duration(s: float) -> str:
    s = int(round(s))
    h, m, sec = s // 3600, (s % 3600) // 60, s % 60
    return f"{h} h {m:02d} min" if h else (f"{m} min {sec:02d} s" if m else f"{sec} s")


def build_report(days: float = 1.0, out: Path | None = None) -> Path:
    con = connect()
    since = time.time() - days * 86400
    rows = con.execute(
        "SELECT dt, x, y, app, title, rel_x, rel_y, win_w, win_h, session FROM samples "
        "WHERE ts >= ? ORDER BY ts", (since,)).fetchall()
    screens = dict((sid, (w, h)) for sid, w, h in
                   con.execute("SELECT id, screen_w, screen_h FROM sessions"))
    con.close()

    total = defaultdict(float)
    titles = defaultdict(lambda: defaultdict(float))
    per_app = defaultdict(lambda: ([], [], [], [], []))   # rel_x, rel_y, w, win_w, win_h
    sx, sy, sw = [], [], []
    for dt, x, y, app, title, rx, ry, ww, wh, sid in rows:
        dt = dt or 0.0
        total[app] += dt
        if app in (ABSENT, OFFSCREEN) or x is None:
            continue
        if title:
            titles[app][title] += dt
        scr = screens.get(sid) or (1920, 1080)
        sx.append(x / scr[0]); sy.append(y / scr[1]); sw.append(dt)
        if rx is not None and 0 <= rx <= 1 and 0 <= ry <= 1:
            a = per_app[app]
            a[0].append(rx); a[1].append(ry); a[2].append(dt); a[3].append(ww); a[4].append(wh)

    looked = {a: t for a, t in total.items() if a not in (ABSENT, OFFSCREEN)}
    grand = sum(total.values()) or 1.0
    looked_total = sum(looked.values()) or 1.0
    apps = sorted(looked.items(), key=lambda kv: -kv[1])

    app_data = []
    for app, t in apps[:12]:
        rx, ry, w, ww, wh = per_app.get(app, ([], [], [], [], []))
        aspect = (float(np.median(ww)) / float(np.median(wh))) if ww and np.median(wh) else 16 / 9
        top_titles = sorted(titles[app].items(), key=lambda kv: -kv[1])[:5]
        app_data.append(dict(name=app, seconds=t, pct=100 * t / looked_total,
                             aspect=max(0.4, min(aspect, 4.0)), heat=_heat(rx, ry, w, GRID_APP),
                             titles=[(ti, fmt_duration(s)) for ti, s in top_titles]))
    data = dict(
        apps=app_data,
        screen=_heat(sx, sy, sw, GRID_SCREEN),
        absent=total.get(ABSENT, 0.0), offscreen=total.get(OFFSCREEN, 0.0),
        looked=sum(looked.values()), grand=grand,
    )
    out = Path(out or "rapport-regard.html")
    out.write_text(_render(data, days), encoding="utf-8")
    return out


def _render(d: dict, days: float) -> str:
    period = "dernières 24 h" if days == 1 else f"{days:g} derniers jours"
    rows = []
    for a in d["apps"]:
        rows.append(
            f"<tr><td class=app>{html.escape(a['name'])}</td><td class=num>{fmt_duration(a['seconds'])}</td>"
            f"<td class=bar><span style='width:{a['pct']:.1f}%'></span></td>"
            f"<td class=num>{a['pct']:.0f} %</td></tr>")
    cards = []
    for i, a in enumerate(d["apps"]):
        t = "".join(f"<li><span>{html.escape(ti[:90])}</span><em>{dur}</em></li>" for ti, dur in a["titles"])
        cards.append(
            f"<section class=card><h3>{html.escape(a['name'])} <small>{fmt_duration(a['seconds'])}</small></h3>"
            f"<div class=frame style='aspect-ratio:{a['aspect']:.3f}'><canvas data-heat='app{i}'></canvas></div>"
            f"<ul>{t}</ul></section>")
    heat_json = json.dumps({"screen": d["screen"], **{f"app{i}": a["heat"] for i, a in enumerate(d["apps"])}})
    empty = "" if d["apps"] else "<p class=empty>Aucune donnée sur la période. Lancez <code>python regard.py track</code>.</p>"
    return f"""<!doctype html>
<html lang="fr"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Rapport Regard</title>
<style>
:root{{--bg:#f7f7f5;--card:#fff;--fg:#1d1f21;--mute:#6b6f76;--line:#e3e3df;--accent:#d9483b}}
@media (prefers-color-scheme:dark){{:root{{--bg:#151719;--card:#1e2124;--fg:#e8eaed;--mute:#9aa0a6;--line:#30343a;--accent:#ff6b5e}}}}
*{{box-sizing:border-box}}body{{margin:0;background:var(--bg);color:var(--fg);font:15px/1.5 system-ui,-apple-system,"Segoe UI",sans-serif}}
main{{max-width:1100px;margin:0 auto;padding:32px 16px 64px}}h1{{margin:0 0 4px;font-size:28px}}
.sub{{color:var(--mute);margin:0 0 28px}}.kpis{{display:grid;grid-template-columns:repeat(auto-fit,minmax(180px,1fr));gap:12px;margin-bottom:28px}}
.kpi{{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:14px 16px}}.kpi b{{display:block;font-size:22px}}.kpi span{{color:var(--mute);font-size:13px}}
table{{width:100%;border-collapse:collapse;background:var(--card);border:1px solid var(--line);border-radius:10px;overflow:hidden}}
td{{padding:9px 14px;border-top:1px solid var(--line)}}tr:first-child td{{border-top:0}}.num{{text-align:right;white-space:nowrap;font-variant-numeric:tabular-nums}}
.bar{{width:45%}}.bar span{{display:block;height:8px;border-radius:4px;background:var(--accent)}}.app{{font-weight:600}}
h2{{margin:36px 0 12px;font-size:19px}}.grid{{display:grid;grid-template-columns:repeat(auto-fill,minmax(320px,1fr));gap:14px;align-items:start}}
.card{{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:14px}}.card h3{{margin:0 0 10px;font-size:16px}}.card small{{color:var(--mute);font-weight:400}}
.frame{{width:100%;border:1px solid var(--line);border-radius:6px;overflow:hidden;background:repeating-linear-gradient(45deg,transparent 0 8px,rgba(127,127,127,.06) 8px 16px)}}
.frame canvas{{width:100%;height:100%;display:block}}.screen{{aspect-ratio:16/9;max-width:760px}}
ul{{list-style:none;margin:10px 0 0;padding:0;font-size:13px}}li{{display:flex;justify-content:space-between;gap:8px;padding:3px 0;color:var(--mute)}}
li span{{overflow:hidden;text-overflow:ellipsis;white-space:nowrap}}li em{{font-style:normal;white-space:nowrap}}.empty{{color:var(--mute)}}
</style></head><body><main>
<h1>Où se pose votre regard</h1><p class=sub>Période : {period}</p>
<div class=kpis>
<div class=kpi><b>{fmt_duration(d['looked'])}</b><span>regard sur l'écran</span></div>
<div class=kpi><b>{len(d['apps'])}</b><span>applications regardées</span></div>
<div class=kpi><b>{fmt_duration(d['offscreen'])}</b><span>regard hors de l'écran</span></div>
<div class=kpi><b>{fmt_duration(d['absent'])}</b><span>visage non détecté</span></div>
</div>
{empty}
<table>{''.join(rows)}</table>
<h2>Sur l'écran entier</h2>
<div class="frame screen"><canvas data-heat="screen"></canvas></div>
<h2>Dans chaque application</h2>
<p class=sub>Position du regard à l'intérieur de la fenêtre (haut = haut de la fenêtre).</p>
<div class=grid>{''.join(cards)}</div>
</main>
<script>
const HEAT={heat_json};
function color(v){{const s=[[0,[40,90,255,0]],[.25,[40,140,255,.45]],[.5,[60,220,120,.6]],[.75,[255,210,40,.75]],[1,[235,50,40,.85]]];
for(let i=1;i<s.length;i++)if(v<=s[i][0]){{const[a,ca]=s[i-1],[b,cb]=s[i],t=(v-a)/(b-a);return ca.map((c,k)=>c+(cb[k]-c)*t)}}return s[4][1]}}
document.querySelectorAll('canvas[data-heat]').forEach(cv=>{{const g=HEAT[cv.dataset.heat];if(!g||!g.length)return;
const h=g.length,w=g[0].length,small=document.createElement('canvas');small.width=w;small.height=h;
const sx=small.getContext('2d'),img=sx.createImageData(w,h);
for(let y=0;y<h;y++)for(let x=0;x<w;x++){{const c=color(g[y][x]),i=(y*w+x)*4;img.data[i]=c[0];img.data[i+1]=c[1];img.data[i+2]=c[2];img.data[i+3]=c[3]*255}}
sx.putImageData(img,0,0);const r=cv.getBoundingClientRect();cv.width=Math.max(1,r.width*2);cv.height=Math.max(1,r.height*2);
const ctx=cv.getContext('2d');ctx.imageSmoothingEnabled=true;ctx.imageSmoothingQuality='high';ctx.drawImage(small,0,0,cv.width,cv.height)}});
</script></body></html>"""
