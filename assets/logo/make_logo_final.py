"""Logo 01 with "RadMaps" set on an arc just below the globe (two curvatures)."""
import math, os, subprocess, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from make_globe_logos import globe, svg, W_BOLD, FONT, NAVY, DKGREEN

OUT = os.path.dirname(os.path.abspath(__file__))


def arc_word(cx, cy, radius, size, spacing=1.5):
    """'RadMaps' on the bottom of a circle (reading left to right), Rad navy and Maps green.
    *radius* is the baseline: letters rise from it towards the centre."""
    s = "RadMaps"
    cols = [NAVY] * 3 + [DKGREEN] * 4
    widths = [W_BOLD.get(c, 600) * size / 1000 + spacing for c in s]
    total = sum(widths) - spacing
    pos = -total / radius / 2
    out = []
    for c, w, col in zip(s, widths, cols):
        mid = pos + (w - spacing) / 2 / radius
        a = math.pi / 2 - mid                      # bottom of the circle, left to right
        x, y = cx + radius * math.cos(a), cy + radius * math.sin(a)
        out.append(f'<text x="{x:.2f}" y="{y:.2f}" font-family="{FONT}" font-weight="700" font-size="{size:.2f}" '
                   f'fill="{col}" text-anchor="middle" '
                   f'transform="rotate({-math.degrees(mid):.2f} {x:.2f} {y:.2f})">{c}</text>')
        pos += w / radius
    return "".join(out)


def logo(curve):
    cx, cy, R, size = 150, 110, 100, 50
    rim = R + 3.5
    r_text = rim + 10 + size * 0.72      # baseline: cap height (0.72 em) plus a small gap below the rim
    body = globe(cx, cy, R)
    if curve == "concentric":           # follows the globe's own circle
        body += arc_word(cx, cy, r_text, size)
    else:                               # gentler arc: larger circle, same lowest point
        r2 = 2.4 * r_text
        body += arc_word(cx, cy + r_text - r2, r2, size)
    return svg(body, 300, 305, "RadMaps")


for name in ():  # the stacked variants (logo 01) are no longer generated
    p = os.path.join(OUT, f"logo_final_{name}.svg")
    open(p, "w").write(logo(name))
    for w in (300, 48):
        subprocess.run(["rsvg-convert", "-w", str(w), "-o", os.path.join(OUT, "png", f"logo_final_{name}_{w}.png"), p],
                       check=True)
print("ok")


def horizontal():
    """Logo 08 (globe left, wordmark and version right), with Rad in navy and Maps in logo 10's gold."""
    from make_globe_logos import word, text_width, YELLOW
    body = globe(80, 80, 68, cell_frac=0.2)
    body += word(170, 104, 74, rad=NAVY, maps=YELLOW)
    vx = 170 + text_width("RadMaps", 74) + 14
    body += (f'<text x="{vx:.2f}" y="104" font-family="{FONT}" font-weight="400" '
             f'font-size="28" fill="#8a93a6">v1.1</text>')
    return svg(body, 600, 160, "RadMaps v1.1")


p = os.path.join(OUT, "radmaps_logo.svg")
open(p, "w").write(horizontal())
subprocess.run(["rsvg-convert", "-w", "900", "-o", os.path.join(OUT, "radmaps_logo.png"), p], check=True)


def favicon():
    """The logo's globe on its own, for the browser tab."""
    return svg(globe(80, 80, 72, cell_frac=0.2), 160, 160, "RadMaps")


p = os.path.join(OUT, "radmaps_favicon.svg")
open(p, "w").write(favicon())
subprocess.run(["rsvg-convert", "-w", "64", "-o", os.path.join(OUT, "radmaps_favicon.png"), p], check=True)
