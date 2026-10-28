"""RadMaps hex-globe lockups (v2): 10 SVG variations + favicon + contact sheet."""
import math, os, subprocess
from PIL import Image, ImageDraw, ImageFont

OUT = os.path.dirname(os.path.abspath(__file__))
RED, ORANGE, YELLOW, GREEN, DKGREEN = "#dc3232", "#e6781e", "#f0c81e", "#50be3c", "#1e781e"
NAVY, PURPLE, WHITE = "#193782", "#8032c8", "#ffffff"
SEA = "#2c4a9a"          # lighter navy tint for sea cells
DARKBG = "#0e1117"       # Streamlit dark theme background
FONT = "Helvetica, Arial, sans-serif"
S3 = math.sqrt(3)

# Helvetica advance widths (per 1000 em) for layout / arc text
W_BOLD = {"R": 722, "a": 556, "d": 611, "M": 833, "p": 611, "s": 556, " ": 278,
          "v": 556, "1": 556, ".": 278, "A": 722, "C": 722, "E": 667, "S": 667,
          "T": 611, "O": 778, "I": 278, "H": 722, "P": 667, "Y": 667, "G": 778,
          "L": 611, "D": 722, "B": 722, "N": 722, "U": 722, "X": 667, "·": 278}


def text_width(s, size):
    return sum(W_BOLD.get(c, 600) for c in s) * size / 1000


def hex_pts(cx, cy, r, flat=False):
    off = 0 if flat else 30
    return [(cx + r * math.cos(math.radians(60 * k + off)),
             cy + r * math.sin(math.radians(60 * k + off))) for k in range(6)]


def hexagon(cx, cy, r, fill, flat=False, extra=""):
    p = " ".join(f"{x:.2f},{y:.2f}" for x, y in hex_pts(cx, cy, r, flat))
    return f'<polygon points="{p}" fill="{fill}" {extra}/>'


def svg(body, w, h, title, bg=None):
    b = f'<rect width="{w}" height="{h}" fill="{bg}"/>' if bg else ""
    return (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w:.0f} {h:.0f}" '
            f'width="{w:.0f}" height="{h:.0f}" role="img" aria-label="{title}">'
            f'<title>{title}</title>{b}{body}</svg>\n')


# Abstract "continents" in unit-disc coordinates (x, y, radius)
BLOBS = [(-0.47, -0.35, 0.37), (-0.29, 0.29, 0.30), (0.35, -0.41, 0.35),
         (0.52, 0.12, 0.28), (0.23, 0.52, 0.21)]
# Notional radiotherapy centres (unit-disc coords) for the access gradient
CENTRES = [(-0.45, -0.38), (0.40, -0.40)]

_uid = [0]


def globe(cx, cy, R, cell_frac=0.175, land="diag", sea=SEA, rim=NAVY,
          rim_w=None, gap_frac=0.10, marker=False):
    """Disc of pointy-top hex cells clipped to a circle.

    land: 'diag'   original dark-green -> orange diagonal
          'access' green near notional centres -> red far away
          'white'  white land
          'green'  two-tone green
    """
    _uid[0] += 1
    cid = f"g{_uid[0]}"
    size = R * cell_frac
    gap = size * gap_frac
    rim_w = R * 0.07 if rim_w is None else rim_w
    out = [f'<defs><clipPath id="{cid}"><circle cx="{cx:.2f}" cy="{cy:.2f}" r="{R:.2f}"/></clipPath></defs>',
           f'<circle cx="{cx:.2f}" cy="{cy:.2f}" r="{R + rim_w / 2:.2f}" fill="{rim}"/>',
           f'<g clip-path="url(#{cid})">']
    n = int(R / size) + 3
    for q in range(-n, n + 1):
        for r_ in range(-n, n + 1):
            x = size * S3 * (q + r_ / 2); y = size * 1.5 * r_
            if math.hypot(x, y) > R + size:
                continue
            ux, uy = x / R, y / R
            is_land = any(math.hypot(ux - bx, uy - by) < br for bx, by, br in BLOBS)
            if not is_land:
                col = sea
            elif land == "diag":
                t = (ux + uy) / 2
                col = DKGREEN if t < -0.25 else GREEN if t < 0.05 else YELLOW if t < 0.25 else ORANGE
            elif land == "access":
                d = min(math.hypot(ux - a, uy - b) for a, b in CENTRES)
                col = (DKGREEN if d < 0.2 else GREEN if d < 0.42 else YELLOW if d < 0.62
                       else ORANGE if d < 0.8 else RED)
            elif land == "white":
                col = WHITE
            else:  # green
                col = DKGREEN if (ux + uy) < -0.1 else GREEN
            out.append(hexagon(cx + x, cy + y, size - gap, col))
    out.append("</g>")
    out.append(f'<circle cx="{cx:.2f}" cy="{cy:.2f}" r="{R:.2f}" fill="none" stroke="{rim}" stroke-width="{rim_w:.2f}"/>')
    if marker:
        for a, b in CENTRES:
            out.append(hexagon(cx + a * R, cy + b * R, size * 0.55, PURPLE, extra=f'stroke="{WHITE}" stroke-width="{size*0.18:.2f}"'))
    return "".join(out)


def word(x, y, size, rad=NAVY, maps=DKGREEN, anchor="start", weight=700, extra=""):
    """'RadMaps' with two-tone Rad/Maps."""
    return (f'<text x="{x:.2f}" y="{y:.2f}" font-family="{FONT}" font-weight="{weight}" '
            f'font-size="{size:.2f}" text-anchor="{anchor}" letter-spacing="0" {extra}>'
            f'<tspan fill="{rad}">Rad</tspan><tspan fill="{maps}">Maps</tspan></text>')


def arc_text(s, cx, cy, radius, size, fill, top=True, spacing=0.0):
    """Place characters individually along a circle (no textPath dependency)."""
    widths = [W_BOLD.get(c, 600) * size / 1000 + spacing for c in s]
    total = sum(widths) - spacing
    ang_total = total / radius
    out = []
    pos = -ang_total / 2
    for c, w in zip(s, widths):
        mid = pos + (w - spacing) / 2 / radius
        if top:
            a = -math.pi / 2 + mid
            x, y = cx + radius * math.cos(a), cy + radius * math.sin(a)
            rot = math.degrees(mid)
        else:
            a = math.pi / 2 - mid
            x, y = cx + radius * math.cos(a), cy + radius * math.sin(a)
            rot = -math.degrees(mid)
        out.append(f'<text x="{x:.2f}" y="{y:.2f}" font-family="{FONT}" font-weight="700" '
                   f'font-size="{size:.2f}" fill="{fill}" text-anchor="middle" '
                   f'dominant-baseline="central" transform="rotate({rot:.2f} {x:.2f} {y:.2f})">{c}</text>')
        pos += w / radius
    return "".join(out)


# ------------------------------------------------------------------ logos
def g01():
    """Stacked lockup: globe above, wordmark below."""
    W, H = 300, 330
    body = globe(150, 118, 100)
    body += word(150, 300, 58, anchor="middle")
    return svg(body, W, H, "RadMaps - stacked")


def g02():
    """Horizontal lockup for the sidebar header."""
    W, H = 560, 160
    body = globe(80, 80, 68, cell_frac=0.2)
    body += word(170, 104, 74)
    return svg(body, W, H, "RadMaps - horizontal")


def g03():
    """Seal: wordmark on top arc, tagline on bottom arc, around a ring."""
    W = H = 320; c = 160
    body = f'<circle cx="{c}" cy="{c}" r="156" fill="{NAVY}"/>'
    body += f'<circle cx="{c}" cy="{c}" r="152" fill="none" stroke="{WHITE}" stroke-width="2" stroke-opacity="0.5"/>'
    body += globe(c, c, 92, rim=WHITE, rim_w=4, sea=SEA)
    body += arc_text("RadMaps", c, c, 122, 38, WHITE, top=True, spacing=3)
    body += arc_text("RADIOTHERAPY ACCESS", c, c, 124, 17, YELLOW, top=False, spacing=2.2)
    for side in (-1, 1):
        x = c + side * 123
        body += hexagon(x, c, 6, GREEN)
    return svg(body, W, H, "RadMaps - seal")


def g04():
    """Globe replaces the 'a' of 'Rad': globe = bowl, navy bar = stem of a single-storey 'a'."""
    fs = 110; base = 118
    xh = fs * 0.532                      # Helvetica Bold x-height
    R_ = xh / 2
    x = 16
    body = f'<text x="{x}" y="{base}" font-family="{FONT}" font-weight="700" font-size="{fs}" fill="{NAVY}">R</text>'
    x += W_BOLD["R"] * fs / 1000 + 3
    gcx, gcy = x + R_, base - R_
    body += globe(gcx, gcy, R_ - 2, cell_frac=0.3, rim_w=5, gap_frac=0.1)
    stem_w = 15
    sx = gcx + R_ - 4
    body += f'<rect x="{sx:.2f}" y="{base - xh:.2f}" width="{stem_w}" height="{xh:.2f}" fill="{NAVY}"/>'
    x = sx + stem_w + 6
    body += (f'<text x="{x:.2f}" y="{base}" font-family="{FONT}" font-weight="700" font-size="{fs}">'
             f'<tspan fill="{NAVY}">d</tspan><tspan fill="{DKGREEN}">Maps</tspan></text>')
    W = x + text_width("dMaps", fs) + 16
    return svg(body, W, 150, "RadMaps - globe as a")


def g05():
    """Clean globe: few large cells (survives small sizes), stacked."""
    W, H = 300, 330
    body = globe(150, 118, 100, cell_frac=0.29, land="green", gap_frac=0.09, rim_w=9)
    body += word(150, 300, 58, anchor="middle")
    return svg(body, W, H, "RadMaps - clean globe")


def g06():
    """Access-ratio land (green near centres -> red far), purple centre markers, horizontal."""
    W, H = 560, 160
    body = globe(80, 80, 68, cell_frac=0.16, land="access", marker=True)
    body += word(170, 104, 74)
    return svg(body, W, H, "RadMaps - access gradient")


def g07():
    """White land on navy sea; monochrome navy wordmark, stacked."""
    W, H = 300, 330
    body = globe(150, 118, 100, cell_frac=0.2, land="white", sea=NAVY, rim=NAVY, gap_frac=0.14)
    body += word(150, 300, 58, rad=NAVY, maps=NAVY, anchor="middle")
    return svg(body, W, H, "RadMaps - white land")


def g08():
    """Horizontal lockup with a subtle version tag."""
    W, H = 600, 160
    body = globe(80, 80, 68, cell_frac=0.2)
    body += word(170, 104, 74)
    vx = 170 + text_width("RadMaps", 74) + 14
    body += (f'<text x="{vx:.2f}" y="104" font-family="{FONT}" font-weight="400" '
             f'font-size="28" fill="#8a93a6">v1.1</text>')
    return svg(body, W, H, "RadMaps v1.1")


def g09():
    """Dark-background lockup (Streamlit dark theme)."""
    W, H = 560, 160
    body = globe(80, 80, 66, cell_frac=0.2, sea="#24408f", rim="#3b5bb5", rim_w=4)
    body += word(170, 104, 74, rad=WHITE, maps=GREEN)
    return svg(body, W, H, "RadMaps - dark", bg=DARKBG)


def g10():
    """Banner: navy band across the lower globe carrying the wordmark."""
    W, H = 340, 300
    c, R = 170, 128
    body = globe(c, 142, R, cell_frac=0.16, land="access")
    bw, bh, by = 310, 66, 196
    body += f'<rect x="{c - bw/2}" y="{by}" width="{bw}" height="{bh}" rx="10" fill="{NAVY}"/>'
    body += word(c, by + 48, 50, rad=WHITE, maps=YELLOW, anchor="middle")
    return svg(body, W, H, "RadMaps - banner")


def favicon():
    """Globe-only favicon: few large cells, thick rim."""
    return svg(globe(50, 50, 44, cell_frac=0.29, land="green", gap_frac=0.09, rim_w=6),
               100, 100, "RadMaps")


LOGOS = [
    (g01, "Stacked lockup"), (g02, "Horizontal lockup"), (g03, "Seal / badge"),
    (g04, "Globe as the 'a'"), (g05, "Clean globe, stacked"), (g06, "Access-gradient land"),
    (g07, "White land on navy"), (g08, "Horizontal + v1.1"), (g09, "Dark background"),
    (g10, "Banner across globe"),
]


def render(svg_path, png_path, w=None, h=None):
    args = ["rsvg-convert"]
    if w: args += ["-w", str(w)]
    if h: args += ["-h", str(h)]
    subprocess.run(args + ["-o", png_path, svg_path], check=True)
    return Image.open(png_path).convert("RGBA")


def main():
    pngdir = os.path.join(OUT, "png"); os.makedirs(pngdir, exist_ok=True)
    tiles = []
    for i, (fn, name) in enumerate(LOGOS, 1):
        p = os.path.join(OUT, f"logo_globe_{i:02d}.svg")
        with open(p, "w") as f:
            f.write(fn())
        big = render(p, os.path.join(pngdir, f"logo_globe_{i:02d}_300w.png"), w=300)
        small = render(p, os.path.join(pngdir, f"logo_globe_{i:02d}_40h.png"), h=40)
        tiles.append((i, name, big, small))
    fp = os.path.join(OUT, "favicon_globe.svg")
    with open(fp, "w") as f:
        f.write(favicon())
    fav = {s: render(fp, os.path.join(pngdir, f"favicon_globe_{s}.png"), w=s) for s in (16, 32, 64)}
    orig = os.path.join(OUT, "..", "logo_10.svg")
    ofav = render(orig, os.path.join(pngdir, "orig_logo_10_32.png"), w=32) if os.path.exists(orig) else None

    font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 22)
    sfont = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 14)
    TW, TH = 680, 400
    sheet = Image.new("RGB", (TW * 2, TH * 5 + 60 + 150), "#f0f2f6")
    d = ImageDraw.Draw(sheet)
    d.text((20, 18), "RadMaps hex-globe lockups  -  300 px wide  |  40 px high on Streamlit sidebar (light #f0f2f6 and dark #262730)",
           fill="#333", font=font)
    for idx, (i, name, big, small) in enumerate(tiles):
        x0 = (idx % 2) * TW; y0 = 60 + (idx // 2) * TH
        d.rectangle([x0 + 8, y0 + 4, x0 + TW - 8, y0 + TH - 10], fill="white", outline="#d0d4dc")
        d.text((x0 + 22, y0 + 14), f"{i:02d}  {name}", fill="#111", font=font)
        sheet.paste(big, (x0 + 22, y0 + 50), big)
        sx = x0 + 350
        d.text((sx, y0 + 52), "sidebar size (40 px high):", fill="#666", font=sfont)
        d.rectangle([sx, y0 + 76, sx + 300, y0 + 136], fill="#f0f2f6")
        sheet.paste(small, (sx + 10, y0 + 86), small)
        d.rectangle([sx, y0 + 146, sx + 300, y0 + 206], fill="#262730")
        sheet.paste(small, (sx + 10, y0 + 156), small)
    # favicon strip
    y0 = 60 + 5 * TH
    d.rectangle([8, y0 + 4, TW * 2 - 8, y0 + 140], fill="white", outline="#d0d4dc")
    d.text((22, y0 + 14), "Favicon candidates: new favicon_globe.svg at 16 / 32 / 64 px (and 32 px zoomed x4)   vs   original logo_10 at 32 px (and x4)",
           fill="#111", font=sfont)
    x = 22
    for s in (16, 32, 64):
        im = fav[s]; sheet.paste(im, (x, y0 + 100 - im.height // 2 + 0), im); x += s + 24
    z = fav[32].resize((128, 128), Image.NEAREST);
    z = z.resize((96, 96), Image.NEAREST)
    sheet.paste(z, (x, y0 + 40), z); x += 160
    if ofav:
        sheet.paste(ofav, (x, y0 + 84), ofav); x += 56
        oz = ofav.resize((96, 96), Image.NEAREST); sheet.paste(oz, (x, y0 + 40), oz)
    sheet.save(os.path.join(OUT, "contact_sheet.png"))


if __name__ == "__main__":
    main()
