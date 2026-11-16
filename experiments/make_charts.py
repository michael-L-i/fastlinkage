"""Draw the README charts (SVG, light and dark) from the files in results/.

usage: python experiments/make_charts.py
"""
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"
THEMES = {
    "light": {"ours": "#2a78d6", "base": "#eb6834", "text": "#1f2328", "muted": "#59636e",
              "grid": "#d1d9e0", "bg": "#ffffff"},
    "dark": {"ours": "#3987e5", "base": "#d95926", "text": "#f0f6fc", "muted": "#9198a1",
             "grid": "#3d444d", "bg": "#0d1117"},
}
FONT = "-apple-system, BlinkMacSystemFont, 'Segoe UI', Helvetica, Arial, sans-serif"


def load(name):
    return json.loads((RES / name).read_text())


def human(s):
    if s >= 3600:
        return f"{s / 60:.0f} min"
    if s >= 120:
        return f"{s / 60:.1f} min"
    return f"{s:.0f} s" if s >= 20 else f"{s:.1f} s"


def svg(w, h, body, t):
    return (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}" width="{w}" height="{h}" '
            f'font-family="{FONT}" font-size="13">\n<rect width="{w}" height="{h}" fill="{t["bg"]}"/>\n'
            + "\n".join(body) + "\n</svg>\n")


def legend(x, y, t, base_name):
    return [f'<circle cx="{x}" cy="{y}" r="6" fill="{t["ours"]}"/>',
            f'<text x="{x + 12}" y="{y + 4}" fill="{t["muted"]}">fastlinkage, Apple laptop GPU</text>',
            f'<circle cx="{x + 220}" cy="{y}" r="6" fill="{t["base"]}"/>',
            f'<text x="{x + 232}" y="{y + 4}" fill="{t["muted"]}">{base_name}</text>']


def speed_chart(t):
    e6 = {(r["data"], r["n"], r["method"]): r for r in load("neighbour_lists.json")}
    e5 = {(r["n"], r["method"]): r for r in load("embeddings.json")["same_tree"]}
    e7 = load("large_runs.json")["ward_vs_fastcluster"]
    e8 = load("ward_embeddings_check.json")
    rows = [
        ("20,000 text embeddings", "average linkage, cosine · vs SciPy",
         e6[("agnews384", 20000, "average_cosine")]["fastlinkage_s"], e5[(20000, "average_cosine")]["scipy_s"]),
        ("100,000 points, 32 features", "Ward · vs fastcluster", e7["ours_s"], e7["fastcluster_s"]),
        ("120,000 text embeddings", "Ward · vs fastcluster",
         e6[("agnews384", 120000, "ward")]["fastlinkage_s"], e8["fastcluster_s"]),
    ]
    w, x0, x1, top, rh = 800, 250, 740, 86, 70
    h = top + rh * len(rows) + 38
    lo, hi = -1, 4                                   # 0.1 s .. 10,000 s

    def X(s):
        return x0 + (math.log10(s) - lo) / (hi - lo) * (x1 - x0)

    b = [f'<text x="20" y="28" font-size="16" font-weight="600" fill="{t["text"]}">'
         'Time to build the exact tree (log scale)</text>']
    b += legend(26, 52, t, "SciPy or fastcluster, CPU")
    base = top + rh * len(rows)
    for e, lab in zip(range(lo, hi + 1), ("0.1 s", "1 s", "10 s", "100 s", "1,000 s", "10,000 s"), strict=False):
        x = X(10 ** e)
        b.append(f'<line x1="{x:.1f}" y1="{top - 8}" x2="{x:.1f}" y2="{base}" stroke="{t["grid"]}" stroke-width="1"/>')
        b.append(f'<text x="{x:.1f}" y="{base + 20}" text-anchor="middle" font-size="12" fill="{t["muted"]}">{lab}</text>')
    for i, (name, sub, ours, ref) in enumerate(rows):
        y = top + rh * i + rh / 2
        b.append(f'<text x="20" y="{y - 3}" font-weight="600" fill="{t["text"]}">{name}</text>')
        b.append(f'<text x="20" y="{y + 15}" font-size="12" fill="{t["muted"]}">{sub}</text>')
        xa, xb = X(ours), X(ref)
        b.append(f'<line x1="{xa:.1f}" y1="{y}" x2="{xb:.1f}" y2="{y}" stroke="{t["grid"]}" stroke-width="2"/>')
        b.append(f'<circle cx="{xa:.1f}" cy="{y}" r="7" fill="{t["ours"]}" stroke="{t["bg"]}" stroke-width="2"/>')
        b.append(f'<circle cx="{xb:.1f}" cy="{y}" r="7" fill="{t["base"]}" stroke="{t["bg"]}" stroke-width="2"/>')
        b.append(f'<text x="{xa:.1f}" y="{y + 24}" text-anchor="middle" font-size="12" fill="{t["text"]}">{human(ours)}</text>')
        b.append(f'<text x="{xb:.1f}" y="{y + 24}" text-anchor="middle" font-size="12" fill="{t["text"]}">{human(ref)}</text>')
        b.append(f'<text x="{(xa + xb) / 2:.1f}" y="{y - 9}" text-anchor="middle" font-weight="600" '
                 f'fill="{t["text"]}">{ref / ours:.0f}x faster</text>')
    return svg(w, h, b, t)


def scaling_chart(t):
    e2 = [r for r in load("scaling.json") if r["method"] == "ward"]
    e6 = {(r["data"], r["n"], r["method"]): r for r in load("neighbour_lists.json")}
    e7 = load("large_runs.json")
    ours = [(20000, e6[("blobs32", 20000, "ward")]["fastlinkage_s"]), (100000, e7["ward_vs_fastcluster"]["ours_s"])]
    ours += [(r["n"], r["s"]) for r in e7["timing"] if r["method"] == "ward"]
    ref = [(r["n"], r["fastcluster_s"]) for r in e2 if "fastcluster_s" in r]
    ref.append((100000, e7["ward_vs_fastcluster"]["fastcluster_s"]))
    w, h, x0, x1, y0, y1 = 800, 380, 70, 640, 320, 80
    lx, hx, ly, hy = math.log10(15000), math.log10(600000), -1, 3

    def X(n):
        return x0 + (math.log10(n) - lx) / (hx - lx) * (x1 - x0)

    def Y(s):
        return y0 - (math.log10(s) - ly) / (hy - ly) * (y0 - y1)

    b = [f'<text x="20" y="28" font-size="16" font-weight="600" fill="{t["text"]}">'
         'Exact Ward tree, 32 features: seconds against number of points (log-log)</text>']
    b += legend(26, 52, t, "fastcluster, CPU")
    for e, lab in zip(range(ly, hy + 1), ("0.1 s", "1 s", "10 s", "100 s", "1,000 s"), strict=False):
        y = Y(10 ** e)
        b.append(f'<line x1="{x0}" y1="{y:.1f}" x2="{x1}" y2="{y:.1f}" stroke="{t["grid"]}" stroke-width="1"/>')
        b.append(f'<text x="{x0 - 8}" y="{y + 4:.1f}" text-anchor="end" font-size="12" fill="{t["muted"]}">{lab}</text>')
    for n in (20000, 50000, 100000, 200000, 500000):
        b.append(f'<text x="{X(n):.1f}" y="{y0 + 20}" text-anchor="middle" font-size="12" fill="{t["muted"]}">{n:,}</text>')
    b.append(f'<text x="{(x0 + x1) / 2}" y="{y0 + 42}" text-anchor="middle" font-size="12" fill="{t["muted"]}">points</text>')
    for pts, col, where in ((ref, t["base"], -12), (ours, t["ours"], 22)):
        path = " ".join(f'{"M" if i == 0 else "L"}{X(n):.1f},{Y(s):.1f}' for i, (n, s) in enumerate(pts))
        b.append(f'<path d="{path}" fill="none" stroke="{col}" stroke-width="2" stroke-linejoin="round"/>')
        for n, s in pts:
            b.append(f'<circle cx="{X(n):.1f}" cy="{Y(s):.1f}" r="5" fill="{col}" stroke="{t["bg"]}" stroke-width="2"/>')
            b.append(f'<text x="{X(n):.1f}" y="{Y(s) + where:.1f}" text-anchor="middle" font-size="12" '
                     f'fill="{t["text"]}">{human(s)}</text>')
    n, s = ref[-1]
    b.append(f'<text x="{X(n) + 14:.1f}" y="{Y(s) + 4:.1f}" font-size="12" fill="{t["muted"]}">fastcluster (single thread)</text>')
    n, s = ours[-1]
    b.append(f'<text x="{X(n) + 14:.1f}" y="{Y(s) + 4:.1f}" font-size="12" fill="{t["muted"]}">fastlinkage</text>')
    return svg(w, h, b, t)


def main():
    out = ROOT / "assets"
    out.mkdir(exist_ok=True)
    for name, t in THEMES.items():
        (out / f"speed-{name}.svg").write_text(speed_chart(t))
        (out / f"scaling-{name}.svg").write_text(scaling_chart(t))
    print("wrote", sorted(p.name for p in out.glob("*.svg")))


if __name__ == "__main__":
    main()
