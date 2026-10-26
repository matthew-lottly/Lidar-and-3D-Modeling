"""Figure plates: consistent editorial layout (title, map panels, source footer) and an automatic text-overlap audit."""
from __future__ import annotations

import textwrap
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle
from matplotlib.text import Text

INK, MUTED, FAINT, RULE, BLUE, RUST, GREEN = "#111111", "#555555", "#8A8A8A", "#CFCFCF", "#1F4E79", "#9A3B12", "#2F6B3F"
plt.rcParams.update({"font.family": ["Liberation Sans", "DejaVu Sans"], "text.parse_math": False, "axes.edgecolor": "#777777", "axes.linewidth": 0.8})
W_IN, H_IN, DPI = 16.0, 10.0, 150
OUT = Path(__file__).resolve().parents[2] / "outputs"


class Plate:
    def __init__(self, title: str, sub: str, sources: str, note: str = ""):
        self.fig = plt.figure(figsize=(W_IN, H_IN), dpi=DPI, facecolor="white")
        self.bg = self.fig.add_axes((0, 0, 1, 1))
        self.bg.set_xlim(0, W_IN); self.bg.set_ylim(0, H_IN); self.bg.set_axis_off()
        self.bg.text(0.6, 9.4, title, fontsize=24, fontweight="bold", color=INK, va="center")
        self.bg.text(0.6, 8.9, sub, fontsize=12.5, color=MUTED, va="center")
        self.bg.plot([0.6, W_IN - 0.6], [8.55, 8.55], color=INK, lw=1.3)
        self.bg.plot([0.6, W_IN - 0.6], [0.78, 0.78], color=RULE, lw=0.8)
        self.bg.text(0.6, 0.55, textwrap.fill(sources, 210), fontsize=7.2, color=FAINT, va="top", linespacing=1.3)
        if note:
            self.bg.text(W_IN - 0.6, 0.55, note, fontsize=7.2, color=FAINT, va="top", ha="right")

    def ax(self, l, b, w, h, **kw):
        return self.fig.add_axes((l / W_IN, b / H_IN, w / W_IN, h / H_IN), **kw)

    def label(self, x, y, text, **kw):
        kw.setdefault("fontsize", 9.5); kw.setdefault("color", MUTED); kw.setdefault("fontweight", "bold")
        return self.bg.text(x, y, text.upper(), va="center", **kw)

    def text(self, x, y, s, **kw):
        kw.setdefault("fontsize", 9.5); kw.setdefault("color", INK); kw.setdefault("va", "top")
        return self.bg.text(x, y, s, **kw)

    def save(self, name: str, audit_on=True) -> list[str]:
        issues = audit(self.fig) if audit_on else []
        OUT.mkdir(parents=True, exist_ok=True)
        self.fig.savefig(OUT / name, facecolor="white")
        plt.close(self.fig)
        for i in issues:
            print(f"  [{name}] {i}")
        return issues


def audit(fig) -> list[str]:
    """Text overlapping other text, or leaving the page."""
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    Wp, Hp = fig.canvas.get_width_height()
    skip = set()
    for a in fig.axes:
        if a.axison:
            for axis in (a.xaxis, a.yaxis):
                skip.update(id(t) for t in axis.get_ticklabels(which="both"))
    items = []
    for t in fig.findobj(Text):
        s = t.get_text().strip()
        if not s or not t.get_visible() or id(t) in skip:
            continue
        bb = t.get_window_extent(r)
        if bb.width < 1 or bb.height < 1:
            continue
        items.append((s.replace("\n", " ")[:34], bb))
    out = []
    for i, (a, ba) in enumerate(items):
        if ba.x0 < -2 or ba.y0 < -2 or ba.x1 > Wp + 2 or ba.y1 > Hp + 2:
            out.append(f"off page: {a!r}")
        for b, bb in items[i + 1:]:
            x = min(ba.x1, bb.x1) - max(ba.x0, bb.x0); y = min(ba.y1, bb.y1) - max(ba.y0, bb.y0)
            if x > 1 and y > 1 and x * y > 0.12 * min(ba.width * ba.height, bb.width * bb.height):
                out.append(f"overlap: {a!r} x {b!r}")
    return out


def map_frame(ax, extent, step=200.0, grid=True, labels=True):
    x0, x1, y0, y1 = extent
    ax.set_xlim(x0, x1); ax.set_ylim(y0, y1); ax.set_aspect("equal")
    xs = np.arange(np.ceil(x0 / step) * step, x1, step); ys = np.arange(np.ceil(y0 / step) * step, y1, step)
    ax.set_xticks(xs); ax.set_yticks(ys)
    ax.set_xticklabels([f"{v:,.0f}" for v in xs] if labels else [], fontsize=6.5, color=MUTED)
    ax.set_yticklabels([f"{v:,.0f}" for v in ys] if labels else [], fontsize=6.5, color=MUTED, rotation=90, va="center")
    ax.tick_params(length=2.5, width=0.6, colors="#777777", pad=2)
    if grid:
        ax.grid(color="white", lw=0.35, alpha=0.35)


def scalebar(ax, extent, length=200.0, loc="bl", color=INK):
    x0, x1, y0, y1 = extent
    w, h = x1 - x0, y1 - y0
    bx, by = (x0 + 0.04 * w, y0 + 0.05 * h) if loc == "bl" else (x1 - 0.04 * w - length, y0 + 0.05 * h)
    ax.add_patch(Rectangle((bx - 0.015 * w, by - 0.025 * h), length + 0.03 * w + 0.08 * w, 0.085 * h, fc="white", alpha=0.85, ec="none", zorder=40))
    for i in range(4):
        ax.add_patch(Rectangle((bx + i * length / 4, by), length / 4, 0.01 * h, fc=color if i % 2 == 0 else "white", ec=color, lw=0.6, zorder=41))
    for i in (0, 2, 4):
        ax.text(bx + i * length / 4, by + 0.018 * h, f"{i * length / 4:g}", fontsize=6.5, ha="center", va="bottom", color=INK, zorder=42)
    ax.text(bx + length + 0.012 * w, by + 0.005 * h, "m", fontsize=6.5, va="center", color=INK, zorder=42)


def north_arrow(ax, extent):
    x0, x1, y0, y1 = extent
    w, h = x1 - x0, y1 - y0
    cx, cy = x1 - 0.045 * w, y1 - 0.085 * h
    ax.add_patch(Rectangle((cx - 0.03 * w, cy - 0.055 * h), 0.06 * w, 0.115 * h, fc="white", alpha=0.85, ec="none", zorder=40))
    ax.annotate("", xy=(cx, cy + 0.045 * h), xytext=(cx, cy - 0.04 * h), arrowprops=dict(arrowstyle="-|>", color=INK, lw=1.3), zorder=42)
    ax.text(cx, cy + 0.05 * h, "N", fontsize=8, fontweight="bold", ha="center", va="bottom", color=INK, zorder=42)


def colorbar(plate: Plate, cmap, vmin, vmax, l, b, w, h, label, ticks=None, orient="h"):
    ax = plate.ax(l, b, w, h)
    grad = np.linspace(0, 1, 256)[None, :] if orient == "h" else np.linspace(1, 0, 256)[:, None]
    ax.imshow(grad, aspect="auto", cmap=cmap, extent=(vmin, vmax, 0, 1) if orient == "h" else (0, 1, vmin, vmax))
    if orient == "h":
        ax.set_yticks([]); ax.tick_params(labelsize=7, length=2, colors=MUTED)
        if ticks is not None:
            ax.set_xticks(ticks)
    else:
        ax.set_xticks([]); ax.yaxis.tick_right(); ax.tick_params(labelsize=7, length=2, colors=MUTED)
    for s in ax.spines.values():
        s.set_linewidth(0.4)
    plate.text(l, b + h + 0.1, label, fontsize=7.5, color=MUTED, va="bottom")
    return ax
