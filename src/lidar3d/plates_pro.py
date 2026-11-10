"""Professional plate layout for the White River figures: kicker, title, one-line deck, panel titles with letter badges, headline-number tiles, measured text wrapping (so paragraphs fit their boxes and stack without overlap),
flowing legends, and a footer with sources and the plate number. Subclass of `plates.Plate`, so `ax`, `text` and `save` (with the text-overlap audit) are the same."""
from __future__ import annotations

import textwrap

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.font_manager import FontProperties
from matplotlib.patches import Circle, Rectangle
from matplotlib.textpath import TextPath

from .plates import DPI, H_IN, W_IN, Plate

INK, TEXT, MUTED, FAINT, RULE, ACCENT, ACCENT2 = "#15191D", "#2B3138", "#5B6470", "#8B929C", "#D5D9DE", "#1F4E79", "#B0298A"
FONT = ["Liberation Sans", "DejaVu Sans"]
LEFT, RIGHT = 0.6, W_IN - 0.6
TOP, BOTTOM = 8.28, 0.98


def text_width(s, fs, bold=False):
    prop = FontProperties(family=FONT, weight="bold" if bold else "normal")
    return TextPath((0, 0), s, size=fs, prop=prop).get_extents().width / 72.0


def wrap(text, width_in, fs, bold=False):
    """Greedy word wrap to a width in inches, measured with the plate's font."""
    lines = []
    for para in str(text).split("\n"):
        cur = ""
        for word in para.split():
            trial = (cur + " " + word).strip()
            if text_width(trial, fs, bold) <= width_in or not cur:
                cur = trial
            else:
                lines.append(cur)
                cur = word
        lines.append(cur)
    return lines


class ProPlate(Plate):
    def __init__(self, number, kicker, title, deck, sources, note=""):
        self.number = number
        self.fig = plt.figure(figsize=(W_IN, H_IN), dpi=DPI, facecolor="white")
        self.bg = self.fig.add_axes((0, 0, 1, 1))
        self.bg.set_xlim(0, W_IN); self.bg.set_ylim(0, H_IN); self.bg.set_axis_off()
        bg = self.bg
        bg.text(LEFT, 9.62, kicker.upper(), fontsize=7.6, color=ACCENT, fontweight="bold", va="center", family=FONT)
        bg.text(LEFT, 9.2, title, fontsize=23, fontweight="bold", color=INK, va="center", family=FONT)
        fs = 11.5
        while text_width(deck, fs) > RIGHT - LEFT and fs > 8.5:
            fs -= 0.25
        bg.text(LEFT, 8.74, deck, fontsize=fs, color=MUTED, va="center", family=FONT)
        bg.plot([LEFT, RIGHT], [8.46, 8.46], color=INK, lw=1.2)
        bg.plot([LEFT, RIGHT], [0.80, 0.80], color=RULE, lw=0.8)
        lines = wrap(sources, 12.4, 6.6)
        bg.text(LEFT, 0.66, "\n".join(lines[:3]), fontsize=6.6, color=FAINT, va="top", linespacing=1.35, family=FONT)
        bg.text(RIGHT, 0.66, f"Plate {number}", fontsize=8, fontweight="bold", color=MUTED, ha="right", va="top", family=FONT)
        if note:
            bg.text(RIGHT, 0.46, "\n".join(wrap(note, 2.6, 6.6)[:2]), fontsize=6.6, color=FAINT, ha="right", va="top", linespacing=1.35, family=FONT)

    # ------------------------------------------------------------------ pieces
    def panel_title(self, x, y, letter, text, w=None):
        """Letter badge and bold title; an optional hairline underneath across `w` inches."""
        self.bg.add_patch(Circle((x + 0.105, y), 0.105, fc=ACCENT, ec="none", zorder=5))
        self.bg.text(x + 0.105, y - 0.003, letter, fontsize=7.2, color="white", fontweight="bold", ha="center", va="center", zorder=6, family=FONT)
        self.bg.text(x + 0.3, y, text, fontsize=9.2, color=INK, fontweight="bold", va="center", family=FONT)
        if w:
            self.bg.plot([x, x + w], [y - 0.17, y - 0.17], color=RULE, lw=0.7)

    def caption(self, x, y, w, text, fs=7.6, color=MUTED):
        lines = wrap(text, w, fs)
        self.bg.text(x, y, "\n".join(lines), fontsize=fs, color=color, va="top", linespacing=1.35, family=FONT)
        return y - 0.135 * len(lines) * fs / 7.6 - 0.04

    def tiles(self, x, y, w, items, num_fs=21):
        """Headline numbers: (value, caption) pairs in one row. Returns the y below the tiles."""
        n = len(items)
        tw = w / n
        h = 0.0
        for i, (val, cap) in enumerate(items):
            x0 = x + i * tw
            self.bg.text(x0 + 0.02, y, val, fontsize=num_fs, fontweight="bold", color=ACCENT, va="top", family=FONT)
            lines = wrap(cap, tw - 0.18, 7.4)
            self.bg.text(x0 + 0.02, y - 0.42, "\n".join(lines), fontsize=7.4, color=MUTED, va="top", linespacing=1.3, family=FONT)
            h = max(h, 0.42 + 0.125 * len(lines))
            if i:
                self.bg.plot([x0 - 0.08, x0 - 0.08], [y - 0.04, y - 0.1 - h * 0.9], color=RULE, lw=0.7)
        return y - h - 0.2

    def notes(self, x, y, w, items, fs=8.5, gap=0.17):
        """Headed paragraphs stacked from y downwards, wrapped to w inches. Returns the y below the last one."""
        for head, body in items:
            self.bg.text(x, y, head, fontsize=9.2, fontweight="bold", color=INK, va="top", family=FONT)
            y -= 0.215
            lines = wrap(body, w, fs)
            self.bg.text(x, y, "\n".join(lines), fontsize=fs, color=TEXT, va="top", linespacing=1.4, family=FONT)
            y -= 0.158 * len(lines) * fs / 8.5 + gap
        if y < BOTTOM - 0.05:
            print(f"  [plate {self.number}] text runs {BOTTOM - y:.2f} in below the page margin")
        return y

    def legend(self, x, y, w, items, fs=7.4, row_h=0.2, title=None):
        """Flowing legend. items = (kind, colour, label); kinds: line, dash, dot, square, tri, patch, ring. Returns the y below it."""
        if title:
            self.bg.text(x, y, title, fontsize=7.4, color=MUTED, fontweight="bold", va="top", family=FONT)
            y -= 0.19
        cx, cy = x, y
        for kind, col, label in items:
            lw = text_width(label, fs) + 0.4
            if cx + lw > x + w and cx > x:
                cx = x; cy -= row_h
            yy = cy - 0.07
            if kind in ("line", "dash"):
                self.bg.plot([cx, cx + 0.22], [yy, yy], color=col, lw=1.6, ls="-" if kind == "line" else (0, (2, 1.5)))
            elif kind == "dot":
                self.bg.plot([cx + 0.11], [yy], "o", color=col, ms=4)
            elif kind == "ring":
                self.bg.plot([cx + 0.11], [yy], "o", mfc="none", mec=col, ms=5, mew=1.2)
            elif kind == "square":
                self.bg.plot([cx + 0.11], [yy], "s", color=col, ms=4.5)
            elif kind == "tri":
                self.bg.plot([cx + 0.11], [yy], "^", mfc="none", mec=col, ms=5, mew=1.0)
            else:
                self.bg.add_patch(Rectangle((cx, yy - 0.055), 0.22, 0.11, fc=col, ec="#888", lw=0.3))
            self.bg.text(cx + 0.3, yy, label, fontsize=fs, color=TEXT, va="center", family=FONT)
            cx += lw + 0.22
        return cy - row_h

    def image(self, im, l, b, w, h, frame=True):
        a = self.ax(l, b, w, h)
        a.imshow(im, aspect="auto"); a.set_xticks([]); a.set_yticks([])
        for sp in a.spines.values():
            sp.set_color("#B7BCC3" if frame else "none"); sp.set_linewidth(0.6)
        return a
