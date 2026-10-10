r"""Spin and orbital structure of the f fbar pair in decays of a spin-zero boson (the book, the chapter on the Higgs decays;
the paper of N. Borodin and D. Naumov, 2026).

Parts of the film:
 1. back-to-back pairs build a wave: the amplitude M(n) to emit the pair along n (both spins down on the z axis: M ~ n_x + i n_y);
    adding the plane waves exp(i p n.r) of more and more directions gives the wave function of the relative coordinate r = x1 - x2,
    a vortex psi ~ j_1(pr) (rhat_x + i rhat_y): a pure P-wave, the phase winds once (colour = phase, brightness = modulus);
 2. scalar decay H(0+) -> f fbar: with both spin projections fixed on one axis s the pair is in a state of definite orbital
    projection m = -(rho1 + rho2)/2 (J_s = 0): m = 0 (two lobes), m = -+1 (a ring around the axis with the phase winding once);
    the surfaces |psi|^2 = const coloured by the phase; the probability of the four spin outcomes along a direction at the polar angle
    theta: (1/2)[1 + rho1 rho2 (1 - 2 cos^2 theta)], i.e. (1 - rho1 rho2) at theta = 0 and (1 + rho1 rho2) at theta = pi/2;
 3. pseudoscalar decay: psi ~ j_0(pr) chi^dag eta, a pure S-wave, the spin singlet, no dependence on rhat;
 4. a mixed state eps1 psi_S + eps2 psi_P: the interference term eps1 eps2 beta j_0 j_1 xi1.(rhat x xi2) makes the cloud lopsided; the
    exchange of the spins of f and fbar (C P) reverses the lean; on the axis the S- and P-waves add on one side and cancel on the other;
 5. what a detector sees: the projection on momentum eigenstates loses the phase of M(n); what survives are spin-angle correlations.

The physics is in ffbar.py (tests: test_ffbar.py), the surface renderer in isorender.py.  All the numbers are in config.toml and all
the words in texts.toml (see ../dvconfig.py for --config / --set).

Usage:
    python higgs_ff_spin.py --lang en            # film -> media/higgs_ff_spin_en.mp4
    python higgs_ff_spin.py --lang ru
    python higgs_ff_spin.py --lang en --snapshot 40
"""

from __future__ import annotations

import argparse
import hashlib
import math
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dvconfig import load_config, load_texts  # noqa: E402

import ffbar as fb  # noqa: E402
import isorender as ir  # noqa: E402

HERE = Path(__file__).resolve().parent
CFG = load_config(HERE)

CARD_S = CFG.timeline.card_s
CARD_AT = list(CFG.timeline.card_at)
CARD_KEYS = [f"h{i}" for i in range(len(CARD_AT))]
_PB = list(CFG.timeline.part_bounds)
CONTENT_TOTAL = _PB[-1]
TOTAL = CONTENT_TOTAL + CARD_S * len(CARD_AT)
T_P1, T_P2, T_P3, T_P4, T_P5 = ((_PB[i], _PB[i + 1]) for i in range(5))

BETA = CFG.physics.beta
P_PHASE = complex(np.exp(1j * math.radians(CFG.physics.p_phase_deg)))
TEXT = {lang: load_texts(HERE, lang) for lang in ("en", "ru")}


def timeline(tf: float) -> tuple[float, int | None, float]:
    """(content time, index of the chapter card or None, progress of the card in [0, 1]) at the film time tf."""
    done = 0
    for i, ca in enumerate(CARD_AT):
        a = ca + i * CARD_S
        if a <= tf < a + CARD_S:
            return ca, i, (tf - a) / CARD_S
        if tf >= a + CARD_S:
            done += 1
    return tf - done * CARD_S, None, 0.0


def film_time(tc: float) -> float:
    return tc + CARD_S * sum(1 for ca in CARD_AT if ca <= tc)


def chapter_of(tc: float) -> int:
    return max(i for i, ca in enumerate(CARD_AT) if ca <= tc)


def smooth(x: float, a: float, b: float) -> float:
    u = min(max((x - a) / (b - a), 0.0), 1.0)
    return u * u * (3 - 2 * u)


def num(x: float, fmt: str, lang: str) -> str:
    s = format(x, fmt)
    return s.replace(".", "{,}") if lang == "ru" else s


def window(t: float, a: float, b: float, fade: float) -> float:
    """1 inside [a, b] with smooth fades of the length `fade` at both ends."""
    return smooth(t, a, a + fade) * (1.0 - smooth(t, b - fade, b))


def scheduled(items, tl: float, fade: float) -> list[tuple[str, float, int]]:
    """[(key, opacity, row)] of the items {key, from, to, row} visible at the local time tl."""
    out = []
    for it in items:
        a = window(tl, it["from"], it["to"], fade)
        if a > 0.0:
            out.append((it["key"], a, it.get("row", 0)))
    return out


def segments(tl: float, segs) -> float:
    """Piecewise smooth parameter: segs = [[t0, t1, v0, v1], ...]; before the first segment v0 of the first, after the last v1 of the last."""
    if tl <= segs[0][0]:
        return float(segs[0][2])
    for t0, t1, v0, v1 in segs:
        if tl < t1:
            u = smooth(tl, t0, t1)
            return float(v0 + (v1 - v0) * u)
    return float(segs[-1][3])


# ------------------------------------------------------------------------------------------------- palette

def palette(phase, delta: float = 0.0) -> np.ndarray:
    """The phase wheel c_ch(phi + delta) = offset + amp cos(phi + delta - 2 pi ch / 3): colours (..., 3)."""
    ph = np.asarray(phase, dtype=float)[..., None] + delta - 2.0 * math.pi * np.arange(3) / 3.0
    return CFG.palette.offset + CFG.palette.amp * np.cos(ph)


def hue_shift(t: float) -> float:
    """The common phase factor exp(-i omega t) of the state: the colour of every point is c(arg psi + delta) with delta = -omega t."""
    return -2.0 * math.pi * t / CFG.palette.hue_period_s


def bg_rgb() -> np.ndarray:
    h = CFG.style.background.lstrip("#")
    return np.array([int(h[i:i + 2], 16) for i in (0, 2, 4)], dtype=float) / 255.0


# ------------------------------------------------------------------------------------ spin states of the cells

def eps_of(ratio: float) -> tuple[float, float]:
    """The couplings of the mixed state: eps1 = 1 (the scalar part is kept), eps2 = ratio."""
    return 1.0, ratio


def cell_psi(kind: str, r1: int, r2: int, ratio: float = 0.0, swap: bool = False):
    """The wave function psi(P) (P: x = p r, shape (..., 3)) of a cell: 'S' scalar, 'P' pseudoscalar, 'mix' the mixed state of the part 4."""
    if kind == "mix":
        s1, s2 = list(CFG.physics.mixed_xi1), list(CFG.physics.mixed_xi2)
        if swap:
            s1, s2 = s2, s1
        st = fb.PairState(tuple(fb.unit(s1)), tuple(fb.unit(s2)), 1, 1)
        e1, e2 = eps_of(ratio)
    else:
        st = fb.common_axis_state(CFG.physics.spin_axis, r1, r2)
        e1, e2 = (1.0, 0.0) if kind == "S" else (0.0, 1.0)
    return lambda P: fb.psi_mixed(P, st, e1, e2, BETA, P_PHASE)


def iso_camera():
    return ir.camera(CFG.iso.elevation_deg, CFG.iso.azimuth_deg)


def iso_shading() -> ir.Shading:
    S = CFG.iso
    return ir.Shading(S.ambient, S.diffuse, S.specular, S.spec_power, tuple(S.light))


_cache: dict = {}
CACHE_DIR = HERE / "cache"


def iso_image(kind: str, r1: int, r2: int, size: int, ratio: float = 0.0, swap: bool = False, disk: bool = False) -> ir.IsoImage:
    """The rendered surfaces of a cell (cached in memory, optionally on disk: the images do not depend on the language)."""
    S = CFG.iso
    key = (kind, r1, r2, size, round(ratio, 4), swap)
    if key in _cache:
        return _cache[key]
    path = None
    if disk:
        tag = hashlib.sha1(repr((key, S.to_dict(), CFG.physics.to_dict())).encode()).hexdigest()[:16]
        path = CACHE_DIR / f"iso_{tag}.npz"
        if path.exists():
            try:
                d = np.load(path)
                img = ir.IsoImage(d["s0"], d["s1"], d["spec"], d["cover"])
                _cache[key] = img
                return img
            except Exception:
                pass
    img = ir.render(cell_psi(kind, r1, r2, ratio, swap), iso_camera(), list(S.levels), list(S.alphas), iso_shading(), S.mix_r_max if kind == "mix" else S.r_max,
                    S.mix_view if kind == "mix" else S.view, size, S.samples, S.supersample, S.edge, S.grad_step)
    _cache[key] = img
    if path is not None:
        CACHE_DIR.mkdir(exist_ok=True)
        tmp = path.with_suffix(f".{id(img)}.tmp.npz")
        np.savez(tmp, s0=img.s0, s1=img.s1, spec=img.spec, cover=img.cover)
        tmp.replace(path)
    return img


_sums: dict = {}


def plane_wave_sum() -> fb.PlaneWaveSum:
    if "sum" not in _sums:
        Q = CFG.partial_sums
        pts = fb.halton_sphere(Q.n_directions)
        _sums["sum"] = fb.PlaneWaveSum(pts, pts[:, 0] + 1j * pts[:, 1], Q.grid, Q.half_width)
    return _sums["sum"]


def n_directions(tl: float) -> int:
    """Number of the plane waves added at the local time tl of the part 1: doubling every `octave_s` seconds."""
    Q = CFG.partial_sums
    k = max(0.0, (tl - Q.start_s) / Q.octave_s)
    return int(min(Q.n_directions, max(1, round(2.0 ** k))))


# ----------------------------------------------------------------------------------------------- the canvas

@dataclass
class CellGeom:
    cx: float
    cy: float
    half: float
    view: float = 0.0

    def to_px(self, scr) -> np.ndarray:
        s = np.asarray(scr, dtype=float)
        return np.stack([self.cx + s[..., 0] / self.view * self.half, self.cy + s[..., 1] / self.view * self.half], axis=-1)


class Canvas:
    """One frame: a single axes that spans the figure, in pixels (y up); everything is placed by fractions of the frame."""

    def __init__(self, size: tuple[int, int], lang: str) -> None:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        self.plt = plt
        self.W, self.H = size
        V = CFG.video
        self.sc = self.H / V.reference_height
        self.lang = lang
        self.tx = TEXT[lang]
        self.bg = bg_rgb()
        self.fig = plt.figure(figsize=(self.W / V.dpi, self.H / V.dpi), dpi=V.dpi, facecolor=CFG.style.background)
        self.ax = self.fig.add_axes([0, 0, 1, 1], facecolor="none")
        self.reset()

    def reset(self) -> None:
        self.ax.clear()
        self.ax.set_facecolor("none")
        self.ax.set_xlim(0, self.W)
        self.ax.set_ylim(0, self.H)
        self.ax.axis("off")
        self.texts: list = []

    def px(self, fx: float, fy: float) -> np.ndarray:
        return np.array([fx * self.W, fy * self.H])

    def color(self, name: str, alpha: float = 1.0) -> tuple:
        c = getattr(CFG.style, name)
        return (*c, alpha) if len(c) == 3 else (*c[:3], c[3] * alpha)

    def text(self, fx: float, fy: float, s: str, size: float, color: tuple, ha: str = "left", va: str = "center", maxw: float | None = None,
             zorder: int = 20, rotation: float = 0.0):
        t = self.ax.text(fx * self.W, fy * self.H, s, color=color, fontsize=size * self.sc, ha=ha, va=va, zorder=zorder, rotation=rotation)
        if maxw is not None:
            bb = t.get_window_extent(self.fig.canvas.get_renderer())
            if bb.width > maxw * self.W:
                t.set_fontsize(size * self.sc * maxw * self.W / bb.width)
        self.texts.append(t)
        return t

    def text_px(self, p, s: str, size: float, color: tuple, ha: str = "center", va: str = "center", zorder: int = 20):
        t = self.ax.text(p[0], p[1], s, color=color, fontsize=size * self.sc, ha=ha, va=va, zorder=zorder)
        self.texts.append(t)
        return t

    def line(self, pts, color: tuple, lw: float, dash=None, zorder: int = 5, cap: str = "round"):
        p = np.asarray(pts, dtype=float)
        kw = {"ls": (0, tuple(d * self.sc for d in dash))} if dash else {}
        self.ax.plot(p[:, 0], p[:, 1], color=color, lw=lw * self.sc, solid_capstyle=cap, zorder=zorder, **kw)

    def arrow(self, p0, p1, color: tuple, lw: float, head: float, zorder: int = 6) -> None:
        """An arrow in pixels: a line with a filled triangular head (length `head`, half-width head * head_aspect)."""
        from matplotlib.patches import Polygon
        p0, p1 = np.asarray(p0, float), np.asarray(p1, float)
        d = p1 - p0
        L = float(np.hypot(*d))
        if L < 1e-6:
            return
        u = d / L
        n = np.array([-u[1], u[0]])
        hl = min(head * self.sc, 0.7 * L)
        hw = hl * CFG.glyph.head_aspect
        self.line([p0, p1 - u * hl * CFG.glyph.head_overlap], color, lw, zorder=zorder)
        self.ax.add_patch(Polygon([p1, p1 - u * hl + n * hw, p1 - u * hl - n * hw], closed=True, facecolor=color, edgecolor="none", zorder=zorder))

    def image(self, rgb: np.ndarray, cx: float, cy: float, half: float, alpha: float = 1.0, zorder: int = 2, interp: str = "bilinear") -> None:
        if alpha < 1.0:
            rgb = rgb * alpha + self.bg * (1.0 - alpha)
        self.ax.imshow(rgb, extent=(cx - half, cx + half, cy - half, cy + half), origin="upper", interpolation=interp, zorder=zorder, aspect="auto")


# ------------------------------------------------------------------------------------------------ elements

def phase_wheel(cv: Canvas, c: np.ndarray, r_out: float, delta: float, alpha: float, label: bool = True) -> None:
    """The legend: the colour of the phase phi (counter-clockwise from the x axis) at the present time."""
    W = CFG.wheel
    n = W.pixels
    a = np.linspace(-1.0, 1.0, n)
    X, Y = np.meshgrid(a, a[::-1])
    R = np.hypot(X, Y)
    rgb = palette(np.arctan2(Y, X), delta)
    mask = ((R <= 1.0) & (R >= W.inner)).astype(float)
    rgba = np.dstack([rgb, mask * alpha])
    cv.ax.imshow(rgba, extent=(c[0] - r_out, c[0] + r_out, c[1] - r_out, c[1] + r_out), origin="upper", interpolation="bilinear", zorder=3, aspect="auto")
    if label and alpha > 0.05:
        for k, key in enumerate(("w0", "w1", "w2", "w3")):
            ang = 0.5 * math.pi * k
            p = c + (r_out + W.label_gap * cv.sc) * np.array([math.cos(ang), math.sin(ang)])
            ha = "left" if k == 0 else ("right" if k == 2 else "center")
            va = "bottom" if k == 1 else ("top" if k == 3 else "center")
            cv.text_px(p, cv.tx[key], W.label_size, cv.color("dim", alpha), ha=ha, va=va)


def spin_glyph(cv: Canvas, c: np.ndarray, up1: int, up2: int, alpha: float) -> None:
    """Two vertical arrows: the spin of the fermion (blue) and of the antifermion (warm)."""
    G = CFG.glyph
    L = G.length * cv.sc
    for k, (up, name, key) in enumerate(((up1, "blue", "f"), (up2, "warm", "fbar"))):
        x = c[0] + (k - 0.5) * G.gap * cv.sc
        y0, y1 = c[1] - 0.5 * L * up, c[1] + 0.5 * L * up
        cv.arrow((x, y0), (x, y1), cv.color(name, alpha), G.width, G.head)
        cv.text_px((x, c[1] - 0.5 * L - G.label_gap * cv.sc), cv.tx[key], G.label_size, cv.color(name, alpha), va="top")


def draw_cell(cv: Canvas, geom: CellGeom, img: ir.IsoImage, delta: float, alpha: float, overlays: bool = True) -> None:
    S = CFG.iso
    rgb = img.rgb(delta, CFG.palette.offset, CFG.palette.amp, cv.bg)
    cv.image(rgb, geom.cx, geom.cy, geom.half, alpha)
    if overlays and alpha > 0.02:
        O = CFG.overlay
        cam = iso_camera()
        col = cv.color("dim", O.alpha * alpha)
        ang = np.linspace(0.0, 2.0 * math.pi, O.ring_points)
        ring = np.stack([O.ring_radius * np.cos(ang), O.ring_radius * np.sin(ang), np.zeros_like(ang)], axis=-1)
        cv.line(geom.to_px(ir.project(ring, cam)), col, O.width, dash=O.dash, zorder=4)
        top = geom.to_px(ir.project(np.array([[0, 0, O.axis_length], [0, 0, -O.axis_length]]), cam))
        cv.line(top, col, O.width, dash=O.dash, zorder=4)
        cv.arrow(geom.to_px(ir.project(np.array([0, 0, O.axis_length - O.axis_head]), cam)), top[0], cv.color("dim", O.alpha * alpha), O.width, O.head, zorder=4)
        cv.text_px(top[0] + np.array([O.axis_label_dx, O.axis_label_dy]) * cv.sc, cv.tx["axis_s"], O.label_size, cv.color("dim", alpha))


def winding_arc(cv: Canvas, geom: CellGeom, m: int, alpha: float) -> None:
    """An arrow along increasing phi (m = +1) or decreasing phi (m = -1) around the axis: the direction in which the phase grows."""
    O = CFG.overlay
    cam = iso_camera()
    s = np.sign(m)
    a0 = math.radians(O.arc_start_deg)
    ang = a0 + s * np.linspace(0.0, math.radians(O.arc_span_deg), O.ring_points)
    pts = np.stack([O.ring_radius * np.cos(ang), O.ring_radius * np.sin(ang), np.full_like(ang, O.arc_z)], axis=-1)
    px = geom.to_px(ir.project(pts, cam))
    cv.line(px, cv.color("gold", alpha), O.arc_width, zorder=7)
    cv.arrow(px[-3], px[-1], cv.color("gold", alpha), O.arc_width, O.arc_head, zorder=7)


def probe(cv: Canvas, geom: CellGeom, theta: float, alpha: float) -> None:
    """The direction rhat at the polar angle theta from the spin axis (in a fixed meridian plane)."""
    O = CFG.overlay
    cam = iso_camera()
    ph = math.radians(CFG.iso.azimuth_deg + O.probe_azimuth_offset_deg)
    rhat = np.array([math.sin(theta) * math.cos(ph), math.sin(theta) * math.sin(ph), math.cos(theta)])
    tip = geom.to_px(ir.project(O.probe_length * rhat, cam))
    o = geom.to_px(ir.project(np.zeros(3), cam))
    cv.arrow(o, tip, cv.color("gold", alpha), O.probe_width, O.probe_head, zorder=8)
    cv.ax.scatter([tip[0]], [tip[1]], s=O.probe_dot * cv.sc ** 2, c=[cv.color("gold", alpha)], linewidths=0, zorder=9)
    # the arc of the polar angle theta from the axis
    if theta > math.radians(CFG.overlay.arc_min_deg):
        t_ = np.linspace(0.0, theta, O.theta_points)
        arc = np.stack([O.theta_radius * np.sin(t_) * math.cos(ph), O.theta_radius * np.sin(t_) * math.sin(ph), O.theta_radius * np.cos(t_)], axis=-1)
        cv.line(geom.to_px(ir.project(arc, cam)), cv.color("gold", O.theta_alpha * alpha), O.theta_width, zorder=8)


def mini_plot(cv: Canvas, rect, xlim, ylim, curves, alpha: float, xticks, yticks, xlabel: str = "", ylabel: str = "") -> tuple:
    """A small plot in pixels: rect = (x0, y0, x1, y1) fractions; curves = [(x, y, colour, dash)]; returns the mapping function."""
    P = CFG.plot
    x0, y0, x1, y1 = rect[0] * cv.W, rect[1] * cv.H, rect[2] * cv.W, rect[3] * cv.H
    mp = lambda x, y: (x0 + (np.asarray(x) - xlim[0]) / (xlim[1] - xlim[0]) * (x1 - x0), y0 + (np.asarray(y) - ylim[0]) / (ylim[1] - ylim[0]) * (y1 - y0))
    dim = cv.color("dim", alpha * P.axis_alpha)
    cv.line([(x0, y0), (x1, y0)], dim, P.axis_width, zorder=4)
    cv.line([(x0, y0), (x0, y1)], dim, P.axis_width, zorder=4)
    for xv, lab in xticks:
        px_, py_ = mp(xv, ylim[0])
        cv.line([(px_, py_), (px_, py_ - P.tick * cv.sc)], dim, P.axis_width, zorder=4)
        cv.text_px((px_, py_ - (P.tick + P.tick_gap) * cv.sc), lab, P.tick_size, cv.color("dim", alpha), va="top")
    for yv, lab in yticks:
        px_, py_ = mp(xlim[0], yv)
        cv.line([(px_, py_), (px_ - P.tick * cv.sc, py_)], dim, P.axis_width, zorder=4)
        cv.text_px((px_ - (P.tick + P.tick_gap) * cv.sc, py_), lab, P.tick_size, cv.color("dim", alpha), ha="right")
    for x, y, col, dash in curves:
        gx, gy = mp(x, y)
        cv.line(np.stack([gx, gy], axis=-1), (*col[:3], col[3] * alpha) if len(col) == 4 else (*col, alpha), P.curve_width, dash=dash, zorder=5)
    if xlabel:
        cv.text_px((x1 + P.xlabel_dx * cv.sc, y0), xlabel, P.label_size, cv.color("dim", alpha), ha="left", va="center")
    if ylabel:
        cv.text_px((x0, y1 + P.ylabel_rise * cv.sc), ylabel, P.label_size, cv.color("dim", alpha), ha="left", va="bottom")
    return mp


def value_label(cv: Canvas, dx: float, dy: float, theta: float, label: str, alpha: float) -> None:
    """The value at the moving dot of a mini plot: to the right of the dot, to the left of it near the end of the axis."""
    P = CFG.plot
    left = theta > P.flip_fraction * 0.5 * math.pi
    sign = -1.0 if left else 1.0
    cv.text_px((dx + sign * P.value_dx * cv.sc, dy + P.value_dy * cv.sc), label, P.value_size, cv.color("gold", alpha), ha="right" if left else "left")


# ---------------------------------------------------------------------------------------------- the scenes

def formulas_and_captions(cv: Canvas, items_f, items_c, tl: float, fade: float) -> None:
    F, LY, S = CFG.fonts, CFG.layout, CFG.style
    for key, a, row in scheduled(items_f, tl, fade):
        cv.text(LY.formula_pos[0], LY.formula_pos[1] - LY.formula_row * row, cv.tx[key], F.formula, cv.color("text", S.formula_alpha * a), maxw=LY.formula_maxw)
    for key, a, row in scheduled(items_c, tl, fade):
        cv.text(LY.caption_pos[0], LY.caption_pos[1], cv.tx[key], F.caption, cv.color("text", S.caption_alpha * a), va="bottom", maxw=LY.caption_maxw)


def scene_p1(cv: Canvas, tl: float, t: float) -> None:
    P, F, LY = CFG.p1, CFG.fonts, CFG.layout
    delta = hue_shift(t)
    a_in = smooth(tl, *P.panels_in)
    Ls, Rs = P.left_side * cv.H, P.right_side * cv.H
    lc, rc = cv.px(*P.left_center), cv.px(*P.right_center)
    # momentum space: the directions of the pair
    N = n_directions(tl)
    sums = plane_wave_sum()
    pts = sums.n[:N]
    cv.ax.add_patch(cv.plt.Circle(lc, 0.5 * Ls * P.disc_fraction, fill=False, edgecolor=cv.color("dim", P.disc_alpha * a_in), lw=P.disc_width * cv.sc, zorder=3))
    for d in ((1, 0), (0, 1)):
        e = 0.5 * Ls * P.disc_fraction * np.array(d)
        cv.line([lc - e, lc + e], cv.color("dim", P.cross_alpha * a_in), P.disc_width, dash=P.cross_dash, zorder=3)
    w = pts[:, 0] + 1j * pts[:, 1]
    cols = palette(np.angle(w), delta)
    amp = np.clip(np.abs(w), 0.0, 1.0)
    rgba = np.column_stack([cols, a_in * (P.dot_alpha_min + (1.0 - P.dot_alpha_min) * amp)])
    boost = 1.0 + P.dot_boost * max(0.0, 1.0 - math.log2(max(N, 1)) / math.log2(P.boost_n))
    cv.ax.scatter(lc[0] + pts[:, 0] * 0.5 * Ls * P.disc_fraction, lc[1] + pts[:, 1] * 0.5 * Ls * P.disc_fraction,
                  s=(P.dot_size * cv.sc ** 2) * boost * (P.dot_scale_min + (1 - P.dot_scale_min) * amp), c=rgba, linewidths=0, zorder=5)
    cv.text_px(lc + np.array([0.0, 0.5 * Ls * P.disc_fraction + P.title_gap * cv.sc]), cv.tx["p1_left"], P.title_size, cv.color("text", a_in), va="bottom")
    # coordinate space: the partial sum of N plane waves, then the exact integral
    ex = sums.exact()
    f_ex = smooth(tl, *P.exact_fade)
    psi = sums.get(N)
    psi = (1.0 - f_ex) * psi + f_ex * ex
    peak = 4.0 * math.pi * float(fb.j1(fb.j1_first_maximum()))
    amp_img = np.clip(np.abs(psi) / peak, 0.0, 1.0) ** P.gamma
    rgb = palette(np.angle(psi), delta) * amp_img[..., None] + cv.bg * (1.0 - amp_img[..., None])
    cv.image(rgb, rc[0], rc[1], 0.5 * Rs, a_in)
    cv.text_px(rc + np.array([0.0, 0.5 * Rs + P.title_gap * cv.sc]), cv.tx["p1_right"], P.title_size, cv.color("text", a_in), va="bottom")
    # the scale bar: p r = 4
    half = CFG.partial_sums.half_width
    bar = P.scale_bar * (0.5 * Rs) / half
    y_bar = rc[1] - 0.5 * Rs - P.scale_gap * cv.sc
    cv.line([(rc[0] - 0.5 * Rs, y_bar), (rc[0] - 0.5 * Rs + bar, y_bar)], cv.color("dim", a_in), P.bar_width)
    cv.text_px((rc[0] - 0.5 * Rs + 0.5 * bar, y_bar - P.bar_label_gap * cv.sc), rf"$pr={int(P.scale_bar)}$", P.bar_size, cv.color("dim", a_in), va="top")
    # the counter and the arrow between the panels
    mid = 0.5 * (lc[0] + 0.5 * Ls + rc[0] - 0.5 * Rs)
    cv.arrow((mid - P.arrow_len * cv.sc * 0.5, rc[1]), (mid + P.arrow_len * cv.sc * 0.5, rc[1]), cv.color("dim", a_in), P.arrow_width, P.arrow_head)
    cv.text_px((mid, rc[1] + P.arrow_label_gap * cv.sc), cv.tx["p1_sum"], P.arrow_size, cv.color("text", a_in), va="bottom")
    count = rf"$N={N}$" if f_ex < 0.5 else r"$N\to\infty$"
    cv.text_px((rc[0], rc[1] - 0.5 * Rs - P.counter_drop * cv.sc), count, P.counter_size, cv.color("gold", a_in), va="top")
    # the legend and the notes
    phase_wheel(cv, cv.px(*P.wheel_center), P.wheel_radius * cv.H, delta, a_in)
    cv.text_px(cv.px(*P.wheel_center) + np.array([0.0, P.wheel_title_dy * cv.H]), cv.tx["p1_phase"], P.note_size, cv.color("text", a_in), va="bottom")
    for key, a, row in scheduled(P.notes, tl, CFG.layout.fade_s):
        cv.text(P.note_pos[0], P.note_pos[1] - P.note_row * row, cv.tx[key], P.note_size, cv.color("text", a), ha="left", va="top", maxw=P.note_maxw)
    formulas_and_captions(cv, P.formulas, P.captions, tl, LY.fade_s)


def scene_p2(cv: Canvas, tl: float, t: float) -> None:
    P, F, LY, S = CFG.p2, CFG.fonts, CFG.layout, CFG.style
    delta = hue_shift(t)
    side = P.cell_side * cv.H
    size = int(round(side))
    theta = math.radians(segments(tl, P.sweep))
    probe_a = smooth(tl, *P.probe_in)
    for k, col in enumerate(P.columns):
        a = smooth(tl, col["appear"], col["appear"] + P.appear_s)
        if a <= 0.0:
            continue
        r1, r2 = col["rho1"], col["rho2"]
        m = -(r1 + r2) // 2
        cx = P.columns_x[k] * cv.W
        geom = CellGeom(cx, P.cell_y * cv.H, 0.5 * side, CFG.iso.view)
        draw_cell(cv, geom, iso_image("S", r1, r2, size), delta, a)
        if m != 0:
            winding_arc(cv, geom, m, a * smooth(tl, *P.arcs_in))
        if probe_a > 0.0:
            probe(cv, geom, theta, probe_a)
        cv.text(P.columns_x[k] + P.header_dx, P.header_y, cv.tx[col["header"]], F.header, cv.color("text", a), ha="left")
        spin_glyph(cv, cv.px(P.columns_x[k] + P.glyph_dx, P.header_y + P.glyph_dy), col["up1"], col["up2"], a)
        if probe_a > 0.0:
            f = 0.5 * (1.0 + r1 * r2 * (1.0 - 2.0 * math.cos(theta) ** 2))
            th = np.linspace(0.0, 0.5 * math.pi, CFG.plot.samples)
            fx = 0.5 * (1.0 + r1 * r2 * (1.0 - 2.0 * np.cos(th) ** 2))
            rect = (P.columns_x[k] - 0.5 * P.plot_w, P.plot_y0, P.columns_x[k] + 0.5 * P.plot_w, P.plot_y1)
            mp = mini_plot(cv, rect, (0.0, 0.5 * math.pi), (0.0, 1.0), [(th, fx, cv.color("blue" if r1 * r2 < 0 else "warm"), None)], probe_a,
                           [(0.0, "$0$"), (0.5 * math.pi, r"$\pi/2$")], [(0.0, "$0$"), (1.0, "$1$")], xlabel=r"$\theta$")
            dx, dy = mp(theta, f)
            cv.ax.scatter([dx], [dy], s=CFG.plot.dot * cv.sc ** 2, c=[cv.color("gold", probe_a)], linewidths=0, zorder=9)
            value_label(cv, dx, dy, theta, rf"${num(f, '.2f', cv.lang)}$", probe_a)
    if probe_a > 0.05:
        cv.text(P.plot_title_x, P.plot_title_y, cv.tx["p2_plot"], F.header_small, cv.color("dim", probe_a), ha="left")
    if probe_a > 0.0:
        cv.text_px(cv.px(*P.theta_pos), rf"$\theta={int(round(math.degrees(theta)))}^\circ$", F.theta, cv.color("gold", probe_a), ha="left")
    phase_wheel(cv, cv.px(*P.wheel_center), P.wheel_radius * cv.H, delta, smooth(tl, *P.wheel_in))
    cv.text_px(cv.px(*P.wheel_center) + np.array([0.0, P.wheel_title_dy * cv.H]), cv.tx["p2_phase"], F.note, cv.color("text", smooth(tl, *P.wheel_in)), va="bottom")
    formulas_and_captions(cv, P.formulas, P.captions, tl, LY.fade_s)


def scene_p3(cv: Canvas, tl: float, t: float) -> None:
    P, F, LY = CFG.p3, CFG.fonts, CFG.layout
    delta = hue_shift(t)
    side = CFG.p2.cell_side * cv.H
    size = int(round(side))
    theta = math.radians(segments(tl, P.sweep))
    morph = smooth(tl, *P.morph)                         # 0: scalar, 1: pseudoscalar
    probe_a = smooth(tl, *P.probe_in)
    for k, col in enumerate(CFG.p2.columns):
        r1, r2 = col["rho1"], col["rho2"]
        cx = CFG.p2.columns_x[k] * cv.W
        geom = CellGeom(cx, CFG.p2.cell_y * cv.H, 0.5 * side, CFG.iso.view)
        m = -(r1 + r2) // 2
        if morph < 1.0:
            draw_cell(cv, geom, iso_image("S", r1, r2, size), delta, 1.0 - morph)
            if m != 0:
                winding_arc(cv, geom, m, (1.0 - morph))
        if morph > 0.0:
            if r1 == -r2:
                draw_cell(cv, geom, iso_image("P", r1, r2, size), delta, morph, overlays=morph > 0.99)
            else:
                cv.text_px((geom.cx, geom.cy), cv.tx["p3_zero"], F.zero, cv.color("dim", morph))
        if probe_a > 0.0:
            probe(cv, geom, theta, probe_a)
        hdr_s, hdr_p = cv.tx[col["header"]], cv.tx[col["header_p"]]
        cv.text(CFG.p2.columns_x[k] + CFG.p2.header_dx, CFG.p2.header_y, hdr_s, F.header, cv.color("text", 1.0 - morph), ha="left")
        cv.text(CFG.p2.columns_x[k] + CFG.p2.header_dx, CFG.p2.header_y, hdr_p, F.header, cv.color("text", morph), ha="left")
        spin_glyph(cv, cv.px(CFG.p2.columns_x[k] + CFG.p2.glyph_dx, CFG.p2.header_y + CFG.p2.glyph_dy), col["up1"], col["up2"], 1.0)
        th = np.linspace(0.0, 0.5 * math.pi, CFG.plot.samples)
        fs = 0.5 * (1.0 + r1 * r2 * (1.0 - 2.0 * np.cos(th) ** 2))
        fp = np.full_like(th, 0.5 * (1.0 - r1 * r2))
        fx = (1.0 - morph) * fs + morph * fp
        f_now = float((1.0 - morph) * 0.5 * (1.0 + r1 * r2 * (1.0 - 2.0 * math.cos(theta) ** 2)) + morph * 0.5 * (1.0 - r1 * r2))
        rect = (CFG.p2.columns_x[k] - 0.5 * CFG.p2.plot_w, CFG.p2.plot_y0, CFG.p2.columns_x[k] + 0.5 * CFG.p2.plot_w, CFG.p2.plot_y1)
        mp = mini_plot(cv, rect, (0.0, 0.5 * math.pi), (0.0, 1.0), [(th, fx, cv.color("blue" if r1 * r2 < 0 else "warm"), None)], 1.0,
                       [(0.0, "$0$"), (0.5 * math.pi, r"$\pi/2$")], [(0.0, "$0$"), (1.0, "$1$")], xlabel=r"$\theta$")
        dx, dy = mp(theta, f_now)
        cv.ax.scatter([dx], [dy], s=CFG.plot.dot * cv.sc ** 2, c=[cv.color("gold", 1.0)], linewidths=0, zorder=9)
        value_label(cv, dx, dy, theta, rf"${num(f_now, '.2f', cv.lang)}$", 1.0)
    cv.text(CFG.p2.plot_title_x, CFG.p2.plot_title_y, cv.tx["p2_plot"], F.header_small, cv.color("dim"), ha="left")
    cv.text_px(cv.px(*CFG.p2.theta_pos), rf"$\theta={int(round(math.degrees(theta)))}^\circ$", F.theta, cv.color("gold"), ha="left")
    phase_wheel(cv, cv.px(*CFG.p2.wheel_center), CFG.p2.wheel_radius * cv.H, delta, 1.0)
    cv.text_px(cv.px(*CFG.p2.wheel_center) + np.array([0.0, CFG.p2.wheel_title_dy * cv.H]), cv.tx["p2_phase"], F.note, cv.color("text"), va="bottom")
    formulas_and_captions(cv, P.formulas, P.captions, tl, LY.fade_s)


def scene_p4(cv: Canvas, tl: float, t: float) -> None:
    P, F, LY, S = CFG.p4, CFG.fonts, CFG.layout, CFG.style
    delta = hue_shift(t)
    side = P.cell_side * cv.H
    size = int(round(side))
    ratio = segments(tl, P.ramp)
    e1, e2 = eps_of(ratio)
    a_l = smooth(tl, *P.left_in)
    a_r = smooth(tl, *P.right_in)
    geom_l = CellGeom(P.left_x * cv.W, P.cell_y * cv.H, 0.5 * side, CFG.iso.mix_view)
    geom_r = CellGeom(P.right_x * cv.W, P.cell_y * cv.H, 0.5 * side, CFG.iso.mix_view)
    img_l = iso_image("mix", 1, 1, size, ratio, False, disk=True)
    draw_cell(cv, geom_l, img_l, delta, a_l)
    if a_r > 0.0:
        draw_cell(cv, geom_r, iso_image("mix", 1, 1, size, ratio, True, disk=True), delta, a_r)
    for g, a, key, swap in ((geom_l, a_l, "p4_state", False), (geom_r, a_r, "p4_swapped", True)):
        if a > 0.0:
            axis_arrows(cv, g, swap, a * smooth(tl, *P.plot_in))
            cv.text(g.cx / cv.W + P.header_dx, P.header_y, cv.tx[key], F.header, cv.color("text", a), ha="center")
            triad(cv, cv.px(g.cx / cv.W + P.triad_dx, P.header_y + P.triad_dy), swap, a)
    # the mixing readout
    a_mix = smooth(tl, *P.mix_in)
    if a_mix > 0.0:
        cv.text_px(cv.px(*P.mix_pos), rf"$\varepsilon_2/\varepsilon_1={num(ratio, '.2f', cv.lang)}$",
                   F.header, cv.color("gold", a_mix), ha="left")
    # the density on the axis
    a_p = smooth(tl, *P.plot_in)
    if a_p > 0.0:
        e1f, e2f = eps_of(P.ramp[-1][3])
        x = np.linspace(0.0, P.plot_xmax, CFG.plot.samples)
        a_ = e1f * BETA * fb.j1(x)
        b_ = e2f * fb.j0(x)
        y_add, y_sub, y_avg = 0.5 * (a_ + b_) ** 2, 0.5 * (a_ - b_) ** 2, 0.5 * (a_ ** 2 + b_ ** 2)
        ymax = float(y_add.max()) * P.plot_ymargin
        rect = (P.plot_rect[0], P.plot_rect[1], P.plot_rect[2], P.plot_rect[3])
        ticks_x = [(float(v), f"${int(v)}$") for v in P.plot_xticks]
        mp = mini_plot(cv, rect, (0.0, P.plot_xmax), (0.0, ymax), [(x, y_add, cv.color("warm"), None), (x, y_sub, cv.color("blue"), None),
                                                                (x, y_avg, cv.color("dim", CFG.plot.avg_alpha), CFG.plot.dash)], a_p,
                       ticks_x, [(0.0, "$0$")], xlabel=r"$pr$")
        # where the amplitudes are equal: the cancellation
        sgn = np.sign(b_ - a_)
        eq = x[int(np.argmax(sgn[1:] != sgn[0])) + 1]            # the first radius where the amplitudes are equal
        gx, gy = mp(eq, 0.0)
        cv.ax.scatter([gx], [gy], s=CFG.plot.dot * cv.sc ** 2, c=[cv.color("gold", a_p)], linewidths=0, zorder=9)
        cv.text_px((gx + P.cancel_dx * cv.sc, gy + P.cancel_dy * cv.sc), cv.tx["p4_cancel"], F.note, cv.color("gold", a_p), ha="left", va="bottom")
        ly = P.plot_rect[3] + P.legend_dy
        for k, (key, col) in enumerate((("p4_lg_add", "warm"), ("p4_lg_sub", "blue"), ("p4_lg_avg", "dim"))):
            yy = ly - k * P.legend_row
            cv.line([cv.px(P.legend_x, yy), cv.px(P.legend_x + P.legend_line, yy)], cv.color(col, a_p), CFG.plot.curve_width,
                    dash=CFG.plot.dash if key == "p4_lg_avg" else None)
            cv.text(P.legend_x + P.legend_line + P.legend_gap, yy, cv.tx[key], F.note, cv.color("text", a_p), ha="left", maxw=P.legend_maxw)
    phase_wheel(cv, cv.px(*P.wheel_center), P.wheel_radius * cv.H, delta, 1.0)
    cv.text_px(cv.px(*P.wheel_center) + np.array([0.0, P.wheel_title_dy * cv.H]), cv.tx["p2_phase"], F.note, cv.color("text"), va="bottom")
    formulas_and_captions(cv, P.formulas, P.captions, tl, LY.fade_s)


def axis_arrows(cv: Canvas, geom: CellGeom, swapped: bool, alpha: float) -> None:
    """The two directions of the plot: along xi_2 x xi_1 (the waves add, warm) and opposite to it (they cancel, blue)."""
    O = CFG.overlay
    cam = iso_camera()
    v1, v2 = np.array(fb.unit(CFG.physics.mixed_xi1)), np.array(fb.unit(CFG.physics.mixed_xi2))
    if swapped:
        v1, v2 = v2, v1
    n = fb.unit(np.cross(v2, v1))
    o = geom.to_px(ir.project(np.zeros(3), cam))
    for sign, name in ((1.0, "warm"), (-1.0, "blue")):
        tip = geom.to_px(ir.project(sign * O.axis_arrow_length * n, cam))
        cv.arrow(o, tip, cv.color(name, alpha), O.probe_width, O.probe_head, zorder=8)


def triad(cv: Canvas, c: np.ndarray, swapped: bool, alpha: float) -> None:
    """The spin vectors xi_1 (blue) and xi_2 (warm) of the mixed state seen with the camera of the cells."""
    G = CFG.glyph
    cam = iso_camera()
    v1, v2 = np.array(fb.unit(CFG.physics.mixed_xi1)), np.array(fb.unit(CFG.physics.mixed_xi2))
    if swapped:
        v1, v2 = v2, v1
    for v, name, key in ((v1, "blue", "xi1"), (v2, "warm", "xi2")):
        tip = c + ir.project(v, cam) * G.triad_scale * cv.sc
        cv.arrow(c, tip, cv.color(name, alpha), G.width, G.head)
        cv.text_px(tip + (tip - c) / max(float(np.hypot(*(tip - c))), 1e-9) * G.triad_label_gap * cv.sc, cv.tx[key], G.label_size, cv.color(name, alpha))


def scene_p5(cv: Canvas, tl: float, t: float) -> None:
    P, F, LY = CFG.p5, CFG.fonts, CFG.layout
    delta = hue_shift(t)
    a_in = smooth(tl, *P.panels_in)
    Ls = P.side * cv.H
    sums = plane_wave_sum()
    pts = sums.n[:P.dots]
    w = pts[:, 0] + 1j * pts[:, 1]
    amp = np.clip(np.abs(w), 0.0, 1.0)
    f_gray = smooth(tl, *P.gray)
    for key, cxy, gray in (("p5_left", P.left_center, False), ("p5_right", P.right_center, True)):
        c = cv.px(*cxy)
        R = 0.5 * Ls * CFG.p1.disc_fraction
        cv.ax.add_patch(cv.plt.Circle(c, R, fill=False, edgecolor=cv.color("dim", CFG.p1.disc_alpha * a_in), lw=CFG.p1.disc_width * cv.sc, zorder=3))
        cols = palette(np.angle(w), delta)
        if gray:
            lum = P.gray_level * (amp ** 2) ** P.gamma
            cols = np.repeat(lum[:, None], 3, axis=1)
            alpha = a_in * (P.dot_alpha_min + (1 - P.dot_alpha_min) * amp ** 2)
        else:
            alpha = a_in * (CFG.p1.dot_alpha_min + (1 - CFG.p1.dot_alpha_min) * amp)
        cv.ax.scatter(c[0] + pts[:, 0] * R, c[1] + pts[:, 1] * R, s=(P.dot_size * cv.sc ** 2) * (CFG.p1.dot_scale_min + (1 - CFG.p1.dot_scale_min) * amp),
                      c=np.column_stack([cols, alpha]), linewidths=0, zorder=5)
        cv.text_px(c + np.array([0.0, R + CFG.p1.title_gap * cv.sc]), cv.tx[key], CFG.p1.title_size, cv.color("text", a_in), va="bottom")
    mid = 0.5 * (cv.px(*P.left_center)[0] + cv.px(*P.right_center)[0])
    cv.arrow((mid - P.arrow_len * cv.sc * 0.5, cv.px(*P.left_center)[1]), (mid + P.arrow_len * cv.sc * 0.5, cv.px(*P.left_center)[1]), cv.color("dim", a_in), CFG.p1.arrow_width, CFG.p1.arrow_head)
    cv.text_px((mid, cv.px(*P.left_center)[1] + CFG.p1.arrow_label_gap * cv.sc), cv.tx["p5_proj"], CFG.p1.arrow_size, cv.color("text", a_in), va="bottom")
    phase_wheel(cv, cv.px(*P.wheel_center), P.wheel_radius * cv.H, delta, a_in * (1.0 - 0.0 * f_gray))
    formulas_and_captions(cv, P.formulas, P.captions, tl, LY.fade_s)


SCENES = [scene_p1, scene_p2, scene_p3, scene_p4, scene_p5]
BOUNDS = [T_P1, T_P2, T_P3, T_P4, T_P5]


def draw_titles(cv: Canvas, t: float) -> None:
    F, LY, S = CFG.fonts, CFG.layout, CFG.style
    cv.text(LY.title_pos[0], LY.title_pos[1], cv.tx["title"], F.title, cv.color("title_color"), maxw=LY.title_maxw)
    ch = chapter_of(t)
    if ch >= 1:
        cv.text(LY.chapter_label_pos[0], LY.chapter_label_pos[1], f"{ch}/{len(CARD_AT) - 1}   " + cv.tx[CARD_KEYS[ch]], F.chapter_label,
                cv.color("dim", S.chapter_label_alpha), ha="right")


def draw_card(cv: Canvas, card: int, cprog: float) -> None:
    F, LY, S = CFG.fonts, CFG.layout, CFG.style
    a_c = min(smooth(cprog, *CFG.timeline.card_fade_in), 1.0 - smooth(cprog, *CFG.timeline.card_fade_out))
    if card == 0:
        cv.text(0.5, LY.card_title_y, cv.tx["h0"], F.card_title, cv.color("card_title_color", a_c), ha="center", maxw=LY.card_maxw)
        cv.text(0.5, LY.card_subtitle_y, cv.tx["h0s"], F.card_subtitle, cv.color("dim", a_c), ha="center")
    else:
        cv.text(0.5, LY.card_number_y, f"{card}", F.card_number, cv.color("gold", S.card_number_alpha * a_c), ha="center")
        cv.text(0.5, LY.card_chapter_y, cv.tx[CARD_KEYS[card]], F.card_chapter, cv.color("card_title_color", a_c), ha="center", maxw=LY.card_maxw)
        cv.text(0.5, LY.card_chapter_sub_y, cv.tx[CARD_KEYS[card] + "s"], F.card_chapter_sub, cv.color("dim", a_c), ha="center", maxw=LY.card_maxw)


def draw_frame(cv: Canvas, tf_nom: float) -> None:
    """Draw the frame at the nominal film time tf_nom (cards included) into the canvas."""
    t, card, cprog = timeline(tf_nom)
    cv.reset()
    if card is not None:
        draw_card(cv, card, cprog)
        return
    draw_titles(cv, t)
    for scene, (a, b) in zip(SCENES, BOUNDS):
        if a <= t < b:
            scene(cv, t - a, t)
            return


def render(out: Path, size: tuple[int, int], fps: int, total: float, lang: str, snap: float | None = None) -> None:
    if snap is None and not shutil.which("ffmpeg"):
        sys.exit("ffmpeg is required")
    V = CFG.video
    W, H = size
    cv = Canvas(size, lang)
    k = total / TOTAL
    frames = int(round(total * fps))
    ids = [int(round(snap * fps))] if snap is not None else range(frames)
    writer = None
    if snap is None:
        writer = subprocess.Popen(
            ["ffmpeg", "-y", "-v", "error", "-f", "rawvideo", "-pix_fmt", "rgba", "-s", f"{W}x{H}",
             "-r", str(fps), "-i", "-", "-c:v", "libx264", "-preset", V.preset, "-crf", str(V.crf),
             "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(out)], stdin=subprocess.PIPE)
    bg = np.append(cv.bg * 255.0, 255.0).astype(np.float32)
    for k_ in ids:
        t_film = k_ / fps
        draw_frame(cv, t_film / k)
        fade_io = min(smooth(t_film, 0.0, V.fade_s), 1.0 - smooth(t_film, total - V.fade_s, total))
        cv.fig.canvas.draw()
        frame = np.asarray(cv.fig.canvas.buffer_rgba()).astype(np.float32)
        if fade_io < 1.0:
            frame = bg + (frame - bg) * fade_io
        frame = frame.clip(0, 255).astype(np.uint8)
        if writer is None:
            from PIL import Image
            Image.fromarray(frame).save(out)
        else:
            writer.stdin.write(frame.tobytes())
    if writer is not None:
        writer.stdin.close()
        writer.wait()
    cv.plt.close(cv.fig)
    print(f"wrote {out}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--lang", choices=("en", "ru"), default="en")
    ap.add_argument("--preview", action="store_true")
    ap.add_argument("--snapshot", type=float, default=None, help="one PNG at this film time (s)")
    ap.add_argument("--seconds", type=float, default=TOTAL)
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--config", type=Path, default=None, help="another configuration file instead of config.toml")
    ap.add_argument("--set", action="append", default=[], metavar="SECTION.KEY=VALUE", help="override one configuration value")
    args = ap.parse_args()
    V = CFG.video
    out = args.out or HERE / "media" / f"higgs_ff_spin_{args.lang}.mp4"
    out.parent.mkdir(parents=True, exist_ok=True)
    if args.snapshot is not None:
        render(out.with_suffix(".png"), (V.width, V.height), V.fps, args.seconds, args.lang, snap=args.snapshot)
    elif args.preview:
        render(out.with_name(out.stem + "_preview.mp4"), (V.preview_width, V.preview_height), V.preview_fps, args.seconds, args.lang)
    else:
        render(out, (V.width, V.height), V.fps, args.seconds, args.lang)


if __name__ == "__main__":
    main()
