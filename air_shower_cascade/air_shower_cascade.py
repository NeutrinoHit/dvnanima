r"""A cosmic-ray air shower: the Heitler-Matthews toy model as a Monte-Carlo cascade.

Parts of the film (the physics is in shower_model.py, see its docstring; the numbers are in config.toml, the words in texts.toml):
 1. Heitler's electromagnetic cascade: one photon, after d = X_0 ln 2 every particle splits in two with half the energy,
    N = 2^n, E = E_0/2^n, until E = E_c: N_max = E_0/E_c at X_max = X_0 ln(E_0/E_c);
 2. a cosmic proton of E_0 = 10^15 eV: the hadronic cascade (pions, pi0 -> two photons feed electromagnetic cascades, charged
    pions of E <= E_dec decay into muons and neutrinos) as a Monte Carlo with a fixed seed; the longitudinal profile N(X) builds
    up in real time, the Heitler estimates are marked;
 3. on the ground: a detector array (hits per station, lateral distribution of e+- and muons) and a short card on the light
    of the shower (fluorescence, Cherenkov).

Everything schematic is labelled in the film: the lateral scale is exaggerated, the drawn tracks are a random sample.

Usage:
    python air_shower_cascade.py --lang en            # film -> media/air_shower_cascade_en.mp4
    python air_shower_cascade.py --lang ru
    python air_shower_cascade.py --lang en --snapshot 40
"""

from __future__ import annotations

import argparse
import math
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass
from fractions import Fraction
from functools import lru_cache
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dvconfig import load_config, load_texts  # noqa: E402

import shower_model as sm  # noqa: E402

HERE = Path(__file__).resolve().parent
CFG = load_config(HERE)
U = CFG.units
PH = CFG.physics

# ----------------------------------------------------------------------------------------------- timeline (content time and film time)
CARD_S = CFG.timeline.card_s
TITLE_CARD_S = CFG.timeline.title_card_s
CARD_AT = list(CFG.timeline.card_at)
CARD_KEYS = [f"h{i}" for i in range(len(CARD_AT))]
_PB = list(CFG.timeline.part_bounds)
CONTENT_TOTAL = _PB[-1]
TOTAL = CONTENT_TOTAL + TITLE_CARD_S + CARD_S * (len(CARD_AT) - 1)
T_TREE, T_SHOWER, T_GROUND, T_LIGHT = ((_PB[i], _PB[i + 1]) for i in range(4))


def card_len(i: int) -> float:
    return TITLE_CARD_S if i == 0 else CARD_S


def timeline(tf: float) -> tuple[float, int | None, float]:
    """(content time, index of the chapter card or None, progress of the card in [0, 1]) at the film time tf."""
    done = 0.0
    for i, ca in enumerate(CARD_AT):
        a = ca + done
        if a <= tf < a + card_len(i):
            return ca, i, (tf - a) / card_len(i)
        if tf >= a + card_len(i):
            done += card_len(i)
    return tf - done, None, 0.0


def film_time(tc: float) -> float:
    return tc + sum(card_len(i) for i, ca in enumerate(CARD_AT) if ca <= tc)


def chapter_of(tc: float) -> int:
    """1 = the toy cascade, 2 = the shower, 3 = the ground and the light."""
    if tc < T_SHOWER[0]:
        return 1
    return 2 if tc < T_GROUND[0] else 3


def smooth(x: float, a: float, b: float) -> float:
    u = min(max((x - a) / (b - a), 0.0), 1.0)
    return u * u * (3 - 2 * u)


# ----------------------------------------------------------------------------------------------- texts
def num(x: float, fmt: str, lang: str) -> str:
    s = format(x, fmt)
    return s.replace(".", "{,}") if lang == "ru" else s


def sci(x: float, lang: str, digits: int = 1) -> str:
    """x as a mathtext fragment (no dollars): 1.2\\times10^{7}."""
    if x <= 0:
        return "0"
    e = int(math.floor(math.log10(x)))
    m = x / 10.0 ** e
    if round(m, digits) >= 10.0:
        m, e = m / 10.0, e + 1
    if round(m, digits) == 1.0:
        return rf"10^{{{e}}}"
    return rf"{num(m, f'.{digits}f', lang)}\times10^{{{e}}}"


def sci_int(x: float, lang: str) -> str:
    """An integer below 100 as it is, larger numbers in the powers of ten (one digit)."""
    return f"{int(round(x))}" if x < 100.0 else sci(x, lang, 0)


TEXT = {lang: load_texts(HERE, lang) for lang in ("en", "ru")}


def fill(s: str, **kw: str) -> str:
    """Replace <key> placeholders (the braces of mathtext stay untouched)."""
    for k, v in kw.items():
        s = s.replace(f"<{k}>", v)
    return s


# ----------------------------------------------------------------------------------------------- the model
def make_params(**over) -> sm.Params:
    d = dict(e0_ev=PH.e0_ev, x0=PH.radiation_length, e_c_ev=PH.critical_energy_ev, lam_i=PH.interaction_length, n_ch=PH.n_charged,
             e_dec_ev=PH.decay_energy_ev, lam_dec=PH.decay_length, x_ground=PH.ground_depth, h_km=PH.scale_height_km,
             pt_ev=PH.mean_pt_ev, e_ms_ev=PH.scattering_energy_ev, m_pion_ev=PH.pion_mass_ev, m_muon_ev=PH.muon_mass_ev,
             e_thin_ev=PH.thin_energy_ev, path=PH.path, x_sea=PH.sea_level_depth)
    d.update(over)
    return sm.Params(**d)


@dataclass
class Model:
    P: sm.Params
    sh: sm.Shower
    grid: np.ndarray
    n_em: np.ndarray
    n_mu: np.ndarray
    n_had: np.ndarray
    n_em_drawn: np.ndarray
    x_first: float
    x_max: float
    n_max: float
    x_max_h: float                     # Heitler-Matthews estimate of X_max of the proton shower
    n_max_h: float                     # Heitler: E_0/E_c
    analytic: dict
    s: dict                            # the drawn sample, sorted by the start depth
    glow: np.ndarray                   # (depth, lateral) track density of e+-
    glow_view: np.ndarray              # the brightness of the glow image in [0, 1]
    glow_lat: tuple[float, float]
    m_levels: int


def _glow_image(sh: sm.Shower, half_m: float) -> tuple[np.ndarray, tuple[float, float]]:
    """Track density of the e+- in (depth, lateral x) from the weighted tracks: three points per track."""
    tr = sh.tr
    m = tr["kind"] == sm.ELECTRON
    x0, x1, w = tr["x0"][m], tr["x1"][m], tr["w"][m]
    p0, p1 = tr["px0"][m], tr["px1"][m]
    db, dd = CFG.mc.glow_bins[0], CFG.mc.glow_bins[1]
    lat_edges = np.arange(-half_m, half_m + db, db)
    dep_edges = np.arange(0.0, sh.p.x_ground + dd, dd)
    pts = [(1.0 / 6.0), (3.0 / 6.0), (5.0 / 6.0)]
    depth = np.concatenate([x0 + t * (x1 - x0) for t in pts])
    lat = np.concatenate([p0 + t * (p1 - p0) for t in pts])
    wt = np.concatenate([w * (x1 - x0) / len(pts) for _ in pts])
    img, _, _ = np.histogram2d(depth, lat, bins=[dep_edges, lat_edges], weights=wt)
    from scipy.ndimage import gaussian_filter
    img = gaussian_filter(img, sigma=(CFG.mc.glow_sigma_bins[1], CFG.mc.glow_sigma_bins[0]))
    return img / img.max(), (float(lat_edges[0]), float(lat_edges[-1]))


def _glow_view(glow: np.ndarray) -> np.ndarray:
    """Brightness: the lateral shape of every depth row (normalised to its maximum, power gamma_shape) times the number of
    particles of the row relative to the maximum (power gamma_depth): the cone is visible at every depth, the brightness
    follows the longitudinal development."""
    C = CFG.column
    row_max = glow.max(axis=1, keepdims=True)
    shape = (glow / np.maximum(row_max, 1e-300)) ** C.glow_gamma_shape
    row_n = glow.sum(axis=1, keepdims=True)
    return shape * (row_n / row_n.max()) ** C.glow_gamma_depth

@lru_cache(maxsize=None)
def get_model() -> Model:
    P = make_params()
    dx = CFG.mc.profile_dx
    sh = sm.simulate(P, CFG.mc.seed, dx)
    spec = sm.DrawSpec(CFG.mc.leaf_pions, tuple(CFG.mc.seed_fraction), tuple(CFG.mc.tree_tracks), CFG.mc.draw_seed)
    s = sm.draw_sample(sh, spec)
    order = np.argsort(s["x0"], kind="stable")
    s = {k: v[order] for k, v in s.items()}
    grid = sh.grid
    n_em = sh.count((sm.GAMMA, sm.ELECTRON))
    em_s = np.isin(s["kind"], (sm.GAMMA, sm.ELECTRON))
    n_drawn = sm.count_profile(s["x0"][em_s], s["x1"][em_s], np.ones(int(em_s.sum())), s["ground"][em_s], grid)
    x_max, n_max = sh.peak()
    half = CFG.column.lateral_half_m
    glow, lat = _glow_image(sh, half)
    view = _glow_view(glow)
    return Model(P=P, sh=sh, grid=grid, n_em=n_em, n_mu=sh.count((sm.MUON,)), n_had=sh.count((sm.PROTON, sm.PION)), n_em_drawn=n_drawn,
                 x_first=sh.x_first(), x_max=x_max, n_max=n_max, x_max_h=sm.matthews_x_max_proton(P, P.lam_i),
                 n_max_h=sm.heitler_n_max(P.e0_ev, P.e_c_ev), analytic=sm.heitler_matthews_profile(P, grid), s=s, glow=glow, glow_view=view,
                 glow_lat=lat, m_levels=sm.hadron_levels(P.e0_ev, P.e_dec_ev, P.n_ch))


# ----------------------------------------------------------------------------------------------- the canvas
class Canvas:
    """The figure, three axes (their geometry is set by the part that is drawn) and an overlay in figure fractions."""

    def __init__(self, W: int, H: int, lang: str) -> None:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        self.plt = plt
        V, S = CFG.video, CFG.style
        self.W, self.H, self.lang = W, H, lang
        self.dpi = V.dpi
        self.sc = H / V.reference_height
        self.px = 72.0 / self.dpi * self.sc                  # points per pixel (lines and markers given in px)
        self.tx = TEXT[lang]
        self.bg = S.background
        self.fig = plt.figure(figsize=(W / self.dpi, H / self.dpi), dpi=self.dpi, facecolor=self.bg)
        self.a = self.fig.add_axes([0, 0, 1, 1], facecolor="none")
        self.b = self.fig.add_axes([0, 0, 1, 1], facecolor="none")
        self.c = self.fig.add_axes([0, 0, 1, 1], facecolor="none")
        self.ov = self.fig.add_axes([0, 0, 1, 1], facecolor="none", zorder=20)
        self.bg_rgba = np.array([int(self.bg[1:3], 16), int(self.bg[3:5], 16), int(self.bg[5:7], 16), 255], np.float32)

    def reset(self) -> None:
        self.fig.texts.clear()
        for ax in (self.a, self.b, self.c, self.ov):
            ax.clear()
            ax.set_facecolor("none")
            ax.axis("off")
        self.ov.set_xlim(0, 1)
        self.ov.set_ylim(0, 1)

    def place(self, ax, rect) -> None:
        ax.set_position(rect)

    def text(self, x: float, y: float, s: str, size: float, color, **kw):
        kw.setdefault("ha", "left")
        kw.setdefault("va", "baseline")
        return self.fig.text(x, y, s, color=color, fontsize=size * self.sc, **kw)

    def renderer(self):
        return self.fig.canvas.get_renderer()

    def width(self, s: str, size: float, **kw) -> float:
        """Width of a text in figure fractions, measured."""
        t = self.fig.text(0, 0, s, fontsize=size * self.sc, **kw)
        w = t.get_window_extent(self.renderer()).width / self.W
        t.remove()
        return w

    def wrap(self, s: str, size: float, max_w: float) -> list[str]:
        """Break a text into the fewest lines that fit; breaks only at spaces outside $...$; the lines are balanced."""
        toks, cur, inmath = [], "", False
        for ch in s:
            if ch == "$":
                inmath = not inmath
            if ch == " " and not inmath:
                toks.append(cur)
                cur = ""
            else:
                cur += ch
        toks.append(cur)
        if self.width(s, size) <= max_w:
            return [s]
        best = None
        for k in range(1, len(toks)):
            a, b = " ".join(toks[:k]), " ".join(toks[k:])
            w = max(self.width(a, size), self.width(b, size))
            if best is None or w < best[0]:
                best = (w, [a, b])
        if best is not None and best[0] <= max_w:
            return best[1]
        lines, cur = [], ""
        for tk in toks:
            trial = (cur + " " + tk) if cur else tk
            if cur and self.width(trial, size) > max_w:
                lines.append(cur)
                cur = tk
            else:
                cur = trial
        lines.append(cur)
        return lines

    def caption(self, s: str, alpha: float = 1.0) -> None:
        L, F, S = CFG.layout, CFG.fonts, CFG.style
        lines = self.wrap(s, F.caption, L.caption_max_width)
        for i, line in enumerate(reversed(lines)):
            self.text(L.caption_pos[0], L.caption_pos[1] + i * L.caption_line_gap, line, F.caption, (*S.text, S.caption_alpha * alpha))

    def style_axes(self, ax, left: bool = True, bottom: bool = False) -> None:
        S = CFG.style
        ax.axis("on")
        for k, sp in ax.spines.items():
            sp.set_visible((k == "left" and left) or (k == "bottom" and bottom))
            sp.set_color(tuple(S.spine_colour))
            sp.set_linewidth(S.spine_width * self.sc)
        ax.tick_params(colors=tuple(S.tick_colour), labelsize=CFG.fonts.tick * self.sc, length=S.tick_length * self.sc, width=S.spine_width * self.sc)
        ax.patch.set_visible(False)
        if self.lang == "ru":
            from matplotlib.ticker import FuncFormatter
            fmt = FuncFormatter(lambda v, _: f"{v:g}".replace(".", ",").replace("-", "−"))
            ax.xaxis.set_major_formatter(fmt)
            ax.yaxis.set_major_formatter(fmt)

    def frame_out(self, fade: float) -> np.ndarray:
        self.fig.canvas.draw()
        frame = np.asarray(self.fig.canvas.buffer_rgba()).astype(np.float32)
        if fade < 1.0:
            frame = self.bg_rgba + (frame - self.bg_rgba) * fade
        return frame.clip(0, 255).astype(np.uint8)


# ----------------------------------------------------------------------------------------------- part 2: the shower
def kind_rgb() -> dict[int, tuple]:
    S = CFG.style
    return {sm.PROTON: tuple(S.primary), sm.PION: tuple(S.hadron), sm.GAMMA: tuple(S.gamma), sm.ELECTRON: tuple(S.electron),
            sm.MUON: tuple(S.muon), sm.NEUTRINO: tuple(S.neutrino)}


def front_schedule(M: Model) -> tuple[np.ndarray, np.ndarray]:
    """(times, depths) of the knots of the front of the shower: top of the picture, first interaction, maximum, ground."""
    st = CFG.scene2.stage_s
    ts = np.concatenate([[0.0], np.cumsum(st)])
    xs = np.array([-CFG.column.top_margin, M.x_first, M.x_first, M.x_max, M.x_max, M.P.x_ground, M.P.x_ground])
    return ts, xs


def front_depth(M: Model, t: float) -> float:
    ts, xs = front_schedule(M)
    return float(np.interp(t, ts, xs))


def caption_key(table: list, t: float) -> str:
    key = table[0][1]
    for a, k in table:
        if t >= a:
            key = k
    return key


def lateral_at(x0: np.ndarray, x1: np.ndarray, p0: np.ndarray, p1: np.ndarray, x: np.ndarray) -> np.ndarray:
    """Lateral coordinate of a track at the depth x (x0 <= x <= x1): the drift is proportional to ln(x / x0)."""
    xa = np.maximum(x0, sm.X_FLOOR)
    den = np.log(np.maximum(x1, xa * (1.0 + 1e-12)) / xa)
    return p0 + (p1 - p0) * np.clip(np.log(np.maximum(x, xa) / xa) / den, 0.0, 1.0)


def exaggeration(cv: Canvas, rect, ylim_span: float, half_m: float, depth: float, h_m: float) -> float:
    """Horizontal px per metre divided by the vertical px per metre at the given depth (X = X_g exp(-z/H): dX/dz = X/H)."""
    px_h = rect[2] * cv.W / (2.0 * half_m)
    px_v = rect[3] * cv.H / ylim_span * depth / h_m
    return px_h / px_v


def energy_scale(P: sm.Params, e: np.ndarray) -> np.ndarray:
    """0 for E = E_c, 1 for the energy of the photons of the first pi0 decays: log scale (the brightness of a track)."""
    e_top = P.e0_ev / (2.0 * P.n_sec)
    return np.clip(np.log(np.maximum(e, P.e_c_ev) / P.e_c_ev) / math.log(e_top / P.e_c_ev), 0.0, 1.0)


def stage_times() -> np.ndarray:
    return np.concatenate([[0.0], np.cumsum(CFG.scene2.stage_s)])


def draw_shower(cv: Canvas, ts: float, lang: str) -> None:
    """The shower at the content time ts of part 2: tracks, glow, profile, legend, readouts, rules."""
    from matplotlib.collections import LineCollection
    M = get_model()
    C, PR, SC2, L, F, S = CFG.column, CFG.profile, CFG.scene2, CFG.layout, CFG.fonts, CFG.style
    tx = cv.tx
    TXT, DIM = tuple(S.text), tuple(S.dim)
    rgb = kind_rgb()
    P = M.P
    xg = P.x_ground
    xf = front_depth(M, ts)
    half = C.lateral_half_m
    y_top, y_bot = -C.top_margin, xg + C.bottom_margin
    st = stage_times()
    # --------------------------------------------------------------- the column
    ax = cv.a
    cv.place(ax, L.column_axes)
    cv.style_axes(ax, left=True)
    ax.set_xlim(-half, half)
    ax.set_ylim(y_bot, y_top)
    ax.set_yticks(CFG.atmosphere.depth_ticks)
    ax.set_xticks([])
    for h in CFG.atmosphere.altitude_ticks_km:
        xh = float(sm.depth_of_altitude(P, h))
        if xh > xg:
            continue
        ax.plot([-half, half], [xh, xh], color=(1, 1, 1, C.altitude_line_alpha), lw=C.altitude_line_width * cv.sc, zorder=0)
        ax.plot([half, half * 1.03], [xh, xh], color=tuple(S.spine_colour), lw=S.spine_width * cv.sc, clip_on=False)
        ax.text(half * 1.05, xh, rf"${num(h, 'g', lang)}$", color=DIM, fontsize=F.tick * cv.sc, ha="left", va="center", clip_on=False)
    ax.text(half * 1.05, y_top - 1.8 * C.top_margin, tx["alt_unit"], color=DIM, fontsize=F.tick * cv.sc, ha="left", va="center", clip_on=False)
    ax.text(-half * 1.0, y_top - 1.8 * C.top_margin, tx["depth_unit"], color=DIM, fontsize=F.tick * cv.sc, ha="left", va="center", clip_on=False)
    # glow: the density of the e+-, up to the front
    nd, nl = M.glow.shape
    dd = CFG.mc.glow_bins[1]
    cdep = (np.arange(nd) + 0.5) * dd
    row_a = np.clip((xf - cdep) / C.glow_edge, 0.0, 1.0)
    row_a = row_a * row_a * (3 - 2 * row_a)
    img = np.zeros((nd, nl, 4))
    img[..., :3] = tuple(C.glow_colour)
    img[..., 3] = M.glow_view * C.glow_alpha * row_a[:, None]
    ax.imshow(img, extent=(M.glow_lat[0], M.glow_lat[1], nd * dd, 0.0), origin="upper", aspect="auto", interpolation="bilinear", zorder=1)
    ax.fill_between([-half, half], xg, y_bot, facecolor="none", edgecolor=(*DIM, C.ground_hatch_alpha), hatch="////", lw=0, zorder=1)
    ax.plot([-half, half], [xg, xg], color=(*TXT, 0.9), lw=C.ground_line_width * cv.sc, zorder=6)
    ax.text(half * 0.98, xg - C.ground_label_dy, fill(tx["ground_label"], x=num(xg, ".0f", lang), h=num(float(sm.altitude_km(P, xg)), ".1f", lang)), color=DIM,
            fontsize=F.tick * cv.sc, ha="right", va="bottom", zorder=8)
    # tracks
    s = M.s
    n = int(np.searchsorted(s["x0"], xf, side="left"))
    x0, x1, p0, p1, kd, en = s["x0"][:n], s["x1"][:n], s["px0"][:n], s["px1"][:n], s["kind"][:n], s["e"][:n]
    end = np.minimum(x1, xf)
    lat_end = lateral_at(x0, x1, p0, p1, end)
    done = x1 <= xf
    age = np.where(done, xf - x1, 0.0)
    floors = {sm.GAMMA: C.fade_floor_em, sm.ELECTRON: C.fade_floor_em, sm.PION: C.fade_floor_hadron, sm.PROTON: 1.0,
              sm.MUON: C.fade_floor_muon, sm.NEUTRINO: C.fade_floor_neutrino}
    widths = {sm.GAMMA: C.track_width_px, sm.ELECTRON: C.track_width_px, sm.PION: C.hadron_width_px, sm.PROTON: C.primary_width_px,
              sm.MUON: C.muon_width_px, sm.NEUTRINO: C.neutrino_width_px}
    scale = {sm.GAMMA: C.gamma_alpha_scale, sm.ELECTRON: C.em_alpha_scale, sm.PION: 1.0, sm.PROTON: 1.0, sm.MUON: 1.0, sm.NEUTRINO: 1.0}
    em_kinds = (sm.GAMMA, sm.ELECTRON)
    bright = energy_scale(P, en)                                            # the energy of a track encoded in brightness and width
    e_alpha = C.energy_alpha_min + (1.0 - C.energy_alpha_min) * bright ** C.energy_alpha_exp
    groups = [((sm.GAMMA, sm.ELECTRON), 3, None), ((sm.NEUTRINO,), 4, tuple(C.neutrino_dash)), ((sm.PION, sm.PROTON), 5, None), ((sm.MUON,), 4, None)]
    for kinds, z, dash in groups:
        for k in kinds:
            m = kd == k
            if not m.any():
                continue
            fl = floors[k]
            base = np.where(done[m], fl + (1.0 - fl) * np.exp(-age[m] / C.fade_tau), 1.0) * scale[k]
            if k in em_kinds:
                base = base * e_alpha[m]
                lw = widths[k] * (1.0 + C.energy_width_gain * bright[m]) * cv.px
            else:
                lw = widths[k] * cv.px
            segs = np.stack([np.column_stack([p0[m], x0[m]]), np.column_stack([lat_end[m], end[m]])], axis=1)
            cols = np.column_stack([np.tile(rgb[k], (int(m.sum()), 1)), base])
            kw = dict(linestyles=[(0, tuple(d * cv.px for d in dash))]) if dash else {}
            ax.add_collection(LineCollection(segs, colors=cols, linewidths=lw, capstyle="round", zorder=z, **kw))
    inter = (done & ((kd == sm.PROTON) | ((kd == sm.PION) & (en > SC2.vertex_min_energy_ev))))
    if inter.any():
        ax.scatter(lat_end[inter], end[inter], s=(SC2.vertex_size * cv.px) ** 2, c=[(*S.pi0, 0.9)], edgecolors="none", zorder=6)
    ax.plot([0.0, 0.0], [y_top, min(xf, 0.0)], color=(*rgb[sm.PROTON], 1.0), lw=C.primary_width_px * cv.px, solid_capstyle="round", zorder=5)
    act = ~done
    for k, size in ((sm.GAMMA, C.head_size), (sm.ELECTRON, C.head_size), (sm.PION, C.head_size_hadron), (sm.PROTON, C.head_size_hadron),
                    (sm.MUON, C.head_size_muon), (sm.NEUTRINO, C.head_size)):
        m = act & (kd == k)
        if m.any():
            a = C.head_alpha * (scale[k] * e_alpha[m] if k in em_kinds else np.ones(int(m.sum())))
            ax.scatter(lat_end[m], end[m], s=(size * cv.px) ** 2, c=np.column_stack([np.tile(rgb[k], (int(m.sum()), 1)), a]), edgecolors="none", zorder=7)
    if xf < 0.0:
        ax.scatter([0.0], [xf], s=(C.head_size_hadron * cv.px) ** 2, c=[(*rgb[sm.PROTON], 1.0)], edgecolors="none", zorder=7)
    gm = (s["kind"] == sm.MUON) & (xf >= xg)
    if gm.any():
        ax.scatter(s["px1"][gm], np.full(int(gm.sum()), xg), s=(C.head_size_muon * cv.px) ** 2, c=[(*rgb[sm.MUON], 0.9)], edgecolors="none", zorder=7)
    ax.plot([-half, half], [xf, xf], color=(*TXT, C.front_alpha), lw=C.front_width_px * cv.px, zorder=2)
    if st[1] <= ts < st[1] + SC2.flash_s:
        u = (ts - st[1]) / SC2.flash_s
        ax.scatter([0.0], [M.x_first], s=(2 * SC2.flash_radius_px * u * cv.px) ** 2, facecolors="none", edgecolors=[(*rgb[sm.PROTON], 1.0 - u)],
                   linewidths=1.6 * cv.sc, zorder=8)
    # scale bar and the exaggeration of the horizontal scale
    xbar0 = -half * C.scale_bar_x
    ybar = C.scale_bar_y
    ax.plot([xbar0, xbar0 + C.scale_bar_m], [ybar, ybar], color=(*TXT, 0.95), lw=C.scale_bar_width * cv.sc, solid_capstyle="butt", zorder=8)
    ax.text(xbar0 + C.scale_bar_m + C.scale_bar_gap_m, ybar, rf"${num(C.scale_bar_m, '.0f', lang)}\,$" + tx["m_unit"], color=TXT, fontsize=F.tick * cv.sc,
            ha="left", va="center", zorder=8)
    ex = exaggeration(cv, L.column_axes, y_bot - y_top, half, M.x_max, P.h_m)
    if ts >= SC2.exagg_label_at:
        ax.text(xbar0, ybar + C.exagg_gap, fill(tx["exagg"], k=num(round(ex), ".0f", lang)), color=DIM, fontsize=F.tick * cv.sc, ha="left", va="center", zorder=8)
    if ts < st[2]:
        ax.text(half * C.primary_label_dx, max(min(xf, M.x_first) * 0.45, C.primary_label_min_y), fill(tx["primary"], e=sci(P.e0_ev, lang, 0)),
                color=(*rgb[sm.PROTON], 1.0), fontsize=F.annotation * cv.sc, ha="left", va="center", zorder=9)
    # --------------------------------------------------------------- the profile
    ax = cv.b
    cv.place(ax, L.profile_axes)
    cv.style_axes(ax, left=True, bottom=True)
    ax.set_xscale("log")
    ax.set_xlim(*PR.xlim)
    ax.set_ylim(y_bot, y_top)
    ax.set_yticks([])
    ax.set_xticks(list(PR.ticks))
    ax.set_xticklabels([rf"$10^{{{int(round(math.log10(v)))}}}$" for v in PR.ticks])
    ax.minorticks_off()
    i = int(np.searchsorted(M.grid, xf, side="right"))
    g = M.grid[:i]
    for arr, col, al in ((M.n_had, rgb[sm.PION], PR.hadron_alpha), (M.n_mu, rgb[sm.MUON], PR.mu_alpha), (M.n_em, TXT, PR.em_alpha)):
        v = np.where(arr[:i] >= 1.0, arr[:i], np.nan)
        ax.plot(v, g, color=(*col, al), lw=PR.curve_width_px * cv.px, solid_capstyle="round", zorder=4)
    ax.plot([PR.xlim[0], PR.xlim[1]], [xg, xg], color=(*TXT, 0.9), lw=C.ground_line_width * cv.sc, zorder=6)
    ax.fill_between([PR.xlim[0], PR.xlim[1]], xg, y_bot, facecolor="none", edgecolor=(*DIM, C.ground_hatch_alpha), hatch="////", lw=0, zorder=1)
    for h in CFG.atmosphere.altitude_ticks_km:
        xh = float(sm.depth_of_altitude(P, h))
        if xh > xg:
            continue
        ax.plot(PR.xlim, [xh, xh], color=(1, 1, 1, C.altitude_line_alpha), lw=C.altitude_line_width * cv.sc, zorder=0)
    a_mark = smooth(ts, st[3], st[3] + SC2.overlay_in_s)
    if a_mark > 0.0:
        ax.plot([M.n_max_h] * 2, [y_top, xg], color=(*TXT, PR.heitler_line_alpha * a_mark), lw=PR.heitler_line_width * cv.px, ls=(0, tuple(PR.heitler_dash)), zorder=3)
        ax.plot([PR.xlim[0], PR.xlim[1]], [M.x_max_h] * 2, color=(*TXT, PR.heitler_line_alpha * a_mark), lw=PR.heitler_line_width * cv.px, ls=(0, tuple(PR.heitler_dash)), zorder=3)
        ax.text(M.n_max_h * 0.9, y_top + 6.0, r"$E_0/E_c$", color=(*TXT, a_mark), fontsize=F.annotation * cv.sc, ha="right", va="top")
        ax.text(PR.xlim[1] * 0.97, M.x_max_h - 8.0, r"$X_{max}$", color=(*TXT, a_mark), fontsize=F.annotation * cv.sc, ha="right", va="bottom")
        ax.plot([M.n_max], [M.x_max], "o", color=(*TXT, a_mark), ms=PR.peak_marker_size * cv.px, zorder=8)
    ax.text(*PR.title_pos, tx["prof_title"], color=DIM, fontsize=F.panel_title * cv.sc, transform=ax.transAxes, ha="left", va="bottom")
    for r, (col, key) in enumerate(((TXT, "pl_em"), (rgb[sm.MUON], "pl_mu"), (rgb[sm.PION], "pl_had"))):
        y = PR.legend_y0 + r * PR.legend_dy
        ax.plot([PR.legend_x0, PR.legend_x0 * PR.legend_len], [y, y], color=(*col, 0.95), lw=PR.curve_width_px * cv.px, solid_capstyle="round")
        ax.text(PR.legend_x0 * PR.legend_len * PR.legend_gap, y, tx[key], color=TXT, fontsize=F.panel_title * cv.sc, ha="left", va="center")
    draw_info(cv, M, ts, xf, lang)


def right_column(cv: Canvas) -> tuple[float, float]:
    I = CFG.info
    return I.legend_x, 1.0 - I.right_margin - I.legend_x


def put_block(cv: Canvas, lines: list[tuple[str, float, tuple, float]], x: float, y: float, max_w: float) -> float:
    """Text lines (text, size, colour, gap after it) from the top y downwards; a line wider than max_w is wrapped; returns the next y."""
    for s, size, col, gap in lines:
        for part in cv.wrap(s, size, max_w):
            cv.text(x, y, part, size, col)
            y -= CFG.info.line_height * size / CFG.fonts.legend
        y -= gap
    return y


def draw_info(cv: Canvas, M: Model, ts: float, xf: float, lang: str) -> None:
    I, F, S, SC2 = CFG.info, CFG.fonts, CFG.style, CFG.scene2
    tx = cv.tx
    TXT, DIM = tuple(S.text), tuple(S.dim)
    rgb = kind_rgb()
    P = M.P
    st = stage_times()
    x0i, width = right_column(cv)
    items = [(rgb[sm.PROTON], "-", "leg_had"), (tuple(S.pi0), "o", "leg_pi0"), (rgb[sm.GAMMA], "-", "leg_gamma"),
             (rgb[sm.ELECTRON], "-", "leg_e"), (rgb[sm.MUON], "-", "leg_mu"), (rgb[sm.NEUTRINO], ":", "leg_nu")]
    for row, (col, sty, key) in enumerate(items):
        y = I.legend_y0 - row * I.legend_dy
        if sty == "o":
            cv.ov.plot([x0i + 0.5 * I.legend_line[1]], [y + I.legend_mark_dy], "o", color=col, ms=5.0 * cv.px, mec="none")
        else:
            cv.ov.plot([x0i, x0i + I.legend_line[1]], [y + I.legend_mark_dy] * 2, color=col, lw=2.0 * cv.px, ls=("-" if sty == "-" else (0, tuple(I.dot_dash))),
                       solid_capstyle="round")
        cv.text(x0i + I.legend_text_dx, y, tx[key], F.legend, TXT)
    # brightness = energy
    yb = I.legend_y0 - len(items) * I.legend_dy
    for j in range(I.bright_steps):
        a = CFG.column.energy_alpha_min + (1 - CFG.column.energy_alpha_min) * (j / max(I.bright_steps - 1, 1)) ** CFG.column.energy_alpha_exp
        xa = x0i + I.legend_line[1] * j / I.bright_steps
        cv.ov.plot([xa, xa + I.legend_line[1] / I.bright_steps], [yb + I.legend_mark_dy] * 2, color=(*rgb[sm.ELECTRON], a), lw=2.0 * cv.px, solid_capstyle="butt")
    cv.text(x0i + I.legend_text_dx, yb, tx["leg_bright"], F.legend, TXT)
    # readouts
    em_now = float(np.interp(xf, M.grid, M.n_em))
    mu_now = float(np.interp(xf, M.grid, M.n_mu))
    dr_now = float(np.interp(xf, M.grid, M.n_em_drawn))
    ro = [fill(tx["ro_x"], x=num(max(xf, 0.0), ".0f", lang), h=(num(float(sm.altitude_km(P, max(xf, sm.X_FLOOR))), ".1f", lang) if xf > 0 else r"\infty")),
          fill(tx["ro_em"], n=sci(em_now, lang)),
          fill(tx["ro_mu"], n=sci(mu_now, lang)),
          fill(tx["ro_k"], k=sci_int(max(em_now / max(dr_now, 1.0), 1.0), lang))]
    y = I.readout_y0
    for s_ in ro:
        for part in cv.wrap(s_, F.readout, width):
            cv.text(x0i, y, part, F.readout, TXT)
            y -= I.readout_dy
    y -= I.panel_gap
    # rules and results
    frac = Fraction(P.n_ch, P.n_sec)
    m = sm.hadron_levels(P.e0_ev, P.e_dec_ev, P.n_ch)
    ph = dict(nch=f"{P.n_ch}", npi0=f"{P.n_pi0}", nsec=f"{P.n_sec}", lam=num(P.lam_i, ".0f", lang), frac=rf"\frac{{{frac.numerator}}}{{{frac.denominator}}}",
              edec=num(P.e_dec_ev / U.gev, ".0f", lang), m=f"{m}", nmu=sci(float(P.n_ch ** m), lang, 0), x0=num(P.x0, ".0f", lang),
              ec=num(P.e_c_ev / U.mev, ".0f", lang), emu=num(100.0 * (P.n_ch / P.n_sec) ** m, ".0f", lang), n2=f"{2 * P.n_sec}", nh=sci(M.n_max_h, lang), xh=num(M.x_max_h, ".0f", lang))
    panels = []
    for pre, t0, t1 in SC2.rules:
        if t0 <= ts < t1:
            panels = [f"{pre}_{k}" for k in ("t", "1", "2", "3", "4")]
    if panels:
        lines = [(fill(tx[panels[0]], **ph), F.panel_title, DIM, I.panel_gap_small)] + [(fill(tx[k], **ph), F.panel, TXT, I.panel_gap_small) for k in panels[1:]]
        put_block(cv, lines, x0i, y, width)
    if ts >= SC2.summary_at:
        mc_x, mc_n = M.x_max, M.n_max
        mu_end = float(M.n_mu[-1])
        ph2 = dict(xmc=num(mc_x, ".0f", lang), xh=num(M.x_max_h, ".0f", lang), nmc=sci(mc_n, lang), nh=sci(M.n_max_h, lang),
                   mumc=sci(mu_end, lang), muh=sci(sm.matthews_n_mu(P.e0_ev, P.e_dec_ev, P.n_ch), lang), beta=num(sm.matthews_beta(P.n_ch), ".2f", lang),
                   ratio=num(mc_n / M.n_max_h, ".2f", lang))
        lines = [(tx["sm_t"], F.panel_title, DIM, I.panel_gap_small)]
        for k in ("sm_x", "sm_n", "sm_mu"):
            lines.append((fill(tx[k], **ph2), F.panel, TXT, I.panel_gap_small))
        lines.append((fill(tx["sm_note"], **ph2), F.panel_title, DIM, 0.0))
        put_block(cv, lines, x0i, y, width)


# ----------------------------------------------------------------------------------------------- part 1: Heitler's toy cascade
@lru_cache(maxsize=None)
def tree_data() -> dict:
    """The fixed-step cascade of one photon: for every particle (g, k) its start and end points (lateral in [-1, 1], depth)."""
    P = make_params()
    n = PH.tree_generations
    d = P.x0 * math.log(2.0)
    hw = CFG.tree.lateral_half_width

    def lat(g: int, k: int) -> float:
        return ((k + 0.5) / 2 ** g - 0.5) * 2.0 * hw

    kinds = [np.array([sm.GAMMA])]
    for g in range(n):
        prev = kinds[-1]
        nxt = np.empty(2 * len(prev), int)
        nxt[0::2] = sm.ELECTRON
        nxt[1::2] = np.where(prev == sm.GAMMA, sm.ELECTRON, sm.GAMMA)
        kinds.append(nxt)
    edges = []
    for g in range(n + 1):
        e_g = P.e_c_ev * 2.0 ** (n - g)
        for k in range(2 ** g):
            a = (lat(g - 1, k // 2) if g > 0 else 0.0, g * d)
            y_end = (g + 1) * d if g < n else n * d + P.x0 * e_g / P.e_c_ev
            b = (lat(g, k), y_end)
            edges.append(dict(g=g, k=k, kind=int(kinds[g][k]), a=a, b=b, final=(g == n)))
    return dict(P=P, n=n, d=d, edges=edges, e0=P.e_c_ev * 2.0 ** n, x_end=n * d + P.x0)


def tree_front(t: float) -> float:
    T = tree_data()
    return float(np.clip((t - CFG.tree.start_s) / CFG.tree.step_s * T["d"], 0.0, T["x_end"]))


def draw_tree(cv: Canvas, t: float, lang: str) -> None:
    from matplotlib.collections import LineCollection
    T = tree_data()
    TR, F, S, L = CFG.tree, CFG.fonts, CFG.style, CFG.layout
    tx = cv.tx
    TXT, DIM = tuple(S.text), tuple(S.dim)
    rgb = kind_rgb()
    P, n, d = T["P"], T["n"], T["d"]
    xf = tree_front(t)
    y_bot = T["x_end"] + TR.depth_margin * d
    y_top = -0.35 * d
    a_in = smooth(t, 0.0, 1.0)
    ax = cv.a
    cv.place(ax, L.tree_axes)
    cv.style_axes(ax, left=True)
    ax.set_xlim(-1, 1)
    ax.set_ylim(y_bot, y_top)
    ticks = [g * d for g in range(n + 1)]
    ax.set_yticks(ticks)
    ax.set_yticklabels([num(v, ".0f", lang) for v in ticks])
    ax.set_xticks([])
    ax.text(-1.0, y_top - 0.25 * d, tx["depth_unit"], color=DIM, fontsize=F.tick * cv.sc, ha="left", va="center", clip_on=False)
    segs, cols, lws = [], [], []
    for e in T["edges"]:
        ya, yb = e["a"][1], e["b"][1]
        if xf <= ya:
            continue
        fr = min((xf - ya) / (yb - ya), 1.0)
        pa = np.array(e["a"])
        pb = np.array(e["b"])
        pe = pa + (pb - pa) * fr
        segs.append([(pa[0], pa[1]), (pe[0], pe[1])])
        al = TR.final_alpha if e["final"] else 1.0
        cols.append((*rgb[e["kind"]], al))
        lws.append(TR.line_width * cv.px)
        if fr >= 1.0 and not e["final"]:
            ax.scatter([pb[0]], [pb[1]], s=(TR.node_size * cv.px) ** 2, c=[(1, 1, 1, 0.9)], edgecolors="none", zorder=6)
        elif fr < 1.0:
            ax.scatter([pe[0]], [pe[1]], s=(TR.head_size * cv.px) ** 2, c=[(*rgb[e["kind"]], 1.0)], edgecolors="none", zorder=7)
    if segs:
        ax.add_collection(LineCollection(segs, colors=cols, linewidths=lws, capstyle="round", zorder=4))
    for g in range(n + 1):
        if xf >= g * d and t > 0.5:
            ax.plot([-1, 1], [g * d, g * d], color=(1, 1, 1, TR.guide_alpha), lw=0.7 * cv.sc, zorder=0)
            lab = rf"$n={g}$:  $N={2 ** g}$,  $E=E_0$" if g == 0 else rf"$n={g}$:  $N={2 ** g}$,  $E=E_0/{2 ** g}$"
            ax.text(TR.label_x, g * d - 0.12 * d, lab, color=TXT, fontsize=TR.label_size * cv.sc, ha="left", va="bottom")
    if xf >= n * d + 0.5 * P.x0:
        ax.text(0.0, n * d + P.x0 + 0.8 * d, tx["tr_stop"], color=DIM, fontsize=TR.label_size * cv.sc, ha="center", va="center")
    ax.text(0.04, 0.35 * d, rf"$\gamma$,  $E_0={num(T['e0'] / U.gev, '.1f', lang)}\,$" + tx["gev"], color=(*rgb[sm.GAMMA], a_in), fontsize=F.annotation * cv.sc, ha="left", va="center")
    # profile
    ax = cv.b
    cv.place(ax, L.tree_profile_axes)
    cv.style_axes(ax, left=True, bottom=True)
    ax.set_xscale("log")
    ax.set_xlim(*TR.profile_xlim)
    ax.set_ylim(y_bot, y_top)
    ax.set_yticks([])
    ax.set_xticks(list(TR.profile_ticks))
    ax.set_xticklabels([num(v, ".0f", lang) for v in TR.profile_ticks])
    ax.minorticks_off()
    yy = np.linspace(0.0, xf, max(int(xf / TR.profile_dy) + 2, 3))
    nn = sm.heitler_profile(yy, T["e0"], d, P.x0, P.e_c_ev)
    nn = np.where(nn > 0, nn, np.nan)
    ax.plot(nn, yy, color=(*TXT, 0.95), lw=TR.stairs_width * cv.px, zorder=4)
    ye = np.linspace(0.0, n * d, 50)
    ax.plot(2.0 ** (ye / d), ye, color=(*rgb[sm.ELECTRON], TR.exp_line_alpha), lw=TR.exp_line_width * cv.px, ls=(0, tuple(TR.exp_dash)), zorder=3)
    ax.text(2.0 ** (0.55 * n), 0.5 * n * d, tx["tr_exp"], color=(*rgb[sm.ELECTRON], 1.0), fontsize=F.annotation * cv.sc, ha="left", va="center")
    ax.text(0.0, 1.02, tx["prof_title"], color=DIM, fontsize=F.panel_title * cv.sc, transform=ax.transAxes, ha="left", va="bottom")
    a_r = smooth(t, TR.results_at, TR.results_at + 1.0)
    if a_r > 0:
        nmax, xmax = sm.heitler_n_max(T["e0"], P.e_c_ev), sm.heitler_x_max(P.x0, T["e0"], P.e_c_ev)
        ax.plot([nmax] * 2, [y_top, y_bot], color=(*TXT, 0.6 * a_r), lw=1.2 * cv.px, ls=(0, (4, 3)), zorder=3)
        ax.plot(list(TR.profile_xlim), [xmax] * 2, color=(*TXT, 0.6 * a_r), lw=1.2 * cv.px, ls=(0, (4, 3)), zorder=3)
        ax.text(nmax * 0.93, y_top + 0.1 * d, r"$E_0/E_c$", color=(*TXT, a_r), fontsize=F.annotation * cv.sc, ha="right", va="top")
        ax.text(TR.profile_xlim[1] * 0.95, xmax - 0.15 * d, r"$X_{max}$", color=(*TXT, a_r), fontsize=F.annotation * cv.sc, ha="right", va="bottom")
        ax.plot([nmax], [xmax], "o", color=(*TXT, a_r), ms=TR.marker_size * cv.px, zorder=8)
    # right column
    I = CFG.info
    x0i, width = right_column(cv)
    for row, (col, key) in enumerate(((rgb[sm.GAMMA], "leg_gamma"), (rgb[sm.ELECTRON], "leg_e"))):
        y = I.legend_y0 - row * I.legend_dy
        cv.ov.plot([x0i, x0i + I.legend_line[1]], [y + I.legend_mark_dy] * 2, color=col, lw=2.0 * cv.px, solid_capstyle="round")
        cv.text(x0i + I.legend_text_dx, y, tx[key], F.legend, TXT)
    ph = dict(d=num(d, ".1f", lang), n=f"{n}", e0=num(T["e0"] / U.gev, ".1f", lang), nmax=f"{int(2 ** n)}", xmax=num(sm.heitler_x_max(P.x0, T["e0"], P.e_c_ev), ".0f", lang))
    for pre, t0, t1 in TR.rules:
        if t0 <= t < t1:
            keys = [f"{pre}_{k}" for k in ("t", "1", "2", "3", "4")]
            lines = [(fill(tx[keys[0]], **ph), F.panel_title, DIM, I.panel_gap_small)] + [(fill(tx[k], **ph), F.panel, TXT, I.panel_gap_small) for k in keys[1:]]
            put_block(cv, lines, x0i, I.tree_panel_y, width)
    cv.caption(fill(tx[caption_key(TR.captions, t)], d=ph["d"], e0=ph["e0"], x0=num(P.x0, ".0f", lang)))


# ----------------------------------------------------------------------------------------------- part 3: the ground
@lru_cache(maxsize=None)
def ground_data() -> dict:
    M = get_model()
    P, sh, G = M.P, M.sh, CFG.ground
    edges = np.array(G.radial_edges_m)
    centres = np.sqrt(np.maximum(edges[:-1], 1e-9) * edges[1:])
    centres[0] = 0.5 * edges[1]
    rho_e = sh.radial_density((sm.ELECTRON,), edges)
    rho_m = sh.radial_density((sm.MUON,), edges)
    n_e, n_m = sh.n_ground((sm.ELECTRON,)), sh.n_ground((sm.MUON,))
    half = G.array_half_m
    n = int(half // G.station_spacing_m)
    ij = np.arange(-n, n + 1) * G.station_spacing_m
    sx, sy = np.meshgrid(ij, ij)
    sx, sy = sx.ravel(), sy.ravel()
    core = np.array(G.core_offset_m)
    r = np.hypot(sx - core[0], sy - core[1])
    floor = rho_e[rho_e > 0].min() * 1e-3
    mean = np.exp(np.interp(np.log(np.maximum(r, centres[0])), np.log(centres), np.log(np.maximum(rho_e, floor)))) * G.station_area_m2
    hits = np.random.default_rng(G.seed).poisson(mean)
    tr = sh.tr
    mm = (tr["kind"] == sm.MUON) & tr["ground"].astype(bool)
    mx, my = tr["px1"][mm] + core[0], tr["py1"][mm] + core[1]
    sp = G.station_spacing_m
    side = math.sqrt(G.station_area_m2)
    ix, iy = np.rint(mx / sp), np.rint(my / sp)
    hit = (np.abs(mx - ix * sp) <= side / 2) & (np.abs(my - iy * sp) <= side / 2) & (np.abs(ix) <= n) & (np.abs(iy) <= n)
    mu_hits = np.zeros((2 * n + 1, 2 * n + 1), int)
    np.add.at(mu_hits, ((iy[hit] + n).astype(int), (ix[hit] + n).astype(int)), 1)
    mu_hits = mu_hits.ravel()
    inside = (np.abs(mx) <= half) & (np.abs(my) <= half)
    r_m = sm.moliere_radius_m(P, P.x_ground)
    age = sm.shower_age(P.x_ground, M.x_max)
    rr = np.geomspace(G.radial_xlim[0], G.radial_xlim[1], 120)
    nkg = sm.nkg_density(rr, n_e, age, r_m)
    ratio = [float(np.interp(math.log(r_), np.log(centres), rho_m / np.maximum(rho_e, 1e-300))) for r_ in G.ratio_radii_m]
    return dict(edges=edges, centres=centres, rho_e=rho_e, rho_m=rho_m, n_e=n_e, n_m=n_m, sx=sx, sy=sy, r=r, hits=hits, mu_hits=mu_hits, r_m=r_m, age=age, rr=rr, nkg=nkg, ratio=ratio, core=core, n_inside=int(inside.sum()))


def draw_ground(cv: Canvas, t: float, lang: str) -> None:
    import matplotlib.patches as mp
    D = ground_data()
    G, F, S, L, I = CFG.ground, CFG.fonts, CFG.style, CFG.layout, CFG.info
    tx = cv.tx
    TXT, DIM = tuple(S.text), tuple(S.dim)
    rgb = kind_rgb()
    a_in = smooth(t, 0.0, G.t_array_in)
    half = G.array_half_m
    ax = cv.a
    cv.place(ax, L.ground_array_axes)
    cv.style_axes(ax, left=True, bottom=True)
    ax.set_xlim(-half, half)
    ax.set_ylim(-half, half)
    ax.set_aspect("equal")
    ax.set_xticks([-200, -100, 0, 100, 200])
    ax.set_yticks([-200, -100, 0, 100, 200])
    ax.text(0.0, 1.02, tx["gr_title"], color=DIM, fontsize=F.panel_title * cv.sc, transform=ax.transAxes, ha="left", va="bottom")
    pos = ax.get_position()
    px_per_m = pos.width * cv.W / (2.0 * half)
    side = G.station_spacing_m * px_per_m * G.station_fill
    R = G.reveal_speed * max(t - G.reveal_start, 0.0)
    f = np.clip((R - D["r"]) / G.reveal_width, 0.0, 1.0) * a_in
    lt = np.clip(np.log10(1.0 + D["hits"]) / G.hit_log_max, 0.0, 1.0)
    lo, hi = np.array(G.hit_colour_low), np.array(G.hit_colour_high)
    col = lo[None, :] + (hi - lo)[None, :] * lt[:, None]
    alpha = np.where(D["hits"] > 0, 0.35 + 0.65 * lt, G.station_empty_alpha) * np.where(D["hits"] > 0, f, a_in)
    ax.scatter(D["sx"], D["sy"], s=(side * 72.0 / cv.dpi) ** 2, marker="s", c=np.column_stack([col, alpha]), edgecolors="none", zorder=3)
    mh = D["mu_hits"] > 0
    ax.scatter(D["sx"][mh], D["sy"][mh], s=(G.muon_dot_size * cv.px) ** 2, c=np.column_stack([np.tile(rgb[sm.MUON], (int(mh.sum()), 1)), G.muon_alpha * f[mh]]),
               edgecolors="none", zorder=5)
    cx, cy = D["core"]
    ax.plot([cx], [cy], "+", color=(1, 1, 1, a_in), ms=G.core_marker_size * cv.px, mew=1.2 * cv.sc, zorder=6)
    ax.add_patch(mp.Circle((cx, cy), D["r_m"], fill=False, ec=(1, 1, 1, G.moliere_alpha * a_in), lw=G.moliere_width * cv.px, ls=(0, tuple(G.moliere_dash)), zorder=4))
    ax.text(cx + D["r_m"] * 0.71, cy + D["r_m"] * 0.71 + 6.0, r"$r_M$", color=(*TXT, a_in), fontsize=F.annotation * cv.sc, ha="left", va="bottom", zorder=7)
    # colour bar
    cb = L.ground_colourbar
    nb = 60
    for j in range(nb):
        u = j / (nb - 1)
        cv.ov.add_patch(mp.Rectangle((cb[0] + cb[2] * j / nb, cb[1]), cb[2] / nb + 1e-4, cb[3], fc=(*(lo + (hi - lo) * u), a_in), ec="none"))
    for tk in G.colourbar_ticks:
        xx = cb[0] + cb[2] * tk / G.hit_log_max
        cv.text(xx, cb[1] - 0.026, rf"${num(10 ** tk, '.0f', lang)}$", F.tick, (*DIM, a_in), ha="center")
    cv.text(cb[0] + cb[2] + 0.012, cb[1] - 0.001, tx["gr_hits"], F.tick, (*DIM, a_in))
    # readouts
    ro = [fill(tx["gr_ro_e"], n=sci(D["n_e"], lang)), fill(tx["gr_ro_mu"], n=sci(D["n_m"], lang), k=f"{int((D['mu_hits'] > 0).sum())}"),
          fill(tx["gr_ro_rm"], r=num(D["r_m"], ".0f", lang)),
          fill(tx["gr_ro_samp"], p=num(100.0 * G.station_area_m2 / G.station_spacing_m ** 2, ".0f", lang)),
          fill(tx["gr_ro_cross"], a=num(D["ratio"][0], ".2f", lang), b=num(D["ratio"][1], ".2f", lang), r1=num(G.ratio_radii_m[0], ".0f", lang), r2=num(G.ratio_radii_m[1], ".0f", lang))]
    y = L.ground_readout_y0
    for s_ in ro:
        for part in cv.wrap(s_, F.readout, 1.0 - L.ground_readout_x - 0.015):
            cv.text(L.ground_readout_x, y, part, F.readout, (*TXT, a_in))
            y -= L.ground_readout_dy
    # radial plot
    a_p = smooth(t, G.t_plot[0], G.t_plot[1])
    if a_p > 0:
        bx = cv.b
        cv.place(bx, L.ground_radial_axes)
        cv.style_axes(bx, left=True, bottom=True)
        bx.set_xscale("log")
        bx.set_yscale("log")
        bx.set_xlim(*G.radial_xlim)
        bx.set_ylim(*G.radial_ylim)
        bx.set_xticks(list(G.radial_ticks))
        bx.set_xticklabels([num(v, ".0f", lang) for v in G.radial_ticks])
        bx.set_yticks(list(G.radial_yticks))
        bx.set_yticklabels([rf"$10^{{{int(round(math.log10(v)))}}}$" for v in G.radial_yticks])
        bx.minorticks_off()
        ok = D["hits"] > 0
        bx.scatter(D["r"][ok], D["hits"][ok] / G.station_area_m2, s=(G.dots_size * cv.px) ** 2, c=[(*rgb[sm.ELECTRON], G.dots_alpha * a_p)], edgecolors="none", zorder=3)
        ce = D["centres"]
        for rho, key in ((D["rho_e"], sm.ELECTRON), (D["rho_m"], sm.MUON)):
            m = rho > 0
            bx.plot(ce[m], rho[m], color=(*rgb[key], a_p), lw=G.curve_width_px * cv.px, solid_capstyle="round", zorder=4)
        a_n = smooth(t, G.t_nkg, G.t_nkg + 1.0)
        if a_n > 0:
            bx.plot(D["rr"], D["nkg"], color=(*TXT, G.nkg_alpha * a_n), lw=1.3 * cv.px, ls=(0, tuple(G.nkg_dash)), zorder=5)
            bx.text(G.radial_xlim[0] * 1.15, G.radial_ylim[0] * 3.0, tx["gr_nkg"].replace("<s>", num(D["age"], ".2f", lang)), color=(*TXT, a_n), fontsize=F.annotation * cv.sc, ha="left", va="bottom")
        bx.axvline(D["r_m"], color=(1, 1, 1, 0.4 * a_p), lw=1.0 * cv.px, ls=(0, (2, 3)))
        bx.text(D["r_m"] * 1.06, G.radial_ylim[1] * 0.5, r"$r_M$", color=(*TXT, a_p), fontsize=F.annotation * cv.sc, ha="left", va="top")
        bx.text(0.0, 1.03, tx["gr_rad_title"], color=(*DIM, a_p), fontsize=F.panel_title * cv.sc, transform=bx.transAxes, ha="left", va="bottom")
        bx.text(1.0, -0.12, tx["gr_r"], color=(*DIM, a_p), fontsize=F.axis_label * cv.sc, transform=bx.transAxes, ha="right", va="top")
        for k, (col, key) in enumerate(((rgb[sm.ELECTRON], "gr_l_e"), (rgb[sm.MUON], "gr_l_mu"))):
            bx.text(0.97, 0.93 - 0.075 * k, tx[key], color=(*col, a_p), fontsize=F.annotation * cv.sc, transform=bx.transAxes, ha="right", va="top")
        bx.text(0.97, 0.93 - 0.075 * 2, tx["gr_l_st"], color=(*rgb[sm.ELECTRON], 0.7 * a_p), fontsize=F.annotation * cv.sc, transform=bx.transAxes, ha="right", va="top")
    cv.caption(fill(tx[caption_key(G.captions, t)], r=num(D["r_m"], ".0f", lang)))


# ----------------------------------------------------------------------------------------------- the light of the shower
def draw_light(cv: Canvas, t: float, lang: str) -> None:
    M = get_model()
    Lg, F, S, L = CFG.light, CFG.fonts, CFG.style, CFG.layout
    tx = cv.tx
    TXT, DIM = tuple(S.text), tuple(S.dim)
    rgb = kind_rgb()
    P = M.P
    a_in = smooth(t, *Lg.t_in)
    a_c = smooth(t, Lg.t_cherenkov, Lg.t_cherenkov + 1.0)
    # fluorescence: the energy deposit
    ax = cv.a
    cv.place(ax, Lg.profile_axes)
    ax.set_aspect("auto")                    # the axes were square-equal in the ground view
    cv.style_axes(ax, left=True, bottom=True)
    dep = M.sh.deposit_profile() / U.tev
    ax.set_xlim(0.0, 1.12 * dep.max())
    xg = P.x_ground
    ax.set_ylim(xg, 0.0)
    ax.set_yticks(CFG.atmosphere.depth_ticks)
    ax.fill_betweenx(M.grid, 0.0, dep, color=(*rgb[sm.ELECTRON], 0.28 * a_in), lw=0)
    ax.plot(dep, M.grid, color=(*rgb[sm.ELECTRON], a_in), lw=Lg.curve_width_px * cv.px, zorder=4)
    ax.text(0.0, 1.03, tx["lt_fl"], color=(*DIM, a_in), fontsize=F.panel_title * cv.sc, transform=ax.transAxes, ha="left", va="bottom")
    ax.text(-0.17, 1.03, tx["depth_unit"], color=DIM, fontsize=F.tick * cv.sc, transform=ax.transAxes, ha="left", va="bottom")
    ef = M.sh.deposited_energy() / P.e0_ev
    ax.text(0.97, 0.90, fill(tx["lt_int"], f=num(ef, ".2f", lang)), color=(*TXT, a_in), fontsize=F.annotation * cv.sc, transform=ax.transAxes, ha="right", va="center")
    # Cherenkov threshold and angle against the altitude
    h = np.linspace(0.0, Lg.altitude_max_km, Lg.samples)
    nm1 = Lg.n_minus_one_sea * sm.depth_of_altitude(P, h) / P.x_sea
    eth = sm.cherenkov_threshold_ev(nm1, Lg.electron_mass_ev) / U.mev
    th = np.degrees(sm.cherenkov_angle(nm1))
    for rect, y, key, unit_key, col in ((Lg.energy_axes, eth, "lt_e", "lt_e_unit", rgb[sm.GAMMA]), (Lg.angle_axes, th, "lt_a", "lt_a_unit", rgb[sm.MUON])):
        bx = cv.b if y is eth else cv.c
        if a_c < CFG.style.panel_cutoff:
            continue
        cv.place(bx, rect)
        bx.set_aspect("auto")
        cv.style_axes(bx, left=True, bottom=True)
        bx.set_xlim(0.0, Lg.altitude_max_km)
        bx.set_ylim(0.0, 1.15 * y.max())
        bx.plot(h, y, color=(*col, a_c), lw=Lg.curve_width_px * cv.px)
        bx.plot([0.0], [y[0]], "o", color=(*col, a_c), ms=Lg.marker_size * cv.px)
        bx.text(0.0, 1.05, tx[key], color=(*DIM, a_c), fontsize=F.panel_title * cv.sc, transform=bx.transAxes, ha="left", va="bottom")
        bx.text(0.02 * Lg.altitude_max_km, Lg.label_level * y.max() if y is th else Lg.label_level_e * y.max(), rf"${num(y[0], '.1f', lang)}$ " + tx[unit_key], color=(*TXT, a_c),
                fontsize=F.annotation * cv.sc, ha="left", va="center")
    cv.caption(tx[caption_key(Lg.captions, t)])


# ----------------------------------------------------------------------------------------------- frames, cards, the film
def draw_card(cv: Canvas, card: int, cprog: float) -> None:
    L, F, S, T = CFG.layout, CFG.fonts, CFG.style, CFG.timeline
    tx = cv.tx
    a_c = min(smooth(cprog, *T.card_fade_in), 1.0 - smooth(cprog, *T.card_fade_out))
    DIM = tuple(S.dim)
    if card == 0:
        cv.text(0.5, L.card_title_y, tx["h0"], F.card_title, (*S.card_title_color, a_c), ha="center", va="center")
        cv.text(0.5, L.card_subtitle_y, tx["h0s"], F.card_subtitle, (*DIM, a_c), ha="center", va="center")
    else:
        num_ = 1 + card
        cv.text(0.5, L.card_number_y, f"{num_}", F.card_number, (*S.gold, S.card_number_alpha * a_c), ha="center", va="center")
        cv.text(0.5, L.card_chapter_y, tx[CARD_KEYS[card]], F.card_chapter, (*S.card_title_color, a_c), ha="center", va="center")
        cv.text(0.5, L.card_chapter_sub_y, tx[CARD_KEYS[card] + "s"], F.card_chapter_sub, (*DIM, a_c), ha="center", va="center")


def draw_frame(cv: Canvas, tc: float, lang: str) -> None:
    L, F, S = CFG.layout, CFG.fonts, CFG.style
    tx = cv.tx
    cv.text(*L.title_pos, tx["title"], F.title, tuple(S.title_color))
    ch = chapter_of(tc)
    cv.text(*L.chapter_label_pos, f"{ch}/3   " + tx[f"ch{ch}"], F.chapter_label, (*S.dim, S.chapter_label_alpha), ha="right")
    cv.text(*CFG.info.tag_pos, tx["tag"], F.tag, (*S.dim, S.chapter_label_alpha), ha="right")
    if tc < T_SHOWER[0]:
        draw_tree(cv, tc - T_TREE[0], lang)
    elif tc < T_GROUND[0]:
        draw_shower(cv, tc - T_SHOWER[0], lang)
        M = get_model()
        cv.caption(fill(tx[caption_key(CFG.scene2.captions, tc - T_SHOWER[0])], e=sci(M.P.e0_ev, lang, 0), nmu=sci(float(M.n_mu[-1]), lang),
                        nem=sci(float(M.n_em[-1]), lang)))
    elif tc < T_LIGHT[0]:
        draw_ground(cv, tc - T_GROUND[0], lang)
    else:
        draw_light(cv, tc - T_LIGHT[0], lang)


def layout_problems(cv: Canvas, min_alpha: float = 0.08) -> list[str]:
    """Measured text boxes of the current frame: the texts that leave the frame and the pairs of texts that overlap."""
    r = cv.renderer()
    items, out_left = [], []
    arts = list(cv.fig.texts)
    for ax in (cv.a, cv.b, cv.c, cv.ov):
        arts += list(ax.texts)
        if ax.axison:
            arts += [t for t in ax.get_xticklabels() + ax.get_yticklabels() if t.get_visible()]
    for t in arts:
        if not t.get_text().strip() or not t.get_visible():
            continue
        col = t.get_color()
        al = t.get_alpha()
        try:
            from matplotlib.colors import to_rgba
            a = to_rgba(col)[3] if al is None else al
        except ValueError:
            a = 1.0
        if a < min_alpha:
            continue
        bb = t.get_window_extent(r)
        if bb.width < 1 or bb.height < 1:
            continue
        items.append((t.get_text()[:40], bb))
        if re.search(r"<\w+>", t.get_text()):
            out_left.append(f"placeholder left: {t.get_text()[:40]!r}")
    out = list(out_left)
    for name, bb in items:
        if bb.x0 < -1 or bb.y0 < -1 or bb.x1 > cv.W + 1 or bb.y1 > cv.H + 1:
            out.append(f"outside: {name!r}")
    for i in range(len(items)):
        for j in range(i + 1, len(items)):
            a, b = items[i][1], items[j][1]
            if a.x0 < b.x1 - 1 and b.x0 < a.x1 - 1 and a.y0 < b.y1 - 1 and b.y0 < a.y1 - 1:
                out.append(f"overlap: {items[i][0]!r} / {items[j][0]!r}")
    return out


def frame_at(cv: Canvas, tc: float, lang: str) -> None:
    """Draw the frame of the content time tc (no chapter card) on the canvas."""
    cv.reset()
    draw_frame(cv, tc, lang)


def render(out: Path, size: tuple[int, int], fps: int, total: float, lang: str, snap: float | None = None, snap_content: float | None = None) -> None:
    V = CFG.video
    if snap is None and snap_content is None and not shutil.which("ffmpeg"):
        sys.exit("ffmpeg is required")
    W, H = size
    cv = Canvas(W, H, lang)
    k = total / TOTAL
    frames = int(round(total * fps))
    ids = [int(round(snap * fps))] if snap is not None else ([0] if snap_content is not None else range(frames))
    writer = None
    if snap is None and snap_content is None:
        writer = subprocess.Popen(
            ["ffmpeg", "-y", "-v", "error", "-f", "rawvideo", "-pix_fmt", "rgba", "-s", f"{W}x{H}",
             "-r", str(fps), "-i", "-", "-c:v", "libx264", "-preset", V.preset, "-crf", str(V.crf),
             "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(out)], stdin=subprocess.PIPE)
    for k_ in ids:
        t_film = k_ / fps
        if snap_content is not None:
            tc, card, cprog = snap_content, None, 0.0
        else:
            tc, card, cprog = timeline(t_film / k)
        cv.reset()
        if card is not None:
            draw_card(cv, card, cprog)
        else:
            draw_frame(cv, tc, lang)
        fade_io = min(smooth(t_film, 0.0, V.fade_s), 1.0 - smooth(t_film, total - V.fade_s, total)) if snap_content is None else 1.0
        frame = cv.frame_out(fade_io)
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
    ap.add_argument("--snapshot", type=float, default=None, help="one PNG at this film time")
    ap.add_argument("--content", type=float, default=None, help="one PNG at this content time (the chapter cards are not counted)")
    ap.add_argument("--seconds", type=float, default=TOTAL)
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--config", type=Path, default=None, help="another configuration file instead of config.toml")
    ap.add_argument("--set", action="append", default=[], metavar="SECTION.KEY=VALUE", help="override one configuration value")
    args = ap.parse_args()
    V = CFG.video
    out = args.out or HERE / "media" / f"air_shower_cascade_{args.lang}.mp4"
    out.parent.mkdir(parents=True, exist_ok=True)
    if args.snapshot is not None:
        render(out.with_suffix(".png"), (V.width, V.height), V.fps, args.seconds, args.lang, snap=args.snapshot)
    elif args.content is not None:
        render(out.with_suffix(".png"), (V.width, V.height), V.fps, args.seconds, args.lang, snap_content=args.content)
    elif args.preview:
        render(out.with_name(out.stem + "_preview.mp4"), (V.preview_width, V.preview_height), V.preview_fps, args.seconds, args.lang)
    else:
        render(out, (V.width, V.height), V.fps, args.seconds, args.lang)


if __name__ == "__main__":
    main()
