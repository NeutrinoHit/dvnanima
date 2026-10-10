r"""Asymmetry in Z-boson production and decay (the book: "Weak decays of the W and Z bosons", e+e- -> Z -> f fbar).

Notation of the book: the neutral current of a fermion is  fbar gamma_mu (g_v - g_a gamma_5) f = fbar gamma_mu (g_L P_L + g_R P_R) f  with

    g_v = t_3 - 2 q sin^2(theta_W),   g_a = t_3,   g_L = g_v + g_a = 2 t_3 - 2 q sin^2(theta_W),   g_R = g_v - g_a = -2 q sin^2(theta_W).

e+e- -> f fbar (massless fermions): the amplitude is a sum of four chiral currents rho_ab [vbar gamma^mu P_a u][ubar gamma_mu P_b v],
a, b = L, R, with rho_ab = -e e_f / s + (g / 2 cos theta_W)^2 g_a^e g_b^f / (s - m_Z^2 + i m_Z Gamma_Z)   (the book's eq. for rho_ab);
at the Z pole the photon is negligible and rho_ab ~ g_a^e g_b^f.  The spin-summed square (the book's eq. for |M|^2, massless limit,
t = -s (1 - cos theta)/2, theta = angle between the e- and the f)

    dsigma/dcos(theta) = N_c s / (128 pi) [ (|rho_LL|^2 + |rho_RR|^2) (1 + cos theta)^2 + (|rho_LR|^2 + |rho_RL|^2) (1 - cos theta)^2 ].

So a channel with equal chiralities (e_L f_L, e_R f_R: the Z has J_z = -1 and the pair spin -1 on its own axis, or both reversed) goes
forward, (1 + cos theta)^2; the mixed channels go backward, (1 - cos theta)^2.  Normalising the sum of the weights to one,

    dN/dcos(theta) ~ 1 + cos^2(theta) + (8/3) A_FB cos(theta),     A_FB = (3/4) A_LR^e A_LR^f,   A_LR^f = (g_L^2 - g_R^2) / (g_L^2 + g_R^2)

(the polarization asymmetry of the book's problem, A_LR^f = [Gamma(Z -> f_L fbar_R) - Gamma(Z -> f_R fbar_L)] / [sum]).

Parts of the film:
 1. the couplings g_L, g_R of nu, e, u, d and the asymmetries A_LR^f (bars);
 2. the four helicity channels, their angular distributions, weights, the sum and A_FB = (3/4) A_LR^e A_LR^f (b quarks);
 3. the slider sin^2(theta_W): 1/4 (no asymmetry) -> 0.2316;
 4. Monte-Carlo events: the histogram of cos(theta), A_FB = (N_F - N_B)/(N_F + N_B) with its error (b quarks, then muons);
 5. away from the pole: A_FB(sqrt s) of e+e- -> mu+mu- from the full rho_ab with photon and Z exchange.

All the numbers are in config.toml and all the words in texts.toml (see ../dvconfig.py for --config / --set).

Usage:
    python z_decay_asymmetry.py --lang en            # film -> media/z_decay_asymmetry_en.mp4
    python z_decay_asymmetry.py --lang ru
    python z_decay_asymmetry.py --lang en --preview
    python z_decay_asymmetry.py --lang en --snapshot 50   # one PNG at film time 50 s
"""

from __future__ import annotations

import argparse
import math
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dvconfig import load_config, load_texts  # noqa: E402

HERE = Path(__file__).resolve().parent
CFG = load_config(HERE)

# ----------------------------------------------------------------------------------- the physics

S2W = CFG.physics.sin2_theta_w
ALPHA = 1.0 / CFG.physics.alpha_inv_mz
MZ = CFG.physics.m_z
GZ = CFG.physics.gamma_z
CHANNELS = ("LL", "LR", "RL", "RR")        # (chirality of the electron, chirality of the fermion f), the order of the film
FORWARD = ("LL", "RR")                      # equal chiralities: (1 + cos theta)^2


def _frac(pair) -> float:
    return pair[0] / pair[1]


FERMIONS = {name: dict(t3=_frac(v["t3"]), q=_frac(v["q"]), nc=v["nc"]) for name, v in CFG.physics.fermions.to_dict().items()}


def couplings(f: str, s2: float = S2W) -> dict:
    """g_v, g_a, g_L, g_R of the family f ("nu", "e", "u", "d") for sin^2(theta_W) = s2 (the book's definitions)."""
    t3, q = FERMIONS[f]["t3"], FERMIONS[f]["q"]
    gv = t3 - 2.0 * q * s2
    ga = t3
    return dict(gv=gv, ga=ga, L=gv + ga, R=gv - ga)


def a_lr_from(gl: float, gr: float) -> float:
    return (gl * gl - gr * gr) / (gl * gl + gr * gr)


def a_lr(f: str, s2: float = S2W) -> float:
    """The polarization asymmetry A_LR^f = (g_L^2 - g_R^2) / (g_L^2 + g_R^2) = [Gamma(Z -> f_L fbar_R) - Gamma(Z -> f_R fbar_L)] / sum."""
    c = couplings(f, s2)
    return a_lr_from(c["L"], c["R"])


def channel_weights(f: str, s2: float = S2W, e: str = "e") -> dict:
    """The weight (g_a^e g_b^f)^2 of every channel (a: the chirality of the electron, b: of the fermion f), not normalised."""
    ce, cf = couplings(e, s2), couplings(f, s2)
    return {ab: (ce[ab[0]] * cf[ab[1]]) ** 2 for ab in CHANNELS}


def normalise(w: dict) -> dict:
    tot = sum(w.values())
    return {k: v / tot for k, v in w.items()}


def channel_shares(f: str, s2: float = S2W) -> dict:
    return normalise(channel_weights(f, s2))


def lobe(ab: str, c):
    """Angular distribution of one helicity channel: (1 + cos theta)^2 if the chiralities are equal, (1 - cos theta)^2 otherwise."""
    c = np.asarray(c, dtype=float)
    return (1.0 + c) ** 2 if ab in FORWARD else (1.0 - c) ** 2


def distribution(c, shares: dict):
    """Sum over the channels, normalised with sum(shares) = 1:  1 + cos^2 + 2 (S_F - S_B) cos."""
    return sum(shares[ab] * lobe(ab, c) for ab in CHANNELS)


def forward_backward(shares: dict) -> tuple[float, float]:
    sf = sum(v for k, v in shares.items() if k in FORWARD)
    sb = sum(v for k, v in shares.items() if k not in FORWARD)
    return sf, sb


def a_fb_from_shares(shares: dict) -> float:
    sf, sb = forward_backward(shares)
    return 0.75 * (sf - sb) / (sf + sb)


def a_fb(f: str, s2: float = S2W) -> float:
    """A_FB = (3/4) A_LR^e A_LR^f at the pole."""
    return 0.75 * a_lr("e", s2) * a_lr(f, s2)


def shape(c, afb: float):
    """The normalised shape 1 + cos^2 + (8/3) A_FB cos (area 8/3)."""
    c = np.asarray(c, dtype=float)
    return 1.0 + c * c + (8.0 / 3.0) * afb * c


def sample_cos(rng: np.random.Generator, n: int, afb: float) -> np.ndarray:
    """cos(theta) of n events from 1 + cos^2 + (8/3) A_FB cos (rejection sampling)."""
    k = (8.0 / 3.0) * afb
    top = 2.0 + abs(k)
    out = np.empty(0)
    while out.size < n:
        c = rng.uniform(-1.0, 1.0, 2 * n)
        u = rng.uniform(0.0, top, 2 * n)
        out = np.concatenate([out, c[u < 1.0 + c * c + k * c]])
    return out[:n]


def estimate(c: np.ndarray) -> tuple[float, float, int, int]:
    """(A_FB, its error, N_F, N_B): A_FB = (N_F - N_B)/(N_F + N_B), error sqrt((1 - A^2)/N)."""
    n = c.size
    nf = int(np.count_nonzero(c > 0.0))
    nb = n - nf
    a = (nf - nb) / n
    return a, math.sqrt(max(1.0 - a * a, 0.0) / n), nf, nb


# --- the full amplitude with the photon and the Z (away from the pole), the book's rho_ab

E2 = 4.0 * math.pi * ALPHA
G2 = E2 / S2W


def rho(sqrt_s: float, f: str, a: str, b: str, photon: bool = True, z: bool = True, s2: float = S2W) -> complex:
    """rho_ab = -e e_f / s + (g / 2 cos theta_W)^2 g_a^e g_b^f / (s - m_Z^2 + i m_Z Gamma_Z); e_f = e q_f, a chirality of the electron."""
    s = sqrt_s * sqrt_s
    out = 0.0 + 0.0j
    if photon:
        out += -E2 * FERMIONS[f]["q"] / s
    if z:
        g2 = E2 / s2
        out += (g2 / (4.0 * (1.0 - s2))) * couplings("e", s2)[a] * couplings(f, s2)[b] / (s - MZ * MZ + 1j * MZ * GZ)
    return out


def rho_shares(sqrt_s: float, f: str, **kw) -> dict:
    """|rho_ab|^2 normalised to one."""
    return normalise({ab: abs(rho(sqrt_s, f, ab[0], ab[1], **kw)) ** 2 for ab in CHANNELS})


def dsigma_dcos(sqrt_s: float, f: str, c, **kw):
    """The massless dsigma/dcos(theta) in GeV^-2: N_c s / (128 pi) [S_F (1 + c)^2 + S_B (1 - c)^2]."""
    s = sqrt_s * sqrt_s
    w = {ab: abs(rho(sqrt_s, f, ab[0], ab[1], **kw)) ** 2 for ab in CHANNELS}
    return FERMIONS[f]["nc"] * s / (128.0 * math.pi) * sum(w[ab] * lobe(ab, c) for ab in CHANNELS)


def sigma_total(sqrt_s: float, f: str, **kw) -> float:
    """sigma = N_c s (sum |rho|^2) / (48 pi)  (GeV^-2), the massless limit of the book's total cross section."""
    s = sqrt_s * sqrt_s
    return FERMIONS[f]["nc"] * s * sum(abs(rho(sqrt_s, f, ab[0], ab[1], **kw)) ** 2 for ab in CHANNELS) / (48.0 * math.pi)


def a_fb_scan(sqrt_s: float, f: str, **kw) -> float:
    return a_fb_from_shares(rho_shares(sqrt_s, f, **kw))


# ----------------------------------------------------------------------------------- texts and the timeline

def num(x: float, fmt: str, lang: str, math_mode: bool = True) -> str:
    """A number with the decimal comma in Russian ({,} inside a formula, to keep the spacing)."""
    s = format(x, fmt)
    if lang == "ru":
        s = s.replace(".", "{,}" if math_mode else ",")
    return s


def signed(x: float, fmt: str, lang: str, math_mode: bool = True) -> str:
    """A number with an explicit minus sign (a real minus inside formulas, a plain one outside)."""
    s = num(abs(x), fmt, lang, math_mode)
    return ("-" if math_mode else "−") + s if x < 0 else s


TEXT = {lang: load_texts(HERE, lang) for lang in ("en", "ru")}          # texts.toml

CARD_S = list(CFG.timeline.card_s)
CARD_AT = list(CFG.timeline.card_at)
CARD_CHAPTER = [0, 2, 3, 4, 5]                                         # the chapter shown on every card (0 = the title card)
_PB = list(CFG.timeline.part_bounds)
CONTENT_TOTAL = _PB[-1]
TOTAL = CONTENT_TOTAL + sum(CARD_S)
PARTS = [(_PB[i], _PB[i + 1]) for i in range(len(_PB) - 1)]
N_PARTS = len(PARTS)


def timeline(tf: float) -> tuple[float, int | None, float]:
    """(content time, index of the card or None, progress of the card in [0, 1]) at the film time tf (the cards are inserted)."""
    off = 0.0
    for i, (ca, ln) in enumerate(zip(CARD_AT, CARD_S)):
        a = ca + off
        if tf < a:
            break
        if tf < a + ln:
            return ca, i, (tf - a) / ln
        off += ln
    return tf - off, None, 0.0


def film_time(tc: float) -> float:
    """Film time of the content time tc (the cards before it are counted)."""
    return tc + sum(ln for ca, ln in zip(CARD_AT, CARD_S) if ca <= tc)


def part_of(tc: float) -> int:
    """Index (1..5) of the part at the content time tc."""
    return 1 + sum(1 for b in _PB[1:-1] if tc >= b)


def smooth(x: float, a: float, b: float) -> float:
    u = min(max((x - a) / (b - a), 0.0), 1.0)
    return u * u * (3 - 2 * u)


def lerp(a: float, b: float, u: float) -> float:
    return a + (b - a) * u


# ----------------------------------------------------------------------------------- the painter (a pixel canvas)

class Painter:
    """A full-frame axes whose data coordinates are pixels (y downwards); sizes given in 720 p pixels are scaled by sc."""

    def __init__(self, W: int, H: int) -> None:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib import patches

        self.plt, self.patches = plt, patches
        V, S = CFG.video, CFG.style
        self.W, self.H = W, H
        self.dpi = V.dpi
        self.sc = H / V.reference_height
        self.fig = plt.figure(figsize=(W / self.dpi, H / self.dpi), dpi=self.dpi, facecolor=S.background)
        self.ax = self.fig.add_axes([0, 0, 1, 1])
        self.texts: list = []
        self.bg = np.array([int(S.background[1:3], 16), int(S.background[3:5], 16), int(S.background[5:7], 16), 255], np.float32)

    # -- geometry
    def X(self, f: float) -> float:
        return f * self.W

    def Y(self, f: float) -> float:
        return f * self.H

    def pt(self, f) -> tuple[float, float]:
        return f[0] * self.W, f[1] * self.H

    def s(self, v: float) -> float:
        return v * self.sc

    def begin(self) -> None:
        self.ax.clear()
        self.ax.set_xlim(0, self.W)
        self.ax.set_ylim(self.H, 0)
        self.ax.axis("off")
        self.texts = []

    # -- primitives (colours are RGB tuples, alpha separate)
    def text(self, x: float, y: float, s: str, size: float, color, alpha: float = 1.0, ha: str = "left", va: str = "center", z: int = 20, **kw):
        if alpha <= 0.004 or not s:
            return None
        t = self.ax.text(x, y, s, fontsize=size * self.sc, color=(*tuple(color)[:3], alpha), ha=ha, va=va, zorder=z, **kw)
        self.texts.append((t, alpha))
        return t

    def poly(self, pts, fc=None, alpha: float = 1.0, ec=None, lw: float = 0.0, z: int = 3, ealpha: float | None = None) -> None:
        if alpha <= 0.004 and (ec is None or (ealpha if ealpha is not None else alpha) <= 0.004):
            return
        fcol = (*tuple(fc)[:3], alpha) if fc is not None else "none"
        ecol = (*tuple(ec)[:3], ealpha if ealpha is not None else alpha) if ec is not None else "none"
        self.ax.add_patch(self.patches.Polygon(np.asarray(pts, float), closed=True, facecolor=fcol, edgecolor=ecol, linewidth=lw * self.sc, zorder=z,
                                               joinstyle="round"))

    def line(self, pts, color, alpha: float = 1.0, lw: float = 1.0, ls="-", z: int = 4, cap: str = "round") -> None:
        if alpha <= 0.004:
            return
        p = np.asarray(pts, float)
        self.ax.plot(p[:, 0], p[:, 1], color=(*tuple(color)[:3], alpha), lw=lw * self.sc, ls=ls, zorder=z, solid_capstyle=cap, dash_capstyle=cap)

    def circle(self, c, r: float, color, alpha: float = 1.0, fill: bool = True, lw: float = 1.0, z: int = 5) -> None:
        if alpha <= 0.004:
            return
        col = (*tuple(color)[:3], alpha)
        self.ax.add_patch(self.patches.Circle(tuple(c), r * self.sc, facecolor=col if fill else "none", edgecolor="none" if fill else col,
                                              linewidth=lw * self.sc, zorder=z))

    def glow(self, c, r: float, color, alpha: float = 1.0, z: int = 4) -> None:
        for k, a in ((1.0, 0.07), (0.7, 0.12), (0.45, 0.22), (0.25, 0.45)):
            self.circle(c, r * k, color, alpha * a, z=z)

    def rect(self, x: float, y: float, w: float, h: float, color, alpha: float = 1.0, z: int = 3, ec=None, lw: float = 0.0, ealpha: float = 1.0,
             radius: float = 0.0) -> None:
        if alpha <= 0.004 and (ec is None or ealpha <= 0.004):
            return
        fc = (*tuple(color)[:3], alpha) if color is not None else "none"
        e = (*tuple(ec)[:3], ealpha) if ec is not None else "none"
        if radius > 0:
            self.ax.add_patch(self.patches.FancyBboxPatch((x + radius * self.sc, y + radius * self.sc), w - 2 * radius * self.sc, h - 2 * radius * self.sc,
                                                          boxstyle=f"round,pad={radius * self.sc},rounding_size={radius * self.sc}", facecolor=fc, edgecolor=e,
                                                          linewidth=lw * self.sc, zorder=z))
        else:
            self.ax.add_patch(self.patches.Rectangle((x, y), w, h, facecolor=fc, edgecolor=e, linewidth=lw * self.sc, zorder=z))

    def arrow(self, p0, p1, color, alpha: float = 1.0, w: float = 2.0, hl: float = 9.0, hw: float = 8.0, z: int = 6) -> None:
        """A filled arrow from p0 to p1 (pixels); the width, head length and head width are in 720 p pixels."""
        if alpha <= 0.004:
            return
        p0, p1 = np.asarray(p0, float), np.asarray(p1, float)
        d = p1 - p0
        L = float(np.hypot(*d))
        if L < 1e-6:
            return
        u = d / L
        n = np.array([-u[1], u[0]])
        w, hl, hw = w * self.sc, min(hl * self.sc, 0.75 * L), hw * self.sc
        base = p1 - u * hl
        pts = [p0 + n * w / 2, base + n * w / 2, base + n * hw / 2, p1, base - n * hw / 2, base - n * w / 2, p0 - n * w / 2]
        self.poly(pts, fc=color, alpha=alpha, z=z)

    def finish(self) -> np.ndarray:
        self.fig.canvas.draw()
        return np.asarray(self.fig.canvas.buffer_rgba()).astype(np.float32)

    def text_boxes(self) -> list[tuple[str, tuple[float, float, float, float], float]]:
        """(string, display bounding box, opacity) of every text drawn in the frame (call after finish())."""
        r = self.fig.canvas.get_renderer()
        out = []
        for t, a in self.texts:
            bb = t.get_window_extent(r)
            out.append((t.get_text(), (bb.x0, bb.y0, bb.x1, bb.y1), a))
        return out


def mix(c1, c2, u: float):
    return tuple(a + (b - a) * u for a, b in zip(c1, c2))


def rgb(name: str):
    return tuple(CFG.style[name])[:3]


CH_COLOR = {ab: tuple(CFG.style[f"ch_{ab}"]) for ab in CHANNELS}


# ----------------------------------------------------------------------------------- pictograms

def vec(ang: float) -> np.ndarray:
    """Unit vector on the screen (y downwards) of the mathematical angle ang (y up)."""
    return np.array([math.cos(ang), -math.sin(ang)])


def perp(ang: float) -> np.ndarray:
    return vec(ang + 0.5 * math.pi)


def spin_arrow(P: Painter, ctr, d, alpha: float = 1.0, scale: float = 1.0, length: float | None = None) -> None:
    """A spin arrow (gold) centred at ctr along the unit screen vector d."""
    IC = CFG.icon
    half = 0.5 * (IC.spin_len if length is None else length) * scale * P.sc
    ctr, d = np.asarray(ctr, float), np.asarray(d, float)
    P.arrow(ctr - d * half, ctr + d * half, rgb("gold"), alpha, w=IC.spin_width * scale, hl=IC.spin_head[0] * scale, hw=IC.spin_head[1] * scale, z=9)


def particle(P: Painter, c, color, alpha: float = 1.0, r: float | None = None, hollow: bool = False) -> None:
    IC = CFG.icon
    r = IC.radius if r is None else r
    P.glow(c, IC.glow_radius * r / IC.radius, color, alpha * IC.glow_alpha / 0.16 * 0.9, z=7)
    if hollow:
        P.circle(c, r, color, alpha * IC.particle_alpha, fill=False, lw=1.8, z=8)
        P.circle(c, r * 0.55, color, alpha * 0.35, z=8)
    else:
        P.circle(c, r, color, alpha * IC.particle_alpha, z=8)


def lepton(P: Painter, c, mom, spin, side, hollow: bool = False, alpha: float = 1.0, scale: float = 1.0, color=None) -> None:
    """A fermion at c: a sphere (hollow for an antiparticle), the momentum (white arrow from the sphere along mom) and the spin (gold arrow
    along spin, drawn to the side of the sphere, side = unit vector)."""
    IC = CFG.icon
    c, mom, side = np.asarray(c, float), np.asarray(mom, float), np.asarray(side, float)
    white = rgb("white") if color is None else color
    sc = P.sc * scale
    p0 = c + mom * (IC.radius + IC.momentum_gap) * sc
    p1 = c + mom * (IC.radius + IC.momentum_gap + IC.momentum_len) * sc
    P.arrow(p0, p1, white, alpha * 0.9, w=IC.momentum_width * scale, hl=IC.momentum_head[0] * scale, hw=IC.momentum_head[1] * scale, z=8)
    particle(P, c, white, alpha, r=IC.radius * scale, hollow=hollow)
    spin_arrow(P, c + side * IC.spin_offset * sc, spin, alpha, scale)


def pair_icon(P: Painter, ctr, ang: float, spin_sign: float, alpha: float = 1.0, scale: float = 1.0, side_sign: float = 1.0) -> None:
    """The pair f fbar flying apart along the direction ang (f at +, solid; fbar at -, hollow): the momenta are white, both spins are gold;
    spin_sign = -1: the spins point against the flight of f (f_L fbar_R), +1: along it (f_R fbar_L)."""
    IC = CFG.icon
    ctr = np.asarray(ctr, float)
    n = vec(ang)
    side = perp(ang) * side_sign
    a = IC.half_length * scale * P.sc
    for sgn, hollow in ((+1.0, False), (-1.0, True)):
        lepton(P, ctr + sgn * n * a, sgn * n, spin_sign * n, side, hollow, alpha, scale)


def beams_icon(P: Painter, ctr, spin_dir: float, alpha: float = 1.0, scale: float = 1.0, half: float | None = None, color=None) -> None:
    """e- (left, solid) and e+ (right, hollow) approaching each other along x; both spins point along spin_dir * x (the Z gets J_z = spin_dir)."""
    IC = CFG.icon
    ctr = np.asarray(ctr, float)
    a = (IC.beam_half_length if half is None else half) * scale * P.sc
    up = np.array([0.0, -1.0])
    ex = np.array([1.0, 0.0])
    lepton(P, ctr - ex * a, ex, spin_dir * ex, up, False, alpha, scale, color)
    lepton(P, ctr + ex * a, -ex, spin_dir * ex, up, True, alpha, scale, color)


# ----------------------------------------------------------------------------------- the frame furniture

def draw_title(P: Painter, tx: dict, part: int | None) -> None:
    L, F, S = CFG.layout, CFG.fonts, CFG.style
    P.text(P.X(L.title_pos[0]), P.Y(L.title_pos[1]), tx["title"], F.title, S.title_color, S.title_color[3])
    if part is not None:
        P.text(P.X(L.chapter_label_pos[0]), P.Y(L.chapter_label_pos[1]), f"{part}/{N_PARTS}   " + tx[f"ch{part}"], F.chapter_label, S.dim,
               S.chapter_label_alpha, ha="right")


def draw_caption(P: Painter, tx: dict, key: str | None, alpha: float = 1.0) -> None:
    if key:
        L, F, S = CFG.layout, CFG.fonts, CFG.style
        P.text(P.X(L.caption_pos[0]), P.Y(L.caption_pos[1]), tx[key], F.caption, S.text, S.caption_alpha * alpha, va="bottom", linespacing=1.35)


def draw_formula(P: Painter, tx: dict, key: str, pos, alpha: float, size: float | None = None, color=None) -> None:
    if alpha > CFG.layout.formula_min_alpha:
        S = CFG.style
        P.text(P.X(pos[0]), P.Y(pos[1]), tx[key], size or CFG.fonts.formula, color or S.text, S.formula_alpha * alpha)


def draw_card(P: Painter, tx: dict, card: int, prog: float) -> None:
    L, F, S, T = CFG.layout, CFG.fonts, CFG.style, CFG.timeline
    a = min(smooth(prog, *T.card_fade_in), 1.0 - smooth(prog, *T.card_fade_out))
    ch = CARD_CHAPTER[card]
    cx = 0.5 * P.W
    if card == 0:
        P.text(cx, P.Y(L.card_title_y), tx["h0"], F.card_title, S.card_title_color, a, ha="center")
        P.text(cx, P.Y(L.card_subtitle_y), tx["h0s"], F.card_subtitle, S.dim, a, ha="center")
    else:
        P.text(cx, P.Y(L.card_number_y), f"{ch}", F.card_number, S.gold, S.card_number_alpha * a, ha="center")
        P.text(cx, P.Y(L.card_chapter_y), tx[f"ch{ch}"], F.card_chapter, S.card_title_color, a, ha="center")
        P.text(cx, P.Y(L.card_chapter_sub_y), tx[f"ch{ch}s"], F.card_chapter_sub, S.dim, a, ha="center")


# ----------------------------------------------------------------------------------- part 1: the couplings

def draw_part1(P: Painter, t: float, tx: dict, lang: str) -> str:
    C, LY, F, S, IC = CFG.couplings, CFG.couplings.layout, CFG.fonts, CFG.style, CFG.icon
    blue, violet, gold, white, dim = rgb("left_c"), rgb("right_c"), rgb("gold"), rgb("white"), rgb("dim")
    sc = P.sc
    for (a, b), pos, key, size in zip(C.formula_in, LY.formula_pos, ("f1_va", "f1_L", "f1_R", "f1_cur"), (F.formula, F.formula, F.formula, F.formula_small)):
        draw_formula(P, tx, key, pos, smooth(t, a, b), size)
    # the legend: the pair f fbar with both spins to the right (f_L fbar_R) or to the left (f_R fbar_L)
    la = smooth(t, *C.legend_in)
    half = (IC.half_length + IC.radius + IC.momentum_gap + IC.momentum_len) * LY.legend_icon_scale
    for k, (key, col, sgn) in enumerate((("leg_L", blue, -1.0), ("leg_R", violet, +1.0))):
        x0, y0 = P.pt(LY.legend_pos[k])
        pair_icon(P, (x0 + half * sc, y0), math.pi, sgn, la, scale=LY.legend_icon_scale, side_sign=-1.0)
        P.text(x0 + (2 * half + LY.legend_text_gap) * sc, y0, tx[key], F.legend, col, la)
    # the bars
    base, unit, bw, gap = P.Y(LY.baseline_y), P.Y(LY.unit_height), P.X(LY.bar_width), P.X(LY.bar_gap)
    first = min(C.group_start)
    P.line([(P.X(0.065), base), (P.X(0.925), base)], white, LY.baseline_alpha * smooth(t, first - 0.6, first), LY.baseline_width)
    hl = smooth(t, *C.highlight_at)
    for i, fam in enumerate(C.groups):
        cx = P.X(LY.group_x[i])
        c = couplings(fam)
        t0 = C.group_start[i]
        p = smooth(t, t0, t0 + C.group_grow_s)
        ra = smooth(t, t0 + C.readout_delay, t0 + C.readout_delay + 0.6)
        if fam == "e" and hl > 0:
            pd_ = [v * sc for v in C.highlight_pad]
            P.rect(cx - bw - gap / 2 - pd_[0], base - unit * max(c["L"] ** 2, c["R"] ** 2) - pd_[1] - 24 * sc,
                   2 * bw + gap + pd_[0] + pd_[2], unit * max(c["L"] ** 2, c["R"] ** 2) + pd_[1] + 24 * sc + (LY.a_y - LY.baseline_y) * P.H + pd_[3] + 6 * sc,
                   None, ec=gold, lw=1.4, ealpha=C.highlight_alpha * hl, radius=10, z=2)
        for side, key_, val, col in ((-1, "g_L", c["L"] ** 2, blue), (+1, "g_R", c["R"] ** 2, violet)):
            x0 = cx - gap / 2 - bw if side < 0 else cx + gap / 2
            h = val * unit * p
            P.rect(x0, base - h, bw, h, col, LY.bar_alpha * (1.0 if h > 0.5 else 0.0), z=3)
            lab = "0" if val < 5e-4 else num(val, f".{C.value_decimals}f", lang, math_mode=False)
            P.text(x0 + bw / 2, base - h + P.Y(LY.value_dy), lab, F.value, col, smooth(p, 0.5, 1.0), ha="center", va="bottom")
            P.text(x0 + bw / 2, base + P.Y(LY.lr_label_dy), tx[key_], F.label_small, col, p, ha="center")
        pa = smooth(t, t0 - 0.2, t0 + 0.5)
        hot = fam == "e" and hl > 0
        P.text(cx, P.Y(LY.name_y), tx[f"n_{fam}"], F.group_name, S.text, pa, ha="center")
        P.text(cx, P.Y(LY.charge_y), tx[f"q_{fam}"], F.group_sub, dim, pa, ha="center")
        P.text(cx, P.Y(LY.a_y), tx[f"a_{fam}"].replace("@v@", num(a_lr(fam), f".{C.a_decimals}f" if abs(a_lr(fam) - 1) > 1e-9 else ".0f", lang)),
               F.readout, mix(S.text, gold, 0.7 if hot else 0.0), ra, ha="center")
    k = max(i for i, a in enumerate(C.caption_at) if t >= a)
    return ("c1a", "c1b", "c1c", "c1d")[k]


# ----------------------------------------------------------------------------------- part 2: the four helicity channels

def lobe_arc(ctr, unit: float, r_of_cos, phi_end: float, n: int) -> np.ndarray:
    """Points ctr + unit r(cos phi) (cos phi, -sin phi) for phi in [-phi_end, phi_end] (the forward direction is +x on the screen)."""
    phis = np.linspace(-phi_end, phi_end, max(3, int(n * phi_end / math.pi) + 3))
    r = r_of_cos(np.cos(phis))
    return np.asarray(ctr, float) + unit * r[:, None] * np.stack([np.cos(phis), -np.sin(phis)], axis=1)


def draw_lobe(P: Painter, ctr, unit: float, r_of_cos, phi_end: float, color, fill_alpha: float, line_alpha: float, lw: float, n: int, z: int = 3) -> None:
    if phi_end <= 1e-6:
        return
    arc = lobe_arc(ctr, unit, r_of_cos, phi_end, n)
    P.poly(np.vstack([np.asarray(ctr, float)[None, :], arc]), fc=color, alpha=fill_alpha, z=z)
    P.line(arc, color, line_alpha, lw, z=z + 1)


def text_width(P: Painter, t) -> float:
    return t.get_window_extent(P.fig.canvas.get_renderer()).width if t is not None else 0.0


def draw_part2(P: Painter, t: float, tx: dict, lang: str) -> str:
    C, LY, CL, F, S = CFG.channels, CFG.channels.layout, CFG.channels.cells, CFG.fonts, CFG.style
    sc = P.sc
    gold, white, dim = rgb("gold"), rgb("white"), rgb("dim")
    shares = channel_shares(C.fermion)
    ctr = np.array(P.pt(LY.stage_centre))
    unit = LY.stage_unit * sc
    cell_unit = CL.unit * sc
    ca = smooth(t, *C.cells_in)
    sweeps = dict(zip(CHANNELS, C.sweeps))
    summed = smooth(t, *C.sum_in)
    # ------------------------------------------------------------------ formulas
    draw_formula(P, tx, "f2_M", LY.formula_pos[0], ca, CFG.fonts.formula_small)
    draw_formula(P, tx, "f2_dn", LY.formula_pos[1], smooth(t, *C.dn_in), CFG.fonts.formula_small)
    a_afb = smooth(t, *C.afb_in)
    t_afb = P.text(P.X(LY.formula_pos[2][0]), P.Y(LY.formula_pos[2][1]), tx["f2_afb"], F.formula, S.text, S.formula_alpha * a_afb)
    a_num = smooth(t, *C.afb_num_in)
    if t_afb is not None and a_num > 0:
        x1 = P.X(LY.formula_pos[2][0]) + text_width(P, t_afb) + 14 * sc
        val = lambda x: num(x, ".3f", lang)
        P.text(x1, P.Y(LY.formula_pos[2][1]), tx["f2_num"].replace("@e@", val(a_lr("e"))).replace("@f@", val(a_lr(C.fermion)))
               .replace("@v@", val(a_fb(C.fermion))), F.formula, gold, S.formula_alpha * a_num)
    # ------------------------------------------------------------------ the beam axis, forward / backward, the legend
    ax_l, ax_r = ctr - np.array([LY.axis_half * sc, 0.0]), ctr + np.array([LY.axis_half * sc, 0.0])
    P.line([ax_l, ax_r], white, LY.axis_alpha * ca, LY.axis_width, ls=(0, tuple(LY.axis_dash)), z=2)
    P.text(ax_r[0] - LY.fb_label_inset * sc, ax_r[1] + LY.fb_label_dy * sc, tx["forward"], F.label_small, dim, ca, ha="right", va="bottom")
    P.text(ax_l[0] + LY.fb_label_inset * sc, ax_l[1] + LY.fb_label_dy * sc, tx["backward"], F.label_small, dim, ca, ha="left", va="bottom")
    bm = smooth(t, *C.mini_in)
    for end, sgn, key in ((ax_l, +1.0, "e_minus"), (ax_r, -1.0, "e_plus")):
        a0 = end + np.array([sgn * 6 * sc, LY.beam_marker_dy * sc])
        P.arrow(a0, a0 + np.array([sgn * LY.beam_marker_len * sc, 0.0]), white, 0.8 * bm, w=2.0, hl=9, hw=8, z=8)
        P.text(a0[0] + sgn * 0.5 * LY.beam_marker_len * sc, a0[1] + 16 * sc, tx[key], F.label_small, white, bm, ha="center")
    lg = np.array(P.pt(LY.legend_pos))
    P.arrow(lg, lg + np.array([LY.legend_arrow_len * sc, 0.0]), white, 0.9 * ca, w=2.0, hl=9, hw=8, z=8)
    P.text(lg[0] + (LY.legend_arrow_len + 8) * sc, lg[1], tx["lg_momentum"], F.label_small, dim, ca, ha="left")
    lg2 = lg + np.array([LY.legend_gap * sc, 0.0])
    P.arrow(lg2, lg2 + np.array([LY.legend_arrow_len * sc, 0.0]), gold, ca, w=3.4, hl=11, hw=11, z=8)
    P.text(lg2[0] + (LY.legend_arrow_len + 8) * sc, lg2[1], tx["lg_spin"], F.label_small, dim, ca, ha="left")
    # ------------------------------------------------------------------ the initial state: the beams come together and form the Z
    mv = smooth(t, *C.beams_move)
    ba = smooth(t, *C.beams_in) * (1.0 - smooth(t, *C.z_spin_out))
    if ba > 0:
        d = lerp(LY.beam_start_x, 0.0, mv) * sc
        fade = ba * (1.0 - smooth(t, C.beams_move[1] - 0.25, C.beams_move[1] + 0.25))
        up = np.array([0.0, -1.0])
        ex = np.array([1.0, 0.0])
        lepton(P, ctr - ex * d, ex, -ex, up, False, fade)
        lepton(P, ctr + ex * d, -ex, -ex, up, True, fade)
        la_ = fade * smooth(d, 70 * sc, 130 * sc)
        P.text(ctr[0] - d, ctr[1] + LY.beam_label_dy * sc, tx["e_minus_L"], F.label, white, la_, ha="center")
        P.text(ctr[0] + d, ctr[1] + LY.beam_label_dy * sc, tx["e_plus_R"], F.label, white, la_, ha="center")
    za = smooth(t, *C.z_in)
    if za > 0:
        P.glow(ctr, LY.z_radius * sc * (0.6 + 0.4 * za), (0.85, 0.92, 1.0), LY.z_alpha * za, z=10)
        zs = za * (1.0 - smooth(t, *C.z_spin_out))
        if zs > 0:
            spin_arrow(P, ctr + np.array([0.0, -34.0 * sc]), np.array([-1.0, 0.0]), zs, 1.0, length=LY.j_spin_len)
            P.text(ctr[0], ctr[1] + LY.z_label_dy * sc - 6 * sc, tx["jz_m"], F.label, gold, zs, ha="center")
            P.text(ctr[0] + 30 * sc, ctr[1] + 8 * sc, "Z", F.label, white, zs, ha="left", style="italic")
    # ------------------------------------------------------------------ the initial-state pictogram (top left of the stage)
    mini = smooth(t, *C.mini_in)
    if mini > 0:
        J = -1.0 + 2.0 * smooth(t, *C.flip)
        mp = np.array(P.pt(LY.mini_pos))
        mc = mp
        beams_icon(P, mc, J, mini, LY.mini_scale)
        lab = tx["init_L"] if J < 0 else tx["init_R"]
        P.text(mc[0], mc[1] - 44 * sc * LY.mini_scale, lab, F.label, white, mini, ha="center")
        ax0 = mc + np.array([LY.mini_arrow_gap * sc * LY.mini_scale, 0.0])
        P.arrow(ax0, ax0 + np.array([LY.mini_arrow_len * sc, 0.0]), white, mini * 0.9, w=2.0, hl=9, hw=8, z=8)
        zc = ax0 + np.array([(LY.mini_arrow_len + LY.mini_z_gap) * sc, 0.0])
        P.glow(zc, 11 * sc, (0.85, 0.92, 1.0), mini, z=7)
        spin_arrow(P, zc + np.array([0.0, -20.0 * sc]), np.array([J, 0.0]), mini * (1.0 if abs(J) > 0.15 else 0.0), 1.0, length=LY.j_spin_len)
        P.text(zc[0] + LY.mini_text_gap * sc, zc[1], tx["jz_m"] if J < 0 else tx["jz_p"], F.label, gold, mini, ha="left")
    # ------------------------------------------------------------------ the cells, the lobes of the channels
    ox, oy = P.pt(CL.origin)
    cw, chh = P.X(CL.size[0]), P.Y(CL.size[1])
    active_k = None
    for k, ab in enumerate(CHANNELS):
        r, c_ = divmod(k, 2)
        x = ox + c_ * (cw + P.X(CL.gap[0]))
        y = oy + r * (chh + P.Y(CL.gap[1]))
        t0, t1 = sweeps[ab]
        u = smooth(t, t0, t1)
        active = t0 <= t < t1
        if active:
            active_k = k
        col = CH_COLOR[ab][:3]
        started = smooth(t, t0 - 0.3, t0 + 0.3)
        P.rect(x, y, cw, chh, None, ec=col, lw=CL.border_width, ealpha=ca * (CL.border_alpha_idle + (CL.border_alpha_active - CL.border_alpha_idle) * (1.0 if active else started * 0.6)),
               radius=CL.radius, z=2)
        P.text(x + 14 * sc, y + CL.title_dy * sc, tx[f"cell_{ab}"], F.cell_title, col, ca * (CL.title_idle_alpha + (1 - CL.title_idle_alpha) * started))
        cc = np.array([x + 0.5 * cw, y + 0.5 * chh + CL.lobe_dy * sc])
        P.line([cc - np.array([0.42 * cw, 0.0]), cc + np.array([0.42 * cw, 0.0])], white, CL.axis_alpha * ca, 0.9, ls=(0, (3, 3)), z=2)
        phi_end = math.pi * u
        draw_lobe(P, cc, cell_unit, lambda cs, ab=ab: shares[ab] * lobe(ab, cs), phi_end, col, LY.lobe_fill_alpha, LY.lobe_line_alpha, 1.4, 120)
        wa = smooth(t, t0, t0 + 0.7)
        P.text(x + 14 * sc, y + chh - CL.weight_dy * sc, tx[f"w_{ab}"], F.label_small, dim, ca * wa)
        P.text(x + cw - 14 * sc, y + chh - CL.weight_dy * sc, num(100 * shares[ab], f".{CL.share_decimals}f", lang, math_mode=False) + " %", F.label, col, ca * wa, ha="right")
        bw_ = (cw - 2 * CL.bar_inset * sc) * shares[ab] * wa
        P.rect(x + CL.bar_inset * sc, y + chh - CL.bar_dy * sc, max(bw_, 0.0), CL.bar_height * sc, col, 0.9 * ca, z=3)
    # ------------------------------------------------------------------ the lobes on the stage
    fill_a = lerp(LY.lobe_fill_alpha, LY.lobe_fill_alpha_summed, summed)
    for ab in CHANNELS:
        t0, t1 = sweeps[ab]
        phi_end = math.pi * smooth(t, t0, t1)
        draw_lobe(P, ctr, unit, lambda cs, ab=ab: shares[ab] * lobe(ab, cs), phi_end, CH_COLOR[ab][:3], fill_a, LY.lobe_line_alpha * lerp(1.0, 0.55, summed), LY.lobe_line_width, LY.lobe_points)
    if summed > 0:
        n_ = LY.lobe_points
        arc = lobe_arc(ctr, unit, lambda cs: distribution(cs, shares), math.pi, n_)
        P.line(arc, white, LY.sum_alpha * summed, LY.sum_line_width, z=6)
        sym = lobe_arc(ctr, unit, lambda cs: 1.0 + cs * cs, math.pi, n_)
        P.line(sym, white, LY.sym_alpha * summed, LY.sym_line_width, ls=(0, tuple(LY.sym_dash)), z=5)
    # ------------------------------------------------------------------ the sweeping pair
    if active_k is not None:
        ab = CHANNELS[active_k]
        t0, t1 = sweeps[ab]
        phi = math.pi * smooth(t, t0, t1)
        col = CH_COLOR[ab][:3]
        n_ = vec(phi)
        r_tip = shares[ab] * float(lobe(ab, math.cos(phi))) * unit
        P.line([ctr, ctr + n_ * r_tip], col, LY.ray_alpha, LY.ray_width, z=4)
        P.circle(ctr + n_ * r_tip, LY.trace_dot * sc, col, 1.0, z=7)
        pair_icon(P, ctr, phi, -1.0 if ab[1] == "L" else +1.0, 1.0, scale=LY.icon_scale, side_sign=-1.0)
        arc = lobe_arc(ctr, LY.arc_radius * sc, lambda cs: np.ones_like(cs), phi, 40)
        P.line(arc[len(arc) // 2:], gold, LY.arc_alpha, LY.arc_width, z=7)
        deg = int(round(math.degrees(phi)))
        mid = vec(0.5 * phi)
        P.text(*(ctr + mid * LY.theta_label_radius * sc), tx["theta_fmt"].replace("@v@", str(deg)), F.label_small, gold, 1.0, ha="left")
    k_cap = max(i for i, a in enumerate(C.caption_at) if t >= a)
    return ("c2a", "c2b", "c2c", "c2d", "c2e", "c2f", "c2g")[k_cap]


# ----------------------------------------------------------------------------------- plot boxes

class Box:
    """A rectangle of the frame (pixels) with its own data coordinates."""

    def __init__(self, P: Painter, frac, xlim, ylim) -> None:
        self.P = P
        self.x0, self.y0, self.x1, self.y1 = frac[0] * P.W, frac[1] * P.H, frac[2] * P.W, frac[3] * P.H
        self.xlim, self.ylim = xlim, ylim

    def px(self, x, y):
        x, y = np.asarray(x, float), np.asarray(y, float)
        return self.x0 + (x - self.xlim[0]) / (self.xlim[1] - self.xlim[0]) * (self.x1 - self.x0), \
            self.y1 - (y - self.ylim[0]) / (self.ylim[1] - self.ylim[0]) * (self.y1 - self.y0)

    def curve(self, x, y) -> np.ndarray:
        X, Y = self.px(x, y)
        return np.stack([X, Y], axis=1)


def draw_box_axes(P: Painter, box: Box, xticks, yticks, xfmt, yfmt, alpha: float, tick_len: float, axis_alpha: float, axis_width: float,
                  grid_alpha: float, xlabels: bool = True, ytick_fontsize: float = 11.0, y_zero_line: bool = False) -> None:
    dim = rgb("dim")
    white = rgb("white")
    P.line([(box.x0, box.y1), (box.x1, box.y1)], white, axis_alpha * alpha, axis_width, z=2)
    P.line([(box.x0, box.y0), (box.x0, box.y1)], white, axis_alpha * alpha, axis_width, z=2)
    for xt in xticks:
        X, Y = box.px(xt, box.ylim[0])
        P.line([(X, box.y1), (X, box.y1 + tick_len * P.sc)], white, axis_alpha * alpha, axis_width, z=2)
        P.line([(X, box.y0), (X, box.y1)], white, grid_alpha * alpha, 0.8, z=1)
        if xlabels:
            P.text(X, box.y1 + (tick_len + 11) * P.sc, xfmt(xt), ytick_fontsize, dim, alpha, ha="center")
    for yt in yticks:
        X, Y = box.px(box.xlim[0], yt)
        P.line([(box.x0 - tick_len * P.sc, Y), (box.x0, Y)], white, axis_alpha * alpha, axis_width, z=2)
        P.line([(box.x0, Y), (box.x1, Y)], white, grid_alpha * alpha * (2.0 if (y_zero_line and abs(yt) < 1e-12) else 1.0), 0.8, z=1)
        P.text(box.x0 - (tick_len + 5) * P.sc, Y, yfmt(yt), ytick_fontsize, dim, alpha, ha="right")


# ----------------------------------------------------------------------------------- part 3: the weak mixing angle

def draw_part3(P: Painter, t: float, tx: dict, lang: str) -> str:
    W3, LY, F, S, CL = CFG.weak_angle, CFG.weak_angle.layout, CFG.fonts, CFG.style, CFG.channels
    sc = P.sc
    gold, white, dim, blue, violet = rgb("gold"), rgb("white"), rgb("dim"), rgb("left_c"), rgb("right_c")
    mu_c, b_c = rgb("mu_c"), rgb("b_c")
    pa = smooth(t, *W3.plots_in)
    k = smooth(t, *W3.slide)
    s2 = lerp(W3.s2_start, S2W, k)
    fm = {"e": ("mu", mu_c), "d": ("b", b_c)}
    # ------------------------------------------------------------------ formulas
    draw_formula(P, tx, "f3_g", LY.formula_pos[0], smooth(t, *W3.formula_in[0]), F.formula_small)
    draw_formula(P, tx, "f3_a", LY.formula_pos[1], smooth(t, *W3.formula_in[1]), F.formula_small)
    # ------------------------------------------------------------------ the distributions
    box = Box(P, LY.dist_box, (-1.0, 1.0), LY.dist_ylim)
    draw_box_axes(P, box, LY.xticks, LY.dist_yticks, lambda v: "0" if v == 0 else num(v, "g", lang, math_mode=False).replace("-", "\u2212"),
                  lambda v: num(v, "g", lang, math_mode=False), pa, LY.tick_len, LY.axis_alpha, LY.axis_width, LY.grid_alpha, xlabels=False)
    box2 = Box(P, LY.asym_box, (-1.0, 1.0), LY.asym_ylim)
    draw_box_axes(P, box2, LY.xticks, LY.asym_yticks, lambda v: ("\u2212" if v < 0 else "") + num(abs(v), "g", lang, math_mode=False),
                  lambda v: ("\u2212" if v < 0 else "") + num(abs(v), "g", lang, math_mode=False), pa, LY.tick_len, LY.axis_alpha, LY.axis_width, LY.grid_alpha,
                  y_zero_line=True)
    P.text(box.x0 + 8 * sc, box.y0 + 12 * sc, tx["dn_label"], F.label_small, dim, pa, ha="left")
    P.text(box2.x0 + 8 * sc, box2.y0 - 12 * sc, tx["asym_label"], F.label_small, dim, pa, ha="left", va="bottom")
    P.text(box2.x1, box2.y1 + 38 * sc, tx["cos_label"], F.label_small, dim, pa, ha="right")
    cs = np.linspace(-1.0, 1.0, LY.curve_samples)
    P.line(box.curve(cs, 1.0 + cs * cs), white, LY.sym_alpha * pa, LY.sym_width, ls=(0, tuple(LY.sym_dash)), z=3)
    P.line(box2.curve(cs, 0.0 * cs), white, 0.0, 1.0)
    for f in W3.a_fb_fermions:
        name, col = fm[f]
        a = a_fb(f, s2)
        P.line(box.curve(cs, shape(cs, a)), col, pa, LY.curve_width, z=5)
        P.line(box2.curve(cs, (8.0 / 3.0) * a * cs), col, pa, LY.curve_width, z=5)
    # the legend above the plot
    ly = P.Y(LY.legend_y)
    for xf, name, col, dash in ((LY.legend_x[0], "b", b_c, None), (LY.legend_x[1], "mu", mu_c, None), (LY.legend_x[2], "sym", white, (0, tuple(LY.sym_dash)))):
        x = P.X(xf)
        P.line([(x, ly), (x + LY.legend_line_len * sc, ly)], col, pa * (LY.sym_alpha if dash else 1.0), LY.curve_width if not dash else LY.sym_width, ls=dash or "-", z=5)
        P.text(x + (LY.legend_line_len + 8) * sc, ly, tx[f"leg3_{name}"], F.label_small, col, pa, ha="left")
    # ------------------------------------------------------------------ the slider
    sx0, sx1 = P.X(LY.slider_x[0]), P.X(LY.slider_x[1])
    sy = P.Y(LY.slider_y)
    lo, hi = W3.s2_range

    def xs(v: float) -> float:
        return sx0 + (v - lo) / (hi - lo) * (sx1 - sx0)

    band = smooth(t, *W3.err_band_in)
    if band > 0:
        xa, xb = xs(S2W - CFG.physics.sin2_theta_w_err), xs(S2W + CFG.physics.sin2_theta_w_err)
        P.rect(xa, sy - 0.5 * LY.err_band_height * sc, xb - xa, LY.err_band_height * sc, gold, 0.28 * band, z=2)
    P.line([(sx0, sy), (sx1, sy)], white, LY.slider_track_alpha * pa, LY.slider_track_width, z=3)
    for v in W3.s2_ticks:
        P.line([(xs(v), sy - LY.slider_tick_len * sc), (xs(v), sy + LY.slider_tick_len * sc)], white, LY.slider_track_alpha * pa, 1.2, z=3)
        P.text(xs(v), sy + LY.slider_tick_label_dy * sc, num(v, ".2f", lang, math_mode=False), F.label_small, dim, pa, ha="center")
    qx = xs(W3.s2_start)
    P.line([(qx, sy - 12 * sc), (qx, sy + 12 * sc)], gold, pa, 2.0, z=4)
    P.text(qx, sy + LY.quarter_label_dy * sc, tx["quarter"], F.label_small, gold, pa, ha="center")
    kx = xs(s2)
    P.glow((kx, sy), LY.slider_knob * sc * 2.0, gold, pa, z=5)
    P.circle((kx, sy), LY.slider_knob * sc, gold, pa, z=6)
    P.text(kx, sy + LY.slider_label_dy * sc, tx["s2_fmt"].replace("@v@", num(s2, f".{W3.decimals_s2}f", lang)), F.label, gold, pa, ha="center")
    if band > 0:
        P.text(xs(S2W), sy + (LY.quarter_label_dy + 0) * sc, tx["book_value"], F.label_small, gold, band, ha="center")
    # ------------------------------------------------------------------ the couplings of the electron
    base = P.Y(LY.bars_baseline_y)
    unit = P.Y(LY.bars_unit)
    ce = couplings("e", s2)
    P.line([(P.X(LY.bars_x[0]) - 10 * sc, base), (P.X(LY.bars_x[1]) + P.X(LY.bar_width) + 10 * sc, base)], white, LY.slider_track_alpha * pa, 1.0, z=3)
    for xf, key, val, col in ((LY.bars_x[0], "ge_L", ce["L"] ** 2, blue), (LY.bars_x[1], "ge_R", ce["R"] ** 2, violet)):
        x = P.X(xf)
        h = val * unit
        P.rect(x, base - h, P.X(LY.bar_width), h, col, 0.9 * pa, z=3)
        P.text(x + 0.5 * P.X(LY.bar_width), base - h - 6 * sc, num(val, ".3f", lang, math_mode=False), F.value, col, pa, ha="center", va="bottom")
        P.text(x + 0.5 * P.X(LY.bar_width), base + P.Y(LY.bar_label_dy), tx[key], F.label_small, col, pa, ha="center")
    # ------------------------------------------------------------------ the readouts
    rows = (("r_alr", white, a_lr("e", s2)), ("r_mu", mu_c, a_fb("e", s2)), ("r_b", b_c, a_fb("d", s2)))
    for (key, col, val), yf in zip(rows, LY.readout_y):
        P.text(P.X(LY.readout_x), P.Y(yf), tx[key].replace("@v@", num(val, f".{W3.decimals_a}f", lang)), F.readout + 2, col, pa, ha="left")
    k_cap = max(i for i, a in enumerate(W3.caption_at) if t >= a)
    return ("c3a", "c3b", "c3c")[k_cap]


# ----------------------------------------------------------------------------------- part 4: events and the statistical error

_cache: dict = {}


def event_data() -> dict:
    """The Monte-Carlo events of the film (the same for every language): cos(theta) of the b quarks and of the muons, the display signs, bin indices."""
    if "ev" not in _cache:
        E, SM = CFG.events, CFG.sampling
        nb_ = E.hist_bins
        rng = np.random.default_rng(SM.seed_b)
        cb = sample_cos(rng, SM.n_b, a_fb(E.fermion_b))
        sb = rng.choice([-1.0, 1.0], size=cb.size)
        rng2 = np.random.default_rng(SM.seed_mu)
        cm = sample_cos(rng2, SM.n_mu, a_fb(E.fermion_mu))
        idx = lambda c: np.clip(((c + 1.0) * 0.5 * nb_).astype(np.int64), 0, nb_ - 1)
        _cache["ev"] = dict(cb=cb, sb=sb, cm=cm, ib=idx(cb), im=idx(cm), cum_f_mu=np.cumsum(cm > 0.0))
    return _cache["ev"]


def n_b_at(t: float) -> int:
    """Number of b events shown at the part time t: the first ones one by one, then a logarithmic growth."""
    E = CFG.events
    k = sum(1 for ts in E.slow_events if t >= ts)
    if t < E.ramp_b[0]:
        return k
    n0 = len(E.slow_events)
    u = smooth(t, *E.ramp_b)
    return int(round(n0 * (CFG.sampling.n_b / n0) ** u))


def n_mu_at(t: float) -> int:
    E = CFG.events
    u = smooth(t, *E.ramp_mu)
    return int(round(E.n_mu_start * (CFG.sampling.n_mu / E.n_mu_start) ** u))


def thousands(n: int, lang: str, math_mode: bool = True) -> str:
    """12345 -> 12\\,345 in a formula (a thin space otherwise)."""
    s = f"{n:,}"
    return s.replace(",", r"\," if math_mode else "\u2009")


def nice_step(ymax: float) -> float:
    e = 10.0 ** math.floor(math.log10(ymax / 3.0))
    for m in (1.0, 2.0, 2.5, 5.0, 10.0):
        if ymax / (m * e) <= 5.0:
            return m * e
    return 10.0 * e


def draw_hist(P: Painter, t: float, tx: dict, lang: str, counts: np.ndarray, n: int, afb_model: float, color, alpha: float) -> None:
    E, LY, F = CFG.events, CFG.events.layout, CFG.fonts
    sc = P.sc
    nb = E.hist_bins
    dc = 2.0 / nb
    cs = np.linspace(-1.0, 1.0, 161)
    exp = n * (3.0 / 8.0) * shape(cs, afb_model) * dc
    ymax = max(E.hist_ymin_top, E.hist_ymargin * max(float(counts.max()), float(exp.max())))
    box = Box(P, LY.hist_box, (-1.0, 1.0), (0.0, ymax))
    step = nice_step(ymax)
    yt = np.arange(0.0, ymax, step)
    draw_box_axes(P, box, LY.hist_xticks, yt, lambda v: ("\u2212" if v < 0 else "") + num(abs(v), "g", lang, math_mode=False),
                  lambda v: thousands(int(round(v)), lang, math_mode=False), alpha, 5.0, 0.55, 1.0, 0.10)
    P.text(box.x1, box.y1 + 38 * sc, tx["cos_label"], F.label_small, rgb("dim"), alpha, ha="right")
    P.text(box.x0, box.y0 - 14 * sc, tx["hist_y"], F.label_small, rgb("dim"), alpha, ha="left", va="bottom")
    edges = np.linspace(-1.0, 1.0, nb + 1)
    for j in range(nb):
        x0, x1 = edges[j] + 0.5 * LY.hist_bar_gap * dc, edges[j + 1] - 0.5 * LY.hist_bar_gap * dc
        X0, Y0 = box.px(x0, 0.0)
        X1, Y1 = box.px(x1, counts[j])
        if counts[j] > 0:
            P.rect(X0, Y1, X1 - X0, Y0 - Y1, color, LY.hist_bar_alpha * alpha, z=3)
    P.line(box.curve(cs, exp), rgb("white"), LY.hist_curve_alpha * alpha, LY.hist_curve_width, ls=(0, (5, 3)), z=5)


def draw_event_display(P: Painter, t: float, tx: dict, lang: str, alpha: float, n: int, ev: dict) -> None:
    E, LY, F = CFG.events, CFG.events.layout, CFG.fonts
    sc = P.sc
    cx, cy = P.pt(LY.ring_centre)
    R = LY.ring_radius * sc
    white, dim, gold, b_c = rgb("white"), rgb("dim"), rgb("gold"), rgb("b_c")
    fcol = b_c
    P.circle((cx, cy), R, white, LY.ring_alpha * alpha, fill=False, lw=LY.ring_width, z=2)
    for deg in range(0, 360, LY.ring_ticks_deg):
        v = vec(math.radians(deg))
        P.line([(cx, cy) + v * R, (cx, cy) + v * (R + LY.tick_len * sc)], white, LY.ring_alpha * alpha, 1.0, z=2)
    P.line([(cx - R - 10 * sc, cy), (cx + R + 10 * sc, cy)], white, LY.axis_alpha * alpha, 1.0, ls=(0, (4, 4)), z=2)
    P.circle((cx, cy), LY.vertex_radius * sc, white, 0.9 * alpha, z=6)
    P.text(cx, cy + LY.label_y * P.H, tx["species_b"], F.label, white, alpha, ha="center")
    # the counters
    if n > 0:
        a_, err, nf, nbk = estimate(ev["cb"][:n])
    else:
        nf = nbk = 0
    P.text(cx + R + LY.counter_dx * sc, cy + LY.counter_y[0] * P.H, tx["forward_short"], F.label_small, dim, alpha, ha="left")
    P.text(cx + R + LY.counter_dx * sc, cy + LY.counter_y[1] * P.H, tx["nf_fmt"].replace("@v@", thousands(nf, lang)), F.readout, white, alpha, ha="left")
    P.text(cx - R - LY.counter_dx * sc, cy + LY.counter_y[0] * P.H, tx["backward_short"], F.label_small, dim, alpha, ha="right")
    P.text(cx - R - LY.counter_dx * sc, cy + LY.counter_y[1] * P.H, tx["nb_fmt"].replace("@v@", thousands(nbk, lang)), F.readout, white, alpha, ha="right")
    # the rays: the first events one by one, then the latest ones of the burst
    def ray(c: float, sgn: float, a: float, grow: float) -> None:
        phi = sgn * math.acos(min(1.0, max(-1.0, c)))
        d = vec(phi)
        r1 = R * grow
        P.line([(cx, cy), (cx, cy) - d * r1], white, LY.ray_alpha_fbar * a * alpha, LY.ray_width_fbar, z=3)
        P.line([(cx, cy), (cx, cy) + d * r1], fcol, a * alpha, LY.ray_width_f, z=4)
        if grow > 0.98:
            P.circle((cx, cy) + d * R, LY.hit_radius * sc, fcol, a * alpha, z=5)
            P.circle((cx, cy) - d * R, LY.hit_radius * sc * 0.8, white, 0.6 * a * alpha, fill=False, lw=1.2, z=5)
    slow = list(E.slow_events)
    for i, ts in enumerate(slow):
        age = t - ts
        if age < 0 or age > E.persist[1]:
            continue
        a = 1.0 - smooth(age, *E.persist)
        ray(float(ev["cb"][i]), float(ev["sb"][i]), a, smooth(age, 0.0, E.flight_s))
        fl = 1.0 - smooth(age, 0.0, 0.3)
        P.glow((cx, cy), LY.flash_radius * sc, gold, fl * alpha, z=5)
    if t >= E.ramp_b[0] and n > len(slow):
        K = min(E.burst_events, n - len(slow))
        for j in range(K):
            i = n - 1 - j
            ray(float(ev["cb"][i]), float(ev["sb"][i]), E.burst_alpha * (1.0 - j / K), 1.0)


def draw_convergence(P: Painter, t: float, tx: dict, lang: str, n: int, ev: dict, alpha: float) -> None:
    E, LY, F = CFG.events, CFG.events.layout, CFG.fonts
    sc = P.sc
    box = Box(P, LY.conv_box, tuple(E.conv_xlog), tuple(E.conv_ylim))
    mu_c, gold, white, dim = rgb("mu_c"), rgb("gold"), rgb("white"), rgb("dim")
    draw_box_axes(P, box, E.conv_xticks, E.conv_yticks, lambda v: rf"$10^{{{int(v)}}}$",
                  lambda v: ("\u2212" if v < 0 else "") + num(abs(v), ".1f" if abs(v) > 1e-9 else ".0f", lang, math_mode=False), alpha, 5.0, 0.55, 1.0, 0.10,
                  y_zero_line=True)
    P.text(box.x1, box.y1 + 38 * sc, tx["conv_x"], F.label_small, dim, alpha, ha="right")
    P.text(box.x0, box.y0 - 14 * sc, tx["conv_y"], F.label_small, dim, alpha, ha="left", va="bottom")
    model = a_fb(E.fermion_mu)
    lo = 10.0 ** E.conv_xlog[0]
    ns = np.unique(np.round(np.logspace(math.log10(E.n_mu_start), math.log10(max(n, E.n_mu_start + 1)), E.conv_samples)).astype(np.int64))
    est = (2.0 * ev["cum_f_mu"][ns - 1] - ns) / ns
    err = np.sqrt(np.maximum(1.0 - est * est, 0.0) / ns)
    X = np.log10(ns)
    up, dn = np.clip(est + err, *E.conv_ylim), np.clip(est - err, *E.conv_ylim)
    pts = np.vstack([box.curve(X, up), box.curve(X, dn)[::-1]])
    P.poly(pts, fc=mu_c, alpha=LY.conv_band_alpha * alpha, z=3)
    P.line(box.curve(X, np.clip(est, *E.conv_ylim)), mu_c, alpha, LY.conv_line_width, z=5)
    xl, yl = box.px(E.conv_xlog[0], model)
    xr, _ = box.px(E.conv_xlog[1], model)
    P.line([(xl, yl), (xr, yl)], gold, LY.conv_ref_alpha * alpha, 1.5, ls=(0, (5, 3)), z=4)
    P.text(xr - 4 * sc, yl - 8 * sc, tx["conv_model"].replace("@v@", num(model, ".3f", lang, math_mode=False)), F.label_small, gold, alpha, ha="right", va="bottom")
    P.text(box.x0 + 0.5 * (box.x1 - box.x0), box.y0 - 52 * sc, tx["species_mu"], F.label, white, alpha, ha="center")


def draw_part4(P: Painter, t: float, tx: dict, lang: str) -> str:
    E, LY, F, S = CFG.events, CFG.events.layout, CFG.fonts, CFG.style
    sc = P.sc
    gold, white, dim, b_c, mu_c = rgb("gold"), rgb("white"), rgb("dim"), rgb("b_c"), rgb("mu_c")
    ev = event_data()
    draw_formula(P, tx, "f4_dn", LY.formula_pos[0], smooth(t, *E.formula_in[0]), CFG.fonts.formula_small)
    draw_formula(P, tx, "f4_afb", LY.formula_pos[1], smooth(t, *E.formula_in[1]), CFG.fonts.formula_small)
    mid = 0.5 * (E.switch[0] + E.switch[1])
    a_b = (1.0 - smooth(t, E.switch[0], mid)) * smooth(t, *E.display_in)
    a_m = smooth(t, mid, E.switch[1])
    nb_ = n_b_at(t)
    nm_ = n_mu_at(t)
    bins = E.hist_bins
    if a_b > 0:
        counts = np.bincount(ev["ib"][:nb_], minlength=bins)
        draw_event_display(P, t, tx, lang, a_b, nb_, ev)
        draw_hist(P, t, tx, lang, counts, max(nb_, 0), a_fb(E.fermion_b), b_c, a_b)
        n_, color, model, a_ro = nb_, b_c, a_fb(E.fermion_b), a_b
        c_all = ev["cb"][:nb_]
    if a_m > 0:
        counts = np.bincount(ev["im"][:nm_], minlength=bins)
        draw_convergence(P, t, tx, lang, nm_, ev, a_m)
        draw_hist(P, t, tx, lang, counts, nm_, a_fb(E.fermion_mu), mu_c, a_m)
    # the readouts: N and A_FB with the statistical error of the species that is on the screen
    for n, c_all, model, col, a_ro in ((nb_, ev["cb"], a_fb(E.fermion_b), b_c, a_b), (nm_, ev["cm"], a_fb(E.fermion_mu), mu_c, a_m)):
        if a_ro <= 0 or n < 1:
            continue
        a_, err, nf, nbk = estimate(c_all[:n])
        P.text(P.X(LY.readout_n[0]), P.Y(LY.readout_n[1]), tx["n_fmt"].replace("@v@", thousands(n, lang)), CFG.fonts.readout + 2, white, a_ro, ha="left")
        P.text(P.X(LY.readout_a[0]), P.Y(LY.readout_a[1]), tx["a_fmt"].replace("@a@", signed(a_, f".{E.a_decimals}f", lang)).replace("@e@", num(err, f".{E.sigma_decimals}f", lang)),
               CFG.fonts.readout + 2, col, a_ro, ha="left")
        P.text(P.X(LY.readout_model[0]), P.Y(LY.readout_model[1]), tx["model_fmt"].replace("@v@", num(model, f".{E.a_decimals}f", lang, math_mode=False)), CFG.fonts.readout - 1, dim, a_ro, ha="right")
    k_cap = max(i for i, a in enumerate(E.caption_at) if t >= a)
    return ("c4a", "c4b", "c4c", "c4d", "c4e")[k_cap]


# ----------------------------------------------------------------------------------- part 5: away from the pole

def sqrt_s_at(t: float) -> float:
    """The sqrt(s) of the sweep (GeV) at the part time t: a monotone smooth path through the points of the config."""
    from scipy.interpolate import PchipInterpolator
    if "path" not in _cache:
        _cache["path"] = PchipInterpolator(CFG.scan.path_t, CFG.scan.path_s)
    ts = CFG.scan.path_t
    return float(_cache["path"](min(max(t, ts[0]), ts[-1])))


def scan_curve() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """sqrt(s), A_FB and sigma / sigma_max over the range of the plot (the full gamma/Z amplitude of the book)."""
    if "scan" not in _cache:
        SC = CFG.scan
        xs = np.linspace(SC.x_range[0], SC.x_range[1], SC.curve_samples)
        a = np.array([a_fb_scan(x, SC.fermion) for x in xs])
        sg = np.array([sigma_total(x, SC.fermion) for x in xs])
        _cache["scan"] = (xs, a, sg / sg.max())
    return _cache["scan"]


def draw_part5(P: Painter, t: float, tx: dict, lang: str) -> str:
    SC, LY, F = CFG.scan, CFG.scan.layout, CFG.fonts
    sc = P.sc
    gold, white, dim, mu_c = rgb("gold"), rgb("white"), rgb("dim"), rgb("mu_c")
    draw_formula(P, tx, "f5_rho", LY.formula_pos[0], smooth(t, *SC.formula_in[0]), F.formula_small)
    draw_formula(P, tx, "f5_dsig", LY.formula_pos[1], smooth(t, *SC.formula_in[1]), F.formula_small)
    pa = smooth(t, *SC.plots_in)
    sq = sqrt_s_at(t)
    a_now = a_fb_scan(sq, SC.fermion)
    # ------------------------------------------------------------------ the distribution at the current energy
    ctr = np.array(P.pt(LY.lobe_centre))
    unit = LY.lobe_unit * sc
    P.line([ctr - np.array([LY.axis_half * sc, 0.0]), ctr + np.array([LY.axis_half * sc, 0.0])], white, LY.axis_alpha * pa, 1.0, ls=(0, (4, 4)), z=2)
    P.text(ctr[0] + LY.axis_half * sc, ctr[1] - 14 * sc, tx["forward_short"], F.label_small, dim, pa, ha="right", va="bottom")
    P.text(ctr[0] - LY.axis_half * sc, ctr[1] - 14 * sc, tx["backward_short"], F.label_small, dim, pa, ha="left", va="bottom")
    P.text(ctr[0], P.Y(LY.lobe_label_y), tx["lobe_label"], F.label, white, pa, ha="center")
    draw_lobe(P, ctr, unit, lambda cs: shape(cs, a_now), math.pi, mu_c, LY.lobe_fill_alpha * pa, pa, LY.lobe_line_width, LY.lobe_points)
    sym = lobe_arc(ctr, unit, lambda cs: 1.0 + cs * cs, math.pi, LY.lobe_points)
    P.line(sym, white, LY.sym_alpha * pa, LY.sym_width, ls=(0, (5, 4)), z=5)
    P.circle(ctr, 3.2 * sc, white, 0.9 * pa, z=6)
    # ------------------------------------------------------------------ A_FB(sqrt s)
    box = Box(P, LY.plot_box, tuple(SC.x_range), tuple(SC.y_range))
    draw_box_axes(P, box, SC.x_ticks, SC.y_ticks, lambda v: num(v, ".0f", lang, math_mode=False),
                  lambda v: ("\u2212" if v < -1e-9 else "") + num(abs(v), ".1f" if abs(v) > 1e-9 else ".0f", lang, math_mode=False), pa, 5.0, 0.55, 1.0, 0.10, y_zero_line=True)
    P.text(box.x1, box.y1 + 38 * sc, tx["sqrt_s_axis"], F.label_small, dim, pa, ha="right")
    P.text(box.x0, box.y0 - 14 * sc, tx["afb_axis"], F.label_small, dim, pa, ha="left", va="bottom")
    xs, a, sg = scan_curve()
    # the cross section of the Z resonance in the background
    ys = SC.y_range[0] + sg * SC.sigma_fraction * (SC.y_range[1] - SC.y_range[0])
    cur = box.curve(xs, ys)
    P.poly(np.vstack([[cur[0, 0], box.y1], cur, [cur[-1, 0], box.y1]]), fc=white, alpha=LY.sigma_alpha * pa, z=1)
    P.line(cur, white, LY.sigma_line_alpha * pa, 1.0, z=2)
    xm, _ = box.px(MZ, 0.0)
    P.line([(xm, box.y0), (xm, box.y1)], gold, LY.pole_line_alpha * pa, 1.2, ls=(0, (3, 3)), z=2)
    P.text(xm, box.y1 + 22 * sc, tx["mz_label"], F.label_small, gold, pa, ha="center")
    P.text(xm + 10 * sc, box.y1 - 16 * sc, tx["sigma_label"], F.label_small, dim, pa, ha="left", va="bottom")
    done = xs <= sq
    if done.sum() > 1:
        P.line(box.curve(xs[done], a[done]), mu_c, pa, LY.curve_width, z=5)
    X, Y = box.px(sq, a_now)
    P.glow((X, Y), LY.dot_radius * sc * 2.2, gold, pa, z=5)
    P.circle((X, Y), LY.dot_radius * sc, gold, pa, z=6)
    # the pole value
    pl = smooth(t, *SC.pole_label_in) * (1.0 - smooth(t, *SC.pole_label_out))
    if pl > 0:
        a_pole = a_fb(SC.fermion)
        Xp, Yp = box.px(MZ, a_pole)
        P.text(xm + 14 * sc, Yp - 24 * sc, tx["pole_value"].replace("@v@", num(a_pole, f".{SC.decimals_a}f", lang)), F.label, gold, pl, ha="left", va="bottom")
    # the readouts
    P.text(P.X(LY.readout_s[0]), P.Y(LY.readout_s[1]), tx["s_fmt"].replace("@v@", num(sq, f".{SC.decimals_s}f", lang)), F.readout + 2, white, pa)
    P.text(P.X(LY.readout_a[0]), P.Y(LY.readout_a[1]), tx["a5_fmt"].replace("@v@", signed(a_now, f".{SC.decimals_a}f", lang)), F.readout + 2, mu_c, pa)
    k_cap = max(i for i, c in enumerate(SC.caption_at) if t >= c)
    return ("c5a", "c5b", "c5c", "c5d")[k_cap]


# ----------------------------------------------------------------------------------- the film

def draw_frame(P: Painter, tf: float, lang: str) -> None:
    """Everything that is drawn at the nominal film time tf (the cards included)."""
    tx = TEXT[lang]
    P.begin()
    t, card, prog = timeline(tf)
    if card is not None:
        draw_card(P, tx, card, prog)
        return
    part = part_of(t)
    cap = None
    if part == 1:
        cap = draw_part1(P, t - PARTS[0][0], tx, lang)
    elif part == 2:
        cap = draw_part2(P, t - PARTS[1][0], tx, lang)
    elif part == 3:
        cap = draw_part3(P, t - PARTS[2][0], tx, lang)
    elif part == 4:
        cap = draw_part4(P, t - PARTS[3][0], tx, lang)
    elif part == 5:
        cap = draw_part5(P, t - PARTS[4][0], tx, lang)
    draw_title(P, tx, part)
    draw_caption(P, tx, cap)


def render(out: Path, size: tuple[int, int], fps: int, total: float, lang: str, snap: float | None = None) -> None:
    if snap is None and not shutil.which("ffmpeg"):
        sys.exit("ffmpeg is required")
    V = CFG.video
    W, H = size
    P = Painter(W, H)
    k = total / TOTAL
    frames = int(round(total * fps))
    ids = [int(round(snap * fps))] if snap is not None else range(frames)
    writer = None
    if snap is None:
        writer = subprocess.Popen(
            ["ffmpeg", "-y", "-v", "error", "-f", "rawvideo", "-pix_fmt", "rgba", "-s", f"{W}x{H}",
             "-r", str(fps), "-i", "-", "-c:v", "libx264", "-preset", V.preset, "-crf", str(V.crf),
             "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(out)], stdin=subprocess.PIPE)
    for k_ in ids:
        t_film = k_ / fps
        draw_frame(P, t_film / k, lang)
        frame = P.finish()
        fade_io = min(smooth(t_film, 0.0, V.fade_s), 1.0 - smooth(t_film, total - V.fade_s, total))
        if fade_io < 1.0:
            frame = P.bg + (frame - P.bg) * fade_io
        frame = frame.clip(0, 255).astype(np.uint8)
        if writer is None:
            from PIL import Image
            Image.fromarray(frame).save(out)
        else:
            writer.stdin.write(frame.tobytes())
    if writer is not None:
        writer.stdin.close()
        writer.wait()
    P.plt.close(P.fig)
    print(f"wrote {out}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--lang", choices=("en", "ru"), default="en")
    ap.add_argument("--preview", action="store_true")
    ap.add_argument("--snapshot", type=float, default=None, help="film time of one PNG frame (cards included)")
    ap.add_argument("--content", action="store_true", help="with --snapshot: the time is a content time (the cards are not counted)")
    ap.add_argument("--seconds", type=float, default=TOTAL)
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--config", type=Path, default=None, help="another configuration file instead of config.toml")
    ap.add_argument("--set", action="append", default=[], metavar="SECTION.KEY=VALUE", help="override one configuration value")
    args = ap.parse_args()
    V = CFG.video
    out = args.out or HERE / "media" / f"z_decay_asymmetry_{args.lang}.mp4"
    out.parent.mkdir(parents=True, exist_ok=True)
    if args.snapshot is not None:
        snap = film_time(args.snapshot) if args.content else args.snapshot
        render(out.with_suffix(".png"), (V.width, V.height), V.fps, args.seconds, args.lang, snap=snap)
    elif args.preview:
        render(out.with_name(out.stem + "_preview.mp4"), (V.preview_width, V.preview_height), V.preview_fps, args.seconds, args.lang)
    else:
        render(out, (V.width, V.height), V.fps, args.seconds, args.lang)


if __name__ == "__main__":
    main()
