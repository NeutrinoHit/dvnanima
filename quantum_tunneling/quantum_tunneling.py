r"""Tunnelling: a point body and an extended body (the book, the Klein-Fock-Gordon chapter, the paragraph on quantum tunnelling).

Parts of the film (the physics is in tunnel_physics.py; every number is in config.toml, every word in texts.toml):
 1. A point body and a train.  A track y = h(x) (a Gaussian hill), cars joined rigidly along the track, one degree of freedom (the arc length S of the
    head car), L = (1/2) N m Sdot^2 - U(S), U(S) = sum_i m g h(x(S - i l)).  A single car with E = V/2 turns back; the train of N cars at the same speed
    passes, because its barrier is max U(S) < N V.  The scan over N: the threshold per car falls from V (N = 1) to the 'hill area per train length'.
 2. A quantum packet on the same hill: i hbar psi_t = [-(hbar^2/2m) psi_xx + V(x)] psi (split-step Fourier), the exact transmission T(E) of the hill,
    the packet transmits  P_T = int |phi(E)|^2 T(E) dE, a heavier particle with the same E/V; what the train analogy explains and what it does not.
 3. A macroscopic 'train': the junction phase delta of a Josephson junction in the tilted washboard U = -E_J cos(delta) - (hbar I / 2e) delta;
    the bias current is the knob that lowers the barrier: controlled tunnelling (illustrative parameters, T = 0, no damping).

Usage:
    python quantum_tunneling.py --lang en             # film -> media/quantum_tunneling_en.mp4
    python quantum_tunneling.py --lang ru
    python quantum_tunneling.py --lang en --snapshot 20
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dvconfig import load_config, load_texts  # noqa: E402

import tunnel_physics as tp  # noqa: E402

HERE = Path(__file__).resolve().parent
CFG = load_config(HERE)

TL = CFG.timeline
CARD_S = TL.card_s
CARD_AT = list(TL.card_at)
CARD_KEYS = [f"h{i}" for i in range(len(CARD_AT))]
PARTS = {k: tuple(getattr(TL, k)) for k in ("a1", "a2", "a3", "w1", "w2", "w3", "m1", "m2", "outro")}
# the parts of the first version (a packet, the Josephson circuit): the code is kept below, the parts are not in the film
LEGACY = {k: tuple(getattr(CFG.timeline_unused, k)) for k in ("b1", "b2", "b3", "lim", "c1", "c2", "outro_josephson")}
CONTENT_TOTAL = PARTS["outro"][1]
TOTAL = CONTENT_TOTAL + CARD_S * len(CARD_AT)


def timeline(tf: float) -> tuple[float, int | None, float]:
    """(content time, index of the card or None, progress of the card in [0, 1]) at the film time tf."""
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


def part_of(t: float) -> str:
    for k, (a, b) in PARTS.items():
        if a <= t < b:
            return k
    return "outro"


def smooth(x: float, a: float, b: float) -> float:
    u = min(max((x - a) / (b - a), 0.0), 1.0)
    return u * u * (3 - 2 * u)


def num(x: float, fmt: str, lang: str) -> str:
    s = format(x, fmt)
    return s.replace(".", "{,}") if lang == "ru" else s


def fill(s: str, /, **kw) -> str:
    """Replace the tokens @name@ of a text (braces are left alone: they belong to the mathtext)."""
    for k, v in kw.items():
        s = s.replace(f"@{k}@", str(v))
    return s


def numt(x: float, fmt: str, lang: str) -> str:
    """A number outside the formulas: the decimal comma in Russian."""
    s = format(x, fmt)
    return s.replace(".", ",") if lang == "ru" else s


TEXT = {lang: load_texts(HERE, lang) for lang in ("en", "ru")}

# ======================================================================================================================= models (lazy, cached)

_cache: dict = {}
CL = CFG.classical
QU = CFG.quantum


def track() -> tp.Track:
    if "track" not in _cache:
        _cache["track"] = tp.Track(CL.height, CL.sigma, CL.track_x_lim, CL.track_step)
    return _cache["track"]


def train(n: int) -> tp.Train:
    if ("train", n) not in _cache:
        _cache[("train", n)] = tp.Train(track(), n, CL.spacing, CL.mass, CL.g)
    return _cache[("train", n)]


def arc0() -> float:
    return float(track().arc(0.0))


def v0_flat() -> float:
    return math.sqrt(2.0 * CL.energy_fraction * CL.g * CL.height)


def v_unit() -> float:
    """V = m g H per car."""
    return CL.mass * CL.g * CL.height


def classical_run(n: int):
    """(t, S relative to the top of the hill, Sdot) of the launch with the speed v0 on the flat ground."""
    key = ("run", n)
    if key not in _cache:
        tr = train(n)
        s0 = float(track().arc(CL.start_head_x))
        t_end = CL.single_t_end if n == 1 else CL.train_t_end
        t, s, v, e = tr.simulate(s0, v0_flat(), t_end, CL.dt, CL.stride)
        _cache[key] = (t, s - arc0(), v, e)
    return _cache[key]


def u_curve(n: int, s_rel: np.ndarray) -> np.ndarray:
    """U_train(S) / (N V) on the grid of the arc positions (relative to the top)."""
    return train(n).potential(s_rel + arc0()) / (n * v_unit())


def threshold_ratio(n: int) -> float:
    """max_S U(S) / (N V)."""
    tr = train(n)
    s = np.linspace(tr.s_start(CL.margin_sigma), tr.s_end(CL.margin_sigma), CFG.scan.analytic_samples)
    return float(tr.potential(s).max() / (n * v_unit()))


def n_cross() -> int:
    """The smallest number of cars whose threshold per car is below E / V."""
    d = scan_data()
    return int(d["n"][np.argmax(d["eta"] < CL.energy_fraction)])


def scan_data() -> dict:
    """The analytic threshold per car for n = 1..analytic_n_max and the numerically found one for the counts of the film (cached on disk)."""
    if "scan" in _cache:
        return _cache["scan"]
    SC = CFG.scan
    ns = np.arange(1, SC.analytic_n_max + 1)
    ana = np.array([threshold_ratio(int(n)) for n in ns])
    sig = json.dumps([CL.to_dict(), SC.to_dict()], sort_keys=True)
    path = HERE / ".cache" / f"scan_{hashlib.md5(sig.encode()).hexdigest()[:12]}.json"
    if path.exists():
        num_eta = json.loads(path.read_text())
    else:
        num_eta = []
        for n in SC.counts:
            tr = train(n)
            vc = tr.critical_speed(CL.margin_sigma)
            vb = tp.critical_speed_by_bisection(tr, tr.s_start(CL.margin_sigma), tr.s_end(CL.margin_sigma), (1 - SC.bracket) * vc, (1 + SC.bracket) * vc,
                                                SC.bisect_dt, SC.bisect_t_max, SC.bisect_rel_tol)
            num_eta.append(0.5 * vb ** 2 / (CL.g * CL.height))              # E / V per car at the numerical threshold speed
        path.parent.mkdir(exist_ok=True)
        path.write_text(json.dumps(num_eta))
    _cache["scan"] = {"n": ns, "eta": ana, "counts": list(SC.counts), "eta_num": np.array(num_eta)}
    return _cache["scan"]


# ======================================================================================================================= drawing helpers

class Frame:
    """One frame: the figure, the scale of the fonts and the helpers that place text and axes."""

    def __init__(self, fig, size: tuple[int, int], lang: str):
        self.fig, self.W, self.H, self.lang = fig, size[0], size[1], lang
        self.sc = size[1] / CFG.video.reference_height
        self.tx = TEXT[lang]
        S = CFG.style
        self.TXT, self.DIM = tuple(S.text), tuple(S.dim)
        self.BLUE, self.WARM, self.GOLD, self.GREEN = tuple(S.blue), tuple(S.warm), tuple(S.gold), tuple(S.green)
        self._r = fig.canvas.get_renderer()

    def text(self, x: float, y: float, s: str, size: float, color, ha: str = "left", va: str = "baseline", max_w: float | None = None, **kw):
        """Figure text; if max_w (fraction of the width) is given the font shrinks (down to min_scale) until the text fits."""
        t = self.fig.text(x, y, s, color=color, fontsize=size * self.sc, ha=ha, va=va, **kw)
        if max_w is not None:
            w = t.get_window_extent(self._r).width / self.W
            if w > max_w:
                t.set_fontsize(size * self.sc * max(max_w / w, CFG.fonts.min_scale))
        return t

    def paragraph(self, x: float, y: float, s: str, size: float, color, max_w: float, line: float, va: str = "top") -> int:
        """Text wrapped to the width max_w (a fraction of the frame); words inside $...$ stay together. Returns the number of lines."""
        words, cur, inside = [], "", False
        for ch in s:
            if ch == "$":
                inside = not inside
            if ch == " " and not inside:
                words.append(cur)
                cur = ""
            else:
                cur += ch
        words.append(cur)
        lines, row = [], ""
        for w in words:
            trial = (row + " " + w).strip()
            t = self.fig.text(0.0, 0.0, trial, fontsize=size * self.sc)
            width = t.get_window_extent(self._r).width / self.W
            t.remove()
            if width > max_w and row:
                lines.append(row)
                row = w
            else:
                row = trial
        lines.append(row)
        for i, ln in enumerate(lines):
            self.fig.text(x, y - i * line, ln, color=color, fontsize=size * self.sc, ha="left", va=va)
        return len(lines)

    def axes(self, rect, **kw):
        ax = self.fig.add_axes(rect, facecolor="none", **kw)
        return ax

    def style_axes(self, ax, ticks: bool = True, left: bool = True, bottom: bool = True) -> None:
        S = CFG.style
        for k, sp in ax.spines.items():
            sp.set_color(tuple(S.axes_edge))
            sp.set_linewidth(S.axes_edge_width * self.sc)
            sp.set_visible((k == "left" and left) or (k == "bottom" and bottom))
        ax.tick_params(colors=self.DIM, labelsize=CFG.fonts.tick * self.sc, length=3 * self.sc, width=0.8 * self.sc)
        if not ticks:
            ax.set_xticks([])
            ax.set_yticks([])

    def label(self, ax, xy, s, size, color, ha="left", va="bottom", **kw):
        return ax.text(xy[0], xy[1], s, color=color, fontsize=size * self.sc, ha=ha, va=va, transform=ax.transAxes, **kw)


def draw_header(f: Frame, t: float, card: int | None, cprog: float) -> None:
    LY, F, S = CFG.layout, CFG.fonts, CFG.style
    tx = f.tx
    if card is None:
        f.text(*LY.title_pos, tx["title"], F.title, tuple(S.title_color), max_w=0.66)
        ch = chapter_of(t)
        if ch >= 1:
            f.text(*LY.chapter_label_pos, f"{ch}/{len(CARD_AT) - 1}   " + tx[CARD_KEYS[ch]], F.chapter_label, (*f.DIM, S.chapter_label_alpha), ha="right", max_w=0.30)
    else:
        a_c = min(smooth(cprog, *TL.card_fade_in), 1.0 - smooth(cprog, *TL.card_fade_out))
        if card == 0:
            f.text(0.5, LY.card_title_y, tx["h0"], F.card_title, (*S.card_title_color, a_c), ha="center", va="center", max_w=0.92)
            f.text(0.5, LY.card_subtitle_y, tx["h0s"], F.card_subtitle, (*f.DIM, a_c), ha="center", va="center", max_w=0.9)
        else:
            f.text(0.5, LY.card_number_y, f"{card}", F.card_number, (*f.GOLD, S.card_number_alpha * a_c), ha="center", va="center")
            f.text(0.5, LY.card_chapter_y, tx[CARD_KEYS[card]], F.card_chapter, (*S.card_title_color, a_c), ha="center", va="center", max_w=0.92)
            f.text(0.5, LY.card_chapter_sub_y, tx[CARD_KEYS[card] + "s"], F.card_chapter_sub, (*f.DIM, a_c), ha="center", va="center", max_w=0.9)


def caption(f: Frame, key: str, key2: str | None = None, **fmt) -> None:
    LY, F, S = CFG.layout, CFG.fonts, CFG.style
    f.text(*LY.caption_pos, fill(f.tx[key], **fmt), F.caption, (*f.TXT, S.caption_alpha), max_w=LY.caption_max_width)
    if key2:
        f.text(*LY.sub_caption_pos, fill(f.tx[key2], **fmt), F.sub_caption, (*f.DIM, S.caption_alpha), max_w=LY.caption_max_width)


def params(lang: str) -> dict:
    """The numbers of the film that appear inside the texts (decimal comma in Russian)."""
    n = lambda x, fm: num(x, fm, lang)
    return {"m": n(CL.mass, ".0f"), "g": n(CL.g, ".2f"), "H": n(CL.height, ".0f"), "l": n(CL.spacing, ".0f"), "v0": n(v0_flat(), ".1f"),
            "eta": n(CL.energy_fraction, ".1f"), "N": f"{CL.cars}", "sx": n(QU.packet_sigma, ".1f"), "w": n(QU.hill_width, ".1f"),
            "mu": n(QU.heavy_factor, ".0f"), "lam_inv": n(1.0 / CFG.josephson.lam, ".0f")}


def formulas(f: Frame, keys: list[tuple[str, float]], size: float | None = None) -> None:
    LY, F, S = CFG.layout, CFG.fonts, CFG.style
    P = params(f.lang)
    for row, (k, a) in enumerate(keys):
        if a > 0.01:
            f.text(LY.formula_pos[0], LY.formula_pos[1] - LY.formula_row * row, fill(f.tx[k], **P), (size or F.formula) if row == 0 else F.formula_small, (*f.TXT, S.formula_alpha * a), max_w=0.94)


# ---------------------------------------------------------------------------------------------------------------- models of part 2

def q_mass(kind: str) -> float:
    return QU.mass * (QU.heavy_factor if kind == "heavy" else 1.0)


def q_mean_energy() -> float:
    return QU.energy_fraction * QU.hill_height


def q_sigma_k() -> float:
    return 1.0 / (2.0 * QU.packet_sigma)


def q_k0(kind: str) -> float:
    """The carrier wave number such that the MEAN energy of the packet, hbar^2 (k0^2 + sigma_k^2) / 2m, equals E = V/2 exactly."""
    return math.sqrt(2.0 * q_mass(kind) * q_mean_energy() / QU.hbar ** 2 - q_sigma_k() ** 2)


def q_potential(x):
    return tp.gauss_hill(x, QU.hill_height, QU.hill_width)


def q_sim(kind: str) -> tp.SplitStep1D:
    sim = tp.SplitStep1D(QU.grid_n, QU.grid_length, q_mass(kind), QU.hbar, QU.dt, lambda t, v=None: v, QU.absorber_width, QU.absorber_strength)
    v = q_potential(sim.x)
    sim.potential = lambda t, v=v: v
    sim.set_state(tp.gaussian_packet(sim.x, QU.packet_x0, q_k0(kind), QU.packet_sigma))
    sim.vpot = v
    return sim


def quantum_run(kind: str) -> dict:
    """The packet evolution sampled at samples_per_s states per film second: density in the view, <T>, <V>, probabilities left / right of the hill."""
    key = ("qrun", kind)
    if key in _cache:
        return _cache[key]
    sim = q_sim(kind)
    rate = QU.light_rate if kind == "light" else QU.heavy_rate
    t_end = QU.light_t_end if kind == "light" else QU.heavy_t_end
    n_s = int(round(t_end / rate * QU.samples_per_s)) + 1
    xv = (sim.x >= QU.view_x[0]) & (sim.x <= QU.view_x[1])
    idx = np.where(xv)[0][:: QU.x_store_stride]
    dens = np.zeros((n_s, idx.size))
    kin, pot, pl, pr, nrm = (np.zeros(n_s) for _ in range(5))
    xb = QU.side_x_factor * QU.hill_width
    for i in range(n_s):
        sim.run(i * rate / QU.samples_per_s)
        d = sim.density()
        dens[i] = d[idx]
        kin[i], pot[i] = sim.mean_kinetic(), sim.mean_potential(sim.vpot)
        pl[i] = d[sim.x < -xb].sum() * sim.dx
        pr[i] = d[sim.x > xb].sum() * sim.dx
        nrm[i] = sim.norm()
    out = {"x": sim.x[idx], "dens": dens, "kin": kin, "pot": pot, "pl": pl, "pr": pr, "norm": nrm, "n": n_s, "dx": sim.dx}
    _cache[key] = out
    return out


def te_curve(mass: float) -> tuple[np.ndarray, np.ndarray]:
    """Exact T(E) of the hill for the given mass, E / V from curve_e_min to te_e_max."""
    key = ("te", mass)
    if key not in _cache:
        e = np.linspace(QU.curve_e_min, QU.te_e_max, QU.curve_points)
        v = QU.hill_height
        t = np.array([tp.transmission(float(en * v), q_potential, mass, QU.hbar, QU.ode_half_width) for en in e])
        _cache[key] = (e, t)
    return _cache[key]


def packet_numbers(kind: str) -> dict:
    """The transmission of the packet from the exact T(E): P_T and its under- / over-barrier parts, and the weight above the top."""
    key = ("pn", kind)
    if key not in _cache:
        sim = q_sim(kind)
        psi0 = sim.psi
        pt, under, over, w_over, w_neg = tp.packet_transmission(psi0, sim.dx, q_mass(kind), QU.hbar, q_potential, QU.ode_half_width, QU.hill_height, QU.energy_samples)
        _cache[key] = {"pt": pt, "under": under, "over": over, "w_over": w_over, "w_neg": w_neg}
    return _cache[key]


def packet_distribution(kind: str, e_grid: np.ndarray) -> np.ndarray:
    """|phi(E)|^2 of the packet on the grid of E (units of V), normalised to the peak value 1: Gaussian in k with sigma_k."""
    m, hb, k0, sk = q_mass(kind), QU.hbar, q_k0(kind), q_sigma_k()
    k = np.sqrt(2.0 * m * e_grid * QU.hill_height) / hb
    w = np.exp(-((k - k0) ** 2) / (2.0 * sk ** 2)) * (m / (hb ** 2 * k))            # |phi(k)|^2 dk/dE
    return w / w.max()


def stationary_state(xs: np.ndarray) -> np.ndarray:
    key = ("stat", xs.size)
    if key not in _cache:
        e = q_mean_energy()
        _cache[key] = np.abs(tp.stationary_state(e, q_potential, QU.mass, QU.hbar, QU.ode_half_width, xs)) ** 2
    return _cache[key]


# ---------------------------------------------------------------------------------------------------------------- part 1: the train

def hill_xy(n: int = 700):
    VW = CFG.view
    x = np.linspace(VW.scene_x[0], VW.scene_x[1], n)
    return x, track().h(x)


def scene_axes(f: Frame):
    """The axes of the train scene with equal scales in x and y (metres)."""
    VW, LY = CFG.view, CFG.layout
    rect = LY.scene_axes
    ax = f.axes(rect)
    ax.set_xlim(*VW.scene_x)
    wpx, hpx = rect[2] * f.W, rect[3] * f.H
    span_y = (VW.scene_x[1] - VW.scene_x[0]) * hpx / wpx
    ax.set_ylim(VW.scene_floor, VW.scene_floor + span_y)
    ax.axis("off")
    return ax


def draw_hill(f: Frame, ax) -> None:
    S, VW = CFG.style, CFG.view
    x, y = hill_xy()
    ax.fill_between(x, VW.scene_floor, y, color=(*S.hill_fill, S.hill_fill_alpha), lw=0, zorder=1)
    ax.plot(x, y, color=(*S.hill_edge, 1.0), lw=S.hill_edge_width * f.sc, zorder=2, solid_capstyle="round")


def draw_cars(f: Frame, ax, n: int, s_head_rel: float, color_head=True) -> None:
    from matplotlib.collections import LineCollection, PolyCollection
    S = CFG.style
    tr = train(n)
    xs, ys, th = tr.car_geometry(s_head_rel + arc0())
    L, Hc, clr = CL.car_length, CL.car_height, CL.car_clearance
    c, s_ = np.cos(th), np.sin(th)
    corners = np.array([[-0.5 * L, clr], [0.5 * L, clr], [0.5 * L, clr + Hc], [-0.5 * L, clr + Hc]])        # local (along, normal)
    # the normal of the track at angle th is (-sin th, cos th)
    px = xs[:, None] + corners[None, :, 0] * c[:, None] - corners[None, :, 1] * s_[:, None]
    py = ys[:, None] + corners[None, :, 0] * s_[:, None] + corners[None, :, 1] * c[:, None]
    polys = [np.column_stack([px[i], py[i]]) for i in range(n)]
    cols = [(*f.BLUE, S.car_alpha)] * n
    if color_head and n > 1:
        cols[0] = (*S.head_car_color, 1.0)
    if n > 1:
        yy = clr + CL.coupler_height * Hc
        # the coupler is a line through the car centres at the height of the couplers
        gx = xs - yy * s_
        gy = ys + yy * c
        ax.plot(gx, gy, color=(*S.coupler_color, 1.0), lw=S.coupler_width * f.sc, zorder=3, solid_capstyle="butt")
    ax.add_collection(PolyCollection(polys, facecolors=cols, edgecolors=(*S.car_edge, 1.0), linewidths=S.car_edge_width * f.sc, zorder=4))


def bars_panel(f: Frame, rect, ke: float, pe: float, thresh: float | None, n: int, alpha: float = 1.0) -> None:
    """Energy per car in units of V: kinetic + potential = E (gold), the threshold max U / N V (warm), N V (dashed)."""
    S, F, VW = CFG.style, CFG.fonts, CFG.view
    tx = f.tx
    ax = f.axes(rect)
    f.style_axes(ax, ticks=False, bottom=False)
    ax.set_xlim(0, 1.6)
    ax.set_ylim(*VW.bars_y)
    ax.bar([0.45], [max(ke, 0.0)], width=S.bar_width, bottom=0.0, color=(*f.BLUE, S.bar_alpha * alpha), lw=0)
    ax.bar([0.45], [max(pe, 0.0)], width=S.bar_width, bottom=max(ke, 0.0), color=(*S.hill_edge, 0.75 * alpha), lw=0)
    e = CL.energy_fraction
    ax.plot([0.0, 1.6], [e, e], color=(*f.GOLD, alpha), lw=S.energy_width * f.sc)
    ax.plot([0.0, 1.6], [1.0, 1.0], color=(*f.DIM, 0.7 * alpha), lw=S.level_width * f.sc, ls=(0, tuple(S.level_dash)))
    if thresh is not None:
        ax.plot([0.0, 1.6], [thresh, thresh], color=(*f.WARM, alpha), lw=S.thresh_width * f.sc)
    fs = F.legend * f.sc
    if n > 1:
        ax.text(0.78, 1.0, tx["b_nv"], color=(*f.DIM, alpha), fontsize=fs, va="bottom", ha="left")
    ax.text(0.78, e, tx["b_e"], color=(*f.GOLD, alpha), fontsize=fs, va="bottom" if thresh is None or thresh < e else "top", ha="left")
    if thresh is not None and abs(thresh - 1.0) > 0.03:
        ax.text(0.78, thresh, tx["b_thr"], color=(*f.WARM, alpha), fontsize=fs, va="top" if thresh < e else "bottom", ha="left")
    elif thresh is not None:
        ax.text(0.78, 1.0, tx["b_thr1"], color=(*f.WARM, alpha), fontsize=fs, va="top", ha="left")
    if ke > 0.08:
        ax.text(0.45, 0.5 * ke, tx["b_ke"], color=(0.02, 0.04, 0.08), fontsize=fs, va="center", ha="center")
    if pe > 0.08:
        ax.text(0.45, ke + 0.5 * pe, tx["b_pe"], color=(0.02, 0.04, 0.08), fontsize=fs, va="center", ha="center")
    f.label(ax, (0.0, 1.04), tx["b_title"], F.panel_title, (*f.DIM, alpha), ha="left", va="bottom")


def u_panel(f: Frame, rect, n: int, s_rel: float | None, ke: float | None, thresh_line: bool, ghost: bool, alpha: float = 1.0, turn: bool = False) -> None:
    """U(S) / N V against S with the energy line, the moving point and the kinetic energy as the gap to the line."""
    S, F, VW = CFG.style, CFG.fonts, CFG.view
    tx = f.tx
    ax = f.axes(rect)
    f.style_axes(ax)
    ax.set_xlim(*VW.u_s)
    ax.set_ylim(*VW.u_y)
    ax.set_yticks([0.0, 0.5, 1.0])
    ax.set_yticklabels(["0", "0,5" if f.lang == "ru" else "0.5", "1"])
    sg = np.linspace(VW.u_s[0], VW.u_s[1], 900)
    if ghost and n > 1:
        ax.fill_between(sg, 0.0, u_curve(1, sg), color=(*S.hill_fill, 0.35 * alpha), lw=0)
        ax.plot(sg, u_curve(1, sg), color=(*S.hill_edge, S.ghost_alpha * alpha), lw=1.2 * f.sc, ls=(0, tuple(S.level_dash)))
    u = u_curve(n, sg)
    col = S.hill_edge if n == 1 else f.WARM
    ax.fill_between(sg, 0.0, u, color=(*(S.hill_fill if n == 1 else f.WARM), (S.hill_fill_alpha if n == 1 else S.fill_alpha) * alpha), lw=0)
    ax.plot(sg, u, color=(*col, alpha), lw=S.curve_width * f.sc)
    e = CL.energy_fraction
    ax.plot(VW.u_s, [e, e], color=(*f.GOLD, alpha), lw=S.energy_width * f.sc, ls=(0, tuple(S.level_dash)))
    if thresh_line:
        th = threshold_ratio(n)
        ax.plot(VW.u_s, [th, th], color=(*f.WARM, 0.9 * alpha), lw=1.2 * f.sc, ls=(0, (1, 2)))
    if s_rel is not None:
        us = float(u_curve(n, np.array([s_rel]))[0])
        ax.plot([s_rel, s_rel], [us, us + (ke if ke is not None else 0.0)], color=(*f.BLUE, alpha), lw=3.0 * f.sc, solid_capstyle="butt")
        ax.scatter([s_rel], [us], s=S.dot_size * f.sc ** 2, color=(*f.TXT, alpha), zorder=5, edgecolors="none")
    f.label(ax, (0.0, 1.04), tx["u_title_1"] if n == 1 else tx["u_title_n"], F.panel_title, (*f.DIM, alpha), ha="left", va="bottom")
    f.label(ax, (1.0, -0.20), tx["u_xlabel"], F.axis_label, (*f.DIM, alpha), ha="right", va="top")


def classical_frame(f: Frame, t: float) -> None:
    S, F, VW = CFG.style, CFG.fonts, CFG.view
    lang, tx = f.lang, f.tx
    part = part_of(t)
    LY = CFG.layout
    a1, a2, a3 = PARTS["a1"], PARTS["a2"], PARTS["a3"]
    fa = smooth(t, 0.0, 1.2)
    formulas(f, [("fA1", fa), ("fA2", fa)])
    ax = scene_axes(f)
    draw_hill(f, ax)
    V = v_unit()
    if part in ("a1", "a2"):
        n = 1 if part == "a1" else CL.cars
        t0 = a1[0] if part == "a1" else a2[0]
        rate = CL.single_rate if part == "a1" else CL.train_rate
        tt, ss, vv, ee = classical_run(n)
        tau = min((t - t0) * rate, tt[-1])
        s_rel = float(np.interp(tau, tt, ss))
        v = float(np.interp(tau, tt, vv))
        draw_cars(f, ax, n, s_rel)
        ke = 0.5 * v * v / (CL.g * CL.height)                             # per car, in units of V
        pe = float(u_curve(n, np.array([s_rel]))[0])
        thresh = threshold_ratio(n)
        if n == 1:
            hlevel = CL.energy_fraction * CL.height
            ax.plot(VW.scene_x, [hlevel, hlevel], color=(*f.GOLD, 0.8), lw=S.level_width * f.sc, ls=(0, tuple(S.level_dash)), zorder=0)
            ax.text(VW.scene_x[1] - 0.4, hlevel + 0.12, tx["lvl_reach"], color=(*f.GOLD, 0.9), fontsize=F.legend * f.sc, ha="right", va="bottom")
            ax.text(0.0, CL.height + 0.45, tx["hill_label"], color=(*f.DIM, 0.9), fontsize=F.legend * f.sc, ha="center", va="bottom")
            turned = float(np.max(ss[: np.searchsorted(tt, tau) + 1]))
            tv = tt[int(np.argmax(vv < 0))] if np.any(vv < 0) else None
            if tv is not None and tau > tv:
                xtp = float(track().x_at(float(np.max(ss)) + arc0()))
                ax.plot([xtp, xtp], [0.0, track().h(xtp)], color=(*f.WARM, 0.9), lw=1.4 * f.sc, ls=(0, (2, 2)), zorder=1)
                ax.text(xtp, track().h(xtp) + 0.2, tx["turning_point"], color=(*f.WARM, 0.95), fontsize=F.legend * f.sc, ha="center", va="bottom")
                a = smooth(tau, tv + CL.stamp_delay_s, tv + CL.stamp_delay_s + 0.5)
                ax.text(VW.scene_x[0] + 1.2, CL.height + 1.0, tx["st_back"], color=(*f.WARM, a), fontsize=F.stamp * f.sc, ha="left", va="center", weight="bold")
        else:
            ax.text(VW.scene_x[0] + 0.8, CL.height + 0.6, fill(tx["train_label"], n=n), color=(*f.TXT, 0.9), fontsize=F.legend * f.sc, ha="left", va="center")
            tr = train(n)
            tail_ok = s_rel + arc0() - CL.spacing * (n - 1) > float(track().arc(CL.passed_margin_sigma * CL.sigma))
            if tail_ok:
                ax.text(VW.scene_x[1] - 1.0, CL.height + 1.0, tx["st_pass"], color=(*f.GREEN, smooth(tau, 0, 1)), fontsize=F.stamp * f.sc, ha="right", va="center", weight="bold")
        bars_panel(f, LY.bars_axes, ke, pe, thresh, n)
        u_panel(f, LY.u_axes, n, s_rel, ke, thresh_line=(n > 1), ghost=(n > 1))
        if part == "a1":
            caption(f, "cA1q" if t - t0 < 4.0 else "cA1a", "cA1s" if t - t0 >= 4.0 else None)
        else:
            caption(f, "cA2q" if t - t0 < 4.0 else "cA2a", "cA2s" if t - t0 >= 4.0 else None)
        return
    # part a3: the scan over the number of cars
    dat = scan_data()
    tau = t - a3[0]
    cnt = dat["counts"]
    k = 0 if tau < CFG.scan.first_dwell_s else min(int((tau - CFG.scan.first_dwell_s) / CFG.scan.dwell_s) + 1, len(cnt) - 1)
    n = cnt[k]
    tr = train(n)
    s_head = float(track().arc(-0.9 * CL.sigma * 3.0)) - arc0()           # the train waits in front of the hill
    draw_cars(f, ax, n, s_head)
    ax.text(VW.scene_x[0] + 0.8, CL.height + 0.6, fill(tx["train_label"], n=n), color=(*f.TXT, 0.9), fontsize=F.legend * f.sc, ha="left", va="center")
    scan_panel(f, LY.scan_axes, k, tau)
    u_panel(f, LY.u_axes, n, None, None, thresh_line=True, ghost=True)
    caption(f, "cA3q" if tau < 4.0 else "cA3a", "cA3s" if tau >= 4.0 else None, ncross=n_cross())


def scan_panel(f: Frame, rect, k: int, tau: float) -> None:
    S, F = CFG.style, CFG.fonts
    tx = f.tx
    dat = scan_data()
    ax = f.axes(rect)
    f.style_axes(ax)
    nmax = CFG.scan.analytic_n_max
    ax.set_xlim(0, nmax + 1)
    ax.set_ylim(0.0, 1.08)
    ax.set_yticks([0.0, 0.5, 1.0])
    ax.set_yticklabels(["0", "0,5" if f.lang == "ru" else "0.5", "1"])
    cnt = dat["counts"]
    n_now = cnt[k]
    ax.plot(dat["n"], dat["eta"], color=(*f.WARM, 0.35), lw=1.2 * f.sc)
    e = CL.energy_fraction
    ax.plot([0, nmax + 1], [e, e], color=(*f.GOLD, 1.0), lw=S.energy_width * f.sc, ls=(0, tuple(S.level_dash)))
    ax.plot([0, nmax + 1], [1.0, 1.0], color=(*f.DIM, 0.6), lw=S.level_width * f.sc, ls=(0, tuple(S.level_dash)))
    # the limit  (m g / l) int h sqrt(1 + h'^2) dx / (N m g H)  ~ 1/N
    asym = train(CL.cars).asymptotic_threshold() / v_unit()
    nn = np.linspace(max(1.0, asym), nmax, 200)
    ax.plot(nn, asym / nn, color=(*f.DIM, 0.8), lw=1.2 * f.sc, ls=(0, (1, 2)))
    ax.text(nmax * 0.98, asym / nmax + 0.045, tx["scan_asym"], color=(*f.DIM, 0.95), fontsize=F.legend * f.sc, ha="right", va="bottom")
    shown = k + 1
    ax.scatter(cnt[:shown], dat["eta_num"][:shown], s=S.dot_size * 0.55 * f.sc ** 2, color=(*f.WARM, 0.95), edgecolors="none", zorder=4)
    cur = k
    ax.scatter([cnt[cur]], [dat["eta"][cnt[cur] - 1]], s=S.dot_size * 1.9 * f.sc ** 2, facecolors="none", edgecolors=(*f.TXT, 1.0), linewidths=1.6 * f.sc, zorder=5)
    ax.text(0.5, 1.0 + 0.02, tx["scan_n1"], color=(*f.DIM, 0.95), fontsize=F.legend * f.sc, ha="left", va="bottom")
    ax.text(nmax + 0.5, e + 0.02, tx["scan_e"], color=(*f.GOLD, 1.0), fontsize=F.legend * f.sc, ha="right", va="bottom")
    f.label(ax, (0.0, 1.04), tx["scan_title"], F.panel_title, (*f.DIM, 1.0), ha="left", va="bottom")
    f.label(ax, (1.0, -0.20), tx["scan_xlabel"], F.axis_label, (*f.DIM, 1.0), ha="right", va="top")
    eta_now = dat["eta"][cnt[cur] - 1]
    f.label(ax, (0.98, 0.80), fill(tx["scan_read"], n=cnt[cur], thr=num(eta_now, ".2f", f.lang)), F.legend, (*f.TXT, 1.0), ha="right", va="top")
    f.label(ax, (0.98, 0.64), tx["scan_pass"] if eta_now < e else tx["scan_back"], F.legend, (*(f.GREEN if eta_now < e else f.WARM), 1.0), ha="right", va="top")


# ---------------------------------------------------------------------------------------------------------------- part 2: the packet

def q_main(f: Frame, kind: str, i: int, magnify: bool = False) -> None:
    S, F, LY = CFG.style, CFG.fonts, CFG.layout
    tx = f.tx
    run = quantum_run(kind)
    ax = f.axes(LY.b_main_axes)
    f.style_axes(ax)
    ax.set_xlim(*QU.view_x)
    ax.set_ylim(0.0, 1.5 * QU.hill_height)
    ax.set_yticks([0.0, QU.hill_height])
    ax.set_yticklabels(["0", "$V$"])
    ax.set_xticks([])
    xg = np.linspace(QU.view_x[0], QU.view_x[1], 800)
    vg = q_potential(xg)
    e = q_mean_energy()
    ax.fill_between(xg, 0.0, vg, color=(*S.hill_fill, S.hill_fill_alpha), lw=0, zorder=1)
    ax.fill_between(xg, e, vg, where=vg > e, color=(*f.WARM, QU.forbid_alpha), lw=0, zorder=1)
    ax.plot(xg, vg, color=(*S.hill_edge, 1.0), lw=S.hill_edge_width * f.sc, zorder=2)
    ax.plot(QU.view_x, [e, e], color=(*f.GOLD, 0.9), lw=S.level_width * f.sc, ls=(0, tuple(S.level_dash)), zorder=3)
    xtp = tp.turning_point_gauss(e, QU.hill_height, QU.hill_width)
    ax.plot([-xtp, xtp], [e, e], "o", color=f.WARM, ms=5 * f.sc, zorder=5)
    ax.text(0.0, QU.hill_height + 0.04, tx["q_forbid"], color=(*f.WARM, 0.95), fontsize=F.legend * f.sc, ha="center", va="bottom", zorder=6)
    ax.text(QU.view_x[1] - 0.3, e + 0.02, tx["q_den_e"], color=(*f.GOLD, 0.9), fontsize=F.legend * f.sc, ha="right", va="bottom")
    d = run["dens"][i]
    sc = QU.density_height / (1.0 / (math.sqrt(2.0 * math.pi) * QU.packet_sigma))
    y = e + sc * d
    ax.fill_between(run["x"], e, y, color=(*f.BLUE, QU.density_alpha), lw=0, zorder=4)
    ax.plot(run["x"], y, color=(*f.BLUE, 1.0), lw=1.8 * f.sc, zorder=5)
    if magnify:
        xb = QU.side_x_factor * QU.hill_width
        sel = run["x"] > xb
        ym = e + QU.magnify_heavy * sc * d[sel]
        ax.plot(run["x"][sel], ym, color=(*f.GREEN, 0.95), lw=1.5 * f.sc, ls=(0, (3, 2)), zorder=5)
        ax.text(QU.view_x[1] - 0.3, e + 0.5 * QU.hill_height * 0.6, fill(tx["q_mag"], mag=num(QU.magnify_heavy, ".0f", f.lang)), color=(*f.GREEN, 0.95), fontsize=F.legend * f.sc, ha="right", va="bottom")
    f.label(ax, (0.0, 1.04), tx["q_den"], F.panel_title, (*f.DIM, 1.0), ha="left", va="bottom")
    pl, pr = run["pl"][i], run["pr"][i]
    done = i > 0.75 * run["n"]
    f.label(ax, (0.03, 0.97), tx["q_refl"] if done else tx["q_left"], F.readout, (*f.BLUE, 1.0), ha="left", va="top")
    f.label(ax, (0.03, 0.88), f"{numt(100 * pl, '.1f', f.lang)} %", F.readout, (*f.TXT, 1.0), ha="left", va="top")
    f.label(ax, (0.97, 0.97), tx["q_trans"] if done else tx["q_right"], F.readout, (*f.BLUE, 1.0), ha="right", va="top")
    f.label(ax, (0.97, 0.88), f"{numt(100 * pr, '.1f', f.lang)} %" if kind == "light" else f"{numt(100 * pr, '.2f', f.lang)} %", F.readout, (*f.TXT, 1.0), ha="right", va="top")


def q_bars(f: Frame, kind: str, i: int) -> None:
    S, F, LY = CFG.style, CFG.fonts, CFG.layout
    tx = f.tx
    run = quantum_run(kind)
    ax = f.axes(LY.b_bars_axes)
    f.style_axes(ax, ticks=False, bottom=False)
    ax.set_xlim(0, 1.6)
    ax.set_ylim(*CFG.view.bars_y)
    kin, pot = run["kin"][i] / QU.hill_height, run["pot"][i] / QU.hill_height
    ax.bar([0.45], [kin], width=S.bar_width, color=(*f.BLUE, S.bar_alpha), lw=0)
    ax.bar([0.45], [pot], width=S.bar_width, bottom=kin, color=(*S.hill_edge, 0.75), lw=0)
    e = q_mean_energy() / QU.hill_height
    ax.plot([0.0, 1.6], [e, e], color=(*f.GOLD, 1.0), lw=S.energy_width * f.sc)
    ax.plot([0.0, 1.6], [1.0, 1.0], color=(*f.DIM, 0.7), lw=S.level_width * f.sc, ls=(0, tuple(S.level_dash)))
    fs = F.legend * f.sc
    ax.text(0.78, 1.0, tx["q_b_top"], color=(*f.DIM, 1.0), fontsize=fs, va="bottom", ha="left")
    ax.text(0.78, e, tx["q_b_e"], color=(*f.GOLD, 1.0), fontsize=fs, va="bottom", ha="left")
    ax.text(0.45, 0.5 * kin, tx["q_b_t"], color=(0.02, 0.04, 0.08), fontsize=fs, va="center", ha="center")
    if pot > 0.03:
        ax.text(0.78, kin + 0.5 * pot, tx["q_b_v"], color=(*S.hill_edge, 1.0), fontsize=fs, va="center", ha="left")
    f.label(ax, (0.0, 1.04), tx["q_b_title"], F.panel_title, (*f.DIM, 1.0), ha="left", va="bottom")


def te_panel(f: Frame, rect, kinds: list[str], dist_kind: str, i: int | None = None) -> None:
    S, F = CFG.style, CFG.fonts
    tx = f.tx
    ax = f.axes(rect)
    f.style_axes(ax)
    ax.set_xlim(0.0, QU.te_e_max)
    ax.set_ylim(0.0, 1.08)
    ax.set_yticks([0.0, 0.5, 1.0])
    ax.set_yticklabels(["0", "0,5" if f.lang == "ru" else "0.5", "1"])
    eg = np.linspace(QU.curve_e_min, QU.te_e_max, 300)
    pd = packet_distribution(dist_kind, eg)
    ax.fill_between(eg, 0.0, QU.dist_height * pd, color=(*f.GOLD, 0.22), lw=0)
    ax.plot([0.0, 1.0, 1.0, QU.te_e_max], [0.0, 0.0, 1.0, 1.0], color=(*f.DIM, 0.9), lw=1.4 * f.sc, ls=(0, tuple(S.level_dash)))
    for kind in kinds:
        e, t = te_curve(q_mass(kind))
        ax.plot(e, t, color=(*(f.BLUE if kind == "light" else f.WARM), 1.0), lw=S.curve_width * f.sc)
        ym = float(np.interp(0.0 + q_mean_energy() / QU.hill_height, e, t))
    e_m = q_mean_energy() / QU.hill_height
    ax.plot([e_m, e_m], [0.0, 1.05], color=(*f.GOLD, 0.9), lw=1.2 * f.sc)
    ax.text(e_m + 0.02, 1.0, tx["te_mean"], color=(*f.GOLD, 1.0), fontsize=F.legend * f.sc, ha="left", va="top")
    ax.text(QU.te_e_max - 0.03, 0.20, tx["te_classical"], color=(*f.DIM, 1.0), fontsize=F.legend * f.sc, ha="right", va="top")
    ax.text(0.02, QU.dist_height + 0.03, tx["te_dist"], color=(*f.GOLD, 0.9), fontsize=F.legend * f.sc, ha="left", va="bottom")
    if len(kinds) > 1:
        ax.text(QU.te_e_max - 0.03, 0.40, tx["te_light"], color=(*f.BLUE, 1.0), fontsize=F.legend * f.sc, ha="right", va="top")
        ax.text(QU.te_e_max - 0.03, 0.27, fill(tx["te_heavy"], mu=num(QU.heavy_factor, ".0f", f.lang)), color=(*f.WARM, 1.0), fontsize=F.legend * f.sc, ha="right", va="top")
    f.label(ax, (0.0, 1.04), tx["te_title"], F.panel_title, (*f.DIM, 1.0), ha="left", va="bottom")
    f.label(ax, (1.0, -0.20), tx["te_xlabel"], F.axis_label, (*f.DIM, 1.0), ha="right", va="top")


def te_big(f: Frame, tau: float) -> None:
    """Part 2b: the exact T(E), the energy distribution of the packet and the split of P_T into the under- and over-barrier parts."""
    S, F, LY = CFG.style, CFG.fonts, CFG.layout
    tx = f.tx
    ax = f.axes(LY.b_te_big_axes)
    f.style_axes(ax)
    ax.set_xlim(0.0, QU.te_e_max)
    ax.set_ylim(0.0, 1.08)
    ax.set_yticks([0.0, 0.5, 1.0])
    ax.set_yticklabels(["0", "0,5" if f.lang == "ru" else "0.5", "1"])
    eg = np.linspace(QU.curve_e_min, QU.te_e_max, 400)
    pd = packet_distribution("light", eg)
    a_d = smooth(tau, *QU.big_dist_in)
    a_p = smooth(tau, *QU.big_prod_in)
    ax.plot([0.0, 1.0, 1.0, QU.te_e_max], [0.0, 0.0, 1.0, 1.0], color=(*f.DIM, 0.9), lw=1.6 * f.sc, ls=(0, tuple(S.level_dash)))
    ax.text(QU.te_e_max - 0.03, 0.12, tx["te_classical"], color=(*f.DIM, 1.0), fontsize=F.legend * f.sc, ha="right", va="top")
    e, t = te_curve(QU.mass)
    reveal = smooth(tau, *QU.big_curve_in)
    sel = e <= QU.curve_e_min + reveal * (QU.te_e_max - QU.curve_e_min)
    ax.plot(e[sel], t[sel], color=(*f.BLUE, 1.0), lw=S.curve_width * 1.2 * f.sc)
    ax.text(1.25, 0.97, tx["te_exact"], color=(*f.BLUE, reveal), fontsize=F.legend * f.sc, ha="left", va="top")
    ax.fill_between(eg, 0.0, QU.dist_height * pd, color=(*f.GOLD, 0.20 * a_d), lw=0)
    ax.plot(eg, QU.dist_height * pd, color=(*f.GOLD, 0.8 * a_d), lw=1.2 * f.sc)
    ax.text(0.03, QU.dist_height + 0.03, tx["te_dist"], color=(*f.GOLD, a_d), fontsize=F.legend * f.sc, ha="left", va="bottom")
    tg = np.interp(eg, e, t)
    prod = QU.dist_height * pd * tg
    under = eg < 1.0
    ax.fill_between(eg[under], 0.0, prod[under], color=(*f.BLUE, 0.9 * a_p), lw=0)
    ax.fill_between(eg[~under], 0.0, prod[~under], color=(*f.WARM, 0.95 * a_p), lw=0)
    e_m = q_mean_energy() / QU.hill_height
    ax.plot([e_m, e_m], [0.0, 1.05], color=(*f.GOLD, 0.9), lw=1.2 * f.sc)
    ax.text(e_m + 0.02, 1.0, tx["te_mean"], color=(*f.GOLD, 1.0), fontsize=F.legend * f.sc, ha="left", va="top")
    f.label(ax, (0.0, 1.04), tx["te_big_title"], F.panel_title, (*f.DIM, 1.0), ha="left", va="bottom")
    f.label(ax, (1.0, -0.12), tx["te_xlabel"], F.axis_label, (*f.DIM, 1.0), ha="right", va="top")
    pn = packet_numbers("light")
    run = quantum_run("light")
    P = {"under": num(pn["under"], ".3f", f.lang), "over": num(pn["over"], ".3f", f.lang), "pt": num(pn["pt"], ".3f", f.lang),
         "ps": num(float(run["pr"][-1]), ".3f", f.lang), "wo": num(pn["w_over"], ".4f", f.lang)}
    y0 = LY.b_te_text_y
    f.text(LY.b_te_text_x, y0, fill(tx["te_split_head"], **P), F.lim_text, (*f.TXT, a_p))
    f.text(LY.b_te_text_x, y0 - LY.b_te_text_row, fill(tx["te_split_under"], **P), F.lim_text, (*f.BLUE, a_p))
    f.text(LY.b_te_text_x, y0 - 2.6 * LY.b_te_text_row, fill(tx["te_split_over"], **P), F.lim_text, (*f.WARM, a_p))
    f.text(LY.b_te_text_x, y0 - 4.3 * LY.b_te_text_row, fill(tx["te_sim"], **P), F.lim_text, (*f.TXT, smooth(tau, QU.big_prod_in[1], QU.big_prod_in[1] + 1.0)))


def quantum_frame(f: Frame, t: float) -> None:
    LY = CFG.layout
    part = part_of(t)
    if part == "b1":
        tau = t - LEGACY["b1"][0]
        run = quantum_run("light")
        i = min(int(tau * QU.samples_per_s), run["n"] - 1)
        formulas(f, [("fB1", smooth(t, LEGACY["b1"][0], LEGACY["b1"][0] + 1.0)), ("fB2", smooth(t, LEGACY["b1"][0], LEGACY["b1"][0] + 1.0))])
        q_main(f, "light", i)
        q_bars(f, "light", i)
        te_panel(f, LY.b_te_axes, ["light"], "light")
        caption(f, "cB1q" if tau < 5.0 else "cB1a", "cB1s" if tau >= 9.0 else None)
    elif part == "b2":
        tau = t - LEGACY["b2"][0]
        formulas(f, [("fB3", 1.0)])
        te_big(f, tau)
        caption(f, "cB2q" if tau < 3.0 else "cB2a", "cB2s" if tau >= 7.0 else None)
    elif part == "b3":
        tau = t - LEGACY["b3"][0]
        run = quantum_run("heavy")
        i = min(int(tau * QU.samples_per_s), run["n"] - 1)
        formulas(f, [("fB1", 1.0), ("fB2h", 1.0)])
        q_main(f, "heavy", i, magnify=True)
        q_bars(f, "heavy", i)
        te_panel(f, LY.b_te_axes, ["light", "heavy"], "heavy")
        if tau > 7.0:
            pl, ph = packet_numbers("light"), packet_numbers("heavy")
            f.text(0.5, 0.435, fill(f.tx["heavy_cmp"], pl=numt(100 * pl["pt"], ".1f", f.lang) + " %", ph=numt(100 * ph["pt"], ".2f", f.lang) + " %", mu=num(QU.heavy_factor, ".0f", f.lang)),
                   CFG.fonts.readout, (*f.TXT, smooth(tau, 7.0, 8.0)), ha="center")
        caption(f, "cB3q" if tau < 3.0 else "cB3a", "cB3s" if tau >= 5.0 else None)


def lim_frame(f: Frame, t: float) -> None:
    S, F, LY = CFG.style, CFG.fonts, CFG.layout
    tx = f.tx
    tau = t - LEGACY["lim"][0]
    ax = f.axes(LY.lim_state_axes)
    f.style_axes(ax)
    L = CFG.limits
    ax.set_xlim(*L.view_x)
    ax.set_ylim(0.0, 1.5 * QU.hill_height)
    ax.set_yticks([0.0, QU.hill_height])
    ax.set_yticklabels(["0", "$V$"])
    ax.set_xticks([])
    xs = np.linspace(L.view_x[0], L.view_x[1], L.samples)
    vg = q_potential(xs)
    e = q_mean_energy()
    ax.fill_between(xs, 0.0, vg, color=(*S.hill_fill, S.hill_fill_alpha), lw=0)
    ax.plot(xs, vg, color=(*S.hill_edge, 1.0), lw=S.hill_edge_width * f.sc)
    ax.plot(L.view_x, [e, e], color=(*f.GOLD, 0.9), lw=S.level_width * f.sc, ls=(0, tuple(S.level_dash)))
    d = stationary_state(xs)
    y = e + QU.stationary_density_height * QU.hill_height * d
    ax.fill_between(xs, e, y, color=(*f.BLUE, QU.density_alpha), lw=0)
    ax.plot(xs, y, color=(*f.BLUE, 1.0), lw=1.8 * f.sc)
    f.label(ax, (0.0, 1.04), tx["lim_state"], F.panel_title, (*f.DIM, 1.0), ha="left", va="bottom")
    f.text(0.05, 0.89, tx["lim_title"], F.lim_head, (*f.TXT, smooth(tau, 0.0, 0.8)), max_w=0.9)
    items = [("lim_y_head", "lim_y1", f.GREEN, 1.0), ("lim_n_head", "lim_n1", f.WARM, 3.2), ("", "lim_n2", f.WARM, 5.2), ("", "lim_n3", f.WARM, 6.9)]
    y = LY.lim_text_y
    for head, body, col, t_in in items:
        a = smooth(tau, t_in, t_in + 0.8)
        if head:
            y -= LY.lim_head_pre if head == "lim_n_head" else 0.0
            f.text(LY.lim_text_x, y, tx[head], F.lim_head, (*col, a))
            y -= LY.lim_head_gap
        n = f.paragraph(LY.lim_text_x, y, tx[body], F.lim_text, (*f.TXT, a), LY.lim_text_width, LY.lim_line)
        y -= n * LY.lim_line + LY.lim_par_gap
    caption(f, "cL")


# ======================================================================================================================= part 3: the circuit

JC = CFG.josephson
CI = CFG.circuit


def washboard_u(delta, s_bias: float):
    """U / E_J of the tilted washboard."""
    return tp.washboard(delta, s_bias)


def circuit_s(t_sim: float) -> float:
    """The bias s = I / I_c as a function of the simulation time (held at s_start, then raised smoothly to s_end)."""
    c2 = LEGACY["c2"][1] - LEGACY["c2"][0]
    t_hold = JC.ramp_hold_s / c2 * JC.t_total
    return JC.s_start + (JC.s_end - JC.s_start) * smooth(t_sim, t_hold, JC.t_total)


def circuit_run() -> dict:
    """The phase packet in the washboard that is tilted with time, sampled at samples_per_s states per film second (cached on the disk)."""
    if "circ" in _cache:
        return _cache["circ"]
    key = hashlib.sha1(json.dumps([dict(JC.__dict__) if hasattr(JC, "__dict__") else str(JC), LEGACY["c2"]], default=str, sort_keys=True).encode()).hexdigest()[:12]
    path = HERE / CI.cache_dir / f"circuit_{key}.npz"
    c2 = LEGACY["c2"][1] - LEGACY["c2"][0]
    n_s = int(round(c2 * JC.samples_per_s)) + 1
    if path.exists():
        z = np.load(path)
        out = {k: z[k] for k in z.files}
    else:
        lam = JC.lam
        mass = 1.0 / lam ** 2                                   # hbar = E_J = 1: M = hbar^2 / (lam^2 E_J)
        length = JC.grid_to - JC.grid_from
        sim = tp.SplitStep1D(JC.grid_n, length, mass, 1.0, JC.dt, lambda t: 0.0, JC.absorber_width, JC.absorber_strength, x_left=JC.grid_from)
        sim.potential = lambda t: washboard_u(sim.x, circuit_s(t))
        psi0, _ = tp.well_ground_state(sim.x, JC.s_start, lam)
        sim.set_state(psi0)
        stride = max(1, JC.grid_n // 1000)
        idx = np.arange(0, JC.grid_n, stride)
        dens = np.zeros((n_s, idx.size))
        s_arr, p_esc, nrm = np.zeros(n_s), np.zeros(n_s), np.zeros(n_s)
        for i in range(n_s):
            sim.run(i * JC.t_total / (n_s - 1))
            d = sim.density()
            s_arr[i] = circuit_s(sim.t)
            top = math.pi - math.asin(s_arr[i])
            nrm[i] = sim.norm()
            p_esc[i] = float(d[sim.x > top].sum() * sim.dx) + (1.0 - nrm[i])
            dens[i] = d[idx]
        out = {"x": sim.x[idx], "dens": dens, "s": s_arr, "p_esc": p_esc, "norm": nrm}
        path.parent.mkdir(exist_ok=True)
        np.savez(path, **out)
    out = dict(out)
    out["n"] = n_s
    _cache["circ"] = out
    return out


def rate_curves() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Gamma / omega_p against s: WKB (zero-point level) and the cubic formula; Gamma = (omega_p / 2 pi) exp(-S) and (omega_p / 2 pi) a_q exp(-7.2 x)."""
    if "rate" not in _cache:
        sg = np.linspace(JC.rate_s_range[0], JC.rate_s_range[1], JC.rate_samples)
        def s_wkb(v: float) -> float:
            try:
                return tp.wkb_exponent_washboard(v, JC.lam)
            except ValueError:                                  # the zero-point level is above the top of the barrier: no forbidden region left
                return 0.0

        wkb = np.array([math.exp(-s_wkb(float(v))) / (2.0 * math.pi) for v in sg])
        cub = np.array([tp.rate_cubic(tp.barrier_over_hbar_omega(float(v), JC.lam)) / (2.0 * math.pi) for v in sg])
        _cache["rate"] = (sg, wkb, cub)
    return _cache["rate"]


def junction_panel(f: Frame, tau: float, alpha: float) -> None:
    S, F, LY = CFG.style, CFG.fonts, CFG.layout
    tx = f.tx
    ax = f.axes(LY.c1_junction_axes)
    ax.set_xlim(*CI.junction_xlim)
    ax.set_ylim(*CI.junction_ylim)
    ax.set_axis_off()
    by = CI.block_y
    for (x0, x1), col, key in ((CI.sc_blocks[0], f.BLUE, "jc_s1"), (CI.ins_block, f.WARM, "jc_ins"), (CI.sc_blocks[1], f.BLUE, "jc_s1")):
        ax.fill_between([x0, x1], by[0], by[1], color=(*col, CI.block_alpha * alpha), lw=0)
        ax.plot([x0, x1, x1, x0, x0], [by[0], by[0], by[1], by[1], by[0]], color=(*col, CI.block_edge_alpha * alpha), lw=1.4 * f.sc)
        ax.text(0.5 * (x0 + x1), by[0] - 0.08, tx[key], color=(*f.DIM, alpha), fontsize=F.legend * f.sc, ha="center", va="top")
    rng = np.random.default_rng(CI.pair_seed)
    for (x0, x1) in CI.sc_blocks:
        px = x0 + 0.25 + (x1 - x0 - 0.5) * rng.random(CI.n_pairs)
        py = by[0] + 0.25 + (by[1] - by[0] - 0.5) * rng.random(CI.n_pairs)
        ax.scatter(px, py, s=(CI.pair_size * f.sc) ** 2, color=(*f.BLUE, CI.pair_alpha * alpha), edgecolors="none", zorder=3)
    delta = math.asin(CI.c1_s)
    for k, ((cx, cy), key) in enumerate(zip(CI.clock_centres, ("jc_phase1", "jc_phase2"))):
        ang = CI.clock_rate * tau + (delta if k == 1 else 0.0)
        r = CI.clock_radius
        th = np.linspace(0, 2 * math.pi, 100)
        ax.plot(cx + r * np.cos(th), cy + r * np.sin(th), color=(*f.DIM, 0.8 * alpha), lw=1.0 * f.sc)
        ax.annotate("", xy=(cx + r * math.cos(ang), cy + r * math.sin(ang)), xytext=(cx, cy),
                    arrowprops=dict(arrowstyle="-|>", color=(*f.GOLD, alpha), lw=2.0 * f.sc, shrinkA=0, shrinkB=0))
        ax.text(cx + r + 0.15, cy, tx[key], color=(*f.GOLD, alpha), fontsize=F.legend * f.sc, ha="left", va="center")
    ax.text(0.5 * (CI.sc_blocks[0][0] + CI.sc_blocks[1][1]), by[0] - 0.75, tx["jc_pairs"], color=(*f.TXT, alpha), fontsize=F.legend * f.sc, ha="center", va="top")


def well_panel(f: Frame, tau: float, alpha: float) -> None:
    S, F, LY = CFG.style, CFG.fonts, CFG.layout
    tx = f.tx
    if alpha < 0.02:
        return
    ax = f.axes(LY.c1_well_axes)
    f.style_axes(ax, ticks=False)
    ax.set_xlim(*CI.c1_xlim)
    ax.set_ylim(*CI.c1_ylim)
    d = np.linspace(CI.c1_xlim[0], CI.c1_xlim[1], CI.c1_samples)
    u = washboard_u(d, CI.c1_s)
    ax.fill_between(d, CI.c1_ylim[0], u, color=(*S.hill_fill, S.hill_fill_alpha * alpha), lw=0)
    ax.plot(d, u, color=(*S.hill_edge, alpha), lw=S.hill_edge_width * f.sc)
    d0 = math.asin(CI.c1_s)
    ball = d0 + CI.ball_jiggle * math.sin(CI.ball_rate * tau)
    ax.plot([ball], [float(washboard_u(ball, CI.c1_s))], "o", color=(*f.GOLD, alpha), ms=CI.ball_size * f.sc, zorder=5)
    f.label(ax, (0.0, 1.03), tx["jc_well"], F.panel_title, (*f.DIM, alpha), ha="left", va="bottom")
    f.label(ax, (1.0, -0.04), tx["delta_abs_axis"], F.axis_label, (*f.DIM, alpha), ha="right", va="top")
    f.label(ax, (0.02, 0.04), tx["jc_note"], F.legend, (*f.TXT, alpha), ha="left", va="bottom")


def c1_frame(f: Frame, t: float) -> None:
    tau = t - LEGACY["c1"][0]
    formulas(f, [("fC1", smooth(tau, 0.0, 1.0)), ("fC2", smooth(tau, 3.2, 4.2))])
    a = smooth(tau, 0.2, 1.2)
    junction_panel(f, tau, a)
    well_panel(f, tau, smooth(tau, 2.5, 3.5))
    caption(f, "cC1q" if tau < 3.0 else "cC1a", "cC1s" if tau >= 4.5 else None)


def c2_frame(f: Frame, t: float) -> None:
    S, F, LY = CFG.style, CFG.fonts, CFG.layout
    tx = f.tx
    tau = t - LEGACY["c2"][0]
    run = circuit_run()
    i = min(int(round(tau * JC.samples_per_s)), run["n"] - 1)
    s_b = float(run["s"][i])
    lam = JC.lam
    d0 = math.asin(s_b)
    top = math.pi - d0
    pr = tp.plasma_ratio(s_b)
    x_hw = tp.barrier_over_hbar_omega(s_b, lam)
    formulas(f, [("fC2", 1.0), ("fC3", smooth(tau, 0.4, 1.4))])
    u_min = float(washboard_u(d0, s_b))
    x0, x1 = d0 + JC.view_delta[0], d0 + JC.view_delta[1]
    # potential
    ax = f.axes(LY.c2_pot_axes)
    f.style_axes(ax, bottom=False)
    ax.set_xticks([])
    ax.set_xlim(*JC.view_delta)
    ax.set_ylim(*JC.view_u)
    d = np.linspace(x0, x1, 900)
    u = (washboard_u(d, s_b) - u_min) / lam
    ax.fill_between(d - d0, JC.view_u[0], u, color=(*S.hill_fill, S.hill_fill_alpha), lw=0)
    ax.plot(d - d0, u, color=(*S.hill_edge, 1.0), lw=S.hill_edge_width * f.sc)
    e0 = 0.5 * pr
    ax.plot(JC.view_delta, [e0, e0], color=(*f.GOLD, CI.level_alpha), lw=S.level_width * f.sc, ls=(0, tuple(S.level_dash)))
    ax.text(JC.view_delta[1] - 0.05, e0 + 0.25, tx["c2_level"], color=(*f.GOLD, 1.0), fontsize=F.legend * f.sc, ha="right", va="bottom")
    ax.plot([top - d0] * 2, [0.0, x_hw * pr], color=(*f.WARM, 0.9), lw=1.2 * f.sc, ls=(0, (2, 2)))
    f.label(ax, (0.0, 1.03), tx["c2_pot_title"], F.panel_title, (*f.DIM, 1.0), ha="left", va="bottom")
    # density
    bx = f.axes(LY.c2_den_axes)
    f.style_axes(bx)
    bx.set_yscale("log")
    bx.set_xlim(*JC.view_delta)
    bx.set_ylim(JC.density_floor, JC.density_ceiling)
    xs = run["x"]
    dn = np.maximum(run["dens"][i], JC.density_floor)
    bx.fill_between(xs - d0, JC.density_floor, dn, color=(*f.BLUE, CI.den_alpha), lw=0)
    bx.plot(xs - d0, dn, color=(*f.BLUE, 1.0), lw=1.6 * f.sc)
    bx.plot([top - d0] * 2, [JC.density_floor, JC.density_ceiling], color=(*f.WARM, 0.9), lw=1.2 * f.sc, ls=(0, (2, 2)))
    f.label(bx, (0.0, 1.04), tx["c2_den_title"], F.panel_title, (*f.DIM, 1.0), ha="left", va="bottom")
    f.label(bx, (1.0, -0.2), tx["delta_axis"], F.axis_label, (*f.DIM, 1.0), ha="right", va="top")
    # rate
    sg, wkb, cub = rate_curves()
    cx = f.axes(LY.c2_rate_axes)
    f.style_axes(cx)
    cx.set_yscale("log")
    cx.set_xlim(JC.rate_s_range[0], JC.rate_s_range[1])
    cx.set_ylim(*CI.rate_ylim)
    cx.plot(sg, np.maximum(wkb, JC.rate_floor), color=(*f.BLUE, 1.0), lw=CI.rate_line_width * f.sc)
    cx.plot(sg, np.maximum(cub, JC.rate_floor), color=(*f.WARM, 0.9), lw=CI.rate_line_width * f.sc, ls=(0, (4, 3)))
    w_now = float(np.interp(s_b, sg, wkb))
    cx.plot([s_b], [max(w_now, JC.rate_floor)], "o", color=(*f.GOLD, 1.0), ms=CI.rate_dot_size ** 0.5 * f.sc)
    cx.text(0.03, 0.28, tx["c2_wkb"], color=(*f.BLUE, 1.0), fontsize=F.legend * f.sc, transform=cx.transAxes, ha="left", va="center")
    cx.text(0.03, 0.12, tx["c2_cubic"], color=(*f.WARM, 1.0), fontsize=F.legend * f.sc, transform=cx.transAxes, ha="left", va="center")
    f.label(cx, (0.0, 1.05), tx["c2_rate_title"], F.panel_title, (*f.DIM, 1.0), ha="left", va="bottom")
    f.label(cx, (1.0, -0.2), tx["c2_xlabel"], F.axis_label, (*f.DIM, 1.0), ha="right", va="top")
    # readout
    lang = f.lang
    vals = {"s": num(s_b, ".3f", lang), "dU": num(tp.barrier_height(s_b), ".4f", lang), "x": num(x_hw, ".2f", lang), "wp": num(pr, ".3f", lang),
            "P": (numt(100.0 * float(run["p_esc"][i]), ".1f", lang) + " %") if run["p_esc"][i] >= CI.p_show_min else ("< " + numt(100.0 * CI.p_show_min, ".1f", lang) + " %")}
    for k, key in enumerate(("c2_s", "c2_dU", "c2_wp", "c2_p")):
        f.text(LY.c2_text_x, LY.c2_text_y - LY.c2_text_row * k, fill(tx[key], **vals), F.readout, (*f.TXT, CI.c2_text_alpha), max_w=0.30)
    caption(f, "cC2q" if tau < 4.0 else "cC2a", "cC2s" if tau >= 6.0 else None, **params(lang))


# ======================================================================================================================= part 2: a classical wave on a chain

WV = CFG.wave
OB = CFG.objects
MP = CFG.mass_plot


def wave_kappa() -> float:
    return math.sqrt(WV.big_omega ** 2 - WV.omega ** 2)                       # c = 1


def wave_t_exact(a: float) -> float:
    """T of a segment of thickness a: the same formula as the rectangular barrier, E -> omega^2, V -> Omega^2, 2m / hbar^2 -> 1."""
    return tp.rectangle_transmission(WV.omega ** 2, WV.big_omega ** 2, a, 0.5, 1.0)


def wave_run(i: int) -> dict:
    """The driven chain with the segment [0, a_i]: leapfrog, a soft point source (transparent for the waves), damping layers at the ends.
    Stored at samples_per_s states per film second of the parts w1 + w2: the displacement on the drawn masses and the amplitude A = sqrt(phi^2 + (phi_t / omega)^2)."""
    key = ("wave", i)
    if key in _cache:
        return _cache[key]
    a = WV.widths[i]
    n = int(round((WV.x_right - WV.x_left) / WV.dx)) + 1
    x = WV.x_left + WV.dx * np.arange(n)
    k0 = np.where((x > -0.5 * WV.dx) & (x < a - 0.5 * WV.dx), WV.big_omega ** 2, 0.0)           # a / dx sites: the width of the segment is exactly a
    dl = np.clip((x[0] + WV.absorber - x) / WV.absorber, 0.0, 1.0)
    dr = np.clip((x - (x[-1] - WV.absorber)) / WV.absorber, 0.0, 1.0)
    sig = WV.absorber_strength * (dl ** 2 + dr ** 2)
    g = np.exp(-0.5 * ((x - WV.source_x) / WV.source_width) ** 2)
    g /= g.sum() * WV.dx
    w, dt, dx = WV.omega, WV.dt, WV.dx
    steps = int(round(WV.t_total / dt))
    d_film = (PARTS["w2"][1] - PARTS["w1"][0])
    n_s = int(round(d_film * WV.samples_per_s)) + 1
    marks = set(int(round(k * steps / (n_s - 1))) for k in range(n_s))
    view = (x >= WV.view_x[0]) & (x <= WV.view_x[1])
    vi = np.where(view)[0]
    phi_prev = np.zeros(n)
    phi = np.zeros(n)
    phi_s = np.zeros((n_s, vi[::WV.dot_stride].size))
    amp_s = np.zeros((n_s, vi.size))
    amp_full_last = None
    k_s = 0
    ramp_t = WV.ramp_periods * 2.0 * math.pi / w
    for step in range(steps + 1):
        t = step * dt
        u = min(max(t / ramp_t, 0.0), 1.0)
        ramp, d_ramp = u * u * (3 - 2 * u), 6.0 * u * (1.0 - u) / ramp_t
        src = 2.0 * WV.source_amp * (d_ramp * math.sin(w * t) + ramp * w * math.cos(w * t))      # the force is the time derivative: no static offset is radiated
        lap = np.zeros(n)
        lap[1:-1] = (phi[2:] + phi[:-2] - 2.0 * phi[1:-1]) / dx ** 2
        nxt = 2.0 * phi - phi_prev + dt ** 2 * (lap - k0 * phi + g * src) - dt * sig * (phi - phi_prev)
        nxt[0] = nxt[-1] = 0.0
        if step in marks:
            vt = (nxt - phi_prev) / (2.0 * dt)
            amp = np.sqrt(phi ** 2 + (vt / w) ** 2)
            phi_s[k_s] = phi[vi][:: WV.dot_stride]
            amp_s[k_s] = amp[vi]
            amp_full_last = (x.copy(), amp.copy())
            k_s += 1
        phi_prev, phi = phi, nxt
    xa, am = amp_full_last
    ml = (xa >= WV.measure_left[0]) & (xa <= WV.measure_left[1])
    lam = 2.0 * math.pi / w
    mr = (xa >= a + WV.measure_right_margin) & (xa <= a + WV.measure_right_margin + lam)
    amax, amin = am[ml].max(), am[ml].min()
    inc, refl = 0.5 * (amax + amin), 0.5 * (amax - amin)
    t_sim = float((am[mr].mean() / inc) ** 2)
    r_sim = float((refl / inc) ** 2)
    out = {"a": a, "x_view": x[vi], "x_dots": x[vi][:: WV.dot_stride], "phi": phi_s, "amp": amp_s, "n": n_s, "t": t_sim, "r": r_sim, "inc": inc,
           "t_exact": wave_t_exact(a), "amp_x": xa, "amp_last": am}
    _cache[key] = out
    return out


def wave_fill(tx: str, **kw) -> str:
    return fill(tx, **kw)


def wave_scene(f: Frame, run: dict, k: int, alpha: float = 1.0) -> None:
    S, F, LY = CFG.style, CFG.fonts, CFG.layout
    tx = f.tx
    ax = f.axes(LY.wave_scene_axes)
    ax.set_xlim(*WV.view_x)
    ax.set_ylim(*WV.scene_ylim)
    ax.set_axis_off()
    a = run["a"]
    ax.axvspan(0.0, a, color=(*f.WARM, WV.segment_alpha * alpha), lw=0)
    ax.plot([WV.view_x[0], WV.view_x[1]], [WV.floor_y, WV.floor_y], color=(*f.DIM, 0.4 * alpha), lw=1.0 * f.sc)
    xd, yd = run["x_dots"], WV.amp_scale * run["phi"][k]
    ax.plot(xd, yd, color=(*f.BLUE, 0.55 * alpha), lw=1.2 * f.sc, zorder=3)
    ax.scatter(xd, yd, s=(S.dot_size ** 0.5 * 0.34 * f.sc) ** 2, color=(*f.BLUE, alpha), edgecolors="none", zorder=4)
    inside = (xd > -0.5 * WV.dx) & (xd < a - 0.5 * WV.dx)
    for xx, yy in zip(xd[inside], yd[inside]):
        ax.plot([xx, xx], [yy, WV.floor_y], color=(*f.WARM, WV.spring_alpha * alpha), lw=1.4 * f.sc, zorder=2)
    f.label(ax, (0.0, 1.0), tx["w_scene"], F.panel_title, (*f.DIM, alpha), ha="left", va="bottom")
    ax.text(a + 0.8, WV.floor_y + 0.12, tx["w_floor"], color=(*f.WARM, alpha), fontsize=F.legend * f.sc, ha="left", va="bottom")


def wave_env(f: Frame, run: dict, k: int, alpha: float = 1.0, show_fit: bool = True) -> None:
    S, F, LY = CFG.style, CFG.fonts, CFG.layout
    tx = f.tx
    ax = f.axes(LY.wave_env_axes)
    f.style_axes(ax)
    ax.set_yscale("log")
    ax.set_xlim(*WV.view_x)
    ax.set_ylim(*WV.env_ylim)
    a = run["a"]
    xv, am = run["x_view"], np.maximum(run["amp"][k], WV.env_ylim[0] * 1.001)
    ax.axvspan(0.0, a, color=(*f.WARM, WV.segment_alpha * alpha), lw=0)
    ax.plot(xv, am, color=(*f.BLUE, alpha), lw=S.curve_width * f.sc)
    if show_fit:
        i0 = int(np.argmin(np.abs(xv - 0.0)))
        xs = np.linspace(0.0, WV.view_x[1], 200)
        ax.plot(xs, np.maximum(am[i0] * np.exp(-wave_kappa() * xs), 1e-12), color=(*f.WARM, 0.95 * alpha), lw=1.6 * f.sc, ls=(0, tuple(S.level_dash)))
        ax.text(0.5 * (0.0 + WV.view_x[1]) + 3.0, WV.env_ylim[0] * 4.0, tx["w_exp"], color=(*f.WARM, alpha), fontsize=F.legend * f.sc, ha="left")
    f.label(ax, (0.0, 1.03), tx["w_env"], F.panel_title, (*f.DIM, alpha), ha="left", va="bottom")
    f.label(ax, (1.0, -0.2), tx["w_x"], F.axis_label, (*f.DIM, alpha), ha="right", va="top")


def wave_readout(f: Frame, run: dict, alpha: float) -> None:
    LY, F = CFG.layout, CFG.fonts
    tx, lang = f.tx, f.lang
    k, a = wave_kappa(), run["a"]
    vals = {"k": num(k, ".2f", lang), "a": num(a, ".1f", lang), "ka": num(k * a, ".1f", lang), "ts": num(run["t"], ".4f", lang), "te": num(run["t_exact"], ".4f", lang),
            "r": num(run["r"], ".3f", lang), "tr": num(run["t"] + run["r"], ".3f", lang), "w": num(WV.omega / WV.big_omega, ".1f", lang)}
    for row, key in enumerate(("w_r_k", "w_r_a", "w_r_t", "w_r_te", "w_r_r")):
        f.text(LY.wave_text_x, LY.wave_text_y - LY.wave_text_row * row, fill(tx[key], **vals), F.readout, (*f.TXT, alpha), max_w=0.30)


def wave_t_panel(f: Frame, rect, n_runs: int, alpha: float = 1.0) -> None:
    S, F = CFG.style, CFG.fonts
    tx = f.tx
    ax = f.axes(rect)
    f.style_axes(ax)
    ax.set_yscale("log")
    k = wave_kappa()
    xmax = 1.15 * k * max(WV.widths)
    ka = np.linspace(0.0, xmax, 300)
    te = np.array([wave_t_exact(v / k) for v in ka])
    pref = 16.0 * WV.omega ** 2 * (WV.big_omega ** 2 - WV.omega ** 2) / WV.big_omega ** 4
    ax.set_xlim(0.0, xmax)
    ax.set_ylim(1.0e-4 / 3.0, 1.3)
    ax.plot(ka, te, color=(*f.BLUE, alpha), lw=S.curve_width * f.sc)
    ax.plot(ka, pref * np.exp(-2.0 * ka), color=(*f.WARM, 0.9 * alpha), lw=1.4 * f.sc, ls=(0, tuple(S.level_dash)))
    ax.text(0.04, 0.12, tx["w_exact"], color=(*f.BLUE, alpha), fontsize=F.legend * f.sc, transform=ax.transAxes)
    ax.text(0.04, 0.04, tx["w_asym"], color=(*f.WARM, alpha), fontsize=F.legend * f.sc, transform=ax.transAxes)
    for i in range(n_runs):
        r = wave_run(i)
        ax.plot([k * r["a"]], [r["t"]], "o", color=(*f.GOLD, alpha), ms=S.dot_size ** 0.5 * 0.9 * f.sc, zorder=5)
    f.label(ax, (0.0, 1.04), tx["w_t_title"], F.panel_title, (*f.DIM, alpha), ha="left", va="bottom")
    f.label(ax, (1.0, -0.2), tx["w_t_x"], F.axis_label, (*f.DIM, alpha), ha="right", va="top")


def wave_frame(f: Frame, t: float) -> None:
    S, F, LY = CFG.style, CFG.fonts, CFG.layout
    part = part_of(t)
    run = wave_run(WV.featured)
    P = {"w": num(WV.omega / WV.big_omega, ".1f", f.lang)}
    if part in ("w1", "w2"):
        tau = t - PARTS["w1"][0]
        k = min(int(round(tau * WV.samples_per_s)), run["n"] - 1)
        f.text(CFG.layout.formula_pos[0], CFG.layout.formula_pos[1], fill(f.tx["fW1"], **P), F.formula, (*f.TXT, S.formula_alpha * smooth(tau, 0.0, 1.0)), max_w=0.94)
        f.text(CFG.layout.formula_pos[0], CFG.layout.formula_pos[1] - LY.formula_row, fill(f.tx["fW1b"], **P), F.formula_small, (*f.TXT, S.formula_alpha * smooth(tau, 1.0, 2.0)), max_w=0.94)
        wave_scene(f, run, k)
        a_env = smooth(tau, 4.0, 6.0)
        if a_env > 0.01:
            wave_env(f, run, k, a_env, show_fit=tau >= PARTS["w2"][0] - PARTS["w1"][0])
        if part == "w1":
            caption(f, "cW1q" if tau < 5.0 else "cW1a")
        else:
            f.text(CFG.layout.formula_pos[0], CFG.layout.formula_pos[1] - 2 * LY.formula_row + 0.02, fill(f.tx["fW2"], **P), F.formula_small, (*f.TXT, S.formula_alpha * smooth(tau, 12.0, 13.0)), max_w=0.94)
            wave_readout(f, run, smooth(tau, 16.0, 18.0))
            caption(f, "cW2q" if tau < 17.0 else "cW2a")
    else:
        tau = t - PARTS["w3"][0]
        loop = int(round(WV.loop_periods * 2.0 * math.pi / WV.omega / (WV.t_total / (run["n"] - 1))))      # whole periods: the loop has no jump of the phase
        k = run["n"] - loop + int(tau * WV.samples_per_s) % loop
        f.text(CFG.layout.formula_pos[0], CFG.layout.formula_pos[1], fill(f.tx["fW3"], **P), F.formula_small * 1.15, (*f.TXT, S.formula_alpha * smooth(tau, 0.0, 1.0)), max_w=0.94)
        wave_scene(f, run, k)
        n_runs = 1 + int(tau / 2.0)
        wave_t_panel(f, LY.wave_env_axes, min(n_runs, len(WV.widths)))
        for row in range(min(n_runs, len(WV.widths))):
            r = wave_run(row)
            vals = {"a": num(r["a"], ".1f", f.lang), "ka": num(wave_kappa() * r["a"], ".1f", f.lang), "ts": num(r["t"], ".4f", f.lang), "te": num(r["t_exact"], ".4f", f.lang)}
            f.text(LY.wave_text_x, LY.wave_text_y - 1.6 * LY.wave_text_row * row, fill(f.tx["w_r_a"], **vals) + "   " + fill(f.tx["w_r_t"], **vals), F.readout, (*f.TXT, 1.0), max_w=0.30)
        caption(f, "cW3q" if tau < 2.0 else "cW3a")


# ======================================================================================================================= part 3: the matter wave


def pow10(x: float, lang: str) -> str:
    """10^x for a negative x as mathtext (the decimal comma in Russian); a huge exponent is written as m x 10^n."""
    ax_ = abs(x)
    if ax_ < 1.0e3:
        m = format(ax_, ".1f")
        return "10^{-" + (m.replace(".", "{,}") if lang == "ru" else m) + "}"
    e = int(math.floor(math.log10(ax_)))
    m = format(ax_ / 10 ** e, ".1f")
    return "10^{-" + (m.replace(".", "{,}") if lang == "ru" else m) + r"\times10^{" + str(e) + "}}"


def object_exponents() -> dict:
    """log10 of exp(-2 kappa a) for V - E = v_minus_e_ev and the width width_nm."""
    out = {}
    for key, m in (("e", OB.electron_kg), ("p", OB.proton_kg), ("g", OB.grain_kg)):
        kappa = math.sqrt(2.0 * m * OB.v_minus_e_ev * OB.ev) / OB.hbar
        out[key] = -2.0 * kappa * OB.width_nm * 1.0e-9 / math.log(10.0)
    return out


def mass_data() -> dict:
    if "mass" not in _cache:
        e = q_mean_energy()
        t = [tp.transmission(e, q_potential, QU.mass * mu, QU.hbar, QU.ode_half_width) for mu in MP.factors]
        _cache["mass"] = {"mu": np.array(MP.factors), "t": np.array(t)}
    return _cache["mass"]


def matter_hill(f: Frame, ax, tau: float, alpha: float = 1.0) -> None:
    S, F = CFG.style, CFG.fonts
    tx = f.tx
    f.style_axes(ax, ticks=False)
    xs = np.linspace(QU.view_x[0], QU.view_x[1], QU.stationary_samples)
    e = q_mean_energy()
    ax.set_xlim(*QU.view_x)
    ax.set_ylim(0.0, 1.5 * QU.hill_height)
    vg = q_potential(xs)
    ax.fill_between(xs, 0.0, vg, color=(*S.hill_fill, S.hill_fill_alpha * alpha), lw=0)
    ax.plot(xs, vg, color=(*S.hill_edge, alpha), lw=S.hill_edge_width * f.sc)
    ax.plot(QU.view_x, [e, e], color=(*f.GOLD, 0.9 * alpha), lw=S.level_width * f.sc, ls=(0, tuple(S.level_dash)))
    psi = tp.stationary_state(e, q_potential, QU.mass, QU.hbar, QU.ode_half_width, xs)
    d = np.abs(psi) ** 2
    y = e + QU.stationary_density_height * QU.hill_height * d
    ax.fill_between(xs, e, y, color=(*f.BLUE, QU.density_alpha * alpha), lw=0)
    ax.plot(xs, y, color=(*f.BLUE, alpha), lw=1.8 * f.sc)
    re = np.real(psi * np.exp(-1j * e / QU.hbar * tau * CFG.quantum.stationary_phase_rate))
    ax.plot(xs, e + QU.stationary_density_height * QU.hill_height * re * 0.8, color=(1.0, 1.0, 1.0, 0.7 * alpha), lw=1.0 * f.sc)
    f.label(ax, (0.0, 1.03), tx["m_dens"], F.panel_title, (*f.DIM, alpha), ha="left", va="bottom")
    f.label(ax, (0.62, 0.95), fill(tx["m_hill"], ), F.legend, (*f.DIM, alpha), ha="left", va="top")


def matter_frame(f: Frame, t: float) -> None:
    S, F, LY = CFG.style, CFG.fonts, CFG.layout
    tx, lang = f.tx, f.lang
    part = part_of(t)
    if part == "m1":
        tau = t - PARTS["m1"][0]
        f.text(LY.formula_pos[0], LY.formula_pos[1], tx["fM1"], F.formula, (*f.TXT, S.formula_alpha * smooth(tau, 0.0, 1.0)), max_w=0.94)
        f.text(LY.formula_pos[0], LY.formula_pos[1] - LY.formula_row, tx["fM1b"], F.formula_small, (*f.TXT, S.formula_alpha * smooth(tau, 2.0, 3.0)), max_w=0.94)
        ax = f.axes(LY.m1_axes)
        matter_hill(f, ax, tau, smooth(tau, 0.2, 1.2))
        tt = tp.transmission(q_mean_energy(), q_potential, QU.mass, QU.hbar, QU.ode_half_width)
        f.text(0.70, 0.40, fill(tx["m_T"], t=num(tt, ".3f", lang)), F.lim_head, (*f.TXT, smooth(tau, 6.0, 7.0)))
        caption(f, "cM1q" if tau < 5.0 else "cM1a")
    else:
        tau = t - PARTS["m2"][0]
        f.text(LY.formula_pos[0], LY.formula_pos[1], tx["fM2"], F.formula, (*f.TXT, S.formula_alpha * smooth(tau, 0.0, 1.0)), max_w=0.94)
        d = mass_data()
        ax = f.axes(LY.m2_axes)
        f.style_axes(ax)
        xs = np.sqrt(d["mu"])
        ax.set_xlim(0.0, 1.1 * xs.max())
        ax.set_ylim(*MP.log_ylim)
        ly = np.log10(d["t"])
        reveal = smooth(tau, 0.5, 4.0)
        sel = xs <= xs.min() + reveal * (xs.max() - xs.min()) + 1e-9
        ax.plot(xs[sel], ly[sel], "-o", color=(*f.BLUE, 1.0), lw=S.curve_width * f.sc, ms=S.dot_size ** 0.5 * 0.7 * f.sc)
        for mu in MP.marked:
            i = int(np.argmin(np.abs(d["mu"] - mu)))
            if sel[i]:
                ax.text(xs[i] + 0.1, ly[i] + 0.35, f"${num(mu, '.0f', lang)}\\,m_0$", color=(*f.GOLD, 1.0), fontsize=F.legend * f.sc, ha="left", va="bottom")
        f.label(ax, (0.0, 1.04), tx["m_plot_title"], F.panel_title, (*f.DIM, 1.0), ha="left", va="bottom")
        f.label(ax, (1.0, -0.12), tx["m_plot_x"], F.axis_label, (*f.DIM, 1.0), ha="right", va="top")
        ex = object_exponents()
        a_o = smooth(tau, 5.0, 6.0)
        vals = {"te": pow10(ex["e"], lang), "tp": pow10(ex["p"], lang), "tg": pow10(ex["g"], lang)}
        f.text(LY.m2_text_x, LY.m2_text_y, tx["m_obj_head"], F.lim_head, (*f.DIM, a_o))
        for row, key in enumerate(("m_obj_e", "m_obj_p", "m_obj_g")):
            f.text(LY.m2_text_x, LY.m2_text_y - LY.m2_text_row * (row + 1), fill(tx[key], **vals), F.lim_text, (*f.TXT, smooth(tau, 5.5 + 0.8 * row, 6.5 + 0.8 * row)), max_w=0.38)
        caption(f, "cM2q" if tau < 3.0 else "cM2a", "cM2s" if tau >= 5.0 else None, v=num(OB.v_minus_e_ev, ".0f", lang), a=num(OB.width_nm, ".1f", lang))


def outro_frame(f: Frame, t: float) -> None:
    LY, F = CFG.layout, CFG.fonts
    tau = t - PARTS["outro"][0]
    for k, key in enumerate(("s1", "s2", "s3", "s4")):
        a = smooth(tau, CI.outro_in[k] * 0.7, CI.outro_in[k] * 0.7 + CI.outro_fade)
        f.paragraph(LY.outro_x, LY.outro_y - LY.outro_row * k, f.tx[key], F.outro * 0.85, (*f.TXT, a), 0.86, 0.045)


def josephson_outro_frame(f: Frame, t: float) -> None:
    LY, F = CFG.layout, CFG.fonts
    tau = t - LEGACY["outro_josephson"][0]
    for k, key in enumerate(("o1", "o2", "o3", "o4")):
        a = smooth(tau, CI.outro_in[k], CI.outro_in[k] + CI.outro_fade)
        f.text(LY.outro_x, LY.outro_y - LY.outro_row * k, f.tx[key], F.outro, (*f.TXT, a), max_w=0.88)


# ======================================================================================================================= the film

def render_frame(f: Frame, tf: float, total: float) -> None:
    k = total / TOTAL
    t, card, cprog = timeline(tf / k)
    draw_header(f, t, card, cprog)
    if card is not None:
        return
    part = part_of(t)
    if part in ("a1", "a2", "a3"):
        classical_frame(f, t)
    elif part in ("w1", "w2", "w3"):
        wave_frame(f, t)
    elif part in ("m1", "m2"):
        matter_frame(f, t)
    elif part == "outro":
        outro_frame(f, t)


def render(out: Path, size: tuple[int, int], fps: int, total: float, lang: str, snap: float | None = None) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if snap is None and not shutil.which("ffmpeg"):
        sys.exit("ffmpeg is required")
    V = CFG.video
    W, H = size
    fig = plt.figure(figsize=(W / V.dpi, H / V.dpi), dpi=V.dpi, facecolor=CFG.style.background)
    bg = CFG.style.background
    bg_rgba = np.array([int(bg[1:3], 16), int(bg[3:5], 16), int(bg[5:7], 16), 255], np.float32)
    frames = int(round(total * fps))
    ids = [int(round(snap * fps))] if snap is not None else range(frames)
    writer = None
    if snap is None:
        writer = subprocess.Popen(
            ["ffmpeg", "-y", "-v", "error", "-f", "rawvideo", "-pix_fmt", "rgba", "-s", f"{W}x{H}", "-r", str(fps), "-i", "-", "-c:v", "libx264",
             "-preset", V.preset, "-crf", str(V.crf), "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(out)], stdin=subprocess.PIPE)
    for k_ in ids:
        tf = k_ / fps
        fig.clear()
        fig.patch.set_facecolor(bg)
        f = Frame(fig, size, lang)
        render_frame(f, tf, total)
        fade = min(smooth(tf, 0.0, V.fade_s), 1.0 - smooth(tf, total - V.fade_s, total))
        fig.canvas.draw()
        frame = np.asarray(fig.canvas.buffer_rgba()).astype(np.float32)
        if fade < 1.0:
            frame = bg_rgba + (frame - bg_rgba) * fade
        frame = frame.clip(0, 255).astype(np.uint8)
        if writer is None:
            from PIL import Image
            Image.fromarray(frame).save(out)
        else:
            writer.stdin.write(frame.tobytes())
    if writer is not None:
        writer.stdin.close()
        writer.wait()
    plt.close(fig)
    print(f"wrote {out}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--lang", choices=("en", "ru"), default="en")
    ap.add_argument("--preview", action="store_true")
    ap.add_argument("--snapshot", type=float, default=None, help="one PNG at this CONTENT time (seconds, the chapter cards are not counted)")
    ap.add_argument("--film-snapshot", type=float, default=None, help="one PNG at this film time (the cards are counted)")
    ap.add_argument("--seconds", type=float, default=TOTAL)
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--config", type=Path, default=None, help="another configuration file instead of config.toml")
    ap.add_argument("--set", action="append", default=[], metavar="SECTION.KEY=VALUE", help="override one configuration value")
    args = ap.parse_args()
    V = CFG.video
    out = args.out or HERE / "media" / f"quantum_tunneling_{args.lang}.mp4"
    out.parent.mkdir(parents=True, exist_ok=True)
    if args.snapshot is not None or args.film_snapshot is not None:
        tsnap = film_time(args.snapshot) if args.snapshot is not None else args.film_snapshot
        render(out.with_suffix(".png"), (V.width, V.height), V.fps, args.seconds, args.lang, snap=tsnap)
    elif args.preview:
        render(out.with_name(out.stem + "_preview.mp4"), (V.preview_width, V.preview_height), V.preview_fps, args.seconds, args.lang)
    else:
        render(out, (V.width, V.height), V.fps, args.seconds, args.lang)


if __name__ == "__main__":
    main()
