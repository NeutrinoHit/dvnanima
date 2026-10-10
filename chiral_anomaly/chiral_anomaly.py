r"""The chiral anomaly: where do the particles come from?  (the book, the chapter "Quantum anomalies", the section "The chiral anomaly")

The film shows the anomaly as a flow of the levels of the Dirac sea (spectral flow); every number on the screen is counted from the levels
(see spectral_flow.py and its tests):

 1. a massless fermion on a ring: right movers E = +p, left movers E = -p, the quantised levels, the filled Dirac sea, the charges
    Q = N_R + N_L and Q_5 = N_R - N_L, both conserved without a field;
 2. an electric field shifts p -> p + eEt: the levels flow, the levels that cross E = 0 are counted, N_R = -N_L = eEL t / 2 pi,
    dQ_5/dt = eEL/pi, d_mu j^mu_5 = (e/pi) E;
 3. the cut-off: an energy window (gauge invariant) lets filled levels in at the bottom of the right branch and out at the bottom of the left
    one; a cut in fixed labels moves with the gauge potential; a crystal (E = sin k) is a regularisation without infinities;
 4. 3+1 dimensions: the lowest Landau level of a massless fermion is a 1+1 chiral branch with eB/2pi states per unit area, a stack of identical
    ladders, dn_5/dt = e^2 E.B / (2 pi^2) = (2 alpha / pi) E.B;
 5. the triangle diagram and the decay pi0 -> gamma gamma.

All the numbers are in config.toml and all the words in texts.toml (see ../dvconfig.py for --config / --set).

Usage:
    python chiral_anomaly.py --lang en            # film -> media/chiral_anomaly_en.mp4
    python chiral_anomaly.py --lang ru
    python chiral_anomaly.py --lang en --snapshot 40
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
sys.path.insert(0, str(HERE))
import spectral_flow as SF  # noqa: E402

CFG = load_config(HERE)
PH, STY, LY, FN = CFG.physics, CFG.style, CFG.layout, CFG.fonts
MN, RG, ST = CFG.main, CFG.ring, CFG.stair
PARTCFG = [CFG.part1, CFG.part2, CFG.part3, CFG.part4, CFG.part5]

CARD_S = CFG.timeline.card_s
CARD_AT = list(CFG.timeline.card_at)
CARD_KEYS = [f"h{i}" for i in range(len(CARD_AT))]
_PB = list(CFG.timeline.part_bounds)
CONTENT_TOTAL = _PB[-1]
TOTAL = CONTENT_TOTAL + CARD_S * len(CARD_AT)
PARTS = [(_PB[i], _PB[i + 1]) for i in range(len(_PB) - 1)]


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


def part_of(tc: float) -> int:
    for i, (a, b) in enumerate(PARTS):
        if a <= tc < b:
            return i + 1
    return len(PARTS)


def smooth(x: float, a: float, b: float) -> float:
    u = min(max((x - a) / (b - a), 0.0), 1.0)
    return u * u * (3 - 2 * u)


def num(x: float, fmt: str, lang: str) -> str:
    s = format(x, fmt)
    return s.replace(".", "{,}") if lang == "ru" else s


def signed(n: int) -> str:
    return f"+{n}" if n > 0 else (f"{n}" if n < 0 else "0")


def pick(items, u: float):
    """The last [from, value] of a sorted list whose start does not exceed u."""
    out = items[0][1]
    for a, v in items:
        if a <= u:
            out = v
    return out


TEXT = {lang: load_texts(HERE, lang) for lang in ("en", "ru")}


def fill(text: str, lang: str, **kw) -> str:
    """Replace @name@ by a number (a tuple (value, format)) or a string."""
    for k, v in kw.items():
        text = text.replace(f"@{k}@", num(v[0], v[1], lang) if isinstance(v, tuple) else str(v))
    return text


# ------------------------------------------------------------------------------------------------------ the physics of the film

def ring_of(rate: float) -> SF.Ring:
    """The ring of the film: the spacing of the levels is PH.spacing, the level rate eEL/2pi is ``rate`` (levels per film second)."""
    return SF.Ring(length=2.0 * math.pi / PH.spacing, theta=PH.theta, e=PH.charge, field=rate * PH.spacing / PH.charge)


_cache: dict = {}


def staircase(rate: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """N_R and -N_L against the shift of the levels (in spacings), counted level by level on a fine grid."""
    key = ("stair", rate)
    if key not in _cache:
        ring = ring_of(rate)
        d = np.linspace(0.0, ST.x_max, ST.samples)
        t = d / rate
        nr = np.array([SF.particles_minus_holes(ring, "R", float(x), PH.n_labels) for x in t])
        nl = np.array([SF.particles_minus_holes(ring, "L", float(x), PH.n_labels) for x in t])
        _cache[key] = (d, nr, -nl)
    return _cache[key]


def flow_time(u: float, P) -> float:
    """The time since the field was switched on (0 before)."""
    return max(u - P.field_on, 0.0)


def counters_at(ring: SF.Ring, t: float) -> dict:
    return SF.charges(ring, t, PH.n_labels)


def last_crossing_age(ring: SF.Ring, t: float) -> float:
    tc = SF.crossing_times(ring, t, PH.n_labels)
    return float(t - tc[-1]) if len(tc) else 1e9


# ------------------------------------------------------------------------------------------------------ the painter

def _hex(c: str) -> np.ndarray:
    return np.array([int(c[1:3], 16), int(c[3:5], 16), int(c[5:7], 16), 255], np.float32)


class Painter:
    """Draws the frames (one Figure that is reused); ``text_boxes`` reports where every text of the last frame is."""

    def __init__(self, lang: str, size: tuple[int, int]) -> None:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        self.plt = plt
        self.lang = lang
        self.tx = TEXT[lang]
        V = CFG.video
        self.W, self.H = size
        self.sc = self.H / V.reference_height
        self.fig = plt.figure(figsize=(self.W / V.dpi, self.H / V.dpi), dpi=V.dpi, facecolor=STY.background)
        mk = lambda box: self.fig.add_axes(box, facecolor="none")
        self.ax_main, self.ax_ring, self.ax_stair = mk(LY.main_axes), mk(LY.ring_axes), mk(LY.stair_axes)
        self.ax_plate, self.ax_tri, self.ax_gam = mk(LY.plate_axes), mk(LY.triangle_axes), mk(LY.gamma_axes)
        self.axes = [self.ax_main, self.ax_ring, self.ax_stair, self.ax_plate, self.ax_tri, self.ax_gam]
        self.part_fade = 1.0
        self.bg = _hex(STY.background)

    # ---- helpers
    def T(self, key: str, **kw) -> str:
        return fill(self.tx[key], self.lang, **kw)

    def nm(self, x: float, fmt: str) -> tuple[float, str]:
        return (x, fmt)

    def put(self, x: float, y: float, text: str, size: float, color, alpha: float = 1.0, ax=None, **kw):
        if alpha <= STY.panel_cutoff:
            return None
        tgt = ax if ax is not None else self.fig
        extra = {"transform": ax.transData} if ax is not None else {}
        return tgt.text(x, y, text, color=(*color, alpha), fontsize=size * self.sc, **extra, **kw)

    def sz(self, s: float) -> float:
        """A marker area given in pt^2 at 720 p -> pt^2 at the current frame height."""
        return s * self.sc * self.sc

    # ---- the frame
    def draw(self, t_film: float, total: float) -> np.ndarray:
        fig = self.fig
        TXT, DIM, GOLD = tuple(STY.text), tuple(STY.dim), tuple(STY.gold)
        t, card, cprog = timeline(t_film / (total / TOTAL))
        fig.texts.clear()
        for ar in self.axes:
            ar.clear()
            ar.axis("off")
        sc = self.sc
        if card is not None:
            a_c = min(smooth(cprog, *CFG.timeline.card_fade_in), 1.0 - smooth(cprog, *CFG.timeline.card_fade_out))
            tx = self.tx
            if card == 0:
                fig.text(0.5, LY.card_title_y, tx["h0"], color=(*STY.card_title_color, a_c), fontsize=FN.card_title * sc, ha="center", va="center", linespacing=1.15)
                fig.text(0.5, LY.card_subtitle_y, tx["h0s"], color=(*DIM, a_c), fontsize=FN.card_subtitle * sc, ha="center", va="center")
            else:
                fig.text(0.5, LY.card_number_y, f"{card}", color=(*GOLD, STY.card_number_alpha * a_c), fontsize=FN.card_number * sc, ha="center", va="center")
                fig.text(0.5, LY.card_chapter_y, tx[CARD_KEYS[card]], color=(*STY.card_title_color, a_c), fontsize=FN.card_chapter * sc, ha="center", va="center")
                fig.text(0.5, LY.card_chapter_sub_y, tx[CARD_KEYS[card] + "s"], color=(*DIM, a_c), fontsize=FN.card_chapter_sub * sc, ha="center", va="center")
        else:
            part = part_of(t)
            u = t - PARTS[part - 1][0]
            self.draw_common(part, u)
            [self.part1, self.part2, self.part3, self.part4, self.part5][part - 1](u)
            ga = smooth(u, *CFG.timeline.part_fade_in)
            if ga < 1.0:
                self.part_fade = ga
        return self.finish(t_film, total)

    def finish(self, t_film: float, total: float) -> np.ndarray:
        V = CFG.video
        self.fig.canvas.draw()
        frame = np.asarray(self.fig.canvas.buffer_rgba()).astype(np.float32)
        if self.part_fade < 1.0:
            k0 = int(round(CFG.timeline.part_fade_top * self.H))
            frame[k0:] = self.bg + (frame[k0:] - self.bg) * self.part_fade
            self.part_fade = 1.0
        fade_io = min(smooth(t_film, 0.0, V.fade_s), 1.0 - smooth(t_film, total - V.fade_s, total))
        if fade_io < 1.0:
            frame = self.bg + (frame - self.bg) * fade_io
        return frame.clip(0, 255).astype(np.uint8)

    def draw_common(self, part: int, u: float) -> None:
        """The title, the chapter label, the formulas and the caption of the part."""
        P = PARTCFG[part - 1]
        tx, sc, fig = self.tx, self.sc, self.fig
        TXT, DIM = tuple(STY.text), tuple(STY.dim)
        fig.text(*LY.title_pos, tx["title"], color=tuple(STY.title_color), fontsize=FN.title * sc)
        fig.text(*LY.chapter_label_pos, f"{part}/{len(PARTS)}   " + tx[CARD_KEYS[part]], color=(*DIM, STY.chapter_label_alpha), fontsize=FN.chapter_label * sc, ha="right")
        for f in P.formulas:
            a = smooth(u, *f["appear"]) * (1.0 - (smooth(u, *f["vanish"]) if "vanish" in f else 0.0))
            if a > STY.panel_cutoff:
                fig.text(f["x"], LY.formula_y - LY.formula_row * f["row"], self.formula_text(f["key"]), color=(*TXT, STY.formula_alpha * a), fontsize=FN.formula * sc)
        cap = pick(P.captions, u)
        if cap:
            fig.text(*LY.caption_pos, self.T(cap, **self.cap_numbers()), color=(*TXT, STY.caption_alpha), fontsize=FN.caption * sc, va="bottom", linespacing=1.25)

    def formula_text(self, key: str) -> str:
        return self.T(key, **self.cap_numbers())

    def cap_numbers(self) -> dict:
        return {"G": self.nm(self.gamma_th(), ".2f"), "Gm": self.nm(self.gamma_meas(), ".2f"), "dG": self.nm(PN.gamma_meas_err, ".2f"),
                "Gold": self.nm(self.gamma_old(), ".4f")}

    def gamma_th(self) -> float:
        return SF.pi0_width(PN.alpha, PN.m_pi_ev, PN.f_pi_ev)

    def gamma_meas(self) -> float:
        return PN.branching * PN.hbar_ev_s / PN.tau_s

    def gamma_old(self) -> float:
        return PN.hbar_ev_s / PN.tau_no_anomaly_s

    # ------------------------------------------------------------------------------------------ pieces: the ring
    def draw_ring(self, ring_state: dict, a: float, field_a: float, phase: float, nr: int, nl: int, ring_t: float) -> None:
        ax = self.ax_ring
        ax.set_xlim(-RG.lim, RG.lim)
        ax.set_ylim(-RG.lim, RG.lim)
        ax.set_aspect("equal", adjustable="datalim")
        if a <= STY.panel_cutoff:
            return
        WARM, COLD, GOLD, DIM = tuple(STY.warm), tuple(STY.cold), tuple(STY.gold), tuple(STY.dim)
        th = np.linspace(0.0, 2.0 * math.pi, RG.points)
        for r, col in ((RG.r_out, WARM), (RG.r_in, COLD)):
            ax.plot(r * np.cos(th), r * np.sin(th), color=(*col, RG.track_alpha * a), lw=RG.track_width * self.sc, solid_capstyle="round")
        # the filled sea: dim dots that run around with the speed of light (right movers counter-clockwise, left movers clockwise)
        for r, col, sgn in ((RG.r_out, WARM, +1.0), (RG.r_in, COLD, -1.0)):
            ang = sgn * phase + 2.0 * math.pi * np.arange(RG.sea_dots) / RG.sea_dots
            ax.scatter(r * np.cos(ang), r * np.sin(ang), s=self.sz(RG.sea_size), color=(*col, RG.sea_alpha * a), edgecolors="none", zorder=3)
        # the excitations: particles (right) and holes (left)
        if nr > 0:
            ang = phase + 2.0 * math.pi * (np.arange(nr) + RG.excitation_offset) / nr
            ax.scatter(RG.r_out * np.cos(ang), RG.r_out * np.sin(ang), s=self.sz(RG.particle_glow), color=(*WARM, RG.glow_alpha * a), edgecolors="none", zorder=4)
            ax.scatter(RG.r_out * np.cos(ang), RG.r_out * np.sin(ang), s=self.sz(RG.particle_size), color=(*WARM, a), edgecolors=(1, 1, 1, a), linewidths=RG.particle_edge * self.sc, zorder=5)
        if nl < 0:
            m = -nl
            ang = -phase + 2.0 * math.pi * (np.arange(m) + RG.excitation_offset) / m
            ax.scatter(RG.r_in * np.cos(ang), RG.r_in * np.sin(ang), s=self.sz(RG.particle_glow), color=(*COLD, RG.glow_alpha * a), edgecolors="none", zorder=4)
            ax.scatter(RG.r_in * np.cos(ang), RG.r_in * np.sin(ang), s=self.sz(RG.particle_size), facecolors=(*STY.hole_fill, a), edgecolors=(*COLD, a),
                       linewidths=RG.hole_edge * self.sc, zorder=5)
        # the electric field along the ring
        if field_a > STY.panel_cutoff:
            from matplotlib.patches import FancyArrowPatch
            for j in range(RG.field_arrows):
                a0 = math.radians(RG.field_start_deg) + 2.0 * math.pi * j / RG.field_arrows
                d = math.radians(RG.field_span_deg) / 2.0
                p0 = (RG.r_field * math.cos(a0 - d), RG.r_field * math.sin(a0 - d))
                p1 = (RG.r_field * math.cos(a0 + d), RG.r_field * math.sin(a0 + d))
                ax.add_patch(FancyArrowPatch(p0, p1, connectionstyle=f"arc3,rad={RG.field_rad}", arrowstyle="-|>", mutation_scale=RG.field_head * self.sc,
                                             color=(*GOLD, field_a * a), lw=RG.field_width * self.sc))
            fa = math.radians(RG.field_label_deg)
            ax.text(RG.field_label_r * math.cos(fa), RG.field_label_r * math.sin(fa), r"$E$", color=(*GOLD, field_a * a), fontsize=RG.field_label_size * self.sc, ha="center", va="center")
            ax.text(0.0, 0.0, self.T("flux"), color=(*DIM, field_a * a), fontsize=RG.flux_size * self.sc, ha="center", va="center")
        la = math.radians(RG.label_deg)
        ax.text((RG.r_out + RG.label_gap) * math.cos(la), (RG.r_out + RG.label_gap) * math.sin(la), r"$R$", color=(*WARM, a), fontsize=RG.label_size * self.sc, ha="center", va="center")
        ax.text((RG.r_in - RG.label_gap_in) * math.cos(la), (RG.r_in - RG.label_gap_in) * math.sin(la), r"$L$", color=(*COLD, a), fontsize=RG.label_size * self.sc, ha="center", va="center")
        ax.text(RG.note_pos[0], RG.note_pos[1], self.tx["schematic"], color=(*DIM, RG.note_alpha * a), fontsize=RG.note_size * self.sc, ha="center", va="center")

    # ------------------------------------------------------------------------------------------ pieces: the staircase
    def draw_stair(self, rate: float, t: float, a: float) -> None:
        ax = self.ax_stair
        ax.set_xlim(-ST.margin_x, ST.x_max)
        ax.set_ylim(-ST.margin_y, ST.y_max)
        if a <= STY.panel_cutoff:
            return
        WARM, COLD, DIM = tuple(STY.warm), tuple(STY.cold), tuple(STY.dim)
        d, nr, nlm = staircase(rate)
        now = rate * t
        sel = d <= now
        ax.plot([0, ST.x_max], [0, 0], color=(1, 1, 1, ST.axis_alpha * a), lw=ST.axis_width * self.sc)
        ax.plot([0, 0], [0, ST.y_max], color=(1, 1, 1, ST.axis_alpha * a), lw=ST.axis_width * self.sc)
        ax.plot([0, ST.x_max], [0, ST.x_max], color=(*DIM, ST.line_alpha * a), lw=ST.line_width * self.sc, ls=(0, tuple(ST.line_dash)))
        ax.step(d[sel], nr[sel], where="post", color=(*WARM, a), lw=ST.step_width * self.sc)
        ax.step(d[sel], nlm[sel], where="post", color=(*COLD, ST.minus_alpha * a), lw=ST.step_width * self.sc, ls=(0, tuple(ST.minus_dash)))
        if sel.any():
            ax.scatter([now], [nr[sel][-1]], s=self.sz(ST.dot_size), color=(*WARM, a), edgecolors="none", zorder=5)
        for v in range(1, int(ST.y_max) + 1, ST.tick_stride):
            ax.text(-ST.tick_gap, v, f"{v}", color=(*DIM, a), fontsize=ST.tick_size * self.sc, ha="right", va="center")
            ax.plot([-ST.tick_len, 0], [v, v], color=(1, 1, 1, ST.axis_alpha * a), lw=ST.axis_width * self.sc)
        ax.text(*ST.title_pos, self.tx["stair_title"], color=(*DIM, a), fontsize=ST.title_size * self.sc, ha="left", va="bottom", transform=ax.transAxes)
        ax.text(*ST.xlabel_pos, r"$t/T$", color=(*DIM, a), fontsize=ST.label_size * self.sc, ha="right", va="top", transform=ax.transAxes)
        ax.text(*ST.line_label_pos, self.tx["stair_line"], color=(*DIM, a), fontsize=ST.note_size * self.sc, ha="left", va="center")
        ax.text(*ST.r_label_pos, r"$N_R$", color=(*WARM, a), fontsize=ST.note_size * self.sc, ha="left", va="center")
        ax.text(*ST.l_label_pos, r"$-N_L$", color=(*COLD, a), fontsize=ST.note_size * self.sc, ha="left", va="center")

    # ------------------------------------------------------------------------------------------ pieces: the counters
    def draw_counters(self, vals: dict, a: float, age: float, rows: list[str], title: str | None = None) -> None:
        """Right column: the counters ``rows`` (keys of vals: NR, NL, Q, Q5), the colour pulses when a level has just crossed."""
        if a <= STY.panel_cutoff:
            return
        if title:
            self.fig.text(LY.counter_x, LY.counter_y + CFG.counters.title_dy, self.tx[title], color=(*tuple(STY.dim), a), fontsize=CFG.counters.title_size * self.sc, ha="left", va="center")
        pulse = max(0.0, 1.0 - age / CFG.counters.pulse_s)
        WARM, COLD, TXT, GOLD = tuple(STY.warm), tuple(STY.cold), tuple(STY.text), tuple(STY.gold)
        base = {"NR": WARM, "NL": COLD, "Q": TXT, "Q5": TXT}
        for i, key in enumerate(rows):
            col = np.array(base[key]) * (1 - pulse * CFG.counters.pulse_gain) + np.array(GOLD) * pulse * CFG.counters.pulse_gain
            txt = self.T("cnt_" + key, v=signed(vals[key]))
            size = CFG.counters.size * (1.0 + CFG.counters.pulse_grow * pulse) if key in ("NR", "NL") or vals[key] != 0 else CFG.counters.size
            self.fig.text(LY.counter_x, LY.counter_y - LY.counter_row * i, txt, color=(*tuple(col), a), fontsize=size * self.sc, ha="left", va="center")

    # ------------------------------------------------------------------------------------------ pieces: the levels
    def draw_levels(self, ring: SF.Ring, t: float, a: float = 1.0, *, sweep: float = 1.0, cut: float | None = None, label_cut: float | None = None,
                    edge_fade: bool = True, flashes: bool = True) -> dict:
        """The two branches with the levels (dots) of the ring at the time t; returns the positions of the floors for the label cut."""
        ax = self.ax_main
        ax.set_xlim(*MN.xlim)
        ax.set_ylim(*MN.ylim)
        WARM, COLD, GOLD, DIM = tuple(STY.warm), tuple(STY.cold), tuple(STY.gold), tuple(STY.dim)
        info = {}
        n = SF.labels(PH.n_labels)
        x0, x1 = MN.xlim
        y0, y1 = MN.ylim
        for br, col in (("R", WARM), ("L", COLD)):
            k, e_ = ring.k(n, t) / PH.spacing, ring.energy(br, n, t) / PH.spacing
            occ = ring.occupied(br, n)
            vis = (np.abs(k) <= x1 + 1.0) & (np.abs(e_) <= y1 + 0.5)
            alpha = np.full(len(n), a)
            if edge_fade:
                alpha = alpha * (1.0 - np.array([smooth(abs(v), *MN.fade_energy) for v in e_]))
            if cut is not None:
                alpha = alpha * np.where(np.abs(e_) <= cut, 1.0, MN.outside_alpha)
            if label_cut is not None:
                e0 = ring.energy(br, n, 0.0) / PH.spacing
                inside = np.abs(e0) <= label_cut
                vis = vis & inside
                if inside.any() and (occ & inside).any():
                    info["floor_" + br] = float(np.min(e_[occ & inside]))
                    info["floor_k_" + br] = float(k[occ & inside][np.argmin(e_[occ & inside])])
            if sweep < 1.0:
                alpha = alpha * np.clip((sweep * MN.sweep_reach - np.abs(k)) / MN.sweep_width, 0.0, 1.0)
            alpha = alpha * vis
            ex_fill = occ & (e_ > 0)                       # particles
            ex_hole = (~occ) & (e_ < 0)                    # holes
            sea = occ & (e_ <= 0)
            vac = (~occ) & (e_ >= 0)
            for m_, kind in ((ex_fill | ex_hole, "ex"), (sea, "sea"), (vac, "vac")):
                m = m_ & (alpha > STY.panel_cutoff)
                if not m.any():
                    continue
                rgba = np.column_stack([np.tile(col, (int(m.sum()), 1)), alpha[m]])
                if kind == "ex":
                    ax.scatter(k[m], e_[m], s=self.sz(MN.dot_size * MN.glow_scale), color=np.column_stack([np.tile(col, (int(m.sum()), 1)), alpha[m] * MN.glow_alpha]), edgecolors="none", zorder=4)
                    big = self.sz(MN.dot_size * MN.particle_scale)
                    fill_mask = ex_fill[m]
                    if fill_mask.any():
                        ax.scatter(k[m][fill_mask], e_[m][fill_mask], s=big, color=rgba[fill_mask], edgecolors=np.column_stack([np.ones((int(fill_mask.sum()), 3)), alpha[m][fill_mask]]),
                                   linewidths=MN.particle_edge * self.sc, zorder=6)
                    hole_mask = ~fill_mask
                    if hole_mask.any():
                        ax.scatter(k[m][hole_mask], e_[m][hole_mask], s=big, facecolors=np.column_stack([np.tile(STY.hole_fill, (int(hole_mask.sum()), 1)), alpha[m][hole_mask]]),
                                   edgecolors=rgba[hole_mask], linewidths=MN.hole_edge * self.sc, zorder=6)
                elif kind == "sea":
                    ax.scatter(k[m], e_[m], s=self.sz(MN.dot_size), color=rgba, edgecolors=np.column_stack([np.ones((int(m.sum()), 3)), alpha[m] * MN.edge_alpha]),
                               linewidths=MN.dot_edge * self.sc, zorder=5)
                else:
                    ax.scatter(k[m], e_[m], s=self.sz(MN.dot_size), facecolors="none", edgecolors=rgba, linewidths=MN.vacant_edge * self.sc, zorder=5)
        if flashes and a > STY.panel_cutoff:
            for tc in SF.crossing_times(ring, t, PH.n_labels):
                age = (t - tc) / MN.flash_s
                if 0.0 <= age < 1.0:
                    ax.scatter([0.0], [0.0], s=self.sz(MN.flash_from + (MN.flash_to - MN.flash_from) * age), facecolors="none",
                               edgecolors=(*GOLD, (1.0 - age) * a), linewidths=MN.flash_width * self.sc, zorder=7)
        return info

    def draw_axes_frame(self, a: float, label_a: float, sea_a: float, fermi_a: float = 1.0) -> None:
        """The two dispersion lines E = +p (right, warm) and E = -p (left, cold), the Fermi level E = 0 and the shaded sea."""
        ax = self.ax_main
        WARM, COLD, DIM = tuple(STY.warm), tuple(STY.cold), tuple(STY.dim)
        x0, x1 = MN.xlim
        y0, y1 = MN.ylim
        if sea_a > STY.panel_cutoff:
            ax.fill_between([x0, x1], [y0, y0], [0.0, 0.0], color=(*STY.sea, MN.sea_alpha * sea_a), lw=0, zorder=0)
        ax.plot([x0, x1], [0, 0], color=(1, 1, 1, MN.fermi_alpha * a * fermi_a), lw=MN.fermi_width * self.sc, ls=(0, tuple(MN.fermi_dash)), zorder=1)
        ax.plot([0, 0], [y0, y1], color=(1, 1, 1, MN.kaxis_alpha * a), lw=MN.fermi_width * self.sc, zorder=1)
        top = min(x1, y1)
        ax.plot([-y1, y1], [-y1, y1], color=(*WARM, MN.line_alpha * a), lw=MN.line_width * self.sc, zorder=2, solid_capstyle="round")
        ax.plot([-y1, y1], [y1, -y1], color=(*COLD, MN.line_alpha * a), lw=MN.line_width * self.sc, zorder=2, solid_capstyle="round")
        if label_a > STY.panel_cutoff:
            ax.text(*MN.r_label_pos, self.tx["lab_R"], color=(*WARM, label_a), fontsize=MN.label_size * self.sc, ha="left", va="center", linespacing=1.2)
            ax.text(*MN.l_label_pos, self.tx["lab_L"], color=(*COLD, label_a), fontsize=MN.label_size * self.sc, ha="right", va="center", linespacing=1.2)
            ax.text(x1 - MN.axis_label_gap[0], MN.axis_label_gap[1], r"$p$", color=(*DIM, label_a), fontsize=MN.axis_size * self.sc, ha="right", va="bottom")
            ax.text(MN.e_label_pos[0], y1 + MN.e_label_pos[1], r"$E$", color=(*DIM, label_a), fontsize=MN.axis_size * self.sc, ha="left", va="top")
            ax.text(x0 + MN.fermi_label_pos[0], MN.fermi_label_pos[1], r"$E=0$", color=(*DIM, label_a * fermi_a), fontsize=MN.fermi_size * self.sc, ha="left", va="bottom")

    # ------------------------------------------------------------------------------------------ part 1
    def part1(self, u: float) -> None:
        P = CFG.part1
        ring = ring_of(P.rate)
        a_ax = smooth(u, *P.axes_in)
        self.ax_main.set_xlim(*MN.xlim)
        self.ax_main.set_ylim(*MN.ylim)
        self.draw_axes_frame(a_ax, smooth(u, *P.labels_in), smooth(u, *P.sea_in))
        sweep = smooth(u, *P.dots_in)
        self.draw_levels(ring, 0.0, smooth(u, P.dots_in[0], P.dots_in[0] + 0.5), sweep=sweep)
        ax = self.ax_main
        DIM, TXT = tuple(STY.dim), tuple(STY.text)
        sea_a = smooth(u, *P.sea_text_in)
        if sea_a > STY.panel_cutoff:
            ax.text(*MN.sea_label_pos, self.tx["lab_sea"], color=(*DIM, sea_a), fontsize=MN.sea_label_size * self.sc, ha="center", va="center", linespacing=1.25)
        vac_a = smooth(u, *P.vac_text_in)
        if vac_a > STY.panel_cutoff:
            ax.text(*MN.vac_label_pos, self.tx["lab_vac"], color=(*DIM, vac_a), fontsize=MN.sea_label_size * self.sc, ha="center", va="center", linespacing=1.25)
        self.draw_legend(smooth(u, *P.legend_in))
        a_ring = smooth(u, *P.ring_in)
        self.draw_ring({}, a_ring, 0.0, RG.speed * u, 0, 0, 0.0)
        self.draw_counters(counters_at(ring, 0.0), smooth(u, *P.counters_in), 1e9, ["NR", "NL", "Q", "Q5"])

    def draw_legend(self, a: float) -> None:
        """The key to the dots, in the left column (the axes of the staircase are free in the part 1)."""
        if a <= STY.panel_cutoff:
            return
        ax = self.ax_stair
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        LG = CFG.legend
        DIM, TXT = tuple(STY.dim), tuple(STY.text)
        for i, (kind, key) in enumerate((("sea", "leg_filled"), ("vac", "leg_empty"), ("part", "leg_particle"), ("hole", "leg_hole"))):
            x, y = LG.pos[0], LG.pos[1] - LG.row * i
            if kind == "sea":
                ax.scatter([x], [y], s=self.sz(MN.dot_size), color=(*TXT, a), edgecolors="none")
            elif kind == "vac":
                ax.scatter([x], [y], s=self.sz(MN.dot_size), facecolors="none", edgecolors=(*TXT, a), linewidths=MN.vacant_edge * self.sc)
            elif kind == "part":
                ax.scatter([x], [y], s=self.sz(MN.dot_size * MN.glow_scale), color=(*TXT, a * MN.glow_alpha), edgecolors="none")
                ax.scatter([x], [y], s=self.sz(MN.dot_size * MN.particle_scale), color=(*TXT, a), edgecolors=(1, 1, 1, a), linewidths=MN.particle_edge * self.sc)
            else:
                ax.scatter([x], [y], s=self.sz(MN.dot_size * MN.glow_scale), color=(*TXT, a * MN.glow_alpha), edgecolors="none")
                ax.scatter([x], [y], s=self.sz(MN.dot_size * MN.particle_scale), facecolors=(*STY.hole_fill, a), edgecolors=(*TXT, a), linewidths=MN.hole_edge * self.sc)
            ax.text(x + LG.gap, y, self.tx[key], color=(*DIM, a), fontsize=LG.size * self.sc, ha="left", va="center")

    # ------------------------------------------------------------------------------------------ part 2
    def part2(self, u: float) -> None:
        P = CFG.part2
        ring = ring_of(P.rate)
        t = flow_time(u, P)
        self.draw_axes_frame(1.0, 1.0, 1.0)
        self.draw_levels(ring, t, 1.0)
        a_f = smooth(u, *P.field_in)
        nr = counters_at(ring, t)
        self.draw_ring({}, 1.0, a_f, RG.speed * u, nr["NR"], nr["NL"], t)
        self.draw_stair(P.rate, t, smooth(u, *P.stair_in))
        self.draw_counters(nr, 1.0, last_crossing_age(ring, t), ["NR", "NL", "Q", "Q5"])

    # ------------------------------------------------------------------------------------------ part 3
    def part3(self, u: float) -> None:
        P = CFG.part3
        seg = 0 if u < P.seg_bounds[0] else (1 if u < P.seg_bounds[1] else 2)
        if seg == 2:
            self.part3_lattice(u)
            return
        ring = ring_of(P.rate)
        if seg == 0:
            t = max(u - P.field_on_a, 0.0)
            t = min(t, P.t_max_a)
            a = 1.0 - smooth(u, *P.fade_a)
            a_cut = smooth(u, *P.cut_in)
            self.draw_axes_frame(1.0, 1.0, 1.0)
            x0, x1 = MN.xlim
            DIM, GOLD = tuple(STY.dim), tuple(STY.gold)
            for sgn, al in ((-1, 1.0), (+1, P.top_cut_alpha)):
                self.ax_main.plot([x0, x1], [sgn * P.cut, sgn * P.cut], color=(*GOLD, al * a_cut * a), lw=P.cut_width * self.sc, ls=(0, tuple(P.cut_dash)), zorder=3)
            self.ax_main.text(x0 + P.cut_label_pos[0], -P.cut + P.cut_label_pos[1], self.tx["lab_cut"], color=(*GOLD, a_cut * a), fontsize=P.cut_label_size * self.sc, ha="left", va="top")
            self.draw_levels(ring, t, a, cut=P.cut if a_cut > 0.5 else None, edge_fade=False)
            self.draw_flow_arrows(a * smooth(u, *P.arrows_in), ring, t)
            filled = SF.window_filled(ring, t, "energy", P.cut, nmax=PH.n_labels)
            start = SF.window_filled(ring, 0.0, "energy", P.cut, nmax=PH.n_labels)
            self.draw_ledger_a(filled, start, a * smooth(u, *P.ledger_in))
            nr = counters_at(ring, t)
            self.draw_counters(nr, a * smooth(u, *P.counters_in), last_crossing_age(ring, t), ["NR", "NL", "Q", "Q5"], "cnt_t_cut")
        else:
            t = max(u - P.field_on_b, 0.0)
            a = smooth(u, *P.fade_b_in) * (1.0 - smooth(u, *P.fade_b_out))
            self.draw_axes_frame(1.0, 1.0, 1.0)
            x0, x1 = MN.xlim
            GOLD, DIM = tuple(STY.gold), tuple(STY.dim)
            self.ax_main.plot([x0, x1], [-P.cut, -P.cut], color=(*GOLD, P.ghost_alpha * a), lw=P.cut_width * self.sc, ls=(0, tuple(P.cut_dash)), zorder=3)
            info = self.draw_levels(ring, t, a, label_cut=P.cut, edge_fade=False)
            for br, col in (("R", tuple(STY.warm)), ("L", tuple(STY.cold))):
                if "floor_" + br in info and a > STY.panel_cutoff:
                    fy, fk = info["floor_" + br], info["floor_k_" + br]
                    self.ax_main.plot([fk - P.floor_half, fk + P.floor_half], [fy - P.floor_gap, fy - P.floor_gap], color=(*col, a), lw=P.floor_width * self.sc, solid_capstyle="round", zorder=4)
            if a > STY.panel_cutoff:
                self.ax_main.text(x0 + P.cut_label_pos[0], -P.cut + P.cut_label_pos[1], self.tx["lab_cut"], color=(*GOLD, P.ghost_alpha * a), fontsize=P.cut_label_size * self.sc, ha="left", va="top")
            fill_ = SF.window_filled(ring, t, "label", P.cut, nmax=PH.n_labels)
            floor_r = info.get("floor_R", 0.0)
            floor_l = info.get("floor_L", 0.0)
            self.draw_ledger_b(fill_, floor_r, floor_l, a * smooth(u, *P.ledger_b_in), t)
            nr = counters_at(ring, t)
            # with the label cut the regularised axial charge does not change
            wc = SF.window_counts(ring, t, "label", P.cut, nmax=PH.n_labels)
            self.draw_counters({"NR": wc["R"], "NL": wc["L"], "Q": wc["Q"], "Q5": wc["Q5"]}, a * smooth(u, *P.counters_in), 1e9, ["NR", "NL", "Q", "Q5"], "cnt_t_label")

    def draw_flow_arrows(self, a: float, ring: SF.Ring, t: float) -> None:
        """Where the filled levels enter and leave: the texts at the corners of the diagram and the rings at the two gates of the cut-off."""
        if a <= STY.panel_cutoff:
            return
        P = CFG.part3
        WARM, COLD, GOLD = tuple(STY.warm), tuple(STY.cold), tuple(STY.gold)
        self.fig.text(*P.arr_in_pos, self.tx["arr_in"], color=(*WARM, a), fontsize=P.arr_size * self.sc, ha="right", va="bottom", linespacing=1.2)
        self.fig.text(*P.arr_out_pos, self.tx["arr_out"], color=(*COLD, a), fontsize=P.arr_size * self.sc, ha="left", va="bottom", linespacing=1.2)
        for tc in SF.crossing_times(ring, t, PH.n_labels):
            age = (t - tc) / MN.flash_s
            if 0.0 <= age < 1.0:
                for gx in (-P.cut, +P.cut):
                    self.ax_main.scatter([gx], [-P.cut], s=self.sz(P.gate_from + (P.gate_to - P.gate_from) * age), facecolors="none", edgecolors=(*GOLD, (1.0 - age) * a),
                                         linewidths=MN.flash_width * self.sc, zorder=7)

    def draw_ledger_a(self, filled: dict, start: dict, a: float) -> None:
        if a <= STY.panel_cutoff:
            return
        WARM, COLD, TXT, DIM = tuple(STY.warm), tuple(STY.cold), tuple(STY.text), tuple(STY.dim)
        L = CFG.ledger
        x, y = L.x, L.y
        self.fig.text(x, y, self.tx["led_title"], color=(*DIM, a), fontsize=L.title_size * self.sc, ha="left", va="center")
        rows = [("led_R", WARM, filled["R"], start["R"]), ("led_L", COLD, filled["L"], start["L"])]
        for i, (key, col, now, st0) in enumerate(rows):
            self.fig.text(x, y - L.row * (i + 1), self.T(key, a=str(st0), b=str(now)), color=(*col, a), fontsize=L.size * self.sc, ha="left", va="center")
        self.fig.text(x, y - L.row * 3, self.T("led_sum", s=str(filled["R"] + filled["L"])), color=(*TXT, a), fontsize=L.size * self.sc, ha="left", va="center")
        self.fig.text(x, y - L.row * 4, self.T("led_diff", d=signed((filled["R"] - filled["L"]) - (start["R"] - start["L"]))), color=(*TXT, a), fontsize=L.size * self.sc, ha="left", va="center")

    def draw_ledger_b(self, fill_: dict, floor_r: float, floor_l: float, a: float, t: float) -> None:
        if a <= STY.panel_cutoff:
            return
        WARM, COLD, TXT, DIM = tuple(STY.warm), tuple(STY.cold), tuple(STY.text), tuple(STY.dim)
        L = CFG.ledger
        x, y = L.x, L.y
        self.fig.text(x, y, self.tx["led_title_b"], color=(*DIM, a), fontsize=L.title_size * self.sc, ha="left", va="center")
        self.fig.text(x, y - L.row * 1, self.T("led_fixed", r=str(fill_["R"]), l=str(fill_["L"])), color=(*TXT, a), fontsize=L.size * self.sc, ha="left", va="center")
        self.fig.text(x, y - L.row * 2, self.T("led_floorR", f=self.nm(floor_r, ".1f")), color=(*WARM, a), fontsize=L.size * self.sc, ha="left", va="center")
        self.fig.text(x, y - L.row * 3, self.T("led_floorL", f=self.nm(floor_l, ".1f")), color=(*COLD, a), fontsize=L.size * self.sc, ha="left", va="center")

    def part3_lattice(self, u: float) -> None:
        P = CFG.part3
        LT = CFG.lattice
        ax = self.ax_main
        WARM, COLD, GOLD, DIM, TXT = tuple(STY.warm), tuple(STY.cold), tuple(STY.gold), tuple(STY.dim), tuple(STY.text)
        a = smooth(u, *P.fade_c_in)
        u3 = u - P.seg_bounds[1]
        zoom = smooth(u, *LT.zoom)
        half = LT.half_width * (1.0 - zoom) + LT.zoom_half_width * zoom
        ymax = LT.y_max * (1.0 - zoom) + LT.zoom_y_max * zoom
        ax.set_xlim(-half, half)
        ax.set_ylim(-ymax, ymax)
        flow = max(u - P.field_on_c, 0.0) * LT.rate
        flow = min(flow, LT.flow_max)
        # the band
        kk = np.linspace(-math.pi, math.pi, LT.curve_points)
        e_ = np.sin(kk)
        ax.fill_between([-half, half], [-ymax, -ymax], [0.0, 0.0], color=(*STY.sea, MN.sea_alpha * a), lw=0, zorder=0)
        ax.plot([-half, half], [0, 0], color=(1, 1, 1, MN.fermi_alpha * a), lw=MN.fermi_width * self.sc, ls=(0, tuple(MN.fermi_dash)), zorder=1)
        right = np.cos(kk) > 0
        ax.plot(np.where(right, kk, np.nan), np.where(right, e_, np.nan), color=(*WARM, MN.line_alpha * a), lw=MN.line_width * self.sc, zorder=2, solid_capstyle="round")
        ax.plot(np.where(~right & (kk > 0), kk, np.nan), np.where(~right & (kk > 0), e_, np.nan), color=(*COLD, MN.line_alpha * a), lw=MN.line_width * self.sc, zorder=2, solid_capstyle="round")
        ax.plot(np.where(~right & (kk < 0), kk, np.nan), np.where(~right & (kk < 0), e_, np.nan), color=(*COLD, MN.line_alpha * a), lw=MN.line_width * self.sc, zorder=2, solid_capstyle="round")
        # the levels
        k0 = SF.lattice_levels(LT.n_sites, LT.theta, 0.0)
        k = SF.lattice_levels(LT.n_sites, LT.theta, flow)
        filled = np.sin(k0) < 0.0
        e_k = np.sin(k)
        is_right = np.cos(k) > 0.0
        cols = np.where(is_right[:, None], np.array(WARM)[None, :], np.array(COLD)[None, :])
        part = filled & (e_k > 0)
        hole = (~filled) & (e_k < 0)
        sea = filled & (e_k <= 0)
        vac = (~filled) & (e_k >= 0)
        sizes = self.sz(LT.dot_size)
        if sea.any():
            ax.scatter(k[sea], e_k[sea], s=sizes, color=np.column_stack([cols[sea], np.full(int(sea.sum()), a)]), edgecolors=(1, 1, 1, a * MN.edge_alpha), linewidths=MN.dot_edge * self.sc, zorder=5)
        if vac.any():
            ax.scatter(k[vac], e_k[vac], s=sizes, facecolors="none", edgecolors=np.column_stack([cols[vac], np.full(int(vac.sum()), a)]), linewidths=MN.vacant_edge * self.sc, zorder=5)
        ex = part | hole
        if ex.any():
            ax.scatter(k[ex], e_k[ex], s=self.sz(LT.dot_size * MN.glow_scale), color=np.column_stack([cols[ex], np.full(int(ex.sum()), a * MN.glow_alpha)]), edgecolors="none", zorder=4)
        if part.any():
            ax.scatter(k[part], e_k[part], s=self.sz(LT.dot_size * MN.particle_scale), color=np.column_stack([cols[part], np.full(int(part.sum()), a)]),
                       edgecolors=(1, 1, 1, a), linewidths=MN.particle_edge * self.sc, zorder=6)
        if hole.any():
            ax.scatter(k[hole], e_k[hole], s=self.sz(LT.dot_size * MN.particle_scale), facecolors=(*STY.hole_fill, a), edgecolors=np.column_stack([cols[hole], np.full(int(hole.sum()), a)]),
                       linewidths=MN.hole_edge * self.sc, zorder=6)
        # the flashes at the two Fermi points
        ring_rate = LT.rate
        for j in range(int(flow + LT.theta) + 1):
            tc = (j + LT.theta) / ring_rate if j + LT.theta > 0 else 0.0
            age = (max(u - P.field_on_c, 0.0) - tc) / MN.flash_s
            if 0.0 <= age < 1.0 and j + LT.theta <= flow + 1e-9:
                for xx in (0.0, math.pi, -math.pi):
                    if abs(xx) <= half:
                        ax.scatter([xx], [0.0], s=self.sz(MN.flash_from + (LT.flash_to - MN.flash_from) * age), facecolors="none", edgecolors=(*GOLD, (1.0 - age) * a), linewidths=MN.flash_width * self.sc, zorder=7)
        # labels
        ax.text(LT.r_label_pos[0], LT.r_label_pos[1] * ymax, self.tx["lat_R"], color=(*WARM, a), fontsize=MN.label_size * self.sc, ha="center", va="center", linespacing=1.2)
        if half > math.pi - LT.edge_margin:
            ax.text(math.pi - LT.l_label_dx, LT.l_label_pos[1] * ymax, self.tx["lat_L"], color=(*COLD, a * (1.0 - zoom)), fontsize=MN.label_size * self.sc, ha="right", va="center", linespacing=1.2)
            ax.text(-math.pi + LT.l_label_dx, LT.l_label_pos[1] * ymax, self.tx["lat_L"], color=(*COLD, a * (1.0 - zoom)), fontsize=MN.label_size * self.sc, ha="left", va="center", linespacing=1.2)
        ax.text(half - LT.axis_dx, -LT.axis_dy * ymax, r"$ka$", color=(*DIM, a), fontsize=MN.axis_size * self.sc, ha="right", va="top")
        ax.text(-half + LT.axis_dx, ymax * LT.e_label_y, r"$E$", color=(*DIM, a), fontsize=MN.axis_size * self.sc, ha="left", va="center")
        if zoom < 0.5:
            wrap_a = a * (1.0 - zoom) * smooth(u, *LT.wrap_in)
            self.put(0.0, ymax * LT.wrap_y, self.tx["lat_wrap"], LT.wrap_size, DIM, wrap_a, ax=ax, ha="center", va="center")
        # counters of the lattice: the electrons are conserved
        c = SF.lattice_counts(LT.n_sites, LT.theta, flow)
        self.draw_counters({"NR": c["NR"], "NL": c["NL"], "Q": c["Q"], "Q5": c["Q5"]}, a * smooth(u, *LT.counters_in), 1e9, ["NR", "NL", "Q", "Q5"], "cnt_t_lat")
        L = CFG.ledger
        self.fig.text(L.x, L.y, self.tx["led_lat_title"], color=(*DIM, a * smooth(u, *LT.counters_in)), fontsize=L.title_size * self.sc, ha="left", va="center")
        self.fig.text(L.x, L.y - L.row, self.T("led_lat", n=str(c["filled"])), color=(*TXT, a * smooth(u, *LT.counters_in)), fontsize=L.size * self.sc, ha="left", va="center")
        self.fig.text(L.x, L.y - 2 * L.row, self.tx["led_lat_zoom"], color=(*DIM, a * zoom), fontsize=L.size * self.sc, ha="left", va="center")

    # ------------------------------------------------------------------------------------------ part 4
    def part4(self, u: float) -> None:
        P = CFG.part4
        LD = CFG.landau
        ring = ring_of(P.rate)
        t = flow_time(u, P)
        nr = counters_at(ring, t)
        self.ax_main.set_position(LY.landau_axes)
        ax = self.ax_main
        WARM, COLD, GOLD, DIM, TXT, GREY = tuple(STY.warm), tuple(STY.cold), tuple(STY.gold), tuple(STY.dim), tuple(STY.text), tuple(STY.grey)
        x0, x1 = LD.xlim
        y0, y1 = LD.ylim
        ax.set_xlim(x0, x1)
        ax.set_ylim(y0, y1)
        a_sp = smooth(u, *P.spectrum_in)
        a_hi = smooth(u, *P.higher_in)
        # axes, the Fermi level and the shaded sea
        ax.fill_between([x0, x1], [y0, y0], [0.0, 0.0], color=(*STY.sea, MN.sea_alpha * a_sp), lw=0, zorder=0)
        ax.plot([x0, x1], [0, 0], color=(1, 1, 1, MN.fermi_alpha * a_sp), lw=MN.fermi_width * self.sc, ls=(0, tuple(MN.fermi_dash)), zorder=1)
        ax.plot([0, 0], [y0, y1], color=(1, 1, 1, MN.kaxis_alpha * a_sp), lw=MN.fermi_width * self.sc, zorder=1)
        pz = np.linspace(x0, x1, LD.curve_points)
        for n_, sgn in ((1, 1), (1, -1), (2, 1), (2, -1)):
            ax.plot(pz, sgn * np.sqrt(pz ** 2 + n_ * LD.gap ** 2), color=(*GREY, LD.higher_alpha * a_hi), lw=LD.higher_width * self.sc, ls=(0, tuple(LD.higher_dash)), zorder=2)
        ax.plot([x0, x1], [x0, x1], color=(*WARM, MN.line_alpha * a_sp), lw=LD.lll_width * self.sc, zorder=3, solid_capstyle="round")
        ax.plot([x0, x1], [-x0, -x1], color=(*COLD, MN.line_alpha * a_sp), lw=LD.lll_width * self.sc, zorder=3, solid_capstyle="round")
        # the levels: the lowest Landau level (the 1+1 ladder) and the higher ones
        n = SF.labels(PH.n_labels)
        kz = ring.k(n, t) / PH.spacing
        occ_r = ring.occupied("R", n)
        for br, col, sgn in (("R", WARM, +1), ("L", COLD, -1)):
            e_ = sgn * kz
            occ = ring.occupied(br, n)
            vis = np.abs(kz) <= x1 + 0.5
            vis &= np.abs(e_) <= y1 + 0.3
            part = vis & occ & (e_ > 0)
            hole = vis & (~occ) & (e_ < 0)
            sea = vis & occ & (e_ <= 0)
            vac = vis & (~occ) & (e_ >= 0)
            al = a_sp
            if sea.any():
                ax.scatter(kz[sea], e_[sea], s=self.sz(LD.dot_size), color=(*col, al), edgecolors=(1, 1, 1, al * MN.edge_alpha), linewidths=MN.dot_edge * self.sc, zorder=5)
            if vac.any():
                ax.scatter(kz[vac], e_[vac], s=self.sz(LD.dot_size), facecolors="none", edgecolors=(*col, al), linewidths=MN.vacant_edge * self.sc, zorder=5)
            for m, kind in ((part, "p"), (hole, "h")):
                if m.any():
                    ax.scatter(kz[m], e_[m], s=self.sz(LD.dot_size * MN.glow_scale), color=(*col, al * MN.glow_alpha), edgecolors="none", zorder=4)
                    if kind == "p":
                        ax.scatter(kz[m], e_[m], s=self.sz(LD.dot_size * MN.particle_scale), color=(*col, al), edgecolors=(1, 1, 1, al), linewidths=MN.particle_edge * self.sc, zorder=6)
                    else:
                        ax.scatter(kz[m], e_[m], s=self.sz(LD.dot_size * MN.particle_scale), facecolors=(*STY.hole_fill, al), edgecolors=(*col, al), linewidths=MN.hole_edge * self.sc, zorder=6)
        # dots on the higher levels: they slide along the hyperbolas, no level crosses E = 0
        if a_hi > STY.panel_cutoff:
            for n_ in (1, 2):
                for sgn in (+1, -1):
                    e_h = sgn * np.sqrt(kz ** 2 + n_ * LD.gap ** 2)
                    vis = (np.abs(kz) <= x1) & (np.abs(e_h) <= y1)
                    filled = sgn < 0
                    if filled:
                        ax.scatter(kz[vis], e_h[vis], s=self.sz(LD.dot_size_high), color=(*GREY, LD.higher_dot_alpha * a_hi), edgecolors="none", zorder=4)
                    else:
                        ax.scatter(kz[vis], e_h[vis], s=self.sz(LD.dot_size_high), facecolors="none", edgecolors=(*GREY, LD.higher_dot_alpha * a_hi), linewidths=MN.vacant_edge * self.sc * LD.high_edge_scale, zorder=4)
        # flashes at the crossings
        for tc in SF.crossing_times(ring, t, PH.n_labels):
            age = (t - tc) / MN.flash_s
            if 0.0 <= age < 1.0:
                ax.scatter([0.0], [0.0], s=self.sz(MN.flash_from + (MN.flash_to - MN.flash_from) * age), facecolors="none", edgecolors=(*GOLD, (1.0 - age) * a_sp), linewidths=MN.flash_width * self.sc, zorder=7)
        # labels
        ax.text(*LD.lab_R_pos, self.tx["lab_R0"], color=(*WARM, a_sp), fontsize=MN.label_size * self.sc, ha="left", va="center", linespacing=1.2)
        ax.text(*LD.lab_L_pos, self.tx["lab_L0"], color=(*COLD, a_sp), fontsize=MN.label_size * self.sc, ha="right", va="center", linespacing=1.2)
        ax.text(*LD.lab_n1_pos, r"$n=1$", color=(*GREY, a_hi), fontsize=LD.n_label_size * self.sc, ha="left", va="center")
        ax.text(*LD.lab_n2_pos, r"$n=2$", color=(*GREY, a_hi), fontsize=LD.n_label_size * self.sc, ha="left", va="center")
        ax.text(x1 - MN.axis_label_gap[0], MN.axis_label_gap[1], r"$p_z$", color=(*DIM, a_sp), fontsize=MN.axis_size * self.sc, ha="right", va="bottom")
        ax.text(MN.e_label_pos[0], y1 + MN.e_label_pos[1], r"$E$", color=(*DIM, a_sp), fontsize=MN.axis_size * self.sc, ha="left", va="top")
        # the plate with the states of a Landau level
        self.draw_plate(u, t, nr)
        # the ledger
        L = CFG.ledger4
        a_l = smooth(u, *P.ledger_in)
        n_phi = LD.rows * LD.cols
        self.fig.text(L.x, L.y, self.tx["led4_title"], color=(*DIM, a_l), fontsize=L.title_size * self.sc, ha="left", va="center")
        self.fig.text(L.x, L.y - L.row, self.T("led4_one", n=str(nr["NR"])), color=(*WARM, a_l), fontsize=L.size * self.sc, ha="left", va="center")
        self.fig.text(L.x, L.y - 2 * L.row, self.T("led4_phi", m=str(n_phi)), color=(*TXT, a_l), fontsize=L.size * self.sc, ha="left", va="center")
        self.fig.text(L.x, L.y - 3 * L.row, self.T("led4_total", m=str(n_phi), n=str(nr["NR"]), t=str(n_phi * nr["NR"])), color=(*GOLD, a_l), fontsize=L.size * self.sc, ha="left", va="center")

    def draw_plate(self, u: float, t: float, nr: dict) -> None:
        P, LD = CFG.part4, CFG.landau
        ax = self.ax_plate
        a = smooth(u, *P.plate_in)
        if a <= STY.panel_cutoff:
            return
        from matplotlib.patches import Rectangle
        WARM, COLD, GOLD, DIM, TXT = tuple(STY.warm), tuple(STY.cold), tuple(STY.gold), tuple(STY.dim), tuple(STY.text)
        ax.set_xlim(-LD.plate_lim_x, LD.plate_lim_x)
        ax.set_ylim(-LD.plate_lim_y, LD.plate_lim_y)
        ax.set_aspect("equal", adjustable="datalim")
        w, h = LD.plate_w, LD.plate_h
        ax.add_patch(Rectangle((-w / 2, -h / 2), w, h, fill=True, facecolor=(*STY.sea, LD.plate_fill_alpha * a), edgecolor=(*DIM, a), lw=LD.plate_edge * self.sc))
        xs = (np.arange(LD.cols) + 0.5) / LD.cols * w - w / 2
        ys = (np.arange(LD.rows) + 0.5) / LD.rows * h - h / 2
        X, Y = np.meshgrid(xs, ys)
        age = last_crossing_age(ring_of(CFG.part4.rate), t)
        pulse = max(0.0, 1.0 - age / MN.flash_s) if t > 0 else 0.0
        ax.scatter(X.ravel(), Y.ravel(), s=self.sz(LD.state_ring), facecolors="none", edgecolors=(*COLD, a), linewidths=LD.state_ring_edge * self.sc, zorder=3)
        col = np.array(WARM) * (1 - pulse) + np.array(GOLD) * pulse
        ax.scatter(X.ravel(), Y.ravel(), s=self.sz(LD.state_dot * (1.0 + LD.pulse_grow * pulse)), color=(*tuple(col), a), edgecolors="none", zorder=4)
        ax.text(-w / 2, h / 2 + LD.plate_label_gap, self.tx["plate_title"], color=(*DIM, a), fontsize=LD.plate_label_size * self.sc, ha="left", va="bottom")
        ax.text(w / 2 + LD.b_dx, LD.b_y * h, r"$\odot\ \mathbf{B},\ \mathbf{E}$", color=(*GOLD, a), fontsize=LD.b_size * self.sc, ha="left", va="center")
        ax.text(0.0, -h / 2 - LD.plate_label_gap, r"$S$", color=(*DIM, a), fontsize=LD.plate_label_size * self.sc, ha="center", va="top")
        ax.text(0.0, -h / 2 - LD.nphi_gap, self.T("plate_nphi", m=str(LD.rows * LD.cols)), color=(*TXT, a), fontsize=LD.plate_label_size * self.sc, ha="center", va="top")

    # ------------------------------------------------------------------------------------------ part 5
    def part5(self, u: float) -> None:
        P = CFG.part5
        TR = CFG.triangle
        ax = self.ax_tri
        WARM, COLD, GOLD, DIM, TXT = tuple(STY.warm), tuple(STY.cold), tuple(STY.gold), tuple(STY.dim), tuple(STY.text)
        a_ans = smooth(u, *P.answer_in)
        a = smooth(u, *P.triangle_in) * (1.0 - a_ans)
        ax.set_xlim(*TR.xlim)
        ax.set_ylim(*TR.ylim)
        ax.set_aspect("equal", adjustable="datalim")
        if a > STY.panel_cutoff:
            from matplotlib.patches import FancyArrowPatch
            va, vb, vc = np.array(TR.v_axial), np.array(TR.v_up), np.array(TR.v_down)
            for p0, p1 in ((va, vb), (vb, vc), (vc, va)):
                ax.plot([p0[0], p1[0]], [p0[1], p1[1]], color=(*TXT, TR.line_alpha * a), lw=TR.line_width * self.sc, solid_capstyle="round", zorder=2)
                mid = 0.5 * (p0 + p1)
                d = (p1 - p0) / np.linalg.norm(p1 - p0)
                ax.add_patch(FancyArrowPatch(tuple(mid - 0.02 * d), tuple(mid + 0.02 * d), arrowstyle="-|>", mutation_scale=TR.head * self.sc, color=(*TXT, a), lw=0, zorder=3))
            # the axial vertex
            ax.scatter([va[0]], [va[1]], s=self.sz(TR.axial_size), marker="s", color=(*GOLD, a), zorder=4)
            ax.plot([va[0] - TR.axial_leg, va[0]], [va[1], va[1]], color=(*GOLD, a), lw=TR.line_width * self.sc, ls=(0, tuple(TR.axial_dash)), zorder=1)
            ax.text(va[0] - TR.axial_leg, va[1] + TR.label_dy, r"$j_5^\mu$", color=(*GOLD, a), fontsize=TR.label_size * self.sc, ha="left", va="bottom")
            ax.text(va[0] + TR.axial_text_dx, va[1] + TR.axial_text_dy, r"$\gamma^\mu\gamma_5$", color=(*GOLD, a), fontsize=TR.label_size * self.sc, ha="left", va="bottom")
            # the photons
            for v, sgn in ((vb, +1), (vc, -1)):
                xs = np.linspace(0.0, TR.photon_len, TR.photon_points)
                amp = TR.photon_amp * np.minimum(xs / TR.photon_ramp, 1.0)
                ang = math.radians(sgn * TR.photon_angle_deg)
                dirv = np.array([math.cos(ang), math.sin(ang)])
                nrm = np.array([-dirv[1], dirv[0]])
                pts = v[None, :] + xs[:, None] * dirv[None, :] + (amp * np.sin(2.0 * math.pi * TR.photon_waves * xs / TR.photon_len))[:, None] * nrm[None, :]
                ax.plot(pts[:, 0], pts[:, 1], color=(*COLD, a), lw=TR.photon_width * self.sc, zorder=2)
                end = v + TR.photon_len * dirv
                ax.text(end[0] + TR.photon_label_dx, end[1], r"$\gamma$", color=(*COLD, a), fontsize=TR.label_size * self.sc, ha="left", va="center")
            ax.text(*TR.loop_label_pos, self.tx["tri_loop"], color=(*DIM, a), fontsize=TR.note_size * self.sc, ha="center", va="top", linespacing=1.2)
        # the right-hand side: rows
        for r in P.rows:
            ar = smooth(u, *r["appear"]) * (1.0 - a_ans)
            if ar > STY.panel_cutoff:
                self.fig.text(r["x"], r["y"], self.T(r["key"], **self.cap_numbers()), color=(*TXT, STY.formula_alpha * ar), fontsize=FN.row * self.sc, va="center")
        self.draw_gamma(smooth(u, *P.gamma_in) * (1.0 - a_ans))
        a_txt = smooth(u, *P.answer_text_in)
        if a_txt > STY.panel_cutoff:
            self.fig.text(*LY.answer_q_pos, self.tx["answer_q"], color=(*GOLD, a_txt), fontsize=FN.answer_q * self.sc, ha="center", va="center")
            self.fig.text(*LY.answer_pos, self.tx["answer"], color=(*TXT, a_txt), fontsize=FN.answer * self.sc, ha="center", va="center", linespacing=P.answer_spacing)

    def draw_gamma(self, a: float) -> None:
        ax = self.ax_gam
        GM = CFG.gamma
        if a <= STY.panel_cutoff:
            return
        WARM, COLD, GOLD, DIM, TXT = tuple(STY.warm), tuple(STY.cold), tuple(STY.gold), tuple(STY.dim), tuple(STY.text)
        ax.set_xlim(math.log10(GM.xmin), math.log10(GM.xmax))
        ax.set_ylim(*GM.ylim)
        lg = lambda v: math.log10(v)
        ax.plot([lg(GM.xmin), lg(GM.xmax)], [0, 0], color=(1, 1, 1, GM.axis_alpha * a), lw=GM.axis_width * self.sc)
        for e in range(int(math.log10(GM.xmin)), int(math.log10(GM.xmax)) + 1):
            ax.plot([e, e], [0, -GM.tick_len], color=(1, 1, 1, GM.axis_alpha * a), lw=GM.axis_width * self.sc)
            ax.text(e, -GM.tick_len - GM.tick_gap, f"$10^{{{e}}}$", color=(*DIM, a), fontsize=GM.tick_size * self.sc, ha="center", va="top")
        ax.text(lg(GM.xmax), GM.unit_y, r"$\Gamma(\pi^0\to\gamma\gamma)$, eV", color=(*DIM, a), fontsize=GM.unit_size * self.sc, ha="right", va="bottom")
        pts = [(self.gamma_old(), DIM, "g_old", GM.old_y, "left"), (self.gamma_th(), GOLD, "g_th", GM.th_y, "right"), (self.gamma_meas(), TXT, "g_meas", GM.meas_y, "right")]
        for v, col, key, y, ha in pts:
            ax.scatter([lg(v)], [y], s=self.sz(GM.dot_size), color=(*col, a), edgecolors="none", zorder=4)
            ax.text(lg(v) + (GM.label_dx if ha == "left" else -GM.label_dx), y, self.T(key, **self.cap_numbers()), color=(*col, a), fontsize=GM.label_size * self.sc, ha=ha, va="center")
        # the error bar of the measurement
        lo, hi = self.gamma_meas() - PN.gamma_meas_err, self.gamma_meas() + PN.gamma_meas_err
        ax.plot([lg(lo), lg(hi)], [GM.meas_y, GM.meas_y], color=(*TXT, a), lw=GM.err_width * self.sc, zorder=3)

    def text_boxes(self, min_alpha: float = 0.15) -> list[tuple[str, tuple[float, float, float, float]]]:
        """(string, (x0, y0, x1, y1) in pixels) of every visible text of the last frame."""
        r = self.fig.canvas.get_renderer()
        out = []
        arts = list(self.fig.texts) + [t for ax in self.fig.axes for t in ax.texts]
        for t in arts:
            if not t.get_text().strip() or not t.get_visible():
                continue
            col = t.get_color()
            if isinstance(col, tuple) and len(col) == 4 and col[3] < min_alpha:
                continue
            bb = t.get_window_extent(r)
            out.append((t.get_text(), (bb.x0, bb.y0, bb.x1, bb.y1)))
        return out


PN = CFG.pion


def render(out: Path, size: tuple[int, int], fps: int, total: float, lang: str, snap: float | None = None) -> None:
    if snap is None and not shutil.which("ffmpeg"):
        sys.exit("ffmpeg is required")
    W, H = size
    V = CFG.video
    painter = Painter(lang, size)
    frames = int(round(total * fps))
    ids = [int(round(snap * fps))] if snap is not None else range(frames)
    writer = None
    if snap is None:
        writer = subprocess.Popen(
            ["ffmpeg", "-y", "-v", "error", "-f", "rawvideo", "-pix_fmt", "rgba", "-s", f"{W}x{H}",
             "-r", str(fps), "-i", "-", "-c:v", "libx264", "-preset", V.preset, "-crf", str(V.crf),
             "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(out)], stdin=subprocess.PIPE)
    for k in ids:
        frame = painter.draw(k / fps, total)
        if writer is None:
            from PIL import Image
            Image.fromarray(frame).save(out)
        else:
            writer.stdin.write(frame.tobytes())
    if writer is not None:
        writer.stdin.close()
        writer.wait()
    painter.plt.close(painter.fig)
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
    out = args.out or HERE / "media" / f"chiral_anomaly_{args.lang}.mp4"
    out.parent.mkdir(parents=True, exist_ok=True)
    if args.snapshot is not None:
        render(out.with_suffix(".png"), (V.width, V.height), V.fps, args.seconds, args.lang, snap=args.snapshot)
    elif args.preview:
        render(out.with_name(out.stem + "_preview.mp4"), (V.preview_width, V.preview_height), V.preview_fps, args.seconds, args.lang)
    else:
        render(out, (V.width, V.height), V.fps, args.seconds, args.lang)


if __name__ == "__main__":
    main()
