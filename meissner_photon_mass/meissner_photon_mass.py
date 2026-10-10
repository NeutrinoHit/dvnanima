r"""The Meissner effect and the photon mass (the book: "Where does a particle get its mass?", the Abelian Higgs model and superconductivity).

A fixed condensate |phi|^2 = v^2/2 gives the gauge field the mass m_A = e v (units hbar = c = 1).  Three parts, each preceded by a chapter card:

 1. Static, 2D.  A disc of the condensate in a uniform magnetic field.  The field is the exact equilibrium solution of
    (nabla^2 - m_A^2) A = 0 inside the disc and nabla^2 A = 0 outside (a superconducting cylinder in a transverse field); the condensate
    grows (v and hence m_A = e v increase) and the field lines are pushed out; along a cut the tangential field decays as exp(-x / lambda),
    lambda = 1 / m_A (the London penetration depth).  Every frame is the equilibrium field for the current v (quasi-static).
 2. Dynamic, 1D.  A wave packet of the gauge field meets the boundary of the condensate.  The equation d_t^2 A - d_x^2 A + m_A^2(x) A = 0,
    m_A(x) = m_A Theta(x), is integrated numerically (leapfrog, absorbing layers).  omega < m_A: the packet is reflected and an evanescent
    tail exp(-kappa x), kappa = sqrt(m_A^2 - omega^2), enters; omega > m_A: the packet is transmitted with k = sqrt(omega^2 - m_A^2) and the
    group velocity v_g = k / omega < 1.  The numbers on the screen are measured in the simulation and set against the formulas.
 3. The dispersion curve omega^2 = k^2 + m_A^2 against the light cone, the gap, v_g, and the polarizations: 2 + 2 = 1 + 3.

The condensate is a fixed background (no back-reaction).  All the numbers are in config.toml and all the words in texts.toml
(see ../dvconfig.py for --config / --set).

Usage:
    python meissner_photon_mass.py --lang en            # film -> media/meissner_photon_mass_en.mp4
    python meissner_photon_mass.py --lang ru
    python meissner_photon_mass.py --lang en --snapshot 20
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

import meissner_physics as mp  # noqa: E402

HERE = Path(__file__).resolve().parent
CFG = load_config(HERE)

CARD_D = list(CFG.timeline.card_durations)
CARD_AT = list(CFG.timeline.card_at)
CARD_KEYS = [f"h{i}" for i in range(len(CARD_AT))]
PB = list(CFG.timeline.part_bounds)
CONTENT_TOTAL = PB[-1]
TOTAL = CONTENT_TOTAL + sum(CARD_D)
T_P1, T_P2, T_P3 = ((PB[i], PB[i + 1]) for i in range(3))
SMOOTH = mp.smooth


def timeline(tf: float) -> tuple[float, int | None, float]:
    """(content time, index of the chapter card or None, progress of the card in [0, 1]) at the film time tf."""
    shown = 0.0
    for i, ca in enumerate(CARD_AT):
        a = ca + shown
        if a <= tf < a + CARD_D[i]:
            return ca, i, (tf - a) / CARD_D[i]
        if tf >= a + CARD_D[i]:
            shown += CARD_D[i]
    return tf - shown, None, 0.0


def film_time(tc: float) -> float:
    return tc + sum(d for ca, d in zip(CARD_AT, CARD_D) if ca <= tc)


def chapter_of(tc: float) -> int:
    return max(i for i, ca in enumerate(CARD_AT) if ca <= tc)


# ----------------------------------------------------------------------------------- texts

TEXT = {lang: load_texts(HERE, lang) for lang in ("en", "ru")}


def num(x: float, nd: int, lang: str, math_mode: bool = True) -> str:
    """A number with nd decimals; in Russian the decimal comma ({,} inside a formula)."""
    s = f"{x:.{nd}f}"
    if lang == "ru":
        s = s.replace(".", "{,}" if math_mode else ",")
    return s


def fill(text: str, lang: str, **vals: float | str) -> str:
    """Replace @name@ by the value (a number is given as (value, decimals) and formatted with the decimal comma in Russian)."""
    for k, v in vals.items():
        s = num(v[0], v[1], lang) if isinstance(v, tuple) else str(v)
        text = text.replace(f"@{k}@", s)
    return text


# ----------------------------------------------------------------------------------- the physics of the film

_cache: dict = {}


def packet_run(kind: str) -> mp.Run:
    if kind not in _cache:
        p = getattr(CFG.packets, kind)
        _cache[kind] = mp.simulate(p.omega, p.sigma, p.x0, p.amplitude, p.duration_s)
    return _cache[kind]


def sample_index(run: mp.Run, tau: float) -> int:
    return int(min(max(round(tau * CFG.physics.proca.samples_per_s), 0), len(run.t) - 1))


def part1_s(t: float) -> float:
    Cy = CFG.physics.cylinder
    return Cy.s_start + (Cy.s_final - Cy.s_start) * SMOOTH(t, *CFG.part1.ramp)


def fmt_ratio(x: float, lang: str) -> str:
    return num(x, 0 if x >= 10 else 1 if x >= 1 else 2, lang)


def packet_measurements(kind: str, tau: float) -> dict:
    """What is read off the simulation at the packet time tau (s): used for the numbers on the screen and by the tests."""
    M = CFG.measure
    rate = CFG.physics.proca.wave_rate
    run = packet_run(kind)
    om = run.omega
    i = sample_index(run, tau)
    out: dict = {"omega": om, "index": i}
    if kind == "below":
        kap = mp.kappa(om)
        out["depth_analytic"] = 1.0 / kap
        lo, hi = (v / kap for v in M.depth_fit)
        out["depth_sim"] = mp.measure_penetration(run, lo, hi, i) if tau >= M.depth_show_s else None
        e_tot = run.u_left[i] + run.u_right[i]
        out["reflected_analytic"] = 1.0
        out["reflected_sim"] = float(run.u_left[i] / e_tot) if tau >= M.energy_done_s else None
    else:
        k = mp.wavenumber_in_condensate(om)
        out["k_analytic"] = k
        out["vg_analytic"] = mp.group_velocity(om)
        frac = run.u_right / (run.u_left + run.u_right)
        inside = np.where(frac >= M.speed_inside_fraction)[0]
        i0 = int(inside[0]) if len(inside) else None
        out["vg_sim"] = None
        if i0 is not None and run.t[i] >= run.t[i0] + M.speed_min_s * rate:
            out["vg_sim"] = mp.measure_speed(run.t, run.xc_right, float(run.t[i0]), float(run.t[i]))
        w = M.speed_vacuum_window
        out["v_vacuum_sim"] = mp.measure_speed(run.t, run.xc_all, w[0] * rate, w[1] * rate)
        out["k_sim"] = mp.measure_wavenumber(run, *M.wavenumber_x, index=i) if tau >= M.wavenumber_s else None
        out["transmitted_analytic"] = mp.transmitted_energy_fraction(om)
        out["transmitted_sim"] = float(run.u_right[i] / (run.u_left[i] + run.u_right[i])) if tau >= M.energy_done_s else None
    return out


# ----------------------------------------------------------------------------------- the picture

class Canvas:
    """The figure of the film: all the axes are created once, every frame redraws the ones that are needed."""

    def __init__(self, size: tuple[int, int], lang: str) -> None:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.colors import LinearSegmentedColormap

        self.plt = plt
        self.lang = lang
        self.tx = TEXT[lang]
        S, L, V = CFG.style, CFG.layout, CFG.video
        self.W, self.H = size
        self.sc = self.H / V.reference_height
        self.BG = S.background
        self.TXT, self.DIM = tuple(S.text), tuple(S.dim)
        self.BLUE, self.WARM, self.GOLD, self.VIOLET = tuple(S.blue), tuple(S.warm), tuple(S.gold), tuple(S.violet)
        self.fig = plt.figure(figsize=(self.W / V.dpi, self.H / V.dpi), dpi=V.dpi, facecolor=self.BG)
        mk = lambda rect: self.fig.add_axes(rect, facecolor="none")  # noqa: E731
        self.ax_disc, self.ax_leg = mk(L.disc_axes), mk(CFG.part1.legend_rect)
        self.ax_prof, self.ax_bar = mk(L.profile_axes), mk(L.bar_axes)
        self.ax_stage, self.ax_disp, self.ax_zoom, self.ax_world = mk(L.stage_axes), mk(L.disp_axes), mk(L.zoom_axes), mk(L.worldline_axes)
        self.ax_disp3, self.ax_triad, self.ax_dof = mk(L.disp3_axes), mk(L.triad_axes), mk(L.dof_axes)
        self.axes = [self.ax_disc, self.ax_leg, self.ax_prof, self.ax_bar, self.ax_stage, self.ax_disp, self.ax_zoom, self.ax_world,
                     self.ax_disp3, self.ax_triad, self.ax_dof]
        P1 = CFG.part1
        self.cmap = LinearSegmentedColormap.from_list("bfield", [(c[0], tuple(c[1:])) for c in P1.cmap], N=P1.cmap_samples)
        # the field map of part 1
        axw, axh = self.W * L.disc_axes[2], self.H * L.disc_axes[3]
        R = CFG.physics.cylinder.radius
        self.yh = P1.y_half * R
        self.xh = self.yh * axw / axh
        ny = P1.grid_ny
        nx = int(round(ny * axw / axh))
        self.gx, self.gy = np.meshgrid(np.linspace(-self.xh, self.xh, nx), np.linspace(-self.yh, self.yh, ny))
        nl = int(self.yh // (P1.line_spacing * R))
        self.levels = np.arange(-nl, nl + 1) * P1.line_spacing * R              # symmetric about the line A = 0
        self.bg_rgb = tuple(int(self.BG[i:i + 2], 16) / 255.0 for i in (1, 3, 5))

    # ----------------------------------------------------------------- helpers
    def ftext(self, x: float, y: float, s: str, size: float, color, alpha: float = 1.0, **kw):
        return self.fig.text(x, y, s, color=(*color, alpha), fontsize=size * self.sc, **kw)

    def atext(self, ax, x: float, y: float, s: str, size: float, color, alpha: float = 1.0, **kw):
        kw.setdefault("transform", ax.transAxes)
        return ax.text(x, y, s, color=(*color, alpha), fontsize=size * self.sc, **kw)

    def edge_fade(self, ax, x0: float, x1: float, y0: float, y1: float, w: float, h: float, zorder: int) -> None:
        """Gradient strips of the background colour along the four borders: the picture fades into the frame (the limits of the axes are kept)."""
        ramp = np.linspace(1.0, 0.0, 64)
        strip = np.zeros((1, 64, 4))
        strip[..., :3] = self.bg_rgb
        strip[..., 3] = ramp
        ax.imshow(strip, extent=(x0, x0 + w, y0, y1), aspect="auto", zorder=zorder)
        ax.imshow(strip[:, ::-1], extent=(x1 - w, x1, y0, y1), aspect="auto", zorder=zorder)
        col = np.transpose(strip, (1, 0, 2))
        ax.imshow(col, extent=(x0, x1, y1 - h, y1), aspect="auto", zorder=zorder)           # origin "upper": row 0 (opaque) is at the top
        ax.imshow(col[::-1], extent=(x0, x1, y0, y0 + h), aspect="auto", zorder=zorder)
        ax.set_xlim(x0, x1)
        ax.set_ylim(y0, y1)

    def reset(self, ax, visible: bool = True) -> None:
        ax.clear()
        ax.set_facecolor("none")
        ax.axis("off")
        ax.set_visible(visible)

    def text_boxes(self) -> list[tuple[str, object]]:
        """(string, bounding box in pixels) of every visible text of the current frame (for the layout tests)."""
        from matplotlib.colors import to_rgba
        self.fig.canvas.draw()
        rend = self.fig.canvas.get_renderer()
        items = list(self.fig.texts)
        for ax in self.axes:
            if ax.get_visible():
                items += list(ax.texts)
        return [(t.get_text(), t.get_window_extent(rend)) for t in items if t.get_text().strip() and to_rgba(t.get_color())[3] >= 0.05]

    # ----------------------------------------------------------------- one frame
    def draw(self, t_film: float, total: float) -> None:
        S, L, F = CFG.style, CFG.layout, CFG.fonts
        tx, DIM, TXT, GOLD = self.tx, self.DIM, self.TXT, self.GOLD
        k = total / TOTAL
        t, card, cprog = timeline(t_film / k)
        self.fig.texts.clear()
        for ax in self.axes:
            self.reset(ax, visible=False)
        if card is not None:
            a_c = min(SMOOTH(cprog, *CFG.timeline.card_fade_in), 1.0 - SMOOTH(cprog, *CFG.timeline.card_fade_out))
            if card == 0:
                self.ftext(0.5, L.card_title_y, tx["h0"], F.card_title, S.card_title_color, a_c, ha="center", va="center")
                self.ftext(0.5, L.card_subtitle_y, tx["h0s"], F.card_subtitle, DIM, a_c, ha="center", va="center")
            else:
                self.ftext(0.5, L.card_number_y, f"{card}", F.card_number, GOLD, S.card_number_alpha * a_c, ha="center", va="center")
                self.ftext(0.5, L.card_chapter_y, tx[CARD_KEYS[card]], F.card_chapter, S.card_title_color, a_c, ha="center", va="center")
                self.ftext(0.5, L.card_chapter_sub_y, tx[CARD_KEYS[card] + "s"], F.card_chapter_sub, DIM, a_c, ha="center", va="center")
            return
        self.ftext(*L.title_pos, tx["title"], F.title, S.title_color[:3], S.title_color[3])
        ch = chapter_of(t)
        self.ftext(*L.chapter_label_pos, f"{ch}/{len(CARD_AT) - 1}   " + tx[CARD_KEYS[ch]], F.chapter_label, DIM, S.chapter_label_alpha, ha="right")
        if t < T_P1[1]:
            self.part1(t)
        elif t < T_P2[1]:
            self.part2(t)
        else:
            self.part3(t)

    def formula(self, key: str, row: int, alpha: float, **vals) -> None:
        L, F, S = CFG.layout, CFG.fonts, CFG.style
        if alpha > L.formula_min_alpha:
            self.ftext(L.formula_pos[0], L.formula_pos[1] - L.formula_row * row, fill(self.tx[key], self.lang, **vals), F.formula, self.TXT,
                       S.formula_alpha * alpha)

    def caption(self, key: str, **vals) -> None:
        self.ftext(*CFG.layout.caption_pos, fill(self.tx[key], self.lang, **vals), CFG.fonts.caption, self.TXT, CFG.style.caption_alpha)

    # ================================================================= part 1: the disc
    def part1(self, t: float) -> None:
        P, L, F, tx, lang = CFG.part1, CFG.layout, CFG.fonts, self.tx, self.lang
        Cy = CFG.physics.cylinder
        R, sc = Cy.radius, self.sc
        s = part1_s(t)
        m = s / R
        dens = (s / Cy.s_final) ** 2                       # |phi|^2 / |phi_final|^2 = (v / v_final)^2
        VIO, BLUE, GOLD, DIM, TXT = self.VIOLET, self.BLUE, self.GOLD, self.DIM, self.TXT
        # formulas and captions
        self.formula("f1a", 0, SMOOTH(t, *P.formula1_in))
        self.formula("f1b", 1, SMOOTH(t, *P.formula2_in))
        ct = P.caption_times
        cap = "c1a" if t < ct[0] else "c1b" if t < ct[1] else "c1c" if t < ct[2] else "c1d"
        self.caption(cap)
        # the field map
        ax = self.ax_disc
        self.reset(ax)
        ax.set_xlim(-self.xh, self.xh)
        ax.set_ylim(-self.yh, self.yh)
        bmag = mp.field_strength(self.gx, self.gy, m, R)
        ax.imshow(bmag, extent=(-self.xh, self.xh, -self.yh, self.yh), origin="lower", cmap=self.cmap, vmin=0.0, vmax=P.vmax,
                  interpolation="bilinear", aspect="auto", zorder=1)
        from matplotlib.patches import Circle
        ea = P.disc_edge_alpha[0] + (P.disc_edge_alpha[1] - P.disc_edge_alpha[0]) * dens
        ax.add_patch(Circle((0, 0), R, facecolor=(*VIO, P.disc_fill_alpha * dens), edgecolor=(*VIO, ea), linewidth=P.disc_edge_width * sc,
                            linestyle=(0, tuple(P.disc_edge_dash)) if dens < P.disc_dash_until else "-", zorder=2))
        a_z = mp.flux_function(self.gx, self.gy, m, R)
        a_z = np.ma.masked_where(np.hypot(self.gx, self.gy) < R - P.mask_depth / m, a_z)       # no lines deep inside: B = 0 there
        ax.contour(self.gx, self.gy, a_z, levels=self.levels, colors=[(1.0, 1.0, 1.0, P.line_alpha)], linewidths=P.line_width * sc, negative_linestyles="solid", zorder=3)
        # the cut through the top of the disc
        ax.plot([0, 0], [R + P.cut_range[0] * R, R - P.cut_range[1] * R], color=(*GOLD, P.cut_alpha * SMOOTH(t, *P.profile_in)), lw=P.cut_width * sc,
                ls=(0, tuple(P.cut_dash)), zorder=4)
        a_lab = SMOOTH(dens, P.disc_label_alpha_from, P.disc_label_alpha_from + 0.35)
        if a_lab > 0.01:
            ax.text(L.disc_label_pos[0] * R, L.disc_label_pos[1] * R + 0.16 * R, tx["condensate"], color=(*VIO, a_lab), fontsize=F.label * sc, ha="center", va="center", zorder=5)
            ax.text(L.disc_label_pos[0] * R, L.disc_label_pos[1] * R - 0.16 * R, tx["phi2"], color=(*TXT, a_lab), fontsize=F.label * sc, ha="center", va="center", zorder=5)
        # the screening current j = -m_A^2 A_z: out of the page (dot) where A_z < 0, into the page (cross) where A_z > 0
        a_j = SMOOTH(t, *P.current_in)
        if a_j > 0.01:
            ang = (np.arange(P.current_count) + 0.5) * 2.0 * math.pi / P.current_count
            rs = R - P.current_offset * R
            wgt = np.abs(np.sin(ang))
            pos, neg = np.sin(ang) < 0, np.sin(ang) > 0
            for sel, col, marker in ((pos, self.WARM, "o"), (neg, TXT, "x")):
                ax.scatter(rs * np.cos(ang[sel]), rs * np.sin(ang[sel]), s=(P.current_size * sc) ** 2 * (P.current_min + (1.0 - P.current_min) * wgt[sel]) ** 2,
                           facecolors="none", edgecolors=[(*col, a_j)], linewidths=1.2 * sc, marker="o", zorder=5)
                ax.scatter(rs * np.cos(ang[sel]), rs * np.sin(ang[sel]), s=(P.current_size * sc) ** 2 * (P.current_min + (1.0 - P.current_min) * wgt[sel]) ** 2 * (0.12 if marker == "o" else 0.5),
                           c=[(*col, a_j)], marker=marker, linewidths=1.0 * sc, zorder=5)
            ax.text(self.xh - P.current_label_offset[0] * R, -self.yh + P.current_label_offset[1] * R, tx["current"], color=(*TXT, a_j), fontsize=F.label * sc, ha="right", va="bottom", zorder=9)
        # B_0 arrows
        x_a = -self.xh + P.arrow_x0 * R
        for y in P.arrow_y:
            ax.annotate("", xy=(x_a + P.arrow_length * R, y * R), xytext=(x_a, y * R),
                        arrowprops=dict(arrowstyle="-|>", color=(*TXT, 0.95), lw=P.arrow_width * sc, shrinkA=0, shrinkB=0), zorder=6)
        ax.text(x_a + L.b0_label_offset[0] * R, P.arrow_y[1] * R + L.b0_label_offset[1] * R, tx["B0"], color=(*TXT, 0.95), fontsize=F.label * sc, va="bottom", zorder=6)
        self.edge_fade(ax, -self.xh, self.xh, -self.yh, self.yh, P.edge_fade[0] * R, P.edge_fade[1] * R, 8)
        # the colour legend of |B|
        al = self.ax_leg
        self.reset(al)
        al.imshow(np.linspace(0, 1, 128)[None, :], extent=(0, P.vmax, 0, 1), aspect="auto", cmap=self.cmap, vmin=0, vmax=1)
        al.set_xlim(0, P.vmax)
        for tk in P.legend_ticks:
            al.text(tk, -0.9, f"${num(tk, 0, lang) if tk != 1 else ''}B_0$" if tk > 0 else "$0$", color=(*DIM, 1), fontsize=F.tick * sc, ha="center", va="top")
        al.text(-0.05, 0.5, tx["Babs"], color=(*DIM, 1), fontsize=F.tick * sc, ha="right", va="center")
        # the readouts
        a_r = SMOOTH(t, *P.readout_in)
        self.ftext(*L.bar_label_pos, tx["growth"], F.label, DIM, a_r)
        ab = self.ax_bar
        self.reset(ab)
        ab.set_xlim(0, 1)
        ab.set_ylim(0, 1)
        from matplotlib.patches import Rectangle
        ab.add_patch(Rectangle((0, 0), 1, 1, facecolor="none", edgecolor=(*DIM, 0.8 * a_r), lw=1.0 * sc))
        ab.add_patch(Rectangle((0, 0), s / Cy.s_final, 1, facecolor=(*VIO, P.bar_fill_alpha * a_r), edgecolor="none"))
        lam = 1.0 / s
        self.ftext(L.readout_pos[0], L.readout_pos[1], rf"$m_A R={fmt_ratio(s, lang)}$", F.readout, TXT, a_r)
        self.ftext(L.readout_pos[0], L.readout_pos[1] - L.readout_row, rf"$\lambda_L=1/m_A={fmt_ratio(lam, lang)}\,R$", F.readout, TXT, a_r)
        # the profile of the tangential field along the cut
        a_p = SMOOTH(t, *P.profile_in)
        ap = self.ax_prof
        self.reset(ap)
        d = np.linspace(P.profile_xlim[0], P.profile_xlim[1], P.profile_samples)
        prof = mp.cut_profile(d * R, m, R)
        ap.set_xlim(*P.profile_xlim)
        ap.set_ylim(*P.profile_ylim)
        ap.axvspan(0.0, P.profile_xlim[1], color=(*VIO, P.profile_shade_alpha * a_p), lw=0)
        ap.plot([0, 0], P.profile_ylim, color=(*VIO, 0.8 * a_p), lw=1.0 * sc)
        ap.plot(P.profile_xlim, [1.0, 1.0], color=(*DIM, 0.6 * a_p), lw=1.0 * sc, ls=(0, (2, 3)))
        ap.plot(d, prof, color=(*BLUE, a_p), lw=P.profile_line_width * sc, solid_capstyle="round", zorder=4)
        bs = float(mp.cut_profile(np.array([0.0]), m, R)[0])
        dd = d[d >= 0.0]
        a_l = SMOOTH(t, P.lambda_mark_from, P.lambda_mark_from + 1.5) * a_p
        ap.plot(dd, bs * np.exp(-m * R * dd), color=(*GOLD, a_l), lw=1.4 * sc, ls=(0, tuple(P.profile_dash)), zorder=3)
        if lam / R < P.profile_xlim[1] * 0.9:
            ap.plot([lam / R, lam / R], [0.0, bs / math.e], color=(*GOLD, P.lambda_mark_alpha * a_l), lw=1.0 * sc, ls=(0, (1, 2)))
            ap.text(lam / R, bs / math.e + 0.07, r"$\lambda_L$", color=(*GOLD, a_l), fontsize=F.label * sc, ha="left", va="bottom")
        self.atext(ap, -0.03, 1.0 / P.profile_ylim[1], tx["B0"], F.panel_label, DIM, a_p, ha="right", va="center")
        self.atext(ap, *L.profile_title_pos, tx["prof_title"], F.panel_title, DIM, a_p, ha="left", va="bottom")
        self.atext(ap, 0.02, 0.04, tx["vacuum"], F.panel_label, DIM, a_p, ha="left", va="bottom")
        self.atext(ap, 0.98, 0.04, tx["condensate"], F.panel_label, VIO, a_p, ha="right", va="bottom")
        self.atext(ap, *L.profile_xlabel_pos, tx["depth"], F.panel_label, DIM, a_p, ha="right", va="top")

    # ================================================================= part 2: the packet at the boundary
    def part2(self, t: float) -> None:
        P = CFG.part2
        if t < P.b_start:
            self.packet_scene("below", t - P.a_start)
        else:
            self.packet_scene("above", t - P.b_start)

    def packet_scene(self, kind: str, tau: float) -> None:
        P, L, F, tx, lang, sc = CFG.part2, CFG.layout, CFG.fonts, self.tx, self.lang, self.sc
        VIO, BLUE, WARM, GOLD, DIM, TXT = self.VIOLET, self.BLUE, self.WARM, self.GOLD, self.DIM, self.TXT
        run = packet_run(kind)
        i = sample_index(run, tau)
        meas = packet_measurements(kind, tau)
        om = run.omega
        below = kind == "below"
        cap_t = P.caption_a if below else P.caption_b
        if below:
            cap = "c2a" if tau < cap_t[1] else "c2b" if tau < cap_t[2] else "c2c"
        else:
            cap = "c2d" if tau < cap_t[1] else "c2e"
        self.caption(cap, omega=(om, 2), vg=(mp.group_velocity(om), 2))
        self.formula("f2a", 0, SMOOTH(tau, *P.formula_a_in))
        self.formula("f2b" if below else "f2c", 1, SMOOTH(tau, *P.formula_b_in))
        # ---- the stage
        ax = self.ax_stage
        self.reset(ax)
        x0, x1 = P.stage_xlim
        y0, y1 = P.stage_ylim
        ax.set_xlim(x0, x1)
        ax.set_ylim(y0, y1)
        ax.axvspan(0.0, x1, color=(*VIO, P.condensate_alpha), lw=0, zorder=1)
        ax.plot([0, 0], [y0, y1], color=(*VIO, P.boundary_alpha), lw=P.boundary_width * sc, ls=(0, tuple(P.boundary_dash)), zorder=2)
        sel = (run.x >= x0 - 1.0) & (run.x <= x1 + 1.0)
        xs, ys = run.x[sel], run.a[i][sel]
        ax.fill_between(xs, 0, ys, where=ys >= 0, color=(*WARM, P.fill_alpha), lw=0, interpolate=True, zorder=3)
        ax.fill_between(xs, 0, ys, where=ys < 0, color=(*BLUE, P.fill_alpha), lw=0, interpolate=True, zorder=3)
        ax.plot(xs, ys, color=(0.92, 0.96, 1.0, 0.95), lw=P.line_width * sc, zorder=4)
        ax.plot([x0, x1], [0, 0], color=(*DIM, 0.35), lw=0.8 * sc, zorder=2)
        grad = np.zeros((1, 64, 4))
        grad[..., :3] = self.bg_rgb
        grad[..., 3] = np.linspace(1.0, 0.0, 64)
        ax.imshow(grad, extent=(x0, x0 + P.fade_edge, y0, y1), aspect="auto", zorder=6)
        ax.set_xlim(x0, x1)
        self.atext(ax, *L.stage_label_pos, tx["vacuum_m0"], F.panel_label, DIM, 1.0, ha="left", va="top")
        self.atext(ax, *L.stage_label2_pos, tx["condensate_line"], F.panel_label, VIO, 1.0, ha="left", va="top")
        self.atext(ax, 0.995, 0.06, tx["stage_axis"], F.panel_label, DIM, 1.0, ha="right", va="bottom")
        if not below:
            xc = float(run.xc_all[i])
            ax.scatter([xc], [P.marker_y], s=P.marker_size * sc * sc, color=GOLD, zorder=7, edgecolors="none")
        # ---- the dispersion panel
        self.dispersion_panel(self.ax_disp, om, 1.0, small=True)
        # ---- the lower middle panel
        if below:
            self.zoom_panel(i, om)
        else:
            self.world_panel(run, i, tau, meas)
        # ---- the table of numbers
        self.table(kind, tau, meas)

    def dispersion_panel(self, ax, omega: float, alpha: float, small: bool = True) -> None:
        P, L, F, tx, sc = CFG.part2, CFG.layout, CFG.fonts, self.tx, self.sc
        VIO, GOLD, DIM, TXT = self.VIOLET, self.GOLD, self.DIM, self.TXT
        self.reset(ax)
        kl, wl = P.disp_klim, P.disp_wlim
        ax.set_xlim(*kl)
        ax.set_ylim(*wl)
        k = np.linspace(kl[0], kl[1], CFG.part3.curve_samples)
        ax.axhspan(0, mp.M_A, color=(*VIO, CFG.part3.gap_alpha * alpha), lw=0)
        ax.plot(k, k, color=(*TXT, CFG.part3.cone_alpha * alpha), lw=CFG.part3.cone_width * sc, ls=(0, tuple(CFG.part3.cone_dash)))
        ax.plot(k, mp.omega_of_k(k), color=(*VIO, alpha), lw=2.0 * sc)
        ax.plot(kl, [omega, omega], color=(*GOLD, 0.9 * alpha), lw=1.2 * sc, ls=(0, (1, 2)))
        ax.plot([kl[0], kl[1]], [0, 0], color=(*DIM, 0.4), lw=0.8 * sc)
        ax.scatter([omega], [omega], s=40 * sc * sc, color=TXT, zorder=5)               # the wave in vacuum: k = omega
        if omega > mp.M_A:
            ax.scatter([mp.wavenumber_in_condensate(omega)], [omega], s=40 * sc * sc, color=VIO, zorder=5)
        else:
            ax.text(0.97, (omega + 0.07) / wl[1], tx["no_wave"], color=(*self.WARM, alpha), fontsize=F.panel_label * sc, ha="right", va="bottom", transform=ax.transAxes)
        ax.text(0.015, (omega + 0.05) / wl[1], r"$\omega$", color=(*GOLD, alpha), fontsize=F.panel_label * sc, ha="left", va="bottom", transform=ax.transAxes)
        self.atext(ax, 0.0, L.panel_title_y, tx["disp_title"], F.panel_title, DIM, alpha, ha="left", va="bottom")
        self.atext(ax, 0.97, 0.04, tx["k_axis"], F.panel_label, DIM, alpha, ha="right", va="bottom")
        self.atext(ax, 0.03, 0.05, tx["gap"], F.panel_label, VIO, alpha, ha="left", va="bottom")

    def zoom_panel(self, i: int, om: float) -> None:
        P, L, F, tx, sc = CFG.part2, CFG.layout, CFG.fonts, self.tx, self.sc
        VIO, BLUE, WARM, GOLD, DIM = self.VIOLET, self.BLUE, self.WARM, self.GOLD, self.DIM
        run = packet_run("below")
        ax = self.ax_zoom
        self.reset(ax)
        x0, x1 = P.zoom_xlim
        ax.set_xlim(x0, x1)
        ax.set_ylim(*P.zoom_ylim)
        ax.axvspan(0.0, x1, color=(*VIO, P.condensate_alpha), lw=0)
        ax.plot([0, 0], P.zoom_ylim, color=(*VIO, P.boundary_alpha), lw=P.boundary_width * sc, ls=(0, tuple(P.boundary_dash)))
        sel = (run.x >= x0) & (run.x <= x1)
        xs, ys = run.x[sel], run.a[i][sel]
        ax.fill_between(xs, 0, ys, where=ys >= 0, color=(*WARM, P.fill_alpha), lw=0, interpolate=True, zorder=3)
        ax.fill_between(xs, 0, ys, where=ys < 0, color=(*BLUE, P.fill_alpha), lw=0, interpolate=True, zorder=3)
        ax.plot(xs, ys, color=(0.92, 0.96, 1.0, 0.95), lw=P.line_width * sc, zorder=4)
        ax.plot([x0, x1], [0, 0], color=(*DIM, 0.35), lw=0.8 * sc)
        kap = mp.kappa(om)
        xe = np.linspace(0.0, x1, P.envelope_samples)
        amp = 2.0 * om / mp.M_A * CFG.packets.below.amplitude              # |A(0)| of a monochromatic wave: |tau| a_0, tau = 2 omega / m_A
        a_env = SMOOTH(float(np.abs(run.a[i][sel]).max()), 0.05, 0.3)
        for sgn in (1.0, -1.0):
            ax.plot(xe, sgn * amp * np.exp(-kap * xe), color=(*GOLD, P.envelope_alpha * a_env), lw=P.envelope_width * sc, ls=(0, tuple(P.envelope_dash)), zorder=5)
        self.atext(ax, 0.0, L.panel_title_y, tx["zoom_title"], F.panel_title, DIM, 1.0, ha="left", va="bottom")
        self.atext(ax, 0.62, 0.9, tx["tail_label"], F.panel_label, GOLD, a_env, ha="left", va="top")

    def world_panel(self, run: mp.Run, i: int, tau: float, meas: dict) -> None:
        P, L, F, tx, sc = CFG.part2, CFG.layout, CFG.fonts, self.tx, self.sc
        VIO, GOLD, DIM, TXT = self.VIOLET, self.GOLD, self.DIM, self.TXT
        ax = self.ax_world
        self.reset(ax)
        ax.set_xlim(*P.worldline_xlim)
        ax.set_ylim(*P.worldline_ylim)
        rate = CFG.physics.proca.wave_rate
        ax.axhspan(0.0, P.worldline_ylim[1], color=(*VIO, P.condensate_alpha), lw=0)
        ax.plot(P.worldline_xlim, [0, 0], color=(*VIO, P.boundary_alpha), lw=P.boundary_width * sc, ls=(0, tuple(P.boundary_dash)))
        ts = run.t[: i + 1] / rate
        ax.plot(ts, run.xc_all[: i + 1], color=GOLD, lw=P.worldline_width * sc, zorder=4)
        # the references: the light cone from the start and the group velocity in the condensate from the crossing
        t_ref = np.array(P.worldline_xlim)
        ax.plot(t_ref, run.xc_all[0] + rate * t_ref, color=(*TXT, P.ref_alpha), lw=P.ref_width * sc, ls=(0, tuple(P.ref_dash)), zorder=3)
        cross = np.where(run.xc_all >= 0.0)[0]
        if len(cross):
            tc = run.t[cross[0]] / rate
            ax.plot(t_ref, mp.group_velocity(run.omega) * rate * (t_ref - tc), color=(*VIO, 0.9), lw=P.ref_width * sc, ls=(0, tuple(P.ref_dash)), zorder=3)
        self.atext(ax, 0.0, L.panel_title_y, tx["world_title"], F.panel_title, DIM, 1.0, ha="left", va="bottom")
        self.atext(ax, 0.97, 0.04, tx["t_axis"], F.panel_label, DIM, 1.0, ha="right", va="bottom")
        self.atext(ax, 0.03, 0.6, tx["c_line"], F.panel_label, TXT, 1.0, ha="left", va="center")
        self.atext(ax, *L.vg_line_pos, tx["vg_line"], F.panel_label, VIO, 1.0, ha="left", va="center")

    def table(self, kind: str, tau: float, meas: dict) -> None:
        P, L, F, tx, lang = CFG.part2, CFG.layout, CFG.fonts, self.tx, self.lang
        DIM, TXT, GOLD = self.DIM, self.TXT, self.GOLD
        a = SMOOTH(tau, *P.table_in)
        x_lab, x_an, x_sim = L.table_cols
        y = L.table_pos[1]
        row = L.table_row
        self.ftext(x_an, y, tx["analytic"], F.table, DIM, a, ha="right")
        self.ftext(x_sim, y, tx["simulation"], F.table, DIM, a, ha="right")
        om = meas["omega"]
        rows: list[tuple[str, str, str | None]] = [(rf"$\omega={num(om, 2, lang)}\,m_A$", "", None)]
        if kind == "below":
            rows.append((tx["row_depth"], num(meas["depth_analytic"], 2, lang), num(meas["depth_sim"], 2, lang) if meas["depth_sim"] is not None else None))
            rows.append((tx["row_reflected"], num(100.0 * meas["reflected_analytic"], 0, lang) + r"\,\%", (num(100.0 * meas["reflected_sim"], 1, lang) + r"\,\%") if meas["reflected_sim"] is not None else None))
        else:
            rows.append((tx["row_k"], num(meas["k_analytic"], 2, lang), num(meas["k_sim"], 2, lang) if meas["k_sim"] is not None else None))
            rows.append((tx["row_vg"], num(meas["vg_analytic"], 3, lang), num(meas["vg_sim"], 3, lang) if meas["vg_sim"] is not None else None))
            rows.append((tx["row_transmitted"], num(100.0 * meas["transmitted_analytic"], 1, lang) + r"\,\%", (num(100.0 * meas["transmitted_sim"], 1, lang) + r"\,\%") if meas["transmitted_sim"] is not None else None))
        for j, (lab, an, sim) in enumerate(rows):
            yy = y - row * (j + 1)
            self.ftext(x_lab, yy, lab, F.table, TXT, a)
            if an:
                self.ftext(x_an, yy, f"${an}$" if "%" not in an else f"${an}$", F.table, TXT, a, ha="right")
            if sim is not None:
                self.ftext(x_sim, yy, f"${sim}$", F.table, GOLD, a, ha="right")

    # ================================================================= part 3: dispersion and polarizations
    def part3(self, t: float) -> None:
        P, L, F, tx, lang, sc = CFG.part3, CFG.layout, CFG.fonts, self.tx, self.lang, self.sc
        VIO, BLUE, GOLD, DIM, TXT, WARM = self.VIOLET, self.BLUE, self.GOLD, self.DIM, self.TXT, self.WARM
        tau = t - P.a_start
        ct = P.caption_times
        cap = "c3a" if tau < ct[0] else "c3b" if tau < ct[1] else "c3c" if tau < ct[2] else "c3d" if tau < ct[3] else "c3e" if tau < ct[4] else "c3f"
        self.caption(cap)
        self.formula("f3a", 0, SMOOTH(tau, *P.hyper_in))
        self.formula("f3b", 1, SMOOTH(tau, *P.slider_in))
        # ---- the dispersion
        ax = self.ax_disp3
        self.reset(ax)
        ax.set_xlim(*P.k_lim)
        ax.set_ylim(*P.w_lim)
        k = np.linspace(P.k_lim[0], P.k_lim[1], P.curve_samples)
        a_c, a_h, a_g, a_s = SMOOTH(tau, *P.cone_in), SMOOTH(tau, *P.hyper_in), SMOOTH(tau, *P.gap_in), SMOOTH(tau, *P.slider_in)
        ax.axhspan(0, mp.M_A, color=(*VIO, P.gap_alpha * a_g), lw=0)
        ax.plot(k, k, color=(*TXT, P.cone_alpha * a_c), lw=P.cone_width * sc, ls=(0, tuple(P.cone_dash)))
        kk = k[: max(int(a_h * len(k)), 2)]
        ax.plot(kk, mp.omega_of_k(kk), color=(*VIO, a_h if a_h > 0 else 0), lw=P.hyper_width * sc, solid_capstyle="round")
        for kind in ("below", "above"):
            w = getattr(CFG.packets, kind).omega
            ax.plot(P.k_lim, [w, w], color=(*GOLD, P.packet_line_alpha * a_g), lw=P.packet_line_width * sc, ls=(0, tuple(P.packet_line_dash)))
        ax.plot(P.k_lim, [0, 0], color=(*DIM, 0.4), lw=0.8 * sc)
        ax.plot([0, 0], P.w_lim, color=(*DIM, 0.4), lw=0.8 * sc)
        self.atext(ax, 0.0, L.panel_title_y, tx["disp3_title"], F.panel_title, DIM, 1.0, ha="left", va="bottom")
        self.atext(ax, 1.0, -0.015, tx["k_axis"], F.panel_label, DIM, 1.0, ha="right", va="top")
        self.atext(ax, *L.cone_label_pos, tx["cone_label"], F.panel_label, TXT, a_c, ha="left", va="center")
        self.atext(ax, *L.hyper_label_pos, tx["hyper_label"], F.panel_label, VIO, a_h, ha="left", va="center")
        self.atext(ax, 0.02, 0.05, tx["gap"], F.panel_label, VIO, a_g, ha="left", va="bottom")
        for kind, key in (("below", "packet1"), ("above", "packet2")):
            w = getattr(CFG.packets, kind).omega
            self.atext(ax, 0.985, w / P.w_lim[1] + 0.012, tx[key], F.dof_label, GOLD, P.packet_line_alpha * a_g * 1.4, ha="right", va="bottom")
        ax.scatter([0.0], [mp.M_A], s=P.point_size * 0.5 * sc * sc, color=GOLD, zorder=6, alpha=a_g)
        ax.text(0.08, mp.M_A + 0.05, tx["mA_label"], color=(*GOLD, a_g), fontsize=F.panel_label * sc, ha="left", va="bottom")
        # the slider: a point on the hyperbola, its tangent (v_g) and the chord (v_p)
        if a_s > 0.01:
            ph = (tau - P.slider_in[0]) / P.slider_period_s
            ks = P.slider_k[0] + (P.slider_k[1] - P.slider_k[0]) * 0.5 * (1.0 - math.cos(2.0 * math.pi * ph))
            ws = float(mp.omega_of_k(ks))
            vg = ks / ws
            h = P.tangent_half
            ax.plot([ks - h, ks + h], [ws - vg * h, ws + vg * h], color=(*GOLD, a_s), lw=P.tangent_width * sc, solid_capstyle="round", zorder=5)
            ax.plot([0, ks], [0, ws], color=(*DIM, P.chord_alpha * a_s), lw=1.0 * sc, ls=(0, (2, 2)), zorder=4)
            ax.scatter([ks], [ws], s=P.point_size * sc * sc, color=GOLD, zorder=6, alpha=a_s)
            self.atext(ax, *L.slider_text_pos, rf"$v_g=k/\omega={num(vg, 2, lang)}$", F.readout, GOLD, a_s, ha="left", va="center")
        # ---- the polarizations
        self.triad(tau)
        self.dof(tau)

    def triad(self, tau: float) -> None:
        P, L, F, tx, sc = CFG.part3, CFG.layout, CFG.fonts, self.tx, self.sc
        BLUE, GOLD, DIM, TXT = self.BLUE, self.GOLD, self.DIM, self.TXT
        ax = self.ax_triad
        self.reset(ax)
        axw, axh = self.W * L.triad_axes[2], self.H * L.triad_axes[3]
        xl = P.triad_xlim
        yh = (xl[1] - xl[0]) * axh / axw                    # equal scale in x and y
        ax.set_xlim(*xl)
        ax.set_ylim(0.0, yh)
        a2, a3 = SMOOTH(tau, *P.pol_in), SMOOTH(tau, *P.pol3_in)
        if a2 < 0.01:
            return
        osc = lambda ph: math.cos(2.0 * math.pi * (tau / P.osc_period_s + ph))  # noqa: E731
        sy, ky = P.triad_symbol_y * yh, P.triad_k_y * yh
        amp = P.osc_amp
        from matplotlib.patches import Circle
        for j, (key, col, al) in enumerate((("e1", BLUE, a2), ("e2", BLUE, a2), ("L", GOLD, a3))):
            if al < 0.01:
                continue
            cx = P.triad_columns[j]
            c = osc(P.triad_phase[j])
            ax.annotate("", xy=(cx + P.triad_k_half, ky), xytext=(cx - P.triad_k_half, ky),
                        arrowprops=dict(arrowstyle="-|>", color=(*DIM, 0.85 * al), lw=1.3 * sc, shrinkA=0, shrinkB=0))
            ax.text(cx + P.triad_k_half, ky + 0.06 * yh, r"$\mathbf{k}$", color=(*TXT, al), fontsize=F.label * sc, ha="right", va="bottom")
            if key == "e1":
                ax.plot([cx, cx], [sy - amp, sy + amp], color=(*col, P.axis_alpha * al), lw=P.axis_width * sc)
                ax.plot([cx, cx], [sy, sy + amp * c], color=(*col, al), lw=P.vector_width * sc, solid_capstyle="round")
                ax.scatter([cx], [sy + amp * c], s=P.dot_size * sc * sc, color=col, alpha=al, zorder=5)
            elif key == "L":
                ax.plot([cx - amp, cx + amp], [sy, sy], color=(*col, P.axis_alpha * al), lw=P.axis_width * sc)
                ax.plot([cx, cx + amp * c], [sy, sy], color=(*col, al), lw=P.vector_width * sc, solid_capstyle="round")
                ax.scatter([cx + amp * c], [sy], s=P.dot_size * sc * sc, color=col, alpha=al, zorder=5)
            else:                                           # out of the plane: a dot (towards us) or a cross (away)
                ax.add_patch(Circle((cx, sy), amp, facecolor="none", edgecolor=(*col, P.axis_alpha * al), lw=P.axis_width * sc))
                r = amp * abs(c)
                if c >= 0:
                    ax.add_patch(Circle((cx, sy), r, facecolor=(*col, al), edgecolor="none"))
                else:
                    ax.plot([cx - r * P.cross_size, cx + r * P.cross_size], [sy - r * P.cross_size, sy + r * P.cross_size], color=(*col, al), lw=P.vector_width * sc)
                    ax.plot([cx - r * P.cross_size, cx + r * P.cross_size], [sy + r * P.cross_size, sy - r * P.cross_size], color=(*col, al), lw=P.vector_width * sc)
            lab = {"e1": r"$\epsilon_1$", "e2": r"$\epsilon_2$", "L": r"$\epsilon_L$"}[key]
            ax.text(cx, P.triad_label_y * yh, lab, color=(*col, al), fontsize=F.label * sc, ha="center", va="center")
            ax.text(cx, P.triad_word_y * yh, tx["pol_long"] if key == "L" else tx["pol_trans"], color=(*DIM, al), fontsize=F.dof_label * sc, ha="center", va="center")
        title = tx["triad_massive"] if a3 > 0.5 else tx["triad_massless"]
        self.atext(ax, *L.triad_title_pos, title, F.panel_title, GOLD if a3 > 0.5 else DIM, 1.0, ha="center", va="bottom")

    def dof(self, tau: float) -> None:
        P, L, F, tx, sc = CFG.part3, CFG.layout, CFG.fonts, self.tx, self.sc
        BLUE, GOLD, DIM, TXT, VIO = self.BLUE, self.GOLD, self.DIM, self.TXT, self.VIOLET
        a = SMOOTH(tau, *P.dof_in)
        if a < 0.01:
            return
        ax = self.ax_dof
        self.reset(ax)
        ax.set_xlim(*P.dof_xlim)
        ax.set_ylim(*P.dof_ylim)
        from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
        nums = ["2", "2", "1", "3"]
        labs = ["dof_phi", "dof_A0", "dof_chi", "dof_A3"]
        cols = [VIO, BLUE, VIO, BLUE]
        for j in range(4):
            bx = P.box_x[j]
            ax.add_patch(FancyBboxPatch((bx, P.box_y), P.box_w, P.box_h, boxstyle="round,pad=0.0,rounding_size=0.03", facecolor=(*cols[j], 0.10 * a),
                                        edgecolor=(*cols[j], P.box_alpha * a), lw=P.box_width * sc))
            ax.text(bx + P.box_w / 2, P.box_y + P.box_h / 2, nums[j], color=(*TXT, a), fontsize=F.dof_number * sc, ha="center", va="center")
            ax.text(bx + P.box_w / 2, L.dof_label_y, tx[labs[j]], color=(*cols[j], a), fontsize=F.dof_label * sc, ha="center", va="center")
        for sx, sg in zip(P.sign_x, ("+", "=", "+")):
            ax.text(sx, P.box_y + P.box_h / 2, sg, color=(*TXT, a), fontsize=F.dof_number * sc, ha="center", va="center")
        aa = SMOOTH(tau, *P.arrow_in)
        if aa > 0.01:
            p0 = (P.box_x[0] + P.box_w / 2, P.box_y + P.box_h)
            p1 = (P.box_x[3] + P.box_w / 2, P.box_y + P.box_h)
            ax.add_patch(FancyArrowPatch(p0, p1, connectionstyle=f"arc3,rad={P.arrow_rad}", arrowstyle="-|>", mutation_scale=14 * sc, color=(*GOLD, aa), lw=2.0 * sc))
            ax.text(0.5, P.goldstone_y, tx["goldstone"], color=(*GOLD, aa), fontsize=F.panel_label * sc, ha="center", va="bottom")
        self.atext(ax, *L.dof_title_pos, tx["dof_title"], F.panel_title, DIM, a, ha="left", va="bottom")


# ----------------------------------------------------------------------------------- the film

def render(out: Path, size: tuple[int, int], fps: int, total: float, lang: str, snap: float | None = None) -> None:
    V = CFG.video
    if snap is None and not shutil.which("ffmpeg"):
        sys.exit("ffmpeg is required")
    cv = Canvas(size, lang)
    W, H = size
    frames = int(round(total * fps))
    ids = [int(round(snap * fps))] if snap is not None else range(frames)
    writer = None
    if snap is None:
        writer = subprocess.Popen(
            ["ffmpeg", "-y", "-v", "error", "-f", "rawvideo", "-pix_fmt", "rgba", "-s", f"{W}x{H}",
             "-r", str(fps), "-i", "-", "-c:v", "libx264", "-preset", V.preset, "-crf", str(V.crf),
             "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(out)], stdin=subprocess.PIPE)
    bg = np.array([int(cv.BG[1:3], 16), int(cv.BG[3:5], 16), int(cv.BG[5:7], 16), 255], np.float32)
    for k_ in ids:
        t_film = k_ / fps
        cv.draw(t_film, total)
        fade_io = min(SMOOTH(t_film, 0.0, V.fade_s), 1.0 - SMOOTH(t_film, total - V.fade_s, total))
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
    out = args.out or HERE / "media" / f"meissner_photon_mass_{args.lang}.mp4"
    out.parent.mkdir(parents=True, exist_ok=True)
    if args.snapshot is not None:
        render(out.with_suffix(".png"), (V.width, V.height), V.fps, args.seconds, args.lang, snap=args.snapshot)
    elif args.preview:
        render(out.with_name(out.stem + "_preview.mp4"), (V.preview_width, V.preview_height), V.preview_fps, args.seconds, args.lang)
    else:
        render(out, (V.width, V.height), V.fps, args.seconds, args.lang)


if __name__ == "__main__":
    main()
