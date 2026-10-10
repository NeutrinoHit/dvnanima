r"""Oscillations of neutral kaons (the book: "The Mysteries of Neutral Kaons", the section "Strangeness oscillations" and "Regeneration").

Notation of the book: K0 = |d sbar> (S = +1), K0bar = |dbar s> (S = -1); CP is neglected at first, so the states of definite mass and
lifetime are the CP eigenstates  K_S = K_1 = (K0 - K0bar) / sqrt 2  and  K_L = K_2 = (K0 + K0bar) / sqrt 2,  Delta m = m_L - m_S.

Parts of the film:
 1. a K0 is born in the strong reaction pi- p -> Lambda K0 (strangeness is conserved): in the plane (K0, K0bar) the state arrow is split into K_S and K_L,
    the lifetimes tau_S, tau_L and the mass difference Delta m;
 2. the oscillation: first for stable kaons (full oscillation over the period 2 pi / Delta m), then with the real widths; two phasors (the K_S part and the K_L part
    of the amplitude) add to the amplitude of K0 and subtract to the amplitude of K0bar; the time axis is then rescaled from tau_S to tau_L:
    K_S is gone and the K_L beam is a 50:50 mixture of K0 and K0bar.  Everything is a numerical integration of i d psi / dt = (M - i Gamma / 2) psi;
 3. what the detector sees: thin targets along the beam (K0bar N -> Lambda pi is allowed, K0 N -> Lambda pi is forbidden by strangeness) and the charge of the lepton
    in K -> l nu pi (the asymmetry A_l(t) = cos(Delta m t) / cosh(Delta Gamma t / 2));
 4. regeneration: a thin absorber in a K_L beam absorbs K0bar more than K0 and a K_S component appears behind it (an illustrative absorber).

All the numbers are in config.toml and all the words in texts.toml (see ../dvconfig.py for --config / --set).

Usage:
    python kaon_oscillations.py --lang en            # film -> media/kaon_oscillations_en.mp4
    python kaon_oscillations.py --lang ru
    python kaon_oscillations.py --lang en --preview
    python kaon_oscillations.py --lang en --snapshot 50   # one PNG at film time 50 s
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

import kaon_physics as kp  # noqa: E402

HERE = Path(__file__).resolve().parent
CFG = kp.CFG

TL = CFG.timeline
CARD_S = list(TL.card_s)
CARD_AT = list(TL.card_at)
PB = list(TL.part_bounds)
CONTENT_TOTAL = PB[-1]
TOTAL = CONTENT_TOTAL + sum(CARD_S)
CARD_KEYS = ["h0", "ch2", "ch3", "ch4"]
PART_KEYS = ["ch1", "ch2", "ch3", "ch4"]

TAU_L = kp.TAU_L
DM = kp.DM


def timeline(tf: float) -> tuple[float, int | None, float]:
    """(content time, index of the card or None, progress of the card in [0, 1]) at the film time tf."""
    shown = 0.0
    for i, ca in enumerate(CARD_AT):
        a = ca + shown
        if a <= tf < a + CARD_S[i]:
            return ca, i, (tf - a) / CARD_S[i]
        if tf >= a + CARD_S[i]:
            shown += CARD_S[i]
    return tf - shown, None, 0.0


def film_time(tc: float) -> float:
    """Film time of the content time tc (the cards before it are counted)."""
    return tc + sum(s for s, ca in zip(CARD_S, CARD_AT) if ca <= tc)


def part_of(tc: float) -> int:
    return max(i for i in range(len(PB) - 1) if PB[i] <= tc)


def smooth(x: float, a: float, b: float) -> float:
    u = min(max((x - a) / (b - a), 0.0), 1.0)
    return u * u * (3 - 2 * u)


def fade(t: float, pair) -> float:
    return smooth(t, pair[0], pair[1])


def lerp(a: float, b: float, u: float) -> float:
    return a + (b - a) * u


# ----------------------------------------------------------------------------------- texts

def num(x: float, fmt: str, lang: str) -> str:
    s = format(x, fmt)
    return s.replace(".", "{,}") if lang == "ru" else s


def plain(x: float, fmt: str, lang: str) -> str:
    """A number outside a formula: the decimal comma in Russian."""
    s = format(x, fmt)
    return s.replace(".", ",") if lang == "ru" else s


def sci(x: float, digits: int, lang: str) -> str:
    """a x 10^b inside a formula, with `digits` significant digits."""
    m, e = f"{x:.{digits - 1}e}".split("e")
    m = m.replace(".", "{,}") if lang == "ru" else m
    return rf"{m}\times10^{{{int(e)}}}"


def values(lang: str) -> dict[str, str]:
    P = CFG.physics
    ct_s = P.c_m_per_s * P.tau_s_s * 100.0           # cm
    ct_l = P.c_m_per_s * P.tau_l_s                   # m
    return {
        "tau_s": sci(P.tau_s_s, 3, lang), "tau_l": sci(P.tau_l_s, 3, lang), "dm_ev": sci(P.delta_m_ev, 2, lang),
        "dm_hbar": sci(P.delta_m_ev / P.hbar_ev_s, 2, lang), "ratio": f"{P.tau_l_s / P.tau_s_s:.0f}",
        "period": num(kp.PERIOD, ".1f", lang), "dmtau": num(kp.DM, ".2f", lang),
        "ct_s": num(ct_s, ".1f", lang), "ct_l": num(ct_l, ".0f", lang), "ct_l_cm": sci(ct_l, 2, lang),
        "f": num(P.f_k0, ".1f", lang), "fb": num(P.f_k0bar, ".1f", lang),
        "ptau_s": sci(P.tau_s_s, 2, lang), "ptau_l": sci(P.tau_l_s, 2, lang),
        "slab_t": num(P.slab_time, ".0f", lang), "cp": sci(P.cp_violation, 1, lang),
    }


TEXT = {lang: load_texts(HERE, lang) for lang in ("en", "ru")}


def texts(lang: str) -> dict[str, str]:
    vals = values(lang)
    out = {}
    for k, v in TEXT[lang].items():
        for name, rep in vals.items():
            v = v.replace(f"@{name}@", rep)
        out[k] = v
    return out


# ----------------------------------------------------------------------------------- sampling of the physics

def tgrid(hi: float, lo: float = 0.0) -> np.ndarray:
    C = CFG.curves
    parts = [np.linspace(lo, hi, C.samples_lin)]
    if lo == 0.0 and hi > 4 * C.geo_start:
        parts.append(np.geomspace(C.geo_start, hi, C.samples_geo))
    return np.unique(np.concatenate(parts))


def beam_i(t, stable: bool = False):
    """(I_K0, I_K0bar, a0, abar) of the beam at the times t (the numerical integration)."""
    run = kp.stable_run() if stable else kp.beam_run()
    a0, ab = run.at(t)
    i0, ib = kp.intensities(a0, ab)
    return i0, ib, a0, ab


def regen_arrays() -> dict:
    """The regeneration run: intensities in the K_S / K_L basis before and behind the absorber, and the amplitudes behind it."""
    R = kp.regeneration()
    ps_b, pl_b = kp.project(R["before"])
    ps_a, pl_a = kp.project(R["after"])
    return {"R": R, "tb": R["tb"], "Sb": np.abs(ps_b) ** 2, "Lb": np.abs(pl_b) ** 2, "ta": R["ta"], "Sa": np.abs(ps_a) ** 2, "La": np.abs(pl_a) ** 2}


# ----------------------------------------------------------------------------------- checks of a frame

def text_boxes(fig):
    """[(string, bbox in pixels)] of every visible text of the figure (and of its axes) that is not transparent."""
    from matplotlib.colors import to_rgba
    from matplotlib.text import Text
    r = fig.canvas.get_renderer()
    out = []
    for tobj in fig.findobj(Text):
        s = tobj.get_text()
        if not s.strip() or not tobj.get_visible():
            continue
        alpha = to_rgba(tobj.get_color())[3] * (tobj.get_alpha() if tobj.get_alpha() is not None else 1.0)
        if alpha < CFG.check.min_alpha:
            continue
        out.append((s, tobj.get_window_extent(r)))
    return out


def frame_problems(fig) -> list[str]:
    """Overlapping texts and texts that leave the frame."""
    boxes = text_boxes(fig)
    W, H = fig.canvas.get_width_height()
    pad = CFG.check.pad_px
    bad = []
    for i, (s, b) in enumerate(boxes):
        if b.x0 < -1 or b.y0 < -1 or b.x1 > W + 1 or b.y1 > H + 1:
            bad.append(f"outside the frame: {s!r}")
        for s2, b2 in boxes[i + 1:]:
            if b.x0 < b2.x1 - pad and b2.x0 < b.x1 - pad and b.y0 < b2.y1 - pad and b2.y0 < b.y1 - pad:
                bad.append(f"overlap: {s!r} / {s2!r}")
    return bad


# ----------------------------------------------------------------------------------- the picture

def render(out: Path, size: tuple[int, int], fps: int, total: float, lang: str, snap: float | None = None, hook=None) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle, FancyBboxPatch, Rectangle
    from matplotlib.transforms import blended_transform_factory

    if snap is None and not shutil.which("ffmpeg"):
        sys.exit("ffmpeg is required")
    tx = texts(lang)
    V, ST, LY, F = CFG.video, CFG.style, CFG.layout, CFG.fonts
    CU, TB, PHc, PL, RE, BX, FO, LP = CFG.curves, CFG.tube, CFG.phasor, CFG.plane, CFG.reaction, CFG.box, CFG.foils, CFG.lepton
    P1, P2, P3, P4 = CFG.part1, CFG.part2, CFG.part3, CFG.part4
    W, H = size
    dpi = V.dpi
    sc = H / V.reference_height
    BG = ST.background
    TXT, DIM = tuple(ST.text), tuple(ST.dim)
    GOLD, K0C, KBC, KSC, KLC = tuple(ST.gold), tuple(ST.k0), tuple(ST.k0bar), tuple(ST.ks), tuple(ST.kl)
    GOOD, BAD, WHITE = tuple(ST.good), tuple(ST.bad), tuple(ST.white)
    fig = plt.figure(figsize=(W / dpi, H / dpi), dpi=dpi, facecolor=BG)

    def mk(rect, name):
        a = fig.add_axes(rect, facecolor="none")
        a.set_label(name)
        return a

    ax_cur = mk(LY.curve_axes, "curves")
    ax_tube = mk(LY.tube_axes, "tube")
    ax_ph = [mk(r, f"phasor{i}") for i, r in enumerate(LY.phasor_axes)]
    ax_re = mk(LY.reaction_axes, "reaction")
    ax_re2 = mk(LY.reaction2_axes, "reaction2")
    ax_pl = mk(LY.plane_axes, "plane")
    ax_cn = mk(LY.counter_axes, "counters")
    ax_ov = mk([0.0, 0.0, 1.0, 1.0], "overlay")
    all_axes = [ax_cur, ax_tube, *ax_ph, ax_re, ax_re2, ax_pl, ax_cn, ax_ov]

    k = total / TOTAL
    frames = int(round(total * fps))
    ids = [int(round(snap * fps))] if snap is not None else range(frames)
    writer = None
    if snap is None:
        writer = subprocess.Popen(
            ["ffmpeg", "-y", "-v", "error", "-f", "rawvideo", "-pix_fmt", "rgba", "-s", f"{W}x{H}",
             "-r", str(fps), "-i", "-", "-c:v", "libx264", "-preset", V.preset, "-crf", str(V.crf),
             "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(out)], stdin=subprocess.PIPE)
    bg_rgba = np.array([int(BG[1:3], 16), int(BG[3:5], 16), int(BG[5:7], 16), 255], np.float32)

    def col(c, a=1.0):
        return (*c[:3], a)

    def ftext(x, y, s, color, size, alpha=1.0, **kw):
        if alpha <= ST.panel_cutoff:
            return None
        return fig.text(x, y, s, color=col(color, alpha), fontsize=size * sc, **kw)

    def arrow(ax, p0, p1, color, lw, alpha=1.0, head=None, z=5, ls="-"):
        hd = head or PHc.head
        ax.annotate("", xy=tuple(p1), xytext=tuple(p0), zorder=z,
                    arrowprops=dict(arrowstyle=f"-|>,head_length={hd[0]},head_width={hd[1]}", color=col(color, alpha), lw=lw * sc,
                                    shrinkA=0, shrinkB=0, linestyle=ls, mutation_scale=ST.arrow_scale * sc))

    def line(ax, xs, ys, color, lw, alpha=1.0, dash=None, z=3, **kw):
        ax.plot(xs, ys, color=col(color, alpha), lw=lw * sc, ls=(0, tuple(dash)) if dash else "-", zorder=z, **kw)

    def nice_step(span: float) -> float:
        raw = span / CU.tick_target
        e = 10.0 ** math.floor(math.log10(raw))
        for m in (1.0, 2.0, 5.0, 10.0):
            if m * e >= raw:
                return m * e
        return 10.0 * e

    # ------------------------------------------------------------------------------------------- the time axis and the beam
    def time_axis(ax, lo, hi, ylo, yhi, w_s, w_l, ytk, xlabel_s, xlabel_l, t_zero=0.0):
        """Axes in data units of tau_S; ticks are labelled in tau_S (weight w_s) and in tau_L (weight w_l)."""
        ax.axis("off")
        ax.set_xlim(lo, hi)
        ax.set_ylim(ylo, yhi)
        tr = blended_transform_factory(ax.transData, ax.transAxes)
        line(ax, [lo, hi], [ylo, ylo], WHITE, CU.axis_width, CU.axis_alpha, z=2, clip_on=False)
        line(ax, [t_zero, t_zero], [ylo, yhi], WHITE, CU.axis_width, CU.axis_alpha * CU.axis_zero_k, z=2)
        for unit, w, label in ((1.0, w_s, xlabel_s), (TAU_L, w_l, xlabel_l)):
            if w <= ST.panel_cutoff:
                continue
            step = nice_step((hi - t_zero) / unit)
            n0 = math.ceil((lo - t_zero) / unit / step - 1e-9)
            n1 = math.floor((hi - t_zero) / unit / step + 1e-9)
            for n in range(n0, n1 + 1):
                x = t_zero + n * step * unit
                line(ax, [x, x], [ylo, ylo + CU.tick_len * (yhi - ylo)], WHITE, CU.axis_width, CU.axis_alpha * w, z=2)
                line(ax, [x, x], [ylo, yhi], WHITE, CU.grid_width, CU.grid_alpha * w, z=1)
                lab = plain(n * step, ".0f" if abs(step) >= 1 else f".{max(1, -int(math.floor(math.log10(step))))}f", lang).replace("-", "\u2212")
                ax.text(x, -CU.label_pad, lab, transform=tr, color=col(DIM, w), fontsize=F.tick * sc, ha="center", va="top")
            ax.text(*CU.xlabel_pos, label, transform=ax.transAxes, color=col(DIM, w), fontsize=F.axis_label * sc, ha="right", va="top")
        for y in ytk:
            ax.text(CU.ytick_dx, y, plain(y, ".1f" if y != int(y) else ".0f", lang).replace("-", "\u2212"), transform=blended_transform_factory(ax.transAxes, ax.transData),
                    color=col(DIM), fontsize=F.tick * sc, ha="right", va="center")
            line(ax, [lo, hi], [y, y], WHITE, CU.grid_width, CU.grid_alpha, z=1)

    def draw_tube(ax, lo, hi, t_end, bands_fn, w_s, w_l, label_s, label_l, show_target=True, front=True, t_start=0.0):
        """The beam as a tube: the thickness is proportional to the total intensity, the bands are the composition."""
        ax.axis("off")
        ax.set_xlim(lo, hi)
        ax.set_ylim(-TB.y_half, TB.y_half)
        if t_end > t_start + 1e-9:
            tt = tgrid(t_end, 0.0) if t_start == 0.0 else np.linspace(t_start, t_end, CU.samples_lin)
            bands = bands_fn(tt)
            tot = sum(h for _c, h in bands)
            base = -0.5 * tot
            for c, h in bands:
                ax.fill_between(tt, base, base + h, color=col(c, TB.alpha), lw=0, zorder=3)
                base = base + h
            ax.plot(tt, 0.5 * tot, color=col(WHITE, TB.edge_alpha), lw=TB.edge_width * sc, zorder=4)
            ax.plot(tt, -0.5 * tot, color=col(WHITE, TB.edge_alpha), lw=TB.edge_width * sc, zorder=4)
            if front:
                yt = float(0.5 * tot[-1])
                line(ax, [t_end, t_end], [-yt, yt], WHITE, TB.front_width, TB.front_alpha, z=6)
        if show_target:
            tw = TB.target_w * (hi - lo)
            ax.add_patch(Rectangle((t_start - tw, -0.5), tw, 1.0, facecolor=col(DIM, TB.target_fill), edgecolor=col(WHITE, TB.target_edge), lw=TB.target_lw * sc, zorder=5))
        for w, lab in ((w_s, label_s), (w_l, label_l)):
            if w > ST.panel_cutoff and lab:
                ax.text(TB.label_pos[0], TB.label_pos[1], lab, transform=ax.transAxes, color=col(DIM, w), fontsize=F.legend * sc, ha="right", va="top")

    def legend_row(entries):
        """entries: (kind, colour, text); kind: 'line', 'dash', 'ghost', 'dot'."""
        ll = CU.legend_line_len
        for (kind, c, key, alpha), x0 in zip(entries, LY.legend_x):
            y = LY.legend_row_y
            if alpha <= ST.panel_cutoff:
                continue
            ax_leg = ax_ov
            if kind == "line":
                ax_leg.plot([x0, x0 + ll], [y, y], color=col(c, alpha), lw=CU.line_width * sc, solid_capstyle="round")
            elif kind == "dash":
                ax_leg.plot([x0, x0 + ll], [y, y], color=col(c, alpha * CU.total_alpha), lw=CU.total_width * sc, ls=(0, tuple(CU.total_dash)))
            elif kind == "ghost":
                ax_leg.plot([x0, x0 + ll], [y, y], color=col(c, alpha * CU.ghost_alpha * CU.ghost_legend_k), lw=CU.ghost_width * sc, ls=(0, tuple(CU.ghost_dash)))
            elif kind == "dot":
                ax_leg.scatter([x0 + ll / 2], [y], s=CU.cursor_dot * sc * sc, color=col(c, alpha), zorder=5)
            elif kind == "ring":
                ax_leg.scatter([x0 + ll / 2], [y], s=CU.cursor_dot * sc * sc * CU.ring_scale, facecolors="none", edgecolors=[col(c, alpha)], linewidths=CU.ring_lw * sc, zorder=5)
            fig.text(x0 + ll + CU.legend_gap, y, tx[key], color=col(c, alpha), fontsize=F.legend * sc, va="center", ha="left")

    # ------------------------------------------------------------------------------------------- phasor panels
    def phasor_panels(a0, ab, alpha, lim, stage_stable, zoom_tag=None, trail=None, trail_alpha=1.0):
        s, l = kp.s_l(a0, ab)
        i0, ib = abs(a0) ** 2, abs(ab) ** 2
        for idx, ax in enumerate(ax_ph):
            ax.axis("off")
            ax.set_xlim(-lim, lim)
            ax.set_ylim(-lim, lim)
            ax.set_aspect("equal", adjustable="box")
            if alpha <= ST.panel_cutoff:
                continue
            for r in PHc.ring_radii:
                if r > PHc.ring_clip * lim:
                    continue
                ax.add_patch(Circle((0, 0), r, fill=False, edgecolor=col(WHITE, PHc.ring_alpha * alpha), lw=PHc.ring_width * sc, zorder=1))
            line(ax, [-lim, lim], [0, 0], WHITE, PHc.cross_width, PHc.axis_alpha * alpha, z=1)
            line(ax, [0, 0], [-lim, lim], WHITE, PHc.cross_width, PHc.axis_alpha * alpha, z=1)
            if trail is not None and trail_alpha > ST.panel_cutoff:
                ts, tl = kp.s_l(*trail)
                pairs = ((ts, KSC), (ts + tl, K0C)) if idx == 0 else ((tl, KLC), (tl - ts, KBC))
                for arr, cc_ in pairs:
                    ax.plot(arr.real, arr.imag, color=col(cc_, PHc.trail_alpha * alpha * trail_alpha), lw=PHc.trail_width * sc, zorder=2)
            if idx == 0:
                p1 = (s.real, s.imag)
                tip = (s + l)
                arrow(ax, (0, 0), p1, KSC, PHc.arrow_lw, alpha)
                arrow(ax, p1, (tip.real, tip.imag), KLC, PHc.arrow_lw, alpha)
                line(ax, [0, l.real, tip.real], [0, l.imag, tip.imag], KLC, PHc.join_width, PHc.join_alpha * alpha, PHc.join_dash)
                res, rc = tip, K0C
            else:
                p1 = (l.real, l.imag)
                tip = l - s
                arrow(ax, (0, 0), p1, KLC, PHc.arrow_lw, alpha)
                arrow(ax, p1, (tip.real, tip.imag), KSC, PHc.arrow_lw, alpha)
                line(ax, [0, -s.real, tip.real], [0, -s.imag, tip.imag], KSC, PHc.join_width, PHc.join_alpha * alpha, PHc.join_dash)
                res, rc = tip, KBC
            arrow(ax, (0, 0), (res.real, res.imag), rc, PHc.sum_lw, alpha * PHc.sum_alpha, z=4)
            ax.scatter([res.real], [res.imag], s=PHc.tip_dot * sc * sc, color=col(rc, alpha), zorder=8)
            ax.set_facecolor("none")
        if alpha > ST.panel_cutoff:
            xs = [(LY.phasor_axes[i][0] + LY.phasor_axes[i][2] / 2) for i in (0, 1)]
            ftext(xs[0], LY.phasor_title_y, tx["ph_t0"], K0C, F.phasor_title, alpha, ha="center", va="center")
            ftext(xs[1], LY.phasor_title_y, tx["ph_t1"], KBC, F.phasor_title, alpha, ha="center", va="center")
            ftext(xs[0], LY.readout_y[0], tx["ph_p0"] + "$=" + num(i0, f".{PHc.value_decimals}f", lang) + "$", K0C, F.readout, alpha, ha="center", va="center")
            ftext(xs[1], LY.readout_y[0], tx["ph_p1"] + "$=" + num(ib, f".{PHc.value_decimals}f", lang) + "$", KBC, F.readout, alpha, ha="center", va="center")
            if zoom_tag is not None and zoom_tag > PHc.tag_min:
                for x in xs:
                    ftext(x + LY.phasor_axes[0][2] / 2 - PHc.tag_inset, LY.phasor_axes[0][1] + PHc.tag_inset, "$\\times" + num(zoom_tag, ".1f", lang) + "$", DIM, F.legend, alpha, ha="right", va="bottom")

    # ------------------------------------------------------------------------------------------- the plane (K0, K0bar)
    def plane_picture(a_axes, a_eig, a_proj, vec, vec_color, label_vec, comps, a_arrow, label_dir=1):
        ax = ax_pl
        ax.axis("off")
        ax.set_xlim(-PL.limit, PL.limit)
        ax.set_ylim(-PL.limit, PL.limit)
        ax.set_aspect("equal", adjustable="box")
        L = PL.limit
        r2 = 1.0 / math.sqrt(2.0)
        if a_axes > ST.panel_cutoff:
            arrow(ax, (-L * PL.axis_neg, 0), (L * PL.axis_pos, 0), K0C, PL.axis_width, a_axes * PL.axis_alpha, PL.head)
            arrow(ax, (0, -L * PL.axis_neg), (0, L * PL.axis_pos), KBC, PL.axis_width, a_axes * PL.axis_alpha, PL.head)
            ax.text(L * PL.axis_pos, -PL.k0_label_dy, tx["pl_k0"], color=col(K0C, a_axes), fontsize=F.plane_label * sc, ha="right", va="top")
            ax.text(-PL.kb_label_dx, L * PL.axis_pos, tx["pl_kb"], color=col(KBC, a_axes), fontsize=F.plane_label * sc, ha="right", va="top")
        if a_eig > ST.panel_cutoff:
            e = L * PL.eigen_extent * r2
            line(ax, [-e, e], [-e, e], KLC, PL.eigen_width, a_eig * PL.eigen_alpha, PL.eigen_dash, z=2)
            line(ax, [-e, e], [e, -e], KSC, PL.eigen_width, a_eig * PL.eigen_alpha, PL.eigen_dash, z=2)
            ax.text(e + PL.eigen_label_gap, e, tx["pl_kl"], color=col(KLC, a_eig), fontsize=F.plane_label * sc, ha="left", va="center")
            ax.text(e + PL.eigen_label_gap, -e, tx["pl_ks"], color=col(KSC, a_eig), fontsize=F.plane_label * sc, ha="left", va="center")
        if a_proj > ST.panel_cutoff:
            for (cx, cy), cc_, key in comps:
                tip = np.array(vec)
                foot = np.array([cx, cy])
                line(ax, [tip[0], foot[0]], [tip[1], foot[1]], cc_, PL.drop_width, PL.drop_alpha * a_proj, PL.drop_dash, z=3)
                arrow(ax, (0, 0), (cx, cy), cc_, PL.comp_lw, a_proj, PL.head, z=4)
        if a_arrow > ST.panel_cutoff:
            arrow(ax, (0, 0), tuple(vec), vec_color, PL.arrow_lw, a_arrow, PL.head, z=6)
            ax.scatter([0], [0], s=PL.state_dot * sc * sc, color=col(WHITE, a_arrow), zorder=7)
            if label_vec:
                ax.text(vec[0] + PL.label_gap, vec[1] + label_dir * PL.label_gap * PL.state_label_k, label_vec, color=col(vec_color, a_arrow), fontsize=F.plane_label * sc,
                        ha="left", va="bottom" if label_dir > 0 else "top")

    # ------------------------------------------------------------------------------------------- the reaction picture of part 1
    def reaction_picture(ax, geo, a_pic, a_s, keys, colors):
        """A two-body reaction a + target -> o1 + o2 at a vertex; keys: texts of (a, target, o1, o2, interaction, S of a, S of o1, S of o2); colors of (a, o1, o2)."""
        ax.axis("off")
        ax.set_xlim(*geo.x_lim)
        ax.set_ylim(*geo.y_lim)
        ax.set_aspect("equal", adjustable="box")
        if a_pic <= ST.panel_cutoff:
            return
        k_in, k_t, k_o1, k_o2, k_str, k_s_in, k_s_o1, k_s_o2 = keys
        c_in, c_o1, c_o2 = colors
        v = np.array(geo.vertex)
        b0, b1 = np.array(geo.beam[0]), np.array(geo.beam[1])
        arrow(ax, b0, b1, c_in, RE.arrow_lw, a_pic, RE.head)
        ax.add_patch(Circle(tuple(v), geo.vertex_radius, facecolor=col(GOLD, RE.vertex_fill * a_pic), edgecolor=col(GOLD, a_pic), lw=RE.vertex_lw * sc, zorder=4))
        d1, d2 = np.array(geo.out1), np.array(geo.out2)
        for d, c in ((d1, c_o1), (d2, c_o2)):
            u = (d - v) / np.linalg.norm(d - v)
            arrow(ax, v + u * geo.vertex_radius, d, c, RE.arrow_lw, a_pic, RE.head)
        ax.text(b0[0], b0[1] + RE.in_label_dy, tx[k_in], color=col(c_in, a_pic), fontsize=F.reaction_label * sc, ha="left", va="bottom")
        ax.text(v[0], v[1] - geo.vertex_radius - RE.target_label_dy, tx[k_t], color=col(DIM, a_pic), fontsize=F.reaction_small * sc, ha="center", va="top")
        ax.text(d1[0] + RE.label_dx, d1[1], tx[k_o1], color=col(c_o1, a_pic), fontsize=F.reaction_label * sc, ha="left", va="center")
        ax.text(d2[0] + RE.label_dx, d2[1], tx[k_o2], color=col(c_o2, a_pic), fontsize=F.reaction_label * sc, ha="left", va="center")
        ax.text(v[0] + RE.str_dx, v[1] + geo.str_dy, tx[k_str], color=col(GOLD, a_pic), fontsize=F.reaction_small * sc, ha="center", va="bottom" if geo.str_dy > 0 else "top")
        if a_s > ST.panel_cutoff:
            ax.text(b0[0], b0[1] - RE.s_in_dy, tx[k_s_in], color=col(DIM, a_s), fontsize=F.reaction_small * sc, ha="left", va="top")
            ax.text(d1[0] + RE.label_dx, d1[1] - RE.s_dy, tx[k_s_o1], color=col(DIM, a_s), fontsize=F.reaction_small * sc, ha="left", va="center")
            ax.text(d2[0] + RE.label_dx, d2[1] - RE.s_dy, tx[k_s_o2], color=col(c_o2, a_s), fontsize=F.reaction_small * sc, ha="left", va="center")

    def box_card(i, a, title_key, value_key, note_key, color):
        if a <= ST.panel_cutoff:
            return
        x, y, w, h = LY.cards_x[i], LY.cards_y, LY.cards_w, LY.cards_h
        axb = ax_ov
        axb.add_patch(FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0,rounding_size={BX.rounding}", transform=axb.transAxes, facecolor=col(color, BX.alpha_fill * a),
                                      edgecolor=col(color, BX.edge_alpha * a), lw=BX.edge_width * sc))
        ftext(x + BX.pad, y + h - BX.title_dy, tx[title_key], color, F.box_title, a, ha="left", va="center")
        ftext(x + BX.pad, y + h * BX.value_at, tx[value_key], TXT, F.box_value, a, ha="left", va="center")
        ftext(x + BX.pad, y + BX.note_pad, tx[note_key], DIM, F.box_note, a, ha="left", va="bottom", linespacing=BX.linespacing)

    # ================================================================================================ the frames
    for k_ in ids:
        t_film = k_ / fps
        tf_nom = t_film / k
        t, card, cprog = timeline(tf_nom)
        for a in all_axes:
            a.clear()
            a.set_facecolor("none")
            a.axis("off")
        ax_ov.set_xlim(0, 1)
        ax_ov.set_ylim(0, 1)
        fig.texts.clear()
        # ----- cards
        if card is not None:
            a_c = min(smooth(cprog, *TL.card_fade_in), 1.0 - smooth(cprog, *TL.card_fade_out))
            if card == 0:
                ftext(0.5, LY.card_title_y, tx["h0"], tuple(ST.card_title_color), F.card_title, a_c, ha="center", va="center")
                ftext(0.5, LY.card_subtitle_y, tx["h0s"], DIM, F.card_subtitle, a_c, ha="center", va="center")
            else:
                ftext(0.5, LY.card_number_y, f"{part_of(t) + 1}", GOLD, F.card_number, ST.card_number_alpha * a_c, ha="center", va="center")
                ftext(0.5, LY.card_chapter_y, tx[CARD_KEYS[card]], tuple(ST.card_title_color), F.card_chapter, a_c, ha="center", va="center")
                ftext(0.5, LY.card_chapter_sub_y, tx[CARD_KEYS[card] + "s"], DIM, F.card_chapter_sub, a_c, ha="center", va="center")
        else:
            part = part_of(t)
            ftext(*LY.title_pos, tx["title"], tuple(ST.title_color[:3]), F.title, ST.title_color[3], ha="left", va="center")
            ftext(*LY.chapter_label_pos, f"{part + 1}/{len(PART_KEYS)}   " + tx[PART_KEYS[part]], DIM, F.chapter_label, ST.chapter_label_alpha, ha="right", va="center")
            cap_key, formulas = None, []

            def caption_at(times, keys):
                out = keys[0]
                for tt, kk in zip(times, keys):
                    if t >= tt:
                        out = kk
                return out

            # ============================================================ part 1
            if part == 0:
                cap_key = caption_at(P1.caption_at, ["c1a", "c1b", "c1c", "c1d", "c1e"])
                formulas = [("f1a", fade(t, P1.formula_in[0])), ("f1b", fade(t, P1.formula_in[1])), ("f1c", fade(t, P1.formula_in[2]))]
                reaction_picture(ax_re, RE, fade(t, P1.reaction_in), fade(t, P1.strangeness_in),
                                 ("rx_pi", "rx_p", "rx_l", "rx_k", "rx_strong", "rx_s_pi", "rx_s_l", "rx_s_k"), (WHITE, WHITE, K0C))
                a_plane = fade(t, P1.plane_in)
                a_arrow = fade(t, P1.arrow_in)
                a_eig = fade(t, P1.eigen_in)
                a_proj = fade(t, P1.proj_in)
                r2 = 1.0 / math.sqrt(2.0)
                comps = [((0.5, 0.5), KLC, "kl"), ((0.5, -0.5), KSC, "ks")]
                plane_picture(a_plane, a_eig, a_proj, (1.0, 0.0), K0C, tx["pl_state"], comps, a_arrow)
                if a_proj > ST.panel_cutoff:
                    ax_pl.text(*PL.comp_label_l, tx["pl_c_l"], color=col(KLC, a_proj), fontsize=F.plane_small * sc, ha="right", va="center")
                    ax_pl.text(*PL.comp_label_s, tx["pl_c_s"], color=col(KSC, a_proj), fontsize=F.plane_small * sc, ha="right", va="center")
                box_card(0, fade(t, P1.cards_in[0]), "b_s", "b_s_v", "b_s_n", KSC)
                box_card(1, fade(t, P1.cards_in[1]), "b_l", "b_l_v", "b_l_n", KLC)
                box_card(2, fade(t, P1.cards_in[2]), "b_m", "b_m_v", "b_m_n", GOLD)
            # ============================================================ part 2
            elif part == 1:
                t_st, t_re, t_zm = P2.stable, P2.real, P2.zoom
                xhe, ce = P2.x_hi_early, P2.cursor_real_end
                stable_stage = t < t_re[0]
                cap_key = caption_at(P2.caption_at, ["c2a", "c2b", "c2c", "c2d", "c2e"])
                if t < t_re[0]:
                    cursor = lerp(0.0, kp.PERIOD, smooth(t, t_st[0] + P2.stable_ramp[0], t_st[1] - P2.stable_ramp[1]))
                    xhi = xhe
                elif t < t_zm[0]:
                    cursor = ce * ((t - t_re[0]) / (t_re[1] - t_re[0])) ** P2.cursor_power
                    xhi = xhe
                else:
                    u = smooth(t, *t_zm)
                    cursor = math.exp(lerp(math.log(ce), math.log(P2.cursor_zoom_end * TAU_L), u))
                    xhi = cursor * xhe / ce
                if t < t_re[0]:
                    formulas = [("f2a1", fade(t, [t_st[0] + e for e in P2.formula_in[0]])), ("f2a2", fade(t, [t_st[0] + e for e in P2.formula_in[1]]))]
                elif t < t_zm[0]:
                    ft = t - t_re[0]
                    formulas = [("f2b1", fade(ft, P2.formula_in[0])), ("f2b2", fade(ft, P2.formula_in[1])), ("f2b3", fade(ft, P2.formula_in[2]))]
                else:
                    formulas = [("f2c1", fade(t - t_zm[0], P2.formula_in[0])), ("f2c2", fade(t - t_zm[0], P2.formula_in[1]))]
                # tick weights
                w_s = 1.0 - smooth(xhi, *P2.tick_switch)
                w_l = smooth(xhi / TAU_L, *P2.tick_switch_l)
                xlo = -CU.x_lo_frac * xhi
                time_axis(ax_cur, xlo, xhi, 0.0, CU.y_hi, w_s, w_l, CU.y_ticks, tx["xl_s"], tx["xl_l"])
                tt = tgrid(cursor)
                # the curves up to the cursor
                if stable_stage:
                    i0, ib, a0, ab = beam_i(tt, stable=True)
                    line(ax_cur, tt, i0, K0C, CU.line_width, 1.0, z=5)
                    line(ax_cur, tt, ib, KBC, CU.line_width, 1.0, z=5)
                    line(ax_cur, [0, xhi], [1.0, 1.0], WHITE, CU.total_width, CU.total_alpha * CU.sum1_k, CU.total_dash, z=3)
                    ghost_a = 0.0
                else:
                    i0, ib, a0, ab = beam_i(tt)
                    gtt = tgrid(kp.PH.stable_t_end)
                    gi0, gib, _, _ = beam_i(gtt, stable=True)
                    ghost_a = 1.0 - P2.ghost_fade_to * smooth(t, t_zm[0], t_zm[0] + P2.ghost_fade_s)
                    if xhi < P2.ghost_hide_above:
                        line(ax_cur, gtt, gi0, K0C, CU.ghost_width, CU.ghost_alpha * ghost_a, CU.ghost_dash, z=2)
                        line(ax_cur, gtt, gib, KBC, CU.ghost_width, CU.ghost_alpha * ghost_a, CU.ghost_dash, z=2)
                    line(ax_cur, tt, i0, K0C, CU.line_width, 1.0, z=5)
                    line(ax_cur, tt, ib, KBC, CU.line_width, 1.0, z=5)
                    line(ax_cur, tt, i0 + ib, WHITE, CU.total_width, CU.total_alpha, CU.total_dash, z=4)
                ci0, cib, ca0, cab = beam_i(np.array([cursor]), stable=stable_stage)
                line(ax_cur, [cursor, cursor], [0.0, CU.y_hi], GOLD, CU.cursor_width, CU.cursor_alpha, z=4)
                ax_cur.scatter([cursor, cursor], [ci0[0], cib[0]], s=CU.cursor_dot * sc * sc, color=[col(K0C), col(KBC)], zorder=7)
                # legend
                if stable_stage:
                    legend_row([("line", K0C, "lg_k0", 1.0), ("line", KBC, "lg_kb", 1.0), ("dash", WHITE, "lg_sum1", 1.0)])
                else:
                    legend_row([("line", K0C, "lg_k0", 1.0), ("line", KBC, "lg_kb", 1.0), ("dash", WHITE, "lg_sum", 1.0), ("ghost", WHITE, "lg_stable", ghost_a * (1 - smooth(xhi, *P2.ghost_legend_fade)))])
                # tube
                # K0bar below, K0 above
                tube_fn = (lambda x: [(KBC, beam_i(x, stable_stage)[1]), (K0C, beam_i(x, stable_stage)[0])])
                draw_tube(ax_tube, xlo, xhi, cursor, tube_fn, w_s, w_l, tx["tb_s"], tx["tb_l"])
                # phasors
                alpha_ph = smooth(t, PB[1] + P2.phasor_in[0], PB[1] + P2.phasor_in[1])
                tot_now = float(ci0[0] + cib[0])
                factor = 1.0 / max(P2.phasor_zoom_floor, math.sqrt(tot_now))       # the panels are enlarged as the amplitudes shrink
                lim = PHc.limit / factor
                zoom_tag = factor
                tr_t = np.linspace(0.0, min(cursor, P2.trail_max), PHc.trail_samples)
                tr = beam_i(tr_t, stable_stage)
                tr_a = 1.0 - (smooth(t, *t_zm) if t >= t_zm[0] else 0.0)
                phasor_panels(ca0[0], cab[0], alpha_ph, lim, stable_stage, zoom_tag, (tr[2], tr[3]), tr_a)
                ftext(LY.phasor_text_x, LY.phasor_legend_y[0], tx["ph_ls0" if stable_stage else "ph_ls"], KSC, F.legend, alpha_ph, ha="left", va="center")
                ftext(LY.phasor_text_x, LY.phasor_legend_y[1], tx["ph_ll0" if stable_stage else "ph_ll"], KLC, F.legend, alpha_ph, ha="left", va="center")
                ftext(LY.phasor_text_x, LY.readout_y[1], tx["ph_sum"], DIM, F.legend, alpha_ph, ha="left", va="center")
            # ============================================================ part 3
            elif part == 2:
                lepton_stage = t >= P3.lepton[0]
                cap_key = caption_at(P3.caption_at, ["c3a", "c3b", "c3c", "c3d", "c3e"])
                xhi = P3.lepton_x_hi if lepton_stage else P3.x_hi
                xlo = -CU.x_lo_frac * xhi
                time_axis(ax_cur, xlo, xhi, CU.y_lo_asym if lepton_stage else 0.0, CU.y_hi, 1.0, 0.0,
                          CU.y_ticks_asym if lepton_stage else CU.y_ticks, tx["xl_s"], tx["xl_l"])
                tt = tgrid(xhi)
                i0, ib, a0, ab = beam_i(tt)
                tube_fn = lambda x: [(KBC, beam_i(x)[1]), (K0C, beam_i(x)[0])]
                if not lepton_stage:
                    formulas = [("f3a1", fade(t, P3.formula_in[0])), ("f3a2", fade(t, P3.formula_in[1]))]
                    line(ax_cur, tt, i0, K0C, CU.line_width * FO.curve_width_k, FO.k0_curve_alpha, z=3)
                    line(ax_cur, tt, ib, KBC, CU.line_width * FO.curve_width_k, FO.kb_curve_alpha, z=4)
                    foil_a = fade(t, P3.foils_in)
                    draw_tube(ax_tube, xlo, xhi, xhi, tube_fn, 1.0, 0.0, tx["tb_s"], "", front=False)
                    ft = np.array(P3.foils)
                    fi0, fib, _, _ = beam_i(ft)
                    for n, (tp, y0, yb) in enumerate(zip(ft, fi0, fib)):
                        a_f = foil_a
                        a_y = smooth(t, P3.foil_start + n * P3.foil_dt, P3.foil_start + n * P3.foil_dt + FO.stem_grow_s)
                        hf = TB.foil_half
                        ax_tube.add_patch(Rectangle((tp - 0.5 * TB.foil_width * (xhi - xlo), -hf), TB.foil_width * (xhi - xlo), 2 * hf,
                                                     facecolor=col(WHITE, TB.foil_alpha * a_f), edgecolor="none", zorder=8))
                        if a_y > ST.panel_cutoff:
                            line(ax_cur, [tp, tp], [0.0, yb * a_y], KBC, FO.stem_width, 1.0, z=6)
                            ax_cur.scatter([tp], [yb * a_y], s=FO.dot_size * sc * sc, color=col(KBC), zorder=8)
                            ax_cur.text(tp, yb * a_y + FO.value_dy, plain(yb, f".{FO.value_decimals}f", lang), color=col(KBC, a_y), fontsize=FO.label_size * sc, ha="center", va="bottom")
                            ax_cur.plot([tp], [0.0], marker="o", ms=FO.k0_marker_size * sc, mfc="none", mec=col(K0C, a_y), mew=FO.k0_marker_lw * sc, zorder=8, clip_on=False)
                    legend_row([("line", K0C, "lg_k0", 1.0), ("line", KBC, "lg_kb", 1.0), ("dot", KBC, "lg_lambda", fade(t, P3.foils_in)), ("ring", K0C, "lg_nolambda", fade(t, P3.foils_in))])
                    reaction_picture(ax_re2, CFG.reaction2, fade(t, P3.table_in[0]), fade(t, P3.table_in[1]),
                                     ("rx2_in", "rx2_n", "rx2_l", "rx2_p", "rx_strong", "rx2_s_in", "rx2_s_l", "rx2_s_p"), (KBC, WHITE, WHITE))
                    # the table of the reactions
                    ta = [fade(t, P3.table_in[0]), fade(t, P3.table_in[1])]
                    for n, key in enumerate(("tb_bar", "tb_k0")):
                        yy = LY.table_y[n]
                        okc = GOOD if n == 0 else BAD
                        ftext(LY.table_x, yy, tx[key], KBC if n == 0 else K0C, F.table_row, ta[n], ha="left", va="center")
                        ftext(LY.table_x, yy - LY.table_dy[0], tx[key + "_s"], DIM, F.table_note, ta[n], ha="left", va="center")
                        ftext(LY.table_x, yy - LY.table_dy[1], tx[key + "_v"], okc, F.table_note, ta[n], ha="left", va="center")
                else:
                    formulas = [("f3b1", fade(t, P3.formula_in[2])), ("f3b2", fade(t, [P3.formula_in[2][0] + P3.formula_delay, P3.formula_in[2][1] + P3.formula_delay]))]
                    u = (t - P3.lepton[0]) / (P3.lepton[1] - P3.lepton[0])
                    cursor = lerp(0.0, P3.lepton_cursor_end, smooth(t, P3.lepton[0] + P3.cursor_ramp[0], P3.lepton[1] - P3.cursor_ramp[1]))
                    a_c = fade(t, P3.lepton_curve_in)
                    line(ax_cur, [xlo, xhi], [0, 0], WHITE, LP.zero_width, LP.zero_alpha, LP.zero_dash, z=2)
                    ct = tgrid(cursor)
                    ci0, cib, _, _ = beam_i(ct)
                    line(ax_cur, ct, np.cos(DM * ct), WHITE, CU.ghost_width, LP.ref_alpha * a_c, CU.ghost_dash, z=3)
                    envt = tgrid(xhi)
                    env = 1.0 / np.cosh(0.5 * (kp.GAMMA_S - kp.GAMMA_L) * envt)
                    line(ax_cur, envt, env, GOLD, CU.ghost_width, LP.ref_alpha * a_c, CU.total_dash, z=3)
                    line(ax_cur, envt, -env, GOLD, CU.ghost_width, LP.ref_alpha * a_c, CU.total_dash, z=3)
                    asym = (ci0 - cib) / (ci0 + cib)
                    line(ax_cur, ct, asym, WHITE, CU.line_width, a_c, z=6)
                    c0, cb, _, _ = beam_i(np.array([cursor]))
                    ac = float((c0 - cb)[0] / (c0 + cb)[0])
                    line(ax_cur, [cursor, cursor], [CU.y_lo_asym, CU.y_hi], GOLD, CU.cursor_width, CU.cursor_alpha, z=4)
                    ax_cur.scatter([cursor], [ac], s=CU.cursor_dot * sc * sc, color=col(WHITE), zorder=7)
                    legend_row([("line", WHITE, "lg_asym", a_c), ("ghost", WHITE, "lg_cos", a_c), ("dash", GOLD, "lg_env", a_c)])
                    draw_tube(ax_tube, xlo, xhi, cursor, tube_fn, 1.0, 0.0, tx["tb_s"], "")
                    # the counters
                    ac_a = fade(t, P3.counters_in)
                    axc = ax_cn
                    axc.axis("off")
                    axc.set_xlim(0, LP.x_hi)
                    axc.set_ylim(0, LP.y_hi)
                    if ac_a > ST.panel_cutoff:
                        for xb, val, c_, key in ((LP.bar_x[0], float(c0[0]), K0C, "cn_plus"), (LP.bar_x[1], float(cb[0]), KBC, "cn_minus")):
                            axc.bar([xb], [val], width=LP.bar_width, color=col(c_, LP.bar_alpha * ac_a))
                            axc.text(xb, -LP.label_dy, tx[key], color=col(c_, ac_a), fontsize=F.counter_label * sc, ha="center", va="top", clip_on=False)
                            axc.text(xb, val + LP.value_dy, plain(val, ".2f", lang), color=col(c_, ac_a), fontsize=F.counter_label * sc, ha="center", va="bottom")
                        line(axc, [0, LP.x_hi], [0, 0], WHITE, LP.base_width, LP.base_alpha * ac_a)
                        ftext(LY.counter_axes[0] + LY.counter_axes[2] / 2, LY.counter_axes[1] + LY.counter_readout_dy, tx["cn_a"] + "$=" + num(ac, ".2f", lang) + "$", WHITE, F.readout, ac_a, ha="center", va="center")
                    for n, (key, cc_) in enumerate((("lp_plus", K0C), ("lp_minus", KBC))):
                        ftext(LY.table_x, LY.lepton_rows[n], tx[key], cc_, F.table_row, ac_a, ha="left", va="center")
                    ftext(LY.table_x, LY.lepton_rows[2], tx["lp_rule"], DIM, F.table_note, ac_a, ha="left", va="center")
            # ============================================================ part 4
            else:
                cap_key = caption_at(P4.caption_at, ["c4a", "c4b", "c4c", "c4d", "c4e"])
                RA = regen_arrays()
                R = RA["R"]
                xlo, xhi = P4.x_lo, P4.x_hi
                a_slab = fade(t, P4.slab_in)
                u_prog = smooth(t, *P4.progress)
                sweeping = t >= P4.sweep[0]
                time_axis(ax_cur, xlo, xhi, 0.0, CU.y_hi, 1.0, 0.0, CU.y_ticks, tx["xl_re"], tx["xl_l"])
                cursor = min(max(lerp(0.0, P4.sweep_end, (t - P4.sweep[0]) / (P4.sweep[1] - P4.sweep[0])), 0.0), P4.sweep_end) if sweeping else 0.0
                formulas = [("f4a", fade(t, P4.formula_in[0])), ("f4b", fade(t, P4.formula_in[1])), ("f4c", fade(t, P4.formula_in[2]))]
                # the state of the beam in the slab: the amplitudes of K0 and K0bar are multiplied by f^u and fbar^u (real parts after removing the common phase)
                psi_in = R["psi_in"]
                ph = np.exp(-1j * np.angle(kp.project(psi_in)[1]))
                v_now = (kp.slab_matrix(u_prog) @ psi_in * ph).real
                s_amp = float((v_now[0] - v_now[1]) / math.sqrt(2.0))
                l_amp = float((v_now[0] + v_now[1]) / math.sqrt(2.0))
                i_l_now, i_s_now = l_amp ** 2, s_amp ** 2
                # intensities: before the absorber the beam is K_L; behind it K_S is regenerated
                line(ax_cur, RA["tb"], RA["Lb"], KLC, CU.line_width, 1.0, z=5)
                line(ax_cur, RA["tb"], RA["Sb"], KSC, CU.line_width, 1.0, z=5)
                if sweeping:
                    m = RA["ta"] <= cursor + 1e-9
                    line(ax_cur, RA["ta"][m], RA["La"][m], KLC, CU.line_width, 1.0, z=5)
                    line(ax_cur, RA["ta"][m], RA["Sa"][m], KSC, CU.line_width, 1.0, z=5)
                    line(ax_cur, [0, 0], [float(RA["Lb"][-1]), float(RA["La"][0])], KLC, CU.line_width, P4.jump_alpha, z=4)
                    line(ax_cur, [0, 0], [float(RA["Sb"][-1]), float(RA["Sa"][0])], KSC, CU.line_width, P4.jump_alpha, z=4)
                    line(ax_cur, [cursor, cursor], [0.0, CU.y_hi], GOLD, CU.cursor_width, CU.cursor_alpha, z=4)
                    ci = int(round(cursor / CFG.integrator.fine_dt))
                    ax_cur.scatter([cursor, cursor], [RA["La"][ci], RA["Sa"][ci]], s=CU.cursor_dot * sc * sc, color=[col(KLC), col(KSC)], zorder=7)
                elif u_prog > 0.0:
                    line(ax_cur, [0, 0], [float(RA["Lb"][-1]), i_l_now], KLC, CU.line_width, P4.pass_alpha, z=4)
                    line(ax_cur, [0, 0], [float(RA["Sb"][-1]), i_s_now], KSC, CU.line_width, P4.pass_alpha, z=4)
                    ax_cur.scatter([0, 0], [i_l_now, i_s_now], s=CU.cursor_dot * sc * sc, color=[col(KLC), col(KSC)], zorder=7)
                legend_row([("line", KLC, "lg_kl", 1.0), ("line", KSC, "lg_ks", 1.0)])
                # the tube: K_L below, K_S above; the absorber at 0
                ax_tube.axis("off")
                ax_tube.set_xlim(xlo, xhi)
                ax_tube.set_ylim(-TB.y_half, TB.y_half)
                if sweeping:
                    keep = RA["ta"] <= cursor + 1e-9
                    full_t = np.concatenate([RA["tb"], RA["ta"][keep]])
                    lb = np.concatenate([RA["Lb"], RA["La"][keep]])
                    sb = np.concatenate([RA["Sb"], RA["Sa"][keep]])
                else:
                    stub = np.array([0.0, P4.stub])
                    full_t = np.concatenate([RA["tb"], stub])
                    lb = np.concatenate([RA["Lb"], [i_l_now, i_l_now]])
                    sb = np.concatenate([RA["Sb"], [i_s_now, i_s_now]])
                tot = lb + sb
                base = -0.5 * tot
                ax_tube.fill_between(full_t, base, base + lb, color=col(KLC, TB.alpha), lw=0, zorder=3)
                ax_tube.fill_between(full_t, base + lb, base + lb + sb, color=col(KSC, TB.alpha), lw=0, zorder=3)
                ax_tube.plot(full_t, 0.5 * tot, color=col(WHITE, TB.edge_alpha), lw=TB.edge_width * sc, zorder=4)
                ax_tube.plot(full_t, -0.5 * tot, color=col(WHITE, TB.edge_alpha), lw=TB.edge_width * sc, zorder=4)
                sw = TB.slab_width * (xhi - xlo)
                ax_tube.add_patch(Rectangle((-0.5 * sw, -TB.slab_half), sw, 2 * TB.slab_half, facecolor=col(WHITE, TB.slab_fill * a_slab),
                                             edgecolor=col(WHITE, TB.slab_edge * a_slab), lw=TB.slab_lw * sc, zorder=8))
                ax_tube.text(TB.label_pos[0], TB.label_pos[1], tx["tb_re"], transform=ax_tube.transAxes, color=col(DIM), fontsize=F.legend * sc, ha="right", va="top")
                ftext(LY.curve_axes[0] + LY.curve_axes[2] * (0.0 - xlo) / (xhi - xlo), LY.slab_label_y, tx["slab"], WHITE, F.legend, a_slab, ha="center", va="center")
                # the right column: the plane first, then the phasors
                a_ph = smooth(t, *P4.phasor_in)
                a_pl = (1.0 - a_ph) * a_slab
                if a_pl > ST.panel_cutoff:
                    rr = 1.0 / math.sqrt(2.0)
                    comps = [((l_amp * rr, l_amp * rr), KLC, "kl"), ((s_amp * rr, -s_amp * rr), KSC, "ks")]
                    plane_picture(a_pl, a_pl, a_pl, (float(v_now[0]), float(v_now[1])), WHITE, tx["pl_state4"], comps, a_pl, -1)
                    ax_pl.text(*PL.comp_label_l4, tx["pl_c_l4"], color=col(KLC, a_pl), fontsize=F.plane_small * sc, ha="right", va="center")
                    if abs(s_amp) > PL.min_label_amp:
                        ax_pl.text(s_amp * rr - PL.comp_gap, -s_amp * rr - PL.comp_gap, tx["pl_c_s4"], color=col(KSC, a_pl), fontsize=F.plane_small * sc, ha="right", va="top")
                if a_ph > ST.panel_cutoff:
                    ci_ = int(round(cursor / CFG.integrator.fine_dt))
                    ca0, cab = R["after"][ci_, 0] * ph, R["after"][ci_, 1] * ph
                    phasor_panels(ca0, cab, a_ph, PHc.limit / P4.phasor_zoom, False, P4.phasor_zoom, (R["after"][:ci_ + 1, 0] * ph, R["after"][:ci_ + 1, 1] * ph), 1.0)
                    ftext(LY.phasor_text_x, LY.phasor_legend_y[0], tx["ph_ks4"], KSC, F.legend, a_ph, ha="left", va="center")
                    ftext(LY.phasor_text_x, LY.phasor_legend_y[1], tx["ph_kl4"], KLC, F.legend, a_ph, ha="left", va="center")

            for key, alpha in formulas:
                if alpha > LY.formula_min_alpha:
                    row = [kk for kk, _ in formulas].index(key)
                    ftext(LY.formula_x, LY.formula_rows[row], tx[key], TXT, F.formula, ST.formula_alpha * alpha, ha="left", va="center")
            if cap_key:
                ftext(*LY.caption_pos, tx[cap_key], TXT, F.caption, ST.caption_alpha, ha="left", va="center")
        fade_io = min(smooth(t_film, 0.0, V.fade_s), 1.0 - smooth(t_film, total - V.fade_s, total))
        fig.canvas.draw()
        if hook is not None:
            hook(fig, t, card)
        frame = np.asarray(fig.canvas.buffer_rgba()).astype(np.float32)
        if fade_io < 1.0:
            frame = bg_rgba + (frame - bg_rgba) * fade_io
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
    ap.add_argument("--snapshot", type=float, default=None, help="film time in seconds (the cards are counted)")
    ap.add_argument("--content", type=float, default=None, help="snapshot at this CONTENT time (no cards)")
    ap.add_argument("--seconds", type=float, default=TOTAL)
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--config", type=Path, default=None, help="another configuration file instead of config.toml")
    ap.add_argument("--set", action="append", default=[], metavar="SECTION.KEY=VALUE", help="override one configuration value")
    args = ap.parse_args()
    V = CFG.video
    out = args.out or HERE / "media" / f"kaon_oscillations_{args.lang}.mp4"
    out.parent.mkdir(parents=True, exist_ok=True)
    if args.content is not None:
        args.snapshot = film_time(args.content)
    if args.snapshot is not None:
        render(out.with_suffix(".png"), (V.width, V.height), V.fps, args.seconds, args.lang, snap=args.snapshot)
    elif args.preview:
        render(out.with_name(out.stem + "_preview.mp4"), (V.preview_width, V.preview_height), V.preview_fps, args.seconds, args.lang)
    else:
        render(out, (V.width, V.height), V.fps, args.seconds, args.lang)


if __name__ == "__main__":
    main()
