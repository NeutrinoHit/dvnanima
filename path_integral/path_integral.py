r"""The path integral: from slits to all paths, and why the classical path wins.

Scene 1.  Amplitudes are added over the paths source -> slit -> ... -> detector through screens with
3, 7 and 5 slits (3, 21 and 105 paths).  The phase of a path is 2 pi L / lambda (L is its length);
the amplitude at the detector is the sum of the unit phasors, drawn head to tail.  The brightness
profile |A|^2 along the detector is computed from the same sum.

Scene 2.  With infinitely many screens and slits all paths contribute.  For a free particle the
paths x_a(t) = a sin(pi t / T) from (0, 0) to (T, 0) have the action

    S(a) = (m/2) int_0^T (dx/dt)^2 dt = (m pi^2 / (4 T)) a^2,

so the phase S / hbar = c a^2 grows quadratically with the deviation a from the classical path
(a = 0).  The phasors e^{i c a^2} added head to tail draw the Cornu spiral: near a = 0 they point the
same way (stationary action), far from it they curl up and cancel.  The sum converges to
sqrt(pi / c) e^{i pi / 4}.

Scene 3.  When hbar decreases, c grows, the spiral shrinks and only the paths within
|a| < sqrt(pi / c) matter: the classical limit.

Schematic: the family of paths is one-parameter (a), the action is quadratic in a, and the values of
c are chosen for clarity.

All the numbers are in config.toml and all the words in texts.toml (see ../dvconfig.py for --config / --set).

Usage:
    python path_integral.py --lang en                # film -> media/path_integral_en.mp4
    python path_integral.py --lang ru
    python path_integral.py --lang en --preview
    python path_integral.py --lang en --snapshot 12  # one PNG at film time 12 s
"""

from __future__ import annotations

import argparse
import itertools
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

SCREENS = tuple(zip(CFG.model.screen_x, CFG.model.slit_counts))     # (x of the screen, number of slits)
SLIT_SPACING = CFG.model.slit_spacing
WAVELENGTH = CFG.model.wavelength
C_PHASE = CFG.model.c_phase                 # c = m pi^2 / (4 T hbar) in scene 2, 1/a^2 units
A_MAX = CFG.model.a_max
DETECTOR_X = CFG.model.detector_x
T_SCENES = tuple(CFG.timeline.scene_starts)  # starts of scenes 1, 2, 3 (seconds, for the nominal film length)
FILM_LENGTH = CFG.timeline.film_length


# ---------------------------------------------------------------- scene 1 maths

def slit_positions(n: int) -> np.ndarray:
    return (np.arange(n) - (n - 1) / 2.0) * SLIT_SPACING


def paths_to_detector(n_screens: int, y_det: float) -> tuple[np.ndarray, np.ndarray]:
    """All paths S -> slits of the first n screens -> detector at (detector_x, y_det).

    Returns the vertices (n_paths, n_screens + 2, 2) and the phases 2 pi L / lambda."""
    ys = [slit_positions(n) for _, n in SCREENS[:n_screens]]
    xs = [x for x, _ in SCREENS[:n_screens]]
    rows = []
    for combo in itertools.product(*ys):
        pts = [(0.0, 0.0)] + [(x, y) for x, y in zip(xs, combo)] + [(DETECTOR_X, y_det)]
        rows.append(pts)
    verts = np.array(rows)
    seg = np.diff(verts, axis=1)
    length = np.sqrt((seg ** 2).sum(axis=2)).sum(axis=1)
    return verts, 2 * math.pi * length / WAVELENGTH


def amplitude(n_screens: int, y_det: float) -> complex:
    _, ph = paths_to_detector(n_screens, y_det)
    return complex(np.exp(1j * ph).sum())


def intensity_profile(n_screens: int, ys: np.ndarray) -> np.ndarray:
    return np.array([abs(amplitude(n_screens, y)) ** 2 for y in ys])


# ---------------------------------------------------------------- scene 2 maths

def action_phase(a: np.ndarray | float, c: float = C_PHASE) -> np.ndarray | float:
    """S(a) / hbar = c a^2."""
    return c * np.asarray(a) ** 2


def cornu_sum(a_max: float, c: float = C_PHASE, n: int = CFG.model.cornu_points) -> complex:
    a = np.linspace(-a_max, a_max, n)
    return complex(np.trapezoid(np.exp(1j * action_phase(a, c)), a))


def stationary_value(c: float) -> complex:
    return complex(math.sqrt(math.pi / c) * np.exp(1j * math.pi / 4))


# ---------------------------------------------------------------------- film

TEXT = {lang: load_texts(HERE, lang) for lang in ("en", "ru")}     # texts.toml


def num(x: float, fmt: str, lang: str) -> str:
    """A number for use in a text: decimal comma in Russian (also inside $...$, as {,})."""
    s = format(x, fmt)
    return s.replace(".", "{,}") if lang == "ru" else s


def put(text: str, **values: str) -> str:
    """Replace the placeholders {name} of a text (LaTeX braces stay untouched)."""
    for k, v in values.items():
        text = text.replace("{" + k + "}", v)
    return text


def smooth(x: float, a: float, b: float) -> float:
    u = min(max((x - a) / (b - a), 0.0), 1.0)
    return u * u * (3 - 2 * u)


def render(out: Path, size: tuple[int, int], fps: int, total: float, lang: str = "en", snap: float | None = None) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.colors as mcolors
    import matplotlib.pyplot as plt
    from matplotlib import cm
    from matplotlib.collections import LineCollection

    if snap is None and not shutil.which("ffmpeg"):
        sys.exit("ffmpeg is required")
    V, MD, TL, LY, F, FM, ST = CFG.video, CFG.model, CFG.timeline, CFG.layout, CFG.fonts, CFG.formats, CFG.style
    tx = TEXT[lang]
    W, H = size
    dpi = V.dpi
    sc = H / V.reference_height
    BG = ST.background
    k = total / FILM_LENGTH
    t1, t2, t3 = (x * k for x in T_SCENES)
    fig = plt.figure(figsize=(W / dpi, H / dpi), dpi=dpi, facecolor=BG)
    ax = fig.add_axes([0, 0, 1, 1])
    pz = fig.add_axes(LY.spiral_axes, facecolor="none")        # phasor / spiral panel

    # scene 1 geometry in pixels
    X0, XS = LY.slit_origin[0] * W, LY.slit_size[0] * W
    YC, YS = LY.slit_origin[1] * H, LY.slit_size[1] * H
    HALF = MD.screen_half_height
    ys_prof = np.linspace(-MD.profile_half_width, MD.profile_half_width, MD.profile_points)
    profiles = {n: intensity_profile(n, ys_prof) for n in (1, 2, 3)}
    for n in profiles:
        profiles[n] = profiles[n] / profiles[n].max()
    stage_t = tuple(t1 + s * k for s in TL.stage_starts) + (t2,)       # stage starts and end of scene 1

    def px(x, y):
        return X0 + XS * np.asarray(x), YC + YS * np.asarray(y)

    # scene 2/3 geometry: paths panel
    a_grid = np.linspace(-A_MAX, A_MAX, MD.a_grid_points)
    t_grid = np.linspace(0, 1, MD.path_points)

    frames = int(round(total * fps))
    ids = [int(round(snap * fps))] if snap is not None else range(frames)
    writer = None
    if snap is None:
        writer = subprocess.Popen(
            ["ffmpeg", "-y", "-v", "error", "-f", "rawvideo", "-pix_fmt", "rgba", "-s", f"{W}x{H}",
             "-r", str(fps), "-i", "-", "-c:v", "libx264", "-preset", V.preset, "-crf", str(V.crf),
             "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(out)], stdin=subprocess.PIPE)

    def hue(ph):
        return cm.hsv((np.asarray(ph) / (2 * math.pi)) % 1.0)

    def say(key: str, **fills: str) -> str:
        return put(tx[key], **fills)

    for k_ in ids:
        t = k_ / fps
        ax.clear()
        ax.set_xlim(0, W)
        ax.set_ylim(0, H)
        ax.axis("off")
        ax.add_patch(matplotlib.patches.Rectangle((0, 0), W, H, color=BG, zorder=0, lw=0))
        pz.clear()
        pz.set_facecolor("none")
        fig.texts.clear()
        scene = 1 if t < t2 else 2 if t < t3 else 3
        # fade between the scenes
        fade = 1.0
        for tb in (t2, t3):
            fade = min(fade, smooth(abs(t - tb), 0.0, TL.scene_fade_s * k)) if abs(t - tb) < TL.scene_fade_s * k else fade

        if scene == 1:
            stage = 1 if t < stage_t[1] else 2 if t < stage_t[2] else 3
            n_scr = stage
            ts = t - stage_t[stage - 1]
            y_det = MD.detector_amplitude * math.sin(2 * math.pi * ts / (TL.stage_length * k) * MD.detector_sweep + MD.detector_phase)
            # screens
            for i, (xs_, n_sl) in enumerate(SCREENS):
                x, _ = px(xs_, 0)
                active = i < n_scr
                alpha = ST.screen_alpha_on if active else ST.screen_alpha_off
                sl = slit_positions(n_sl)
                edges = [-HALF] + [v for s_ in sl for v in (s_ - MD.slit_half_width, s_ + MD.slit_half_width)] + [HALF]
                for e0, e1 in zip(edges[0::2], edges[1::2]):
                    _, ya = px(0, e0)
                    _, yb = px(0, e1)
                    ax.plot([x, x], [ya, yb], color=(*ST.screen_colour, alpha), lw=ST.screen_width * sc, solid_capstyle="butt", zorder=3)
            # source and detector
            xsrc, ysrc = px(0, 0)
            ax.scatter([xsrc], [ysrc], s=(ST.source_size * sc * 72 / dpi) ** 2, c=[tuple(ST.source_colour)], linewidths=0, zorder=6)
            xd, yd = px(DETECTOR_X, y_det)
            ax.plot([xd, xd], [YC - HALF * YS, YC + HALF * YS], color=tuple(ST.detector_colour), lw=ST.detector_width * sc, zorder=2)
            # paths
            verts, ph = paths_to_detector(n_scr, y_det)
            few = len(verts) < ST.few_paths
            segs = [np.column_stack(px(v[:, 0], v[:, 1])) for v in verts]
            cols = [(*c[:3], ST.path_alpha_few if few else ST.path_alpha_many) for c in hue(ph)]
            ax.add_collection(LineCollection(segs, colors=cols, linewidths=(ST.path_width_few if few else ST.path_width_many) * sc, zorder=4))
            # brightness profile along the detector
            prof = profiles[n_scr]
            xp = xd + LY.profile_scale * W * prof
            yp = YC + YS * ys_prof
            ax.fill_betweenx(yp, xd, xp, color=tuple(ST.profile_fill), zorder=3, lw=0)
            ax.plot(xp, yp, color=tuple(ST.profile_line), lw=ST.profile_width * sc, zorder=5)
            ax.scatter([xd], [yd], s=(ST.detector_marker_size * sc * 72 / dpi) ** 2, c=[tuple(ST.detector_marker)], linewidths=0, zorder=7)
            ax.text(xd + LY.intensity_label_pos[0] * W, YC - HALF * YS + LY.intensity_label_pos[1] * sc, tx["intensity_label"],
                    color=tuple(ST.intensity_colour), fontsize=F.intensity_label * sc, ha="center")
            # phasors
            vec = np.exp(1j * ph)
            pts = np.concatenate([[0], np.cumsum(vec)])
            tot = pts[-1]
            lim = max(LY.phasor_lim_min, min(abs(pts).max() * LY.phasor_lim_margin, LY.phasor_lim_max))
            pz.set_xlim(lim * LY.phasor_xlim[0], lim * LY.phasor_xlim[1])
            pz.set_ylim(-lim * LY.phasor_ylim, lim * LY.phasor_ylim)
            pz.set_aspect("equal")
            pz.axis("off")
            seg_p = [[(pts[i].real, pts[i].imag), (pts[i + 1].real, pts[i + 1].imag)] for i in range(len(vec))]
            pz.add_collection(LineCollection(seg_p, colors=hue(ph), linewidths=ST.phasor_width * sc))
            pz.annotate("", xy=(tot.real, tot.imag), xytext=(0, 0), arrowprops=dict(arrowstyle="-|>", color=ST.arrow_colour, lw=ST.arrow_width * sc))
            fig.text(*LY.phasor_title_pos, tx["phasor_title"], color=tuple(ST.phasor_title_colour), fontsize=F.phasor_title * sc)
            fig.text(*LY.readout_pos, say("readout_sum", amp=num(abs(tot), FM.amp, lang), amp2=num(abs(tot) ** 2, FM.amp2, lang)),
                     color=tuple(ST.readout_colour), fontsize=F.readout * sc, family="monospace")
            chip = f"chip_{stage}"
            fig.text(*LY.chip_pos, tx[chip], color=tuple(ST.chip_colour), fontsize=F.chip * sc)
            fig.text(*LY.formula_pos, tx["formula_sum"], color=tuple(ST.formula_colour), fontsize=F.formula * sc)
        else:
            c = C_PHASE if scene == 2 else C_PHASE * (1 + MD.c_growth * smooth(t, t3 + TL.classical_limit[0] * k, t3 + TL.classical_limit[1] * k))
            # paths panel on the left
            Xp0, Xp1 = LY.paths_panel_x[0] * W, LY.paths_panel_x[1] * W
            Yp0, Yp1 = LY.paths_panel_y[0] * H, LY.paths_panel_y[1] * H
            xs_pix = Xp0 + (Xp1 - Xp0) * t_grid
            a_reveal = A_MAX * (smooth(t, t2 + TL.reveal[0] * k, t2 + TL.reveal[1] * k) if scene == 2 else 1.0)
            r_stat = math.sqrt(math.pi / c)
            ax.plot([Xp0, Xp0], [Yp0, Yp1], color=tuple(ST.paths_axis_colour), lw=ST.paths_axis_width * sc)
            ax.plot([Xp1, Xp1], [Yp0, Yp1], color=tuple(ST.paths_axis_colour), lw=ST.paths_axis_width * sc)
            ymid = 0.5 * (Yp0 + Yp1)
            ax.scatter([Xp0, Xp1], [ymid, ymid], s=(ST.point_dot_size * sc * 72 / dpi) ** 2, c=[tuple(ST.point_a_dot), tuple(ST.point_b_dot)], linewidths=0, zorder=6)
            ax.text(Xp0 - LY.point_label_offset[0] * sc, ymid + LY.point_label_offset[1] * sc, tx["point_a"], color=tuple(ST.point_a_colour),
                    fontsize=F.point_label * sc, ha="center")
            ax.text(Xp1 + LY.point_label_offset[0] * sc, ymid + LY.point_label_offset[1] * sc, tx["point_b"], color=tuple(ST.point_b_colour),
                    fontsize=F.point_label * sc, ha="center")
            show = np.abs(a_grid) <= a_reveal
            segs, cols, lws = [], [], []
            for a in a_grid[show][::MD.path_stride]:
                path_y = ymid + (Yp1 - Yp0) * 0.5 * (a / A_MAX) * np.sin(math.pi * t_grid)
                segs.append(np.column_stack([xs_pix, path_y]))
                ph = action_phase(a, c)
                inside = abs(a) < r_stat
                cols.append((*hue(ph)[:3], ST.path_alpha_inside if inside else (ST.path_alpha_outside_scene3 if scene == 3 else ST.path_alpha_outside_scene2)))
                lws.append((ST.path_width_inside if inside else ST.path_width_outside) * sc)
            if segs:
                ax.add_collection(LineCollection(segs, colors=cols, linewidths=lws, zorder=4))
            ax.plot(xs_pix, np.full_like(xs_pix, ymid), color=ST.classical_colour, lw=ST.classical_width * sc, alpha=ST.classical_alpha, zorder=5)
            ax.text(0.5 * (Xp0 + Xp1), ymid + LY.classical_label_dy * sc, tx["classical_label"], color=tuple(ST.classical_label_colour),
                    fontsize=F.classical_label * sc, ha="center", zorder=8)
            # spiral
            n = MD.spiral_points
            aa = np.linspace(-a_reveal, a_reveal, n)
            z = np.cumsum(np.exp(1j * action_phase(aa, c))) * (aa[1] - aa[0] if a_reveal > 0 else 0)
            pz.set_xlim(*LY.spiral_xlim)
            pz.set_ylim(*LY.spiral_ylim)
            pz.set_aspect("equal")
            pz.axis("off")
            if a_reveal > 0:
                seg_z = np.column_stack([z.real, z.imag])
                lc = LineCollection(np.stack([seg_z[:-1], seg_z[1:]], axis=1),
                                    colors=hue(action_phase(aa[:-1], c) * ST.spiral_hue_rate + ST.spiral_hue_offset), linewidths=ST.spiral_width * sc)
                pz.add_collection(lc)
                tot = z[-1]
                pz.annotate("", xy=(tot.real, tot.imag), xytext=(z[0].real, z[0].imag),
                            arrowprops=dict(arrowstyle="-|>", color=ST.arrow_colour, lw=ST.spiral_arrow_width * sc))
                st = stationary_value(c)
                pz.plot([0, st.real], [0, st.imag], color=tuple(ST.stationary_colour), lw=ST.stationary_width * sc, ls=(0, tuple(ST.stationary_dash)))
            fig.text(*LY.spiral_title_pos, tx["spiral_title"], color=tuple(ST.spiral_title_colour), fontsize=F.spiral_title * sc)
            fig.text(*LY.spiral_title_sub_pos, tx["spiral_title_sub"], color=tuple(ST.spiral_title_sub_colour), fontsize=F.spiral_title_sub * sc)
            amp = abs(z[-1]) if a_reveal > 0 else 0.0
            fig.text(*LY.spiral_readout_pos, say("readout_spiral", amp=num(amp, FM.amp_spiral, lang), limit=num(r_stat, FM.limit, lang)),
                     color=tuple(ST.spiral_readout_colour), fontsize=F.spiral_readout * sc, family="monospace")
            fig.text(*LY.spiral_c_pos, say("readout_c", c_value=num(c, FM.c, lang)), color=tuple(ST.spiral_c_colour), fontsize=F.spiral_c * sc, family="monospace")
            fig.text(*LY.action_title_pos, tx["action_title"], color=tuple(ST.action_title_colour), fontsize=F.action_title * sc)
            fig.text(*LY.action_formula_pos, tx["action_formula"], color=tuple(ST.action_formula_colour), fontsize=F.action_formula * sc)
        # captions
        if scene == 1:
            cap = "cap_slits"
            if t > t2 - TL.caption_change * k:
                cap = "cap_all_paths"
        elif scene == 2:
            cap = "cap_cancel"
        else:
            cap = "cap_classical"
        ax.add_patch(matplotlib.patches.Rectangle((0, 0), LY.caption_box[0] * W, LY.caption_box[1] * H, color=tuple(ST.caption_box_colour), zorder=11, lw=0))
        fig.text(*LY.caption_pos, tx[cap], color=tuple(ST.caption_colour), fontsize=F.caption * sc)
        if fade < 1.0:
            ax.add_patch(matplotlib.patches.Rectangle((0, 0), W, H, color=(*ST.fade_colour, 1 - fade), zorder=20, lw=0))
        fig.canvas.draw()
        if writer is None:
            fig.savefig(out, dpi=dpi, facecolor=fig.get_facecolor())
        else:
            writer.stdin.write(np.asarray(fig.canvas.buffer_rgba()).tobytes())
    if writer is not None:
        writer.stdin.close()
        writer.wait()
    plt.close(fig)
    print(f"wrote {out}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--lang", choices=("en", "ru"), default="en")
    ap.add_argument("--preview", action="store_true")
    ap.add_argument("--snapshot", type=float, default=None)
    ap.add_argument("--seconds", type=float, default=FILM_LENGTH)
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--config", type=Path, default=None, help="another configuration file instead of config.toml")
    ap.add_argument("--set", action="append", default=[], metavar="SECTION.KEY=VALUE", help="override one configuration value")
    args = ap.parse_args()
    V = CFG.video
    out = args.out or HERE / "media" / f"path_integral_{args.lang}.mp4"
    out.parent.mkdir(parents=True, exist_ok=True)
    if args.snapshot is not None:
        render(out.with_suffix(".png"), (V.width, V.height), V.fps, args.seconds, args.lang, snap=args.snapshot)
    elif args.preview:
        render(out.with_name(out.stem + "_preview.mp4"), (V.preview_width, V.preview_height), V.preview_fps, args.seconds, args.lang)
    else:
        render(out, (V.width, V.height), V.fps, args.seconds, args.lang)


if __name__ == "__main__":
    main()
