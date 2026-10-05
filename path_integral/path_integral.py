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

Usage:
    python path_integral.py                 # film -> media/path_integral.mp4
    python path_integral.py --preview
    python path_integral.py --snapshot 12   # one PNG at film time 12 s
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

HERE = Path(__file__).resolve().parent

SCREENS = ((0.25, 3), (0.50, 7), (0.75, 5))
SLIT_SPACING = 0.085
WAVELENGTH = 0.0165
C_PHASE = 9.0                 # c = m pi^2 / (4 T hbar) in scene 2, 1/a^2 units
A_MAX = 1.8
T_SCENES = (0.0, 15.0, 29.0)  # starts of scenes 1, 2, 3 (seconds, for a 40 s film)


# ---------------------------------------------------------------- scene 1 maths

def slit_positions(n: int) -> np.ndarray:
    return (np.arange(n) - (n - 1) / 2.0) * SLIT_SPACING


def paths_to_detector(n_screens: int, y_det: float) -> tuple[np.ndarray, np.ndarray]:
    """All paths S -> slits of the first n screens -> detector at (1, y_det).

    Returns the vertices (n_paths, n_screens + 2, 2) and the phases 2 pi L / lambda."""
    ys = [slit_positions(n) for _, n in SCREENS[:n_screens]]
    xs = [x for x, _ in SCREENS[:n_screens]]
    rows = []
    for combo in itertools.product(*ys):
        pts = [(0.0, 0.0)] + [(x, y) for x, y in zip(xs, combo)] + [(1.0, y_det)]
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


def cornu_sum(a_max: float, c: float = C_PHASE, n: int = 4001) -> complex:
    a = np.linspace(-a_max, a_max, n)
    return complex(np.trapezoid(np.exp(1j * action_phase(a, c)), a))


def stationary_value(c: float) -> complex:
    return complex(math.sqrt(math.pi / c) * np.exp(1j * math.pi / 4))


# ---------------------------------------------------------------------- film

def smooth(x: float, a: float, b: float) -> float:
    u = min(max((x - a) / (b - a), 0.0), 1.0)
    return u * u * (3 - 2 * u)


def render(out: Path, size: tuple[int, int], fps: int, total: float, snap: float | None = None) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.colors as mcolors
    import matplotlib.pyplot as plt
    from matplotlib import cm
    from matplotlib.collections import LineCollection

    if snap is None and not shutil.which("ffmpeg"):
        sys.exit("ffmpeg is required")
    W, H = size
    dpi = 100
    sc = H / 720.0
    k = total / 40.0
    t1, t2, t3 = (x * k for x in T_SCENES)
    fig = plt.figure(figsize=(W / dpi, H / dpi), dpi=dpi, facecolor="#03060c")
    ax = fig.add_axes([0, 0, 1, 1])
    pz = fig.add_axes([0.705, 0.25, 0.285, 0.52], facecolor="none")        # phasor / spiral panel

    # scene 1 geometry in pixels
    X0, XS = 0.06 * W, 0.50 * W
    YC, YS = 0.52 * H, 0.70 * H
    ys_prof = np.linspace(-0.45, 0.45, 181)
    profiles = {n: intensity_profile(n, ys_prof) for n in (1, 2, 3)}
    for n in profiles:
        profiles[n] = profiles[n] / profiles[n].max()
    stage_t = (t1, t1 + 4.8 * k, t1 + 9.6 * k, t2)       # stage starts and end of scene 1

    def px(x, y):
        return X0 + XS * np.asarray(x), YC + YS * np.asarray(y)

    # scene 2/3 geometry: paths panel
    a_grid = np.linspace(-A_MAX, A_MAX, 401)
    t_grid = np.linspace(0, 1, 60)

    frames = int(round(total * fps))
    ids = [int(round(snap * fps))] if snap is not None else range(frames)
    writer = None
    if snap is None:
        writer = subprocess.Popen(
            ["ffmpeg", "-y", "-v", "error", "-f", "rawvideo", "-pix_fmt", "rgba", "-s", f"{W}x{H}",
             "-r", str(fps), "-i", "-", "-c:v", "libx264", "-preset", "slow", "-crf", "19",
             "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(out)], stdin=subprocess.PIPE)

    def hue(ph):
        return cm.hsv((np.asarray(ph) / (2 * math.pi)) % 1.0)

    for k_ in ids:
        t = k_ / fps
        ax.clear()
        ax.set_xlim(0, W)
        ax.set_ylim(0, H)
        ax.axis("off")
        ax.add_patch(matplotlib.patches.Rectangle((0, 0), W, H, color="#03060c", zorder=0, lw=0))
        pz.clear()
        pz.set_facecolor("none")
        fig.texts.clear()
        scene = 1 if t < t2 else 2 if t < t3 else 3
        # fade between the scenes
        fade = 1.0
        for tb in (t2, t3):
            fade = min(fade, smooth(abs(t - tb), 0.0, 0.5 * k)) if abs(t - tb) < 0.5 * k else fade

        if scene == 1:
            stage = 1 if t < stage_t[1] else 2 if t < stage_t[2] else 3
            n_scr = stage
            ts = t - stage_t[stage - 1]
            y_det = 0.36 * math.sin(2 * math.pi * ts / (4.8 * k) * 0.9 + 0.4)
            # screens
            for i, (xs_, n_sl) in enumerate(SCREENS):
                x, _ = px(xs_, 0)
                active = i < n_scr
                alpha = 0.9 if active else 0.12
                sl = slit_positions(n_sl)
                edges = [-0.5] + [v for s_ in sl for v in (s_ - 0.012, s_ + 0.012)] + [0.5]
                for e0, e1 in zip(edges[0::2], edges[1::2]):
                    _, ya = px(0, e0)
                    _, yb = px(0, e1)
                    ax.plot([x, x], [ya, yb], color=(0.85, 0.9, 1.0, alpha), lw=3.0 * sc, solid_capstyle="butt", zorder=3)
            # source and detector
            xsrc, ysrc = px(0, 0)
            ax.scatter([xsrc], [ysrc], s=(14 * sc * 72 / dpi) ** 2, c=[(1, 0.85, 0.4, 1)], linewidths=0, zorder=6)
            xd, yd = px(1.0, y_det)
            ax.plot([xd, xd], [YC - 0.5 * YS, YC + 0.5 * YS], color=(0.7, 0.8, 0.95, 0.35), lw=1.0 * sc, zorder=2)
            # paths
            verts, ph = paths_to_detector(n_scr, y_det)
            segs = [np.column_stack(px(v[:, 0], v[:, 1])) for v in verts]
            cols = [(*c[:3], 0.50 if len(verts) < 30 else 0.30) for c in hue(ph)]
            ax.add_collection(LineCollection(segs, colors=cols, linewidths=(1.5 if len(verts) < 30 else 0.9) * sc, zorder=4))
            # brightness profile along the detector
            prof = profiles[n_scr]
            xp = xd + 0.16 * W * prof
            yp = YC + YS * ys_prof
            ax.fill_betweenx(yp, xd, xp, color=(1.0, 0.8, 0.4, 0.35), zorder=3, lw=0)
            ax.plot(xp, yp, color=(1.0, 0.85, 0.5, 0.9), lw=1.6 * sc, zorder=5)
            ax.scatter([xd], [yd], s=(11 * sc * 72 / dpi) ** 2, c=[(1, 1, 1, 1)], linewidths=0, zorder=7)
            ax.text(xd + 0.17 * W, YC - 0.5 * YS - 18 * sc, r"$|\mathcal{A}|^{2}$", color=(1.0, 0.85, 0.5, 0.95), fontsize=13 * sc, ha="center")
            # phasors
            vec = np.exp(1j * ph)
            pts = np.concatenate([[0], np.cumsum(vec)])
            tot = pts[-1]
            lim = max(3.0, min(abs(pts).max() * 1.1, 110.0))
            pz.set_xlim(-lim * 0.35, lim * 1.0)
            pz.set_ylim(-lim * 0.7, lim * 0.7)
            pz.set_aspect("equal")
            pz.axis("off")
            seg_p = [[(pts[i].real, pts[i].imag), (pts[i + 1].real, pts[i + 1].imag)] for i in range(len(vec))]
            pz.add_collection(LineCollection(seg_p, colors=hue(ph), linewidths=2.0 * sc))
            pz.annotate("", xy=(tot.real, tot.imag), xytext=(0, 0), arrowprops=dict(arrowstyle="-|>", color="white", lw=2.6 * sc))
            fig.text(0.715, 0.835, "sum of phasors / сумма фазоров", color=(0.82, 0.9, 1, 0.95), fontsize=12 * sc)
            fig.text(0.715, 0.795, rf"$|\mathcal{{A}}|$ = {abs(tot):.1f}     $|\mathcal{{A}}|^{{2}}$ = {abs(tot) ** 2:.0f}", color=(1.0, 0.85, 0.5, 1), fontsize=12 * sc, family="monospace")
            n_paths = len(vec)
            chips = {1: "3 slits: 3 paths", 2: "3 × 7 = 21 paths", 3: "3 × 7 × 5 = 105 paths"}[stage]
            fig.text(0.03, 0.915, chips, color=(0.9, 0.95, 1, 0.95), fontsize=18 * sc)
            fig.text(0.03, 0.872, {1: "3 щели: 3 пути", 2: "3 × 7 = 21 путь", 3: "3 × 7 × 5 = 105 путей"}[stage], color=(0.6, 0.68, 0.8, 0.9), fontsize=12 * sc)
            cap = ("the amplitude is the sum over all paths; each path contributes a phase 2πL/λ",
                   "амплитуда есть сумма по всем путям; каждый путь даёт фазу 2πL/λ")
            fig.text(0.715, 0.20, r"$\mathcal{A}=\sum\,\mathcal{A}_{1,i}\,\mathcal{A}_{2,j}\,\mathcal{A}_{3,k}$", color=(0.9, 0.95, 1, 0.9), fontsize=12.5 * sc)
        else:
            c = C_PHASE if scene == 2 else C_PHASE * (1 + 9 * smooth(t, t3 + 1.0 * k, t3 + 8.5 * k))
            # paths panel on the left
            Xp0, Xp1 = 0.07 * W, 0.62 * W
            Yp0, Yp1 = 0.23 * H, 0.86 * H
            xs_pix = Xp0 + (Xp1 - Xp0) * t_grid
            a_reveal = A_MAX * (smooth(t, t2 + 0.8 * k, t2 + 11.5 * k) if scene == 2 else 1.0)
            r_stat = math.sqrt(math.pi / c)
            ax.plot([Xp0, Xp0], [Yp0, Yp1], color=(0.7, 0.8, 0.95, 0.3), lw=1.0 * sc)
            ax.plot([Xp1, Xp1], [Yp0, Yp1], color=(0.7, 0.8, 0.95, 0.3), lw=1.0 * sc)
            ymid = 0.5 * (Yp0 + Yp1)
            ax.scatter([Xp0, Xp1], [ymid, ymid], s=(13 * sc * 72 / dpi) ** 2, c=[(1, 0.85, 0.4, 1), (0.6, 0.9, 1, 1)], linewidths=0, zorder=6)
            ax.text(Xp0 - 6 * sc, ymid - 26 * sc, "A", color=(1, 0.85, 0.5, 1), fontsize=13 * sc, ha="center")
            ax.text(Xp1 + 6 * sc, ymid - 26 * sc, "B", color=(0.6, 0.9, 1, 1), fontsize=13 * sc, ha="center")
            show = np.abs(a_grid) <= a_reveal
            segs, cols, lws = [], [], []
            for a in a_grid[show][::2]:
                ypix = ymid + (Yp1 - Yp0) * 0.5 * (a / A_MAX) * math.sin(math.pi * 1.0) * 0 + 0
                path_y = ymid + (Yp1 - Yp0) * 0.5 * (a / A_MAX) * np.sin(math.pi * t_grid)
                segs.append(np.column_stack([xs_pix, path_y]))
                ph = action_phase(a, c)
                inside = abs(a) < r_stat
                cols.append((*hue(ph)[:3], 0.9 if inside else (0.28 if scene == 3 else 0.5)))
                lws.append((2.2 if inside else 0.8) * sc)
            if segs:
                ax.add_collection(LineCollection(segs, colors=cols, linewidths=lws, zorder=4))
            ax.plot(xs_pix, np.full_like(xs_pix, ymid), color="white", lw=2.6 * sc, alpha=0.95, zorder=5)
            ax.text(0.5 * (Xp0 + Xp1), ymid + 12 * sc, "classical path / классический путь (a = 0)", color=(1, 1, 1, 0.9), fontsize=10 * sc, ha="center", zorder=8)
            # spiral
            n = 1200
            aa = np.linspace(-a_reveal, a_reveal, n)
            z = np.cumsum(np.exp(1j * action_phase(aa, c))) * (aa[1] - aa[0] if a_reveal > 0 else 0)
            lim = 0.9 if scene == 2 else 0.9
            sc_z = 1.0
            pz.set_xlim(-0.40, 0.80)
            pz.set_ylim(-0.55, 0.55)
            pz.set_aspect("equal")
            pz.axis("off")
            if a_reveal > 0:
                seg_z = np.column_stack([z.real, z.imag])
                lc = LineCollection(np.stack([seg_z[:-1], seg_z[1:]], axis=1), colors=hue(action_phase(aa[:-1], c) * 0.3 + 1.0), linewidths=1.8 * sc)
                pz.add_collection(lc)
                tot = z[-1]
                pz.annotate("", xy=(tot.real, tot.imag), xytext=(z[0].real, z[0].imag), arrowprops=dict(arrowstyle="-|>", color="white", lw=2.4 * sc))
                st = stationary_value(c)
                pz.plot([0, st.real], [0, st.imag], color=(1, 0.85, 0.4, 0.6), lw=1.0 * sc, ls=(0, (3, 3)))
            fig.text(0.715, 0.835, r"Cornu spiral: $\sum e^{iS/\hbar}$", color=(0.82, 0.9, 1, 0.95), fontsize=12.5 * sc)
            fig.text(0.715, 0.795, "спираль Корню: сумма по путям", color=(0.6, 0.68, 0.8, 0.9), fontsize=10.5 * sc)
            amp = abs(z[-1]) if a_reveal > 0 else 0.0
            fig.text(0.715, 0.215, rf"$|\mathcal{{A}}|$ = {amp:.2f}   (limit $\sqrt{{\pi/c}}$ = {r_stat:.2f})", color=(1.0, 0.85, 0.5, 1), fontsize=10.5 * sc, family="monospace")
            fig.text(0.715, 0.17, f"ħ ∝ 1/c;   c = {c:5.1f}", color=(0.8, 0.88, 1.0, 0.95), fontsize=10.5 * sc, family="monospace")
            fig.text(0.03, 0.915, r"$S=\int L\,dt$,   phase $=S/\hbar$", color=(0.9, 0.95, 1, 0.95), fontsize=18 * sc)
            fig.text(0.03, 0.872, r"$S(a)=\frac{m\pi^{2}}{4T}\,a^{2}$  →  phase $=c\,a^{2}$", color=(0.6, 0.68, 0.8, 0.9), fontsize=11 * sc)
        # captions
        if scene == 1:
            cap = ("paths through screens with 3, 7, 5 slits: 3 × 7 × 5 = 105 amplitudes", "пути через экраны с 3, 7, 5 щелями: 3 × 7 × 5 = 105 амплитуд")
            if t > t2 - 2.2 * k:
                cap = ("infinitely many screens with infinitely many slits: all paths", "бесконечно много экранов с бесконечным числом щелей: все пути")
        elif scene == 2:
            cap = ("far from the classical path the phases rotate quickly and cancel", "вдали от классического пути фазы быстро вращаются и взаимно гасятся")
        else:
            cap = ("ħ → 0: only paths near the one of least action survive", "ħ → 0: остаются лишь пути вблизи пути наименьшего действия")
        ax.add_patch(matplotlib.patches.Rectangle((0, 0), 0.70 * W, 0.105 * H, color=(0.01, 0.02, 0.05, 0.8), zorder=11, lw=0))
        fig.text(0.03, 0.053, cap[0], color=(0.88, 0.93, 1.0, 0.93), fontsize=11.5 * sc)
        fig.text(0.03, 0.02, cap[1], color=(0.6, 0.68, 0.8, 0.9), fontsize=9.8 * sc)
        if fade < 1.0:
            ax.add_patch(matplotlib.patches.Rectangle((0, 0), W, H, color=(0.01, 0.02, 0.05, 1 - fade), zorder=20, lw=0))
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
    ap.add_argument("--preview", action="store_true")
    ap.add_argument("--snapshot", type=float, default=None)
    ap.add_argument("--seconds", type=float, default=40.0)
    ap.add_argument("--out", type=Path, default=HERE / "media" / "path_integral.mp4")
    args = ap.parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    if args.snapshot is not None:
        render(args.out.with_suffix(".png"), (1280, 720), 30, args.seconds, snap=args.snapshot)
    elif args.preview:
        render(args.out.with_name("path_integral_preview.mp4"), (640, 360), 15, args.seconds)
    else:
        render(args.out, (1280, 720), 30, args.seconds)


if __name__ == "__main__":
    main()
