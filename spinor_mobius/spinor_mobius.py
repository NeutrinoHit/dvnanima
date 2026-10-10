"""Spin 1/2: a rotation by 360 degrees changes the sign of a spinor, 720 degrees restore it.

Three linked views of one rotation angle phi (0 -> 720 degrees):

* a dial: a body rotated by phi about the z axis;
* a Moebius strip: a vector lying across the strip is carried once around the core
  circle when the body turns by 360 degrees.  It comes back reversed, and only after
  the second turn (720 degrees) it coincides with the starting vector;
* the complex plane: the spinor component psi = exp(-i phi / 2) of the Dirac equation
  example in the book.  phi = 360 degrees gives psi = -1, phi = 720 degrees gives psi = +1.

Strip: P(u, v) = ((1 + v cos(u/2)) cos u, (1 + v cos(u/2)) sin u, v sin(u/2)),
vector across the strip at u: w(u) = (cos(u/2) cos u, cos(u/2) sin u, sin(u/2)),
so that w(u + 2 pi) = -w(u) and w(u + 4 pi) = w(u).  Here u = phi.

All the numbers are in config.toml and all the words in texts.toml (see ../dvconfig.py for --config / --set).

Usage:
    python spinor_mobius.py --lang en                # film -> media/spinor_mobius_en.mp4
    python spinor_mobius.py --lang ru
    python spinor_mobius.py --lang en --preview
    python spinor_mobius.py --lang en --snapshot 6   # one PNG at film time 6 s
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
HALF_WIDTH = CFG.model.half_width            # half-width of the strip
TEXT = {lang: load_texts(HERE, lang) for lang in ("en", "ru")}     # texts.toml


def strip_point(u: np.ndarray, v: np.ndarray) -> np.ndarray:
    r = 1.0 + v * np.cos(u / 2)
    return np.stack([r * np.cos(u), r * np.sin(u), v * np.sin(u / 2)], axis=-1)


def across(u: float) -> np.ndarray:
    """Unit vector across the strip at parameter u (flips sign after u -> u + 2 pi)."""
    return np.array([math.cos(u / 2) * math.cos(u), math.cos(u / 2) * math.sin(u), math.sin(u / 2)])


def spinor_component(phi: float) -> complex:
    return complex(math.cos(phi / 2), -math.sin(phi / 2))


def angle_at(t: float, total: float) -> float:
    """phi(t) in radians: 0 -> 2pi -> 4pi with a short hold after each turn."""
    hold = CFG.timeline.hold_s
    move = (total - 2 * hold - CFG.timeline.slack_s) / 2
    t = max(t - CFG.timeline.start_delay_s, 0.0)
    def ease(x: float) -> float:
        x = min(max(x, 0.0), 1.0)
        return x * x * (3 - 2 * x)
    if t < move:
        return 2 * math.pi * ease(t / move)
    if t < move + hold:
        return 2 * math.pi
    if t < 2 * move + hold:
        return 2 * math.pi + 2 * math.pi * ease((t - move - hold) / move)
    return 4 * math.pi


def render(out: Path, size: tuple[int, int], fps: int, seconds: float, lang: str = "en", snap: float | None = None) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import cm
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

    if snap is None and not shutil.which("ffmpeg"):
        sys.exit("ffmpeg is required")
    tx = TEXT[lang]
    V, TL, MD, CM, LY, F, ST = CFG.video, CFG.timeline, CFG.model, CFG.camera, CFG.layout, CFG.fonts, CFG.style
    W, H = size
    dpi = V.dpi
    sc = H / V.reference_height
    bg, fg, dim = ST.background, ST.text, ST.dim
    fig = plt.figure(figsize=(W / dpi, H / dpi), dpi=dpi, facecolor=bg)
    ax3 = fig.add_axes(LY.axes_3d, projection="3d", facecolor=bg, computed_zorder=False)
    axd = fig.add_axes(LY.axes_dial, facecolor=bg)
    axc = fig.add_axes(LY.axes_spinor, facecolor=bg)
    axt = fig.add_axes(LY.axes_text, facecolor=bg)
    u_grid, v_grid = np.meshgrid(np.linspace(0, 2 * np.pi, MD.u_samples), np.linspace(-HALF_WIDTH, HALF_WIDTH, MD.v_samples))
    P = strip_point(u_grid, v_grid)
    lo, span = ST.colormap_range
    colors = getattr(cm, ST.colormap)(lo + span * (u_grid / (2 * np.pi)))
    colors[..., 3] = ST.surface_alpha

    frames = int(round(seconds * fps))
    ids = [int(round(snap * fps))] if snap is not None else range(frames)
    writer = None
    if snap is None:
        writer = subprocess.Popen(
            ["ffmpeg", "-y", "-v", "error", "-f", "rawvideo", "-pix_fmt", "rgba", "-s", f"{W}x{H}",
             "-r", str(fps), "-i", "-", "-c:v", "libx264", "-preset", V.preset, "-crf", str(V.crf),
             "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(out)], stdin=subprocess.PIPE)
    trace: list[complex] = []
    last_k = -1
    for k in ids:
        t = k / fps
        phi = angle_at(t, seconds)
        # --- 3D strip
        ax3.clear()
        ax3.set_facecolor(bg)
        ax3.plot_surface(P[..., 0], P[..., 1], P[..., 2], facecolors=colors, shade=False,
                         rstride=1, cstride=1, linewidth=0, antialiased=True, zorder=1)
        ax3.plot_wireframe(P[..., 0], P[..., 1], P[..., 2], rstride=ST.mesh_stride[0], cstride=ST.mesh_stride[1], color=ST.mesh,
                           linewidth=ST.mesh_width, alpha=ST.mesh_alpha)
        core = strip_point(np.linspace(0, 2 * np.pi, MD.core_samples), np.zeros(MD.core_samples))
        ax3.plot(core[:, 0], core[:, 1], core[:, 2], color=ST.mesh, lw=ST.core_width, alpha=ST.core_alpha)
        c0 = strip_point(np.array(0.0), np.array(0.0))
        w0 = across(0.0)
        a0, b0 = c0 - HALF_WIDTH * w0, c0 + HALF_WIDTH * w0
        ax3.plot([a0[0], b0[0]], [a0[1], b0[1]], [a0[2], b0[2]], color=ST.reference, lw=ST.reference_width, alpha=ST.reference_alpha, zorder=10)
        ax3.scatter(*b0, color=ST.reference, s=ST.reference_marker_size, marker="^", zorder=11)
        cu = strip_point(np.array(phi), np.array(0.0))
        wu = across(phi)
        au, bu = cu - HALF_WIDTH * wu, cu + HALF_WIDTH * wu
        ax3.plot([au[0], bu[0]], [au[1], bu[1]], [au[2], bu[2]], color=ST.accent, lw=ST.vector_width, zorder=12)
        ax3.scatter(*bu, color=ST.accent, s=ST.vector_marker_size, marker="^", zorder=13)
        ax3.set_axis_off()
        ax3.set_xlim(*CM.xlim)
        ax3.set_ylim(*CM.ylim)
        ax3.set_zlim(*CM.zlim)
        ax3.set_box_aspect(tuple(CM.box_aspect))
        ax3.view_init(elev=CM.elevation_deg, azim=CM.azimuth_deg + CM.azimuth_swing_deg * math.sin(2 * math.pi * t / seconds))
        # --- dial
        axd.clear()
        axd.set_facecolor(bg)
        ang = np.linspace(0, 2 * np.pi, MD.dial_samples)
        axd.plot(np.cos(ang), np.sin(ang), color=ST.circle, lw=ST.circle_width)
        axd.annotate("", xy=(1, 0), xytext=(0, 0), arrowprops=dict(arrowstyle="-|>", color=ST.reference, lw=ST.dial_reference_width, alpha=ST.dial_reference_alpha))
        axd.annotate("", xy=(math.cos(phi), math.sin(phi)), xytext=(0, 0),
                     arrowprops=dict(arrowstyle="-|>", color=ST.accent, lw=ST.dial_arrow_width))
        turns = np.linspace(0, phi, MD.turn_samples)
        axd.plot(LY.dial_arc_radius * np.cos(turns), LY.dial_arc_radius * np.sin(turns), color=ST.accent, lw=ST.dial_arc_width, alpha=ST.dial_arc_alpha)
        axd.set_xlim(-LY.dial_lim, LY.dial_lim)
        axd.set_ylim(-LY.dial_lim, LY.dial_lim)
        axd.set_aspect("equal")
        axd.axis("off")
        axd.set_title(tx["dial_title"], color=fg, fontsize=F.panel_title * sc)
        # --- spinor
        axc.clear()
        axc.set_facecolor(bg)
        psi = spinor_component(phi)
        if k != last_k and (not trace or abs(trace[-1] - psi) > 1e-9):
            trace.append(psi)
        last_k = k
        axc.plot(np.cos(ang), np.sin(ang), color=ST.circle, lw=ST.circle_width)
        if len(trace) > 1 and snap is None:
            tr = np.array(trace)
            axc.plot(tr.real, tr.imag, color=ST.blue, lw=ST.trace_width, alpha=ST.trace_alpha)
        elif snap is not None:
            ph = np.linspace(0, phi, MD.spinor_samples)
            axc.plot(np.cos(ph / 2), -np.sin(ph / 2), color=ST.blue, lw=ST.trace_width, alpha=ST.trace_alpha)
        axc.annotate("", xy=(psi.real, psi.imag), xytext=(0, 0),
                     arrowprops=dict(arrowstyle="-|>", color=ST.blue, lw=ST.spinor_arrow_width))
        axc.scatter([1, -1], [0, 0], s=ST.point_size, color=dim)
        axc.text(*LY.plus_one_pos, tx["plus_one"], color=dim, fontsize=F.plus_minus * sc)
        axc.text(*LY.minus_one_pos, tx["minus_one"], color=dim, fontsize=F.plus_minus * sc)
        axc.set_xlim(-LY.spinor_lim, LY.spinor_lim)
        axc.set_ylim(-LY.spinor_lim, LY.spinor_lim)
        axc.set_aspect("equal")
        axc.axis("off")
        axc.set_title(tx["spinor_title"], color=fg, fontsize=F.panel_title * sc)
        # --- text
        axt.clear()
        axt.axis("off")
        deg = math.degrees(phi)
        if deg < TL.message_tol_deg:
            msg = tx["msg_start"]
        elif abs(deg - 360) < TL.message_tol_deg:
            msg = tx["msg_360"]
        elif abs(deg - 720) < TL.message_tol_deg:
            msg = tx["msg_720"]
        else:
            msg = ""
        axt.text(*LY.phi_pos, tx["phi_line"].replace("{deg}", f"{deg:.0f}"), color=ST.accent, fontsize=F.phi * sc)
        axt.text(*LY.message_pos, msg, color=fg, fontsize=F.message * sc)
        fig.texts.clear()
        fig.text(*LY.title_pos, tx["title"], color=ST.title_colour, fontsize=F.title * sc)
        fig.text(*LY.footer_pos, tx["footer"], color=dim, fontsize=F.footer * sc)
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
    ap.add_argument("--seconds", type=float, default=CFG.timeline.film_length)
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--config", type=Path, default=None, help="another configuration file instead of config.toml")
    ap.add_argument("--set", action="append", default=[], metavar="SECTION.KEY=VALUE", help="override one configuration value")
    args = ap.parse_args()
    V = CFG.video
    out = args.out or HERE / "media" / f"spinor_mobius_{args.lang}.mp4"
    out.parent.mkdir(parents=True, exist_ok=True)
    if args.snapshot is not None:
        render(out.with_suffix(".png"), (V.width, V.height), V.fps, args.seconds, args.lang, snap=args.snapshot)
    elif args.preview:
        render(out.with_name(out.stem + "_preview.mp4"), (V.preview_width, V.preview_height), V.preview_fps, CFG.timeline.preview_length, args.lang)
    else:
        render(out, (V.width, V.height), V.fps, args.seconds, args.lang)


if __name__ == "__main__":
    main()
