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

Usage:
    python spinor_mobius.py                # film -> media/spinor_mobius.mp4
    python spinor_mobius.py --preview
    python spinor_mobius.py --snapshot 6   # one PNG at film time 6 s
"""

from __future__ import annotations

import argparse
import math
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
HALF_WIDTH = 0.5


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
    hold = 1.6
    move = (total - 2 * hold - 1.0) / 2
    t = max(t - 0.5, 0.0)
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


def render(out: Path, size: tuple[int, int], fps: int, seconds: float, snap: float | None = None) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import cm
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

    if snap is None and not shutil.which("ffmpeg"):
        sys.exit("ffmpeg is required")
    W, H = size
    dpi = 100
    sc = H / 720.0
    bg, fg, dim = "#0a0d14", "#e8edf5", "#8b96ad"
    fig = plt.figure(figsize=(W / dpi, H / dpi), dpi=dpi, facecolor=bg)
    ax3 = fig.add_axes([0.0, 0.03, 0.56, 0.86], projection="3d", facecolor=bg, computed_zorder=False)
    axd = fig.add_axes([0.60, 0.50, 0.17, 0.30], facecolor=bg)
    axc = fig.add_axes([0.80, 0.50, 0.17, 0.30], facecolor=bg)
    axt = fig.add_axes([0.60, 0.08, 0.37, 0.28], facecolor=bg)
    u_grid, v_grid = np.meshgrid(np.linspace(0, 2 * np.pi, 120), np.linspace(-HALF_WIDTH, HALF_WIDTH, 9))
    P = strip_point(u_grid, v_grid)
    colors = cm.twilight_shifted(0.15 + 0.7 * (u_grid / (2 * np.pi)))
    colors[..., 3] = 0.42

    frames = int(round(seconds * fps))
    ids = [int(round(snap * fps))] if snap is not None else range(frames)
    writer = None
    if snap is None:
        writer = subprocess.Popen(
            ["ffmpeg", "-y", "-v", "error", "-f", "rawvideo", "-pix_fmt", "rgba", "-s", f"{W}x{H}",
             "-r", str(fps), "-i", "-", "-c:v", "libx264", "-preset", "slow", "-crf", "20",
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
        ax3.plot_wireframe(P[..., 0], P[..., 1], P[..., 2], rstride=8, cstride=6, color="#c8d3e6",
                           linewidth=0.25, alpha=0.35)
        core = strip_point(np.linspace(0, 2 * np.pi, 200), np.zeros(200))
        ax3.plot(core[:, 0], core[:, 1], core[:, 2], color="#c8d3e6", lw=0.8, alpha=0.6)
        c0 = strip_point(np.array(0.0), np.array(0.0))
        w0 = across(0.0)
        a0, b0 = c0 - HALF_WIDTH * w0, c0 + HALF_WIDTH * w0
        ax3.plot([a0[0], b0[0]], [a0[1], b0[1]], [a0[2], b0[2]], color="#ffffff", lw=3.0, alpha=0.9, zorder=10)
        ax3.scatter(*b0, color="#ffffff", s=70, marker="^", zorder=11)
        cu = strip_point(np.array(phi), np.array(0.0))
        wu = across(phi)
        au, bu = cu - HALF_WIDTH * wu, cu + HALF_WIDTH * wu
        ax3.plot([au[0], bu[0]], [au[1], bu[1]], [au[2], bu[2]], color="#ff9a1f", lw=4.0, zorder=12)
        ax3.scatter(*bu, color="#ff9a1f", s=110, marker="^", zorder=13)
        ax3.set_axis_off()
        ax3.set_xlim(-1.35, 1.35)
        ax3.set_ylim(-1.35, 1.35)
        ax3.set_zlim(-0.9, 0.9)
        ax3.set_box_aspect((1, 1, 0.55))
        ax3.view_init(elev=42, azim=-62 + 14 * math.sin(2 * math.pi * t / seconds))
        # --- dial
        axd.clear()
        axd.set_facecolor(bg)
        ang = np.linspace(0, 2 * np.pi, 200)
        axd.plot(np.cos(ang), np.sin(ang), color="#3a445c", lw=1.2)
        axd.annotate("", xy=(1, 0), xytext=(0, 0), arrowprops=dict(arrowstyle="-|>", color="white", lw=1.6, alpha=0.8))
        axd.annotate("", xy=(math.cos(phi), math.sin(phi)), xytext=(0, 0),
                     arrowprops=dict(arrowstyle="-|>", color="#ff9a1f", lw=2.6))
        turns = np.linspace(0, phi, 80)
        axd.plot(0.35 * np.cos(turns), 0.35 * np.sin(turns), color="#ff9a1f", lw=1.0, alpha=0.7)
        axd.set_xlim(-1.25, 1.25)
        axd.set_ylim(-1.25, 1.25)
        axd.set_aspect("equal")
        axd.axis("off")
        axd.set_title("rotation φ / поворот φ", color=fg, fontsize=12 * sc)
        # --- spinor
        axc.clear()
        axc.set_facecolor(bg)
        psi = spinor_component(phi)
        if k != last_k and (not trace or abs(trace[-1] - psi) > 1e-9):
            trace.append(psi)
        last_k = k
        axc.plot(np.cos(ang), np.sin(ang), color="#3a445c", lw=1.2)
        if len(trace) > 1 and snap is None:
            tr = np.array(trace)
            axc.plot(tr.real, tr.imag, color="#7fd1ff", lw=1.0, alpha=0.5)
        elif snap is not None:
            ph = np.linspace(0, phi, 120)
            axc.plot(np.cos(ph / 2), -np.sin(ph / 2), color="#7fd1ff", lw=1.0, alpha=0.5)
        axc.annotate("", xy=(psi.real, psi.imag), xytext=(0, 0),
                     arrowprops=dict(arrowstyle="-|>", color="#7fd1ff", lw=2.4))
        axc.scatter([1, -1], [0, 0], s=14, color=dim)
        axc.text(1.12, 0.06, "+1", color=dim, fontsize=10 * sc)
        axc.text(-1.42, 0.06, "−1", color=dim, fontsize=10 * sc)
        axc.set_xlim(-1.3, 1.3)
        axc.set_ylim(-1.3, 1.3)
        axc.set_aspect("equal")
        axc.axis("off")
        axc.set_title(r"spinor $\psi=e^{-i\varphi/2}$", color=fg, fontsize=12 * sc)
        # --- text
        axt.clear()
        axt.axis("off")
        deg = math.degrees(phi)
        if deg < 1:
            msg1, msg2 = "start", "старт"
        elif abs(deg - 360) < 1:
            msg1, msg2 = "360°: ψ → −ψ", "360°: ψ меняет знак"
        elif abs(deg - 720) < 1:
            msg1, msg2 = "720°: ψ → ψ", "720°: ψ возвращается"
        else:
            msg1, msg2 = "", ""
        axt.text(0.0, 0.85, f"φ = {deg:5.0f}°", color="#ff9a1f", fontsize=22 * sc, family="monospace")
        axt.text(0.0, 0.45, msg1, color=fg, fontsize=17 * sc)
        axt.text(0.0, 0.12, msg2, color=dim, fontsize=14 * sc)
        fig.texts.clear()
        fig.text(0.02, 0.93, "Spin 1/2 and the Möbius strip / Спин 1/2 и лента Мёбиуса", color="white", fontsize=19 * sc)
        fig.text(0.02, 0.02, "the vector across the strip comes back reversed after 360° and restored after 720° / "
                 "вектор на ленте возвращается обращённым после 360° и прежним после 720°", color=dim, fontsize=9.5 * sc)
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
    ap.add_argument("--seconds", type=float, default=20.0)
    ap.add_argument("--out", type=Path, default=HERE / "media" / "spinor_mobius.mp4")
    args = ap.parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    if args.snapshot is not None:
        render(args.out.with_suffix(".png"), (1280, 720), 30, args.seconds, snap=args.snapshot)
    elif args.preview:
        render(args.out.with_name("spinor_mobius_preview.mp4"), (640, 360), 15, 6.0)
    else:
        render(args.out, (1280, 720), 30, args.seconds)


if __name__ == "__main__":
    main()
