r"""A kick to an electron: where the electromagnetic wave front comes from.

A charge at rest is instantaneously given the velocity v along +x at t = 0 (c = 1).
The exact field of this process (Lienard-Wiechert) has three zones:

* r > ct: the information about the kick has not arrived, the field is the Coulomb field of
  the charge at rest at the origin, the field lines are radial from the origin;
* r < ct: the field of a charge that moves uniformly with velocity v.  Its field lines are
  straight and radial from the present position x_p = v t, but crowded towards the plane
  perpendicular to v (Lorentz contraction);
* r = ct: a thin shell, the front of the electromagnetic wave.  It carries the field
  transverse to the radius that joins the two fields.

Gauss's law fixes how the lines are joined.  The flux inside a cone of half-angle psi around
the velocity, seen from the present position, is

    (1/2) (1 - cos(psi) / sqrt(1 - beta^2 sin^2 psi))      (in units of the charge),

and it must equal the flux (1 - cos(theta)) / 2 inside the cone of half-angle theta seen from
the origin.  Hence a line that leaves the charge at the angle psi runs to the shell, goes along
the shell (an arc of the circle r = ct) and continues outwards, radially from the origin, at
the angle theta with

    cos(theta) = cos(psi) / sqrt(1 - beta^2 sin^2 psi),    i.e.   tan(theta) = tan(psi) / gamma.

The lines are drawn with equal flux, so their density is the strength of the field, and the
shell shines where many lines run along it, that is where the radiation field is strong.

Usage:
    python electron_kick.py                 # film -> media/electron_kick.mp4
    python electron_kick.py --preview
    python electron_kick.py --snapshot 6    # one PNG at film time 6 s
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


def psi_of_theta(theta: np.ndarray | float, beta: float) -> np.ndarray | float:
    """Angle (from the velocity, seen from the charge) of the inner line that joins the outer
    line leaving the origin at the angle theta."""
    theta = np.asarray(theta, dtype=float)
    c = np.cos(theta)
    gamma = 1.0 / math.sqrt(1.0 - beta ** 2)
    cos_psi = c / (gamma * np.sqrt(1.0 - (c * beta) ** 2))
    return np.arccos(np.clip(cos_psi, -1.0, 1.0))


def theta_of_psi(psi: np.ndarray | float, beta: float) -> np.ndarray | float:
    psi = np.asarray(psi, dtype=float)
    c = np.cos(psi) / np.sqrt(1.0 - (beta * np.sin(psi)) ** 2)
    return np.arccos(np.clip(c, -1.0, 1.0))


def flux_fraction_moving(psi: np.ndarray | float, beta: float) -> np.ndarray | float:
    """Flux of the field of a uniformly moving charge inside the cone of half-angle psi."""
    psi = np.asarray(psi, dtype=float)
    return 0.5 * (1.0 - np.cos(psi) / np.sqrt(1.0 - (beta * np.sin(psi)) ** 2))


def line_geometry(theta: float, beta: float, t: float) -> dict[str, np.ndarray]:
    """Pieces of the field line that is radial at the angle theta far away (t > 0)."""
    radius = t                                 # c = 1
    x_p = beta * t
    psi = float(psi_of_theta(theta, beta))
    cp, sp = math.cos(psi), math.sin(psi)
    s = -x_p * cp + math.sqrt(radius ** 2 - (x_p * sp) ** 2)
    hit = np.array([x_p + s * cp, s * sp])
    theta_in = math.atan2(hit[1], hit[0])
    return dict(start=np.array([x_p, 0.0]), hit=hit, theta_in=theta_in, theta_out=theta, psi=psi, radius=radius)


# ---------------------------------------------------------------------- film

def smooth(x: float, a: float, b: float) -> float:
    u = min(max((x - a) / (b - a), 0.0), 1.0)
    return u * u * (3 - 2 * u)


RUNS = ((0.30, 0.0, 10.0), (0.85, 10.0, 20.0))     # (beta, start, end) in film seconds
T_KICK = 1.6                                        # seconds after the start of a run


def render(out: Path, size: tuple[int, int], fps: int, total: float, snap: float | None = None) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.colors as mcolors
    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection

    if snap is None and not shutil.which("ffmpeg"):
        sys.exit("ffmpeg is required")
    W, H = size
    dpi = 100
    sc = H / 720.0
    cx0, cy0 = 0.20 * W, 0.215 * H
    unit = 80.0 * sc                            # pixels travelled by light in one film second
    n_lines = 22
    theta_k = np.arccos(1.0 - 2.0 * (np.arange(n_lines) + 0.5) / n_lines)   # equal flux, upper half-plane
    r_far = 620.0 * sc

    fig = plt.figure(figsize=(W / dpi, H / dpi), dpi=dpi, facecolor="#03060c")
    ax = fig.add_axes([0, 0, 1, 1])
    clip = matplotlib.patches.Rectangle((0, 0), 0.715 * W, H, transform=ax.transData)

    frames = int(round(total * fps))
    ids = [int(round(snap * fps))] if snap is not None else range(frames)
    writer = None
    if snap is None:
        writer = subprocess.Popen(
            ["ffmpeg", "-y", "-v", "error", "-f", "rawvideo", "-pix_fmt", "rgba", "-s", f"{W}x{H}",
             "-r", str(fps), "-i", "-", "-c:v", "libx264", "-preset", "slow", "-crf", "19",
             "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(out)], stdin=subprocess.PIPE)

    def glow(X, Y, color, d, alpha=1.0, z=8):
        for scale, al in ((4.2, 0.07), (2.4, 0.16), (1.3, 0.4), (0.6, 1.0)):
            ax.scatter([X], [Y], s=(d * scale * 72 / dpi) ** 2, linewidths=0, zorder=z,
                       facecolors=[mcolors.to_rgba(color, al * alpha)])

    for k_ in ids:
        tf = k_ / fps
        run = next((r for r in RUNS if r[1] <= tf < r[2]), RUNS[-1])
        beta, t_start, t_end = run
        gamma = 1.0 / math.sqrt(1 - beta ** 2)
        t = min(tf - t_start - T_KICK, 6.2)       # time since the kick, film seconds (frozen at the end of a run)
        fade = min(smooth(tf - t_start, 0.0, 0.6), 1 - smooth(tf, t_end - 0.6, t_end)) if t_end < total - 1e-9 else smooth(tf - t_start, 0, 0.6)
        ax.clear()
        ax.set_xlim(0, W)
        ax.set_ylim(0, H)
        ax.axis("off")
        ax.add_patch(matplotlib.patches.Rectangle((0, 0), W, H, color="#03060c", zorder=0))
        ax.set_clip_path(clip)
        segs_out, segs_in, arcs = [], [], []
        if t <= 0.0:
            for th in theta_k:
                for sgn in (1,):
                    segs_out.append([(cx0, cy0), (cx0 + r_far * math.cos(th), cy0 + sgn * r_far * math.sin(th))])
            x_charge, radius = 0.0, 0.0
        else:
            tt = t * unit / unit                  # in light-seconds: c = 1
            radius = t
            x_charge = beta * t
            for th in theta_k:
                g = line_geometry(float(th), beta, tt)
                for sgn in (1,):
                    p0 = (cx0 + unit * g["start"][0], cy0)
                    p1 = (cx0 + unit * g["hit"][0], cy0 + sgn * unit * g["hit"][1])
                    segs_in.append([p0, p1])
                    a0, a1 = g["theta_in"], g["theta_out"]
                    aa = np.linspace(a0, a1, 24)
                    arcs.append([(cx0 + unit * radius * math.cos(a), cy0 + sgn * unit * radius * math.sin(a)) for a in aa])
                    far = (cx0 + unit * radius * math.cos(a1), cy0 + sgn * unit * radius * math.sin(a1))
                    segs_out.append([far, (cx0 + r_far * math.cos(a1), cy0 + sgn * r_far * math.sin(a1))])
        a_f = fade
        ax.add_collection(LineCollection(segs_out, colors=[(0.45, 0.55, 0.78, 0.55 * a_f)] * len(segs_out), linewidths=1.3 * sc, zorder=2))
        if segs_in:
            ax.add_collection(LineCollection(segs_in, colors=[(0.35, 0.88, 1.0, 0.85 * a_f)] * len(segs_in), linewidths=1.5 * sc, zorder=3))
        if arcs:
            ax.add_collection(LineCollection(arcs, colors=[(1.0, 0.78, 0.30, 0.38 * a_f)] * len(arcs), linewidths=3.0 * sc, zorder=4, capstyle="round"))
        # light circle r = ct
        if radius > 0:
            a_ = np.linspace(0.0, math.pi, 300)
            ax.plot(cx0 + unit * radius * np.cos(a_), cy0 + unit * radius * np.sin(a_), color=(1, 0.85, 0.5, 0.35 * a_f), lw=1.0 * sc, ls=(0, (3, 4)), zorder=4)
        ax.plot([0, 0.715 * W], [cy0, cy0], color=(0.7, 0.8, 0.95, 0.28 * a_f), lw=1.0 * sc, zorder=1)
        ax.text(cx0 - 14 * sc, cy0 - 26 * sc, "x = 0", color=(0.75, 0.82, 0.95, 0.7 * a_f), fontsize=10 * sc, zorder=10)
        # the charge, the origin and the kick
        ax.scatter([cx0], [cy0], s=(7 * sc * 72 / dpi) ** 2, facecolors="none", edgecolors=[(0.8, 0.85, 1, 0.5 * a_f)], linewidths=1.0 * sc, zorder=6)
        glow(cx0 + unit * x_charge, cy0, "#7fd6ff", 15 * sc, a_f)
        ax.plot([cx0 + unit * x_charge - 3.5 * sc, cx0 + unit * x_charge + 3.5 * sc], [cy0, cy0], color="#04203a", lw=2.2 * sc, alpha=a_f, zorder=9, solid_capstyle="round")
        if -0.35 < t < 0.35:
            k_a = (1 - abs(t) / 0.35)
            ax.annotate("", xy=(cx0 - 12 * sc, cy0), xytext=(cx0 - 12 * sc - 95 * sc, cy0),
                        arrowprops=dict(arrowstyle="-|>", color=(1.0, 0.55, 0.15, k_a * a_f), lw=5 * sc, mutation_scale=22 * sc), zorder=10)
            ax.text(cx0 - 120 * sc, cy0 + 24 * sc, "kick / пинок", color=(1.0, 0.7, 0.35, k_a * a_f), fontsize=14 * sc, zorder=10)
        if t > 0 and x_charge > 0.4:
            ax.annotate("", xy=(cx0 + unit * x_charge + 60 * sc, cy0 + 24 * sc), xytext=(cx0 + unit * x_charge + 8 * sc, cy0 + 24 * sc),
                        arrowprops=dict(arrowstyle="-|>", color=(0.6, 0.95, 1.0, 0.9 * a_f), lw=2 * sc), zorder=10)
            ax.text(cx0 + unit * x_charge + 30 * sc, cy0 + 32 * sc, "v", color=(0.6, 0.95, 1.0, 0.9 * a_f), fontsize=13 * sc, zorder=10)
        # labels of the zones
        if t > 1.2:
            ax.text(cx0 + unit * radius * 0.62, cy0 + unit * radius * 0.80, "radiation front r = ct", color=(1.0, 0.82, 0.4, 0.9 * a_f), fontsize=11 * sc, zorder=10)
        # panel with the formulas
        fig.texts.clear()
        fig.text(0.735, 0.90, "electron kick / пинок электрону", color=(0.88, 0.93, 1, a_f), fontsize=14 * sc)
        fig.text(0.735, 0.835, f"β = v/c = {beta:.2f},   γ = {gamma:.2f}", color=(0.6, 0.9, 1.0, a_f), fontsize=14 * sc, family="monospace")
        items = [
            ((0.45, 0.55, 0.78), "r > ct", "field of the charge at rest\nполе покоящегося заряда"),
            ((1.0, 0.78, 0.30), "r = ct", "radiation front: field ⊥ radius\nфронт волны: поле ⊥ радиусу"),
            ((0.35, 0.88, 1.0), "r < ct", "field of the moving charge,\ncrowded to the plane ⊥ v\nполе движущегося заряда"),
        ]
        for i, (col, head, body) in enumerate(items):
            y = 0.74 - 0.145 * i
            fig.text(0.735, y, "━ " + head, color=(*col, a_f), fontsize=12.5 * sc, family="monospace")
            fig.text(0.735, y - 0.052, body, color=(0.72, 0.78, 0.9, 0.9 * a_f), fontsize=9.5 * sc, linespacing=1.35, va="top")
        fig.text(0.735, 0.29, "Gauss: flux conservation / сохранение потока", color=(0.88, 0.93, 1, 0.9 * a_f), fontsize=9.8 * sc)
        fig.text(0.735, 0.235, r"$\cos\theta=\dfrac{\cos\psi}{\sqrt{1-\beta^{2}\sin^{2}\psi}}$",
                 color=(1.0, 0.82, 0.4, 0.95 * a_f), fontsize=14 * sc)
        fig.text(0.735, 0.178, r"$\tan\theta=\tan\psi\,/\,\gamma$", color=(1.0, 0.82, 0.4, 0.95 * a_f), fontsize=12 * sc)
        fig.text(0.735, 0.145, "θ: angle at the origin (r > ct)\nψ: angle at the charge (r < ct)\nθ: угол из начала, ψ: угол из заряда",
                 color=(0.6, 0.68, 0.8, 0.85 * a_f), fontsize=8.5 * sc, linespacing=1.4, va="top")
        ax.add_patch(matplotlib.patches.Rectangle((0, 0), 0.715 * W, 0.075 * H, color=(0.01, 0.02, 0.05, 0.85), zorder=11))
        cap = ("an electron at rest: the field lines are radial", "покоящийся электрон: линии поля радиальны") if t <= 0 else (
            ("the field near the electron has changed, the far field has not yet heard about the kick",
             "вблизи электрона поле уже изменилось, дальнее ещё «не знает» о толчке") if t < 5.4 else
            ("the lines are joined along the shell: the front of the electromagnetic wave", "линии соединяются вдоль оболочки: фронт электромагнитной волны"))
        fig.text(0.03, 0.043, cap[0], color=(0.88, 0.93, 1.0, 0.93 * a_f), fontsize=11.5 * sc)
        fig.text(0.03, 0.012, cap[1], color=(0.6, 0.68, 0.8, 0.9 * a_f), fontsize=9.5 * sc)
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
    ap.add_argument("--out", type=Path, default=HERE / "media" / "electron_kick.mp4")
    args = ap.parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    if args.snapshot is not None:
        render(args.out.with_suffix(".png"), (1280, 720), 30, args.seconds, snap=args.snapshot)
    elif args.preview:
        render(args.out.with_name("electron_kick_preview.mp4"), (640, 360), 15, args.seconds)
    else:
        render(args.out, (1280, 720), 30, args.seconds)


if __name__ == "__main__":
    main()
