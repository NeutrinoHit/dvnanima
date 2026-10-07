r"""Cherenkov radiation from the Huygens construction.

A charge moving along x with speed v = beta c in a medium of refractive index n.  At every moment it
emits a spherical wavelet; the wave speed is c / n.  Units: c = 1 length unit per second of film.

  * beta n < 1 (v < c/n): the wavelets are nested circles and never overlap constructively.
  * beta n > 1 (v > c/n): the wavelets have a common envelope, a cone with half-angle
        psi = arcsin(1 / (n beta)),
    and light travels perpendicular to it, at the Cherenkov angle to the track,
        cos(theta) = 1 / (n beta),     theta = 90 deg - psi.
  * The threshold is beta = 1/n; the largest angle (beta -> 1) is arccos(1/n).
  * A detector at distance L records a ring of radius R = L tan(theta).

Water, n = 1.33: threshold beta = 0.752 (kinetic energy 0.26 MeV for an electron, 54.6 MeV for a muon),
theta_max = 41.2 deg.

Scenes (film time, s): 0-10 beta = 0.55 (below threshold); 10-19 beta = 0.85; 19-28 beta = 0.99;
28-40 beta rises slowly 0.80 -> 0.99 and the ring grows.

Usage:
    python cherenkov_radiation.py                 # film -> media/cherenkov_radiation.mp4
    python cherenkov_radiation.py --preview
    python cherenkov_radiation.py --snapshot 22
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

N_INDEX = 1.33
M_ELECTRON, M_MUON = 0.511, 105.658       # MeV
L_DET = 1.0                               # radiator-to-detector distance, units of the ring panel
EMIT_DT = 0.10                            # wavelet emission period, s
WINDOW_BACK, WINDOW_FRONT = 4.4, 1.3      # world window relative to the particle
# (film time, beta) knots: beta(t) is linear between knots
BETA_KNOTS = ((0.0, 0.55), (10.0, 0.55), (11.5, 0.85), (19.0, 0.85), (20.5, 0.99), (28.0, 0.99),
              (29.5, 0.80), (40.0, 0.99))


# ------------------------------------------------------------------ physics

def threshold_beta(n: float = N_INDEX) -> float:
    return 1.0 / n


def cherenkov_angle(beta: float, n: float = N_INDEX) -> float | None:
    """Angle between the photon direction and the track in radians, None below threshold."""
    x = 1.0 / (n * beta)
    return math.acos(x) if x < 1.0 else None


def mach_half_angle(beta: float, n: float = N_INDEX) -> float | None:
    x = 1.0 / (n * beta)
    return math.asin(x) if x < 1.0 else None


def threshold_kinetic_energy(mass_mev: float, n: float = N_INDEX) -> float:
    gamma = 1.0 / math.sqrt(1.0 - 1.0 / n ** 2)
    return (gamma - 1.0) * mass_mev


def ring_radius(beta: float, length: float = L_DET, n: float = N_INDEX) -> float | None:
    th = cherenkov_angle(beta, n)
    return None if th is None else length * math.tan(th)


def beta_of_t(t: float) -> float:
    ts = [k[0] for k in BETA_KNOTS]
    bs = [k[1] for k in BETA_KNOTS]
    if t >= ts[-1]:
        return bs[-1]
    for i in range(1, len(ts)):
        if t <= ts[i]:
            return float(np.interp(t, ts[i - 1:i + 1], bs[i - 1:i + 1]))
    return bs[-1]


def in_transition(t: float) -> bool:
    return 10.0 <= t < 11.5 or 19.0 <= t < 20.5 or 28.0 <= t < 29.5


def trajectory(t_end: float, dt: float = 1 / 120) -> tuple[np.ndarray, np.ndarray]:
    ts = np.arange(0.0, t_end + dt, dt)
    beta = np.array([beta_of_t(t) for t in ts])
    x = np.concatenate([[0.0], np.cumsum(0.5 * (beta[1:] + beta[:-1]) * dt)])
    return ts, x


def wavelets(t: float, ts: np.ndarray, xs: np.ndarray, dt_emit: float = EMIT_DT, n: float = N_INDEX):
    """Centres and radii of the wavelets emitted before time t."""
    te = np.arange(0.0, t, dt_emit)
    xe = np.interp(te, ts, xs)
    return xe, (t - te) / n


def smooth(v: float, a: float, b: float) -> float:
    u = min(max((v - a) / (b - a), 0.0), 1.0)
    return u * u * (3 - 2 * u)


# --------------------------------------------------------------------- film

def render(out: Path, size: tuple[int, int], fps: int, total: float, snap: float | None = None) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.patches as mp
    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection

    if snap is None and not shutil.which("ffmpeg"):
        sys.exit("ffmpeg is required")
    W, H = size
    dpi = 100
    sc = H / 720.0
    BG = "#03060c"
    TXT = (0.88, 0.93, 1.0, 0.95)
    DIM = (0.6, 0.68, 0.8, 0.9)
    fig = plt.figure(figsize=(W / dpi, H / dpi), dpi=dpi, facecolor=BG)
    ww = WINDOW_BACK + WINDOW_FRONT
    ax_w = fig.add_axes([0.03, 0.17, 0.62, 0.70])
    wh = ww * (0.70 * H) / (0.62 * W)
    ax_a = fig.add_axes([0.715, 0.585, 0.27, 0.275], facecolor="none")     # theta(beta)
    ax_r = fig.add_axes([0.715, 0.15, 0.27, 0.31], facecolor="none")     # ring on the detector

    k = total / 40.0
    ts, xs = trajectory(total / k)
    frames = int(round(total * fps))
    ids = [int(round(snap * fps))] if snap is not None else range(frames)
    writer = None
    if snap is None:
        writer = subprocess.Popen(
            ["ffmpeg", "-y", "-v", "error", "-f", "rawvideo", "-pix_fmt", "rgba", "-s", f"{W}x{H}",
             "-r", str(fps), "-i", "-", "-c:v", "libx264", "-preset", "slow", "-crf", "19",
             "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(out)], stdin=subprocess.PIPE)
    bgrid = np.linspace(threshold_beta() + 1e-6, 0.9999, 200)
    th_curve = np.array([math.degrees(cherenkov_angle(b)) for b in bgrid])
    th_max = math.degrees(math.acos(1 / N_INDEX))

    def style(ax):
        for s in ax.spines.values():
            s.set_color((0.5, 0.6, 0.75, 0.6))
            s.set_linewidth(0.8 * sc)
        ax.tick_params(colors=(0.65, 0.72, 0.85), labelsize=9 * sc, length=3 * sc)

    for k_ in ids:
        t_film = k_ / fps
        t = t_film / k
        beta = beta_of_t(t)
        xp = float(np.interp(t, ts, xs))
        th = cherenkov_angle(beta)
        psi = mach_half_angle(beta)
        above = th is not None

        fig.texts.clear()
        fig.patches.clear()
        ax_w.clear()
        ax_w.set_facecolor("none")
        ax_w.set_xlim(xp - WINDOW_BACK, xp + WINDOW_FRONT)
        ax_w.set_ylim(-wh / 2, wh / 2)
        ax_w.axis("off")
        # medium
        ax_w.add_patch(mp.Rectangle((xp - WINDOW_BACK, -wh / 2), ww, wh, color=(0.10, 0.22, 0.34, 0.45), lw=0, zorder=0))
        # wavelets
        xe, r = wavelets(t, ts, xs)
        ang = np.linspace(0, 2 * math.pi, 80)
        keep = r < 1.25 * WINDOW_BACK
        segs = [np.column_stack([c + rr * np.cos(ang), rr * np.sin(ang)]) for c, rr in zip(xe[keep], r[keep])]
        age = r[keep] * N_INDEX
        cols = [(0.55, 0.8, 1.0, 0.55 * math.exp(-a / 4.0) + 0.08) for a in age]
        ax_w.add_collection(LineCollection(segs, colors=cols, linewidths=1.0 * sc, zorder=2))
        # track and particle
        ax_w.plot([xp - WINDOW_BACK, xp], [0, 0], color=(1.0, 0.85, 0.4, 0.35), lw=1.0 * sc, ls=":", zorder=3)
        ax_w.plot([xp], [0], "o", color=(1.0, 0.8, 0.3), ms=9 * sc, zorder=8)
        ax_w.annotate("", xy=(xp + 0.95, 0), xytext=(xp + 0.25, 0),
                      arrowprops=dict(arrowstyle="-|>", color=(1.0, 0.8, 0.3), lw=1.6 * sc), zorder=8)
        ax_w.text(xp + 0.62, 0.18, "v", color=(1.0, 0.8, 0.3), fontsize=12 * sc, ha="center", zorder=8)
        if above:
            fade = 1.0 - smooth(abs(t - 10.75), 0.0, 0.75) if False else (0.25 if in_transition(t) else 1.0)
            # envelope (cone) and photon rays
            for sgn in (+1, -1):
                ex = xp - 5.0 * math.cos(psi)
                ey = sgn * 5.0 * math.sin(psi)
                ax_w.plot([xp, ex], [0, ey], color=(1.0, 1.0, 1.0, 0.9 * fade), lw=2.2 * sc, zorder=6)
            for lag in (0.7, 1.5, 2.3, 3.1):
                for sgn in (+1, -1):
                    x0 = xp - lag
                    ax_w.annotate("", xy=(x0 + 0.62 * math.cos(th), sgn * 0.62 * math.sin(th)), xytext=(x0, 0),
                                  arrowprops=dict(arrowstyle="-|>", color=(0.45, 0.85, 1.0, 0.9 * fade), lw=1.5 * sc), zorder=5)
            # angle arc
            arc = np.linspace(0, th, 30)
            ax_w.plot(xp - 2.3 + 0.38 * np.cos(arc), 0.38 * np.sin(arc), color=(0.45, 0.85, 1.0, fade), lw=1.2 * sc, zorder=6)
            ax_w.text(xp - 1.78, 0.12, "θ", color=(0.45, 0.85, 1.0, fade), fontsize=12 * sc, zorder=6)
            lab = f"cos θ = 1/(nβ) = {math.cos(th):.3f}   θ = {math.degrees(th):.1f}°"
        else:
            lab = "βn < 1: the wavelets do not overlap, no light"
        # panel text
        n_txt = f"n = {N_INDEX:g}   β = {beta:.3f}   βn = {beta * N_INDEX:.2f}"
        ax_w.text(xp - WINDOW_BACK + 0.12, wh / 2 - 0.2, n_txt, color=TXT, fontsize=12 * sc, va="top", zorder=9)
        ax_w.text(xp - WINDOW_BACK + 0.12, wh / 2 - 0.55, lab, color=(0.45, 0.85, 1.0) if above else DIM,
                  fontsize=11.5 * sc, va="top", zorder=9)

        # ------------------------------------------------ theta(beta)
        ax_a.clear()
        ax_a.set_facecolor("none")
        style(ax_a)
        ax_a.plot(bgrid, th_curve, color=(0.45, 0.85, 1.0, 0.95), lw=1.8 * sc)
        ax_a.axvline(threshold_beta(), color=(1, 1, 1, 0.45), lw=0.9 * sc, ls="--")
        ax_a.axhline(th_max, color=(1, 1, 1, 0.3), lw=0.8 * sc, ls=":")
        ax_a.set_xlim(0.5, 1.0)
        ax_a.set_ylim(0, 52)
        ax_a.set_xlabel("β", color=DIM, fontsize=10 * sc)
        ax_a.set_ylabel("θ, deg", color=DIM, fontsize=10 * sc)
        ax_a.text(threshold_beta() + 0.01, 3, "threshold\nβ = 1/n", color=DIM, fontsize=9 * sc)
        ax_a.text(0.995, th_max + 1.5, f"θmax = {th_max:.1f}°", color=DIM, fontsize=9 * sc, ha="right")
        if above:
            ax_a.plot([beta], [math.degrees(th)], "o", color=(1.0, 0.8, 0.3), ms=7 * sc)
        else:
            ax_a.plot([beta], [0], "o", color=(1.0, 0.8, 0.3), ms=7 * sc)
        ax_a.set_title("Cherenkov angle", color=TXT, fontsize=10.5 * sc, loc="left")

        # ------------------------------------------------ ring on the detector
        ax_r.clear()
        ax_r.set_facecolor("none")
        ax_r.set_xlim(-1.35, 1.35)
        ax_r.set_ylim(-1.35, 1.35)
        ax_r.set_aspect("equal")
        for s in ax_r.spines.values():
            s.set_visible(False)
        ax_r.set_xticks([])
        ax_r.set_yticks([])
        ax_r.add_patch(mp.Circle((0, 0), 1.25, fill=False, ec=(0.5, 0.6, 0.75, 0.6), lw=1.0 * sc))
        ax_r.plot([0], [0], "+", color=(1.0, 0.8, 0.3), ms=8 * sc)
        R = ring_radius(beta)
        if R is not None:
            phis = np.linspace(0, 2 * math.pi, 200)
            ax_r.plot(R * np.cos(phis), R * np.sin(phis), color=(0.45, 0.85, 1.0, 0.95), lw=3.0 * sc)
            ax_r.add_patch(mp.Circle((0, 0), R, fill=False, ec=(0.45, 0.85, 1.0, 0.25), lw=9 * sc))
            ax_r.text(0, -1.33, f"R = L tan θ = {R:.2f} L", color=TXT, fontsize=10.5 * sc, ha="center")
        else:
            ax_r.text(0, -1.33, "no ring below threshold", color=DIM, fontsize=10.5 * sc, ha="center")
        ax_r.set_title("ring on the detector", color=TXT, fontsize=10.5 * sc, loc="left")

        # ------------------------------------------------ titles and captions
        if t < 10.0:
            cap = ("v < c/n: the wavelets of successive positions lie inside one another",
                   "v < c/n: волны от последовательных положений вложены друг в друга")
        elif t < 28.0:
            cap = ("v > c/n: the wavelets add up along a common envelope, a cone; light goes perpendicular to it",
                   "v > c/n: волны складываются вдоль общей огибающей, конуса; свет идёт перпендикулярно ей")
        else:
            cap = (f"the faster the charge, the wider the cone and the ring; threshold in water: e⁻ {threshold_kinetic_energy(M_ELECTRON):.2f} MeV, μ {threshold_kinetic_energy(M_MUON):.0f} MeV",
                   f"чем быстрее заряд, тем шире конус и кольцо; порог в воде: e⁻ {threshold_kinetic_energy(M_ELECTRON):.2f} МэВ, μ {threshold_kinetic_energy(M_MUON):.0f} МэВ")
        fig.text(0.03, 0.058, cap[0], color=TXT, fontsize=11.0 * sc)
        fig.text(0.03, 0.022, cap[1], color=DIM, fontsize=9.8 * sc)
        fig.text(0.03, 0.915, "Cherenkov radiation", color=TXT, fontsize=16 * sc)
        fig.text(0.03, 0.885, "Излучение Вавилова — Черенкова", color=DIM, fontsize=11 * sc)
        fade_io = min(smooth(t_film, 0.0, 0.5), 1.0 - smooth(t_film, total - 0.5, total))
        fig.canvas.draw()
        frame = np.asarray(fig.canvas.buffer_rgba()).astype(np.float32)
        if fade_io < 1.0:
            bg = np.array([3, 6, 12, 255], np.float32)
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
    plt.close(fig)
    print(f"wrote {out}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--preview", action="store_true")
    ap.add_argument("--snapshot", type=float, default=None)
    ap.add_argument("--seconds", type=float, default=40.0)
    ap.add_argument("--out", type=Path, default=HERE / "media" / "cherenkov_radiation.mp4")
    args = ap.parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    if args.snapshot is not None:
        render(args.out.with_suffix(".png"), (1280, 720), 30, args.seconds, snap=args.snapshot)
    elif args.preview:
        render(args.out.with_name("cherenkov_radiation_preview.mp4"), (640, 360), 15, args.seconds)
    else:
        render(args.out, (1280, 720), 30, args.seconds)


if __name__ == "__main__":
    main()
