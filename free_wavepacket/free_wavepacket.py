r"""A free Gaussian wave packet: motion, spreading and what stays unchanged in momentum space.

Units hbar = m = 1.  The initial state is the minimum-uncertainty packet

    psi(x, 0) = (2 pi s0^2)^(-1/4) exp(-x^2 / (4 s0^2) + i k0 x),     s0 = sigma_x(0).

Free evolution multiplies the momentum amplitude by a phase,

    psi~(p, t) = psi~(p, 0) exp(-i E_p t),     E_p = p^2 / 2,

so |psi~(p)|^2 never changes, the centre moves with the group velocity v = k0 and the width grows as

    sigma_x(t) = s0 sqrt(1 + (t / 2 s0^2)^2).

The film evolves psi exactly (FFT), it does not use the Gaussian formula, and the tests compare the two.

Scene 1.  One packet: Re psi (the carrier wave), |psi|^2 (the probability density), the momentum
distribution with the phase of psi~(p) as colour, and sigma_x(t).
Scene 2.  A narrow and a wide packet side by side: the narrower one spreads faster (sigma_x sigma_p = 1/2).

Usage:
    python free_wavepacket.py                 # film -> media/free_wavepacket.mp4
    python free_wavepacket.py --preview
    python free_wavepacket.py --snapshot 12
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

K0 = 2.5                      # central wave number (= group velocity, hbar = m = 1)
S1 = 1.0                      # sigma_x(0) of the main / narrow packet
S2 = 2.6                      # sigma_x(0) of the wide packet in scene 2
T_MAX = 15.0                  # model time covered by one scene
X_LO, X_HI = -12.0, 52.0
N_GRID, L_BOX = 4096, 240.0   # FFT grid: points, box length (centred on x = 12)
SCENE2_START = 17.0


# ------------------------------------------------------------------ maths

def grid() -> tuple[np.ndarray, np.ndarray]:
    x = (np.arange(N_GRID) - N_GRID // 2) * (L_BOX / N_GRID) + 12.0
    p = 2 * math.pi * np.fft.fftfreq(N_GRID, d=L_BOX / N_GRID)
    return x, p


def initial_state(x: np.ndarray, s0: float, k0: float = K0) -> np.ndarray:
    return (2 * math.pi * s0 ** 2) ** -0.25 * np.exp(-((x - 0.0) ** 2) / (4 * s0 ** 2) + 1j * k0 * x)


def evolve(psi0: np.ndarray, p: np.ndarray, t: float) -> np.ndarray:
    """Exact free evolution (hbar = m = 1) by FFT."""
    return np.fft.ifft(np.fft.fft(psi0) * np.exp(-0.5j * p ** 2 * t))


def sigma_x_exact(s0: float, t: float) -> float:
    return s0 * math.sqrt(1 + (t / (2 * s0 ** 2)) ** 2)


def moments(x: np.ndarray, psi: np.ndarray) -> tuple[float, float]:
    rho = np.abs(psi) ** 2
    dx = x[1] - x[0]
    norm = rho.sum() * dx
    mean = (x * rho).sum() * dx / norm
    var = ((x - mean) ** 2 * rho).sum() * dx / norm
    return mean, math.sqrt(var)


def momentum_density(psi: np.ndarray, p: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    ph = np.fft.fft(psi)
    order = np.argsort(p)
    return p[order], ph[order]


def smooth(v: float, a: float, b: float) -> float:
    u = min(max((v - a) / (b - a), 0.0), 1.0)
    return u * u * (3 - 2 * u)


# ------------------------------------------------------------------- film

def render(out: Path, size: tuple[int, int], fps: int, total: float, snap: float | None = None) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import cm

    if snap is None and not shutil.which("ffmpeg"):
        sys.exit("ffmpeg is required")
    W, H = size
    dpi = 100
    sc = H / 720.0
    BG = "#03060c"
    fig = plt.figure(figsize=(W / dpi, H / dpi), dpi=dpi, facecolor=BG)
    ax_x = fig.add_axes([0.06, 0.20, 0.61, 0.66], facecolor="none")     # position space
    ax_p = fig.add_axes([0.725, 0.50, 0.25, 0.36], facecolor="none")    # momentum space
    ax_s = fig.add_axes([0.725, 0.20, 0.25, 0.17], facecolor="none")    # sigma_x(t)
    ax_t = fig.add_axes([0.06, 0.175, 0.61, 0.17], facecolor="none")    # |psi|^2 history strip (not used in scene 1)

    x, p = grid()
    psi_a = initial_state(x, S1)
    psi_b = initial_state(x, S2)
    frames = int(round(total * fps))
    ids = [int(round(snap * fps))] if snap is not None else range(frames)
    writer = None
    if snap is None:
        writer = subprocess.Popen(
            ["ffmpeg", "-y", "-v", "error", "-f", "rawvideo", "-pix_fmt", "rgba", "-s", f"{W}x{H}",
             "-r", str(fps), "-i", "-", "-c:v", "libx264", "-preset", "slow", "-crf", "19",
             "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(out)], stdin=subprocess.PIPE)

    k = total / 34.0
    t_scene2 = SCENE2_START * k
    tt = np.linspace(0, T_MAX, 300)
    sig_a = np.array([sigma_x_exact(S1, v) for v in tt])
    sig_b = np.array([sigma_x_exact(S2, v) for v in tt])
    TXT = (0.88, 0.93, 1.0, 0.95)
    DIM = (0.6, 0.68, 0.8, 0.9)

    def style(ax):
        for s in ax.spines.values():
            s.set_color((0.5, 0.6, 0.75, 0.6))
            s.set_linewidth(0.8 * sc)
        ax.tick_params(colors=(0.65, 0.72, 0.85), labelsize=9 * sc, length=3 * sc)

    for k_ in ids:
        t_film = k_ / fps
        scene = 1 if t_film < t_scene2 else 2
        loc = t_film if scene == 1 else t_film - t_scene2
        span = (t_scene2 if scene == 1 else total - t_scene2)
        t = T_MAX * min(1.0, max(0.0, (loc - 0.04 * span) / (0.92 * span)))
        t = T_MAX * smooth(loc, 0.0, span) * 0 + t       # linear in time (no easing)

        fig.texts.clear()
        for a in (ax_x, ax_p, ax_s, ax_t):
            a.clear()
            a.set_facecolor("none")
            style(a)
        ax_t.axis("off")

        packets = [(psi_a, S1, (1.0, 0.72, 0.3))]
        if scene == 2:
            packets = [(psi_a, S1, (1.0, 0.72, 0.3)), (psi_b, S2, (0.35, 0.8, 1.0))]

        # ----------------------------------------------------- position space
        ax_x.set_xlim(X_LO, X_HI)
        ax_x.set_ylim(-0.95, 0.95) if scene == 1 else ax_x.set_ylim(-0.1, 0.62)
        ax_x.set_xlabel("x", color=DIM, fontsize=10 * sc)
        ax_x.axhline(0, color=(0.5, 0.6, 0.75, 0.3), lw=0.7 * sc)
        for psi0, s0, col in packets:
            psi = evolve(psi0, p, t)
            rho = np.abs(psi) ** 2
            mean, sig = moments(x, psi)
            if scene == 1:
                ax_x.plot(x, psi.real, color=(0.55, 0.78, 1.0, 0.9), lw=1.1 * sc)
                ax_x.fill_between(x, 0, rho, color=(*col, 0.25), lw=0)
                ax_x.plot(x, rho, color=(*col, 1.0), lw=1.9 * sc)
                ax_x.plot([mean, mean], [0, rho.max()], color=(1, 1, 1, 0.5), lw=0.9 * sc, ls=":")
                ax_x.annotate("", xy=(mean + sig, 0.66), xytext=(mean - sig, 0.66),
                              arrowprops=dict(arrowstyle="<->", color=(1, 1, 1, 0.8), lw=1.2 * sc))
                ax_x.text(mean, 0.72, f"2σ = {2 * sig:.1f}", color=TXT, fontsize=10.5 * sc, ha="center")
            else:
                ax_x.fill_between(x, 0, rho, color=(*col, 0.22), lw=0)
                ax_x.plot(x, rho, color=(*col, 1.0), lw=2.0 * sc)
        if scene == 1:
            ax_x.text(X_LO + 1, 0.82, "Re ψ", color=(0.55, 0.78, 1.0), fontsize=11 * sc)
            ax_x.text(X_LO + 8, 0.82, "|ψ|²", color=(1.0, 0.72, 0.3), fontsize=11 * sc)
            ax_x.text(X_HI - 1, 0.82, f"t = {t:4.1f}", color=TXT, fontsize=12 * sc, ha="right")
        else:
            ax_x.text(X_HI - 1, 0.55, f"t = {t:4.1f}", color=TXT, fontsize=12 * sc, ha="right")
            ax_x.text(X_LO + 1, 0.55, f"σ₀ = {S1:g}", color=(1.0, 0.72, 0.3), fontsize=11 * sc)
            ax_x.text(X_LO + 1, 0.49, f"σ₀ = {S2:g}", color=(0.35, 0.8, 1.0), fontsize=11 * sc)
            ax_x.set_ylabel("|ψ|²", color=DIM, fontsize=10 * sc)

        # ----------------------------------------------------- momentum space
        for i, (psi0, s0, col) in enumerate(packets):
            psi = evolve(psi0, p, t)
            pp, ph = momentum_density(psi, p)
            dens = np.abs(ph) ** 2
            dens = dens / dens.max()
            m = (pp > 0.2) & (pp < 4.8)
            if scene == 1:
                phase = np.angle(ph[m])
                colors = cm.hsv((phase / (2 * math.pi)) % 1.0)
                ax_p.bar(pp[m], dens[m], width=(pp[1] - pp[0]) * 1.05, color=colors, lw=0)
            else:
                ax_p.fill_between(pp[m], 0, dens[m] * (S1 / s0 if False else 1.0), color=(*col, 0.25), lw=0)
                ax_p.plot(pp[m], dens[m], color=(*col, 1.0), lw=1.8 * sc)
        ax_p.set_xlim(0.2, 4.8)
        ax_p.set_ylim(0, 1.12)
        ax_p.set_xlabel("p", color=DIM, fontsize=10 * sc)
        ax_p.set_title(r"$|\tilde\psi(p)|^2$ does not change" if scene == 1 else r"$|\tilde\psi(p)|^2$: narrower in $x$ = broader in $p$",
                       color=TXT, fontsize=10.5 * sc, loc="left")
        if scene == 1:
            ax_p.text(4.7, 1.0, r"colour = phase of $\tilde\psi(p)$", color=DIM, fontsize=9 * sc, ha="right")

        # ----------------------------------------------------- sigma_x(t)
        ax_s.plot(tt, sig_a, color=(1.0, 0.72, 0.3, 0.9), lw=1.6 * sc)
        ax_s.plot([t], [sigma_x_exact(S1, t)], "o", color=(1.0, 0.72, 0.3), ms=6 * sc)
        if scene == 2:
            ax_s.plot(tt, sig_b, color=(0.35, 0.8, 1.0, 0.9), lw=1.6 * sc)
            ax_s.plot([t], [sigma_x_exact(S2, t)], "o", color=(0.35, 0.8, 1.0), ms=6 * sc)
        ax_s.set_xlim(0, T_MAX)
        ax_s.set_ylim(0, max(sig_a.max(), sig_b.max() if scene == 2 else 0) * 1.08)
        ax_s.set_xlabel("t", color=DIM, fontsize=10 * sc)
        ax_s.set_title(r"$\sigma_x(t)=\sigma_0\sqrt{1+(t/2\sigma_0^2)^2}$", color=TXT, fontsize=10.5 * sc, loc="left")

        # ----------------------------------------------------- captions
        if scene == 1:
            cap = ("the centre moves with the group velocity, the envelope broadens and its height falls: the phases $e^{-iE_p t}$ get out of step",
                   "центр движется с групповой скоростью, огибающая расширяется, высота падает: фазы $e^{-iE_p t}$ расходятся")
        else:
            cap = ("the narrower in x, the broader in p, the faster the spreading: $\\sigma_x\\sigma_p=\\hbar/2$ at $t=0$",
                   "чем уже по x, тем шире по p и тем быстрее расплывание: $\\sigma_x\\sigma_p=\\hbar/2$ при $t=0$")
        fig.patches.append(matplotlib.patches.Rectangle((0, 0), 0.92, 0.105, transform=fig.transFigure,
                                                         color=(0.01, 0.02, 0.05, 0.8), zorder=0, lw=0))
        fig.text(0.03, 0.058, cap[0], color=TXT, fontsize=11.0 * sc)
        fig.text(0.03, 0.022, cap[1], color=DIM, fontsize=9.8 * sc)
        fig.text(0.06, 0.915, "A free Gaussian wave packet" if scene == 1 else "Narrow and wide packets",
                 color=TXT, fontsize=16 * sc)
        fig.text(0.06, 0.885, "Свободный гауссов волновой пакет" if scene == 1 else "Узкий и широкий пакеты",
                 color=DIM, fontsize=11 * sc)
        fade = min(smooth(t_film, 0.0, 0.5), 1.0 - smooth(t_film, total - 0.5, total))
        if scene == 1 and loc > span - 0.5 * k or scene == 2 and loc < 0.5 * k:
            fade = min(fade, 0.35 + 0.65 * abs(loc - (span if scene == 1 else 0)) / (0.5 * k))
        fig.canvas.draw()
        frame = np.asarray(fig.canvas.buffer_rgba()).astype(np.float32)
        if fade < 1.0:
            bg = np.array([3, 6, 12, 255], np.float32)
            frame = bg + (frame - bg) * fade
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
    ap.add_argument("--seconds", type=float, default=34.0)
    ap.add_argument("--out", type=Path, default=HERE / "media" / "free_wavepacket.mp4")
    args = ap.parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    if args.snapshot is not None:
        render(args.out.with_suffix(".png"), (1280, 720), 30, args.seconds, snap=args.snapshot)
    elif args.preview:
        render(args.out.with_name("free_wavepacket_preview.mp4"), (640, 360), 15, args.seconds)
    else:
        render(args.out, (1280, 720), 30, args.seconds)


if __name__ == "__main__":
    main()
