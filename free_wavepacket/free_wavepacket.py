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

All the numbers are in config.toml and all the words in texts.toml (see ../dvconfig.py for --config / --set).

Usage:
    python free_wavepacket.py --lang en       # film -> media/free_wavepacket_en.mp4
    python free_wavepacket.py --lang ru
    python free_wavepacket.py --lang en --snapshot 12
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

K0 = CFG.model.k0                          # central wave number (= group velocity, hbar = m = 1)
S1 = CFG.model.sigma_narrow                # sigma_x(0) of the main / narrow packet
S2 = CFG.model.sigma_wide                  # sigma_x(0) of the wide packet in scene 2
T_MAX = CFG.model.t_max                    # model time covered by one scene
X_LO, X_HI = CFG.model.x_lo, CFG.model.x_hi
N_GRID, L_BOX = CFG.model.grid_points, CFG.model.box_length   # FFT grid: points, box length (centred on box_centre)
BOX_CENTRE = CFG.model.box_centre
SCENE2_START = CFG.timeline.scene2_start
FILM_LENGTH = CFG.timeline.film_length


# ------------------------------------------------------------------ maths

def grid() -> tuple[np.ndarray, np.ndarray]:
    x = (np.arange(N_GRID) - N_GRID // 2) * (L_BOX / N_GRID) + BOX_CENTRE
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

TEXT = {lang: load_texts(HERE, lang) for lang in ("en", "ru")}     # texts.toml


def num(x: float, fmt: str, lang: str) -> str:
    """A number for use inside $...$: decimal comma in Russian."""
    s = format(x, fmt)
    return s.replace(".", "{,}") if lang == "ru" else s


def render(out: Path, size: tuple[int, int], fps: int, total: float, lang: str = "en", snap: float | None = None) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import cm

    if snap is None and not shutil.which("ffmpeg"):
        sys.exit("ffmpeg is required")
    tx = TEXT[lang]
    V, LY, F, ST, TL, MD = CFG.video, CFG.layout, CFG.fonts, CFG.style, CFG.timeline, CFG.model
    W, H = size
    dpi = V.dpi
    sc = H / V.reference_height
    BG = ST.background
    bg_rgba = np.array([int(BG[1:3], 16), int(BG[3:5], 16), int(BG[5:7], 16), 255], np.float32)
    fig = plt.figure(figsize=(W / dpi, H / dpi), dpi=dpi, facecolor=BG)
    ax_x = fig.add_axes(LY.position_axes, facecolor="none")     # position space
    ax_p = fig.add_axes(LY.momentum_axes, facecolor="none")    # momentum space
    ax_s = fig.add_axes(LY.sigma_axes, facecolor="none")    # sigma_x(t)
    ax_t = fig.add_axes(LY.strip_axes, facecolor="none")    # |psi|^2 history strip (switched off)

    x, p = grid()
    psi_a = initial_state(x, S1)
    psi_b = initial_state(x, S2)
    frames = int(round(total * fps))
    ids = [int(round(snap * fps))] if snap is not None else range(frames)
    writer = None
    if snap is None:
        writer = subprocess.Popen(
            ["ffmpeg", "-y", "-v", "error", "-f", "rawvideo", "-pix_fmt", "rgba", "-s", f"{W}x{H}",
             "-r", str(fps), "-i", "-", "-c:v", "libx264", "-preset", V.preset, "-crf", str(V.crf),
             "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(out)], stdin=subprocess.PIPE)

    k = total / FILM_LENGTH
    t_scene2 = SCENE2_START * k
    tt = np.linspace(0, T_MAX, MD.sigma_curve_samples)
    sig_a = np.array([sigma_x_exact(S1, v) for v in tt])
    sig_b = np.array([sigma_x_exact(S2, v) for v in tt])
    TXT = tuple(ST.text)
    DIM = tuple(ST.dim)
    COL_A, COL_B = tuple(ST.packet_narrow), tuple(ST.packet_wide)
    COL_RE = tuple(ST.real_part)

    from matplotlib.ticker import FuncFormatter

    def style(ax):
        if lang == "ru":                         # decimal comma on the tick labels
            fmt = FuncFormatter(lambda v, _: f"{v:g}".replace(".", ",").replace("-", "\u2212"))
            ax.xaxis.set_major_formatter(fmt)
            ax.yaxis.set_major_formatter(fmt)
        for s in ax.spines.values():
            s.set_color(tuple(ST.spine_colour))
            s.set_linewidth(ST.spine_width * sc)
        ax.tick_params(colors=tuple(ST.tick_colour), labelsize=F.tick * sc, length=ST.tick_length * sc)

    for k_ in ids:
        t_film = k_ / fps
        scene = 1 if t_film < t_scene2 else 2
        loc = t_film if scene == 1 else t_film - t_scene2
        span = (t_scene2 if scene == 1 else total - t_scene2)
        t = T_MAX * min(1.0, max(0.0, (loc - TL.time_margin_start * span) / (TL.time_margin_span * span)))       # linear in time (no easing)

        fig.texts.clear()
        for a in (ax_x, ax_p, ax_s, ax_t):
            a.clear()
            a.set_facecolor("none")
            style(a)
        ax_t.axis("off")

        packets = [(psi_a, S1, COL_A)]
        if scene == 2:
            packets = [(psi_a, S1, COL_A), (psi_b, S2, COL_B)]

        # ----------------------------------------------------- position space
        ax_x.set_xlim(X_LO, X_HI)
        ax_x.set_ylim(*LY.ylim_scene1) if scene == 1 else ax_x.set_ylim(*LY.ylim_scene2)
        ax_x.set_xlabel(r"$x$", color=DIM, fontsize=F.axis_label * sc)
        ax_x.axhline(0, color=tuple(ST.axis_line), lw=ST.axis_line_width * sc)
        for psi0, s0, col in packets:
            psi = evolve(psi0, p, t)
            rho = np.abs(psi) ** 2
            mean, sig = moments(x, psi)
            if scene == 1:
                ax_x.plot(x, psi.real, color=(*COL_RE, ST.real_part_alpha), lw=ST.real_part_width * sc)
                ax_x.fill_between(x, 0, rho, color=(*col, ST.fill_alpha_scene1), lw=0)
                ax_x.plot(x, rho, color=(*col, 1.0), lw=ST.density_width_scene1 * sc)
                ax_x.plot([mean, mean], [0, rho.max()], color=tuple(ST.mean_line), lw=ST.mean_line_width * sc, ls=":")
                ax_x.annotate("", xy=(mean + sig, LY.width_arrow_y), xytext=(mean - sig, LY.width_arrow_y),
                              arrowprops=dict(arrowstyle="<->", color=tuple(ST.width_arrow), lw=ST.width_arrow_lw * sc))
                ax_x.text(mean, LY.width_label_y, rf"$2\sigma={num(2 * sig, '.1f', lang)}$", color=TXT, fontsize=F.width_label * sc, ha="center")
            else:
                ax_x.fill_between(x, 0, rho, color=(*col, ST.fill_alpha_scene2), lw=0)
                ax_x.plot(x, rho, color=(*col, 1.0), lw=ST.density_width_scene2 * sc)
        if scene == 1:
            ax_x.text(X_LO + LY.legend1_pos[0], LY.legend1_pos[1], r"$\mathrm{Re}\,\psi$", color=COL_RE, fontsize=F.legend * sc)
            ax_x.text(X_LO + LY.legend1_pos2[0], LY.legend1_pos2[1], r"$|\psi|^2$", color=COL_A, fontsize=F.legend * sc)
            ax_x.text(X_HI - LY.time_label_scene1[0], LY.time_label_scene1[1], rf"$t={num(t, '.1f', lang)}$", color=TXT, fontsize=F.time_label * sc, ha="right")
        else:
            ax_x.text(X_HI - LY.time_label_scene2[0], LY.time_label_scene2[1], rf"$t={num(t, '.1f', lang)}$", color=TXT, fontsize=F.time_label * sc, ha="right")
            ax_x.text(X_LO + LY.sigma0_pos[0][0], LY.sigma0_pos[0][1], rf"$\sigma_0={num(S1, 'g', lang)}$", color=COL_A, fontsize=F.legend * sc)
            ax_x.text(X_LO + LY.sigma0_pos[1][0], LY.sigma0_pos[1][1], rf"$\sigma_0={num(S2, 'g', lang)}$", color=COL_B, fontsize=F.legend * sc)
            ax_x.set_ylabel(r"$|\psi|^2$", color=DIM, fontsize=F.axis_label * sc)

        # ----------------------------------------------------- momentum space
        for i, (psi0, s0, col) in enumerate(packets):
            psi = evolve(psi0, p, t)
            pp, ph = momentum_density(psi, p)
            dens = np.abs(ph) ** 2
            dens = dens / dens.max()
            m = (pp > MD.momentum_window[0]) & (pp < MD.momentum_window[1])
            if scene == 1:
                phase = np.angle(ph[m])
                colors = cm.hsv((phase / (2 * math.pi)) % 1.0)
                ax_p.bar(pp[m], dens[m], width=(pp[1] - pp[0]) * ST.momentum_bar_width, color=colors, lw=0)
            else:
                ax_p.fill_between(pp[m], 0, dens[m], color=(*col, ST.momentum_fill_alpha), lw=0)
                ax_p.plot(pp[m], dens[m], color=(*col, 1.0), lw=ST.momentum_width * sc)
        ax_p.set_xlim(*MD.momentum_window)
        ax_p.set_ylim(0, MD.momentum_ymax)
        ax_p.set_xlabel(r"$p$", color=DIM, fontsize=F.axis_label * sc)
        ax_p.set_title(tx["p_title1"] if scene == 1 else tx["p_title2"],
                       color=TXT, fontsize=F.panel_title * sc, loc="left")
        if scene == 1:
            ax_p.text(*LY.momentum_colour_note, tx["p_colour"], color=DIM, fontsize=F.colour_note * sc, ha="right")

        # ----------------------------------------------------- sigma_x(t)
        ax_s.plot(tt, sig_a, color=(*COL_A, ST.sigma_curve_alpha), lw=ST.sigma_curve_width * sc)
        ax_s.plot([t], [sigma_x_exact(S1, t)], "o", color=COL_A, ms=ST.sigma_marker * sc)
        if scene == 2:
            ax_s.plot(tt, sig_b, color=(*COL_B, ST.sigma_curve_alpha), lw=ST.sigma_curve_width * sc)
            ax_s.plot([t], [sigma_x_exact(S2, t)], "o", color=COL_B, ms=ST.sigma_marker * sc)
        ax_s.set_xlim(0, T_MAX)
        ax_s.set_ylim(0, max(sig_a.max(), sig_b.max() if scene == 2 else 0) * LY.sigma_ylim_factor)
        ax_s.set_xlabel(r"$t$", color=DIM, fontsize=F.axis_label * sc)
        ax_s.set_title(tx["sigma_title"], color=TXT, fontsize=F.panel_title * sc, loc="left")

        # ----------------------------------------------------- captions
        cap = tx["cap1"] if scene == 1 else tx["cap2"]
        fig.patches.append(matplotlib.patches.Rectangle((0, 0), *LY.caption_box, transform=fig.transFigure,
                                                         color=tuple(LY.caption_box_colour), zorder=0, lw=0))
        fig.text(*LY.caption_pos, cap, color=TXT, fontsize=F.caption * sc)
        fig.text(*LY.title_pos, tx["title1"] if scene == 1 else tx["title2"], color=TXT, fontsize=F.title * sc)
        fade = min(smooth(t_film, 0.0, V.fade_s), 1.0 - smooth(t_film, total - V.fade_s, total))
        if scene == 1 and loc > span - TL.scene_fade_s * k or scene == 2 and loc < TL.scene_fade_s * k:
            fade = min(fade, TL.scene_fade_floor + (1.0 - TL.scene_fade_floor) * abs(loc - (span if scene == 1 else 0)) / (TL.scene_fade_s * k))
        fig.canvas.draw()
        frame = np.asarray(fig.canvas.buffer_rgba()).astype(np.float32)
        if fade < 1.0:
            frame = bg_rgba + (frame - bg_rgba) * fade
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
    ap.add_argument("--snapshot", type=float, default=None)
    ap.add_argument("--seconds", type=float, default=FILM_LENGTH)
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--config", type=Path, default=None, help="another configuration file instead of config.toml")
    ap.add_argument("--set", action="append", default=[], metavar="SECTION.KEY=VALUE", help="override one configuration value")
    args = ap.parse_args()
    V = CFG.video
    out = args.out or HERE / "media" / f"free_wavepacket_{args.lang}.mp4"
    out.parent.mkdir(parents=True, exist_ok=True)
    if args.snapshot is not None:
        render(out.with_suffix(".png"), (V.width, V.height), V.fps, args.seconds, args.lang, snap=args.snapshot)
    elif args.preview:
        render(out.with_name(out.stem + "_preview.mp4"), (V.preview_width, V.preview_height), V.preview_fps, args.seconds, args.lang)
    else:
        render(out, (V.width, V.height), V.fps, args.seconds, args.lang)


if __name__ == "__main__":
    main()
