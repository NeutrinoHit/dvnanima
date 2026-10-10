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

All the numbers are in config.toml and all the words in texts.toml (see ../dvconfig.py for --config / --set).

Usage:
    python cherenkov_radiation.py --lang en       # film -> media/cherenkov_radiation_en.mp4
    python cherenkov_radiation.py --lang ru
    python cherenkov_radiation.py --lang en --snapshot 22
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

N_INDEX = CFG.physics.n_index
M_ELECTRON, M_MUON = CFG.physics.electron_mass_mev, CFG.physics.muon_mass_mev       # MeV
L_DET = CFG.physics.detector_distance                               # radiator-to-detector distance, units of the ring panel
EMIT_DT = CFG.physics.emit_dt                                       # wavelet emission period, s
WINDOW_BACK, WINDOW_FRONT = CFG.physics.window_back, CFG.physics.window_front      # world window relative to the particle
# (film time, beta) knots: beta(t) is linear between knots
BETA_KNOTS = tuple(tuple(k) for k in CFG.timeline.beta_knots)
FILM_LENGTH = CFG.timeline.film_length
TRANSITIONS = [tuple(w) for w in CFG.timeline.transitions]


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
    return any(a <= t < b for a, b in TRANSITIONS)


def trajectory(t_end: float, dt: float = 1 / CFG.physics.trajectory_per_s) -> tuple[np.ndarray, np.ndarray]:
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

TEXT = {lang: load_texts(HERE, lang) for lang in ("en", "ru")}     # texts.toml


def num(x: float, fmt: str, lang: str) -> str:
    s = format(x, fmt)
    return s.replace(".", "{,}") if lang == "ru" else s


def render(out: Path, size: tuple[int, int], fps: int, total: float, lang: str = "en", snap: float | None = None) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.patches as mp
    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection

    if snap is None and not shutil.which("ffmpeg"):
        sys.exit("ffmpeg is required")
    tx = TEXT[lang]
    V, LY, ST, WV, SC, AP, RP = CFG.video, CFG.layout, CFG.style, CFG.wavelets, CFG.scene, CFG.angle_panel, CFG.ring_panel
    W, H = size
    dpi = V.dpi
    sc = H / V.reference_height
    BG = ST.background
    bg_rgba = np.array([int(BG[1:3], 16), int(BG[3:5], 16), int(BG[5:7], 16), 255], np.float32)
    TXT = tuple(ST.text)
    DIM = tuple(ST.dim)
    PART, LIGHT = tuple(ST.particle), tuple(ST.light)
    fig = plt.figure(figsize=(W / dpi, H / dpi), dpi=dpi, facecolor=BG)
    ww = WINDOW_BACK + WINDOW_FRONT
    ax_w = fig.add_axes(LY.world_axes)
    wh = ww * (LY.world_axes[3] * H) / (LY.world_axes[2] * W)
    ax_a = fig.add_axes(AP.axes, facecolor="none")     # theta(beta)
    ax_r = fig.add_axes(RP.axes, facecolor="none")     # ring on the detector

    k = total / FILM_LENGTH
    ts, xs = trajectory(total / k)
    frames = int(round(total * fps))
    ids = [int(round(snap * fps))] if snap is not None else range(frames)
    writer = None
    if snap is None:
        writer = subprocess.Popen(
            ["ffmpeg", "-y", "-v", "error", "-f", "rawvideo", "-pix_fmt", "rgba", "-s", f"{W}x{H}",
             "-r", str(fps), "-i", "-", "-c:v", "libx264", "-preset", V.preset, "-crf", str(V.crf),
             "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(out)], stdin=subprocess.PIPE)
    bgrid = np.linspace(threshold_beta() + AP.beta_epsilon, AP.beta_max, AP.grid_points)
    th_curve = np.array([math.degrees(cherenkov_angle(b)) for b in bgrid])
    th_max = math.degrees(math.acos(1 / N_INDEX))

    from matplotlib.ticker import FuncFormatter

    def style(ax):
        if lang == "ru":
            fmt = FuncFormatter(lambda v, _: f"{v:g}".replace(".", ",").replace("-", "\u2212"))
            ax.xaxis.set_major_formatter(fmt)
            ax.yaxis.set_major_formatter(fmt)
        for s in ax.spines.values():
            s.set_color(tuple(ST.spine_colour))
            s.set_linewidth(ST.spine_width * sc)
        ax.tick_params(colors=tuple(ST.tick_colour), labelsize=ST.tick_size * sc, length=ST.tick_length * sc)

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
        ax_w.add_patch(mp.Rectangle((xp - WINDOW_BACK, -wh / 2), ww, wh, color=tuple(WV.medium_colour), lw=0, zorder=0))
        # wavelets
        xe, r = wavelets(t, ts, xs)
        ang = np.linspace(0, 2 * math.pi, WV.points)
        keep = r < WV.keep_factor * WINDOW_BACK
        segs = [np.column_stack([c + rr * np.cos(ang), rr * np.sin(ang)]) for c, rr in zip(xe[keep], r[keep])]
        age = r[keep] * N_INDEX
        cols = [(*WV.colour, WV.alpha_gain * math.exp(-a / WV.decay_time) + WV.alpha_base) for a in age]
        ax_w.add_collection(LineCollection(segs, colors=cols, linewidths=WV.width * sc, zorder=2))
        # track and particle
        ax_w.plot([xp - WINDOW_BACK, xp], [0, 0], color=(*ST.track, SC.track_alpha), lw=SC.track_width * sc, ls=SC.track_dash, zorder=3)
        ax_w.plot([xp], [0], "o", color=PART, ms=SC.particle_size * sc, zorder=8)
        ax_w.annotate("", xy=(xp + SC.velocity_arrow["x1"], 0), xytext=(xp + SC.velocity_arrow["x0"], 0),
                      arrowprops=dict(arrowstyle="-|>", color=PART, lw=SC.velocity_arrow["lw"] * sc), zorder=8)
        ax_w.text(xp + SC.velocity_label["dx"], SC.velocity_label["y"], r"$\mathbf{v}$", color=PART, fontsize=SC.velocity_label["size"] * sc, ha="center", zorder=8)
        if above:
            fade = CFG.timeline.transition_alpha if in_transition(t) else 1.0
            # envelope (cone) and photon rays
            for sgn in (+1, -1):
                ex = xp - SC.cone_length * math.cos(psi)
                ey = sgn * SC.cone_length * math.sin(psi)
                ax_w.plot([xp, ex], [0, ey], color=(*ST.cone, SC.cone_alpha * fade), lw=SC.cone_width * sc, zorder=6)
            for lag in SC.ray_lags:
                for sgn in (+1, -1):
                    x0 = xp - lag
                    ax_w.annotate("", xy=(x0 + SC.ray_length * math.cos(th), sgn * SC.ray_length * math.sin(th)), xytext=(x0, 0),
                                  arrowprops=dict(arrowstyle="-|>", color=(*LIGHT, SC.ray_alpha * fade), lw=SC.ray_width * sc), zorder=5)
            # angle arc
            arc = np.linspace(0, th, SC.arc_points)
            ax_w.plot(xp + SC.arc_centre_dx + SC.arc_radius * np.cos(arc), SC.arc_radius * np.sin(arc), color=(*LIGHT, fade), lw=SC.arc_width * sc, zorder=6)
            ax_w.text(xp + SC.theta_label["dx"], SC.theta_label["y"], r"$\theta$", color=(*LIGHT, fade), fontsize=SC.theta_label["size"] * sc, zorder=6)
            lab = rf"$\cos\theta=1/(n\beta)={num(math.cos(th), '.3f', lang)}\quad\theta={num(math.degrees(th), '.1f', lang)}^\circ$"
        else:
            lab = rf"$\beta n<1$: {tx['nolight']}"
        # panel text
        n_txt = rf"$n={num(N_INDEX, 'g', lang)}\quad\beta={num(beta, '.3f', lang)}\quad\beta n={num(beta * N_INDEX, '.2f', lang)}$"
        ax_w.text(xp - WINDOW_BACK + SC.info1_offset[0], wh / 2 - SC.info1_offset[1], n_txt, color=TXT, fontsize=SC.info1_size * sc, va="top", zorder=9)
        ax_w.text(xp - WINDOW_BACK + SC.info2_offset[0], wh / 2 - SC.info2_offset[1], lab, color=LIGHT if above else DIM,
                  fontsize=SC.info2_size * sc, va="top", zorder=9)

        # ------------------------------------------------ theta(beta)
        ax_a.clear()
        ax_a.set_facecolor("none")
        style(ax_a)
        ax_a.plot(bgrid, th_curve, color=(*LIGHT, AP.curve_alpha), lw=AP.curve_width * sc)
        ax_a.axvline(threshold_beta(), color=(*ST.guide, AP.threshold_alpha), lw=AP.threshold_width * sc, ls="--")
        ax_a.axhline(th_max, color=(*ST.guide, AP.max_alpha), lw=AP.max_width * sc, ls=":")
        ax_a.set_xlim(*AP.xlim)
        ax_a.set_ylim(*AP.ylim)
        ax_a.set_xlabel(r"$\beta$", color=DIM, fontsize=AP.label_size * sc)
        ax_a.set_ylabel(rf"$\theta$, {tx['deg']}", color=DIM, fontsize=AP.label_size * sc)
        ax_a.text(threshold_beta() + AP.threshold_label["dx"], AP.threshold_label["y"], tx["thr"] + "\n" + r"$\beta=1/n$", color=DIM, fontsize=AP.threshold_label["size"] * sc)
        ax_a.text(AP.max_label["x"], th_max + AP.max_label["dy"], rf"$\theta_{{\max}}={num(th_max, '.1f', lang)}^\circ$", color=DIM, fontsize=AP.max_label["size"] * sc, ha="right")
        if above:
            ax_a.plot([beta], [math.degrees(th)], "o", color=PART, ms=AP.marker_size * sc)
        else:
            ax_a.plot([beta], [0], "o", color=PART, ms=AP.marker_size * sc)
        ax_a.set_title(tx["ang_title"], color=TXT, fontsize=AP.title_size * sc, loc="left")

        # ------------------------------------------------ ring on the detector
        ax_r.clear()
        ax_r.set_facecolor("none")
        ax_r.set_xlim(-RP.lim, RP.lim)
        ax_r.set_ylim(-RP.lim, RP.lim)
        ax_r.set_aspect("equal")
        for s in ax_r.spines.values():
            s.set_visible(False)
        ax_r.set_xticks([])
        ax_r.set_yticks([])
        ax_r.add_patch(mp.Circle((0, 0), RP.detector_radius, fill=False, ec=(*ST.detector, RP.detector_alpha), lw=RP.detector_width * sc))
        ax_r.plot([0], [0], "+", color=PART, ms=RP.centre_marker_size * sc)
        R = ring_radius(beta)
        if R is not None:
            phis = np.linspace(0, 2 * math.pi, RP.ring_points)
            ax_r.plot(R * np.cos(phis), R * np.sin(phis), color=(*LIGHT, RP.ring_alpha), lw=RP.ring_width * sc)
            ax_r.add_patch(mp.Circle((0, 0), R, fill=False, ec=(*LIGHT, RP.halo_alpha), lw=RP.halo_width * sc))
            ax_r.text(0, RP.text_y, rf"$R=L\,\tan\theta={num(R, '.2f', lang)}\,L$", color=TXT, fontsize=RP.radius_size * sc, ha="center")
        else:
            ax_r.text(0, RP.text_y, tx["noring"], color=DIM, fontsize=RP.none_size * sc, ha="center")
        ax_r.set_title(tx["ring_title"], color=TXT, fontsize=RP.title_size * sc, loc="left")

        # ------------------------------------------------ titles and captions
        if t < CFG.timeline.caption_switch[0]:
            cap = tx["cap1"]
        elif t < CFG.timeline.caption_switch[1]:
            cap = tx["cap2"]
        else:
            cap = tx["cap3"].format(e=rf"${num(threshold_kinetic_energy(M_ELECTRON), '.2f', lang)}$", mu=rf"${num(threshold_kinetic_energy(M_MUON), '.0f', lang)}$")
        fig.text(*LY.caption_pos, cap, color=TXT, fontsize=LY.caption_size * sc)
        fig.text(*LY.title_pos, tx["title"], color=TXT, fontsize=LY.title_size * sc)
        fade_io = min(smooth(t_film, 0.0, V.fade_s), 1.0 - smooth(t_film, total - V.fade_s, total))
        fig.canvas.draw()
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
    ap.add_argument("--snapshot", type=float, default=None)
    ap.add_argument("--seconds", type=float, default=FILM_LENGTH)
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--config", type=Path, default=None, help="another configuration file instead of config.toml")
    ap.add_argument("--set", action="append", default=[], metavar="SECTION.KEY=VALUE", help="override one configuration value")
    args = ap.parse_args()
    V = CFG.video
    out = args.out or HERE / "media" / f"cherenkov_radiation_{args.lang}.mp4"
    out.parent.mkdir(parents=True, exist_ok=True)
    if args.snapshot is not None:
        render(out.with_suffix(".png"), (V.width, V.height), V.fps, args.seconds, args.lang, snap=args.snapshot)
    elif args.preview:
        render(out.with_name(out.stem + "_preview.mp4"), (V.preview_width, V.preview_height), V.preview_fps, args.seconds, args.lang)
    else:
        render(out, (V.width, V.height), V.fps, args.seconds, args.lang)


if __name__ == "__main__":
    main()
