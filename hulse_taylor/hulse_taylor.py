"""Hulse-Taylor binary pulsar PSR B1913+16: orbit, gravitational waves and the periastron shift.

Two neutron stars (1.438 and 1.390 solar masses) move on an eccentric Kepler orbit
(P_b = 7.75 h, e = 0.617).  They radiate gravitational waves, lose energy and spiral in, so
the orbital period decreases and the periastron arrives earlier and earlier than a model with
a constant period predicts.  The accumulated shift after a time t is

    delta(t) = (1/2) (dP_b/dt / P_b) t^2          (dP_b/dt < 0),

about -38 s after 30 years.  The film draws this curve next to the orbit.

Facts used (Weisberg and Huang, ApJ 829, 55 (2016), the numbers quoted in the book):
    P_b = 0.322997448918 d, e = 0.6171334, m1 = 1.438, m2 = 1.390 solar masses,
    dP_b/dt (GR)        = -2.40263e-12,
    dP_b/dt (intrinsic) = -2.398e-12   (observed, kinematic contribution subtracted),
    dP_b/dt (observed)  = -2.423e-12.
The orbit is a Kepler ellipse; the semi-major axis follows from Kepler's third law.

Not to scale (and labelled so in the film): the orbit shrinks by 5e-8 in 30 years and the
stars run ahead of the constant-period model by 0.5 degree (drawn 120 times larger); both are exaggerated.  The
gravitational-wave pattern is schematic: a rotating quadrupole with retarded phase, the
wavelength is about a thousand times smaller than the real one, and the amplitude
follows the instantaneous separation so that the bursts come at periastron.

Usage:
    python hulse_taylor.py                 # film -> media/hulse_taylor.mp4
    python hulse_taylor.py --preview
    python hulse_taylor.py --snapshot 20   # one PNG at film time 20 s
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

G_M_SUN = 1.32712440018e20          # m^3 s^-2
P_B = 0.322997448918 * 86400.0      # s
ECC = 0.6171334
M1, M2 = 1.438, 1.390               # solar masses
PDOT_GR = -2.40263e-12
PDOT_INTR = -2.398e-12
PDOT_OBS = -2.423e-12
YEAR = 365.25 * 86400.0
Y0, Y1 = 1975.0, 2005.0
SHRINK_VIEW = 0.075                 # drawn relative shrink of the orbit over 30 years
PHASE_EXAG = 120.0                   # exaggeration of the phase shift


def semi_major_axis() -> float:
    """Relative semi-major axis (m) from Kepler's third law."""
    mu = G_M_SUN * (M1 + M2)
    return (mu * P_B ** 2 / (4 * math.pi ** 2)) ** (1.0 / 3.0)


def periastron_shift(years: np.ndarray | float, pdot: float = PDOT_GR) -> np.ndarray | float:
    """Accumulated shift of the periastron time (s) relative to a constant period."""
    t = np.asarray(years, dtype=float) * YEAR
    return 0.5 * (pdot / P_B) * t ** 2


def fractional_shrink(years: float) -> float:
    """da/a after a number of years: a ~ P^(2/3) so da/a = (2/3) dP/P."""
    return (2.0 / 3.0) * (PDOT_GR / P_B) * years * YEAR


def kepler(mean_anomaly: np.ndarray, e: float = ECC) -> np.ndarray:
    """Eccentric anomaly from the mean anomaly (Newton iterations)."""
    m = np.mod(mean_anomaly, 2 * np.pi)
    E = np.where(e < 0.8, m + e * np.sin(m), np.pi * np.ones_like(m))
    for _ in range(12):
        E = E - (E - e * np.sin(E) - m) / (1 - e * np.cos(E))
    return E


def relative_orbit(mean_anomaly: np.ndarray, a: float = 1.0, e: float = ECC) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Relative position (x, y) with the periastron on the +x axis, and the separation."""
    E = kepler(mean_anomaly, e)
    x = a * (np.cos(E) - e)
    y = a * math.sqrt(1 - e * e) * np.sin(E)
    return x, y, a * (1 - e * np.cos(E))


# ---------------------------------------------------------------------- film

def smooth(x: float, a: float, b: float) -> float:
    t = min(max((x - a) / (b - a), 0.0), 1.0)
    return t * t * (3 - 2 * t)


def timeline(total: float) -> dict[str, tuple[float, float]]:
    k = total / 36.0
    return dict(waves=(3.0 * k, 6.5 * k), panel=(7.0 * k, 9.0 * k), years=(10.0 * k, 30.0 * k))


def year_at(t: float, tl: dict[str, tuple[float, float]]) -> float:
    a, b = tl["years"]
    u = min(max((t - a) / (b - a), 0.0), 1.0)
    return Y0 + (Y1 - Y0) * u


def render(out: Path, size: tuple[int, int], fps: int, total: float, snap: float | None = None) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.colors as mcolors
    import matplotlib.pyplot as plt
    from scipy.ndimage import gaussian_filter

    if snap is None and not shutil.which("ffmpeg"):
        sys.exit("ffmpeg is required")
    W, H = size
    dpi = 100
    sc = H / 720.0
    tl = timeline(total)
    cx0, cy0 = 0.335 * W, 0.50 * H
    A = 178.0 * sc                          # semi-major axis in pixels
    p_vis = 2.4                             # film seconds per orbit
    m_tot = M1 + M2
    f1, f2 = M2 / m_tot, M1 / m_tot         # star 1 (pulsar) moves on the smaller circle
    rng = np.random.default_rng(3)
    # stars in the background
    star_x = rng.uniform(0, W, 160)
    star_y = rng.uniform(0, H, 160)
    star_a = rng.uniform(0.15, 0.7, 160)
    # field grid for the gravitational waves (low resolution, upsampled)
    gw = int(0.62 * W / 4)
    gh = int(H / 4)
    gx = (np.arange(gw) * 4 + 2) - (cx0 - 0.0)
    gy = (np.arange(gh) * 4 + 2) - cy0
    GX, GY = np.meshgrid(gx, gy[::-1] * 1.0)
    rho = np.hypot(GX, GY)
    psi = np.arctan2(GY, GX)
    edge = np.clip(np.minimum(np.minimum(np.arange(gw)[None, :], gw - 1 - np.arange(gw)[None, :]),
                              np.minimum(np.arange(gh)[:, None], gh - 1 - np.arange(gh)[:, None])) / 22.0, 0.0, 1.0)
    edge = edge * edge * (3 - 2 * edge)
    c_vis = 330.0 * sc / p_vis * 0.62       # pixels per film second

    fig = plt.figure(figsize=(W / dpi, H / dpi), dpi=dpi, facecolor="#03060c")
    ax = fig.add_axes([0, 0, 1, 1])
    pa = fig.add_axes([0.675, 0.24, 0.285, 0.47], facecolor="none")
    cmap = mcolors.LinearSegmentedColormap.from_list("gw", ["#d1397a", "#0b1020", "#2fd0c4"])

    frames = int(round(total * fps))
    ids = [int(round(snap * fps))] if snap is not None else range(frames)
    writer = None
    if snap is None:
        writer = subprocess.Popen(
            ["ffmpeg", "-y", "-v", "error", "-f", "rawvideo", "-pix_fmt", "rgba", "-s", f"{W}x{H}",
             "-r", str(fps), "-i", "-", "-c:v", "libx264", "-preset", "slow", "-crf", "19",
             "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(out)], stdin=subprocess.PIPE)

    def glow(X, Y, color, d, alpha=1.0, z=6):
        for scale, al in ((5.0, 0.05), (2.8, 0.13), (1.6, 0.32), (0.75, 1.0)):
            ax.scatter([X], [Y], s=(d * scale * 72 / dpi) ** 2, c=[mcolors.to_rgba(color, al * alpha)],
                       linewidths=0, zorder=z)

    ya_all = np.linspace(Y0, Y1, 300)
    shift_gr = periastron_shift(ya_all - Y0)
    shift_intr = periastron_shift(ya_all - Y0, PDOT_INTR)

    for k in ids:
        t = k / fps
        year = year_at(t, tl)
        dyr = year - Y0
        u = dyr / (Y1 - Y0)
        a_now = A * (1 - SHRINK_VIEW * u)
        # exaggerated phase advance of the real orbit relative to the constant-period model
        dphase = -2 * math.pi * periastron_shift(dyr) / P_B * PHASE_EXAG
        phase = 2 * math.pi * t / p_vis
        ax.clear()
        ax.set_xlim(0, W)
        ax.set_ylim(0, H)
        ax.axis("off")
        ax.add_patch(matplotlib.patches.Rectangle((0, 0), W, H, color="#03060c", zorder=0, lw=0))
        ax.scatter(star_x, star_y, s=0.8 * sc ** 2 * 6, c=[(0.8, 0.88, 1.0, a_) for a_ in star_a], linewidths=0, zorder=1)
        # gravitational waves: rotating quadrupole, retarded phase, amplitude follows the separation
        w_alpha = smooth(t, *tl["waves"])
        if w_alpha > 0.01:
            t_ret = t - rho / c_vis
            M_ret = 2 * math.pi * t_ret / p_vis + dphase
            xr, yr, sep = relative_orbit(M_ret, 1.0)
            ang = np.arctan2(yr, xr)
            amp = 0.4 + 0.6 * ((1.0 - ECC) / sep) ** 2.2
            field = 1.6 * amp * np.cos(2 * (psi - ang)) / np.sqrt(1.0 + rho / (90 * sc))
            field *= np.clip((rho - 28 * sc) / (50 * sc), 0, 1)
            field *= np.exp(-rho / (520 * sc)) * edge
            img = cmap(0.5 + 0.5 * np.clip(field * 2.0, -1, 1))
            img[..., 3] = np.clip(np.abs(field) * 1.5, 0, 0.55) * w_alpha
            ax.imshow(img, extent=(cx0 + gx[0] - 2, cx0 + gx[-1] + 2, cy0 - gy[0] - 2, cy0 - gy[-1] + 2), origin="upper",
                      interpolation="bicubic", zorder=2, aspect="auto")
        # full orbit (dashed) of both stars
        Mo = np.linspace(0, 2 * math.pi, 400)
        xo, yo, _ = relative_orbit(Mo, a_now)
        ax.plot(cx0 + f1 * xo, cy0 + f1 * yo, color="#ffd27a", lw=0.8 * sc, alpha=0.28, ls=(0, (2, 4)), zorder=3)
        ax.plot(cx0 - f2 * xo, cy0 - f2 * yo, color="#8fe3ff", lw=0.8 * sc, alpha=0.28, ls=(0, (2, 4)), zorder=3)
        # the stars
        xs, ys, sep = relative_orbit(np.array([phase + dphase]), a_now)
        p1 = (cx0 + f1 * xs[0], cy0 + f1 * ys[0])
        p2 = (cx0 - f2 * xs[0], cy0 - f2 * ys[0])
        # constant-period model (ghost) when the shift is visible
        if dphase != 0.0 and abs(dphase) > 0.03:
            xg, yg, _ = relative_orbit(np.array([phase]), A)
            g_alpha = min(1.0, abs(dphase) / 0.2)
            for (gxp, gyp, col) in ((cx0 + f1 * xg[0], cy0 + f1 * yg[0], "#ffd27a"), (cx0 - f2 * xg[0], cy0 - f2 * yg[0], "#8fe3ff")):
                ax.scatter([gxp], [gyp], s=(16 * sc * 72 / dpi) ** 2, facecolors="none",
                           edgecolors=[mcolors.to_rgba(col, 0.7 * g_alpha)], linewidths=1.2 * sc, zorder=5)
        # trails
        for frac, col, (px, py) in ((f1, "#ffd27a", p1), (-f2, "#8fe3ff", p2)):
            Mt = phase + dphase - np.linspace(0, 1.1, 40)
            xt, yt, _ = relative_orbit(Mt, a_now)
            tx = cx0 + frac * xt
            ty = cy0 + frac * yt
            for j in range(39):
                ax.plot(tx[j:j + 2], ty[j:j + 2], color=col, lw=2.2 * sc, alpha=0.55 * (1 - j / 39) ** 1.5, zorder=4, solid_capstyle="round")
        glow(p1[0], p1[1], "#ffd27a", 17 * sc)
        glow(p2[0], p2[1], "#8fe3ff", 16.5 * sc)
        # periastron mark and labels
        xp, yp, _ = relative_orbit(np.array([0.0]), a_now)
        ax.scatter([cx0 + f1 * xp[0] - f2 * xp[0] * 0], [cy0], s=0, zorder=3)
        lab_a = smooth(t, 0.8, 2.0) * (1 - smooth(t, 8.0, 9.5))
        if lab_a > 0.01:
            ax.annotate("pulsar 1.44 M$_\\odot$", (p1[0], p1[1]), xytext=(p1[0] + 24 * sc, p1[1] + 22 * sc), color=(1.0, 0.85, 0.5, lab_a),
                        fontsize=11 * sc, arrowprops=dict(arrowstyle="-", color=(1, 0.85, 0.5, 0.5 * lab_a), lw=0.8), zorder=9)
            ax.annotate("1.39 M$_\\odot$", (p2[0], p2[1]), xytext=(p2[0] - 70 * sc, p2[1] - 30 * sc), color=(0.6, 0.9, 1.0, lab_a),
                        fontsize=11 * sc, arrowprops=dict(arrowstyle="-", color=(0.6, 0.9, 1, 0.5 * lab_a), lw=0.8), zorder=9)
        # graph
        pa.clear()
        pa.set_facecolor("none")
        pn = smooth(t, *tl["panel"])
        fig.texts.clear()
        if pn > 0.01:
            pa.set_xlim(Y0, Y1)
            pa.set_ylim(-43, 2)
            for name, sp in pa.spines.items():
                sp.set_visible(name in ("left", "bottom"))
                sp.set_color((0.75, 0.82, 0.92, 0.55 * pn))
            pa.axhline(0, color=(0.75, 0.82, 0.92, 0.3 * pn), lw=0.9 * sc, ls=(0, (3, 4)))
            m = ya_all <= year
            if t >= tl["years"][0] - 0.01:
                pa.fill_between(ya_all[m], shift_gr[m], shift_intr[m], color=(1.0, 0.65, 0.25, 0.25 * pn))
                pa.plot(ya_all[m], shift_gr[m], color=(0.35, 0.85, 0.9, pn), lw=3.0 * sc, solid_capstyle="round")
                pa.plot(ya_all[m], shift_intr[m], color=(1.0, 0.68, 0.3, pn), lw=1.8 * sc, ls=(0, (4, 3)))
                now = float(periastron_shift(dyr))
                pa.scatter([year], [now], s=(11 * sc) ** 2 * 0.6, c=[(1, 1, 1, pn)], zorder=5, linewidths=0)
                pa.scatter([year], [now], s=(26 * sc) ** 2 * 0.6, c=[(0.35, 0.85, 0.9, 0.25 * pn)], zorder=4, linewidths=0)
            else:
                now = 0.0
            pa.set_xticks([1975, 1985, 1995, 2005])
            pa.set_xticklabels(["1975", "1985", "1995", "2005"], color=(0.82, 0.9, 1.0, 0.85 * pn), fontsize=10 * sc)
            pa.set_yticks([0, -10, -20, -30, -40])
            pa.set_yticklabels(["0", "−10", "−20", "−30", "−40"], color=(0.82, 0.9, 1.0, 0.85 * pn), fontsize=10 * sc)
            pa.tick_params(colors=(0.75, 0.82, 0.92, 0.5 * pn), length=3)
            pa.set_xlabel("year / год", color=(0.82, 0.9, 1.0, 0.85 * pn), fontsize=10.5 * sc, labelpad=2)
            fig.text(0.675, 0.865, "periastron shift / сдвиг периастра", color=(0.82, 0.9, 1.0, pn), fontsize=14 * sc)
            fig.text(0.675, 0.775, f"{now:6.1f} s", color=(0.45, 0.9, 0.95, pn), fontsize=34 * sc, family="monospace")
            fig.text(0.855, 0.790, f"{year:6.0f}", color=(1.0, 0.82, 0.55, pn), fontsize=24 * sc, family="monospace")
            fig.text(0.675, 0.13, "—  GR prediction / предсказание ОТО", color=(0.35, 0.85, 0.9, 0.9 * pn), fontsize=10 * sc)
            fig.text(0.675, 0.095, "- -  measured Ṗ$_b$ (corrected) / измеренное Ṗ$_b$", color=(1.0, 0.68, 0.3, 0.9 * pn), fontsize=10 * sc)
            fig.text(0.675, 0.052, "ratio / отношение 0.9983 ± 0.0016", color=(0.7, 0.78, 0.9, 0.85 * pn), fontsize=10 * sc)
        else:
            pa.axis("off")
        # captions
        if t < tl["panel"][0]:
            cap = ("PSR B1913+16: two neutron stars on an eccentric orbit, P = 7.75 h, e = 0.62",
                   "PSR B1913+16: две нейтронные звезды на вытянутой орбите, период 7,75 ч, e = 0.62")
        elif t < tl["years"][0]:
            cap = ("they radiate gravitational waves and lose energy", "они излучают гравитационные волны и теряют энергию")
        elif year < Y1 - 0.01:
            cap = ("the orbit shrinks, the period falls, the periastron comes earlier than a constant period predicts",
                   "орбита сжимается, период падает, периастр приходит раньше, чем при постоянном периоде")
        else:
            cap = ("30 years of timing agree with general relativity", "30 лет наблюдений согласуются с ОТО")
        ax.add_patch(matplotlib.patches.Rectangle((0, 0), 0.66 * W, 0.12 * H, color=(0.01, 0.02, 0.05, 0.6), zorder=11, lw=0))
        fig.text(0.03, 0.062, cap[0], color=(0.88, 0.93, 1.0, 0.93), fontsize=12 * sc)
        fig.text(0.03, 0.024, cap[1], color=(0.6, 0.68, 0.8, 0.9), fontsize=10.5 * sc)
        fig.text(0.03, 0.935, "not to scale: orbit shrink and phase shift exaggerated, wave pattern schematic / "
                 "не в масштабе: сжатие орбиты и сдвиг фазы преувеличены, волны схематичны", color=(0.5, 0.58, 0.72, 0.8),
                 fontsize=8.5 * sc)
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
    ap.add_argument("--seconds", type=float, default=36.0)
    ap.add_argument("--out", type=Path, default=HERE / "media" / "hulse_taylor.mp4")
    args = ap.parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    if args.snapshot is not None:
        render(args.out.with_suffix(".png"), (1280, 720), 30, args.seconds, snap=args.snapshot)
    elif args.preview:
        render(args.out.with_name("hulse_taylor_preview.mp4"), (640, 360), 15, 8.0)
    else:
        render(args.out, (1280, 720), 30, args.seconds)


if __name__ == "__main__":
    main()
