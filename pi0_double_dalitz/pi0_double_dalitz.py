"""The decay pi0 -> e+ e- e+ e-: the planes of the two pairs and the angle between them.

A neutral pion decays into two photons and each virtual photon turns into an e+ e- pair
(the "double Dalitz" decay).  Every pair spans a plane that contains the photon axis.  From one
event to the next these planes have a different orientation; what the experiment measures is the
angle phi between them, and its distribution reveals the parity of the pion.

For a pseudoscalar pion the rate is proportional to |E1 . B2|^2: it is largest for orthogonal planes,

    dN/dphi  ~  1 - a cos(2 phi),      phi in [0, 2 pi),

with two humps at phi = pi/2 and 3 pi/2 (for a scalar particle the humps would be at 0 and pi).
In the film a = 0.35 reproduces the height of the humps of the KTeV histogram (fig. 40.6 of the
book); the amplitude is illustrative.  The azimuth of the first plane is uniform, the second is rotated
by phi drawn from this distribution (rejection sampling).

The picture is schematic: the opening angle of a pair is exaggerated and the events are accelerated.

Usage:
    python pi0_double_dalitz.py                 # film -> media/pi0_double_dalitz.mp4
    python pi0_double_dalitz.py --preview
    python pi0_double_dalitz.py --snapshot 20   # one PNG at film time 20 s
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
A_AMP = 0.35
D_CONV = 1.5            # distance at which a photon converts (units of the scene)
Z_END = 3.1
VEL = 5.0               # speed of the particles, scene units per second
N_BINS = 24


def density(phi: np.ndarray | float, a: float = A_AMP) -> np.ndarray | float:
    """dN/dphi normalized on [0, 2 pi): (1 - a cos 2 phi) / (2 pi)."""
    return (1.0 - a * np.cos(2.0 * np.asarray(phi))) / (2.0 * math.pi)


def sample_phi(rng: np.random.Generator, n: int, a: float = A_AMP) -> np.ndarray:
    out: list[float] = []
    while len(out) < n:
        phi = rng.uniform(0, 2 * math.pi, 2 * n)
        u = rng.uniform(0, 1 + a, 2 * n)
        out.extend(phi[u < 1 - a * np.cos(2 * phi)])
    return np.array(out[:n])


def make_events(seed: int, total: float):
    rng = np.random.default_rng(seed)
    events = []
    t = 2.5
    events.append(dict(t=t, slow=2.2))
    t = 9.0
    dt = 0.75
    while t < total - 2.2:
        events.append(dict(t=t, slow=1.0))
        t += dt
        dt = max(0.14, dt * 0.93)
    phis = sample_phi(rng, len(events))
    for ev, phi in zip(events, phis):
        ev["phi"] = float(phi)
        ev["azim1"] = float(rng.uniform(0, 2 * math.pi))
    events[0]["phi"] = 0.5 * math.pi + 0.25      # the first, slow event: planes nearly orthogonal
    events[0]["azim1"] = 0.6
    return events


def smooth(x: float, a: float, b: float) -> float:
    u = min(max((x - a) / (b - a), 0.0), 1.0)
    return u * u * (3 - 2 * u)


def render(out: Path, size: tuple[int, int], fps: int, total: float, snap: float | None = None) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.colors as mcolors
    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection, PolyCollection

    if snap is None and not shutil.which("ffmpeg"):
        sys.exit("ffmpeg is required")
    W, H = size
    dpi = 100
    sc = H / 720.0
    cx0, cy0 = 0.335 * W, 0.50 * H
    S = 86.0 * sc
    az, el = math.radians(36.0), math.radians(20.0)

    def proj(x, y, z):
        x, y, z = np.asarray(x, float), np.asarray(y, float), np.asarray(z, float)
        X = x * math.cos(az) - y * math.sin(az)
        Y = z * math.cos(el) + (x * math.sin(az) + y * math.cos(az)) * math.sin(el)
        return cx0 + S * X, cy0 + S * Y

    events = make_events(6, total)
    t_conv = [e["t"] + D_CONV / VEL * e["slow"] for e in events]
    phis_all = np.array([e["phi"] for e in events])
    edges = np.linspace(0, 2 * math.pi, N_BINS + 1)
    centers = 0.5 * (edges[1:] + edges[:-1])

    fig = plt.figure(figsize=(W / dpi, H / dpi), dpi=dpi, facecolor="#03060c")
    ax = fig.add_axes([0, 0, 1, 1])
    dial = fig.add_axes([0.745, 0.585, 0.20, 0.30], facecolor="none")
    hist = fig.add_axes([0.745, 0.215, 0.225, 0.26], facecolor="none")
    rng_bg = np.random.default_rng(2)
    stars = rng_bg.uniform(0, 1, (110, 2))

    frames = int(round(total * fps))
    ids = [int(round(snap * fps))] if snap is not None else range(frames)
    writer = None
    if snap is None:
        writer = subprocess.Popen(
            ["ffmpeg", "-y", "-v", "error", "-f", "rawvideo", "-pix_fmt", "rgba", "-s", f"{W}x{H}",
             "-r", str(fps), "-i", "-", "-c:v", "libx264", "-preset", "slow", "-crf", "19",
             "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(out)], stdin=subprocess.PIPE)

    def glow(X, Y, color, d, alpha=1.0, z=8):
        X = np.atleast_1d(X)
        Y = np.atleast_1d(Y)
        al = np.broadcast_to(np.atleast_1d(alpha), X.shape)
        for scale, a_ in ((4.0, 0.07), (2.2, 0.17), (1.2, 0.42), (0.55, 1.0)):
            ax.scatter(X, Y, s=(d * scale * 72 / dpi) ** 2, linewidths=0, zorder=z,
                       facecolors=[mcolors.to_rgba(color, a_ * float(x)) for x in al])

    col_pos, col_neg, col_p1, col_p2 = "#ff8a3a", "#4cc3ff", "#ff6b9a", "#6b8cff"

    for k_ in ids:
        t = k_ / fps
        ax.clear()
        ax.set_xlim(0, W)
        ax.set_ylim(0, H)
        ax.axis("off")
        ax.add_patch(matplotlib.patches.Rectangle((0, 0), W, H, color="#03060c", zorder=0, lw=0))
        ax.scatter(stars[:, 0] * W, stars[:, 1] * H, s=1.0 * sc ** 2 * 5, c=[(0.8, 0.88, 1, 0.12)] * len(stars), linewidths=0, zorder=1)
        # the axis of the photons, faint
        xa, ya = proj([0, 0], [0, 0], [-3.4, 3.4])
        ax.plot(xa, ya, color=(0.7, 0.8, 0.95, 0.18), lw=1.0 * sc, ls=(0, (3, 5)), zorder=2)
        polys, pcols = [], []
        lines, lcols, lws = [], [], []
        last = None
        n_done = 0
        for ev, tc in zip(events, t_conv):
            slow = ev["slow"]
            tau = (t - ev["t"]) / slow
            if tau < 0 or tau > 1.9:
                if t >= tc:
                    n_done += 1
                continue
            if t >= tc:
                n_done += 1
            last = ev
            fade = min(smooth(tau, 0.0, 0.1), 1 - smooth(tau, 1.3, 1.9))
            phi1, phi2 = ev["azim1"], ev["azim1"] + ev["phi"]
            tau_c = D_CONV / VEL
            # photons (wavy lines)
            if tau > 0.05:
                for sgn in (1, -1):
                    L = min(VEL * (tau - 0.05), D_CONV)
                    zz = np.linspace(0, L, 60)
                    wav = 0.07 * np.sin(zz * 16)
                    X, Y = proj(wav, 0 * zz, sgn * zz)
                    lines.append(np.column_stack([X, Y]))
                    lcols.append((1.0, 0.95, 0.6, 0.8 * fade))
                    lws.append(2.0 * sc)
            if tau >= tau_c:
                f2 = smooth(tau, tau_c, tau_c + 0.15) * fade
                for sgn, phi, col in ((1, phi1, col_p1), (-1, phi2, col_p2)):
                    u = np.array([math.cos(phi), math.sin(phi)])
                    z0, z1, w = D_CONV * 0.55, Z_END, 1.15
                    corners = []
                    for (zz, ww) in ((z0, -w), (z0, w), (z1, w), (z1, -w)):
                        X, Y = proj(ww * u[0], ww * u[1], sgn * zz)
                        corners.append((float(X), float(Y)))
                    polys.append(corners)
                    pcols.append(mcolors.to_rgba(col, 0.13 * f2))
                    # the pair
                    run = VEL * (tau - tau_c)
                    for (s_t, colr) in ((+1, col_pos), (-1, col_neg)):
                        ang = 0.34
                        zpos = D_CONV + run * math.cos(ang)
                        tr = run * math.sin(ang) * s_t
                        X, Y = proj(tr * u[0], tr * u[1], sgn * zpos)
                        X0, Y0 = proj(0 * u[0], 0 * u[1], sgn * D_CONV)
                        lines.append(np.array([[float(X0), float(Y0)], [float(X), float(Y)]]))
                        lcols.append(mcolors.to_rgba(colr, 0.55 * fade))
                        lws.append(2.2 * sc)
                        glow(X, Y, colr, 11 * sc, fade, z=9)
        if polys:
            ax.add_collection(PolyCollection(polys, facecolors=pcols, edgecolors=[(1, 1, 1, 0.12)] * len(polys), linewidths=0.7 * sc, zorder=3))
        if lines:
            ax.add_collection(LineCollection(lines, colors=lcols, linewidths=lws, zorder=4, capstyle="round"))
        # pi0 and the decay flash
        x0, y0 = proj(0, 0, 0)
        pi_alpha = 1.0
        glow(float(x0), float(y0), "#c9d4e6", 26 * sc, pi_alpha, z=10)
        ax.text(float(x0) + 18 * sc, float(y0) - 30 * sc, "π⁰", color=(0.85, 0.9, 1, 0.95), fontsize=15 * sc, zorder=12)
        for ev in events:
            tau = (t - ev["t"]) / ev["slow"]
            if 0 <= tau < 0.2:
                f = 1 - tau / 0.2
                glow(float(x0), float(y0), "#fff4d0", (30 + 90 * (1 - f)) * sc, 0.9 * f, z=11)
        # the dial: end view along the photon axis
        dial.clear()
        dial.set_facecolor("none")
        dial.set_xlim(-1.35, 1.35)
        dial.set_ylim(-1.35, 1.35)
        dial.set_aspect("equal")
        dial.axis("off")
        a_ = np.linspace(0, 2 * math.pi, 200)
        dial.plot(np.cos(a_), np.sin(a_), color=(0.7, 0.8, 0.95, 0.35), lw=1.0 * sc)
        if last is not None:
            phi1, phi2 = last["azim1"], last["azim1"] + last["phi"]
            for ph, col in ((phi1, col_p1), (phi2, col_p2)):
                dial.plot([-math.cos(ph), math.cos(ph)], [-math.sin(ph), math.sin(ph)], color=col, lw=3.0 * sc, alpha=0.9, solid_capstyle="round")
                dial.annotate("", xy=(1.12 * math.cos(ph), 1.12 * math.sin(ph)), xytext=(0.7 * math.cos(ph), 0.7 * math.sin(ph)),
                              arrowprops=dict(arrowstyle="-|>", color=col, lw=2.0 * sc))
            arc = np.linspace(phi1, phi2, 60)
            dial.plot(0.45 * np.cos(arc), 0.45 * np.sin(arc), color=(1, 1, 1, 0.8), lw=1.4 * sc)
            ph_deg = math.degrees(last["phi"]) % 360
            dial.text(0, -1.33, f"φ = {ph_deg:5.0f}°", color=(1, 0.88, 0.5, 1), fontsize=13 * sc, ha="center", va="top", family="monospace")
        fig.texts.clear()
        fig.text(0.745, 0.905, "planes of the pairs, end view", color=(0.82, 0.9, 1, 0.9), fontsize=9.5 * sc)
        fig.text(0.745, 0.883, "плоскости пар, вид вдоль оси", color=(0.6, 0.68, 0.8, 0.9), fontsize=8.5 * sc)
        # the histogram
        hist.clear()
        hist.set_facecolor("none")
        done = phis_all[[tc <= t for tc in t_conv]]
        counts, _ = np.histogram(done, bins=edges)
        hist.bar(centers / math.pi, counts, width=(edges[1] - edges[0]) / math.pi * 0.86, color=(0.55, 0.9, 0.78, 0.85))
        n = len(done)
        grid = np.linspace(0, 2 * math.pi, 200)
        hist.plot(grid / math.pi, density(grid) * (edges[1] - edges[0]) * max(n, 1), color=(1.0, 0.82, 0.4, 0.95), lw=2.0 * sc)
        hist.set_xlim(0, 2)
        hist.set_ylim(0, max(4, 1.5 * (1 + A_AMP) / (2 * math.pi) * (edges[1] - edges[0]) * max(n, 1) + 2))
        for name, sp in hist.spines.items():
            sp.set_visible(name in ("left", "bottom"))
            sp.set_color((0.75, 0.82, 0.92, 0.5))
        hist.set_xticks([0, 0.5, 1, 1.5, 2])
        hist.set_xticklabels(["0", "0.5", "1", "1.5", "2"], color=(0.82, 0.9, 1, 0.85), fontsize=9.5 * sc)
        hist.set_yticks([])
        hist.set_xlabel("φ / π", color=(0.82, 0.9, 1, 0.9), fontsize=10 * sc, labelpad=0)
        fig.text(0.745, 0.505, f"events / события: {n}", color=(0.55, 1.0, 0.8, 1), fontsize=12 * sc, family="monospace")
        fig.text(0.745, 0.115, "—  1 − a cos 2φ   (pseudoscalar / псевдоскаляр)", color=(1.0, 0.82, 0.4, 0.95), fontsize=9.5 * sc)
        fig.text(0.745, 0.078, "a is illustrative / амплитуда условная", color=(0.6, 0.68, 0.8, 0.85), fontsize=8.5 * sc)
        fig.text(0.03, 0.925, "π⁰ → e⁺e⁻ e⁺e⁻", color=(0.9, 0.95, 1, 0.95), fontsize=19 * sc)
        # captions
        if t < events[1]["t"] - 0.4:
            cap = ("each virtual photon becomes an e⁺e⁻ pair; every pair spans a plane", "каждый виртуальный фотон превращается в пару e⁺e⁻, каждая пара задаёт плоскость")
        else:
            cap = ("in every event the planes are oriented differently; the distribution of the angle φ builds up",
                   "в каждом событии плоскости ориентированы по-своему; копится распределение по углу φ")
        ax.add_patch(matplotlib.patches.Rectangle((0, 0), 0.70 * W, 0.105 * H, color=(0.01, 0.02, 0.05, 0.8), zorder=13, lw=0))
        fig.text(0.03, 0.053, cap[0], color=(0.88, 0.93, 1.0, 0.93), fontsize=11.5 * sc)
        fig.text(0.03, 0.02, cap[1], color=(0.6, 0.68, 0.8, 0.9), fontsize=9.8 * sc)
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
    ap.add_argument("--seconds", type=float, default=34.0)
    ap.add_argument("--out", type=Path, default=HERE / "media" / "pi0_double_dalitz.mp4")
    args = ap.parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    if args.snapshot is not None:
        render(args.out.with_suffix(".png"), (1280, 720), 30, args.seconds, snap=args.snapshot)
    elif args.preview:
        render(args.out.with_name("pi0_double_dalitz_preview.mp4"), (640, 360), 15, 12.0)
    else:
        render(args.out, (1280, 720), 30, args.seconds)


if __name__ == "__main__":
    main()
