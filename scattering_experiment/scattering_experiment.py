"""A collider scattering experiment: e+ e- -> mu+ mu-.

The three stages of a typical scattering experiment, as in the chapter on the S-matrix:

1. preparation of the initial state: two counter-propagating beams are accelerated;
2. interaction: the particles meet in a localized region; most pass through each other,
   rarely an e+ e- pair annihilates and a mu+ mu- pair is born;
3. measurement of the final state: a detector around the interaction region records where
   the muons go.

Many bunch crossings are then repeated: the muon directions build up the angular
distribution of e+ e- -> mu+ mu- at high energy,

    dsigma/dOmega  ~  1 + cos^2(theta),       dN/dcos(theta) = N (3/8) (1 + cos^2 theta),

theta being the angle between mu- and the e- beam, and the number of events per crossing is
the cross section times the flux (the definition of the cross section in the book).

The muon directions are drawn from this distribution (rejection sampling, azimuth uniform) and
the picture is a side view: the beam axis is horizontal, the detector is a sphere seen from the
side.  Everything else is schematic: the rate of events is hugely exaggerated and the bunches are
drawn as a few tens of particles.

Usage:
    python scattering_experiment.py                 # film -> media/scattering_experiment.mp4
    python scattering_experiment.py --preview
    python scattering_experiment.py --snapshot 15   # one PNG at film time 15 s
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
N_BUNCH = 56
RING = 250.0            # detector radius in pixels at 720p
SIGMA_X = 30.0
SIGMA_Y = 8.0


def sample_cos_theta(rng: np.random.Generator, n: int) -> np.ndarray:
    """cos(theta) with density (3/8)(1 + c^2) on [-1, 1] (rejection sampling)."""
    out: list[float] = []
    while len(out) < n:
        c = rng.uniform(-1, 1, 2 * n)
        u = rng.uniform(0, 2, 2 * n)
        out.extend(c[u < 1 + c * c])
    return np.array(out[:n])


def angular_density(c: np.ndarray) -> np.ndarray:
    return 0.375 * (1.0 + c * c)


def make_events(seed: int, total: float):
    """List of bunch crossings with their events."""
    rng = np.random.default_rng(seed)
    crossings = []
    t_first = 4.6
    times = [t_first] + list(np.arange(10.3, total - 2.4, 0.78))
    for k, t_c in enumerate(times):
        dx_e = rng.normal(0, SIGMA_X, N_BUNCH)
        dy_e = rng.normal(0, SIGMA_Y, N_BUNCH)
        dx_p = rng.normal(0, SIGMA_X, N_BUNCH)
        dy_p = rng.normal(0, SIGMA_Y, N_BUNCH)
        n_ev = 1 if k == 0 else int(rng.poisson(5.0))
        events = []
        used_e: set[int] = set()
        used_p: set[int] = set()
        order_e = np.argsort(np.abs(dx_e))
        order_p = np.argsort(np.abs(dx_p))
        for _ in range(n_ev):
            ie = int(rng.choice([i for i in order_e[:30] if i not in used_e]))
            ip = int(rng.choice([i for i in order_p[:30] if i not in used_p]))
            used_e.add(ie)
            used_p.add(ip)
            c = float(sample_cos_theta(rng, 1)[0])
            phi = rng.uniform(0, 2 * math.pi)
            if k == 0:
                c, phi = 0.62, 0.9               # a clear, readable first event
            events.append(dict(ie=ie, ip=ip, c=c, phi=phi))
        crossings.append(dict(t_c=float(t_c), first=(k == 0), dx_e=dx_e, dy_e=dy_e, dx_p=dx_p, dy_p=dy_p, events=events))
    return crossings


def xc_of(t: float, cr: dict, v: float) -> float:
    """Centre of the e- bunch (the e+ bunch is at -xc)."""
    t_c = cr["t_c"]
    if cr["first"]:
        # accelerating beam: x = -d (1 - (t/t_c)^1.8), after the crossing it keeps its speed
        d = 400.0
        if t <= t_c:
            return -d * (1 - (max(t, 0.0) / t_c) ** 1.8)
        return d * 1.8 / t_c * (t - t_c)
    return v * (t - t_c)


def event_time_and_point(ev: dict, cr: dict, v: float) -> tuple[float, float, float]:
    """Time and place where e-_i and e+_j meet."""
    dxi, dxj = cr["dx_e"][ev["ie"]], cr["dx_p"][ev["ip"]]
    xc = 0.5 * (dxj - dxi)                  # e-: xc + dxi = e+: -xc + dxj
    t_c = cr["t_c"]
    if cr["first"]:
        d = 400.0
        speed = d * 1.8 / t_c
        t = t_c + xc / speed
    else:
        t = t_c + xc / v
    ym = 0.5 * (cr["dy_e"][ev["ie"]] + cr["dy_p"][ev["ip"]])
    return float(t), 0.5 * (dxi + dxj), float(ym)


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
    cx0, cy0 = 0.355 * W, 0.51 * H
    v_fast = 560.0 * sc
    ring = RING * sc
    crossings = make_events(4, total)
    # flatten events with absolute times
    flat = []
    for k, cr in enumerate(crossings):
        v = v_fast
        for ev in cr["events"]:
            t_ev, xm, ym = event_time_and_point(ev, cr, v / sc)
            slow = 2.6 if cr["first"] else 1.0
            nx, ny, nz = ev["c"], math.sqrt(1 - ev["c"] ** 2) * math.cos(ev["phi"]), math.sqrt(1 - ev["c"] ** 2) * math.sin(ev["phi"])
            speed3 = 760.0 * sc / slow
            t_hit = t_ev + ring / speed3
            flat.append(dict(k=k, t=t_ev, xm=xm, ym=ym, n=(nx, ny, nz), speed=speed3, t_hit=t_hit, c=ev["c"], first=cr["first"]))
    t_hits = np.array([e["t_hit"] for e in flat])
    cs = np.array([e["c"] for e in flat])

    fig = plt.figure(figsize=(W / dpi, H / dpi), dpi=dpi, facecolor="#03050b")
    ax = fig.add_axes([0, 0, 1, 1])
    ph = fig.add_axes([0.725, 0.27, 0.245, 0.34], facecolor="none")
    rng_bg = np.random.default_rng(8)
    stars = rng_bg.uniform(0, 1, (120, 2))
    edges = np.linspace(-1, 1, 17)
    centers = 0.5 * (edges[1:] + edges[:-1])

    frames = int(round(total * fps))
    ids = [int(round(snap * fps))] if snap is not None else range(frames)
    writer = None
    if snap is None:
        writer = subprocess.Popen(
            ["ffmpeg", "-y", "-v", "error", "-f", "rawvideo", "-pix_fmt", "rgba", "-s", f"{W}x{H}",
             "-r", str(fps), "-i", "-", "-c:v", "libx264", "-preset", "slow", "-crf", "19",
             "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(out)], stdin=subprocess.PIPE)

    def glow(X, Y, color, d, alpha, z=6):
        X = np.atleast_1d(X)
        Y = np.atleast_1d(Y)
        al = np.broadcast_to(np.atleast_1d(alpha), X.shape)
        for scale, a_ in ((4.0, 0.06), (2.2, 0.15), (1.2, 0.4), (0.55, 1.0)):
            ax.scatter(X, Y, s=(d * scale * 72 / dpi) ** 2, linewidths=0, zorder=z,
                       facecolors=[mcolors.to_rgba(color, a_ * float(x)) for x in al])

    col_e, col_p, col_mu_m, col_mu_p = "#4cc3ff", "#ff9a3a", "#7dffb0", "#ff6bd0"

    for k_ in ids:
        t = k_ / fps
        ax.clear()
        ax.set_xlim(0, W)
        ax.set_ylim(0, H)
        ax.axis("off")
        ax.add_patch(matplotlib.patches.Rectangle((0, 0), W, H, color="#03050b", zorder=0, lw=0))
        ax.scatter(stars[:, 0] * W, stars[:, 1] * H, s=1.0 * sc ** 2 * 5, c=[(0.8, 0.88, 1, 0.12)] * len(stars), linewidths=0, zorder=1)
        # beam pipe, interaction region and detector
        for yy in (-11, 11):
            ax.plot([0, 0.70 * W], [cy0 + yy * sc, cy0 + yy * sc], color="#6f86a8", lw=1.0 * sc, alpha=0.35, zorder=2)
        a_ = np.linspace(0, 2 * math.pi, 300)
        ax.plot(cx0 + ring * np.cos(a_), cy0 + ring * np.sin(a_), color="#9fb7d6", lw=2.2 * sc, alpha=0.30, zorder=2)
        ax.plot(cx0 + (ring + 9 * sc) * np.cos(a_), cy0 + (ring + 9 * sc) * np.sin(a_), color="#9fb7d6", lw=0.7 * sc, alpha=0.2, zorder=2)
        ax.add_patch(matplotlib.patches.Circle((cx0, cy0), 38 * sc, fill=False, ec="#d8e6ff", lw=0.9 * sc, ls=(0, (2, 3)), alpha=0.35, zorder=2))
        # accelerating cavities along the pipe (stage 1)
        for i, xx in enumerate((-390, -320, -250, 250, 320, 390)):
            pulse = 0.5 + 0.5 * math.sin(6.0 * t - 0.7 * abs(xx) / 70 * (1 if xx < 0 else -1))
            a_cav = (0.55 + 0.35 * pulse) * (1.0 if t < 4.6 else 0.45)
            ax.add_patch(matplotlib.patches.Rectangle((cx0 + xx * sc - 14 * sc, cy0 - 20 * sc), 28 * sc, 40 * sc,
                                                      fc=(0.2, 0.35, 0.6, 0.16 * a_cav), ec=(0.5, 0.7, 1, 0.55 * a_cav), lw=0.9 * sc, zorder=2))
        # persistent hits
        for e in flat:
            if t >= e["t_hit"]:
                age = t - e["t_hit"]
                nx, ny, nz = e["n"]
                for sgn, col in ((1, col_mu_m), (-1, col_mu_p)):
                    hx, hy = cx0 + sgn * ring * nx, cy0 + sgn * ring * ny
                    base = 0.42 if age > 0.8 else 1.0 - 0.58 * age / 0.8
                    glow(hx, hy, col, (11 + 3.5 * sgn * nz) * sc, base, z=5)
        # bunches
        for k, cr in enumerate(crossings):
            t_c = cr["t_c"]
            xc = xc_of(t, cr, v_fast / sc) * sc
            if abs(xc) > 760 * sc:
                continue
            gone_e = {ev["ie"] for ev, fe in zip(cr["events"], [e for e in flat if e["k"] == k]) if t >= fe["t"]}
            gone_p = {ev["ip"] for ev, fe in zip(cr["events"], [e for e in flat if e["k"] == k]) if t >= fe["t"]}
            idx_e = [i for i in range(N_BUNCH) if i not in gone_e]
            idx_p = [i for i in range(N_BUNCH) if i not in gone_p]
            xe = cx0 + xc + cr["dx_e"][idx_e] * sc
            ye = cy0 + cr["dy_e"][idx_e] * sc
            xp = cx0 - xc + cr["dx_p"][idx_p] * sc
            yp = cy0 + cr["dy_p"][idx_p] * sc
            keep_e = np.abs(xe - cx0) < 430 * sc
            keep_p = np.abs(xp - cx0) < 430 * sc
            xe, ye, xp, yp = xe[keep_e], ye[keep_e], xp[keep_p], yp[keep_p]
            if len(xe) == 0 and len(xp) == 0:
                continue
            speed = (v_fast if not cr["first"] else (400 * 1.8 / t_c * (max(t, 0) / t_c) ** 0.8 * sc if t <= t_c else 400 * 1.8 / t_c * sc))
            streak = min(speed * 0.05, 36 * sc)
            segs_e = [[(a, b), (a - streak, b)] for a, b in zip(xe, ye)]
            segs_p = [[(a, b), (a + streak, b)] for a, b in zip(xp, yp)]
            ax.add_collection(LineCollection(segs_e, colors=[mcolors.to_rgba(col_e, 0.30)] * len(segs_e), linewidths=1.6 * sc, zorder=4))
            ax.add_collection(LineCollection(segs_p, colors=[mcolors.to_rgba(col_p, 0.30)] * len(segs_p), linewidths=1.6 * sc, zorder=4))
            glow(xe, ye, col_e, 8.5 * sc, 0.95)
            glow(xp, yp, col_p, 8.5 * sc, 0.95)
        # events: flash and muons
        for e in flat:
            dt = t - e["t"]
            if dt < 0 or dt > 2.5:
                continue
            ox, oy = cx0 + e["xm"] * sc, cy0 + e["ym"] * sc
            slow = 2.6 if e["first"] else 1.0
            if dt < 0.5 * slow:
                f = 1 - dt / (0.5 * slow)
                glow(ox, oy, "#fff1c8", (40 + 90 * (1 - f)) * sc, 0.85 * f, z=8)
                ax.add_patch(matplotlib.patches.Circle((ox, oy), (8 + 85 * (1 - f)) * sc, fill=False, ec=mcolors.to_rgba("#fff1c8", 0.6 * f),
                                                       lw=1.4 * sc, zorder=8))
            nx, ny, nz = e["n"]
            travel = min(dt * e["speed"], ring + 4 * sc)
            if dt * e["speed"] < ring + 40 * sc:
                for sgn, col in ((1, col_mu_m), (-1, col_mu_p)):
                    px, py = ox + sgn * nx * travel, oy + sgn * ny * travel
                    ax.plot([ox, px], [oy, py], color=mcolors.to_rgba(col, 0.55), lw=3.0 * sc, zorder=7, solid_capstyle="round")
                    glow(px, py, col, (13 + 3 * sgn * nz) * sc, 1.0, z=9)
        # stage chips and captions
        stage = 1 if t < 3.8 else 2 if t < 5.4 else 3 if t < 9.6 else 0
        chips = (("1  preparation / подготовка", 1), ("2  interaction / взаимодействие", 2), ("3  measurement / измерение", 3))
        for i, (txt, st) in enumerate(chips):
            on = (stage == st) or (stage == 0)
            a_c = 1.0 if stage == st else (0.55 if stage == 0 else 0.28)
            ax.text(0.03 * W, H * (0.925 - 0.052 * i), txt, color=(0.62, 0.86, 1.0, a_c) if on else (0.5, 0.58, 0.72, a_c),
                    fontsize=(13.5 if stage == st else 11.5) * sc, zorder=12)
        k_cross = sum(1 for cr in crossings[1:] if t >= cr["t_c"])
        n_ev = int(np.sum(t_hits <= t))
        if t < 3.8:
            cap = ("beams are accelerated and focused", "пучки ускоряют и фокусируют")
        elif t < 5.4:
            cap = ("the particles meet in a localized region: e⁺e⁻ → μ⁺μ⁻", "частицы встречаются в малой области: e⁺e⁻ → μ⁺μ⁻")
        elif t < 9.6:
            cap = ("the detector records where the muons go", "детектор регистрирует, куда летят мюоны")
        else:
            cap = (f"bunch crossings repeated: most particles pass through, rare events", "пучки встречаются снова и снова: почти все частицы пролетают насквозь")
        ax.add_patch(matplotlib.patches.Rectangle((0, 0), W, 0.115 * H, color=(0.01, 0.02, 0.05, 0.6), zorder=11, lw=0))
        fig.texts.clear()
        fig.text(0.03, 0.058, cap[0], color=(0.88, 0.93, 1.0, 0.93), fontsize=13 * sc)
        fig.text(0.03, 0.022, cap[1], color=(0.6, 0.68, 0.8, 0.9), fontsize=11 * sc)
        fig.text(0.03, 0.005 + 0.115, "", fontsize=1)
        ax.text(cx0 - 420 * sc, cy0 + 24 * sc, "e⁻ →", color=col_e, fontsize=13 * sc, alpha=0.9, zorder=12)
        ax.text(cx0 + 340 * sc, cy0 + 24 * sc, "← e⁺", color=col_p, fontsize=13 * sc, alpha=0.9, zorder=12)
        ax.text(cx0 - 60 * sc, H * 0.925, "μ⁻ ●", color=col_mu_m, fontsize=13 * sc, alpha=0.85, zorder=12)
        ax.text(cx0 + 20 * sc, H * 0.925, "μ⁺ ●", color=col_mu_p, fontsize=13 * sc, alpha=0.85, zorder=12)
        # histogram panel
        ph.clear()
        ph.set_facecolor("none")
        p_a = min(1.0, max(0.0, (t - 5.2) / 1.2))
        if p_a > 0.01:
            counts, _ = np.histogram(cs[t_hits <= t], bins=edges)
            ph.bar(centers, counts, width=(edges[1] - edges[0]) * 0.86, color=(0.5, 0.9, 0.75, 0.85 * p_a))
            n_all = max(1, len(cs))
            curve = np.linspace(-1, 1, 100)
            ph.plot(curve, angular_density(curve) * (edges[1] - edges[0]) * max(n_ev, 1), color=(1.0, 0.82, 0.4, p_a), lw=2.2 * sc)
            ph.set_xlim(-1, 1)
            ph.set_ylim(0, max(4, 1.15 * angular_density(1.0) * (edges[1] - edges[0]) * max(n_ev, 1) * 1.05))
            for name, sp in ph.spines.items():
                sp.set_visible(name in ("left", "bottom"))
                sp.set_color((0.75, 0.82, 0.92, 0.5 * p_a))
            ph.set_xticks([-1, 0, 1])
            ph.set_xticklabels(["−1", "0", "1"], color=(0.82, 0.9, 1, 0.85 * p_a), fontsize=10 * sc)
            ph.set_yticks([])
            ph.set_xlabel("cos θ  (angle to the e⁻ beam / угол к пучку e⁻)", color=(0.82, 0.9, 1, 0.85 * p_a), fontsize=9.5 * sc, labelpad=2)
            fig.text(0.725, 0.79, "events / события", color=(0.82, 0.9, 1, p_a), fontsize=13 * sc)
            fig.text(0.725, 0.69, f"N = {n_ev}", color=(0.55, 1.0, 0.8, p_a), fontsize=30 * sc, family="monospace")
            fig.text(0.725, 0.645, f"bunch crossings / встреч пучков: {k_cross + (1 if t >= crossings[0]['t_c'] else 0)}",
                     color=(0.7, 0.78, 0.9, 0.9 * p_a), fontsize=9.5 * sc)
            fig.text(0.725, 0.19, "—  dN/dcosθ ∝ 1 + cos²θ   (QED)", color=(1.0, 0.82, 0.4, 0.95 * p_a), fontsize=10 * sc)
            fig.text(0.725, 0.145, "σ = N / (flux × targets)   cross section / сечение", color=(0.7, 0.78, 0.9, 0.9 * p_a), fontsize=9.5 * sc)
        else:
            ph.axis("off")
        fig.text(0.725, 0.935 - 0.0, "schematic: event rate exaggerated / схема: частота событий завышена", color=(0.5, 0.58, 0.72, 0.8),
                 fontsize=8 * sc)
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
    ap.add_argument("--seconds", type=float, default=28.0)
    ap.add_argument("--out", type=Path, default=HERE / "media" / "scattering_experiment.mp4")
    args = ap.parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    if args.snapshot is not None:
        render(args.out.with_suffix(".png"), (1280, 720), 30, args.seconds, snap=args.snapshot)
    elif args.preview:
        render(args.out.with_name("scattering_experiment_preview.mp4"), (640, 360), 15, 10.0)
    else:
        render(args.out, (1280, 720), 30, args.seconds)


if __name__ == "__main__":
    main()
