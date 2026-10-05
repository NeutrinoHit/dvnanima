"""Running charge: virtual electron-positron pairs screen an electron.

The vacuum is full of short-lived e+e- pairs.  Far from any charge their orientation
is random.  Near an electron the pairs are polarized: each positron is pulled in and each
electron pushed out, so the pairs line up radially and screen the charge.  A test charge
that flies in measures a larger charge the closer it gets.

Model (lengths in the reduced Compton wavelength lambda_C = hbar / m_e c):

* pairs are created uniformly in the plane at random times, live a random time of order
  one unit, and are shown as short dipoles;
* the orientation of a new pair relative to the radial direction is drawn from a von Mises
  distribution with concentration kappa(r) = s * K0 / (1 + (r / R0)^2).  s is the strength
  of the charge (switched on smoothly), so the alignment fades with distance;
* the charge seen at distance r follows the one-loop form

      Q(r) / e = 1 / (1 - b L(r)),    L(r) = (1/2) ln(1 + 1/r^2),

  which tends to 1 for r >> lambda_C and to 1 / (1 - b ln(1/r)) for r << lambda_C.  For the
  electron b = 2 alpha / (3 pi) = 0.0015.  The film uses b = 0.25 so that the effect is
  visible; the real change is about 1 % at r = 1e-3 lambda_C.

Usage:
    python running_charge.py                 # film -> media/running_charge.mp4
    python running_charge.py --preview       # 5 s low-resolution film
    python running_charge.py --snapshot 18   # one PNG at film time 18 s
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

B_REAL = 2 * (1 / 137.036) / (3 * math.pi)
B_FILM = 0.25
K0 = 16.0
R0 = 1.2
R_SCENE = 3.3          # half-height of the scene in lambda_C
PAIR_DENSITY = 0.85     # live pairs per unit area
MEAN_LIFE = 1.0


def charge_ratio(r: np.ndarray | float, b: float = B_FILM) -> np.ndarray | float:
    """Q(r)/e for the one-loop running."""
    r = np.asarray(r, dtype=float)
    return 1.0 / (1.0 - b * 0.5 * np.log1p(1.0 / r ** 2))


def smooth(x: float, a: float, b: float) -> float:
    t = min(max((x - a) / (b - a), 0.0), 1.0)
    return t * t * (3 - 2 * t)


# ------------------------------------------------------------------ the pairs

def make_pairs(seed: int, total: float) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    area = math.pi * (R_SCENE * 1.15) ** 2
    rate = PAIR_DENSITY * area / MEAN_LIFE
    n = int(rate * (total + 3.0))
    t0 = np.sort(rng.uniform(-2.0, total, n))
    life = np.clip(rng.gamma(3.0, MEAN_LIFE / 3.0, n), 0.45, 2.6)
    r = np.sqrt(rng.uniform(0.2 ** 2, (R_SCENE * 1.15) ** 2, n))
    near = rng.uniform(size=n) < 0.22                       # extra pairs close to the charge (polarization cloud)
    r = np.where(near, np.sqrt(rng.uniform(0.22 ** 2, 1.5 ** 2, n)), r)
    phi = rng.uniform(0, 2 * math.pi, n)
    d0 = rng.uniform(0.16, 0.42, n)
    noise = rng.normal(size=n)          # standard normal, turned into an angle with kappa later
    unif = rng.uniform(-math.pi, math.pi, n)
    vm_seed = rng.integers(0, 2 ** 31, n)
    return dict(t0=t0, life=life, r=r, phi=phi, d0=d0, noise=noise, unif=unif, vm_seed=vm_seed)


def pair_ends(pairs: dict[str, np.ndarray], t: float, s_of) -> tuple[np.ndarray, ...]:
    """Positions of the positron and electron ends and an opacity for every living pair."""
    t0, life = pairs["t0"], pairs["life"]
    alive = (t >= t0) & (t <= t0 + life)
    idx = np.where(alive)[0]
    age = (t - t0[idx]) / life[idx]
    r = pairs["r"][idx]
    phi = pairs["phi"][idx]
    s = np.array([s_of(t0[i]) for i in idx]) if len(idx) else np.zeros(0)
    kappa = s * K0 / (1.0 + (r / R0) ** 2)
    # orientation relative to the radial direction: von Mises, drawn reproducibly per pair
    psi = np.array([np.random.default_rng(int(pairs["vm_seed"][i])).vonmises(0.0, k) if k > 1e-6
                    else pairs["unif"][i] for i, k in zip(idx, kappa)]) if len(idx) else np.zeros(0)
    # the pair stretches in the field of the charge
    stretch = 1.0 + 1.4 * s_of(t) / (1.0 + (r / 0.9) ** 2)
    d = pairs["d0"][idx] * stretch * (0.7 + 0.3 * np.sin(math.pi * np.clip(age, 0, 1)))
    cx, cy = r * np.cos(phi), r * np.sin(phi)
    ux, uy = np.cos(phi + psi), np.sin(phi + psi)           # from positron to electron
    pos = np.stack([cx - 0.5 * d * ux, cy - 0.5 * d * uy], axis=1)
    ele = np.stack([cx + 0.5 * d * ux, cy + 0.5 * d * uy], axis=1)
    fade_in = np.clip(age * life[idx] / 0.28, 0, 1)
    fade_out = np.clip((1 - age) * life[idx] / 0.28, 0, 1)
    return pos, ele, np.minimum(fade_in, fade_out), age, life[idx]


# ---------------------------------------------------------------------- film

def timeline(total: float) -> dict[str, float]:
    k = total / 26.0
    return dict(charge_on=(5.0 * k, 8.5 * k), panel=(8.5 * k, 10.0 * k), probe=(10.0 * k, 23.0 * k))


def probe_radius(t: float, tl: dict[str, float]) -> float:
    a, b = tl["probe"]
    u = min(max((t - a) / (b - a), 0.0), 1.0)
    e = 0.5 - 0.5 * math.cos(math.pi * u)
    return math.exp(math.log(3.0) + (math.log(0.13) - math.log(3.0)) * e)


def render(out: Path, size: tuple[int, int], fps: int, total: float, snap: float | None = None,
           chrome: bool = True) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection
    from scipy.ndimage import gaussian_filter

    if snap is None and not shutil.which("ffmpeg"):
        sys.exit("ffmpeg is required")
    W, H = size
    dpi = 100
    sc = H / 720.0
    unit = H / (2 * R_SCENE)               # pixels per lambda_C
    cx0, cy0 = 0.37 * W, 0.5 * H
    tl = timeline(total)
    pairs = make_pairs(5, total)
    s_of = lambda t: smooth(t, *tl["charge_on"])

    fig = plt.figure(figsize=(W / dpi, H / dpi), dpi=dpi, facecolor="#04070d")
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, W)
    ax.set_ylim(0, H)
    ax.axis("off")
    pa = fig.add_axes([0.715, 0.25, 0.255, 0.38], facecolor="none")

    # background: vignette and slowly drifting vacuum texture (low-resolution, upsampled)
    yy, xx = np.mgrid[0:H // 8, 0:W // 8]
    rr = np.hypot(xx * 8 - cx0, yy * 8 - cy0) / (0.55 * W)
    vignette = np.clip(1.0 - 0.75 * rr ** 1.6, 0.0, 1.0)
    rng = np.random.default_rng(2)
    tex = [gaussian_filter(rng.normal(size=(H // 8, W // 8)), 6) for _ in range(3)]
    tex = [(t_ - t_.min()) / (t_.max() - t_.min()) for t_ in tex]
    halo_y, halo_x = np.mgrid[0:H // 8, 0:W // 8]
    halo_r = np.hypot((halo_x * 8 - cx0) / unit, (halo_y * 8 - cy0) / unit)
    cloud = np.exp(-((halo_r - 0.55) / 0.45) ** 2)          # polarization cloud around the charge

    def glow(points: np.ndarray, color: str, diameter: float, opacity: np.ndarray, z: int) -> None:
        if len(points) == 0:
            return
        X = cx0 + unit * points[:, 0]
        Y = cy0 + unit * points[:, 1]
        for scale, alpha in ((4.2, 0.06), (2.4, 0.14), (1.35, 0.35), (0.62, 1.0)):
            s = (diameter * scale * 72 / dpi) ** 2
            ax.scatter(X, Y, s=s, linewidths=0, zorder=z,
                       edgecolors="none", facecolors=[matplotlib.colors.to_rgba(color, alpha * o) for o in opacity])

    frames = int(round(total * fps))
    ids = [int(round(snap * fps))] if snap is not None else range(frames)
    writer = None
    if snap is None:
        writer = subprocess.Popen(
            ["ffmpeg", "-y", "-v", "error", "-f", "rawvideo", "-pix_fmt", "rgba", "-s", f"{W}x{H}",
             "-r", str(fps), "-i", "-", "-c:v", "libx264", "-preset", "slow", "-crf", "19",
             "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(out)], stdin=subprocess.PIPE)

    rs_curve = np.geomspace(0.1, 3.3, 200)
    for k in ids:
        t = k / fps
        s = s_of(t)
        ax.clear()
        ax.set_xlim(0, W)
        ax.set_ylim(0, H)
        ax.axis("off")
        # background
        phase = 0.5 + 0.5 * math.sin(2 * math.pi * t / 11.0)
        mix = (1 - phase) * tex[0] + phase * tex[1]
        mix = 0.6 * mix + 0.4 * tex[2]
        img = np.zeros(mix.shape + (3,))
        img[..., 0] = 0.015 + 0.020 * mix
        img[..., 1] = 0.030 + 0.060 * mix
        img[..., 2] = 0.060 + 0.130 * mix
        img = img * vignette[..., None]
        img[..., 0] += 0.10 * s * cloud
        img[..., 1] += 0.045 * s * cloud
        ax.imshow(np.clip(img, 0, 1), extent=(0, W, 0, H), origin="lower", interpolation="bicubic", zorder=0, aspect="auto")
        # lambda_C circle
        ang = np.linspace(0, 2 * math.pi, 240)
        ax.plot(cx0 + unit * np.cos(ang), cy0 + unit * np.sin(ang), color="#8fb4d8", lw=0.7 * sc, alpha=0.22, ls=(0, (2, 4)), zorder=1)
        ax.text(cx0 + unit * 0.71, cy0 + unit * 0.78, "λ$_C$", color="#8fb4d8", alpha=0.55, fontsize=11 * sc, zorder=1)
        # field lines whose strength follows Q(r)
        if s > 0.01:
            segs, widths, cols = [], [], []
            n_lines = 28
            rr_ = np.geomspace(0.14, R_SCENE * 1.1, 70)
            for i in range(n_lines):
                a = 2 * math.pi * i / n_lines + 0.05 * math.sin(0.3 * t + i)
                pts = np.stack([cx0 + unit * rr_ * math.cos(a), cy0 + unit * rr_ * math.sin(a)], axis=1)
                q = charge_ratio(rr_)
                for j in range(len(rr_) - 1):
                    segs.append(pts[j:j + 2])
                    widths.append((0.5 + 2.1 * (q[j] - 1.0) * 1.1 + 0.6) * sc)
                    alpha = s * (0.10 + 0.55 * min(1.0, (q[j] - 1.0) / 1.2)) / (1 + 0.5 * rr_[j])
                    cols.append((0.45, 0.80, 1.0, min(alpha, 0.8)))
            ax.add_collection(LineCollection(segs, linewidths=widths, colors=cols, zorder=2, capstyle="round"))
        # pairs
        pos, ele, op, age, life = pair_ends(pairs, t, s_of)
        if len(pos):
            lines = [[(cx0 + unit * p[0], cy0 + unit * p[1]), (cx0 + unit * e[0], cy0 + unit * e[1])] for p, e in zip(pos, ele)]
            ax.add_collection(LineCollection(lines, colors=[(0.8, 0.85, 1.0, 0.45 * o) for o in op],
                                             linewidths=1.1 * sc, zorder=3))
            glow(ele, "#4cc3ff", 0.075 * unit * 1.6, op, 4)
            glow(pos, "#ff8f3a", 0.075 * unit * 1.6, op, 4)
            # creation / annihilation flashes
            born = np.clip(1.0 - age * life / 0.22, 0, 1)
            died = np.clip(1.0 - (1 - age) * life / 0.18, 0, 1)
            mids = 0.5 * (pos + ele)
            for flash, color in ((born, "#bfe6ff"), (died, "#ffffff")):
                m = flash > 0.02
                if m.any():
                    X = cx0 + unit * mids[m, 0]
                    Y = cy0 + unit * mids[m, 1]
                    rad = (0.06 + 0.22 * (1 - flash[m])) * unit if color != "#ffffff" else (0.04 + 0.12 * (1 - flash[m])) * unit
                    ax.scatter(X, Y, s=(2 * rad * 72 / dpi) ** 2, facecolors="none",
                               edgecolors=[matplotlib.colors.to_rgba(color, 0.40 * f) for f in flash[m]],
                               linewidths=1.2 * sc, zorder=5)
        # the electron in the centre
        ax.scatter([cx0], [cy0], s=(0.9 * unit * 72 / dpi) ** 2, c="#2a7fff", alpha=0.05 * s, linewidths=0, zorder=6)
        ax.scatter([cx0], [cy0], s=(0.5 * unit * 72 / dpi) ** 2, c="#38a8ff", alpha=0.18 * s, linewidths=0, zorder=6)
        ax.scatter([cx0], [cy0], s=(0.27 * unit * 72 / dpi) ** 2, c="#4cc3ff", alpha=0.9 * s, linewidths=0, zorder=7)
        ax.plot([cx0 - 0.07 * unit, cx0 + 0.07 * unit], [cy0, cy0], color="#05203a", lw=2.4 * sc, alpha=s, zorder=8, solid_capstyle="round")
        # probe
        pa.clear()
        pa.set_facecolor("none")
        p_alpha = smooth(t, *tl["panel"]) if chrome else 0.0
        r_p = probe_radius(t, tl)
        if t >= tl["probe"][0] - 0.8:
            pa_ = smooth(t, tl["probe"][0] - 0.8, tl["probe"][0] + 0.2)
            ang_p = math.radians(-18)
            ts = np.linspace(0, 1, 60)
            path_r = np.geomspace(3.0, max(r_p, 0.13), 60)
            ax.plot(cx0 + unit * path_r * math.cos(ang_p), cy0 + unit * path_r * math.sin(ang_p),
                    color="#ffffff", lw=0.9 * sc, alpha=0.18 * pa_, zorder=9)
            px, py = r_p * math.cos(ang_p), r_p * math.sin(ang_p)
            glow(np.array([[px, py]]), "#c6ffd2", 0.075 * unit * 1.9, np.array([pa_]), 10)
            if chrome:
                ax.text(cx0 + unit * (px + 0.20), cy0 + unit * (py - 0.30), "test charge / пробный заряд", color="#d9ffe2",
                        alpha=0.8 * pa_, fontsize=10 * sc, zorder=10)
        # panel with the measured charge
        if p_alpha > 0.01:
            pa.set_xscale("log")
            pa.set_xlim(3.3, 0.1)
            pa.set_ylim(0.95, 2.35)
            for sp in pa.spines.values():
                sp.set_visible(False)
            pa.spines["bottom"].set_visible(True)
            pa.spines["bottom"].set_color((0.7, 0.8, 0.9, 0.5 * p_alpha))
            pa.spines["left"].set_visible(True)
            pa.spines["left"].set_color((0.7, 0.8, 0.9, 0.5 * p_alpha))
            q_all = charge_ratio(rs_curve)
            pa.plot(rs_curve, q_all, color=(1, 1, 1, 0.12 * p_alpha), lw=1.4 * sc)
            shown = rs_curve >= r_p
            if t >= tl["probe"][0]:
                pa.plot(rs_curve[shown], q_all[shown], color=(1.0, 0.62, 0.25, p_alpha), lw=3.0 * sc, solid_capstyle="round")
            pa.axhline(1.0, color=(0.7, 0.8, 0.9, 0.35 * p_alpha), lw=1.0 * sc, ls=(0, (3, 4)))
            pa.set_xticks([0.1, 0.3, 1.0, 3.0])
            pa.set_xticklabels(["0.1", "0.3", "1", "3"], color=(0.8, 0.88, 1.0, 0.8 * p_alpha), fontsize=10 * sc)
            pa.set_yticks([1.0, 1.5, 2.0])
            pa.set_yticklabels(["1", "1.5", "2"], color=(0.8, 0.88, 1.0, 0.8 * p_alpha), fontsize=10 * sc)
            pa.minorticks_off()
            pa.tick_params(colors=(0.7, 0.8, 0.9, 0.5 * p_alpha), length=3)
            pa.set_xlabel("r / λ$_C$", color=(0.8, 0.88, 1.0, 0.85 * p_alpha), fontsize=11 * sc, labelpad=2)
            pa.text(1.0, 1.0, "", transform=pa.transAxes)
            if t >= tl["probe"][0]:
                q_now = float(charge_ratio(r_p))
                pa.scatter([r_p], [q_now], s=(14 * sc) ** 2 * 0.6, c="#ffe0b8", zorder=5, linewidths=0)
                pa.scatter([r_p], [q_now], s=(30 * sc) ** 2 * 0.6, c="#ff9f40", alpha=0.25, zorder=4, linewidths=0)
            else:
                q_now = 1.0
            fig.texts.clear()
            fig.text(0.715, 0.86, "Q(r) / e", color=(0.8, 0.88, 1.0, p_alpha), fontsize=16 * sc)
            fig.text(0.715, 0.735, f"{q_now:.2f}", color=(1.0, 0.72, 0.38, p_alpha), fontsize=40 * sc, family="monospace")
            fig.text(0.715, 0.685, "charge seen at distance r / заряд на расстоянии r", color=(0.62, 0.7, 0.82, 0.9 * p_alpha), fontsize=9.5 * sc)
            fig.text(0.715, 0.135, "schematic scale: effect exaggerated ~150×\n(real electron: +1 % at r ≈ 10⁻³ λ$_C$)\nусловный масштаб: эффект увеличен ~150×",
                     color=(0.55, 0.62, 0.75, 0.85 * p_alpha), fontsize=8.5 * sc, linespacing=1.45, va="top")
        else:
            pa.axis("off")
            fig.texts.clear()
        if chrome:
            # captions on a soft dark strip
            strip = np.linspace(0.0, 0.62, 30)[::-1]
            ax.imshow(np.tile(strip[:, None], (1, 2)), extent=(0, W, 0, 0.14 * H), origin="upper", cmap="gray_r", alpha=None, zorder=11, aspect="auto", interpolation="bilinear", vmin=0, vmax=1) if False else None
            ax.add_patch(matplotlib.patches.Rectangle((0, 0), W, 0.12 * H, color=(0.01, 0.02, 0.05, 0.62), zorder=11, lw=0))
            if t < tl["charge_on"][0]:
                cap = ("vacuum: virtual e⁺e⁻ pairs appear and vanish, orientation random", "вакуум: виртуальные пары e⁺e⁻ рождаются и исчезают, ориентация случайна")
            elif t < tl["probe"][0]:
                cap = ("near the electron the pairs line up: positrons in, electrons out, the charge is screened",
                       "вблизи электрона пары выстраиваются: позитроны внутрь, электроны наружу, заряд экранируется")
            else:
                cap = ("the closer, the less screening: a test charge sees a larger charge", "чем ближе, тем слабее экранировка: пробный заряд видит больший заряд")
            fig.text(0.03, 0.058, cap[0], color=(0.86, 0.92, 1.0, 0.92), fontsize=13 * sc)
            fig.text(0.03, 0.022, cap[1], color=(0.58, 0.67, 0.8, 0.9), fontsize=11 * sc)
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
    ap.add_argument("--still", type=float, default=None,
                    help="clean 2560x1440 still without captions and graph at this film time (for book previews)")
    ap.add_argument("--seconds", type=float, default=26.0)
    ap.add_argument("--out", type=Path, default=HERE / "media" / "running_charge.mp4")
    args = ap.parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    if args.still is not None:
        from PIL import Image
        target = args.out.with_name(args.out.stem + "_still.png")
        render(target, (2560, 1440), 30, args.seconds, snap=args.still, chrome=False)
        img = Image.open(target)
        img.crop((0, 0, int(0.70 * img.width), img.height)).save(target)
        print(f"cropped {target}")
    elif args.snapshot is not None:
        render(args.out.with_suffix(".png"), (1280, 720), 30, args.seconds, snap=args.snapshot)
    elif args.preview:
        render(args.out.with_name("running_charge_preview.mp4"), (640, 360), 15, 6.0)
    else:
        render(args.out, (1280, 720), 30, args.seconds)


if __name__ == "__main__":
    main()
