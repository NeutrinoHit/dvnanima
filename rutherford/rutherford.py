"""Rutherford scattering of alpha particles by a thin foil (Geiger-Marsden experiment).

A beam of alpha particles crosses a few atomic layers.  Every alpha particle is
a classical point charge moving in the Coulomb field of fixed point nuclei
(3D, trajectories integrated numerically).  Nothing is imposed: the scattering
angles follow from the integration and are compared with the Rutherford formula.

Units: alpha mass m = 1, initial speed v0 = 1, so E = 1/2.  The potential energy is
U(r) = K exp(-r/a) / r with K = r_min * E, where r_min is the head-on distance of
closest approach and a is the atomic screening length (the atoms are neutral).  With impact parameter b the deflection angle is
    tan(theta/2) = r_min / (2 b)      (Rutherford).
For b much smaller than a (angles above about r_min/a) single scattering gives, per particle and per unit area density n of nuclei,
    dN/dtheta = n * (pi * r_min^2 / 4) * cos(theta/2) / sin^3(theta/2).

The picture is schematic: nuclei are drawn much larger than they are and the foil
is only a few layers thick, so that rare large-angle events appear within a short
film.  In the real experiment about one alpha particle in 8000 turns back.

Usage:
    python rutherford.py --simulate            # trajectories -> media/rutherford_tracks.npz
    python rutherford.py                       # film -> media/rutherford.mp4
    python rutherford.py --preview             # 4 s low-resolution film
"""

from __future__ import annotations

import argparse
import math
import shutil
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
MEDIA = HERE / "media"

# ---------------------------------------------------------------- parameters
R_MIN = 1.0                    # head-on distance of closest approach (length unit)
LAYERS = 4
LAYER_GAP = 14.0
SPACING = 50.0                 # mean spacing of nuclei inside a layer
HALF = 125.0                   # lateral half-size of the foil (x-z/y plane of the layer)
BEAM_HALF = 90.0               # beam cross section: |y|, |z| <= BEAM_HALF
X_START = -150.0
R_STOP = 260.0                 # trajectories stop at this distance from the foil centre
N_ALPHA = 6000
SCREEN = 25.0                  # atomic screening length: electrons make the atoms neutral


def make_foil(rng: np.random.Generator, layers: int = LAYERS) -> np.ndarray:
    """Nuclei: a jittered square lattice in each of the LAYERS layers."""
    n = int(round(2 * HALF / SPACING))
    grid = (np.arange(n) + 0.5) * SPACING - HALF
    pts = []
    for k in range(layers):
        x = (k - (layers - 1) / 2) * LAYER_GAP
        shift = rng.uniform(-SPACING / 2, SPACING / 2, size=2)
        yy, zz = np.meshgrid(grid + shift[0], grid + shift[1])
        jitter = rng.uniform(-0.2 * SPACING, 0.2 * SPACING, size=(yy.size, 2))
        pts.append(np.column_stack([np.full(yy.size, x), yy.ravel() + jitter[:, 0],
                                    zz.ravel() + jitter[:, 1]]))
    return np.vstack(pts)


def acceleration(r: np.ndarray, nuclei: np.ndarray, k: float) -> np.ndarray:
    """Screened Coulomb force, U = K exp(-r/SCREEN) / r.

    The atoms are neutral: the electrons screen the nucleus beyond about SCREEN.
    Without screening the field of the whole foil would reflect the beam."""
    d = r - nuclei
    dist = np.sqrt(np.einsum("ij,ij->i", d, d))
    mag = k * np.exp(-dist / SCREEN) * (1.0 / dist ** 2 + 1.0 / (SCREEN * dist))
    return np.sum(d * (mag / dist)[:, None], axis=0)


def one_track(args: tuple[np.ndarray, np.ndarray, float, float]) -> tuple[np.ndarray, np.ndarray, float]:
    """Integrate one alpha particle; return sampled path, times and the final velocity angle."""
    nuclei, start, dt_out, k = args
    from scipy.integrate import solve_ivp

    def rhs(_t, s):
        return np.concatenate([s[3:], acceleration(s[:3], nuclei, k)])

    def leave(_t, s):
        return np.linalg.norm(s[:3]) - R_STOP
    leave.terminal = True
    leave.direction = 1

    s0 = np.concatenate([start, [1.0, 0.0, 0.0]])
    sol = solve_ivp(rhs, (0.0, 4 * (R_STOP - X_START)), s0, method="DOP853",
                    rtol=1e-9, atol=1e-9, events=leave, dense_output=True)
    t_end = sol.t[-1]
    t = np.arange(0.0, t_end, dt_out)
    path = sol.sol(t)[:3].T
    v = sol.y[3:, -1]
    theta = math.degrees(math.acos(np.clip(v[0] / np.linalg.norm(v), -1, 1)))
    return path.astype(np.float32), t.astype(np.float32), theta


def simulate(seed: int, n_alpha: int, out: Path, r_min: float = R_MIN, layers: int = LAYERS) -> None:
    rng = np.random.default_rng(seed)
    nuclei = make_foil(rng, layers)
    starts = np.column_stack([np.full(n_alpha, X_START),
                              rng.uniform(-BEAM_HALF, BEAM_HALF, n_alpha),
                              rng.uniform(-BEAM_HALF, BEAM_HALF, n_alpha)])
    dt_out = 1.0
    with ProcessPoolExecutor() as pool:
        results = list(pool.map(one_track, [(nuclei, s, dt_out, r_min * 0.5) for s in starts], chunksize=8))
    paths = [r[0] for r in results]
    lengths = np.array([len(p) for p in paths])
    flat = np.concatenate(paths)
    angles = np.array([r[2] for r in results])
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, nuclei=nuclei, starts=starts, flat=flat, lengths=lengths,
                        angles=angles, dt=dt_out, r_min=r_min, layers=layers,
                        beam_half=BEAM_HALF, spacing=SPACING)
    print(f"{n_alpha} tracks, {np.sum(angles > 90)} turned by more than 90 degrees -> {out}")


def rutherford_curve(theta_deg: np.ndarray, n_alpha: int, r_min: float, layers: int) -> np.ndarray:
    """Expected number of particles per degree, single scattering."""
    th = np.radians(theta_deg)
    density = layers / SPACING ** 2                      # nuclei per unit area of the beam
    per_rad = n_alpha * density * (math.pi * r_min ** 2 / 4) * np.cos(th / 2) / np.sin(th / 2) ** 3
    return per_rad * math.pi / 180.0


# ------------------------------------------------------------------- the film

def render(data_path: Path, out: Path, size: tuple[int, int], fps: int, seconds: float | None,
           snap: float | None = None) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection

    if not shutil.which("ffmpeg"):
        sys.exit("ffmpeg is required")
    d = np.load(data_path)
    nuclei, flat, lengths, angles = d["nuclei"], d["flat"], d["lengths"], d["angles"]
    n = len(lengths)
    offsets = np.concatenate([[0], np.cumsum(lengths)])
    dt = float(d["dt"])
    W, H = size
    dpi = 100
    speed = 80.0                        # simulation units per film second
    window = 150.0                      # half-size of the foil view
    launch_span = 17.0                  # film seconds over which alphas are fired
    rng = np.random.default_rng(1)
    t0 = np.sort(rng.uniform(0.0, launch_span, n))
    # film time at which each alpha leaves the foil view (for the histogram)
    exit_t = np.empty(n)
    for i in range(n):
        p = flat[offsets[i]:offsets[i + 1]]
        far = np.where(np.linalg.norm(p, axis=1) > window)[0]
        far = far[p[far, 0] > -window] if len(far) else far
        exit_t[i] = t0[i] + (dt * (far[0] if len(far) else len(p) - 1)) / speed
    total = float(seconds) if seconds else float(exit_t.max() + 3.0)
    frames = int(round(total * fps))
    turned = angles > 90.0
    big = angles > 10.0
    bins = np.linspace(0, 180, 91)
    theory_x = np.linspace(1.0, 179.0, 400)
    theory_y = rutherford_curve(theory_x, n, float(d["r_min"]), int(d["layers"])) * 2.0          # per 2-degree bin

    fig = plt.figure(figsize=(W / dpi, H / dpi), dpi=dpi, facecolor="#0a0d14")
    sc = H / 720.0                                   # font scale
    ax = fig.add_axes([0.02, 0.08, 0.80 * H / W, 0.80])   # square foil view
    axh = fig.add_axes([0.575, 0.20, 0.395, 0.55], facecolor="#10141f")
    gold = "#e6b422"

    def setup_scene() -> None:
        ax.set_facecolor("#05070c")
        ax.set_xlim(-window, window)
        ax.set_ylim(-window, window)
        ax.set_aspect("equal")
        ax.set_xticks([])
        ax.set_yticks([])
        for s in ax.spines.values():
            s.set_color("#2a3142")

    writer = None if snap is not None else subprocess.Popen(
        ["ffmpeg", "-y", "-v", "error", "-f", "rawvideo", "-pix_fmt", "rgba", "-s", f"{W}x{H}",
         "-r", str(fps), "-i", "-", "-c:v", "libx264", "-preset", "slow", "-crf", "20",
         "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(out)], stdin=subprocess.PIPE)

    layer_x = np.unique(np.round(nuclei[:, 0], 6))
    frame_ids = [int(round(snap * fps))] if snap is not None else range(frames)
    for k in frame_ids:
        tf = k / fps
        ax.clear()
        setup_scene()
        # foil: translucent gold slab and nuclei (projection on the x-y plane)
        ax.axvspan(layer_x.min() - 4, layer_x.max() + 4, color=gold, alpha=0.10, lw=0)
        ax.scatter(nuclei[:, 0], nuclei[:, 1], s=34, color=gold, edgecolors="#fff2b0",
                   linewidths=0.4, zorder=3)
        active = np.where((tf >= t0) & (tf < exit_t + 0.4))[0]
        pts, cols, sizes = [], [], []
        segs, segcols = [], []
        for i in active:
            idx = int((tf - t0[i]) * speed / dt)
            idx = min(idx, lengths[i] - 1)
            p = flat[offsets[i]:offsets[i] + idx + 1]
            pos = p[-1]
            color = "#ff5a4d" if turned[i] else ("#7fd1ff" if big[i] else "#c8d3e6")
            pts.append(pos[:2])
            cols.append(color)
            sizes.append(26 if turned[i] else 7)
            if (big[i] or turned[i]) and len(p) > 2:
                tail = p[max(0, idx - 60):, :2]
                segs.append(tail)
                segcols.append(color)
        if segs:
            ax.add_collection(LineCollection(segs, colors=segcols, linewidths=1.0, alpha=0.7, zorder=2))
        if pts:
            pts = np.array(pts)
            ax.scatter(pts[:, 0], pts[:, 1], s=sizes, c=cols, zorder=4, linewidths=0)
        ax.text(-window + 6, window - 14, "α", color="#c8d3e6", fontsize=16 * sc)
        ax.annotate("", xy=(-window + 52, window - 38), xytext=(-window + 8, window - 38),
                    arrowprops=dict(arrowstyle="->", color="#c8d3e6", lw=1.4))
        ax.text(layer_x.max() + 8, -window + 8, "gold foil / золотая фольга", color=gold, fontsize=11 * sc)
        # histogram of the particles that have already left
        done = exit_t <= tf
        axh.clear()
        axh.set_facecolor("#10141f")
        counts, _ = np.histogram(angles[done], bins=bins)
        axh.bar(bins[:-1] + 1.0, counts, width=1.8, color="#7fd1ff", log=True)
        axh.axvspan(90, 180, color="#ff5a4d", alpha=0.10, lw=0)
        sel = theory_x >= 12.0
        axh.plot(theory_x[sel], theory_y[sel], color=gold, lw=1.8, label="Rutherford 1/sin⁴(θ/2)")
        axh.set_xlim(0, 180)
        axh.set_yscale("log")
        axh.set_ylim(0.5, 1500)
        axh.set_xlabel("scattering angle θ, degrees / угол рассеяния θ, градусы", color="#c8d3e6", fontsize=11 * sc)
        axh.set_ylabel("particles per 2° / частиц на 2°", color="#c8d3e6", fontsize=11 * sc)
        axh.tick_params(colors="#c8d3e6", labelsize=10 * sc)
        for s in axh.spines.values():
            s.set_color("#2a3142")
        axh.legend(loc="upper right", frameon=False, labelcolor=gold, fontsize=11 * sc)
        axh.text(92, 0.45, "θ > 90°", color="#ff8a7d", fontsize=11 * sc)
        fig.texts.clear()
        fig.text(0.02, 0.925, "Rutherford scattering / Рассеяние Резерфорда", color="white", fontsize=19 * sc)
        fig.text(0.585, 0.86, f"fired / выпущено: {int(np.sum(t0 <= tf))}", color="#c8d3e6", fontsize=14 * sc)
        fig.text(0.585, 0.805, f"turned back (θ > 90°) / назад: {int(np.sum(turned & done))}", color="#ff8a7d", fontsize=14 * sc)
        fig.text(0.02, 0.025, "schematic: 4 atomic layers, nuclei drawn much larger than they are; "
                 "in a real foil about 1 in 8000 turns back / схема: 4 слоя атомов, ядра увеличены",
                 color="#7f8aa3", fontsize=9 * sc)
        fig.canvas.draw()
        if writer is None:
            fig.savefig(out, dpi=dpi, facecolor=fig.get_facecolor())
        else:
            writer.stdin.write(np.asarray(fig.canvas.buffer_rgba()).tobytes())
    if writer is not None:
        writer.stdin.close()
        writer.wait()
    plt.close(fig)
    print(f"wrote {out}" + (f" ({frames} frames)" if writer is not None else ""))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--simulate", action="store_true", help="only integrate the trajectories")
    ap.add_argument("--seed", type=int, default=11)
    ap.add_argument("--alphas", type=int, default=N_ALPHA)
    ap.add_argument("--r-min", type=float, default=R_MIN)
    ap.add_argument("--layers", type=int, default=LAYERS)
    ap.add_argument("--preview", action="store_true")
    ap.add_argument("--snapshot", type=float, default=None, help="write one PNG at this film time (s)")
    ap.add_argument("--data", type=Path, default=MEDIA / "rutherford_tracks.npz")
    ap.add_argument("--out", type=Path, default=MEDIA / "rutherford.mp4")
    args = ap.parse_args()
    if args.simulate or not args.data.exists():
        simulate(args.seed, args.alphas, args.data, args.r_min, args.layers)
        if args.simulate:
            return
    if args.snapshot is not None:
        render(args.data, args.out.with_suffix(".png"), (1280, 720), 30, None, snap=args.snapshot)
    elif args.preview:
        render(args.data, args.out.with_name("rutherford_preview.mp4"), (640, 360), 15, 4.0)
    else:
        render(args.data, args.out, (1280, 720), 30, None)


if __name__ == "__main__":
    main()
