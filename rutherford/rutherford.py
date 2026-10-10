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

All the numbers are in config.toml and all the words in texts.toml (see ../dvconfig.py for --config / --set).
The tracks are computed once (--simulate) and stored in media/rutherford_tracks.npz; the film only reads them.

Usage:
    python rutherford.py --simulate            # trajectories -> media/rutherford_tracks.npz
    python rutherford.py --lang en             # film -> media/rutherford_en.mp4
    python rutherford.py --lang ru             # film -> media/rutherford_ru.mp4
    python rutherford.py --preview             # 4 s low-resolution film
    python rutherford.py --snapshot 14         # one PNG at film time 14 s
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

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dvconfig import load_config, load_texts  # noqa: E402

HERE = Path(__file__).resolve().parent
MEDIA = HERE / "media"
CFG = load_config(HERE)

# ---------------------------------------------------------------- parameters
R_MIN = CFG.foil.r_min                    # head-on distance of closest approach (length unit)
LAYERS = CFG.foil.layers
LAYER_GAP = CFG.foil.layer_gap
SPACING = CFG.foil.spacing                # mean spacing of nuclei inside a layer
HALF = CFG.foil.half                      # lateral half-size of the foil (x-z/y plane of the layer)
BEAM_HALF = CFG.foil.beam_half            # beam cross section: |y|, |z| <= BEAM_HALF
X_START = CFG.foil.x_start
R_STOP = CFG.foil.r_stop                  # trajectories stop at this distance from the foil centre
N_ALPHA = CFG.foil.n_alpha
SCREEN = CFG.foil.screen                  # atomic screening length: electrons make the atoms neutral

TEXT = {lang: load_texts(HERE, lang) for lang in ("en", "ru")}     # texts.toml


def num(x: float, fmt: str, lang: str) -> str:
    """A number for use inside $...$: decimal comma in Russian."""
    s = format(x, fmt)
    return s.replace(".", "{,}") if lang == "ru" else s


def make_foil(rng: np.random.Generator, layers: int = LAYERS) -> np.ndarray:
    """Nuclei: a jittered square lattice in each of the LAYERS layers."""
    n = int(round(2 * HALF / SPACING))
    grid = (np.arange(n) + 0.5) * SPACING - HALF
    pts = []
    for k in range(layers):
        x = (k - (layers - 1) / 2) * LAYER_GAP
        shift = rng.uniform(-SPACING / 2, SPACING / 2, size=2)
        yy, zz = np.meshgrid(grid + shift[0], grid + shift[1])
        jitter = rng.uniform(-CFG.foil.jitter * SPACING, CFG.foil.jitter * SPACING, size=(yy.size, 2))
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
    SIM = CFG.simulation

    def rhs(_t, s):
        return np.concatenate([s[3:], acceleration(s[:3], nuclei, k)])

    def leave(_t, s):
        return np.linalg.norm(s[:3]) - R_STOP
    leave.terminal = True
    leave.direction = 1

    s0 = np.concatenate([start, [SIM.speed, 0.0, 0.0]])
    sol = solve_ivp(rhs, (0.0, SIM.time_limit_factor * (R_STOP - X_START) / SIM.speed), s0, method=SIM.method,
                    rtol=SIM.rtol, atol=SIM.atol, events=leave, dense_output=True)
    t_end = sol.t[-1]
    t = np.arange(0.0, t_end, dt_out)
    path = sol.sol(t)[:3].T
    v = sol.y[3:, -1]
    theta = math.degrees(math.acos(np.clip(v[0] / np.linalg.norm(v), -1, 1)))
    return path.astype(np.float32), t.astype(np.float32), theta


def simulate(seed: int, n_alpha: int, out: Path, r_min: float = R_MIN, layers: int = LAYERS) -> None:
    SIM = CFG.simulation
    rng = np.random.default_rng(seed)
    nuclei = make_foil(rng, layers)
    starts = np.column_stack([np.full(n_alpha, X_START),
                              rng.uniform(-BEAM_HALF, BEAM_HALF, n_alpha),
                              rng.uniform(-BEAM_HALF, BEAM_HALF, n_alpha)])
    dt_out = SIM.dt_out
    with ProcessPoolExecutor() as pool:
        results = list(pool.map(one_track, [(nuclei, s, dt_out, r_min * SIM.energy) for s in starts], chunksize=SIM.chunksize))
    paths = [r[0] for r in results]
    lengths = np.array([len(p) for p in paths])
    flat = np.concatenate(paths)
    angles = np.array([r[2] for r in results])
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, nuclei=nuclei, starts=starts, flat=flat, lengths=lengths,
                        angles=angles, dt=dt_out, r_min=r_min, layers=layers,
                        beam_half=BEAM_HALF, spacing=SPACING)
    print(f"{n_alpha} tracks, {np.sum(angles > CFG.film.turned_deg)} turned by more than {CFG.film.turned_deg:g} degrees -> {out}")


def rutherford_curve(theta_deg: np.ndarray, n_alpha: int, r_min: float, layers: int) -> np.ndarray:
    """Expected number of particles per degree, single scattering."""
    th = np.radians(theta_deg)
    density = layers / SPACING ** 2                      # nuclei per unit area of the beam
    per_rad = n_alpha * density * (math.pi * r_min ** 2 / 4) * np.cos(th / 2) / np.sin(th / 2) ** 3
    return per_rad * math.pi / 180.0


def smooth(v: float, a: float, b: float) -> float:
    u = min(max((v - a) / (b - a), 0.0), 1.0)
    return u * u * (3 - 2 * u)


# ------------------------------------------------------------------- the film

def render(data_path: Path, out: Path, size: tuple[int, int], fps: int, seconds: float | None,
           lang: str = "en", snap: float | None = None) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection

    if not shutil.which("ffmpeg"):
        sys.exit("ffmpeg is required")
    tx = TEXT[lang]
    V, FM, LY, F, ST = CFG.video, CFG.film, CFG.layout, CFG.fonts, CFG.style
    d = np.load(data_path)
    nuclei, flat, lengths, angles = d["nuclei"], d["flat"], d["lengths"], d["angles"]
    n = len(lengths)
    offsets = np.concatenate([[0], np.cumsum(lengths)])
    dt = float(d["dt"])
    W, H = size
    dpi = V.dpi
    speed = FM.speed                    # simulation units per film second
    window = FM.window                  # half-size of the foil view
    launch_span = FM.launch_span        # film seconds over which alphas are fired
    rng = np.random.default_rng(FM.launch_seed)
    t0 = np.sort(rng.uniform(0.0, launch_span, n))
    # film time at which each alpha leaves the foil view (for the histogram)
    exit_t = np.empty(n)
    for i in range(n):
        p = flat[offsets[i]:offsets[i + 1]]
        far = np.where(np.linalg.norm(p, axis=1) > window)[0]
        far = far[p[far, 0] > -window] if len(far) else far
        exit_t[i] = t0[i] + (dt * (far[0] if len(far) else len(p) - 1)) / speed
    total = float(seconds) if seconds else float(exit_t.max() + FM.end_pad)
    frames = int(round(total * fps))
    turned = angles > FM.turned_deg
    big = angles > FM.big_deg
    bins = np.linspace(0, FM.angle_max, FM.hist_bins + 1)
    bin_width = FM.angle_max / FM.hist_bins
    theory_x = np.linspace(*FM.theory_range, FM.theory_samples)
    theory_y = rutherford_curve(theory_x, n, float(d["r_min"]), int(d["layers"])) * bin_width          # per bin
    words = dict(turned=num(FM.turned_deg, "g", lang), bin=num(bin_width, "g", lang), layers=int(d["layers"]), one_in=FM.real_one_in)

    BG = ST.figure_background
    bg_rgba = np.array([int(BG[1:3], 16), int(BG[3:5], 16), int(BG[5:7], 16), 255], np.float32)
    fig = plt.figure(figsize=(W / dpi, H / dpi), dpi=dpi, facecolor=BG)
    sc = H / V.reference_height                      # font scale
    ax = fig.add_axes([LY.scene_pos[0], LY.scene_pos[1], LY.scene_height * H / W, LY.scene_height])   # square foil view
    axh = fig.add_axes(LY.hist_axes, facecolor=ST.hist_background)
    gold = ST.gold

    def setup_scene() -> None:
        ax.set_facecolor(ST.scene_background)
        ax.set_xlim(-window, window)
        ax.set_ylim(-window, window)
        ax.set_aspect("equal")
        ax.set_xticks([])
        ax.set_yticks([])
        for s in ax.spines.values():
            s.set_color(ST.spine)

    writer = None if snap is not None else subprocess.Popen(
        ["ffmpeg", "-y", "-v", "error", "-f", "rawvideo", "-pix_fmt", "rgba", "-s", f"{W}x{H}",
         "-r", str(fps), "-i", "-", "-c:v", "libx264", "-preset", V.preset, "-crf", str(V.crf),
         "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(out)], stdin=subprocess.PIPE)

    layer_x = np.unique(np.round(nuclei[:, 0], 6))
    frame_ids = [int(round(snap * fps))] if snap is not None else range(frames)
    for k in frame_ids:
        tf = k / fps
        ax.clear()
        setup_scene()
        # foil: translucent gold slab and nuclei (projection on the x-y plane)
        ax.axvspan(layer_x.min() - LY.slab_margin, layer_x.max() + LY.slab_margin, color=gold, alpha=ST.slab_alpha, lw=0)
        ax.scatter(nuclei[:, 0], nuclei[:, 1], s=ST.nucleus_size, color=gold, edgecolors=ST.nucleus_edge,
                   linewidths=ST.nucleus_edge_width, zorder=3)
        active = np.where((tf >= t0) & (tf < exit_t + FM.exit_delay))[0]
        pts, cols, sizes = [], [], []
        segs, segcols = [], []
        for i in active:
            idx = int((tf - t0[i]) * speed / dt)
            idx = min(idx, lengths[i] - 1)
            p = flat[offsets[i]:offsets[i] + idx + 1]
            pos = p[-1]
            color = ST.turned_colour if turned[i] else (ST.big_colour if big[i] else ST.alpha_colour)
            pts.append(pos[:2])
            cols.append(color)
            sizes.append(ST.point_size_turned if turned[i] else ST.point_size)
            if (big[i] or turned[i]) and len(p) > 2:
                tail = p[max(0, idx - FM.tail):, :2]
                segs.append(tail)
                segcols.append(color)
        if segs:
            ax.add_collection(LineCollection(segs, colors=segcols, linewidths=ST.tail_width, alpha=ST.tail_alpha, zorder=2))
        if pts:
            pts = np.array(pts)
            ax.scatter(pts[:, 0], pts[:, 1], s=sizes, c=cols, zorder=4, linewidths=0)
        ax.text(-window + LY.alpha_label_offset[0], window - LY.alpha_label_offset[1], tx["alpha_label"], color=ST.alpha_colour,
                fontsize=F.alpha_label * sc)
        ax.annotate("", xy=(-window + LY.arrow_offset[1], window - LY.arrow_offset[2]),
                    xytext=(-window + LY.arrow_offset[0], window - LY.arrow_offset[2]),
                    arrowprops=dict(arrowstyle=ST.arrow_style, color=ST.alpha_colour, lw=ST.arrow_width))
        ax.text(layer_x.max() + LY.foil_label_offset[0], -window + LY.foil_label_offset[1], tx["foil_label"], color=gold,
                fontsize=F.foil_label * sc)
        # histogram of the particles that have already left
        done = exit_t <= tf
        axh.clear()
        axh.set_facecolor(ST.hist_background)
        counts, _ = np.histogram(angles[done], bins=bins)
        axh.bar(bins[:-1] + bin_width / 2, counts, width=ST.bar_width, color=ST.bar_colour, log=True)
        axh.axvspan(FM.turned_deg, FM.angle_max, color=ST.turned_colour, alpha=ST.turned_region_alpha, lw=0)
        sel = theory_x >= FM.theory_min_deg
        axh.plot(theory_x[sel], theory_y[sel], color=gold, lw=ST.theory_width, label=tx["legend"])
        axh.set_xlim(0, FM.angle_max)
        axh.set_yscale("log")
        axh.set_ylim(*LY.hist_ylim)
        axh.set_xlabel(tx["xlabel"], color=ST.text, fontsize=F.axis_label * sc)
        axh.set_ylabel(tx["ylabel"].format(**words), color=ST.text, fontsize=F.axis_label * sc)
        axh.tick_params(colors=ST.text, labelsize=F.tick * sc)
        for s in axh.spines.values():
            s.set_color(ST.spine)
        axh.legend(loc=LY.legend_loc, frameon=False, labelcolor=gold, fontsize=F.legend * sc)
        axh.text(*LY.turned_label_pos, tx["turned_label"].format(**words), color=ST.turned_text, fontsize=F.turned_label * sc)
        fig.texts.clear()
        fig.text(*LY.title_pos, tx["title"], color=ST.title, fontsize=F.title * sc)
        fig.text(*LY.fired_pos, tx["fired"].format(n=int(np.sum(t0 <= tf))), color=ST.text, fontsize=F.counter * sc)
        fig.text(*LY.turned_pos, tx["turned"].format(n=int(np.sum(turned & done)), **words), color=ST.turned_text, fontsize=F.counter * sc)
        fig.text(*LY.note_pos, tx["note"].format(**words), color=ST.note_colour, fontsize=F.note * sc)
        fig.canvas.draw()
        frame = np.asarray(fig.canvas.buffer_rgba())
        if V.fade_s > 0:
            fade = min(smooth(tf, 0.0, V.fade_s), 1.0 - smooth(tf, total - V.fade_s, total))
            if fade < 1.0:
                frame = (bg_rgba + (frame.astype(np.float32) - bg_rgba) * fade).clip(0, 255).astype(np.uint8)
        if writer is None:
            from PIL import Image
            Image.fromarray(frame).save(out)
        else:
            writer.stdin.write(frame.tobytes())
    if writer is not None:
        writer.stdin.close()
        writer.wait()
    plt.close(fig)
    print(f"wrote {out}" + (f" ({frames} frames)" if writer is not None else ""))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--simulate", action="store_true", help="only integrate the trajectories")
    ap.add_argument("--seed", type=int, default=CFG.simulation.seed)
    ap.add_argument("--alphas", type=int, default=N_ALPHA)
    ap.add_argument("--r-min", type=float, default=R_MIN)
    ap.add_argument("--layers", type=int, default=LAYERS)
    ap.add_argument("--lang", choices=("en", "ru"), default="en")
    ap.add_argument("--preview", action="store_true")
    ap.add_argument("--snapshot", type=float, default=None, help="write one PNG at this film time (s)")
    ap.add_argument("--data", type=Path, default=MEDIA / "rutherford_tracks.npz")
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--config", type=Path, default=None, help="another configuration file instead of config.toml")
    ap.add_argument("--set", action="append", default=[], metavar="SECTION.KEY=VALUE", help="override one configuration value")
    args = ap.parse_args()
    V = CFG.video
    out = args.out or MEDIA / f"rutherford_{args.lang}.mp4"
    if args.simulate or not args.data.exists():
        simulate(args.seed, args.alphas, args.data, args.r_min, args.layers)
        if args.simulate:
            return
    if args.snapshot is not None:
        render(args.data, out.with_suffix(".png"), (V.width, V.height), V.fps, None, args.lang, snap=args.snapshot)
    elif args.preview:
        render(args.data, out.with_name(out.stem + "_preview.mp4"), (V.preview_width, V.preview_height), V.preview_fps, V.preview_seconds, args.lang)
    else:
        render(args.data, out, (V.width, V.height), V.fps, None, args.lang)


if __name__ == "__main__":
    main()
