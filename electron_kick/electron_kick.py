r"""A kick to an electron: where the electromagnetic wave front comes from.

A charge at rest is instantaneously given the velocity v along +x at t = 0 (c = 1).
The exact field of this process (Lienard-Wiechert) has three zones:

* r > ct: the information about the kick has not arrived, the field is the Coulomb field of
  the charge at rest at the origin, the field lines are radial from the origin;
* r < ct: the field of a charge that moves uniformly with velocity v.  Its field lines are
  straight and radial from the present position x_p = v t, but crowded towards the plane
  perpendicular to v (Lorentz contraction);
* r = ct: a thin shell, the front of the electromagnetic wave.  It carries the field
  transverse to the radius that joins the two fields.

Gauss's law fixes how the lines are joined.  The flux inside a cone of half-angle psi around
the velocity, seen from the present position, is

    (1/2) (1 - cos(psi) / sqrt(1 - beta^2 sin^2 psi))      (in units of the charge),

and it must equal the flux (1 - cos(theta)) / 2 inside the cone of half-angle theta seen from
the origin.  Hence a line that leaves the charge at the angle psi runs to the shell, goes along
the shell (an arc of the circle r = ct) and continues outwards, radially from the origin, at
the angle theta with

    cos(theta) = cos(psi) / sqrt(1 - beta^2 sin^2 psi),    i.e.   tan(theta) = tan(psi) / gamma.

The lines are drawn with equal flux, so their density is the strength of the field, and the
shell shines where many lines run along it, that is where the radiation field is strong.

All the numbers are in config.toml and all the words in texts.toml (see ../dvconfig.py for --config / --set).

Usage:
    python electron_kick.py --lang en       # film -> media/electron_kick_en.mp4
    python electron_kick.py --lang ru
    python electron_kick.py --lang en --preview
    python electron_kick.py --lang en --snapshot 6    # one PNG at film time 6 s
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


def psi_of_theta(theta: np.ndarray | float, beta: float) -> np.ndarray | float:
    """Angle (from the velocity, seen from the charge) of the inner line that joins the outer
    line leaving the origin at the angle theta."""
    theta = np.asarray(theta, dtype=float)
    c = np.cos(theta)
    gamma = 1.0 / math.sqrt(1.0 - beta ** 2)
    cos_psi = c / (gamma * np.sqrt(1.0 - (c * beta) ** 2))
    return np.arccos(np.clip(cos_psi, -1.0, 1.0))


def theta_of_psi(psi: np.ndarray | float, beta: float) -> np.ndarray | float:
    psi = np.asarray(psi, dtype=float)
    c = np.cos(psi) / np.sqrt(1.0 - (beta * np.sin(psi)) ** 2)
    return np.arccos(np.clip(c, -1.0, 1.0))


def flux_fraction_moving(psi: np.ndarray | float, beta: float) -> np.ndarray | float:
    """Flux of the field of a uniformly moving charge inside the cone of half-angle psi."""
    psi = np.asarray(psi, dtype=float)
    return 0.5 * (1.0 - np.cos(psi) / np.sqrt(1.0 - (beta * np.sin(psi)) ** 2))


def line_geometry(theta: float, beta: float, t: float) -> dict[str, np.ndarray]:
    """Pieces of the field line that is radial at the angle theta far away (t > 0)."""
    radius = t                                 # c = 1
    x_p = beta * t
    psi = float(psi_of_theta(theta, beta))
    cp, sp = math.cos(psi), math.sin(psi)
    s = -x_p * cp + math.sqrt(radius ** 2 - (x_p * sp) ** 2)
    hit = np.array([x_p + s * cp, s * sp])
    theta_in = math.atan2(hit[1], hit[0])
    return dict(start=np.array([x_p, 0.0]), hit=hit, theta_in=theta_in, theta_out=theta, psi=psi, radius=radius)


# ---------------------------------------------------------------------- film

TEXT = {lang: load_texts(HERE, lang) for lang in ("en", "ru")}     # texts.toml


def smooth(x: float, a: float, b: float) -> float:
    u = min(max((x - a) / (b - a), 0.0), 1.0)
    return u * u * (3 - 2 * u)


def num(x: float, fmt: str, lang: str) -> str:
    """A number for use inside $...$: decimal comma in Russian."""
    s = format(x, fmt)
    return s.replace(".", "{,}") if lang == "ru" else s


def rgba(c, alpha: float = 1.0) -> tuple:
    """A colour of the configuration [r, g, b] or [r, g, b, a] with its opacity multiplied by alpha."""
    return (*c[:3], (c[3] if len(c) > 3 else 1.0) * alpha)


def render(out: Path, size: tuple[int, int], fps: int, total: float, lang: str = "en", snap: float | None = None) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.colors as mcolors
    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection

    if snap is None and not shutil.which("ffmpeg"):
        sys.exit("ffmpeg is required")
    tx = TEXT[lang]
    V, MD, TL, LY, F, ST = CFG.video, CFG.model, CFG.timeline, CFG.layout, CFG.fonts, CFG.style
    runs = [tuple(r) for r in TL.runs]               # (beta, start, end) in film seconds
    W, H = size
    dpi = V.dpi
    sc = H / V.reference_height
    cx0, cy0 = LY.origin[0] * W, LY.origin[1] * H
    unit = LY.unit_px * sc                      # pixels travelled by light in one film second
    n_lines = MD.n_lines
    theta_k = np.arccos(1.0 - 2.0 * (np.arange(n_lines) + 0.5) / n_lines)   # equal flux, upper half-plane
    r_far = LY.r_far_px * sc
    BG = ST.background
    scene_w = LY.scene_width * W

    fig = plt.figure(figsize=(W / dpi, H / dpi), dpi=dpi, facecolor=BG)
    ax = fig.add_axes([0, 0, 1, 1])
    clip = matplotlib.patches.Rectangle((0, 0), scene_w, H, transform=ax.transData)

    frames = int(round(total * fps))
    ids = [int(round(snap * fps))] if snap is not None else range(frames)
    writer = None
    if snap is None:
        writer = subprocess.Popen(
            ["ffmpeg", "-y", "-v", "error", "-f", "rawvideo", "-pix_fmt", "rgba", "-s", f"{W}x{H}",
             "-r", str(fps), "-i", "-", "-c:v", "libx264", "-preset", V.preset, "-crf", str(V.crf),
             "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(out)], stdin=subprocess.PIPE)

    def glow(X, Y, color, d, alpha=1.0, z=8):
        for scale, al in ST.glow:
            ax.scatter([X], [Y], s=(d * scale * 72 / dpi) ** 2, linewidths=0, zorder=z,
                       facecolors=[mcolors.to_rgba(color, al * alpha)])

    for k_ in ids:
        tf = k_ / fps
        run = next((r for r in runs if r[1] <= tf < r[2]), runs[-1])
        beta, t_start, t_end = run
        gamma = 1.0 / math.sqrt(1 - beta ** 2)
        t = min(tf - t_start - TL.t_kick, TL.t_front_max)       # time since the kick, film seconds (frozen at the end of a run)
        fade = (min(smooth(tf - t_start, 0.0, V.fade_s), 1 - smooth(tf, t_end - V.fade_s, t_end)) if t_end < total - 1e-9
                else smooth(tf - t_start, 0, V.fade_s))
        ax.clear()
        ax.set_xlim(0, W)
        ax.set_ylim(0, H)
        ax.axis("off")
        ax.add_patch(matplotlib.patches.Rectangle((0, 0), W, H, color=BG, zorder=0))
        ax.set_clip_path(clip)
        segs_out, segs_in, arcs = [], [], []
        if t <= 0.0:
            for th in theta_k:
                segs_out.append([(cx0, cy0), (cx0 + r_far * math.cos(th), cy0 + r_far * math.sin(th))])
            x_charge, radius = 0.0, 0.0
        else:
            radius = t                            # in light-seconds: c = 1
            x_charge = beta * t
            for th in theta_k:
                g = line_geometry(float(th), beta, t)
                p0 = (cx0 + unit * g["start"][0], cy0)
                p1 = (cx0 + unit * g["hit"][0], cy0 + unit * g["hit"][1])
                segs_in.append([p0, p1])
                a0, a1 = g["theta_in"], g["theta_out"]
                aa = np.linspace(a0, a1, MD.arc_samples)
                arcs.append([(cx0 + unit * radius * math.cos(a), cy0 + unit * radius * math.sin(a)) for a in aa])
                far = (cx0 + unit * radius * math.cos(a1), cy0 + unit * radius * math.sin(a1))
                segs_out.append([far, (cx0 + r_far * math.cos(a1), cy0 + r_far * math.sin(a1))])
        a_f = fade
        ax.add_collection(LineCollection(segs_out, colors=[rgba(ST.outer, ST.outer_alpha * a_f)] * len(segs_out), linewidths=ST.outer_width * sc, zorder=2))
        if segs_in:
            ax.add_collection(LineCollection(segs_in, colors=[rgba(ST.inner, ST.inner_alpha * a_f)] * len(segs_in), linewidths=ST.inner_width * sc, zorder=3))
        if arcs:
            ax.add_collection(LineCollection(arcs, colors=[rgba(ST.front, ST.arc_alpha * a_f)] * len(arcs), linewidths=ST.arc_width * sc, zorder=4, capstyle="round"))
        # light circle r = ct
        if radius > 0:
            a_ = np.linspace(0.0, math.pi, MD.circle_samples)
            ax.plot(cx0 + unit * radius * np.cos(a_), cy0 + unit * radius * np.sin(a_), color=rgba(ST.circle, a_f), lw=ST.circle_width * sc,
                    ls=(0, tuple(ST.circle_dash)), zorder=4)
        ax.plot([0, scene_w], [cy0, cy0], color=rgba(ST.axis_line, a_f), lw=ST.axis_line_width * sc, zorder=1)
        ax.text(cx0 + LY.x_label_offset[0] * sc, cy0 + LY.x_label_offset[1] * sc, tx["x_label"], color=rgba(ST.origin_label, a_f), fontsize=F.x_label * sc, zorder=10)
        # the charge, the origin and the kick
        ax.scatter([cx0], [cy0], s=(ST.origin_ring_size * sc * 72 / dpi) ** 2, facecolors="none", edgecolors=[rgba(ST.origin_ring, a_f)],
                   linewidths=ST.origin_ring_width * sc, zorder=6)
        glow(cx0 + unit * x_charge, cy0, ST.charge, ST.charge_size * sc, a_f)
        ax.plot([cx0 + unit * x_charge - LY.core_half * sc, cx0 + unit * x_charge + LY.core_half * sc], [cy0, cy0], color=ST.charge_core,
                lw=ST.core_width * sc, alpha=a_f, zorder=9, solid_capstyle="round")
        if -TL.kick_window < t < TL.kick_window:
            k_a = (1 - abs(t) / TL.kick_window)
            ax.annotate("", xy=(cx0 - LY.kick_arrow_gap * sc, cy0), xytext=(cx0 - LY.kick_arrow_gap * sc - LY.kick_arrow_len * sc, cy0),
                        arrowprops=dict(arrowstyle="-|>", color=rgba(ST.kick_colour, k_a * a_f), lw=ST.kick_arrow_width * sc,
                                        mutation_scale=ST.kick_arrow_mutation * sc), zorder=10)
            ax.text(cx0 + LY.kick_text_offset[0] * sc, cy0 + LY.kick_text_offset[1] * sc, tx["kick"], color=rgba(ST.kick_text_colour, k_a * a_f),
                    fontsize=F.kick * sc, zorder=10)
        if t > 0 and x_charge > TL.v_arrow_min_x:
            ax.annotate("", xy=(cx0 + unit * x_charge + LY.v_arrow_to * sc, cy0 + LY.v_arrow_dy * sc),
                        xytext=(cx0 + unit * x_charge + LY.v_arrow_from * sc, cy0 + LY.v_arrow_dy * sc),
                        arrowprops=dict(arrowstyle="-|>", color=rgba(ST.v_colour, a_f), lw=ST.v_arrow_width * sc), zorder=10)
            ax.text(cx0 + unit * x_charge + LY.v_label_offset[0] * sc, cy0 + LY.v_label_offset[1] * sc, tx["v_label"],
                    color=rgba(ST.v_colour, a_f), fontsize=F.v_label * sc, zorder=10)
        # labels of the zones
        if t > TL.front_label_t:
            ax.text(cx0 + unit * radius * LY.front_label[0], cy0 + unit * radius * LY.front_label[1], tx["front_label"],
                    color=rgba(ST.front_label_colour, a_f), fontsize=F.front_label * sc, zorder=10)
        # panel with the formulas
        fig.texts.clear()
        fig.text(LY.panel_x, LY.title_y, tx["title"], color=rgba(ST.title_colour, a_f), fontsize=F.title * sc)
        beta_line = tx["beta_line"].replace("{beta}", num(beta, ".2f", lang)).replace("{gamma}", num(gamma, ".2f", lang))
        fig.text(LY.panel_x, LY.beta_y, beta_line, color=rgba(ST.beta_colour, a_f), fontsize=F.beta * sc, family=F.mono_family)
        items = [
            (ST.outer, tx["zone_outer"], tx["body_outer"]),
            (ST.front, tx["zone_front"], tx["body_front"]),
            (ST.inner, tx["zone_inner"], tx["body_inner"]),
        ]
        for i, (col, head, body) in enumerate(items):
            y = LY.items_y0 - LY.items_dy * i
            fig.text(LY.panel_x, y, tx["marker"] + " " + head, color=rgba(col, a_f), fontsize=F.zone_name * sc, family=F.mono_family)
            fig.text(LY.panel_x, y - LY.body_dy, body, color=rgba(ST.zone_body_colour, a_f), fontsize=F.zone_body * sc,
                     linespacing=F.body_linespacing, va="top")
        fig.text(LY.panel_x, LY.gauss_y, tx["gauss"], color=rgba(ST.gauss_colour, a_f), fontsize=F.gauss * sc)
        fig.text(LY.panel_x, LY.formula1_y, tx["formula1"], color=rgba(ST.formula_colour, a_f), fontsize=F.formula1 * sc)
        fig.text(LY.panel_x, LY.formula2_y, tx["formula2"], color=rgba(ST.formula_colour, a_f), fontsize=F.formula2 * sc)
        fig.text(LY.panel_x, LY.note_y, tx["note"], color=rgba(ST.note_colour, a_f), fontsize=F.note * sc, linespacing=F.note_linespacing, va="top")
        ax.add_patch(matplotlib.patches.Rectangle((0, 0), scene_w, LY.caption_box_height * H, color=tuple(ST.caption_box), zorder=11))
        cap = tx["cap_rest"] if t <= 0 else (tx["cap_near"] if t < TL.caption_front_t else tx["cap_front"])
        fig.text(*LY.caption_pos, cap, color=rgba(ST.caption_colour, a_f), fontsize=F.caption * sc)
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
    ap.add_argument("--lang", choices=("en", "ru"), default="en")
    ap.add_argument("--preview", action="store_true")
    ap.add_argument("--snapshot", type=float, default=None)
    ap.add_argument("--seconds", type=float, default=CFG.timeline.film_length)
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--config", type=Path, default=None, help="another configuration file instead of config.toml")
    ap.add_argument("--set", action="append", default=[], metavar="SECTION.KEY=VALUE", help="override one configuration value")
    args = ap.parse_args()
    V = CFG.video
    out = args.out or HERE / "media" / f"electron_kick_{args.lang}.mp4"
    out.parent.mkdir(parents=True, exist_ok=True)
    if args.snapshot is not None:
        render(out.with_suffix(".png"), (V.width, V.height), V.fps, args.seconds, args.lang, snap=args.snapshot)
    elif args.preview:
        render(out.with_name(out.stem + "_preview.mp4"), (V.preview_width, V.preview_height), V.preview_fps, args.seconds, args.lang)
    else:
        render(out, (V.width, V.height), V.fps, args.seconds, args.lang)


if __name__ == "__main__":
    main()
