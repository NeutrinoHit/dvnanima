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

All the numbers are in config.toml and all the words in texts.toml (see ../dvconfig.py for --config / --set).

Usage:
    python running_charge.py --lang en           # film -> media/running_charge_en.mp4
    python running_charge.py --lang ru           # film -> media/running_charge_ru.mp4
    python running_charge.py --preview           # 6 s low-resolution film
    python running_charge.py --snapshot 18       # one PNG at film time 18 s
    python running_charge.py --still             # clean still without captions and graph (book previews)
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

B_REAL = 2 * (1 / CFG.model.alpha_inverse) / (3 * math.pi)
B_FILM = CFG.model.b_film
K0 = CFG.model.k0
R0 = CFG.model.r0
R_SCENE = CFG.model.r_scene             # half-height of the scene in lambda_C
PAIR_DENSITY = CFG.model.pair_density    # live pairs per unit area
MEAN_LIFE = CFG.model.mean_life

TEXT = {lang: load_texts(HERE, lang) for lang in ("en", "ru")}     # texts.toml


def num(x: float, fmt: str, lang: str) -> str:
    """A number for plain text: decimal comma in Russian."""
    s = format(x, fmt)
    return s.replace(".", ",") if lang == "ru" else s


def charge_ratio(r: np.ndarray | float, b: float = B_FILM) -> np.ndarray | float:
    """Q(r)/e for the one-loop running."""
    r = np.asarray(r, dtype=float)
    return 1.0 / (1.0 - b * 0.5 * np.log1p(1.0 / r ** 2))


def smooth(x: float, a: float, b: float) -> float:
    t = min(max((x - a) / (b - a), 0.0), 1.0)
    return t * t * (3 - 2 * t)


# ------------------------------------------------------------------ the pairs

def make_pairs(seed: int, total: float) -> dict[str, np.ndarray]:
    P = CFG.pairs
    rng = np.random.default_rng(seed)
    r_out = R_SCENE * P.area_margin
    area = math.pi * r_out ** 2
    rate = PAIR_DENSITY * area / MEAN_LIFE
    n = int(rate * (total + P.count_pad_s))
    t0 = np.sort(rng.uniform(P.t_start, total, n))
    life = np.clip(rng.gamma(P.life_shape, MEAN_LIFE / P.life_shape, n), *P.life_clip)
    r = np.sqrt(rng.uniform(P.r_inner ** 2, r_out ** 2, n))
    near = rng.uniform(size=n) < P.near_fraction            # extra pairs close to the charge (polarization cloud)
    r = np.where(near, np.sqrt(rng.uniform(P.near_range[0] ** 2, P.near_range[1] ** 2, n)), r)
    phi = rng.uniform(0, 2 * math.pi, n)
    d0 = rng.uniform(*P.dipole_length, n)
    noise = rng.normal(size=n)          # standard normal, turned into an angle with kappa later
    unif = rng.uniform(-math.pi, math.pi, n)
    vm_seed = rng.integers(0, 2 ** 31, n)
    return dict(t0=t0, life=life, r=r, phi=phi, d0=d0, noise=noise, unif=unif, vm_seed=vm_seed)


def pair_ends(pairs: dict[str, np.ndarray], t: float, s_of) -> tuple[np.ndarray, ...]:
    """Positions of the positron and electron ends and an opacity for every living pair."""
    P = CFG.pairs
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
    stretch = 1.0 + P.stretch_gain * s_of(t) / (1.0 + (r / P.stretch_radius) ** 2)
    d = pairs["d0"][idx] * stretch * (P.breathing[0] + P.breathing[1] * np.sin(math.pi * np.clip(age, 0, 1)))
    cx, cy = r * np.cos(phi), r * np.sin(phi)
    ux, uy = np.cos(phi + psi), np.sin(phi + psi)           # from positron to electron
    pos = np.stack([cx - 0.5 * d * ux, cy - 0.5 * d * uy], axis=1)
    ele = np.stack([cx + 0.5 * d * ux, cy + 0.5 * d * uy], axis=1)
    fade_in = np.clip(age * life[idx] / P.edge_fade_s, 0, 1)
    fade_out = np.clip((1 - age) * life[idx] / P.edge_fade_s, 0, 1)
    return pos, ele, np.minimum(fade_in, fade_out), age, life[idx]


# ---------------------------------------------------------------------- film

def timeline(total: float) -> dict[str, tuple[float, float]]:
    TL = CFG.timeline
    k = total / TL.film_length
    return {name: (TL[name][0] * k, TL[name][1] * k) for name in ("charge_on", "panel", "probe")}


def probe_radius(t: float, tl: dict[str, tuple[float, float]]) -> float:
    PR = CFG.probe
    a, b = tl["probe"]
    u = min(max((t - a) / (b - a), 0.0), 1.0)
    e = 0.5 - 0.5 * math.cos(math.pi * u)
    return math.exp(math.log(PR.r_start) + (math.log(PR.r_end) - math.log(PR.r_start)) * e)


def render(out: Path, size: tuple[int, int], fps: int, total: float, lang: str = "en", snap: float | None = None,
           chrome: bool = True) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection
    from scipy.ndimage import gaussian_filter

    if snap is None and not shutil.which("ffmpeg"):
        sys.exit("ffmpeg is required")
    tx = TEXT[lang]
    V, TLN, PR, SC, FL, EL, GL = CFG.video, CFG.timeline, CFG.probe, CFG.scene, CFG.field, CFG.electron, CFG.glow
    PS, FX, PN, PT, CP, F, ST = CFG.pair_style, CFG.flash, CFG.panel, CFG.panel_text, CFG.caption, CFG.fonts, CFG.style
    W, H = size
    dpi = V.dpi
    sc = H / V.reference_height
    unit = H / (2 * R_SCENE)               # pixels per lambda_C
    cx0, cy0 = SC.centre[0] * W, SC.centre[1] * H
    tl = timeline(total)
    pairs = make_pairs(CFG.pairs.seed, total)
    s_of = lambda t: smooth(t, *tl["charge_on"])
    BG = ST.background
    bg_rgba = np.array([int(BG[1:3], 16), int(BG[3:5], 16), int(BG[5:7], 16), 255], np.float32)

    fig = plt.figure(figsize=(W / dpi, H / dpi), dpi=dpi, facecolor=BG)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, W)
    ax.set_ylim(0, H)
    ax.axis("off")
    pa = fig.add_axes(PN.axes, facecolor="none")

    # background: vignette and slowly drifting vacuum texture (low-resolution, upsampled)
    cell = SC.texture_cell
    yy, xx = np.mgrid[0:H // cell, 0:W // cell]
    rr = np.hypot(xx * cell - cx0, yy * cell - cy0) / (SC.vignette_radius * W)
    vignette = np.clip(1.0 - SC.vignette_strength * rr ** SC.vignette_power, 0.0, 1.0)
    rng = np.random.default_rng(SC.texture_seed)
    tex = [gaussian_filter(rng.normal(size=(H // cell, W // cell)), SC.texture_blur) for _ in range(SC.texture_count)]
    tex = [(t_ - t_.min()) / (t_.max() - t_.min()) for t_ in tex]
    halo_y, halo_x = np.mgrid[0:H // cell, 0:W // cell]
    halo_r = np.hypot((halo_x * cell - cx0) / unit, (halo_y * cell - cy0) / unit)
    cloud = np.exp(-((halo_r - SC.cloud_centre) / SC.cloud_width) ** 2)          # polarization cloud around the charge

    def glow(points: np.ndarray, color: str, diameter: float, opacity: np.ndarray, z: int) -> None:
        if len(points) == 0:
            return
        X = cx0 + unit * points[:, 0]
        Y = cy0 + unit * points[:, 1]
        for scale, alpha in GL.layers:
            s = (diameter * scale * 72 / dpi) ** 2
            ax.scatter(X, Y, s=s, linewidths=0, zorder=z,
                       edgecolors="none", facecolors=[matplotlib.colors.to_rgba(color, alpha * o) for o in opacity])

    frames = int(round(total * fps))
    ids = [int(round(snap * fps))] if snap is not None else range(frames)
    writer = None
    if snap is None:
        writer = subprocess.Popen(
            ["ffmpeg", "-y", "-v", "error", "-f", "rawvideo", "-pix_fmt", "rgba", "-s", f"{W}x{H}",
             "-r", str(fps), "-i", "-", "-c:v", "libx264", "-preset", V.preset, "-crf", str(V.crf),
             "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(out)], stdin=subprocess.PIPE)

    rs_curve = np.geomspace(PN.r_range[0], PN.r_range[1], CFG.model.curve_points)
    for k in ids:
        t = k / fps
        s = s_of(t)
        ax.clear()
        ax.set_xlim(0, W)
        ax.set_ylim(0, H)
        ax.axis("off")
        # background
        phase = 0.5 + 0.5 * math.sin(2 * math.pi * t / SC.texture_period_s)
        mix = (1 - phase) * tex[0] + phase * tex[1]
        mix = SC.texture_mix[0] * mix + SC.texture_mix[1] * tex[2]
        img = np.zeros(mix.shape + (3,))
        img[..., 0] = SC.base[0] + SC.gain[0] * mix
        img[..., 1] = SC.base[1] + SC.gain[1] * mix
        img[..., 2] = SC.base[2] + SC.gain[2] * mix
        img = img * vignette[..., None]
        img[..., 0] += SC.cloud_tint[0] * s * cloud
        img[..., 1] += SC.cloud_tint[1] * s * cloud
        ax.imshow(np.clip(img, 0, 1), extent=(0, W, 0, H), origin="lower", interpolation="bicubic", zorder=0, aspect="auto")
        # lambda_C circle
        ang = np.linspace(0, 2 * math.pi, SC.circle_points)
        ax.plot(cx0 + unit * np.cos(ang), cy0 + unit * np.sin(ang), color=ST.circle_colour, lw=ST.circle_width * sc,
                alpha=ST.circle_alpha, ls=(0, tuple(ST.circle_dash)), zorder=1)
        ax.text(cx0 + unit * SC.circle_label[0], cy0 + unit * SC.circle_label[1], tx["lambda_label"], color=ST.circle_colour,
                alpha=ST.circle_label_alpha, fontsize=F.circle_label * sc, zorder=1)
        # field lines whose strength follows Q(r)
        if s > FL.visible_above:
            segs, widths, cols = [], [], []
            rr_ = np.geomspace(FL.radius_min, R_SCENE * FL.radius_factor, FL.points)
            for i in range(FL.lines):
                a = 2 * math.pi * i / FL.lines + FL.wobble * math.sin(FL.wobble_rate * t + i)
                pts = np.stack([cx0 + unit * rr_ * math.cos(a), cy0 + unit * rr_ * math.sin(a)], axis=1)
                q = charge_ratio(rr_)
                for j in range(len(rr_) - 1):
                    segs.append(pts[j:j + 2])
                    widths.append((FL.width_base + FL.width_gain * (q[j] - 1.0) * FL.width_boost + FL.width_add) * sc)
                    alpha = s * (FL.alpha_base + FL.alpha_gain * min(1.0, (q[j] - 1.0) / FL.alpha_q_scale)) / (1 + FL.alpha_decay * rr_[j])
                    cols.append((*ST.field_colour, min(alpha, FL.alpha_max)))
            ax.add_collection(LineCollection(segs, linewidths=widths, colors=cols, zorder=2, capstyle="round"))
        # pairs
        pos, ele, op, age, life = pair_ends(pairs, t, s_of)
        if len(pos):
            lines = [[(cx0 + unit * p[0], cy0 + unit * p[1]), (cx0 + unit * e[0], cy0 + unit * e[1])] for p, e in zip(pos, ele)]
            ax.add_collection(LineCollection(lines, colors=[(*PS.bond_colour, PS.bond_alpha * o) for o in op],
                                             linewidths=PS.bond_width * sc, zorder=3))
            glow(ele, GL.electron_colour, GL.base_diameter * unit * GL.pair_factor, op, 4)
            glow(pos, GL.positron_colour, GL.base_diameter * unit * GL.pair_factor, op, 4)
            # creation / annihilation flashes
            born = np.clip(1.0 - age * life / FX.born_time, 0, 1)
            died = np.clip(1.0 - (1 - age) * life / FX.died_time, 0, 1)
            mids = 0.5 * (pos + ele)
            for flash, color, rad0 in ((born, FX.born_colour, FX.born_radius), (died, FX.died_colour, FX.died_radius)):
                m = flash > FX.visible_above
                if m.any():
                    X = cx0 + unit * mids[m, 0]
                    Y = cy0 + unit * mids[m, 1]
                    rad = (rad0[0] + rad0[1] * (1 - flash[m])) * unit
                    ax.scatter(X, Y, s=(2 * rad * 72 / dpi) ** 2, facecolors="none",
                               edgecolors=[matplotlib.colors.to_rgba(color, FX.alpha * f) for f in flash[m]],
                               linewidths=FX.line_width * sc, zorder=5)
        # the electron in the centre
        for diam, col, alpha, z in zip(EL.glow_diameters, EL.glow_colours, EL.glow_alphas, (6, 6, 7)):
            ax.scatter([cx0], [cy0], s=(diam * unit * 72 / dpi) ** 2, c=col, alpha=alpha * s, linewidths=0, zorder=z)
        ax.plot([cx0 - EL.sign_half_length * unit, cx0 + EL.sign_half_length * unit], [cy0, cy0], color=EL.sign_colour,
                lw=EL.sign_width * sc, alpha=s, zorder=8, solid_capstyle="round")
        # probe
        pa.clear()
        pa.set_facecolor("none")
        p_alpha = smooth(t, *tl["panel"]) if chrome else 0.0
        r_p = probe_radius(t, tl)
        if t >= tl["probe"][0] + TLN.probe_fade[0]:
            pa_ = smooth(t, tl["probe"][0] + TLN.probe_fade[0], tl["probe"][0] + TLN.probe_fade[1])
            ang_p = math.radians(PR.angle_deg)
            path_r = np.geomspace(PR.r_start, max(r_p, PR.r_end), PR.path_points)
            ax.plot(cx0 + unit * path_r * math.cos(ang_p), cy0 + unit * path_r * math.sin(ang_p),
                    color=ST.probe_path_colour, lw=PR.path_width * sc, alpha=PR.path_alpha * pa_, zorder=9)
            px, py = r_p * math.cos(ang_p), r_p * math.sin(ang_p)
            glow(np.array([[px, py]]), PR.glow_colour, GL.base_diameter * unit * PR.glow_factor, np.array([pa_]), 10)
            if chrome:
                ax.text(cx0 + unit * (px + PR.label_offset[0]), cy0 + unit * (py + PR.label_offset[1]), tx["probe_label"],
                        color=PR.label_colour, alpha=PR.label_alpha * pa_, fontsize=F.probe_label * sc, zorder=10)
        # panel with the measured charge
        if p_alpha > PN.visible_above:
            pa.set_xscale("log")
            pa.set_xlim(PN.r_range[1], PN.r_range[0])
            pa.set_ylim(*PN.y_range)
            for sp in pa.spines.values():
                sp.set_visible(False)
            pa.spines["bottom"].set_visible(True)
            pa.spines["bottom"].set_color((*PN.axis_colour, PN.spine_alpha * p_alpha))
            pa.spines["left"].set_visible(True)
            pa.spines["left"].set_color((*PN.axis_colour, PN.spine_alpha * p_alpha))
            q_all = charge_ratio(rs_curve)
            pa.plot(rs_curve, q_all, color=(1, 1, 1, PN.ghost_alpha * p_alpha), lw=PN.ghost_width * sc)
            shown = rs_curve >= r_p
            if t >= tl["probe"][0]:
                pa.plot(rs_curve[shown], q_all[shown], color=(*PN.curve_colour, p_alpha), lw=PN.curve_width * sc, solid_capstyle="round")
            pa.axhline(1.0, color=(*PN.axis_colour, PN.level_alpha * p_alpha), lw=PN.level_width * sc, ls=(0, tuple(PN.level_dash)))
            tick_col = (*PN.tick_colour, PN.tick_alpha * p_alpha)
            pa.set_xticks(PN.xticks)
            pa.set_xticklabels([num(v, "g", lang) for v in PN.xticks], color=tick_col, fontsize=F.tick * sc)
            pa.set_yticks(PN.yticks)
            pa.set_yticklabels([num(v, "g", lang) for v in PN.yticks], color=tick_col, fontsize=F.tick * sc)
            pa.minorticks_off()
            pa.tick_params(colors=(*PN.axis_colour, PN.tick_mark_alpha * p_alpha), length=PN.tick_length)
            pa.set_xlabel(tx["panel_xlabel"], color=(*PN.tick_colour, PN.label_alpha * p_alpha), fontsize=F.axis_label * sc,
                          labelpad=PN.label_pad)
            if t >= tl["probe"][0]:
                q_now = float(charge_ratio(r_p))
                pa.scatter([r_p], [q_now], s=(PN.marker_inner * sc) ** 2 * PN.marker_area, c=PN.marker_inner_colour, zorder=5, linewidths=0)
                pa.scatter([r_p], [q_now], s=(PN.marker_outer * sc) ** 2 * PN.marker_area, c=PN.marker_outer_colour,
                           alpha=PN.marker_outer_alpha, zorder=4, linewidths=0)
            else:
                q_now = 1.0
            fig.texts.clear()
            fig.text(PT.x, PT.title_y, tx["panel_title"], color=(*PT.title_colour, p_alpha), fontsize=F.panel_title * sc)
            fig.text(PT.x, PT.number_y, num(q_now, PT.number_format, lang), color=(*PT.number_colour, p_alpha),
                     fontsize=F.panel_number * sc, family=ST.number_family)
            fig.text(PT.x, PT.sub_y, tx["panel_sub"], color=(*PT.sub_colour, PT.sub_alpha * p_alpha), fontsize=F.panel_sub * sc)
            fig.text(PT.x, PT.note_y, tx["panel_note"], color=(*PT.note_colour, PT.note_alpha * p_alpha),
                     fontsize=F.panel_note * sc, linespacing=PT.note_linespacing, va="top")
        else:
            pa.axis("off")
            fig.texts.clear()
        if chrome:
            # captions on a soft dark strip
            ax.add_patch(matplotlib.patches.Rectangle((0, 0), W, CP.strip_height * H, color=tuple(CP.strip_colour), zorder=11, lw=0))
            if t < tl["charge_on"][0]:
                cap = tx["cap_vacuum"]
            elif t < tl["probe"][0]:
                cap = tx["cap_screen"]
            else:
                cap = tx["cap_probe"]
            for line, y, size, col in zip(cap.split("\n"), CP.y, CP.size, CP.colour):
                fig.text(CP.x, y, line, color=tuple(col), fontsize=size * sc)
        fig.canvas.draw()
        frame = np.asarray(fig.canvas.buffer_rgba())
        if V.fade_s > 0:
            fade = min(smooth(t, 0.0, V.fade_s), 1.0 - smooth(t, total - V.fade_s, total))
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
    print(f"wrote {out}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--lang", choices=("en", "ru"), default="en")
    ap.add_argument("--preview", action="store_true")
    ap.add_argument("--snapshot", type=float, default=None)
    ap.add_argument("--still", type=float, nargs="?", const=CFG.video.still_time, default=None,
                    help="clean still (video.still_width x still_height) without captions and graph at this film time (for book previews)")
    ap.add_argument("--seconds", type=float, default=CFG.timeline.film_length)
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--config", type=Path, default=None, help="another configuration file instead of config.toml")
    ap.add_argument("--set", action="append", default=[], metavar="SECTION.KEY=VALUE", help="override one configuration value")
    args = ap.parse_args()
    V = CFG.video
    out = args.out or HERE / "media" / f"running_charge_{args.lang}.mp4"
    out.parent.mkdir(parents=True, exist_ok=True)
    if args.still is not None:
        from PIL import Image
        target = out.with_name(out.stem + "_still.png")
        render(target, (V.still_width, V.still_height), V.fps, args.seconds, args.lang, snap=args.still, chrome=False)
        img = Image.open(target)
        img.crop((0, 0, int(V.still_crop * img.width), img.height)).save(target)
        print(f"cropped {target}")
    elif args.snapshot is not None:
        render(out.with_suffix(".png"), (V.width, V.height), V.fps, args.seconds, args.lang, snap=args.snapshot)
    elif args.preview:
        render(out.with_name(out.stem + "_preview.mp4"), (V.preview_width, V.preview_height), V.preview_fps, V.preview_seconds, args.lang)
    else:
        render(out, (V.width, V.height), V.fps, args.seconds, args.lang)


if __name__ == "__main__":
    main()
