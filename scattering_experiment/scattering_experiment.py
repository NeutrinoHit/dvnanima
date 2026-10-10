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

All the numbers are in config.toml and all the words in texts.toml (see ../dvconfig.py for --config / --set).

Usage:
    python scattering_experiment.py --lang en                # film -> media/scattering_experiment_en.mp4
    python scattering_experiment.py --lang ru
    python scattering_experiment.py --lang en --preview
    python scattering_experiment.py --lang en --snapshot 15  # one PNG at film time 15 s
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

N_BUNCH = CFG.model.bunch_size
RING = CFG.model.detector_radius     # detector radius in pixels at 720p
SIGMA_X = CFG.model.sigma_x
SIGMA_Y = CFG.model.sigma_y
FILM_LENGTH = CFG.timeline.film_length


def sample_cos_theta(rng: np.random.Generator, n: int) -> np.ndarray:
    """cos(theta) with density (3/8)(1 + c^2) on [-1, 1] (rejection sampling)."""
    out: list[float] = []
    while len(out) < n:
        c = rng.uniform(-1, 1, 2 * n)
        u = rng.uniform(0, 2, 2 * n)
        out.extend(c[u < 1 + c * c])
    return np.array(out[:n])


def angular_density(c: np.ndarray) -> np.ndarray:
    return 3.0 / 8.0 * (1.0 + c * c)


def make_events(seed: int, total: float):
    """List of bunch crossings with their events."""
    MD, TL = CFG.model, CFG.timeline
    rng = np.random.default_rng(seed)
    crossings = []
    t_first = TL.first_crossing
    times = [t_first] + list(np.arange(TL.crossing_start, total - TL.crossing_end_margin, TL.crossing_period))
    for k, t_c in enumerate(times):
        dx_e = rng.normal(0, SIGMA_X, N_BUNCH)
        dy_e = rng.normal(0, SIGMA_Y, N_BUNCH)
        dx_p = rng.normal(0, SIGMA_X, N_BUNCH)
        dy_p = rng.normal(0, SIGMA_Y, N_BUNCH)
        n_ev = 1 if k == 0 else int(rng.poisson(MD.events_per_crossing_mean))
        events = []
        used_e: set[int] = set()
        used_p: set[int] = set()
        order_e = np.argsort(np.abs(dx_e))
        order_p = np.argsort(np.abs(dx_p))
        for _ in range(n_ev):
            ie = int(rng.choice([i for i in order_e[:MD.nearest_particles] if i not in used_e]))
            ip = int(rng.choice([i for i in order_p[:MD.nearest_particles] if i not in used_p]))
            used_e.add(ie)
            used_p.add(ip)
            c = float(sample_cos_theta(rng, 1)[0])
            phi = rng.uniform(0, 2 * math.pi)
            if k == 0:
                c, phi = MD.first_event_cos, MD.first_event_phi               # a clear, readable first event
            events.append(dict(ie=ie, ip=ip, c=c, phi=phi))
        crossings.append(dict(t_c=float(t_c), first=(k == 0), dx_e=dx_e, dy_e=dy_e, dx_p=dx_p, dy_p=dy_p, events=events))
    return crossings


def xc_of(t: float, cr: dict, v: float) -> float:
    """Centre of the e- bunch (the e+ bunch is at -xc)."""
    t_c = cr["t_c"]
    if cr["first"]:
        # accelerating beam: x = -d (1 - (t/t_c)^p), after the crossing it keeps its speed
        d = CFG.model.first_start_distance
        if t <= t_c:
            return -d * (1 - (max(t, 0.0) / t_c) ** CFG.model.first_exponent)
        return d * CFG.model.first_exponent / t_c * (t - t_c)
    return v * (t - t_c)


def event_time_and_point(ev: dict, cr: dict, v: float) -> tuple[float, float, float]:
    """Time and place where e-_i and e+_j meet."""
    dxi, dxj = cr["dx_e"][ev["ie"]], cr["dx_p"][ev["ip"]]
    xc = 0.5 * (dxj - dxi)                  # e-: xc + dxi = e+: -xc + dxj
    t_c = cr["t_c"]
    if cr["first"]:
        d = CFG.model.first_start_distance
        speed = d * CFG.model.first_exponent / t_c
        t = t_c + xc / speed
    else:
        t = t_c + xc / v
    ym = 0.5 * (cr["dy_e"][ev["ie"]] + cr["dy_p"][ev["ip"]])
    return float(t), 0.5 * (dxi + dxj), float(ym)


# ---------------------------------------------------------------------- film

TEXT = {lang: load_texts(HERE, lang) for lang in ("en", "ru")}     # texts.toml


def put(text: str, **values: str) -> str:
    """Replace the placeholders {name} of a text (LaTeX braces stay untouched)."""
    for k, v in values.items():
        text = text.replace("{" + k + "}", v)
    return text


def render(out: Path, size: tuple[int, int], fps: int, total: float, lang: str = "en", snap: float | None = None) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.colors as mcolors
    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection

    if snap is None and not shutil.which("ffmpeg"):
        sys.exit("ffmpeg is required")
    V, MD, TL, LY, F, ST = CFG.video, CFG.model, CFG.timeline, CFG.layout, CFG.fonts, CFG.style
    tx = TEXT[lang]
    W, H = size
    dpi = V.dpi
    sc = H / V.reference_height
    BG = ST.background
    cx0, cy0 = LY.centre[0] * W, LY.centre[1] * H
    v_fast = MD.bunch_speed * sc
    ring = RING * sc
    D0, EXP = MD.first_start_distance, MD.first_exponent
    crossings = make_events(MD.events_seed, total)
    # flatten events with absolute times
    flat = []
    for k, cr in enumerate(crossings):
        v = v_fast
        for ev in cr["events"]:
            t_ev, xm, ym = event_time_and_point(ev, cr, v / sc)
            slow = MD.first_slowdown if cr["first"] else 1.0
            nx, ny, nz = ev["c"], math.sqrt(1 - ev["c"] ** 2) * math.cos(ev["phi"]), math.sqrt(1 - ev["c"] ** 2) * math.sin(ev["phi"])
            speed3 = MD.muon_speed * sc / slow
            t_hit = t_ev + ring / speed3
            flat.append(dict(k=k, t=t_ev, xm=xm, ym=ym, n=(nx, ny, nz), speed=speed3, t_hit=t_hit, c=ev["c"], first=cr["first"]))
    t_hits = np.array([e["t_hit"] for e in flat])
    cs = np.array([e["c"] for e in flat])

    fig = plt.figure(figsize=(W / dpi, H / dpi), dpi=dpi, facecolor=BG)
    ax = fig.add_axes([0, 0, 1, 1])
    ph = fig.add_axes(LY.hist_axes, facecolor="none")
    rng_bg = np.random.default_rng(LY.star_seed)
    stars = rng_bg.uniform(0, 1, (LY.star_count, 2))
    edges = np.linspace(-1, 1, MD.histogram_bins + 1)
    centers = 0.5 * (edges[1:] + edges[:-1])

    frames = int(round(total * fps))
    ids = [int(round(snap * fps))] if snap is not None else range(frames)
    writer = None
    if snap is None:
        writer = subprocess.Popen(
            ["ffmpeg", "-y", "-v", "error", "-f", "rawvideo", "-pix_fmt", "rgba", "-s", f"{W}x{H}",
             "-r", str(fps), "-i", "-", "-c:v", "libx264", "-preset", V.preset, "-crf", str(V.crf),
             "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(out)], stdin=subprocess.PIPE)

    def glow(X, Y, color, d, alpha, z=6):
        X = np.atleast_1d(X)
        Y = np.atleast_1d(Y)
        al = np.broadcast_to(np.atleast_1d(alpha), X.shape)
        for scale, a_ in ST.glow_layers:
            ax.scatter(X, Y, s=(d * scale * 72 / dpi) ** 2, linewidths=0, zorder=z,
                       facecolors=[mcolors.to_rgba(color, a_ * float(x)) for x in al])

    col_e, col_p, col_mu_m, col_mu_p = ST.colour_e, ST.colour_p, ST.colour_mu_m, ST.colour_mu_p
    stage_ends = TL.stage_ends

    for k_ in ids:
        t = k_ / fps
        ax.clear()
        ax.set_xlim(0, W)
        ax.set_ylim(0, H)
        ax.axis("off")
        ax.add_patch(matplotlib.patches.Rectangle((0, 0), W, H, color=BG, zorder=0, lw=0))
        ax.scatter(stars[:, 0] * W, stars[:, 1] * H, s=ST.star_size * sc ** 2 * ST.star_area, c=[tuple(ST.star_colour)] * len(stars), linewidths=0, zorder=1)
        # beam pipe, interaction region and detector
        for yy in (-LY.pipe_half_gap, LY.pipe_half_gap):
            ax.plot([0, LY.pipe_length * W], [cy0 + yy * sc, cy0 + yy * sc], color=ST.pipe_colour, lw=ST.pipe_width * sc, alpha=ST.pipe_alpha, zorder=2)
        a_ = np.linspace(0, 2 * math.pi, LY.ring_samples)
        ax.plot(cx0 + ring * np.cos(a_), cy0 + ring * np.sin(a_), color=ST.ring_colour, lw=ST.ring_width * sc, alpha=ST.ring_alpha, zorder=2)
        ax.plot(cx0 + (ring + LY.ring_outer_offset * sc) * np.cos(a_), cy0 + (ring + LY.ring_outer_offset * sc) * np.sin(a_), color=ST.ring_colour,
                lw=ST.ring_outer_width * sc, alpha=ST.ring_outer_alpha, zorder=2)
        ax.add_patch(matplotlib.patches.Circle((cx0, cy0), LY.interaction_radius * sc, fill=False, ec=ST.interaction_colour, lw=ST.interaction_width * sc,
                                               ls=(0, tuple(ST.interaction_dash)), alpha=ST.interaction_alpha, zorder=2))
        # accelerating cavities along the pipe (stage 1)
        cw, ch = LY.cavity_size
        for xx in LY.cavity_x:
            pulse = 0.5 + 0.5 * math.sin(TL.cavity_rate * t - TL.cavity_lag * abs(xx) / TL.cavity_length * (1 if xx < 0 else -1))
            a_cav = (ST.cavity_base + ST.cavity_swing * pulse) * (1.0 if t < TL.first_crossing else ST.cavity_idle)
            ax.add_patch(matplotlib.patches.Rectangle((cx0 + xx * sc - cw / 2 * sc, cy0 - ch / 2 * sc), cw * sc, ch * sc,
                                                      fc=(*ST.cavity_face, ST.cavity_face_alpha * a_cav), ec=(*ST.cavity_edge, ST.cavity_edge_alpha * a_cav),
                                                      lw=ST.cavity_width * sc, zorder=2))
        # persistent hits
        for e in flat:
            if t >= e["t_hit"]:
                age = t - e["t_hit"]
                nx, ny, nz = e["n"]
                for sgn, col in ((1, col_mu_m), (-1, col_mu_p)):
                    hx, hy = cx0 + sgn * ring * nx, cy0 + sgn * ring * ny
                    base = ST.hit_rest if age > TL.hit_fade else ST.hit_start - ST.hit_drop * age / TL.hit_fade
                    glow(hx, hy, col, (LY.hit_size[0] + LY.hit_size[1] * sgn * nz) * sc, base, z=5)
        # bunches
        for k, cr in enumerate(crossings):
            t_c = cr["t_c"]
            xc = xc_of(t, cr, v_fast / sc) * sc
            if abs(xc) > LY.cut_distance * sc:
                continue
            gone_e = {ev["ie"] for ev, fe in zip(cr["events"], [e for e in flat if e["k"] == k]) if t >= fe["t"]}
            gone_p = {ev["ip"] for ev, fe in zip(cr["events"], [e for e in flat if e["k"] == k]) if t >= fe["t"]}
            idx_e = [i for i in range(N_BUNCH) if i not in gone_e]
            idx_p = [i for i in range(N_BUNCH) if i not in gone_p]
            xe = cx0 + xc + cr["dx_e"][idx_e] * sc
            ye = cy0 + cr["dy_e"][idx_e] * sc
            xp = cx0 - xc + cr["dx_p"][idx_p] * sc
            yp = cy0 + cr["dy_p"][idx_p] * sc
            keep_e = np.abs(xe - cx0) < LY.keep_distance * sc
            keep_p = np.abs(xp - cx0) < LY.keep_distance * sc
            xe, ye, xp, yp = xe[keep_e], ye[keep_e], xp[keep_p], yp[keep_p]
            if len(xe) == 0 and len(xp) == 0:
                continue
            speed = (v_fast if not cr["first"] else (D0 * EXP / t_c * (max(t, 0) / t_c) ** (EXP - 1.0) * sc if t <= t_c else D0 * EXP / t_c * sc))
            streak = min(speed * LY.streak_factor, LY.streak_max * sc)
            segs_e = [[(a, b), (a - streak, b)] for a, b in zip(xe, ye)]
            segs_p = [[(a, b), (a + streak, b)] for a, b in zip(xp, yp)]
            ax.add_collection(LineCollection(segs_e, colors=[mcolors.to_rgba(col_e, ST.streak_alpha)] * len(segs_e), linewidths=ST.streak_width * sc, zorder=4))
            ax.add_collection(LineCollection(segs_p, colors=[mcolors.to_rgba(col_p, ST.streak_alpha)] * len(segs_p), linewidths=ST.streak_width * sc, zorder=4))
            glow(xe, ye, col_e, ST.particle_size * sc, ST.particle_alpha)
            glow(xp, yp, col_p, ST.particle_size * sc, ST.particle_alpha)
        # events: flash and muons
        for e in flat:
            dt = t - e["t"]
            if dt < 0 or dt > TL.event_life:
                continue
            ox, oy = cx0 + e["xm"] * sc, cy0 + e["ym"] * sc
            slow = MD.first_slowdown if e["first"] else 1.0
            if dt < TL.flash_duration * slow:
                f = 1 - dt / (TL.flash_duration * slow)
                glow(ox, oy, ST.flash_colour, (LY.flash_start[0] + LY.flash_start[1] * (1 - f)) * sc, ST.flash_alpha * f, z=8)
                ax.add_patch(matplotlib.patches.Circle((ox, oy), (LY.flash_ring[0] + LY.flash_ring[1] * (1 - f)) * sc, fill=False,
                                                       ec=mcolors.to_rgba(ST.flash_colour, ST.flash_ring_alpha * f), lw=ST.flash_ring_width * sc, zorder=8))
            nx, ny, nz = e["n"]
            travel = min(dt * e["speed"], ring + LY.muon_stop * sc)
            if dt * e["speed"] < ring + LY.muon_gone * sc:
                for sgn, col in ((1, col_mu_m), (-1, col_mu_p)):
                    px, py = ox + sgn * nx * travel, oy + sgn * ny * travel
                    ax.plot([ox, px], [oy, py], color=mcolors.to_rgba(col, ST.muon_line_alpha), lw=ST.muon_line_width * sc, zorder=7, solid_capstyle="round")
                    glow(px, py, col, (LY.muon_size[0] + LY.muon_size[1] * sgn * nz) * sc, 1.0, z=9)
        # stage chips and captions
        stage = 1 if t < stage_ends[0] else 2 if t < stage_ends[1] else 3 if t < stage_ends[2] else 0
        for i, st in enumerate((1, 2, 3)):
            on = (stage == st) or (stage == 0)
            a_c = ST.chip_alpha_active if stage == st else (ST.chip_alpha_idle if stage == 0 else ST.chip_alpha_other)
            ax.text(LY.chip_x * W, H * (LY.chip_top - LY.chip_step * i), tx[f"stage_{st}"], color=(*ST.chip_on, a_c) if on else (*ST.chip_off, a_c),
                    fontsize=(F.chip_active if stage == st else F.chip) * sc, zorder=12)
        k_cross = sum(1 for cr in crossings[1:] if t >= cr["t_c"])
        n_ev = int(np.sum(t_hits <= t))
        cap = f"cap_{stage if stage else 4}"
        ax.add_patch(matplotlib.patches.Rectangle((0, 0), W, LY.caption_box_height * H, color=tuple(ST.caption_box_colour), zorder=11, lw=0))
        fig.texts.clear()
        fig.text(*LY.caption_pos, tx[cap], color=tuple(ST.caption_colour), fontsize=F.caption * sc)
        ax.text(cx0 + LY.label_beam_e[0] * sc, H * LY.label_beam_e[1], tx["label_e"], color=col_e, fontsize=F.beam_label * sc, alpha=ST.beam_label_alpha, zorder=12)
        ax.text(cx0 + LY.label_beam_p[0] * sc, H * LY.label_beam_p[1], tx["label_p"], color=col_p, fontsize=F.beam_label * sc, alpha=ST.beam_label_alpha, zorder=12)
        ax.text(cx0 + LY.label_mu_m[0] * sc, H * LY.label_mu_m[1], tx["label_mu_m"], color=col_mu_m, fontsize=F.muon_label * sc, alpha=ST.muon_label_alpha, zorder=12)
        ax.text(cx0 + LY.label_mu_p[0] * sc, H * LY.label_mu_p[1], tx["label_mu_p"], color=col_mu_p, fontsize=F.muon_label * sc, alpha=ST.muon_label_alpha, zorder=12)
        # histogram panel
        ph.clear()
        ph.set_facecolor("none")
        p_a = min(1.0, max(0.0, (t - TL.histogram_start) / TL.histogram_fade))
        if p_a > ST.visible_alpha:
            counts, _ = np.histogram(cs[t_hits <= t], bins=edges)
            ph.bar(centers, counts, width=(edges[1] - edges[0]) * LY.hist_bar_width, color=(*ST.bar_colour, ST.bar_alpha * p_a))
            curve = np.linspace(-1, 1, LY.curve_samples)
            ph.plot(curve, angular_density(curve) * (edges[1] - edges[0]) * max(n_ev, 1), color=(*ST.curve_colour, p_a), lw=ST.curve_width * sc)
            ph.set_xlim(-1, 1)
            ph.set_ylim(0, max(LY.hist_min_ymax, LY.hist_ymax_factor * angular_density(1.0) * (edges[1] - edges[0]) * max(n_ev, 1) * LY.hist_ymax_margin))
            for name, sp in ph.spines.items():
                sp.set_visible(name in ("left", "bottom"))
                sp.set_color((*ST.hist_axis, ST.hist_axis_alpha * p_a))
            text_col = (*ST.hist_text, ST.hist_text_alpha * p_a)
            ph.set_xticks(list(LY.hist_ticks))
            ph.set_xticklabels([str(v).replace("-", "−") for v in LY.hist_ticks], color=text_col, fontsize=F.hist_tick * sc)
            ph.set_yticks([])
            ph.set_xlabel(tx["hist_xlabel"], color=text_col, fontsize=F.hist_xlabel * sc, labelpad=LY.hist_xlabel_pad)
            fig.text(*LY.hist_title_pos, tx["hist_title"], color=(*ST.hist_text, p_a), fontsize=F.hist_title * sc)
            fig.text(*LY.count_pos, put(tx["count"], n_events=str(n_ev)), color=(*ST.count_colour, p_a), fontsize=F.count * sc, family="monospace")
            fig.text(*LY.crossings_pos, put(tx["crossings"], n_crossings=str(k_cross + (1 if t >= crossings[0]["t_c"] else 0))),
                     color=(*ST.crossings_colour, ST.crossings_alpha * p_a), fontsize=F.crossings * sc)
            fig.text(*LY.legend_curve_pos, tx["legend_curve"], color=(*ST.curve_colour, ST.legend_curve_alpha * p_a), fontsize=F.legend_curve * sc)
            fig.text(*LY.legend_sigma_pos, tx["legend_sigma"], color=(*ST.legend_sigma_colour, ST.legend_sigma_alpha * p_a), fontsize=F.legend_sigma * sc)
        else:
            ph.axis("off")
        fig.text(*LY.note_pos, tx["note"], color=tuple(ST.note_colour), fontsize=F.note * sc)
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
    ap.add_argument("--seconds", type=float, default=FILM_LENGTH)
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--config", type=Path, default=None, help="another configuration file instead of config.toml")
    ap.add_argument("--set", action="append", default=[], metavar="SECTION.KEY=VALUE", help="override one configuration value")
    args = ap.parse_args()
    V = CFG.video
    out = args.out or HERE / "media" / f"scattering_experiment_{args.lang}.mp4"
    out.parent.mkdir(parents=True, exist_ok=True)
    if args.snapshot is not None:
        render(out.with_suffix(".png"), (V.width, V.height), V.fps, args.seconds, args.lang, snap=args.snapshot)
    elif args.preview:
        render(out.with_name(out.stem + "_preview.mp4"), (V.preview_width, V.preview_height), V.preview_fps, V.preview_seconds, args.lang)
    else:
        render(out, (V.width, V.height), V.fps, args.seconds, args.lang)


if __name__ == "__main__":
    main()
