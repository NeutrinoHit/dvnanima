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

All the numbers are in config.toml and all the words in texts.toml (see ../dvconfig.py for --config / --set).

Usage:
    python hulse_taylor.py --lang en                # film -> media/hulse_taylor_en.mp4
    python hulse_taylor.py --lang ru
    python hulse_taylor.py --lang en --preview
    python hulse_taylor.py --lang en --snapshot 20  # one PNG at film time 20 s
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

_M = CFG.model
G_M_SUN = _M.gm_sun                          # m^3 s^-2
P_B = _M.period_days * _M.day_s              # s
ECC = _M.eccentricity
M1, M2 = _M.mass_pulsar, _M.mass_companion   # solar masses
PDOT_GR = _M.pdot_gr
PDOT_INTR = _M.pdot_intrinsic
PDOT_OBS = _M.pdot_observed
YEAR = _M.year_days * _M.day_s
Y0, Y1 = _M.year_start, _M.year_end
SHRINK_VIEW = _M.shrink_view                 # drawn relative shrink of the orbit over 30 years
PHASE_EXAG = _M.phase_exaggeration           # exaggeration of the phase shift
KEPLER_ITERATIONS = _M.kepler_iterations
FILM_LENGTH = CFG.timeline.film_length


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
    for _ in range(KEPLER_ITERATIONS):
        E = E - (E - e * np.sin(E) - m) / (1 - e * np.cos(E))
    return E


def relative_orbit(mean_anomaly: np.ndarray, a: float = 1.0, e: float = ECC) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Relative position (x, y) with the periastron on the +x axis, and the separation."""
    E = kepler(mean_anomaly, e)
    x = a * (np.cos(E) - e)
    y = a * math.sqrt(1 - e * e) * np.sin(E)
    return x, y, a * (1 - e * np.cos(E))


# ---------------------------------------------------------------------- film

TEXT = {lang: load_texts(HERE, lang) for lang in ("en", "ru")}     # texts.toml


def num(x: float, fmt: str, lang: str) -> str:
    """A number for use in a text: decimal comma in Russian (also inside $...$, as {,})."""
    s = format(x, fmt)
    return s.replace(".", "{,}") if lang == "ru" else s


def put(text: str, **values: str) -> str:
    """Replace the placeholders {name} of a text (LaTeX braces stay untouched)."""
    for k, v in values.items():
        text = text.replace("{" + k + "}", v)
    return text


def smooth(x: float, a: float, b: float) -> float:
    t = min(max((x - a) / (b - a), 0.0), 1.0)
    return t * t * (3 - 2 * t)


def timeline(total: float) -> dict[str, tuple[float, float]]:
    T = CFG.timeline
    k = total / FILM_LENGTH
    return dict(waves=(T.waves[0] * k, T.waves[1] * k), panel=(T.panel[0] * k, T.panel[1] * k), years=(T.years[0] * k, T.years[1] * k))


def year_at(t: float, tl: dict[str, tuple[float, float]]) -> float:
    a, b = tl["years"]
    u = min(max((t - a) / (b - a), 0.0), 1.0)
    return Y0 + (Y1 - Y0) * u


def render(out: Path, size: tuple[int, int], fps: int, total: float, lang: str = "en", snap: float | None = None) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.colors as mcolors
    import matplotlib.pyplot as plt

    if snap is None and not shutil.which("ffmpeg"):
        sys.exit("ffmpeg is required")
    V, LY, WV, F, ST, TLN, FM = CFG.video, CFG.layout, CFG.waves, CFG.fonts, CFG.style, CFG.timeline, CFG.formats
    tx = TEXT[lang]
    W, H = size
    dpi = V.dpi
    sc = H / V.reference_height
    BG = ST.background
    tl = timeline(total)
    cx0, cy0 = LY.orbit_centre[0] * W, LY.orbit_centre[1] * H
    A = LY.semi_major_px * sc               # semi-major axis in pixels
    p_vis = TLN.orbit_period                # film seconds per orbit
    m_tot = M1 + M2
    f1, f2 = M2 / m_tot, M1 / m_tot         # star 1 (pulsar) moves on the smaller circle
    rng = np.random.default_rng(LY.star_seed)
    # stars in the background
    star_x = rng.uniform(0, W, LY.star_count)
    star_y = rng.uniform(0, H, LY.star_count)
    star_a = rng.uniform(LY.star_alpha[0], LY.star_alpha[1], LY.star_count)
    # field grid for the gravitational waves (low resolution, upsampled)
    cell = LY.wave_cell_px
    gw = int(LY.wave_width * W / cell)
    gh = int(H / cell)
    gx = (np.arange(gw) * cell + cell // 2) - (cx0 - 0.0)
    gy = (np.arange(gh) * cell + cell // 2) - cy0
    GX, GY = np.meshgrid(gx, gy[::-1] * 1.0)
    rho = np.hypot(GX, GY)
    psi = np.arctan2(GY, GX)
    edge = np.clip(np.minimum(np.minimum(np.arange(gw)[None, :], gw - 1 - np.arange(gw)[None, :]),
                              np.minimum(np.arange(gh)[:, None], gh - 1 - np.arange(gh)[:, None])) / LY.wave_edge_cells, 0.0, 1.0)
    edge = edge * edge * (3 - 2 * edge)
    c_vis = LY.wave_speed_px * sc / p_vis * LY.wave_speed_factor       # pixels per film second

    fig = plt.figure(figsize=(W / dpi, H / dpi), dpi=dpi, facecolor=BG)
    ax = fig.add_axes([0, 0, 1, 1])
    pa = fig.add_axes(LY.panel_axes, facecolor="none")
    cmap = mcolors.LinearSegmentedColormap.from_list("gw", list(WV.colours))

    frames = int(round(total * fps))
    ids = [int(round(snap * fps))] if snap is not None else range(frames)
    writer = None
    if snap is None:
        writer = subprocess.Popen(
            ["ffmpeg", "-y", "-v", "error", "-f", "rawvideo", "-pix_fmt", "rgba", "-s", f"{W}x{H}",
             "-r", str(fps), "-i", "-", "-c:v", "libx264", "-preset", V.preset, "-crf", str(V.crf),
             "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(out)], stdin=subprocess.PIPE)

    def glow(X, Y, color, d, alpha=1.0, z=6):
        for scale, al in ST.glow_layers:
            ax.scatter([X], [Y], s=(d * scale * 72 / dpi) ** 2, c=[mcolors.to_rgba(color, al * alpha)],
                       linewidths=0, zorder=z)

    ya_all = np.linspace(Y0, Y1, _M.shift_samples)
    shift_gr = periastron_shift(ya_all - Y0)
    shift_intr = periastron_shift(ya_all - Y0, PDOT_INTR)
    COL_1, COL_2 = ST.colour_pulsar, ST.colour_companion
    PT, PTA = tuple(ST.panel_text), ST.panel_text_alpha            # colour and opacity of the texts of the panel
    PS = tuple(ST.panel_spine)
    # the numbers of the model that appear in the texts
    fills = dict(m1=num(M1, FM.mass, lang), m2=num(M2, FM.mass, lang), period_h=num(P_B / _M.hour_s, FM.period_hours, lang),
                 ecc=num(ECC, FM.eccentricity, lang), years=format(Y1 - Y0, FM.years), ratio=num(_M.pdot_ratio, FM.ratio, lang),
                 error=num(_M.pdot_ratio_error, FM.ratio, lang))

    def say(key: str) -> str:
        return put(tx[key], **fills)

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
        ax.add_patch(matplotlib.patches.Rectangle((0, 0), W, H, color=BG, zorder=0, lw=0))
        ax.scatter(star_x, star_y, s=ST.star_size * sc ** 2 * ST.star_area, c=[(*ST.star_colour, a_) for a_ in star_a], linewidths=0, zorder=1)
        # gravitational waves: rotating quadrupole, retarded phase, amplitude follows the separation
        w_alpha = smooth(t, *tl["waves"])
        if w_alpha > ST.visible_alpha:
            t_ret = t - rho / c_vis
            M_ret = 2 * math.pi * t_ret / p_vis + dphase
            xr, yr, sep = relative_orbit(M_ret, 1.0)
            ang = np.arctan2(yr, xr)
            amp = WV.amp_base + WV.amp_peak * ((1.0 - ECC) / sep) ** WV.amp_exponent
            field = WV.gain * amp * np.cos(2 * (psi - ang)) / np.sqrt(1.0 + rho / (WV.decay_px * sc))
            field *= np.clip((rho - WV.inner_start_px * sc) / (WV.inner_width_px * sc), 0, 1)
            field *= np.exp(-rho / (WV.outer_px * sc)) * edge
            img = cmap(0.5 + 0.5 * np.clip(field * WV.colour_gain, -1, 1))
            img[..., 3] = np.clip(np.abs(field) * WV.alpha_gain, 0, WV.alpha_max) * w_alpha
            half = cell / 2
            ax.imshow(img, extent=(cx0 + gx[0] - half, cx0 + gx[-1] + half, cy0 - gy[0] - half, cy0 - gy[-1] + half), origin="upper",
                      interpolation="bicubic", zorder=2, aspect="auto")
        # full orbit (dashed) of both stars
        Mo = np.linspace(0, 2 * math.pi, LY.orbit_samples)
        xo, yo, _ = relative_orbit(Mo, a_now)
        dash = (0, tuple(ST.orbit_dash))
        ax.plot(cx0 + f1 * xo, cy0 + f1 * yo, color=COL_1, lw=ST.orbit_width * sc, alpha=ST.orbit_alpha, ls=dash, zorder=3)
        ax.plot(cx0 - f2 * xo, cy0 - f2 * yo, color=COL_2, lw=ST.orbit_width * sc, alpha=ST.orbit_alpha, ls=dash, zorder=3)
        # the stars
        xs, ys, sep = relative_orbit(np.array([phase + dphase]), a_now)
        p1 = (cx0 + f1 * xs[0], cy0 + f1 * ys[0])
        p2 = (cx0 - f2 * xs[0], cy0 - f2 * ys[0])
        # constant-period model (ghost) when the shift is visible
        if dphase != 0.0 and abs(dphase) > ST.ghost_threshold:
            xg, yg, _ = relative_orbit(np.array([phase]), A)
            g_alpha = min(1.0, abs(dphase) / ST.ghost_full)
            for (gxp, gyp, col) in ((cx0 + f1 * xg[0], cy0 + f1 * yg[0], COL_1), (cx0 - f2 * xg[0], cy0 - f2 * yg[0], COL_2)):
                ax.scatter([gxp], [gyp], s=(ST.ghost_diameter * sc * 72 / dpi) ** 2, facecolors="none",
                           edgecolors=[mcolors.to_rgba(col, ST.ghost_alpha * g_alpha)], linewidths=ST.ghost_width * sc, zorder=5)
        # trails
        n_tr = LY.trail_samples
        for frac, col, (px, py) in ((f1, COL_1, p1), (-f2, COL_2, p2)):
            Mt = phase + dphase - np.linspace(0, LY.trail_length, n_tr)
            xt, yt, _ = relative_orbit(Mt, a_now)
            tx_ = cx0 + frac * xt
            ty_ = cy0 + frac * yt
            for j in range(n_tr - 1):
                ax.plot(tx_[j:j + 2], ty_[j:j + 2], color=col, lw=ST.trail_width * sc, alpha=ST.trail_alpha * (1 - j / (n_tr - 1)) ** ST.trail_power,
                        zorder=4, solid_capstyle="round")
        glow(p1[0], p1[1], COL_1, ST.star_diameter_pulsar * sc)
        glow(p2[0], p2[1], COL_2, ST.star_diameter_companion * sc)
        # labels of the stars
        lab_a = smooth(t, *TLN.label_fade_in) * (1 - smooth(t, *TLN.label_fade_out))
        if lab_a > ST.visible_alpha:
            o1, o2 = LY.label_pulsar_offset, LY.label_companion_offset
            ax.annotate(say("label_pulsar"), (p1[0], p1[1]), xytext=(p1[0] + o1[0] * sc, p1[1] + o1[1] * sc), color=(*ST.label_pulsar, lab_a),
                        fontsize=F.label * sc, arrowprops=dict(arrowstyle="-", color=(*ST.label_pulsar, ST.label_line_alpha * lab_a), lw=ST.label_line_width * sc), zorder=9)
            ax.annotate(say("label_companion"), (p2[0], p2[1]), xytext=(p2[0] + o2[0] * sc, p2[1] + o2[1] * sc), color=(*ST.label_companion, lab_a),
                        fontsize=F.label * sc, arrowprops=dict(arrowstyle="-", color=(*ST.label_companion, ST.label_line_alpha * lab_a), lw=ST.label_line_width * sc), zorder=9)
        # graph
        pa.clear()
        pa.set_facecolor("none")
        pn = smooth(t, *tl["panel"])
        fig.texts.clear()
        if pn > ST.visible_alpha:
            pa.set_xlim(Y0, Y1)
            pa.set_ylim(*LY.panel_ylim)
            for name, sp in pa.spines.items():
                sp.set_visible(name in ("left", "bottom"))
                sp.set_color((*PS, ST.panel_spine_alpha * pn))
            pa.axhline(0, color=(*PS, ST.panel_zero_alpha * pn), lw=ST.panel_zero_width * sc, ls=(0, tuple(ST.panel_zero_dash)))
            m = ya_all <= year
            if t >= tl["years"][0] - TLN.start_margin_s:
                pa.fill_between(ya_all[m], shift_gr[m], shift_intr[m], color=(*ST.fill_colour, ST.fill_alpha * pn))
                pa.plot(ya_all[m], shift_gr[m], color=(*ST.curve_gr, pn), lw=ST.curve_gr_width * sc, solid_capstyle="round")
                pa.plot(ya_all[m], shift_intr[m], color=(*ST.curve_intrinsic, pn), lw=ST.curve_intrinsic_width * sc, ls=(0, tuple(ST.curve_dash)))
                now = float(periastron_shift(dyr))
                pa.scatter([year], [now], s=(ST.marker_size * sc) ** 2 * ST.marker_area, c=[(1, 1, 1, pn)], zorder=5, linewidths=0)
                pa.scatter([year], [now], s=(ST.marker_glow_size * sc) ** 2 * ST.marker_area, c=[(*ST.curve_gr, ST.marker_glow_alpha * pn)], zorder=4, linewidths=0)
            else:
                now = 0.0
            tick_col = (*PT, PTA * pn)
            pa.set_xticks(list(LY.year_ticks))
            pa.set_xticklabels([str(v) for v in LY.year_ticks], color=tick_col, fontsize=F.tick * sc)
            pa.set_yticks(list(LY.shift_ticks))
            pa.set_yticklabels([str(v).replace("-", "−") for v in LY.shift_ticks], color=tick_col, fontsize=F.tick * sc)
            pa.tick_params(colors=(*PS, ST.panel_tick_alpha * pn), length=LY.tick_length)
            pa.set_xlabel(tx["axis_year"], color=tick_col, fontsize=F.axis_label * sc, labelpad=LY.xlabel_pad)
            fig.text(*LY.panel_title_pos, tx["panel_title"], color=(*PT, pn), fontsize=F.panel_title * sc)
            now_s = format(now, FM.shift)
            fig.text(*LY.shift_value_pos, f"{(now_s.replace('.', ',') if lang == 'ru' else now_s).replace('-', '−')} {tx['unit_s']}", color=(*ST.shift_value_colour, pn),
                     fontsize=F.shift_value * sc, family="monospace")
            fig.text(*LY.year_pos, format(year, FM.year), color=(*ST.year_colour, pn), fontsize=F.year * sc, family="monospace")
            fig.text(*LY.legend_gr_pos, tx["legend_gr"], color=(*ST.curve_gr, ST.legend_alpha * pn), fontsize=F.legend * sc)
            fig.text(*LY.legend_pdot_pos, tx["legend_pdot"], color=(*ST.curve_intrinsic, ST.legend_alpha * pn), fontsize=F.legend * sc)
            fig.text(*LY.ratio_pos, say("ratio"), color=(*ST.ratio_colour, ST.ratio_alpha * pn), fontsize=F.ratio * sc)
        else:
            pa.axis("off")
        # captions
        if t < tl["panel"][0]:
            key = "cap_intro"
        elif t < tl["years"][0]:
            key = "cap_waves"
        elif year < Y1 - TLN.end_margin_years:
            key = "cap_shrink"
        else:
            key = "cap_done"
        ax.add_patch(matplotlib.patches.Rectangle((0, 0), LY.caption_box[0] * W, LY.caption_box[1] * H, color=tuple(ST.caption_box_colour), zorder=11, lw=0))
        fig.text(*LY.caption_pos, say(key), color=tuple(ST.caption_colour), fontsize=F.caption * sc)
        fig.text(*LY.note_pos, say("note"), color=tuple(ST.note_colour), fontsize=F.note * sc)
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
    out = args.out or HERE / "media" / f"hulse_taylor_{args.lang}.mp4"
    out.parent.mkdir(parents=True, exist_ok=True)
    if args.snapshot is not None:
        render(out.with_suffix(".png"), (V.width, V.height), V.fps, args.seconds, args.lang, snap=args.snapshot)
    elif args.preview:
        render(out.with_name(out.stem + "_preview.mp4"), (V.preview_width, V.preview_height), V.preview_fps, V.preview_seconds, args.lang)
    else:
        render(out, (V.width, V.height), V.fps, args.seconds, args.lang)


if __name__ == "__main__":
    main()
