r"""The decay pi0 -> e+ e- e+ e-: E and B of the two photons, the planes of the pairs, and what changes for a scalar.

A neutral pion decays into two photons and each virtual photon turns into an e+ e- pair (the "double
Dalitz" decay).  Every pair spans a plane that contains the polarization (the electric field E) of its
photon; B is perpendicular to the plane.  From one event to the next the planes have a different
orientation; the experiment measures the angle phi between them.

Pseudoscalar (pion, P = -1).  The effective interaction and the amplitude are

    L = (alpha / pi f_pi) pi0 E.B,       M  ~  E1.B2 + E2.B1  =  (m^2 / 2) khat . (e1 x e2),

so |M|^2 ~ sin^2 phi, and with the width Gamma = alpha^2 m^3 / (64 pi^3 f_pi^2) = 7.8 eV (the book's
formula).  For photons going back to back B_i = khat_i x E_i, hence E1.B2 = khat.(E1 x E2) ~ sin(phi):
perpendicular polarizations, perpendicular planes.

Scalar (P = +1), L = (g/4) S F F = (g/2) S (E^2 - B^2):

    M  ~  E1.E2 - B1.B2  =  2 E1.E2  ~  cos(phi),       Gamma = g^2 m^3 / (64 pi),

so the planes prefer to be parallel.

The rates dN/dphi ~ 1 -+ a cos(2 phi) are used with the illustrative a = 0.35 (it reproduces the height of
the humps of the KTeV histogram of the book, fig. 40.6): the real e+e- pair is a diluted analyzer of the
photon polarization, an ideal analyzer would give a = 1.  The azimuth of the first plane is uniform, the
second is rotated by phi drawn from the distribution (rejection sampling).

The picture is schematic: the opening angle of a pair is exaggerated and the events are accelerated.

All the numbers are in config.toml and all the words in texts.toml (see ../dvconfig.py for --config / --set).

Usage:
    python pi0_double_dalitz.py --lang en            # film -> media/pi0_double_dalitz_en.mp4
    python pi0_double_dalitz.py --lang ru
    python pi0_double_dalitz.py --lang en --preview
    python pi0_double_dalitz.py --lang en --snapshot 12   # one PNG at film time 12 s
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
A_AMP = CFG.model.a_amplitude
D_CONV = CFG.model.conversion_distance   # distance at which a photon converts (units of the scene)
Z_END = CFG.model.z_end
VEL = CFG.model.velocity                 # speed of the particles, scene units per second
N_BINS = CFG.model.bins
K_WAVE = CFG.model.wave_number           # wave number of the field arrows
T_B = CFG.timeline.scalar_start          # the scalar part starts
T_SUM = CFG.timeline.summary_start       # the summary starts
TOTAL = CFG.timeline.total
STRETCH = CFG.model.stretch              # the photon stage of the two slow (explanatory) events is slowed down by this factor

TEXT = {lang: {k: (tuple(v) if isinstance(v, list) else v) for k, v in load_texts(HERE, lang).items()} for lang in ("en", "ru")}


def density(phi: np.ndarray | float, a: float = A_AMP, mode: str = "ps") -> np.ndarray | float:
    """dN/dphi normalized on [0, 2 pi): (1 -+ a cos 2 phi) / (2 pi); minus for ps, plus for s."""
    sign = -1.0 if mode == "ps" else 1.0
    return (1.0 + sign * a * np.cos(2.0 * np.asarray(phi))) / (2.0 * math.pi)


def sample_phi(rng: np.random.Generator, n: int, a: float = A_AMP, mode: str = "ps") -> np.ndarray:
    sign = -1.0 if mode == "ps" else 1.0
    out: list[float] = []
    while len(out) < n:
        phi = rng.uniform(0, 2 * math.pi, 2 * n)
        u = rng.uniform(0, 1 + a, 2 * n)
        out.extend(phi[u < 1 + sign * a * np.cos(2 * phi)])
    return np.array(out[:n])


def amplitude_squared(phi: float, mode: str) -> float:
    """|E1.B2|^2 / max (ps) or (E1.E2 - B1.B2)^2 / max (s) for polarizations at the angle phi."""
    return math.sin(phi) ** 2 if mode == "ps" else math.cos(phi) ** 2


def field_vectors(phi_pol: float, direction: int) -> tuple[np.ndarray, np.ndarray]:
    """Unit E (along the polarization) and B = khat x E of a photon going along direction * z."""
    e = np.array([math.cos(phi_pol), math.sin(phi_pol), 0.0])
    k = np.array([0.0, 0.0, float(direction)])
    return e, np.cross(k, e)


def triple_product_check(phi: float) -> tuple[float, float]:
    """Returns (E1.B2 + E2.B1, E1.E2 - B1.B2) for back-to-back photons with polarizations 0 and phi."""
    e1, b1 = field_vectors(0.0, +1)
    e2, b2 = field_vectors(phi, -1)
    return float(e1 @ b2 + e2 @ b1), float(e1 @ e2 - b1 @ b2)


def make_events(seed: int = CFG.events.seed, total: float = TOTAL):
    rng = np.random.default_rng(seed)
    E_ = CFG.events
    events: list[dict] = []
    for mode, t_start, t_end, slow_t in (("ps", E_.ps_start, T_B - E_.ps_end_margin, E_.slow_event_slowness),
                                         ("s", T_B + E_.scalar_offset, T_SUM - E_.scalar_end_margin, E_.slow_event_slowness)):
        events.append(dict(t=t_start, slow=slow_t, mode=mode, stretch=STRETCH))
        t = t_start + E_.first_gap
        dt = E_.gap
        while t < t_end:
            events.append(dict(t=t, slow=E_.fast_slowness, mode=mode))
            t += dt
            dt = max(E_.gap_min, dt * E_.gap_decay)
    for mode in ("ps", "s"):
        idx = [i for i, e in enumerate(events) if e["mode"] == mode]
        phis = sample_phi(rng, len(idx), mode=mode)
        for i, phi in zip(idx, phis):
            events[i]["phi"] = float(phi)
            events[i]["azim1"] = float(rng.uniform(0, 2 * math.pi))
        events[idx[0]]["phi"] = 0.5 * math.pi + E_.first_ps_phi_offset if mode == "ps" else E_.first_s_phi
        events[idx[0]]["azim1"] = E_.first_ps_azimuth if mode == "ps" else E_.first_s_azimuth
    return events


def smooth(x: float, a: float, b: float) -> float:
    u = min(max((x - a) / (b - a), 0.0), 1.0)
    return u * u * (3 - 2 * u)



def tau_of(ev: dict, t: float) -> float:
    """Physical event time in units of the event slowness; the photon stage of slow events is stretched."""
    raw = (t - ev["t"]) / ev["slow"]
    s = ev.get("stretch", 1.0)
    tau_c = D_CONV / VEL
    if s == 1.0:
        return raw
    return raw / s if raw < s * tau_c else raw - (s - 1.0) * tau_c


def conversion_time(ev: dict) -> float:
    return ev["t"] + ev["slow"] * ev.get("stretch", 1.0) * D_CONV / VEL


def render(out: Path, size: tuple[int, int], fps: int, total: float, lang: str, snap: float | None = None) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.colors as mcolors
    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection, PolyCollection

    if snap is None and not shutil.which("ffmpeg"):
        sys.exit("ffmpeg is required")
    tx = TEXT[lang]
    V, ST, CA, PH, PR, TR, PT, DL, HI, SM, LY, EV = (CFG.video, CFG.style, CFG.camera, CFG.photons, CFG.pairs, CFG.triad, CFG.particle, CFG.dial,
                                                      CFG.hist, CFG.summary, CFG.layout, CFG.events)
    W, H = size
    dpi = V.dpi
    sc = H / V.reference_height
    k_t = total / TOTAL
    cx0, cy0 = CA.centre[0] * W, CA.centre[1] * H
    S = CA.scale * sc
    az, el = math.radians(CA.azimuth_deg), math.radians(CA.elevation_deg)

    def proj(x, y, z):
        x, y, z = np.asarray(x, float), np.asarray(y, float), np.asarray(z, float)
        X = x * math.cos(az) - y * math.sin(az)
        Y = z * math.cos(el) + (x * math.sin(az) + y * math.cos(az)) * math.sin(el)
        return cx0 + S * X, cy0 + S * Y

    events = make_events()
    for e in events:
        e["t"] *= k_t
        e["slow"] *= k_t
    t_conv = [conversion_time(e) for e in events]
    phis_ps = np.array([e["phi"] for e in events if e["mode"] == "ps"])
    phis_s = np.array([e["phi"] for e in events if e["mode"] == "s"])
    tc_ps = np.array([tc for tc, e in zip(t_conv, events) if e["mode"] == "ps"])
    tc_s = np.array([tc for tc, e in zip(t_conv, events) if e["mode"] == "s"])
    edges = np.linspace(0, 2 * math.pi, N_BINS + 1)
    centers = 0.5 * (edges[1:] + edges[:-1])
    bw = edges[1] - edges[0]

    BG = ST.background
    bg_rgba = np.array([int(BG[1:3], 16), int(BG[3:5], 16), int(BG[5:7], 16), 255], np.float32)
    fig = plt.figure(figsize=(W / dpi, H / dpi), dpi=dpi, facecolor=BG)
    ax = fig.add_axes([0, 0, 1, 1])
    dial = fig.add_axes(DL.axes, facecolor="none")
    hist = fig.add_axes(HI.axes, facecolor="none")
    rng_bg = np.random.default_rng(ST.star_seed)
    stars = rng_bg.uniform(0, 1, (ST.stars, 2))

    frames = int(round(total * fps))
    ids = [int(round(snap * fps))] if snap is not None else range(frames)
    writer = None
    if snap is None:
        writer = subprocess.Popen(
            ["ffmpeg", "-y", "-v", "error", "-f", "rawvideo", "-pix_fmt", "rgba", "-s", f"{W}x{H}",
             "-r", str(fps), "-i", "-", "-c:v", "libx264", "-preset", V.preset, "-crf", str(V.crf),
             "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(out)], stdin=subprocess.PIPE)

    def glow(X, Y, color, d, alpha=1.0, z=8):
        X = np.atleast_1d(X)
        Y = np.atleast_1d(Y)
        al = np.broadcast_to(np.atleast_1d(alpha), X.shape)
        for scale, a_ in ST.glow_layers:
            ax.scatter(X, Y, s=(d * scale * 72 / dpi) ** 2, linewidths=0, zorder=z,
                       facecolors=[mcolors.to_rgba(color, a_ * float(x)) for x in al])

    col_pos, col_neg, col_p1, col_p2 = ST.positron, ST.electron, ST.photon1, ST.photon2
    col_B = tuple(ST.b_field)
    TXT = tuple(ST.text)
    DIM = tuple(ST.dim)

    def arrow2d(x0, y0, x1, y1, color, lw, z=12, ls="-"):
        ax.annotate("", xy=(x1, y1), xytext=(x0, y0),
                    arrowprops=dict(arrowstyle="-|>", color=color, lw=lw, mutation_scale=ST.arrow_mutation * sc, linestyle=ls), zorder=z)

    for k_ in ids:
        t = k_ / fps
        fig.texts.clear()
        ax.clear()
        ax.set_xlim(0, W)
        ax.set_ylim(0, H)
        ax.axis("off")
        ax.add_patch(matplotlib.patches.Rectangle((0, 0), W, H, color=BG, zorder=0, lw=0))
        summary = t >= T_SUM * k_t
        mode_now = "ps" if t < T_B * k_t else "s"
        ax.scatter(stars[:, 0] * W, stars[:, 1] * H, s=ST.star_size * sc ** 2, c=[tuple(ST.star_colour)] * len(stars), linewidths=0, zorder=1)
        # the axis of the photons, faint
        xa, ya = proj([0, 0], [0, 0], [-CA.axis_half_length, CA.axis_half_length])
        if not summary:
            ax.plot(xa, ya, color=(*ST.guide, CA.axis_alpha), lw=CA.axis_width * sc, ls=(0, tuple(CA.axis_dash)), zorder=2)
        polys, pcols = [], []
        lines, lcols, lws = [], [], []
        tipx, tipy, tipc = [], [], []
        last = None
        for ev in events:
            slow = ev["slow"]
            tau = tau_of(ev, t)
            if tau < 0 or tau > EV.duration_limit or summary:
                continue
            last = ev
            fade = min(smooth(tau, *EV.fade_in), 1 - smooth(tau, *EV.fade_out))
            phi1, phi2 = ev["azim1"], ev["azim1"] + ev["phi"]
            tau_c = D_CONV / VEL
            slow_event = ev.get("stretch", 1.0) > 1.0
            # photons: E and B arrows along the path
            if slow_event and EV.photon_start < tau < tau_c + EV.photon_start:
                for sgn, phi, colE in ((1, phi1, col_p1), (-1, phi2, col_p2)):
                    Lp = min(VEL * (tau - EV.photon_start), D_CONV)
                    zz = np.arange(PH.first_arrow, Lp, PH.arrow_step_slow if slow_event else PH.arrow_step_fast)
                    if len(zz) == 0:
                        continue
                    amp = (PH.amplitude_slow if slow_event else PH.amplitude_fast) * np.sin(K_WAVE * (zz - VEL * tau))
                    e, b = field_vectors(phi, sgn)
                    X0, Y0 = proj(0 * zz, 0 * zz, sgn * zz)
                    XE, YE = proj(amp * e[0], amp * e[1], sgn * zz)
                    XB, YB = proj(PH.b_factor * amp * b[0], PH.b_factor * amp * b[1], sgn * zz)
                    zf = np.linspace(0.0, Lp, PH.field_line_samples)
                    ampf = PH.field_line_amplitude * np.sin(K_WAVE * (zf - VEL * tau)) * smooth(Lp, 0.0, PH.field_line_ramp)
                    for ampv, vec, colr in ((1.0, e, mcolors.to_rgba(colE, PH.field_line_alpha_e * fade)), (PH.b_factor, b, (*col_B, PH.field_line_alpha_b * fade))):
                        XC, YC = proj(ampf * ampv * vec[0], ampf * ampv * vec[1], sgn * zf)
                        lines.append(np.column_stack([XC, YC]))
                        lcols.append(colr)
                        lws.append(PH.field_line_width * sc)
                    for xs_, ys_, colr, wl in ((XE, YE, mcolors.to_rgba(colE, PH.arrow_alpha_e * fade), PH.arrow_width_e), (XB, YB, (*col_B, PH.arrow_alpha_b * fade), PH.arrow_width_b)):
                        for i in range(len(zz)):
                            lines.append(np.array([[X0[i], Y0[i]], [xs_[i], ys_[i]]]))
                            lcols.append(colr)
                            lws.append(wl * sc)
                        tipx.extend(xs_)
                        tipy.extend(ys_)
                        tipc.extend([colr] * len(zz))
                    # the photon line itself
                    XL, YL = proj([0.0, 0.0], [0.0, 0.0], sgn * np.array([0.0, Lp]))
                    lines.append(np.column_stack([XL, YL]))
                    lcols.append((*ST.photon_line, PH.line_alpha * fade))
                    lws.append(PH.line_width * sc)
            elif tau >= EV.photon_start:
                for sgn in (1, -1):
                    XL, YL = proj([0, 0], [0, 0], sgn * np.array([0.0, min(VEL * (tau - EV.photon_start), D_CONV)]))
                    lines.append(np.column_stack([XL, YL]))
                    lcols.append((*ST.photon_line, PH.line_alpha * fade))
                    lws.append(PH.line_width * sc)
            if tau >= tau_c:
                f2 = smooth(tau, tau_c, tau_c + PR.conversion_ramp) * fade
                for sgn, phi, col in ((1, phi1, col_p1), (-1, phi2, col_p2)):
                    u = np.array([math.cos(phi), math.sin(phi)])
                    z0, z1, w = D_CONV * PR.plane_start_factor, Z_END, PR.plane_half_width
                    corners = []
                    for (zz, ww) in ((z0, -w), (z0, w), (z1, w), (z1, -w)):
                        X, Y = proj(ww * u[0], ww * u[1], sgn * zz)
                        corners.append((float(X), float(Y)))
                    polys.append(corners)
                    pcols.append(mcolors.to_rgba(col, PR.plane_alpha * f2))
                    run = VEL * (tau - tau_c)
                    for (s_t, colr) in ((+1, col_pos), (-1, col_neg)):
                        ang = PR.opening_angle
                        zpos = D_CONV + run * math.cos(ang)
                        tr = run * math.sin(ang) * s_t
                        X, Y = proj(tr * u[0], tr * u[1], sgn * zpos)
                        X0, Y0 = proj(0 * u[0], 0 * u[1], sgn * D_CONV)
                        lines.append(np.array([[float(X0), float(Y0)], [float(X), float(Y)]]))
                        lcols.append(mcolors.to_rgba(colr, PR.track_alpha * fade))
                        lws.append(PR.track_width * sc)
                        glow(X, Y, colr, PR.glow_size * sc, fade, z=9)
        if polys:
            ax.add_collection(PolyCollection(polys, facecolors=pcols, edgecolors=[(1, 1, 1, PR.plane_edge_alpha)] * len(polys), linewidths=PR.plane_edge_width * sc, zorder=3))
        if lines:
            ax.add_collection(LineCollection(lines, colors=lcols, linewidths=lws, zorder=4, capstyle="round"))
        if tipx:
            ax.scatter(tipx, tipy, s=(PH.tip_size * sc * 72 / dpi) ** 2, c=tipc, linewidths=0, zorder=5)

        # triads E_i, B_i at the photons of the latest event (as in the book figure)
        if last is not None and not summary:
            tau = tau_of(last, t)
            if tau >= D_CONV / VEL + TR.appear[0]:
                fade = min(smooth(tau, D_CONV / VEL + TR.appear[0], D_CONV / VEL + TR.appear[1]), 1 - smooth(tau, *EV.fade_out))
                phi1, phi2 = last["azim1"], last["azim1"] + last["phi"]
                for i, (sgn, phi, colE, lab) in enumerate(((1, phi1, col_p1, "1"), (-1, phi2, col_p2, "2"))):
                    e, b = field_vectors(phi, sgn)
                    zc = sgn * TR.z
                    X0, Y0 = proj(0, 0, zc)
                    XE, YE = proj(TR.length * e[0], TR.length * e[1], zc)
                    XB, YB = proj(TR.length * b[0], TR.length * b[1], zc)
                    arrow2d(float(X0), float(Y0), float(XE), float(YE), mcolors.to_rgba(colE, fade), TR.e_width * sc)
                    arrow2d(float(X0), float(Y0), float(XB), float(YB), (*col_B, fade), TR.b_width * sc)
                    for (xx, yy), nm, colr in (((XE, YE), "E", mcolors.to_rgba(colE, fade)), ((XB, YB), "B", (*col_B, fade))):
                        dx = float(xx - X0)
                        dy = float(yy - Y0)
                        nrm = math.hypot(dx, dy) or 1.0
                        ax.text(float(xx) + TR.label_offset * sc * dx / nrm, float(yy) + TR.label_offset * sc * dy / nrm, rf"$\mathbf{{{nm}}}_{lab}$",
                                color=colr, fontsize=TR.label_size * sc, ha="center", va="center", zorder=13)

        # particle and the decay flash
        x0, y0 = proj(0, 0, 0)
        if not summary:
            pc = ST.pion if mode_now == "ps" else ST.scalar
            glow(float(x0), float(y0), pc, PT.glow_size * sc, 1.0, z=10)
            ax.text(float(x0) + PT.label_offset[0] * sc, float(y0) + PT.label_offset[1] * sc, r"$\pi^0$" if mode_now == "ps" else r"$S$",
                    color=tuple(PT.label_color), fontsize=PT.label_size * sc, zorder=12)
            for ev in events:
                tau = tau_of(ev, t)
                if 0 <= tau < PT.flash_duration:
                    f = 1 - tau / PT.flash_duration
                    glow(float(x0), float(y0), ST.flash, (PT.flash_size_base + PT.flash_size_gain * (1 - f)) * sc, PT.flash_alpha * f, z=11)

        if summary:
            f_in = smooth(t, T_SUM * k_t, T_SUM * k_t + CFG.timeline.summary_fade_s * k_t)
            for cxp, mode_s, phi_s, colh in ((SM.dials_x[0] * W, "ps", 0.5 * math.pi, tuple(HI.curve_colour_ps)), (SM.dials_x[1] * W, "s", 0.0, tuple(HI.curve_colour_s))):
                cyp, Rp = SM.dial_y * H, SM.dial_radius * H
                aa = np.linspace(0, 2 * math.pi, HI.curve_samples)
                ax.plot(cxp + Rp * np.cos(aa), cyp + Rp * np.sin(aa), color=(*ST.guide, SM.circle_alpha * f_in), lw=1.0 * sc, zorder=5)
                ph1 = 0.5 * math.pi - 0.5 * phi_s + SM.rotation_offset
                for ph, colE, sg, nm in ((ph1, col_p1, +1, "1"), (ph1 + phi_s, col_p2, -1, "2")):
                    ph_tip = ph + (math.pi if (mode_s == "s" and nm == "2") else 0.0)
                    ux, uy = math.cos(ph_tip), math.sin(ph_tip)
                    ax.plot([cxp - Rp * ux, cxp + Rp * ux], [cyp - Rp * uy, cyp + Rp * uy], color=mcolors.to_rgba(colE, SM.plane_alpha_ps * f_in if mode_s == "ps" else SM.plane_alpha_s * f_in), lw=SM.plane_width * sc, zorder=6, solid_capstyle="round")
                    arrow2d(cxp + SM.arrow_start * Rp * ux, cyp + SM.arrow_start * Rp * uy, cxp + SM.arrow_end * Rp * ux, cyp + SM.arrow_end * Rp * uy, mcolors.to_rgba(colE, f_in), SM.arrow_width * sc, z=7)
                    ax.text(cxp + SM.e_label_radius * Rp * ux, cyp + SM.e_label_radius * Rp * uy, rf"$\mathbf{{E}}_{nm}$", color=mcolors.to_rgba(colE, f_in), fontsize=SM.e_label_size * sc, ha="center", va="center", zorder=8)
                    e_, b_ = field_vectors(ph, sg)
                    arrow2d(cxp, cyp, cxp + SM.b_length * Rp * b_[0], cyp + SM.b_length * Rp * b_[1], (*col_B, SM.b_alpha * f_in), SM.b_width * sc, z=7, ls=(0, tuple(SM.b_dash)))
                    ax.text(cxp + SM.b_label_radius * Rp * b_[0], cyp + SM.b_label_radius * Rp * b_[1], rf"$\mathbf{{B}}_{nm}$", color=(*col_B, f_in), fontsize=SM.b_label_size * sc, ha="center", va="center", zorder=8)
                title = tx["leg_ps"] if mode_s == "ps" else tx["leg_s"]
                fig.text(cxp / W, SM.title_y, title, color=(*colh, f_in), fontsize=SM.title_size * sc, ha="center")
                fml, words = tx["sum_ps"] if mode_s == "ps" else tx["sum_s"]
                fig.text(cxp / W, SM.formula_y, fml, color=(*colh, f_in), fontsize=SM.formula_size * sc, ha="center", va="top")
                fig.text(cxp / W, SM.words_y, words, color=(*colh, f_in), fontsize=SM.words_size * sc, ha="center", va="top")
        # the dial: end view along the photon axis
        dial.clear()
        dial.set_facecolor("none")
        dial.set_xlim(-DL.lim, DL.lim)
        dial.set_ylim(-DL.lim, DL.lim)
        dial.set_aspect("equal")
        dial.axis("off")
        a_ = np.linspace(0, 2 * math.pi, HI.curve_samples)
        if not summary:
            dial.plot(np.cos(a_), np.sin(a_), color=(*ST.guide, DL.circle_alpha), lw=DL.circle_width * sc)
        readout = None
        if last is not None and not summary:
            phi1, phi2 = last["azim1"], last["azim1"] + last["phi"]
            for ph, col in ((phi1, col_p1), (phi2, col_p2)):
                dial.plot([-math.cos(ph), math.cos(ph)], [-math.sin(ph), math.sin(ph)], color=col, lw=DL.plane_width * sc, alpha=DL.plane_alpha, solid_capstyle="round")
                dial.annotate("", xy=(DL.arrow_end * math.cos(ph), DL.arrow_end * math.sin(ph)), xytext=(DL.arrow_start * math.cos(ph), DL.arrow_start * math.sin(ph)),
                              arrowprops=dict(arrowstyle="-|>", color=col, lw=DL.arrow_width * sc))
            for i_, (ph, col, sg) in enumerate(((phi1, col_p1, +1), (phi2, col_p2, -1))):
                e_, b_ = field_vectors(ph, sg)
                dial.annotate("", xy=(DL.b_length * b_[0], DL.b_length * b_[1]), xytext=(0, 0),
                              arrowprops=dict(arrowstyle="-|>", color=col_B, lw=DL.b_width * sc, linestyle=(0, tuple(DL.b_dash))))
                dial.text(DL.label_radius * math.cos(ph), DL.label_radius * math.sin(ph), rf"$\mathbf{{E}}_{i_ + 1}$", color=col, fontsize=DL.label_size * sc, ha="center", va="center")
            arc = np.linspace(phi1, phi2, DL.arc_points)
            dial.plot(DL.arc_radius * np.cos(arc), DL.arc_radius * np.sin(arc), color=(1, 1, 1, DL.arc_alpha), lw=DL.arc_width * sc)
            ph_deg = math.degrees(last["phi"]) % 360
            dial.text(0, DL.angle_pos, rf"$\varphi={ph_deg:.0f}^\circ$", color=tuple(DL.angle_color), fontsize=DL.angle_size * sc, ha="center", va="top")
            readout = amplitude_squared(last["phi"], last["mode"])
        if not summary:
            fig.text(*DL.title_pos, tx["dial"], color=tuple(DL.title_color), fontsize=DL.title_size * sc)
            if readout is not None:
                fig.text(*DL.readout_pos, tx["read_ps" if last["mode"] == "ps" else "read_s"] + (f"{readout:.2f}".replace(".", "{,}") if lang == "ru" else f"{readout:.2f}") + "$",
                         color=tuple(DL.readout_color), fontsize=DL.readout_size * sc)

        # the histogram
        hist.clear()
        hist.set_facecolor("none")
        grid = np.linspace(0, 2 * math.pi, HI.curve_samples)

        def bars(phis, tcs, mode, colr, offset=0.0, width_frac=HI.bar_width):
            done = phis[tcs <= t]
            counts, _ = np.histogram(done, bins=edges)
            hist.bar(centers / math.pi + offset, counts, width=bw / math.pi * width_frac, color=colr)
            return len(done), counts.max() if len(done) else 0

        n_ps = int((tc_ps <= t).sum())
        n_s = int((tc_s <= t).sum())
        if summary:
            f_in = smooth(t, T_SUM * k_t, T_SUM * k_t + CFG.timeline.summary_fade_s * k_t)
            n1, m1 = bars(phis_ps, np.array([0.0] * len(phis_ps)), "ps", (*HI.curve_colour_ps, SM.hist_bar_alpha * f_in), -bw / (4 * math.pi), SM.hist_bar_width)
            n2, m2 = bars(phis_s, np.array([0.0] * len(phis_s)), "s", (*HI.curve_colour_s, SM.hist_bar_alpha * f_in), bw / (4 * math.pi), SM.hist_bar_width)
            scale = max(n1, n2)
            hist.plot(grid / math.pi, density(grid, mode="ps") * bw * n1, color=(*HI.curve_colour_ps, f_in), lw=HI.curve_width * sc)
            hist.plot(grid / math.pi, density(grid, mode="s") * bw * n2, color=(*HI.curve_colour_s, f_in), lw=HI.curve_width * sc)
            top = SM.hist_top_factor * (1 + A_AMP) / (2 * math.pi) * bw * scale + HI.top_offset
        else:
            if mode_now == "ps":
                n, m = bars(phis_ps, tc_ps, "ps", (*HI.bar_colour_ps, HI.bar_alpha))
                hist.plot(grid / math.pi, density(grid, mode="ps") * bw * max(n, 1), color=(*HI.curve_colour_ps, HI.curve_alpha), lw=HI.curve_width * sc)
            else:
                n, m = bars(phis_s, tc_s, "s", (*HI.bar_colour_s, HI.bar_alpha))
                hist.plot(grid / math.pi, density(grid, mode="s") * bw * max(n, 1), color=(*HI.curve_colour_s, HI.curve_alpha), lw=HI.curve_width * sc)
            top = max(HI.top_min, HI.top_factor * (1 + A_AMP) / (2 * math.pi) * bw * max(n, 1) + HI.top_offset)
        hist.set_xlim(0, 2)
        hist.set_ylim(0, top)
        for name, sp in hist.spines.items():
            sp.set_visible(name in ("left", "bottom"))
            sp.set_color((*HI.spine_colour, HI.spine_alpha))
        hist.set_xticks(list(HI.ticks))
        hist.set_xticklabels([format(x, "g").replace(".", ",") if lang == "ru" else format(x, "g") for x in HI.ticks], color=tuple(HI.tick_colour), fontsize=HI.tick_size * sc)
        hist.set_yticks([])
        hist.set_xlabel(r"$\varphi/\pi$", color=tuple(HI.xlabel_colour), fontsize=HI.xlabel_size * sc, labelpad=0)
        if summary:
            fig.text(*SM.hist_title_pos, tx["sum_title"], color=TXT, fontsize=SM.hist_title_size * sc)
            fig.text(*SM.legend_ps_pos, "—  " + tx["leg_ps"], color=(*HI.curve_colour_ps, HI.curve_alpha), fontsize=SM.legend_size * sc)
            fig.text(*SM.legend_s_pos, "—  " + tx["leg_s"], color=(*HI.curve_colour_s, HI.curve_alpha), fontsize=SM.legend_size * sc)
        else:
            fig.text(*HI.count_pos, rf"{tx['events']}: ${n}$", color=tuple(HI.count_colour), fontsize=HI.count_size * sc)
            curve = tx["curve_ps"] if mode_now == "ps" else tx["curve_s"]
            fig.text(*HI.legend_pos, "—  " + curve, color=(*HI.curve_colour_ps, HI.curve_alpha) if mode_now == "ps" else (*HI.curve_colour_s, HI.curve_alpha), fontsize=HI.legend_size * sc)
            fig.text(*HI.note_pos, tx["a_note"], color=tuple(HI.note_colour), fontsize=HI.note_size * sc)

        # titles and formulas
        if summary:
            fig.text(*LY.title_pos, tx["title_summary"], color=tuple(ST.title_color), fontsize=LY.summary_title_size * sc)
        else:
            fig.text(*LY.title_pos, tx["title_ps"] if mode_now == "ps" else tx["title_s"], color=tuple(ST.title_color), fontsize=LY.title_size * sc)
            fig.text(LY.title_pos[0] + (LY.subtitle_x_ps if mode_now == "ps" else LY.subtitle_x_s), LY.subtitle_y, tx["sub_ps"] if mode_now == "ps" else tx["sub_s"],
                     color=DIM, fontsize=LY.subtitle_size * sc)
            fig.text(*LY.formula_pos, tx["ps_formula"] if mode_now == "ps" else tx["s_formula"], color=tuple(ST.formula_color), fontsize=LY.formula_size * sc)
        fig.text(LY.key_e1_x, LY.keys_y, tx["key_e1"], color=col_p1, fontsize=LY.keys_size * sc)
        fig.text(LY.key_e2_x, LY.keys_y, tx["key_e2"], color=col_p2, fontsize=LY.keys_size * sc)
        fig.text(LY.key_b_x, LY.keys_y, tx["key_b"], color=col_B, fontsize=LY.keys_size * sc)

        # captions
        if summary:
            cap = tx["cap5"]
        elif mode_now == "ps":
            if t < events[1]["t"] - CFG.timeline.caption_before_second:
                cap = tx["cap1"] if t < conversion_time(events[0]) + CFG.timeline.caption_after_conversion * k_t else tx["cap2"]
            else:
                cap = tx["cap3"]
        else:
            cap = tx["cap4"] if t < [e for e in events if e["mode"] == "s"][1]["t"] - CFG.timeline.caption_before_second else tx["cap3"]
        ax.add_patch(matplotlib.patches.Rectangle((0, 0), LY.caption_box[0] * W, LY.caption_box[1] * H, color=tuple(LY.caption_box_colour), zorder=13, lw=0))
        fig.text(*LY.caption_pos, cap[0], color=tuple(ST.caption_color), fontsize=LY.caption_size * sc)
        fade_io = min(smooth(t, 0.0, V.fade_s), 1.0 - smooth(t, total - V.fade_s, total))
        fig.canvas.draw()
        frame = np.asarray(fig.canvas.buffer_rgba()).astype(np.float32)
        if fade_io < 1.0:
            frame = bg_rgba + (frame - bg_rgba) * fade_io
        frame = frame.clip(0, 255).astype(np.uint8)
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
    ap.add_argument("--seconds", type=float, default=TOTAL)
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--config", type=Path, default=None, help="another configuration file instead of config.toml")
    ap.add_argument("--set", action="append", default=[], metavar="SECTION.KEY=VALUE", help="override one configuration value")
    args = ap.parse_args()
    V = CFG.video
    out = args.out or HERE / "media" / f"pi0_double_dalitz_{args.lang}.mp4"
    out.parent.mkdir(parents=True, exist_ok=True)
    if args.snapshot is not None:
        render(out.with_suffix(".png"), (V.width, V.height), V.fps, args.seconds, args.lang, snap=args.snapshot)
    elif args.preview:
        render(out.with_name(out.stem + "_preview.mp4"), (V.preview_width, V.preview_height), V.preview_fps, args.seconds, args.lang)
    else:
        render(out, (V.width, V.height), V.fps, args.seconds, args.lang)


if __name__ == "__main__":
    main()
