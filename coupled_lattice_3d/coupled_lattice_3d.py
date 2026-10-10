r"""A 3D lattice of coupled oscillators (the 3D mattress of the book): from the normal modes to the field.

Chapters (the same as in ../coupled_chain and ../coupled_lattice_2d):
 1. the box lattice of NX x NY x NZ masses joined by springs (fixed boundary), the Lagrangian and the equation of motion;
 2. the normal modes  q ~ sin(a pi i/(NX+1)) sin(b pi j/(NY+1)) sin(c pi k/(NZ+1)) cos(omega_abc t); the spectrum of the
    mode energies E(omega) and the dispersion omega(|k|) show which modes take part;
 3. a bump is released at the centre: a spherical wave, many modes, constant mode energies;
 4. two wave packets collide in the linear lattice and pass through each other (free particles);
 5. the same collision with the anharmonic terms (cubic and quartic): the energy is redistributed among the normal
    modes - the interaction;
 6. N -> infinity: the lattice is refined (14x8x8, 28x16x16, 56x32x32 masses, ever smaller dots) and becomes the field
    phi(x, y, z, t), omega = c |k|.

The masses are drawn as dots whose size and brightness are the displacement (a quiet mass is a tiny dim dot),
the colour is the sign.

All the numbers are in config.toml and all the words in texts.toml (see ../dvconfig.py for --config / --set).

Usage:
    python coupled_lattice_3d.py --lang en            # film -> media/coupled_lattice_3d_en.mp4
    python coupled_lattice_3d.py --lang ru
    python coupled_lattice_3d.py --lang en --snapshot 70
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

import lattice3d_physics as P  # noqa: E402

HERE = Path(__file__).resolve().parent
CFG = load_config(HERE)
NX, NY, NZ = P.NX, P.NY, P.NZ

CARD_S = CFG.timeline.card_s
CARD_AT = list(CFG.timeline.card_at)
CARD_KEYS = [f"h{i}" for i in range(len(CARD_AT))]
_PB = list(CFG.timeline.part_bounds)
CONTENT_TOTAL = _PB[-1]
TOTAL = CONTENT_TOTAL + CARD_S * len(CARD_AT)

T_INTRO, T_MODES, T_PLUCK, T_LIN, T_NONLIN, T_CONT = ((_PB[i], _PB[i + 1]) for i in range(6))

MODES = [((m["a"] or NX, m["b"] or NY, m["c"] or NZ), m["from"], m["to"], m["caption"]) for m in CFG.modes.list]
PERIOD_111 = CFG.modes.period_s                            # film seconds per period of the mode (1, 1, 1)
PHYS_PER_S = (2 * math.pi / PERIOD_111) / P.omega(1, 1, 1)
WAVE_RATE = CFG.physics.wave_rate
MODE_AMP = CFG.modes.amplitude
QREF = CFG.modes.reference_displacement                    # displacement of a full-size dot


def smooth(x: float, a: float, b: float) -> float:
    u = min(max((x - a) / (b - a), 0.0), 1.0)
    return u * u * (3 - 2 * u)


def timeline(tf: float) -> tuple[float, int | None, float]:
    done = 0
    for i, ca in enumerate(CARD_AT):
        a = ca + i * CARD_S
        if a <= tf < a + CARD_S:
            return ca, i, (tf - a) / CARD_S
        if tf >= a + CARD_S:
            done += 1
    return tf - done * CARD_S, None, 0.0


def film_time(tc: float) -> float:
    return tc + CARD_S * sum(1 for ca in CARD_AT if ca <= tc)


def chapter_of(tc: float) -> int:
    return max(i for i, ca in enumerate(CARD_AT) if ca <= tc)


def num(x: float, fmt: str, lang: str) -> str:
    s = format(x, fmt)
    return s.replace(".", "{,}") if lang == "ru" else s


# ------------------------------------------------------------------- the field of the continuum part

LXF, LYF, LZF = NX + 1.0, NY + 1.0, NZ + 1.0
NXF, NYF, NZF = CFG.limit.fine_grid                        # a fine grid with the spacing 1/4 of a site
_cache: dict = {}


def _fine_sines(n: int) -> np.ndarray:
    return np.sqrt(2.0 / (n + 1)) * np.sin(np.outer(np.arange(1, n + 1), np.arange(1, n + 1)) * math.pi / (n + 1))


def fine_field(t: float) -> np.ndarray:
    """phi(x, y, z, t) of the continuum limit (omega = c |k|, fixed boundary) for the bump released at rest."""
    if "fine" not in _cache:
        sx, sy, sz = _fine_sines(NXF), _fine_sines(NYF), _fine_sines(NZF)
        x = np.arange(1, NXF + 1) * LXF / (NXF + 1)
        y = np.arange(1, NYF + 1) * LYF / (NYF + 1)
        z = np.arange(1, NZF + 1) * LZF / (NZF + 1)
        i0, j0, k0, gx, gy, gz, amp = P.BUMP
        phi0 = amp * (np.exp(-((x - i0) ** 2) / (2 * gx ** 2))[:, None, None] * np.exp(-((y - j0) ** 2) / (2 * gy ** 2))[None, :, None]
                      * np.exp(-((z - k0) ** 2) / (2 * gz ** 2))[None, None, :])
        om = math.pi * np.sqrt((np.arange(1, NXF + 1)[:, None, None] / LXF) ** 2 + (np.arange(1, NYF + 1)[None, :, None] / LYF) ** 2
                               + (np.arange(1, NZF + 1)[None, None, :] / LZF) ** 2)
        c0 = np.einsum("ai,bj,ck,ijk->abc", sx, sy, sz, phi0, optimize=True)
        _cache["fine"] = (sx, sy, sz, c0, om)
    sx, sy, sz, c0, om = _cache["fine"]
    return np.einsum("ai,bj,ck,abc->ijk", sx, sy, sz, c0 * np.cos(om * t), optimize=True)


def sample_fine(phi: np.ndarray, nx: int, ny: int, nz: int) -> np.ndarray:
    xi = np.clip(np.round(np.arange(1, nx + 1) * (NXF + 1) / (nx + 1)).astype(int) - 1, 0, NXF - 1)
    yi = np.clip(np.round(np.arange(1, ny + 1) * (NYF + 1) / (ny + 1)).astype(int) - 1, 0, NYF - 1)
    zi = np.clip(np.round(np.arange(1, nz + 1) * (NZF + 1) / (nz + 1)).astype(int) - 1, 0, NZF - 1)
    return phi[np.ix_(xi, yi, zi)]


# -------------------------------------------------------------------------------- texts (texts.toml)

TEXT = {lang: load_texts(HERE, lang) for lang in ("en", "ru")}


# ----------------------------------------------------------------------------------- the film state

def nonlinear_run() -> tuple[np.ndarray, np.ndarray]:
    if "nl" not in _cache:
        q0, v0 = P.collision_initial()
        rate = CFG.physics.nonlinear_samples_per_s
        n = int(round((T_NONLIN[1] - T_NONLIN[0]) * rate)) + 1
        _cache["nl"] = P.integrate(q0, v0, P.ALPHA, P.BETA, np.arange(n) / float(rate) * WAVE_RATE, dt=CFG.physics.integrator_dt)
    return _cache["nl"]


def bump_initial() -> np.ndarray:
    return P.gauss3(*P.BUMP)


NB = CFG.spectrum.bins                                     # bins of the spectrum


def spectrum(e: np.ndarray) -> np.ndarray:
    """Energy of the modes in the bins of omega (the shares add up to 1)."""
    h, _ = np.histogram(P.OMEGA.ravel(), bins=NB, range=(0.0, P.OMEGA_MAX), weights=e.ravel())
    return h / max(float(e.sum()), 1e-12)


def bin_of(w: float) -> int:
    return int(min(NB - 1, w / P.OMEGA_MAX * NB))


def ymax_of(kind: str) -> float:
    if ("ym", kind) not in _cache:
        if kind == "pluck":
            vals = [spectrum(P.mode_energies(bump_initial(), np.zeros((NX, NY, NZ)))).max()]
        elif kind == "lin":
            vals = [spectrum(P.mode_energies(*P.collision_initial())).max()]
        else:
            Q, V = nonlinear_run()
            s = CFG.spectrum.ymax_stride
            vals = [spectrum(P.mode_energies(q, v)).max() for q, v in zip(Q[::s], V[::s])]
        _cache[("ym", kind)] = CFG.spectrum.ymax_margin * max(vals)
    return _cache[("ym", kind)]


def state(t: float) -> dict:
    I, M, PL, L, N, C = CFG.intro, CFG.modes, CFG.pluck, None, CFG.nonlinear, CFG.limit
    st = {"q": np.zeros((NX, NY, NZ)), "E": None, "active": None, "tag": None, "ymax": 1.0, "formulas": [], "cap": "c0",
          "chain_a": smooth(t, *I.lattice_appear), "panels": 0.0, "kind": "intro", "cont": None, "delta": None}
    if t < T_MODES[0]:
        st["formulas"] = [("fL1", smooth(t, *I.formula1_in) * (1 - smooth(t, *I.formula1_out)), 0), ("fEq", smooth(t, *I.formula2_in), 0)]
        st["cap"] = "c0" if t < I.caption_switch else "c1"
        return st
    st["panels"] = smooth(t, T_MODES[0], T_MODES[0] + M.panels_fade_s)
    if t < T_MODES[1]:
        st["kind"] = "modes"
        for m_, a, b, cap in MODES:
            if a <= t < b:
                w = smooth(t, a, a + M.fade_s) * (1.0 - smooth(t, b - M.fade_s, b))
                ph = (t - a) * PHYS_PER_S
                st["q"] = MODE_AMP * w * P.mode_shape(*m_) * math.cos(P.omega(*m_) * ph)
                E = np.zeros((NX, NY, NZ))
                E[m_[0] - 1, m_[1] - 1, m_[2] - 1] = 1.0
                st["E"], st["active"], st["wmode"], st["ymax"] = E, m_, w, 1.0
                st["cap"] = cap
        st["formulas"] = [("fM1", 1.0, 0), ("fM2", 1.0, 1)]
        return st
    if t < T_PLUCK[1]:
        st["kind"] = "pluck"
        q, v = P.linear_state(bump_initial(), np.zeros((NX, NY, NZ)), (t - T_PLUCK[0]) * WAVE_RATE)
        st["q"], st["E"], st["ymax"] = q, P.mode_energies(q, v), ymax_of("pluck")
        st["cap"] = "cp" if t < T_PLUCK[0] + PL.caption_switch else "cp2"
        st["formulas"] = [("fP", 1.0, 0)]
        return st
    if t < T_LIN[1]:
        st["kind"] = "lin"
        q, v = P.linear_state(*P.collision_initial(), (t - T_LIN[0]) * WAVE_RATE)
        st["q"], st["E"], st["ymax"] = q, P.mode_energies(q, v), ymax_of("lin")
        st["delta"], st["tag"], st["cap"] = 0.0, "lin", "cl"
        st["formulas"] = [("fEq", 1.0, 0)]
        return st
    if t < T_NONLIN[1]:
        st["kind"] = "nonlin"
        Q, V = nonlinear_run()
        i = int(min(round((t - T_NONLIN[0]) * CFG.physics.nonlinear_samples_per_s), len(Q) - 1))
        E = P.mode_energies(Q[i], V[i])
        st["q"], st["E"], st["ymax"] = Q[i], E, ymax_of("nonlin")
        e0 = P.mode_energies(*P.collision_initial())
        st["delta"] = 0.5 * float(np.abs(P.fractions(E) - P.fractions(e0)).sum())
        st["tag"] = "nonlin"
        st["cap"] = "cn" if t < T_NONLIN[0] + N.caption_switch else "cn2"
        st["formulas"] = [("fN", 1.0, 0)]
        return st
    st["kind"] = "cont"
    st["panels"] = 1.0
    st["formulas"] = [("fC", smooth(t, T_CONT[0] + C.formula_in[0], T_CONT[0] + C.formula_in[1]), 0)]
    st["cap"] = "cc" if t < T_CONT[0] + C.caption_switch else "cc2"
    levels = [tuple(lv) for lv in C.levels]
    k = int(min(len(levels) - 1, (t - T_CONT[0]) // C.level_s))
    st["cont"] = (levels[k], (t - T_CONT[0]) * C.time_rate + C.time_offset,
                  smooth(t, T_CONT[0] + C.label_switch[0], T_CONT[0] + C.label_switch[1]))
    return st


# ----------------------------------------------------------------------------------- the picture

def render(out: Path, size: tuple[int, int], fps: int, total: float, lang: str, snap: float | None = None) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection

    if snap is None and not shutil.which("ffmpeg"):
        sys.exit("ffmpeg is required")
    tx = TEXT[lang]
    V, S, LY, F, CM, SP, DS, LM, CD = CFG.video, CFG.style, CFG.layout, CFG.fonts, CFG.camera, CFG.spectrum, CFG.dispersion, CFG.limit, CFG.modes
    W, H = size
    dpi = V.dpi
    sc = H / V.reference_height
    BG = S.background
    TXT, DIM = tuple(S.text), tuple(S.dim)
    BLUE, WARM, GOLD = np.array(S.blue), np.array(S.warm), tuple(S.gold)
    MID = np.array(S.mid)
    fig = plt.figure(figsize=(W / dpi, H / dpi), dpi=dpi, facecolor=BG)
    ax = fig.add_axes(LY.lattice_axes, facecolor="none")
    axm = fig.add_axes(LY.spectrum_axes, facecolor="none")
    axd = fig.add_axes(LY.dispersion_axes, facecolor="none")
    k = total / TOTAL
    frames = int(round(total * fps))
    ids = [int(round(snap * fps))] if snap is not None else range(frames)
    writer = None
    if snap is None:
        writer = subprocess.Popen(
            ["ffmpeg", "-y", "-v", "error", "-f", "rawvideo", "-pix_fmt", "rgba", "-s", f"{W}x{H}",
             "-r", str(fps), "-i", "-", "-c:v", "libx264", "-preset", V.preset, "-crf", str(V.crf),
             "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(out)], stdin=subprocess.PIPE)

    AZ, EL = math.radians(CM.azimuth_deg), math.radians(CM.elevation_deg)
    CX, CY, CZ = 0.5 * (NX + 1), 0.5 * (NY + 1), 0.5 * (NZ + 1)
    bg_rgba = np.array([int(BG[1:3], 16), int(BG[3:5], 16), int(BG[5:7], 16), 255], np.float32)

    def proj(x, y, z):
        x, y, z = np.asarray(x, float) - CX, np.asarray(y, float) - CY, np.asarray(z, float) - CZ
        xr = x * math.cos(AZ) - y * math.sin(AZ)
        yr = x * math.sin(AZ) + y * math.cos(AZ)
        return xr, yr * math.sin(EL) + z * math.cos(EL), yr

    def signed_colors(q: np.ndarray, qref: float) -> np.ndarray:
        a = np.clip(q / qref, -1, 1)[..., None]
        return np.where(a >= 0, MID * (1 - a) + WARM * a, MID * (1 + a) + BLUE * (-a))

    def draw_cube(alpha: float, lx: float = NX + 1.0, ly: float = NY + 1.0, lz: float = NZ + 1.0):
        c = [(x, y, z) for x in (0, lx) for y in (0, ly) for z in (0, lz)]
        edges = [(a, b) for a in range(8) for b in range(a + 1, 8) if sum(u != v for u, v in zip(c[a], c[b])) == 1]
        segs = []
        for a, b in edges:
            X, Ys, _ = proj([c[a][0], c[b][0]], [c[a][1], c[b][1]], [c[a][2], c[b][2]])
            segs.append(np.column_stack([X, Ys]))
        ax.add_collection(LineCollection(segs, colors=[(*S.cube_color, S.cube_alpha * alpha)] * len(segs), linewidths=S.cube_width * sc, zorder=3))

    def draw_cloud(Xg, Yg, Zg, q, qref, alpha, appear, lines: bool, dot_scale: float):
        """Masses as dots (size and brightness = displacement, colour = sign) and thin springs."""
        zs = Zg + S.dot_shift * np.clip(q / qref, -S.dot_shift_clip, S.dot_shift_clip)
        Xp, Yp, Dp = proj(Xg, Yg, zs)
        if lines:
            segs = []
            for ax_ in range(3):
                sl_a = [slice(None)] * 3
                sl_b = [slice(None)] * 3
                sl_a[ax_], sl_b[ax_] = slice(0, -1), slice(1, None)
                a_, b_ = tuple(sl_a), tuple(sl_b)
                segs.append(np.stack([np.stack([Xp[a_], Yp[a_]], -1), np.stack([Xp[b_], Yp[b_]], -1)], axis=-2).reshape(-1, 2, 2))
            segs = np.concatenate(segs)
            ax.add_collection(LineCollection(segs, colors=[(*S.spring_color, S.spring_alpha * alpha)] * len(segs), linewidths=S.spring_width * sc, zorder=2))
        mag = np.clip(np.abs(q) / qref, 0.0, 1.0)
        col = signed_colors(q, qref)
        sizes = (dot_scale * (S.dot_base + S.dot_gain * mag)) * sc * sc
        a_d = alpha * appear * (S.dot_alpha_base + S.dot_alpha_gain * mag)
        order = np.argsort(-Dp.reshape(-1))
        rgba = np.column_stack([col.reshape(-1, 3), a_d.reshape(-1)])
        ax.scatter(Xp.reshape(-1)[order], Yp.reshape(-1)[order], s=sizes.reshape(-1)[order], c=rgba[order], edgecolors="none", zorder=5)

    for k_ in ids:
        t_film = k_ / fps
        tf_nom = t_film / k
        t, card, cprog = timeline(tf_nom)
        st = state(t)
        if card is not None:
            st.update({"q": np.zeros((NX, NY, NZ)), "E": None, "active": None, "formulas": [], "cap": None, "chain_a": 0.0,
                       "panels": 0.0, "kind": "intro", "tag": None, "delta": None})
        fig.texts.clear()
        for ar in (ax, axm, axd):
            ar.clear()
            ar.set_facecolor("none")
            ar.axis("off")
        ax.set_xlim(*CM.xlim)
        ax.set_ylim(*CM.ylim)
        if card is None:
            fig.text(*LY.title_pos, tx["title"], color=tuple(S.title_color), fontsize=F.title * sc)
            ch = chapter_of(t)
            if ch >= 1:
                fig.text(*LY.chapter_label_pos, f"{ch}/{len(CARD_AT) - 1}   " + tx[CARD_KEYS[ch]], color=(*DIM, S.chapter_label_alpha),
                         fontsize=F.chapter_label * sc, ha="right")
        else:
            a_c = min(smooth(cprog, *CFG.timeline.card_fade_in), 1.0 - smooth(cprog, *CFG.timeline.card_fade_out))
            if card == 0:
                fig.text(0.5, LY.card_title_y, tx["h0"], color=(*S.card_title_color, a_c), fontsize=F.card_title * sc, ha="center", va="center")
                fig.text(0.5, LY.card_subtitle_y, tx["h0s"], color=(*DIM, a_c), fontsize=F.card_subtitle * sc, ha="center", va="center")
            else:
                fig.text(0.5, LY.card_number_y, f"{card}", color=(*GOLD, S.card_number_alpha * a_c), fontsize=F.card_number * sc, ha="center", va="center")
                fig.text(0.5, LY.card_chapter_y, tx[CARD_KEYS[card]], color=(*S.card_title_color, a_c), fontsize=F.card_chapter * sc, ha="center", va="center")
                fig.text(0.5, LY.card_chapter_sub_y, tx[CARD_KEYS[card] + "s"], color=(*DIM, a_c), fontsize=F.card_chapter_sub * sc, ha="center", va="center")
        for key, alpha, row in st["formulas"]:
            if alpha > LY.formula_min_alpha:
                fig.text(LY.formula_pos[0], LY.formula_pos[1] - LY.formula_row * row, tx[key], color=(*TXT, S.formula_alpha * alpha), fontsize=F.formula * sc)
        if st["cap"]:
            fig.text(*LY.caption_pos, tx[st["cap"]], color=(*TXT, S.caption_alpha), fontsize=F.caption * sc)
        if st["kind"] == "modes" and st.get("active") is not None:
            a_, b_, c_ = st["active"]
            fig.text(*LY.mode_label_pos, f"{tx['mode']} $(a,b,c)=({a_},{b_},{c_})$,  $\\omega={num(P.omega(a_, b_, c_), '.2f', lang)}\\,\\sqrt{{k/m}}$",
                     color=(*GOLD, st.get("wmode", 1.0)), fontsize=F.mode_label * sc)
        # ---------------- the lattice
        a_ch = st["chain_a"]
        if st["kind"] != "cont":
            if a_ch > S.lattice_cutoff:
                Xg, Yg, Zg = np.meshgrid(P.IX.astype(float), P.IY.astype(float), P.IZ.astype(float), indexing="ij")
                appear = np.clip((a_ch * (NX + CFG.intro.sweep_lead) - Xg) / CFG.intro.sweep_width, 0.0, 1.0)
                draw_cloud(Xg, Yg, Zg, st["q"], QREF, 1.0, appear, True, S.dot_size)
                draw_cube(a_ch)
            if st["tag"]:
                fig.text(*LY.tag_pos, tx[st["tag"]], color=(*(BLUE if st["tag"] == "lin" else WARM), 1.0), fontsize=F.tag * sc, ha="left", va="top")
        else:
            lev, tc, a_field = st["cont"]
            draw_cube(1.0)
            phi_f = fine_field(tc)
            nx, ny, nz = lev
            Xg, Yg, Zg = np.meshgrid(np.arange(1, nx + 1) * LXF / (nx + 1), np.arange(1, ny + 1) * LYF / (ny + 1),
                                     np.arange(1, nz + 1) * LZF / (nz + 1), indexing="ij")
            phi = sample_fine(phi_f, nx, ny, nz)
            ref = LM.reference_n / nx
            draw_cloud(Xg, Yg, Zg, phi, LM.reference_displacement, ref ** LM.dot_alpha_exponent, np.ones_like(phi), nx <= LM.lines_max_n,
                       S.dot_size * ref ** LM.dot_size_exponent)
            fig.text(*LY.n_label_pos, rf"$N={nx}\times{ny}\times{nz}$" if a_field < LM.label_threshold else r"$N\to\infty$", color=(*TXT, 1.0),
                     fontsize=F.n_label * sc, ha="right", va="top")
        # ---------------- the spectrum and the dispersion
        pa = st["panels"]
        if pa > S.panel_cutoff and st["kind"] != "cont":
            axm.axis("on")
            for sp in axm.spines.values():
                sp.set_visible(False)
            axm.set_xticks([])
            axm.set_yticks([])
            E = st["E"] if st["E"] is not None else np.zeros((NX, NY, NZ))
            sp_ = spectrum(E)
            edges = np.linspace(0.0, P.OMEGA_MAX, NB + 1)
            colb = [(*BLUE, SP.bar_alpha * pa)] * NB
            if st.get("active") is not None:
                colb[bin_of(P.omega(*st["active"]))] = (*GOLD, SP.active_alpha * pa)
            axm.bar(0.5 * (edges[:-1] + edges[1:]), sp_, width=SP.bar_width * P.OMEGA_MAX / NB, color=colb, lw=0)
            axm.set_xlim(0, P.OMEGA_MAX)
            axm.set_ylim(0, st["ymax"])
            axm.plot([0, P.OMEGA_MAX], [0, 0], color=(1, 1, 1, SP.baseline_alpha * pa), lw=SP.baseline_width * sc)
            axm.text(*SP.title_pos, tx["E"], color=(*DIM, pa), fontsize=SP.title_size * sc, transform=axm.transAxes, ha="left", va="bottom")
            axm.text(*SP.xlabel_pos, r"$\omega$", color=(*DIM, pa), fontsize=SP.xlabel_size * sc, transform=axm.transAxes, ha="right", va="top")
            if st["delta"] is not None:
                fig.text(*SP.delta_pos, tx["delta"] + rf" ${100 * st['delta']:.0f}\,\%$", color=(*TXT, pa), fontsize=SP.delta_size * sc,
                         ha="left", va="bottom", linespacing=SP.delta_linespacing)
        if pa > S.panel_cutoff:
            axd.axis("on")
            for sp in axd.spines.values():
                sp.set_visible(False)
            axd.set_xticks([])
            axd.set_yticks([])
            kmax = float(P.KMAG.max()) * DS.k_margin
            axd.set_xlim(0, kmax)
            axd.set_ylim(0, DS.y_max)
            kk = np.linspace(0, kmax, DS.line_samples)
            if st["kind"] == "cont":
                a_d = smooth(t, T_CONT[0] + LM.dispersion_switch[0], T_CONT[0] + LM.dispersion_switch[1])
                axd.scatter(P.KMAG.ravel(), P.OMEGA.ravel(), s=DS.limit_dot_size * sc * sc,
                            c=[(*BLUE, DS.limit_dot_alpha * (1 - DS.limit_dot_fade * a_d))], edgecolors="none")
                axd.plot(kk, kk, color=(*BLUE, a_d), lw=DS.limit_line_width * sc, ls=(0, tuple(DS.limit_line_dash)))
                axd.text(*DS.limit_label_pos, tx["l_cont"], color=(*BLUE, a_d), fontsize=DS.limit_label_size * sc, ha="left")
                axd.text(kmax + DS.lattice_label_offset[0], DS.lattice_label_offset[1], tx["l_latt"], color=(*DIM, a_d),
                         fontsize=DS.limit_label_size * sc, ha="right")
            else:
                if st["E"] is not None and st["E"].max() > 0:
                    fr = (st["E"] / st["E"].max()).ravel()
                else:
                    fr = np.zeros(NX * NY * NZ)
                sizes = (DS.dot_size_base + DS.dot_size_gain * fr) * sc * sc
                colors = [(*BLUE, pa * (DS.dot_alpha_base + DS.dot_alpha_gain * f)) for f in fr]
                if st.get("active") is not None:
                    a_, b_, c_ = st["active"]
                    idx = ((a_ - 1) * NY + (b_ - 1)) * NZ + (c_ - 1)
                    colors[idx] = (*GOLD, pa)
                    sizes[idx] = DS.active_size * sc * sc
                axd.scatter(P.KMAG.ravel(), P.OMEGA.ravel(), s=sizes, c=colors, edgecolors="none", zorder=4)
                axd.plot(kk, kk, color=(1, 1, 1, DS.cone_alpha * pa), lw=DS.cone_width * sc, ls=(0, tuple(DS.cone_dash)))
            axd.plot([0, kmax], [0, 0], color=(1, 1, 1, DS.baseline_alpha * pa), lw=DS.baseline_width * sc)
            axd.text(*DS.title_pos, tx["disp"], color=(*DIM, pa), fontsize=DS.title_size * sc, transform=axd.transAxes, ha="left", va="bottom")
            axd.text(*DS.xlabel_pos, r"$|\mathbf{k}|$", color=(*DIM, pa), fontsize=DS.xlabel_size * sc, transform=axd.transAxes, ha="right", va="top")
        fade_io = min(smooth(t_film, 0.0, V.fade_s), 1.0 - smooth(t_film, total - V.fade_s, total))
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
    out = args.out or HERE / "media" / f"coupled_lattice_3d_{args.lang}.mp4"
    out.parent.mkdir(parents=True, exist_ok=True)
    if args.snapshot is not None:
        render(out.with_suffix(".png"), (V.width, V.height), V.fps, args.seconds, args.lang, snap=args.snapshot)
    elif args.preview:
        render(out.with_name(out.stem + "_preview.mp4"), (V.preview_width, V.preview_height), V.preview_fps, args.seconds, args.lang)
    else:
        render(out, (V.width, V.height), V.fps, args.seconds, args.lang)


if __name__ == "__main__":
    main()
