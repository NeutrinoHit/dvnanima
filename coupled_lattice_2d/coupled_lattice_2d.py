r"""A 2D lattice of coupled oscillators (the mattress of the book): from the normal modes to the field.

Chapters (the same as in ../coupled_chain):
 1. the lattice of NX x NY masses joined by springs (fixed edge), the Lagrangian and the equation of motion;
 2. the normal modes  q_ij ~ sin(a pi i/(NX+1)) sin(b pi j/(NY+1)) cos(omega_ab t); the map of the mode energies E_ab
    and the dispersion omega(|k|) show which modes take part;
 3. a bump is released: circular waves, many modes, constant mode energies;
 4. two wave packets collide in the linear lattice and pass through each other (free particles);
 5. the same collision with the anharmonic terms (cubic and quartic) of the Lagrangian: the energy is redistributed
    among the normal modes - the interaction;
 6. N -> infinity: the lattice becomes the field phi(x, y, t), omega = c |k|.

All the numbers are in config.toml and all the words in texts.toml (see ../dvconfig.py for --config / --set).

Usage:
    python coupled_lattice_2d.py --lang en            # film -> media/coupled_lattice_2d_en.mp4
    python coupled_lattice_2d.py --lang ru
    python coupled_lattice_2d.py --lang en --snapshot 60
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

import lattice2d_physics as P  # noqa: E402

HERE = Path(__file__).resolve().parent
CFG = load_config(HERE)
NX, NY = P.NX, P.NY

CARD_S = CFG.timeline.card_s
CARD_AT = list(CFG.timeline.card_at)
CARD_KEYS = [f"h{i}" for i in range(len(CARD_AT))]
_PB = list(CFG.timeline.part_bounds)
CONTENT_TOTAL = _PB[-1]
TOTAL = CONTENT_TOTAL + CARD_S * len(CARD_AT)

T_INTRO, T_MODES, T_PLUCK, T_LIN, T_NONLIN, T_CONT = ((_PB[i], _PB[i + 1]) for i in range(6))

MODES = [((m["a"] or NX, m["b"] or NY), m["from"], m["to"], m["caption"]) for m in CFG.modes.list]
PERIOD_11 = CFG.modes.period_s                             # film seconds per period of the mode (1, 1)
PHYS_PER_S = (2 * math.pi / PERIOD_11) / P.omega(1, 1)
WAVE_RATE = CFG.physics.wave_rate
MODE_AMP = CFG.modes.amplitude                             # the peak height of a mode is about 1.1 sites


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

LXF, LYF = NX + 1.0, NY + 1.0
NXF, NYF = CFG.limit.fine_grid                             # a fine grid with the spacing 1/8 of a site
_cache: dict = {}


def fine_field(t: float) -> np.ndarray:
    """phi(x, y, t) of the continuum limit (omega = c |k|, fixed edge) for the bump released at rest."""
    if "fine" not in _cache:
        ix = np.arange(1, NXF + 1)
        iy = np.arange(1, NYF + 1)
        sx = np.sqrt(2.0 / (NXF + 1)) * np.sin(np.outer(np.arange(1, NXF + 1), ix) * math.pi / (NXF + 1))
        sy = np.sqrt(2.0 / (NYF + 1)) * np.sin(np.outer(np.arange(1, NYF + 1), iy) * math.pi / (NYF + 1))
        x = ix * LXF / (NXF + 1)
        y = iy * LYF / (NYF + 1)
        i0, j0, gx, gy, amp = P.BUMP
        phi0 = amp * np.exp(-((x[:, None] - i0) ** 2) / (2 * gx ** 2) - ((y[None, :] - j0) ** 2) / (2 * gy ** 2))
        om = math.pi * np.sqrt((np.arange(1, NXF + 1)[:, None] / LXF) ** 2 + (np.arange(1, NYF + 1)[None, :] / LYF) ** 2)
        _cache["fine"] = (sx, sy, sx @ phi0 @ sy.T, om)
    sx, sy, c0, om = _cache["fine"]
    return sx.T @ (c0 * np.cos(om * t)) @ sy


def sample_fine(phi: np.ndarray, nx: int, ny: int) -> np.ndarray:
    """phi at the points of an nx x ny lattice (the nearest fine grid points)."""
    xi = np.clip(np.round(np.arange(1, nx + 1) * (NXF + 1) / (nx + 1)).astype(int) - 1, 0, NXF - 1)
    yi = np.clip(np.round(np.arange(1, ny + 1) * (NYF + 1) / (ny + 1)).astype(int) - 1, 0, NYF - 1)
    return phi[np.ix_(xi, yi)]


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
    i0, j0, gx, gy, amp = P.BUMP
    return P.gauss2(i0, j0, gx, gy, amp)


def emax_of(kind: str) -> float:
    """The colour scale of the map of the mode energies (the largest fraction of one mode)."""
    if ("em", kind) not in _cache:
        if kind == "pluck":
            vals = [P.fractions(P.mode_energies(bump_initial(), np.zeros((NX, NY)))).max()]
        elif kind == "lin":
            vals = [P.fractions(P.mode_energies(*P.collision_initial())).max()]
        else:
            Q, V = nonlinear_run()
            vals = [P.fractions(P.mode_energies(q, v)).max() for q, v in zip(Q[::CFG.map.ymax_stride], V[::CFG.map.ymax_stride])]
        _cache[("em", kind)] = max(vals)
    return _cache[("em", kind)]


def state(t: float) -> dict:
    I, M, PL, N, C = CFG.intro, CFG.modes, CFG.pluck, CFG.nonlinear, CFG.limit
    HS = CFG.collision.height_scale
    st = {"q": np.zeros((NX, NY)), "E": None, "active": None, "scale": 1.0, "tag": None, "emax": 1.0,
          "formulas": [], "cap": "c0", "chain_a": smooth(t, *I.lattice_appear), "panels": 0.0, "kind": "intro", "cont": None,
          "delta": None}
    if t < T_MODES[0]:
        st["formulas"] = [("fL1", smooth(t, *I.formula1_in) * (1 - smooth(t, *I.formula1_out)), 0), ("fEq", smooth(t, *I.formula2_in), 0)]
        st["cap"] = "c0" if t < I.caption_switch else "c1"
        return st
    st["panels"] = smooth(t, T_MODES[0], T_MODES[0] + M.panels_fade_s)
    if t < T_MODES[1]:
        st["kind"] = "modes"
        for (a_, b_), a, b, cap in MODES:
            if a <= t < b:
                w = smooth(t, a, a + M.fade_s) * (1.0 - smooth(t, b - M.fade_s, b))
                ph = (t - a) * PHYS_PER_S
                st["q"] = MODE_AMP * w * P.mode_shape(a_, b_) * math.cos(P.omega(a_, b_) * ph)
                E = np.zeros((NX, NY))
                E[a_ - 1, b_ - 1] = 1.0
                st["E"], st["active"], st["wmode"] = E, (a_, b_), w
                st["cap"] = cap
        st["formulas"] = [("fM1", 1.0, 0), ("fM2", 1.0, 1)]
        return st
    if t < T_PLUCK[1]:
        st["kind"] = "pluck"
        q, v = P.linear_state(bump_initial(), np.zeros((NX, NY)), (t - T_PLUCK[0]) * WAVE_RATE)
        st["q"], st["E"], st["emax"] = q, P.fractions(P.mode_energies(q, v)), emax_of("pluck")
        st["cap"] = "cp" if t < T_PLUCK[0] + PL.caption_switch else "cp2"
        st["formulas"] = [("fP", 1.0, 0)]
        return st
    if t < T_LIN[1]:
        st["kind"] = "lin"
        q, v = P.linear_state(*P.collision_initial(), (t - T_LIN[0]) * WAVE_RATE)
        st["q"], st["E"], st["scale"], st["emax"] = q, P.fractions(P.mode_energies(q, v)), HS, emax_of("lin")
        st["delta"], st["tag"], st["cap"] = 0.0, "lin", "cl"
        st["formulas"] = [("fEq", 1.0, 0)]
        return st
    if t < T_NONLIN[1]:
        st["kind"] = "nonlin"
        Q, V = nonlinear_run()
        i = int(min(round((t - T_NONLIN[0]) * CFG.physics.nonlinear_samples_per_s), len(Q) - 1))
        E = P.fractions(P.mode_energies(Q[i], V[i]))
        st["q"], st["E"], st["scale"], st["emax"] = Q[i], E, HS, emax_of("nonlin")
        st["delta"] = 0.5 * float(np.abs(E - P.fractions(P.mode_energies(*P.collision_initial()))).sum())
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
    from matplotlib.collections import LineCollection, PolyCollection
    from matplotlib.colors import LinearSegmentedColormap

    if snap is None and not shutil.which("ffmpeg"):
        sys.exit("ffmpeg is required")
    tx = TEXT[lang]
    V, S, LY, F, CM, G, MP, DS, LM = CFG.video, CFG.style, CFG.layout, CFG.fonts, CFG.camera, CFG.grid, CFG.map, CFG.dispersion, CFG.limit
    W, H = size
    dpi = V.dpi
    sc = H / V.reference_height
    BG = S.background
    TXT, DIM = tuple(S.text), tuple(S.dim)
    BLUE, WARM, GOLD = np.array(S.blue), np.array(S.warm), tuple(S.gold)
    MID = np.array(S.mid)
    cmap_e = LinearSegmentedColormap.from_list("e", [tuple(c) for c in MP.colormap])
    fig = plt.figure(figsize=(W / dpi, H / dpi), dpi=dpi, facecolor=BG)
    ax = fig.add_axes(LY.lattice_axes, facecolor="none")
    axm = fig.add_axes(LY.map_axes, facecolor="none")
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
    CX, CY = 0.5 * (NX + 1), 0.5 * (NY + 1)
    bg_rgba = np.array([int(BG[1:3], 16), int(BG[3:5], 16), int(BG[5:7], 16), 255], np.float32)

    def proj(x, y, z):
        x, y, z = np.asarray(x, float) - CX, np.asarray(y, float) - CY, np.asarray(z, float)
        xr = x * math.cos(AZ) - y * math.sin(AZ)
        yr = x * math.sin(AZ) + y * math.cos(AZ)
        return xr, yr * math.sin(EL) + z * math.cos(EL), yr

    def spring(ax_, p0, p1, turns: int = G.spring_turns, amp_px: float = G.spring_amplitude_px, lead_px: float = G.spring_lead_px, pts: int = G.spring_points) -> np.ndarray:
        P0 = ax_.transData.transform(p0)
        P1 = ax_.transData.transform(p1)
        d = P1 - P0
        L = float(np.hypot(*d))
        if L < 1e-6:
            return np.array([p0, p1], float)
        u = d / L
        n = np.array([-u[1], u[0]])
        lead = min(lead_px, G.spring_lead_max * L)
        s_ = np.linspace(0.0, L, pts)
        body = np.clip((s_ - lead) / max(L - 2 * lead, 1e-6), 0.0, 1.0)
        ramp = np.clip(np.minimum(body, 1.0 - body) / G.spring_ramp, 0.0, 1.0)
        off = amp_px * ramp * np.sin(2.0 * math.pi * turns * body) * ((s_ >= lead) & (s_ <= L - lead))
        pix = P0[None, :] + s_[:, None] * u[None, :] + off[:, None] * n[None, :]
        return ax_.transData.inverted().transform(pix)

    def hcolors(z: np.ndarray, scale: float) -> np.ndarray:
        a = np.clip(z / (G.height_color_scale * scale), -1, 1)[..., None]
        return np.where(a >= 0, MID * (1 - a) + WARM * a, MID * (1 + a) + BLUE * (-a))

    def draw_grid(Xs, Ys, Z, mode, alpha, scale, appear=None):
        """Z has the shape (nx + 2, ny + 2) with the zero edge; mode 'springs', 'mesh' or 'surface'."""
        nxp, nyp = Z.shape
        XX, YY = np.meshgrid(Xs, Ys, indexing="ij")
        Xp, Yp, Dp = proj(XX, YY, Z)
        col = hcolors(Z, scale)
        # the translucent surface under the springs
        quads = np.stack([np.stack([Xp[:-1, :-1], Yp[:-1, :-1]], -1), np.stack([Xp[1:, :-1], Yp[1:, :-1]], -1),
                          np.stack([Xp[1:, 1:], Yp[1:, 1:]], -1), np.stack([Xp[:-1, 1:], Yp[:-1, 1:]], -1)], axis=2).reshape(-1, 4, 2)
        if mode == "surface":
            hx = np.gradient(Z, Xs, axis=0)
            hy = np.gradient(Z, Ys, axis=1)
            nrm = np.stack([-hx, -hy, np.ones_like(hx)], axis=-1)
            nrm /= np.linalg.norm(nrm, axis=-1, keepdims=True)
            lam = np.clip(nrm @ np.array(G.light) / G.light_norm, 0.0, 1.0)
            col = np.clip(col * (G.shade_base + G.shade_gain * lam[..., None]), 0.0, 1.0)
        cq = ((col[:-1, :-1] + col[1:, :-1] + col[1:, 1:] + col[:-1, 1:]) / 4.0).reshape(-1, 3)
        dq = Dp[:-1, :-1].reshape(-1)
        srt = np.argsort(-dq)
        a_s = (G.surface_alpha if mode == "surface" else G.under_surface_alpha) * alpha
        fc = np.column_stack([cq[srt], np.full(len(srt), a_s)])
        ax.add_collection(PolyCollection(quads[srt], facecolors=fc, edgecolors=fc if mode == "surface" else "none", linewidths=G.surface_edge_width * sc, zorder=1))
        if mode == "surface":
            return
        if mode == "springs":
            segs, cols = [], []
            ap = np.ones((nxp, nyp)) if appear is None else appear
            for i in range(nxp - 1):
                for j in range(1, nyp - 1):
                    segs.append(spring(ax, (Xp[i, j], Yp[i, j]), (Xp[i + 1, j], Yp[i + 1, j])))
                    cols.append((*G.spring_color, G.spring_alpha * alpha * min(ap[i, j], ap[i + 1, j])))
            for i in range(1, nxp - 1):
                for j in range(nyp - 1):
                    segs.append(spring(ax, (Xp[i, j], Yp[i, j]), (Xp[i, j + 1], Yp[i, j + 1])))
                    cols.append((*G.spring_color, G.spring_alpha * alpha * min(ap[i, j], ap[i, j + 1])))
            ax.add_collection(LineCollection(segs, colors=cols, linewidths=G.spring_width * sc, zorder=2))
        else:
            step_i = max(1, (nxp - 2) // G.mesh_steps[0])
            step_j = max(1, (nyp - 2) // G.mesh_steps[1])
            segs = [np.column_stack([Xp[i, :], Yp[i, :]]) for i in range(0, nxp, step_i)]
            segs += [np.column_stack([Xp[:, j], Yp[:, j]]) for j in range(0, nyp, step_j)]
            ax.add_collection(LineCollection(segs, colors=[(*G.mesh_color, G.mesh_alpha * alpha)] * len(segs), linewidths=G.mesh_width * sc, zorder=2))
        # the masses
        inner = (slice(1, nxp - 1), slice(1, nyp - 1))
        ms = np.full(Xp[inner].size, G.dot_size if mode == "springs" else (G.mesh_dot_size if nxp < G.dense_n else G.dense_dot_size)) * sc * sc
        cm = np.column_stack([col[inner].reshape(-1, 3), (np.ones(Xp[inner].size) if appear is None else appear[inner].reshape(-1)) * alpha])
        ax.scatter(Xp[inner].reshape(-1), Yp[inner].reshape(-1), s=ms, c=cm, edgecolors="none", zorder=5)
        # the fixed edge
        ex = np.concatenate([Xp[:, 0], Xp[-1, :], Xp[::-1, -1], Xp[0, ::-1]])
        ey = np.concatenate([Yp[:, 0], Yp[-1, :], Yp[::-1, -1], Yp[0, ::-1]])
        ax.plot(ex, ey, color=(*G.edge_color, G.edge_alpha * alpha), lw=G.edge_width * sc, zorder=3, solid_joinstyle="round")

    for k_ in ids:
        t_film = k_ / fps
        tf_nom = t_film / k
        t, card, cprog = timeline(tf_nom)
        st = state(t)
        if card is not None:
            st.update({"q": np.zeros((NX, NY)), "E": None, "active": None, "formulas": [], "cap": None, "chain_a": 0.0,
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
            a_, b_ = st["active"]
            fig.text(*LY.mode_label_pos, f"{tx['mode']} $(a,b)=({a_},{b_})$,  $\\omega_{{ab}}={num(P.omega(a_, b_), '.2f', lang)}\\,\\sqrt{{k/m}}$",
                     color=(*GOLD, st.get("wmode", 1.0)), fontsize=F.mode_label * sc)
        # ---------------- the lattice
        a_ch = st["chain_a"]
        if st["kind"] != "cont":
            Z = np.pad(st["q"] * st["scale"], 1)
            Xs, Ys = np.arange(NX + 2, dtype=float), np.arange(NY + 2, dtype=float)
            ii, jj = np.meshgrid(np.arange(NX + 2), np.arange(NY + 2), indexing="ij")
            appear = np.clip((a_ch * (NX + NY + CFG.intro.sweep_lead) - (ii + jj)) / CFG.intro.sweep_width, 0.0, 1.0)
            if a_ch > S.lattice_cutoff:
                draw_grid(Xs, Ys, Z, "springs", 1.0, st["scale"] if st["kind"] in ("lin", "nonlin") else 1.0, appear)
            if st["tag"]:
                fig.text(*LY.tag_pos, tx[st["tag"]], color=(*(BLUE if st["tag"] == "lin" else WARM), 1.0), fontsize=F.tag * sc, ha="left", va="top")
        else:
            (nx, ny), tc, a_field = st["cont"]
            phi = sample_fine(fine_field(tc), nx, ny)
            Z = np.pad(phi * LM.height_scale, 1)
            Xs = np.concatenate([[0.0], np.arange(1, nx + 1) * LXF / (nx + 1), [LXF]])
            Ys = np.concatenate([[0.0], np.arange(1, ny + 1) * LYF / (ny + 1), [LYF]])
            if nx <= LM.springs_max_n:
                draw_grid(Xs, Ys, Z, "springs", 1 - a_field, LM.height_scale)
            elif nx <= LM.mesh_max_n:
                draw_grid(Xs, Ys, Z, "mesh", 1 - a_field, LM.height_scale)
            if a_field > LM.surface_cutoff or nx > LM.mesh_max_n:
                st_ = LM.surface_stride
                fine = fine_field(tc)[::st_, ::st_]
                Xf = np.concatenate([[0.0], (np.arange(1, NXF + 1)[::st_]) * LXF / (NXF + 1), [LXF]])
                Yf = np.concatenate([[0.0], (np.arange(1, NYF + 1)[::st_]) * LYF / (NYF + 1), [LYF]])
                draw_grid(Xf, Yf, np.pad(fine * LM.height_scale, 1), "surface", max(a_field, 0.0 if nx <= LM.mesh_max_n else 1.0), LM.surface_color_scale)
            fig.text(*LY.n_label_pos, rf"$N={nx}\times{ny}$" if a_field < LM.label_threshold else r"$N\to\infty$", color=(*TXT, 1.0), fontsize=F.n_label * sc, ha="right", va="top")
        # ---------------- the map of the mode energies and the dispersion
        pa = st["panels"]
        if pa > S.panel_cutoff and st["kind"] != "cont":
            axm.axis("on")
            for sp in axm.spines.values():
                sp.set_color((1, 1, 1, MP.border_alpha * pa))
            axm.set_xticks([])
            axm.set_yticks([])
            E = st["E"] if st["E"] is not None else np.zeros((NX, NY))
            axm.imshow(E.T, origin="lower", extent=(0.5, NX + 0.5, 0.5, NY + 0.5), cmap=cmap_e, vmin=0, vmax=max(st["emax"], 1e-6), aspect="auto", alpha=pa)
            if st.get("active") is not None:
                a_, b_ = st["active"]
                axm.add_patch(plt.Rectangle((a_ - 0.5, b_ - 0.5), 1, 1, fill=False, ec=GOLD, lw=MP.highlight_width * sc))
            axm.text(*MP.title_pos, tx["E"], color=(*DIM, pa), fontsize=MP.title_size * sc, transform=axm.transAxes, ha="left", va="bottom")
            axm.text(*MP.a_label_pos, r"$a$", color=(*DIM, pa), fontsize=MP.label_size * sc, transform=axm.transAxes, ha="right", va="top")
            axm.text(*MP.b_label_pos, r"$b$", color=(*DIM, pa), fontsize=MP.label_size * sc, transform=axm.transAxes, ha="right", va="top")
            if st["delta"] is not None:
                fig.text(*MP.delta_pos, tx["delta"] + rf" ${100 * st['delta']:.0f}\,\%$", color=(*TXT, pa), fontsize=MP.delta_size * sc,
                         ha="left", va="bottom", linespacing=MP.delta_linespacing)
        if pa > S.panel_cutoff:
            axd.axis("on")
            for sp in axd.spines.values():
                sp.set_visible(False)
            axd.set_xticks([])
            axd.set_yticks([])
            axd.set_xlim(0, DS.k_max)
            axd.set_ylim(0, DS.y_max)
            kk = np.linspace(0, DS.k_max, DS.line_samples)
            if st["kind"] == "cont":
                a_d = smooth(t, T_CONT[0] + LM.dispersion_switch[0], T_CONT[0] + LM.dispersion_switch[1])
                axd.scatter(P.KMAG.ravel(), P.OMEGA.ravel(), s=DS.limit_dot_size * sc * sc,
                            c=[(*BLUE, DS.limit_dot_alpha * (1 - DS.limit_dot_fade * a_d))], edgecolors="none")
                axd.plot(kk, kk, color=(*BLUE, a_d), lw=DS.limit_line_width * sc, ls=(0, tuple(DS.limit_line_dash)))
                axd.text(*DS.limit_label_pos, tx["l_cont"], color=(*BLUE, a_d), fontsize=DS.limit_label_size * sc, ha="left")
                axd.text(*DS.lattice_label_pos, tx["l_latt"], color=(*DIM, a_d), fontsize=DS.limit_label_size * sc, ha="right")
            else:
                if st["E"] is not None and st["E"].max() > 0:
                    fr = (st["E"] / st["E"].max()).ravel()
                else:
                    fr = np.full(NX * NY, 0.0)
                sizes = (DS.dot_size_base + DS.dot_size_gain * fr) * sc * sc
                colors = [(*BLUE, pa * (DS.dot_alpha_base + DS.dot_alpha_gain * f)) for f in fr]
                if st.get("active") is not None:
                    a_, b_ = st["active"]
                    colors[(a_ - 1) * NY + (b_ - 1)] = (*GOLD, pa)
                    sizes[(a_ - 1) * NY + (b_ - 1)] = DS.active_size * sc * sc
                axd.scatter(P.KMAG.ravel(), P.OMEGA.ravel(), s=sizes, c=colors, edgecolors="none", zorder=4)
                axd.plot(kk, kk, color=(1, 1, 1, DS.cone_alpha * pa), lw=DS.cone_width * sc, ls=(0, tuple(DS.cone_dash)))
            axd.plot([0, DS.k_max], [0, 0], color=(1, 1, 1, DS.baseline_alpha * pa), lw=DS.baseline_width * sc)
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
    out = args.out or HERE / "media" / f"coupled_lattice_2d_{args.lang}.mp4"
    out.parent.mkdir(parents=True, exist_ok=True)
    if args.snapshot is not None:
        render(out.with_suffix(".png"), (V.width, V.height), V.fps, args.seconds, args.lang, snap=args.snapshot)
    elif args.preview:
        render(out.with_name(out.stem + "_preview.mp4"), (V.preview_width, V.preview_height), V.preview_fps, args.seconds, args.lang)
    else:
        render(out, (V.width, V.height), V.fps, args.seconds, args.lang)


if __name__ == "__main__":
    main()
