r"""A chain of N coupled oscillators: from the normal modes to the field (the book, the classical field theory chapter).

Parts of the film:
 1. the chain of N masses m joined by springs k (fixed ends), the Lagrangian and the equations of motion;
 2. the normal modes  x_j ~ sin(s pi j / (N+1)) cos(omega_s t),  omega_s = 2 sqrt(k/m) sin(s pi / (2 (N+1))): every mass
    oscillates with the same frequency; the position of the mode on the dispersion curve is marked, the bars show
    the energy of every mode;
 3. one bump is released: it splits into two waves, many modes take part, the energy of every mode is constant;
 4. two wave packets collide: in a linear chain they pass through each other (the modes do not interact: free particles);
 5. the same collision with the cubic term of the Lagrangian (V = k d^2/2 + alpha d^3/3 + beta d^4/4, d = x_{j+1} - x_j,
    beta is added to keep the potential bounded): the modes exchange energy, the packets change - an interaction;
 6. the limit N -> infinity: the masses become dense, the chain becomes the field phi(x, t), the dispersion becomes omega = c k.

All the numbers are in config.toml and all the words in texts.toml (see ../dvconfig.py for --config / --set).

Usage:
    python coupled_chain.py --lang en            # film -> media/coupled_chain_en.mp4
    python coupled_chain.py --lang ru
    python coupled_chain.py --lang en --snapshot 50
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

N = CFG.physics.n                # the masses
KC = CFG.physics.k               # spring constant, m = 1
ALPHA, BETA = CFG.physics.alpha, CFG.physics.beta          # cubic and quartic couplings (alpha^2 < 3 beta: the potential is convex)
CARD_S = CFG.timeline.card_s     # length of a chapter card
CARD_AT = list(CFG.timeline.card_at)                       # a card is shown before the content time
CARD_KEYS = [f"h{i}" for i in range(len(CARD_AT))]
_PB = list(CFG.timeline.part_bounds)
CONTENT_TOTAL = _PB[-1]          # the film without the title cards
TOTAL = CONTENT_TOTAL + CARD_S * len(CARD_AT)

T_INTRO, T_MODES, T_PLUCK, T_LIN, T_NONLIN, T_CONT = ((_PB[i], _PB[i + 1]) for i in range(6))

MODES = [(m["s"] or N, m["from"], m["to"], m["caption"]) for m in CFG.modes.list]   # s, from, to, caption
PERIOD_S1 = CFG.modes.period_s   # film seconds per period of the mode s = 1
PHYS_PER_S = (2 * math.pi / PERIOD_S1) / (2.0 * math.sin(math.pi / (2 * (N + 1))))   # physical time per film second in the mode part
WAVE_RATE = CFG.physics.wave_rate                          # physical time per film second for the waves (c = 1 site per unit time)


def timeline(tf: float) -> tuple[float, int | None, float]:
    """(content time, index of the chapter card or None, progress of the card in [0, 1]) at the film time tf."""
    done = 0
    for i, ca in enumerate(CARD_AT):
        a = ca + i * CARD_S
        if a <= tf < a + CARD_S:
            return ca, i, (tf - a) / CARD_S
        if tf >= a + CARD_S:
            done += 1
    return tf - done * CARD_S, None, 0.0


def film_time(tc: float) -> float:
    """Film time of the content time tc (the cards before it are counted)."""
    return tc + CARD_S * sum(1 for ca in CARD_AT if ca <= tc)


def chapter_of(tc: float) -> int:
    return max(i for i, ca in enumerate(CARD_AT) if ca <= tc)


def smooth(x: float, a: float, b: float) -> float:
    u = min(max((x - a) / (b - a), 0.0), 1.0)
    return u * u * (3 - 2 * u)


# -------------------------------------------------------------------------------- the physics

J = np.arange(1, N + 1)
S = np.sqrt(2.0 / (N + 1)) * np.sin(np.outer(np.arange(1, N + 1), J) * math.pi / (N + 1))       # orthogonal: rows are the modes
OMEGA = 2.0 * math.sqrt(KC) * np.sin(np.arange(1, N + 1) * math.pi / (2 * (N + 1)))


def omega(s: float, n: int = N) -> float:
    """Frequency of the mode s of a chain of n masses (k = m = 1)."""
    return 2.0 * math.sqrt(KC) * math.sin(s * math.pi / (2 * (n + 1)))


def mode_shape(s: int) -> np.ndarray:
    return S[s - 1]


def linear_state(x0: np.ndarray, v0: np.ndarray, t: float) -> tuple[np.ndarray, np.ndarray]:
    """Exact evolution of the linear chain from the normal modes."""
    a = S @ x0
    b = (S @ v0) / OMEGA
    q = a * np.cos(OMEGA * t) + b * np.sin(OMEGA * t)
    p = -a * OMEGA * np.sin(OMEGA * t) + b * OMEGA * np.cos(OMEGA * t)
    return S.T @ q, S.T @ p


def mode_energies(x: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Energy of every normal mode, E_s = (p_s^2 + omega_s^2 q_s^2) / 2 (harmonic part)."""
    return 0.5 * (S @ v) ** 2 + 0.5 * OMEGA ** 2 * (S @ x) ** 2


def force(x: np.ndarray, alpha: float, beta: float) -> np.ndarray:
    xp = np.concatenate([[0.0], x, [0.0]])
    d = np.diff(xp)
    f = KC * d + alpha * d ** 2 + beta * d ** 3
    return f[1:] - f[:-1]


def potential_energy(x: np.ndarray, alpha: float, beta: float) -> float:
    d = np.diff(np.concatenate([[0.0], x, [0.0]]))
    return float(np.sum(0.5 * KC * d ** 2 + alpha * d ** 3 / 3.0 + beta * d ** 4 / 4.0))


def integrate(x0: np.ndarray, v0: np.ndarray, alpha: float, beta: float, times: np.ndarray, dt: float = CFG.physics.default_dt):
    """Velocity-Verlet; returns x and v at the given (increasing) physical times."""
    x, v = x0.copy(), v0.copy()
    a = force(x, alpha, beta)
    t = 0.0
    xs, vs = [], []
    for tt in times:
        while t < tt - 1e-12:
            h = min(dt, tt - t)
            v = v + 0.5 * h * a
            x = x + h * v
            a = force(x, alpha, beta)
            v = v + 0.5 * h * a
            t += h
        xs.append(x.copy())
        vs.append(v.copy())
    return np.array(xs), np.array(vs)


def packet(j1: float, k0: float, sig: float, amp: float, sgn: float) -> tuple[np.ndarray, np.ndarray]:
    """A wave packet that travels to the right (sgn = +1) or to the left (-1); (displacements, velocities)."""
    env = np.exp(-((J - j1) ** 2) / (2 * sig ** 2))
    w = 2.0 * math.sin(k0 / 2.0)
    return amp * np.cos(k0 * (J - j1)) * env, sgn * w * amp * np.sin(k0 * (J - j1)) * env


def bump(j0: float, sig: float, amp: float) -> np.ndarray:
    return amp * np.exp(-((J - j0) ** 2) / (2 * sig ** 2))


PK_A, PK_K0, PK_SIG = CFG.physics.packet.amplitude, CFG.physics.packet.carrier_k, CFG.physics.packet.sigma
PK_LEFT, PK_RIGHT = CFG.physics.packet.left_j, CFG.physics.packet.right_j
BUMP_J0, BUMP_SIG, BUMP_A = CFG.physics.bump.centre_j, CFG.physics.bump.sigma, CFG.physics.bump.amplitude


def collision_initial() -> tuple[np.ndarray, np.ndarray]:
    x1, v1 = packet(PK_LEFT, PK_K0, PK_SIG, PK_A, +1.0)
    x2, v2 = packet(PK_RIGHT, PK_K0, PK_SIG, PK_A, -1.0)
    return x1 + x2, v1 + v2


def continuum_field(x: np.ndarray, t: float, length: float) -> np.ndarray:
    """d'Alembert solution of the wave equation with fixed ends: a bump released at rest."""
    def g(y: np.ndarray) -> np.ndarray:                                 # odd 2L-periodic extension of the bump
        y = (y + length) % (2 * length) - length
        base = lambda z: BUMP_A * np.exp(-((z - BUMP_J0 * length / (N + 1)) ** 2) / (2 * (BUMP_SIG * length / (N + 1)) ** 2))
        return base(y) - base(-y)
    return 0.5 * (g(x - t) + g(x + t))


# ----------------------------------------------------------------------------------- texts

def num(x: float, fmt: str, lang: str) -> str:
    s = format(x, fmt)
    return s.replace(".", "{,}") if lang == "ru" else s


TEXT = {lang: load_texts(HERE, lang) for lang in ("en", "ru")}     # texts.toml


# ----------------------------------------------------------------------------------- the film state

_cache: dict = {}


def nonlinear_run() -> tuple[np.ndarray, np.ndarray]:
    """The collision with the cubic term at the film rate of 30 frames per second (index = film frame inside the part)."""
    if "nl" not in _cache:
        x0, v0 = collision_initial()
        rate = CFG.physics.nonlinear_samples_per_s
        n = int(round((T_NONLIN[1] - T_NONLIN[0]) * rate)) + 1
        _cache["nl"] = integrate(x0, v0, ALPHA, BETA, np.arange(n) / float(rate) * WAVE_RATE)
    return _cache["nl"]


def fractions(E: np.ndarray) -> np.ndarray:
    return E / max(float(E.sum()), 1e-12)


def ymax_of(kind: str) -> float:
    if ("ym", kind) not in _cache:
        if kind == "pluck":
            vals = [fractions(mode_energies(*linear_state(bump(BUMP_J0, BUMP_SIG, BUMP_A), np.zeros(N), 0.0))).max()]
        elif kind == "lin":
            vals = [fractions(mode_energies(*linear_state(*collision_initial(), 0.0))).max()]
        else:
            X, V = nonlinear_run()
            vals = [fractions(mode_energies(x, v)).max() for x, v in zip(X[::CFG.spectrum.ymax_stride], V[::CFG.spectrum.ymax_stride])]
        _cache[("ym", kind)] = CFG.spectrum.y_margin * max(vals)
    return _cache[("ym", kind)]


def state(t: float) -> dict:
    """Everything that is drawn at the film time t: displacements, mode fractions, the marked modes, the texts."""
    I, M, PL, N_, C = CFG.intro, CFG.modes, CFG.pluck, CFG.nonlinear, CFG.limit
    HS = CFG.collision.height_scale
    st = {"x": np.zeros(N), "E": None, "active": None, "scale": 1.0, "tag": None, "ymax": CFG.spectrum.y_margin, "ghost": None,
          "formulas": [], "cap": "c0", "chain_a": smooth(t, *I.chain_appear), "panels": 0.0, "kind": "intro", "cont": None}
    if t < T_MODES[0]:
        st["formulas"] = [("fL", smooth(t, *I.formula1_in) * (1 - smooth(t, *I.formula1_out)), 0), ("fEq", smooth(t, *I.formula2_in), 0)]
        st["cap"] = "c0" if t < I.caption_switch else "c1"
        return st
    st["panels"] = smooth(t, T_MODES[0], T_MODES[0] + M.panels_fade_s)
    if t < T_MODES[1]:
        st["kind"] = "modes"
        for s, a, b, cap in MODES:
            if a <= t < b:
                w = smooth(t, a, a + M.fade_s) * (1.0 - smooth(t, b - M.fade_s, b))
                ph = (t - a) * PHYS_PER_S
                st["x"] = M.amplitude * w * mode_shape(s) * math.cos(omega(s) * ph)
                E = np.zeros(N)
                E[s - 1] = 1.0
                st["E"], st["active"], st["mode_s"] = E, s, s
                st["ymax"] = CFG.spectrum.y_margin
                st["cap"] = cap
                st["wmode"] = w
        st["formulas"] = [("fM1", 1.0, 0), ("fM2", 1.0, 1)]
        return st
    if t < T_PLUCK[1]:
        st["kind"] = "pluck"
        tp = (t - T_PLUCK[0]) * WAVE_RATE
        x0 = bump(BUMP_J0, BUMP_SIG, BUMP_A)
        x, v = linear_state(x0, np.zeros(N), max(0.0, tp))
        st["x"], st["E"], st["ymax"] = x, fractions(mode_energies(x, v)), ymax_of("pluck")
        st["cap"] = "cp" if t < T_PLUCK[0] + PL.caption_switch else "cp2"
        st["formulas"] = [("fP", 1.0, 0)]
        return st
    if t < T_LIN[1]:
        st["kind"] = "lin"
        tp = (t - T_LIN[0]) * WAVE_RATE
        x, v = linear_state(*collision_initial(), tp)
        st["x"], st["E"], st["scale"], st["ymax"] = x, fractions(mode_energies(x, v)), HS, ymax_of("lin")
        st["delta"] = 0.0
        st["tag"], st["cap"] = "lin", "cl"
        st["formulas"] = [("fEq", 1.0, 0)]
        return st
    if t < T_NONLIN[1]:
        st["kind"] = "nonlin"
        X, V = nonlinear_run()
        i = int(min(round((t - T_NONLIN[0]) * CFG.physics.nonlinear_samples_per_s), len(X) - 1))
        st["x"], st["E"], st["scale"], st["ymax"] = X[i], fractions(mode_energies(X[i], V[i])), HS, ymax_of("nonlin")
        st["delta"] = 0.5 * float(np.abs(st["E"] - fractions(mode_energies(*collision_initial()))).sum())
        xl, vl = linear_state(*collision_initial(), (t - T_NONLIN[0]) * WAVE_RATE)
        st["E_lin"] = fractions(mode_energies(xl, vl))
        st["ghost"] = xl
        st["tag"] = "nonlin"
        st["cap"] = "cn" if t < T_NONLIN[0] + N_.caption_switch else "cn2"
        st["formulas"] = [("fN", 1.0, 0)]
        return st
    st["kind"] = "cont"
    st["panels"] = 1.0
    st["formulas"] = [("fC", smooth(t, T_CONT[0] + C.formula_in[0], T_CONT[0] + C.formula_in[1]), 0)]
    st["cap"] = "cc" if t < T_CONT[0] + C.caption_switch else "cc2"
    nd = list(C.levels)
    k = int(min(len(nd) - 1, (t - T_CONT[0]) // C.level_s))
    st["cont"] = (nd[k], (t - T_CONT[0]) * C.time_rate + C.time_offset, smooth(t, T_CONT[0] + C.label_switch[0], T_CONT[0] + C.label_switch[1]))
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
    V, S_, LY, F, CM, CH, SP, DS, LM = CFG.video, CFG.style, CFG.layout, CFG.fonts, CFG.camera, CFG.chain, CFG.spectrum, CFG.dispersion, CFG.limit
    W, H = size
    dpi = V.dpi
    sc = H / V.reference_height
    BG = S_.background
    TXT, DIM = tuple(S_.text), tuple(S_.dim)
    BLUE, WARM, GOLD = tuple(S_.blue), tuple(S_.warm), tuple(S_.gold)
    fig = plt.figure(figsize=(W / dpi, H / dpi), dpi=dpi, facecolor=BG)
    ax = fig.add_axes(LY.chain_axes, facecolor="none")
    axb = fig.add_axes(LY.spectrum_axes, facecolor="none")
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

    s_axis = np.arange(1, N + 1)
    s_fine = np.linspace(0.0, N + 1, DS.curve_samples)
    LEN = N + 1.0
    bg_rgba = np.array([int(BG[1:3], 16), int(BG[3:5], 16), int(BG[5:7], 16), 255], np.float32)

    def spring(ax_, p0, p1, turns: int = CH.spring_turns, amp_px: float = CH.spring_amplitude_px, lead_px: float = CH.spring_lead_px, pts: int = CH.spring_points) -> np.ndarray:
        """A coil spring between two points of the axes; the geometry is built in pixels, so a stretched spring
        (a long, inclined one) has the same coil radius and longer pitch."""
        P0 = ax_.transData.transform(p0)
        P1 = ax_.transData.transform(p1)
        d = P1 - P0
        L = float(np.hypot(*d))
        if L < 1e-6:
            return np.array([p0, p1], float)
        u = d / L
        n = np.array([-u[1], u[0]])
        lead = min(lead_px, CH.spring_lead_max * L)
        s_ = np.linspace(0.0, L, pts)
        body = np.clip((s_ - lead) / max(L - 2 * lead, 1e-6), 0.0, 1.0)
        ramp = np.clip(np.minimum(body, 1.0 - body) / CH.spring_ramp, 0.0, 1.0)
        off = amp_px * ramp * np.sin(2.0 * math.pi * turns * body) * ((s_ >= lead) & (s_ <= L - lead))
        pix = P0[None, :] + s_[:, None] * u[None, :] + off[:, None] * n[None, :]
        return ax_.transData.inverted().transform(pix)

    def mass_colors(y: np.ndarray, scale: float) -> np.ndarray:
        a = np.clip(y / (CH.height_color_scale * scale if scale else 1.0), -1, 1)[:, None]
        neg = np.array(BLUE)[None, :]
        pos = np.array(WARM)[None, :]
        mid = np.array(S_.mid)[None, :]
        return np.where(a >= 0, mid * (1 - a) + pos * a, mid * (1 + a) + neg * (-a))

    for k_ in ids:
        t_film = k_ / fps
        tf_nom = t_film / k
        t, card, cprog = timeline(tf_nom)
        st = state(t)
        if card is not None:
            st.update({"x": np.zeros(N), "E": None, "active": None, "formulas": [], "cap": None, "chain_a": 0.0, "panels": 0.0,
                       "kind": "intro", "tag": None, "ghost": None})
        fig.texts.clear()
        for ar in (ax, axb, axd):
            ar.clear()
            ar.set_facecolor("none")
        # ---------------- titles, formulas, captions
        if card is None:
            fig.text(*LY.title_pos, tx["title"], color=tuple(S_.title_color), fontsize=F.title * sc)
            ch = chapter_of(t)
            if ch >= 1:
                fig.text(*LY.chapter_label_pos, f"{ch}/{len(CARD_AT) - 1}   " + tx[CARD_KEYS[ch]], color=(*DIM, S_.chapter_label_alpha),
                         fontsize=F.chapter_label * sc, ha="right")
        else:
            a_c = min(smooth(cprog, *CFG.timeline.card_fade_in), 1.0 - smooth(cprog, *CFG.timeline.card_fade_out))
            if card == 0:
                fig.text(0.5, LY.card_title_y, tx["h0"], color=(*S_.card_title_color, a_c), fontsize=F.card_title * sc, ha="center", va="center")
                fig.text(0.5, LY.card_subtitle_y, tx["h0s"], color=(*DIM, a_c), fontsize=F.card_subtitle * sc, ha="center", va="center")
            else:
                fig.text(0.5, LY.card_number_y, f"{card}", color=(*GOLD, S_.card_number_alpha * a_c), fontsize=F.card_number * sc, ha="center", va="center")
                fig.text(0.5, LY.card_chapter_y, tx[CARD_KEYS[card]], color=(*S_.card_title_color, a_c), fontsize=F.card_chapter * sc, ha="center", va="center")
                fig.text(0.5, LY.card_chapter_sub_y, tx[CARD_KEYS[card] + "s"], color=(*DIM, a_c), fontsize=F.card_chapter_sub * sc, ha="center", va="center")
        for key, alpha, row in st["formulas"]:
            if alpha > LY.formula_min_alpha:
                fig.text(LY.formula_pos[0], LY.formula_pos[1] - LY.formula_row * row, tx[key], color=(*TXT, S_.formula_alpha * alpha), fontsize=F.formula * sc)
        if st["cap"]:
            fig.text(*LY.caption_pos, tx[st["cap"]], color=(*TXT, S_.caption_alpha), fontsize=F.caption * sc)
        if st["kind"] == "modes" and st.get("active") is not None:
            s = st["active"]
            w = st.get("wmode", 1.0)
            ttxt = f"{tx['mode']} $s={s}$,  $\\omega_s={num(omega(s), '.2f', lang)}\\,\\sqrt{{k/m}}$"
            fig.text(*LY.mode_label_pos, ttxt, color=(*GOLD, w), fontsize=F.mode_label * sc)
        # ---------------- the chain
        ax.axis("off")
        ax.set_xlim(-CM.xlim_margin, LEN + CM.xlim_margin)
        ax.set_ylim(*CM.ylim)
        a_ch = st["chain_a"]
        if st["kind"] != "cont":
            # the masses appear one after another
            appear = np.clip((a_ch * (N + CFG.intro.sweep_lead) - np.arange(N)) / CFG.intro.sweep_width, 0.0, 1.0)
            y = st["x"] * st["scale"]
            xs = np.concatenate([[0.0], J.astype(float), [LEN]])
            ys = np.concatenate([[0.0], y, [0.0]])
            segs = [spring(ax, (xs[i], ys[i]), (xs[i + 1], ys[i + 1]), amp_px=CH.spring_amplitude_px * sc, lead_px=CH.spring_lead_px * sc) for i in range(N + 1)]
            col_sp = [(*CH.spring_color, CH.spring_alpha * min(appear[min(i, N - 1)], appear[max(i - 1, 0)])) for i in range(N + 1)]
            ax.add_collection(LineCollection(segs, colors=col_sp, linewidths=CH.spring_width * sc, capstyle="round"))
            for xw in (0.0, LEN):
                ax.plot([xw, xw], [-CH.wall_half_height, CH.wall_half_height], color=(*CH.wall_color, CH.wall_alpha * a_ch), lw=CH.wall_width * sc, solid_capstyle="round")
            cols = mass_colors(y, st["scale"])
            rgba = np.column_stack([cols, appear])
            ax.scatter(J, y, s=(CH.mass_size * sc * sc) * appear, c=rgba, edgecolors=tuple(CH.mass_edge_color), linewidths=CH.mass_edge_width * sc, zorder=5)
            ax.plot([0, LEN], [0, 0], color=(1, 1, 1, CH.axis_line_alpha * a_ch), lw=CH.axis_line_width * sc, ls=(0, tuple(CH.axis_line_dash)))
            if st.get("ghost") is not None:
                from scipy.interpolate import PchipInterpolator
                xg = np.linspace(0.0, LEN, CFG.physics.ghost_samples)
                yg = PchipInterpolator(xs, np.concatenate([[0.0], st["ghost"] * st["scale"], [0.0]]))(xg)
                ax.plot(xg, yg, color=(1, 1, 1, CH.ghost_alpha), lw=CH.ghost_width * sc, ls=(0, tuple(CH.ghost_dash)), zorder=4)
            if st["tag"]:
                lab = tx[st["tag"]]
                ax.text(LEN + CH.tag_pos[0], CH.tag_pos[1], lab, color=(*(BLUE if st["tag"] == "lin" else WARM), 1.0), fontsize=CH.tag_size * sc, ha="right", va="top")
        else:
            nd, tc, a_field = st["cont"]
            xm = np.arange(1, nd + 1) * LEN / (nd + 1)
            ym = continuum_field(xm, tc, LEN)
            xs = np.concatenate([[0.0], xm, [LEN]])
            ys = np.concatenate([[0.0], ym, [0.0]])
            xf = np.linspace(0.0, LEN, LM.field_samples)
            yf = continuum_field(xf, tc, LEN)
            if nd <= LM.springs_max_n:
                segs = [spring(ax, (xs[i], ys[i]), (xs[i + 1], ys[i + 1]), amp_px=CH.spring_amplitude_px * sc, lead_px=CH.spring_lead_px * sc) for i in range(nd + 1)]
                ax.add_collection(LineCollection(segs, colors=[(*CH.spring_color, LM.spring_alpha * (1 - a_field))] * len(segs), linewidths=CH.spring_width * sc))
            else:
                ax.plot(xs, ys, color=(*CH.spring_color, LM.line_alpha * (1 - a_field)), lw=LM.chain_line_width * sc)
            ax.plot(xf, yf, color=(*BLUE, a_field), lw=LM.field_line_width * sc, solid_capstyle="round")
            cols = mass_colors(ym, 1.0)
            ax.scatter(xm, ym, s=max(LM.dot_min_size, LM.dot_size * LM.levels[0] / nd) * sc * sc * (1 - LM.dot_fade * a_field),
                       c=np.column_stack([cols, np.full(nd, 1 - LM.dot_alpha_fade * a_field)]), edgecolors="none", zorder=5)
            for xw in (0.0, LEN):
                ax.plot([xw, xw], [-CH.wall_half_height, CH.wall_half_height], color=(*CH.wall_color, CH.wall_alpha), lw=CH.wall_width * sc, solid_capstyle="round")
            ax.text(LEN + LM.n_label_pos[0], LM.n_label_pos[1], rf"$N={nd}$" if a_field < LM.label_threshold else r"$N\to\infty$", color=(*TXT, 1.0),
                    fontsize=LM.n_label_size * sc, ha="right", va="top")
        # ---------------- the energies of the modes
        pa = st["panels"]
        axb.axis("off")
        axd.axis("off")
        if pa > S_.panel_cutoff and st["kind"] != "cont":
            axb.set_xlim(SP.xlim[0], N + SP.xlim[1])
            ymx = st["ymax"]
            axb.set_ylim(0, ymx)
            E = st["E"] if st["E"] is not None else np.zeros(N)
            colb = [(*GOLD, SP.active_alpha * pa) if (st.get("active") == s) else (*BLUE, SP.bar_alpha * pa) for s in s_axis]
            axb.bar(s_axis, E, width=SP.bar_width, color=colb, lw=0)
            if st["kind"] == "nonlin":
                axb.step(np.concatenate([[0.5], s_axis + 0.5]), np.concatenate([st["E_lin"][:1], st["E_lin"]]), where="pre", color=(1, 1, 1, SP.ghost_alpha * pa),
                         lw=SP.ghost_width * sc, ls=(0, tuple(SP.ghost_dash)))
            axb.plot([SP.xlim[0], N + SP.xlim[1]], [0, 0], color=(1, 1, 1, SP.baseline_alpha * pa), lw=SP.baseline_width * sc)
            axb.text(*SP.title_pos, tx["E"], color=(*DIM, pa), fontsize=SP.title_size * sc, transform=axb.transAxes, ha="left", va="bottom")
            axb.text(*SP.xlabel_pos, r"$s$", color=(*DIM, pa), fontsize=SP.xlabel_size * sc, transform=axb.transAxes, ha="right", va="top")
            if st["kind"] in ("lin", "nonlin"):
                axb.text(*SP.delta_pos, tx["delta"] + rf" ${100 * st['delta']:.0f}\,\%$", color=(*TXT, pa), fontsize=SP.delta_size * sc, transform=axb.transAxes, ha="right", va="bottom")
            # the dispersion curve
            axd.set_xlim(0, N + 1)
            axd.set_ylim(0, DS.y_max)
            axd.plot(s_fine, [omega(v) for v in s_fine], color=(*DIM, DS.curve_alpha * pa), lw=DS.curve_width * sc)
            axd.plot([0, N + 1], [0, math.pi], color=(1, 1, 1, DS.cone_alpha * pa), lw=DS.cone_width * sc, ls=(0, tuple(DS.cone_dash)))
            if st["E"] is not None:
                fr = st["E"] / max(float(st["E"].max()), 1e-9)
                sizes = (DS.dot_size_base + DS.dot_size_gain * fr) * sc * sc
                colors = [(*GOLD, pa) if st.get("active") == s else (*BLUE, pa * (DS.dot_alpha_base + DS.dot_alpha_gain * fr[s - 1])) for s in s_axis]
            else:
                sizes, colors = DS.dot_size_base * sc * sc, (*BLUE, pa)
            axd.scatter(s_axis, [omega(s) for s in s_axis], s=sizes, c=colors, edgecolors="none", zorder=4)
            axd.plot([0, N + 1], [0, 0], color=(1, 1, 1, DS.baseline_alpha * pa), lw=DS.baseline_width * sc)
            axd.text(*DS.title_pos, tx["disp"], color=(*DIM, pa), fontsize=DS.title_size * sc, transform=axd.transAxes, ha="left", va="bottom")
            axd.text(*DS.xlabel_pos, r"$s$", color=(*DIM, pa), fontsize=DS.xlabel_size * sc, transform=axd.transAxes, ha="right", va="top")
        elif st["kind"] == "cont":
            a_d = smooth(t, T_CONT[0] + LM.dispersion_switch[0], T_CONT[0] + LM.dispersion_switch[1])
            axd.set_xlim(0, 1)
            kk = np.linspace(0, 1, DS.limit_samples)
            axd.plot(kk, np.sin(0.5 * math.pi * kk), color=(*DIM, DS.limit_curve_alpha * a_d), lw=DS.limit_curve_width * sc)
            axd.plot(kk, 0.5 * math.pi * kk, color=(*BLUE, a_d), lw=DS.limit_line_width * sc, ls=(0, tuple(DS.limit_line_dash)))
            axd.set_ylim(0, DS.limit_y_max)
            axd.text(*DS.title_pos, tx["disp"], color=(*DIM, a_d), fontsize=DS.title_size * sc, transform=axd.transAxes, ha="left", va="bottom")
            axd.plot([0, 1], [0, 0], color=(1, 1, 1, DS.baseline_alpha * a_d), lw=DS.baseline_width * sc)
            axd.text(*DS.xlabel_pos, r"$k$", color=(*DIM, a_d), fontsize=DS.xlabel_size * sc, transform=axd.transAxes, ha="right", va="top")
            axd.text(*DS.limit_label_pos, tx["l_cont"], color=(*BLUE, a_d), fontsize=DS.limit_label_size * sc, ha="right", va="center")
            axd.text(*DS.limit_lattice_label_pos, tx["l_latt"], color=(*DIM, a_d), fontsize=DS.limit_label_size * sc, ha="right", va="top")
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
    out = args.out or HERE / "media" / f"coupled_chain_{args.lang}.mp4"
    out.parent.mkdir(parents=True, exist_ok=True)
    if args.snapshot is not None:
        render(out.with_suffix(".png"), (V.width, V.height), V.fps, args.seconds, args.lang, snap=args.snapshot)
    elif args.preview:
        render(out.with_name(out.stem + "_preview.mp4"), (V.preview_width, V.preview_height), V.preview_fps, args.seconds, args.lang)
    else:
        render(out, (V.width, V.height), V.fps, args.seconds, args.lang)


if __name__ == "__main__":
    main()
