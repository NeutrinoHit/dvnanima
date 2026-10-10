r"""Chiral waves and the fermion mass (the book, the chapter "Where does a particle get its mass?", the section "Coupled chiral waves give birth to mass").

The (1+1)-dimensional Dirac equation in the chiral basis (psi_L, psi_R) is solved exactly (Fourier modes, see dirac1d.py):

    i (d_t + d_x) psi_R = m psi_L,      i (d_t - d_x) psi_L = m psi_R,      m = f v / sqrt(2)   (Yukawa constant f, Higgs condensate v).

Parts of the film:
 1. m = 0: psi_R runs to the right and psi_L to the left, both at the speed of light, independently (chirality = helicity = direction);
 2. the condensate switches on (a dial, shown in slow motion): psi_R becomes a source of psi_L and back, the chirality <gamma_5> = P_R - P_L
    oscillates with the frequency 2E, the packet slows down, a small part runs back on the lower branch E = -sqrt(p^2+m^2);
 3. a pure psi_R wave at rest for three Yukawa constants: full flips <gamma_5> = cos(2 m t), the heavier the faster (two coupled pendulums);
 4. wave packets made of the eigenvectors of H: psi_L and psi_R travel together, <gamma_5> = p/E = v is constant, the heavier the slower;
 5. why the "spring" exists: the mass term couples an SU(2) doublet to a singlet and is forbidden; the Yukawa term with the Higgs
    doublet gives m = f v / sqrt(2).

All the numbers are in config.toml and all the words in texts.toml (see ../dvconfig.py for --config / --set).

Usage:
    python chiral_fermion_mass.py --lang en            # film -> media/chiral_fermion_mass_en.mp4
    python chiral_fermion_mass.py --lang ru
    python chiral_fermion_mass.py --lang en --snapshot 40
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
sys.path.insert(0, str(HERE))
import dirac1d as D  # noqa: E402

CFG = load_config(HERE)

GR, HG, PK, SG, STY = CFG.grid, CFG.higgs, CFG.packet, CFG.stage, CFG.style
SR = GR.sample_rate
GRID = D.Grid(GR.length, GR.n)
DP = 2.0 * math.pi / GR.length                       # the spacing of the momenta

CARD_S = CFG.timeline.card_s
CARD_AT = list(CFG.timeline.card_at)                 # a card is shown before the content time
CARD_KEYS = [f"h{i}" for i in range(len(CARD_AT))]
_PB = list(CFG.timeline.part_bounds)
CONTENT_TOTAL = _PB[-1]
TOTAL = CONTENT_TOTAL + CARD_S * len(CARD_AT)
PARTS = [(_PB[i], _PB[i + 1]) for i in range(len(_PB) - 1)]
PART_CFG = [None, CFG.part1, CFG.part2, CFG.part3, CFG.part4, CFG.part5]


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


def part_of(tc: float) -> int:
    """1-based index of the part that contains the content time tc."""
    for i, (a, b) in enumerate(PARTS):
        if a <= tc < b:
            return i + 1
    return len(PARTS)


def smooth(x: float, a: float, b: float) -> float:
    u = min(max((x - a) / (b - a), 0.0), 1.0)
    return u * u * (3 - 2 * u)


def num(x: float, fmt: str, lang: str) -> str:
    s = format(x, fmt)
    return s.replace(".", "{,}") if lang == "ru" else s


def mass_of(f: float) -> float:
    """m = f v / sqrt(2)."""
    return f * HG.v / math.sqrt(2.0)


def phys_time(segments, u: float) -> float:
    """Physical time at the film time u when the rate (phys time per film second) is piecewise constant."""
    t = 0.0
    for a, b, r in segments:
        if u <= a:
            break
        t += (min(u, b) - a) * r
    return t


def pick(items, u: float):
    """The last [from, value] of a sorted list whose start does not exceed u."""
    out = items[0][1]
    for a, v in items:
        if a <= u:
            out = v
    return out


TEXT = {lang: load_texts(HERE, lang) for lang in ("en", "ru")}     # texts.toml


def fill(s: str, lang: str, **kw: float | str) -> str:
    """Replace @name@ by a number (tuple (value, format)) or a string."""
    for k, v in kw.items():
        s = s.replace(f"@{k}@", num(v[0], v[1], lang) if isinstance(v, tuple) else str(v))
    return s


# -------------------------------------------------------------------------------------------- the simulations

X_SEL = np.where(np.abs(GRID.x) <= SG.view_half + SG.view_pad)[0]
XD = GRID.x[X_SEL]
_order = np.argsort(GRID.p)
P_IDX = np.array([i for i in _order if abs(GRID.p[i]) <= CFG.dispersion.p_max])
P_D = GRID.p[P_IDX]

_SIM: dict = {}


def run_field(psi0: np.ndarray, mass_fn, times: np.ndarray) -> dict:
    """Evolve the real-space state psi0 (2, n) exactly and store everything the pictures need at every time of ``times``."""
    outk = D.evolve(D.to_fourier(psi0), GRID.p, mass_fn, times, GR.dt)
    n = len(times)
    R = {"t": np.asarray(times, float), "m": np.array([mass_fn(t) for t in times]), "PR": np.empty(n), "PL": np.empty(n), "xc": np.empty(n),
         "psiL": np.empty((n, len(X_SEL)), complex), "psiR": np.empty((n, len(X_SEL)), complex),
         "wp": np.empty((n, len(P_IDX))), "wm": np.empty((n, len(P_IDX))), "xp": np.empty(n), "xm": np.empty(n), "Wm": np.empty(n)}
    for k in range(n):
        px = D.to_real(outk[k])
        R["PR"][k], R["PL"][k] = D.chirality_weights(px, GRID.dx)
        R["xc"][k] = D.centroid(px, GRID.x, GRID.dx)
        R["psiL"][k], R["psiR"][k] = px[0][X_SEL], px[1][X_SEL]
        wp, wm = D.branch_weights(outk[k], GRID.p, R["m"][k], GRID.dx)
        R["wp"][k], R["wm"][k] = wp[P_IDX] / DP, wm[P_IDX] / DP        # the probability density in p
        R["xp"][k], _ = D.branch_centroid(outk[k], GRID.p, R["m"][k], GRID.x, GRID.dx, +1)
        R["xm"][k], R["Wm"][k] = D.branch_centroid(outk[k], GRID.p, R["m"][k], GRID.x, GRID.dx, -1)
    R["g"] = R["PR"] - R["PL"]
    R["wref"] = float(max(R["wp"].max(), R["wm"].max()))
    R["ymax"] = SG.headroom * float(max(np.abs(R["psiR"][0]).max(), np.abs(R["psiL"][0]).max()))
    R["mean_g"] = float(np.sum((R["wp"][-1] - R["wm"][-1]) * DP * D.velocity_ratio(GRID.p[P_IDX], R["m"][-1]))) if len(R["m"]) else 0.0
    return R


def frames_of(duration: float) -> int:
    return int(round(duration * SR)) + 1


def sim1() -> dict:
    if "s1" not in _SIM:
        P = CFG.part1
        n = frames_of(PARTS[0][1] - PARTS[0][0])
        t = P.rate * np.arange(n) / SR
        zero = lambda tt: 0.0
        r = run_field(D.chiral_packet(GRID, P.r_start, PK.sigma, +PK.carrier_p, "R"), zero, t)
        l = run_field(D.chiral_packet(GRID, P.l_start, PK.sigma, -PK.carrier_p, "L"), zero, t)
        s = dict(r)
        for k in ("psiL", "psiR", "wp", "wm"):
            s[k] = r[k] + l[k]
        s["gR"], s["gL"] = r["g"], l["g"]
        s["wref"] = float(max(s["wp"].max(), s["wm"].max()))
        _SIM["s1"] = s
    return _SIM["s1"]


def sim2() -> dict:
    if "s2" not in _SIM:
        P = CFG.part2
        n = frames_of(PARTS[1][1] - PARTS[1][0])
        t = np.array([phys_time(P.segments, k / SR) for k in range(n)])
        m_end = mass_of(P.f)
        mass = lambda tt: m_end * D.smoothstep(tt, P.ramp_start, P.ramp_start + P.ramp_duration)
        s = run_field(D.chiral_packet(GRID, PK.start_x, PK.sigma, PK.carrier_p, "R"), mass, t)
        s["m_end"] = m_end
        _SIM["s2"] = s
    return _SIM["s2"]


def rest_packet() -> np.ndarray:
    return D.chiral_packet(GRID, 0.0, PK.rest_sigma, 0.0, "R", math.radians(PK.rest_phase_deg))


def sim3(i: int) -> dict:
    key = ("s3", i)
    if key not in _SIM:
        P = CFG.part3
        run = P.runs[i]
        n = frames_of(run["to"] - run["from"])
        m = mass_of(run["f"])
        _SIM[key] = run_field(rest_packet(), lambda tt: m, P.rate * np.arange(n) / SR)
    return _SIM[key]


def sim4(i: int) -> dict:
    key = ("s4", i)
    if key not in _SIM:
        P = CFG.part4
        n = frames_of(PARTS[3][1] - PARTS[3][0])
        m = mass_of(P.f_values[i])
        u = np.arange(n) / SR
        t = P.rate * np.maximum(u - P.lead_s, 0.0)
        _SIM[key] = run_field(D.eigen_packet(GRID, PK.start_x, PK.sigma, PK.carrier_p, m, +1), lambda tt: m, t)
    return _SIM[key]


def sim5() -> dict:
    if "s5" not in _SIM:
        P = CFG.part5
        n = frames_of(PARTS[4][1] - PARTS[4][0])
        m = mass_of(P.f)
        _SIM["s5"] = run_field(rest_packet(), lambda tt: m, P.rate * np.arange(n) / SR)
    return _SIM["s5"]


def idx(u: float, n: int) -> int:
    return int(min(max(round(u * SR), 0), n - 1))


# -------------------------------------------------------------------------------------------- the state of a frame

def formulas_of(P, u: float) -> list:
    return [(f["key"], smooth(u, *f["appear"]), f["x"], f["row"], f["color"]) for f in P.formulas]


def dial_of(v: float, f: float, alpha: float) -> dict:
    return {"v": v, "f": f, "m": f * v / math.sqrt(2.0), "alpha": alpha}


def state1(u: float) -> dict:
    P, S = CFG.part1, sim1()
    i = idx(u, len(S["t"]))
    t = S["t"][: i + 1]
    xr, xl = P.r_start + S["t"][i], P.l_start - S["t"][i]
    return {
        "cap": pick(P.captions, u),
        "formulas": formulas_of(P, u),
        "dial": dial_of(0.0, HG.f_ref, 1.0),
        "lanes": {"x": XD, "psiR": S["psiR"][i], "psiL": S["psiL"][i], "ymax": S["ymax"], "m": 0.0, "labels": ("lab_R_arrow", "lab_L_arrow"), "light": None,
                  "cx": None, "alpha": 1.0},
        "chir": {"xlim": (0.0, float(S["t"][-1])), "curves": [{"t": t, "g": S["gR"][: i + 1], "color": STY.warm, "fill": False}, {"t": t, "g": S["gL"][: i + 1], "color": STY.cold, "fill": False}],
                 "flat": [("flat_R", +1, STY.warm), ("flat_L", -1, STY.cold)], "value": None, "alpha": smooth(u, 1.0, 2.0)},
        "disp": {"kind": "chiral", "m": 0.0, "wp": S["wp"][i], "wm": S["wm"][i], "wref": S["wref"], "cone_labels": True, "alpha": smooth(u, 0.3, 1.3)},
        "slow": 0.0,
    }


def state2(u: float) -> dict:
    P, S = CFG.part2, sim2()
    i = idx(u, len(S["t"]))
    tnow = S["t"][i]
    m = S["m"][i]
    dm = m / S["m_end"]
    x_light = PK.start_x + tnow
    slow = smooth(u, P.slow_label[0] - 0.3, P.slow_label[0]) * (1.0 - smooth(u, P.slow_label[1], P.slow_label[1] + 0.3))
    lo = max(0.0, tnow - P.window + P.window_lead)
    sel = S["t"][: i + 1] >= lo
    t, g = S["t"][: i + 1][sel], S["g"][: i + 1][sel]
    mean_a = smooth(u, P.mean_from, P.mean_from + 1.0)
    E = math.sqrt(PK.carrier_p ** 2 + S["m_end"] ** 2)
    return {
        "cap": pick(P.captions, u),
        "formulas": formulas_of(P, u),
        "dial": dial_of(HG.v_max * dm, P.f, 1.0),
        "lanes": {"x": XD, "psiR": S["psiR"][i], "psiL": S["psiL"][i], "ymax": S["ymax"], "m": m, "labels": ("lab_R", "lab_L"), "light": x_light,
                  "cx": float(S["xc"][i]), "cx_alpha": smooth(u, P.centroid_from, P.centroid_from + 0.8), "alpha": 1.0,
                  "branches": (float(S["xp"][i]), float(S["xm"][i]), smooth(u, P.branch_from, P.branch_from + 0.8))},
        "chir": {"xlim": (lo, lo + P.window), "curves": [{"t": t, "g": g, "color": STY.light, "fill": True}], "value": float(g[-1]),
                 "mean": S["mean_g"], "mean_alpha": mean_a, "mean_from": P.ramp_start + P.ramp_duration, "freq": 2.0 * E, "freq_alpha": smooth(u, P.freq_from, P.freq_from + 1.0), "alpha": 1.0},
        "disp": {"kind": "chiral", "m": m, "wp": S["wp"][i], "wm": S["wm"][i], "wref": S["wref"], "cone_labels": False, "alpha": 1.0},
        "slow": slow,
    }


def run_index3(u: float) -> int:
    runs = CFG.part3.runs
    return max(i for i, r in enumerate(runs) if r["from"] <= u)


def state3(u: float) -> dict:
    P = CFG.part3
    k = run_index3(u)
    run = P.runs[k]
    S = sim3(k)
    ul = u - run["from"]
    i = idx(ul, len(S["t"]))
    dur = run["to"] - run["from"]
    a = smooth(ul, 0.0, P.fade_s) * (1.0 - smooth(ul, dur - P.fade_s, dur))
    # the dial turns to the next Yukawa constant at the end of the run
    f = run["f"]
    if k + 1 < len(P.runs):
        f = f + (P.runs[k + 1]["f"] - f) * smooth(ul, dur - P.dial_turn_s, dur)
    curves = []
    labels = []
    for j in range(k + 1):
        Sj = sim3(j)
        last = j == k
        n = i + 1 if last else len(Sj["t"])
        curves.append({"t": Sj["t"][:n], "g": Sj["g"][:n], "color": STY.mass_colors[j], "fill": last, "old": not last})
        labels.append((j, mass_of(P.runs[j]["f"])))
    return {
        "cap": pick(P.captions, u),
        "formulas": formulas_of(P, u),
        "dial": dial_of(HG.v, f, 1.0),
        "lanes": {"x": XD, "psiR": S["psiR"][i], "psiL": S["psiL"][i], "ymax": S["ymax"], "m": S["m"][i], "labels": ("lab_R", "lab_L"), "light": None, "cx": None,
                  "alpha": a},
        "chir": {"xlim": (0.0, float(sim3(k)["t"][-1])), "curves": curves, "value": float(S["g"][i]), "runs": labels, "alpha": 1.0},
        "disp": {"kind": "chiral", "m": S["m"][i], "wp": S["wp"][i], "wm": S["wm"][i], "wref": S["wref"], "cone_labels": False, "alpha": 1.0, "gap_always": True},
        "slow": 0.0,
        "run": k,
    }


def state4(u: float) -> dict:
    P = CFG.part4
    sims = [sim4(i) for i in range(len(P.f_values))]
    i = idx(u, len(sims[0]["t"]))
    tnow = sims[0]["t"][i]
    ro = smooth(u, P.readout_from, P.readout_from + 0.8)
    tr = []
    for j, S in enumerate(sims):
        tr.append({"psiR": S["psiR"][i], "psiL": S["psiL"][i], "m": S["m"][i], "f": P.f_values[j], "xc": float(S["xc"][i]), "g": float(S["g"][i]),
                   "color": STY.mass_colors[j]})
    curves = [{"t": S["t"][: i + 1], "g": S["g"][: i + 1], "color": STY.mass_colors[j], "fill": False} for j, S in enumerate(sims)]
    return {
        "cap": pick(P.captions, u),
        "formulas": formulas_of(P, u),
        "dial": None,
        "tracks": {"items": tr, "x_light": PK.start_x + tnow, "ro": ro, "rho_ref": float(np.abs(sims[0]["psiR"][0]).max() ** 2)},
        "chir": {"xlim": (0.0, float(sims[0]["t"][-1])), "curves": curves, "value": None, "alpha": 1.0, "tracks": True},
        "disp": {"kind": "tracks", "ms": [S["m"][0] for S in sims], "p0": PK.carrier_p, "colors": STY.mass_colors, "alpha": 1.0},
        "slow": 0.0,
    }


def state5(u: float) -> dict:
    P, S = CFG.part5, sim5()
    i = idx(u, len(S["t"]))
    rel = smooth(u, *P.relabel)
    rows = [(r["key"], smooth(u, r["from"], r["from"] + P.row_fade_s), r["y"]) for r in P.rows]
    return {
        "cap": pick(P.captions, u),
        "formulas": [],
        "rows": rows,
        "dial": dial_of(HG.v, P.f, 1.0),
        "lanes": {"x": XD, "psiR": S["psiR"][i], "psiL": S["psiL"][i], "ymax": S["ymax"], "m": S["m"][i], "labels": ("lab_R", "lab_L"), "relabel": rel,
                  "light": None, "cx": None, "alpha": 1.0},
        "chir": None,
        "disp": None,
        "pot": smooth(u, P.rows[-1]["from"], P.rows[-1]["from"] + P.row_fade_s),
        "slow": 0.0,
    }


STATES = [None, state1, state2, state3, state4, state5]


def state(t: float) -> dict:
    """Everything that is drawn at the content time t."""
    part = part_of(t)
    st = STATES[part](t - PARTS[part - 1][0])
    st["part"] = part
    st["t"] = t
    return st


# -------------------------------------------------------------------------------------------- the picture

def clusters(dens: np.ndarray, wref: float, cutoff: float) -> list[tuple[float, float]]:
    """(mean momentum, probability) of every group of neighbouring momenta whose density exceeds cutoff * wref."""
    mask = dens > cutoff * wref
    out, i, n = [], 0, len(dens)
    while i < n:
        if mask[i]:
            j = i
            while j + 1 < n and mask[j + 1]:
                j += 1
            w = float(np.sum(dens[i:j + 1]))
            out.append((float(np.sum(P_D[i:j + 1] * dens[i:j + 1]) / w), w * DP))
            i = j + 1
        else:
            i += 1
    return out


def mix(a, b, w: float) -> np.ndarray:
    """(1 - w) a + w b for two colours."""
    return (1.0 - w) * np.asarray(a, float) + w * np.asarray(b, float)


def chirality_colors(c: np.ndarray) -> np.ndarray:
    """cold (psi_L, c = -1) to warm (psi_R, c = +1)."""
    w = 0.5 * (1.0 + np.clip(c, -1.0, 1.0))
    return (1.0 - w)[:, None] * np.array(STY.cold)[None, :] + w[:, None] * np.array(STY.warm)[None, :]


class Painter:
    """Draws the frames (one Figure that is reused); ``text_boxes`` reports where every text of the last frame is."""

    def __init__(self, lang: str, size: tuple[int, int]) -> None:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        self.plt = plt
        self.lang = lang
        self.tx = TEXT[lang]
        V, LY = CFG.video, CFG.layout
        self.W, self.H = size
        self.sc = self.H / V.reference_height
        dpi = V.dpi
        self.fig = plt.figure(figsize=(self.W / dpi, self.H / dpi), dpi=dpi, facecolor=STY.background)
        mk = lambda box: self.fig.add_axes(box, facecolor="none")
        self.ax_r, self.ax_l, self.ax_gap = mk(LY.lane_r_axes), mk(LY.lane_l_axes), mk(LY.gap_axes)
        self.ax_tr, self.ax_ch, self.ax_ds, self.ax_dl = mk(LY.track_axes), mk(LY.chirality_axes), mk(LY.dispersion_axes), mk(LY.dial_axes)
        self.ax_pot = mk(LY.potential_axes)
        self.part_fade = 1.0
        self.bg = np.array([int(STY.background[1:3], 16), int(STY.background[3:5], 16), int(STY.background[5:7], 16), 255], np.float32)

    # ---- helpers
    def T(self, key: str, **kw) -> str:
        return fill(self.tx[key], self.lang, **kw)

    def n(self, x: float, fmt: str) -> tuple[float, str]:
        return (x, fmt)

    def spring(self, ax, p0, p1) -> np.ndarray:
        """A coil spring between two points of the axes, built in pixels (the same coil radius for any length)."""
        S = CFG.springs
        P0, P1 = ax.transData.transform(p0), ax.transData.transform(p1)
        d = P1 - P0
        L = float(np.hypot(*d))
        u = d / L
        nrm = np.array([-u[1], u[0]])
        lead = min(S.lead_px * self.sc, S.lead_max * L)
        s_ = np.linspace(0.0, L, S.points)
        body = np.clip((s_ - lead) / max(L - 2 * lead, 1e-6), 0.0, 1.0)
        ramp = np.clip(np.minimum(body, 1.0 - body) / S.ramp, 0.0, 1.0)
        off = S.amplitude_px * self.sc * ramp * np.sin(2.0 * math.pi * S.turns * body) * ((s_ >= lead) & (s_ <= L - lead))
        pix = P0[None, :] + s_[:, None] * u[None, :] + off[:, None] * nrm[None, :]
        return ax.transData.inverted().transform(pix)

    # ---- the frame
    def draw(self, t_film: float, total: float) -> np.ndarray:
        V, LY, F = CFG.video, CFG.layout, CFG.fonts
        tx, sc, fig = self.tx, self.sc, self.fig
        TXT, DIM, GOLD = tuple(STY.text), tuple(STY.dim), tuple(STY.gold)
        t, card, cprog = timeline(t_film / (total / TOTAL))
        fig.texts.clear()
        for ar in (self.ax_r, self.ax_l, self.ax_gap, self.ax_tr, self.ax_ch, self.ax_ds, self.ax_dl, self.ax_pot):
            ar.clear()
            ar.axis("off")
        if card is not None:
            a_c = min(smooth(cprog, *CFG.timeline.card_fade_in), 1.0 - smooth(cprog, *CFG.timeline.card_fade_out))
            if card == 0:
                fig.text(0.5, LY.card_title_y, tx["h0"], color=(*STY.card_title_color, a_c), fontsize=F.card_title * sc, ha="center", va="center")
                fig.text(0.5, LY.card_subtitle_y, tx["h0s"], color=(*DIM, a_c), fontsize=F.card_subtitle * sc, ha="center", va="center")
            else:
                fig.text(0.5, LY.card_number_y, f"{card}", color=(*GOLD, STY.card_number_alpha * a_c), fontsize=F.card_number * sc, ha="center", va="center")
                fig.text(0.5, LY.card_chapter_y, tx[CARD_KEYS[card]], color=(*STY.card_title_color, a_c), fontsize=F.card_chapter * sc, ha="center", va="center")
                fig.text(0.5, LY.card_chapter_sub_y, tx[CARD_KEYS[card] + "s"], color=(*DIM, a_c), fontsize=F.card_chapter_sub * sc, ha="center", va="center")
        else:
            st = state(t)
            self.draw_content(st, t)
            ga = smooth(t - PARTS[st["part"] - 1][0], *CFG.timeline.part_fade_in)
            if ga < 1.0:
                self.part_fade = ga
        return self.finish(t_film, total)

    def finish(self, t_film: float, total: float) -> np.ndarray:
        V = CFG.video
        self.fig.canvas.draw()
        frame = np.asarray(self.fig.canvas.buffer_rgba()).astype(np.float32)
        if self.part_fade < 1.0:
            k0 = int(round(CFG.timeline.part_fade_top * self.H))
            frame[k0:] = self.bg + (frame[k0:] - self.bg) * self.part_fade
            self.part_fade = 1.0
        fade_io = min(smooth(t_film, 0.0, V.fade_s), 1.0 - smooth(t_film, total - V.fade_s, total))
        if fade_io < 1.0:
            frame = self.bg + (frame - self.bg) * fade_io
        return frame.clip(0, 255).astype(np.uint8)

    def draw_content(self, st: dict, t: float) -> None:
        LY, F = CFG.layout, CFG.fonts
        tx, sc, fig = self.tx, self.sc, self.fig
        TXT, DIM = tuple(STY.text), tuple(STY.dim)
        part = st["part"]
        fig.text(*LY.title_pos, tx["title"], color=tuple(STY.title_color), fontsize=F.title * sc)
        fig.text(*LY.chapter_label_pos, f"{part}/{len(PARTS)}   " + tx[CARD_KEYS[part]], color=(*DIM, STY.chapter_label_alpha), fontsize=F.chapter_label * sc, ha="right")
        for key, alpha, x, row, color in st["formulas"]:
            if alpha > LY.formula_min_alpha:
                fig.text(x, LY.formula_y - LY.formula_row * row, tx[key], color=(*STY[color], STY.formula_alpha * alpha), fontsize=F.formula * sc)
        for key, alpha, y in st.get("rows", []):
            if alpha > LY.formula_min_alpha:
                fig.text(LY.rows_x, y, tx[key], color=(*TXT, STY.formula_alpha * alpha), fontsize=F.row * sc, va="center")
        if st["cap"]:
            fig.text(*LY.caption_pos, self.cap_text(st), color=(*TXT, STY.caption_alpha), fontsize=F.caption * sc)
        fig.text(SG.note_pos[0], SG.note_pos[1], tx["note"], color=(*DIM, SG.note_alpha), fontsize=SG.note_size * sc, ha="left", va="center")
        if st.get("lanes"):
            self.draw_lanes(st["lanes"], st["slow"])
        if st.get("tracks"):
            self.draw_tracks(st["tracks"])
        if st.get("chir"):
            self.draw_chirality(st["chir"])
        if st.get("disp"):
            self.draw_dispersion(st["disp"])
        if st.get("pot") is not None:
            self.draw_potential(st["pot"])
        self.draw_dial(st["dial"])

    def cap_text(self, st: dict) -> str:
        return self.T(st["cap"])

    # ---- the lanes
    def draw_lanes(self, L: dict, slow: float) -> None:
        sc, tx, LY = self.sc, self.tx, CFG.layout
        ymax, a = L["ymax"], L["alpha"]
        for ax, z, col, name, side in ((self.ax_r, L["psiR"], STY.warm, L["labels"][0], "R"), (self.ax_l, L["psiL"], STY.cold, L["labels"][1], "L")):
            ax.set_xlim(-SG.view_half, SG.view_half)
            ax.set_ylim(-ymax, ymax)
            ax.axis("off")
            ax.add_patch(self.plt.Rectangle((-SG.view_half, -ymax), 2 * SG.view_half, 2 * ymax, facecolor=(*col, SG.sheet_alpha), edgecolor="none", zorder=0))
            ax.plot([-SG.view_half, SG.view_half], [0, 0], color=(1, 1, 1, SG.baseline_alpha), lw=SG.baseline_width * sc, ls=(0, tuple(SG.baseline_dash)), zorder=1)
            edge_y = -ymax if side == "R" else ymax
            ax.plot([-SG.view_half, SG.view_half], [edge_y, edge_y], color=(*col, SG.sheet_edge_alpha), lw=SG.sheet_edge_width * sc, zorder=1, clip_on=False)
            env = np.abs(z)
            ax.fill_between(L["x"], -env, env, color=(*col, SG.envelope_alpha * a), lw=0, zorder=2)
            ax.plot(L["x"], env, color=(*col, SG.outline_alpha * a), lw=SG.outline_width * sc, zorder=3)
            ax.plot(L["x"], -env, color=(*col, SG.outline_alpha * a), lw=SG.outline_width * sc, zorder=3)
            ax.plot(L["x"], z.real, color=(*mix(col, (1, 1, 1), 0.25), a), lw=SG.line_width * sc, zorder=4)
            if L["light"] is not None and abs(L["light"]) <= SG.view_half:
                ax.plot([L["light"]] * 2, [-ymax, ymax], color=(1, 1, 1, SG.light_alpha), lw=SG.light_width * sc, ls=(0, tuple(SG.light_dash)), zorder=5)
            if L.get("cx") is not None and abs(L["cx"]) <= SG.view_half:
                ca = L["cx_alpha"]
                ax.plot([L["cx"]] * 2, [-ymax, ymax], color=(*STY.gold, SG.centroid_alpha * ca), lw=SG.centroid_width * sc, zorder=5)
        # the labels of the lanes: in the left margin
        for ax, col, key, side in ((self.ax_r, STY.warm, L["labels"][0], "R"), (self.ax_l, STY.cold, L["labels"][1], "L")):
            bb = ax.get_position()
            yc = bb.y0 + 0.5 * bb.height
            rel = L.get("relabel")
            k2 = ("lab_e_R" if side == "R" else "lab_L_doublet")
            if rel is None:
                self.fig.text(LY.lane_label_x, yc, tx[key], color=(*col, 1.0), fontsize=SG.label_size * sc, ha="right", va="center")
            else:
                self.fig.text(LY.lane_label_x, yc, tx[key], color=(*col, 1.0 - rel), fontsize=SG.label_size * sc, ha="right", va="center")
                self.fig.text(LY.lane_label_x, yc, tx[k2], color=(*col, rel), fontsize=SG.label_size * sc, ha="right", va="center")
                if rel > STY.panel_cutoff:
                    ax.text(SG.label_pos[0], SG.label_pos[1], tx["tag_R" if side == "R" else "tag_L"], color=(*col, rel), fontsize=SG.tag_size * sc, ha="left", va="center",
                            transform=ax.transAxes, zorder=7)
        if L["light"] is not None and abs(L["light"]) <= SG.view_half:
            self.ax_r.text(L["light"] - 0.15, ymax * SG.light_label_y, tx["light"], color=(1, 1, 1, SG.light_alpha + 0.3), fontsize=SG.light_label_size * sc, ha="right", va="top", zorder=6)
        if L.get("cx") is not None and L["cx_alpha"] > CFG.style.panel_cutoff and abs(L["cx"]) <= SG.view_half:
            self.ax_l.text(L["cx"] + 0.15, -ymax * SG.light_label_y, tx["centroid"], color=(*STY.gold, L["cx_alpha"]), fontsize=SG.centroid_label_size * sc, ha="left", va="bottom", zorder=6)
        if L.get("branches") is not None and L["branches"][2] > STY.panel_cutoff:
            xp, xm, ba = L["branches"]
            for xb, key in ((xp, "br_up"), (xm, "br_dn")):
                if abs(xb) <= SG.view_half:
                    self.ax_l.text(xb, ymax * SG.branch_label_y, tx[key], color=(*STY.text, ba), fontsize=SG.branch_label_size * sc, ha="center", va="center", zorder=6)
        if slow > STY.panel_cutoff:
            self.ax_r.text(SG.slow_pos[0], SG.slow_pos[1], tx["slow"], color=(*STY.gold, slow), fontsize=SG.slow_size * sc, ha="right", va="top", transform=self.ax_r.transAxes)
        # springs: the coupling by the condensate
        axg = self.ax_gap
        axg.set_xlim(-SG.view_half, SG.view_half)
        axg.set_ylim(0.0, 1.0)
        axg.axis("off")
        S = CFG.springs
        s = float(np.clip(L["m"] / S.m_ref, 0.0, 1.6))
        if s > 1e-3:
            from matplotlib.collections import LineCollection
            xs = np.linspace(-SG.view_half + S.x_margin, SG.view_half - S.x_margin, S.count)
            segs = [self.spring(axg, (x, 1.0), (x, 0.0)) for x in xs]
            axg.add_collection(LineCollection(segs, colors=[(*S.color, min(S.alpha_min + S.alpha_gain * s, 1.0) * L["alpha"])] * len(segs),
                                              linewidths=(S.width_min + S.width_gain * min(s, 1.0)) * sc, capstyle="round"))

    # ---- the tracks (part 4)
    def draw_tracks(self, T: dict) -> None:
        sc, tx, LY = self.sc, self.tx, CFG.layout
        ax = self.ax_tr
        ax.set_xlim(-SG.view_half, SG.view_half)
        ax.set_ylim(0.0, float(len(T["items"])))
        ax.axis("off")
        for it, yb in zip(T["items"], SG.track_y):
            col = it["color"]
            ax.plot([-SG.view_half, SG.view_half], [yb, yb], color=(1, 1, 1, SG.track_base_alpha), lw=SG.baseline_width * sc, ls=(0, tuple(SG.baseline_dash)), zorder=1)
            rR = np.abs(it["psiR"]) ** 2 / T["rho_ref"] * SG.density_scale * 0.5
            rL = np.abs(it["psiL"]) ** 2 / T["rho_ref"] * SG.density_scale * 0.5
            ax.fill_between(XD, yb, yb + rR, color=(*STY.warm, SG.track_ribbon_alpha), lw=0, zorder=2)
            ax.fill_between(XD, yb - rL, yb, color=(*STY.cold, SG.track_ribbon_alpha), lw=0, zorder=2)
            ax.plot(XD, yb + rR, color=(*STY.warm, SG.track_outline_alpha), lw=SG.track_outline_width * sc, zorder=3)
            ax.plot(XD, yb - rL, color=(*STY.cold, SG.track_outline_alpha), lw=SG.track_outline_width * sc, zorder=3)
            if abs(it["xc"]) <= SG.view_half:
                ax.scatter([it["xc"]], [yb], s=SG.track_dot_size * sc * sc, color=(*col, 1.0), zorder=6, edgecolors="none")
            ym = ax.transData.transform((0, yb))[1] / self.H
            self.fig.text(SG.track_label_x, ym + SG.track_label_dy, self.T("tr_m", m=self.n(it["m"], ".1f")), color=(*col, 1.0), fontsize=SG.track_label_size * sc, ha="right", va="center")
            if T["ro"] > STY.panel_cutoff:
                self.fig.text(SG.track_label_x, ym - SG.track_label_dy, self.T("tr_v", v=self.n(it["g"], ".2f")), color=(*col, T["ro"]), fontsize=SG.track_readout_size * sc, ha="right", va="center")
        xl = T["x_light"]
        if abs(xl) <= SG.view_half:
            ax.plot([xl, xl], [0.0, float(len(T["items"]))], color=(1, 1, 1, SG.track_light_alpha), lw=SG.light_width * sc, ls=(0, tuple(SG.light_dash)), zorder=5)
            ax.text(xl - 0.15, float(len(T["items"])) - 0.04, tx["light"], color=(1, 1, 1, SG.light_alpha + 0.3), fontsize=SG.light_label_size * sc, ha="right", va="top", zorder=6)

    # ---- the chirality plot
    def draw_chirality(self, C: dict) -> None:
        sc, tx, CH = self.sc, self.tx, CFG.chirality
        ax = self.ax_ch
        a = C["alpha"]
        if a <= STY.panel_cutoff:
            return
        ax.set_xlim(*C["xlim"])
        ax.set_ylim(-CH.ylim, CH.ylim)
        ax.axis("off")
        ax.plot(C["xlim"], [0, 0], color=(1, 1, 1, CH.base_alpha * a), lw=CH.base_width * sc, ls=(0, tuple(CH.base_dash)))
        yt = ax.get_yaxis_transform()
        for yv in CH.tick_values:
            col = STY.warm if yv > 0 else (STY.cold if yv < 0 else STY.dim)
            ax.text(CH.tick_x, yv, f"${'+' if yv > 0 else ('-' if yv < 0 else '')}{abs(int(yv))}$", color=(*col, a), fontsize=CH.tick_size * sc, ha="right", va="center", transform=yt)
        ax.text(*CH.title_pos, tx["chir_title"], color=(*STY.dim, a), fontsize=CH.title_size * sc, transform=ax.transAxes, ha="left", va="bottom")
        if C.get("mean") is not None and C["mean_alpha"] > STY.panel_cutoff:
            m0 = C["mean_from"]
            ax.plot([max(m0, C["xlim"][0]), C["xlim"][1]], [C["mean"]] * 2, color=(*STY.gold, CH.mean_alpha * C["mean_alpha"]), lw=CH.mean_width * sc, ls=(0, tuple(CH.mean_dash)), zorder=4)
            ax.text(*CH.mean_label_pos, self.T("chir_mean", g=self.n(C["mean"], ".2f")), color=(*STY.gold, C["mean_alpha"]), fontsize=CH.mean_label_size * sc, transform=ax.transAxes,
                    ha="right", va="bottom")
        if C.get("freq") is not None and C["freq_alpha"] > STY.panel_cutoff:
            ax.text(*CH.freq_pos, self.T("chir_freq", w=self.n(C["freq"], ".2f")), color=(*TXT_(), C["freq_alpha"]), fontsize=CH.freq_size * sc, transform=ax.transAxes, ha="right", va="bottom")
        if C.get("value") is not None:
            ax.text(*CH.value_pos, self.T("chir_val", g=self.n(C["value"], "+.2f")), color=(*STY.text, a), fontsize=CH.value_size * sc, transform=ax.transAxes, ha="right", va="bottom")
        for c in C["curves"]:
            t, g = np.asarray(c["t"]), np.asarray(c["g"])
            if len(t) < 2:
                continue
            if c.get("old"):
                ax.plot(t, g, color=(*c["color"], CH.old_alpha * a), lw=CH.old_width * sc, ls=(0, tuple(CH.old_dash)), zorder=2)
                continue
            if c.get("fill"):
                ax.fill_between(t, 0, g, where=g >= 0, interpolate=True, color=(*STY.warm, CH.fill_alpha * a), lw=0, zorder=2)
                ax.fill_between(t, 0, g, where=g < 0, interpolate=True, color=(*STY.cold, CH.fill_alpha * a), lw=0, zorder=2)
            ax.plot(t, g, color=(*c["color"], a), lw=CH.line_width * sc, zorder=3, solid_capstyle="round")
        for key, yv, col in C.get("flat", []):
            ax.text(C["xlim"][1] * CH.flat_x, yv * CH.flat_y, tx[key], color=(*col, CH.flat_alpha * a), fontsize=CH.flat_label_size * sc, ha="right", va="center")
        for rj, mj in C.get("runs", []):
            ax.text(CH.run_label_x[rj], CH.run_label_y, self.T("run_label", m=self.n(mj, ".1f")), color=(*STY.mass_colors[rj], a),
                    fontsize=CH.run_label_size * sc, transform=ax.transAxes, ha="right", va="bottom")

    # ---- the dispersion plot
    def draw_dispersion(self, Dd: dict) -> None:
        from matplotlib.collections import LineCollection
        sc, tx, DS = self.sc, self.tx, CFG.dispersion
        ax, a = self.ax_ds, Dd["alpha"]
        if a <= STY.panel_cutoff:
            return
        ax.set_xlim(-DS.p_max, DS.p_max)
        ax.set_ylim(-DS.e_max, DS.e_max)
        ax.axis("off")
        ax.plot([-DS.p_max, DS.p_max], [0, 0], color=(1, 1, 1, DS.axis_alpha * a), lw=DS.axis_width * sc)
        ax.plot([0, 0], [-DS.e_max, DS.e_max], color=(1, 1, 1, DS.axis_alpha * a), lw=DS.axis_width * sc)
        ax.text(DS.xlabel_pos[0] * DS.p_max, DS.xlabel_pos[1], r"$p$", color=(*STY.dim, a), fontsize=DS.xlabel_size * sc, ha="right", va="top")
        ax.text(DS.ylabel_pos[0], DS.ylabel_pos[1] * DS.e_max, r"$E$", color=(*STY.dim, a), fontsize=DS.ylabel_size * sc, ha="left", va="top")
        pp = np.linspace(-DS.p_max, DS.p_max, DS.curve_points)
        if Dd["kind"] == "chiral":
            ax.text(*DS.title_pos, tx["disp_title"], color=(*STY.text, a), fontsize=DS.title_size * sc, transform=ax.transAxes, ha="left", va="bottom")
            m = Dd["m"]
            for sign, name in ((1, "up"), (-1, "dn")):
                ax.plot(pp, sign * pp, color=(1, 1, 1, DS.bare_alpha * a), lw=DS.bare_width * sc, ls=(0, tuple(DS.bare_dash)), zorder=1)
            E = np.sqrt(pp ** 2 + m ** 2)
            pts_u, pts_d = np.column_stack([pp, E]), np.column_stack([pp, -E])
            with np.errstate(invalid="ignore", divide="ignore"):
                cu = np.where(E > 1e-9, pp / np.where(E > 1e-9, E, 1.0), 0.0)
            for pts, c, wb, dens in ((pts_u, cu, +1, Dd["wp"]), (pts_d, -cu, -1, Dd["wm"])):
                seg = np.stack([pts[:-1], pts[1:]], axis=1)
                ax.add_collection(LineCollection(seg, colors=[(*c_, DS.curve_alpha * a) for c_ in chirality_colors(0.5 * (c[:-1] + c[1:]))], linewidths=DS.curve_width * sc, zorder=3,
                                                 capstyle="round"))
                # where the wave packet sits: a dot on the branch, its area is the probability of that branch
                for pm, w in clusters(dens, Dd["wref"], DS.cluster_cutoff):
                    em = wb * math.sqrt(pm * pm + m * m)
                    col = chirality_colors(np.array([wb * pm / max(abs(em), 1e-9)]))[0]
                    ax.scatter([pm], [em], s=DS.dot_area * w * sc * sc, color=(*col, a), edgecolors=(1, 1, 1, DS.dot_edge_alpha * a), linewidths=DS.dot_edge_width * sc, zorder=6)
            if Dd["cone_labels"]:
                ax.text(DS.cone_label_p, DS.cone_label_e, tx["cone_R"], color=(*STY.warm, a), fontsize=DS.cone_label_size * sc, ha="center", va="center")
                ax.text(-DS.cone_label_p, DS.cone_label_e, tx["cone_L"], color=(*STY.cold, a), fontsize=DS.cone_label_size * sc, ha="center", va="center")
            if m > DS.gap_min:
                ax.annotate("", xy=(0.0, m), xytext=(0.0, -m), arrowprops={"arrowstyle": "<->", "color": (*STY.gold, DS.gap_alpha * a), "lw": DS.gap_width * sc, "shrinkA": 0, "shrinkB": 0}, zorder=6)
                ax.text(DS.gap_label_dx, 0.0, self.T("gap", m=self.n(2.0 * m, ".1f")), color=(*STY.gold, a), fontsize=DS.gap_label_size * sc, ha="left", va="center", zorder=7,
                        bbox={"facecolor": STY.background, "edgecolor": "none", "pad": 1.0, "alpha": 0.85})
        else:
            ax.text(*DS.title_pos, tx["disp_title_up"], color=(*STY.text, a), fontsize=DS.title_size * sc, transform=ax.transAxes, ha="left", va="bottom")
            for j, (m, col) in enumerate(zip(Dd["ms"], Dd["colors"])):
                E = np.sqrt(pp ** 2 + m ** 2)
                ax.plot(pp, E, color=(*col, DS.curve_alpha * a), lw=DS.curve_width * sc, zorder=3)
                ax.plot(pp, -E, color=(*col, DS.lower_alpha * a), lw=DS.curve_width * sc * DS.lower_width, ls=(0, tuple(DS.bare_dash)), zorder=2)
                p0 = Dd["p0"]
                E0 = math.sqrt(p0 * p0 + m * m)
                v = p0 / E0
                h = DS.tangent_half
                ax.plot([p0 - h, p0 + h], [E0 - v * h, E0 + v * h], color=(*col, DS.tangent_alpha * a), lw=DS.tangent_width * sc, zorder=4)
                ax.scatter([p0], [E0], s=DS.dot_area * sc * sc, color=(*col, a), zorder=6, edgecolors=(1, 1, 1, DS.dot_edge_alpha * a), linewidths=DS.dot_edge_width * sc)

    # ---- the Higgs potential (part 5, schematic)
    def draw_potential(self, a: float) -> None:
        sc, tx, PT = self.sc, self.tx, CFG.potential
        if a <= STY.panel_cutoff:
            return
        ax = self.ax_pot
        phi = np.linspace(-PT.phi_max, PT.phi_max, PT.points)
        V = (phi ** 2 - 1.0) ** 2
        ax.set_xlim(-PT.phi_max, PT.phi_max)
        ax.set_ylim(-PT.y_below, PT.y_max)
        ax.axis("off")
        ax.plot(phi, V, color=(*STY.dim, PT.curve_alpha * a), lw=PT.curve_width * sc)
        ax.plot([-PT.phi_max, PT.phi_max], [0, 0], color=(1, 1, 1, PT.axis_alpha * a), lw=PT.axis_width * sc)
        ax.scatter([1.0], [0.0], s=PT.ball_size * sc * sc, color=(*STY.gold, a), zorder=5, edgecolors="none")
        ax.text(*PT.title_pos, tx["pot_title"], color=(*STY.dim, a), fontsize=PT.title_size * sc, transform=ax.transAxes, ha="left", va="bottom")
        ax.text(1.0, -0.5 * PT.y_below, tx["pot_label"], color=(*STY.gold, a), fontsize=PT.label_size * sc, ha="center", va="center")

    # ---- the dial
    def draw_dial(self, d: dict | None) -> None:
        if d is None:
            return
        sc, tx, DL = self.sc, self.tx, CFG.dial
        ax, a = self.ax_dl, d["alpha"]
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis("off")
        ax.text(*DL.title_pos, tx["dial_title"], color=(*STY.dim, a), fontsize=DL.title_size * sc, transform=ax.transAxes, ha="left", va="bottom")
        for y, name, val, vmax in ((DL.row_y[0], "$v$", d["v"], HG.v_max), (DL.row_y[1], "$f$", d["f"], HG.f_max)):
            x0, x1 = DL.track_x
            ax.plot([x0, x1], [y, y], color=(*STY.dim, DL.track_alpha * a), lw=DL.track_width * sc, solid_capstyle="round")
            xk = x0 + (x1 - x0) * float(np.clip(val / vmax, 0.0, 1.0))
            ax.plot([x0, xk], [y, y], color=(*STY.gold, a), lw=DL.track_width * sc, solid_capstyle="round")
            ax.scatter([xk], [y], s=DL.knob_size * sc * sc, color=(*STY.gold, a), zorder=5, edgecolors="none")
            ax.text(DL.label_x, y, name, color=(*STY.gold, a), fontsize=DL.label_size * sc, ha="left", va="center")
            ax.text(DL.value_x, y, f"${num(val, '.2f', self.lang)}$", color=(*STY.text, a), fontsize=DL.value_size * sc, ha="left", va="center")
        ax.text(*DL.m_pos, self.T("dial_m", m=self.n(d["m"], ".2f")), color=(*STY.gold, a), fontsize=DL.m_size * sc, ha="left", va="bottom")

    def text_boxes(self, min_alpha: float = 0.15) -> list[tuple[str, tuple[float, float, float, float]]]:
        """(string, (x0, y0, x1, y1) in pixels) of every visible text of the last frame."""
        r = self.fig.canvas.get_renderer()
        out = []
        arts = list(self.fig.texts) + [t for ax in self.fig.axes for t in ax.texts]
        for t in arts:
            if not t.get_text().strip() or not t.get_visible():
                continue
            col = t.get_color()
            if isinstance(col, tuple) and len(col) == 4 and col[3] < min_alpha:
                continue
            bb = t.get_window_extent(r)
            out.append((t.get_text(), (bb.x0, bb.y0, bb.x1, bb.y1)))
        return out


def TXT_() -> tuple:
    return tuple(STY.text)


def render(out: Path, size: tuple[int, int], fps: int, total: float, lang: str, snap: float | None = None) -> None:
    if snap is None and not shutil.which("ffmpeg"):
        sys.exit("ffmpeg is required")
    W, H = size
    V = CFG.video
    painter = Painter(lang, size)
    frames = int(round(total * fps))
    ids = [int(round(snap * fps))] if snap is not None else range(frames)
    writer = None
    if snap is None:
        writer = subprocess.Popen(
            ["ffmpeg", "-y", "-v", "error", "-f", "rawvideo", "-pix_fmt", "rgba", "-s", f"{W}x{H}",
             "-r", str(fps), "-i", "-", "-c:v", "libx264", "-preset", V.preset, "-crf", str(V.crf),
             "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(out)], stdin=subprocess.PIPE)
    for k in ids:
        frame = painter.draw(k / fps, total)
        if writer is None:
            from PIL import Image
            Image.fromarray(frame).save(out)
        else:
            writer.stdin.write(frame.tobytes())
    if writer is not None:
        writer.stdin.close()
        writer.wait()
    painter.plt.close(painter.fig)
    print(f"wrote {out}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--lang", choices=("en", "ru"), default="en")
    ap.add_argument("--preview", action="store_true")
    ap.add_argument("--snapshot", type=float, default=None, help="one PNG at this film time (s)")
    ap.add_argument("--seconds", type=float, default=TOTAL)
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--config", type=Path, default=None, help="another configuration file instead of config.toml")
    ap.add_argument("--set", action="append", default=[], metavar="SECTION.KEY=VALUE", help="override one configuration value")
    args = ap.parse_args()
    V = CFG.video
    out = args.out or HERE / "media" / f"chiral_fermion_mass_{args.lang}.mp4"
    out.parent.mkdir(parents=True, exist_ok=True)
    if args.snapshot is not None:
        render(out.with_suffix(".png"), (V.width, V.height), V.fps, args.seconds, args.lang, snap=args.snapshot)
    elif args.preview:
        render(out.with_name(out.stem + "_preview.mp4"), (V.preview_width, V.preview_height), V.preview_fps, args.seconds, args.lang)
    else:
        render(out, (V.width, V.height), V.fps, args.seconds, args.lang)


if __name__ == "__main__":
    main()
