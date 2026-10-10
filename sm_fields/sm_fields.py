r"""The fields of the Standard Model as sheets of one stack, and what a particle, an interaction and an annihilation are.

The film follows the section "Quantum fields" of the first chapter of the book.

1. The table of the 17 fields of the Standard Model (6 quark, 3 charged-lepton, 3 neutrino fields, the gauge
   fields gamma, g, W, Z and the Higgs field) lifts into a stack of 17 sheets: every field is a separate sheet that
   fills the whole space.  A line through the stack marks one point of space: every field is present in every point.
2. All fields fluctuate a little; a particle is a wave packet on the sheet of its own field.  Two electron waves pass
   through each other: free waves of one field do not notice each other (linear superposition).
3. Zoom: the other fifteen sheets fly away up and down, two solid sheets are left, the electron field (height =
   |psi|^2) and the electromagnetic field (height = A_0).  The packets of the scalar-QED model of the book
   (qed_pair.py, the code of ../fields/scalar_qed): every packet creates its Coulomb-like field A^mu on the photon
   sheet, the field of the other packet shifts the phase of the wave, the gradient of the phase deflects the packet:
   e- e- repel, e- e+ attract.
4. e- e+ -> gamma gamma and back: the packets disappear from the electron sheet and photon waves appear on the photon
   sheet.  This part is schematic: the energy shares are E_e = cos^2(theta) E_0, E_gamma = sin^2(theta) E_0 with a
   smooth step theta(t); it is an illustration of the exchange of energy between two fields, not a solution of an
   equation.

Everything is schematic: the sheets are an image of different fields, not a literal layering of the vacuum.

All the numbers are in config.toml and all the words in texts.toml (see ../dvconfig.py for --config / --set).

Usage:
    python sm_fields.py --lang en            # film -> media/sm_fields_en.mp4
    python sm_fields.py --lang ru
    python sm_fields.py --lang en --snapshot 20
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

import qed_pair as qp  # noqa: E402

HERE = Path(__file__).resolve().parent
CFG = load_config(HERE)

_T = CFG.timeline
TOTAL = _T.total
T_CAM = tuple(_T.camera)        # the camera turns from the top view to the oblique one
T_LIFT = tuple(_T.lift)         # the tiles lift one after another into the stack (a cascade)
T_POINT = tuple(_T.point)       # a point of space pierces all the sheets
T_PULSE = _T.pulse              # a particle on the electron sheet
T_PASS = tuple(_T.passing)      # two electron waves pass through each other
T_ZOOM = tuple(_T.zoom)         # the other sheets fly away, two solid sheets are left
T_A = tuple(_T.part_a)          # e- e-: repulsion (scalar QED)
T_B = tuple(_T.part_b)          # e- e+: attraction (scalar QED)
T_C = tuple(_T.part_c)          # e- e+ -> gamma gamma and back
ZOOM_HALF, Z_E, Z_G = CFG.zoom.half, CFG.zoom.z_electron, CFG.zoom.z_photon      # half-size and heights of the two sheets in the zoom
A0_BASE = {"pp": CFG.heights.a0_base_equal, "pm": CFG.heights.a0_base_opposite}   # the far field of two equal charges is a constant: a gauge shift
H_E, H_G = CFG.heights.electron, CFG.heights.photon        # heights: scene units per unit of |psi|^2 and of A_0
SPACING = CFG.stack.spacing     # distance between the sheets of the stack
SIZE_TABLE, SIZE_STACK = CFG.stack.size_table, CFG.stack.size_stack


# ------------------------------------------------------------------- the fields

GROUPS = {name: tuple(rgb) for name, rgb in CFG.groups.to_dict().items()}

# name, group, column and row of the table (the Higgs sits in its own column)
FIELDS = [(f["name"], f["group"], f["col"], f["row"]) for f in CFG.fields]
STACK_ORDER = list(CFG.table.stack_order)
LABEL = {f["name"]: f["label"] for f in CFG.fields}
FIELD_SYMBOL = {f["name"]: f["symbol"] for f in CFG.fields}


def f1(a) -> float:
    return float(np.ravel(a)[0])


def lift_progress(name: str, t: float) -> float:
    """Progress (0 = in the table, 1 = in the stack) of the sheet 'name': the sheets lift one after another."""
    i = STACK_ORDER.index(name)
    n = len(STACK_ORDER) - 1
    dur = CFG.stack.lift_duration
    start = T_LIFT[0] + (T_LIFT[1] - T_LIFT[0] - dur) * i / n
    return smooth(t, start, start + dur)


def smooth(x: float, a: float, b: float) -> float:
    u = min(max((x - a) / (b - a), 0.0), 1.0)
    return u * u * (3 - 2 * u)


def table_center(col: float, row: float) -> tuple[float, float]:
    """Centre of a tile in the table (x to the right, y up), units of the sheet half-width."""
    S_ = CFG.stack
    return (col - S_.table_col_centre) * S_.table_col_pitch + S_.table_x_shift, (S_.table_row_centre - row) * S_.table_row_pitch + S_.table_y_shift


def stack_z(name: str, spacing: float = SPACING) -> float:
    i = STACK_ORDER.index(name)
    return (len(STACK_ORDER) / 2 - i) * spacing


# ------------------------------------------------------------------ wave shapes

def fluctuation(u: np.ndarray, v: np.ndarray, t: float, seed: int, amp: float = CFG.fluctuation.amplitude) -> np.ndarray:
    """Small random ripples: a few plane waves with random directions, frequencies and phases."""
    rng = np.random.default_rng(seed)
    h = np.zeros_like(u)
    FL = CFG.fluctuation
    for _ in range(FL.waves):
        th = rng.uniform(0, 2 * math.pi)
        k = rng.uniform(*FL.k_range)
        w = rng.uniform(*FL.omega_range)
        ph = rng.uniform(0, 2 * math.pi)
        h += np.sin(k * (u * math.cos(th) + v * math.sin(th)) - w * t + ph)
    return amp * h / 2.0


def packet(u: np.ndarray, v: np.ndarray, t: float, x0: float, y0: float, vx: float, sigma: float, k: float,
           amp: float) -> np.ndarray:
    """A moving wave packet: Gaussian envelope and a carrier along the motion."""
    xc = x0 + vx * t
    r2 = (u - xc) ** 2 + (v - y0) ** 2
    return amp * np.exp(-r2 / (2 * sigma ** 2)) * np.cos(k * (u - xc))


def ring(u: np.ndarray, v: np.ndarray, t: float, x0: float, y0: float, t0: float, c: float, w: float, k: float,
         amp: float) -> np.ndarray:
    """A ring wave spreading from (x0, y0), launched at t0."""
    if t < t0:
        return np.zeros_like(u)
    r = np.hypot(u - x0, v - y0)
    s = r - c * (t - t0)
    return amp * np.exp(-(s / w) ** 2) * np.cos(k * s) / np.sqrt(1.0 + CFG.stack_waves.ring_decay * r)


_AN = CFG.annihilation
T_C1, T_C2 = _AN.t_annihilation, _AN.t_creation          # times of the annihilation and of the pair creation, from the start of the part C
V_E, V_G = _AN.electron_speed, _AN.photon_speed          # speeds of the electron and of the photon packets on the sheets (plane units / s)
X_HALF = qp.LX / 2              # the half-size of a sheet in the plane units of the model


def mixing_angle(tt: float, width: float = _AN.mix_width) -> float:
    """theta = 0 (all the energy on the electron sheet), pi/2 (all on the photon sheet): e- e+ -> gamma gamma at T_C1,
    and back, gamma gamma -> e- e+, at T_C2 (smooth steps of the given width)."""
    up = smooth(tt, T_C1 - 0.5 * width, T_C1 + 0.5 * width)
    down = smooth(tt, T_C2 - 0.5 * width, T_C2 + 0.5 * width)
    return 0.5 * math.pi * (up - down)


def energy_shares(theta: float) -> tuple[float, float]:
    """(share of the energy on the electron sheet, on the photon sheet): cos^2, sin^2."""
    return math.cos(theta) ** 2, math.sin(theta) ** 2


def electron_x(tt: float) -> float:
    """Distance of the e- (the e+ is at minus this) from the centre: they approach until T_C1 and part after T_C2."""
    return V_E * (T_C1 - tt) if tt <= T_C1 else (V_E * (tt - T_C2) if tt >= T_C2 else 0.0)


def photon_x_out(tt: float) -> float:
    """The photons born at T_C1 fly apart."""
    return V_G * max(0.0, tt - T_C1)


def photon_x_in(tt: float) -> float:
    """The photons that meet at T_C2 fly towards each other."""
    return V_G * max(0.0, T_C2 - tt)


def wave_on_sheet(name: str, U: np.ndarray, V: np.ndarray, t: float) -> np.ndarray:
    """The waves of the stack part of the film (before the zoom) on the sheet 'name', in the units of the sheet."""
    SW = CFG.stack_waves
    add = np.zeros_like(U)
    if name == "e":
        if T_PULSE <= t < T_PASS[0]:
            for j in range(SW.ring_count):
                add += ring(U, V, t, 0.0, 0.0, T_PULSE + SW.ring_spacing * j, SW.ring_speed, SW.ring_width, SW.ring_k,
                            SW.ring_amplitude - SW.ring_amplitude_step * j)
        if T_PASS[0] <= t < T_PASS[1]:
            tt = t - T_PASS[0]
            for p in SW.packets:
                add += packet(U, V, tt, p["x0"], p["y0"], p["vx"], SW.packet_sigma, SW.packet_k, SW.packet_amplitude)
    return add


# ----------------------------------------------------------- the zoom: the two fields and their interaction

def sim_time(t: float, win: tuple[float, float]) -> float:
    """Time of the scalar-QED simulation at the film time t (linear over the window, held outside)."""
    return qp.T_SIM * min(max((t - win[0]) / (win[1] - win[0]), 0.0), 1.0)


def stage_weights(t: float) -> tuple[float, float, float]:
    """Presence of the stages A (e- e-), B (e- e+) and C (annihilation): cross-fades of 1.2 s."""
    SG = CFG.stages
    wa = smooth(t, T_A[0], T_A[0] + SG.a_in) * (1.0 - smooth(t, T_B[0] - SG.b_in_before, T_B[0] + SG.b_in_after))
    wb = smooth(t, T_B[0] - SG.b_in_before, T_B[0] + SG.b_in_after) * (1.0 - smooth(t, T_C[0] - SG.c_in_before, T_C[0] + SG.c_in_after))
    wc = smooth(t, T_C[0] - SG.c_in_before, T_C[0] + SG.c_in_after) * (1.0 - smooth(t, T_C[1] + SG.c_out[0], T_C[1] + SG.c_out[1]))
    return wa, wb, wc


def annihilation_fields(X: np.ndarray, Y: np.ndarray, tt: float) -> tuple[np.ndarray, np.ndarray]:
    """Heights (scene units) on the electron and the photon sheet of the schematic e- e+ <-> gamma gamma.

    First event: e- e+ come along the x axis, annihilate at T_C1, the two photons fly away along x.  Second, separate
    event: two other photons come along the y axis, meet at T_C2 and create a pair that flies away along y."""
    ee, eg = energy_shares(mixing_angle(tt))
    xe = electron_x(tt)
    along_x = tt < 0.5 * (T_C1 + T_C2)
    ze = np.zeros_like(X)
    zg = np.zeros_like(X)
    for sg in (-1.0, 1.0):
        cx_, cy_ = (sg * xe, 0.0) if along_x else (0.0, sg * xe)
        ze += _AN.electron_amplitude * math.sqrt(ee) * np.exp(-((X - cx_) ** 2 + (Y - cy_) ** 2) / (2 * _AN.electron_sigma ** 2))
        xp = photon_x_out(tt)                                        # the photons of the first event: along x
        zg += _AN.photon_amplitude * math.sqrt(eg) * np.exp(-((X - sg * xp) ** 2 + Y ** 2) / (2 * _AN.photon_sigma ** 2)) * np.cos(_AN.photon_k * (X - sg * xp))
        yp = photon_x_in(tt)                                         # the photons of the second event: along y
        zg += _AN.photon_amplitude * math.sqrt(eg) * np.exp(-(X ** 2 + (Y - sg * yp) ** 2) / (2 * _AN.photon_sigma ** 2)) * np.cos(_AN.photon_k * (Y - sg * yp))
    return ze, zg


def pair_fields(case: str, s: float, step: int = qp.GRID_STEP) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Heights (scene units) on the electron and the photon sheet from the scalar-QED model and the packet centres."""
    low, up, c = qp.frame(case, s, step=step)
    return H_E * low, H_G * (up - A0_BASE[case]), c


def sheet_grid(step: int = qp.GRID_STEP) -> np.ndarray:
    """The 76 points of the drawn grid along one axis (the plane units), the last one at the edge of the sheet."""
    return np.append(qp.axes(step), X_HALF)


def edge_pad(a: np.ndarray) -> np.ndarray:
    return np.pad(a, ((0, 1), (0, 1)), mode="edge")


def zoom_heights(t: float, grid: np.ndarray) -> tuple[np.ndarray, np.ndarray, list]:
    """(electron sheet, photon sheet) heights on the grid x grid (index [ix, iy]) and the markers (x, y, label) of the packets."""
    X, Y = np.meshgrid(grid, grid, indexing="ij")
    ze = np.zeros_like(X)
    zg = np.zeros_like(X)
    marks: list = []
    wa, wb, wc = stage_weights(t)
    for w, case, win in ((wa, "pp", T_A), (wb, "pm", T_B)):
        if w > CFG.stages.weight_min:
            pe, pg, c = pair_fields(case, sim_time(t, win))
            ze += w * edge_pad(pe)
            zg += w * edge_pad(pg)
            marks.append((w, c, case))
    if wc > CFG.stages.weight_min:
        ce, cg = annihilation_fields(X, Y, min(max(t - T_C[0], 0.0), T_C[1] - T_C[0]))
        ze += wc * ce
        zg += wc * cg
    return ze, zg, marks


TEXT = {lang: load_texts(HERE, lang) for lang in ("en", "ru")}     # texts.toml


def render(out: Path, size: tuple[int, int], fps: int, total: float, lang: str, snap: float | None = None) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection, PolyCollection

    if snap is None and not shutil.which("ffmpeg"):
        sys.exit("ffmpeg is required")
    tx = TEXT[lang]
    V, S_, LY, F, WI, SU, LB, LG, PS, CH, PN, BR, CM, ZM, SG = (CFG.video, CFG.style, CFG.layout, CFG.fonts, CFG.wire, CFG.surface, CFG.labels,
                                                                 CFG.legend, CFG.point_of_space, CFG.charges, CFG.panel, CFG.bars, CFG.camera,
                                                                 CFG.zoom, CFG.stages)
    W, H = size
    dpi = V.dpi
    sc = H / V.reference_height
    BG = S_.background
    TXT = tuple(S_.text)
    DIM = tuple(S_.dim)
    bg_rgba = np.array([int(BG[1:3], 16), int(BG[3:5], 16), int(BG[5:7], 16), 255], np.float32)
    fig = plt.figure(figsize=(W / dpi, H / dpi), dpi=dpi, facecolor=BG)
    ax = fig.add_axes(LY.axes, facecolor="none")
    ax_b = fig.add_axes(BR.axes, facecolor="none")
    k = total / TOTAL
    frames = int(round(total * fps))
    ids = [int(round(snap * fps))] if snap is not None else range(frames)
    writer = None
    if snap is None:
        writer = subprocess.Popen(
            ["ffmpeg", "-y", "-v", "error", "-f", "rawvideo", "-pix_fmt", "rgba", "-s", f"{W}x{H}",
             "-r", str(fps), "-i", "-", "-c:v", "libx264", "-preset", V.preset, "-crf", str(V.crf),
             "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(out)], stdin=subprocess.PIPE)

    grid_u = sheet_grid() / X_HALF                               # 76 points in [-1, 1]
    NG = len(grid_u)
    GU, GV = np.meshgrid(grid_u, grid_u, indexing="ij")           # index [iu, iv]
    line_idx = np.arange(0, NG, WI.line_step)                    # every n-th grid line is drawn in the wire view
    seeds = {name: CFG.fluctuation.seed_base + CFG.fluctuation.seed_step * i for i, (name, *_r) in enumerate(FIELDS)}
    info = {name: (grp, col, row) for name, grp, col, row in FIELDS}
    light = np.array(SU.light)
    light /= np.linalg.norm(light)

    def camera(tau: float) -> tuple[float, float]:
        return math.radians(0.0 + CM.azimuth_deg * tau), math.radians(CM.elevation_top_deg - CM.elevation_drop_deg * tau)       # azimuth, elevation

    for k_ in ids:
        t_film = k_ / fps
        t = t_film / k
        fig.texts.clear()
        for art in list(fig.artists):
            art.remove()
        ax.clear()
        ax.set_facecolor("none")
        ax.axis("off")
        ax_b.clear()
        ax_b.axis("off")
        ax.set_xlim(*CM.xlim)
        ax.set_ylim(*CM.ylim)
        ax.set_aspect("equal")
        tau = min(lift_progress(nm, t) for nm in STACK_ORDER)          # all the sheets are in the stack when tau = 1
        az, el = camera(smooth(t, *T_CAM))
        zp = smooth(t, *T_ZOOM)                                        # the zoom
        fly = smooth(t, T_ZOOM[0] + ZM.fly_start, T_ZOOM[1]) ** 2      # the other sheets fly away
        solid = smooth(t, T_ZOOM[0] + ZM.solid_from, T_ZOOM[1] + ZM.solid_to)            # the two sheets become solid

        def proj(x, y, z):
            x, y, z = np.asarray(x, float), np.asarray(y, float), np.asarray(z, float)
            xr = x * math.cos(az) - y * math.sin(az)
            yr = x * math.sin(az) + y * math.cos(az)
            return xr, yr * math.sin(el) + z * math.cos(el)

        # what is highlighted (the other sheets are dimmed)
        if t < T_PASS[0] - WI.focus_lead:
            focus = set(info) if t < T_POINT[0] else {"e"}
            if t >= T_PULSE:
                focus = {"e"}
        elif t < T_ZOOM[0]:
            focus = {"e"}
        else:
            focus = {"e", "gamma"}
        dim_t = smooth(t, T_POINT[1] + WI.dim_ramp[0], T_POINT[1] + WI.dim_ramp[1])        # the dimming of the non-focus sheets
        up_group = set(STACK_ORDER[:STACK_ORDER.index("gamma")])     # the sheets above the photon sheet fly up
        zoom_on = t >= T_ZOOM[0] and tau > CFG.stack.complete_at
        if t >= T_A[0] - ZM.fields_from:
            ze_g, zg_g, marks = zoom_heights(t, grid_u * X_HALF)
        else:
            ze_g = zg_g = np.zeros((NG, NG))
            marks = []

        # ---- sheet geometry for all fields
        order = []
        for name, grp, col, row in FIELDS:
            tx_, ty_ = table_center(col, row)
            pr = lift_progress(name, t)
            cx = tx_ * (1 - pr)
            cy = ty_ * (1 - pr)
            cz = stack_z(name) * pr
            half = SIZE_TABLE * (1 - pr) + SIZE_STACK * pr
            if zoom_on:
                if name == "e":
                    cz += (Z_E - stack_z("e")) * zp
                    half += (ZOOM_HALF - SIZE_STACK) * zp
                elif name == "gamma":
                    cz += (Z_G - stack_z("gamma")) * zp
                    half += (ZOOM_HALF - SIZE_STACK) * zp
                else:
                    cz += (ZM.fly_distance if name in up_group else -ZM.fly_distance) * fly
            order.append((cz, name, cx, cy, half, pr))
        # draw far sheets first: for the stack, bigger z is nearer to the top; use the camera depth
        order.sort(key=lambda r: r[0])
        for cz, name, cx, cy, half, pr in order:
            grp = info[name][0]
            colr = GROUPS[grp]
            amp_scale = half / SIZE_STACK
            is_pair = name in ("e", "gamma")
            alpha = WI.alpha
            if tau > CFG.stack.complete_at and name not in focus:
                alpha = WI.alpha - WI.dim_drop * dim_t
                if zoom_on:                                    # for a moment the sheets light up, then fly away
                    alpha = (ZM.light_up_base + ZM.light_up_gain * smooth(t, T_ZOOM[0], T_ZOOM[0] + ZM.light_up_s)) * (1.0 - fly)
            if alpha < WI.alpha_min:
                continue
            h_fl = CFG.fluctuation.height_factor * amp_scale * fluctuation(GU, GV, t, seeds[name])
            h_ex = CFG.stack_waves.height_factor * amp_scale * wave_on_sheet(name, GU, GV, t) if (is_pair and tau > CFG.stack.complete_at) else 0.0
            h = h_fl + h_ex
            if is_pair and zoom_on and t >= T_A[0] - ZM.fields_from:
                h = h_fl * (1.0 - ZM.fluctuation_damping * zp) + (ze_g if name == "e" else zg_g)
            Xw = cx + half * GU
            Yw = cy + half * GV
            Zw = cz + h
            wire_a = alpha * WI.alpha_gain * (1.0 - WI.solid_wire_fade * solid * (1.0 if is_pair else 0.0))
            segs = []
            for i in line_idx:
                X_, Y_ = proj(Xw[:, i], Yw[:, i], Zw[:, i])
                segs.append(np.column_stack([X_, Y_]))
                X_, Y_ = proj(Xw[i, :], Yw[i, :], Zw[i, :])
                segs.append(np.column_stack([X_, Y_]))
            lw = WI.width_focus if (name in focus and tau > CFG.stack.complete_at and dim_t > WI.dim_threshold) else WI.width
            if is_pair and solid > ZM.solid_cutoff:
                # ---- the solid surface: lit quads, far ones first
                hx = np.gradient(Zw, 2 * half / (NG - 1), axis=0)
                hy = np.gradient(Zw, 2 * half / (NG - 1), axis=1)
                nrm = np.stack([-hx, -hy, np.ones_like(hx)], axis=-1)
                nrm /= np.linalg.norm(nrm, axis=-1, keepdims=True)
                lam = np.clip(nrm @ light, 0.0, 1.0)
                rel = (Zw - cz)
                if name == "e":
                    top = np.clip(rel / SU.electron_tint_scale, 0.0, 1.0)[..., None]
                    base = np.array(colr)
                    col_v = base * (SU.electron_shade_base + SU.electron_shade_gain * lam[..., None]) + SU.electron_tint_gain * top * np.array([1, 1, 1])
                    a_s = SU.electron_alpha
                else:
                    sgn = np.clip(-rel / SU.photon_negative_scale, 0.0, 1.0)[..., None]       # 0 = positive A_0 (pink), 1 = negative (violet)
                    base = np.array(colr)
                    cold = np.array(SU.photon_cold)
                    base_v = base * (1.0 - sgn) + cold * sgn
                    col_v = base_v * (SU.photon_shade_base + SU.photon_shade_gain * lam[..., None])
                    a_s = SU.photon_alpha
                col_v = np.clip(col_v, 0.0, 1.0)
                P = np.stack(proj(Xw, Yw, Zw), axis=-1)                        # [iu, iv, 2]
                quads = np.stack([P[:-1, :-1], P[1:, :-1], P[1:, 1:], P[:-1, 1:]], axis=2).reshape(-1, 4, 2)
                cq = ((col_v[:-1, :-1] + col_v[1:, :-1] + col_v[1:, 1:] + col_v[:-1, 1:]) / 4.0).reshape(-1, 3)
                depth = ((Xw[:-1, :-1] * math.sin(az) + Yw[:-1, :-1] * math.cos(az))).reshape(-1)
                srt = np.argsort(-depth)
                fc = np.column_stack([cq[srt], np.full(len(srt), a_s * solid * min(1.0, alpha * SU.alpha_boost))])
                ax.add_collection(PolyCollection(quads[srt], facecolors=fc, edgecolors=fc, linewidths=SU.edge_width * sc, zorder=2))
            ax.add_collection(LineCollection(segs, colors=[(*colr, wire_a)] * len(segs), linewidths=lw * sc, capstyle="round", zorder=2))
            # the cross-section of the waves along the central line of the sheet, drawn thick (the stack part)
            wave_sheet = is_pair and tau > CFG.stack.complete_at and t < T_ZOOM[0] + WI.cross_section_until
            if wave_sheet:
                sl = grid_u
                wv = wave_on_sheet(name, sl, np.zeros_like(sl), t)
                if np.any(np.abs(wv) > WI.cross_section_threshold):
                    hc = CFG.fluctuation.height_factor * fluctuation(sl, np.zeros_like(sl), t, seeds[name]) + CFG.stack_waves.height_factor * wv
                    Xc, Yc = proj(cx + half * sl, cy + 0.0 * sl, cz + amp_scale * hc)
                    ax.plot(Xc, Yc, color=(*colr, 1.0), lw=WI.cross_section_width * sc, zorder=4, solid_capstyle="round")
            # the frame of the sheet
            corners = np.array([[-1, -1], [1, -1], [1, 1], [-1, 1], [-1, -1]], float)
            Xf, Yf = proj(cx + half * corners[:, 0], cy + half * corners[:, 1], cz * np.ones(5))
            ax.plot(Xf, Yf, color=(*colr, min(1.0, alpha + WI.frame_alpha_gain)), lw=WI.frame_width * sc, zorder=3)
            # the label
            if pr < CFG.stack.complete_at:
                mult = LB.multiplicity.to_dict().get(grp) if name not in LB.no_multiplicity else None
                if mult is not None:
                    Xm_, Ym_ = proj(cx + LB.multiplicity_pos[0] * half, cy + LB.multiplicity_pos[1] * half, cz)
                    ax.text(f1(Xm_), f1(Ym_), mult, color=(*colr, 1.0 - pr), fontsize=LB.multiplicity_size * sc, ha="center", va="center", zorder=5)
                Xl, Yl = proj(cx, cy, cz)
                ax.text(f1(Xl), f1(Yl) + LB.tile_dy, LABEL[name], color=(*colr, 1.0 - pr), fontsize=LB.tile_size * sc, ha="center", va="center", zorder=5)
            elif t < T_ZOOM[0] and (name in focus or dim_t < WI.dim_threshold):
                Xl, Yl = proj(cx + half * 1.0, cy - half * 1.0, cz)
                ax.text(f1(Xl) + LB.symbol_offset[0], f1(Yl) + LB.symbol_offset[1], FIELD_SYMBOL[name], color=(*colr, alpha), fontsize=LB.symbol_size * sc, ha="left", va="center", zorder=5)

        # ---- the legend of the table groups (a row of colour keys)
        if tau < LG.hide_after:
            keys = [tuple(kv) for kv in CFG.table.legend]
            x0 = LG.x0
            rend = fig.canvas.get_renderer()
            ax.apply_aspect()
            for grp, key in keys:
                ax.plot([x0], [LG.y], "s", color=(*GROUPS[grp], 1.0 - tau), ms=LG.marker_size * sc, zorder=5)
                tt_ = ax.text(x0 + LG.text_dx, LG.y, tx[key], color=(*GROUPS[grp], 1.0 - tau), fontsize=LG.text_size * sc, ha="left", va="center", zorder=5)
                wd = tt_.get_window_extent(rend).transformed(ax.transData.inverted()).width     # width of the text in the units of the axes
                x0 += LG.text_dx + wd + LG.gap
        # ---- one point of space pierces all the sheets
        if tau > CFG.stack.complete_at and T_POINT[0] <= t < T_POINT[1] + PS.visible_until:
            a_pt = smooth(t, T_POINT[0], T_POINT[0] + PS.fade_in) * (1 - smooth(t, T_POINT[1] + PS.fade_out[0], T_POINT[1] + PS.fade_out[1]))
            px_, py_ = PS.position
            zt, zb = stack_z("H") + PS.z_margin, stack_z("b") - PS.z_margin
            Xa, Ya = proj(np.array([px_, px_]), np.array([py_, py_]), np.array([zt, zb]))
            ax.plot(Xa, Ya, color=(*S_.point_color, PS.line_alpha * a_pt), lw=PS.line_width * sc, ls=(0, tuple(PS.line_dash)), zorder=6)
            for name, *_ in FIELDS:
                Xd, Yd = proj(np.array([px_]), np.array([py_]), np.array([stack_z(name)]))
                ax.plot(Xd, Yd, "o", color=(*S_.point_color, a_pt), ms=PS.dot_size * sc, zorder=7)
            Xt, Yt = proj(np.array([px_]), np.array([py_]), np.array([zt]))
            ax.text(f1(Xt) + PS.text_offset[0], f1(Yt) + PS.text_offset[1], tx["point"], color=(*S_.point_color, a_pt), fontsize=PS.text_size * sc, zorder=7)
        # ---- the annotations of the two sheets in the zoom
        if zoom_on and t >= T_ZOOM[0] + ZM.label_from:
            a_l = smooth(t, T_ZOOM[0] + ZM.label_from, T_ZOOM[1] + ZM.label_to)
            for name, key, zc in (("e", "e_el" if t < T_C[0] - SG.c_in_before else "e_el2", Z_E), ("gamma", "e_ph" if t < T_C[0] - SG.c_in_before else "e_ph2", Z_G)):
                cs = np.array([[-1, -1], [1, -1], [1, 1], [-1, 1]], float)
                Xc, Yc = proj(ZOOM_HALF * cs[:, 0], ZOOM_HALF * cs[:, 1], zc * np.ones(4))
                j = int(np.argmin(Xc))
                ax.text(float(Xc[j]) + LB.sheet_annotation_dx, float(Yc[j]), tx[key], color=(*GROUPS[info[name][0]], a_l), fontsize=LB.sheet_annotation_size * sc,
                        ha="right", va="center", zorder=6, linespacing=LB.sheet_annotation_linespacing)
        # ---- dotted lines from the charges to their fields and the charge labels (parts A and B)
        for w, c, case in marks:
            hs = 2 * ZOOM_HALF / (2 * X_HALF)
            for idx, lab in enumerate(("e-", "e-" if case == "pp" else "e+")):
                px_, py_ = float(c[idx][0]), float(c[idx][1])
                iu = int(np.clip(round((px_ / X_HALF + 1) / 2 * (NG - 1)), 0, NG - 1))
                iv = int(np.clip(round((py_ / X_HALF + 1) / 2 * (NG - 1)), 0, NG - 1))
                xw_, yw_ = hs * px_, hs * py_
                zlo = Z_E + ze_g[iu, iv]
                zhi = Z_G + zg_g[iu, iv]
                Xd, Yd = proj(np.array([xw_, xw_]), np.array([yw_, yw_]), np.array([zlo, zhi]))
                ax.plot(Xd, Yd, color=(1, 1, 1, CH.dotted_alpha * w * solid), lw=CH.dotted_width * sc, ls=(0, tuple(CH.dotted_dash)), zorder=6)
                Xm, Ym = proj(np.array([xw_]), np.array([yw_]), np.array([zlo + CH.label_dz]))
                ax.text(f1(Xm), f1(Ym) + CH.label_dy, r"$e^-$" if lab == "e-" else r"$e^+$", color=(1, 1, 1, w * solid), fontsize=CH.label_size * sc,
                        ha="center", va="center", zorder=8)
        # ---- the panel of the interaction (parts A and B)
        wa, wb, wc = stage_weights(t)
        for w, key_pair, key_kind, col_k in ((wa, "pair_pp", "rep", tuple(PN.repulsion_color)), (wb, "pair_pm", "att", tuple(PN.attraction_color))):
            if w > SG.panel_min:
                fig.text(PN.x, PN.pair["y"], tx[key_pair], color=(*TXT[:3], w), fontsize=PN.pair["size"] * sc)
                fig.text(PN.x, PN.kind["y"], tx[key_kind], color=(*col_k, w), fontsize=PN.kind["size"] * sc)
                fig.text(PN.x, PN.maxwell["y"], tx["maxwell"], color=(*TXT[:3], w), fontsize=PN.maxwell["size"] * sc)
                fig.text(PN.x, PN.maxwell_note["y"], tx["f1d"], color=(*DIM[:3], w), fontsize=PN.maxwell_note["size"] * sc)
                fig.text(PN.x, PN.phase["y"], tx["phase"], color=(*TXT[:3], w), fontsize=PN.phase["size"] * sc)
                fig.text(PN.x, PN.phase_note["y"], tx["f2d"], color=(*DIM[:3], w), fontsize=PN.phase_note["size"] * sc)
        if wb > SG.panel_min or wa > SG.panel_min:
            fig.text(PN.x, PN.positive["y"], tx["pos"], color=(*GROUPS["gauge"], max(wa, wb) * solid), fontsize=PN.positive["size"] * sc)
            fig.text(PN.x, PN.negative["y"], tx["neg"], color=(*PN.negative_color, wb * solid), fontsize=PN.negative["size"] * sc)
        # ---- the energy bars of the annihilation
        if wc > SG.panel_min:
            tt = min(max(t - T_C[0], 0.0), T_C[1] - T_C[0])
            ee, eg = energy_shares(mixing_angle(tt))
            a_b = wc
            ax_b.set_visible(True)
            ax_b.set_xlim(0, BR.xlim)
            ax_b.set_ylim(0, BR.ylim)
            ax_b.bar(list(BR.centres), [ee, eg], width=BR.width, color=[(*GROUPS["lepton"], BR.alpha * a_b), (*GROUPS["gauge"], BR.alpha * a_b)], lw=0)
            ax_b.plot([0, BR.xlim], [0, 0], color=(1, 1, 1, BR.baseline_alpha * a_b), lw=BR.baseline_width * sc)
            ax_b.text(BR.centres[0], BR.label_y, tx["bar_e"], color=(*GROUPS["lepton"], a_b), fontsize=BR.label_size * sc, ha="center", va="top")
            ax_b.text(BR.centres[1], BR.label_y, tx["bar_g"], color=(*GROUPS["gauge"], a_b), fontsize=BR.label_size * sc, ha="center", va="top")
            ax_b.text(*BR.sum_pos, tx["bar_s"], color=(*TXT[:3], a_b), fontsize=BR.sum_size * sc, ha="center", va="top")
            fig.text(*BR.title_pos, tx["ann"], color=(*TXT[:3], a_b), fontsize=BR.title_size * sc)
        # ---- captions
        if t < T_LIFT[0] - _T.lift_caption_lead:
            cap = tx["s0"]
        elif t < T_POINT[0]:
            cap = tx["s1"]
        elif t < T_PULSE - ZM.caption_lead:
            cap = tx["s2"]
        elif t < T_PASS[0]:
            cap = tx["s3"]
        elif t < T_ZOOM[0]:
            cap = tx["s4"]
        elif t < T_A[0] - ZM.caption_lead:
            cap = tx["s5"]
        elif t < T_B[0] - ZM.caption_lead:
            cap = tx["sA"]
        elif t < T_C[0] - ZM.caption_lead:
            cap = tx["sB"]
        elif t < T_C[0] + T_C1 + SG.caption_c1_after:
            cap = tx["sC1"]
        elif t < T_C[0] + SG.caption_c2_until:
            cap = tx["sC2"]
        else:
            cap = tx["sC3"]
        fig.text(*LY.title_pos, tx["title"], color=tuple(S_.title_color), fontsize=F.title * sc)
        fig.text(*LY.subtitle_pos, tx["sub"], color=DIM, fontsize=F.subtitle * sc)
        fig.text(*LY.caption_pos, cap, color=TXT, fontsize=F.caption * sc)
        if t < T_POINT[1]:
            fig.text(*LY.colour_note_pos, tx["col"], color=(*DIM[:3], S_.colour_note_alpha * (1.0 - smooth(t, T_POINT[1] + S_.colour_note_fade[0], T_POINT[1] + S_.colour_note_fade[1]))),
                     fontsize=F.colour_note * sc)
        if tau > CFG.stack.complete_at:
            fig.text(*LY.note_pos, tx["note"], color=tuple(S_.note_color), fontsize=F.note * sc, ha="left", wrap=True)
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
    out = args.out or HERE / "media" / f"sm_fields_{args.lang}.mp4"
    out.parent.mkdir(parents=True, exist_ok=True)
    if args.snapshot is not None:
        render(out.with_suffix(".png"), (V.width, V.height), V.fps, args.seconds, args.lang, snap=args.snapshot)
    elif args.preview:
        render(out.with_name(out.stem + "_preview.mp4"), (V.preview_width, V.preview_height), V.preview_fps, args.seconds, args.lang)
    else:
        render(out, (V.width, V.height), V.fps, args.seconds, args.lang)


if __name__ == "__main__":
    main()
