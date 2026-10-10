r"""Attraction and repulsion as the interference of waves (the book, the chapter "Attraction and Repulsion").

Two Gaussian wave packets in the plane (the scalar-QED model of scalar_qed_numerics.py: analytic free packets, the potential A_0 of each packet,
the eikonal phase of the interaction; the centres follow from the gradient of that phase). Charges of opposite signs: attraction; of the same sign: repulsion.
 1. the packets and their fields (the contours of A_0),
 2. the motion: trails, the free (straight) motion dashed, the force, the distance and the transverse position against time,
 3. the interference: along the line to the other packet the phase chi = -s q^2 theta of the interaction and the distribution over the momenta of the free wave and
    of the wave with the interaction; the shift Delta P = <grad chi> is the force integrated over time.
Every number is in config.toml, every word in texts.toml (the cases attraction / repulsion differ by the signs of the charges).

Usage:
    python attraction_repulsion.py --case attraction --lang en      # media/attraction_en.mp4
    python attraction_repulsion.py --case repulsion --lang ru       # media/repulsion_ru.mp4
    python attraction_repulsion.py --case attraction --lang en --snapshot 30       # one PNG at the content time 30 s
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dvconfig import load_config, load_texts  # noqa: E402

import scalar_qed_numerics as N  # noqa: E402

HERE = Path(__file__).resolve().parent
CFG = load_config(HERE)
TEXT = {lang: load_texts(HERE, lang) for lang in ("en", "ru")}
SQ, VW, TL, ST, FT, LY = CFG.scalar_qed, CFG.view, CFG.timeline, CFG.style, CFG.fonts, CFG.layout

CASE = "attraction"
CARD_AT = list(TL.card_at)
CARD_KEYS = ["h0", "h1", "h2", "h3"]
PARTS = {k: tuple(getattr(TL, k)) for k in ("intro", "motion", "interf", "outro")}
CONTENT_TOTAL = PARTS["outro"][1]
TOTAL = CONTENT_TOTAL + TL.card_s * len(CARD_AT)
_cache: dict = {}


def set_case(case: str) -> None:
    global CASE
    CASE = case
    _cache.clear()


def timeline(tf: float) -> tuple[float, int | None, float]:
    """(content time, index of the card or None, progress of the card) at the film time tf."""
    done = 0
    for i, ca in enumerate(CARD_AT):
        a = ca + i * TL.card_s
        if a <= tf < a + TL.card_s:
            return ca, i, (tf - a) / TL.card_s
        if tf >= a + TL.card_s:
            done += 1
    return tf - done * TL.card_s, None, 0.0


def chapter_of(tc: float) -> int:
    return max(i for i, ca in enumerate(CARD_AT) if ca <= tc)


def smooth(x: float, a: float, b: float) -> float:
    u = min(max((x - a) / (b - a), 0.0), 1.0)
    return u * u * (3 - 2 * u)


def num(x: float, fmt: str, lang: str) -> str:
    s = format(x, fmt)
    return s.replace(".", "{,}") if lang == "ru" else s


def fill(s: str, /, **kw) -> str:
    for k, v in kw.items():
        s = s.replace(f"@{k}@", str(v))
    return s


def key_of(name: str) -> str:
    """The text key of the current case: name_attraction / name_repulsion."""
    return f"{name}_{CASE}"


# ======================================================================================================================= physics (cached)

def make_cfg(case: str) -> N.ScalarQEDConfig:
    signs = list(getattr(CFG.cases, case).charge_signs)
    packets = tuple(N._load_packet(dict(name=f"packet_{i + 1}", charge_ratio=SQ.charge_ratio * signs[i], sigma=SQ.sigma,
                                        px=(1 if i == 0 else -1) * SQ.momentum, py=0.0, x0=(-1 if i == 0 else 1) * SQ.x0,
                                        y0=(1 if i == 0 else -1) * SQ.y0, phase0=0.0), i) for i in range(2))
    return N.ScalarQEDConfig(grid=N.GridConfig(SQ.nx, SQ.ny, SQ.lx, SQ.ly), time=N.TimeConfig(0.0, SQ.t_end, SQ.sim_samples, SQ.sim_samples),
                             physics=N.PhysicsConfig(SQ.mass, SQ.charge_q), observables=N.ObservableConfig("phi_abs2", "a0"),
                             render=N.RenderConfig(1.0, 1.0, 1.0, 1.0, 0.0, 1.0), packets=packets)


def sample_line(arr: np.ndarray, x_axis: np.ndarray, y_axis: np.ndarray, px: np.ndarray, py: np.ndarray) -> np.ndarray:
    from scipy.ndimage import map_coordinates
    ix = (px - x_axis[0]) / (x_axis[1] - x_axis[0])
    iy = (py - y_axis[0]) / (y_axis[1] - y_axis[0])
    return map_coordinates(arr, [ix, iy], order=1, mode="nearest")


def compute(case: str) -> dict:
    """The simulation of the model and everything the drawing needs, cached on the disk."""
    key = hashlib.sha1(json.dumps([case, SQ.to_dict(), VW.to_dict(), PARTS["motion"], PARTS["interf"], TL.interf_t0, TL.interf_t1], sort_keys=True, default=str).encode()).hexdigest()[:12]
    path = HERE / VW.cache_dir / f"{case}_{key}.npz"
    if path.exists():
        z = np.load(path)
        return {k: z[k] for k in z.files}
    cfg = make_cfg(case)
    bundle = N.build_simulation_bundle(cfg)
    x_axis, y_axis = bundle["x_axis"], bundle["y_axis"]
    st = bundle["simulation_times"]
    centers = np.asarray(bundle["corrected_centers"])                     # (nt, 2, 2)
    straight = np.asarray(N._build_straight_trajectories(cfg, st))
    x_grid, y_grid = np.meshgrid(x_axis, y_axis, indexing="ij")
    ix = np.where((x_axis >= VW.x[0]) & (x_axis <= VW.x[1]))[0]
    iy = np.where((y_axis >= VW.y[0]) & (y_axis <= VW.y[1]))[0]
    d_motion = PARTS["motion"][1] - PARTS["motion"][0]
    tm = np.linspace(0.0, SQ.t_end, int(round(d_motion * VW.frame_samples_per_s)) + 1)
    n_i = int(round((PARTS["interf"][1] - PARTS["interf"][0]) * VW.interf_samples_per_s)) + 1
    ti = np.linspace(TL.interf_t0, TL.interf_t1, n_i)
    need = {float(t): ("map", k) for k, t in enumerate(tm)}
    dens1, dens2 = np.zeros((tm.size, ix.size, iy.size), np.float16), np.zeros((tm.size, ix.size, iy.size), np.float16)
    a0 = np.zeros((tm.size, ix.size, iy.size), np.float16)
    npts, half = VW.line_points, VW.line_half_length
    s_axis = np.linspace(-half, half, npts)
    ds = s_axis[1] - s_axis[0]
    chi_line = np.zeros((n_i, npts))
    free_line = np.zeros((n_i, npts), complex)
    d_p = np.zeros(n_i)
    nvec = np.zeros((n_i, 2))
    sign_t = 1.0
    half_pix = int(VW.window_pixels) // 2
    dxg, dyg = x_axis[1] - x_axis[0], y_axis[1] - y_axis[0]

    def snapshot_at(t: float):
        c_at = N.sample_frames(st, bundle["corrected_centers"], t)
        frames = N._packet_frames_for_centers(cfg, t, c_at, 0.0)
        pdata = N._packet_snapshot_data(cfg, x_grid, y_grid, t, frames, 0.0)
        th_t = N._sample_packet_field_history(st, bundle["theta_total_history"], t)
        th_i = N._sample_packet_field_history(st, bundle["theta_int_history"], t)
        return N._compose_snapshot(cfg, x_grid, y_grid, t, pdata, th_t, th_i, frames, 0.0), frames

    for k, t in enumerate(tm):
        snap, _ = snapshot_at(float(t))
        dens1[k] = snap["density_1"][np.ix_(ix, iy)]
        dens2[k] = snap["density_2"][np.ix_(ix, iy)]
        a0[k] = snap["a0_physical"][np.ix_(ix, iy)]
    for k, t in enumerate(ti):
        snap, frames = snapshot_at(float(t))
        c1, c2 = frames[0].center, frames[1].center
        pmom = frames[0].momentum
        n = np.array([-pmom[1], pmom[0]]) / float(np.hypot(*pmom))             # the transverse direction of packet 1 (the model moves the centre by the transverse force)
        if np.dot(n, c2 - c1) < 0.0 and k == 0:
            sign_t = -1.0
        elif k == 0:
            sign_t = 1.0
        n = sign_t * n                                                           # oriented toward the transverse side of the other packet at t = 0, the same for all t
        nvec[k] = n
        px_, py_ = c1[0] + s_axis * n[0], c1[1] + s_axis * n[1]
        chi_line[k] = -sample_line(snap["phase_1"], x_axis, y_axis, px_, py_)
        ref = np.exp(1j * frames[0].energy * float(t))                      # removes the global phase E t: the crest of the free wave stays at the centre
        free_line[k] = (sample_line(snap["phi_1_free_real"], x_axis, y_axis, px_, py_) + 1j * sample_line(snap["phi_1_free_imag"], x_axis, y_axis, px_, py_)) * ref
        ci = int(np.argmin(np.abs(x_axis - c1[0])))
        cj = int(np.argmin(np.abs(y_axis - c1[1])))
        win = (slice(max(ci - half_pix, 0), ci + half_pix), slice(max(cj - half_pix, 0), cj + half_pix))
        chi2d = -snap["phase_1"][win]
        gx, gy = np.gradient(chi2d, dxg, dyg)
        w = np.abs(snap["phi_1_free_real"][win] + 1j * snap["phi_1_free_imag"][win]) ** 2
        d_p[k] = float((w * (gx * n[0] + gy * n[1])).sum() / w.sum())          # <Delta P> along the line: the density-weighted gradient of the phase
    peak = float(np.abs(free_line).max())
    free_line /= peak
    out = {"x_axis": x_axis[ix], "y_axis": y_axis[iy], "st": st, "centers": centers, "straight": straight, "tm": tm, "dens1": dens1, "dens2": dens2, "a0": a0,
           "ti": ti, "s_axis": s_axis, "chi_line": chi_line, "free_re": free_line.real, "free_im": free_line.imag, "d_p": d_p, "nvec": nvec,
           "signs": np.array([p.charge_sign for p in cfg.packets])}
    path.parent.mkdir(exist_ok=True)
    np.savez(path, **out)
    return out


def get_data() -> dict:
    if "data" not in _cache:
        _cache["data"] = compute(CASE)
    return _cache["data"]


def momentum_distributions(D: dict, k: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """Along the transverse line to the other packet: the momentum axis (relative to the mean of the free packet), the distribution of the free wave and of
    the free wave with the interaction phase linear across the packet, exp(i g s) phi_free, g = <grad chi> (the free wave plus the wave of the interaction): the free
    distribution shifted by g; the free peak = 1."""
    free = D["free_re"][k] + 1j * D["free_im"][k]
    s_ax = D["s_axis"]
    g = float(D["d_p"][k])
    n = free.size * int(VW.pad_factor)
    ds = float(s_ax[1] - s_ax[0])
    pk = 2.0 * np.pi * np.fft.fftshift(np.fft.fftfreq(n, d=ds))
    df = np.abs(np.fft.fftshift(np.fft.fft(free, n=n))) ** 2
    dg = np.abs(np.fft.fftshift(np.fft.fft(free * np.exp(1j * g * s_ax), n=n))) ** 2
    mean_f = float((pk * df).sum() / df.sum())
    return pk - mean_f, df / df.max(), dg / df.max(), g


def smooth_series(a: np.ndarray, sigma: float) -> np.ndarray:
    from scipy.ndimage import gaussian_filter1d
    return gaussian_filter1d(a, sigma, axis=0, mode="nearest")


def force_series(D: dict) -> np.ndarray:
    """m * d^2 R / dt^2 of the two packets (smoothed), shape (nt, 2, 2)."""
    st, c = D["st"], D["centers"]
    dt = st[1] - st[0]
    v = np.gradient(smooth_series(c, 4.0), dt, axis=0)
    a = np.gradient(smooth_series(v, 4.0), dt, axis=0)
    return SQ.mass * a


# ======================================================================================================================= drawing helpers

class Frame:
    def __init__(self, fig, size: tuple[int, int], lang: str):
        self.fig, self.W, self.H, self.lang = fig, size[0], size[1], lang
        self.sc = size[1] / CFG.video.reference_height
        self.tx = TEXT[lang]
        self.TXT, self.DIM = tuple(ST.text), tuple(ST.dim)
        self.WARM, self.BLUE, self.GOLD, self.GREEN = tuple(ST.warm), tuple(ST.blue), tuple(ST.gold), tuple(ST.green)
        self._r = fig.canvas.get_renderer()

    def text(self, x, y, s, size, color, ha="left", va="baseline", max_w=None, **kw):
        t = self.fig.text(x, y, s, color=color, fontsize=size * self.sc, ha=ha, va=va, **kw)
        if max_w is not None:
            w = t.get_window_extent(self._r).width / self.W
            if w > max_w:
                t.set_fontsize(size * self.sc * max(max_w / w, FT.min_scale))
        return t

    def paragraph(self, x, y, s, size, color, max_w, line, va="top") -> int:
        words, cur, inside = [], "", False
        for ch in s:
            if ch == "$":
                inside = not inside
            if ch == " " and not inside:
                words.append(cur)
                cur = ""
            else:
                cur += ch
        words.append(cur)
        lines, row = [], ""
        for w in words:
            trial = (row + " " + w).strip()
            t = self.fig.text(0.0, 0.0, trial, fontsize=size * self.sc)
            width = t.get_window_extent(self._r).width / self.W
            t.remove()
            if width > max_w and row:
                lines.append(row)
                row = w
            else:
                row = trial
        lines.append(row)
        for i, ln in enumerate(lines):
            self.fig.text(x, y - i * line, ln, color=color, fontsize=size * self.sc, ha="left", va=va)
        return len(lines)

    def axes(self, rect, **kw):
        return self.fig.add_axes(rect, facecolor="none", **kw)

    def style_axes(self, ax, left=True, bottom=True):
        for k, sp in ax.spines.items():
            sp.set_color(tuple(ST.axes_edge))
            sp.set_linewidth(ST.axes_edge_width * self.sc)
            sp.set_visible((k == "left" and left) or (k == "bottom" and bottom))
        ax.tick_params(colors=self.DIM, labelsize=FT.tick * self.sc, length=3 * self.sc, width=0.8 * self.sc)

    def label(self, ax, xy, s, size, color, ha="left", va="bottom", **kw):
        return ax.text(xy[0], xy[1], s, color=color, fontsize=size * self.sc, ha=ha, va=va, transform=ax.transAxes, **kw)


def caption(f: Frame, key: str, key2: str | None = None) -> None:
    f.text(*LY.caption_pos, f.tx[key], FT.caption, (*f.TXT, ST.caption_alpha), max_w=LY.caption_max_width)
    if key2:
        f.text(*LY.sub_caption_pos, f.tx[key2], FT.sub_caption, (*f.DIM, ST.caption_alpha), max_w=LY.caption_max_width)


def draw_header(f: Frame, t: float, card: int | None, cprog: float) -> None:
    tx = f.tx
    if card is None:
        f.text(*LY.title_pos, tx[key_of("title")], FT.title, tuple(ST.title_color), max_w=0.66)
        ch = chapter_of(t)
        if ch >= 1:
            f.text(*LY.chapter_label_pos, f"{ch}/3   " + tx[key_of(CARD_KEYS[ch]) if CARD_KEYS[ch] == "h2" else CARD_KEYS[ch]], FT.chapter_label, (*f.DIM, ST.chapter_label_alpha), ha="right", max_w=0.30)
        f.text(*LY.tag_pos, tx["tag"], FT.tag, (*f.DIM, 0.7), max_w=0.6)
    else:
        a_c = min(smooth(cprog, *TL.card_fade_in), 1.0 - smooth(cprog, *TL.card_fade_out))
        if card == 0:
            f.text(0.5, LY.card_title_y, tx[key_of("title")], FT.card_title, (*ST.card_title_color, a_c), ha="center", va="center", max_w=0.92)
            f.text(0.5, LY.card_subtitle_y, tx[key_of("h0s")], FT.card_subtitle, (*f.DIM, a_c), ha="center", va="center", max_w=0.9)
        else:
            name = "h2" if card == 2 else CARD_KEYS[card]
            kk = key_of(name) if card == 2 else name
            sub = key_of("h2s") if card == 2 else name + "s"
            f.text(0.5, LY.card_number_y, f"{card}", FT.card_number, (*f.GOLD, ST.card_number_alpha * a_c), ha="center", va="center")
            f.text(0.5, LY.card_chapter_y, tx[kk], FT.card_chapter, (*ST.card_title_color, a_c), ha="center", va="center", max_w=0.92)
            f.text(0.5, LY.card_chapter_sub_y, tx[sub], FT.card_chapter_sub, (*f.DIM, a_c), ha="center", va="center", max_w=0.9)


def formulas(f: Frame, items: list[tuple[str, float]]) -> None:
    for row, (k, a) in enumerate(items):
        if a > 0.01:
            f.text(LY.formula_pos[0], LY.formula_pos[1] - LY.formula_row * row, f.tx[k], FT.formula if row == 0 else FT.formula_small, (*f.TXT, ST.formula_alpha * a), max_w=0.94)


# ======================================================================================================================= the map

def frame_index(D: dict, t_sim: float) -> tuple[int, int, float]:
    tm = D["tm"]
    k = int(np.clip(np.searchsorted(tm, t_sim, side="right") - 1, 0, tm.size - 2))
    w = float((t_sim - tm[k]) / (tm[k + 1] - tm[k]))
    return k, k + 1, min(max(w, 0.0), 1.0)


def sim_index(D: dict, t_sim: float) -> int:
    return int(np.clip(round((t_sim - D["st"][0]) / (D["st"][1] - D["st"][0])), 0, D["st"].size - 1))


def draw_map(f: Frame, t_sim: float, a_fields: float, a_force: float, a_trail: float) -> None:
    D = get_data()
    ax = f.axes(LY.map_axes)
    ax.set_xlim(*VW.x)
    ax.set_ylim(*VW.y)
    ax.set_aspect("equal")
    ax.set_axis_off()
    k0, k1, w = frame_index(D, t_sim)
    d1 = (1 - w) * D["dens1"][k0].astype(np.float32) + w * D["dens1"][k1].astype(np.float32)
    d2 = (1 - w) * D["dens2"][k0].astype(np.float32) + w * D["dens2"][k1].astype(np.float32)
    a0 = (1 - w) * D["a0"][k0].astype(np.float32) + w * D["a0"][k1].astype(np.float32)
    ref1 = max(float(D["dens1"].max()), 1e-9)
    ref2 = max(float(D["dens2"].max()), 1e-9)
    bg = np.array([int(ST.background[i:i + 2], 16) for i in (1, 3, 5)], np.float32) / 255.0
    cols = [np.array(f.WARM if s > 0 else f.BLUE, np.float32) for s in D["signs"]]
    img = np.zeros(d1.shape + (3,), np.float32) + bg
    for dd, ref, col in ((d1, ref1, cols[0]), (d2, ref2, cols[1])):
        g = np.clip(dd / ref, 0.0, 1.0)
        g = np.clip((g - ST.map_floor) / (1.0 - ST.map_floor), 0.0, 1.0) ** ST.map_gamma * ST.map_gain
        layer = np.clip(g[..., None] * col[None, None, :], 0.0, 1.0)
        img = 1.0 - (1.0 - img) * (1.0 - layer)                                  # the screen blend: the overlap does not burn out
    img = np.clip(img, 0.0, 1.0)
    xa, ya = D["x_axis"], D["y_axis"]
    ax.imshow(np.transpose(img, (1, 0, 2)), origin="lower", extent=[xa[0], xa[-1], ya[0], ya[-1]], interpolation="bilinear", zorder=1)
    if a_fields > 0.01:
        a_ref = float(np.percentile(np.abs(D["a0"]), 99.7))
        lv = np.array(ST.potential_levels) * a_ref
        X, Y = np.meshgrid(xa, ya, indexing="ij")
        pos = [l for l in lv if a0.max() > l]
        neg = [l for l in lv if a0.min() < -l]
        if pos:
            ax.contour(X, Y, a0, levels=pos, colors=[(*f.WARM, ST.potential_alpha * a_fields)], linewidths=ST.potential_width * f.sc, zorder=2)
        if neg:
            ax.contour(X, Y, a0, levels=sorted(-np.array(neg)), colors=[(*f.BLUE, ST.potential_alpha * a_fields)], linewidths=ST.potential_width * f.sc, linestyles="solid", zorder=2)
    st, c, sc_ = D["st"], D["centers"], D["straight"]
    i = sim_index(D, t_sim)
    for p in range(2):
        col = f.WARM if D["signs"][p] > 0 else f.BLUE
        j0 = max(0, i - int(ST.trail_seconds / (st[1] - st[0])))
        if a_trail > 0.01:
            ax.plot(sc_[:, p, 0], sc_[:, p, 1], color=(*f.DIM, ST.free_alpha * a_trail), lw=1.2 * f.sc, ls=(0, tuple(ST.level_dash)), zorder=3)
            ax.plot(c[j0:i + 1, p, 0], c[j0:i + 1, p, 1], color=(*col, ST.trail_alpha * a_trail), lw=ST.trail_width * f.sc, zorder=4)
        ax.plot([c[i, p, 0]], [c[i, p, 1]], "o", ms=ST.marker_size * f.sc, mfc="none", mec=(*col, 0.95), mew=1.4 * f.sc, zorder=5)
        ax.text(c[i, p, 0], c[i, p, 1], "+" if D["signs"][p] > 0 else "−", color=(*col, 1.0), fontsize=FT.sign * f.sc, ha="center", va="center", zorder=6)
    if a_force > 0.01:
        F = force_series(D)
        for p in range(2):
            col = f.WARM if D["signs"][p] > 0 else f.BLUE
            v = F[i, p]
            mag = float(np.hypot(*v))
            if mag > 1e-4:
                length = ST.force_scale * min(mag / ST.force_ref, 1.0) * 1.0
                u = v / mag
                ax.annotate("", xy=(c[i, p, 0] + u[0] * length, c[i, p, 1] + u[1] * length), xytext=(c[i, p, 0], c[i, p, 1]),
                            arrowprops=dict(arrowstyle="-|>", color=(*f.GOLD, a_force), lw=ST.force_width * f.sc, shrinkA=6 * f.sc, shrinkB=0), zorder=7)
    for p, key in enumerate(("p1", "p2")):
        ax.text(c[i, p, 0], c[i, p, 1] + (0.75 if p == 0 else -0.75), f.tx[key], color=(*f.DIM, 0.9), fontsize=FT.legend * f.sc, ha="center", va="center", zorder=6)
    if a_fields > 0.5:
        f.text(LY.map_axes[0] + 0.01, LY.map_axes[1] + 0.012, f.tx["a0_label"], FT.legend, (*f.DIM, 0.85 * a_fields))


def panel_distance(f: Frame, t_sim: float) -> None:
    D = get_data()
    ax = f.axes(LY.panel_a_axes)
    f.style_axes(ax)
    st, c, sc_ = D["st"], D["centers"], D["straight"]
    d = np.hypot(c[:, 0, 0] - c[:, 1, 0], c[:, 0, 1] - c[:, 1, 1])
    d_free = np.hypot(sc_[:, 0, 0] - sc_[:, 1, 0], sc_[:, 0, 1] - sc_[:, 1, 1])
    i = sim_index(D, t_sim)
    ax.set_xlim(0.0, st[-1])
    ax.set_ylim(0.0, 1.08 * max(d.max(), d_free.max()))
    ax.plot(st, d_free, color=(*f.DIM, 0.9), lw=1.4 * f.sc, ls=(0, tuple(ST.level_dash)))
    ax.plot(st[: i + 1], d[: i + 1], color=(*f.GOLD, 1.0), lw=ST.curve_width * f.sc)
    ax.plot([st[i]], [d[i]], "o", color=(*f.GOLD, 1.0), ms=6 * f.sc)
    f.label(ax, (0.0, 1.04), f.tx["pa_title"], FT.panel_title, (*f.DIM, 1.0))
    f.label(ax, (1.0, -0.17), f.tx["p_t"], FT.axis_label, (*f.DIM, 1.0), ha="right", va="top")
    ax.text(0.04, 0.20, f.tx["pa_free"], color=(*f.DIM, 1.0), fontsize=FT.legend * f.sc, transform=ax.transAxes, ha="left", va="center")
    ax.text(0.04, 0.08, f.tx["pa_int"], color=(*f.GOLD, 1.0), fontsize=FT.legend * f.sc, transform=ax.transAxes, ha="left", va="center")


def panel_transverse(f: Frame, t_sim: float) -> None:
    D = get_data()
    ax = f.axes(LY.panel_b_axes)
    f.style_axes(ax)
    st, c, sc_ = D["st"], D["centers"], D["straight"]
    i = sim_index(D, t_sim)
    lo = min(c[:, 0, 1].min(), sc_[:, 0, 1].min())
    hi = max(c[:, 0, 1].max(), sc_[:, 0, 1].max())
    pad = 0.12 * (hi - lo + 1e-6)
    ax.set_xlim(0.0, st[-1])
    ax.set_ylim(lo - pad, hi + pad)
    ax.plot(st, sc_[:, 0, 1], color=(*f.DIM, 0.9), lw=1.4 * f.sc, ls=(0, tuple(ST.level_dash)))
    col = f.WARM if D["signs"][0] > 0 else f.BLUE
    ax.plot(st[: i + 1], c[: i + 1, 0, 1], color=(*col, 1.0), lw=ST.curve_width * f.sc)
    ax.plot([st[i]], [c[i, 0, 1]], "o", color=(*col, 1.0), ms=6 * f.sc)
    f.label(ax, (0.0, 1.04), f.tx["pb_title"], FT.panel_title, (*f.DIM, 1.0))
    f.label(ax, (1.0, -0.17), f.tx["p_t"], FT.axis_label, (*f.DIM, 1.0), ha="right", va="top")


def panels_interference(f: Frame, t_sim: float, alpha: float) -> None:
    D = get_data()
    k = int(np.clip(round((t_sim - D["ti"][0]) / (D["ti"][1] - D["ti"][0])), 0, D["ti"].size - 1))
    s = D["s_axis"]
    ax = f.axes(LY.panel_a_axes)
    f.style_axes(ax)
    chi = D["chi_line"][k] - D["chi_line"][k][s.size // 2]
    lim = max(np.abs(D["chi_line"] - D["chi_line"][:, s.size // 2:s.size // 2 + 1]).max(), 1e-6) * 1.1
    ax.set_xlim(s[0], s[-1])
    ax.set_ylim(-lim, lim)
    ax.plot([s[0], s[-1]], [0, 0], color=(*f.DIM, 0.5 * alpha), lw=0.8 * f.sc)
    ax.plot([0, 0], [-lim, lim], color=(*f.DIM, 0.4 * alpha), lw=0.8 * f.sc, ls=(0, tuple(ST.level_dash)))
    ax.plot(s, chi, color=(*f.GOLD, alpha), lw=ST.curve_width * f.sc)
    ax.plot(s, float(D["d_p"][k]) * s, color=(*f.GREEN, alpha), lw=1.4 * f.sc, ls=(0, tuple(ST.level_dash)))
    ax.text(0.04, 0.93, f.tx["pc_tangent"], color=(*f.GREEN, alpha), fontsize=FT.legend * f.sc, transform=ax.transAxes, ha="left", va="top")
    ax.annotate("", xy=(s[-1] * 0.82, 0.0), xytext=(s[-1] * 0.45, 0.0), arrowprops=dict(arrowstyle="-|>", color=(*f.DIM, alpha), lw=1.2 * f.sc))
    ax.text(s[-1] * 0.82, -0.09 * lim, f.tx["pd_toward"], color=(*f.DIM, alpha), fontsize=FT.legend * f.sc, ha="right", va="top")
    f.label(ax, (0.0, 1.04), f.tx["pc_title"], FT.panel_title, (*f.DIM, alpha), va="bottom")
    f.label(ax, (1.0, -0.17), f.tx["pc_x"], FT.axis_label, (*f.DIM, alpha), ha="right", va="top")
    bx = f.axes(LY.panel_b_axes)
    f.style_axes(bx)
    axis, dfree, dsum, g = momentum_distributions(D, k)
    sel = np.abs(axis) <= VW.momentum_range
    bx.set_xlim(-VW.momentum_range, VW.momentum_range)
    bx.set_ylim(-VW.diff_ylim, 1.25)
    bx.plot([-VW.momentum_range, VW.momentum_range], [0, 0], color=(*f.DIM, 0.4 * alpha), lw=0.8 * f.sc)
    bx.fill_between(axis[sel], 0, dfree[sel], color=(*f.DIM, 0.16 * alpha), lw=0)
    bx.plot(axis[sel], dfree[sel], color=(*f.DIM, alpha), lw=1.4 * f.sc, ls=(0, tuple(ST.level_dash)))
    bx.plot(axis[sel], dsum[sel], color=(*f.GOLD, alpha), lw=ST.curve_width * f.sc)
    diff = (dsum - dfree) * VW.diff_gain
    bx.fill_between(axis[sel], 0, diff[sel], where=diff[sel] >= 0, color=(*f.GREEN, 0.55 * alpha), lw=0)
    bx.fill_between(axis[sel], 0, diff[sel], where=diff[sel] < 0, color=(*f.WARM, 0.55 * alpha), lw=0)
    bx.annotate("", xy=(g, 1.14), xytext=(0, 1.14), arrowprops=dict(arrowstyle="-|>", color=(*f.GOLD, alpha), lw=2.0 * f.sc, shrinkA=0, shrinkB=0))
    bx.text(0.03, 0.93, f.tx["pd_free"], color=(*f.DIM, alpha), fontsize=FT.legend * f.sc, transform=bx.transAxes, ha="left", va="top")
    bx.text(0.03, 0.82, f.tx["pd_full"], color=(*f.GOLD, alpha), fontsize=FT.legend * f.sc, transform=bx.transAxes, ha="left", va="top")
    bx.text(0.03, 0.71, f.tx["pd_gen"], color=(*f.GREEN, alpha), fontsize=FT.legend * f.sc, transform=bx.transAxes, ha="left", va="top")
    f.label(bx, (0.0, 1.04), f.tx["pd_title"], FT.panel_title, (*f.DIM, alpha))
    f.label(bx, (1.0, -0.17), f.tx["pd_x"], FT.axis_label, (*f.DIM, alpha), ha="right", va="top")
    bx.text(0.97, 0.93, fill(f.tx["dp_readout"], dp=num(g, "+.2f", f.lang)), color=(*f.GOLD, alpha), fontsize=FT.readout * f.sc, transform=bx.transAxes, ha="right", va="top")


# ======================================================================================================================= the film

def sim_time_of(t: float) -> float:
    a, b = PARTS["motion"]
    if t < PARTS["intro"][1]:
        return 0.0
    if t < PARTS["interf"][0]:
        return SQ.t_end * (t - a) / (b - a)
    if t < PARTS["outro"][0]:
        a2, b2 = PARTS["interf"]
        return TL.interf_t0 + (TL.interf_t1 - TL.interf_t0) * (t - a2) / (b2 - a2)
    return SQ.t_end


def draw_content(f: Frame, t: float) -> None:
    tx = f.tx
    part = "intro" if t < PARTS["intro"][1] else "motion" if t < PARTS["interf"][0] else "interf" if t < PARTS["outro"][0] else "outro"
    t_sim = sim_time_of(t)
    if part == "intro":
        tau = t - PARTS["intro"][0]
        formulas(f, [("f1", smooth(tau, 0.5, 1.5)), ("f1b", smooth(tau, 3.0, 4.0))])
        draw_map(f, 0.0, smooth(tau, 1.5, 4.5), 0.0, 0.0)
        caption(f, "cI1" if tau < 6.0 else key_of("cI2"))
    elif part == "motion":
        tau = t - PARTS["motion"][0]
        formulas(f, [("f2", smooth(tau, 6.0, 7.0))])
        draw_map(f, t_sim, 1.0, smooth(tau, 6.0, 8.0), smooth(tau, 0.5, 2.0))
        panel_distance(f, t_sim)
        panel_transverse(f, t_sim)
        cap = "cM1" if tau < 10.0 else key_of("cM2") if tau < 34.0 else "cM3"
        caption(f, cap)
    elif part == "interf":
        tau = t - PARTS["interf"][0]
        formulas(f, [("f3", smooth(tau, 0.3, 1.3)), ("f3b", smooth(tau, 8.0, 9.0))])
        draw_map(f, t_sim, 1.0, 0.0, 1.0)
        panels_interference(f, t_sim, smooth(tau, 0.3, 1.5))
        cap = "cF1" if tau < 9.0 else key_of("cF2") if tau < 19.0 else key_of("cF3")
        caption(f, cap)
    else:
        tau = t - PARTS["outro"][0]
        for k in range(4):
            a = smooth(tau, TL.outro_in[k], TL.outro_in[k] + TL.outro_fade)
            f.paragraph(LY.outro_x, LY.outro_y - LY.outro_row * k, tx[key_of(f"o{k + 1}")], FT.outro * 0.85, (*f.TXT, a), LY.outro_width, LY.outro_line)


def render_frame(f: Frame, tf: float, total: float) -> None:
    k = total / TOTAL
    t, card, cprog = timeline(tf / k)
    draw_header(f, t, card, cprog)
    if card is None:
        draw_content(f, t)


def render(out: Path, size: tuple[int, int], fps: int, total: float, lang: str, snap: float | None = None) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if snap is None and not shutil.which("ffmpeg"):
        sys.exit("ffmpeg is required")
    V = CFG.video
    W, H = size
    fig = plt.figure(figsize=(W / V.dpi, H / V.dpi), dpi=V.dpi, facecolor=ST.background)
    bg = ST.background
    bg_rgba = np.array([int(bg[1:3], 16), int(bg[3:5], 16), int(bg[5:7], 16), 255], np.float32)
    frames = int(round(total * fps))
    ids = [int(round(snap * fps))] if snap is not None else range(frames)
    writer = None
    if snap is None:
        writer = subprocess.Popen(["ffmpeg", "-y", "-v", "error", "-f", "rawvideo", "-pix_fmt", "rgba", "-s", f"{W}x{H}", "-r", str(fps), "-i", "-", "-c:v", "libx264",
                                   "-preset", V.preset, "-crf", str(V.crf), "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(out)], stdin=subprocess.PIPE)
    for k_ in ids:
        tf = k_ / fps
        fig.clear()
        fig.patch.set_facecolor(bg)
        f = Frame(fig, size, lang)
        render_frame(f, tf, total)
        fade = min(smooth(tf, 0.0, V.fade_s), 1.0 - smooth(tf, total - V.fade_s, total))
        fig.canvas.draw()
        frame = np.asarray(fig.canvas.buffer_rgba()).astype(np.float32)
        if fade < 1.0:
            frame = bg_rgba + (frame - bg_rgba) * fade
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
    ap.add_argument("--case", choices=("attraction", "repulsion"), default="attraction")
    ap.add_argument("--lang", choices=("en", "ru"), default="en")
    ap.add_argument("--preview", action="store_true")
    ap.add_argument("--snapshot", type=float, default=None, help="one PNG at this CONTENT time (seconds, the cards are not counted)")
    ap.add_argument("--seconds", type=float, default=TOTAL)
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--config", type=Path, default=None, help="another configuration file instead of config.toml")
    ap.add_argument("--set", action="append", default=[], metavar="SECTION.KEY=VALUE", help="override one configuration value")
    args = ap.parse_args()
    set_case(args.case)
    V = CFG.video
    size = (V.preview_width, V.preview_height) if args.preview else (V.width, V.height)
    fps = V.preview_fps if args.preview else V.fps
    name = f"{args.case}_{args.lang}{'_preview' if args.preview else ''}"
    out = args.out or (HERE / "media" / f"{name}.mp4")
    out.parent.mkdir(parents=True, exist_ok=True)
    if args.snapshot is not None:
        t_film = args.snapshot + TL.card_s * sum(1 for ca in CARD_AT if ca <= args.snapshot)
        render(out.with_suffix(".png"), size, fps, args.seconds, args.lang, snap=t_film * args.seconds / TOTAL)
    else:
        render(out, size, fps, args.seconds, args.lang)


if __name__ == "__main__":
    main()
