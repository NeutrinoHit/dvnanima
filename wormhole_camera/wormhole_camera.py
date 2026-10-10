r"""A camera at a wormhole (the book: chapter "General Relativity", the exercises "Wormhole" and "A photograph of the Universe at a wormhole").

The Morris-Thorne wormhole is an exact solution of Einstein's equations (it needs a negative energy density); light moves along the geodesics of its embedding surface
r = b cosh(z/b).  A camera that flies around it collects light from both universes.  Five parts:

 1. The solution of Einstein's equations: the metric, the surface, the throat of radius b, the exotic matter and its energy (the numbers of the book).
 2. Rays of light on the surface: the impact parameter p, rays with p < b cross the throat, rays with p > b turn back; the cone of the rays from the other side.
 3. A view from afar: the ray-traced picture, the other universe in a round window, the camera flies around the wormhole.
 4. Through the throat: the window grows, the camera crosses the throat and looks back.
 5. How to measure the throat: the angular radius of the window as a function of the distance.

All the numbers are in config.toml and all the words in texts.toml (see ../dvconfig.py for --config / --set).

Usage:
    python wormhole_camera.py --lang en            # film -> media/wormhole_camera_en.mp4
    python wormhole_camera.py --lang ru --workers 3
    python wormhole_camera.py --lang en --snapshot 40
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

import wormhole_physics as wp  # noqa: E402

HERE = Path(__file__).resolve().parent
CFG = load_config(HERE)
PB = list(CFG.timeline.part_bounds)
CARD_D = CFG.timeline.title_card
CONTENT_TOTAL = PB[-1]
TOTAL = CONTENT_TOTAL + CARD_D
B = CFG.physics.b
SMOOTH = wp.smooth
TEXT = {lang: load_texts(HERE, lang) for lang in ("en", "ru")}


def timeline(tf: float) -> tuple[float, bool, float]:
    """(content time, is the title card shown, progress of the card) at the film time tf."""
    if tf < CARD_D:
        return 0.0, True, tf / CARD_D
    return tf - CARD_D, False, 0.0


def part_of(tc: float) -> int:
    return max(i for i in range(len(PB) - 1) if PB[i] <= tc) + 1


def num(x: float, nd: int, lang: str) -> str:
    s = f"{x:.{nd}f}"
    if float(s) == 0.0:
        s = s.lstrip("-")
    return s.replace(".", "{,}") if lang == "ru" else s


def fill(text: str, lang: str, **vals) -> str:
    for k, v in vals.items():
        s = num(v[0], v[1], lang) if isinstance(v, tuple) else str(v)
        text = text.replace(f"@{k}@", s)
    return text


def _greedy(toks: list[str], width: int) -> list[str]:
    lines, line = [], ""
    for t in toks:
        vis = len(t) if "$" not in t else max(len(t) // 2, 3)
        if line and len(line) + 1 + vis > width:
            lines.append(line)
            line = t
        else:
            line = (line + " " + t) if line else t
    lines.append(line)
    return lines


def wrap(text: str, width: int) -> str:
    """Break a caption into lines of similar length (at most about `width` characters); a formula $...$ is never split."""
    toks, cur, inmath = [], "", False
    for ch in text:
        if ch == "$":
            inmath = not inmath
        if ch == " " and not inmath:
            toks.append(cur)
            cur = ""
        else:
            cur += ch
    toks.append(cur)
    lines = _greedy(toks, width)
    if len(lines) > 1:
        total = len(text) - 2 * text.count("$") // 2
        lines = _greedy(toks, int(total / len(lines) * 1.12) + 2)
    return "\n".join(lines)


# ----------------------------------------------------------------------------------- numbers of the film

def physics_numbers() -> dict:
    P = CFG.physics
    e_side = wp.side_energy_si(P.b_si_m, P.c_si, P.g_si)
    return {"e_side": -e_side / 1e47, "ratio": 2 * abs(e_side) / P.sun_rest_energy_j, "b_km": P.b_si_m / 1000.0}


def camera_state(tc: float) -> dict:
    """The film camera at the content time tc: l/b, the orbit angle, yaw, pitch, roll (radians for the angles)."""
    from scipy.interpolate import PchipInterpolator
    K = np.array(CFG.camera.frames, float)
    t = float(np.clip(tc, K[0, 0], K[-1, 0]))
    f = [PchipInterpolator(K[:, 0], K[:, i])(t) for i in range(1, 7)]
    return {"l": float(f[0]) * B, "alpha": math.radians(f[1]), "yaw": math.radians(f[2]), "pitch": math.radians(f[3]), "roll": math.radians(f[4]), "fov": float(f[5])}


def view_keys(part: str, t: float) -> tuple[float, float]:
    from scipy.interpolate import PchipInterpolator
    K = np.array(CFG.surface_views[part], float)
    tt = float(np.clip(t, K[0, 0], K[-1, 0]))
    return float(PchipInterpolator(K[:, 0], K[:, 1])(tt)), float(PchipInterpolator(K[:, 0], K[:, 2])(tt))


# ----------------------------------------------------------------------------------- the canvas

class Canvas:
    def __init__(self, size: tuple[int, int], lang: str) -> None:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        self.plt = plt
        self.lang = lang
        self.tx = TEXT[lang]
        S, L, V = CFG.style, CFG.layout, CFG.video
        self.W, self.H = size
        self.sc = self.H / V.reference_height
        self.BG = S.background
        self.TXT, self.DIM = tuple(S.text), tuple(S.dim)
        self.BLUE, self.WARM, self.GOLD, self.VIOLET, self.CYAN = tuple(S.blue), tuple(S.warm), tuple(S.gold), tuple(S.violet), tuple(S.cyan)
        self.fig = plt.figure(figsize=(self.W / V.dpi, self.H / V.dpi), dpi=V.dpi, facecolor=self.BG)
        mk = lambda rect, z=0: self.fig.add_axes(rect, facecolor="none", zorder=z)  # noqa: E731
        self.ax_bg = mk([0, 0, 1, 1], 0)
        self.ax_ov = mk([0, 0, 1, 1], 1)
        self.ax_surf = mk(L.surface_axes, 2)
        self.ax_inset = mk(L.inset_axes, 3)
        self.ax_graph = mk(L.graph_axes, 4)
        self.axes = [self.ax_bg, self.ax_ov, self.ax_surf, self.ax_inset, self.ax_graph]
        self.img_artist = None
        self.geom = self._geometry()
        rng = np.random.default_rng(CFG.surface.stars_seed)
        self.stars = (rng.random((CFG.surface.stars_count, 2)), rng.uniform(0.3, 1.0, CFG.surface.stars_count), rng.uniform(0.2, 0.7, CFG.surface.stars_count))
        self._strip = self._strip_image()

    # ----------------------------------------------------------------- helpers
    def ftext(self, x, y, s, size, color, alpha=1.0, **kw):
        return self.fig.text(x, y, s, color=(*color, alpha), fontsize=size * self.sc, **kw)

    def reset(self, ax, visible=True):
        ax.clear()
        ax.set_facecolor("none")
        ax.axis("off")
        ax.set_visible(visible)

    def _strip_image(self):
        """A dark gradient behind the captions at the bottom of the frame (RGBA, shown on top of the camera view)."""
        S = CFG.style
        h = 256
        a = np.clip(1.0 - np.linspace(0, 1, h) / S.strip_height, 0, 1) ** 1.3 * S.strip_alpha
        img = np.zeros((h, 2, 4))
        img[..., :3] = 0.0
        img[..., 3] = a[::-1, None] if False else a[:, None]
        return img[::-1]

    def _geometry(self):
        G = CFG.surface
        u = np.linspace(-G.u_max, G.u_max, G.n_u + 1)
        ph = np.linspace(0, 2 * np.pi, G.n_phi + 1)
        U, PH = np.meshgrid(u, ph, indexing="ij")
        R = B * np.cosh(U / B * 0 + U)           # u is z/b
        P = np.stack([B * np.cosh(U) * np.cos(PH), B * np.cosh(U) * np.sin(PH), B * U], -1)
        uc = 0.25 * (U[:-1, :-1] + U[1:, :-1] + U[1:, 1:] + U[:-1, 1:])
        pc = 0.25 * (PH[:-1, :-1] + PH[1:, :-1] + PH[1:, 1:] + PH[:-1, 1:])
        N = np.stack([-np.cos(pc), -np.sin(pc), np.sinh(uc)], -1) / np.cosh(uc)[..., None]
        return {"u": u, "ph": ph, "P": P, "uc": uc, "pc": pc, "N": N}

    # ----------------------------------------------------------------- projection and drawing of the surface
    @staticmethod
    def proj(P, el, az):
        """Orthographic projection: P (..., 3) -> screen x, screen y, depth (larger = farther)."""
        ce, se, ca, sa = math.cos(math.radians(el)), math.sin(math.radians(el)), math.cos(math.radians(az)), math.sin(math.radians(az))
        x1 = P[..., 0] * ca - P[..., 1] * sa
        y1 = P[..., 0] * sa + P[..., 1] * ca
        return x1, P[..., 2] * ce + y1 * se, y1 * ce - P[..., 2] * se

    def surface_to_world(self, u, ph):
        return np.stack([B * np.cosh(u) * np.cos(ph), B * np.cosh(u) * np.sin(ph), B * u], -1)

    def draw_surface(self, ax, rect, el, az, half_h, alpha=1.0, exotic=0.0, throat=0.0, rays=(), cam=None, cam_size=70.0, pulse_t=None, radius_line=0.0, show_axis=0.0, edge=1.0, reveal=1.0) -> None:
        from matplotlib.collections import LineCollection, PolyCollection
        G = CFG.surface
        g = self.geom
        self.reset(ax)
        aspect = (rect[2] * self.W) / (rect[3] * self.H)
        ax.set_xlim(-half_h * aspect, half_h * aspect)
        ax.set_ylim(-half_h, half_h)
        ca, sa = math.cos(math.radians(az)), math.sin(math.radians(az))
        ce, se = math.cos(math.radians(el)), math.sin(math.radians(el))
        v_w = np.array([-ce * sa, -ce * ca, se])
        lx, ly, lz = G.light_dir
        L_w = np.array([lx * ca + ly * sa, -lx * sa + ly * ca, lz])
        L_w /= np.linalg.norm(L_w)
        N = g["N"]
        diff = np.abs(N @ L_w)
        rim = (1.0 - np.abs(N @ v_w)) ** G.rim_power
        s = SMOOTH_ARR(g["uc"] / G.tint_width)
        base = (1 - s)[..., None] * np.array(G.cool) + s[..., None] * np.array(G.warm)
        col = base * (G.ambient + (1 - G.ambient) * diff)[..., None] + G.rim_weight * rim[..., None] * np.array([0.7, 0.85, 1.0])
        if exotic > 0:
            col = col + (exotic * G.exotic_gain / np.cosh(g["uc"]) ** 4)[..., None] * np.array(G.exotic_color)
        col = np.clip(col, 0, 1)
        sx, sy, dp = self.proj(g["P"], el, az)
        quad = np.stack([np.stack([sx[:-1, :-1], sy[:-1, :-1]], -1), np.stack([sx[1:, :-1], sy[1:, :-1]], -1),
                         np.stack([sx[1:, 1:], sy[1:, 1:]], -1), np.stack([sx[:-1, 1:], sy[:-1, 1:]], -1)], 2)
        depth = 0.25 * (dp[:-1, :-1] + dp[1:, :-1] + dp[1:, 1:] + dp[:-1, 1:])
        order = np.argsort(-depth.ravel())
        quad = quad.reshape(-1, 4, 2)[order]
        qa = np.clip((reveal * (G.u_max + 0.3) - np.abs(g["uc"])) / 0.3, 0, 1).ravel()[order]
        fc = np.concatenate([col.reshape(-1, 3)[order], (G.face_alpha * alpha * qa)[:, None]], 1)
        ec = np.stack([np.ones(len(order)), np.ones(len(order)), np.ones(len(order)), G.edge_alpha * alpha * edge * qa], 1)
        ax.add_collection(PolyCollection(quad, facecolors=fc, edgecolors=ec, linewidths=G.edge_width * self.sc, zorder=1))
        # the axis and the radius b
        if show_axis > 0:
            a = self.proj(np.array([[0, 0, -G.u_max * B], [0, 0, G.u_max * B]]), el, az)
            ax.plot(a[0], a[1], color=(*self.DIM, 0.45 * show_axis * alpha), lw=0.8 * self.sc, ls=(0, (4, 4)), zorder=2)
        if radius_line > 0:
            pts = self.proj(np.array([[0, 0, 0], [B * radius_line, 0, 0]]), el, az)
            ax.plot(pts[0], pts[1], color=(*self.GOLD, 0.95 * alpha), lw=2.2 * self.sc, zorder=6)
            ax.scatter(pts[0][:1], pts[1][:1], s=(5 * self.sc) ** 2, color=[(*self.GOLD, alpha)], zorder=7)
        # the throat circle
        if throat > 0:
            c = self.surface_to_world(np.zeros(181), np.linspace(0, 2 * np.pi, 181))
            cx, cy, _ = self.proj(c, el, az)
            for w, a_ in ((G.throat_glow_width, 0.12), (G.throat_glow_width * 0.55, 0.20), (G.throat_width, 0.95)):
                ax.plot(cx, cy, color=(*self.GOLD, a_ * throat * alpha), lw=w * self.sc, zorder=5, solid_capstyle="round")
        # rays
        for r in rays:
            xyz = self.surface_to_world(r["u"], r["ph"])
            rx, ry, _ = self.proj(xyz, el, az)
            n = len(rx)
            k = max(2, int(n * r.get("reveal", 1.0)))
            for w, a_ in ((G.ray_width * 3.2, 0.10), (G.ray_width * 1.8, 0.22), (G.ray_width, G.ray_alpha)):
                ax.plot(rx[:k], ry[:k], color=(*r["color"], a_ * r.get("alpha", 1.0) * alpha), lw=w * self.sc, zorder=8, solid_capstyle="round")
            if pulse_t is not None and r.get("pulses", True) and k > 20:
                for j in range(int(G.pulse_count)):
                    f = (pulse_t * G.pulse_speed + r.get("offset", 0.0) + j / G.pulse_count) % 1.0
                    idx = int((1.0 - f) * (k - 1))
                    ax.scatter(rx[idx:idx + 1], ry[idx:idx + 1], s=(G.pulse_size * self.sc) ** 2 / 36, color=[(1, 1, 1, 0.95 * r.get("alpha", 1.0) * alpha)], zorder=9, linewidths=0)
                    ax.scatter(rx[idx:idx + 1], ry[idx:idx + 1], s=(G.pulse_size * self.sc) ** 2 / 9, color=[(*r["color"], 0.30 * r.get("alpha", 1.0) * alpha)], zorder=9, linewidths=0)
        # the camera
        if cam is not None:
            p = self.proj(self.surface_to_world(np.array([cam[0]]), np.array([cam[1]])), el, az)
            ax.scatter(p[0], p[1], s=(cam_size * self.sc) ** 2 / 6, color=[(*self.GOLD, 0.25 * alpha)], zorder=11, linewidths=0)
            ax.scatter(p[0], p[1], s=(cam_size * self.sc) ** 2 / 20, color=[(1, 0.97, 0.85, alpha)], edgecolors=[(*self.GOLD, alpha)], linewidths=1.6 * self.sc, zorder=12)
            self._cam_screen = (float(p[0][0]), float(p[1][0]))

    # ----------------------------------------------------------------- rays for the drawing
    def ray_set(self, l_cam: float, phi0: float, psis, reveal=1.0):
        out = []
        for i, psi in enumerate(psis):
            u, ph, thr = wp.ray_path(psi, l_cam, B, CFG.surface.u_max + 0.15, int(CFG.surface.ray_points))
            keep = np.abs(u) <= CFG.surface.u_max
            if not keep.all():
                last = np.argmax(~keep)
                u, ph = u[:last], ph[:last]
            col = self.CYAN if thr else self.WARM
            out.append({"u": u, "ph": ph + phi0, "color": col, "reveal": reveal, "offset": (i * 0.137) % 1.0})
        return out

    # ----------------------------------------------------------------- text helpers
    def formula(self, key, row, alpha, size="formula", **vals):
        L, F = CFG.layout, CFG.fonts
        if alpha > L.formula_min_alpha:
            self.ftext(L.formula_x, L.formula_y[row], fill(self.tx[key], self.lang, **vals), F[size], self.TXT, alpha, ha="left", va="center")

    def caption(self, t, times, keys, **vals):
        """keys[i] is shown during times[i] = [t0, t1] with a short fade."""
        L, F = CFG.layout, CFG.fonts
        for (t0, t1), key in zip(times, keys):
            a = min(SMOOTH(t, t0, t0 + 0.7), 1.0 - SMOOTH(t, t1 - 0.7, t1))
            if a > 0.01:
                self.ftext(*L.caption_pos, wrap(fill(self.tx[key], self.lang, **vals), 108), F.caption, self.TXT, a, ha="center", va="bottom", linespacing=L.caption_spacing)

    def dark_background(self, alpha=1.0):
        ax = self.ax_bg
        self.reset(ax)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        p, sz, al = self.stars
        ax.scatter(p[:, 0], p[:, 1], s=(sz * 1.6 * self.sc) ** 2, color=[(*self.TXT, a * alpha) for a in al], linewidths=0)

    # ----------------------------------------------------------------- one frame
    def draw(self, t_film: float, total: float) -> None:
        S, L, F = CFG.style, CFG.layout, CFG.fonts
        tx, DIM = self.tx, self.DIM
        k = total / TOTAL
        t, card, cprog = timeline(t_film / k)
        self.fig.texts.clear()
        for ax in self.axes:
            self.reset(ax, visible=False)
        if card:
            a_c = min(SMOOTH(cprog, *CFG.timeline.card_fade_in), 1.0 - SMOOTH(cprog, *CFG.timeline.card_fade_out))
            self.ftext(0.5, L.card_title_y, tx["title"], F.card_title, S.title_color[:3], a_c, ha="center", va="center")
            self.ftext(0.5, L.card_subtitle_y, tx["subtitle"], F.card_subtitle, DIM, a_c, ha="center", va="center")
            return
        p = part_of(t)
        tl = t - PB[p - 1]
        {1: self.part1, 2: self.part2, 3: self.part3, 4: self.part4, 5: self.part5}[p](tl, t)
        self.ftext(*L.title_pos, tx["title"], F.title, S.title_color[:3], S.title_color[3] * 0.0 if p >= 3 else S.title_color[3])
        lab_a = min(SMOOTH(tl, 0.0, 0.8), 1.0 - SMOOTH(tl, PB[p] - PB[p - 1] - 0.8, PB[p] - PB[p - 1]))
        self.ftext(*L.chapter_label_pos, f"{p}/5   " + tx[f"p{p}"], F.chapter_label, DIM, 0.9 * max(lab_a, 0.55 if p >= 3 else 0.0), ha="right")

    # ================================================================= part 1: the solution
    def part1(self, t: float, tc: float) -> None:
        P, G, L, F = CFG.part1, CFG.surface, CFG.layout, CFG.fonts
        nums = physics_numbers()
        el, az = view_keys("part1", t)
        a_s = SMOOTH(t, *P.surface_fade) * (1.0 - SMOOTH(t, *P.fade_out))
        self.dark_background(1.0 - SMOOTH(t, *P.fade_out))
        thr = SMOOTH(t, *P.throat_times) * 1.0
        rad = SMOOTH(t, *P.throat_times)
        exo = SMOOTH(t, *P.exotic_times)
        self.draw_surface(self.ax_surf, L.surface_axes, el, az, G.view_half_height, alpha=a_s, exotic=exo, throat=thr, radius_line=rad, show_axis=rad, reveal=SMOOTH(t, 0.3, 5.5))
        ax = self.ax_surf
        fa = 1.0 - SMOOTH(t, *P.fade_out)
        self.formula("metric", 0, SMOOTH(t, *P.metric_times) * fa)
        self.formula("profile", 1, SMOOTH(t, *P.profile_times) * fa)
        a_sh = SMOOTH(t, *P.sheet_label_times) * (1.0 - SMOOTH(t, P.exotic_times[0], P.exotic_times[0] + 1.5)) * fa
        # sheet labels at the screen positions of the two sheets
        top = self.proj(np.array([[0, 0, 1.55 * B]]), el, az)
        bot = self.proj(np.array([[0, 0, -1.55 * B]]), el, az)
        self.sheet_label(ax, top, self.tx["side_up"], self.WARM, a_sh)
        self.sheet_label(ax, bot, self.tx["side_down"], self.BLUE, a_sh)
        if rad > 0.02:
            q = self.proj(np.array([[0.5 * B, 0, 0]]), el, az)
            ax.text(float(q[0][0]), float(q[1][0]) - 0.28, "$b$", color=(*self.GOLD, rad * fa), fontsize=F.label * 1.3 * self.sc, ha="center", va="top", zorder=20)
            tl = self.proj(np.array([[-1.0 * B, 0, 0]]), el, az)
            ax.text(float(tl[0][0]) - 0.35, float(tl[1][0]) + 0.05, fill(self.tx["throat_label"], self.lang), color=(*self.GOLD, rad * fa), fontsize=F.label * self.sc, ha="right", va="center", zorder=20,
                    bbox=dict(boxstyle="round,pad=0.35", facecolor=(0.01, 0.02, 0.05, 0.62 * rad * fa), edgecolor=(*self.GOLD, 0.35 * rad * fa), linewidth=0.8 * self.sc))
        self.formula("rho", 2, SMOOTH(t, *P.exotic_formula_times) * fa, size="formula")
        self.formula("e_side", 3, SMOOTH(t, *P.energy_times) * fa, **{"e_side": (nums["e_side"], 2)})
        a_e = SMOOTH(t, P.energy_times[0] + 1.0, P.energy_times[1] + 1.0) * fa
        if a_e > 0.02:
            self.ftext(L.formula_x, L.formula_y[3] - 0.085, fill(self.tx["e_total"], self.lang, ratio=(nums["ratio"], 1), b_km=(nums["b_km"], 0)), F.formula * 0.9, self.GOLD, a_e, ha="left", va="center")
        self.caption(t, P.caption_times, ["c1a", "c1b", "c1c", "c1d", "c1e"], b_km=(nums["b_km"], 0))

    def sheet_label(self, ax, pt, text, color, alpha):
        if alpha > 0.02:
            ax.text(float(pt[0][0]), float(pt[1][0]), text, color=(*color, alpha), fontsize=CFG.fonts.label * self.sc, ha="center", va="center", zorder=20)

    # ================================================================= part 2: the rays
    def part2(self, t: float, tc: float) -> None:
        P, G, L, F = CFG.part2, CFG.surface, CFG.layout, CFG.fonts
        el, az = view_keys("part2", t)
        fa = SMOOTH(t, 0.0, 1.0) * (1.0 - SMOOTH(t, *P.fade_out))
        self.dark_background(1.0 - SMOOTH(t, *P.fade_out))
        K = np.array(P.camera_l, float)
        l_c = float(np.interp(t, K[:, 0], K[:, 1]))
        l_c = float(np.interp(t, K[:, 0], K[:, 1]))
        # smooth the camera motion
        from scipy.interpolate import PchipInterpolator
        l_c = float(PchipInterpolator(K[:, 0], K[:, 1])(np.clip(t, K[0, 0], K[-1, 0])))
        psis = np.concatenate([-np.linspace(0.0, math.radians(G.ray_psi_max_deg), (int(G.ray_count) + 1) // 2)[::-1], np.linspace(0.0, math.radians(G.ray_psi_max_deg), (int(G.ray_count) + 1) // 2)[1:]])
        psis = psis[np.abs(psis) > 1e-3] if False else psis
        reveal = SMOOTH(t, *P.fan_times)
        rays = self.ray_set(l_c, 0.0, psis, reveal=1.0)
        # the fan opens from the middle
        for r, psi in zip(rays, psis):
            r["alpha"] = SMOOTH(reveal, 0.0, 1.0) * SMOOTH(abs(psi) / (math.radians(G.ray_psi_max_deg) + 1e-9), 0.0, 1.0) * 0 + SMOOTH(reveal * 1.4 - abs(psi) / math.radians(G.ray_psi_max_deg) * 0.4, 0.0, 1.0)
            r["reveal"] = min(1.0, reveal * 1.15)
        cls = SMOOTH(t, *P.class_times)
        self.draw_surface(self.ax_surf, L.surface_axes, el, az, G.view_half_height, alpha=fa, rays=rays, cam=(math.asinh(l_c / B), 0.0), cam_size=G.camera_size, pulse_t=t if reveal > 0.9 else None, throat=0.55 * fa)
        ax = self.ax_surf
        if hasattr(self, "_cam_screen"):
            ax.text(self._cam_screen[0], self._cam_screen[1] + 0.32, self.tx["camera"], color=(*self.GOLD, fa), fontsize=F.label * self.sc, ha="center", va="bottom", zorder=20)
        self.formula("p_def", 0, SMOOTH(t, *P.fan_times) * fa, size="formula")
        self.formula("p_through", 1, cls * fa)
        self.formula("p_back", 2, cls * fa)
        if cls * fa > 0.02:
            self.ftext(L.formula_x - 0.012, L.formula_y[1], "●", F.formula, self.CYAN, cls * fa, ha="right", va="center")
            self.ftext(L.formula_x - 0.012, L.formula_y[2], "●", F.formula, self.WARM, cls * fa, ha="right", va="center")
        cone = SMOOTH(t, *P.cone_times)
        self.formula("cone", 3, cone * fa, size="formula_big")
        if cone * fa > 0.02:
            psi_c = math.degrees(wp.critical_angle(l_c, B))
            self.ftext(L.formula_x, L.formula_y[3] - 0.09, fill(self.tx["r_psi"], self.lang, psi=(psi_c, 1)), F.readout, self.GOLD, cone * fa, ha="left", va="center")
        self.caption(t, P.caption_times, ["c2a", "c2b", "c2c"])

    # ================================================================= the camera parts 3, 4, 5
    def view_frame(self, tc: float, alpha_view: float, dim: float = 1.0):
        st = camera_state(tc)
        import wormhole_view as wv
        img, _ = wv.render_view(st["l"], st["alpha"], st["yaw"], st["pitch"], st["roll"], self.W, self.H, frame_seed=int(tc * 30), fov=st["fov"])
        ax = self.ax_bg
        self.reset(ax)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.imshow(img * dim, extent=(0, 1, 0, 1), aspect="auto", interpolation="nearest", alpha=alpha_view, zorder=0)
        axo = self.ax_ov
        self.reset(axo)
        axo.set_xlim(0, 1)
        axo.set_ylim(0, 1)
        axo.imshow(self._strip, extent=(0, 1, 0, CFG.style.strip_height), aspect="auto", interpolation="bilinear", alpha=alpha_view)
        return st

    def inset(self, st: dict, alpha: float) -> None:
        I, G, L = CFG.inset, CFG.surface, CFG.layout
        if alpha < 0.01:
            return
        r = L.inset_axes
        pad = I.panel_pad
        from matplotlib.patches import FancyBboxPatch
        axo = self.ax_ov
        axo.add_patch(FancyBboxPatch((r[0] - pad, r[1] - pad), r[2] + 2 * pad, r[3] + 2 * pad, boxstyle="round,pad=0,rounding_size=0.01", transform=axo.transAxes,
                                     facecolor=(*I.bg_color, I.bg_alpha * alpha), edgecolor=(*self.DIM, 0.25 * alpha), linewidth=0.8 * self.sc, zorder=0))
        l = st["l"]
        l_lim = CFG.inset.l_max
        if abs(l) > l_lim:
            # the camera is beyond the part of the surface that is drawn: it waits at the rim
            u_rim = math.copysign(0.97 * G.u_max, l)
            self.draw_surface(self.ax_inset, L.inset_axes, I.elevation, I.azimuth, I.view_half_height, alpha=alpha, cam=(u_rim, st["alpha"]), cam_size=I.camera_size, throat=0.5 * alpha, edge=0.8)
            return
        a_r = SMOOTH(l_lim - abs(l), 0.0, 0.5)
        psis = [s * math.radians(a) for a in I.ray_psi_deg for s in ((1,) if a == 0 else (1, -1))]
        rays = self.ray_set(l, st["alpha"], psis)
        for rr in rays:
            rr["reveal"] = 1.0
            rr["alpha"] = a_r
        self.draw_surface(self.ax_inset, L.inset_axes, I.elevation, I.azimuth, I.view_half_height, alpha=alpha, rays=rays, cam=(math.asinh(l / B), st["alpha"]), cam_size=I.camera_size,
                          throat=0.5 * alpha, edge=0.8, pulse_t=None)

    def readout(self, st: dict, alpha: float) -> None:
        L, F = CFG.layout, CFG.fonts
        if alpha < 0.02:
            return
        psi_c = math.degrees(wp.critical_angle(st["l"], B))
        side = self.tx["side_ours"] if st["l"] >= 0 else self.tx["side_theirs"]
        x, y = L.readout_pos
        self.ftext(x, y, side, F.readout_small, self.DIM, alpha, ha="left", va="center")
        self.ftext(x, y - L.readout_row, fill(self.tx["r_l"], self.lang, l=(abs(st["l"]) / B, 1)), F.readout, self.TXT, alpha, ha="left", va="center")
        self.ftext(x, y - 2 * L.readout_row, fill(self.tx["r_psi"], self.lang, psi=(psi_c, 1)), F.readout, self.GOLD, alpha, ha="left", va="center")

    def part3(self, t: float, tc: float) -> None:
        P = CFG.part3
        a = SMOOTH(t, *P.fade_in)
        st = self.view_frame(tc, a)
        self.inset(st, a * SMOOTH(t, *CFG.inset.alpha_in))
        self.readout(st, SMOOTH(t, *P.readout_times))
        self.caption(t, P.caption_times, ["c3a", "c3b"])

    def part4(self, t: float, tc: float) -> None:
        P = CFG.part4
        st = self.view_frame(tc, 1.0)
        self.inset(st, 1.0)
        self.readout(st, 1.0)
        self.caption(t, P.caption_times, ["c4a", "c4b"])

    def part5(self, t: float, tc: float) -> None:
        P, L, F = CFG.part5, CFG.layout, CFG.fonts
        fo = 1.0 - SMOOTH(t, *P.fade_out)
        d0, d1, dm = P.summary_dim_view
        dim = 1.0 - (1.0 - dm) * SMOOTH(t, d0, d1)
        st = self.view_frame(tc, fo, dim)
        self.inset(st, 1.0 - SMOOTH(t, d0, d1))
        self.readout(st, 1.0 - SMOOTH(t, d0, d1))
        self.graph(st, SMOOTH(t, *P.graph_times) * (1.0 - SMOOTH(t, d0, d1)))
        self.caption(t, P.caption_times, ["c4c", "c5a", "c5b"])
        for key, (t0, t1), y, fk, gold in zip(P.summary_lines, P.summary_times, P.summary_y, P.summary_fonts, P.summary_gold):
            a = SMOOTH(t, t0, t1) * fo
            if a > 0.01:
                self.ftext(0.5, y, self.tx[key], F[fk], self.GOLD if gold else self.TXT, a, ha="center", va="center")

    def graph(self, st: dict, alpha: float) -> None:
        if alpha < 0.02:
            return
        Gr, L, F = CFG.graph, CFG.layout, CFG.fonts
        ax = self.ax_graph
        self.reset(ax)
        ax.set_visible(True)
        ax.set_xlim(0, Gr.l_max)
        ax.set_ylim(0, Gr.psi_max_deg)
        from matplotlib.patches import FancyBboxPatch
        r = L.graph_axes
        axo = self.ax_ov
        axo.add_patch(FancyBboxPatch((r[0] - 0.045, r[1] - 0.075), r[2] + 0.06, r[3] + 0.14, boxstyle="round,pad=0,rounding_size=0.01", transform=axo.transAxes,
                                     facecolor=(*CFG.inset.bg_color, CFG.inset.bg_alpha * alpha), edgecolor=(*self.DIM, 0.25 * alpha), linewidth=0.8 * self.sc, zorder=0))
        x = np.linspace(0, Gr.l_max, 300)
        ax.plot(x, np.degrees(np.arcsin(B / np.hypot(x, B))), color=(*self.BLUE, alpha), lw=Gr.line_width * self.sc)
        l = abs(st["l"]) / B
        psi = math.degrees(wp.critical_angle(st["l"], B))
        ax.plot([l, l], [0, psi], color=(*self.GOLD, 0.5 * alpha), lw=0.9 * self.sc, ls=(0, (3, 3)))
        ax.scatter([min(l, Gr.l_max)], [psi], s=(Gr.dot_size * self.sc) ** 2, color=[(*self.GOLD, alpha)], zorder=5)
        self.atext(ax, 0.5, 1.07, fill(self.tx["graph_title"], self.lang), F.label, self.TXT, alpha, ha="center", va="bottom")
        self.atext(ax, 0.5, -0.15, self.tx["graph_x"], F.readout_small, self.DIM, alpha, ha="center", va="top")
        self.atext(ax, -0.15, 0.5, self.tx["graph_y"], F.readout_small, self.DIM, alpha, ha="right", va="center", rotation=90)
        ax.tick_params(colors=(*self.DIM, alpha), labelsize=F.tick * self.sc, length=2)
        for sp in ax.spines.values():
            sp.set_visible(True)
            sp.set_color((*self.DIM, 0.5 * alpha))
        ax.axis("on")
        ax.set_facecolor("none")

    def atext(self, ax, x, y, s, size, color, alpha=1.0, **kw):
        kw.setdefault("transform", ax.transAxes)
        return ax.text(x, y, s, color=(*color, alpha), fontsize=size * self.sc, **kw)


def SMOOTH_ARR(x):
    t = np.clip(x, -1, 1) * 0.5 + 0.5
    return t * t * (3 - 2 * t)


# ----------------------------------------------------------------------------------- the film

def render(out: Path, size, fps, total, lang, snap=None, frame_range=None) -> None:
    V = CFG.video
    if snap is None and not shutil.which("ffmpeg"):
        sys.exit("ffmpeg is required")
    cv = Canvas(size, lang)
    W, H = size
    frames = int(round(total * fps))
    if snap is not None:
        ids = [int(round(snap * fps))]
    else:
        a, b_ = frame_range if frame_range else (0, frames)
        ids = range(a, b_)
    writer = None
    if snap is None:
        writer = subprocess.Popen(["ffmpeg", "-y", "-v", "error", "-f", "rawvideo", "-pix_fmt", "rgba", "-s", f"{W}x{H}", "-r", str(fps), "-i", "-", "-c:v", "libx264",
                                   "-preset", V.preset, "-crf", str(V.crf), "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(out)], stdin=subprocess.PIPE)
    bg = np.array([int(cv.BG[1:3], 16), int(cv.BG[3:5], 16), int(cv.BG[5:7], 16), 255], np.float32)
    for k_ in ids:
        t_film = k_ / fps
        cv.draw(t_film, total)
        fade_io = min(SMOOTH(t_film, 0.0, V.fade_s), 1.0 - SMOOTH(t_film, total - V.fade_s, total))
        cv.fig.canvas.draw()
        frame = np.asarray(cv.fig.canvas.buffer_rgba()).astype(np.float32)
        if fade_io < 1.0:
            frame = bg + (frame - bg) * fade_io
        frame = frame.clip(0, 255).astype(np.uint8)
        if writer is None:
            from PIL import Image
            Image.fromarray(frame).save(out)
        else:
            writer.stdin.write(frame.tobytes())
    if writer is not None:
        writer.stdin.close()
        writer.wait()
    cv.plt.close(cv.fig)
    print(f"wrote {out}")


def render_parallel(out: Path, size, fps, total, lang, workers: int, extra: list[str]) -> None:
    frames = int(round(total * fps))
    cuts = np.linspace(0, frames, workers + 1).astype(int)
    parts = []
    procs = []
    for i in range(workers):
        part = out.with_name(f"{out.stem}.part{i}.mp4")
        parts.append(part)
        cmd = [sys.executable, str(Path(__file__).resolve()), "--lang", lang, "--out", str(part), "--range", str(cuts[i]), str(cuts[i + 1]), "--seconds", str(total)] + extra
        procs.append(subprocess.Popen(cmd))
    for p in procs:
        if p.wait() != 0:
            sys.exit("a worker failed")
    lst = out.with_name(out.stem + ".parts.txt")
    lst.write_text("".join(f"file '{p.name}'\n" for p in parts))
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-f", "concat", "-safe", "0", "-i", str(lst), "-c", "copy", "-movflags", "+faststart", str(out)], check=True)
    for p in parts:
        p.unlink()
    lst.unlink()
    print(f"wrote {out}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--lang", choices=("en", "ru"), default="en")
    ap.add_argument("--preview", action="store_true")
    ap.add_argument("--snapshot", type=float, default=None, help="one PNG at this film time (s)")
    ap.add_argument("--seconds", type=float, default=TOTAL)
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--workers", type=int, default=1, help="render the film in this many parallel pieces")
    ap.add_argument("--range", type=int, nargs=2, default=None, help=argparse.SUPPRESS)
    ap.add_argument("--small", action="store_true", help=argparse.SUPPRESS)
    ap.add_argument("--config", type=Path, default=None, help="another configuration file instead of config.toml")
    ap.add_argument("--set", action="append", default=[], metavar="SECTION.KEY=VALUE", help="override one configuration value")
    args = ap.parse_args()
    V = CFG.video
    out = args.out or HERE / "media" / f"wormhole_camera_{args.lang}.mp4"
    out.parent.mkdir(parents=True, exist_ok=True)
    extra = []
    if args.config:
        extra += ["--config", str(args.config)]
    for s in args.set:
        extra += ["--set", s]
    full = (V.width, V.height, V.fps)
    small = (V.preview_width, V.preview_height, V.preview_fps)
    if args.snapshot is not None:
        render(out.with_suffix(".png"), full[:2], full[2], args.seconds, args.lang, snap=args.snapshot)
    elif args.range:
        w, h, f = small if args.small else full
        render(out, (w, h), f, args.seconds, args.lang, frame_range=tuple(args.range))
    elif args.preview:
        w, h, f = small
        target = out.with_name(out.stem + "_preview.mp4")
        if args.workers > 1:
            render_parallel(target, (w, h), f, args.seconds, args.lang, args.workers, extra + ["--small"])
        else:
            render(target, (w, h), f, args.seconds, args.lang)
    elif args.workers > 1:
        render_parallel(out, full[:2], full[2], args.seconds, args.lang, args.workers, extra)
    else:
        render(out, full[:2], full[2], args.seconds, args.lang)


if __name__ == "__main__":
    main()
