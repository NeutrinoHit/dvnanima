r"""The gauge principle: two friends and a field (the book: chapter "Gauge Invariance", the box "A Shift of Phase" and the sections "QED in a New Way",
"The Covariant Derivative" and "The Geometry of Gauge Invariance").

The phase of the wave function is a convention.  Two friends (an experimenter and a theorist) may choose it independently at every point, and then
the derivative has to be replaced by the covariant one, D = d + i q A, with a gauge field A that transforms as A -> A - d(alpha)/q.  Six parts:

 1. Two friends and a phase (an illustration; the numbers on the screen are computed).
 2. Two paths: a common phase of both paths does not move the interference fringes, a relative phase does (exact paraxial beams).
 3. A convention at every point: a packet on a lattice, rephased by alpha(x), obeys the free equation no longer (exact time evolution).
 4. The compensating field: the lattice Wilson-line links U = exp(+i q theta) make the equation covariant; the packet follows the theorist's prediction.
 5. The field strength and the loop: a pure-gauge field has F = 0; a flux tube; the Wilson loop is q times the flux (Stokes) and it moves the fringes
    (the Aharonov-Bohm effect) whatever the gauge.
 6. Summary.

All the numbers are in config.toml and all the words in texts.toml (see ../dvconfig.py for --config / --set).

Usage:
    python gauge_principle.py --lang en            # film -> media/gauge_principle_en.mp4
    python gauge_principle.py --lang ru
    python gauge_principle.py --lang en --snapshot 20
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

import gauge_physics as gp  # noqa: E402

HERE = Path(__file__).resolve().parent
CFG = load_config(HERE)

CARD_D = list(CFG.timeline.card_durations)
CARD_AT = list(CFG.timeline.card_at)
CARD_KEYS = [f"h{i}" for i in range(len(CARD_AT))]
PB = list(CFG.timeline.part_bounds)
CONTENT_TOTAL = PB[-1]
TOTAL = CONTENT_TOTAL + sum(CARD_D)
SMOOTH = gp.smooth
TWO_PI = gp.TWO_PI


def timeline(tf: float) -> tuple[float, int | None, float]:
    """(content time, index of the chapter card or None, progress of the card in [0, 1]) at the film time tf."""
    shown = 0.0
    for i, ca in enumerate(CARD_AT):
        a = ca + shown
        if a <= tf < a + CARD_D[i]:
            return ca, i, (tf - a) / CARD_D[i]
        if tf >= a + CARD_D[i]:
            shown += CARD_D[i]
    return tf - shown, None, 0.0


def film_time(tc: float) -> float:
    return tc + sum(d for ca, d in zip(CARD_AT, CARD_D) if ca <= tc)


def chapter_of(tc: float) -> int:
    return max(i for i, ca in enumerate(CARD_AT) if ca <= tc)


def part_of(tc: float) -> int:
    """1 .. 6: the part of the film at the content time tc."""
    return max(i for i in range(len(PB) - 1) if PB[i] <= tc) + 1


# ----------------------------------------------------------------------------------- texts

TEXT = {lang: load_texts(HERE, lang) for lang in ("en", "ru")}


def num(x: float, nd: int, lang: str, math_mode: bool = True) -> str:
    """A number with nd decimals; in Russian the decimal comma ({,} inside a formula)."""
    s = f"{x:.{nd}f}"
    if float(s) == 0.0:
        s = s.lstrip("-")
    if lang == "ru":
        s = s.replace(".", "{,}" if math_mode else ",")
    return s


def sci(x: float, lang: str, nd: int = 1) -> str:
    """x as 3.1\\cdot10^{-14} (for use inside a formula)."""
    if x <= 0.0:
        return "0"
    m, e = gp.sci_parts(x)
    if round(m, nd) >= 10.0:
        m, e = 1.0, e + 1
    return f"{num(m, nd, lang)}\\cdot10^{{{e}}}"


def fill(text: str, lang: str, **vals) -> str:
    """Replace @name@ by the value (a number is given as (value, decimals) and formatted with the decimal comma in Russian)."""
    for k, v in vals.items():
        s = num(v[0], v[1], lang) if isinstance(v, tuple) else str(v)
        text = text.replace(f"@{k}@", s)
    return text


# ----------------------------------------------------------------------------------- the physics of the film

class Interferometer:
    """Two slits, a screen: exact paraxial beams (see gauge_physics.beam); the geometry in the plane of the film."""

    def __init__(self) -> None:
        I = CFG.interf
        self.k = TWO_PI / I.wavelength
        self.sigma = I.slit_sigma
        self.slit_y = [I.slit_separation / 2.0, -I.slit_separation / 2.0]        # path 1 (upper), path 2 (lower)
        self.barrier_x = I.barrier_x
        self.screen_x = I.screen_x
        self.length = I.screen_x - I.barrier_x
        self.y_screen = np.linspace(-I.screen_half, I.screen_half, I.screen_samples)
        ref = np.abs(self.screen_field(self.y_screen, [0.0, 0.0])) ** 2
        self.i_ref_max = float(ref.max())
        self.peak0 = gp.peak_near(self.y_screen, ref)
        self.period = gp.fringe_period(self.y_screen, ref)

    def screen_field(self, y, phases):
        return gp.two_path_field(self.length, y, self.slit_y, self.k, self.sigma, phases)

    def screen_intensity(self, phases) -> np.ndarray:
        return np.abs(self.screen_field(self.y_screen, phases)) ** 2 / self.i_ref_max

    def screen_intensity_phases(self, phase_funcs) -> np.ndarray:
        """phase_funcs[j] = array (over y_screen) of the phase of path j at the screen (Wilson line + the phase at the slit)."""
        psi = 0.0
        for y0, ph in zip(self.slit_y, phase_funcs):
            psi = psi + np.exp(1j * ph) * gp.beam(self.length, self.y_screen, y0, self.k, self.sigma)
        return np.abs(psi) ** 2 / self.i_ref_max

    def central_shift(self, intensity: np.ndarray, expected: float = 0.0) -> float:
        """The displacement of the maximum that is nearest to the expected position, in fringe periods."""
        pk = gp.peak_near(self.y_screen, intensity * self.i_ref_max, near=self.peak0 + expected * self.period)
        return (pk - self.peak0) / self.period

    def field_map(self, x: np.ndarray, y: np.ndarray, phases) -> np.ndarray:
        """psi on the grid (x, y) (meshgrid arrays) between the barrier and the screen."""
        return gp.two_path_field(x - self.barrier_x, y, self.slit_y, self.k, self.sigma, phases) / math.sqrt(self.i_ref_max)


class ChainSim:
    """The lattice particle of parts 3 and 4: the true state, the naively rephased one and the covariantly evolved one."""

    def __init__(self) -> None:
        C = CFG.chain
        self.q = C.charge
        self.ch = gp.Chain(C.n, C.half_width, C.mass, self.q)
        x = self.ch.x
        self.kappa, self.beta, self.kosc = C.alpha_slope, C.alpha_amp, C.alpha_k
        self.alpha = self.kappa * x + self.beta * np.sin(self.kosc * x)
        self.psi0 = self.ch.packet(C.packet_x0, C.packet_sigma, C.packet_k0)
        self.psi0_new = np.exp(1j * self.alpha) * self.psi0                         # the same state in the experimenter's convention
        self.theta_a = self.ch.links(self.a_of)                                     # the links of the compensating field A = -alpha'/q
        self.ev_free = self.ch.evolver(None)
        self.ev_cov = self.ch.evolver(self.theta_a)
        self.c_true = self.ev_free.coefficients(self.psi0)
        self.c_naive = self.ev_free.coefficients(self.psi0_new)
        self.c_cov = self.ev_cov.coefficients(self.psi0_new)
        self.mean_dalpha = self.ch.mean_of(self.dalpha(x), self.psi0)
        self.t_end = C.t_end

    def dalpha(self, x):
        return self.kappa + self.beta * self.kosc * np.cos(self.kosc * np.asarray(x, dtype=float))

    def a_of(self, x):
        return -self.dalpha(x) / self.q

    def states(self, tau: float):
        return self.ev_free.at(self.c_true, tau), self.ev_free.at(self.c_naive, tau), self.ev_cov.at(self.c_cov, tau)

    def packet_speed(self, which: str) -> float:
        """The exact drift velocity of the centre: (k_0 + <alpha'>)/m for the naive packet and k_0/m for the true one (free evolution: d<x>/dt = <p>/m)."""
        C = CFG.chain
        return (C.packet_k0 + (self.mean_dalpha if which == "naive" else 0.0)) / C.mass

    def derivative_defects(self, psi_true: np.ndarray, psi_new: np.ndarray) -> tuple[float, float]:
        """Relative defects || d psi' - e^{i alpha} d psi || / || d psi || of the naive and of the covariant derivative."""
        ch = self.ch
        ph = np.exp(1j * self.alpha)
        dn = np.linalg.norm(ch.derivative(psi_new) - ph * ch.derivative(psi_true)) / np.linalg.norm(ch.derivative(psi_true))
        dc = np.linalg.norm(ch.covariant_derivative(psi_new, self.theta_a) - ph * ch.covariant_derivative(psi_true, np.zeros_like(self.theta_a))) / np.linalg.norm(
            ch.covariant_derivative(psi_true, np.zeros_like(self.theta_a)))
        return float(dn), float(dc)


class PlaneSim:
    """Part 5: a flux tube in the plane of the interferometer, gauge transformations, loops and fringes."""

    def __init__(self, itf: Interferometer) -> None:
        P = CFG.plane
        self.itf = itf
        self.q = P.charge
        self.flux = P.flux_over_pi * math.pi / self.q
        self.field = gp.PlaneField(self.flux, P.tube_sigma, tuple(P.tube_centre), [tuple(t) for t in P.alpha_terms], self.q)
        self.source = tuple(P.source)
        I = CFG.interf
        self.slits = [(I.barrier_x, y0) for y0 in itf.slit_y]
        self.screen_pts = (I.screen_x, 0.0)
        self.loop1 = np.array([self.slits[1], self.screen_pts, self.slits[0]])           # lower slit -> screen centre -> upper slit -> back along the barrier
        c2, h2 = P.loop2_centre, P.loop2_half
        self.loop2 = np.array([(c2[0] - h2, c2[1] - h2), (c2[0] + h2, c2[1] - h2), (c2[0] + h2, c2[1] + h2), (c2[0] - h2, c2[1] + h2)])
        self.step = P.line_step
        self.y_pts = np.linspace(-I.screen_half, I.screen_half, P.fringe_samples)

    def a_func(self, flux_scale: float, gauge_scale: float):
        return lambda x, y: self.field.a_field(x, y, flux_scale, gauge_scale)

    def loop_phase(self, loop: np.ndarray, flux_scale: float, gauge_scale: float) -> float:
        """q * oint A_mu dz^mu (counter-clockwise), the argument of the Wilson loop up to the sign."""
        return self.q * gp.polygon_line_integral(self.a_func(flux_scale, gauge_scale), loop, self.step)

    def loop_flux(self, loop: np.ndarray, flux_scale: float) -> float:
        """q * int F_12 dx dy over the loop (a triangle or a rectangle)."""
        f = lambda x, y: self.field.b_field(x, y, flux_scale)   # noqa: E731
        if len(loop) == 3:
            v = loop if gp.polygon_area(loop) > 0 else loop[::-1]
            return self.q * gp.triangle_integral(f, v[0], v[1], v[2], CFG.plane.quadrature_n)
        x0, x1 = float(loop[:, 0].min()), float(loop[:, 0].max())
        y0, y1 = float(loop[:, 1].min()), float(loop[:, 1].max())
        return self.q * gp.rectangle_integral(f, x0, x1, y0, y1, CFG.plane.quadrature_n)

    def dial_angles(self, flux_scale: float, gauge_scale: float) -> list[float]:
        """The phase of the wave at each slit: -q int_{source -> slit} A + g alpha(source) (the emission convention)."""
        a = self.a_func(flux_scale, gauge_scale)
        a_src = float(self.field.alpha(*self.source)) * gauge_scale
        out = []
        for s in self.slits:
            out.append(-self.q * gp.polygon_line_integral(a, np.array([self.source, s]), self.step, closed=False) + a_src)
        return out

    def screen_pattern(self, flux_scale: float, gauge_scale: float) -> np.ndarray:
        """|psi|^2 on the screen: path j has the phase  dial_j - q int_{slit_j -> (x_s, y)} A."""
        a = self.a_func(flux_scale, gauge_scale)
        dials = self.dial_angles(flux_scale, gauge_scale)
        phases = []
        for j, s in enumerate(self.slits):
            w = gp.wilson_phase_to_screen(a, s, CFG.interf.screen_x, self.y_pts, self.q, self.step)
            phases.append(dials[j] + w)
        itf = self.itf
        psi = 0.0
        for y0, ph in zip(itf.slit_y, phases):
            psi = psi + np.exp(1j * ph) * gp.beam(itf.length, self.y_pts, y0, itf.k, itf.sigma)
        return np.abs(psi) ** 2 / itf.i_ref_max

    def peak_shift(self, pattern: np.ndarray, expected: float = 0.0) -> float:
        itf = self.itf
        pk = gp.peak_near(self.y_pts, pattern * itf.i_ref_max, near=itf.peak0 + expected * itf.period)
        return (pk - itf.peak0) / itf.period


_cache: dict = {}


def itf() -> Interferometer:
    if "itf" not in _cache:
        _cache["itf"] = Interferometer()
    return _cache["itf"]


def chain() -> ChainSim:
    if "chain" not in _cache:
        _cache["chain"] = ChainSim()
    return _cache["chain"]


def plane() -> PlaneSim:
    if "plane" not in _cache:
        _cache["plane"] = PlaneSim(itf())
    return _cache["plane"]


def part1_numbers() -> dict:
    """The complex numbers of the two friends (a plane wave psi = e^{i k z} at the two points) and the two differences of part 1."""
    P = CFG.part1
    ph = [-0.5 * P.phase_difference, 0.5 * P.phase_difference]                       # the phases of psi(x), psi(y) in the theorist's convention
    al = list(P.alpha)
    psi = [np.exp(1j * p) for p in ph]
    new = [np.exp(1j * (p + a)) for p, a in zip(ph, al)]
    return {"phase": ph, "alpha": al, "same": abs(psi[0] - psi[1]), "different": abs(new[0] - new[1])}


# ----------------------------------------------------------------------------------- the picture

class Canvas:
    """The figure of the film: all the axes are created once, every frame redraws the ones that are needed."""

    def __init__(self, size: tuple[int, int], lang: str) -> None:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.colors import LinearSegmentedColormap

        self.plt = plt
        self.lang = lang
        self.tx = TEXT[lang]
        S, L, V = CFG.style, CFG.layout, CFG.video
        self.W, self.H = size
        self.sc = self.H / V.reference_height
        self.BG = S.background
        self.TXT, self.DIM = tuple(S.text), tuple(S.dim)
        self.BLUE, self.WARM, self.GOLD, self.VIOLET, self.GREEN = tuple(S.blue), tuple(S.warm), tuple(S.gold), tuple(S.violet), tuple(S.green)
        self.bg_rgb = tuple(int(self.BG[i:i + 2], 16) / 255.0 for i in (1, 3, 5))
        self.fig = plt.figure(figsize=(self.W / V.dpi, self.H / V.dpi), dpi=V.dpi, facecolor=self.BG)
        mk = lambda rect: self.fig.add_axes(rect, facecolor="none")  # noqa: E731
        self.ax_scene = mk(L.scene_axes)
        self.ax_map, self.ax_prof, self.ax_wheel = mk(L.map_axes), mk(L.profile_axes), mk(L.wheel_axes)
        self.rows = [mk(r) for r in L.row_axes]
        self.ax_pa, self.ax_pf, self.ax_p5 = mk(L.plane_a_axes), mk(L.plane_f_axes), mk(L.plane_profile_axes)
        self.axes = [self.ax_scene, self.ax_map, self.ax_prof, self.ax_wheel, *self.rows, self.ax_pa, self.ax_pf, self.ax_p5]
        self.cmap_field = LinearSegmentedColormap.from_list("field", [(c[0], tuple(c[1:])) for c in CFG.part2.cmap], N=CFG.part2.cmap_samples)
        self.cmap_f = LinearSegmentedColormap.from_list("fmap", [(c[0], tuple(c[1:])) for c in CFG.part5.f_cmap], N=CFG.part2.cmap_samples)
        self.ring = self._ring_image()
        self._geom: dict = {}

    # ----------------------------------------------------------------- helpers
    def ftext(self, x: float, y: float, s: str, size: float, color, alpha: float = 1.0, **kw):
        return self.fig.text(x, y, s, color=(*color, alpha), fontsize=size * self.sc, **kw)

    def atext(self, ax, x: float, y: float, s: str, size: float, color, alpha: float = 1.0, **kw):
        kw.setdefault("transform", ax.transAxes)
        return ax.text(x, y, s, color=(*color, alpha), fontsize=size * self.sc, **kw)

    def reset(self, ax, visible: bool = True) -> None:
        ax.clear()
        ax.set_facecolor("none")
        ax.axis("off")
        ax.set_visible(visible)

    def aspect_half_height(self, rect, x_span: float) -> float:
        """Half of the y range of an axes (figure fractions rect) whose x range has the length x_span, for equal scales."""
        return 0.5 * x_span * (rect[3] * self.H) / (rect[2] * self.W)

    def text_boxes(self) -> list[tuple[str, object]]:
        """(string, bounding box in pixels) of every visible text of the current frame (for the layout tests)."""
        from matplotlib.colors import to_rgba
        self.fig.canvas.draw()
        rend = self.fig.canvas.get_renderer()
        items = list(self.fig.texts)
        for ax in self.axes:
            if ax.get_visible():
                items += list(ax.texts)
        return [(t.get_text(), t.get_window_extent(rend)) for t in items if t.get_text().strip() and to_rgba(t.get_color())[3] >= CFG.layout.text_min_alpha]

    def _ring_image(self) -> np.ndarray:
        """RGBA image of a phase dial: a hue ring (red at the right, counter-clockwise), the inside is dark."""
        D = CFG.dial
        n = D.image_size
        c = np.linspace(-1.0, 1.0, n)
        xx, yy = np.meshgrid(c, c)
        r = np.hypot(xx, yy)
        ang = np.arctan2(yy, xx)
        img = np.zeros((n, n, 4))
        img[..., :3] = gp.phase_rgb(ang, D.ring_value, D.ring_saturation)
        edge = 1.5 / n * 2
        outer = np.clip((1.0 - r) / edge, 0, 1)
        inner = np.clip((r - D.inner_radius) / edge, 0, 1)
        img[..., 3] = outer * inner
        # the dark disc inside
        disc = np.clip((D.inner_radius - r) / edge, 0, 1) * D.face_alpha
        img[..., :3] = img[..., :3] * (1 - disc[..., None]) + np.array(self.bg_rgb) * disc[..., None] * 0 + np.array(D.face_color)[None, None, :] * disc[..., None]
        img[..., 3] = np.maximum(img[..., 3], disc)
        return img[::-1]

    def dial(self, ax, cx: float, cy: float, r: float, angle: float, alpha: float = 1.0, ghost: float | None = None, z: int = 5, hand: float | None = None) -> None:
        """A phase dial: the hue ring and a hand that points at `angle` (counter-clockwise from the right); an optional dim ghost hand."""
        from matplotlib.patches import Circle
        D = CFG.dial
        sc = self.sc
        ax.imshow(self.ring, extent=(cx - r, cx + r, cy - r, cy + r), zorder=z, alpha=alpha, interpolation="bilinear", aspect="auto")
        ax.add_patch(Circle((cx, cy), r * D.rim_radius, facecolor="none", edgecolor=(*self.DIM, D.rim_alpha * alpha), linewidth=D.rim_width * sc, zorder=z + 1))
        hl = r * (D.hand_length if hand is None else hand)
        if ghost is not None:
            ax.plot([cx, cx + hl * math.cos(ghost)], [cy, cy + hl * math.sin(ghost)], color=(*self.DIM, D.ghost_alpha * alpha), lw=D.hand_width * sc, ls=(0, tuple(D.ghost_dash)),
                    solid_capstyle="round", zorder=z + 1)
        ax.plot([cx, cx + hl * math.cos(angle)], [cy, cy + hl * math.sin(angle)], color=(*self.TXT, alpha), lw=D.hand_width * sc, solid_capstyle="round", zorder=z + 2)
        ax.scatter([cx + hl * math.cos(angle)], [cy + hl * math.sin(angle)], s=(D.tip_size * sc) ** 2, color=[(*gp.phase_rgb(angle, 1.0, 1.0), alpha)], edgecolors=[(*self.TXT, alpha)],
                   linewidths=D.tip_edge * sc, zorder=z + 3)
        ax.scatter([cx], [cy], s=(D.hub_size * sc) ** 2, color=[(*self.TXT, alpha)], zorder=z + 3)

    # ----------------------------------------------------------------- one frame
    def draw(self, t_film: float, total: float) -> None:
        S, L, F = CFG.style, CFG.layout, CFG.fonts
        tx, DIM, GOLD = self.tx, self.DIM, self.GOLD
        k = total / TOTAL
        t, card, cprog = timeline(t_film / k)
        self.fig.texts.clear()
        for ax in self.axes:
            self.reset(ax, visible=False)
        if card is not None:
            a_c = min(SMOOTH(cprog, *CFG.timeline.card_fade_in), 1.0 - SMOOTH(cprog, *CFG.timeline.card_fade_out))
            if card == 0:
                self.ftext(0.5, L.card_title_y, tx["h0"], F.card_title, S.card_title_color, a_c, ha="center", va="center")
                self.ftext(0.5, L.card_subtitle_y, tx["h0s"], F.card_subtitle, DIM, a_c, ha="center", va="center")
            else:
                self.ftext(0.5, L.card_number_y, f"{card}", F.card_number, GOLD, S.card_number_alpha * a_c, ha="center", va="center")
                self.ftext(0.5, L.card_chapter_y, tx[CARD_KEYS[card]], F.card_chapter, S.card_title_color, a_c, ha="center", va="center")
                self.ftext(0.5, L.card_chapter_sub_y, tx[CARD_KEYS[card] + "s"], F.card_chapter_sub, DIM, a_c, ha="center", va="center")
            return
        self.ftext(*L.title_pos, tx["title"], F.title, S.title_color[:3], S.title_color[3])
        ch = chapter_of(t)
        self.ftext(*L.chapter_label_pos, f"{ch}/{len(CARD_AT) - 1}   " + tx[CARD_KEYS[ch]], F.chapter_label, DIM, S.chapter_label_alpha, ha="right")
        p = part_of(t)
        tl = t - PB[p - 1]
        {1: self.part1, 2: self.part2, 3: self.part3, 4: self.part4, 5: self.part5, 6: self.part6}[p](tl)

    def formula(self, key: str, row: int, alpha: float, pos=None, **vals) -> None:
        L, F, S = CFG.layout, CFG.fonts, CFG.style
        if alpha > L.formula_min_alpha:
            x, y = pos if pos is not None else (L.formula_pos[0], L.formula_pos[1] - L.formula_row * row)
            self.ftext(x, y, fill(self.tx[key], self.lang, **vals), F.formula, self.TXT, S.formula_alpha * alpha)

    def caption(self, key: str, **vals) -> None:
        self.ftext(*CFG.layout.caption_pos, fill(self.tx[key], self.lang, **vals), CFG.fonts.caption, self.TXT, CFG.style.caption_alpha, va="bottom", linespacing=CFG.layout.caption_spacing)

    def pick(self, t: float, times: list[float], keys: list[str]) -> str:
        """The key whose time interval contains t: keys[i] from times[i-1] (times has one entry less than keys)."""
        i = sum(1 for tt in times if t >= tt)
        return keys[i]

    # ================================================================= part 1: two friends and a phase
    def part1(self, t: float) -> None:
        P, L, F, tx, lang, sc = CFG.part1, CFG.layout, CFG.fonts, self.tx, self.lang, self.sc
        from matplotlib.patches import Circle, Polygon
        ax = self.ax_scene
        self.reset(ax)
        xw = P.scene_width
        yh = self.aspect_half_height(L.scene_axes, xw)
        ax.set_xlim(0, xw)
        ax.set_ylim(0, 2 * yh)
        TXT, DIM, GOLD, BLUE, WARM = self.TXT, self.DIM, self.GOLD, self.BLUE, self.WARM
        nums = part1_numbers()
        rng = np.random.default_rng(P.stars_seed)
        sx, sy_ = rng.uniform(0, xw, P.stars), rng.uniform(0, 2 * yh, P.stars)
        ax.scatter(sx, sy_, s=(rng.uniform(P.star_size[0], P.star_size[1], P.stars) * sc) ** 2, color=[(*self.TXT, a) for a in rng.uniform(P.star_alpha[0], P.star_alpha[1], P.stars)], linewidths=0, zorder=0)
        # the Earth (the theorist)
        a_e = SMOOTH(t, *P.earth_in)
        ex, ey = P.earth_centre
        er = P.earth_radius
        for g_r, g_a in P.earth_glow:
            ax.add_patch(Circle((ex, ey), er * g_r, facecolor=(*BLUE, g_a * a_e), edgecolor="none", zorder=1))
        ax.add_patch(Circle((ex, ey), er, facecolor=(*P.ocean_color, a_e), edgecolor=(*BLUE, 0.9 * a_e), linewidth=1.2 * sc, zorder=2))
        for blob in P.land:
            pts = np.array(blob) * er + np.array([ex, ey])
            poly = Polygon(pts, closed=True, facecolor=(*P.land_color, a_e), edgecolor="none", zorder=3)
            ax.add_patch(poly)
            poly.set_clip_path(Circle((ex, ey), er, transform=ax.transData))
        self.atext(ax, ex / xw + P.earth_label_offset[0], (ey + P.earth_label_offset[1]) / (2 * yh), tx["theorist"], F.label, TXT, a_e, ha="left", va="center")
        # the two experimenters: rockets along arcs, then the dials
        targets = [np.array(P.target_x), np.array(P.target_y)]
        names = ["x", "y"]
        a_fly = SMOOTH(t, *P.fly)
        a_dial = SMOOTH(t, *P.dials_in)
        a_hand = SMOOTH(t, *P.hands_in)
        a_turn = SMOOTH(t, *P.turn)
        a_sig = SMOOTH(t, *P.signals)
        for j, tg in enumerate(targets):
            side = -1.0 if j == 0 else 1.0
            start = np.array([ex + side * P.launch_spread, ey + er * P.launch_offset])
            park = tg + np.array([side * P.rocket_offset[0], P.rocket_offset[1]])
            ctrl = np.array([start[0] + side * P.arc_ctrl[0], start[1] + P.arc_ctrl[1]])
            u = a_fly
            pos = (1 - u) ** 2 * start + 2 * u * (1 - u) * ctrl + u * u * park
            tan = 2 * (1 - u) * (ctrl - start) + 2 * u * (park - ctrl)
            ang = math.atan2(tan[1], tan[0])
            # the path (dotted) behind the rocket
            uu = np.linspace(0, u, 60)[:, None]
            path = (1 - uu) ** 2 * start + 2 * uu * (1 - uu) * ctrl + uu ** 2 * park
            if u > 0.01:
                ax.plot(path[:, 0], path[:, 1], color=(*DIM, P.trail_alpha), lw=1.2 * sc, ls=(0, tuple(P.trail_dash)), zorder=2)
            if a_fly > 0.005:
                self.rocket(ax, pos, ang, P.rocket_scale, 1.0 - 0.0 * u)
            # the signal back to the Earth (one-way communication)
            if a_sig > 0.01:
                s0 = tg
                s1 = np.array([ex, ey + er * P.signal_end])
                ax.plot([s0[0], s1[0]], [s0[1], s1[1]], color=(*DIM, P.signal_line_alpha * min(a_sig * 4, 1.0)), lw=1.0 * sc, ls=(0, tuple(P.trail_dash)), zorder=1)
                for m in range(P.signal_dots):                                        # pulses that start one after another and arrive together with the end of the signal
                    w = (a_sig - m * P.signal_gap) / (1.0 - (P.signal_dots - 1) * P.signal_gap)
                    if 0.0 < w < 1.0:
                        pp = s0 + (s1 - s0) * w
                        ax.scatter([pp[0]], [pp[1]], s=(P.signal_size * sc) ** 2, color=[(*GOLD, 0.9)], zorder=6)
            # the dial of the experimenter
            if a_dial > 0.01:
                r = P.dial_radius
                ang0 = nums["phase"][j]
                ang1 = ang0 + nums["alpha"][j] * a_turn
                ghost = ang0 if a_turn > 0.02 else None
                self.dial(ax, tg[0], tg[1], r, ang1 if a_hand > 0.01 else 0.0, alpha=a_dial, ghost=ghost, hand=P.hand_length if a_hand > 0.01 else 0.0)
                if a_turn > 0.02:                                                     # the arc of the rotation alpha
                    arc = np.linspace(ang0, ang1, 40)
                    rr = r * P.arc_radius
                    ax.plot(tg[0] + rr * np.cos(arc), tg[1] + rr * np.sin(arc), color=(*GOLD, 0.95 * a_dial), lw=1.8 * sc, zorder=8)
                    ax.annotate("", xy=(tg[0] + rr * math.cos(ang1), tg[1] + rr * math.sin(ang1)),
                                xytext=(tg[0] + rr * math.cos(ang1 - 0.12 * np.sign(ang1 - ang0)), tg[1] + rr * math.sin(ang1 - 0.12 * np.sign(ang1 - ang0))),
                                arrowprops=dict(arrowstyle="-|>", color=(*GOLD, 0.95 * a_dial), lw=1.8 * sc, shrinkA=0, shrinkB=0), zorder=8)
                self.atext(ax, (tg[0]) / xw, (tg[1] + P.label_offset[0]) / (2 * yh), tx["experimenter_" + names[j]], F.label, TXT, a_dial, ha="center", va="bottom")
                key = "psi_" + names[j] + ("_new" if a_turn > 0.5 else "")
                self.atext(ax, (tg[0]) / xw, (tg[1] - P.label_offset[1]) / (2 * yh), tx[key], F.label, GOLD if a_turn > 0.5 else TXT, a_hand * a_dial, ha="center", va="top")
        # the report: two differences
        a1 = SMOOTH(t, *P.rows_in[0])
        a2 = SMOOTH(t, *P.rows_in[1])
        if a1 > 0.01:
            self.ftext(P.row_label_x, P.row_y[0], tx["row_same"], F.label, DIM, a1, ha="right", va="center")
            self.ftext(P.row_x, P.row_y[0], fill(tx["row_same_f"], lang, v=(nums["same"], 2)), F.formula, TXT, a1, ha="left", va="center")
        if a2 > 0.01:
            self.ftext(P.row_label_x, P.row_y[1], tx["row_diff"], F.label, DIM, a2, ha="right", va="center")
            self.ftext(P.row_x, P.row_y[1], fill(tx["row_diff_f"], lang, v=(nums["different"], 2)), F.formula, GOLD, a2, ha="left", va="center")
        a3 = SMOOTH(t, *P.question_in)
        if a3 > 0.01:
            self.ftext(P.question_pos[0], P.question_pos[1], tx["question"], F.question, TXT, a3, ha="center", va="center")
        self.ftext(*P.tag_pos, tx["illustration"], F.tag, DIM, P.tag_alpha, ha="right", va="top")
        self.caption(self.pick(t, P.caption_times, ["c1a", "c1b", "c1c"]))

    def rocket(self, ax, pos, ang: float, scale: float, alpha: float) -> None:
        """A small rocket (an illustration) from the polygons of the configuration, rotated to the direction ang."""
        from matplotlib.patches import Polygon
        R = CFG.part1.rocket
        ca, sa = math.cos(ang), math.sin(ang)
        rot = np.array([[ca, -sa], [sa, ca]])
        for poly in R.polygons:
            pts = (np.array(poly["pts"]) * scale) @ rot.T + np.asarray(pos)
            col = tuple(CFG.style[poly["color"]]) if isinstance(poly["color"], str) else tuple(poly["color"])
            ax.add_patch(Polygon(pts, closed=True, facecolor=(*col, poly["alpha"] * alpha), edgecolor=(*self.TXT, R.edge_alpha * alpha), linewidth=R.edge_width * self.sc, zorder=7, joinstyle="round"))
        c = (np.array(R.window) * scale) @ rot.T + np.asarray(pos)
        from matplotlib.patches import Circle
        ax.add_patch(Circle(c, R.window_radius * scale, facecolor=(*self.BLUE, alpha), edgecolor=(*self.TXT, R.edge_alpha * alpha), linewidth=R.edge_width * self.sc, zorder=8))

    # ================================================================= part 2: two paths
    def part2(self, t: float) -> None:
        P, L, F, tx, lang, sc = CFG.part2, CFG.layout, CFG.fonts, self.tx, self.lang, self.sc
        I = itf()
        TXT, DIM, GOLD, BLUE, WARM = self.TXT, self.DIM, self.GOLD, self.BLUE, self.WARM
        # the phases of the two paths
        th_common = TWO_PI * SMOOTH(t, *P.common_turn)
        th_rel = TWO_PI * SMOOTH(t, *P.relative_turn)
        th1 = th_common + th_rel
        th2 = th_common
        delta = th1 - th2
        a_in = SMOOTH(t, *P.appear)
        self.formula("f2a", 0, SMOOTH(t, *P.formula_in[0]))
        self.formula("f2b", 1, SMOOTH(t, *P.formula_in[1]))
        self.caption(self.pick(t, P.caption_times, ["c2a", "c2b", "c2c", "c2d"]))
        # the field map
        ax = self.ax_map
        self.reset(ax)
        xmin, xmax = P.view_x
        yh = self.aspect_half_height(L.map_axes, xmax - xmin)
        ax.set_xlim(xmin, xmax)
        ax.set_ylim(-yh, yh)
        self.interference_map(ax, I, [th1, th2], yh, a_in)
        self.apparatus(ax, I, yh, a_in)
        a_ps = SMOOTH(t, *P.shifter_in) * (1.0 - SMOOTH(t, *P.shifter_out))
        if a_ps > 0.01:                                                       # the phase shifter on path 1 (an illustration)
            sx, sy_ = tuple(CFG.plane.source)
            frac = P.shifter_frac
            cx_, cy_ = sx + frac * (I.barrier_x - sx), sy_ + frac * (I.slit_y[0] - sy_)
            ang = math.atan2(I.slit_y[0] - sy_, I.barrier_x - sx)
            hw, hh = P.shifter_size
            c, s_ = math.cos(ang), math.sin(ang)
            from matplotlib.patches import Polygon
            pts = np.array([(-hw, -hh), (hw, -hh), (hw, hh), (-hw, hh)]) @ np.array([[c, s_], [-s_, c]]) + np.array([cx_, cy_])
            ax.add_patch(Polygon(pts, closed=True, facecolor=(*GOLD, P.shifter_fill * a_ps), edgecolor=(*GOLD, a_ps), linewidth=1.6 * sc, zorder=7))
            ax.text(cx_ - P.shifter_label_dx, cy_ + P.shifter_label_dy, tx["shifter"], color=(*GOLD, a_ps), fontsize=F.dial_label * sc, ha="center", va="bottom", zorder=9)
        for j, (th, y0) in enumerate(zip((th1, th2), I.slit_y)):
            self.dial(ax, P.dial_x, np.sign(y0) * P.dial_y, P.dial_radius, th, alpha=a_in)
            ax.plot([P.dial_x + P.dial_radius * 0.9, I.barrier_x], [np.sign(y0) * P.dial_y - np.sign(y0) * P.dial_radius * 0.3, y0], color=(*DIM, P.link_alpha * a_in), lw=1.0 * sc,
                    ls=(0, tuple(P.link_dash)), zorder=3)
            self.atext(ax, (P.dial_x - xmin) / (xmax - xmin), 0.5 + (np.sign(y0) * (P.dial_y + P.dial_radius + P.dial_label_gap)) / (2 * yh), tx[f"path{j + 1}"], F.dial_label, DIM, a_in, ha="center", va="center")
        # the profile on the screen
        pa = self.ax_prof
        self.reset(pa)
        pa.set_xlim(0, P.profile_xmax)
        pa.set_ylim(-yh, yh)
        pattern = I.screen_intensity([th1, th2])
        ref = I.screen_intensity([0.0, 0.0])
        y = I.y_screen
        pa.fill_betweenx(y, 0, pattern, color=(*BLUE, P.profile_fill_alpha * a_in), linewidth=0)
        pa.plot(ref, y, color=(*DIM, P.ghost_alpha * a_in), lw=P.ghost_width * sc, ls=(0, tuple(P.ghost_dash)))
        pa.plot(pattern, y, color=(*TXT, a_in), lw=P.profile_width * sc)
        self.atext(pa, 0.5, P.profile_title_y, tx["screen"], F.panel_title, DIM, a_in, ha="center", va="bottom")
        # the colour wheel and the readouts
        self.wheel(P.wheel_pos, a_in)
        shift = I.central_shift(pattern, expected=delta / TWO_PI)
        rows = [(tx["r_theta1"], f"${num(math.degrees(th1) % 360, 0, lang)}^\\circ$"),
                (tx["r_theta2"], f"${num(math.degrees(th2) % 360, 0, lang)}^\\circ$"),
                (tx["r_delta"], f"${num(math.degrees(delta), 0, lang)}^\\circ$"),
                (tx["r_change"], f"${num(float(np.max(np.abs(pattern - ref))), 3, lang)}$"),
                (tx["r_shift"], f"${num(shift, 2, lang)}$")]
        a_r = SMOOTH(t, *P.readout_in)
        for i, (lab, val) in enumerate(rows):
            yy = P.readout_pos[1] - i * P.readout_row
            self.ftext(P.readout_pos[0], yy + P.readout_label_dy, lab, F.readout_label, DIM, a_r, ha="left", va="bottom")
            self.ftext(P.readout_pos[0], yy, val, F.readout, TXT, a_r, ha="left", va="bottom")

    def wheel(self, pos, alpha: float) -> None:
        """A small phase wheel (the legend of the dial colours)."""
        ax = self.ax_wheel
        self.reset(ax)
        ax.set_xlim(-1, 1)
        ax.set_ylim(-1, 1)
        ax.imshow(self.ring, extent=(-1, 1, -1, 1), aspect="auto", alpha=alpha, interpolation="bilinear")
        ax.set_xlim(-1, 1)
        ax.set_ylim(-1, 1)
        W = CFG.part2
        for lab, ang in (("0", 0.0), ("\\pi/2", math.pi / 2), ("\\pi", math.pi), ("3\\pi/2", 1.5 * math.pi)):
            ax.text(W.wheel_tick * math.cos(ang), W.wheel_tick * math.sin(ang), f"${lab}$", color=(*self.DIM, alpha), fontsize=CFG.fonts.tick * self.sc, ha="center", va="center")
        self.ftext(pos[0], pos[1], self.tx["wheel"], CFG.fonts.readout_label, self.DIM, alpha, ha="center", va="bottom")

    def interference_map(self, ax, I: Interferometer, phases, yh: float, alpha: float) -> None:
        P = CFG.part2
        W, H = self.W * CFG.layout.map_axes[2], self.H * CFG.layout.map_axes[3]
        key = ("map", round(yh, 6))
        if key not in self._geom:
            nx = int(round(W * P.map_scale))
            ny = int(round(H * P.map_scale))
            xs = np.linspace(I.barrier_x, I.screen_x, nx)
            ys = np.linspace(-yh, yh, ny)
            self._geom[key] = np.meshgrid(xs, ys)
        X, Y = self._geom[key]
        psi = I.field_map(X, Y, phases)
        v = np.clip(np.abs(psi) ** 2, 0, 1) ** P.gamma * (1 - P.ripple + P.ripple * np.cos(np.angle(psi)))
        ax.imshow(v, extent=(I.barrier_x, I.screen_x, -yh, yh), origin="lower", cmap=self.cmap_field, vmin=0.0, vmax=1.0, interpolation="bilinear", aspect="auto", zorder=1, alpha=alpha)
        ax.set_xlim(*P.view_x)
        ax.set_ylim(-yh, yh)

    def apparatus(self, ax, I: Interferometer, yh: float, alpha: float, source: bool = True) -> None:
        """The barrier with two slits, the screen and the source (drawn in the data coordinates of the plane)."""
        P = CFG.part2
        sc = self.sc
        gap = P.slit_gap
        bx = I.barrier_x
        edges = [(-yh, I.slit_y[1] - gap), (I.slit_y[1] + gap, I.slit_y[0] - gap), (I.slit_y[0] + gap, yh)]
        for y0, y1 in edges:
            ax.plot([bx, bx], [y0, y1], color=(*self.TXT, P.barrier_alpha * alpha), lw=P.barrier_width * sc, solid_capstyle="butt", zorder=6)
        ax.plot([I.screen_x, I.screen_x], [-yh, yh], color=(*self.GOLD, P.screen_alpha * alpha), lw=P.screen_width * sc, zorder=6)
        if source:
            sx, sy = tuple(CFG.plane.source)
            ax.scatter([sx], [sy], s=(P.source_size * sc) ** 2, color=[(*self.WARM, alpha)], zorder=6)
            for s in I.slit_y:
                ax.plot([sx, bx], [sy, s], color=(*self.DIM, P.link_alpha * alpha), lw=1.0 * sc, ls=(0, tuple(P.link_dash)), zorder=3)

    # ================================================================= parts 3 and 4: the particle on a line
    def part3(self, t: float) -> None:
        self.chain_scene(t, CFG.part3, False)

    def part4(self, t: float) -> None:
        self.chain_scene(t, CFG.part4, True)

    def ribbon(self, ax, x: np.ndarray, psi: np.ndarray, ymax: float, strip_phase: np.ndarray, alpha: float, ghost: np.ndarray | None, strip_label: str, lab_alpha: float) -> None:
        """|psi|^2 as a mountain coloured by the phase of psi, with the strip of the convention (the colour of e^{i alpha}) under it."""
        from matplotlib.collections import LineCollection
        R = CFG.rows
        sc = self.sc
        rho = np.abs(psi) ** 2
        n = len(x)
        ny = R.image_rows
        yy = np.linspace(0.0, ymax, ny)[:, None]
        col = gp.phase_rgb(np.angle(psi), 1.0, R.saturation)
        edge = ymax / ny * R.edge_soft
        cover = np.clip((rho[None, :] - yy) / edge + 0.5, 0.0, 1.0)
        shade = R.fill_bottom + (R.fill_top - R.fill_bottom) * (yy / ymax)
        img = np.zeros((ny, n, 4))
        img[..., :3] = col[None, :, :]
        img[..., 3] = cover * shade * alpha
        sy = R.strip_height * ymax
        ax.imshow(img, extent=(x[0], x[-1], 0.0, ymax), origin="lower", aspect="auto", interpolation="bilinear", zorder=2)
        strip = gp.phase_rgb(strip_phase, 1.0, R.strip_saturation)[None, :, :]
        ax.imshow(strip, extent=(x[0], x[-1], -sy * (1 + R.strip_gap), -sy * R.strip_gap), origin="lower", aspect="auto", interpolation="nearest", zorder=2, alpha=alpha)
        pts = np.column_stack([x, rho])
        segs = np.stack([pts[:-1], pts[1:]], axis=1)
        lc = LineCollection(segs, colors=[(*c, alpha) for c in col[:-1]], linewidths=R.line_width * sc, zorder=4, capstyle="round")
        ax.add_collection(lc)
        if ghost is not None:
            ax.plot(x, ghost, color=(*self.TXT, R.ghost_alpha * alpha), lw=R.ghost_width * sc, ls=(0, tuple(R.ghost_dash)), zorder=5)
        ybot = -sy * (1 + R.strip_gap) - R.label_room * ymax
        ax.set_xlim(-CFG.chain.view_half, CFG.chain.view_half)
        ax.set_ylim(ybot, ymax)
        ax.text(-CFG.chain.view_half + R.strip_label_dx, ybot + R.strip_label_dy * ymax, strip_label, color=(*self.DIM, lab_alpha), fontsize=CFG.fonts.strip_label * sc, ha="left", va="bottom")

    def chain_scene(self, t: float, P, covariant: bool) -> None:
        L, F, tx, lang, sc = CFG.layout, CFG.fonts, self.tx, self.lang, self.sc
        R = CFG.rows
        sim = chain()
        C = CFG.chain
        TXT, DIM, GOLD, VIOLET, BLUE = self.TXT, self.DIM, self.GOLD, self.VIOLET, self.BLUE
        a_in = SMOOTH(t, *P.appear)
        tau = C.t_end * min(max((t - P.run[0]) / (P.run[1] - P.run[0]), 0.0), 1.0)
        psi_true, psi_naive, psi_cov = sim.states(tau)
        psi_b = psi_cov if covariant else psi_naive
        x = sim.ch.x
        rho_true = np.abs(psi_true) ** 2
        ymax = R.ymax
        self.formula("f4a" if covariant else "f3a", 0, SMOOTH(t, *P.formula_in[0]))
        self.formula("f4b" if covariant else "f3b", 1, SMOOTH(t, *P.formula_in[1]))
        self.caption(self.pick(t, P.caption_times, ["c4a", "c4b", "c4c"] if covariant else ["c3a", "c3b", "c3c"]))
        ax1, ax2, ax3 = self.rows
        for ax in self.rows:
            self.reset(ax)
        self.ribbon(ax1, x, psi_true, ymax, np.zeros_like(x), a_in, None, tx["strip_theorist"], a_in)
        self.atext(ax1, R.label_pos[0], R.label_pos[1], tx["row_theorist"], F.row_label, TXT, a_in, ha="left", va="top")
        self.ribbon(ax2, x, psi_b, ymax, sim.alpha, a_in, rho_true, tx["strip_experimenter"], a_in)
        self.atext(ax2, R.label_pos[0], R.label_pos[1], tx["row_exp_cov" if covariant else "row_exp_free"], F.row_label, GOLD if not covariant else self.GREEN, a_in, ha="left", va="top")
        # the third row: the gradient of the convention, then the compensating field
        a3 = SMOOTH(t, *P.row3_in)
        morph = SMOOTH(t, *P.morph) if covariant else 0.0
        da = sim.dalpha(x)
        curve = da * (1 - morph) + (-da / sim.q) * morph
        ax3.set_xlim(-CFG.chain.view_half, CFG.chain.view_half)
        ax3.set_ylim(*R.row3_ylim)
        col3 = tuple(np.array(GOLD) * (1 - morph) + np.array(self.GREEN) * morph)
        ax3.fill_between(x, 0, curve, color=(*col3, R.row3_fill_alpha * a3), linewidth=0, zorder=2)
        ax3.plot(x, curve, color=(*col3, a3), lw=R.row3_width * sc, zorder=3)
        ax3.plot([-CFG.chain.view_half, CFG.chain.view_half], [0, 0], color=(*DIM, R.baseline_alpha * a3), lw=1.0 * sc, zorder=1)
        xq = np.arange(R.arrow_start, CFG.chain.view_half, R.arrow_step)
        aq = np.where(morph > 0.5, -sim.dalpha(xq) / sim.q, sim.dalpha(xq))
        for xv, av in zip(xq, aq):
            half = 0.5 * R.arrow_scale * av
            if abs(half) < R.arrow_min:
                ax3.scatter([xv], [R.arrow_y], s=(R.dot_size * sc) ** 2, color=[(*col3, R.arrow_alpha * a3)], zorder=4)
                continue
            ax3.annotate("", xy=(xv + half, R.arrow_y), xytext=(xv - half, R.arrow_y), arrowprops=dict(arrowstyle="-|>", color=(*col3, R.arrow_alpha * a3), lw=R.arrow_width * sc,
                                                                                                         shrinkA=0, shrinkB=0, mutation_scale=R.arrow_head * sc), zorder=4)
        self.atext(ax3, R.label_pos[0], R.label_pos3, tx["row3_field"] if morph > 0.5 else tx["row3_grad"], F.row_label, col3, a3, ha="left", va="top")
        # the readouts on the right
        ch = sim.ch
        a_r = SMOOTH(t, *P.readout_in)
        xr = L.readout_x
        if not covariant:
            vt, vn = sim.packet_speed("true"), sim.packet_speed("naive")
            x0 = C.packet_x0
            rows = [("", None, None),
                    (tx["r3_head_x"], None, None),
                    (tx["r3_theorist"], ch.centroid(psi_true), x0 + vt * tau),
                    (tx["r3_experimenter"], ch.centroid(psi_naive), x0 + vn * tau),
                    (tx["r3_head_v"], None, None),
                    (tx["r3_theorist"], (ch.centroid(psi_true) - x0) / tau if tau > P.speed_min_t else None, vt),
                    (tx["r3_experimenter"], (ch.centroid(psi_naive) - x0) / tau if tau > P.speed_min_t else None, vn)]
            self.table(rows, P, a_r, 2)
        else:
            err = float(np.max(np.abs(np.abs(psi_cov) ** 2 - rho_true)))
            dn, dc = sim.derivative_defects(psi_true, psi_cov)
            blocks = [(tx["r4_head_rho"], f"${sci(max(err, P.floor), lang)}$"), (tx["r4_head_dn"], f"${num(dn, 2, lang)}$"), (tx["r4_head_dc"], f"${sci(max(dc, P.floor), lang)}$")]
            for i, (lab, val) in enumerate(blocks):
                yy = P.block_y - i * P.block_dy
                self.ftext(xr, yy, lab, F.readout_label, DIM, a_r, ha="left", va="bottom")
                self.ftext(xr, yy - P.block_gap, val, F.readout, (TXT, self.WARM, self.GREEN)[i], a_r, ha="left", va="top")
        # the time
        self.ftext(L.time_pos[0], L.time_pos[1], fill(tx["time"], lang, t=(tau, 1)), F.readout_label, DIM, a_in, ha="right", va="bottom")

    def table(self, rows, P, alpha: float, nd: int) -> None:
        L, F, lang = CFG.layout, CFG.fonts, self.lang
        xr = L.readout_x
        y = P.table_y
        for lab, v1, v2 in rows:
            if v1 is None and v2 is None:
                if lab == "":                                                      # the column heads
                    self.ftext(L.table_cols[0], y, self.tx["col_sim"], F.table_head, self.DIM, alpha, ha="right", va="bottom")
                    self.ftext(L.table_cols[1], y, self.tx["col_formula"], F.table_head, self.DIM, alpha, ha="right", va="bottom")
                else:
                    self.ftext(xr, y, lab, F.readout_label, self.DIM, alpha, ha="left", va="bottom")
            else:
                self.ftext(xr, y, lab, F.readout_label, self.TXT, alpha, ha="left", va="bottom")
                if v1 is not None:
                    self.ftext(L.table_cols[0], y, f"${num(v1, nd, lang)}$", F.readout, self.TXT, alpha, ha="right", va="bottom")
                self.ftext(L.table_cols[1], y, f"${num(v2, nd, lang)}$", F.readout, self.GOLD, alpha, ha="right", va="bottom")
            y -= P.table_row

    # ================================================================= part 5: the field strength and the loop
    def part5(self, t: float) -> None:
        P, L, F, tx, lang, sc = CFG.part5, CFG.layout, CFG.fonts, self.tx, self.lang, self.sc
        pl = plane()
        I = itf()
        TXT, DIM, GOLD, BLUE, WARM, VIOLET = self.TXT, self.DIM, self.GOLD, self.BLUE, self.WARM, self.VIOLET
        g1 = SMOOTH(t, *P.gauge1)                         # pure gauge, first convention
        fl = SMOOTH(t, *P.tube_on)                         # the flux tube
        g2 = SMOOTH(t, *P.gauge2)                          # the second convention (a different alpha: the same function, changed strength)
        gauge = g1 * (1 - SMOOTH(t, *P.gauge1_off)) + g2 * P.gauge2_gain
        a_in = SMOOTH(t, *P.appear)
        self.formula("f5a", 0, SMOOTH(t, *P.formula_in[0]), pos=P.formula_pos[0])
        self.formula("f5b", 1, SMOOTH(t, *P.formula_in[1]), pos=P.formula_pos[1])
        self.caption(self.pick(t, P.caption_times, ["c5a", "c5b", "c5c"]))
        xmin, xmax = CFG.part2.view_x
        yha = self.aspect_half_height(L.plane_a_axes, xmax - xmin)
        yhf = self.aspect_half_height(L.plane_f_axes, xmax - xmin)
        # ---- the panel of A
        ax = self.ax_pa
        self.reset(ax)
        ax.set_xlim(xmin, xmax)
        ax.set_ylim(-yha, yha)
        self.apparatus(ax, I, yha, a_in)
        gx = np.arange(P.arrow_x[0], P.arrow_x[1] + 1e-9, P.arrow_step)
        gy = np.arange(-P.arrow_y_half, P.arrow_y_half + 1e-9, P.arrow_step)
        GX, GY = np.meshgrid(gx, gy)
        ux, uy = pl.field.a_field(GX, GY, fl, gauge)
        ux_s, uy_s = ux * P.arrow_scale, uy * P.arrow_scale
        ln = np.hypot(ux_s, uy_s)
        fac = np.where(ln > P.arrow_max, P.arrow_max / np.maximum(ln, 1e-12), 1.0)
        ux_s, uy_s = ux_s * fac, uy_s * fac
        keep = (np.abs(GX - I.barrier_x) > P.arrow_barrier_gap) & (np.hypot(ux_s, uy_s) > P.arrow_min)
        ax.quiver(GX[keep], GY[keep], ux_s[keep], uy_s[keep], angles="xy", scale_units="xy", scale=1.0, color=(*BLUE, P.arrow_alpha * a_in), width=P.arrow_width * sc / self.W,
                  headwidth=P.arrow_head[0], headlength=P.arrow_head[1], headaxislength=P.arrow_head[2], pivot="tail", zorder=2)
        # the tube
        from matplotlib.patches import Circle, Polygon
        if fl > 0.01:
            ax.add_patch(Circle(tuple(CFG.plane.tube_centre), CFG.plane.tube_sigma * P.tube_mark, facecolor="none", edgecolor=(*WARM, 0.9 * fl), linewidth=1.6 * sc, zorder=5))
        # the loops
        for k_, (loop, colr) in enumerate(((pl.loop1, GOLD), (pl.loop2, self.GREEN))):
            ax.add_patch(Polygon(loop, closed=True, facecolor=(*colr, P.loop_fill_alpha * a_in), edgecolor=(*colr, P.loop_alpha * a_in), linewidth=P.loop_width * sc, zorder=4, joinstyle="round"))
        # the dials at the slits
        dials = pl.dial_angles(fl, gauge)
        for j, (th, y0) in enumerate(zip(dials, I.slit_y)):
            self.dial(ax, CFG.part2.dial_x, np.sign(y0) * CFG.part2.dial_y, CFG.part2.dial_radius, th, alpha=a_in)
        for k_, (lp, colr) in enumerate(zip(P.loop_label_pos, (GOLD, self.GREEN))):
            ax.text(lp[0], lp[1], tx[f"loop{k_ + 1}"], color=(*colr, a_in), fontsize=F.label * sc, ha="center", va="center", zorder=9)
        self.atext(ax, P.panel_title_pos[0], P.panel_title_pos[1], tx["panel_a"], F.panel_title, TXT, a_in, ha="left", va="bottom")
        # ---- the panel of F
        af = self.ax_pf
        self.reset(af)
        af.set_xlim(xmin, xmax)
        af.set_ylim(-yhf, yhf)
        if ("fmap", round(yhf, 6)) not in self._geom:
            nx = int(round(self.W * L.plane_f_axes[2] * P.f_scale))
            ny = int(round(self.H * L.plane_f_axes[3] * P.f_scale))
            self._geom[("fmap", round(yhf, 6))] = np.meshgrid(np.linspace(xmin, xmax, nx), np.linspace(-yhf, yhf, ny))
        FX, FY = self._geom[("fmap", round(yhf, 6))]
        fmap = pl.field.b_field(FX, FY, fl) / pl.field.b_field(np.array(CFG.plane.tube_centre[0]), np.array(CFG.plane.tube_centre[1]), 1.0)
        af.imshow(fmap, extent=(xmin, xmax, -yhf, yhf), origin="lower", cmap=self.cmap_f, vmin=0.0, vmax=1.0, interpolation="bilinear", aspect="auto", zorder=1, alpha=a_in)
        for loop, colr in ((pl.loop1, GOLD), (pl.loop2, self.GREEN)):
            af.add_patch(Polygon(loop, closed=True, facecolor="none", edgecolor=(*colr, P.loop_alpha * a_in), linewidth=P.loop_width * sc, zorder=4, joinstyle="round"))
        af.set_xlim(xmin, xmax)
        af.set_ylim(-yhf, yhf)
        a_zero = a_in * (1.0 - SMOOTH(fl, *P.f_zero_fade))
        if a_zero > 0.01:
            af.text(P.f_zero_pos[0], P.f_zero_pos[1], tx["f_zero"], color=(*DIM, a_zero), fontsize=F.readout * sc, ha="center", va="center", zorder=9)
        self.atext(af, P.f_title_pos[0], P.f_title_pos[1], tx["panel_f"], F.panel_title, TXT, a_in, ha="left", va="bottom")
        # F computed by finite differences from the arrays of A (the same grid for A and for the gauge-transformed A)
        h = P.fd_step
        fx = np.arange(-P.fd_half[0], P.fd_half[0] + 1e-9, h)
        fy = np.arange(-P.fd_half[1], P.fd_half[1] + 1e-9, h)
        QX, QY = np.meshgrid(fx, fy)
        axx, ayy = pl.field.a_field(QX, QY, fl, gauge)
        a0x, a0y = pl.field.a_field(QX, QY, fl, 0.0)
        f_now = gp.curl_fd(axx, ayy, h)
        f_ref = gp.curl_fd(a0x, a0y, h)
        f_peak = float(pl.field.b_field(np.array(CFG.plane.tube_centre[0]), np.array(CFG.plane.tube_centre[1]), 1.0))
        dF = float(np.max(np.abs(f_now - f_ref))) / f_peak                                       # the change of F by the gauge transformation, in units of the peak value
        # ---- the readouts
        phase1 = pl.loop_phase(pl.loop1, fl, gauge)
        phase2 = pl.loop_phase(pl.loop2, fl, gauge)
        flux1 = pl.loop_flux(pl.loop1, fl)
        a_r = SMOOTH(t, *P.readout_in)
        pi_ = math.pi
        rows = [(tx["r5_loop1"], f"${num(phase1 / pi_, 3, lang)}\\pi$", GOLD),
                (tx["r5_flux1"], f"${num(flux1 / pi_, 3, lang)}\\pi$", GOLD),
                (tx["r5_loop2"], f"${num(phase2 / pi_, 3, lang)}\\pi$", self.GREEN),
                (tx["r5_df"], f"${sci(dF, lang)}$" if dF > P.floor else "$0$", TXT)]
        for i, (lab, val, colr) in enumerate(rows):
            yy = P.readout_y - i * P.readout_row
            self.ftext(P.readout_x, yy, lab, F.readout_label, DIM, a_r, ha="left", va="bottom")
            self.ftext(P.readout_x + P.readout_w, yy, val, F.readout, colr, a_r, ha="right", va="bottom")
        # ---- the screen: the fringes
        a5 = self.ax_p5
        self.reset(a5)
        pattern = pl.screen_pattern(fl, gauge)
        ref = pl.screen_pattern(0.0, 0.0)
        a5.set_xlim(-CFG.interf.screen_half, CFG.interf.screen_half)
        a5.set_ylim(0.0, P.profile_ymax)
        a5.fill_between(pl.y_pts, 0, pattern, color=(*BLUE, P.profile_fill_alpha * a_in), linewidth=0)
        a5.plot(pl.y_pts, ref, color=(*DIM, P.ghost_alpha * a_in), lw=P.ghost_width * sc, ls=(0, tuple(P.ghost_dash)))
        a5.plot(pl.y_pts, pattern, color=(*TXT, a_in), lw=P.profile_width * sc)
        expected = (phase1 / TWO_PI) if fl > 0.01 else 0.0
        shift = pl.peak_shift(pattern, expected=-expected if P.shift_sign < 0 else expected)
        pk0 = I.peak0
        pk1 = pk0 + shift * I.period
        a5.plot([pk0, pk0], [0, P.profile_ymax], color=(*DIM, P.peak_alpha * a_in), lw=1.0 * sc, ls=(0, tuple(P.ghost_dash)), zorder=1)
        a5.plot([pk1, pk1], [0, P.profile_ymax], color=(*GOLD, P.peak_alpha * a_in), lw=1.4 * sc, zorder=1)
        self.atext(a5, P.profile_title_pos[0], P.profile_title_pos[1], tx["panel_screen"], F.panel_title, TXT, a_in, ha="left", va="bottom")
        self.atext(a5, P.shift_pos[0], P.shift_pos[1], fill(tx["r5_shift"], lang, s=(shift, 2)), F.readout_label, GOLD, a_r, ha="right", va="top")

    # ================================================================= part 6: summary
    def part6(self, t: float) -> None:
        P, L, F, tx = CFG.part6, CFG.layout, CFG.fonts, self.tx
        for i, key in enumerate(P.lines):
            a = SMOOTH(t, *P.times[i])
            size = F[P.fonts[i]]
            col = self.GOLD if P.gold[i] else self.TXT
            self.ftext(0.5, P.y[i], tx[key], size, col, a, ha="center", va="center", linespacing=L.caption_spacing)


# ----------------------------------------------------------------------------------- the film

def render(out: Path, size: tuple[int, int], fps: int, total: float, lang: str, snap: float | None = None) -> None:
    V = CFG.video
    if snap is None and not shutil.which("ffmpeg"):
        sys.exit("ffmpeg is required")
    cv = Canvas(size, lang)
    W, H = size
    frames = int(round(total * fps))
    ids = [int(round(snap * fps))] if snap is not None else range(frames)
    writer = None
    if snap is None:
        writer = subprocess.Popen(
            ["ffmpeg", "-y", "-v", "error", "-f", "rawvideo", "-pix_fmt", "rgba", "-s", f"{W}x{H}",
             "-r", str(fps), "-i", "-", "-c:v", "libx264", "-preset", V.preset, "-crf", str(V.crf),
             "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(out)], stdin=subprocess.PIPE)
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
    out = args.out or HERE / "media" / f"gauge_principle_{args.lang}.mp4"
    out.parent.mkdir(parents=True, exist_ok=True)
    if args.snapshot is not None:
        render(out.with_suffix(".png"), (V.width, V.height), V.fps, args.seconds, args.lang, snap=args.snapshot)
    elif args.preview:
        render(out.with_name(out.stem + "_preview.mp4"), (V.preview_width, V.preview_height), V.preview_fps, args.seconds, args.lang)
    else:
        render(out, (V.width, V.height), V.fps, args.seconds, args.lang)


if __name__ == "__main__":
    main()
