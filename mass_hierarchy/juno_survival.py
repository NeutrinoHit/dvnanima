from __future__ import annotations

import numpy as np
from manim import *
from manim import ManimColor

# =============================
# Global style
# =============================
BG = ManimColor("#05060A")
SOFT_LIGHT = ManimColor("#E9F2FF")
NEUTRINO_GOLD = ManimColor("#F0E68C")

PALETTE = [
    ManimColor("#3b0f70"),  # deep purple
    ManimColor("#1f4aa8"),  # blue
    ManimColor("#1fa187"),  # teal/green
    ManimColor("#4ac16d"),  # green
    ManimColor("#fde725"),  # yellow
    ManimColor("#f46d43"),  # orange
    ManimColor("#d73027"),  # red
]

def palette_color(x: float) -> ManimColor:
    """x in [0,1] -> color"""
    x = float(np.clip(x, 0.0, 1.0))
    n = len(PALETTE) - 1
    t = x * n
    i = int(np.floor(t))
    a = t - i
    if i >= n:
        return PALETTE[-1]
    return interpolate_color(PALETTE[i], PALETTE[i+1], a)

# =============================
# Physics: 3-flavor vacuum Pee
# =============================
def pee_3flavor_vacuum(
    E_MeV: float,
    L_km: float,
    theta12_deg: float = 33.44,
    theta13_deg: float = 8.57,
    dm21: float = 7.42e-5,
    dm31: float = +2.517e-3,  # NO default; IO will use negative below
) -> float:
    """
    Pee = 1 - cos^4(th13)*sin^2(2th12)*sin^2(Δ21)
          - sin^2(2th13)*(cos^2(th12)*sin^2(Δ31) + sin^2(th12)*sin^2(Δ32))
    Δij = 1.267 * Δm^2_ij[eV^2] * L[km] / E[GeV]
    """
    E_GeV = E_MeV * 1e-3
    th12 = np.deg2rad(theta12_deg)
    th13 = np.deg2rad(theta13_deg)

    s12, c12 = np.sin(th12), np.cos(th12)
    s13, c13 = np.sin(th13), np.cos(th13)

    dm32 = dm31 - dm21

    def sin2(x):  # noqa: E306
        return np.sin(x) ** 2

    D21 = 1.267 * dm21 * L_km / E_GeV
    D31 = 1.267 * dm31 * L_km / E_GeV
    D32 = 1.267 * dm32 * L_km / E_GeV

    term_solar = (c13**4) * (np.sin(2 * th12) ** 2) * sin2(D21)
    term_atm = (np.sin(2 * th13) ** 2) * ((c12**2) * sin2(D31) + (s12**2) * sin2(D32))

    Pee = 1.0 - term_solar - term_atm
    return float(np.clip(Pee, 0.0, 1.0))


# =============================
# Baseline L(t): keyframes
# =============================
def baseline_from_keyframes(u: float, keyframes: list[tuple[float, float]]) -> float:
    """
    u in [0,1]. Piecewise-linear L(u) via np.interp.
    Holds are made by repeating L at adjacent keyframes.
    """
    u = float(np.clip(u, 0.0, 1.0))
    us = np.array([k[0] for k in keyframes], dtype=float)
    Ls = np.array([k[1] for k in keyframes], dtype=float)
    return float(np.interp(u, us, Ls))


# =============================
# JUNO sphere inset (pseudo-3D)
# =============================
def sample_sphere_points(n: int, seed: int = 1) -> np.ndarray:
    """Uniform-ish points on a sphere (Fibonacci spiral). Returns (n,3) array."""
    rng = np.random.default_rng(seed)
    i = np.arange(n)
    phi = (1 + 5**0.5) / 2
    theta = 2 * np.pi * i / phi
    z = 1 - 2 * (i + 0.5) / n
    r = np.sqrt(np.clip(1 - z * z, 0, 1))
    x = r * np.cos(theta)
    y = r * np.sin(theta)
    pts = np.stack([x, y, z], axis=1)
    rng.shuffle(pts)
    return pts


def rot_y(a: float) -> np.ndarray:
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]], dtype=float)


def rot_x(a: float) -> np.ndarray:
    c, s = np.cos(a), np.sin(a)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]], dtype=float)


class JunoSphereInset(VGroup):
    """
    Pseudo-3D sphere with PMT dots.
    Uses an explicit anchor point (world coordinates) so updaters never reset dots to origin.
    """

    def __init__(self, n_pmts: int = 3000, radius: float = 1.05, **kwargs):
        super().__init__(**kwargs)
        self.n_pmts = int(n_pmts)
        self.radius = float(radius)

        self.pts0 = sample_sphere_points(self.n_pmts, seed=2)
        self.theta = 0.0
        self.phi = 0.60  # tilt

        self.depth_arr = np.full(self.n_pmts, 0.5, dtype=float)
        self.hits = np.zeros(self.n_pmts, dtype=int)

        # submobjects
        self.outline = Circle(radius=self.radius).set_stroke(color=SOFT_LIGHT, opacity=0.30, width=2).set_fill(opacity=0)

        self.dots = VGroup(*[
            Dot(radius=0.011, color=SOFT_LIGHT, fill_opacity=0.10)
            for _ in range(self.n_pmts)
        ])

        self.add(self.outline, self.dots)

        # anchor in WORLD coords (where inset is placed)
        self._anchor = ORIGIN.copy()

        # initial layout around anchor
        self._update_positions()
        self.set_hits(self.hits)

    # ---- anchor-aware positioning API ----
    def move_to(self, point_or_mobject, **kwargs):
        super().move_to(point_or_mobject, **kwargs)
        self._anchor = self.get_center()
        return self

    def shift(self, vector):
        super().shift(vector)
        self._anchor = self.get_center()
        return self

    def set_anchor(self, point):
        """Force anchor point (world coords). Call once after final placement."""
        self._anchor = np.array(point, dtype=float)
        return self

    # ---- internal geometry ----
    def _project(self, pts3: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        x, y, z = pts3[:, 0], pts3[:, 1], pts3[:, 2]
        pts2 = np.stack([x, y, np.zeros_like(x)], axis=1)
        depth = (z + 1) / 2
        return pts2, depth

    def _update_positions(self):
        # local rotated sphere
        R = rot_y(self.theta) @ rot_x(self.phi)
        pts = (self.pts0 @ R.T) * self.radius
        pts2, depth = self._project(pts)
        self.depth_arr = depth

        # world anchor (fixed inset position)
        center = self._anchor

        # account for external scaling of the whole group
        # outline width = 2*radius*scale
        scale_factor = self.outline.get_width() / (2.0 * self.radius)

        # keep outline centered at anchor too
        self.outline.move_to(center)

        for i, d in enumerate(self.dots):
            d.move_to(center + pts2[i] * scale_factor)

    def set_hits(self, hits: np.ndarray):
        self.hits = hits.astype(int)
        depth_arr = self.depth_arr
        h = np.clip(self.hits, 0, 20)
        for i, d in enumerate(self.dots):
            depth = float(self.depth_arr[i])
            base_op = 0.05 + 0.22 * depth

            x = float(np.sqrt(h[i] / 20.0))  # 0..1
            col = palette_color(x)

            op = min(0.90, base_op + 0.035 * float(h[i]))
            d.set_color(col)
            d.set_opacity(op)


    def rotate_step(self, dt: float, omega: float = 0.18):
        self.theta += omega * dt
        self._update_positions()
        # не пересэмплируем hits, только перерисовываем их с новым depth
        self.set_hits(self.hits)


# =============================
# Main scene: graph + sphere inset
# =============================
class JunoSurvivalScene(Scene):
    def construct(self):
        self.camera.background_color = BG

        # --- CONFIG ---
        L_FINAL = 53.0
        T_TOTAL = 212.0  # seconds (song)
        E_MIN, E_MAX = 1.8, 8.0
        NPTS = 600

        # If you want strictly uniform speed: make L linear and ignore keyframes
        USE_KEYFRAMES = False

        keyframes = [
            (0.00, 0.0),
            (0.10, 5.3),
            (0.30, 20.0),
            (0.55, 35.0),
            (1.00, L_FINAL),
        ]

        dm31_NO = +2.517e-3
        dm31_IO = -2.498e-3

        # --- Axes ---
        axes = Axes(
            x_range=[E_MIN, E_MAX, 1.0],
            y_range=[0.0, 1.0, 0.2],
            x_length=11.5,
            y_length=5.5,
            tips=False,
        ).set_stroke(opacity=0.25)

        xlab = Text("E (MeV)", font_size=28).set_fill(SOFT_LIGHT, opacity=0.85)
        ylab = Text("Survival Probability",font_size=24,t2c={"Survival": SOFT_LIGHT, "Probability": SOFT_LIGHT}).set_fill(SOFT_LIGHT, opacity=0.85).rotate(PI/2)

        xlab.next_to(axes, DOWN, buff=0.35).shift(RIGHT * 4.2)
        ylab.next_to(axes, LEFT, buff=0.35).shift(UP * 0.2)

        self.add(axes, xlab, ylab)

        # --- Time tracker u in [0,1] ---
        u = ValueTracker(0.0)

        def current_L() -> float:
            if USE_KEYFRAMES:
                return baseline_from_keyframes(u.get_value(), keyframes)
            return float(L_FINAL * u.get_value())

        L_text = always_redraw(
            lambda: Text(f"L = {current_L():4.1f} km", font_size=30)
            .set_fill(SOFT_LIGHT, opacity=0.9)
            .to_corner(UR)
            .shift(LEFT * 0.4 + DOWN * 0.4)
        )
        self.add(L_text)

        # --- Graphs ---
        E_grid = np.linspace(E_MIN, E_MAX, NPTS)

        def make_graph(dm31: float, opacity: float = 0.95) -> VMobject:
            L = max(current_L(), 0.8)  # avoid singular in phases
            ys = [pee_3flavor_vacuum(float(E), L_km=L, dm31=dm31) for E in E_grid]
            pts = [axes.c2p(float(E), float(y)) for E, y in zip(E_grid, ys)]
            g = VMobject()
            g.set_points_smoothly(pts)
            g.set_stroke(SOFT_LIGHT, width=3, opacity=opacity)
            return g

        graph_NO = always_redraw(lambda: make_graph(dm31_NO, opacity=0.95))
        graph_IO = always_redraw(lambda: make_graph(dm31_IO, opacity=0.55))

        leg_NO = Text("NO", font_size=26).set_fill(SOFT_LIGHT, opacity=0.95)
        leg_IO = Text("IO", font_size=26).set_fill(SOFT_LIGHT, opacity=0.55)
        legend = VGroup(leg_NO, leg_IO).arrange(DOWN, aligned_edge=LEFT, buff=0.15)
        legend.to_corner(UL).shift(RIGHT * 0.4 + DOWN * 0.4)

        self.add(graph_NO, graph_IO, legend)

        marker_E = 4.0
        marker = Line(axes.c2p(marker_E, 0.0), axes.c2p(marker_E, 1.0)).set_stroke(
            SOFT_LIGHT, width=2, opacity=0.08
        )
        self.add(marker)

        # --- JUNO sphere inset (bottom-right inside axes) ---
        # Performance note: 3000 is safe. Try 5000 if your machine is fine.
        N_PMT = 3000

        # Visual/physics knobs for "event rate vs distance"
        # At L0 we have ~6000 p.e. (your assumption at E~4 MeV)
        NPE_AT_L0 = 6000.0
        L0 = 5.3      # km reference (so at 53 km ~100x less)
        L_MIN = 0.8   # km clamp
        DT0 = 0.08    # seconds between updates at L=L0 (tune to taste)

        sphere = JunoSphereInset(n_pmts=N_PMT, radius=1.05).scale(1.75)
        sphere.move_to(axes.get_corner(DR) + LEFT * 1.35 + UP * 2.05)

        sphere_label = Text("JUNO", font_size=22).set_fill(SOFT_LIGHT, opacity=0.75)
        sphere_label.next_to(sphere, RIGHT, buff=0.08)

        self.add(sphere, sphere_label)

        acc = {"t": 0.0}
        # --- Event rate & brightness knobs ---
        L_FINAL = 53.0
        L_MIN = 0.8

        NPE_PER_EVENT = 6000.0   # constant brightness per event (your assumption)
        DT_MIN = 0.12            # frequent events (near source)
        DT_MAX = 2.50            # rare events (far), but never zero

        acc = {"t": 0.0}

        def sphere_updater(mobj: JunoSphereInset, dt: float):
            # rotation always
            mobj.rotate_step(dt, omega=0.18)

            L = max(current_L(), L_MIN)

            # event interval grows with L^2, bounded [DT_MIN, DT_MAX]
            frac = (L / L_FINAL) ** 2
            update_dt = DT_MIN + (DT_MAX - DT_MIN) * float(np.clip(frac, 0.0, 1.0))

            acc["t"] += dt
            if acc["t"] < update_dt:
                return
            acc["t"] = 0.0

            # constant per-event brightness; optional Pee modulation
            Pee4 = pee_3flavor_vacuum(4.0, L_km=L, dm31=dm31_NO)
            npe_total = NPE_PER_EVENT * Pee4   # or just NPE_PER_EVENT if you want fully constant

            mu_per_pmt = npe_total / N_PMT
            hits = np.random.poisson(lam=max(mu_per_pmt, 0.0), size=N_PMT)

            mobj.set_hits(hits)

        sphere.add_updater(sphere_updater)

        # --- Animate whole song ---
        self.play(u.animate.set_value(1.0), run_time=T_TOTAL, rate_func=linear)

        sphere.remove_updater(sphere_updater)
        self.wait(0.3)
