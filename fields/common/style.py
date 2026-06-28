from __future__ import annotations

from manim import Surface, VGroup, WHITE

COLORS = {
    "background": "#050608",
    "text": "#F4F5F7",
    "muted_text": "#BCC3D0",
    "accent": "#FFD166",
    "warning": "#FF7B72",
    "phi_fill": "#69B7FF",
    "phi_stroke": "#D9F0FF",
    "phi_secondary": "#4E8FD2",
    "chi_fill": "#FFB267",
    "chi_stroke": "#FFE1BF",
    "plane_stroke": "#8391A7",
}


def style_surface(surface: Surface, fill_color: str, stroke_color: str, fill_opacity: float = 0.62, stroke_width: float = 0.8, stroke_opacity: float = 0.78) -> Surface:
    surface.set_fill(fill_color, opacity=fill_opacity)
    surface.set_stroke(stroke_color, width=stroke_width, opacity=stroke_opacity)
    return surface


def reference_plane(x_range: tuple[float, float], y_range: tuple[float, float], resolution: tuple[int, int], z_level: float = 0.0) -> Surface:
    plane = Surface(
        lambda u, v: [u, v, z_level],
        u_range=x_range,
        v_range=y_range,
        resolution=resolution,
    )
    plane.set_fill(opacity=0.0)
    plane.set_stroke(COLORS["plane_stroke"], width=0.65, opacity=0.55)
    return plane


def stack_surfaces(*surfaces: Surface) -> VGroup:
    return VGroup(*surfaces)
