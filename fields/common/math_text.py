from __future__ import annotations

from manim import DOWN, LEFT, MathTex, Text, UP, VGroup, WHITE


def make_panel(
    title: str,
    lines: list[tuple[str, str, str]],
    title_scale: float,
    line_scale: float,
    top_buff: float,
):
    items = [Text(title, color=WHITE).scale(title_scale)]
    for kind, content, color in lines:
        mob = MathTex(content, color=color).scale(line_scale) if kind == "math" else Text(content, color=color).scale(line_scale)
        items.append(mob)
    group = VGroup(*items).arrange(DOWN, aligned_edge=LEFT, buff=0.08)
    group.to_edge(UP, buff=top_buff)
    return group
