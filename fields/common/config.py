from __future__ import annotations

import os
from manim import config

from .camera_presets import get_profile_preset, get_scene_layout


VALID_PROFILES = {"widescreen", "shorts"}


def current_profile(default: str = "widescreen") -> str:
    profile = os.environ.get("DVN_PROFILE", default).strip().lower()
    return profile if profile in VALID_PROFILES else default


def _apply_aspect_preserving_render_geometry(frame_width: float, frame_height: float) -> None:
    """Keep the CLI-selected quality tier while enforcing the target aspect ratio."""
    aspect = frame_width / frame_height
    long_side = max(int(config.pixel_width), int(config.pixel_height))
    if aspect >= 1.0:
        config.pixel_width = long_side
        config.pixel_height = max(1, int(round(long_side / aspect)))
    else:
        config.pixel_height = long_side
        config.pixel_width = max(1, int(round(long_side * aspect)))
    config.frame_width = frame_width
    config.frame_height = frame_height


def apply_manim_profile(profile: str | None = None) -> str:
    resolved = current_profile(profile or "widescreen")
    preset = get_profile_preset(resolved)
    _apply_aspect_preserving_render_geometry(
        preset["frame_width"],
        preset["frame_height"],
    )
    return resolved


def scene_bundle(scene_key: str, profile: str | None = None) -> dict:
    resolved = current_profile(profile or "widescreen")
    return {
        "profile": resolved,
        "render": get_profile_preset(resolved),
        "layout": get_scene_layout(scene_key, resolved),
    }
