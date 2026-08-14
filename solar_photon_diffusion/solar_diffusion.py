from __future__ import annotations

import configparser
import os
from pathlib import Path

import numpy as np
from manim import *

from physics import (
    BCZ_RADIUS_FRACTION,
    SECONDS_PER_YEAR,
    load_profile,
    mean_exit_time_seconds,
    production_weighted_exit_time_seconds,
    radius_at_delay_fraction,
)


ROOT = Path(__file__).resolve().parent


def load_cfg() -> dict:
    parser = configparser.ConfigParser()
    parser.read(ROOT / "run.cfg")

    def get(section: str, key: str, cast, fallback):
        return cast(parser.get(section, key, fallback=str(fallback)))

    return {
        "manim": {
            "pixel_width": get("manim", "pixel_width", int, 1080),
            "pixel_height": get("manim", "pixel_height", int, 1920),
            "frame_width": get("manim", "frame_width", float, 9.0),
            "frame_height": get("manim", "frame_height", float, 16.0),
            "frame_rate": get("manim", "frame_rate", int, 30),
            "background_color": get(
                "manim", "background_color", str, "#03060d"
            ),
        },
        "scene": {
            key: get("scene", key, float, fallback)
            for key, fallback in {
                "intro_time": 3.8,
                "zone_reveal_time": 2.2,
                "micro_intro_time": 1.4,
                "micro_walk_time": 7.5,
                "transition_time": 1.2,
                "model_intro_time": 0.8,
                "model_step_time": 1.0,
                "model_hold_time": 1.5,
                "model_outro_time": 0.8,
                "main_intro_time": 1.5,
                "radiative_sweep_time": 15.0,
                "bcz_reveal_time": 0.8,
                "bcz_hold_time": 3.0,
                "outer_sweep_time": 5.0,
                "outer_hold_time": 0.8,
                "main_outro_time": 1.0,
                "summary_card_time": 1.2,
                "summary_details_time": 1.5,
                "final_hold_time": 5.7,
            }.items()
        },
        "colors": dict(parser["colors"]),
    }


CFG = load_cfg()
config.pixel_width = CFG["manim"]["pixel_width"]
config.pixel_height = CFG["manim"]["pixel_height"]
config.frame_width = CFG["manim"]["frame_width"]
config.frame_height = CFG["manim"]["frame_height"]
config.frame_rate = CFG["manim"]["frame_rate"]
if os.environ.get("SOLAR_DIFFUSION_PREVIEW") == "1":
    config.pixel_width = 360
    config.pixel_height = 640
    config.frame_rate = 15


LANGUAGE = os.environ.get("SOLAR_DIFFUSION_LANG", "ru").strip().lower()
if LANGUAGE not in {"ru", "en"}:
    raise ValueError("SOLAR_DIFFUSION_LANG must be 'ru' or 'en'")

COPY = {
    "ru": {
        "title": "КАК СВЕТ ВЫХОДИТ ИЗ СОЛНЦА",
        "subtitle": "ОТ ЯДРА ДО ФОТОСФЕРЫ",
        "core_tag": "ядро  0–0,2 R☉",
        "radiative_tag": "лучистая зона",
        "convection_tag": "конвективная зона",
        "micro_label": "в центре:  ℓ = 1/(κᵣρ) ≈ {value} мм",
        "scale_note_prefix": "зигзаг увеличен примерно в",
        "scale_note_suffix": "раз",
        "micro_scale": "микромасштаб",
        "model_title": "КАК ТЕМПЕРАТУРА ВХОДИТ В РАСЧЁТ",
        "model_steps": [
            "частотный перенос",
            "локальное равновесие",
            "среднее по энергии",
            "диффузия",
        ],
        "equation_caption": "диффузия + сферическая геометрия",
        "delay_label": "накопленное среднее время",
        "years": "лет",
        "bcz_label": "до границы конвекции",
        "thousand_years": "{value} тыс. лет",
        "extrapolation": "условное продолжение чистой диффузии",
        "central_card": "из самого центра",
        "weighted_card": "с учётом места рождения энергии",
        "caveat": (
            "Внешний слой переносит энергию преимущественно конвекцией.\n"
            "41,4 тыс. лет — контрольное продолжение чистой диффузии."
        ),
        "earth": "после поверхности:  8 мин 19 с",
        "mfp_title": "локальный шаг  ℓ(r) = 1/(κᵣρ)",
        "cm": "см",
        "half_radius": "0,5",
        "bcz_tick": "0,713",
        "temperature_ticks": ["15,7 MK", "2,2 MK", "5,8 kK"],
        "temperature_note": "цветовой код\nT(r)",
        "still_title": "СВЕТ ИЗ ЦЕНТРА СОЛНЦА",
        "still_subtitle": "модель радиационной диффузии",
        "still_weighted": "усреднение по рождению энергии",
    },
    "en": {
        "title": "HOW LIGHT ESCAPES THE SUN",
        "subtitle": "FROM THE CORE TO THE PHOTOSPHERE",
        "core_tag": "core  0–0.2 R☉",
        "radiative_tag": "radiative zone",
        "convection_tag": "convective zone",
        "micro_label": "at the center:  ℓ = 1/(κᵣρ) ≈ {value} mm",
        "scale_note_prefix": "the random walk is magnified by about",
        "scale_note_suffix": "",
        "micro_scale": "microscopic scale",
        "model_title": "HOW TEMPERATURE ENTERS THE MODEL",
        "model_steps": [
            "frequency transport",
            "local equilibrium",
            "energy average",
            "diffusion",
        ],
        "equation_caption": "diffusion + spherical geometry",
        "delay_label": "accumulated mean time",
        "years": "years",
        "bcz_label": "to the convection boundary",
        "thousand_years": "{value} thousand years",
        "extrapolation": "conditional pure-diffusion extension",
        "central_card": "starting at the center",
        "weighted_card": "weighted by energy-production radius",
        "caveat": (
            "The outer layer transports energy mainly by convection.\n"
            "41.4 kyr is a control extension of pure photon diffusion."
        ),
        "earth": "after the surface:  8 min 19 s",
        "mfp_title": "local step  ℓ(r) = 1/(κᵣρ)",
        "cm": "cm",
        "half_radius": "0.5",
        "bcz_tick": "0.713",
        "temperature_ticks": ["15.7 MK", "2.2 MK", "5.8 kK"],
        "temperature_note": "T color\ncode",
        "still_title": "LIGHT FROM THE SUN'S CORE",
        "still_subtitle": "a radiative-diffusion model",
        "still_weighted": "weighted by energy-production radius",
    },
}[LANGUAGE]


def local_decimal(value: float, digits: int = 1) -> str:
    text = f"{value:.{digits}f}"
    return text.replace(".", ",") if LANGUAGE == "ru" else text


class SolarPhotonDiffusion(Scene):
    """Portrait animation of photon-energy diffusion through the Sun."""

    def construct(self) -> None:
        self.camera.background_color = CFG["manim"]["background_color"]
        colors = CFG["colors"]
        timing = CFG["scene"]
        profile = load_profile()
        hold_driver = ValueTracker(0.0)
        # Keep this first in the scene.  Animating it for visual holds forces
        # Cairo to redraw the whole frame instead of building a broken static
        # cache around always_redraw objects.
        hold_driver.add_updater(lambda mob, dt: None)
        self.add(hold_driver)

        centre_to_surface_years = mean_exit_time_seconds(profile) / SECONDS_PER_YEAR
        centre_to_bcz_years = (
            mean_exit_time_seconds(
                profile, escape_radius_fraction=BCZ_RADIUS_FRACTION
            )
            / SECONDS_PER_YEAR
        )
        weighted_years = (
            production_weighted_exit_time_seconds(profile) / SECONDS_PER_YEAR
        )
        bcz_delay_fraction = centre_to_bcz_years / centre_to_surface_years
        core_mfp_mm = profile.mean_free_path_cm[0] * 10.0

        stars = self._make_stars(colors["muted"])
        title = Text(
            COPY["title"],
            font="Arial",
            font_size=37,
            weight=BOLD,
            color=colors["white"],
        ).move_to([0, 7.28, 0])
        subtitle = Text(
            COPY["subtitle"],
            font="Arial",
            font_size=20,
            weight=BOLD,
            color=colors["muted"],
        ).move_to([0, 6.75, 0])
        title.set_z_index(100)
        subtitle.set_z_index(100)
        sun_center = np.array([0.0, 2.45, 0.0])
        sun_radius = 3.08
        corona = self._make_corona(sun_center, sun_radius, colors)
        sun = self._make_sun(sun_center, sun_radius, colors)
        zone_overlays = self._make_zone_overlays(sun_center, sun_radius, colors)
        convection_texture = self._make_convection_texture(
            sun_center, sun_radius, colors
        )
        bcz_boundary = DashedVMobject(
            Circle(
                radius=sun_radius * BCZ_RADIUS_FRACTION,
                color=colors["convective_zone"],
                stroke_width=2.5,
            ).move_to(sun_center),
            num_dashes=70,
        ).set_opacity(0.92)
        core_boundary = Circle(
            radius=sun_radius * 0.2,
            color=colors["core"],
            stroke_width=1.4,
        ).set_opacity(0.52).move_to(sun_center)

        core_tag = self._layer_tag(
            COPY["core_tag"],
            sun_center + np.array([-1.65, -0.2, 0]),
            sun_center + np.array([-0.56, -0.12, 0]),
            colors["core"],
            colors,
        )
        radiative_tag = self._layer_tag(
            COPY["radiative_tag"],
            sun_center + np.array([-3.25, 1.3, 0]),
            sun_center + np.array([-2.0, 0.78, 0]),
            colors["radiative_zone"],
            colors,
        )
        convection_tag = self._layer_tag(
            COPY["convection_tag"],
            sun_center + np.array([2.7, 2.03, 0]),
            sun_center + np.array([2.48, 1.3, 0]),
            colors["convective_zone"],
            colors,
            align="right",
        )

        self.add(stars)
        self.play(
            FadeIn(corona, scale=0.85),
            FadeIn(sun, scale=0.88),
            FadeIn(convection_texture),
            Write(title),
            FadeIn(subtitle, shift=0.12 * DOWN),
            run_time=timing["intro_time"],
        )
        # Keep the header in Cairo's moving-object pass.  Otherwise dynamic
        # overlays can make the static raster cache omit glyphs in individual
        # partial-movie segments.
        title.add_updater(lambda mob, dt: mob.set_opacity(1.0))
        subtitle.add_updater(lambda mob, dt: mob.set_opacity(1.0))
        self.play(
            FadeIn(zone_overlays),
            Create(bcz_boundary),
            Create(core_boundary),
            FadeIn(core_tag),
            FadeIn(radiative_tag),
            FadeIn(convection_tag),
            run_time=timing["zone_reveal_time"],
        )

        magnifier = self._make_magnifier(colors)
        connector = DashedLine(
            sun_center + np.array([0.24, -0.42, 0]),
            np.array([0.0, -2.85, 0]),
            dash_length=0.08,
            color=colors["muted"],
            stroke_width=1.5,
        ).set_opacity(0.55)
        micro_path = self._make_micro_walk(np.array([0.0, -4.4, 0.0]), colors)
        micro_photon = self._glowing_dot(micro_path.get_start(), colors["cyan"], 0.075)
        micro_label = Text(
            COPY["micro_label"].format(
                value=local_decimal(core_mfp_mm, 3)
            ),
            font="Arial",
            font_size=25,
            color=colors["white"],
        ).move_to([0, -6.28, 0])
        scale_note_parts = [
            Text(
                COPY["scale_note_prefix"],
                font="Arial",
                font_size=19,
                color=colors["muted"],
            ),
            MathTex(r"10^{14}", font_size=25, color=colors["muted"]),
        ]
        if COPY["scale_note_suffix"]:
            scale_note_parts.append(
                Text(
                    COPY["scale_note_suffix"],
                    font="Arial",
                    font_size=19,
                    color=colors["muted"],
                )
            )
        scale_note = VGroup(*scale_note_parts).arrange(RIGHT, buff=0.09)
        scale_note.next_to(micro_label, DOWN, buff=0.22)

        self.play(
            FadeIn(connector),
            FadeIn(magnifier, scale=0.8),
            FadeIn(micro_photon),
            FadeIn(micro_label, shift=0.12 * UP),
            FadeIn(scale_note, shift=0.12 * UP),
            run_time=timing["micro_intro_time"],
        )
        self.play(
            Create(micro_path),
            MoveAlongPath(micro_photon, micro_path),
            run_time=timing["micro_walk_time"],
            rate_func=linear,
        )

        self.play(
            FadeOut(
                VGroup(
                    magnifier,
                    connector,
                    micro_path,
                    micro_photon,
                    micro_label,
                    scale_note,
                ),
                shift=0.15 * DOWN,
            ),
            FadeOut(core_tag),
            FadeOut(radiative_tag),
            FadeOut(convection_tag),
            run_time=timing["transition_time"],
        )

        model_panel, model_title, model_steps, model_arrows = (
            self._make_model_chain(colors)
        )
        self.play(
            FadeIn(model_panel, shift=0.10 * UP),
            FadeIn(model_title),
            run_time=timing["model_intro_time"],
        )
        self.play(
            FadeIn(model_steps[0], shift=0.08 * UP),
            run_time=timing["model_step_time"],
        )
        for index in range(1, len(model_steps)):
            self.play(
                GrowArrow(model_arrows[index - 1]),
                FadeIn(model_steps[index], shift=0.08 * UP),
                run_time=timing["model_step_time"],
            )
        self.play(
            hold_driver.animate.increment_value(1.0),
            run_time=timing["model_hold_time"],
            rate_func=linear,
        )
        self.play(
            FadeOut(VGroup(model_panel, model_title, model_steps, model_arrows)),
            run_time=timing["model_outro_time"],
        )

        equation_panel, equation = self._make_equation_panel(colors)
        delay_label = Text(
            COPY["delay_label"],
            font="Arial",
            font_size=17,
            color=colors["muted"],
        ).move_to([2.05, -1.02, 0])
        years_number = DecimalNumber(
            0,
            num_decimal_places=0,
            group_with_commas=True,
            font_size=52,
            color=colors["white"],
        ).move_to([1.55, -1.67, 0])
        years_unit = Text(
            COPY["years"],
            font="Arial",
            font_size=24,
            color=colors["muted"],
        ).next_to(years_number, RIGHT, buff=0.2)
        progress = ValueTracker(0.0)
        years_number.add_updater(
            lambda mob: mob.set_value(
                progress.get_value() * centre_to_surface_years
            )
        )
        years_unit.add_updater(lambda mob: mob.next_to(years_number, RIGHT, buff=0.2))

        def current_radius_fraction() -> float:
            return float(radius_at_delay_fraction(profile, progress.get_value()))

        def current_temperature_color():
            return self._temperature_color(
                profile,
                current_radius_fraction(),
                colors,
            )

        mean_free_path_chart = self._make_mean_free_path_chart(profile, colors)
        mean_free_path_cursor = always_redraw(
            lambda: self._make_mean_free_path_cursor(
                profile,
                current_radius_fraction(),
                colors,
                accent=current_temperature_color(),
            )
        )

        temperature_legend = self._make_temperature_legend(profile, colors)
        temperature_cursor = always_redraw(
            lambda: self._make_temperature_cursor(
                profile,
                current_radius_fraction(),
                colors,
            )
        )

        diffusion_halo = always_redraw(
            lambda: self._make_diffusion_halo(
                sun_center,
                sun_radius,
                current_radius_fraction(),
                colors,
                front_color=current_temperature_color(),
            )
        )
        radius_readout = always_redraw(
            lambda: self._make_radius_readout(
                current_radius_fraction(),
                colors,
                position=np.array([2.05, -2.42, 0.0]),
                accent=current_temperature_color(),
            )
        )
        self.add(diffusion_halo)

        self.play(
            FadeIn(equation_panel, shift=0.12 * UP),
            Write(equation),
            FadeIn(delay_label),
            FadeIn(years_number),
            FadeIn(years_unit),
            FadeIn(radius_readout),
            FadeIn(mean_free_path_chart, shift=0.08 * UP),
            FadeIn(mean_free_path_cursor),
            FadeIn(temperature_legend, shift=0.08 * LEFT),
            FadeIn(temperature_cursor),
            run_time=timing["main_intro_time"],
        )
        self.play(
            progress.animate.set_value(bcz_delay_fraction),
            run_time=timing["radiative_sweep_time"],
            rate_func=smooth,
        )

        bcz_result = self._result_badge(
            COPY["bcz_label"],
            COPY["thousand_years"].format(
                value=local_decimal(centre_to_bcz_years / 1000)
            ),
            colors["cyan"],
            np.array([0.0, -6.98, 0.0]),
            colors,
        )
        self.play(
            FadeIn(bcz_result, shift=0.12 * UP),
            run_time=timing["bcz_reveal_time"],
        )
        self.play(
            hold_driver.animate.increment_value(1.0),
            run_time=timing["bcz_hold_time"],
            rate_func=linear,
        )
        extrapolation_label = Text(
            COPY["extrapolation"],
            font="Arial",
            font_size=14,
            color=colors["muted"],
        ).move_to(delay_label)
        # Switch regimes cleanly at r_bcz.  Keeping this out of the animated
        # sweep also avoids Cairo cache artefacts on the static overlay.
        self.remove(bcz_result, delay_label)
        delay_label = extrapolation_label
        self.add(delay_label)
        self.play(
            progress.animate.set_value(1.0),
            run_time=timing["outer_sweep_time"],
            rate_func=smooth,
        )
        self.play(
            hold_driver.animate.increment_value(1.0),
            run_time=timing["outer_hold_time"],
            rate_func=linear,
        )

        years_number.clear_updaters()
        years_unit.clear_updaters()
        final_group = VGroup(
            equation_panel,
            equation,
            delay_label,
            years_number,
            years_unit,
            radius_readout,
            mean_free_path_chart,
            mean_free_path_cursor,
            temperature_legend,
            temperature_cursor,
        )
        self.play(
            FadeOut(final_group),
            FadeOut(diffusion_halo),
            run_time=timing["main_outro_time"],
        )

        central_card = self._result_badge(
            COPY["central_card"],
            COPY["thousand_years"].format(
                value=local_decimal(centre_to_surface_years / 1000)
            ),
            colors["surface"],
            np.array([0.0, -2.45, 0.0]),
            colors,
        )
        weighted_card = self._result_badge(
            COPY["weighted_card"],
            COPY["thousand_years"].format(
                value=local_decimal(weighted_years / 1000)
            ),
            colors["cyan"],
            np.array([0.0, -4.03, 0.0]),
            colors,
        )
        earth_card = self._earth_card(colors, np.array([0.0, -6.17, 0.0]))
        caveat = Text(
            COPY["caveat"],
            font="Arial",
            font_size=18,
            line_spacing=0.82,
            color=colors["muted"],
            should_center=True,
        ).move_to([0, -7.43, 0])

        surface_flash = Circle(
            radius=sun_radius * 1.015,
            color=colors["core"],
            stroke_width=7,
        ).set_opacity(0.0).move_to(sun_center)
        self.add(surface_flash)
        self.play(
            surface_flash.animate.set_opacity(0.72).scale(1.055),
            FadeIn(central_card, shift=0.15 * UP),
            run_time=timing["summary_card_time"],
        )
        self.play(
            surface_flash.animate.set_opacity(0.0).scale(1.05),
            FadeIn(weighted_card, shift=0.15 * UP),
            run_time=timing["summary_card_time"],
        )
        self.play(
            FadeIn(earth_card),
            FadeIn(caveat),
            run_time=timing["summary_details_time"],
        )
        flight_pulse = self._glowing_dot(
            earth_card[1].get_start(), colors["core"], 0.035
        )
        self.add(flight_pulse)
        # Cairo's moving-object cache can otherwise redraw only a narrow strip
        # around the travelling pulse.  Mark every persistent layer as moving
        # during this last shot so no intermittent frame loses static content.
        for persistent in [
            stars,
            corona,
            sun,
            zone_overlays,
            convection_texture,
            bcz_boundary,
            core_boundary,
            title,
            subtitle,
            central_card,
            weighted_card,
            earth_card,
            caveat,
            surface_flash,
        ]:
            persistent.add_updater(lambda mob, dt: mob.shift(ORIGIN))
        self.play(
            MoveAlongPath(flight_pulse, earth_card[1]),
            hold_driver.animate.increment_value(1.0),
            run_time=timing["final_hold_time"],
            rate_func=linear,
        )

    @staticmethod
    def _make_stars(color: str) -> VGroup:
        rng = np.random.default_rng(20260813)
        stars = VGroup()
        for _ in range(92):
            point = np.array([rng.uniform(-4.35, 4.35), rng.uniform(-7.8, 7.8), 0])
            radius = rng.uniform(0.006, 0.022)
            star = Dot(point, radius=radius, color=color)
            star.set_opacity(rng.uniform(0.12, 0.55))
            stars.add(star)
        return stars

    @staticmethod
    def _make_corona(center: np.ndarray, radius: float, colors: dict) -> VGroup:
        corona = VGroup()
        for scale, opacity, width in [
            (1.19, 0.035, 20),
            (1.12, 0.06, 14),
            (1.065, 0.13, 8),
            (1.025, 0.36, 3),
        ]:
            ring = Circle(
                radius=radius * scale,
                color=colors["surface"],
                stroke_width=width,
            ).move_to(center)
            ring.set_opacity(opacity)
            corona.add(ring)
        return corona

    @staticmethod
    def _make_sun(center: np.ndarray, radius: float, colors: dict) -> VGroup:
        gradient = color_gradient(
            [colors["surface"], colors["mid_sun"], colors["deep_sun"]], 42
        )
        layers = VGroup()
        for index, color in enumerate(gradient):
            fraction = 1.0 - 0.96 * index / (len(gradient) - 1)
            disk = Circle(radius=radius * fraction, stroke_width=0)
            disk.set_fill(color, opacity=1.0)
            disk.move_to(center)
            layers.add(disk)
        core_glow = Circle(radius=radius * 0.2, stroke_width=0)
        core_glow.set_fill(colors["core"], opacity=0.76).move_to(center)
        layers.add(core_glow)
        return layers

    @staticmethod
    def _make_zone_overlays(
        center: np.ndarray, radius: float, colors: dict
    ) -> VGroup:
        # Stack filled disks from outside inward.  This gives clean radial
        # bands in Cairo; a filled Annulus leaves a small seam at its cut.
        convective = Circle(
            radius=radius * 0.995,
            stroke_width=0,
            fill_color=colors["convective_zone"],
            fill_opacity=0.20,
        ).move_to(center)
        radiative = Circle(
            radius=radius * BCZ_RADIUS_FRACTION,
            stroke_width=0,
            fill_color=colors["radiative_zone"],
            fill_opacity=0.27,
        ).move_to(center)
        core = Circle(
            radius=radius * 0.2,
            stroke_width=0,
            fill_color=colors["core"],
            fill_opacity=0.82,
        ).move_to(center)
        return VGroup(convective, radiative, core)

    @staticmethod
    def _make_convection_texture(
        center: np.ndarray, radius: float, colors: dict
    ) -> VGroup:
        rng = np.random.default_rng(71307)
        arcs = VGroup()
        for ring_fraction in np.linspace(0.76, 0.97, 5):
            count = int(18 + 18 * ring_fraction)
            for index in range(count):
                angle = TAU * (index / count) + rng.uniform(-0.035, 0.035)
                arc = Arc(
                    radius=radius * ring_fraction,
                    start_angle=angle,
                    angle=rng.uniform(0.055, 0.13),
                    stroke_color=colors["convective_zone"],
                    stroke_width=rng.uniform(0.9, 2.0),
                ).move_to(center)
                arc.set_opacity(rng.uniform(0.20, 0.48))
                arcs.add(arc)
        return arcs

    @staticmethod
    def _layer_tag(
        label: str,
        text_point: np.ndarray,
        anchor: np.ndarray,
        color: str,
        colors: dict,
        align: str = "left",
    ) -> VGroup:
        text = Text(
            label,
            font="Arial",
            font_size=17,
            weight=BOLD,
            color=colors["white"],
        )
        if text.width > 2.15:
            text.scale_to_fit_width(2.15)
        text.move_to(text_point)
        badge = SurroundingRectangle(
            text,
            buff=0.10,
            corner_radius=0.10,
            stroke_color=color,
            stroke_width=1.1,
            fill_color=colors["panel"],
            fill_opacity=0.90,
        )
        edge = badge.get_right() if align == "left" else badge.get_left()
        line = Line(edge, anchor, color=color, stroke_width=1.25).set_opacity(0.7)
        dot = Dot(anchor, radius=0.035, color=color)
        return VGroup(line, dot, badge, text)

    @staticmethod
    def _make_magnifier(colors: dict) -> VGroup:
        center = np.array([0.0, -4.4, 0.0])
        rings = VGroup(
            Circle(radius=1.48, color=colors["cyan"], stroke_width=2.2),
            Circle(radius=1.39, color=colors["grid"], stroke_width=1.0),
        ).move_to(center)
        fill = Circle(radius=1.43, stroke_width=0)
        fill.set_fill(colors["panel"], opacity=0.92).move_to(center)
        particles = VGroup()
        rng = np.random.default_rng(51042)
        for _ in range(48):
            rad = 1.22 * np.sqrt(rng.random())
            phi = rng.uniform(0, TAU)
            point = center + np.array([rad * np.cos(phi), rad * np.sin(phi), 0])
            particle = Dot(
                point,
                radius=rng.uniform(0.018, 0.04),
                color=colors["radiative"],
            ).set_opacity(rng.uniform(0.22, 0.68))
            particles.add(particle)
        caption = Text(
            COPY["micro_scale"],
            font="Arial",
            font_size=18,
            color=colors["muted"],
        ).move_to(center + np.array([0, 1.72, 0]))
        return VGroup(fill, particles, rings, caption)

    @staticmethod
    def _make_model_chain(
        colors: dict,
    ) -> tuple[VMobject, Text, VGroup, VGroup]:
        panel = RoundedRectangle(
            width=8.25,
            height=2.85,
            corner_radius=0.20,
            stroke_color=colors["grid"],
            stroke_width=1.5,
            fill_color=colors["panel"],
            fill_opacity=0.95,
        ).move_to([0.0, -4.62, 0.0])
        title = Text(
            COPY["model_title"],
            font="Arial",
            font_size=20,
            weight=BOLD,
            color=colors["white"],
        ).move_to([0.0, -3.52, 0.0])
        if title.width > 7.4:
            title.scale_to_fit_width(7.4)

        formulas = [
            r"I_\nu(\mathbf r,\hat{\mathbf n},t)",
            r"S_\nu=B_\nu[T(r)]",
            r"\kappa_R[T,\rho,X_i]",
            r"D(r)={c\over3\rho\kappa_R}",
        ]
        x_positions = [-3.05, -1.05, 1.05, 3.08]
        steps = VGroup()
        for x, formula, caption_text in zip(
            x_positions, formulas, COPY["model_steps"]
        ):
            formula_mob = MathTex(
                formula,
                font_size=27,
                color=colors["white"],
            ).move_to([x, -4.54, 0.0])
            if formula_mob.width > 1.62:
                formula_mob.scale_to_fit_width(1.62)
            caption = Text(
                caption_text,
                font="Arial",
                font_size=12,
                color=colors["muted"],
            ).move_to([x, -5.33, 0.0])
            if caption.width > 1.72:
                caption.scale_to_fit_width(1.72)
            steps.add(VGroup(formula_mob, caption))

        arrows = VGroup()
        for left, right in zip(steps[:-1], steps[1:]):
            arrow = Arrow(
                left.get_right() + 0.08 * RIGHT,
                right.get_left() + 0.08 * LEFT,
                buff=0.06,
                stroke_width=1.6,
                max_tip_length_to_length_ratio=0.20,
                color=colors["cyan"],
            )
            arrows.add(arrow)
        return panel, title, steps, arrows

    @staticmethod
    def _make_micro_walk(center: np.ndarray, colors: dict) -> VMobject:
        rng = np.random.default_rng(137)
        point = np.array(center, dtype=float)
        points = [point.copy()]
        for _ in range(150):
            angle = rng.uniform(0, TAU)
            candidate = point + 0.115 * np.array([np.cos(angle), np.sin(angle), 0])
            offset = candidate - center
            if np.linalg.norm(offset[:2]) > 1.24:
                normal = offset / np.linalg.norm(offset)
                direction = candidate - point
                direction = direction - 2.0 * np.dot(direction, normal) * normal
                candidate = point + direction
            point = candidate
            points.append(point.copy())
        path = VMobject(stroke_color=colors["cyan"], stroke_width=2.0)
        path.set_points_as_corners(points)
        path.set_stroke(opacity=0.85)
        path.set_fill(opacity=0)
        return path

    @staticmethod
    def _glowing_dot(point: np.ndarray, color: str, radius: float) -> VGroup:
        return VGroup(
            Dot(point, radius=radius * 2.8, color=color).set_opacity(0.12),
            Dot(point, radius=radius * 1.7, color=color).set_opacity(0.3),
            Dot(point, radius=radius, color=WHITE),
        )

    @staticmethod
    def _make_equation_panel(colors: dict) -> tuple[VMobject, MathTex]:
        panel = RoundedRectangle(
            width=8.15,
            height=2.05,
            corner_radius=0.18,
            stroke_color=colors["grid"],
            stroke_width=1.5,
            fill_color=colors["panel"],
            fill_opacity=0.92,
        ).move_to([0, -4.55, 0])
        equation = MathTex(
            r"D(r)={c\over3\kappa_R\rho}",
            r"\qquad",
            r"\langle t\rangle={1\over c}\int_0^R r\,\kappa_R\rho\,dr",
            font_size=32,
            color=colors["white"],
        ).move_to([0, -4.40, 0])
        equation[0].set_color(colors["cyan"])
        caption = Text(
            COPY["equation_caption"],
            font="Arial",
            font_size=18,
            color=colors["muted"],
        ).move_to([0, -5.14, 0])
        panel.add(caption)
        return panel, equation

    @staticmethod
    def _mean_free_path_chart_point(
        radius_fraction: float,
        mean_free_path_cm: float,
    ) -> np.ndarray:
        x_left, x_right = -3.57, -0.27
        y_bottom, y_top = -2.38, -1.06
        log_min, log_max = np.log10(0.0025), 0.0
        radius = np.clip(radius_fraction, 0.0, 0.99)
        log_path = np.clip(np.log10(mean_free_path_cm), log_min, log_max)
        x = x_left + (x_right - x_left) * radius / 0.99
        y = y_bottom + (y_top - y_bottom) * (log_path - log_min) / (
            log_max - log_min
        )
        return np.array([x, y, 0.0])

    @classmethod
    def _make_mean_free_path_chart(cls, profile, colors: dict) -> VGroup:
        panel = RoundedRectangle(
            width=4.08,
            height=2.02,
            corner_radius=0.16,
            stroke_color=colors["grid"],
            stroke_width=1.25,
            fill_color=colors["panel"],
            fill_opacity=0.92,
        ).move_to([-2.06, -1.75, 0.0])
        title = Text(
            COPY["mfp_title"],
            font="Arial",
            font_size=15,
            color=colors["white"],
        ).move_to([-2.05, -0.90, 0.0])

        x_left, x_right = -3.57, -0.27
        y_bottom, y_top = -2.38, -1.06
        axes = VGroup(
            Line([x_left, y_bottom, 0], [x_right, y_bottom, 0]),
            Line([x_left, y_bottom, 0], [x_left, y_top, 0]),
        ).set_stroke(colors["muted"], width=0.85, opacity=0.72)

        grid = VGroup()
        tick_labels = VGroup()
        for value, label in [(0.01, "10⁻²"), (0.1, "10⁻¹"), (1.0, "1")]:
            y = cls._mean_free_path_chart_point(0.0, value)[1]
            grid.add(
                Line([x_left, y, 0], [x_right, y, 0]).set_stroke(
                    colors["grid"], width=0.65, opacity=0.52
                )
            )
            tick_labels.add(
                Text(
                    label,
                    font="Arial",
                    font_size=9,
                    color=colors["muted"],
                ).move_to([x_left - 0.22, y, 0])
            )
        tick_labels.add(
            Text(COPY["cm"], font="Arial", font_size=9, color=colors["muted"]).move_to(
                [x_left - 0.22, y_top + 0.13, 0]
            )
        )

        for radius, label in [
            (0.0, "0"),
            (0.5, COPY["half_radius"]),
            (0.99, "1"),
        ]:
            x = cls._mean_free_path_chart_point(radius, 0.0025)[0]
            axes.add(
                Line([x, y_bottom, 0], [x, y_bottom - 0.045, 0]).set_stroke(
                    colors["muted"], width=0.8, opacity=0.72
                )
            )
            tick_labels.add(
                Text(
                    label,
                    font="Arial",
                    font_size=9,
                    color=colors["muted"],
                ).move_to([x, y_bottom - 0.14, 0])
            )
        tick_labels.add(
            Text(
                "r/R☉",
                font="Arial",
                font_size=9,
                color=colors["muted"],
            ).move_to([(x_left + x_right) / 2.0, y_bottom - 0.29, 0])
        )

        visible = profile.radius_fraction <= 0.99
        radii = np.r_[0.0, profile.radius_fraction[visible]]
        paths = np.r_[profile.mean_free_path_cm[0], profile.mean_free_path_cm[visible]]
        curve = VMobject()
        curve.set_points_as_corners(
            [cls._mean_free_path_chart_point(radius, path) for radius, path in zip(radii, paths)]
        )
        curve.set_stroke(colors["cyan"], width=2.15, opacity=0.96)
        curve.set_fill(opacity=0)

        bcz_x = cls._mean_free_path_chart_point(
            BCZ_RADIUS_FRACTION,
            float(
                np.interp(
                    BCZ_RADIUS_FRACTION,
                    profile.radius_fraction,
                    profile.mean_free_path_cm,
                )
            ),
        )[0]
        bcz_line = DashedLine(
            [bcz_x, y_bottom, 0],
            [bcz_x, y_top, 0],
            dash_length=0.035,
            color=colors["surface"],
            stroke_width=1.0,
        ).set_opacity(0.68)
        bcz_label = Text(
            COPY["bcz_tick"],
            font="Arial",
            font_size=9,
            color=colors["surface"],
        ).move_to([bcz_x, y_bottom - 0.14, 0])

        return VGroup(panel, grid, axes, tick_labels, bcz_line, curve, bcz_label, title)

    @classmethod
    def _make_mean_free_path_cursor(
        cls,
        profile,
        radius_fraction: float,
        colors: dict,
        accent=None,
    ) -> VGroup:
        mean_free_path = float(
            np.interp(
                radius_fraction,
                np.r_[0.0, profile.radius_fraction],
                np.r_[profile.mean_free_path_cm[0], profile.mean_free_path_cm],
            )
        )
        point = cls._mean_free_path_chart_point(radius_fraction, mean_free_path)
        guide = DashedLine(
            [point[0], -2.38, 0],
            point,
            dash_length=0.025,
            color=colors["white"],
            stroke_width=0.8,
        ).set_opacity(0.42)
        marker_color = accent or colors["cyan"]
        halo = Dot(point, radius=0.105, color=marker_color).set_opacity(0.22)
        marker = Dot(point, radius=0.045, color=colors["white"])
        return VGroup(guide, halo, marker)

    @staticmethod
    def _temperature_fraction(profile, radius_fraction: float) -> float:
        temperature = float(
            np.interp(
                radius_fraction,
                np.r_[0.0, profile.radius_fraction],
                np.r_[profile.temperature_k[0], profile.temperature_k],
            )
        )
        log_min = np.log10(profile.temperature_k[-1])
        log_max = np.log10(profile.temperature_k[0])
        return float(
            np.clip(
                (np.log10(temperature) - log_min) / (log_max - log_min),
                0.0,
                1.0,
            )
        )

    @classmethod
    def _temperature_color(
        cls,
        profile,
        radius_fraction: float,
        colors: dict,
    ):
        palette = color_gradient(
            [
                colors["convective"],
                colors["radiative"],
                colors["surface"],
                colors["core"],
            ],
            101,
        )
        index = int(round(100 * cls._temperature_fraction(profile, radius_fraction)))
        return palette[index]

    @classmethod
    def _make_temperature_legend(cls, profile, colors: dict) -> VGroup:
        panel = RoundedRectangle(
            width=1.22,
            height=4.48,
            corner_radius=0.16,
            stroke_color=colors["grid"],
            stroke_width=1.15,
            fill_color=colors["panel"],
            fill_opacity=0.86,
        ).move_to([3.76, 2.50, 0.0])

        x_bar = 3.52
        y_bottom, y_top = 0.82, 4.10
        segments = VGroup()
        palette = color_gradient(
            [
                colors["convective"],
                colors["radiative"],
                colors["surface"],
                colors["core"],
            ],
            42,
        )
        segment_height = (y_top - y_bottom) / len(palette)
        for index, color in enumerate(palette):
            segment = Rectangle(
                width=0.15,
                height=segment_height * 1.04,
                stroke_width=0,
                fill_color=color,
                fill_opacity=1.0,
            ).move_to(
                [x_bar, y_bottom + (index + 0.5) * segment_height, 0.0]
            )
            segments.add(segment)

        title = Text(
            "T(r)",
            font="Arial",
            font_size=14,
            weight=BOLD,
            color=colors["white"],
        ).move_to([3.75, 4.40, 0.0])

        labels = VGroup()
        for radius, text_value in zip(
            [0.0, BCZ_RADIUS_FRACTION, 1.0],
            COPY["temperature_ticks"],
        ):
            y = y_bottom + (y_top - y_bottom) * cls._temperature_fraction(
                profile, radius
            )
            tick = Line(
                [x_bar - 0.11, y, 0.0],
                [x_bar + 0.11, y, 0.0],
                color=colors["white"],
                stroke_width=0.8,
            ).set_opacity(0.72)
            label = Text(
                text_value,
                font="Arial",
                font_size=9,
                color=colors["white"] if radius == 0.0 else colors["muted"],
            ).move_to([3.96, y, 0.0])
            labels.add(tick, label)

        note = Text(
            COPY["temperature_note"],
            font="Arial",
            font_size=8,
            line_spacing=0.78,
            color=colors["muted"],
            should_center=True,
        ).move_to([3.76, 0.47, 0.0])

        legend = VGroup(panel, segments, labels, title, note)
        legend.set_z_index(70)
        return legend

    @classmethod
    def _make_temperature_cursor(
        cls,
        profile,
        radius_fraction: float,
        colors: dict,
    ) -> VGroup:
        x_bar = 3.52
        y_bottom, y_top = 0.82, 4.10
        y = y_bottom + (y_top - y_bottom) * cls._temperature_fraction(
            profile, radius_fraction
        )
        color = cls._temperature_color(profile, radius_fraction, colors)
        glow = Dot([x_bar, y, 0.0], radius=0.105, color=color).set_opacity(0.34)
        marker = Line(
            [x_bar - 0.17, y, 0.0],
            [x_bar + 0.17, y, 0.0],
            color=colors["white"],
            stroke_width=2.2,
        )
        result = VGroup(glow, marker)
        result.set_z_index(80)
        return result

    @staticmethod
    def _make_diffusion_halo(
        center: np.ndarray,
        sun_radius: float,
        radius_fraction: float,
        colors: dict,
        front_color=None,
    ) -> VGroup:
        radius = max(0.025, sun_radius * radius_fraction)
        accent = front_color or colors["cyan"]
        halo = VGroup()
        energy_cloud = Circle(
            radius=radius,
            stroke_width=0,
            fill_color=accent,
            fill_opacity=0.035,
        ).move_to(center)
        halo.add(energy_cloud)
        for scale, opacity, width in [
            (0.34, 0.13, 1.0),
            (0.56, 0.18, 1.1),
            (0.78, 0.24, 1.2),
        ]:
            ripple = Circle(
                radius=radius * scale,
                stroke_color=accent,
                stroke_width=width,
                fill_opacity=0,
            ).set_stroke(opacity=opacity).move_to(center)
            halo.add(ripple)
        ring = Circle(
            radius=radius,
            color=accent,
            stroke_width=2.8,
        ).set_fill(opacity=0).set_stroke(opacity=0.92).move_to(center)
        outer = Circle(
            radius=radius * 1.018,
            color=colors["white"],
            stroke_width=5.5,
        ).set_fill(opacity=0).set_stroke(opacity=0.15).move_to(center)
        halo.add(outer, ring)
        return halo

    @staticmethod
    def _make_radius_readout(
        radius_fraction: float,
        colors: dict,
        position: np.ndarray | None = None,
        accent=None,
    ) -> VGroup:
        readout_color = accent or colors["cyan"]
        label = Text(
            f"r = {local_decimal(radius_fraction, 3)} R☉",
            font="Arial",
            font_size=22,
            color=readout_color,
        )
        pill = RoundedRectangle(
            width=2.72,
            height=0.55,
            corner_radius=0.22,
            stroke_color=readout_color,
            stroke_width=1.0,
            fill_color=colors["panel"],
            fill_opacity=0.88,
        )
        label.move_to(pill)
        if position is None:
            position = np.array([0.0, -3.52, 0.0])
        return VGroup(pill, label).move_to(position)

    @staticmethod
    def _result_badge(
        label_text: str,
        value_text: str,
        accent: str,
        position: np.ndarray,
        colors: dict,
    ) -> VGroup:
        panel = RoundedRectangle(
            width=8.0,
            height=1.25,
            corner_radius=0.2,
            stroke_color=accent,
            stroke_width=1.6,
            fill_color=colors["panel"],
            fill_opacity=0.95,
        )
        accent_bar = RoundedRectangle(
            width=0.08,
            height=0.76,
            corner_radius=0.035,
            stroke_width=0,
            fill_color=accent,
            fill_opacity=1.0,
        ).move_to([-3.61, 0, 0])
        label = Text(
            label_text,
            font="Arial",
            font_size=20,
            color=colors["muted"],
        )
        label.align_to(panel, LEFT).shift(0.62 * RIGHT + 0.23 * UP)
        value = Text(
            value_text,
            font="Arial",
            font_size=35,
            weight=BOLD,
            color=accent,
        )
        value.align_to(panel, RIGHT).shift(0.55 * LEFT + 0.08 * DOWN)
        return VGroup(panel, accent_bar, label, value).move_to(position)

    @staticmethod
    def _earth_card(colors: dict, position: np.ndarray) -> VGroup:
        sun_dot = Dot([-2.9, 0, 0], radius=0.18, color=colors["surface"])
        earth = Circle(
            radius=0.17,
            stroke_color=colors["cyan"],
            stroke_width=1.5,
            fill_color=colors["blue"],
            fill_opacity=1.0,
        ).move_to([2.92, 0, 0])
        beam = Line(
            sun_dot.get_right(),
            earth.get_left(),
            color=colors["core"],
            stroke_width=2.0,
        )
        beam_glow = Line(
            sun_dot.get_right(),
            earth.get_left(),
            color=colors["surface"],
            stroke_width=7.0,
        ).set_opacity(0.14)
        text = Text(
            COPY["earth"],
            font="Arial",
            font_size=23,
            color=colors["white"],
        ).move_to([0, -0.5, 0])
        return VGroup(beam_glow, beam, sun_dot, earth, text).move_to(position)


class SolarPhotonDiffusionStill(SolarPhotonDiffusion):
    """A compact final-state still, useful for covers and quick checks."""

    def construct(self) -> None:
        self.camera.background_color = CFG["manim"]["background_color"]
        colors = CFG["colors"]
        profile = load_profile()
        centre_years = mean_exit_time_seconds(profile) / SECONDS_PER_YEAR
        weighted_years = production_weighted_exit_time_seconds(profile) / SECONDS_PER_YEAR

        center = np.array([0.0, 2.25, 0.0])
        radius = 3.05
        self.add(
            self._make_stars(colors["muted"]),
            self._make_corona(center, radius, colors),
            self._make_sun(center, radius, colors),
            self._make_convection_texture(center, radius, colors),
        )
        title = Text(
            COPY["still_title"],
            font="Arial",
            font_size=39,
            weight=BOLD,
            color=colors["white"],
        ).move_to([0, 7.15, 0])
        subtitle = Text(
            COPY["still_subtitle"],
            font="Arial",
            font_size=23,
            color=colors["muted"],
        ).move_to([0, 6.62, 0])
        central_card = self._result_badge(
            COPY["central_card"],
            COPY["thousand_years"].format(
                value=local_decimal(centre_years / 1000)
            ),
            colors["surface"],
            np.array([0.0, -2.05, 0.0]),
            colors,
        )
        weighted_card = self._result_badge(
            COPY["still_weighted"],
            COPY["thousand_years"].format(
                value=local_decimal(weighted_years / 1000)
            ),
            colors["cyan"],
            np.array([0.0, -3.58, 0.0]),
            colors,
        )
        equation_panel, equation = self._make_equation_panel(colors)
        equation_panel.move_to([0, -5.55, 0])
        equation.move_to([0, -5.4, 0])
        self.add(title, subtitle, central_card, weighted_card, equation_panel, equation)
        self.wait(0.1)
