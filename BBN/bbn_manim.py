# -*- coding: utf-8 -*-
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from manim import *


N_PROTONS = 18
N_NEUTRONS = 3
N_PHOTONS = 12
SEED = 7


config.pixel_width = 1920
config.pixel_height = 1080
config.frame_width = 14.222
config.frame_height = 8.0
config.frame_rate = 30


BG_COLOR = "#07111F"
STAR_COLOR = "#DCEAFF"
TEXT_COLOR = "#EAF2FF"
MUTED_TEXT = "#9DB0C8"
PROTON_COLOR = "#D96A63"
NEUTRON_COLOR = "#9FA9B8"
PHOTON_COLOR = "#F3D46B"
PHOTON_DIM = "#BDA44F"
DEUTERIUM_COLOR = "#86D9FF"
HE3_COLOR = "#9FE7C6"
HE4_COLOR = "#C6E6FF"
HOT_COLOR = "#EAA25C"
WINDOW_COLOR = "#8BE4B5"
COOL_COLOR = "#80B7FF"
TRACE_COLOR = "#B8A6FF"

NUCLEON_R = 0.165
PHOTON_R = 0.36
PLAY_BOUNDS = (-6.45, 4.55, -3.05, 2.55)


@dataclass
class MotionSpec:
    velocity: np.ndarray
    radius: float


def fit_to_width(mobject: Mobject, max_width: float) -> Mobject:
    if mobject.width > max_width:
        mobject.scale_to_fit_width(max_width)
    return mobject


def unit(vector: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    if norm < 1e-9:
        return RIGHT.copy()
    return vector / norm


def angle_of(vector: np.ndarray) -> float:
    direction = unit(vector)
    return math.atan2(float(direction[1]), float(direction[0]))


def random_positions(
    rng: np.random.Generator,
    count: int,
    bounds: tuple[float, float, float, float],
    min_dist: float = 0.44,
) -> list[np.ndarray]:
    xmin, xmax, ymin, ymax = bounds
    points: list[np.ndarray] = []
    attempts = 0
    while len(points) < count and attempts < 5000:
        attempts += 1
        point = np.array(
            [
                rng.uniform(xmin, xmax),
                rng.uniform(ymin, ymax),
                0.0,
            ]
        )
        if all(np.linalg.norm(point - old) >= min_dist for old in points):
            points.append(point)
    while len(points) < count:
        points.append(
            np.array(
                [
                    rng.uniform(xmin, xmax),
                    rng.uniform(ymin, ymax),
                    0.0,
                ]
            )
        )
    return points


def star_field(rng: np.random.Generator, count: int = 95) -> VGroup:
    stars = VGroup()
    for _ in range(count):
        x = rng.uniform(-7.0, 7.0)
        y = rng.uniform(-3.85, 3.85)
        radius = rng.uniform(0.006, 0.018)
        opacity = rng.uniform(0.10, 0.34)
        star = Dot([x, y, 0.0], radius=radius)
        star.set_fill(STAR_COLOR, opacity=opacity)
        star.set_stroke(width=0)
        stars.add(star)
    return stars


def label_text(text: str, font_size: int, color: str = TEXT_COLOR) -> Text:
    return Text(text, font_size=font_size, color=color)


def nucleon_core(kind: str, radius: float = NUCLEON_R, font_size: int = 21) -> VGroup:
    color = PROTON_COLOR if kind == "p" else NEUTRON_COLOR
    glow = Circle(radius=radius * 1.45)
    glow.set_fill(color, opacity=0.16)
    glow.set_stroke(width=0)

    body = Circle(radius=radius)
    body.set_fill(color, opacity=0.96)
    body.set_stroke(WHITE, width=1.2, opacity=0.62)

    letter = Text(kind, font_size=font_size, color=WHITE, weight=BOLD)
    letter.move_to(body)

    particle = VGroup(glow, body, letter)
    particle.kind = kind
    return particle


def particle(kind: str, radius: float = NUCLEON_R) -> VGroup:
    body = nucleon_core(kind, radius=radius, font_size=22)
    body.kind = kind
    body.set_z_index(5)
    return body


def photon(angle: float = 0.0, ray_count: int = 5) -> VGroup:
    ray_count = 5 if ray_count >= 5 else 3
    offsets = np.linspace(-0.16, 0.16, ray_count)
    lengths = np.linspace(0.48, 0.78, ray_count)
    lengths = np.minimum(lengths, lengths[::-1])

    ray_glows = VGroup()
    rays = VGroup()
    for offset, length in zip(offsets, lengths):
        start = np.array([-length, offset, 0.0])
        end = np.array([-0.10, offset * 0.36, 0.0])
        glow_ray = Line(start, end)
        glow_ray.set_stroke(PHOTON_COLOR, width=7.0, opacity=0.10)
        ray = Line(start, end)
        ray.set_stroke(PHOTON_COLOR, width=2.4, opacity=0.88)
        ray_glows.add(glow_ray)
        rays.add(ray)

    head_glow = Circle(radius=0.21).move_to([0.05, 0.0, 0.0])
    head_glow.set_fill(PHOTON_COLOR, opacity=0.15)
    head_glow.set_stroke(width=0)

    head = Circle(radius=0.105).move_to([0.05, 0.0, 0.0])
    head.set_fill(PHOTON_COLOR, opacity=0.96)
    head.set_stroke(WHITE, width=0.9, opacity=0.45)

    gamma = Text("γ", font_size=13, color=BG_COLOR, weight=BOLD)
    gamma.move_to(head)

    mob = VGroup(head_glow, ray_glows, rays, head, gamma)
    mob.kind = "gamma"
    mob.photon_angle = 0.0
    mob.photon_head = head
    mob.photon_rays = rays
    orient_photon(mob, angle)
    mob.set_z_index(4)
    return mob


def photon_head_center(mob: Mobject) -> np.ndarray:
    head = getattr(mob, "photon_head", None)
    if head is None:
        return mob.get_center()
    return head.get_center()


def place_photon_head_at(mob: Mobject, point: np.ndarray) -> Mobject:
    mob.shift(point - photon_head_center(mob))
    return mob


def photon_tail_center(mob: Mobject) -> np.ndarray:
    rays = getattr(mob, "photon_rays", None)
    if rays is None or len(rays) == 0:
        return mob.get_center()
    return np.mean([ray.get_center() for ray in rays], axis=0)


def orient_photon(mob: Mobject, angle: float) -> None:
    current = float(getattr(mob, "photon_angle", 0.0))
    head = photon_head_center(mob)
    mob.rotate(angle - current, about_point=head)
    mob.photon_angle = angle
    direction = np.array([math.cos(angle), math.sin(angle), 0.0])
    if np.dot(photon_head_center(mob) - photon_tail_center(mob), direction) < 0:
        mob.rotate(PI, about_point=photon_head_center(mob))


def reaction_flash(point: np.ndarray, color: str = WHITE) -> VGroup:
    ring = Circle(radius=0.22).move_to(point)
    ring.set_fill(opacity=0)
    ring.set_stroke(color, width=2.4, opacity=0.72)
    halo = Circle(radius=0.34).move_to(point)
    halo.set_fill(color, opacity=0.10)
    halo.set_stroke(width=0)
    return VGroup(halo, ring)


def deuterium() -> VGroup:
    p = nucleon_core("p", radius=0.18, font_size=19).shift(LEFT * 0.18)
    n = nucleon_core("n", radius=0.18, font_size=19).shift(RIGHT * 0.18)
    bond = Line(p.get_center(), n.get_center())
    bond.set_stroke(DEUTERIUM_COLOR, width=3.0, opacity=0.7)
    label = Text("D", font_size=25, color=DEUTERIUM_COLOR, weight=BOLD)
    label.next_to(VGroup(p, n), UP, buff=0.08)

    halo = RoundedRectangle(width=0.92, height=0.58, corner_radius=0.16)
    halo.set_fill(DEUTERIUM_COLOR, opacity=0.09)
    halo.set_stroke(DEUTERIUM_COLOR, width=1.5, opacity=0.35)
    halo.move_to(VGroup(p, n))
    mob = VGroup(halo, bond, p, n, label)
    mob.kind = "D"
    mob.set_z_index(7)
    return mob


def he3() -> VGroup:
    p1 = nucleon_core("p", radius=0.16, font_size=17).shift(LEFT * 0.20 + DOWN * 0.10)
    p2 = nucleon_core("p", radius=0.16, font_size=17).shift(RIGHT * 0.20 + DOWN * 0.10)
    n = nucleon_core("n", radius=0.16, font_size=17).shift(UP * 0.18)
    core = VGroup(p1, p2, n)

    halo = Circle(radius=0.47)
    halo.set_fill(HE3_COLOR, opacity=0.10)
    halo.set_stroke(HE3_COLOR, width=1.5, opacity=0.38)
    halo.move_to(core)

    label = Text("He-3", font_size=22, color=HE3_COLOR, weight=BOLD)
    label.next_to(core, UP, buff=0.08)
    mob = VGroup(halo, core, label)
    mob.kind = "He3"
    mob.set_z_index(7)
    return mob


def he4(scale: float = 1.0) -> VGroup:
    offsets = [
        LEFT * 0.22 + UP * 0.16,
        RIGHT * 0.22 + DOWN * 0.16,
        RIGHT * 0.22 + UP * 0.16,
        LEFT * 0.22 + DOWN * 0.16,
    ]
    kinds = ["p", "p", "n", "n"]
    nucleons = VGroup(
        *[
            nucleon_core(kind, radius=0.17, font_size=17).shift(offset)
            for kind, offset in zip(kinds, offsets)
        ]
    )

    halo = Circle(radius=0.57)
    halo.set_fill(HE4_COLOR, opacity=0.12)
    halo.set_stroke(HE4_COLOR, width=2.1, opacity=0.44)
    halo.move_to(nucleons)

    label = Text("He-4", font_size=24, color=HE4_COLOR, weight=BOLD)
    label.next_to(nucleons, DOWN, buff=0.10)
    mob = VGroup(halo, nucleons, label)
    mob.scale(scale)
    mob.kind = "He4"
    mob.set_z_index(7)
    return mob


def make_temperature_gauge(temp: ValueTracker) -> tuple[VGroup, Mobject, Mobject, Mobject]:
    center = np.array([5.55, -0.15, 0.0])
    height = 4.85

    track = RoundedRectangle(width=0.42, height=height, corner_radius=0.17)
    track.set_fill("#101D30", opacity=0.62)
    track.set_stroke("#8BA2C4", width=1.2, opacity=0.46)
    track.move_to(center)

    bottom = track.get_bottom() + UP * 0.13
    usable_height = height - 0.28

    def current_height() -> float:
        return max(0.10, usable_height * temp.get_value())

    def color_for_temp() -> str:
        t = temp.get_value()
        if t > 0.62:
            return HOT_COLOR
        if t > 0.32:
            return WINDOW_COLOR
        return COOL_COLOR

    fill = always_redraw(
        lambda: Rectangle(width=0.25, height=current_height())
        .set_fill(color_for_temp(), opacity=0.84)
        .set_stroke(width=0)
        .move_to(bottom + UP * current_height() / 2)
    )

    marker = always_redraw(
        lambda: Triangle()
        .scale(0.12)
        .rotate(-PI / 2)
        .set_fill(color_for_temp(), opacity=0.95)
        .set_stroke(width=0)
        .move_to(bottom + UP * current_height() + LEFT * 0.42)
    )

    title = Text("T", font_size=30, color=TEXT_COLOR, weight=BOLD)
    title.next_to(track, UP, buff=0.20)
    high = Text("высокая", font_size=18, color=HOT_COLOR)
    high.next_to(track, RIGHT, buff=0.22).align_to(track, UP).shift(DOWN * 0.15)
    low = Text("ниже", font_size=18, color=COOL_COLOR)
    low.next_to(track, RIGHT, buff=0.22).align_to(track, DOWN).shift(UP * 0.15)

    window = RoundedRectangle(width=0.72, height=1.12, corner_radius=0.12)
    window.set_fill(WINDOW_COLOR, opacity=0.06)
    window.set_stroke(WINDOW_COLOR, width=1.6, opacity=0.72)
    window.move_to(bottom + UP * (usable_height * 0.40))
    window_label = Text("окно BBN", font_size=18, color=WINDOW_COLOR)
    window_label.next_to(window, LEFT, buff=0.16)
    bbn_window = VGroup(window, window_label)
    bbn_window.set_opacity(0)

    gauge = VGroup(track, fill, marker, title, high, low)
    return gauge, bbn_window, track, fill


def composition_panel() -> VGroup:
    title = Text("итоговый состав", font_size=25, color=TEXT_COLOR, weight=BOLD)

    bar_width = 4.15
    h_bar_bg = RoundedRectangle(width=bar_width, height=0.28, corner_radius=0.05)
    h_bar_bg.set_fill("#122036", opacity=0.85)
    h_bar_bg.set_stroke("#7184A0", width=0.6, opacity=0.35)
    h_bar = RoundedRectangle(width=bar_width * 0.75, height=0.28, corner_radius=0.05)
    h_bar.set_fill(PROTON_COLOR, opacity=0.92)
    h_bar.set_stroke(width=0)
    h_bar.align_to(h_bar_bg, LEFT).move_to(h_bar_bg.get_left() + RIGHT * h_bar.width / 2)
    h_label = Text("водород ≈ 75%", font_size=22, color=TEXT_COLOR)
    h_label.next_to(h_bar_bg, LEFT, buff=0.28)

    he_bar_bg = h_bar_bg.copy()
    he_bar = RoundedRectangle(width=bar_width * 0.25, height=0.28, corner_radius=0.05)
    he_bar.set_fill(HE4_COLOR, opacity=0.92)
    he_bar.set_stroke(width=0)
    he_bar.align_to(he_bar_bg, LEFT).move_to(he_bar_bg.get_left() + RIGHT * he_bar.width / 2)
    he_label = Text("гелий-4 ≈ 25%", font_size=22, color=TEXT_COLOR)
    he_label.next_to(he_bar_bg, LEFT, buff=0.28)

    h_row = VGroup(h_label, h_bar_bg, h_bar)
    he_row = VGroup(he_label, he_bar_bg, he_bar)
    rows = VGroup(h_row, he_row).arrange(DOWN, aligned_edge=RIGHT, buff=0.18)

    trace = Text("следы D, He-3, Li-7", font_size=20, color=TRACE_COLOR)
    panel = VGroup(title, rows, trace).arrange(DOWN, buff=0.18, aligned_edge=LEFT)
    panel.move_to(np.array([-0.25, -2.95, 0.0]))
    return panel


class BBNAnimation(Scene):
    def construct(self) -> None:
        self.rng = np.random.default_rng(SEED)
        self.motion: dict[int, MotionSpec] = {}
        self.camera.background_color = BG_COLOR

        temp = ValueTracker(0.96)
        gauge, bbn_window, temp_track, _ = make_temperature_gauge(temp)

        background = Rectangle(width=15.0, height=8.5)
        background.set_fill(BG_COLOR, opacity=1.0)
        background.set_stroke(width=0)

        stars = star_field(self.rng)
        title = Text(
            "Первые минуты: синтез лёгких ядер",
            font_size=36,
            color=TEXT_COLOR,
            weight=BOLD,
        )
        fit_to_width(title, 9.8)
        title.to_edge(UP, buff=0.30).shift(LEFT * 0.76)

        stage_label = Text("слишком горячо", font_size=23, color=HOT_COLOR, weight=BOLD)
        fit_to_width(stage_label, 2.70)
        stage_label.next_to(temp_track, DOWN, buff=0.24).shift(LEFT * 0.10)

        caption = Text(
            "показана малая часть фотонного фона",
            font_size=24,
            color=PHOTON_COLOR,
        )
        fit_to_width(caption, 8.7)
        caption.next_to(title, DOWN, buff=0.18).align_to(title, LEFT)

        watermark = Text("@NeutrinoHit", font_size=20, color=MUTED_TEXT)
        watermark.to_corner(DR, buff=0.20).set_opacity(0.75)

        self.add(background, stars, gauge, watermark)
        self.play(FadeIn(title, shift=DOWN * 0.16), FadeIn(stage_label), run_time=1.0)

        protons, neutrons, photons = self.create_initial_particles()
        all_particles = VGroup(*protons, *neutrons, *photons)
        self.play(
            LaggedStart(*[FadeIn(m, scale=0.88) for m in all_particles], lag_ratio=0.035),
            FadeIn(caption, shift=UP * 0.08),
            run_time=2.4,
        )
        self.wait(0.45)

        reaction_note = Text("·", font_size=25, color=TEXT_COLOR)
        reaction_note.next_to(caption, DOWN, buff=0.18).align_to(caption, LEFT)
        reaction_note.set_opacity(0)
        self.add(reaction_note)

        self.hot_deuterium_stage(protons, neutrons, photons, reaction_note)
        caption = self.replace_text(
            caption,
            "Вселенная остывает",
            color=COOL_COLOR,
            anchor=caption,
        )
        self.cooling_stage(temp, photons, caption, reaction_note)

        stage_label = self.replace_text(
            stage_label,
            "окно синтеза",
            color=WINDOW_COLOR,
            anchor=stage_label,
            font_size=23,
            max_width=2.70,
        )
        self.play(FadeIn(bbn_window), run_time=0.7)

        he4_main, trace_d = self.synthesis_window(protons, neutrons, photons, caption, reaction_note)

        stage_label = self.replace_text(
            stage_label,
            "реакции замирают",
            color=COOL_COLOR,
            anchor=stage_label,
            font_size=23,
            max_width=2.70,
        )
        self.final_composition(
            protons,
            neutrons,
            photons,
            he4_main,
            trace_d,
            temp,
            caption,
            reaction_note,
        )

    def create_initial_particles(self) -> tuple[list[VGroup], list[VGroup], list[VGroup]]:
        p_positions = random_positions(self.rng, N_PROTONS, (-6.0, 3.75, -2.55, 2.0))
        n_positions = random_positions(self.rng, N_NEUTRONS, (-5.6, 3.15, -2.20, 1.75))
        g_positions = random_positions(self.rng, N_PHOTONS, (-6.1, 3.95, -2.65, 2.18), 0.28)

        protons = []
        for pos in p_positions:
            p = particle("p")
            p.move_to(pos)
            speed = self.rng.uniform(0.12, 0.21)
            angle = self.rng.uniform(0.0, TAU)
            self.start_motion(p, speed * np.array([math.cos(angle), math.sin(angle), 0.0]), NUCLEON_R)
            protons.append(p)

        neutrons = []
        for pos in n_positions:
            n = particle("n")
            n.move_to(pos)
            speed = self.rng.uniform(0.10, 0.18)
            angle = self.rng.uniform(0.0, TAU)
            self.start_motion(n, speed * np.array([math.cos(angle), math.sin(angle), 0.0]), NUCLEON_R)
            neutrons.append(n)

        photons = []
        for pos in g_positions:
            angle = self.rng.uniform(0.0, TAU)
            g = photon(angle=angle)
            g.move_to(pos)
            speed = self.rng.uniform(0.34, 0.52)
            self.start_motion(g, speed * np.array([math.cos(angle), math.sin(angle), 0.0]), PHOTON_R)
            photons.append(g)

        return protons, neutrons, photons

    def start_motion(self, mob: Mobject, velocity: np.ndarray, radius: float) -> None:
        self.motion[id(mob)] = MotionSpec(velocity=np.array(velocity, dtype=float), radius=radius)
        self.resume_motion(mob)

    def pause_motion(self, *mobs: Mobject) -> None:
        for mob in mobs:
            mob.clear_updaters()

    def resume_motion(self, *mobs: Mobject) -> None:
        for mob in mobs:
            spec = self.motion.get(id(mob))
            if spec is None:
                continue
            mob.clear_updaters()
            mob.add_updater(self.bounce_updater(spec))

    def set_velocity(self, mob: Mobject, velocity: np.ndarray) -> None:
        spec = self.motion.get(id(mob))
        if spec is not None:
            spec.velocity = np.array(velocity, dtype=float)

    def scale_velocity(self, mob: Mobject, factor: float) -> None:
        spec = self.motion.get(id(mob))
        if spec is not None:
            spec.velocity *= factor

    def bounce_updater(self, spec: MotionSpec):
        xmin, xmax, ymin, ymax = PLAY_BOUNDS

        def update(mob: Mobject, dt: float) -> None:
            pos = mob.get_center()
            new = pos + spec.velocity * dt

            if new[0] < xmin + spec.radius:
                new[0] = xmin + spec.radius
                spec.velocity[0] = abs(spec.velocity[0])
            elif new[0] > xmax - spec.radius:
                new[0] = xmax - spec.radius
                spec.velocity[0] = -abs(spec.velocity[0])

            if new[1] < ymin + spec.radius:
                new[1] = ymin + spec.radius
                spec.velocity[1] = abs(spec.velocity[1])
            elif new[1] > ymax - spec.radius:
                new[1] = ymax - spec.radius
                spec.velocity[1] = -abs(spec.velocity[1])

            mob.shift(new - pos)
            if getattr(mob, "kind", None) == "gamma":
                orient_photon(mob, angle_of(spec.velocity))

        return update

    def replace_text(
        self,
        old: Mobject,
        text: str,
        color: str = TEXT_COLOR,
        anchor: Mobject | None = None,
        font_size: int | None = None,
        max_width: float = 8.7,
        run_time: float = 0.55,
    ) -> Text:
        new = Text(text, font_size=font_size or 24, color=color)
        fit_to_width(new, max_width)
        new.move_to(anchor or old)
        new.align_to(anchor or old, LEFT)
        old.set_opacity(1)
        self.play(Transform(old, new), run_time=run_time)
        return old  # type: ignore[return-value]

    def hot_deuterium_stage(
        self,
        protons: list[VGroup],
        neutrons: list[VGroup],
        photons: list[VGroup],
        reaction_note: Text,
    ) -> None:
        reaction_note = self.replace_text(
            reaction_note,
            "дейтерий рождается, но не выживает",
            color=TEXT_COLOR,
            anchor=reaction_note,
            font_size=25,
        )

        centers = [
            np.array([-2.85, 0.85, 0.0]),
            np.array([-1.15, -0.55, 0.0]),
            np.array([0.70, 0.64, 0.0]),
        ]

        for idx, center in enumerate(centers):
            p = protons[idx]
            n = neutrons[idx % len(neutrons)]
            g = photons[idx]
            self.pause_motion(p, n, g)

            equation = Text("p + n → D", font_size=25, color=DEUTERIUM_COLOR)
            equation.move_to(np.array([-3.75, -3.38, 0.0])).align_to(reaction_note, LEFT)

            self.play(
                FadeIn(equation, shift=UP * 0.08),
                p.animate.move_to(center + LEFT * 0.28),
                n.animate.move_to(center + RIGHT * 0.28),
                run_time=0.95,
                rate_func=smooth,
            )

            d = deuterium().move_to(center)
            flash = reaction_flash(center, DEUTERIUM_COLOR)
            self.play(
                FadeIn(flash, scale=0.85),
                FadeOut(p, scale=0.75),
                FadeOut(n, scale=0.75),
                FadeIn(d, scale=1.05),
                run_time=0.62,
            )
            self.play(
                flash.animate.scale(2.2).set_opacity(0),
                run_time=0.28,
                rate_func=smooth,
            )
            self.remove(flash)

            next_eq = Text("D + γ → p + n", font_size=25, color=PHOTON_COLOR)
            next_eq.move_to(equation).align_to(equation, LEFT)
            self.play(Transform(equation, next_eq), run_time=0.45)

            attack_direction = unit(np.array([-1.0, -0.36 if idx % 2 == 0 else 0.36, 0.0]))
            orient_photon(g, angle_of(attack_direction))
            start = center - attack_direction * 2.35
            place_photon_head_at(g, start)
            self.add(g)
            hit = center + attack_direction * 0.05
            self.play(
                g.animate.shift(hit - photon_head_center(g)),
                run_time=0.56,
                rate_func=linear,
            )

            split_left = center + LEFT * 0.54 + DOWN * 0.07
            split_right = center + RIGHT * 0.54 + UP * 0.07
            p.move_to(split_left)
            n.move_to(split_right)
            burst = reaction_flash(center, PHOTON_COLOR)
            self.play(
                FadeIn(burst, scale=0.72),
                FadeOut(d, scale=0.55),
                FadeOut(g, scale=0.36),
                FadeIn(p, scale=0.92),
                FadeIn(n, scale=0.92),
                run_time=0.72,
            )
            self.play(
                p.animate.shift(LEFT * 0.36),
                n.animate.shift(RIGHT * 0.36),
                burst.animate.scale(2.3).set_opacity(0),
                FadeOut(equation, shift=DOWN * 0.08),
                run_time=0.54,
            )
            self.remove(d, g, burst)
            g.absorbed = True

            self.set_velocity(p, np.array([-0.17, 0.06, 0.0]))
            self.set_velocity(n, np.array([0.12, -0.08, 0.0]))
            self.resume_motion(p, n)
            self.wait(0.05)

        self.wait(0.18)

    def cooling_stage(
        self,
        temp: ValueTracker,
        photons: list[VGroup],
        caption: Text,
        reaction_note: Text,
    ) -> None:
        self.replace_text(
            reaction_note,
            "энергии фотонов уже не хватает, чтобы всё разбивать",
            color=MUTED_TEXT,
            anchor=reaction_note,
            font_size=23,
        )

        photon_anims = []
        for g in photons:
            if getattr(g, "absorbed", False):
                continue
            self.scale_velocity(g, 0.55)
            head_glow, ray_glows, rays, head, gamma = g
            head_center = photon_head_center(g)
            photon_anims.extend(
                [
                    head_glow.animate.set_fill(PHOTON_DIM, opacity=0.10).scale(
                        1.05,
                        about_point=head_center,
                    ),
                    ray_glows.animate.scale(1.22, about_point=head_center).set_stroke(
                        color=PHOTON_DIM,
                        width=6.2,
                        opacity=0.07,
                    ),
                    rays.animate.scale(1.22, about_point=head_center).set_stroke(
                        color=PHOTON_DIM,
                        width=2.0,
                        opacity=0.58,
                    ),
                    head.animate.set_fill(PHOTON_DIM, opacity=0.82).set_stroke(
                        WHITE,
                        width=0.8,
                        opacity=0.30,
                    ),
                    gamma.animate.set_color(PHOTON_DIM).set_opacity(0.72),
                ]
            )

        self.play(
            temp.animate.set_value(0.42),
            *photon_anims,
            run_time=3.8,
            rate_func=smooth,
        )
        self.wait(0.2)

    def synthesis_window(
        self,
        protons: list[VGroup],
        neutrons: list[VGroup],
        photons: list[VGroup],
        caption: Text,
        reaction_note: Text,
    ) -> tuple[VGroup, VGroup]:
        self.replace_text(
            caption,
            "открывается окно синтеза",
            color=WINDOW_COLOR,
            anchor=caption,
        )
        self.replace_text(
            reaction_note,
            "D теперь выживает и становится ступенькой к He-4",
            color=TEXT_COLOR,
            anchor=reaction_note,
            font_size=24,
        )

        for mob in [protons[0], protons[1], protons[2], protons[3], neutrons[0], neutrons[1], neutrons[2]]:
            self.pause_motion(mob)
        for g in photons[:5]:
            self.scale_velocity(g, 0.66)

        d_pos = np.array([-2.65, 0.22, 0.0])
        p0, n0 = protons[0], neutrons[0]
        self.play(
            p0.animate.move_to(d_pos + LEFT * 0.26),
            n0.animate.move_to(d_pos + RIGHT * 0.26),
            run_time=0.90,
        )
        d1 = deuterium().move_to(d_pos)
        flash = reaction_flash(d_pos, WINDOW_COLOR)
        eq = Text("p + n → D", font_size=25, color=DEUTERIUM_COLOR)
        eq.move_to(np.array([-3.75, -3.38, 0.0])).align_to(reaction_note, LEFT)
        self.play(
            FadeIn(eq, shift=UP * 0.08),
            FadeIn(flash, scale=0.85),
            FadeOut(p0, scale=0.75),
            FadeOut(n0, scale=0.75),
            FadeIn(d1, scale=1.06),
            run_time=0.76,
        )
        self.play(flash.animate.scale(2.0).set_opacity(0), run_time=0.30)
        self.remove(flash)
        self.wait(0.15)

        p1 = protons[1]
        he3_pos = np.array([-0.65, 0.22, 0.0])
        self.play(
            d1.animate.move_to(he3_pos + LEFT * 0.22),
            p1.animate.move_to(he3_pos + RIGHT * 0.38),
            Transform(eq, Text("D + p → He-3", font_size=25, color=HE3_COLOR).move_to(eq).align_to(eq, LEFT)),
            run_time=0.95,
        )
        he3_mob = he3().move_to(he3_pos)
        flash = reaction_flash(he3_pos, HE3_COLOR)
        self.play(
            FadeIn(flash, scale=0.85),
            FadeOut(d1, scale=0.75),
            FadeOut(p1, scale=0.75),
            FadeIn(he3_mob, scale=1.05),
            run_time=0.78,
        )
        self.play(flash.animate.scale(2.0).set_opacity(0), run_time=0.30)
        self.remove(flash)

        n1 = neutrons[1]
        he4_pos = np.array([1.55, 0.25, 0.0])
        self.play(
            he3_mob.animate.move_to(he4_pos + LEFT * 0.27),
            n1.animate.move_to(he4_pos + RIGHT * 0.42),
            Transform(eq, Text("He-3 + n → He-4", font_size=25, color=HE4_COLOR).move_to(eq).align_to(eq, LEFT)),
            run_time=1.05,
        )
        he4_main = he4(scale=1.18).move_to(he4_pos)
        flash = reaction_flash(he4_pos, HE4_COLOR)
        self.play(
            FadeIn(flash, scale=0.85),
            FadeOut(he3_mob, scale=0.72),
            FadeOut(n1, scale=0.72),
            FadeIn(he4_main, scale=1.06),
            run_time=0.88,
        )
        self.play(flash.animate.scale(2.4).set_opacity(0), run_time=0.38)
        self.remove(flash)

        self.replace_text(
            reaction_note,
            "свободные нейтроны уходят в связанные ядра",
            color=TEXT_COLOR,
            anchor=reaction_note,
            font_size=24,
        )

        p2, n2 = protons[2], neutrons[2]
        trace_pos = np.array([2.95, -0.72, 0.0])
        self.play(
            p2.animate.move_to(trace_pos + LEFT * 0.20),
            n2.animate.move_to(trace_pos + RIGHT * 0.20),
            run_time=0.90,
        )
        trace_d = deuterium().scale(0.70).move_to(trace_pos)
        trace_label = Text("след D", font_size=18, color=TRACE_COLOR)
        trace_label.next_to(trace_d, DOWN, buff=0.05)
        trace_d.add(trace_label)
        self.play(
            Transform(eq, Text("малые следы остаются", font_size=25, color=TRACE_COLOR).move_to(eq).align_to(eq, LEFT)),
            FadeOut(p2, scale=0.72),
            FadeOut(n2, scale=0.72),
            FadeIn(trace_d, scale=1.06),
            run_time=0.90,
        )
        self.play(FadeOut(eq, shift=DOWN * 0.10), run_time=0.45)
        self.wait(0.3)
        return he4_main, trace_d

    def final_composition(
        self,
        protons: list[VGroup],
        neutrons: list[VGroup],
        photons: list[VGroup],
        he4_main: VGroup,
        trace_d: VGroup,
        temp: ValueTracker,
        caption: Text,
        reaction_note: Text,
    ) -> None:
        self.replace_text(
            caption,
            "BBN задал стартовый состав вещества для будущих звёзд",
            color=TEXT_COLOR,
            anchor=caption,
            font_size=25,
            max_width=9.7,
            run_time=0.70,
        )
        self.replace_text(
            reaction_note,
            "почти все нейтроны связаны; свободных протонов намного больше",
            color=MUTED_TEXT,
            anchor=reaction_note,
            font_size=22,
            max_width=9.5,
            run_time=0.70,
        )

        consumed = {0, 1, 2}
        free_protons = [p for i, p in enumerate(protons) if i not in consumed]
        for p in free_protons:
            self.pause_motion(p)
        for g in photons:
            self.pause_motion(g)

        left_targets = []
        columns = 5
        for i, _ in enumerate(free_protons):
            col = i % columns
            row = i // columns
            left_targets.append(np.array([-5.20 + col * 0.46, 1.20 - row * 0.46, 0.0]))

        he4_targets = [
            np.array([1.35, 1.12, 0.0]),
            np.array([2.50, 1.08, 0.0]),
            np.array([1.95, 0.08, 0.0]),
        ]
        he4_symbol_1 = he4(scale=0.92).move_to(he4_targets[1])
        he4_symbol_2 = he4(scale=0.82).move_to(he4_targets[2])

        he_label = Text("стабильные ядра He-4", font_size=22, color=HE4_COLOR)
        he_label.move_to(np.array([1.92, 1.95, 0.0]))

        photon_fade = [FadeOut(g, shift=0.2 * DOWN) for g in photons]
        proton_moves = [p.animate.move_to(target) for p, target in zip(free_protons, left_targets)]

        self.play(
            temp.animate.set_value(0.20),
            *photon_fade,
            *proton_moves,
            he4_main.animate.scale(0.86).move_to(he4_targets[0]),
            trace_d.animate.scale(0.78).move_to(np.array([3.38, -0.28, 0.0])).set_opacity(0.78),
            run_time=1.7,
            rate_func=smooth,
        )

        proton_label = Text("свободные p: будущий водород", font_size=22, color=PROTON_COLOR)
        proton_label.next_to(VGroup(*free_protons), UP, buff=0.28).align_to(VGroup(*free_protons), LEFT)
        fit_to_width(proton_label, 4.6)

        self.play(
            FadeIn(proton_label, shift=DOWN * 0.08),
            FadeIn(he_label, shift=DOWN * 0.08),
            FadeIn(he4_symbol_1, scale=0.92),
            FadeIn(he4_symbol_2, scale=0.92),
            run_time=0.8,
        )

        panel = composition_panel()
        self.play(FadeIn(panel, shift=UP * 0.15), run_time=0.8)
        self.wait(2.2)
