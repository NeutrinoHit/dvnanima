from manim import *
import numpy as np

from pathlib import Path
import configparser


def load_cfg(path: str = "run.cfg") -> dict:
    cfg = configparser.ConfigParser(inline_comment_prefixes=(";",))
    cfg.read(Path(path))

    def get(section, key, cast=str, fallback=None):
        if fallback is None:
            return cast(cfg[section][key])
        return cast(cfg.get(section, key, fallback=str(fallback)))

    manim_params = {
        "pixel_width": get("manim", "pixel_width", int),
        "pixel_height": get("manim", "pixel_height", int),
        "frame_width": get("manim", "frame_width", float),
        "frame_height": get("manim", "frame_height", float),
        "frame_rate": get("manim", "frame_rate", int, 60),
        "background_color": get("manim", "background_color", str),
    }

    scene = {
        "x_max": get("scene", "x_max", float, 2 * np.pi),
        "run_time": get("scene", "run_time", float, 6.0),
        "tail_wait": get("scene", "tail_wait", float, 0.3),
        "title_text": get("scene", "title_text", str, "Neutrino as Quantum Chameleon"),
    }

    ui = {
        "title_scale": get("ui", "title_scale", float, 0.9),
        "title_buff": get("ui", "title_buff", float, 0.45),
        "axes_x_length": get("ui", "axes_x_length", float, 11.2),
        "top_axes_y_length": get("ui", "top_axes_y_length", float, 2.6),
        "bot_axes_y_length": get("ui", "bot_axes_y_length", float, 2.6),
        "top_axes_shift": get("ui", "top_axes_shift", float, 1.0),
        "bot_axes_shift": get("ui", "bot_axes_shift", float, -2.05),
        "axis_stroke_width": get("ui", "axis_stroke_width", int, 2),
        "wave_stroke_width": get("ui", "wave_stroke_width", int, 7),
        "prob_stroke_width": get("ui", "prob_stroke_width", int, 7),
        "label_color": get("ui", "label_color", str, "#c9d2df"),
        "label_font_size": get("ui", "label_font_size", int, 30),
        "legend_font_size": get("ui", "legend_font_size", int, 44),
        "annot_font_size": get("ui", "annot_font_size", int, 28),
    }

    return {"manim": manim_params, "scene": scene, "ui": ui}


CFG = load_cfg("run.cfg")

config.pixel_width = CFG["manim"]["pixel_width"]
config.pixel_height = CFG["manim"]["pixel_height"]
config.frame_width = CFG["manim"]["frame_width"]
config.frame_height = CFG["manim"]["frame_height"]
config.frame_rate = CFG["manim"]["frame_rate"]

class NeutrinoOscillationPhases(Scene):
    def construct(self):
        scene = CFG["scene"]
        ui = CFG["ui"]
        self.camera.background_color = CFG["manim"]["background_color"]

        title = Text(scene["title_text"], weight=BOLD).scale(ui["title_scale"])
        title.to_edge(UP, buff=ui["title_buff"])
        self.add(title)

        X0 = 0.0
        X1 = scene["x_max"]

        top_axes = Axes(
            x_range=[X0, X1, np.pi],
            y_range=[-1.4, 1.4, 1],
            x_length=ui["axes_x_length"],
            y_length=ui["top_axes_y_length"],
            tips=False,
            axis_config={"include_numbers": False, "stroke_width": ui["axis_stroke_width"]},
        ).shift(UP * ui["top_axes_shift"])

        bot_axes = Axes(
            x_range=[X0, X1, np.pi],
            y_range=[0, 1.05, 0.5],
            x_length=ui["axes_x_length"],
            y_length=ui["bot_axes_y_length"],
            tips=False,
            axis_config={"include_numbers": False, "stroke_width": ui["axis_stroke_width"]},
        ).shift(UP * ui["bot_axes_shift"])

        top_label = Text("Two mass-eigenstate phases", font_size=ui["label_font_size"], weight=BOLD)\
            .set_color(ui["label_color"]).next_to(top_axes, UP, buff=0.22)
        bot_label = Text("Oscillation probabilities", font_size=ui["label_font_size"], weight=BOLD)\
            .set_color(ui["label_color"]).next_to(bot_axes, UP, buff=0.22)

        self.add(top_axes, bot_axes, top_label, bot_label)

        # --- top waves ---
        def nu1(x): return np.sin(3 * x)   # 3 crests
        def nu2(x): return np.sin(4 * x)   # 4 crests

        # Build every curve once over its full, fixed domain.  Replotting with a
        # changing x_range makes Axes.plot choose a slightly different sample
        # grid on every frame, so already visible line segments appear to jump.
        # Create reveals the fixed vector path without changing its geometry.
        wave1 = top_axes.plot(
            nu1, x_range=[X0, X1], color=BLUE, stroke_width=ui["wave_stroke_width"]
        )
        wave2 = top_axes.plot(
            nu2, x_range=[X0, X1], color=ORANGE, stroke_width=ui["wave_stroke_width"]
        )

        leg1 = MathTex(r"\nu_1", font_size=ui["legend_font_size"]).set_color(BLUE)
        leg2 = MathTex(r"\nu_2", font_size=ui["legend_font_size"]).set_color(ORANGE)
        legend = VGroup(
            VGroup(Line(LEFT*0.45, RIGHT*0.45).set_stroke(BLUE, ui["wave_stroke_width"]), leg1).arrange(RIGHT, buff=0.25),
            VGroup(Line(LEFT*0.45, RIGHT*0.45).set_stroke(ORANGE, ui["wave_stroke_width"]), leg2).arrange(RIGHT, buff=0.25),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.22).to_corner(UR, buff=0.65)

        self.add(legend)

        # --- bottom probabilities ---
        def Pee(x): return np.cos(0.5 * x) ** 2
        def Pem(x): return np.sin(0.5 * x) ** 2

        pee_curve = bot_axes.plot(
            Pee, x_range=[X0, X1], color=BLUE, stroke_width=ui["prob_stroke_width"]
        )
        pem_curve = bot_axes.plot(
            Pem, x_range=[X0, X1], color=ORANGE, stroke_width=ui["prob_stroke_width"]
        )

        pee_lbl = MathTex(r"P_{ee}", font_size=ui["legend_font_size"]).set_color(BLUE).next_to(bot_axes, LEFT, buff=0.35).shift(UP*0.65)
        pem_lbl = MathTex(r"P_{e\mu}", font_size=ui["legend_font_size"]).set_color(ORANGE).next_to(bot_axes, LEFT, buff=0.35).shift(DOWN*0.65)

        self.add(pee_lbl, pem_lbl)

        # ---------------- Annotations on top: x=0 and x=pi ----------------
        def vline(x, opacity=0.25):
            return Line(
                top_axes.c2p(x, -1.25),
                top_axes.c2p(x,  1.25),
            ).set_stroke(WHITE, 2, opacity=opacity)

        # vertical markers
        line0  = vline(0.0, opacity=0.20)
        linepi = vline(np.pi, opacity=0.20)

        # arrows + labels (place above the waves)
        def label_with_arrow(text, x, direction=UP):
            lbl = Text(text, font_size=ui["annot_font_size"], weight=BOLD).set_color(ui["label_color"])
            tip = top_axes.c2p(x, 1.05)
            if direction is UP:
                lbl.next_to(tip, UP, buff=0.15)
            else:
                lbl.next_to(tip, UP, buff=0.15)
            arr = Arrow(
                start=lbl.get_bottom() + DOWN*0.05,
                end=tip,
                buff=0.05,
                stroke_width=4,
                max_tip_length_to_length_ratio=0.18
            ).set_color(ui["label_color"])
            return VGroup(arr, lbl)

        in_phase = label_with_arrow("in phase", 0.0)
        out_phase = label_with_arrow("out of phase", np.pi)

        in_phase_marker = VGroup(line0, in_phase)
        out_phase_marker = VGroup(linepi, out_phase)
        self.add(in_phase_marker)

        # ---------------- Animation ----------------
        marker_fade_time = min(0.2, scene["run_time"] / 10)
        delayed_out_phase = Succession(
            Wait(scene["run_time"] / 2),
            FadeIn(out_phase_marker, run_time=marker_fade_time),
            Wait(scene["run_time"] / 2 - marker_fade_time),
        )
        self.play(
            Create(wave1, rate_func=linear),
            Create(wave2, rate_func=linear),
            Create(pee_curve, rate_func=linear),
            Create(pem_curve, rate_func=linear),
            delayed_out_phase,
            run_time=scene["run_time"],
        )
        self.wait(scene["tail_wait"])
