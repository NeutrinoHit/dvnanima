from manim import *
import numpy as np
from pathlib import Path
import configparser


def load_cfg(path: str = "run.cfg") -> dict:
    cfg = configparser.ConfigParser(inline_comment_prefixes=(";",))
    if Path(path).exists():
        cfg.read(Path(path))

    def get(section, key, cast=str, fallback=None):
        if cfg.has_option(section, key):
            return cast(cfg.get(section, key))
        if fallback is None:
            raise KeyError(f"Missing [{section}] {key} in {path}")
        return cast(fallback)

    manim_params = {
        "pixel_width": get("manim", "pixel_width", int, 1080),
        "pixel_height": get("manim", "pixel_height", int, 1920),
        "frame_width": get("manim", "frame_width", float, 9.0),
        "frame_height": get("manim", "frame_height", float, 16.0),
        "frame_rate": get("manim", "frame_rate", int, 60),
        "background_color": get("manim", "background_color", str, "#000000"),
        "renderer": get("manim", "renderer", str, ""),
    }

    scene = {
        "text_font": get("scene", "text_font", str, "PT Sans"),
        "show_coords": get("scene", "show_coords", int, 0) == 1,
        "coord_font_size": get("scene", "coord_font_size", int, 24),
        "left_x_min": get("scene", "left_x_min", float, -1.0),
        "left_x_max": get("scene", "left_x_max", float, 3.0),
        "left_x_step": get("scene", "left_x_step", float, 1.0),
        "left_y_min": get("scene", "left_y_min", float, -2.0),
        "left_y_max": get("scene", "left_y_max", float, 2.0),
        "left_y_step": get("scene", "left_y_step", float, 1.0),
        "grid_stroke_width": get("scene", "grid_stroke_width", float, 1.0),
        "grid_stroke_opacity": get("scene", "grid_stroke_opacity", float, 0.30),
        "axis_stroke_width": get("scene", "axis_stroke_width", float, 2.0),
        "unit_circle_radius": get("scene", "unit_circle_radius", float, 1.0),
        "unit_circle_stroke_width": get("scene", "unit_circle_stroke_width", float, 2.0),
        "unit_circle_stroke_opacity": get("scene", "unit_circle_stroke_opacity", float, 0.50),
        "dot_radius": get("scene", "dot_radius", float, 0.11),
        "dot_shadow": get("scene", "dot_shadow", float, 1.5),
        "start_label_tex": get("scene", "start_label_tex", str, r"i"),
        "start_label_scale": get("scene", "start_label_scale", float, 1.0),
        "start_label_buff": get("scene", "start_label_buff", float, 0.08),
        "top_formula_tex": get("scene", "top_formula_tex", str, r"i^i=\exp(i\,\log i)"),
        "top_formula_scale": get("scene", "top_formula_scale", float, 0.90),
        "top_formula_corner": get("scene", "top_formula_corner", str, "UL"),
        "focus_center_y": get("scene", "focus_center_y", float, -0.25),
        "focus_left_scale": get("scene", "focus_left_scale", float, get("scene", "left_group_scale", float, 1.02)),
        "focus_right_scale": get("scene", "focus_right_scale", float, get("scene", "right_group_scale", float, 0.80)),
        "stage_caption_font_size": get("scene", "stage_caption_font_size", int, 32),
        "stage_caption_buff": get("scene", "stage_caption_buff", float, 0.15),
        "stage1_caption": get("scene", "stage1_caption", str, "1) Start: z = i"),
        "stage2_caption": get("scene", "stage2_caption", str, "2) Log branches"),
        "stage3_caption": get("scene", "stage3_caption", str, "3) x i and exp"),
        "winding_ns": get("scene", "winding_ns", str, "0,1,2"),
        "winding_label_scale": get("scene", "winding_label_scale", float, 0.70),
        "winding_arc_radius_base": get("scene", "winding_arc_radius_base", float, 1.06),
        "winding_arc_radius_step": get("scene", "winding_arc_radius_step", float, 0.14),
        "winding_arc_stroke_width": get("scene", "winding_arc_stroke_width", float, 3.0),
        "winding_arc_stroke_opacity": get("scene", "winding_arc_stroke_opacity", float, 0.80),
        "winding_formula_tex": get("scene", "winding_formula_tex", str, r"i=e^{i(\pi/2+2\pi n)}"),
        "winding_formula_scale": get("scene", "winding_formula_scale", float, 0.72),
        "winding_formula_buff": get("scene", "winding_formula_buff", float, 0.18),
        "log_u_min": get("scene", "log_u_min", float, -15.0),
        "log_u_max": get("scene", "log_u_max", float, 1.0),
        "log_u_step": get("scene", "log_u_step", float, 2.0),
        "log_v_min": get("scene", "log_v_min", float, -8.0),
        "log_v_max": get("scene", "log_v_max", float, 16.0),
        "log_v_step": get("scene", "log_v_step", float, 2.0),
        "log_grid_u_step": get("scene", "log_grid_u_step", float, 1.0),
        "log_grid_v_step": get("scene", "log_grid_v_step", float, 1.0),
        "log_title": get("scene", "log_title", str, "Log-plane: u=Re Log z, v=Im Log z"),
        "log_title_font_size": get("scene", "log_title_font_size", int, 26),
        "log_title_buff": get("scene", "log_title_buff", float, 0.15),
        "branch_ns": get("scene", "branch_ns", str, "-1,0,1,2"),
        "ladder_dot_radius": get("scene", "ladder_dot_radius", float, 0.06),
        "branch_formula_tex": get("scene", "branch_formula_tex", str, r"\log(i)=i(\pi/2+2\pi n)"),
        "branch_formula_scale": get("scene", "branch_formula_scale", float, 0.70),
        "branch_formula_buff": get("scene", "branch_formula_buff", float, 0.20),
        "halo_radius": get("scene", "halo_radius", float, 0.14),
        "halo_stroke_width": get("scene", "halo_stroke_width", float, 3.0),
        "target_dot_radius": get("scene", "target_dot_radius", float, 0.06),
        "rot_text": get("scene", "rot_text", str, "x i  -> rotate by 90 deg"),
        "exp_text": get("scene", "exp_text", str, "exp -> positive real axis"),
        "side_text_font_size": get("scene", "side_text_font_size", int, 28),
        "side_text_buff": get("scene", "side_text_buff", float, 0.12),
        "u_mark_radius": get("scene", "u_mark_radius", float, 0.07),
        "u_mark_tex": get("scene", "u_mark_tex", str, r"-\pi/2"),
        "u_mark_scale": get("scene", "u_mark_scale", float, 0.75),
        "u_mark_buff": get("scene", "u_mark_buff", float, 0.10),
        "exp_dot_radius": get("scene", "exp_dot_radius", float, 0.07),
        "principal_dot_radius": get("scene", "principal_dot_radius", float, 0.09),
        "result_eq_tex": get("scene", "result_eq_tex", str, r"i^i=e^{-\pi/2}"),
        "result_eq_scale": get("scene", "result_eq_scale", float, 0.90),
        "result_eq_buff": get("scene", "result_eq_buff", float, 0.15),
        "principal_text": get("scene", "principal_text", str, "Principal branch"),
        "principal_text_font_size": get("scene", "principal_text_font_size", int, 30),
        "principal_text_buff": get("scene", "principal_text_buff", float, 0.12),
        "value_decimals": get("scene", "value_decimals", int, 6),
        "value_font_size": get("scene", "value_font_size", float, 36.0),
        "ratio_text": get("scene", "ratio_text", str, "Each next branch is about {ratio:.1f}x smaller"),
        "ratio_font_size": get("scene", "ratio_font_size", int, 28),
        "ratio_buff": get("scene", "ratio_buff", float, 0.15),
        "show_zero_zoom": get("scene", "show_zero_zoom", int, 1) == 1,
        "zero_zoom_scale": get("scene", "zero_zoom_scale", float, 1.55),
        "zero_zoom_time": get("scene", "zero_zoom_time", float, 1.0),
        "show_hint": get("scene", "show_hint", int, 1) == 1,
        "hint_text": get("scene", "hint_text", str, "Other branches: e^{-pi/2-2pi n} are real and positive."),
        "hint_font_size": get("scene", "hint_font_size", int, 26),
        "hint_bottom_buff": get("scene", "hint_bottom_buff", float, 0.35),
        "nonprincipal_opacity": get("scene", "nonprincipal_opacity", float, 0.42),
        "others_lag_ratio": get("scene", "others_lag_ratio", float, 0.16),
        "intro_time": get("scene", "intro_time", float, 1.2),
        "title_time": get("scene", "title_time", float, 0.6),
        "transition_time": get("scene", "transition_time", float, 0.8),
        "winding_time": get("scene", "winding_time", float, 1.4),
        "matchcut_time": get("scene", "matchcut_time", float, 1.0),
        "branch_formula_time": get("scene", "branch_formula_time", float, 0.6),
        "halo_time": get("scene", "halo_time", float, 0.35),
        "rotate_text_time": get("scene", "rotate_text_time", float, 0.4),
        "rotate_time": get("scene", "rotate_time", float, 1.2),
        "u_mark_time": get("scene", "u_mark_time", float, 0.6),
        "exp_text_time": get("scene", "exp_text_time", float, 0.4),
        "exp_dots_time": get("scene", "exp_dots_time", float, 0.9),
        "principal_move_time": get("scene", "principal_move_time", float, 0.9),
        "flash_time": get("scene", "flash_time", float, 0.9),
        "result_time": get("scene", "result_time", float, 0.7),
        "hint_time": get("scene", "hint_time", float, 0.6),
        "stage_pause": get("scene", "stage_pause", float, 0.2),
        "final_wait": get("scene", "final_wait", float, 2.8),
        "flash_line_length": get("scene", "flash_line_length", float, 0.18),
        "flash_radius": get("scene", "flash_radius", float, 0.35),
    }

    colors = {
        "text_col": get("colors", "text_col", str, "#FFFFFF"),
        "label_col": get("colors", "label_col", str, "#FFD166"),
        "grid_col": get("colors", "grid_col", str, "#4D6BFF"),
        "axis_col": get("colors", "axis_col", str, "#FFFFFF"),
        "circle_col": get("colors", "circle_col", str, "#7AA6FF"),
        "dot_col": get("colors", "dot_col", str, "#FF4D4D"),
        "ladder_col": get("colors", "ladder_col", str, "#7AA6FF"),
        "target_col": get("colors", "target_col", str, "#FF4D4D"),
        "branch_col": get("colors", "branch_col", str, "#7AA6FF"),
        "value_col": get("colors", "value_col", str, "#FFD166"),
        "flash_col": get("colors", "flash_col", str, "#FFD166"),
        "principal_col": get("colors", "principal_col", str, "#FF4D4D"),
    }
    return {"manim": manim_params, "scene": scene, "colors": colors}


def parse_int_list(value: str, fallback: list[int]) -> list[int]:
    out = []
    for token in value.replace(";", ",").split(","):
        token = token.strip()
        if not token:
            continue
        try:
            out.append(int(token))
        except ValueError:
            continue
    return out if out else fallback


def normalize_tex(expr: str) -> str:
    return expr.replace(r"\Log", r"\log")


CFG = load_cfg(str(Path(__file__).with_name("run.cfg")))
config.pixel_width = CFG["manim"]["pixel_width"]
config.pixel_height = CFG["manim"]["pixel_height"]
config.frame_width = CFG["manim"]["frame_width"]
config.frame_height = CFG["manim"]["frame_height"]
config.frame_rate = CFG["manim"]["frame_rate"]
config.background_color = CFG["manim"]["background_color"]
if CFG["manim"]["renderer"]:
    config.renderer = CFG["manim"]["renderer"]


class ImaginaryPowerScene(Scene):
    def construct(self):
        scene = CFG["scene"]
        colors = CFG["colors"]
        self.camera.background_color = CFG["manim"]["background_color"]
        Text.set_default(font=scene["text_font"], color=colors["text_col"])
        MathTex.set_default(color=colors["text_col"])

        plane = ComplexPlane(
            x_range=[scene["left_x_min"], scene["left_x_max"], scene["left_x_step"]],
            y_range=[scene["left_y_min"], scene["left_y_max"], scene["left_y_step"]],
            background_line_style={
                "stroke_color": colors["grid_col"],
                "stroke_width": scene["grid_stroke_width"],
                "stroke_opacity": scene["grid_stroke_opacity"],
            },
            axis_config={"stroke_color": colors["axis_col"], "stroke_width": scene["axis_stroke_width"]},
        )
        if scene["show_coords"]:
            plane.add_coordinates(font_size=scene["coord_font_size"])

        circle = Circle(
            radius=scene["unit_circle_radius"],
            color=colors["circle_col"],
            stroke_width=scene["unit_circle_stroke_width"],
            stroke_opacity=scene["unit_circle_stroke_opacity"],
        )
        left_group = VGroup(plane, circle).scale(scene["focus_left_scale"]).move_to([0.0, scene["focus_center_y"], 0.0])
        left_group_template = left_group.copy()
        left_plane_template = left_group_template[0]

        title = MathTex(normalize_tex(scene["top_formula_tex"]), color=colors["label_col"]).scale(scene["top_formula_scale"])
        corner = scene["top_formula_corner"].upper()
        if corner == "UR":
            title.to_corner(UR)
        elif corner == "LL":
            title.to_corner(DL)
        elif corner == "LR":
            title.to_corner(DR)
        else:
            title.to_corner(UL)

        def stage_cap(text: str) -> Text:
            return Text(
                text,
                font=scene["text_font"],
                font_size=scene["stage_caption_font_size"],
                color=colors["label_col"],
            ).next_to(title, DOWN, buff=scene["stage_caption_buff"])

        stage_caption = stage_cap(scene["stage1_caption"])

        self.add(left_group)
        self.play(Write(title), run_time=scene["title_time"])
        self.play(FadeIn(stage_caption), run_time=0.3)

        dot_i = Dot(plane.n2p(1j), color=colors["dot_col"], radius=scene["dot_radius"]).set(shadow=scene["dot_shadow"])
        label_i = (
            MathTex(scene["start_label_tex"], color=colors["label_col"])
            .scale(scene["start_label_scale"])
            .next_to(dot_i, UP + RIGHT, buff=scene["start_label_buff"])
        )
        self.play(GrowFromCenter(dot_i), Write(label_i), run_time=scene["intro_time"])

        winding_ns = [n for n in parse_int_list(scene["winding_ns"], fallback=[0, 1, 2]) if n >= 0]
        if not winding_ns:
            winding_ns = [0, 1, 2]
        winding_ns = sorted(set(winding_ns))
        winding_arcs = VGroup()
        winding_labels = VGroup()
        n_to_label = {}
        for idx, n in enumerate(winding_ns):
            ang = PI / 2 + TAU * n
            radius = scene["unit_circle_radius"] * (scene["winding_arc_radius_base"] + idx * scene["winding_arc_radius_step"])
            arc = ParametricFunction(
                lambda t, r=radius: plane.c2p(r * np.cos(t), r * np.sin(t)),
                t_range=[0, ang],
                color=colors["ladder_col"],
                stroke_width=scene["winding_arc_stroke_width"],
            )
            arc.set_stroke(opacity=scene["winding_arc_stroke_opacity"] * (1.0 - 0.12 * idx))
            winding_arcs.add(arc)
            if n == 0:
                tex = r"\pi/2"
            elif n == 1:
                tex = r"\pi/2+2\pi"
            else:
                tex = rf"\pi/2+{2*n}\pi"
            lbl = MathTex(tex, color=colors["value_col"]).scale(scene["winding_label_scale"])
            lbl.next_to(plane.c2p(0, radius), RIGHT, buff=0.08)
            winding_labels.add(lbl)
            n_to_label[n] = lbl

        winding_formula = MathTex(normalize_tex(scene["winding_formula_tex"]), color=colors["value_col"]).scale(
            scene["winding_formula_scale"]
        )
        winding_formula.next_to(left_group, DOWN, buff=scene["winding_formula_buff"])

        self.play(LaggedStart(*[Create(a) for a in winding_arcs], lag_ratio=0.2), run_time=scene["winding_time"])
        self.play(
            LaggedStart(*[FadeIn(l) for l in winding_labels], lag_ratio=0.15),
            FadeIn(winding_formula),
            run_time=0.8,
        )
        self.wait(scene["stage_pause"])

        log_axes = Axes(
            x_range=[scene["log_u_min"], scene["log_u_max"], scene["log_u_step"]],
            y_range=[scene["log_v_min"], scene["log_v_max"], scene["log_v_step"]],
            tips=False,
            axis_config={"stroke_color": colors["axis_col"], "stroke_width": scene["axis_stroke_width"]},
        )
        log_bg = NumberPlane(
            x_range=[scene["log_u_min"], scene["log_u_max"], scene["log_grid_u_step"]],
            y_range=[scene["log_v_min"], scene["log_v_max"], scene["log_grid_v_step"]],
            background_line_style={
                "stroke_color": colors["grid_col"],
                "stroke_width": scene["grid_stroke_width"],
                "stroke_opacity": scene["grid_stroke_opacity"],
            },
            axis_config={"stroke_opacity": 0.0},
        )
        right_group = VGroup(log_bg, log_axes).scale(scene["focus_right_scale"]).move_to([0.0, scene["focus_center_y"], 0.0])
        log_title = Text(scene["log_title"], font_size=scene["log_title_font_size"], color=colors["text_col"])
        log_title.next_to(right_group, UP, buff=scene["log_title_buff"])

        self.play(
            ReplacementTransform(left_group, right_group),
            FadeOut(dot_i),
            FadeOut(label_i),
            FadeIn(log_title, shift=UP * 0.1),
            Transform(stage_caption, stage_cap(scene["stage2_caption"])),
            run_time=scene["transition_time"],
        )

        branch_ns = parse_int_list(scene["branch_ns"], fallback=[-1, 0, 1, 2])
        if 0 not in branch_ns:
            branch_ns = [0] + branch_ns
        entries = []
        for n in branch_ns:
            v = np.pi / 2 + TAU * n
            if scene["log_v_min"] <= v <= scene["log_v_max"]:
                entries.append((n, v))
        if not entries:
            entries = [(0, np.pi / 2)]
        entries.sort(key=lambda nv: (0 if nv[0] == 0 else 1, abs(nv[0])))
        principal_entry = entries[0]
        other_entries = entries[1:]

        principal_ladder = Dot(log_axes.c2p(0.0, principal_entry[1]), radius=scene["ladder_dot_radius"], color=colors["principal_col"])
        other_ladder = VGroup(
            *[Dot(log_axes.c2p(0.0, v), radius=scene["ladder_dot_radius"], color=colors["ladder_col"]) for _, v in other_entries]
        )
        if len(other_ladder) > 0:
            other_ladder.set_opacity(scene["nonprincipal_opacity"])

        pn = principal_entry[0]
        if pn in n_to_label:
            self.play(TransformFromCopy(n_to_label[pn], principal_ladder), run_time=scene["matchcut_time"] * 0.5)
        else:
            self.play(GrowFromCenter(principal_ladder), run_time=scene["halo_time"])
        other_anims = []
        for d, (n, _) in zip(other_ladder, other_entries):
            if n in n_to_label:
                other_anims.append(TransformFromCopy(n_to_label[n], d))
            else:
                other_anims.append(GrowFromCenter(d))
        if other_anims:
            self.play(LaggedStart(*other_anims, lag_ratio=scene["others_lag_ratio"]), run_time=scene["matchcut_time"])
        self.play(FadeOut(winding_arcs), FadeOut(winding_labels), FadeOut(winding_formula), run_time=0.35)

        branch_label = MathTex(normalize_tex(scene["branch_formula_tex"]), color=colors["value_col"]).scale(scene["branch_formula_scale"])
        branch_label.next_to(right_group, DOWN, buff=scene["branch_formula_buff"])
        self.play(Write(branch_label), run_time=scene["branch_formula_time"])

        halo = Circle(radius=scene["halo_radius"], color=colors["flash_col"], stroke_width=scene["halo_stroke_width"]).move_to(principal_ladder)
        self.play(Create(halo), run_time=scene["halo_time"])

        principal_target = Dot(log_axes.c2p(-principal_entry[1], 0.0), radius=scene["target_dot_radius"], color=colors["principal_col"])
        other_targets = VGroup(
            *[Dot(log_axes.c2p(-v, 0.0), radius=scene["target_dot_radius"], color=colors["target_col"]) for _, v in other_entries]
        )
        if len(other_targets) > 0:
            other_targets.set_opacity(scene["nonprincipal_opacity"])

        rot_text = Text(scene["rot_text"], font_size=scene["side_text_font_size"], color=colors["text_col"])
        rot_text.next_to(log_title, DOWN, buff=scene["side_text_buff"])
        self.play(FadeIn(rot_text, shift=DOWN * 0.1), run_time=scene["rotate_text_time"])
        rotate_anims = [Transform(principal_ladder, principal_target), FadeOut(halo)]
        if len(other_ladder) > 0:
            rotate_anims.append(Transform(other_ladder, other_targets))
        self.play(*rotate_anims, run_time=scene["rotate_time"], rate_func=smooth)

        principal_u = -principal_entry[1]
        u_mark = Dot(log_axes.c2p(principal_u, 0.0), radius=scene["u_mark_radius"], color=colors["flash_col"])
        u_lab = MathTex(normalize_tex(scene["u_mark_tex"]), color=colors["flash_col"]).scale(scene["u_mark_scale"])
        u_lab.next_to(u_mark, DOWN, buff=scene["u_mark_buff"])
        self.play(GrowFromCenter(u_mark), Write(u_lab), run_time=scene["u_mark_time"])
        self.wait(scene["stage_pause"])

        principal_val = float(np.exp(-np.pi / 2))
        projector = Arrow(
            start=principal_ladder.get_center(),
            end=left_plane_template.c2p(principal_val, 0.0),
            color=colors["flash_col"],
            stroke_width=3.0,
            buff=0.06,
            max_tip_length_to_length_ratio=0.12,
        )
        self.play(GrowArrow(projector), run_time=0.5)

        left_group_back = left_group_template.copy()
        self.play(
            ReplacementTransform(right_group, left_group_back),
            FadeOut(log_title),
            FadeOut(branch_label),
            FadeOut(rot_text),
            FadeOut(u_mark),
            FadeOut(u_lab),
            FadeOut(principal_ladder),
            *( [FadeOut(other_ladder)] if len(other_ladder) > 0 else [] ),
            FadeOut(projector),
            Transform(stage_caption, stage_cap(scene["stage3_caption"])),
            run_time=scene["transition_time"],
        )

        exp_text = Text(scene["exp_text"], font_size=scene["side_text_font_size"], color=colors["text_col"])
        exp_text.next_to(stage_caption, DOWN, buff=scene["side_text_buff"])
        self.play(FadeIn(exp_text, shift=DOWN * 0.1), run_time=scene["exp_text_time"])

        left_plane_back = left_group_back[0]
        x_min = min(scene["left_x_min"], scene["left_x_max"])
        x_max = max(scene["left_x_min"], scene["left_x_max"])
        principal_exp = Dot(
            left_plane_back.c2p(principal_val, 0.0),
            radius=scene["principal_dot_radius"],
            color=colors["principal_col"],
        ).set(shadow=scene["dot_shadow"])
        other_exp = VGroup()
        for _, v in other_entries:
            x = float(np.exp(-v))
            if x_min <= x <= x_max:
                other_exp.add(Dot(left_plane_back.c2p(x, 0.0), radius=scene["exp_dot_radius"], color=colors["branch_col"]))
        if len(other_exp) > 0:
            other_exp.set_opacity(scene["nonprincipal_opacity"])

        self.play(GrowFromCenter(principal_exp), run_time=scene["principal_move_time"])
        if len(other_exp) > 0:
            self.play(LaggedStart(*[GrowFromCenter(d) for d in other_exp], lag_ratio=scene["others_lag_ratio"]), run_time=scene["exp_dots_time"])

        val_label = DecimalNumber(
            principal_val,
            num_decimal_places=scene["value_decimals"],
            show_ellipsis=False,
            color=colors["value_col"],
            font_size=scene["value_font_size"],
        ).next_to(principal_exp, UP)
        result_eq = MathTex(normalize_tex(scene["result_eq_tex"]), color=colors["text_col"]).scale(scene["result_eq_scale"])
        result_eq.next_to(val_label, UP, buff=scene["result_eq_buff"])
        principal_text = Text(
            scene["principal_text"],
            font_size=scene["principal_text_font_size"],
            color=colors["text_col"],
        ).next_to(result_eq, UP, buff=scene["principal_text_buff"])

        self.play(
            Flash(
                principal_exp,
                color=colors["flash_col"],
                line_length=scene["flash_line_length"],
                flash_radius=scene["flash_radius"],
            ),
            Write(val_label),
            run_time=scene["flash_time"],
        )
        self.play(Write(result_eq), FadeIn(principal_text), run_time=scene["result_time"])
        self.play(FadeOut(exp_text), run_time=0.2)

        ratio_value = float(np.exp(2 * np.pi))
        ratio_text = scene["ratio_text"]
        if "{ratio" in ratio_text:
            ratio_text = ratio_text.format(ratio=ratio_value)
        ratio_label = Text(
            ratio_text,
            font=scene["text_font"],
            font_size=scene["ratio_font_size"],
            color=colors["label_col"],
        ).next_to(result_eq, DOWN, buff=scene["ratio_buff"])
        self.play(FadeIn(ratio_label), run_time=0.45)

        if scene["show_zero_zoom"]:
            zero_pt = left_plane_back.c2p(0.0, 0.0)
            zoom_group = VGroup(left_group_back, principal_exp, other_exp, val_label, result_eq, principal_text, ratio_label)
            self.play(
                zoom_group.animate.scale(scene["zero_zoom_scale"], about_point=zero_pt),
                run_time=scene["zero_zoom_time"],
                rate_func=smooth,
            )

        if scene["show_hint"]:
            hint = Text(scene["hint_text"], font_size=scene["hint_font_size"], color=colors["text_col"])
            hint.to_edge(DOWN, buff=scene["hint_bottom_buff"])
            self.play(FadeIn(hint), run_time=scene["hint_time"])

        self.wait(scene["final_wait"])
