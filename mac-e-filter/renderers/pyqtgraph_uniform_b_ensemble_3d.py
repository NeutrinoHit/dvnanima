#!/usr/bin/env python3
"""Render three exact relativistic electron helices in a uniform B field."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from mac_e_filter.data_io import load_trajectory_dataset  # noqa: E402

import pyqtgraph_katrin_source_to_detector_3d as video_helpers  # noqa: E402
import pyqtgraph_katrin_spectrometer_3d as shared_scene  # noqa: E402


BACKGROUND = shared_scene.BACKGROUND
FOREGROUND = shared_scene.FOREGROUND
FIELD_LINE = (0.18, 0.72, 1.00, 0.22)
FIELD_ARROW = (0.25, 0.82, 1.00, 0.72)
GUIDING_LINE_ALPHA = 0.35


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset",
        type=Path,
        default=PROJECT_ROOT / "datasets" / "uniform_b_ensemble.npz",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Write MP4 instead of opening the interactive viewer.",
    )
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--duration-s", type=float, default=12.0)
    parser.add_argument("--hold-start-s", type=float, default=0.8)
    parser.add_argument("--hold-end-s", type=float, default=1.2)
    parser.add_argument("--window-width", type=int, default=3840)
    parser.add_argument("--window-height", type=int, default=2160)
    parser.add_argument("--camera-distance", type=float, default=15.5)
    parser.add_argument("--camera-azimuth", type=float, default=-66.0)
    parser.add_argument("--camera-elevation", type=float, default=15.0)
    parser.add_argument("--interval-ms", type=int, default=25)
    return parser.parse_args()


def _group_style(font_size_px: int = 17) -> str:
    return (
        "QGroupBox { color: #DCE6F8; border: 1px solid #34445E; "
        "border-radius: 8px; margin-top: 12px; "
        f"font-size: {font_size_px}px; font-weight: 600; }} "
        "QGroupBox::title { subcontrol-origin: margin; left: 9px; "
        "padding: 0 5px; }"
    )


def _build_physics_panel(qt_core, qt_widgets, metadata):
    container = qt_widgets.QWidget()
    layout = qt_widgets.QVBoxLayout(container)
    layout.setContentsMargins(0, 0, 0, 0)
    layout.setSpacing(7)

    field_group = qt_widgets.QGroupBox("Постоянное однородное поле")
    field_group.setStyleSheet(_group_style())
    field_layout = qt_widgets.QVBoxLayout(field_group)
    field_layout.setContentsMargins(12, 22, 12, 12)
    field_t = float(metadata["magnetic_field_t"])
    field_label = qt_widgets.QLabel(
        "<div style='font-size:25px; font-weight:600;'>"
        f"B = {field_t * 1e3:.3f} мТл &nbsp;&nbsp; ↑ +z</div>"
        "<div style='font-size:18px; margin-top:7px;'>"
        "E = 0 &nbsp;·&nbsp; K = 18,600 кэВ</div>"
    )
    field_label.setTextFormat(qt_core.Qt.TextFormat.RichText)
    field_label.setAlignment(qt_core.Qt.AlignmentFlag.AlignCenter)
    field_label.setStyleSheet(
        "color: #EAF4FF; background-color: #071526; "
        "border: 1px solid #268CCC; border-radius: 7px; padding: 13px;"
    )
    field_layout.addWidget(field_label)
    layout.addWidget(field_group)

    equation_group = qt_widgets.QGroupBox("Точное релятивистское решение")
    equation_group.setStyleSheet(_group_style())
    equation_layout = qt_widgets.QVBoxLayout(equation_group)
    equation_layout.setContentsMargins(12, 23, 12, 13)
    equation = qt_widgets.QLabel(
        "ω<sub>c</sub> = |q|B/(γm)<br>"
        "r<sub>L</sub> = p<sub>⊥</sub>/(|q|B)<br>"
        "z(t) = z₀ + v cos(α) t"
    )
    equation.setTextFormat(qt_core.Qt.TextFormat.RichText)
    equation.setAlignment(qt_core.Qt.AlignmentFlag.AlignCenter)
    equation.setStyleSheet(
        "color: #F1F5FF; font-size: 21px; line-height: 145%; "
        "background-color: #0A1220; border-radius: 6px; padding: 13px;"
    )
    equation_layout.addWidget(equation)
    invariant = qt_widgets.QLabel(
        "Магнитное поле не совершает работу:\n"
        "γ, |v| и кинетическая энергия постоянны"
    )
    invariant.setAlignment(qt_core.Qt.AlignmentFlag.AlignCenter)
    invariant.setStyleSheet(
        "color: #AFC4E2; font-size: 15px; padding-top: 7px;"
    )
    equation_layout.addWidget(invariant)
    layout.addWidget(equation_group)
    return container


def _build_track_cards(qt_core, qt_widgets, tracks):
    group = qt_widgets.QGroupBox("Три начальных направления")
    group.setStyleSheet(_group_style())
    layout = qt_widgets.QVBoxLayout(group)
    layout.setContentsMargins(10, 22, 10, 11)
    layout.setSpacing(8)
    cards = []
    for track in tracks:
        card = qt_widgets.QLabel()
        card.setTextFormat(qt_core.Qt.TextFormat.RichText)
        card.setMinimumHeight(72)
        card.setAlignment(qt_core.Qt.AlignmentFlag.AlignVCenter)
        card.setStyleSheet(
            "QLabel { color: #F4F7FF; background-color: #091525; "
            f"border: 1px solid {track['color']}; border-radius: 7px; "
            "font-size: 16px; padding: 8px 10px; }"
        )
        layout.addWidget(card)
        cards.append(card)
    return group, cards


def _field_arrow_segments(
    z_m: float,
    x_m: float,
    y_m: float,
) -> np.ndarray:
    tip = np.array([z_m, x_m, y_m])
    return np.asarray(
        [
            tip,
            tip + np.array([-0.42, +0.13, 0.0]),
            tip,
            tip + np.array([-0.42, -0.13, 0.0]),
        ],
        dtype=np.float32,
    )


def _add_uniform_field(view, gl, z_min: float, z_max: float) -> None:
    line_segments = []
    arrow_segments = []
    for x_m in np.linspace(-2.0, 2.0, 5):
        for y_m in (-0.65, 0.0, 0.65):
            line_segments.append(
                np.array(
                    [[z_min, x_m, y_m], [z_max, x_m, y_m]],
                    dtype=np.float32,
                )
            )
            if y_m == 0.0:
                for z_m in (-4.2, 0.0, 4.2):
                    arrow_segments.append(
                        _field_arrow_segments(z_m, x_m, y_m)
                    )
    field_lines = gl.GLLinePlotItem(
        pos=np.concatenate(line_segments),
        color=FIELD_LINE,
        width=1.1,
        mode="lines",
        antialias=True,
    )
    field_lines.setGLOptions("translucent")
    view.addItem(field_lines)
    arrows = gl.GLLinePlotItem(
        pos=np.concatenate(arrow_segments),
        color=FIELD_ARROW,
        width=2.2,
        mode="lines",
        antialias=True,
    )
    view.addItem(arrows)


def _add_field_label(view, gl, qt_core, qt_gui) -> None:
    font = qt_gui.QFont("Helvetica", 18)
    font.setBold(True)
    label = gl.GLTextItem(
        pos=np.array([4.7, -2.2, 1.05]),
        color=qt_gui.QColor("#54D9FF"),
        text="B = const   → +z",
        font=font,
        alignment=(
            qt_core.Qt.AlignmentFlag.AlignHCenter
            | qt_core.Qt.AlignmentFlag.AlignVCenter
        ),
    )
    view.addItem(label)


def main() -> None:
    args = parse_args()
    if args.fps <= 0:
        raise SystemExit("--fps must be positive")
    if args.window_width % 2 or args.window_height % 2:
        raise SystemExit("MP4 dimensions must be even")

    try:
        import pyqtgraph as pg
        import pyqtgraph.opengl as gl
        from pyqtgraph.Qt import QtCore, QtGui, QtWidgets
    except Exception as exc:
        raise SystemExit(f"pyqtgraph OpenGL/PyQt is unavailable: {exc}")

    surface_format = QtGui.QSurfaceFormat()
    surface_format.setDepthBufferSize(24)
    surface_format.setSamples(4)
    QtGui.QSurfaceFormat.setDefaultFormat(surface_format)

    dataset = load_trajectory_dataset(args.dataset)
    metadata = dataset["metadata"]
    if metadata.get("dataset_type") != "uniform_b_exact_ensemble":
        raise SystemExit("Expected an exact uniform-B ensemble dataset")
    time_s = np.asarray(dataset["time_s"], dtype=float)
    position_m = np.asarray(dataset["position_m"], dtype=float)
    guiding_center_m = np.asarray(
        dataset["guiding_center_m"],
        dtype=float,
    )
    tracks = list(metadata["tracks"])
    track_count = position_m.shape[0]
    scene_paths = video_helpers._scene_coordinates(position_m)
    scene_guiding_centers = video_helpers._scene_coordinates(
        guiding_center_m
    )

    pg.setConfigOption("background", BACKGROUND)
    pg.setConfigOption("foreground", FOREGROUND)
    pg.setConfigOption("antialias", True)
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])

    window = QtWidgets.QMainWindow()
    window.setWindowTitle("Электроны в постоянном магнитном поле")
    if args.out is not None:
        window.setWindowFlag(QtCore.Qt.WindowType.FramelessWindowHint, True)
    central = QtWidgets.QWidget()
    central.setStyleSheet(f"background-color: {BACKGROUND};")
    root = QtWidgets.QVBoxLayout(central)
    root.setContentsMargins(8, 7, 8, 34)
    root.setSpacing(4)
    window.setCentralWidget(central)

    title = QtWidgets.QLabel()
    title.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
    title.setMinimumHeight(96)
    title.setStyleSheet(
        f"color: {FOREGROUND}; font-size: 30px; font-weight: 600; "
        "padding: 4px;"
    )
    root.addWidget(title)

    method_caption = QtWidgets.QLabel(
        "<b>Точные винтовые траектории в однородном поле.</b> "
        "Все электроны имеют одинаковые энергию и физическое время; "
        "радиусы и координаты не увеличены."
    )
    method_caption.setTextFormat(QtCore.Qt.TextFormat.RichText)
    method_caption.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
    method_caption.setStyleSheet(
        "color: #D4E0F3; font-size: 18px; padding: 3px;"
    )
    root.addWidget(method_caption)

    content = QtWidgets.QWidget()
    content_layout = QtWidgets.QHBoxLayout(content)
    content_layout.setContentsMargins(0, 0, 0, 0)
    content_layout.setSpacing(8)

    sidebar = QtWidgets.QWidget()
    sidebar.setFixedWidth(500)
    sidebar_layout = QtWidgets.QVBoxLayout(sidebar)
    sidebar_layout.setContentsMargins(0, 0, 0, 0)
    sidebar_layout.setSpacing(7)
    sidebar_layout.addWidget(
        _build_physics_panel(QtCore, QtWidgets, metadata)
    )
    cards_group, cards = _build_track_cards(
        QtCore,
        QtWidgets,
        tracks,
    )
    sidebar_layout.addWidget(cards_group)
    sidebar_layout.addStretch(1)
    content_layout.addWidget(sidebar)

    view = gl.GLViewWidget()
    view.setBackgroundColor(BACKGROUND)
    view.opts["center"] = QtGui.QVector3D(-1.0, 0.0, 0.0)
    view.setCameraPosition(
        distance=args.camera_distance,
        elevation=args.camera_elevation,
        azimuth=args.camera_azimuth,
    )
    content_layout.addWidget(view, stretch=1)
    root.addWidget(content, stretch=1)

    z_min = float(np.min(position_m[:, :, 2])) - 0.5
    z_max = float(np.max(position_m[:, :, 2])) + 0.6
    _add_uniform_field(view, gl, z_min, z_max)
    _add_field_label(view, gl, QtCore, QtGui)

    colors = [
        tuple(float(value) for value in pg.mkColor(track["color"]).getRgbF())
        for track in tracks
    ]
    trails = []
    particles = []
    halos = []
    for track_index, color in enumerate(colors):
        guiding_line_color = (
            color[0],
            color[1],
            color[2],
            GUIDING_LINE_ALPHA,
        )
        guiding_line = gl.GLLinePlotItem(
            pos=scene_guiding_centers[track_index],
            color=guiding_line_color,
            width=1.3,
            mode="line_strip",
            antialias=True,
        )
        guiding_line.setGLOptions("translucent")
        view.addItem(guiding_line)
        trail = gl.GLLinePlotItem(
            pos=scene_paths[track_index, :1],
            color=color,
            width=3.2,
            mode="line_strip",
            antialias=True,
        )
        particle = gl.GLScatterPlotItem(
            pos=scene_paths[track_index, :1],
            color=color,
            size=15.0,
            pxMode=True,
        )
        halo = gl.GLScatterPlotItem(
            pos=scene_paths[track_index, :1],
            color=(color[0], color[1], color[2], 0.22),
            size=34.0,
            pxMode=True,
        )
        for item in (trail, halo, particle):
            view.addItem(item)
        trails.append(trail)
        particles.append(particle)
        halos.append(halo)

    legend = QtWidgets.QLabel(
        "α — угол между начальной скоростью и B"
        "&nbsp;&nbsp;&nbsp;&nbsp;"
        "одинаковое время ⇒ одинаковое число оборотов"
        "&nbsp;&nbsp;&nbsp;&nbsp;"
        "шаг винта ∝ cos α"
    )
    legend.setTextFormat(QtCore.Qt.TextFormat.RichText)
    legend.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
    legend.setMinimumHeight(56)
    legend.setStyleSheet(
        "color: #CAD4E7; font-size: 22px; padding: 8px 3px;"
    )
    root.addWidget(legend)

    controls = QtWidgets.QWidget()
    controls_layout = QtWidgets.QHBoxLayout(controls)
    controls_layout.setContentsMargins(0, 1, 0, 0)
    play_button = QtWidgets.QPushButton("Старт / пауза")
    slider = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
    slider.setRange(0, time_s.size - 1)
    controls_layout.addWidget(play_button)
    controls_layout.addWidget(slider, stretch=1)
    root.addWidget(controls)
    if args.out is not None:
        controls.hide()

    timer = QtCore.QTimer()
    timer.setInterval(max(1, args.interval_ms))

    def update_frame(index: int) -> None:
        index = int(np.clip(index, 0, time_s.size - 1))
        for track_index, track in enumerate(tracks):
            origin = scene_paths[track_index, index]
            trails[track_index].setData(
                pos=scene_paths[track_index, : index + 1]
            )
            particles[track_index].setData(pos=origin[None, :])
            halos[track_index].setData(pos=origin[None, :])
            cards[track_index].setText(
                f"<span style='color:{track['color']};'>●</span> "
                f"<b>{track['label']}</b>"
                f" &nbsp; r<sub>L</sub> = "
                f"{100.0 * track['larmor_radius_m']:.2f} см<br>"
                f"v<sub>∥</sub>/v = "
                f"{np.cos(np.deg2rad(track['pitch_angle_deg'])):.4f}"
                f" &nbsp;·&nbsp; z = {position_m[track_index, index, 2]:+.2f} м"
                f" &nbsp;·&nbsp; N = "
                f"{track['turns'] * index / (time_s.size - 1):.2f}"
            )
        title.setText(
            "Электроны в постоянном магнитном поле"
            f"<br><span style='font-size:20px; font-weight:400;'>"
            f"физическое время = {time_s[index] * 1e9:7.2f} нс"
            f" &nbsp;·&nbsp; "
            f"N = {tracks[0]['turns'] * index / (time_s.size - 1):.3f}"
            " оборота</span>"
        )
        slider.blockSignals(True)
        slider.setValue(index)
        slider.blockSignals(False)

    def advance() -> None:
        next_index = slider.value() + 1
        if next_index >= time_s.size:
            next_index = 0
        update_frame(next_index)

    def toggle_playback() -> None:
        if timer.isActive():
            timer.stop()
        else:
            timer.start()

    slider.valueChanged.connect(update_frame)
    play_button.clicked.connect(toggle_playback)
    timer.timeout.connect(advance)
    update_frame(0)

    if args.out is None:
        window.resize(1920, 1080)
    else:
        screen = app.primaryScreen()
        device_pixel_ratio = (
            float(screen.devicePixelRatio()) if screen is not None else 1.0
        )
        logical_width = max(
            1,
            int(round(args.window_width / device_pixel_ratio)),
        )
        logical_height = max(
            1,
            int(round(args.window_height / device_pixel_ratio)),
        )
        window.resize(logical_width, logical_height)
        window.move(0, 0)
    window.show()
    for _ in range(6):
        app.processEvents()

    if args.out is None:
        timer.start()
        app.exec()
        return

    try:
        import imageio.v2 as imageio
    except Exception as exc:
        raise SystemExit(f"imageio is required for MP4 export: {exc}")

    frame_indices = video_helpers._frame_indices(
        time_s.size,
        args.fps,
        args.duration_s,
        args.hold_start_s,
        args.hold_end_s,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    writer = imageio.get_writer(
        args.out,
        fps=args.fps,
        codec="libx264",
        quality=9,
        macro_block_size=None,
        pixelformat="yuv420p",
        ffmpeg_log_level="warning",
    )
    native_size_reported = False
    try:
        for frame_number, sample_index in enumerate(frame_indices):
            update_frame(int(sample_index))
            view.update()
            app.processEvents()
            app.processEvents()
            frame_image = window.grab().toImage()
            if not native_size_reported:
                print(
                    "Qt native capture: "
                    f"{frame_image.width()}x{frame_image.height()}; "
                    f"requested: {args.window_width}x{args.window_height}",
                    flush=True,
                )
                native_size_reported = True
            if (
                frame_image.width() != args.window_width
                or frame_image.height() != args.window_height
            ):
                frame_image = frame_image.scaled(
                    args.window_width,
                    args.window_height,
                    QtCore.Qt.AspectRatioMode.IgnoreAspectRatio,
                    QtCore.Qt.TransformationMode.SmoothTransformation,
                )
            writer.append_data(
                video_helpers._qimage_to_rgb_array(frame_image, QtGui)
            )
            if frame_number % max(args.fps, 1) == 0:
                print(
                    f"Rendered {frame_number + 1}/{frame_indices.size} frames",
                    flush=True,
                )
    finally:
        writer.close()
        window.close()

    print(f"Saved exact uniform-B animation: {args.out}")
    print(
        f"resolution={args.window_width}x{args.window_height}, "
        f"frames={frame_indices.size}, fps={args.fps}"
    )
    print(
        "tracks="
        + ", ".join(
            f"{track['label']}:rL={track['larmor_radius_m']:.6g}m"
            for track in tracks
        )
    )


if __name__ == "__main__":
    main()
