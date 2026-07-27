#!/usr/bin/env python3
"""Render KATRIN MAC-E transmission and reflection with electric retardation."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from mac_e_filter.data_io import load_trajectory_dataset  # noqa: E402
from mac_e_filter.katrin_nominal import build_katrin_2013_field  # noqa: E402

import pyqtgraph_katrin_source_to_detector_3d as source_scene  # noqa: E402
import pyqtgraph_katrin_spectrometer_3d as scene  # noqa: E402


BACKGROUND = scene.BACKGROUND
FOREGROUND = scene.FOREGROUND
ELECTRODE = (1.00, 0.29, 0.22, 0.72)
ELECTRODE_EDGE = (1.00, 0.57, 0.28, 0.95)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset",
        type=Path,
        default=PROJECT_ROOT / "datasets" / "katrin_2013_mac_e.npz",
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
    parser.add_argument("--camera-distance", type=float, default=32.0)
    parser.add_argument("--camera-azimuth", type=float, default=-48.0)
    parser.add_argument("--camera-elevation", type=float, default=10.0)
    parser.add_argument("--interval-ms", type=int, default=25)
    return parser.parse_args()


def _group_style(font_size_px: int = 16) -> str:
    return (
        "QGroupBox { color: #DCE6F8; border: 1px solid #34445E; "
        "border-radius: 7px; margin-top: 11px; "
        f"font-size: {font_size_px}px; font-weight: 600; }} "
        "QGroupBox::title { subcontrol-origin: margin; left: 8px; "
        "padding: 0 4px; }"
    )


def _configure_plot(plot, qt_gui, *, left_label: str, units: str) -> None:
    plot.setBackground("#07101D")
    plot.getPlotItem().setMenuEnabled(False)
    plot.setMouseEnabled(x=False, y=False)
    plot.showGrid(x=True, y=True, alpha=0.20)
    label_style = {"font-size": "13pt"}
    plot.setLabel("bottom", "z", units="м", **label_style)
    plot.setLabel("left", left_label, units=units, **label_style)
    tick_font = qt_gui.QFont("Helvetica", 12)
    plot.getAxis("bottom").setStyle(tickFont=tick_font)
    plot.getAxis("left").setStyle(tickFont=tick_font)


def _build_field_and_potential_profiles(
    pg,
    qt_core,
    qt_gui,
    qt_widgets,
    dataset,
):
    container = qt_widgets.QWidget()
    layout = qt_widgets.QVBoxLayout(container)
    layout.setContentsMargins(0, 0, 0, 0)
    layout.setSpacing(5)

    z = np.asarray(dataset["field_line_z_m"], dtype=float)
    magnetic_field = np.asarray(
        dataset["axis_field_magnitude_t"],
        dtype=float,
    )
    path_potential = np.asarray(
        dataset["guiding_path_electric_potential_v"],
        dtype=float,
    )
    metadata = dataset["metadata"]

    magnetic_group = qt_widgets.QGroupBox(
        "Магнитное поле: сильное → слабое → сильное"
    )
    magnetic_group.setStyleSheet(_group_style())
    magnetic_layout = qt_widgets.QVBoxLayout(magnetic_group)
    magnetic_layout.setContentsMargins(7, 18, 7, 7)
    magnetic_plot = pg.PlotWidget()
    magnetic_plot.setFixedHeight(178)
    _configure_plot(
        magnetic_plot,
        qt_gui,
        left_label="|B|",
        units="Тл",
    )
    magnetic_plot.setLogMode(y=True)
    magnetic_plot.plot(
        z,
        magnetic_field,
        pen=pg.mkPen("#FFD166", width=2.2),
    )
    magnetic_cursor = pg.InfiniteLine(
        pos=float(z[0]),
        angle=90,
        movable=False,
        pen=pg.mkPen("#5DEBFF", width=2.0),
    )
    magnetic_plot.addItem(magnetic_cursor)
    magnetic_plot.setXRange(float(z[0]), float(z[-1]), padding=0.0)
    magnetic_plot.setYRange(-4.0, 1.0, padding=0.02)
    magnetic_layout.addWidget(magnetic_plot)
    layout.addWidget(magnetic_group)

    potential_group = qt_widgets.QGroupBox(
        "Электростатический барьер на траектории"
    )
    potential_group.setStyleSheet(_group_style())
    potential_layout = qt_widgets.QVBoxLayout(potential_group)
    potential_layout.setContentsMargins(7, 18, 7, 7)
    potential_plot = pg.PlotWidget()
    potential_plot.setFixedHeight(178)
    _configure_plot(
        potential_plot,
        qt_gui,
        left_label="−ΔΦ",
        units="кВ",
    )
    source_potential = float(path_potential[0])
    retarding_kev = -(path_potential - source_potential) / 1000.0
    potential_plot.plot(
        z,
        retarding_kev,
        pen=pg.mkPen("#FF665C", width=2.6),
        fillLevel=0.0,
        brush=(255, 78, 68, 30),
    )
    source_energy_kev = float(metadata["kinetic_energy_ev"]) / 1000.0
    potential_plot.addLine(
        y=source_energy_kev,
        pen=pg.mkPen("#F7F8FF", width=1.4, style=qt_core.Qt.PenStyle.DashLine),
    )
    potential_cursor = pg.InfiniteLine(
        pos=float(z[0]),
        angle=90,
        movable=False,
        pen=pg.mkPen("#5DEBFF", width=2.0),
    )
    potential_plot.addItem(potential_cursor)
    potential_plot.setXRange(float(z[0]), float(z[-1]), padding=0.0)
    potential_plot.setYRange(0.0, 19.3, padding=0.02)
    potential_layout.addWidget(potential_plot)
    potential_note = qt_widgets.QLabel(
        "красный: qΔΦ для электрона · белый пунктир: K₀ = 18,600 кэВ\n"
        "постускоряющее напряжение FPD не включено"
    )
    potential_note.setAlignment(qt_core.Qt.AlignmentFlag.AlignCenter)
    potential_note.setStyleSheet(
        "color: #C8D4E8; font-size: 13px; padding-top: 2px;"
    )
    potential_layout.addWidget(potential_note)
    layout.addWidget(potential_group)
    return container, magnetic_cursor, potential_cursor


def _build_track_cards(qt_core, qt_widgets, tracks_metadata):
    group = qt_widgets.QGroupBox("Электроны: энергия и результат")
    group.setStyleSheet(_group_style())
    layout = qt_widgets.QVBoxLayout(group)
    layout.setContentsMargins(9, 19, 9, 9)
    layout.setSpacing(7)
    cards = []
    for track in tracks_metadata:
        label = qt_widgets.QLabel()
        label.setTextFormat(qt_core.Qt.TextFormat.RichText)
        label.setMinimumHeight(49)
        label.setAlignment(qt_core.Qt.AlignmentFlag.AlignVCenter)
        border = (
            "#35D5FF"
            if track["outcome"] == "transmitted"
            else "#FFB347"
        )
        label.setStyleSheet(
            "QLabel { color: #F1F5FF; background-color: #0A1525; "
            f"border: 1px solid {border}; border-radius: 6px; "
            "font-size: 14px; padding: 6px 8px; }"
        )
        layout.addWidget(label)
        cards.append(label)
    note = qt_widgets.QLabel(
        "ПРОШЁЛ: пересёк z=0 и достиг FPD\n"
        "ОТРАЖЁН: p∥=0, затем движение к источнику"
    )
    note.setAlignment(qt_core.Qt.AlignmentFlag.AlignCenter)
    note.setStyleSheet(
        "color: #AEBAD0; font-size: 13px; padding-top: 3px;"
    )
    layout.addWidget(note)
    return group, cards


def _add_inner_electrode_rings(view, gl) -> None:
    axial_positions = np.linspace(-9.7, 9.7, 17)
    all_segments = []
    for axial_position in axial_positions:
        radius = 0.90 * float(
            scene._vessel_radius(np.array([axial_position]))[0]
        )
        all_segments.append(
            scene._circle_segments(
                np.array([axial_position]),
                radius,
                samples=112,
            )
        )
    electrodes = gl.GLLinePlotItem(
        pos=np.concatenate(all_segments),
        color=ELECTRODE,
        width=1.8,
        mode="lines",
        antialias=True,
    )
    electrodes.setGLOptions("translucent")
    view.addItem(electrodes)


def _add_reflection_markers(view, gl, dataset, tracks_metadata, colors):
    markers = []
    for track_index, track in enumerate(tracks_metadata):
        if track["outcome"] != "reflected":
            markers.append(None)
            continue
        trajectory = np.asarray(
            dataset["guiding_center_m"][track_index],
            dtype=float,
        )
        turning_index = int(np.argmax(trajectory[:, 2]))
        marker_position = source_scene._scene_coordinates(
            trajectory[turning_index : turning_index + 1]
        )
        marker = gl.GLScatterPlotItem(
            pos=marker_position,
            color=colors[track_index],
            size=18.0,
            pxMode=True,
        )
        marker.hide()
        view.addItem(marker)
        markers.append(marker)
    return markers


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
    if metadata.get("dataset_type") != "katrin_mac_e_adiabatic_electrostatic":
        raise SystemExit("Expected a KATRIN electrostatic MAC-E dataset")

    configuration = str(metadata["field_configuration"])
    build_katrin_2013_field(configuration)
    time_s = np.asarray(dataset["time_s"], dtype=float)
    guiding_center_m = np.asarray(dataset["guiding_center_m"], dtype=float)
    electric_potential_v = np.asarray(
        dataset["electric_potential_v"],
        dtype=float,
    )
    kinetic_energy_ev = np.asarray(dataset["kinetic_energy_ev"], dtype=float)
    pitch_deg = np.asarray(dataset["pitch_angle_deg"], dtype=float)
    parallel_speed = np.asarray(dataset["parallel_speed_m_s"], dtype=float)
    perpendicular_speed = np.asarray(
        dataset["perpendicular_speed_m_s"],
        dtype=float,
    )
    larmor_radius_m = np.asarray(dataset["larmor_radius_m"], dtype=float)
    cumulative_turns = np.asarray(dataset["cumulative_turns"], dtype=float)
    active = np.asarray(dataset["active"], dtype=bool)
    tracks_metadata = list(metadata["tracks"])
    track_count = guiding_center_m.shape[0]
    scene_paths = source_scene._scene_coordinates(guiding_center_m)

    terminal_indices = []
    for track_index in range(track_count):
        first_inactive = np.flatnonzero(~active[track_index])
        terminal_indices.append(
            int(first_inactive[0])
            if first_inactive.size
            else time_s.size - 1
        )
    terminal_indices = np.asarray(terminal_indices, dtype=int)

    pg.setConfigOption("background", BACKGROUND)
    pg.setConfigOption("foreground", FOREGROUND)
    pg.setConfigOption("antialias", True)
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])

    window = QtWidgets.QMainWindow()
    window.setWindowTitle("KATRIN MAC-E: электрический барьер")
    if args.out is not None:
        window.setWindowFlag(QtCore.Qt.WindowType.FramelessWindowHint, True)
    central = QtWidgets.QWidget()
    central.setStyleSheet(f"background-color: {BACKGROUND};")
    root = QtWidgets.QVBoxLayout(central)
    root.setContentsMargins(8, 6, 8, 34)
    root.setSpacing(3)
    window.setCentralWidget(central)

    title = QtWidgets.QLabel()
    title.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
    title.setMinimumHeight(102)
    title.setSizePolicy(
        QtWidgets.QSizePolicy.Policy.Expanding,
        QtWidgets.QSizePolicy.Policy.Fixed,
    )
    title.setStyleSheet(
        f"color: {FOREGROUND}; font-size: 28px; font-weight: 600; "
        "padding: 4px;"
    )
    root.addWidget(title)

    electric_model = metadata["electrostatic_model"]
    retarding_label = (
        f"{abs(electric_model['retarding_difference_v']):,.1f}"
        .replace(",", " ")
        .replace(".", ",")
    )
    method_caption = QtWidgets.QLabel(
        "<b>Электронный барьер: qΔΦ = "
        f"{retarding_label} эВ.</b> "
        "Сохраняются γmc²+qΦ и p<sub>⊥</sub>²/B; "
        "красные кольца — цилиндрический лапласовский суррогат электродов."
    )
    method_caption.setTextFormat(QtCore.Qt.TextFormat.RichText)
    method_caption.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
    method_caption.setStyleSheet(
        "color: #D4E0F3; font-size: 17px; padding: 3px;"
    )
    root.addWidget(method_caption)

    content = QtWidgets.QWidget()
    content_layout = QtWidgets.QHBoxLayout(content)
    content_layout.setContentsMargins(0, 0, 0, 0)
    content_layout.setSpacing(7)

    sidebar = QtWidgets.QWidget()
    sidebar.setFixedWidth(470)
    sidebar_layout = QtWidgets.QVBoxLayout(sidebar)
    sidebar_layout.setContentsMargins(0, 0, 0, 0)
    sidebar_layout.setSpacing(5)
    (
        profile_widget,
        magnetic_cursor,
        potential_cursor,
    ) = _build_field_and_potential_profiles(
        pg,
        QtCore,
        QtGui,
        QtWidgets,
        dataset,
    )
    sidebar_layout.addWidget(profile_widget)
    track_group, track_cards = _build_track_cards(
        QtCore,
        QtWidgets,
        tracks_metadata,
    )
    sidebar_layout.addWidget(track_group)
    sidebar_layout.addStretch(1)
    content_layout.addWidget(sidebar)

    view = gl.GLViewWidget()
    view.setBackgroundColor(BACKGROUND)
    view.opts["center"] = QtGui.QVector3D(2.0, 0.0, -4.0)
    view.setCameraPosition(
        distance=args.camera_distance,
        elevation=args.camera_elevation,
        azimuth=args.camera_azimuth,
    )
    content_layout.addWidget(view, stretch=1)
    root.addWidget(content, stretch=1)

    scene._add_vessel_wireframe(view, gl)
    _add_inner_electrode_rings(view, gl)
    scene._add_lfcs_coils(view, gl, configuration)
    scene._add_analyzing_plane(view, gl, QtCore, QtGui)
    scene._add_segmented_detector(view, gl, QtCore, QtGui)
    source_scene._add_superconducting_magnets(
        view,
        gl,
        QtCore,
        QtGui,
    )
    source_scene._add_flux_tube(
        view,
        gl,
        np.asarray(dataset["field_line_z_m"], dtype=float),
        np.asarray(dataset["flux_radius_m"], dtype=float),
    )
    beam_axis = gl.GLLinePlotItem(
        pos=np.array(
            [[-40.0, 0.0, 0.0], [14.2, 0.0, 0.0]],
            dtype=np.float32,
        ),
        color=scene.AXIS,
        width=1.0,
        mode="lines",
        antialias=True,
    )
    beam_axis.setGLOptions("translucent")
    view.addItem(beam_axis)

    colors = [
        tuple(float(v) for v in pg.mkColor(track["color"]).getRgbF())
        for track in tracks_metadata
    ]
    trails = []
    halos = []
    particles = []
    for track_index, color in enumerate(colors):
        trail = gl.GLLinePlotItem(
            pos=scene_paths[track_index, :1],
            color=color,
            width=3.0,
            mode="line_strip",
            antialias=True,
        )
        halo = gl.GLScatterPlotItem(
            pos=scene_paths[track_index, :1],
            color=(color[0], color[1], color[2], 0.20),
            size=30.0,
            pxMode=True,
        )
        particle = gl.GLScatterPlotItem(
            pos=scene_paths[track_index, :1],
            color=color,
            size=14.0,
            pxMode=True,
        )
        for item in (trail, halo, particle):
            view.addItem(item)
        trails.append(trail)
        halos.append(halo)
        particles.append(particle)

    detector_hits = source_scene._add_detector_hits(
        view,
        gl,
        guiding_center_m,
        colors,
    )
    reflection_markers = _add_reflection_markers(
        view,
        gl,
        dataset,
        tracks_metadata,
        colors,
    )

    legend = QtWidgets.QLabel(
        "<span style='color:#52E3A4;'>●</span> и "
        "<span style='color:#35D5FF;'>●</span> прошли к FPD"
        "&nbsp;&nbsp;&nbsp;&nbsp;"
        "<span style='color:#FFB347;'>●</span> отражён при p<sub>∥</sub>=0"
        "&nbsp;&nbsp;&nbsp;&nbsp;"
        "<span style='color:#5DEBFF;'>●</span> анализирующая плоскость"
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
        reference_frame = min(index, terminal_indices[1])
        cursor_z = float(guiding_center_m[1, reference_frame, 2])
        magnetic_cursor.setValue(cursor_z)
        potential_cursor.setValue(cursor_z)
        status_parts = []
        for track_index, track in enumerate(tracks_metadata):
            frame = min(index, terminal_indices[track_index])
            origin = scene_paths[track_index, frame]
            trails[track_index].setData(
                pos=scene_paths[track_index, : frame + 1]
            )
            halos[track_index].setData(pos=origin[None, :])
            particles[track_index].setData(pos=origin[None, :])

            ended = index >= terminal_indices[track_index]
            transmitted = track["outcome"] == "transmitted"
            detector_hits[track_index].setVisible(ended and transmitted)
            particles[track_index].setVisible(not (ended and transmitted))
            halos[track_index].setVisible(not (ended and transmitted))

            if reflection_markers[track_index] is not None:
                turning_reached = (
                    np.max(
                        guiding_center_m[
                            track_index,
                            : frame + 1,
                            2,
                        ]
                    )
                    >= float(track["turning_z_m"]) - 1.0e-4
                )
                reflection_markers[track_index].setVisible(turning_reached)

            speed_parallel = float(parallel_speed[track_index, frame])
            speed_perpendicular = float(
                perpendicular_speed[track_index, frame]
            )
            speed = np.hypot(speed_parallel, speed_perpendicular)
            direction = "→" if speed_parallel >= 0.0 else "←"
            terminal_text = (
                "ПРОШЁЛ"
                if transmitted
                else (
                    "ОТРАЖЁН"
                    if speed_parallel < 0.0 or ended
                    else "К БАРЬЕРУ"
                )
            )
            track_cards[track_index].setText(
                f"<span style='color:{track['color']};'>●</span> "
                f"<b>{track['label']}</b> &nbsp; {terminal_text}<br>"
                f"K = {kinetic_energy_ev[track_index, frame]:.3f} эВ"
                f" &nbsp;·&nbsp; α = {pitch_deg[track_index, frame]:.3f}°"
                f" &nbsp;·&nbsp; v<sub>∥</sub>/v = "
                f"{speed_parallel / speed:+.4f} {direction}"
            )
            status_parts.append(
                f"<span style='color:{track['color']};'>"
                f"{track['label']}: z="
                f"{guiding_center_m[track_index, frame, 2]:+.2f} м, "
                f"K={kinetic_energy_ev[track_index, frame]:.3f} эВ, "
                f"N={cumulative_turns[track_index, frame]:.0f}</span>"
            )

        title.setText(
            "KATRIN MAC-E: электростатическое прохождение и отражение"
            f"<br><span style='font-size:18px; font-weight:400;'>"
            f"физическое время = {time_s[index] * 1e6:7.3f} мкс"
            "<br>"
            + "&nbsp;&nbsp; | &nbsp;&nbsp;".join(status_parts)
            + "</span>"
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

    frame_indices = source_scene._frame_indices(
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
                source_scene._qimage_to_rgb_array(frame_image, QtGui)
            )
            if frame_number % max(args.fps, 1) == 0:
                print(
                    f"Rendered {frame_number + 1}/{frame_indices.size} frames",
                    flush=True,
                )
    finally:
        writer.close()
        window.close()

    print(f"Saved 4K electrostatic MAC-E animation: {args.out}")
    print(
        f"resolution={args.window_width}x{args.window_height}, "
        f"frames={frame_indices.size}, fps={args.fps}"
    )
    print(
        "outcomes="
        + ", ".join(
            f"{track['label']}:{track['outcome']}"
            for track in tracks_metadata
        )
    )
    print(
        "energy_residual_eV="
        + ", ".join(
            f"{track['total_energy_residual_ev']:.3g}"
            for track in tracks_metadata
        )
    )


if __name__ == "__main__":
    main()
