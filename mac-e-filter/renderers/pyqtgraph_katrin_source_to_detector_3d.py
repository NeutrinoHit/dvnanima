#!/usr/bin/env python3
"""Render source-to-detector KATRIN guiding-centre transport in 3D."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from mac_e_filter.data_io import load_trajectory_dataset  # noqa: E402
from mac_e_filter.katrin_nominal import (  # noqa: E402
    SUPERCONDUCTING_SOURCES_2013,
    build_katrin_2013_field,
)

import pyqtgraph_katrin_spectrometer_3d as scene  # noqa: E402


BACKGROUND = scene.BACKGROUND
FOREGROUND = scene.FOREGROUND
FLUX_TUBE = (0.26, 0.64, 1.00, 0.22)
STRONG_MAGNET = (0.66, 0.28, 1.00, 0.72)
STRONG_MAGNET_EDGE = (0.81, 0.58, 1.00, 0.95)
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset",
        type=Path,
        default=PROJECT_ROOT
        / "datasets"
        / "katrin_2013_source_to_detector.npz",
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


def _qimage_to_rgb_array(image, qt_gui) -> np.ndarray:
    converted = image.convertToFormat(qt_gui.QImage.Format.Format_RGBA8888)
    pointer = converted.bits()
    try:
        pointer.setsize(converted.sizeInBytes())
    except AttributeError:
        pass
    rgba = np.frombuffer(
        pointer,
        dtype=np.uint8,
        count=converted.sizeInBytes(),
    ).reshape(converted.height(), converted.width(), 4)
    return np.ascontiguousarray(rgba[:, :, :3])


def _frame_indices(
    sample_count: int,
    fps: int,
    duration_s: float,
    hold_start_s: float,
    hold_end_s: float,
) -> np.ndarray:
    moving_count = max(2, int(round(duration_s * fps)))
    moving = np.rint(
        np.linspace(0, sample_count - 1, moving_count)
    ).astype(np.int32)
    hold_start = np.zeros(
        max(0, int(round(hold_start_s * fps))),
        dtype=np.int32,
    )
    hold_end = np.full(
        max(0, int(round(hold_end_s * fps))),
        sample_count - 1,
        dtype=np.int32,
    )
    return np.concatenate((hold_start, moving, hold_end))


def _scene_coordinates(position_m: np.ndarray) -> np.ndarray:
    """Map physical (x, y, z) to scene (z, x, y)."""

    return np.ascontiguousarray(
        np.stack(
            (position_m[..., 2], position_m[..., 0], position_m[..., 1]),
            axis=-1,
        ),
        dtype=np.float32,
    )


def _add_superconducting_magnets(view, gl, qt_core, qt_gui) -> None:
    """Show published locations and field values, not undocumented currents."""

    label_names = {"WGTS", "PS2", "PCH", "DET"}
    font = qt_gui.QFont("Helvetica", 14)
    font.setBold(True)
    for source in SUPERCONDUCTING_SOURCES_2013:
        short_name = source.name.split()[0]
        radius = 0.62 if source.center_z_m < -11.7 else 0.92
        axial_positions = source.center_z_m + np.linspace(-0.22, 0.22, 5)
        rings = gl.GLLinePlotItem(
            pos=scene._circle_segments(
                axial_positions,
                radius,
                samples=96,
            ),
            color=STRONG_MAGNET,
            width=2.4,
            mode="lines",
            antialias=True,
        )
        rings.setGLOptions("translucent")
        view.addItem(rings)
        edge = gl.GLLinePlotItem(
            pos=scene._circle_segments(
                np.array([source.center_z_m]),
                radius + 0.05,
                samples=96,
            ),
            color=STRONG_MAGNET_EDGE,
            width=2.8,
            mode="lines",
            antialias=True,
        )
        view.addItem(edge)
        if short_name in label_names:
            label = gl.GLTextItem(
                pos=np.array(
                    [source.center_z_m, -radius - 0.18, radius + 0.30]
                ),
                color=qt_gui.QColor("#D5B7FF"),
                text=(
                    f"{short_name}  "
                    f"{source.typical_max_field_t:g} Тл"
                ),
                font=font,
                alignment=(
                    qt_core.Qt.AlignmentFlag.AlignHCenter
                    | qt_core.Qt.AlignmentFlag.AlignVCenter
                ),
            )
            view.addItem(label)


def _add_flux_tube(
    view,
    gl,
    field_line_z_m: np.ndarray,
    flux_radius_m: np.ndarray,
) -> None:
    stride = max(1, field_line_z_m.size // 4000)
    z = field_line_z_m[::stride]
    radius = flux_radius_m[::stride]
    for phi in np.linspace(0.0, 2.0 * np.pi, 10, endpoint=False):
        line = np.column_stack(
            (
                z,
                radius * np.cos(phi),
                radius * np.sin(phi),
            )
        ).astype(np.float32)
        item = gl.GLLinePlotItem(
            pos=line,
            color=FLUX_TUBE,
            width=1.0,
            mode="line_strip",
            antialias=True,
        )
        item.setGLOptions("translucent")
        view.addItem(item)

    cross_sections = (-38.87, -12.10, 0.0, 12.18, 13.78)
    for axial_position in cross_sections:
        radius_here = float(
            np.interp(axial_position, field_line_z_m, flux_radius_m)
        )
        ring = gl.GLLinePlotItem(
            pos=scene._circle_segments(
                np.array([axial_position]),
                radius_here,
                samples=112,
            ),
            color=(0.26, 0.70, 1.00, 0.42),
            width=1.2,
            mode="lines",
            antialias=True,
        )
        ring.setGLOptions("translucent")
        view.addItem(ring)


def _build_field_profile(pg, qt_core, qt_gui, qt_widgets, dataset):
    widget = qt_widgets.QGroupBox("Поле на оси: сильное → слабое → сильное")
    widget.setStyleSheet(
        "QGroupBox { color: #DCE6F8; border: 1px solid #34445E; "
        "border-radius: 7px; margin-top: 10px; font-size: 15px; "
        "font-weight: 600; } "
        "QGroupBox::title { subcontrol-origin: margin; left: 8px; "
        "padding: 0 4px; }"
    )
    layout = qt_widgets.QVBoxLayout(widget)
    layout.setContentsMargins(7, 17, 7, 8)
    layout.setSpacing(4)
    plot = pg.PlotWidget()
    plot.setFixedHeight(224)
    plot.setBackground("#07101D")
    plot.getPlotItem().setMenuEnabled(False)
    plot.setMouseEnabled(x=False, y=False)
    plot.showGrid(x=True, y=True, alpha=0.20)
    label_style = {"font-size": "13pt"}
    plot.setLabel("bottom", "z", units="м", **label_style)
    plot.setLabel("left", "|B|", units="Тл", **label_style)
    tick_font = qt_gui.QFont("Helvetica", 12)
    plot.getAxis("bottom").setStyle(tickFont=tick_font)
    plot.getAxis("left").setStyle(tickFont=tick_font)
    plot.setLogMode(y=True)

    z = np.asarray(dataset["field_line_z_m"], dtype=float)
    field = np.asarray(dataset["axis_field_magnitude_t"], dtype=float)
    plot.plot(z, field, pen=pg.mkPen("#FFD166", width=2.0))
    plot.plot(
        np.array([-38.87, 0.0, 12.18, 13.78]),
        np.interp(np.array([-38.87, 0.0, 12.18, 13.78]), z, field),
        pen=None,
        symbol="o",
        symbolSize=7,
        symbolBrush="#FF6B6B",
    )
    cursor = pg.InfiniteLine(
        pos=-38.87,
        angle=90,
        movable=False,
        pen=pg.mkPen("#5DEBFF", width=2.0),
    )
    plot.addItem(cursor)
    plot.setXRange(float(z[0]), float(z[-1]), padding=0.0)
    plot.setYRange(-4.0, 1.0, padding=0.02)
    layout.addWidget(plot)
    caption = qt_widgets.QLabel(
        "WGTS 3,6 Тл → анализ 0,347 мТл → PCH 6 Тл\n"
        "токи LFCS ниже — только тонкая коррекция слабого поля"
    )
    caption.setAlignment(qt_core.Qt.AlignmentFlag.AlignCenter)
    caption.setStyleSheet(
        "color: #C8D4E8; font-size: 13px; padding-top: 3px;"
    )
    layout.addWidget(caption)
    return widget, cursor


def _add_detector_hits(view, gl, guiding_center_m: np.ndarray, colors):
    hits = []
    for track_index, color in enumerate(colors):
        final = guiding_center_m[track_index, -1]
        display_position = np.array(
            [
                final[2],
                scene.DETECTOR_DISPLAY_SCALE * final[0],
                scene.DETECTOR_DISPLAY_SCALE * final[1],
            ],
            dtype=np.float32,
        )
        hit = gl.GLScatterPlotItem(
            pos=display_position[None, :],
            color=color,
            size=17.0,
            pxMode=True,
        )
        hit.hide()
        view.addItem(hit)
        hits.append(hit)
    return hits


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
    if metadata.get("dataset_type") != "katrin_source_to_detector_adiabatic":
        raise SystemExit("Expected a source-to-detector adiabatic dataset")
    configuration = str(metadata["field_configuration"])
    field_model = build_katrin_2013_field(configuration)
    time_s = np.asarray(dataset["time_s"], dtype=float)
    guiding_center_m = np.asarray(dataset["guiding_center_m"], dtype=float)
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
    scene_paths = _scene_coordinates(guiding_center_m)
    arrival_indices = []
    for track_index in range(track_count):
        first_inactive = np.flatnonzero(~active[track_index])
        if first_inactive.size:
            # Resampling uses endpoint-hold interpolation, so the first
            # sample after the physical flight time is exactly on the FPD.
            arrival_indices.append(int(first_inactive[0]))
        else:
            arrival_indices.append(time_s.size - 1)
    arrival_indices = np.asarray(arrival_indices, dtype=int)

    pg.setConfigOption("background", BACKGROUND)
    pg.setConfigOption("foreground", FOREGROUND)
    pg.setConfigOption("antialias", True)
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])

    window = QtWidgets.QMainWindow()
    window.setWindowTitle("KATRIN: от сильного поля источника до детектора")
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
    title.setMinimumHeight(98)
    title.setSizePolicy(
        QtWidgets.QSizePolicy.Policy.Expanding,
        QtWidgets.QSizePolicy.Policy.Fixed,
    )
    title.setStyleSheet(
        f"color: {FOREGROUND}; font-size: 27px; font-weight: 600; "
        "padding: 4px;"
    )
    root.addWidget(title)

    method_caption = QtWidgets.QLabel(
        "<b>Расчёт начинается в WGTS: B = 3,603 Тл.</b> "
        "Цветные линии — адиабатические центры орбит; "
        "r<sub>L</sub> и доли скорости приведены в физических единицах."
    )
    method_caption.setTextFormat(QtCore.Qt.TextFormat.RichText)
    method_caption.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
    method_caption.setStyleSheet(
        "color: #D4E0F3; font-size: 16px; padding: 2px;"
    )
    root.addWidget(method_caption)

    content = QtWidgets.QWidget()
    content_layout = QtWidgets.QHBoxLayout(content)
    content_layout.setContentsMargins(0, 0, 0, 0)
    content_layout.setSpacing(7)

    sidebar = QtWidgets.QWidget()
    sidebar.setFixedWidth(420)
    sidebar_layout = QtWidgets.QVBoxLayout(sidebar)
    sidebar_layout.setContentsMargins(0, 0, 0, 0)
    sidebar_layout.setSpacing(4)
    field_widget, field_cursor = _build_field_profile(
        pg,
        QtCore,
        QtGui,
        QtWidgets,
        dataset,
    )
    sidebar_layout.addWidget(field_widget)
    sidebar_layout.addWidget(
        scene._build_current_table(QtCore, QtWidgets, configuration)
    )
    (
        velocity_gauges,
        parallel_bars,
        perpendicular_bars,
    ) = scene._build_velocity_gauges(
        QtCore,
        QtWidgets,
        tracks_metadata,
    )
    sidebar_layout.addWidget(velocity_gauges)
    sidebar_layout.addStretch(1)
    content_layout.addWidget(sidebar)

    view = gl.GLViewWidget()
    view.setBackgroundColor(BACKGROUND)
    # The slight vertical offset keeps the long, inclined beamline centred
    # in the available viewport while retaining a detector-facing azimuth.
    view.opts["center"] = QtGui.QVector3D(2.0, 0.0, -4.0)
    view.setCameraPosition(
        distance=args.camera_distance,
        elevation=args.camera_elevation,
        azimuth=args.camera_azimuth,
    )
    content_layout.addWidget(view, stretch=1)
    root.addWidget(content, stretch=1)

    scene._add_vessel_wireframe(view, gl)
    scene._add_lfcs_coils(view, gl, configuration)
    scene._add_analyzing_plane(view, gl, QtCore, QtGui)
    scene._add_segmented_detector(view, gl, QtCore, QtGui)
    _add_superconducting_magnets(view, gl, QtCore, QtGui)
    _add_flux_tube(
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
            width=2.8,
            mode="line_strip",
            antialias=True,
        )
        halo = gl.GLScatterPlotItem(
            pos=scene_paths[track_index, :1],
            color=(color[0], color[1], color[2], 0.20),
            size=29.0,
            pxMode=True,
        )
        particle = gl.GLScatterPlotItem(
            pos=scene_paths[track_index, :1],
            color=color,
            size=13.0,
            pxMode=True,
        )
        for item in (trail, halo, particle):
            view.addItem(item)
        trails.append(trail)
        halos.append(halo)
        particles.append(particle)
    detector_hits = _add_detector_hits(
        view,
        gl,
        guiding_center_m,
        colors,
    )

    legend = QtWidgets.QLabel(
        "α — угол между импульсом электрона и линией B"
        "&nbsp;&nbsp;&nbsp;&nbsp;"
        "<span style='color:#5DEBFF;'>●</span> анализирующая плоскость"
        "&nbsp;&nbsp;"
        "<span style='color:#FFE66D;'>◉</span> FPD ×35"
    )
    legend.setTextFormat(QtCore.Qt.TextFormat.RichText)
    legend.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
    legend.setMinimumHeight(54)
    legend.setStyleSheet(
        "color: #CAD4E7; font-size: 22px; padding: 8px 3px;"
    )
    root.addWidget(legend)

    validation = metadata["central_full_lorentz_validation"]

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
        status_parts = []
        cursor_z = float(
            guiding_center_m[0, min(index, arrival_indices[0]), 2]
        )
        field_cursor.setValue(cursor_z)
        for track_index in range(track_count):
            frame = min(index, arrival_indices[track_index])
            origin = scene_paths[track_index, frame]
            trails[track_index].setData(
                pos=scene_paths[track_index, : frame + 1]
            )
            halos[track_index].setData(pos=origin[None, :])
            particles[track_index].setData(pos=origin[None, :])

            speed_parallel = parallel_speed[track_index, frame]
            speed_perpendicular = perpendicular_speed[track_index, frame]
            speed = np.hypot(speed_parallel, speed_perpendicular)
            parallel_fraction = speed_parallel / speed
            perpendicular_fraction = speed_perpendicular / speed
            parallel_bars[track_index].setValue(
                int(round(1000.0 * parallel_fraction))
            )
            parallel_bars[track_index].setFormat(
                f"v∥/v = {parallel_fraction:.4f}"
            )
            perpendicular_bars[track_index].setValue(
                int(round(1000.0 * perpendicular_fraction))
            )
            perpendicular_bars[track_index].setFormat(
                f"v⊥/v = {perpendicular_fraction:.4f}"
            )

            arrived = index >= arrival_indices[track_index]
            particles[track_index].setVisible(not arrived)
            halos[track_index].setVisible(not arrived)
            detector_hits[track_index].setVisible(arrived)
            track = tracks_metadata[track_index]
            status_parts.append(
                f"<span style='color:{track['color']};'>"
                f"{track['label']}: z={guiding_center_m[track_index, frame, 2]:+.1f} м, "
                f"α={pitch_deg[track_index, frame]:.3f}°, "
                f"r<sub>L</sub>={1e3*larmor_radius_m[track_index, frame]:.2f} мм, "
                f"N={cumulative_turns[track_index, frame]:.0f}</span>"
            )

        title.setText(
            "KATRIN: от сильного поля источника к детектору"
            f"<br><span style='font-size:17px; font-weight:400;'>"
            f"физическое время = {time_s[index] * 1e9:7.2f} нс; E = 0"
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

    frame_indices = _frame_indices(
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
            writer.append_data(_qimage_to_rgb_array(frame_image, QtGui))
            if frame_number % max(args.fps, 1) == 0:
                print(
                    f"Rendered {frame_number + 1}/{frame_indices.size} frames",
                    flush=True,
                )
    finally:
        writer.close()
        window.close()

    print(f"Saved 4K source-to-detector animation: {args.out}")
    print(
        f"resolution={args.window_width}x{args.window_height}, "
        f"frames={frame_indices.size}, fps={args.fps}"
    )
    print(
        "cyclotron_turns="
        + ", ".join(
            f"{track['label']}:{track['cyclotron_turns']:.1f}"
            for track in tracks_metadata
        )
    )
    print(
        "central_full_lorentz_relative_difference_percent="
        + ", ".join(
            f"{value:.4f}"
            for value in validation["relative_difference_percent"]
        )
    )


if __name__ == "__main__":
    main()
