#!/usr/bin/env python3
"""Interactive viewer and MP4 renderer for a precomputed KATRIN trajectory."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from mac_e_filter.data_io import load_trajectory_dataset  # noqa: E402


BACKGROUND = "#0B1020"
FOREGROUND = "#E8EDF7"
REFERENCE = "#34415A"
TRAIL = "#4CC9F0"
PARTICLE = "#FF4D6D"
START = "#46D6A7"
FIELD = "#4EA8DE"
PITCH = "#F4A261"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset",
        type=Path,
        default=PROJECT_ROOT
        / "datasets"
        / "katrin_2013_electron_18p6kev.npz",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Write MP4 instead of opening the interactive viewer.",
    )
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument(
        "--duration-s",
        type=float,
        default=12.0,
        help="Displayed duration; the physical timestamps remain unchanged.",
    )
    parser.add_argument("--hold-start-s", type=float, default=0.5)
    parser.add_argument("--hold-end-s", type=float, default=1.0)
    parser.add_argument("--window-width", type=int, default=1600)
    parser.add_argument("--window-height", type=int, default=900)
    parser.add_argument(
        "--interval-ms",
        type=int,
        default=25,
        help="Interactive playback timer interval.",
    )
    return parser.parse_args()


def _qimage_to_rgb_array(image, qt_gui) -> np.ndarray:
    converted = image.convertToFormat(qt_gui.QImage.Format.Format_RGBA8888)
    pointer = converted.bits()
    try:
        pointer.setsize(converted.sizeInBytes())
    except AttributeError:
        pass
    array = np.frombuffer(
        pointer,
        dtype=np.uint8,
        count=converted.sizeInBytes(),
    )
    array = array.reshape(converted.height(), converted.width(), 4)
    return np.ascontiguousarray(array[:, :, :3])


def _frame_indices(
    sample_count: int,
    fps: int,
    duration_s: float,
    hold_start_s: float,
    hold_end_s: float,
) -> np.ndarray:
    moving_frames = max(2, int(round(max(duration_s, 0.1) * fps)))
    moving = np.rint(
        np.linspace(0, sample_count - 1, moving_frames)
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


def _padded_limits(values: np.ndarray, fraction: float = 0.08) -> tuple[float, float]:
    lower = float(np.min(values))
    upper = float(np.max(values))
    span = upper - lower
    if span == 0.0:
        span = max(abs(lower), 1.0)
    padding = fraction * span
    return lower - padding, upper + padding


def main() -> None:
    args = parse_args()
    if args.fps <= 0:
        raise SystemExit("--fps must be positive")
    if args.out is not None:
        os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

    try:
        import pyqtgraph as pg
        from pyqtgraph.Qt import QtCore, QtGui, QtWidgets
    except Exception as exc:
        raise SystemExit(f"pyqtgraph/PyQt is unavailable: {exc}")

    dataset = load_trajectory_dataset(args.dataset)
    metadata = dataset["metadata"]
    time_s = np.asarray(dataset["time_s"], dtype=float)
    position = np.asarray(dataset["position_m"], dtype=float)
    field_microtesla = (
        1.0e6 * np.asarray(dataset["magnetic_field_magnitude_t"], dtype=float)
    )
    field_millitesla = field_microtesla / 1000.0
    pitch_deg = np.asarray(dataset["pitch_angle_deg"], dtype=float)
    gyrofrequency = np.asarray(
        dataset["local_gyrofrequency_rad_s"], dtype=float
    )
    if time_s.ndim != 1 or position.shape != (time_s.size, 3):
        raise SystemExit("Unexpected trajectory dataset dimensions")

    z = position[:, 2]
    x = position[:, 0]
    y = position[:, 1]
    rho = np.hypot(x, y)
    cyclotron_frequency_hz = gyrofrequency / (2.0 * np.pi)
    cumulative_turns = np.zeros_like(time_s)
    cumulative_turns[1:] = np.cumsum(
        0.5
        * (cyclotron_frequency_hz[1:] + cyclotron_frequency_hz[:-1])
        * np.diff(time_s)
    )

    pg.setConfigOption("background", BACKGROUND)
    pg.setConfigOption("foreground", FOREGROUND)
    pg.setConfigOption("antialias", True)
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])

    window = QtWidgets.QMainWindow()
    window.setWindowTitle("KATRIN full relativistic electron trajectory")
    central = QtWidgets.QWidget()
    central.setStyleSheet(f"background-color: {BACKGROUND};")
    root = QtWidgets.QVBoxLayout(central)
    root.setContentsMargins(10, 8, 10, 8)
    root.setSpacing(5)
    window.setCentralWidget(central)

    title = QtWidgets.QLabel()
    title.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
    title.setStyleSheet(
        f"color: {FOREGROUND}; font-size: 21px; font-weight: 600; "
        "padding: 5px;"
    )
    root.addWidget(title)

    graphics = pg.GraphicsLayoutWidget()
    graphics.setBackground(BACKGROUND)
    root.addWidget(graphics, stretch=1)

    longitudinal = graphics.addPlot(row=0, col=0)
    transverse = graphics.addPlot(row=0, col=1)
    field_plot = graphics.addPlot(row=1, col=0)
    pitch_plot = graphics.addPlot(row=1, col=1)

    for plot in (longitudinal, transverse, field_plot, pitch_plot):
        plot.showGrid(x=True, y=True, alpha=0.18)
        plot.getAxis("left").setTextPen(FOREGROUND)
        plot.getAxis("bottom").setTextPen(FOREGROUND)
        plot.getAxis("left").setPen(pg.mkPen(FOREGROUND))
        plot.getAxis("bottom").setPen(pg.mkPen(FOREGROUND))
        plot.getAxis("left").enableAutoSIPrefix(False)
        plot.getAxis("bottom").enableAutoSIPrefix(False)

    longitudinal.setTitle("z–y projection (coordinates in metres)")
    longitudinal.setLabel("bottom", "z", units="m")
    longitudinal.setLabel("left", "y", units="m")
    longitudinal.setXRange(*_padded_limits(z, 0.02), padding=0.0)
    longitudinal.setYRange(*_padded_limits(y), padding=0.0)
    longitudinal.disableAutoRange()

    transverse.setTitle("x–y projection (equal scale)")
    transverse.setLabel("bottom", "x", units="m")
    transverse.setLabel("left", "y", units="m")
    transverse.setAspectLocked(True)
    transverse_extent = max(
        abs(float(x.min())),
        abs(float(x.max())),
        abs(float(y.min())),
        abs(float(y.max())),
    )
    transverse_extent = max(1.08 * transverse_extent, 1.0e-3)
    transverse.setXRange(
        -transverse_extent, transverse_extent, padding=0.0
    )
    transverse.setYRange(
        -transverse_extent, transverse_extent, padding=0.0
    )
    transverse.disableAutoRange()

    field_plot.setTitle("Magnetic field sampled by the electron")
    field_plot.setLabel("bottom", "z", units="m")
    field_plot.setLabel("left", "|B|", units="mT")
    field_plot.setXRange(*_padded_limits(z, 0.02), padding=0.0)
    field_plot.setYRange(*_padded_limits(field_millitesla), padding=0.0)
    field_plot.disableAutoRange()

    pitch_plot.setTitle("Instantaneous pitch angle")
    pitch_plot.setLabel("bottom", "z", units="m")
    pitch_plot.setLabel("left", "pitch", units="deg")
    pitch_plot.setXRange(*_padded_limits(z, 0.02), padding=0.0)
    pitch_plot.setYRange(*_padded_limits(pitch_deg), padding=0.0)
    pitch_plot.disableAutoRange()

    thin_reference_pen = pg.mkPen(REFERENCE, width=1.3)
    trail_pen = pg.mkPen(TRAIL, width=3.0)
    field_pen = pg.mkPen(FIELD, width=2.2)
    pitch_pen = pg.mkPen(PITCH, width=2.2)
    particle_brush = pg.mkBrush(PARTICLE)
    particle_pen = pg.mkPen("#FFFFFF", width=1.2)

    longitudinal.plot(z, y, pen=thin_reference_pen)
    longitudinal_trail = longitudinal.plot([], [], pen=trail_pen)
    longitudinal_marker = pg.ScatterPlotItem(
        size=13,
        brush=particle_brush,
        pen=particle_pen,
    )
    longitudinal.addItem(longitudinal_marker)

    transverse.plot(x, y, pen=thin_reference_pen)
    transverse_trail = transverse.plot([], [], pen=trail_pen)
    transverse_start = pg.ScatterPlotItem(
        x=[x[0]],
        y=[y[0]],
        size=10,
        brush=pg.mkBrush(START),
        pen=pg.mkPen(START),
    )
    transverse_marker = pg.ScatterPlotItem(
        size=13,
        brush=particle_brush,
        pen=particle_pen,
    )
    transverse.addItem(transverse_start)
    transverse.addItem(transverse_marker)

    field_plot.plot(z, field_millitesla, pen=pg.mkPen(REFERENCE, width=1.2))
    field_history = field_plot.plot([], [], pen=field_pen)
    field_marker = pg.ScatterPlotItem(
        size=11,
        brush=pg.mkBrush(FIELD),
        pen=particle_pen,
    )
    field_plot.addItem(field_marker)

    pitch_plot.plot(z, pitch_deg, pen=pg.mkPen(REFERENCE, width=1.2))
    pitch_history = pitch_plot.plot([], [], pen=pitch_pen)
    pitch_marker = pg.ScatterPlotItem(
        size=11,
        brush=pg.mkBrush(PITCH),
        pen=particle_pen,
    )
    pitch_plot.addItem(pitch_marker)

    physical_duration_s = float(time_s[-1] - time_s[0])
    if args.out is not None:
        playback_note_text = (
            f"Playback slowed by {args.duration_s / physical_duration_s:.3e}×; "
            "coordinates, cyclotron phase, and orbit radius are not exaggerated."
        )
    else:
        playback_note_text = (
            "Interactive playback is not real-time; physical time is shown above. "
            "Coordinates, cyclotron phase, and orbit radius are not exaggerated."
        )
    playback_note = QtWidgets.QLabel(playback_note_text)
    playback_note.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
    playback_note.setStyleSheet(
        f"color: #AAB6CC; font-size: 13px; padding: 2px;"
    )
    root.addWidget(playback_note)

    controls = QtWidgets.QWidget()
    controls_layout = QtWidgets.QHBoxLayout(controls)
    controls_layout.setContentsMargins(0, 2, 0, 0)
    play_button = QtWidgets.QPushButton("Play / pause")
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
        stop = index + 1
        longitudinal_trail.setData(z[:stop], y[:stop])
        longitudinal_marker.setData(x=[z[index]], y=[y[index]])
        transverse_trail.setData(x[:stop], y[:stop])
        transverse_marker.setData(x=[x[index]], y=[y[index]])
        field_history.setData(z[:stop], field_millitesla[:stop])
        field_marker.setData(
            x=[z[index]], y=[field_millitesla[index]]
        )
        pitch_history.setData(z[:stop], pitch_deg[:stop])
        pitch_marker.setData(x=[z[index]], y=[pitch_deg[index]])
        title.setText(
            "Full relativistic Lorentz orbit — KATRIN 2013 nominal B, E = 0"
            f"<br><span style='font-size:15px; font-weight:400;'>"
            f"t = {time_s[index] * 1e9:7.3f} ns   |   "
            f"z = {z[index]:+6.3f} m   |   "
            f"ρ = {rho[index]:.3f} m   |   "
            f"|B| = {field_microtesla[index]:7.2f} µT   |   "
            f"f<sub>c</sub> = {cyclotron_frequency_hz[index] / 1e6:6.2f} MHz"
            f"   |   phase = {cumulative_turns[index]:.3f} turns"
            "</span>"
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
    window.resize(args.window_width, args.window_height)
    window.show()
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
        quality=8,
        macro_block_size=None,
        ffmpeg_log_level="warning",
    )
    try:
        for frame_number, sample_index in enumerate(frame_indices):
            update_frame(int(sample_index))
            app.processEvents()
            writer.append_data(
                _qimage_to_rgb_array(window.grab().toImage(), QtGui)
            )
            if frame_number % max(args.fps * 2, 1) == 0:
                print(
                    f"Rendered {frame_number + 1}/{frame_indices.size} frames",
                    flush=True,
                )
    finally:
        writer.close()
        window.close()

    visual_slowdown = args.duration_s / max(physical_duration_s, 1.0e-30)
    print(f"Saved PyQtGraph animation: {args.out}")
    print(f"frames={frame_indices.size}, fps={args.fps}")
    print(f"physical_flight_time={time_s[-1] * 1e9:.6f} ns")
    print(f"visual_slowdown={visual_slowdown:.6e}x")
    print(f"integrated_cyclotron_turns={cumulative_turns[-1]:.9f}")
    print(
        "field_model="
        f"{metadata['field_reference']['title']} "
        f"({metadata['scenario']['field_configuration']})"
    )


if __name__ == "__main__":
    main()
