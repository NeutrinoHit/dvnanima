from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import imageio.v2 as imageio
import numpy as np

from fields.scalar_qed.data_io import display_label, load_dataset, resolve_frame_key


LIGHT_THEME = {
    "background": "#f6f3eb",
    "foreground": "#141414",
    "panel_background": "#faf8f2",
    "panel_border_rgba": "rgba(20,20,20,28)",
    "grid_rgba": (36, 36, 36, 52),
}
THREE_D_THEME = {
    "background": "#0f1622",
    "foreground": "#f5f7fb",
    "panel_background": "#162132",
    "panel_border_rgba": "rgba(255,255,255,26)",
}
CENTER_COLORS = ("#d94841", "#c79a00")


@dataclass
class PanelData:
    role: str
    key: str
    frames: np.ndarray
    signed: bool
    fixed_levels: tuple[float, float]
    unicode_label: str
    html_label: str
    height_target: float
    z_multiplier: float
    alpha: float
    alpha_falloff_power: float | None
    alpha_floor: float
    gl_options: str
    camera_center_z_factor: float


@dataclass
class Panel2DState:
    plot: Any
    image: Any
    center_items: list[tuple[Any, Any, np.ndarray]]
    panel: PanelData


@dataclass
class Panel3DState:
    view: Any
    surface: Any
    legend: Any
    panel: PanelData
    lut: np.ndarray
    center_line_items: list[tuple[Any, np.ndarray]]
    center_marker_items: list[tuple[Any, np.ndarray]]
    fixed_peak: float


def _theme_for_mode(mode: str) -> dict[str, object]:
    return THREE_D_THEME if mode == "3d" else LIGHT_THEME


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="pyqtgraph viewer/renderer for a precomputed scalar QED dataset.")
    parser.add_argument("data", type=Path)
    parser.add_argument("--lower-key", type=str, default="phi_abs2")
    parser.add_argument("--upper-key", type=str, default="a0")
    parser.add_argument("--show-upper", action="store_true")
    parser.add_argument("--show-centers", action="store_true")
    parser.add_argument("--mode", choices=("2d", "3d"), default="2d")
    parser.add_argument("--time-stride", type=int, default=1)
    parser.add_argument("--space-stride", type=int, default=1)
    parser.add_argument("--interval-ms", type=int, default=40)
    parser.add_argument("--fps", type=int, default=24)
    parser.add_argument("--out", type=Path, default=None, help="Optional mp4 or gif output.")
    parser.add_argument("--fixed-levels", action="store_true", help="Use global color limits instead of per-frame normalization.")
    parser.add_argument("--z-mode", choices=("dynamic", "fixed"), default="dynamic")
    parser.add_argument("--height-scale", type=float, default=1.0, help="Visual multiplier for 3D surface height.")
    parser.add_argument("--camera-distance", type=float, default=None)
    parser.add_argument("--camera-elevation", type=float, default=19.0)
    parser.add_argument("--camera-azimuth", type=float, default=-48.0)
    parser.add_argument("--upper-camera-distance", type=float, default=None)
    parser.add_argument("--upper-camera-elevation", type=float, default=None)
    parser.add_argument("--upper-camera-azimuth", type=float, default=None)
    parser.add_argument("--window-width", type=int, default=1728)
    parser.add_argument("--window-height", type=int, default=960)
    return parser


def _global_levels(frames: np.ndarray) -> tuple[float, float]:
    vmax = max(float(np.max(np.abs(frames))), 1e-9)
    if float(np.min(frames)) < 0.0:
        return -vmax, vmax
    return 0.0, vmax


def _frame_levels(frame: np.ndarray) -> tuple[float, float]:
    vmax = max(float(np.max(np.abs(frame))), 1e-9)
    if float(np.min(frame)) < 0.0:
        return -vmax, vmax
    return 0.0, vmax


def _frame_peak(frame: np.ndarray) -> float:
    return max(float(np.max(np.abs(frame))), 1e-9)


def _set_center_items(scatter_item, path_item, centers: np.ndarray, frame_idx: int) -> None:
    point = centers[frame_idx]
    scatter_item.setData([float(point[0])], [float(point[1])])
    path_item.setData(centers[: frame_idx + 1, 0], centers[: frame_idx + 1, 1])


def _colormap(pg, signed: bool, *, style: str) -> Any:
    if style == "3d":
        preferred = "CET-D1" if signed else "CET-L17"
        fallback = "CET-D1" if signed else "CET-L9"
    else:
        preferred = "CET-D1" if signed else "CET-L9"
        fallback = "CET-D1" if signed else "CET-L17"
    try:
        return pg.colormap.get(preferred)
    except Exception:
        try:
            return pg.colormap.get(fallback)
        except Exception:
            return pg.colormap.getFromMatplotlib("coolwarm" if signed else "viridis")


def _center_color(index: int) -> str:
    return CENTER_COLORS[index % len(CENTER_COLORS)]


def _qimage_to_rgb_array(image, np_module: np.ndarray, QtGui) -> np.ndarray:
    converted = image.convertToFormat(QtGui.QImage.Format.Format_RGBA8888)
    ptr = converted.bits()
    try:
        ptr.setsize(converted.sizeInBytes())
    except AttributeError:
        pass
    array = np_module.frombuffer(ptr, dtype=np.uint8, count=converted.sizeInBytes())
    array = array.reshape(converted.height(), converted.width(), 4)
    return np_module.ascontiguousarray(array[:, :, :3])


def _save_animation(app, window, update_frame, num_frames: int, out_path: Path, fps: int, np_module: np.ndarray, QtGui) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    writer_kwargs = {"fps": max(1, int(fps))}
    if out_path.suffix.lower() == ".mp4":
        writer_kwargs["macro_block_size"] = None
    writer = imageio.get_writer(out_path, **writer_kwargs)
    try:
        for frame_idx in range(num_frames):
            update_frame(frame_idx)
            app.processEvents()
            rgb = _qimage_to_rgb_array(window.grab().toImage(), np_module, QtGui)
            writer.append_data(rgb)
    finally:
        writer.close()


def _legend_labels(vmin: float, vmax: float) -> dict[str, float]:
    if np.isclose(vmin, vmax):
        return {f"{vmax:.3g}": 1.0}
    mid = 0.5 * (vmin + vmax)
    return {
        f"{vmax:.3g}": 1.0,
        f"{mid:.3g}": 0.5,
        f"{vmin:.3g}": 0.0,
    }


def _frame_colors(frame: np.ndarray, levels: tuple[float, float], lut: np.ndarray, alpha: float) -> np.ndarray:
    vmin, vmax = levels
    if vmax <= vmin:
        norm = np.zeros_like(frame, dtype=np.float32)
    else:
        norm = np.clip((frame - vmin) / (vmax - vmin), 0.0, 1.0)
    indices = np.clip((norm * (lut.shape[0] - 1)).astype(np.int32), 0, lut.shape[0] - 1)
    colors = np.asarray(lut[indices], dtype=np.float32).copy()
    if colors.shape[-1] == 3:
        alpha_channel = np.full(colors.shape[:-1] + (1,), alpha, dtype=np.float32)
        colors = np.concatenate((colors, alpha_channel), axis=-1)
    else:
        colors[..., 3] = alpha
    return np.ascontiguousarray(colors.reshape(-1, 4), dtype=np.float32)


def _apply_alpha_falloff(
    colors: np.ndarray, frame: np.ndarray, max_alpha: float, power: float, alpha_floor: float
) -> np.ndarray:
    peak = _frame_peak(frame)
    strength = np.clip(np.abs(frame) / peak, 0.0, 1.0).astype(np.float32)
    alpha_map = max_alpha * np.power(strength, power)
    alpha_map = np.clip(alpha_map, alpha_floor, max_alpha)
    adjusted = np.asarray(colors, dtype=np.float32).copy().reshape(frame.shape[0], frame.shape[1], 4)
    adjusted[..., 3] = alpha_map
    return np.ascontiguousarray(adjusted.reshape(-1, 4), dtype=np.float32)


def _panel_render_style(*, role: str, signed: bool) -> dict[str, object]:
    if role == "upper" and signed:
        return {
            "z_multiplier": 1.35,
            "alpha": 0.46,
            "alpha_falloff_power": 1.25,
            "alpha_floor": 0.005,
            "gl_options": "translucent",
            "camera_center_z_factor": 0.04,
        }
    return {
        "z_multiplier": 1.0,
        "alpha": 0.96 if not signed else 0.88,
        "alpha_falloff_power": None,
        "alpha_floor": 0.0,
        "gl_options": "opaque",
        "camera_center_z_factor": 0.10,
    }


def _panel_data_from_dataset(dataset: dict, args: argparse.Namespace) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[PanelData]]:
    time_stride = max(1, int(args.time_stride))
    space_stride = max(1, int(args.space_stride))
    lower_key = resolve_frame_key(args.lower_key)
    upper_key = resolve_frame_key(args.upper_key)

    times = dataset["times"][::time_stride]
    x_axis = dataset["x_axis"][::space_stride]
    y_axis = dataset["y_axis"][::space_stride]
    packet_centers = dataset["packet_centers"][::time_stride]

    render_cfg = dataset["metadata"]["config"].get("render", {})
    lower_height_target = float(render_cfg.get("lower_height_target", 1.10))
    upper_height_target = float(render_cfg.get("upper_height_target", 0.92))

    lower_frames = dataset[lower_key][::time_stride, ::space_stride, ::space_stride]
    upper_frames = dataset[upper_key][::time_stride, ::space_stride, ::space_stride]

    lower_signed = float(np.min(lower_frames)) < 0.0
    panels = [
        PanelData(
            role="lower",
            key=lower_key,
            frames=lower_frames,
            signed=lower_signed,
            fixed_levels=_global_levels(lower_frames),
            unicode_label=display_label(lower_key, style="unicode"),
            html_label=display_label(lower_key, style="html"),
            height_target=lower_height_target,
            **_panel_render_style(role="lower", signed=lower_signed),
        )
    ]
    if args.show_upper:
        upper_signed = float(np.min(upper_frames)) < 0.0
        panels.append(
            PanelData(
                role="upper",
                key=upper_key,
                frames=upper_frames,
                signed=upper_signed,
                fixed_levels=_global_levels(upper_frames),
                unicode_label=display_label(upper_key, style="unicode"),
                html_label=display_label(upper_key, style="html"),
                height_target=upper_height_target,
                **_panel_render_style(role="upper", signed=upper_signed),
            )
        )
    return times, x_axis, y_axis, packet_centers, panels


def _build_2d_ui(pg, QtCore, QtWidgets, times: np.ndarray, x_axis: np.ndarray, y_axis: np.ndarray, packet_centers: np.ndarray, panels: list[PanelData], args: argparse.Namespace):
    window = QtWidgets.QMainWindow()
    window.setWindowTitle("Scalar QED Dataset Viewer")
    central = QtWidgets.QWidget()
    layout = QtWidgets.QVBoxLayout(central)
    window.setCentralWidget(central)

    graphics = pg.GraphicsLayoutWidget()
    layout.addWidget(graphics, stretch=1)
    controls = QtWidgets.QHBoxLayout()
    layout.addLayout(controls)

    play_button = QtWidgets.QPushButton("Play")
    controls.addWidget(play_button)
    slider = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
    slider.setMinimum(0)
    slider.setMaximum(len(times) - 1)
    controls.addWidget(slider, stretch=1)
    time_label = QtWidgets.QLabel(f"t = {float(times[0]):.3f}")
    controls.addWidget(time_label)

    x_min = float(x_axis[0])
    y_min = float(y_axis[0])
    x_span = float(x_axis[-1] - x_axis[0])
    y_span = float(y_axis[-1] - y_axis[0])

    states: list[Panel2DState] = []
    for col_idx, panel in enumerate(panels):
        plot = graphics.addPlot(row=0, col=col_idx)
        plot.showGrid(x=True, y=True, alpha=0.22)
        plot.setLabel("bottom", "x")
        plot.setLabel("left", "y")
        plot.setAspectLocked(False)
        plot.setXRange(x_min, x_min + x_span, padding=0.0)
        plot.setYRange(y_min, y_min + y_span, padding=0.0)
        plot.disableAutoRange()

        image = pg.ImageItem()
        image.setRect(QtCore.QRectF(x_min, y_min, x_span, y_span))
        color_map = _colormap(pg, panel.signed, style="2d")
        image.setLookupTable(color_map.getLookupTable(nPts=256))
        plot.addItem(image)

        center_items = []
        if args.show_centers:
            for packet_idx, centers in enumerate(packet_centers.transpose(1, 0, 2)):
                color_name = _center_color(packet_idx)
                scatter = pg.ScatterPlotItem(size=9, brush=pg.mkBrush(color_name), pen=pg.mkPen(color_name))
                path = pg.PlotDataItem(pen=pg.mkPen(color_name, width=2))
                plot.addItem(path)
                plot.addItem(scatter)
                center_items.append((scatter, path, centers))

        states.append(Panel2DState(plot=plot, image=image, center_items=center_items, panel=panel))

    def update_frame(frame_idx: int) -> None:
        frame_idx = int(max(0, min(frame_idx, len(times) - 1)))
        slider.blockSignals(True)
        slider.setValue(frame_idx)
        slider.blockSignals(False)
        time_label.setText(f"t = {float(times[frame_idx]):.3f}")

        for state in states:
            frame = state.panel.frames[frame_idx]
            levels = state.panel.fixed_levels if args.fixed_levels else _frame_levels(frame)
            state.plot.setTitle(
                f'{state.panel.html_label} <span style="color:#666;">[{levels[0]:.3g}, {levels[1]:.3g}]</span>'
            )
            state.image.setImage(frame.T, autoLevels=False)
            state.image.setLevels(levels)
            for scatter, path, centers in state.center_items:
                _set_center_items(scatter, path, centers, frame_idx)

    return window, play_button, slider, update_frame


def _make_panel_box(QtCore, QtWidgets, title: str, view, theme: dict[str, object]) -> Any:
    container = QtWidgets.QFrame()
    container.setStyleSheet(
        "QFrame { "
        f"background: {theme['panel_background']}; "
        f"border: 1px solid {theme['panel_border_rgba']}; "
        "border-radius: 14px; }"
    )
    layout = QtWidgets.QVBoxLayout(container)
    layout.setContentsMargins(8, 8, 8, 8)
    layout.setSpacing(4)

    title_label = QtWidgets.QLabel(title)
    title_label.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
    title_label.setStyleSheet(
        "font-size: 18px; font-weight: 600; "
        f"color: {theme['foreground']}; background: transparent;"
    )
    layout.addWidget(title_label)
    layout.addWidget(view, stretch=1)
    return container


def _make_gl_panel(
    gl,
    pg,
    QtGui,
    panel: PanelData,
    x_axis: np.ndarray,
    y_axis: np.ndarray,
    packet_centers: np.ndarray,
    args: argparse.Namespace,
    theme: dict[str, object],
):
    view = gl.GLViewWidget()
    view.setBackgroundColor(theme["background"])
    view.opts["fov"] = 36
    domain_x = float(x_axis[-1] - x_axis[0])
    domain_y = float(y_axis[-1] - y_axis[0])
    z_target = float(panel.height_target) * float(args.height_scale) * float(panel.z_multiplier)
    if panel.role == "upper":
        distance_arg = args.upper_camera_distance if args.upper_camera_distance is not None else args.camera_distance
        elevation_arg = args.upper_camera_elevation if args.upper_camera_elevation is not None else args.camera_elevation
        azimuth_arg = args.upper_camera_azimuth if args.upper_camera_azimuth is not None else args.camera_azimuth
    else:
        distance_arg = args.camera_distance
        elevation_arg = args.camera_elevation
        azimuth_arg = args.camera_azimuth

    distance = float(distance_arg) if distance_arg is not None else 1.18 * max(domain_x, domain_y)
    camera_center = QtGui.QVector3D(
        float(0.5 * (x_axis[0] + x_axis[-1])),
        float(0.5 * (y_axis[0] + y_axis[-1])),
        float(panel.camera_center_z_factor) * z_target,
    )
    view.setCameraPosition(
        pos=camera_center,
        distance=distance,
        elevation=float(elevation_arg),
        azimuth=float(azimuth_arg),
    )

    color_map = _colormap(pg, panel.signed, style="3d")
    lut = np.asarray(color_map.getLookupTable(nPts=256, alpha=True, mode="float"), dtype=np.float32)
    alpha = float(panel.alpha)
    initial_levels = panel.fixed_levels if args.fixed_levels else _frame_levels(panel.frames[0])
    initial_z = np.asarray((z_target / max(_frame_peak(panel.frames[0]), 1e-9)) * panel.frames[0], dtype=np.float32)
    initial_colors = _frame_colors(panel.frames[0], initial_levels, lut, alpha)
    if panel.alpha_falloff_power is not None:
        initial_colors = _apply_alpha_falloff(
            initial_colors, panel.frames[0], alpha, float(panel.alpha_falloff_power), float(panel.alpha_floor)
        )
    surface = gl.GLSurfacePlotItem(
        x=np.asarray(x_axis, dtype=np.float32),
        y=np.asarray(y_axis, dtype=np.float32),
        z=initial_z,
        colors=initial_colors,
        shader="shaded",
        smooth=False,
        computeNormals=True,
        drawEdges=False,
        glOptions=panel.gl_options,
    )
    view.addItem(surface)

    legend = gl.GLGradientLegendItem(
        pos=(14, 18),
        size=(18, 190),
        gradient=color_map,
        labels=_legend_labels(*initial_levels),
        fontColor=theme["foreground"],
    )
    view.addItem(legend)

    center_line_items: list[tuple[Any, np.ndarray]] = []
    center_marker_items: list[tuple[Any, np.ndarray]] = []
    if args.show_centers:
        z_center = np.float32(0.025 * z_target)
        for packet_idx, centers in enumerate(packet_centers.transpose(1, 0, 2)):
            color_name = _center_color(packet_idx)
            color = pg.mkColor(color_name).getRgbF()
            line = gl.GLLinePlotItem(
                pos=np.zeros((1, 3), dtype=np.float32),
                color=color,
                width=2.5,
                mode="line_strip",
                antialias=True,
                glOptions="translucent",
            )
            marker = gl.GLScatterPlotItem(
                pos=np.zeros((1, 3), dtype=np.float32),
                color=color,
                size=10.0,
                pxMode=True,
                glOptions="translucent",
            )
            view.addItem(line)
            view.addItem(marker)
            center_line_items.append((line, np.asarray(centers, dtype=np.float32)))
            center_marker_items.append((marker, np.asarray(centers, dtype=np.float32)))

    return Panel3DState(
        view=view,
        surface=surface,
        legend=legend,
        panel=panel,
        lut=lut,
        center_line_items=center_line_items,
        center_marker_items=center_marker_items,
        fixed_peak=_frame_peak(panel.frames),
    )


def _build_3d_ui(pg, gl, QtCore, QtGui, QtWidgets, times: np.ndarray, x_axis: np.ndarray, y_axis: np.ndarray, packet_centers: np.ndarray, panels: list[PanelData], args: argparse.Namespace, theme: dict[str, object]):
    window = QtWidgets.QMainWindow()
    window.setWindowTitle("Scalar QED Dataset Viewer")
    central = QtWidgets.QWidget()
    central.setStyleSheet(
        f"background: {theme['background']}; color: {theme['foreground']};"
        "QPushButton { background: rgba(255,255,255,0.08); color: inherit; border: 1px solid rgba(255,255,255,0.18); border-radius: 8px; padding: 6px 12px; }"
        "QPushButton:hover { background: rgba(255,255,255,0.14); }"
        "QSlider::groove:horizontal { height: 8px; background: rgba(255,255,255,0.10); border-radius: 4px; }"
        "QSlider::handle:horizontal { width: 18px; margin: -6px 0; border-radius: 9px; background: rgba(255,255,255,0.92); }"
    )
    root = QtWidgets.QVBoxLayout(central)
    root.setContentsMargins(10, 8, 10, 10)
    root.setSpacing(8)
    window.setCentralWidget(central)

    time_label = QtWidgets.QLabel(f"t = {float(times[0]):.3f}")
    time_label.setStyleSheet(f"font-size: 26px; font-weight: 600; color: {theme['foreground']};")
    root.addWidget(time_label)

    content = QtWidgets.QHBoxLayout()
    content.setSpacing(8)
    root.addLayout(content, stretch=1)

    controls = QtWidgets.QHBoxLayout()
    root.addLayout(controls)
    play_button = QtWidgets.QPushButton("Play")
    controls.addWidget(play_button)
    slider = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
    slider.setMinimum(0)
    slider.setMaximum(len(times) - 1)
    controls.addWidget(slider, stretch=1)

    states: list[Panel3DState] = []
    for panel in panels:
        state = _make_gl_panel(gl, pg, QtGui, panel, x_axis, y_axis, packet_centers, args, theme)
        panel_box = _make_panel_box(QtCore, QtWidgets, panel.unicode_label, state.view, theme)
        content.addWidget(panel_box, stretch=1)
        states.append(state)

    def update_frame(frame_idx: int) -> None:
        frame_idx = int(max(0, min(frame_idx, len(times) - 1)))
        slider.blockSignals(True)
        slider.setValue(frame_idx)
        slider.blockSignals(False)
        time_label.setText(f"t = {float(times[frame_idx]):.3f}")

        for state in states:
            frame = state.panel.frames[frame_idx]
            levels = state.panel.fixed_levels if args.fixed_levels else _frame_levels(frame)
            peak = state.fixed_peak if args.z_mode == "fixed" else _frame_peak(frame)
            target_height = float(state.panel.height_target) * float(args.height_scale) * float(state.panel.z_multiplier)
            z = np.asarray((target_height / max(peak, 1e-9)) * frame, dtype=np.float32)
            alpha = float(state.panel.alpha)
            colors = _frame_colors(frame, levels, state.lut, alpha)
            if state.panel.alpha_falloff_power is not None:
                colors = _apply_alpha_falloff(
                    colors, frame, alpha, float(state.panel.alpha_falloff_power), float(state.panel.alpha_floor)
                )
            state.surface.setData(z=z, colors=colors)
            state.legend.setData(labels=_legend_labels(*levels))

            if args.show_centers:
                z_center = np.float32(0.025 * target_height)
                for line_item, centers in state.center_line_items:
                    path = np.empty((frame_idx + 1, 3), dtype=np.float32)
                    path[:, :2] = centers[: frame_idx + 1]
                    path[:, 2] = z_center
                    line_item.setData(pos=path)
                for marker_item, centers in state.center_marker_items:
                    point = np.array([[centers[frame_idx, 0], centers[frame_idx, 1], z_center]], dtype=np.float32)
                    marker_item.setData(pos=point)

    return window, play_button, slider, update_frame


def main() -> None:
    try:
        import pyqtgraph as pg
        from pyqtgraph.Qt import QtCore, QtGui, QtWidgets
    except Exception as exc:
        raise SystemExit(f"pyqtgraph/PyQt is not available in this interpreter: {exc}")

    args = build_arg_parser().parse_args()
    theme = _theme_for_mode(args.mode)

    pg.setConfigOption("background", theme["background"])
    pg.setConfigOption("foreground", theme["foreground"])
    pg.setConfigOption("antialias", True)
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])

    dataset = load_dataset(args.data)
    times, x_axis, y_axis, packet_centers, panels = _panel_data_from_dataset(dataset, args)

    if args.mode == "3d":
        try:
            import pyqtgraph.opengl as gl
        except Exception as exc:
            raise SystemExit(
                "pyqtgraph OpenGL backend is unavailable in this interpreter: "
                f"{exc}. Install `PyOpenGL` and `PyOpenGL_accelerate` if needed."
            )
        window, play_button, slider, update_frame = _build_3d_ui(
            pg, gl, QtCore, QtGui, QtWidgets, times, x_axis, y_axis, packet_centers, panels, args, theme
        )
    else:
        window, play_button, slider, update_frame = _build_2d_ui(
            pg, QtCore, QtWidgets, times, x_axis, y_axis, packet_centers, panels, args
        )

    timer = QtCore.QTimer()
    timer.setInterval(max(1, int(args.interval_ms)))

    def on_slider(value: int) -> None:
        update_frame(int(value))

    def on_timer() -> None:
        next_idx = (slider.value() + 1) % len(times)
        update_frame(next_idx)

    def toggle_play() -> None:
        if timer.isActive():
            timer.stop()
            play_button.setText("Play")
        else:
            timer.start()
            play_button.setText("Pause")

    slider.valueChanged.connect(on_slider)
    play_button.clicked.connect(toggle_play)
    timer.timeout.connect(on_timer)

    update_frame(0)
    window.resize(int(args.window_width), int(args.window_height))
    window.show()
    if hasattr(window, "raise_"):
        window.raise_()
    app.processEvents()

    if args.out is not None:
        _save_animation(app, window, update_frame, len(times), args.out, args.fps, np, QtGui)
        print(f"Saved pyqtgraph render to {args.out}")
        return

    app.exec()


if __name__ == "__main__":
    main()
