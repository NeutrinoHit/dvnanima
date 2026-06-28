from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parents[2]))

from fields.radiating_charge.data_io import display_label, load_dataset, resolve_frame_key


LIGHT_THEME = {
    "background": "#f6f3eb",
    "foreground": "#141414",
}
THREE_D_THEME = {
    "background": "#000000",
    "foreground": "#f5f7fb",
}
SOURCE_COLOR = "#d94841"


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
    peak: float


@dataclass
class Panel2DState:
    plot: Any
    image: Any
    marker: Any | None
    path: Any | None
    anchor: Any | None
    panel: PanelData


@dataclass
class Panel3DState:
    view: Any
    surface: Any
    legend: Any
    marker: Any | None
    path: Any | None
    anchor: Any | None
    panel: PanelData
    lut: np.ndarray


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="pyqtgraph viewer/renderer for a radiating-charge dataset.")
    parser.add_argument("data", type=Path)
    parser.add_argument("--lower-key", type=str, default="bz_rad")
    parser.add_argument("--upper-key", type=str, default="ez_rad")
    parser.add_argument("--show-upper", action="store_true")
    parser.add_argument("--stacked-planes", action="store_true", help="3D mode: draw one lower trajectory plane and one upper field plane.")
    parser.add_argument("--show-center", action="store_true")
    parser.add_argument("--show-fixed-center", dest="show_fixed_center", action="store_true")
    parser.add_argument("--hide-fixed-center", dest="show_fixed_center", action="store_false")
    parser.add_argument("--mode", choices=("2d", "3d"), default="2d")
    parser.add_argument("--transform", choices=("linear", "log", "signed_log", "tanh"), default="signed_log")
    parser.add_argument("--time-stride", type=int, default=1)
    parser.add_argument("--space-stride", type=int, default=1)
    parser.add_argument("--interval-ms", type=int, default=40)
    parser.add_argument("--fps", type=int, default=24)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--fixed-levels", dest="fixed_levels", action="store_true", help="Use global color limits for all frames.")
    parser.add_argument("--dynamic-levels", dest="fixed_levels", action="store_false", help="Use per-frame color limits.")
    parser.add_argument("--z-mode", choices=("dynamic", "fixed"), default="dynamic")
    parser.add_argument("--height-scale", type=float, default=1.0)
    parser.add_argument("--xy-scale", type=float, default=1.0, help="Visual scale for XY plane/markers in renderer.")
    parser.add_argument("--surface-alpha", type=float, default=0.48)
    parser.add_argument("--surface-smooth", dest="surface_smooth", action="store_true", help="Enable vertex interpolation for smoother 3D surface.")
    parser.add_argument("--surface-faceted", dest="surface_smooth", action="store_false", help="Disable interpolation; use faceted surface.")
    parser.add_argument("--center-size", type=float, default=10.0, help="Marker size for moving charge in 3D mode.")
    parser.add_argument("--fixed-center-size", type=float, default=16.0, help="Marker size for fixed potential center.")
    parser.add_argument("--time-slowdown", type=float, default=1.0, help="Uniform slowdown factor for export by repeating frames.")
    parser.add_argument(
        "--radiation-slowmo",
        type=float,
        default=0.0,
        help="Additional adaptive slowdown during strong radiation peaks (export only).",
    )
    parser.add_argument(
        "--radiation-slowmo-gamma",
        type=float,
        default=2.1,
        help="Shape of adaptive slowdown response; larger values focus on strongest bursts.",
    )
    parser.add_argument(
        "--radiation-slowmo-quantile",
        type=float,
        default=0.997,
        help="Spatial quantile used to estimate per-frame radiation intensity for slowdown.",
    )
    parser.add_argument(
        "--level-max-quantile",
        type=float,
        default=1.0,
        help="Clip color limits to this quantile of |field| (helps reveal weaker waves).",
    )
    parser.add_argument(
        "--height-peak-quantile",
        type=float,
        default=1.0,
        help="Clip height normalization to this quantile of |field| (helps reveal low-amplitude structure).",
    )
    parser.add_argument("--upper-plane-shift", type=float, default=None, help="Override vertical shift of the upper plane in stacked 3D mode.")
    parser.add_argument("--camera-distance-scale", type=float, default=1.45, help="3D camera distance as a multiple of scene span.")
    parser.add_argument("--camera-elevation", type=float, default=24.0, help="3D camera elevation angle in degrees.")
    parser.add_argument("--camera-azimuth", type=float, default=-34.0, help="3D camera azimuth angle in degrees.")
    parser.add_argument("--window-width", type=int, default=1728)
    parser.add_argument("--window-height", type=int, default=960)
    parser.set_defaults(fixed_levels=True, show_fixed_center=True, surface_smooth=True)
    return parser


def _theme_for_mode(mode: str) -> dict[str, object]:
    return THREE_D_THEME if mode == "3d" else LIGHT_THEME


def _robust_abs_peak(values: np.ndarray, quantile: float) -> float:
    q = float(np.clip(quantile, 0.50, 1.0))
    abs_values = np.abs(values)
    if q >= 1.0:
        return max(float(np.max(abs_values)), 1e-9)
    return max(float(np.quantile(abs_values, q=q)), 1e-9)


def _global_levels(frames: np.ndarray, max_quantile: float = 1.0) -> tuple[float, float]:
    vmax = _robust_abs_peak(frames, max_quantile)
    if float(np.min(frames)) < 0.0:
        return -vmax, vmax
    return 0.0, vmax


def _frame_levels(frame: np.ndarray, max_quantile: float = 1.0) -> tuple[float, float]:
    vmax = _robust_abs_peak(frame, max_quantile)
    if float(np.min(frame)) < 0.0:
        return -vmax, vmax
    return 0.0, vmax


def _frame_peak(frame: np.ndarray, peak_quantile: float = 1.0) -> float:
    return _robust_abs_peak(frame, peak_quantile)


def _apply_transform(frame: np.ndarray, mode: str) -> np.ndarray:
    if mode == "linear":
        return frame
    if mode == "log":
        return np.log10(1.0 + np.abs(frame))
    if mode == "signed_log":
        return np.sign(frame) * np.log10(1.0 + np.abs(frame))
    peak = _frame_peak(frame)
    return np.tanh(frame / peak)


def _colormap(pg, signed: bool, *, style: str):
    if style in {"3d", "3d_stacked"} and signed:
        # Signed radiation map: blue (negative) -> black (zero) -> red (positive).
        return pg.ColorMap(
            np.array([0.0, 0.46, 0.50, 0.54, 1.0], dtype=float),
            np.array(
                [
                    (38, 74, 255, 255),
                    (12, 27, 84, 255),
                    (0, 0, 0, 255),
                    (98, 16, 19, 255),
                    (255, 56, 46, 255),
                ],
                dtype=np.ubyte,
            ),
        )
    if style in {"3d", "3d_stacked"} and not signed:
        # Unsigned radiation map with black floor and bright blue/red highlights.
        return pg.ColorMap(
            np.array([0.0, 0.10, 0.42, 0.74, 0.92, 1.0], dtype=float),
            np.array(
                [
                    (0, 0, 0, 255),
                    (6, 10, 32, 255),
                    (35, 86, 255, 255),
                    (120, 26, 144, 255),
                    (255, 56, 46, 255),
                    (255, 221, 221, 255),
                ],
                dtype=np.ubyte,
            ),
        )

    preferred = "RdYlBu_r" if signed else "viridis"
    if style == "3d":
        preferred = "RdYlBu_r" if signed else "turbo"
    if style == "3d_stacked":
        preferred = "coolwarm" if signed else "plasma"
    try:
        return pg.colormap.getFromMatplotlib(preferred)
    except Exception:
        return pg.colormap.get("CET-D1" if signed else "CET-L9")


def _qimage_to_rgb_array(image, np_module, QtGui) -> np.ndarray:
    converted = image.convertToFormat(QtGui.QImage.Format.Format_RGBA8888)
    ptr = converted.bits()
    try:
        ptr.setsize(converted.sizeInBytes())
    except AttributeError:
        pass
    array = np_module.frombuffer(ptr, dtype=np.uint8, count=converted.sizeInBytes())
    array = array.reshape(converted.height(), converted.width(), 4)
    return np_module.ascontiguousarray(array[:, :, :3])


_FONT_5X7 = {
    " ": ("00000", "00000", "00000", "00000", "00000", "00000", "00000"),
    "D": ("11110", "10001", "10001", "10001", "10001", "10001", "11110"),
    "E": ("11111", "10000", "11110", "10000", "10000", "10000", "11111"),
    "G": ("01110", "10001", "10000", "10111", "10001", "10001", "01110"),
    "H": ("10001", "10001", "11111", "10001", "10001", "10001", "10001"),
    "I": ("11111", "00100", "00100", "00100", "00100", "00100", "11111"),
    "L": ("10000", "10000", "10000", "10000", "10000", "10000", "11111"),
    "M": ("10001", "11011", "10101", "10101", "10001", "10001", "10001"),
    "O": ("01110", "10001", "10001", "10001", "10001", "10001", "01110"),
    "U": ("10001", "10001", "10001", "10001", "10001", "10001", "01110"),
    "W": ("10001", "10001", "10001", "10101", "10101", "10101", "01010"),
}


def _draw_bitmap_text(
    rgb: np.ndarray,
    text: str,
    x: int,
    y: int,
    *,
    scale: int = 2,
    color: tuple[int, int, int] = (255, 255, 255),
) -> None:
    h, w = int(rgb.shape[0]), int(rgb.shape[1])
    pen_x = int(x)
    pen_y = int(y)
    pixel = max(1, int(scale))
    glyph_w = 5 * pixel
    glyph_h = 7 * pixel
    gap = pixel

    for raw_ch in str(text).upper():
        glyph = _FONT_5X7.get(raw_ch, _FONT_5X7[" "])
        for gy, row in enumerate(glyph):
            for gx, bit in enumerate(row):
                if bit != "1":
                    continue
                x0 = pen_x + gx * pixel
                y0 = pen_y + gy * pixel
                x1 = min(w, x0 + pixel)
                y1 = min(h, y0 + pixel)
                if x1 <= 0 or y1 <= 0 or x0 >= w or y0 >= h:
                    continue
                rgb[max(y0, 0) : y1, max(x0, 0) : x1, 0] = np.uint8(color[0])
                rgb[max(y0, 0) : y1, max(x0, 0) : x1, 1] = np.uint8(color[1])
                rgb[max(y0, 0) : y1, max(x0, 0) : x1, 2] = np.uint8(color[2])
        pen_x += glyph_w + gap

    _ = glyph_h  # Keep lint quiet if future refactors remove usage.


def _overlay_colorbar(rgb: np.ndarray, lut: np.ndarray) -> np.ndarray:
    if lut is None or lut.size == 0:
        return rgb
    h, w = int(rgb.shape[0]), int(rgb.shape[1])
    if h < 40 or w < 40:
        return rgb

    bar_h = int(np.clip(round(0.26 * h), 120, 300))
    bar_w = int(np.clip(round(0.014 * w), 16, 36))
    margin = int(np.clip(round(0.015 * min(h, w)), 12, 28))
    x0 = margin
    y0 = (h - bar_h) // 2
    y1 = y0 + bar_h
    x1 = min(w - 2, x0 + bar_w)
    y0 = max(0, y0)
    y1 = min(h - 2, y1)
    if x1 <= x0 or y1 <= y0:
        return rgb

    grad_h = y1 - y0
    idx = np.linspace(lut.shape[0] - 1, 0, grad_h).astype(np.int32)
    colors = np.clip(lut[idx, :3], 0.0, 1.0)
    grad_rgb = np.asarray(np.round(255.0 * colors), dtype=np.uint8)
    rgb[y0:y1, x0:x1, :] = grad_rgb[:, None, :]

    # White border for visibility on dark backgrounds.
    xb0, xb1 = max(0, x0 - 1), min(w, x1 + 1)
    yb0, yb1 = max(0, y0 - 1), min(h, y1 + 1)
    rgb[yb0:yb1, xb0, :] = 255
    rgb[yb0:yb1, xb1 - 1, :] = 255
    rgb[yb0, xb0:xb1, :] = 255
    rgb[yb1 - 1, xb0:xb1, :] = 255

    label_scale = 1 if h < 900 else 2
    glyph_h = 7 * label_scale
    label_x = xb1 + int(np.clip(round(0.45 * margin), 6, 18))
    top_y = y0 - glyph_h // 2
    mid_y = y0 + (y1 - y0) // 2 - glyph_h // 2
    bot_y = y1 - glyph_h // 2 - glyph_h
    labels = [("HIGH", top_y), ("MEDIUM", mid_y), ("LOW", bot_y)]
    for text, ly in labels:
        ly_clip = int(np.clip(ly, 0, h - glyph_h - 1))
        _draw_bitmap_text(rgb, text, label_x + 1, ly_clip + 1, scale=label_scale, color=(0, 0, 0))
        _draw_bitmap_text(rgb, text, label_x, ly_clip, scale=label_scale, color=(235, 235, 235))

    return rgb


def _save_animation(
    app,
    window,
    update_frame,
    num_frames: int,
    out_path: Path,
    fps: int,
    np_module,
    QtGui,
    frame_sequence: np.ndarray | None = None,
    colorbar_lut: np.ndarray | None = None,
) -> int:
    try:
        import imageio.v2 as imageio
    except Exception as exc:
        raise SystemExit(
            f"imageio is required to write `{out_path.suffix}` animations. Install imageio first. Details: {exc}"
        )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    writer_kwargs = {"fps": max(1, int(fps))}
    if out_path.suffix.lower() == ".mp4":
        writer_kwargs["macro_block_size"] = None
    writer = imageio.get_writer(out_path, **writer_kwargs)
    if frame_sequence is None:
        frame_indices = np.arange(num_frames, dtype=np.int32)
    else:
        frame_indices = np.asarray(frame_sequence, dtype=np.int32).reshape(-1)
        if frame_indices.size == 0:
            frame_indices = np.arange(num_frames, dtype=np.int32)
        frame_indices = np.clip(frame_indices, 0, max(0, num_frames - 1))
    try:
        for frame_idx in frame_indices:
            update_frame(int(frame_idx))
            app.processEvents()
            rgb = _qimage_to_rgb_array(window.grab().toImage(), np_module, QtGui)
            if colorbar_lut is not None:
                rgb = _overlay_colorbar(rgb, colorbar_lut)
            writer.append_data(rgb)
    finally:
        writer.close()
    return int(frame_indices.size)


def _build_export_frame_sequence(
    panel_frames: np.ndarray,
    num_frames: int,
    *,
    time_slowdown: float,
    radiation_slowmo: float,
    radiation_slowmo_gamma: float,
    radiation_slowmo_quantile: float,
) -> np.ndarray:
    if num_frames <= 0:
        return np.zeros((0,), dtype=np.int32)

    base = max(1.0, float(time_slowdown))
    repeats = np.full((num_frames,), base, dtype=np.float64)

    slowmo = max(0.0, float(radiation_slowmo))
    if slowmo > 0.0:
        q = float(np.clip(radiation_slowmo_quantile, 0.50, 0.99999))
        signal = np.quantile(np.abs(panel_frames[:num_frames]), q=q, axis=(1, 2))
        lo = float(np.quantile(signal, 0.45))
        hi = float(np.quantile(signal, 0.995))
        denom = max(hi - lo, 1e-12)
        norm = np.clip((signal - lo) / denom, 0.0, 1.0)
        gamma = max(1e-6, float(radiation_slowmo_gamma))
        repeats *= 1.0 + slowmo * np.power(norm, gamma)

    repeats_int = np.clip(np.rint(repeats).astype(np.int32), 1, 240)
    return np.repeat(np.arange(num_frames, dtype=np.int32), repeats_int)


def _build_panels(dataset: dict, args) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[PanelData]]:
    time_stride = max(1, int(args.time_stride))
    space_stride = max(1, int(args.space_stride))

    lower_key = resolve_frame_key(args.lower_key, dataset=dataset)
    upper_key = resolve_frame_key(args.upper_key, dataset=dataset)

    times = dataset["times"][::time_stride]
    x_axis = dataset["x_axis"][::space_stride]
    y_axis = dataset["y_axis"][::space_stride]
    centers = dataset["source_centers"][::time_stride]

    render_cfg = dataset["metadata"].get("config", {}).get("render", {})
    lower_frames = dataset[lower_key][::time_stride, ::space_stride, ::space_stride]
    upper_frames = dataset[upper_key][::time_stride, ::space_stride, ::space_stride]

    panels = [
        PanelData(
            role="lower",
            key=lower_key,
            frames=lower_frames,
            signed=float(np.min(lower_frames)) < 0.0,
            fixed_levels=_global_levels(_apply_transform(lower_frames, args.transform), max_quantile=float(args.level_max_quantile)),
            unicode_label=display_label(lower_key, style="unicode"),
            html_label=display_label(lower_key, style="html"),
            height_target=float(render_cfg.get("lower_height_target", 1.0)),
            peak=_frame_peak(_apply_transform(lower_frames, args.transform), peak_quantile=float(args.height_peak_quantile)),
        )
    ]
    if args.show_upper:
        panels.append(
            PanelData(
                role="upper",
                key=upper_key,
                frames=upper_frames,
                signed=float(np.min(upper_frames)) < 0.0,
                fixed_levels=_global_levels(_apply_transform(upper_frames, args.transform), max_quantile=float(args.level_max_quantile)),
                unicode_label=display_label(upper_key, style="unicode"),
                html_label=display_label(upper_key, style="html"),
                height_target=float(render_cfg.get("upper_height_target", 1.0)),
                peak=_frame_peak(_apply_transform(upper_frames, args.transform), peak_quantile=float(args.height_peak_quantile)),
            )
        )
    return times, x_axis, y_axis, centers, panels


def _fixed_center_xy(dataset: dict) -> tuple[float, float] | None:
    meta = dataset.get("metadata", {})
    cfg = meta.get("config", {})
    traj = cfg.get("trajectory", {})
    if not isinstance(traj, dict):
        return None
    model = str(traj.get("model", "")).strip().lower()
    model_cfg = traj.get(model, {})
    if not isinstance(model_cfg, dict):
        return None
    try:
        cx = float(model_cfg.get("center_x", 0.0))
        cy = float(model_cfg.get("center_y", 0.0))
    except Exception:
        return None
    return cx, cy


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
    times, x_axis, y_axis, centers, panels = _build_panels(dataset, args)

    window = QtWidgets.QMainWindow()
    central = QtWidgets.QWidget()
    root = QtWidgets.QVBoxLayout(central)
    window.setCentralWidget(central)

    time_label = QtWidgets.QLabel(f"t = {float(times[0]):.3f}")
    if args.mode == "3d":
        time_label.setStyleSheet(
            "QLabel { color: #ffffff; font-size: 17px; font-weight: 600; "
            "padding: 4px 8px; background: rgba(0, 0, 0, 150); border-radius: 4px; }"
        )
    root.addWidget(time_label)

    controls_widget = QtWidgets.QWidget()
    controls_layout = QtWidgets.QHBoxLayout(controls_widget)
    controls_layout.setContentsMargins(0, 0, 0, 0)
    slider = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
    slider.setMinimum(0)
    slider.setMaximum(len(times) - 1)
    controls_layout.addWidget(slider, stretch=1)
    root.addWidget(controls_widget)
    if args.out is not None:
        controls_widget.hide()

    fixed_center = _fixed_center_xy(dataset)
    xy_scale = max(float(args.xy_scale), 1e-9)
    x_view = np.asarray(x_axis * xy_scale, dtype=np.float32)
    y_view = np.asarray(y_axis * xy_scale, dtype=np.float32)
    centers_view = np.asarray(centers, dtype=np.float32).copy()
    centers_view[..., 0] *= np.float32(xy_scale)
    centers_view[..., 1] *= np.float32(xy_scale)
    fixed_center_view = None
    if fixed_center is not None:
        fixed_center_view = (float(fixed_center[0]) * xy_scale, float(fixed_center[1]) * xy_scale)

    export_colorbar_lut: np.ndarray | None = None

    if args.mode == "2d":
        graphics = pg.GraphicsLayoutWidget()
        root.insertWidget(1, graphics, stretch=1)
        panel_states: list[Panel2DState] = []
        x_min, x_max = float(x_view[0]), float(x_view[-1])
        y_min, y_max = float(y_view[0]), float(y_view[-1])

        for idx, panel in enumerate(panels):
            plot = graphics.addPlot(row=0, col=idx)
            plot.showGrid(x=True, y=True, alpha=0.25)
            plot.setLabel("bottom", "x")
            plot.setLabel("left", "y")
            plot.setXRange(x_min, x_max, padding=0.0)
            plot.setYRange(y_min, y_max, padding=0.0)
            plot.disableAutoRange()
            image = pg.ImageItem()
            image.setRect(QtCore.QRectF(x_min, y_min, x_max - x_min, y_max - y_min))
            cmap = _colormap(pg, panel.signed, style="2d")
            image.setLookupTable(cmap.getLookupTable(nPts=256))
            plot.addItem(image)

            marker = None
            path = None
            anchor = None
            if args.show_center:
                path = pg.PlotDataItem(pen=pg.mkPen(SOURCE_COLOR, width=2))
                marker = pg.ScatterPlotItem(size=9, brush=pg.mkBrush(SOURCE_COLOR), pen=pg.mkPen(SOURCE_COLOR))
                plot.addItem(path)
                plot.addItem(marker)
            if args.show_fixed_center and fixed_center_view is not None:
                anchor_pen = pg.mkPen("#ffffff", width=2)
                anchor = pg.ScatterPlotItem(
                    x=[fixed_center_view[0]],
                    y=[fixed_center_view[1]],
                    size=float(args.fixed_center_size),
                    symbol="x",
                    brush=pg.mkBrush(0, 0, 0, 0),
                    pen=anchor_pen,
                )
                plot.addItem(anchor)
            panel_states.append(Panel2DState(plot=plot, image=image, marker=marker, path=path, anchor=anchor, panel=panel))

        def update_frame(frame_idx: int) -> None:
            frame_idx = int(max(0, min(frame_idx, len(times) - 1)))
            slider.blockSignals(True)
            slider.setValue(frame_idx)
            slider.blockSignals(False)
            time_label.setText(f"t = {float(times[frame_idx]):.3f}")

            for state in panel_states:
                raw = state.panel.frames[frame_idx]
                frame = _apply_transform(raw, args.transform)
                levels = state.panel.fixed_levels if args.fixed_levels else _frame_levels(frame, max_quantile=float(args.level_max_quantile))
                state.image.setImage(frame.T, autoLevels=False)
                state.image.setLevels(levels)
                state.plot.setTitle(f"{state.panel.html_label} [{levels[0]:.3g}, {levels[1]:.3g}]")

                if args.show_center and state.marker is not None and state.path is not None:
                    cx, cy = centers_view[frame_idx, 0]
                    state.marker.setData([float(cx)], [float(cy)])
                    state.path.setData(centers_view[: frame_idx + 1, 0, 0], centers_view[: frame_idx + 1, 0, 1])

    else:
        try:
            import pyqtgraph.opengl as gl
        except Exception as exc:
            raise SystemExit(f"pyqtgraph OpenGL backend unavailable: {exc}")

        content = QtWidgets.QHBoxLayout()
        root.insertLayout(1, content, stretch=1)
        stacked_mode = bool(args.stacked_planes and args.show_upper and len(panels) >= 2)
        show_gl_legend = args.out is None
        span_x = float(x_view[-1] - x_view[0])
        span_y = float(y_view[-1] - y_view[0])
        span = max(span_x, span_y, 1.0)
        camera_distance = max(0.05, float(args.camera_distance_scale) * span)
        if stacked_mode:
            render_cfg = dataset["metadata"].get("config", {}).get("render", {})
            if args.upper_plane_shift is None:
                upper_shift = float(render_cfg.get("upper_plane_shift", 2.0))
            else:
                upper_shift = float(args.upper_plane_shift)
            upper_panel = panels[1]

            view = gl.GLViewWidget()
            view.setBackgroundColor(theme["background"])
            view.setCameraPosition(
                distance=camera_distance,
                elevation=float(args.camera_elevation),
                azimuth=float(args.camera_azimuth),
            )
            view.opts["center"] = QtGui.QVector3D(0.0, 0.0, 0.50 * upper_shift)
            content.addWidget(view, stretch=1)

            base = np.zeros((len(x_view), len(y_view)), dtype=np.float32)
            base_colors = np.zeros((base.size, 4), dtype=np.float32)
            base_colors[:, 3] = 0.28
            floor = gl.GLSurfacePlotItem(
                x=np.asarray(x_view, dtype=np.float32),
                y=np.asarray(y_view, dtype=np.float32),
                z=base,
                colors=base_colors,
                shader="shaded",
                smooth=False,
                computeNormals=False,
                drawEdges=False,
                glOptions="translucent",
            )
            view.addItem(floor)

            upper_base = np.full((len(x_view), len(y_view)), upper_shift, dtype=np.float32)
            upper_base_colors = np.zeros((upper_base.size, 4), dtype=np.float32)
            upper_base_colors[:, 3] = 0.20
            upper_plane = gl.GLSurfacePlotItem(
                x=np.asarray(x_view, dtype=np.float32),
                y=np.asarray(y_view, dtype=np.float32),
                z=upper_base,
                colors=upper_base_colors,
                shader="shaded",
                smooth=False,
                computeNormals=False,
                drawEdges=False,
                glOptions="translucent",
            )
            view.addItem(upper_plane)

            cmap = _colormap(pg, upper_panel.signed, style="3d_stacked")
            if show_gl_legend:
                legend_labels = {"+": 1.0, "0": 0.5, "-": 0.0} if upper_panel.signed else {"hi": 1.0, "mid": 0.55, "0": 0.0}
                legend = gl.GLGradientLegendItem(pos=(12, 16), size=(24, 220), gradient=cmap, labels=legend_labels)
                view.addItem(legend)

            lut = np.asarray(cmap.getLookupTable(nPts=256, alpha=True, mode="float"), dtype=np.float32)
            export_colorbar_lut = lut if export_colorbar_lut is None else export_colorbar_lut
            z0_raw = upper_panel.frames[0]
            z0 = _apply_transform(z0_raw, args.transform)
            z_scale = float(upper_panel.height_target) * float(args.height_scale) / max(
                _frame_peak(z0, peak_quantile=float(args.height_peak_quantile)),
                1e-9,
            )
            levels = (
                upper_panel.fixed_levels
                if args.fixed_levels
                else _frame_levels(z0, max_quantile=float(args.level_max_quantile))
            )
            norm = np.clip((z0 - levels[0]) / max(levels[1] - levels[0], 1e-9), 0.0, 1.0)
            idx = np.clip((norm * (lut.shape[0] - 1)).astype(np.int32), 0, lut.shape[0] - 1)
            colors = np.ascontiguousarray(lut[idx].reshape(-1, 4), dtype=np.float32)
            colors[:, 3] = np.float32(np.clip(args.surface_alpha, 0.05, 1.0))
            upper_surface = gl.GLSurfacePlotItem(
                x=np.asarray(x_view, dtype=np.float32),
                y=np.asarray(y_view, dtype=np.float32),
                z=np.asarray(upper_shift + z_scale * z0, dtype=np.float32),
                colors=colors,
                shader="shaded",
                smooth=bool(args.surface_smooth),
                computeNormals=True,
                drawEdges=False,
                glOptions="translucent",
            )
            view.addItem(upper_surface)

            marker = None
            path = None
            anchor = None
            if args.show_center:
                color = pg.mkColor(SOURCE_COLOR).getRgbF()
                marker = gl.GLScatterPlotItem(
                    pos=np.zeros((1, 3), dtype=np.float32),
                    color=color,
                    size=float(args.center_size),
                    pxMode=True,
                )
                path = gl.GLLinePlotItem(pos=np.zeros((1, 3), dtype=np.float32), color=color, width=2.0, mode="line_strip", antialias=True)
                view.addItem(marker)
                view.addItem(path)
            if args.show_fixed_center and fixed_center_view is not None:
                anchor = gl.GLScatterPlotItem(
                    pos=np.array([[fixed_center_view[0], fixed_center_view[1], 0.0]], dtype=np.float32),
                    color=(0.96, 0.96, 0.96, 1.0),
                    size=float(args.fixed_center_size),
                    pxMode=True,
                )
                view.addItem(anchor)

            def update_frame(frame_idx: int) -> None:
                frame_idx = int(max(0, min(frame_idx, len(times) - 1)))
                slider.blockSignals(True)
                slider.setValue(frame_idx)
                slider.blockSignals(False)
                time_label.setText(f"t = {float(times[frame_idx]):.3f}   ({upper_panel.unicode_label} on upper plane)")

                raw = upper_panel.frames[frame_idx]
                frame = _apply_transform(raw, args.transform)
                levels = (
                    upper_panel.fixed_levels
                    if args.fixed_levels
                    else _frame_levels(frame, max_quantile=float(args.level_max_quantile))
                )
                peak = (
                    upper_panel.peak
                    if args.z_mode == "fixed"
                    else _frame_peak(frame, peak_quantile=float(args.height_peak_quantile))
                )
                z_scale_local = float(upper_panel.height_target) * float(args.height_scale) / max(peak, 1e-9)
                norm = np.clip((frame - levels[0]) / max(levels[1] - levels[0], 1e-9), 0.0, 1.0)
                lut_idx = np.clip((norm * (lut.shape[0] - 1)).astype(np.int32), 0, lut.shape[0] - 1)
                colors = np.ascontiguousarray(lut[lut_idx].reshape(-1, 4), dtype=np.float32)
                colors[:, 3] = np.float32(np.clip(args.surface_alpha, 0.05, 1.0))
                upper_surface.setData(z=np.asarray(upper_shift + z_scale_local * frame, dtype=np.float32), colors=colors)

                if args.show_center and marker is not None and path is not None:
                    z_mark = np.float32(0.03 * float(args.height_scale))
                    cx, cy = centers_view[frame_idx, 0]
                    marker.setData(pos=np.array([[cx, cy, z_mark]], dtype=np.float32))
                    hist = np.zeros((frame_idx + 1, 3), dtype=np.float32)
                    hist[:, 0] = centers_view[: frame_idx + 1, 0, 0]
                    hist[:, 1] = centers_view[: frame_idx + 1, 0, 1]
                    hist[:, 2] = z_mark
                    path.setData(pos=hist)
                if args.show_fixed_center and anchor is not None and fixed_center_view is not None:
                    z_anchor = np.float32(0.03 * float(args.height_scale))
                    anchor.setData(pos=np.array([[fixed_center_view[0], fixed_center_view[1], z_anchor]], dtype=np.float32))
        else:
            panel_states: list[Panel3DState] = []
            for panel in panels:
                view = gl.GLViewWidget()
                view.setBackgroundColor(theme["background"])
                view.setCameraPosition(
                    distance=camera_distance,
                    elevation=float(args.camera_elevation),
                    azimuth=float(args.camera_azimuth),
                )
                view.opts["center"] = QtGui.QVector3D(0.0, 0.0, 0.0)
                cmap = _colormap(pg, panel.signed, style="3d")
                lut = np.asarray(cmap.getLookupTable(nPts=256, alpha=True, mode="float"), dtype=np.float32)
                export_colorbar_lut = lut if export_colorbar_lut is None else export_colorbar_lut
                z0_raw = panel.frames[0]
                z0 = _apply_transform(z0_raw, args.transform)
                z_scale = float(panel.height_target) * float(args.height_scale) / max(
                    _frame_peak(z0, peak_quantile=float(args.height_peak_quantile)),
                    1e-9,
                )
                levels = (
                    panel.fixed_levels
                    if args.fixed_levels
                    else _frame_levels(z0, max_quantile=float(args.level_max_quantile))
                )
                norm = np.clip((z0 - levels[0]) / max(levels[1] - levels[0], 1e-9), 0.0, 1.0)
                idx = np.clip((norm * (lut.shape[0] - 1)).astype(np.int32), 0, lut.shape[0] - 1)
                colors = np.ascontiguousarray(lut[idx].reshape(-1, 4), dtype=np.float32)
                surf = gl.GLSurfacePlotItem(
                    x=np.asarray(x_view, dtype=np.float32),
                    y=np.asarray(y_view, dtype=np.float32),
                    z=np.asarray(z_scale * z0, dtype=np.float32),
                    colors=colors,
                    shader="shaded",
                    smooth=bool(args.surface_smooth),
                    computeNormals=True,
                    drawEdges=False,
                    glOptions="translucent",
                )
                view.addItem(surf)
                legend = None
                if show_gl_legend:
                    legend_labels = {"+": 1.0, "0": 0.5, "-": 0.0} if panel.signed else {"hi": 1.0, "mid": 0.55, "0": 0.0}
                    legend = gl.GLGradientLegendItem(pos=(12, 16), size=(24, 220), gradient=cmap, labels=legend_labels)
                    view.addItem(legend)

                marker = None
                path = None
                anchor = None
                if args.show_center:
                    color = pg.mkColor(SOURCE_COLOR).getRgbF()
                    marker = gl.GLScatterPlotItem(
                        pos=np.zeros((1, 3), dtype=np.float32),
                        color=color,
                        size=float(args.center_size),
                        pxMode=True,
                    )
                    path = gl.GLLinePlotItem(pos=np.zeros((1, 3), dtype=np.float32), color=color, width=2.0, mode="line_strip", antialias=True)
                    view.addItem(marker)
                    view.addItem(path)
                if args.show_fixed_center and fixed_center_view is not None:
                    anchor = gl.GLScatterPlotItem(
                        pos=np.array([[fixed_center_view[0], fixed_center_view[1], 0.0]], dtype=np.float32),
                        color=(0.96, 0.96, 0.96, 1.0),
                        size=float(args.fixed_center_size),
                        pxMode=True,
                    )
                    view.addItem(anchor)

                panel_states.append(
                    Panel3DState(
                        view=view,
                        surface=surf,
                        legend=legend,
                        marker=marker,
                        path=path,
                        anchor=anchor,
                        panel=panel,
                        lut=lut,
                    )
                )
                content.addWidget(view, stretch=1)

            def update_frame(frame_idx: int) -> None:
                frame_idx = int(max(0, min(frame_idx, len(times) - 1)))
                slider.blockSignals(True)
                slider.setValue(frame_idx)
                slider.blockSignals(False)
                time_label.setText(f"t = {float(times[frame_idx]):.3f}")

                for state in panel_states:
                    raw = state.panel.frames[frame_idx]
                    frame = _apply_transform(raw, args.transform)
                    levels = (
                        state.panel.fixed_levels
                        if args.fixed_levels
                        else _frame_levels(frame, max_quantile=float(args.level_max_quantile))
                    )
                    peak = (
                        state.panel.peak
                        if args.z_mode == "fixed"
                        else _frame_peak(frame, peak_quantile=float(args.height_peak_quantile))
                    )
                    z_scale = float(state.panel.height_target) * float(args.height_scale) / max(peak, 1e-9)
                    norm = np.clip((frame - levels[0]) / max(levels[1] - levels[0], 1e-9), 0.0, 1.0)
                    lut_idx = np.clip((norm * (state.lut.shape[0] - 1)).astype(np.int32), 0, state.lut.shape[0] - 1)
                    colors = np.ascontiguousarray(state.lut[lut_idx].reshape(-1, 4), dtype=np.float32)
                    colors[:, 3] = np.float32(np.clip(args.surface_alpha, 0.05, 1.0))
                    state.surface.setData(z=np.asarray(z_scale * frame, dtype=np.float32), colors=colors)

                    if args.show_center and state.marker is not None and state.path is not None:
                        z_mark = np.float32(0.03 * float(state.panel.height_target) * float(args.height_scale))
                        cx, cy = centers_view[frame_idx, 0]
                        state.marker.setData(pos=np.array([[cx, cy, z_mark]], dtype=np.float32))
                        hist = np.zeros((frame_idx + 1, 3), dtype=np.float32)
                        hist[:, 0] = centers_view[: frame_idx + 1, 0, 0]
                        hist[:, 1] = centers_view[: frame_idx + 1, 0, 1]
                        hist[:, 2] = z_mark
                        state.path.setData(pos=hist)
                    if args.show_fixed_center and state.anchor is not None and fixed_center_view is not None:
                        z_anchor = np.float32(0.03 * float(state.panel.height_target) * float(args.height_scale))
                        state.anchor.setData(pos=np.array([[fixed_center_view[0], fixed_center_view[1], z_anchor]], dtype=np.float32))

    def on_slider(value: int) -> None:
        update_frame(int(value))

    slider.valueChanged.connect(on_slider)

    update_frame(0)
    window.resize(int(args.window_width), int(args.window_height))
    window.show()

    if args.out is not None:
        export_sequence = _build_export_frame_sequence(
            panels[0].frames,
            len(times),
            time_slowdown=float(args.time_slowdown),
            radiation_slowmo=float(args.radiation_slowmo),
            radiation_slowmo_gamma=float(args.radiation_slowmo_gamma),
            radiation_slowmo_quantile=float(args.radiation_slowmo_quantile),
        )
        written = _save_animation(
            app,
            window,
            update_frame,
            len(times),
            args.out,
            args.fps,
            np,
            QtGui,
            frame_sequence=export_sequence,
            colorbar_lut=export_colorbar_lut if args.mode == "3d" else None,
        )
        print(f"Saved pyqtgraph render to {args.out}")
        print(f"export_frames = {written} (base={len(times)}, slowdown_factor={written / max(len(times), 1):.2f})")
        return

    app.exec()


if __name__ == "__main__":
    main()
