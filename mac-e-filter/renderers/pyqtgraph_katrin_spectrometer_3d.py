#!/usr/bin/env python3
"""Render a full-orbit electron moving inside a 3D KATRIN spectrometer scene."""

from __future__ import annotations

import argparse
import colorsys
from pathlib import Path
import sys

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from mac_e_filter.data_io import load_trajectory_dataset  # noqa: E402
from mac_e_filter.dynamics import decompose_velocity  # noqa: E402
from mac_e_filter.katrin_nominal import (  # noqa: E402
    LFCS_AXIAL_LENGTH_M,
    LFCS_INNER_RADIUS_M,
    LFCS_RADIAL_THICKNESS_M,
    LFCS_ROWS_2013,
    build_katrin_2013_field,
)


BACKGROUND = "#050914"
FOREGROUND = "#F1F5FF"
VESSEL = (0.58, 0.65, 0.76, 0.30)
VESSEL_RING = (0.68, 0.74, 0.84, 0.42)
AXIS = (0.52, 0.57, 0.66, 0.45)
NEGATIVE_CURRENT = (0.10, 0.62, 1.00, 0.42)
POSITIVE_CURRENT = (1.00, 0.27, 0.23, 0.58)
NEGATIVE_CURRENT_ARROW = (0.10, 0.62, 1.00, 1.00)
POSITIVE_CURRENT_ARROW = (1.00, 0.27, 0.23, 1.00)
ZERO_CURRENT = (0.36, 0.40, 0.48, 0.35)
TRAIL = (0.22, 0.91, 1.00, 1.00)
PARTICLE = (1.00, 0.20, 0.36, 1.00)
HALO = (1.00, 0.20, 0.36, 0.22)
START = (0.25, 0.92, 0.64, 1.00)
PARALLEL_VELOCITY = (0.32, 1.00, 0.48, 1.00)
PERPENDICULAR_VELOCITY = (1.00, 0.28, 0.85, 1.00)
TOTAL_VELOCITY = (1.00, 1.00, 1.00, 0.58)
ANALYZING_PLANE = (0.18, 0.90, 1.00, 0.13)
ANALYZING_PLANE_EDGE = (0.20, 0.92, 1.00, 0.92)
DETECTOR_EDGE = (0.92, 0.96, 1.00, 0.95)

# Published global dimensions of the main spectrometer. The true vessel is not
# an ellipsoid; this pair constrains the explicitly schematic wire envelope.
VESSEL_LENGTH_M = 23.28
VESSEL_MAX_DIAMETER_M = 9.80

# The symmetric 2013 field has its analyzing plane at the spectrometer
# mid-plane. The visible detector is placed in the documented DET-magnet
# region, but its 90 mm active diameter is enlarged by 35 for legibility.
ANALYZING_PLANE_Z_M = 0.0
ANALYZING_PLANE_DISPLAY_RADIUS_M = 4.20
DETECTOR_REGION_Z_M = 13.78
DETECTOR_PHYSICAL_DIAMETER_M = 0.090
DETECTOR_DISPLAY_SCALE = 35.0
DETECTOR_DISPLAY_RADIUS_M = (
    0.5 * DETECTOR_PHYSICAL_DIAMETER_M * DETECTOR_DISPLAY_SCALE
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset",
        type=Path,
        default=PROJECT_ROOT
        / "datasets"
        / "katrin_2013_collimation_ensemble.npz",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Write MP4 instead of opening the interactive viewer.",
    )
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--duration-s", type=float, default=12.0)
    parser.add_argument("--hold-start-s", type=float, default=0.7)
    parser.add_argument("--hold-end-s", type=float, default=1.3)
    parser.add_argument("--window-width", type=int, default=1600)
    parser.add_argument("--window-height", type=int, default=900)
    parser.add_argument("--interval-ms", type=int, default=25)
    parser.add_argument("--camera-distance", type=float, default=34.0)
    parser.add_argument("--camera-azimuth", type=float, default=-70.0)
    parser.add_argument("--camera-elevation", type=float, default=8.0)
    parser.add_argument(
        "--camera-orbit-deg",
        type=float,
        default=0.0,
        help="Optional visual camera orbit; zero keeps projected vector lengths comparable.",
    )
    parser.add_argument(
        "--velocity-vector-length-m",
        type=float,
        default=2.2,
        help="Display length corresponding to the total speed; direction and component fractions are physical.",
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
    start = np.zeros(
        max(0, int(round(hold_start_s * fps))),
        dtype=np.int32,
    )
    end = np.full(
        max(0, int(round(hold_end_s * fps))),
        sample_count - 1,
        dtype=np.int32,
    )
    return np.concatenate((start, moving, end))


def _circle_segments(
    axial_positions_m: np.ndarray,
    radius_m: float,
    *,
    samples: int = 160,
) -> np.ndarray:
    """Return independent line segments for rings around the physical z axis.

    Scene coordinates are ``(physical z, physical x, physical y)``.
    """

    theta = np.linspace(0.0, 2.0 * np.pi, samples + 1)
    circle_y = radius_m * np.cos(theta)
    circle_z = radius_m * np.sin(theta)
    all_segments: list[np.ndarray] = []
    for axial_position in axial_positions_m:
        ring = np.column_stack(
            (
                np.full(theta.size, axial_position),
                circle_y,
                circle_z,
            )
        )
        segments = np.empty((2 * samples, 3), dtype=np.float32)
        segments[0::2] = ring[:-1]
        segments[1::2] = ring[1:]
        all_segments.append(segments)
    return np.concatenate(all_segments)


def _current_arrow_segments(
    axial_position_m: float,
    radius_m: float,
    sign: float,
) -> np.ndarray:
    """Three arrowheads indicating conventional-current direction."""

    result: list[np.ndarray] = []
    for theta in (-0.75, 1.35, 3.45):
        radial = np.array([0.0, np.cos(theta), np.sin(theta)])
        tangent = sign * np.array([0.0, -np.sin(theta), np.cos(theta)])
        tip = np.array(
            [
                axial_position_m,
                radius_m * np.cos(theta),
                radius_m * np.sin(theta),
            ]
        )
        back = tip - 0.42 * tangent
        first_wing = back + 0.16 * radial
        second_wing = back - 0.16 * radial
        result.append(np.vstack((tip, first_wing, tip, second_wing)))
    return np.asarray(np.concatenate(result), dtype=np.float32)


def _vessel_radius(axial_position_m: np.ndarray) -> np.ndarray:
    """Schematic ellipsoidal envelope constrained by published max dimensions."""

    half_length = 0.5 * VESSEL_LENGTH_M
    max_radius = 0.5 * VESSEL_MAX_DIAMETER_M
    normalized = np.clip(axial_position_m / half_length, -1.0, 1.0)
    return max_radius * np.sqrt(np.maximum(0.0, 1.0 - normalized**2))


def _add_vessel_wireframe(view, gl) -> None:
    axial = np.linspace(
        -0.5 * VESSEL_LENGTH_M,
        0.5 * VESSEL_LENGTH_M,
        241,
    )
    radius = _vessel_radius(axial)
    for theta in np.linspace(0.0, 2.0 * np.pi, 14, endpoint=False):
        meridian = np.column_stack(
            (
                axial,
                radius * np.cos(theta),
                radius * np.sin(theta),
            )
        ).astype(np.float32)
        item = gl.GLLinePlotItem(
            pos=meridian,
            color=VESSEL,
            width=1.0,
            mode="line_strip",
            antialias=True,
        )
        item.setGLOptions("translucent")
        view.addItem(item)

    ring_positions = np.linspace(
        -0.42 * VESSEL_LENGTH_M,
        0.42 * VESSEL_LENGTH_M,
        11,
    )
    for axial_position in ring_positions:
        ring_radius = float(_vessel_radius(np.array([axial_position]))[0])
        item = gl.GLLinePlotItem(
            pos=_circle_segments(
                np.array([axial_position]),
                ring_radius,
                samples=120,
            ),
            color=VESSEL_RING,
            width=1.0,
            mode="lines",
            antialias=True,
        )
        item.setGLOptions("translucent")
        view.addItem(item)


def _add_lfcs_coils(view, gl, configuration: str) -> None:
    radius = LFCS_INNER_RADIUS_M + 0.5 * LFCS_RADIAL_THICKNESS_M
    for row in LFCS_ROWS_2013:
        current = row.current(configuration)
        if current > 0.0:
            color = POSITIVE_CURRENT
        elif current < 0.0:
            color = NEGATIVE_CURRENT
        else:
            color = ZERO_CURRENT

        for center in row.centers_z_m:
            turn_offsets = (
                (np.arange(row.turns_each, dtype=float) + 0.5)
                / row.turns_each
                - 0.5
            ) * LFCS_AXIAL_LENGTH_M
            turn_positions = center + turn_offsets
            turns = gl.GLLinePlotItem(
                pos=_circle_segments(turn_positions, radius),
                color=color,
                width=1.0,
                mode="lines",
                antialias=True,
            )
            turns.setGLOptions("translucent")
            view.addItem(turns)

            if current != 0.0:
                arrows = gl.GLLinePlotItem(
                    pos=_current_arrow_segments(
                        center,
                        radius + 0.06,
                        np.sign(current),
                    ),
                    color=(
                        POSITIVE_CURRENT_ARROW
                        if current > 0.0
                        else NEGATIVE_CURRENT_ARROW
                    ),
                    width=2.5,
                    mode="lines",
                    antialias=True,
                )
                view.addItem(arrows)


def _add_analyzing_plane(view, gl, qt_core, qt_gui) -> None:
    """Draw the symmetric analyzing plane at physical z=0."""

    samples = 128
    theta = np.linspace(0.0, 2.0 * np.pi, samples, endpoint=False)
    vertices = np.vstack(
        (
            np.array([[ANALYZING_PLANE_Z_M, 0.0, 0.0]]),
            np.column_stack(
                (
                    np.full(samples, ANALYZING_PLANE_Z_M),
                    ANALYZING_PLANE_DISPLAY_RADIUS_M * np.cos(theta),
                    ANALYZING_PLANE_DISPLAY_RADIUS_M * np.sin(theta),
                )
            ),
        )
    ).astype(np.float32)
    faces = np.array(
        [
            [0, 1 + index, 1 + ((index + 1) % samples)]
            for index in range(samples)
        ],
        dtype=np.uint32,
    )
    face_colors = np.tile(
        np.asarray(ANALYZING_PLANE, dtype=np.float32),
        (samples, 1),
    )
    mesh_data = gl.MeshData(
        vertexes=vertices,
        faces=faces,
        faceColors=face_colors,
    )
    plane = gl.GLMeshItem(
        meshdata=mesh_data,
        smooth=False,
        drawEdges=False,
    )
    plane.setGLOptions("translucent")
    view.addItem(plane)

    edge = gl.GLLinePlotItem(
        pos=_circle_segments(
            np.array([ANALYZING_PLANE_Z_M]),
            ANALYZING_PLANE_DISPLAY_RADIUS_M,
            samples=samples,
        ),
        color=ANALYZING_PLANE_EDGE,
        width=2.4,
        mode="lines",
        antialias=True,
    )
    view.addItem(edge)

    cross = np.array(
        [
            [0.0, -ANALYZING_PLANE_DISPLAY_RADIUS_M, 0.0],
            [0.0, +ANALYZING_PLANE_DISPLAY_RADIUS_M, 0.0],
            [0.0, 0.0, -ANALYZING_PLANE_DISPLAY_RADIUS_M],
            [0.0, 0.0, +ANALYZING_PLANE_DISPLAY_RADIUS_M],
        ],
        dtype=np.float32,
    )
    cross_item = gl.GLLinePlotItem(
        pos=cross,
        color=(0.20, 0.92, 1.00, 0.42),
        width=1.2,
        mode="lines",
        antialias=True,
    )
    cross_item.setGLOptions("translucent")
    view.addItem(cross_item)

    font = qt_gui.QFont("Helvetica", 16)
    font.setBold(True)
    label = gl.GLTextItem(
        pos=np.array([0.0, -4.35, 3.35]),
        color=qt_gui.QColor("#5DEBFF"),
        text="АНАЛИЗИРУЮЩАЯ ПЛОСКОСТЬ  z = 0",
        font=font,
        alignment=(
            qt_core.Qt.AlignmentFlag.AlignHCenter
            | qt_core.Qt.AlignmentFlag.AlignVCenter
        ),
    )
    view.addItem(label)


def _detector_pixel_color(ring: int, sector: int) -> tuple[float, ...]:
    """Color the exact 4 + 12×12 detector segmentation."""

    radial_fraction = ring / 12.0
    hue = 0.61 - 0.49 * radial_fraction
    value = 0.86 + 0.10 * ((sector + ring) % 2)
    red, green, blue = colorsys.hsv_to_rgb(hue, 0.78, value)
    return red, green, blue, 0.93


def _add_segmented_detector(view, gl, qt_core, qt_gui) -> None:
    """Draw the 148-pixel focal-plane detector with explicit display scale."""

    vertices: list[list[float]] = []
    faces: list[list[int]] = []
    face_colors: list[tuple[float, ...]] = []

    def add_triangle(
        first: tuple[float, float],
        second: tuple[float, float],
        third: tuple[float, float],
        color: tuple[float, ...],
    ) -> None:
        start = len(vertices)
        for radius, angle in (first, second, third):
            vertices.append(
                [
                    DETECTOR_REGION_Z_M,
                    radius * np.cos(angle),
                    radius * np.sin(angle),
                ]
            )
        faces.append([start, start + 1, start + 2])
        face_colors.append(color)

    bullseye_radius = DETECTOR_DISPLAY_RADIUS_M * np.sqrt(4.0 / 148.0)
    arc_subdivisions = 4
    for sector in range(4):
        start_angle = 0.5 * np.pi * sector
        end_angle = 0.5 * np.pi * (sector + 1)
        angles = np.linspace(
            start_angle,
            end_angle,
            arc_subdivisions + 1,
        )
        color = _detector_pixel_color(0, sector)
        for angle0, angle1 in zip(angles[:-1], angles[1:], strict=True):
            add_triangle(
                (0.0, angle0),
                (bullseye_radius, angle0),
                (bullseye_radius, angle1),
                color,
            )

    radial_boundaries = [bullseye_radius]
    for ring in range(1, 13):
        inner_radius = (
            DETECTOR_DISPLAY_RADIUS_M
            * np.sqrt((4.0 + 12.0 * (ring - 1)) / 148.0)
        )
        outer_radius = (
            DETECTOR_DISPLAY_RADIUS_M
            * np.sqrt((4.0 + 12.0 * ring) / 148.0)
        )
        radial_boundaries.append(outer_radius)
        for sector in range(12):
            start_angle = 2.0 * np.pi * sector / 12.0
            end_angle = 2.0 * np.pi * (sector + 1) / 12.0
            angles = np.linspace(
                start_angle,
                end_angle,
                arc_subdivisions + 1,
            )
            color = _detector_pixel_color(ring, sector)
            for angle0, angle1 in zip(
                angles[:-1],
                angles[1:],
                strict=True,
            ):
                start = len(vertices)
                vertices.extend(
                    [
                        [
                            DETECTOR_REGION_Z_M,
                            inner_radius * np.cos(angle0),
                            inner_radius * np.sin(angle0),
                        ],
                        [
                            DETECTOR_REGION_Z_M,
                            outer_radius * np.cos(angle0),
                            outer_radius * np.sin(angle0),
                        ],
                        [
                            DETECTOR_REGION_Z_M,
                            outer_radius * np.cos(angle1),
                            outer_radius * np.sin(angle1),
                        ],
                        [
                            DETECTOR_REGION_Z_M,
                            inner_radius * np.cos(angle1),
                            inner_radius * np.sin(angle1),
                        ],
                    ]
                )
                faces.extend(
                    [
                        [start, start + 1, start + 2],
                        [start, start + 2, start + 3],
                    ]
                )
                face_colors.extend((color, color))

    mesh_data = gl.MeshData(
        vertexes=np.asarray(vertices, dtype=np.float32),
        faces=np.asarray(faces, dtype=np.uint32),
        faceColors=np.asarray(face_colors, dtype=np.float32),
    )
    detector = gl.GLMeshItem(
        meshdata=mesh_data,
        smooth=False,
        drawEdges=False,
    )
    detector.setGLOptions("translucent")
    view.addItem(detector)

    for radius in radial_boundaries:
        ring_item = gl.GLLinePlotItem(
            pos=_circle_segments(
                np.array([DETECTOR_REGION_Z_M]),
                radius,
                samples=96,
            ),
            color=DETECTOR_EDGE,
            width=1.0,
            mode="lines",
            antialias=True,
        )
        view.addItem(ring_item)

    radial_segments: list[np.ndarray] = []
    for sector in range(12):
        angle = 2.0 * np.pi * sector / 12.0
        radial_segments.append(
            np.array(
                [
                    [
                        DETECTOR_REGION_Z_M,
                        bullseye_radius * np.cos(angle),
                        bullseye_radius * np.sin(angle),
                    ],
                    [
                        DETECTOR_REGION_Z_M,
                        DETECTOR_DISPLAY_RADIUS_M * np.cos(angle),
                        DETECTOR_DISPLAY_RADIUS_M * np.sin(angle),
                    ],
                ]
            )
        )
    for sector in range(4):
        angle = 0.5 * np.pi * sector
        radial_segments.append(
            np.array(
                [
                    [DETECTOR_REGION_Z_M, 0.0, 0.0],
                    [
                        DETECTOR_REGION_Z_M,
                        bullseye_radius * np.cos(angle),
                        bullseye_radius * np.sin(angle),
                    ],
                ]
            )
        )
    radial_item = gl.GLLinePlotItem(
        pos=np.asarray(
            np.concatenate(radial_segments),
            dtype=np.float32,
        ),
        color=DETECTOR_EDGE,
        width=1.0,
        mode="lines",
        antialias=True,
    )
    view.addItem(radial_item)

    font = qt_gui.QFont("Helvetica", 16)
    font.setBold(True)
    label = gl.GLTextItem(
        pos=np.array([DETECTOR_REGION_Z_M, 0.0, 1.75]),
        color=qt_gui.QColor("#FFE66D"),
        text="FPD: 148 СЕГМЕНТОВ  (размер ×35)",
        font=font,
        alignment=(
            qt_core.Qt.AlignmentFlag.AlignHCenter
            | qt_core.Qt.AlignmentFlag.AlignVCenter
        ),
    )
    view.addItem(label)


def _build_current_table(qt_core, qt_widgets, configuration: str):
    """Build a compact three-column LFCS current table."""

    widget = qt_widgets.QGroupBox("Корректирующие токи LFCS, А")
    widget.setStyleSheet(
        "QGroupBox { color: #DCE6F8; border: 1px solid #34445E; "
        "border-radius: 7px; margin-top: 10px; font-size: 15px; "
        "font-weight: 600; } "
        "QGroupBox::title { subcontrol-origin: margin; left: 8px; "
        "padding: 0 4px; }"
    )
    layout = qt_widgets.QGridLayout(widget)
    layout.setContentsMargins(8, 17, 8, 8)
    layout.setHorizontalSpacing(6)
    layout.setVerticalSpacing(6)
    entries = []
    for row in LFCS_ROWS_2013:
        current = row.current(configuration)
        if current > 0.0:
            background = "#4A1820"
            border = "#FF5B55"
        elif current < 0.0:
            background = "#0C2942"
            border = "#20A8FF"
        else:
            background = "#252A35"
            border = "#697589"
        for subcoil, _center in enumerate(row.centers_z_m):
            suffix = (
                chr(ord("a") + subcoil)
                if len(row.centers_z_m) > 1
                else ""
            )
            entries.append((f"L{row.index}{suffix}", current, background, border))

    for entry_index, (name, current, background, border) in enumerate(entries):
        table_row, table_column = divmod(entry_index, 3)
        chip = qt_widgets.QLabel(
            f"<b>{name}</b>&nbsp;&nbsp;{current:+.1f}"
        )
        chip.setTextFormat(qt_core.Qt.TextFormat.RichText)
        chip.setAlignment(qt_core.Qt.AlignmentFlag.AlignCenter)
        chip.setStyleSheet(
            f"QLabel {{ color: #F4F7FF; background-color: {background}; "
            f"border: 1px solid {border}; border-radius: 5px; "
            "font-size: 13px; padding: 5px 2px; }"
        )
        layout.addWidget(chip, table_row, table_column)

    note = qt_widgets.QLabel(
        "слева → направо: источник → детектор\n"
        "это не сильнополевые соленоиды\n"
        "красные L14a/b — counter-coil"
    )
    note.setAlignment(qt_core.Qt.AlignmentFlag.AlignCenter)
    note.setStyleSheet(
        "color: #AEBAD0; font-size: 12px; padding-top: 4px;"
    )
    layout.addWidget(note, 5, 0, 1, 3)
    return widget


def _build_field_profile(pg, qt_core, qt_widgets, field_model):
    """Show the orders-of-magnitude field scale omitted by the 3D orbit."""

    widget = qt_widgets.QGroupBox("Поле на оси |B(z)|")
    widget.setStyleSheet(
        "QGroupBox { color: #DCE6F8; border: 1px solid #34445E; "
        "border-radius: 7px; margin-top: 10px; font-size: 15px; "
        "font-weight: 600; } "
        "QGroupBox::title { subcontrol-origin: margin; left: 8px; "
        "padding: 0 4px; }"
    )
    layout = qt_widgets.QVBoxLayout(widget)
    layout.setContentsMargins(5, 12, 5, 6)
    layout.setSpacing(2)

    plot = pg.PlotWidget()
    plot.setFixedHeight(150)
    plot.setBackground("#07101D")
    plot.getPlotItem().setMenuEnabled(False)
    plot.setMouseEnabled(x=False, y=False)
    plot.showGrid(x=True, y=True, alpha=0.20)
    plot.setLabel("bottom", "z", units="м")
    plot.setLabel("left", "|B|", units="Тл")
    plot.setLogMode(y=True)

    axial_position = np.linspace(-40.0, 15.0, 2401)
    field_magnitude = np.abs(field_model.total.axis_field(axial_position))
    central_region = pg.LinearRegionItem(
        values=(-6.5, 6.5),
        movable=False,
        brush=(45, 215, 255, 30),
        pen=pg.mkPen("#35D7FF", width=1.0),
    )
    plot.addItem(central_region)
    plot.plot(
        axial_position,
        field_magnitude,
        pen=pg.mkPen("#FFD166", width=2.0),
    )
    marker_z = np.array([-38.87, 0.0, 12.18])
    marker_b = np.abs(field_model.total.axis_field(marker_z))
    plot.plot(
        marker_z,
        marker_b,
        pen=None,
        symbol="o",
        symbolSize=6,
        symbolBrush="#FF6B6B",
    )
    plot.setXRange(-40.0, 15.0, padding=0.0)
    plot.setYRange(-4.0, 1.0, padding=0.02)
    layout.addWidget(plot)

    explanation = qt_widgets.QLabel(
        "WGTS 3.6 Тл · PS2 4.5 Тл · PCH 6 Тл\n"
        "голубая область орбит: 0.35–1.01 мТл"
    )
    explanation.setAlignment(qt_core.Qt.AlignmentFlag.AlignCenter)
    explanation.setStyleSheet(
        "color: #C8D4E8; font-size: 10px; padding-top: 2px;"
    )
    layout.addWidget(explanation)
    return widget


def _build_velocity_gauges(qt_core, qt_widgets, tracks_metadata):
    """Build fixed-screen gauges immune to 3D projection."""

    widget = qt_widgets.QGroupBox("Компоненты скорости")
    widget.setStyleSheet(
        "QGroupBox { color: #DCE6F8; border: 1px solid #34445E; "
        "border-radius: 6px; margin-top: 8px; font-size: 12px; "
        "font-weight: 600; } "
        "QGroupBox::title { subcontrol-origin: margin; left: 8px; "
        "padding: 0 4px; }"
    )
    layout = qt_widgets.QVBoxLayout(widget)
    layout.setContentsMargins(9, 17, 9, 9)
    layout.setSpacing(4)
    parallel_bars = []
    perpendicular_bars = []
    for track in tracks_metadata:
        label = qt_widgets.QLabel(
            f"<span style='color:{track['color']};'>●</span> "
            f"<b>{track['label']}</b>"
        )
        label.setTextFormat(qt_core.Qt.TextFormat.RichText)
        label.setStyleSheet("color: #E7EEFA; font-size: 13px;")
        layout.addWidget(label)

        parallel_bar = qt_widgets.QProgressBar()
        parallel_bar.setRange(0, 1000)
        parallel_bar.setTextVisible(True)
        parallel_bar.setFixedHeight(24)
        parallel_bar.setStyleSheet(
            "QProgressBar { color: #F5FFF7; background-color: #102219; "
            "border: 1px solid #318F49; border-radius: 3px; "
            "text-align: center; font-size: 12px; } "
            "QProgressBar::chunk { background-color: #2BCB55; }"
        )
        layout.addWidget(parallel_bar)
        parallel_bars.append(parallel_bar)

        perpendicular_bar = qt_widgets.QProgressBar()
        perpendicular_bar.setRange(0, 1000)
        perpendicular_bar.setTextVisible(True)
        perpendicular_bar.setFixedHeight(24)
        perpendicular_bar.setStyleSheet(
            "QProgressBar { color: #FFF5FD; background-color: #261124; "
            "border: 1px solid #B735A3; border-radius: 3px; "
            "text-align: center; font-size: 12px; } "
            "QProgressBar::chunk { background-color: #DE42C8; }"
        )
        layout.addWidget(perpendicular_bar)
        perpendicular_bars.append(perpendicular_bar)

    return widget, parallel_bars, perpendicular_bars


def main() -> None:
    args = parse_args()
    if args.fps <= 0:
        raise SystemExit("--fps must be positive")
    if args.out is not None and (
        args.window_width % 2 or args.window_height % 2
    ):
        raise SystemExit(
            "MP4 export requires even --window-width and --window-height"
        )

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
    if metadata.get("dataset_type") == "katrin_collimation_ensemble":
        configuration = str(metadata["field_configuration"])
    else:
        configuration = str(metadata["scenario"]["field_configuration"])
    field_model = build_katrin_2013_field(configuration)
    time_s = np.asarray(dataset["time_s"], dtype=float)
    position = np.asarray(dataset["position_m"], dtype=float)
    normalized_momentum = np.asarray(
        dataset["normalized_momentum"], dtype=float
    )
    velocity = np.asarray(dataset["velocity_m_s"], dtype=float)
    magnetic_field = np.asarray(dataset["magnetic_field_t"], dtype=float)
    field_microtesla = (
        1.0e6 * np.asarray(dataset["magnetic_field_magnitude_t"], dtype=float)
    )
    pitch_deg = np.asarray(dataset["pitch_angle_deg"], dtype=float)
    if position.ndim == 2:
        position = position[None, ...]
        normalized_momentum = normalized_momentum[None, ...]
        velocity = velocity[None, ...]
        magnetic_field = magnetic_field[None, ...]
        field_microtesla = field_microtesla[None, ...]
        pitch_deg = pitch_deg[None, ...]
        active = np.ones(position.shape[:2], dtype=bool)
        tracks_metadata = [
            {
                "label": "локальный α₀=30°",
                "color": "#35D5FF",
                "initial_local_pitch_deg": float(pitch_deg[0, 0]),
            }
        ]
    elif position.ndim == 3:
        active = np.asarray(dataset["active"], dtype=bool)
        tracks_metadata = list(metadata["tracks"])
    else:
        raise SystemExit("Unexpected position_m dimensions")
    track_count = position.shape[0]

    particle_metadata = metadata["particle"]
    mass_kg = float(particle_metadata["mass_kg"])
    charge_c = abs(float(particle_metadata["charge_c"]))
    gamma = np.sqrt(
        1.0
        + np.einsum(
            "tni,tni->tn",
            normalized_momentum,
            normalized_momentum,
        )
    )
    gyrofrequency = (
        charge_c
        * np.linalg.norm(magnetic_field, axis=2)
        / (gamma * mass_kg)
    )
    scene_trajectory = np.ascontiguousarray(
        np.stack(
            (
                position[:, :, 2],
                position[:, :, 0],
                position[:, :, 1],
            ),
            axis=-1,
        ),
        dtype=np.float32,
    )
    cyclotron_frequency_hz = gyrofrequency / (2.0 * np.pi)
    cumulative_turns = np.zeros((track_count, time_s.size), dtype=float)
    cumulative_turns[:, 1:] = np.cumsum(
        0.5
        * (
            cyclotron_frequency_hz[:, 1:]
            + cyclotron_frequency_hz[:, :-1]
        )
        * np.diff(time_s)[None, :],
        axis=1,
    )

    pg.setConfigOption("background", BACKGROUND)
    pg.setConfigOption("foreground", FOREGROUND)
    pg.setConfigOption("antialias", True)
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])

    window = QtWidgets.QMainWindow()
    window.setWindowTitle("KATRIN: полные трёхмерные орбиты электронов")
    central = QtWidgets.QWidget()
    central.setStyleSheet(f"background-color: {BACKGROUND};")
    root = QtWidgets.QVBoxLayout(central)
    root.setContentsMargins(6, 6, 6, 6)
    root.setSpacing(3)
    window.setCentralWidget(central)

    title = QtWidgets.QLabel()
    title.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
    title.setStyleSheet(
        f"color: {FOREGROUND}; font-size: 22px; font-weight: 600; "
        "padding: 5px;"
    )
    root.addWidget(title)

    geometry_caption = QtWidgets.QLabel(
        "<span style='color:#5DEBFF;'>●</span> "
        "<b>анализирующая плоскость z=0</b>"
        "&nbsp;&nbsp;&nbsp;&nbsp;"
        "<span style='color:#FFE66D;'>◉</span> "
        "<b>FPD</b> — отдельный 148-сегментный детектор после спектрометра"
    )
    geometry_caption.setTextFormat(QtCore.Qt.TextFormat.RichText)
    geometry_caption.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
    geometry_caption.setStyleSheet(
        "color: #CFD8EA; font-size: 12px; padding: 1px;"
    )
    root.addWidget(geometry_caption)

    content = QtWidgets.QWidget()
    content_layout = QtWidgets.QHBoxLayout(content)
    content_layout.setContentsMargins(0, 0, 0, 0)
    content_layout.setSpacing(5)

    sidebar = QtWidgets.QWidget()
    sidebar.setFixedWidth(285)
    sidebar_layout = QtWidgets.QVBoxLayout(sidebar)
    sidebar_layout.setContentsMargins(0, 0, 0, 0)
    sidebar_layout.setSpacing(4)
    sidebar_layout.addWidget(
        _build_current_table(QtCore, QtWidgets, configuration)
    )
    sidebar_layout.addWidget(
        _build_field_profile(
            pg,
            QtCore,
            QtWidgets,
            field_model,
        )
    )
    (
        velocity_gauges,
        parallel_speed_bars,
        perpendicular_speed_bars,
    ) = _build_velocity_gauges(
        QtCore,
        QtWidgets,
        tracks_metadata,
    )
    sidebar_layout.addWidget(velocity_gauges)
    sidebar_layout.addStretch(1)
    content_layout.addWidget(sidebar)

    view = gl.GLViewWidget()
    view.setBackgroundColor(BACKGROUND)
    view.opts["center"] = QtGui.QVector3D(1.0, 0.0, 0.0)
    view.setCameraPosition(
        distance=args.camera_distance,
        elevation=args.camera_elevation,
        azimuth=args.camera_azimuth,
    )
    content_layout.addWidget(view, stretch=1)
    root.addWidget(content, stretch=1)

    _add_vessel_wireframe(view, gl)
    _add_lfcs_coils(view, gl, configuration)
    _add_analyzing_plane(view, gl, QtCore, QtGui)
    _add_segmented_detector(view, gl, QtCore, QtGui)

    beam_axis = gl.GLLinePlotItem(
        pos=np.array(
            [
                [-0.5 * VESSEL_LENGTH_M, 0.0, 0.0],
                [DETECTOR_REGION_Z_M, 0.0, 0.0],
            ],
            dtype=np.float32,
        ),
        color=AXIS,
        width=1.0,
        mode="lines",
        antialias=True,
    )
    beam_axis.setGLOptions("translucent")
    view.addItem(beam_axis)

    track_colors = [
        tuple(float(value) for value in pg.mkColor(track["color"]).getRgbF())
        for track in tracks_metadata
    ]
    start_markers = []
    trails = []
    halos = []
    particles = []
    total_velocity_items = []
    parallel_velocity_items = []
    perpendicular_velocity_items = []
    for track_index, color in enumerate(track_colors):
        halo_color = (color[0], color[1], color[2], 0.20)
        start_marker = gl.GLScatterPlotItem(
            pos=scene_trajectory[track_index, :1],
            color=START,
            size=7.0,
            pxMode=True,
        )
        trail = gl.GLLinePlotItem(
            pos=scene_trajectory[track_index, :1],
            color=color,
            width=2.6,
            mode="line_strip",
            antialias=True,
        )
        halo = gl.GLScatterPlotItem(
            pos=scene_trajectory[track_index, :1],
            color=halo_color,
            size=29.0,
            pxMode=True,
        )
        particle = gl.GLScatterPlotItem(
            pos=scene_trajectory[track_index, :1],
            color=color,
            size=13.0,
            pxMode=True,
        )
        total_velocity = gl.GLLinePlotItem(
            pos=np.zeros((2, 3), dtype=np.float32),
            color=TOTAL_VELOCITY,
            width=1.2,
            mode="lines",
            antialias=True,
        )
        parallel_velocity = gl.GLLinePlotItem(
            pos=np.zeros((2, 3), dtype=np.float32),
            color=PARALLEL_VELOCITY,
            width=3.0,
            mode="lines",
            antialias=True,
        )
        perpendicular_velocity = gl.GLLinePlotItem(
            pos=np.zeros((2, 3), dtype=np.float32),
            color=PERPENDICULAR_VELOCITY,
            width=3.0,
            mode="lines",
            antialias=True,
        )
        for item in (
            start_marker,
            trail,
            halo,
            particle,
            total_velocity,
            parallel_velocity,
            perpendicular_velocity,
        ):
            view.addItem(item)
        start_markers.append(start_marker)
        trails.append(trail)
        halos.append(halo)
        particles.append(particle)
        total_velocity_items.append(total_velocity)
        parallel_velocity_items.append(parallel_velocity)
        perpendicular_velocity_items.append(perpendicular_velocity)

    track_legend = "&nbsp;&nbsp;&nbsp;".join(
        (
            f"<span style='color:{track['color']};'>●</span> "
            f"{track['label']}"
        )
        for track in tracks_metadata
    )
    legend = QtWidgets.QLabel(
        track_legend
        + "&nbsp;&nbsp;&nbsp;&nbsp;"
        "<span style='color:#52FF7A;'>━</span> v∥"
        "&nbsp;&nbsp;"
        "<span style='color:#FF47D9;'>━</span> v⊥"
        "&nbsp;&nbsp;"
        "<span style='color:#FFFFFF;'>━</span> v"
        "&nbsp;&nbsp;&nbsp;&nbsp;"
        "α — угол между импульсом <i>p</i> и магнитной линией <i>B</i>"
    )
    legend.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
    legend.setStyleSheet(
        f"color: #CAD4E7; font-size: 14px; padding: 2px;"
    )
    root.addWidget(legend)

    mapping = metadata.get("source_to_analysis_adiabatic_mapping")
    if mapping is not None:
        source_values = mapping["source_pitch_angles_deg"]
        analysis_values = mapping["mapped_analysis_pitch_angles_deg"]
        pairs = ", ".join(
            f"{source:g}°→{analysis:.3f}°"
            for source, analysis in zip(
                source_values, analysis_values, strict=True
            )
        )
        mapping_note = QtWidgets.QLabel(
            "Адиабатическое сопоставление «источник → анализирующая плоскость» "
            f"({mapping['source_field_t']:.1f} T → "
            f"{1e3 * mapping['analyzing_field_t']:.3f} мТл): {pairs}"
        )
        mapping_note.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        mapping_note.setStyleSheet(
            "color: #D5C4FF; font-size: 12px; padding: 1px;"
        )
        root.addWidget(mapping_note)

    physical_duration_s = float(time_s[-1] - time_s[0])
    slowdown = args.duration_s / physical_duration_s
    note = QtWidgets.QLabel(
        f"Воспроизведение замедлено в {slowdown:.3e} раза; координаты орбит "
        "не масштабированы; полные орбиты рассчитаны только для "
        "−6.5 ≤ z ≤ +6.5 м. "
        f"Векторы скорости нормированы к экранной длине {args.velocity_vector_length_m:g} м. "
        "Сильнополевые магниты 3.6–6 Тл находятся вне этого интервала. "
        "FPD: 148 пикселей, физический диаметр 90 мм, показан ×35; "
        "осевое положение схематично в области магнита DET."
    )
    note.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
    note.setStyleSheet(
        "color: #9EABC2; font-size: 12px; padding: 2px;"
    )
    root.addWidget(note)

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
    last_active_indices = np.array(
        [
            int(np.flatnonzero(active[track_index])[-1])
            for track_index in range(track_count)
        ],
        dtype=int,
    )

    def update_frame(index: int) -> None:
        index = int(np.clip(index, 0, time_s.size - 1))
        progress = index / max(time_s.size - 1, 1)
        status_parts = []
        for track_index in range(track_count):
            track_frame = min(index, last_active_indices[track_index])
            current_position = scene_trajectory[
                track_index, track_frame : track_frame + 1
            ]
            trails[track_index].setData(
                pos=scene_trajectory[
                    track_index, : track_frame + 1
                ]
            )
            halos[track_index].setData(pos=current_position)
            particles[track_index].setData(pos=current_position)

            field_vector = magnetic_field[track_index, track_frame]
            velocity_vector = velocity[track_index, track_frame]
            speed = np.linalg.norm(velocity_vector)
            parallel_vector, perpendicular_vector = decompose_velocity(
                velocity_vector,
                field_vector,
            )
            parallel_fraction = np.linalg.norm(parallel_vector) / speed
            perpendicular_fraction = (
                np.linalg.norm(perpendicular_vector) / speed
            )
            parallel_speed_bars[track_index].setValue(
                int(round(1000.0 * parallel_fraction))
            )
            parallel_speed_bars[track_index].setFormat(
                f"v∥/v = {parallel_fraction:.3f}"
            )
            perpendicular_speed_bars[track_index].setValue(
                int(round(1000.0 * perpendicular_fraction))
            )
            perpendicular_speed_bars[track_index].setFormat(
                f"v⊥/v = {perpendicular_fraction:.3f}"
            )
            vector_scale = args.velocity_vector_length_m / speed

            def scene_vector(vector: np.ndarray) -> np.ndarray:
                return np.array(
                    [vector[2], vector[0], vector[1]],
                    dtype=np.float32,
                )

            origin = current_position[0]
            parallel_end = (
                origin + vector_scale * scene_vector(parallel_vector)
            )
            total_end = origin + vector_scale * scene_vector(velocity_vector)
            parallel_velocity_items[track_index].setData(
                pos=np.asarray([origin, parallel_end], dtype=np.float32)
            )
            perpendicular_velocity_items[track_index].setData(
                pos=np.asarray(
                    [parallel_end, total_end],
                    dtype=np.float32,
                )
            )
            total_velocity_items[track_index].setData(
                pos=np.asarray([origin, total_end], dtype=np.float32)
            )

            track = tracks_metadata[track_index]
            status_parts.append(
                f"<span style='color:{track['color']};'>"
                f"{track['label']}: z={position[track_index, track_frame, 2]:+.2f} м, "
                f"α={pitch_deg[track_index, track_frame]:.2f}°, "
                f"N={cumulative_turns[track_index, track_frame]:.2f}</span>"
            )

        view.setCameraPosition(
            distance=args.camera_distance,
            elevation=args.camera_elevation,
            azimuth=(
                args.camera_azimuth
                + args.camera_orbit_deg * (progress - 0.5)
            ),
        )
        title.setText(
            (
                "Магнитная коллимация: центральная область спектрометра KATRIN"
                f"<br><span style='font-size:15px; font-weight:400;'>"
                f"физическое время = {time_s[index] * 1e9:7.3f} нс; E = 0"
                "<br>"
            )
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
        window.resize(args.window_width, args.window_height)
    else:
        screen = app.primaryScreen()
        if screen is None:
            logical_scale = 1.0
        else:
            available = screen.availableGeometry()
            logical_scale = min(
                1.0,
                0.90 * available.width() / args.window_width,
                0.90 * available.height() / args.window_height,
            )
        window.resize(
            max(1, int(round(args.window_width * logical_scale))),
            max(1, int(round(args.window_height * logical_scale))),
        )
    window.show()
    for _ in range(4):
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
            view.update()
            app.processEvents()
            frame_image = window.grab().toImage()
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
                _qimage_to_rgb_array(frame_image, QtGui)
            )
            if frame_number % max(args.fps * 2, 1) == 0:
                print(
                    f"Rendered {frame_number + 1}/{frame_indices.size} frames",
                    flush=True,
                )
    finally:
        writer.close()
        window.close()

    print(f"Saved 3D spectrometer animation: {args.out}")
    print(f"frames={frame_indices.size}, fps={args.fps}")
    print(f"physical_flight_time={time_s[-1] * 1e9:.6f} ns")
    print(f"visual_slowdown={slowdown:.6e}x")
    print(
        "integrated_cyclotron_turns="
        + ", ".join(
            f"{tracks_metadata[index]['label']}:"
            f"{cumulative_turns[index, last_active_indices[index]]:.9f}"
            for index in range(track_count)
        )
    )
    print(f"lfcs_configuration={configuration}")


if __name__ == "__main__":
    main()
