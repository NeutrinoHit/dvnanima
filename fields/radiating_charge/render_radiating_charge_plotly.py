from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parents[2]))

from fields.radiating_charge.data_io import display_label, load_dataset, resolve_frame_key


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Render a radiating-charge dataset with Plotly.")
    parser.add_argument("data", type=Path)
    parser.add_argument("--lower-key", type=str, default="bz_rad")
    parser.add_argument("--upper-key", type=str, default="ez_rad")
    parser.add_argument("--show-upper", action="store_true")
    parser.add_argument("--show-center", action="store_true")
    parser.add_argument("--mode", choices=("2d", "3d"), default="2d")
    parser.add_argument("--transform", choices=("linear", "log", "signed_log", "tanh"), default="signed_log")
    parser.add_argument("--z-mode", choices=("dynamic", "fixed"), default="dynamic")
    parser.add_argument("--time-stride", type=int, default=3)
    parser.add_argument("--space-stride", type=int, default=2)
    parser.add_argument("--width", type=int, default=1400)
    parser.add_argument("--height", type=int, default=760)
    parser.add_argument("--out", type=Path, default=Path("radiating_charge_plotly.html"))
    return parser


def _apply_transform(frame: np.ndarray, mode: str) -> np.ndarray:
    if mode == "linear":
        return frame
    if mode == "log":
        return np.log10(1.0 + np.abs(frame))
    if mode == "signed_log":
        return np.sign(frame) * np.log10(1.0 + np.abs(frame))
    peak = max(float(np.max(np.abs(frame))), 1e-9)
    return np.tanh(frame / peak)


def _levels(frames: np.ndarray) -> tuple[float, float]:
    vmax = max(float(np.max(np.abs(frames))), 1e-9)
    if float(np.min(frames)) < 0.0:
        return -vmax, vmax
    return 0.0, vmax


def _surface_scale(frame: np.ndarray, target: float, mode: str, reference_peak: float) -> float:
    peak = reference_peak if mode == "fixed" else max(float(np.max(np.abs(frame))), 1e-9)
    return target / peak


def _build_panels(dataset: dict, args):
    time_stride = max(1, int(args.time_stride))
    space_stride = max(1, int(args.space_stride))

    lower_key = resolve_frame_key(args.lower_key, dataset=dataset)
    upper_key = resolve_frame_key(args.upper_key, dataset=dataset)

    times = dataset["times"][::time_stride]
    x_axis = dataset["x_axis"][::space_stride]
    y_axis = dataset["y_axis"][::space_stride]
    centers = dataset["source_centers"][::time_stride, 0]

    cfg_render = dataset["metadata"].get("config", {}).get("render", {})
    panels = [
        {
            "key": lower_key,
            "frames": dataset[lower_key][::time_stride, ::space_stride, ::space_stride],
            "label": display_label(lower_key, style="unicode"),
            "target": float(cfg_render.get("lower_height_target", 1.0)),
        }
    ]
    if args.show_upper:
        panels.append(
            {
                "key": upper_key,
                "frames": dataset[upper_key][::time_stride, ::space_stride, ::space_stride],
                "label": display_label(upper_key, style="unicode"),
                "target": float(cfg_render.get("upper_height_target", 1.0)),
            }
        )

    for panel in panels:
        panel["frames_t"] = _apply_transform(panel["frames"], args.transform)
        panel["levels"] = _levels(panel["frames_t"])
        panel["peak"] = max(float(np.max(np.abs(panel["frames_t"]))), 1e-9)
        panel["colorscale"] = "RdBu" if panel["levels"][0] < 0.0 else "Viridis"

    return panels, times, x_axis, y_axis, centers


def _build_frame_data(mode: str, panels: list[dict], x_axis: np.ndarray, y_axis: np.ndarray, centers: np.ndarray, frame_idx: int, show_center: bool, z_mode: str):
    traces = []
    z_ranges = []

    for idx, panel in enumerate(panels):
        frame = panel["frames_t"][frame_idx]
        vmin, vmax = panel["levels"]
        if mode == "2d":
            traces.append(
                go.Heatmap(
                    x=x_axis,
                    y=y_axis,
                    z=frame.T,
                    zmin=vmin,
                    zmax=vmax,
                    colorscale=panel["colorscale"],
                    showscale=True,
                    colorbar={"title": panel["label"], "len": 0.8},
                )
            )
            z_ranges.append((0.0, 0.0))
            if show_center:
                cx, cy = centers[frame_idx]
                hist = centers[: frame_idx + 1]
                traces.append(go.Scatter(x=hist[:, 0], y=hist[:, 1], mode="lines", line={"color": "red", "width": 2}, showlegend=False))
                traces.append(go.Scatter(x=[cx], y=[cy], mode="markers", marker={"color": "red", "size": 8}, showlegend=False))
        else:
            scale = _surface_scale(frame, panel["target"], z_mode, panel["peak"])
            z = (scale * frame).T
            z_min, z_max = float(np.min(z)), float(np.max(z))
            z_pad = max(0.08 * max(abs(z_min), abs(z_max), 1e-9), 1e-3)
            traces.append(
                go.Surface(
                    x=x_axis,
                    y=y_axis,
                    z=z,
                    cmin=vmin,
                    cmax=vmax,
                    colorscale=panel["colorscale"],
                    showscale=True,
                    colorbar={"title": panel["label"], "len": 0.8},
                )
            )
            z_ranges.append((z_min - z_pad, z_max + z_pad))
            if show_center:
                cx, cy = centers[frame_idx]
                hist = centers[: frame_idx + 1]
                z_mark = np.full(frame_idx + 1, 0.03 * panel["target"], dtype=float)
                traces.append(go.Scatter3d(x=hist[:, 0], y=hist[:, 1], z=z_mark, mode="lines", line={"color": "red", "width": 5}, showlegend=False))
                traces.append(go.Scatter3d(x=[cx], y=[cy], z=[z_mark[-1]], mode="markers", marker={"color": "red", "size": 4}, showlegend=False))

    return traces, z_ranges


def main() -> None:
    args = build_arg_parser().parse_args()
    dataset = load_dataset(args.data)
    panels, times, x_axis, y_axis, centers = _build_panels(dataset, args)

    cols = len(panels)
    specs = [[{"type": "scene" if args.mode == "3d" else "heatmap"} for _ in range(cols)]]
    fig = make_subplots(rows=1, cols=cols, specs=specs, subplot_titles=[panel["label"] for panel in panels], horizontal_spacing=0.12)

    init_data, init_z_ranges = _build_frame_data(args.mode, panels, x_axis, y_axis, centers, 0, args.show_center, args.z_mode)
    trace_ptr = 0
    traces_per_panel = 3 if args.show_center else 1
    for col in range(1, cols + 1):
        for _ in range(traces_per_panel):
            fig.add_trace(init_data[trace_ptr], row=1, col=col)
            trace_ptr += 1

    frames = []
    for frame_idx, t_val in enumerate(times):
        frame_data, z_ranges = _build_frame_data(args.mode, panels, x_axis, y_axis, centers, frame_idx, args.show_center, args.z_mode)
        layout = {"title": f"t = {float(t_val):.3f}"}
        if args.mode == "3d":
            for idx in range(cols):
                scene_name = "scene" if idx == 0 else f"scene{idx + 1}"
                layout[scene_name] = {
                    "xaxis": {"title": "x", "range": [float(x_axis[0]), float(x_axis[-1])]},
                    "yaxis": {"title": "y", "range": [float(y_axis[0]), float(y_axis[-1])]},
                    "zaxis": {"range": [float(z_ranges[idx][0]), float(z_ranges[idx][1])]},
                    "camera": {"eye": {"x": 1.55, "y": -1.6, "z": 0.95}},
                }
        frames.append(go.Frame(data=frame_data, name=str(frame_idx), layout=layout))

    fig.frames = frames
    fig.update_layout(title=f"t = {float(times[0]):.3f}", width=int(args.width), height=int(args.height))

    if args.mode == "2d":
        for idx in range(cols):
            x_name = "xaxis" if idx == 0 else f"xaxis{idx + 1}"
            y_name = "yaxis" if idx == 0 else f"yaxis{idx + 1}"
            x_ref = "x" if idx == 0 else f"x{idx + 1}"
            fig.layout[x_name].update(title="x", range=[float(x_axis[0]), float(x_axis[-1])], constrain="domain")
            fig.layout[y_name].update(title="y", range=[float(y_axis[0]), float(y_axis[-1])], scaleanchor=x_ref)
    else:
        for idx in range(cols):
            scene_name = "scene" if idx == 0 else f"scene{idx + 1}"
            fig.layout[scene_name].update(
                xaxis={"title": "x", "range": [float(x_axis[0]), float(x_axis[-1])]},
                yaxis={"title": "y", "range": [float(y_axis[0]), float(y_axis[-1])]},
                zaxis={"range": [float(init_z_ranges[idx][0]), float(init_z_ranges[idx][1])]},
                camera={"eye": {"x": 1.55, "y": -1.6, "z": 0.95}},
            )

    fig.update_layout(
        sliders=[
            {
                "active": 0,
                "currentvalue": {"prefix": "frame: "},
                "steps": [
                    {
                        "args": [[frame.name], {"frame": {"duration": 0, "redraw": True}, "mode": "immediate"}],
                        "label": f"{float(times[idx]):.2f}",
                        "method": "animate",
                    }
                    for idx, frame in enumerate(fig.frames)
                ],
            }
        ],
        updatemenus=[
            {
                "type": "buttons",
                "buttons": [
                    {
                        "label": "Play",
                        "method": "animate",
                        "args": [None, {"frame": {"duration": 40, "redraw": True}, "fromcurrent": True}],
                    },
                    {
                        "label": "Pause",
                        "method": "animate",
                        "args": [[None], {"frame": {"duration": 0, "redraw": False}, "mode": "immediate"}],
                    },
                ],
            }
        ],
    )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(args.out, include_plotlyjs="cdn")
    print(f"Saved Plotly render to {args.out}")


if __name__ == "__main__":
    main()
