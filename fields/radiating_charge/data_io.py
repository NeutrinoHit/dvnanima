from __future__ import annotations

from dataclasses import asdict
import json
from pathlib import Path
from typing import TYPE_CHECKING, Callable

import numpy as np

if TYPE_CHECKING:
    from .numerics import RadiatingChargeConfig


DATASET_VERSION = 1
DEFAULT_FRAME_KEYS = (
    "ex",
    "ey",
    "ez",
    "bx",
    "by",
    "bz",
    "sx",
    "sy",
    "sz",
    "e_mag",
    "b_mag",
    "s_mag",
    "ex_rad",
    "ey_rad",
    "ez_rad",
    "bx_rad",
    "by_rad",
    "bz_rad",
    "sx_rad",
    "sy_rad",
    "sz_rad",
    "e_rad_mag",
    "b_rad_mag",
    "s_rad_mag",
)
FRAME_KEY_ALIASES = {
    "e": "e_mag",
    "b": "b_mag",
    "s": "s_mag",
    "eabs": "e_mag",
    "babs": "b_mag",
    "sabs": "s_mag",
    "erad": "e_rad_mag",
    "brad": "b_rad_mag",
    "srad": "s_rad_mag",
}
FRAME_KEY_DISPLAY = {
    "ex": {"math": r"$E_x$", "unicode": "Ex", "html": "E<sub>x</sub>"},
    "ey": {"math": r"$E_y$", "unicode": "Ey", "html": "E<sub>y</sub>"},
    "ez": {"math": r"$E_z$", "unicode": "Ez", "html": "E<sub>z</sub>"},
    "bx": {"math": r"$B_x$", "unicode": "Bx", "html": "B<sub>x</sub>"},
    "by": {"math": r"$B_y$", "unicode": "By", "html": "B<sub>y</sub>"},
    "bz": {"math": r"$B_z$", "unicode": "Bz", "html": "B<sub>z</sub>"},
    "sx": {"math": r"$S_x$", "unicode": "Sx", "html": "S<sub>x</sub>"},
    "sy": {"math": r"$S_y$", "unicode": "Sy", "html": "S<sub>y</sub>"},
    "sz": {"math": r"$S_z$", "unicode": "Sz", "html": "S<sub>z</sub>"},
    "e_mag": {"math": r"$|\mathbf E|$", "unicode": "|E|", "html": "|E|"},
    "b_mag": {"math": r"$|\mathbf B|$", "unicode": "|B|", "html": "|B|"},
    "s_mag": {"math": r"$|\mathbf S|$", "unicode": "|S|", "html": "|S|"},
    "ex_rad": {"math": r"$E_x^{\mathrm{rad}}$", "unicode": "Ex(rad)", "html": "E<sub>x</sub><sup>rad</sup>"},
    "ey_rad": {"math": r"$E_y^{\mathrm{rad}}$", "unicode": "Ey(rad)", "html": "E<sub>y</sub><sup>rad</sup>"},
    "ez_rad": {"math": r"$E_z^{\mathrm{rad}}$", "unicode": "Ez(rad)", "html": "E<sub>z</sub><sup>rad</sup>"},
    "bx_rad": {"math": r"$B_x^{\mathrm{rad}}$", "unicode": "Bx(rad)", "html": "B<sub>x</sub><sup>rad</sup>"},
    "by_rad": {"math": r"$B_y^{\mathrm{rad}}$", "unicode": "By(rad)", "html": "B<sub>y</sub><sup>rad</sup>"},
    "bz_rad": {"math": r"$B_z^{\mathrm{rad}}$", "unicode": "Bz(rad)", "html": "B<sub>z</sub><sup>rad</sup>"},
    "sx_rad": {"math": r"$S_x^{\mathrm{rad}}$", "unicode": "Sx(rad)", "html": "S<sub>x</sub><sup>rad</sup>"},
    "sy_rad": {"math": r"$S_y^{\mathrm{rad}}$", "unicode": "Sy(rad)", "html": "S<sub>y</sub><sup>rad</sup>"},
    "sz_rad": {"math": r"$S_z^{\mathrm{rad}}$", "unicode": "Sz(rad)", "html": "S<sub>z</sub><sup>rad</sup>"},
    "e_rad_mag": {"math": r"$|\mathbf E_{\mathrm{rad}}|$", "unicode": "|E_rad|", "html": "|E<sub>rad</sub>|"},
    "b_rad_mag": {"math": r"$|\mathbf B_{\mathrm{rad}}|$", "unicode": "|B_rad|", "html": "|B<sub>rad</sub>|"},
    "s_rad_mag": {"math": r"$|\mathbf S_{\mathrm{rad}}|$", "unicode": "|S_rad|", "html": "|S<sub>rad</sub>|"},
}


def _float32_copy(array: np.ndarray) -> np.ndarray:
    return np.asarray(array, dtype=np.float32)


def build_dataset(
    cfg: "RadiatingChargeConfig | None" = None,
    progress_callback: Callable[[int, int, float, float], None] | None = None,
) -> dict:
    from .numerics import build_simulation_bundle

    simulation = build_simulation_bundle(cfg, progress_callback=progress_callback)
    resolved_cfg = simulation["cfg"]

    frame_keys = list(resolved_cfg.observables.keys)
    metadata = {
        "dataset_version": DATASET_VERSION,
        "config": asdict(resolved_cfg),
        "lower_surface": resolved_cfg.observables.lower_surface,
        "upper_surface": resolved_cfg.observables.upper_surface,
        "frame_keys": frame_keys,
    }

    dataset = {
        "metadata": metadata,
        "x_axis": _float32_copy(simulation["x_axis"]),
        "y_axis": _float32_copy(simulation["y_axis"]),
        "times": _float32_copy(simulation["times"]),
        "source_centers": _float32_copy(simulation["source_centers"]),
        "trajectory_xyz": _float32_copy(simulation["trajectory_xyz"]),
        "trajectory_v": _float32_copy(simulation["trajectory_v"]),
        "trajectory_a": _float32_copy(simulation["trajectory_a"]),
        "retarded_residual_max": _float32_copy(simulation["retarded_residual_max"]),
    }
    for key in frame_keys:
        dataset[key] = _float32_copy(simulation["frames"][key])
    return dataset


def save_dataset(path: str | Path, dataset: dict, compressed: bool = False) -> Path:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    frame_keys = dataset["metadata"].get("frame_keys", list(DEFAULT_FRAME_KEYS))
    payload = {
        "metadata_json": np.array(json.dumps(dataset["metadata"]), dtype="<U65535"),
        "x_axis": dataset["x_axis"],
        "y_axis": dataset["y_axis"],
        "times": dataset["times"],
        "source_centers": dataset["source_centers"],
        "trajectory_xyz": dataset["trajectory_xyz"],
        "trajectory_v": dataset["trajectory_v"],
        "trajectory_a": dataset["trajectory_a"],
        "retarded_residual_max": dataset["retarded_residual_max"],
    }
    for key in frame_keys:
        payload[key] = dataset[key]

    save_fn = np.savez_compressed if compressed else np.savez
    save_fn(output_path, **payload)
    return output_path


def load_dataset(path: str | Path) -> dict:
    input_path = Path(path)
    with np.load(input_path, allow_pickle=False) as archive:
        metadata = json.loads(str(archive["metadata_json"]))
        frame_keys = list(metadata.get("frame_keys", list(DEFAULT_FRAME_KEYS)))
        dataset = {
            "metadata": metadata,
            "x_axis": archive["x_axis"],
            "y_axis": archive["y_axis"],
            "times": archive["times"],
            "source_centers": archive["source_centers"],
            "trajectory_xyz": archive["trajectory_xyz"],
            "trajectory_v": archive["trajectory_v"],
            "trajectory_a": archive["trajectory_a"],
            "retarded_residual_max": archive["retarded_residual_max"],
        }
        for key in frame_keys:
            dataset[key] = archive[key]
    return dataset


def available_frame_keys(dataset: dict) -> tuple[str, ...]:
    return tuple(dataset["metadata"].get("frame_keys", list(DEFAULT_FRAME_KEYS)))


def resolve_frame_key(key: str, dataset: dict | None = None) -> str:
    normalized = FRAME_KEY_ALIASES.get(key, key)
    valid = set(DEFAULT_FRAME_KEYS)
    if dataset is not None:
        valid = set(available_frame_keys(dataset))
    if normalized not in valid:
        available = tuple(sorted(valid))
        raise KeyError(f"Unknown frame key `{key}`. Available keys: {available}")
    return normalized


def display_label(key: str, style: str = "unicode") -> str:
    normalized = FRAME_KEY_ALIASES.get(key, key)
    style_map = FRAME_KEY_DISPLAY.get(normalized)
    if style_map is None:
        return normalized
    return style_map.get(style, style_map["unicode"])
