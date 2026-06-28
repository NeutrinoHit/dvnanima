from __future__ import annotations

from dataclasses import asdict
import json
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from .numerics import ScalarQEDConfig


DATASET_VERSION = 1
FRAME_KEYS = (
    "lower_frames",
    "upper_frames",
    "phi_real_frames",
    "phi_abs2_frames",
    "charge_frames",
    "a0_frames",
)
FRAME_KEY_ALIASES = {
    "lower": "lower_frames",
    "upper": "upper_frames",
    "phi_real": "phi_real_frames",
    "phi_abs2": "phi_abs2_frames",
    "rho": "charge_frames",
    "charge": "charge_frames",
    "a0": "a0_frames",
}
FRAME_KEY_DISPLAY = {
    "lower_frames": {
        "math": "Lower field",
        "unicode": "Lower field",
        "html": "Lower field",
    },
    "upper_frames": {
        "math": "Upper field",
        "unicode": "Upper field",
        "html": "Upper field",
    },
    "phi_real_frames": {
        "math": r"$\mathrm{Re}\,\varphi$",
        "unicode": "Re φ",
        "html": "Re &phi;",
    },
    "phi_abs2_frames": {
        "math": r"$|\varphi|^2$",
        "unicode": "|φ|²",
        "html": "|&phi;|<sup>2</sup>",
    },
    "charge_frames": {
        "math": r"$\rho$",
        "unicode": "ρ",
        "html": "&rho;",
    },
    "a0_frames": {
        "math": r"$A_0$",
        "unicode": "A₀",
        "html": "A<sub>0</sub>",
    },
}


def _float32_copy(array: np.ndarray) -> np.ndarray:
    return np.asarray(array, dtype=np.float32)


def build_dataset(cfg: "ScalarQEDConfig | None" = None) -> dict:
    from .numerics import build_animation_bundle_from_simulation, build_simulation_bundle

    simulation = build_simulation_bundle(cfg)
    animation = build_animation_bundle_from_simulation(simulation)
    resolved_cfg = animation["cfg"]

    metadata = {
        "dataset_version": DATASET_VERSION,
        "config": asdict(resolved_cfg),
        "lower_surface": resolved_cfg.observables.lower_surface,
        "upper_surface": resolved_cfg.observables.upper_surface,
        "frame_keys": list(FRAME_KEYS),
    }

    dataset = {
        "metadata": metadata,
        "x_axis": _float32_copy(animation["x_axis"]),
        "y_axis": _float32_copy(animation["y_axis"]),
        "times": _float32_copy(animation["times"]),
        "simulation_times": _float32_copy(animation["simulation_times"]),
        "packet_centers": _float32_copy(animation["packet_centers"]),
    }
    for key in FRAME_KEYS:
        dataset[key] = _float32_copy(animation[key])
    return dataset


def save_dataset(path: str | Path, dataset: dict, compressed: bool = False) -> Path:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "metadata_json": np.array(json.dumps(dataset["metadata"]), dtype="<U65535"),
        "x_axis": dataset["x_axis"],
        "y_axis": dataset["y_axis"],
        "times": dataset["times"],
        "simulation_times": dataset["simulation_times"],
        "packet_centers": dataset["packet_centers"],
    }
    for key in FRAME_KEYS:
        payload[key] = dataset[key]
    save_fn = np.savez_compressed if compressed else np.savez
    save_fn(output_path, **payload)
    return output_path


def load_dataset(path: str | Path) -> dict:
    input_path = Path(path)
    with np.load(input_path, allow_pickle=False) as archive:
        metadata = json.loads(str(archive["metadata_json"]))
        dataset = {
            "metadata": metadata,
            "x_axis": archive["x_axis"],
            "y_axis": archive["y_axis"],
            "times": archive["times"],
            "simulation_times": archive["simulation_times"],
            "packet_centers": archive["packet_centers"],
        }
        for key in FRAME_KEYS:
            dataset[key] = archive[key]
    return dataset


def resolve_frame_key(key: str) -> str:
    normalized = FRAME_KEY_ALIASES.get(key, key)
    if normalized not in FRAME_KEYS:
        raise KeyError(f"Unknown frame key `{key}`. Available keys: {FRAME_KEYS}")
    return normalized


def display_label(key: str, style: str = "unicode") -> str:
    normalized = FRAME_KEY_ALIASES.get(key, key)
    style_map = FRAME_KEY_DISPLAY.get(normalized)
    if style_map is None:
        return normalized
    return style_map.get(style, style_map["unicode"])
