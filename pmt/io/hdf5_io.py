from __future__ import annotations

from dataclasses import asdict
import json
from pathlib import Path
from typing import Any

import numpy as np

from pmt.physics.simulate import SceneResult


DATASET_VERSION = 2


def _require_h5py():
    try:
        import h5py
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("h5py is required to save HDF5 output") from exc
    return h5py


def _cast(array: np.ndarray, dtype: np.dtype | type | None) -> np.ndarray:
    if dtype is None:
        return np.asarray(array)
    return np.asarray(array, dtype=dtype)


def build_hdf5_payload(
    result: SceneResult,
    storage_dtype: np.dtype | type | None = np.float32,
) -> dict[str, Any]:
    metadata = {
        "dataset_version": DATASET_VERSION,
        "scene_name": result.config.scene.name,
        "scene_title": result.config.scene.title,
        "config": asdict(result.config),
        "stats": asdict(result.stats),
        "status_encoding": {
            "1": "collected",
            "2": "outside",
            "3": "hit_cathode",
            "4": "hit_focus",
            "5": "timeout",
        },
        "electrode_encoding": {
            "0": "free",
            "1": "cathode",
            "2": "receiver",
            "3": "focus",
            "4": "boundary",
        },
    }

    return {
        "metadata": metadata,
        "x_axis_m": _cast(result.x_axis_m, storage_dtype),
        "y_axis_m": _cast(result.y_axis_m, storage_dtype),
        "potential": _cast(result.field.potential, storage_dtype),
        "ex": _cast(result.field.ex, storage_dtype),
        "ey": _cast(result.field.ey, storage_dtype),
        "electrode_id": np.asarray(result.geometry.electrode_id, dtype=np.uint8),
        "fixed_mask": np.asarray(result.geometry.fixed_mask, dtype=np.bool_),
        "time_axis_s": _cast(result.tracks.time_axis_s, storage_dtype),
        "launch_positions_m": _cast(result.launch_positions_m, storage_dtype),
        "launch_velocities_m_s": _cast(result.launch_velocities_m_s, storage_dtype),
        "positions_m": _cast(result.tracks.positions_m, storage_dtype),
        "velocities_m_s": _cast(result.tracks.velocities_m_s, storage_dtype),
        "alive": np.asarray(result.tracks.alive, dtype=np.bool_),
        "status": np.asarray(result.tracks.status, dtype=np.int8),
        "impact_step": np.asarray(result.tracks.impact_step, dtype=np.int32),
        "impact_position_m": _cast(result.tracks.impact_position_m, storage_dtype),
        "impact_velocity_m_s": _cast(result.tracks.impact_velocity_m_s, storage_dtype),
        "solver_iterations": np.asarray(result.stats.solver_iterations, dtype=np.int32),
        "solver_residual": np.asarray(result.stats.solver_residual, dtype=np.float64),
        "collection_efficiency": np.asarray(result.stats.collection_efficiency, dtype=np.float64),
        "compute_seconds": np.asarray(result.stats.compute_seconds, dtype=np.float64),
    }


def save_scene_hdf5(
    output_path: str | Path,
    result: SceneResult,
    storage_dtype: np.dtype | type | None = np.float32,
    compression: str | None = "gzip",
    compression_level: int = 4,
) -> Path:
    h5py = _require_h5py()
    payload = build_hdf5_payload(result, storage_dtype=storage_dtype)

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    with h5py.File(out, "w") as h5:
        metadata_json = json.dumps(payload["metadata"], ensure_ascii=False, separators=(",", ":"))
        h5.attrs["dataset_version"] = int(payload["metadata"].get("dataset_version", DATASET_VERSION))
        h5.attrs["metadata_json"] = metadata_json

        for key, value in payload.items():
            if key == "metadata":
                continue
            array = np.asarray(value)
            kwargs: dict[str, Any] = {}
            if compression is not None and array.ndim > 0:
                kwargs = {
                    "compression": compression,
                    "compression_opts": compression_level,
                    "shuffle": True,
                }
            h5.create_dataset(key, data=array, **kwargs)

    return out


def load_scene_hdf5(path: str | Path) -> dict[str, Any]:
    h5py = _require_h5py()
    source = Path(path)

    with h5py.File(source, "r") as h5:
        metadata_json = str(h5.attrs.get("metadata_json", "{}"))
        metadata = json.loads(metadata_json)
        if "dataset_version" not in metadata:
            metadata["dataset_version"] = int(h5.attrs.get("dataset_version", DATASET_VERSION))

        dataset: dict[str, Any] = {"metadata": metadata}
        for key in h5.keys():
            dataset[key] = h5[key][()]
        return dataset
