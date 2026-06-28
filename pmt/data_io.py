from __future__ import annotations

from dataclasses import asdict
import json
from pathlib import Path
from typing import Any

import numpy as np

from .numerics import SimulationConfig, build_simulation_bundle


DATASET_VERSION = 1


def _require_h5py():
    try:
        import h5py
    except ImportError as exc:
        raise RuntimeError("h5py is required for PMT HDF5 export/import") from exc
    return h5py


def _cast_array(array: np.ndarray, dtype: np.dtype | type | None) -> np.ndarray:
    if dtype is None:
        return np.asarray(array)
    return np.asarray(array, dtype=dtype)


def build_dataset(
    cfg: SimulationConfig | None = None,
    storage_dtype: np.dtype | type | None = np.float32,
) -> dict[str, Any]:
    simulation = build_simulation_bundle(cfg)
    resolved_cfg: SimulationConfig = simulation["cfg"]

    metadata = {
        "dataset_version": DATASET_VERSION,
        "config": asdict(resolved_cfg),
        "electrode_encoding": {
            "0": "free",
            "1": "cathode",
            "2": "dynode",
            "3": "boundary",
        },
    }

    dataset: dict[str, Any] = {
        "metadata": metadata,
        "x_axis": _cast_array(simulation["x_axis"], storage_dtype),
        "y_axis": _cast_array(simulation["y_axis"], storage_dtype),
        "potential": _cast_array(simulation["potential"], storage_dtype),
        "ex": _cast_array(simulation["ex"], storage_dtype),
        "ey": _cast_array(simulation["ey"], storage_dtype),
        "fixed_mask": np.asarray(simulation["fixed_mask"], dtype=np.bool_),
        "electrode_id": np.asarray(simulation["electrode_id"], dtype=np.uint8),
        "time_axis": _cast_array(simulation["time_axis"], storage_dtype),
        "launch_positions": _cast_array(simulation["launch_positions"], storage_dtype),
        "launch_velocities": _cast_array(simulation["launch_velocities"], storage_dtype),
        "particle_positions": _cast_array(simulation["particle_positions"], storage_dtype),
        "particle_velocities": _cast_array(simulation["particle_velocities"], storage_dtype),
        "particle_alive": np.asarray(simulation["particle_alive"], dtype=np.bool_),
        "particle_impact_mask": np.asarray(simulation["particle_impact_mask"], dtype=np.bool_),
        "particle_impact_step": np.asarray(simulation["particle_impact_step"], dtype=np.int32),
        "particle_impact_position": _cast_array(simulation["particle_impact_position"], storage_dtype),
        "particle_impact_velocity": _cast_array(simulation["particle_impact_velocity"], storage_dtype),
        "solver_iterations": np.asarray(simulation["solver_iterations"], dtype=np.int32),
        "solver_residual": np.asarray(simulation["solver_residual"], dtype=np.float64),
        "collection_efficiency": np.asarray(simulation["collection_efficiency"], dtype=np.float64),
    }
    return dataset


def save_dataset(
    path: str | Path,
    dataset: dict[str, Any],
    compression: str | None = "gzip",
    compression_level: int = 4,
) -> Path:
    h5py = _require_h5py()

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with h5py.File(output_path, "w") as h5:
        metadata_json = json.dumps(dataset["metadata"], separators=(",", ":"))
        h5.attrs["dataset_version"] = int(dataset["metadata"].get("dataset_version", DATASET_VERSION))
        h5.attrs["metadata_json"] = metadata_json

        for key, value in dataset.items():
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

    return output_path


def load_dataset(path: str | Path) -> dict[str, Any]:
    h5py = _require_h5py()

    input_path = Path(path)
    with h5py.File(input_path, "r") as h5:
        metadata_json = str(h5.attrs.get("metadata_json", "{}"))
        metadata = json.loads(metadata_json)
        if "dataset_version" not in metadata:
            metadata["dataset_version"] = int(h5.attrs.get("dataset_version", DATASET_VERSION))

        dataset: dict[str, Any] = {"metadata": metadata}
        for key in h5.keys():
            dataset[key] = h5[key][()]
    return dataset


__all__ = [
    "DATASET_VERSION",
    "build_dataset",
    "save_dataset",
    "load_dataset",
]
