"""Portable trajectory dataset I/O."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from .dynamics import TrajectorySolution

DATASET_VERSION = 1


def save_trajectory_dataset(
    path: str | Path,
    solution: TrajectorySolution,
    metadata: dict[str, Any],
) -> Path:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    diagnostics = solution.diagnostics
    resolved_metadata = {
        "dataset_version": DATASET_VERSION,
        **metadata,
        "solver": solution.solver_statistics,
        "summary": solution.summary(),
    }
    np.savez_compressed(
        output_path,
        metadata_json=np.array(
            json.dumps(resolved_metadata, ensure_ascii=False),
            dtype="<U65535",
        ),
        time_s=solution.time_s,
        position_m=solution.position_m,
        normalized_momentum=solution.normalized_momentum,
        velocity_m_s=diagnostics.velocity_m_s,
        magnetic_field_t=diagnostics.magnetic_field_t,
        magnetic_field_magnitude_t=diagnostics.magnetic_field_magnitude_t,
        gamma=diagnostics.gamma,
        kinetic_energy_ev=diagnostics.kinetic_energy_ev,
        pitch_angle_deg=diagnostics.pitch_angle_deg,
        magnetic_moment_j_per_t=diagnostics.magnetic_moment_j_per_t,
        local_gyrofrequency_rad_s=diagnostics.local_gyrofrequency_rad_s,
    )
    return output_path


def load_trajectory_dataset(path: str | Path) -> dict[str, Any]:
    with np.load(Path(path), allow_pickle=False) as archive:
        return {
            "metadata": json.loads(str(archive["metadata_json"])),
            **{
                key: archive[key]
                for key in archive.files
                if key != "metadata_json"
            },
        }

