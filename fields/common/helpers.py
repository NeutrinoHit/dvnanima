from __future__ import annotations

import sys
from pathlib import Path

import numpy as np


ROOT_DIR = Path(__file__).resolve().parents[2]


def bootstrap_project_path(script_file: str) -> Path:
    root = Path(script_file).resolve().parents[2]
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    return root


def unit(vec: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vec))
    if norm <= 1e-10:
        return np.array([1.0, 0.0])
    return vec / norm


def perpendicular(vec: np.ndarray) -> np.ndarray:
    return np.array([-vec[1], vec[0]])


def soft_absorber(axis: np.ndarray, edge_fraction: float = 0.16, strength: float = 0.05) -> np.ndarray:
    x = np.linspace(-1.0, 1.0, len(axis))
    edge = 1.0 - edge_fraction
    mask = np.ones_like(x)
    left = x < -edge
    right = x > edge
    mask[left] = np.exp(-strength * ((x[left] + edge) / edge_fraction) ** 2)
    mask[right] = np.exp(-strength * ((x[right] - edge) / edge_fraction) ** 2)
    return mask


def smooth_window_2d(x_axis: np.ndarray, y_axis: np.ndarray, edge_fraction: float = 0.16, strength: float = 0.05) -> np.ndarray:
    return np.outer(soft_absorber(x_axis, edge_fraction, strength), soft_absorber(y_axis, edge_fraction, strength))
