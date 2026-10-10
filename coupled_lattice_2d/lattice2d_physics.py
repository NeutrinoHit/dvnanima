"""Physics of a rectangular lattice of NX x NY equal masses with fixed boundary (the 2D mattress of the book).

Masses m = 1, springs k = 1 between nearest neighbours, displacements q_ij perpendicular to the plane.
The normal modes are products of sines; the linear evolution is exact; with the anharmonic bond potential
V(d) = k d^2/2 + alpha d^3/3 + beta d^4/4 (d = difference of the displacements of two neighbours) the chain of
modes is integrated with the velocity-Verlet scheme.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dvconfig import load_config  # noqa: E402

CFG = load_config(Path(__file__).resolve().parent)
_PH = CFG.physics
NX, NY = _PH.nx, _PH.ny
KC = _PH.k
ALPHA, BETA = _PH.alpha, _PH.beta
DEFAULT_DT = _PH.default_dt

IX = np.arange(1, NX + 1)
IY = np.arange(1, NY + 1)
SX = np.sqrt(2.0 / (NX + 1)) * np.sin(np.outer(np.arange(1, NX + 1), IX) * math.pi / (NX + 1))
SY = np.sqrt(2.0 / (NY + 1)) * np.sin(np.outer(np.arange(1, NY + 1), IY) * math.pi / (NY + 1))
A_ = np.arange(1, NX + 1)[:, None]
B_ = np.arange(1, NY + 1)[None, :]
OMEGA = 2.0 * math.sqrt(KC) * np.sqrt(np.sin(A_ * math.pi / (2 * (NX + 1))) ** 2 + np.sin(B_ * math.pi / (2 * (NY + 1))) ** 2)
KMAG = math.pi * np.sqrt((A_ / (NX + 1)) ** 2 + (B_ / (NY + 1)) ** 2)        # |k| of the mode (site units)


def omega(a: int, b: int) -> float:
    return float(OMEGA[a - 1, b - 1])


def mode_shape(a: int, b: int) -> np.ndarray:
    return np.outer(SX[a - 1], SY[b - 1])


def to_modes(q: np.ndarray) -> np.ndarray:
    return SX @ q @ SY.T


def from_modes(c: np.ndarray) -> np.ndarray:
    return SX.T @ c @ SY


def linear_state(q0: np.ndarray, v0: np.ndarray, t: float) -> tuple[np.ndarray, np.ndarray]:
    a = to_modes(q0)
    b = to_modes(v0) / OMEGA
    c = a * np.cos(OMEGA * t) + b * np.sin(OMEGA * t)
    p = -a * OMEGA * np.sin(OMEGA * t) + b * OMEGA * np.cos(OMEGA * t)
    return from_modes(c), from_modes(p)


def mode_energies(q: np.ndarray, v: np.ndarray) -> np.ndarray:
    return 0.5 * to_modes(v) ** 2 + 0.5 * OMEGA ** 2 * to_modes(q) ** 2


def fractions(e: np.ndarray) -> np.ndarray:
    return e / max(float(e.sum()), 1e-12)


def _bonds(q: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    qp = np.pad(q, 1)
    return qp[1:, 1:-1] - qp[:-1, 1:-1], qp[1:-1, 1:] - qp[1:-1, :-1]       # (NX+1, NY), (NX, NY+1)


def force(q: np.ndarray, alpha: float, beta: float) -> np.ndarray:
    dx, dy = _bonds(q)
    tx = KC * dx + alpha * dx ** 2 + beta * dx ** 3
    ty = KC * dy + alpha * dy ** 2 + beta * dy ** 3
    return tx[1:] - tx[:-1] + ty[:, 1:] - ty[:, :-1]


def potential_energy(q: np.ndarray, alpha: float, beta: float) -> float:
    dx, dy = _bonds(q)
    f = lambda d: float(np.sum(0.5 * KC * d ** 2 + alpha * d ** 3 / 3.0 + beta * d ** 4 / 4.0))
    return f(dx) + f(dy)


def integrate(q0: np.ndarray, v0: np.ndarray, alpha: float, beta: float, times: np.ndarray, dt: float = DEFAULT_DT):
    q, v = q0.copy(), v0.copy()
    a = force(q, alpha, beta)
    t = 0.0
    qs, vs = [], []
    for tt in times:
        while t < tt - 1e-12:
            h = min(dt, tt - t)
            v = v + 0.5 * h * a
            q = q + h * v
            a = force(q, alpha, beta)
            v = v + 0.5 * h * a
            t += h
        qs.append(q.copy())
        vs.append(v.copy())
    return np.array(qs), np.array(vs)


def gauss2(i0: float, j0: float, sx: float, sy: float, amp: float) -> np.ndarray:
    return amp * np.exp(-((IX[:, None] - i0) ** 2) / (2 * sx ** 2) - ((IY[None, :] - j0) ** 2) / (2 * sy ** 2))


def packet(i0: float, j0: float, k0: float, sx: float, sy: float, amp: float, sgn: float) -> tuple[np.ndarray, np.ndarray]:
    """A wave packet along x (carrier k0), moving to the right (sgn = +1) or to the left (-1)."""
    env = gauss2(i0, j0, sx, sy, 1.0)
    w = 2.0 * math.sin(k0 / 2.0)
    ph = k0 * (IX[:, None] - i0)
    return amp * np.cos(ph) * env, sgn * w * amp * np.sin(ph) * env


PK_A, PK_K0, PK_SX, PK_SY = _PH.packet.amplitude, _PH.packet.carrier_k, _PH.packet.sigma_x, _PH.packet.sigma_y
PK_LEFT, PK_RIGHT = _PH.packet.left_x, _PH.packet.right_x
BUMP = (*_PH.bump.centre, *_PH.bump.sigma, _PH.bump.amplitude)             # i0, j0, sigma_x, sigma_y, amplitude


def collision_initial() -> tuple[np.ndarray, np.ndarray]:
    jc = 0.5 * (NY + 1)
    q1, v1 = packet(PK_LEFT, jc, PK_K0, PK_SX, PK_SY, PK_A, +1.0)
    q2, v2 = packet(PK_RIGHT, jc, PK_K0, PK_SX, PK_SY, PK_A, -1.0)
    return q1 + q2, v1 + v2
