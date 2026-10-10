"""Physics of a box lattice of NX x NY x NZ equal masses with a fixed boundary (the 3D mattress of the book).

Masses m = 1, springs k = 1 between nearest neighbours, a scalar displacement q_ijk of every mass.
The normal modes are products of three sines; the linear evolution is exact; with the anharmonic bond potential
V(d) = k d^2/2 + alpha d^3/3 + beta d^4/4 (d = difference of the displacements of two neighbours) the lattice is
integrated with the velocity-Verlet scheme.
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
NX, NY, NZ = _PH.nx, _PH.ny, _PH.nz
KC = _PH.k
ALPHA, BETA = _PH.alpha, _PH.beta
DEFAULT_DT = _PH.default_dt


def _sines(n: int) -> np.ndarray:
    return np.sqrt(2.0 / (n + 1)) * np.sin(np.outer(np.arange(1, n + 1), np.arange(1, n + 1)) * math.pi / (n + 1))


SX, SY, SZ = _sines(NX), _sines(NY), _sines(NZ)
IX, IY, IZ = (np.arange(1, n + 1) for n in (NX, NY, NZ))
_A = np.arange(1, NX + 1)[:, None, None]
_B = np.arange(1, NY + 1)[None, :, None]
_C = np.arange(1, NZ + 1)[None, None, :]
OMEGA = 2.0 * math.sqrt(KC) * np.sqrt(np.sin(_A * math.pi / (2 * (NX + 1))) ** 2 + np.sin(_B * math.pi / (2 * (NY + 1))) ** 2
                                      + np.sin(_C * math.pi / (2 * (NZ + 1))) ** 2)
KMAG = math.pi * np.sqrt((_A / (NX + 1)) ** 2 + (_B / (NY + 1)) ** 2 + (_C / (NZ + 1)) ** 2)
OMEGA_MAX = 2.0 * math.sqrt(3.0 * KC)


def omega(a: int, b: int, c: int) -> float:
    return float(OMEGA[a - 1, b - 1, c - 1])


def mode_shape(a: int, b: int, c: int) -> np.ndarray:
    return SX[a - 1][:, None, None] * SY[b - 1][None, :, None] * SZ[c - 1][None, None, :]


def to_modes(q: np.ndarray) -> np.ndarray:
    return np.einsum("ai,bj,ck,ijk->abc", SX, SY, SZ, q, optimize=True)


def from_modes(m: np.ndarray) -> np.ndarray:
    return np.einsum("ai,bj,ck,abc->ijk", SX, SY, SZ, m, optimize=True)


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


def _bonds(q: np.ndarray):
    qp = np.pad(q, 1)
    return (qp[1:, 1:-1, 1:-1] - qp[:-1, 1:-1, 1:-1], qp[1:-1, 1:, 1:-1] - qp[1:-1, :-1, 1:-1],
            qp[1:-1, 1:-1, 1:] - qp[1:-1, 1:-1, :-1])


def force(q: np.ndarray, alpha: float, beta: float) -> np.ndarray:
    dx, dy, dz = _bonds(q)
    tension = lambda d: KC * d + alpha * d ** 2 + beta * d ** 3
    tx, ty, tz = tension(dx), tension(dy), tension(dz)
    return tx[1:] - tx[:-1] + ty[:, 1:] - ty[:, :-1] + tz[:, :, 1:] - tz[:, :, :-1]


def potential_energy(q: np.ndarray, alpha: float, beta: float) -> float:
    f = lambda d: float(np.sum(0.5 * KC * d ** 2 + alpha * d ** 3 / 3.0 + beta * d ** 4 / 4.0))
    return sum(f(d) for d in _bonds(q))


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


def gauss3(i0: float, j0: float, k0: float, sx: float, sy: float, sz: float, amp: float) -> np.ndarray:
    return amp * (np.exp(-((IX[:, None, None] - i0) ** 2) / (2 * sx ** 2)) * np.exp(-((IY[None, :, None] - j0) ** 2) / (2 * sy ** 2))
                  * np.exp(-((IZ[None, None, :] - k0) ** 2) / (2 * sz ** 2)))


def packet(i0: float, k0x: float, sx: float, sy: float, amp: float, sgn: float) -> tuple[np.ndarray, np.ndarray]:
    """A wave packet along x (carrier k0x), centred in y and z, moving to the right (sgn = +1) or to the left (-1)."""
    env = gauss3(i0, 0.5 * (NY + 1), 0.5 * (NZ + 1), sx, sy, sy, 1.0)
    w = 2.0 * math.sin(k0x / 2.0)
    ph = k0x * (IX[:, None, None] - i0)
    return amp * np.cos(ph) * env, sgn * w * amp * np.sin(ph) * env


PK_A, PK_K0, PK_SX, PK_SY = _PH.packet.amplitude, _PH.packet.carrier_k, _PH.packet.sigma_x, _PH.packet.sigma_yz
PK_LEFT, PK_RIGHT = _PH.packet.left_x, _PH.packet.right_x
BUMP = (*_PH.bump.centre, *_PH.bump.sigma, _PH.bump.amplitude)           # i0, j0, k0, sigmas, amplitude


def collision_initial() -> tuple[np.ndarray, np.ndarray]:
    q1, v1 = packet(PK_LEFT, PK_K0, PK_SX, PK_SY, PK_A, +1.0)
    q2, v2 = packet(PK_RIGHT, PK_K0, PK_SX, PK_SY, PK_A, -1.0)
    return q1 + q2, v1 + v2
