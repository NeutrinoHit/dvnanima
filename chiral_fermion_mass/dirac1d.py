r"""The free Dirac field in 1+1 dimensions in the chiral basis (psi_L, psi_R), solved exactly in Fourier space.

The book (the chapter "Where does a particle get its mass?", the section "Coupled chiral waves give birth to mass"):

    i (d_t + d_x) psi_R = m psi_L,        i (d_t - d_x) psi_L = m psi_R,
    i d_t (psi_L, psi_R)^T = H (psi_L, psi_R)^T,    H(p) = [[-p, m], [m, p]] = p gamma_5 + m sigma_x,   gamma_5 = diag(-1, +1) in this basis,

so that for ``psi ~ exp(ipx)`` the eigenvalues of H are E = +-sqrt(p^2 + m^2).  In 1+1 dimensions the velocity operator is
``dH/dp = gamma_5``: the chirality IS the velocity (psi_R moves to the right with c = 1, psi_L to the left).

Every function takes its numbers as arguments (the film passes them from config.toml).  The evolution over a step ``h`` with a
mass that is constant during the step is the exact unitary matrix ``cos(E h) - i H sin(E h) / E``; a time dependent mass
``m(t)`` is integrated with the midpoint rule over small steps (a product of exact unitaries, so the norm is conserved
to rounding error for any step).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable

import numpy as np


@dataclass(frozen=True)
class Grid:
    """A periodic grid of n points on [-length/2, length/2) and its momenta p = 2 pi k / length (FFT order)."""
    length: float
    n: int

    @property
    def dx(self) -> float:
        return self.length / self.n

    @property
    def x(self) -> np.ndarray:
        return (np.arange(self.n) - self.n // 2) * self.dx

    @property
    def p(self) -> np.ndarray:
        return 2.0 * math.pi * np.fft.fftfreq(self.n, d=self.dx)


def gaussian(grid: Grid, x0: float, sigma: float, p0: float, phase: float = 0.0) -> np.ndarray:
    """exp(-(x-x0)^2 / 4 sigma^2) exp(i p0 x + i phase), normalised to the unit norm; |psi|^2 has the standard deviation sigma."""
    x = grid.x
    f = np.exp(-((x - x0) ** 2) / (4.0 * sigma ** 2) + 1j * (p0 * x + phase))
    return f / math.sqrt(float(np.sum(np.abs(f) ** 2)) * grid.dx)


def energy(p: np.ndarray, m: float) -> np.ndarray:
    return np.sqrt(p ** 2 + m ** 2)


def hamiltonian(p: np.ndarray, m: float) -> np.ndarray:
    """H(p) for every momentum, shape (n, 2, 2) in the basis (L, R)."""
    h = np.zeros((len(p), 2, 2))
    h[:, 0, 0] = -p
    h[:, 1, 1] = p
    h[:, 0, 1] = h[:, 1, 0] = m
    return h


def eigenvector(p: np.ndarray, m: float, sign: int) -> np.ndarray:
    """The normalised eigenvector (u_L, u_R) of H(p) with the eigenvalue sign * sqrt(p^2 + m^2), shape (2, n).

    With cos(phi) = -p / E and sin(phi) = m / E (phi in [0, pi]) it is (cos phi/2, sin phi/2) for E > 0 and
    (-sin phi/2, cos phi/2) for E < 0: smooth in p, so that a packet built from it has no stray phases.
    """
    phi = np.arctan2(m, -p)
    c, s = np.cos(0.5 * phi), np.sin(0.5 * phi)
    return np.array([c, s]) if sign > 0 else np.array([-s, c])


def step(psi: np.ndarray, p: np.ndarray, m: float, h: float) -> np.ndarray:
    """Exact evolution of psi~(p) = (L, R) over the time h with the constant mass m."""
    e = np.sqrt(p ** 2 + m ** 2)
    c = np.cos(e * h)
    sinc = np.where(e > 1e-12, np.sin(e * h) / np.where(e > 1e-12, e, 1.0), h)        # sin(E h) / E
    L, R = psi
    return np.array([c * L - 1j * sinc * (-p * L + m * R), c * R - 1j * sinc * (m * L + p * R)])


def evolve(psi_k: np.ndarray, p: np.ndarray, mass: Callable[[float], float], times: np.ndarray, dt: float, t0: float = 0.0) -> np.ndarray:
    """psi~ (2, n) at the increasing physical ``times`` for the mass m(t) (midpoint rule, steps of at most ``dt``)."""
    out = np.empty((len(times), 2, psi_k.shape[1]), complex)
    psi, t = psi_k.copy(), t0
    for i, tt in enumerate(times):
        span = tt - t
        if span > 1e-14:
            n = max(1, int(math.ceil(span / dt - 1e-9)))
            h = span / n
            for j in range(n):
                psi = step(psi, p, mass(t + (j + 0.5) * h), h)
            t = tt
        out[i] = psi
    return out


# ------------------------------------------------------------------------------------------------ observables

def to_real(psi_k: np.ndarray) -> np.ndarray:
    return np.fft.ifft(psi_k, axis=-1)


def to_fourier(psi_x: np.ndarray) -> np.ndarray:
    return np.fft.fft(psi_x, axis=-1)


def norm(psi_x: np.ndarray, dx: float) -> float:
    return float(np.sum(np.abs(psi_x) ** 2) * dx)


def chirality_weights(psi_x: np.ndarray, dx: float) -> tuple[float, float]:
    """(P_R, P_L): the probabilities of the right and of the left chirality; <gamma_5> = P_R - P_L."""
    return float(np.sum(np.abs(psi_x[1]) ** 2) * dx), float(np.sum(np.abs(psi_x[0]) ** 2) * dx)


def centroid(psi_x: np.ndarray, x: np.ndarray, dx: float) -> float:
    """<x> of the density |psi_L|^2 + |psi_R|^2 (the packet must stay away from the edge of the periodic grid)."""
    rho = np.abs(psi_x[0]) ** 2 + np.abs(psi_x[1]) ** 2
    return float(np.sum(x * rho) * dx / (np.sum(rho) * dx))


def branch_weights(psi_k: np.ndarray, p: np.ndarray, m: float, dx: float) -> tuple[np.ndarray, np.ndarray]:
    """The probability (per momentum cell) on the upper (E > 0) and on the lower (E < 0) branch of H(p) with the mass m;
    their sum over the grid is the norm."""
    n = psi_k.shape[1]
    out = []
    for sign in (+1, -1):
        u = eigenvector(p, m, sign)
        amp = u[0] * psi_k[0] + u[1] * psi_k[1]
        out.append(np.abs(amp) ** 2 * dx / n)
    return out[0], out[1]


def branch_centroid(psi_k: np.ndarray, p: np.ndarray, m: float, x: np.ndarray, dx: float, sign: int) -> tuple[float, float]:
    """(<x>, probability) of the part of the field that lies on the branch E = sign * sqrt(p^2 + m^2) of H(p)."""
    u = eigenvector(p, m, sign)
    amp = u[0] * psi_k[0] + u[1] * psi_k[1]
    comp = np.fft.ifft(np.array([u[0] * amp, u[1] * amp]), axis=-1)
    rho = np.abs(comp[0]) ** 2 + np.abs(comp[1]) ** 2
    w = float(np.sum(rho) * dx)
    return (float(np.sum(x * rho) * dx / w) if w > 1e-12 else 0.0), w


def eigen_packet(grid: Grid, x0: float, sigma: float, p0: float, m: float, sign: int = +1) -> np.ndarray:
    """A wave packet built from the eigenvectors of H(p): (psi_L(x), psi_R(x)) with the sign * E energy and the mean momentum p0."""
    g = gaussian(grid, x0, sigma, p0)
    gk = np.fft.fft(g)
    u = eigenvector(grid.p, m, sign)
    return np.fft.ifft(np.array([u[0] * gk, u[1] * gk]), axis=-1)


def chiral_packet(grid: Grid, x0: float, sigma: float, p0: float, chirality: str, phase: float = 0.0) -> np.ndarray:
    """A pure psi_R ('R') or pure psi_L ('L') packet: (psi_L(x), psi_R(x))."""
    g = gaussian(grid, x0, sigma, p0, phase)
    z = np.zeros_like(g)
    return np.array([z, g]) if chirality == "R" else np.array([g, z])


# ------------------------------------------------------------------------------------------------ analytic results

def group_velocity(p: float, m: float) -> float:
    """v = dE/dp = p / E."""
    return p / math.sqrt(p * p + m * m)


def velocity_ratio(p: np.ndarray, m: float) -> np.ndarray:
    """p / E for an array of momenta (0 where E = 0): the chirality <gamma_5> of the upper-branch eigenvector."""
    e = np.sqrt(p ** 2 + m ** 2)
    return np.where(e > 1e-12, p / np.where(e > 1e-12, e, 1.0), 0.0)


def mean_chirality_plane_wave(p: float, m: float, t: np.ndarray | float) -> np.ndarray | float:
    """<gamma_5>(t) of a plane wave that is pure psi_R at t = 0 and has the mass m for t > 0: p^2/E^2 + (m^2/E^2) cos(2 E t)."""
    e2 = p * p + m * m
    return (p * p + m * m * np.cos(2.0 * math.sqrt(e2) * np.asarray(t))) / e2


def smoothstep(x: float, a: float, b: float) -> float:
    u = min(max((x - a) / (b - a), 0.0), 1.0)
    return u * u * (3 - 2 * u)
