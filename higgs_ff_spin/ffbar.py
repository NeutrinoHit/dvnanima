r"""Physics of the film "Spin and orbital structure of the f fbar pair in Higgs decays" (Borodin and Naumov, 2026).

The decay of a spin-zero boson into a fermion f and an antifermion fbar, written in the coordinate space of the relative
coordinate r = x1 - x2 (units: x = p r, p is the momentum of a fermion in the rest frame of the parent):

    psi_{rho1 rho2}(r; s1, s2) = i m_f p / (16 pi^2 v) * Integral dOmega_n  e^{i p n.r}  [ubar_{rho1}(p n, s1) Gamma v_{rho2}(-p n, s2)]

    Gamma = 1       (scalar, 0+):        ubar v           = -2 p  chi^dag (sigma.n) eta
    Gamma = gamma5  (pseudoscalar, 0-):  ubar gamma5 v    =  2 E  chi^dag eta

With the angular integrals  Int dOmega_n e^{i p n.r} = 4 pi j_0(pr)  and  Int dOmega_n e^{i p n.r} n = 4 pi i j_1(pr) rhat  the two
amplitudes, in the common unit N = m_f p E / (2 pi v), are

    psi_S = beta j_1(pr) chi^dag(s1) (sigma.rhat) eta(s2),              a pure P-wave (L = 1)
    psi_P = phase_P j_0(pr) chi^dag(s1) eta(s2),                        a pure S-wave (L = 0),    beta = p / E.

For the amplitudes iM = i (m_f/v) ubar v and iM = i (m_f/v) ubar gamma5 v, taken literally, phase_P = +i (the tests check it by
integrating the Dirac spinors numerically).  The film uses phase_P = -i: the sign of the pseudoscalar coupling is a convention
(gamma5 -> -gamma5), and this choice gives the sign of the CP-odd term of Eq. (B4) of the paper.  The mixed state is
psi = eps1 psi_S + eps2 psi_P.

Spin vectors: s_i is the axis of the spin projection, rho_i = +-1 the projection on it; xi_i = rho_i s_i is the physical spin direction
(as in the book, Eq. (higgsdecays_5)).  The squared moduli (unit N^2):

    |psi_S|^2 = beta^2 j_1^2 (1/2) [1 + xi1.xi2 - 2 (xi1.rhat)(xi2.rhat)]
    |psi_P|^2 = j_0^2 (1/2) [1 - xi1.xi2]
    2 Re(eps1 eps2^* psi_S psi_P^*) = beta j_0 j_1 [ Re(eps1 eps2^*) xi1.(rhat x xi2) - Im(eps1 eps2^*) (xi1 - xi2).rhat ].

Everything here is free of the film's configuration: the numbers come in as arguments.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

# ---------------------------------------------------------------------------------------------- spinors

SIGMA = np.array([[[0, 1], [1, 0]], [[0, -1j], [1j, 0]], [[1, 0], [0, -1]]], dtype=complex)


def unit(v) -> np.ndarray:
    v = np.asarray(v, dtype=float)
    return v / np.linalg.norm(v)


def sdot(v) -> np.ndarray:
    """sigma . v for a real 3-vector (a 2x2 matrix)."""
    return np.tensordot(np.asarray(v, dtype=float), SIGMA, axes=1)


def _angles(s) -> tuple[float, float]:
    s = unit(s)
    return math.acos(max(-1.0, min(1.0, float(s[2])))), math.atan2(float(s[1]), float(s[0]))


def chi(s, rho: int) -> np.ndarray:
    """Pauli spinor with  (sigma.s) chi = rho chi  (Eq. (A2) of the paper)."""
    th, ph = _angles(s)
    if rho > 0:
        return np.array([math.cos(th / 2), np.exp(1j * ph) * math.sin(th / 2)])
    return np.array([-np.exp(-1j * ph) * math.sin(th / 2), math.cos(th / 2)])


def eta(s, rho: int) -> np.ndarray:
    """Antifermion spinor  eta_rho(s) = i sigma_2 chi_rho^*(s) up to a sign (Eq. (A2)): (sigma.s) eta = -rho eta."""
    th, ph = _angles(s)
    if rho > 0:
        return np.array([np.exp(-1j * ph) * math.sin(th / 2), -math.cos(th / 2)])
    return np.array([-math.cos(th / 2), -np.exp(1j * ph) * math.sin(th / 2)])


# ---------------------------------------------------------------------------------------- spherical Bessel

def j0(x):
    x = np.asarray(x, dtype=float)
    small = np.abs(x) < 1e-4
    xs = np.where(small, 1.0, x)
    return np.where(small, 1.0 - x * x / 6.0, np.sin(xs) / xs)


def j1(x):
    x = np.asarray(x, dtype=float)
    small = np.abs(x) < 1e-3
    xs = np.where(small, 1.0, x)
    return np.where(small, x / 3.0 - x ** 3 / 30.0, np.sin(xs) / xs ** 2 - np.cos(xs) / xs)


def j1_first_maximum() -> float:
    """pr of the first maximum of j_1 (the root of j_1' = 0: tan x = 2 x / (2 - x^2))."""
    lo, hi = 1.5, 2.6
    d = lambda x: float((j1(x + 1e-6) - j1(x - 1e-6)) / 2e-6)
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        if d(lo) * d(mid) <= 0:
            hi = mid
        else:
            lo = mid
    return 0.5 * (lo + hi)


# ------------------------------------------------------------------------------------------- pair states

@dataclass(frozen=True)
class PairState:
    """Spin state of the pair: axes s1, s2 and projections rho1, rho2 = +-1 on them."""
    s1: tuple
    s2: tuple
    rho1: int
    rho2: int

    @property
    def xi1(self) -> np.ndarray:
        return self.rho1 * unit(self.s1)

    @property
    def xi2(self) -> np.ndarray:
        return self.rho2 * unit(self.s2)

    def vector(self) -> np.ndarray:
        """c_a = chi^dag(s1) sigma_a eta(s2): chi^dag (sigma.rhat) eta = rhat . c."""
        ch, et = chi(self.s1, self.rho1), eta(self.s2, self.rho2)
        return np.array([ch.conj() @ SIGMA[a] @ et for a in range(3)])

    def scalar(self) -> complex:
        """d = chi^dag(s1) eta(s2)."""
        return complex(chi(self.s1, self.rho1).conj() @ eta(self.s2, self.rho2))


def common_axis_state(axis, rho1: int, rho2: int) -> PairState:
    a = tuple(float(x) for x in unit(axis))
    return PairState(a, a, int(rho1), int(rho2))


def psi_s(points, state: PairState, beta: float = 1.0) -> np.ndarray:
    """Scalar (P-wave) amplitude psi_S at the points x = p r (shape (..., 3)), in units N: beta j_1(x) chi^dag (sigma.rhat) eta."""
    P = np.asarray(points, dtype=float)
    r = np.linalg.norm(P, axis=-1)
    return beta * j1(r) / np.maximum(r, 1e-12) * (P @ state.vector())


def psi_p(points, state: PairState, phase: complex) -> np.ndarray:
    """Pseudoscalar (S-wave) amplitude psi_P at the points, in units N: phase * j_0(x) chi^dag eta."""
    P = np.asarray(points, dtype=float)
    r = np.linalg.norm(P, axis=-1)
    return phase * j0(r) * state.scalar()


def psi_mixed(points, state: PairState, eps1: complex, eps2: complex, beta: float, phase: complex) -> np.ndarray:
    return eps1 * psi_s(points, state, beta) + eps2 * psi_p(points, state, phase)


# --------------------------------------------------------------------- the closed forms of the squared moduli

def angular_scalar(rhat, xi1, xi2) -> np.ndarray:
    """(1/2) [1 + xi1.xi2 - 2 (xi1.rhat)(xi2.rhat)]   (Eq. (15) with xi_i = rho_i s_i)."""
    rhat = np.asarray(rhat, dtype=float)
    return 0.5 * (1.0 + float(np.dot(xi1, xi2)) - 2.0 * (rhat @ xi1) * (rhat @ xi2))


def angular_pseudoscalar(xi1, xi2) -> float:
    """(1/2) [1 - xi1.xi2]   (Eq. (22))."""
    return 0.5 * (1.0 - float(np.dot(xi1, xi2)))


def triple_product(rhat, xi1, xi2) -> np.ndarray:
    """xi1 . (rhat x xi2) = rhat . (xi2 x xi1)."""
    rhat = np.asarray(rhat, dtype=float)
    return rhat @ np.cross(xi2, xi1)


def density_closed(points, xi1, xi2, eps1: complex, eps2: complex, beta: float) -> np.ndarray:
    """|psi|^2 of the mixed state in units N^2, from the closed forms (the film's phase convention)."""
    P = np.asarray(points, dtype=float)
    r = np.linalg.norm(P, axis=-1)
    rhat = P / np.maximum(r, 1e-12)[..., None]
    a, b = beta * j1(r), j0(r)
    e12 = eps1 * np.conj(eps2)
    xi1, xi2 = np.asarray(xi1, float), np.asarray(xi2, float)
    return (abs(eps1) ** 2 * a * a * angular_scalar(rhat, xi1, xi2)
            + abs(eps2) ** 2 * b * b * angular_pseudoscalar(xi1, xi2)
            + a * b * (e12.real * triple_product(rhat, xi1, xi2) - e12.imag * (rhat @ (xi1 - xi2))))


def density_on_axis(r, eps1: float, eps2: float, beta: float, sign: int) -> np.ndarray:
    """|psi|^2 on the axis rhat = sign * (xi2 x xi1) / |xi2 x xi1| for xi1 perpendicular to xi2 (real eps).

    On this axis (xi1.rhat) = (xi2.rhat) = 0 and xi1.xi2 = 0, so |psi|^2 = (1/2) a^2 + (1/2) b^2 + sign a b = (1/2) (a + sign b)^2
    with a = eps1 beta j_1(pr), b = eps2 j_0(pr): the S and P waves add on one side (sign = +1) and cancel on the other (sign = -1)."""
    a, b = eps1 * beta * j1(r), eps2 * j0(r)
    return 0.5 * (a + sign * b) ** 2


# ------------------------------------------------------------------------------------------- quadratures

def gauss_sphere(n_theta: int, n_phi: int) -> tuple[np.ndarray, np.ndarray]:
    """Points (n, 3) and weights of a product quadrature on the unit sphere (Gauss-Legendre in cos(theta), midpoints in phi)."""
    x, w = np.polynomial.legendre.leggauss(n_theta)
    ph = (np.arange(n_phi) + 0.5) * 2.0 * math.pi / n_phi
    st = np.sqrt(1.0 - x * x)
    pts = np.stack([np.outer(st, np.cos(ph)), np.outer(st, np.sin(ph)), np.outer(x, np.ones(n_phi))], axis=-1).reshape(-1, 3)
    wt = np.outer(w, np.full(n_phi, 2.0 * math.pi / n_phi)).reshape(-1)
    return pts, wt


def radical_inverse(i: int, base: int) -> float:
    f, r = 1.0, 0.0
    while i > 0:
        f /= base
        r += f * (i % base)
        i //= base
    return r


def halton_sphere(n: int) -> np.ndarray:
    """n points on the unit sphere from the Halton sequence (2, 3): every prefix of the sequence is spread over the whole sphere."""
    z = np.array([2.0 * radical_inverse(i + 1, 2) - 1.0 for i in range(n)])
    ph = np.array([2.0 * math.pi * radical_inverse(i + 1, 3) for i in range(n)])
    s = np.sqrt(1.0 - z * z)
    return np.stack([s * np.cos(ph), s * np.sin(ph), z], axis=1)


class PlaneWaveSum:
    """Partial sums  (4 pi / N) sum_{k<=N} w(n_k) exp(i p n_k.r)  on a square grid of the plane z = 0 (x = p r in [-R, R]).

    This is the integral over the directions of the pair with the weights w(n) = the amplitude to emit the pair along n; for the
    weight n_x + i n_y (both spins down on the z axis) it converges to 4 pi i j_1(pr) (rhat_x + i rhat_y)."""

    def __init__(self, directions: np.ndarray, weights: np.ndarray, grid: int, half_width: float) -> None:
        self.n, self.w = directions, weights
        a = np.linspace(-half_width, half_width, grid)
        self.X, self.Y = np.meshgrid(a, a[::-1])
        self.reset()

    def reset(self) -> None:
        self.acc = np.zeros(self.X.shape, dtype=complex)
        self.done = 0

    def get(self, count: int) -> np.ndarray:
        """The partial sum over the first `count` directions (the accumulator is extended when count grows)."""
        if count < self.done:
            self.reset()
        for k in range(self.done, count):
            self.acc += self.w[k] * np.exp(1j * (self.n[k, 0] * self.X + self.n[k, 1] * self.Y))
        self.done = count
        return self.acc * (4.0 * math.pi / count)

    def exact(self) -> np.ndarray:
        """The limit N -> infinity for the weight n_x + i n_y: 4 pi i j_1(pr) exp(i phi_r) (the vector identity (13) of the paper)."""
        rho = np.hypot(self.X, self.Y)
        return 4.0 * math.pi * 1j * j1(rho) * np.exp(1j * np.arctan2(self.Y, self.X))
