r"""Physics of the film "The gauge principle: two friends and a field" (units hbar = 1; the conventions of the book, chapter "Gauge Invariance").

    psi(x)  -> psi'(x) = e^{i alpha(x)} psi(x)
    A_mu(x) -> A'_mu(x) = A_mu(x) - (1/q) d_mu alpha(x)
    D_mu = d_mu + i q A_mu            D_mu psi -> e^{i alpha} D_mu psi
    U(y, x) = exp(-i q int_x^y A_mu dz^mu)  ->  e^{i alpha(y)} U(y, x) e^{-i alpha(x)}      (the Wilson line, the parallel transporter)
    F_mu nu = d_mu A_nu - d_nu A_mu         invariant;      U_loop = exp(-i q oint A_mu dz^mu) = exp(-i q int F_12 dx dy)  (Stokes)

All the components of A are the covariant (lower-index) ones, as in the book; for a spatial direction A_x = -A^x, so that
-(1/2m) D_x^2 = (p - q A^x)^2 / 2m is the usual minimal coupling.  Three groups of exact computations are used by the film.

1. Two paths (the interference of the book's "two experimenters" with a screen).  Each path is a paraxial Gaussian beam leaving a slit,
   an exact solution of 2 i k d_x psi + d_y^2 psi = 0; the screen shows |e^{i th_1} psi_1 + e^{i th_2} psi_2|^2, which depends on th_1 - th_2 only.
2. A quantum particle on a line (a lattice with spacing a, mass m).  The covariant lattice derivative uses the Wilson-line link
   U(x_j, x_{j+1}) = exp(+i q theta_j), theta_j = int_{x_j}^{x_{j+1}} A_x dx, exactly like the continuum definition of D in the book; the
   Hamiltonian -(1/2m) D^2 is evolved EXACTLY in time (diagonalisation, no time step).  A gauge transformation acts on the links by
   theta_j -> theta_j - (alpha_{j+1} - alpha_j) / q, and everything is covariant to rounding errors.
3. A plane with a thin magnetic flux tube.  A = (-g(r) y, g(r) x), g = Phi (1 - exp(-r^2 / 2 s^2)) / (2 pi r^2), F_12 = Phi exp(-r^2 / 2 s^2) / (2 pi s^2);
   gauge transformations A -> A - grad(alpha) / q with a smooth alpha(x, y), line integrals along polygons and the flux integrals (Stokes).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from matplotlib.colors import hsv_to_rgb
from scipy import linalg

TWO_PI = 2.0 * math.pi
TINY = 1e-12


def smooth(x: float, a: float, b: float) -> float:
    u = min(max((x - a) / (b - a), 0.0), 1.0)
    return u * u * (3 - 2 * u)


def phase_rgb(phase, value=1.0, saturation: float = 1.0) -> np.ndarray:
    """The colour of a complex phase: hue = phase / 2 pi (red at phase 0), brightness = value."""
    h = np.asarray(phase, dtype=float) / TWO_PI % 1.0
    v = np.broadcast_to(np.asarray(value, dtype=float), h.shape)
    return hsv_to_rgb(np.stack([h, np.full(h.shape, saturation), v], axis=-1))


# =========================================================================================== 1. two paths (paraxial beams)

def beam(dx, y, y0: float, k: float, sigma: float):
    """Paraxial Gaussian beam of the carrier exp(i k dx) that leaves a slit at y0 (dx = the distance from the barrier, amplitude exp(-(y-y0)^2/4 sigma^2) there)."""
    s = 1.0 + 1j * np.asarray(dx, dtype=float) / (2.0 * k * sigma ** 2)
    return np.exp(1j * k * np.asarray(dx, dtype=float)) * np.exp(-((y - y0) ** 2) / (4.0 * sigma ** 2 * s)) / np.sqrt(s)


def two_path_field(dx, y, slit_y, k: float, sigma: float, phases):
    """psi = sum_j e^{i phases[j]} beam_j: the wave behind a barrier with slits at slit_y."""
    out = 0.0
    for y0, th in zip(slit_y, phases):
        out = out + np.exp(1j * th) * beam(dx, y, y0, k, sigma)
    return out


def local_maxima(y: np.ndarray, intensity: np.ndarray) -> np.ndarray:
    """Positions of the local maxima of a sampled curve (parabolic interpolation)."""
    idx = np.where((intensity[1:-1] >= intensity[:-2]) & (intensity[1:-1] > intensity[2:]))[0] + 1
    out = []
    for i in idx:
        a, b, c = intensity[i - 1], intensity[i], intensity[i + 1]
        den = a - 2 * b + c
        off = 0.5 * (a - c) / den if abs(den) > 0 else 0.0
        out.append(y[i] + off * (y[1] - y[0]))
    return np.array(out)


def peak_near(y: np.ndarray, intensity: np.ndarray, near: float = 0.0) -> float:
    pk = local_maxima(y, intensity)
    return float(pk[np.argmin(np.abs(pk - near))])


def fringe_period(y: np.ndarray, intensity: np.ndarray, near: float = 0.0) -> float:
    """The distance between the two maxima that enclose `near` (measured on the pattern)."""
    pk = local_maxima(y, intensity)
    i = int(np.argmin(np.abs(pk - near)))
    j = i + 1 if i + 1 < len(pk) else i - 1
    return float(abs(pk[j] - pk[i]))


def fringe_period_farfield(wavelength: float, distance: float, separation: float) -> float:
    return wavelength * distance / separation


# =========================================================================================== 2. a particle on a line

@dataclass
class Chain:
    """A lattice x_j = -W + j a, j = 0 .. n-1 (hard walls), mass m, charge q."""

    n: int
    half_width: float
    mass: float = 1.0
    charge: float = 1.0

    def __post_init__(self) -> None:
        self.x = np.linspace(-self.half_width, self.half_width, self.n)
        self.a = float(self.x[1] - self.x[0])

    # ----------------------------------------------------------------- the links (Wilson lines)
    def links(self, a_func, order: int = 6) -> np.ndarray:
        """theta_j = int_{x_j}^{x_{j+1}} A_x dx by Gauss-Legendre quadrature (n-1 values)."""
        xi, w = np.polynomial.legendre.leggauss(order)
        mid = 0.5 * (self.x[1:] + self.x[:-1])
        pts = mid[:, None] + 0.5 * self.a * xi[None, :]
        return 0.5 * self.a * (a_func(pts) * w[None, :]).sum(axis=1)

    def gauge_links(self, alpha_values: np.ndarray) -> np.ndarray:
        """The links of the pure-gauge field A = -(1/q) d alpha: theta_j = -(alpha_{j+1} - alpha_j) / q exactly."""
        return -(alpha_values[1:] - alpha_values[:-1]) / self.charge

    # ----------------------------------------------------------------- derivatives and the Hamiltonian
    def derivative(self, psi: np.ndarray) -> np.ndarray:
        """The naive central difference (psi_{j+1} - psi_{j-1}) / 2a; the two end points are left at zero."""
        out = np.zeros_like(psi)
        out[1:-1] = (psi[2:] - psi[:-2]) / (2.0 * self.a)
        return out

    def covariant_derivative(self, psi: np.ndarray, theta: np.ndarray) -> np.ndarray:
        """(D psi)_j = [U(x_j, x_{j+1}) psi_{j+1} - U(x_j, x_{j-1}) psi_{j-1}] / 2a with U(x_j, x_{j+1}) = exp(+i q theta_j): D = d + i q A."""
        q = self.charge
        out = np.zeros_like(psi)
        out[1:-1] = (np.exp(1j * q * theta[1:]) * psi[2:] - np.exp(-1j * q * theta[:-1]) * psi[:-2]) / (2.0 * self.a)
        return out

    def hamiltonian(self, theta: np.ndarray | None = None) -> np.ndarray:
        """H = -(1/2m) D^2 on the lattice, with the covariant lattice Laplacian [U psi_{j+1} + U^* psi_{j-1} - 2 psi_j] / a^2 (theta = None: A = 0)."""
        c = 1.0 / (2.0 * self.mass * self.a ** 2)
        h = np.zeros((self.n, self.n), dtype=complex if theta is not None else float)
        idx = np.arange(self.n - 1)
        up = -c * (np.exp(1j * self.charge * theta) if theta is not None else np.ones(self.n - 1))
        h[idx, idx + 1] = up
        h[idx + 1, idx] = np.conj(up)
        h[np.arange(self.n), np.arange(self.n)] = 2.0 * c
        return h

    def evolver(self, theta: np.ndarray | None = None) -> "Evolver":
        e, v = linalg.eigh(self.hamiltonian(theta))
        return Evolver(e, v)

    # ----------------------------------------------------------------- states
    def packet(self, x0: float, sigma: float, k0: float) -> np.ndarray:
        psi = np.exp(-((self.x - x0) ** 2) / (4.0 * sigma ** 2) + 1j * k0 * self.x)
        return psi / np.sqrt(np.sum(np.abs(psi) ** 2) * self.a)

    def norm(self, psi: np.ndarray) -> float:
        return float(np.sum(np.abs(psi) ** 2) * self.a)

    def centroid(self, psi: np.ndarray) -> float:
        rho = np.abs(psi) ** 2
        return float(np.sum(self.x * rho) / np.sum(rho))

    def mean_of(self, f: np.ndarray, psi: np.ndarray) -> float:
        rho = np.abs(psi) ** 2
        return float(np.sum(f * rho) / np.sum(rho))


class Evolver:
    """psi(t) = V exp(-i E t) V^dagger psi(0): exact in time."""

    def __init__(self, energies: np.ndarray, vectors: np.ndarray) -> None:
        self.e, self.v = energies, vectors

    def coefficients(self, psi0: np.ndarray) -> np.ndarray:
        return self.v.conj().T @ psi0

    def at(self, coeff: np.ndarray, t: float) -> np.ndarray:
        return self.v @ (np.exp(-1j * self.e * t) * coeff)


# =========================================================================================== 3. a plane with a flux tube

def tube_potential(x, y, flux: float, s: float):
    """(A_x, A_y) of a Gaussian flux tube of total flux `flux` (the covariant components): F_12 = flux exp(-r^2/2s^2) / (2 pi s^2)."""
    r2 = np.asarray(x, dtype=float) ** 2 + np.asarray(y, dtype=float) ** 2
    small = r2 < 1e-12
    r2s = np.where(small, 1.0, r2)
    g = np.where(small, flux / (TWO_PI * 2.0 * s ** 2), flux * (-np.expm1(-r2s / (2.0 * s ** 2))) / (TWO_PI * r2s))
    return -g * y, g * x


def tube_field(x, y, flux: float, s: float):
    r2 = np.asarray(x, dtype=float) ** 2 + np.asarray(y, dtype=float) ** 2
    return flux * np.exp(-r2 / (2.0 * s ** 2)) / (TWO_PI * s ** 2)


def alpha_wave(x, y, terms) -> tuple:
    """alpha = sum_i amp_i sin(kx_i x + ky_i y + ph_i): (alpha, d_x alpha, d_y alpha).  terms = [(amp, kx, ky, ph), ...]."""
    al = np.zeros(np.broadcast(x, y).shape)
    ax = np.zeros_like(al)
    ay = np.zeros_like(al)
    for amp, kx, ky, ph in terms:
        arg = kx * np.asarray(x, dtype=float) + ky * np.asarray(y, dtype=float) + ph
        al = al + amp * np.sin(arg)
        ax = ax + amp * kx * np.cos(arg)
        ay = ay + amp * ky * np.cos(arg)
    return al, ax, ay


class PlaneField:
    """A(x, y) = A_tube(flux) - (g/q) grad alpha, with the strength g of the gauge transformation in [0, 1]."""

    def __init__(self, flux: float, s: float, centre: tuple, terms, charge: float = 1.0) -> None:
        self.flux, self.s, self.centre, self.terms, self.q = flux, s, centre, terms, charge

    def a_field(self, x, y, flux_scale: float = 1.0, gauge_scale: float = 0.0):
        ax_t, ay_t = tube_potential(np.asarray(x, dtype=float) - self.centre[0], np.asarray(y, dtype=float) - self.centre[1], self.flux * flux_scale, self.s)
        _, gx, gy = alpha_wave(x, y, self.terms)
        return ax_t - gauge_scale * gx / self.q, ay_t - gauge_scale * gy / self.q

    def b_field(self, x, y, flux_scale: float = 1.0):
        return tube_field(np.asarray(x, dtype=float) - self.centre[0], np.asarray(y, dtype=float) - self.centre[1], self.flux * flux_scale, self.s)

    def alpha(self, x, y):
        return alpha_wave(x, y, self.terms)[0]


def curl_fd(ax: np.ndarray, ay: np.ndarray, h: float) -> np.ndarray:
    """F_12 = d_x A_y - d_y A_x by central differences on a grid indexed [iy, ix] (the interior points)."""
    dya = (ay[1:-1, 2:] - ay[1:-1, :-2]) / (2 * h)
    dxa = (ax[2:, 1:-1] - ax[:-2, 1:-1]) / (2 * h)
    return dya - dxa


def polygon_line_integral(a_func, vertices: np.ndarray, max_step: float, order: int = 4, closed: bool = True) -> float:
    """int A_x dx + A_y dy along the polygon (straight pieces split into steps <= max_step, Gauss-Legendre of the given order on each)."""
    xi, w = np.polynomial.legendre.leggauss(order)
    pts = np.asarray(vertices, dtype=float)
    segs = list(zip(pts[:-1], pts[1:]))
    if closed:
        segs.append((pts[-1], pts[0]))
    total = 0.0
    for p0, p1 in segs:
        length = float(np.hypot(*(p1 - p0)))
        m = max(1, int(math.ceil(length / max_step)))
        t0 = np.arange(m) / m
        t = (t0[:, None] + (0.5 + 0.5 * xi[None, :]) / m).ravel()
        ww = np.tile(w * 0.5 / m, m)
        px = p0[0] + t * (p1[0] - p0[0])
        py = p0[1] + t * (p1[1] - p0[1])
        ax, ay = a_func(px, py)
        total += float(np.sum(ww * (ax * (p1[0] - p0[0]) + ay * (p1[1] - p0[1]))))
    return total


def polygon_area(vertices: np.ndarray) -> float:
    """The signed area of a polygon (positive for the counter-clockwise order)."""
    v = np.asarray(vertices, dtype=float)
    x, y = v[:, 0], v[:, 1]
    return 0.5 * float(np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y))


def triangle_integral(f, v0, v1, v2, n: int = 300) -> float:
    """int f dA over the triangle v0 v1 v2 (tensor Gauss-Legendre in the collapsed coordinates)."""
    xi, w = np.polynomial.legendre.leggauss(n)
    s = 0.5 * (xi + 1.0)
    ws = 0.5 * w
    v0, v1, v2 = (np.asarray(v, dtype=float) for v in (v0, v1, v2))
    two_area = abs((v1[0] - v0[0]) * (v2[1] - v0[1]) - (v1[1] - v0[1]) * (v2[0] - v0[0]))
    ss, tt = np.meshgrid(s, s, indexing="ij")
    wgt = ws[:, None] * ws[None, :] * (1.0 - ss) * two_area
    u, v = ss, (1.0 - ss) * tt
    px = v0[0] + u * (v1[0] - v0[0]) + v * (v2[0] - v0[0])
    py = v0[1] + u * (v1[1] - v0[1]) + v * (v2[1] - v0[1])
    return float(np.sum(wgt * f(px, py)))


def rectangle_integral(f, x0: float, x1: float, y0: float, y1: float, n: int = 200) -> float:
    xi, w = np.polynomial.legendre.leggauss(n)
    xs = 0.5 * (x1 - x0) * xi + 0.5 * (x1 + x0)
    ys = 0.5 * (y1 - y0) * xi + 0.5 * (y1 + y0)
    xx, yy = np.meshgrid(xs, ys, indexing="ij")
    return float(np.sum(0.25 * (x1 - x0) * (y1 - y0) * w[:, None] * w[None, :] * f(xx, yy)))


def wilson_phase_to_screen(a_func, start: tuple, screen_x: float, screen_y: np.ndarray, charge: float, max_step: float) -> np.ndarray:
    """-q int_{start}^{(screen_x, y)} A_mu dz^mu along the straight lines (the phase of the Wilson line U), for every y of the screen."""
    out = np.empty(len(screen_y))
    p0 = np.array(start, dtype=float)
    for i, yy in enumerate(screen_y):
        out[i] = -charge * polygon_line_integral(a_func, np.array([p0, [screen_x, yy]]), max_step, closed=False)
    return out


def sci_parts(x: float) -> tuple[float, int]:
    """x = m * 10^e with 1 <= m < 10 (x > 0)."""
    e = int(math.floor(math.log10(x)))
    return x / 10.0 ** e, e
