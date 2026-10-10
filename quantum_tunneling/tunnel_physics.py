r"""Physics of the film "Tunnelling: a point body and an extended body" (no pictures, no configuration; every number is an argument).

Three parts, one recurring shape (the Gaussian hill  f(u) = exp(-u^2/2)):

A. Classical.  A track  y = h(x) = H f(x/sigma);  N cars of mass m joined rigidly (fixed separations ``l`` ALONG the track),
   friction-free, one degree of freedom: the arc length S of the head car, the cars sit at S - i l.
       L = (1/2) N m Sdot^2 - U(S),      U(S) = sum_i m g h(x(S - i l)).
   U(S) is the barrier of the whole train.  The train passes iff  E_total > max_S U(S)   (E_total = (N/2) m v0^2 on flat ground).
   For N = 1 this is m g H (the point body); for a long train U_max -> (m g / l) int h sqrt(1 + h'^2) dx.

B. Quantum.  i hbar dpsi/dt = [-(hbar^2/2m) d^2/dx^2 + V(x)] psi, V = V0 f(x/w); split-step Fourier with absorbing edges; the exact
   stationary transmission T(E) (ODE through the hill); a packet transmits  P_T = int |phi(k)|^2 T(E_k) dk.

C. Josephson junction (RCSJ, zero temperature, no damping):  the phase delta moves in the washboard
       U(delta) = -E_J cos(delta) - s E_J delta,   s = I / I_c,     with the "mass" M = hbar^2 / (lambda^2 E_J),  lambda = hbar omega_p0 / E_J.
   Barrier  Delta U(s) = 2 E_J [sqrt(1 - s^2) - s arccos s],   omega_p(s) = omega_p0 (1 - s^2)^(1/4).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from scipy.integrate import solve_ivp
from scipy.interpolate import CubicSpline
from scipy.optimize import brentq


# =================================================================================================== A. the train on the hill

@dataclass(frozen=True)
class Track:
    """The track y = H exp(-x^2 / (2 sigma^2)); the arc length s is counted from x = -x_lim."""
    height: float
    sigma: float
    x_lim: float = 70.0
    step: float = 0.01

    def __post_init__(self) -> None:
        xg = np.arange(-self.x_lim, self.x_lim + self.step / 2, self.step)
        sp = np.sqrt(1.0 + self.slope(xg) ** 2)
        sg = np.concatenate([[0.0], np.cumsum(0.5 * (sp[1:] + sp[:-1]) * np.diff(xg))])      # trapezoid: error ~ step^2 * h'''
        # a Simpson-accurate correction is not needed: both directions of the map use the same table (see the test of the inverse)
        object.__setattr__(self, "_xg", xg)
        object.__setattr__(self, "_sg", sg)
        object.__setattr__(self, "_x_of_s", CubicSpline(sg, xg))
        object.__setattr__(self, "_s_of_x", CubicSpline(xg, sg))

    def h(self, x):
        return self.height * np.exp(-np.asarray(x) ** 2 / (2.0 * self.sigma ** 2))

    def slope(self, x):
        x = np.asarray(x)
        return -x / self.sigma ** 2 * self.h(x)

    def sin_theta(self, x):
        s = self.slope(x)
        return s / np.sqrt(1.0 + s * s)

    def arc(self, x):
        return self._s_of_x(x)

    def x_at(self, s):
        return self._x_of_s(np.clip(s, self._sg[0], self._sg[-1]))

    @property
    def s_min(self) -> float:
        return float(self._sg[0])

    @property
    def s_max(self) -> float:
        return float(self._sg[-1])

    def extra_arc(self) -> float:
        """Arc length of the track minus its horizontal extent."""
        return float(self._sg[-1] - 2 * self.x_lim)


@dataclass(frozen=True)
class Train:
    """N cars of mass m, joined rigidly at the arc-length spacing l, on a Track; g is the free-fall acceleration."""
    track: Track
    n: int
    spacing: float
    mass: float
    g: float

    def x_cars(self, s_head):
        s = np.asarray(s_head, float)[..., None] - self.spacing * np.arange(self.n)
        return self.track.x_at(s)

    def potential(self, s_head):
        return self.mass * self.g * self.track.h(self.x_cars(s_head)).sum(-1)

    def force(self, s_head):
        return -self.mass * self.g * self.track.sin_theta(self.x_cars(s_head)).sum(-1)

    def s_start(self, margin_sigma: float) -> float:
        """Arc position of a head that is ``margin_sigma`` sigma before the top of the hill."""
        return float(self.track.arc(-margin_sigma * self.track.sigma))

    def s_end(self, margin_sigma: float) -> float:
        """Arc position of the head when the tail has left the hill by ``margin_sigma`` sigma."""
        return float(self.track.arc(margin_sigma * self.track.sigma)) + self.spacing * (self.n - 1)

    def u_max(self, margin_sigma: float, samples: int = 200001) -> float:
        """The threshold: max_S U(S), S from the start to the end of the hill (the tail has left it)."""
        s0 = self.s_start(margin_sigma)
        s1 = float(self.track.arc(margin_sigma * self.track.sigma)) + self.spacing * (self.n - 1)
        s = np.linspace(s0, s1, samples)
        return float(self.potential(s).max())

    def critical_speed(self, margin_sigma: float) -> float:
        u0 = float(self.potential(self.s_start(margin_sigma)))
        return math.sqrt(2.0 * (self.u_max(margin_sigma) - u0) / (self.n * self.mass))

    def simulate(self, s0: float, v0: float, t_end: float, dt: float, stride: int = 1):
        """Velocity Verlet in S; returns t, S, Sdot (every ``stride`` steps) and the total energy at every output step."""
        big_m = self.n * self.mass
        steps = int(round(t_end / dt))
        out = steps // stride + 1
        t = np.zeros(out)
        s_arr = np.zeros(out)
        v_arr = np.zeros(out)
        e_arr = np.zeros(out)
        s, v = float(s0), float(v0)
        a = float(self.force(s)) / big_m
        for i in range(steps + 1):
            if i % stride == 0:
                j = i // stride
                t[j], s_arr[j], v_arr[j] = i * dt, s, v
                e_arr[j] = 0.5 * big_m * v * v + float(self.potential(s))
            if i == steps:
                break
            v += 0.5 * dt * a
            s += dt * v
            a = float(self.force(s)) / big_m
            v += 0.5 * dt * a
        return t, s_arr, v_arr, e_arr

    def car_geometry(self, s_head: float):
        """Positions (x, y) and tangent angles of the cars at the head position s_head."""
        x = self.x_cars(s_head)
        return x, self.track.h(x), np.arctan(self.track.slope(x))

    def asymptotic_threshold(self) -> float:
        """lim_{N->inf} max U = (m g / l) int h sqrt(1 + h'^2) dx   (the hill 'area' along the track, per car spacing)."""
        xg = np.linspace(-self.track.x_lim, self.track.x_lim, 400001)
        f = self.track.h(xg) * np.sqrt(1.0 + self.track.slope(xg) ** 2)
        return float(self.mass * self.g / self.spacing * np.trapezoid(f, xg))


def will_pass(train: Train, s0: float, v0: float, dt: float, s_far: float, t_max: float) -> bool:
    """Integrate the motion (velocity Verlet) and stop as soon as the head turns back (Sdot < 0) or the whole train has gone over (S >= s_far)."""
    big_m = train.n * train.mass
    s, v = float(s0), float(v0)
    a = float(train.force(s)) / big_m
    t = 0.0
    while t < t_max:
        v += 0.5 * dt * a
        s += dt * v
        a = float(train.force(s)) / big_m
        v += 0.5 * dt * a
        t += dt
        if v < 0.0:
            return False
        if s >= s_far:
            return True
    return s >= s_far


def critical_speed_by_bisection(train: Train, s0: float, s_far: float, v_lo: float, v_hi: float, dt: float, t_max: float, rel_tol: float = 1e-4) -> float:
    """The launch speed in [v_lo, v_hi] that separates 'turns back' from 'passes', found only by integrating the motion
    (the bracket is checked: the lower end must turn back, the upper end must pass)."""
    if will_pass(train, s0, v_lo, dt, s_far, t_max) or not will_pass(train, s0, v_hi, dt, s_far, t_max):
        raise ValueError("the bracket does not enclose the threshold")
    lo, hi = v_lo, v_hi
    while hi - lo > rel_tol * hi:
        mid = 0.5 * (lo + hi)
        if will_pass(train, s0, mid, dt, s_far, t_max):
            hi = mid
        else:
            lo = mid
    return 0.5 * (lo + hi)


# =================================================================================================== B. the quantum packet

def gauss_hill(x, v0: float, w: float):
    return v0 * np.exp(-np.asarray(x) ** 2 / (2.0 * w ** 2))


def transmission(energy: float, potential, mass: float, hbar: float, half_width: float, breakpoints=()) -> float:
    """Exact stationary transmission probability through V(x) (V -> 0 at +-half_width): the ODE is integrated from a pure
    transmitted wave exp(ikx) at x = +half_width to x = -half_width, where it is decomposed into incident + reflected waves.
    ``breakpoints`` are the points where V is discontinuous (the integration is restarted there)."""
    k = math.sqrt(2.0 * mass * energy) / hbar

    def rhs(x, y):
        return np.array([y[1], -2.0 * mass / hbar ** 2 * (energy - potential(x)) * y[0]], dtype=complex)

    pts = sorted({*breakpoints}, reverse=True)
    stops = [half_width] + [p for p in pts if -half_width < p < half_width] + [-half_width]
    y = np.array([np.exp(1j * k * half_width), 1j * k * np.exp(1j * k * half_width)], dtype=complex)
    for a, b in zip(stops[:-1], stops[1:]):
        sol = solve_ivp(rhs, [a, b], y, method="DOP853", rtol=1e-11, atol=1e-13)
        y = sol.y[:, -1]
    psi, dpsi = y
    amp_in = 0.5 * (psi + dpsi / (1j * k)) * np.exp(1j * k * half_width)         # coefficient of exp(ikx) at x = -half_width
    return float(1.0 / abs(amp_in) ** 2)


def stationary_state(energy: float, potential, mass: float, hbar: float, half_width: float, xs: np.ndarray) -> np.ndarray:
    """psi(x) of the scattering state (incident wave of unit amplitude from the left), on the grid xs."""
    k = math.sqrt(2.0 * mass * energy) / hbar

    def rhs(x, y):
        return np.array([y[1], -2.0 * mass / hbar ** 2 * (energy - potential(x)) * y[0]], dtype=complex)

    y0 = np.array([np.exp(1j * k * half_width), 1j * k * np.exp(1j * k * half_width)], dtype=complex)
    sol = solve_ivp(rhs, [half_width, -half_width], y0, method="DOP853", rtol=1e-10, atol=1e-12, dense_output=True)
    psi = sol.sol(xs)[0]
    psi_l = sol.sol(-half_width)
    amp_in = 0.5 * (psi_l[0] + psi_l[1] / (1j * k)) * np.exp(1j * k * half_width)
    return psi / amp_in


def rectangle_transmission(energy: float, v0: float, a: float, mass: float, hbar: float) -> float:
    """T = [1 + V0^2 sinh^2(kappa a) / (4 E (V0 - E))]^-1 for E < V0, the barrier of width a."""
    if energy < v0:
        kappa = math.sqrt(2.0 * mass * (v0 - energy)) / hbar
        return 1.0 / (1.0 + v0 ** 2 * math.sinh(kappa * a) ** 2 / (4.0 * energy * (v0 - energy)))
    q = math.sqrt(2.0 * mass * (energy - v0)) / hbar
    return 1.0 / (1.0 + v0 ** 2 * math.sin(q * a) ** 2 / (4.0 * energy * (energy - v0)))


def wkb_exponent_gauss(energy: float, v0: float, w: float, mass: float, hbar: float) -> float:
    """S = 2 int kappa dx over the classically forbidden region of V = V0 exp(-x^2/2w^2); T ~ exp(-S)."""
    if energy >= v0:
        return 0.0
    xt = w * math.sqrt(2.0 * math.log(v0 / energy))
    xs = np.linspace(-xt, xt, 20001)
    kap = np.sqrt(np.maximum(2.0 * mass * (gauss_hill(xs, v0, w) - energy), 0.0)) / hbar
    return float(2.0 * np.trapezoid(kap, xs))


def turning_point_gauss(energy: float, v0: float, w: float) -> float:
    return w * math.sqrt(2.0 * math.log(v0 / energy))


class SplitStep1D:
    """psi_t = -i H psi / hbar with H = p^2/2m + V(x, t): Strang splitting (exact FFT kinetic step), absorbing edges.
    The potential may depend on time through ``potential(t)`` (a callable returning the array V on the grid)."""

    def __init__(self, n: int, length: float, mass: float, hbar: float, dt: float, potential, absorber_width: float, absorber_strength: float,
                 x_left: float | None = None):
        self.n, self.mass, self.hbar, self.dt = n, mass, hbar, dt
        x0 = -length / 2 if x_left is None else x_left
        self.x = x0 + length * np.arange(n) / n
        self.dx = length / n
        self.k = 2.0 * np.pi * np.fft.fftfreq(n, d=self.dx)
        self.kin = np.exp(-1j * hbar * self.k ** 2 / (2.0 * mass) * dt)
        self.potential = potential if callable(potential) else (lambda t, v=np.asarray(potential): v)
        d_left = np.clip((self.x[0] + absorber_width - self.x) / absorber_width, 0.0, 1.0)
        d_right = np.clip((self.x - (self.x[-1] - absorber_width)) / absorber_width, 0.0, 1.0)
        self.damp = np.exp(-absorber_strength * (d_left ** 2 + d_right ** 2) * dt)
        self.t = 0.0
        self.psi = np.zeros(n, complex)

    def set_state(self, psi: np.ndarray) -> None:
        self.psi = np.asarray(psi, complex).copy()

    def step(self) -> None:
        half = np.exp(-0.5j * self.dt / self.hbar * self.potential(self.t + 0.5 * self.dt))
        p = half * self.psi
        p = np.fft.ifft(self.kin * np.fft.fft(p))
        p = half * p
        self.psi = p * self.damp
        self.t += self.dt

    def run(self, t_end: float) -> None:
        while self.t < t_end - 1e-12:
            self.step()

    def density(self) -> np.ndarray:
        return np.abs(self.psi) ** 2

    def norm(self) -> float:
        return float(self.density().sum() * self.dx)

    def mean_kinetic(self) -> float:
        pk = np.fft.fft(self.psi)
        w = np.abs(pk) ** 2
        return float((self.hbar ** 2 * self.k ** 2 / (2.0 * self.mass) * w).sum() / w.sum())

    def mean_potential(self, v: np.ndarray) -> float:
        return float((self.density() * v).sum() * self.dx)


def gaussian_packet(x: np.ndarray, x0: float, k0: float, sigma_x: float) -> np.ndarray:
    """Normalised packet; sigma_x is the standard deviation of |psi|^2."""
    psi = np.exp(-((x - x0) ** 2) / (4.0 * sigma_x ** 2) + 1j * k0 * (x - x0))
    return psi / math.sqrt(float((np.abs(psi) ** 2).sum() * (x[1] - x[0])))


def packet_energy_distribution(psi: np.ndarray, dx: float, mass: float, hbar: float):
    """(k > 0 grid, weights |phi_k|^2 dk, energies E_k) of an initial packet; plus the weight of k < 0."""
    n = psi.size
    k = 2.0 * np.pi * np.fft.fftfreq(n, d=dx)
    w = np.abs(np.fft.fft(psi)) ** 2
    w = w / w.sum()
    pos = k > 0
    return k[pos], w[pos], (hbar * k[pos]) ** 2 / (2.0 * mass), float(w[~pos].sum())


def packet_transmission(psi: np.ndarray, dx: float, mass: float, hbar: float, potential, half_width: float, v_top: float, n_energy: int = 400):
    """P_T = sum_k |phi_k|^2 T(E_k) and its split into the under-barrier (E < v_top) and over-barrier (E >= v_top) parts.
    T is computed exactly on a grid of energies and interpolated (it is smooth)."""
    k, w, e, w_neg = packet_energy_distribution(psi, dx, mass, hbar)
    sel = w > 1e-14 * w.max()
    e_lo, e_hi = e[sel].min(), e[sel].max()
    eg = np.linspace(e_lo, e_hi, n_energy)
    tg = np.array([transmission(float(en), potential, mass, hbar, half_width) for en in eg])
    t_of_e = CubicSpline(eg, tg)
    t = np.where(sel, t_of_e(np.clip(e, e_lo, e_hi)), 0.0)
    under = float((w * t)[e < v_top].sum())
    over = float((w * t)[e >= v_top].sum())
    return under + over, under, over, float(w[e >= v_top].sum()), w_neg


# =================================================================================================== C. the Josephson junction

def washboard(delta, s: float):
    """U / E_J = -cos(delta) - s delta."""
    return -np.cos(delta) - s * np.asarray(delta)


def barrier_height(s: float) -> float:
    """Delta U / E_J = 2 [sqrt(1 - s^2) - s arccos s]."""
    return 2.0 * (math.sqrt(1.0 - s * s) - s * math.acos(s))


def plasma_ratio(s: float) -> float:
    """omega_p(s) / omega_p0 = (1 - s^2)^(1/4)."""
    return (1.0 - s * s) ** 0.25


def well_and_top(s: float) -> tuple[float, float]:
    return math.asin(s), math.pi - math.asin(s)


def barrier_over_hbar_omega(s: float, lam: float) -> float:
    """x = Delta U / (hbar omega_p) with lam = hbar omega_p0 / E_J."""
    return barrier_height(s) / (plasma_ratio(s) * lam)


def rate_cubic(x: float) -> float:
    """Gamma / (omega_p / 2 pi) = a_q exp(-36 x / 5),  a_q = sqrt(864 pi x),  x = Delta U / (hbar omega_p)  (cubic potential, T = 0)."""
    return math.sqrt(864.0 * math.pi * x) * math.exp(-7.2 * x)


def wkb_exponent_washboard(s: float, lam: float, energy_over_ej: float | None = None) -> float:
    """S = 2 int sqrt(2 M (U - E)) / hbar d delta over the forbidden region of the washboard; with M = hbar^2 / (lam^2 E_J) the integrand is
    sqrt(2 (U - E) / E_J) / lam.  Default energy: the zero-point level of the harmonic well, E = U_min + lam plasma_ratio(s) / 2 (units of E_J)."""
    d0, d1 = well_and_top(s)
    e = float(washboard(d0, s)) + 0.5 * lam * plasma_ratio(s) if energy_over_ej is None else energy_over_ej
    a, b = washboard_turning_points(s, e)
    ds = np.linspace(a, b, 20001)
    kap = np.sqrt(np.maximum(2.0 * (washboard(ds, s) - e), 0.0)) / lam
    return float(2.0 * np.trapezoid(kap, ds))


def washboard_turning_points(s: float, e: float) -> tuple[float, float]:
    """The two roots of U(delta) = e (E_J units) on the two sides of the barrier top delta_1 = pi - arcsin s, left of the well to the next well."""
    d0, d1 = well_and_top(s)

    def f(d):
        return float(washboard(d, s)) - e

    a = brentq(f, d0, d1)                                  # f(d0) < 0 < f(d1): U rises from the well bottom to the top
    grid = np.linspace(d1, d1 + 8.0 * math.pi, 40001)
    vals = washboard(grid, s) - e
    i = int(np.argmax(vals < 0.0))
    b = brentq(f, grid[i - 1], grid[i])
    return float(a), float(b)


def well_ground_state(grid: np.ndarray, s: float, lam: float) -> np.ndarray:
    """Ground state of H/E_J = -(lam^2/2) d^2/d delta^2 + U(delta) / E_J truncated by a hard wall at the top of the barrier
    (finite differences, normalised on the grid). The quasi-bound state of the well."""
    d0, d1 = well_and_top(s)
    dx = grid[1] - grid[0]
    inside = grid < d1
    gi = grid[inside]
    n = gi.size
    diag = lam ** 2 / dx ** 2 + washboard(gi, s)
    off = -0.5 * lam ** 2 / dx ** 2 * np.ones(n - 1)
    from scipy.linalg import eigh_tridiagonal
    w, v = eigh_tridiagonal(diag, off, select="i", select_range=(0, 0))
    psi = np.zeros(grid.size)
    psi[inside] = np.abs(v[:, 0])
    return psi / math.sqrt(float((psi ** 2).sum() * dx)), float(w[0])


def harmonic_frequency_numeric(s: float, lam: float, h: float = 1e-3) -> float:
    """omega / omega_p0 from the numerical second derivative of U at the minimum: sqrt(U''(delta_0) / E_J)."""
    d0 = math.asin(s)
    u2 = (float(washboard(d0 + h, s)) - 2.0 * float(washboard(d0, s)) + float(washboard(d0 - h, s))) / h ** 2
    return math.sqrt(u2)
