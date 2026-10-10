r"""Physics of the film "The Meissner effect and the photon mass" (units hbar = c = 1, B_0 = 1).

The gauge field in a condensate.  With the phase of the condensate gauged away (the book's eq. symmetry_breaking8-10) the gauge field
A^mu in the region where |phi|^2 = v^2/2 obeys the Proca equation

    (d^2 + m_A^2) A^mu = 0,        m_A = e v,

and the current is the London current  j = -m_A^2 A.  The condensate is a FIXED background here (no back-reaction on v), which is the
approximation behind the London equation as well.  Two reductions of this equation are used in the film.

1. Static, 2D (the Meissner effect).  A = A_z(x, y) e_z, B = (d_y A_z, -d_x A_z); the contours of A_z are the magnetic field lines.  In
   equilibrium

        (nabla^2 - m_A^2(x, y)) A_z = 0,        m_A(x, y) = m_A inside a disc of radius R and 0 outside,

   with a uniform field B_0 = e_x far away.  The exact solution of this problem (a superconducting cylinder in a transverse field):

        outside   A_z = B_0 y (1 + a R^2 / r^2),                 a = 2 I_1(s) / (s I_0(s)) - 1,        s = m_A R
        inside    A_z = B_0 (2 / (m_A I_0(s))) I_1(m_A r) sin(theta)

   (A_z and its normal derivative are continuous at r = R).  Close to the surface the tangential field decays as exp(-d / lambda),
   lambda = 1 / m_A (London penetration depth); for m_A -> infinity the field lines are expelled and a -> -1.

2. Dynamic, 1D (a wave packet at the boundary).  A transverse polarisation A(x, t) of a wave that travels along x obeys exactly

        d_t^2 A - d_x^2 A + m_A^2(x) A = 0,            m_A(x) = m_A Theta(x),

   which is solved here by the leapfrog scheme with absorbing layers at both ends.  For omega < m_A the packet is reflected and leaves the
   evanescent tail exp(-kappa x), kappa = sqrt(m_A^2 - omega^2); for omega > m_A it is transmitted with k = sqrt(omega^2 - m_A^2) and the
   group velocity v_g = k / omega < 1.
"""

from __future__ import annotations

import math
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.special import ive

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dvconfig import load_config  # noqa: E402

HERE = Path(__file__).resolve().parent
CFG = load_config(HERE)

M_A = CFG.physics.m_a                # the mass of the gauge field in the condensate, the unit of frequency and of inverse length
TINY = 1e-12                         # tolerances of the numerics


def smooth(x: float, a: float, b: float) -> float:
    u = min(max((x - a) / (b - a), 0.0), 1.0)
    return u * u * (3 - 2 * u)


# =========================================================================================== static: a disc in a uniform field

def screening_coefficient(m: float, radius: float) -> float:
    """a = 2 I_1(s) / (s I_0(s)) - 1, s = m R: the amplitude of the dipole field outside the disc (a = 0: no effect, a = -1: perfect diamagnet)."""
    s = max(m * radius, TINY)
    return float(2.0 * ive(1, s) / (s * ive(0, s)) - 1.0)


def flux_function(x: np.ndarray, y: np.ndarray, m: float, radius: float) -> np.ndarray:
    """A_z(x, y) of the exact solution, B_0 = 1 (the magnetic field lines are the contours of this function)."""
    m = max(m, TINY)
    r = np.hypot(x, y)
    a = screening_coefficient(m, radius)
    out = y * (1.0 + a * radius ** 2 / np.maximum(r, TINY) ** 2)
    rin = np.minimum(r, radius)
    z = np.maximum(m * rin, TINY)
    ratio = ive(1, z) / ive(0, m * radius) * np.exp(m * rin - m * radius)       # I_1(m r) / I_0(m R) without overflow
    inside = 2.0 * y * ratio / (m * np.maximum(rin, TINY))
    return np.where(r > radius, out, inside)


def magnetic_field(x: np.ndarray, y: np.ndarray, m: float, radius: float) -> tuple[np.ndarray, np.ndarray]:
    """(B_x, B_y) = (d_y A_z, -d_x A_z) of the exact solution, B_0 = 1, analytic derivatives."""
    m = max(m, TINY)
    r = np.hypot(x, y)
    a = screening_coefficient(m, radius)
    ro = np.maximum(r, TINY)
    bx_out = 1.0 + a * radius ** 2 * (x ** 2 - y ** 2) / ro ** 4
    by_out = 2.0 * a * radius ** 2 * x * y / ro ** 4
    rin = np.maximum(np.minimum(r, radius), TINY)
    z = m * rin
    pref = (2.0 / m) * np.exp(z - m * radius) / ive(0, m * radius)             # C I_n(m r) = pref * ive(n, m r)
    i0, i1 = pref * ive(0, z), pref * ive(1, z)
    f = i1 / rin                                                               # C F(r),   F = I_1(m r) / r
    fp = m * i0 / rin - 2.0 * i1 / rin ** 2                                    # C F'(r)
    bx_in = f + (y ** 2 / rin) * fp
    by_in = -(x * y / rin) * fp
    return np.where(r > radius, bx_out, bx_in), np.where(r > radius, by_out, by_in)


def field_strength(x: np.ndarray, y: np.ndarray, m: float, radius: float) -> np.ndarray:
    bx, by = magnetic_field(x, y, m, radius)
    return np.hypot(bx, by)


def cut_profile(depth: np.ndarray, m: float, radius: float) -> np.ndarray:
    """B_x on the vertical line through the centre of the disc (x = 0): there the field is parallel to the surface.
    depth = R - y; depth < 0 is outside (vacuum), depth > 0 is inside the condensate."""
    y = radius - depth
    bx, _ = magnetic_field(np.zeros_like(y), y, m, radius)
    return bx


def planar_london(m: float, x: np.ndarray) -> np.ndarray:
    """B(x) of a uniform field B_0 = 1 in vacuum (x < 0) that meets the condensate m_A(x) = m Theta(x): B = 1, then exp(-m x)."""
    return np.where(x < 0.0, 1.0, np.exp(-m * np.maximum(x, 0.0)))


def solve_planar_static(m: float, x: np.ndarray) -> np.ndarray:
    """Numerical solution of A'' = m^2 Theta(x) A on the grid x (uniform), with A'(x_min) = B_0 = 1 and A(x_max) = 0; returns B = A'."""
    n = len(x)
    dx = float(x[1] - x[0])
    m2 = np.where(x > 0.0, m * m, 0.0)
    lap = np.zeros((n, n))
    idx = np.arange(1, n - 1)
    lap[idx, idx - 1] = 1.0
    lap[idx, idx] = -2.0 - dx * dx * m2[idx]
    lap[idx, idx + 1] = 1.0
    rhs = np.zeros(n)
    lap[0, 0], lap[0, 1] = -1.0, 1.0                    # (A_1 - A_0) / dx = B_0 = 1
    rhs[0] = dx
    lap[-1, -1] = 1.0                                   # A(x_max) = 0
    a = np.linalg.solve(lap, rhs)
    return np.gradient(a, dx)


def decay_length(x: np.ndarray, b: np.ndarray, lo: float, hi: float) -> float:
    """1 / |slope| of ln B(x) on lo <= x <= hi (a straight-line fit): the penetration depth."""
    sel = (x >= lo) & (x <= hi) & (b > 0.0)
    slope = np.polyfit(x[sel], np.log(b[sel]), 1)[0]
    return -1.0 / float(slope)


# ========================================================================================== dynamic: the packet at the boundary

def kappa(omega: float, m: float = M_A) -> float:
    """Inverse penetration depth of a wave below the gap, sqrt(m^2 - omega^2)."""
    return math.sqrt(m * m - omega * omega)


def wavenumber_in_condensate(omega: float, m: float = M_A) -> float:
    return math.sqrt(omega * omega - m * m)


def group_velocity(omega: float, m: float = M_A) -> float:
    """v_g = d omega / d k = k / omega for omega^2 = k^2 + m^2 (zero below the gap: nothing propagates)."""
    return math.sqrt(omega * omega - m * m) / omega if omega > m else 0.0


def omega_of_k(k: np.ndarray | float, m: float = M_A):
    return np.sqrt(np.asarray(k, float) ** 2 + m * m)


def transmitted_energy_fraction(omega: float, m: float = M_A) -> float:
    """Energy transmission of a monochromatic wave at the step m_A Theta(x): 4 omega k / (omega + k)^2 above the gap, 0 below."""
    if omega <= m:
        return 0.0
    k = wavenumber_in_condensate(omega, m)
    return 4.0 * omega * k / (omega + k) ** 2


@dataclass
class Setup:
    x: np.ndarray            # grid
    dx: float
    dt: float
    mass2: np.ndarray        # m_A^2(x)
    gamma: np.ndarray        # absorbing layers: A_tt + gamma A_t - A_xx + m^2 A = 0


def make_setup(m: float = M_A, absorbing: bool = True) -> Setup:
    P = CFG.physics.proca
    n = int(round((P.x_max - P.x_min) / P.dx))
    x = P.x_min + (np.arange(n) + 0.5) * P.dx                  # the step at x = 0 lies half-way between two grid points
    mass2 = np.where(x > 0.0, m * m, 0.0)
    depth = np.maximum(P.x_min + P.sponge_width - x, x - (P.x_max - P.sponge_width))        # how far into an absorbing layer (negative: outside)
    gamma = P.sponge_strength * (np.clip(depth, 0.0, None) / P.sponge_width) ** P.sponge_power if absorbing else np.zeros(n)
    return Setup(x, P.dx, P.cfl * P.dx, mass2, gamma)


def packet_state(su: Setup, omega: float, sigma: float, x0: float, amplitude: float) -> tuple[np.ndarray, np.ndarray]:
    """A right-moving packet in vacuum, A = f(x - t): returns A at t = -dt and t = 0 (two leapfrog levels)."""
    k0 = omega

    def f(s: np.ndarray) -> np.ndarray:
        return amplitude * np.cos(k0 * (s - x0)) * np.exp(-((s - x0) ** 2) / (2.0 * sigma ** 2))

    return f(su.x + su.dt), f(su.x)


def laplacian(a: np.ndarray, dx: float) -> np.ndarray:
    out = np.zeros_like(a)
    out[1:-1] = (a[2:] - 2.0 * a[1:-1] + a[:-2]) / (dx * dx)
    return out


def leapfrog_step(su: Setup, a_old: np.ndarray, a: np.ndarray) -> np.ndarray:
    g = 0.5 * su.gamma * su.dt
    new = (2.0 * a - (1.0 - g) * a_old + su.dt ** 2 * (laplacian(a, su.dx) - su.mass2 * a)) / (1.0 + g)
    new[0] = new[-1] = 0.0
    return new


def discrete_energy(su: Setup, a_old: np.ndarray, a: np.ndarray) -> float:
    """The energy that the leapfrog scheme conserves exactly without absorption:
    sum dx [ ((a - a_old)/dt)^2 / 2 + (D a . D a_old + m^2 a a_old) / 2 ],  D = forward difference / dx."""
    kin = 0.5 * np.sum(((a - a_old) / su.dt) ** 2) * su.dx
    da, da_old = np.diff(a) / su.dx, np.diff(a_old) / su.dx
    pot = 0.5 * (np.sum(da * da_old) * su.dx + np.sum(su.mass2 * a * a_old) * su.dx)
    return float(kin + pot)


def energy_density(su: Setup, a_old: np.ndarray, a_new: np.ndarray) -> np.ndarray:
    """u = (A_t^2 + A_x^2 + m^2 A^2) / 2 at the time of the middle level (central differences)."""
    a = 0.5 * (a_old + a_new)
    at = (a_new - a_old) / (2.0 * su.dt)
    ax = np.gradient(a, su.dx)
    return 0.5 * (at ** 2 + ax ** 2 + su.mass2 * a ** 2)


@dataclass
class Run:
    """One packet, sampled at ``samples_per_s`` per film second (sim time = rate * film time)."""
    x: np.ndarray
    t: np.ndarray             # sim time of the samples
    a: np.ndarray             # samples x grid: A(x, t)
    tail_x: np.ndarray        # the window in which the running maximum of |A| is kept
    tail_max: np.ndarray      # samples x window: max over the time so far of |A(x)|
    u_left: np.ndarray        # energy in x < 0
    u_right: np.ndarray       # energy in x > 0
    xc_left: np.ndarray       # centroid of the energy density in x < 0 (nan if empty)
    xc_right: np.ndarray      # centroid of the energy density in x > 0
    xc_all: np.ndarray        # centroid of the energy density of the whole box
    spec: np.ndarray          # samples x window: the Fourier component of A(x, t) at the carrier omega, summed over the samples so far
    omega: float
    energy0: float


def simulate(omega: float, sigma: float, x0: float, amplitude: float, duration_s: float, absorbing: bool = True) -> Run:
    P = CFG.physics.proca
    su = make_setup(absorbing=absorbing)
    a_old, a = packet_state(su, omega, sigma, x0, amplitude)          # the levels A(-dt) and A(0)
    steps = int(round(P.wave_rate / P.samples_per_s / su.dt))          # leapfrog steps between two samples
    n_samples = int(round(duration_s * P.samples_per_s)) + 1
    lo, hi = P.tail_window
    tail = (su.x >= lo) & (su.x <= hi)
    left, right = su.x < 0.0, su.x > 0.0
    amax = np.abs(a)
    rec = {k: [] for k in ("a", "tail", "ul", "ur", "xl", "xr", "xa", "t", "spec")}
    dt_s = steps * su.dt
    fourier = np.zeros(int(tail.sum()), complex)
    energy0 = None
    for i in range(n_samples):
        a_new = leapfrog_step(su, a_old, a)
        u = energy_density(su, a_old, a_new)                           # at the time of `a`
        if energy0 is None:
            energy0 = float(u.sum() * su.dx)
        ul, ur = float(u[left].sum() * su.dx), float(u[right].sum() * su.dx)
        rec["a"].append(a.copy())
        rec["tail"].append(amax[tail].copy())
        rec["ul"].append(ul)
        rec["ur"].append(ur)
        rec["xl"].append(float((su.x[left] * u[left]).sum() * su.dx / ul) if ul > TINY else math.nan)
        rec["xr"].append(float((su.x[right] * u[right]).sum() * su.dx / ur) if ur > TINY else math.nan)
        rec["xa"].append(float((su.x * u).sum() * su.dx / (ul + ur)))
        fourier = fourier + a[tail] * np.exp(1j * omega * i * dt_s) * dt_s
        rec["spec"].append(fourier.copy())
        rec["t"].append(i * dt_s)
        a_old, a = a, a_new
        amax = np.maximum(amax, np.abs(a))
        for _ in range(steps - 1):
            a_new = leapfrog_step(su, a_old, a)
            a_old, a = a, a_new
            np.maximum(amax, np.abs(a), out=amax)
    return Run(su.x, np.array(rec["t"]), np.array(rec["a"]), su.x[tail], np.array(rec["tail"]), np.array(rec["ul"]), np.array(rec["ur"]),
               np.array(rec["xl"]), np.array(rec["xr"]), np.array(rec["xa"]), np.array(rec["spec"]), omega, energy0)


def spectral_profile(run: Run, index: int | None = None) -> np.ndarray:
    """|A_omega(x)| in the tail window: the modulus of the Fourier component of the simulated field at the carrier frequency (up to the sample
    ``index``).  Every frequency of the packet is attenuated as exp(-kappa(omega) x) separately, so this is free of the bandwidth of the packet."""
    return np.abs(run.spec[-1 if index is None else index])


def measure_penetration(run: Run, lo: float, hi: float, index: int | None = None) -> float:
    """Penetration depth from the simulation: the straight-line fit of ln |A_omega(x)| on lo <= x <= hi (at the sample ``index``, default the last)."""
    prof = spectral_profile(run, index)
    sel = (run.tail_x >= lo) & (run.tail_x <= hi) & (prof > 0.0)
    slope = np.polyfit(run.tail_x[sel], np.log(prof[sel]), 1)[0]
    return -1.0 / float(slope)


def measure_wavenumber(run: Run, lo: float, hi: float, index: int | None = None) -> float:
    """Wave number in the condensate from the simulation: the slope of the phase of the Fourier component A_omega(x) on lo <= x <= hi (once the
    packet has passed): a transmitted wave is exp(i k x), and every frequency of the packet has its own k(omega)."""
    sp = run.spec[-1 if index is None else index]
    sel = (run.tail_x >= lo) & (run.tail_x <= hi)
    return abs(float(np.polyfit(run.tail_x[sel], np.unwrap(np.angle(sp[sel])), 1)[0]))


def measure_speed(t: np.ndarray, xc: np.ndarray, t0: float, t1: float) -> float:
    """Speed of a centroid: the slope of a straight-line fit of xc(t) on t0 <= t <= t1."""
    sel = (t >= t0) & (t <= t1) & np.isfinite(xc)
    return float(np.polyfit(t[sel], xc[sel], 1)[0])
