r"""Neutral kaons K0 - K0bar: the exact two-level evolution with the effective Hamiltonian H = M - i Gamma / 2.

Basis (K0, K0bar).  CP is conserved (the book, the section "Strangeness oscillations"), so M and Gamma are real symmetric
matrices with the eigenvectors

    K_L = (K0 + K0bar) / sqrt(2)   (mass m_L, width Gamma_L),     K_S = (K0 - K0bar) / sqrt(2)   (mass m_S, width Gamma_S),

    M      = m0 + (dm / 2) sigma_x,                  dm = m_L - m_S,
    Gamma  = (Gamma_S + Gamma_L) / 2 + ((Gamma_L - Gamma_S) / 2) sigma_x.

The time is measured in tau_S = 1 / Gamma_S, the energies in hbar / tau_S; m0 = 0 (the phase of the mean mass is a common phase
factor and is removed, "the frame rotating with the mean mass").  The Schroedinger equation i d psi / dt = H psi is integrated
numerically (DOP853); the closed formulas of the book are used only for the tests and the labels, never for the pictures.

All the numbers are in config.toml (see ../dvconfig.py).
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
from scipy.integrate import solve_ivp

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dvconfig import load_config  # noqa: E402

HERE = Path(__file__).resolve().parent
CFG = load_config(HERE)
PH = CFG.physics
IN = CFG.integrator

GAMMA_S = 1.0                                              # 1 / tau_S
GAMMA_L = PH.tau_s_s / PH.tau_l_s                          # tau_S / tau_L
TAU_L = 1.0 / GAMMA_L                                      # tau_L in units of tau_S
DM = PH.delta_m_ev / PH.hbar_ev_s * PH.tau_s_s             # Delta m * tau_S
PERIOD = 2.0 * math.pi / DM                                # period of the oscillation without decays, units of tau_S
T_MAX = PH.t_max_in_tau_l * TAU_L

K0 = np.array([1.0, 0.0], complex)
K0BAR = np.array([0.0, 1.0], complex)
KS = np.array([1.0, -1.0], complex) / math.sqrt(2.0)
KL = np.array([1.0, 1.0], complex) / math.sqrt(2.0)
SX = np.array([[0.0, 1.0], [1.0, 0.0]])
I2 = np.eye(2)


def hamiltonian(gs: float = GAMMA_S, gl: float = GAMMA_L, dm: float = DM, m0: float = 0.0) -> np.ndarray:
    """H = M - i Gamma / 2 in the basis (K0, K0bar); eigenvalues m_{S,L} - i Gamma_{S,L} / 2 with eigenvectors K_S, K_L."""
    mass = m0 * I2 + 0.5 * dm * SX
    width = 0.5 * (gs + gl) * I2 + 0.5 * (gl - gs) * SX
    return mass - 0.5j * width


def integrate(h: np.ndarray, psi0: np.ndarray, times: np.ndarray) -> np.ndarray:
    """psi(t) at the given increasing times, from i d psi / dt = H psi (times[0] may be > 0: psi0 is the state at times[0])."""
    times = np.asarray(times, float)
    if len(times) == 1:
        return np.asarray(psi0, complex)[None, :]
    sol = solve_ivp(lambda _t, y: -1j * (h @ y), (times[0], times[-1]), np.asarray(psi0, complex), t_eval=times,
                    method=IN.method, rtol=IN.rtol, atol=IN.atol)
    if not sol.success:
        raise RuntimeError(sol.message)
    return sol.y.T


def time_grid(t_end: float) -> np.ndarray:
    """A fine grid for t < fine_until and a coarser one afterwards (the oscillation period is 13 tau_S, the grid step is 0.01 / 0.05)."""
    a = np.arange(0.0, min(IN.fine_until, t_end), IN.fine_dt)
    b = np.arange(IN.fine_until, t_end + 0.5 * IN.coarse_dt, IN.coarse_dt)
    return np.unique(np.concatenate([a, b[b <= t_end + 1e-9], [t_end]]))


class Run:
    """A numerically integrated state psi(t) on a grid, with interpolation to any time."""

    def __init__(self, h: np.ndarray, psi0: np.ndarray, t_end: float, t0: float = 0.0) -> None:
        self.t = time_grid(t_end - t0) + t0
        self.psi = integrate(h, psi0, self.t)

    def at(self, t):
        """(a0, abar) at the time(s) t (linear interpolation of the real and imaginary parts on the fine grid)."""
        t = np.asarray(t, float)
        out = []
        for k in range(2):
            out.append(np.interp(t, self.t, self.psi[:, k].real) + 1j * np.interp(t, self.t, self.psi[:, k].imag))
        return out[0], out[1]


_cache: dict = {}


def beam_run() -> Run:
    """A K0 beam (born at t = 0) with decays, up to T_MAX."""
    if "beam" not in _cache:
        _cache["beam"] = Run(hamiltonian(), K0, T_MAX)
    return _cache["beam"]


def stable_run() -> Run:
    """The same beam of stable kaons (Gamma_S = Gamma_L = 0), one period of the oscillation and a little more."""
    if "stable" not in _cache:
        _cache["stable"] = Run(hamiltonian(0.0, 0.0), K0, PH.stable_t_end)
    return _cache["stable"]


def s_l(a0, abar):
    """The K_S and K_L parts of the amplitudes: a0 = s + l, abar = l - s (s = <K_S|psi>/sqrt2, l = <K_L|psi>/sqrt2)."""
    return 0.5 * (a0 - abar), 0.5 * (a0 + abar)


def intensities(a0, abar):
    return np.abs(a0) ** 2, np.abs(abar) ** 2


def project(psi: np.ndarray):
    """(<K_S|psi>, <K_L|psi>) for psi of shape (..., 2)."""
    return psi @ KS.conj(), psi @ KL.conj()


# ----------------------------------------------------------------------------------- the closed formulas of the book (for the tests)

def p_survive(t, gs: float = GAMMA_S, gl: float = GAMMA_L, dm: float = DM):
    """P(K0 -> K0), eq. (neutral_kaons_17) of the book."""
    t = np.asarray(t, float)
    return 0.25 * (np.exp(-gs * t) + np.exp(-gl * t) + 2.0 * np.exp(-0.5 * (gs + gl) * t) * np.cos(dm * t))


def p_appear(t, gs: float = GAMMA_S, gl: float = GAMMA_L, dm: float = DM):
    """P(K0 -> K0bar)."""
    t = np.asarray(t, float)
    return 0.25 * (np.exp(-gs * t) + np.exp(-gl * t) - 2.0 * np.exp(-0.5 * (gs + gl) * t) * np.cos(dm * t))


def asymmetry(t, gs: float = GAMMA_S, gl: float = GAMMA_L, dm: float = DM):
    """A_l(t) = cos(dm t) / cosh(dGamma t / 2), eq. (Kl3_A_final)."""
    t = np.asarray(t, float)
    return np.cos(dm * t) / np.cosh(0.5 * (gs - gl) * t)


def lepton_asymmetry(a0, abar):
    """The charge asymmetry of the semileptonic decays from the amplitudes: (N+ - N-) / (N+ + N-), N+ ~ |a0|^2, N- ~ |abar|^2."""
    i0, ib = intensities(a0, abar)
    return (i0 - ib) / (i0 + ib)


# ----------------------------------------------------------------------------------- regeneration

def slab_matrix(u: float = 1.0, f: float = PH.f_k0, fbar: float = PH.f_k0bar) -> np.ndarray:
    """A thin absorber in the beam in the basis (K0, K0bar): the amplitudes are multiplied by f^u and fbar^u (u = the depth
    in units of the thickness; absorption without a change of the phase: an illustration, not a model of a real nucleus)."""
    return np.diag([f ** u, fbar ** u]).astype(complex)


def regeneration(t_slab: float = PH.slab_time, t_before: float = PH.regen_before, t_after: float = PH.regen_after) -> dict:
    """A K0 beam is born at 0, meets the absorber at t_slab (only K_L is left by then) and goes on.  Returns the amplitudes
    (a0, abar) on a grid before and after the absorber, normalized so that the intensity of the incident beam |psi(t_slab)|^2 = 1."""
    if "regen" in _cache and _cache["regen"]["t_slab"] == t_slab:
        return _cache["regen"]
    h = hamiltonian()
    up = integrate(h, K0, np.array([0.0, t_slab]))[-1]
    norm = float(np.linalg.norm(up))
    up = up / norm
    # the part before the absorber: from the state at t_slab - t_before to t_slab (all numerically)
    t0 = t_slab - t_before
    psi0 = integrate(h, K0, np.array([0.0, t0]))[-1] / norm if t0 > 0 else K0 / norm
    tb = np.arange(t0, t_slab + 0.5 * IN.fine_dt, IN.fine_dt)
    before = integrate(h, psi0, tb)
    after_psi0 = slab_matrix() @ up
    ta = np.arange(0.0, t_after + 0.5 * IN.fine_dt, IN.fine_dt)
    after = integrate(h, after_psi0, ta)
    out = {"t_slab": t_slab, "norm": norm, "psi_in": up, "psi_out": after_psi0, "tb": tb - t_slab, "before": before, "ta": ta, "after": after}
    _cache["regen"] = out
    return out
