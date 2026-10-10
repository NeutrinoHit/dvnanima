r"""Spectral flow of the Dirac sea: the physics behind the film "The chiral anomaly" (the book, the chapter "Quantum anomalies").

Everything the film draws is computed here by counting levels; no anomaly coefficient is typed in.

1+1 dimensions, a massless Dirac fermion on a ring of length L (charge e, hbar = c = 1).
  right movers   E = +k,   left movers   E = -k,   k_n(t) = 2 pi (n + theta) / L + e E t      (kinetic momentum, n = canonical label)
  the vacuum at t = 0: every level with E < 0 is filled (the Dirac sea);
  the occupation numbers belong to the labels n and do not change when the field is switched on (the levels flow, the electrons stay in them).
  N_R = (filled levels above E = 0) - (empty levels below E = 0) for the right branch, the same for the left branch;
  vector charge Q = N_R + N_L,   axial charge Q_5 = N_R - N_L.
The infinite sea has to be regularised: a "window" of levels that are counted.
  * window in the kinetic momentum / energy, -lam_lo <= E <= lam_hi on each branch: gauge invariant,
  * window in the labels n (a fixed set of levels): the cut moves with the gauge potential,
  * a smooth regulator f(E / lam) in the kinetic energy.
A lattice (a crystal, E = sin k) is a regularisation that is finite and exact: the two branches are the two Fermi points of one band.

3+1 dimensions: a Weyl fermion H = chi sigma.(p - e A) in a magnetic field B along z (Landau gauge A_y = B x); the Landau levels are computed by
diagonalising H on a strip (Fourier derivative in x, p_y = 2 pi m / L_y, p_z a parameter) and counted.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


# ------------------------------------------------------------------------------------------------------------- 1+1 ring

@dataclass(frozen=True)
class Ring:
    """A ring of length ``length``; ``theta`` shifts the labels (theta = 1/2: there is no level exactly at E = 0)."""
    length: float
    theta: float
    e: float
    field: float

    @property
    def spacing(self) -> float:
        """The distance between neighbouring levels, 2 pi / L."""
        return 2.0 * math.pi / self.length

    @property
    def level_rate(self) -> float:
        """eEL / 2 pi: the number of levels that pass a given energy per unit time."""
        return self.e * self.field * self.length / (2.0 * math.pi)

    def shift(self, t: float) -> float:
        """e E t: the change of the kinetic momentum of every level."""
        return self.e * self.field * t

    def k0(self, n: np.ndarray) -> np.ndarray:
        return (np.asarray(n) + self.theta) * self.spacing

    def k(self, n: np.ndarray, t: float) -> np.ndarray:
        return self.k0(n) + self.shift(t)

    def energy(self, branch: str, n: np.ndarray, t: float) -> np.ndarray:
        return self.k(n, t) if branch == "R" else -self.k(n, t)

    def occupied(self, branch: str, n: np.ndarray) -> np.ndarray:
        """The vacuum: filled iff the energy at t = 0 is negative (a property of the label, it does not change)."""
        return self.energy(branch, n, 0.0) < 0.0


def labels(nmax: int) -> np.ndarray:
    return np.arange(-nmax, nmax)


def particles_minus_holes(ring: Ring, branch: str, t: float, nmax: int = 400) -> int:
    """N of one branch by direct counting: filled levels above E = 0 minus empty levels below E = 0."""
    n = labels(nmax)
    e_, occ = ring.energy(branch, n, t), ring.occupied(branch, n)
    return int(np.sum(occ & (e_ > 0.0)) - np.sum(~occ & (e_ < 0.0)))


def charges(ring: Ring, t: float, nmax: int = 400) -> dict:
    nr, nl = particles_minus_holes(ring, "R", t, nmax), particles_minus_holes(ring, "L", t, nmax)
    return {"NR": nr, "NL": nl, "Q": nr + nl, "Q5": nr - nl}


def crossing_times(ring: Ring, t_max: float, nmax: int = 400) -> np.ndarray:
    """The times at which the levels pass E = 0 (k_n(t) = 0; the same labels cross in both branches, the right one upwards, the left one downwards)."""
    n = labels(nmax)
    t_cross = -ring.k0(n) / (ring.e * ring.field)                 # d k / dt = e E
    return np.sort(t_cross[(t_cross > 0.0) & (t_cross <= t_max)])


def window_filled(ring: Ring, t: float, kind: str, lam_lo: float, lam_hi: float | None = None, nmax: int = 400) -> dict:
    """The numbers of FILLED levels counted by a window (absolute numbers, the vacuum is not subtracted).

    kind = "energy": the levels with -lam_lo <= E <= lam_hi at the time t (gauge invariant);
    kind = "label": the levels whose energies were inside the same window at t = 0 (a fixed set of levels)."""
    lam_hi = lam_lo if lam_hi is None else lam_hi
    n = labels(nmax)
    out = {}
    for br in ("R", "L"):
        occ = ring.occupied(br, n)
        e_t, e_0 = ring.energy(br, n, t), ring.energy(br, n, 0.0)
        sel = (e_t >= -lam_lo) & (e_t <= lam_hi) if kind == "energy" else (e_0 >= -lam_lo) & (e_0 <= lam_hi)
        out[br] = int(np.sum(occ & sel))
    return out


def window_counts(ring: Ring, t: float, kind: str, lam_lo: float, lam_hi: float | None = None, nmax: int = 400) -> dict:
    """The change of the filled levels of each branch that a window counts, relative to t = 0 (N_R, N_L, Q, Q_5 of the regularised theory)."""
    now, start = window_filled(ring, t, kind, lam_lo, lam_hi, nmax), window_filled(ring, 0.0, kind, lam_lo, lam_hi, nmax)
    out = {"R": now["R"] - start["R"], "L": now["L"] - start["L"]}
    out["Q"], out["Q5"] = out["R"] + out["L"], out["R"] - out["L"]
    return out


def smooth_charges(ring: Ring, t: float, lam: float, nmax: int = 4000) -> dict:
    """Regularised charges with a smooth weight exp(-(E/lam)^2) in the kinetic energy: sum over the levels, with the vacuum subtracted."""
    n = labels(nmax)
    out = {}
    for br in ("R", "L"):
        occ = ring.occupied(br, n)
        w_t = np.exp(-((ring.energy(br, n, t) / lam) ** 2))
        w_0 = np.exp(-((ring.energy(br, n, 0.0) / lam) ** 2))
        out[br] = float(np.sum(w_t[occ]) - np.sum(w_0[occ]))
    out["Q"], out["Q5"] = out["R"] + out["L"], out["R"] - out["L"]
    return out


def vacuum_current(ring: Ring, shift: float, kind: str, lam: float, nmax: int = 400) -> int:
    """The ground state of the ring with a constant gauge potential (the levels are shifted by ``shift``, no electric field): (filled R) - (filled L).

    For a gauge-invariant cut the answer is bounded (a persistent current of at most one level); for a cut in fixed labels it grows with the potential."""
    n = labels(nmax)
    k = ring.k0(n) + shift
    k_ref = ring.k0(n)
    if kind == "energy":
        sel = np.abs(k) <= lam
    else:
        sel = np.abs(k_ref) <= lam
    filled_r = np.sum(sel & (k < 0.0))                # right branch: E = k < 0
    filled_l = np.sum(sel & (k > 0.0))                # left branch: E = -k < 0
    return int(filled_r - filled_l)


# ------------------------------------------------------------------------------------------------------------- the lattice

def lattice_levels(n_sites: int, theta: float, shift_levels: float) -> np.ndarray:
    """The wave numbers k = 2 pi (n + theta) / N + shift (in units of 1/a) folded into (-pi, pi], n = 0..N-1."""
    k = 2.0 * math.pi * (np.arange(n_sites) + theta) / n_sites + shift_levels * 2.0 * math.pi / n_sites
    return (k + math.pi) % (2.0 * math.pi) - math.pi


def lattice_counts(n_sites: int, theta: float, shift_levels: float) -> dict:
    """A tight-binding chain, E = sin k.  Right movers are at cos k > 0 (the point k = 0), left movers at cos k < 0 (the point k = pi).
    The occupation belongs to the label (filled iff E < 0 at the start); returns N_R, N_L and the total number of filled levels."""
    k0 = lattice_levels(n_sites, theta, 0.0)
    k = lattice_levels(n_sites, theta, shift_levels)
    filled = np.sin(k0) < 0.0
    e_ = np.sin(k)
    right = np.cos(k) > 0.0
    tol = 1e-12                                               # a level exactly at E = 0 is counted neither as a particle nor as a hole

    def count(region: np.ndarray) -> int:
        return int(np.sum(filled & (e_ > tol) & region) - np.sum(~filled & (e_ < -tol) & region))
    nr, nl = count(right), count(~right)
    return {"NR": nr, "NL": nl, "Q": nr + nl, "Q5": nr - nl, "filled": int(np.sum(filled))}


# ------------------------------------------------------------------------------------------------------------- Landau levels

def fourier_derivative_matrix(n: int, dx: float) -> np.ndarray:
    """The spectral derivative on a periodic grid of n (odd) points."""
    k = 2.0 * math.pi * np.fft.fftfreq(n, d=dx)
    return np.fft.ifft(1j * k[:, None] * np.fft.fft(np.eye(n), axis=0), axis=0)


def weyl_strip_hamiltonian(eb: float, p_y: float, p_z: float, x: np.ndarray, chirality: int = +1) -> np.ndarray:
    """chi * sigma.pi + (the p_z term) on a strip, pi = p - e A, A_y = B x, p_y a number:  H = chi [[p_z, pi_x - i pi_y], [pi_x + i pi_y, -p_z]]."""
    n = len(x)
    dx = float(x[1] - x[0])
    px = -1j * fourier_derivative_matrix(n, dx)
    py = np.diag(p_y - eb * x).astype(complex)
    h = np.zeros((2 * n, 2 * n), complex)
    h[:n, :n] = p_z * np.eye(n)
    h[n:, n:] = -p_z * np.eye(n)
    h[:n, n:] = px - 1j * py
    h[n:, :n] = px + 1j * py
    return chirality * 0.5 * (h + h.conj().T)


def landau_spectrum(eb: float, length_x: float, length_y: float, n_x: int, p_z: float, m_range: range, chirality: int = +1) -> list[dict]:
    """For every p_y = 2 pi m / L_y: the eigenvalues, the centres <x> and the weight of the lower spinor component."""
    x = (np.arange(n_x) - n_x // 2) * (length_x / n_x)
    out = []
    for m in m_range:
        p_y = 2.0 * math.pi * m / length_y
        ev, vec = np.linalg.eigh(weyl_strip_hamiltonian(eb, p_y, p_z, x, chirality))
        dens = np.abs(vec[:n_x]) ** 2 + np.abs(vec[n_x:]) ** 2
        phase = (dens * np.exp(2j * math.pi * x / length_x)[:, None]).sum(axis=0)        # the centre on the periodic box (a circular mean)
        out.append({"m": m, "p_y": p_y, "centre_expected": p_y / eb, "E": ev, "x": np.angle(phase) * length_x / (2.0 * math.pi)})
    return out


def landau_branch_count(spec: list[dict], energy: float, x_lo: float, x_hi: float, tol: float = 1e-6) -> int:
    """How many states have the energy ``energy`` (within tol) and a centre in [x_lo, x_hi).  States at the seam of the periodic box are not in the interior."""
    c = 0
    for s in spec:
        c += int(np.sum((np.abs(s["E"] - energy) < tol) & (s["x"] >= x_lo - 1e-9) & (s["x"] < x_hi - 1e-9)))
    return c


def landau_energies(eb: float, p_z: float, n_max: int) -> list[float]:
    """The known levels: E_n = +-sqrt(p_z^2 + 2 n eB) (n >= 1) and the single branch +p_z (n = 0, chirality +)."""
    out = [p_z]
    for n in range(1, n_max + 1):
        out += [math.sqrt(p_z ** 2 + 2.0 * n * eb), -math.sqrt(p_z ** 2 + 2.0 * n * eb)]
    return sorted(out)


# ------------------------------------------------------------------------------------------------------------- charges and the pion

def isospin_anomaly_weight(charges_: list[float], t3: list[float], n_colors: int) -> float:
    """N_c * sum_f t3_f Q_f^2 (the weight of the anomaly of the neutral isospin current relative to a single fermion of charge e)."""
    return n_colors * sum(t * q * q for t, q in zip(t3, charges_))


def pi0_amplitude(alpha: float, f_pi: float) -> float:
    return alpha / (math.pi * f_pi)


def pi0_width(alpha: float, m_pi: float, f_pi: float) -> float:
    """Gamma = A^2 m^3 / (64 pi) with A = alpha / (pi f)  =  alpha^2 m^3 / (64 pi^3 f^2)."""
    a = pi0_amplitude(alpha, f_pi)
    return a * a * m_pi ** 3 / (64.0 * math.pi)
