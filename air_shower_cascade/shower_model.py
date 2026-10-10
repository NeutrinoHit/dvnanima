r"""Heitler-Matthews toy model of an extensive air shower, as a Monte-Carlo cascade (no numbers of its own: see config.toml).

The model (Heitler 1944 for the electromagnetic part, Matthews 2005 for the hadronic part):

 * depth is measured along the axis in g/cm^2 (a vertical shower in an isothermal atmosphere, X = X_g exp(-h/H));
 * e+-, gamma (electromagnetic particles): after the depth d = X_0 ln 2 (X_0 is the radiation length, the energy of an electron
   falls as exp(-X/X_0): one halving per X_0 ln 2), exactly (Heitler's original model, ``path = "fixed"``) or with an exponential
   distribution of the mean d (the Monte-Carlo version, ``path = "exponential"``), every particle splits in two with half the
   energy each:
        e -> e + gamma (bremsstrahlung),      gamma -> e+ + e-  (pair production).
   Below the critical energy E_c ionisation wins: the particle does not multiply, it loses its energy at the constant rate
   E_c / X_0 and stops after X_0 E / E_c.
 * hadrons (proton, charged pions): after the depth lambda_I (the mean free path; exponential or fixed, as above) an interaction makes n_ch charged pions and n_ch/2
   neutral pions with equal energies E / (n_ch + n_ch/2) (2/3 of the secondaries are charged), the transverse momenta are
   Rayleigh distributed (sum zero).  A neutral pion decays at once into two photons (the electromagnetic cascade starts),
   a charged pion of energy E <= E_dec decays into a muon and a neutrino instead of interacting (flat energy
   distribution E_mu / E_pi in [m_mu^2 / m_pi^2, 1], exact for an isotropic decay of a scalar).
 * muons and neutrinos go straight to the ground (the muon decay and its ionisation loss are not included).
 * lateral coordinates (metres, two components) follow from the directions: a hadron keeps its direction plus the transverse
   momentum kick, an electron is deflected by multiple scattering (space angle <theta^2> = (E_s / E)^2 s / X_0, E_s = 21.2 MeV: two Gaussian components), a photon goes
   straight; lateral drift over a depth interval [X1, X2] is u H ln(X2 / X1) (isothermal atmosphere).  This is schematic:
   a straight path is assumed to be vertical.

Counts are exact for the model (every particle is followed) except that below the thinning energy E_thin only one of the two
daughters of a split is followed, with a doubled statistical weight (Hillas thinning, unbiased); E_thin = inf switches it off.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace

import numpy as np

GAMMA, ELECTRON, PION, PROTON, MUON, NEUTRINO = range(6)
KIND_COUNT = 6
TOL = 1e-9            # relative tolerance of the "E > E_c" test (a particle with exactly E_c does not split)
X_FLOOR = 1e-6        # g/cm^2: lower limit of the depth in the logarithm of the drift (the primary starts at X = 0)


@dataclass(frozen=True)
class Params:
    e0_ev: float
    x0: float                 # radiation length, g/cm^2
    e_c_ev: float             # critical energy of e+-
    lam_i: float              # hadronic interaction length, g/cm^2
    n_ch: int                 # charged pions per hadronic interaction (even); the neutral ones are n_ch / 2
    e_dec_ev: float           # a charged pion of E <= E_dec decays
    lam_dec: float            # mean decay depth of a pion of E = E_dec, g/cm^2 (proportional to E)
    x_ground: float           # g/cm^2
    h_km: float               # scale height of the atmosphere
    pt_ev: float              # mean transverse momentum of a secondary hadron
    e_ms_ev: float            # multiple scattering constant E_s (21.2 MeV)
    m_pion_ev: float
    m_muon_ev: float
    e_thin_ev: float          # thinning energy (inf: no thinning)
    path: str = "exponential"  # "exponential": random depth with the mean lambda; "fixed": exactly the mean
    x_sea: float = 1030.0      # depth of sea level, g/cm^2 (altitudes are measured from it)

    @property
    def n_pi0(self) -> int:
        return self.n_ch // 2

    @property
    def n_sec(self) -> int:
        return self.n_ch + self.n_ch // 2

    @property
    def muon_fraction_min(self) -> float:
        return (self.m_muon_ev / self.m_pion_ev) ** 2

    @property
    def h_m(self) -> float:
        return self.h_km * 1000.0

    def with_(self, **kw) -> "Params":
        return replace(self, **kw)


# ------------------------------------------------------------------------------ analytic model

def heitler_n_max(e0: float, e_c: float) -> float:
    """Heitler: the number of particles at the shower maximum, N_max = E_0 / E_c."""
    return e0 / e_c


def heitler_x_max(x0: float, e0: float, e_c: float) -> float:
    """Heitler: n_c = ln(E_0/E_c)/ln 2 doublings, each over X_0 ln 2: X_max = X_0 ln(E_0/E_c)."""
    return x0 * math.log(e0 / e_c)


def matthews_beta(n_ch: int) -> float:
    return math.log(n_ch) / math.log(n_ch + n_ch // 2)


def matthews_n_mu(e0: float, e_dec: float, n_ch: int) -> float:
    """Matthews: N_mu = (E_0 / E_dec)^beta,  beta = ln n_ch / ln(1.5 n_ch)."""
    return (e0 / e_dec) ** matthews_beta(n_ch)


def hadron_levels(e0: float, e_dec: float, n_ch: int) -> int:
    """The number m of interaction levels after which the charged pions have E <= E_dec: E_0 / (1.5 n_ch)^m <= E_dec."""
    m = 0
    e = e0
    while e > e_dec * (1.0 + TOL):
        e /= n_ch + n_ch // 2
        m += 1
    return m


def matthews_x_max_proton(p: Params, x_first: float) -> float:
    """X_max of the proton shower: the first-generation pi0 (E_0 / n_sec each, gamma E_0 / (2 n_sec)) start at the first
    interaction depth: X_max = X_1 + X_0 ln(E_0 / (2 n_sec E_c))."""
    return x_first + p.x0 * math.log(p.e0_ev / (2 * p.n_sec * p.e_c_ev))


def electromagnetic_fraction(p: Params) -> float:
    """1 - (2/3)^m: the part of E_0 that ends in the electromagnetic cascade (the rest goes to the muons and the neutrinos)."""
    m = hadron_levels(p.e0_ev, p.e_dec_ev, p.n_ch)
    return 1.0 - (p.n_ch / p.n_sec) ** m


def moliere_radius_m(p: Params, depth: float) -> float:
    """r_M = (E_ms / E_c) X_0 / rho at the depth X, rho = X / H (isothermal atmosphere), in metres."""
    rho = depth / (p.h_m * 100.0)                      # g/cm^3 (H in cm)
    return p.e_ms_ev / p.e_c_ev * p.x0 / rho / 100.0


def nkg_density(r: np.ndarray, n_e: float, age: float, r_m: float) -> np.ndarray:
    """Nishimura-Kamata-Greisen lateral density of the electrons (per m^2) for the size n_e, the age s and the Moliere radius r_m."""
    from scipy.special import gamma
    c = gamma(4.5 - age) / (2.0 * math.pi * gamma(age) * gamma(4.5 - 2.0 * age))
    return n_e * c / r_m ** 2 * (r / r_m) ** (age - 2.0) * (1.0 + r / r_m) ** (age - 4.5)


def shower_age(depth: float, x_max: float) -> float:
    return 3.0 * depth / (depth + 2.0 * x_max)


def cherenkov_threshold_ev(n_minus_one: np.ndarray, m_e_ev: float) -> np.ndarray:
    """Total energy of an electron at the Cherenkov threshold gamma m_e, 1 / gamma^2 = 1 - 1 / n^2."""
    n = 1.0 + n_minus_one
    return m_e_ev / np.sqrt(1.0 - 1.0 / n ** 2)


def cherenkov_angle(n_minus_one: np.ndarray) -> np.ndarray:
    return np.arccos(1.0 / (1.0 + n_minus_one))


def altitude_km(p: Params, depth: float | np.ndarray) -> float | np.ndarray:
    return p.h_km * np.log(p.x_sea / np.maximum(depth, X_FLOOR))


def depth_of_altitude(p: Params, h_km: float | np.ndarray) -> float | np.ndarray:
    return p.x_sea * np.exp(-np.asarray(h_km) / p.h_km)


def drift_m(p: Params, x_b: np.ndarray, x_d: np.ndarray) -> np.ndarray:
    """Vertical length (metres) of the path between the depths x_b and x_d: H ln(x_d / x_b)."""
    return p.h_m * np.log(x_d / np.maximum(x_b, X_FLOOR))


def heitler_generations(e: float, e_c: float) -> int:
    """n: the number of splittings of a photon of energy e until the particles have E <= E_c (a particle with E = E_c does not split)."""
    n = 0
    while e > e_c * (1.0 + TOL):
        e /= 2.0
        n += 1
    return n


def heitler_profile(x: np.ndarray, e: float, d: float, x0: float, e_c: float) -> np.ndarray:
    """Heitler's electromagnetic cascade of one photon (energy e) entering at X = 0, with the fixed step d = X_0 ln 2: the number
    of particles N(X) = 2^g for g d <= X < (g + 1) d (g < n); the n-th generation (E_n = e / 2^n <= E_c) ionises over X_0 E_n / E_c."""
    x = np.asarray(x, float)
    n = heitler_generations(e, e_c)
    g = np.floor(np.maximum(x, 0.0) / d)
    out = np.where(g < n, 2.0 ** g, 0.0)
    last = (x >= n * d) & (x < n * d + x0 * (e / 2.0 ** n) / e_c)
    out = np.where(last, 2.0 ** n, out)
    return np.where(x < 0.0, 0.0, out)


def heitler_matthews_profile(p: Params, x: np.ndarray) -> dict[str, np.ndarray]:
    """The deterministic (fixed-step) Heitler-Matthews proton shower: level j interactions at X_j = j lambda_I produce
    n_pi0 n_ch^(j-1) neutral pions each of E_0 / n_sec^j, i.e. two photons of E_0 / (2 n_sec^j) that start electromagnetic
    cascades; the charged pions of level m decay (mean decay depth lambda_dec E_m / E_dec) into muons.
    Returns the numbers of e+- plus gamma ('em'), hadrons ('had') and muons ('mu') at the depths x."""
    x = np.asarray(x, float)
    m = hadron_levels(p.e0_ev, p.e_dec_ev, p.n_ch)
    d = p.x0 * math.log(2.0)
    em = np.zeros_like(x)
    for j in range(1, m + 1):
        count = 2 * p.n_pi0 * p.n_ch ** (j - 1)
        e_ph = p.e0_ev / (2.0 * p.n_sec ** j)
        em += count * heitler_profile(x - j * p.lam_i, e_ph, d, p.x0, p.e_c_ev)
    had = np.where(x < p.lam_i, 1.0, 0.0)
    for j in range(1, m):
        had += np.where((x >= j * p.lam_i) & (x < (j + 1) * p.lam_i), float(p.n_ch ** j), 0.0)
    e_m = p.e0_ev / p.n_sec ** m
    x_dec = m * p.lam_i + p.lam_dec * e_m / p.e_dec_ev
    had += np.where((x >= m * p.lam_i) & (x < x_dec), float(p.n_ch ** m), 0.0)
    mu = np.where(x >= x_dec, float(p.n_ch ** m), 0.0)
    return dict(em=em, had=had, mu=mu)


# ------------------------------------------------------------------------------ tracks

class Tracks:
    """Append-only table of straight track segments (structure of arrays)."""

    FIELDS = ("x0", "x1", "px0", "py0", "px1", "py1", "kind", "e", "w", "parent", "level", "ground")

    def __init__(self) -> None:
        self._parts: dict[str, list] = {f: [] for f in self.FIELDS}
        self.n = 0

    def add(self, **cols) -> np.ndarray:
        n = len(cols["x0"])
        for f in self.FIELDS:
            self._parts[f].append(np.broadcast_to(np.asarray(cols[f]), (n,)).copy())
        idx = np.arange(self.n, self.n + n)
        self.n += n
        return idx

    def finish(self) -> dict[str, np.ndarray]:
        return {f: (np.concatenate(v) if v else np.zeros(0)) for f, v in self._parts.items()}


def _path(rng: np.random.Generator, p: Params, mean: np.ndarray) -> np.ndarray:
    """Depth to the next interaction: exponential with the given mean (``path = "exponential"``) or exactly the mean (``"fixed"``)."""
    if p.path == "fixed":
        return mean
    return rng.exponential(mean)


# ------------------------------------------------------------------------------ hadronic cascade

def hadron_cascade(p: Params, rng: np.random.Generator, tab: Tracks) -> dict[str, np.ndarray]:
    """The primary proton and the charged pions level by level; returns the photon seeds (from the pi0 decays) of the
    electromagnetic cascade.  The tracks of the hadrons, the muons and the neutrinos are appended to ``tab``."""
    n_sec, n_pi0 = p.n_sec, p.n_pi0
    pop = dict(x=np.zeros(1), e=np.array([p.e0_ev]), pos=np.zeros((1, 2)), u=np.zeros((1, 2)),
               kind=np.array([PROTON]), parent=np.array([-1]), level=np.array([0]))
    seeds: list[dict[str, np.ndarray]] = []
    sigma_c = p.pt_ev * math.sqrt(2.0 / math.pi)         # Rayleigh with the mean pt: sigma of each component
    r_mu = p.muon_fraction_min
    while len(pop["x"]):
        x, e, pos, u, kind, parent, level = (pop[k] for k in ("x", "e", "pos", "u", "kind", "parent", "level"))
        n = len(x)
        decays = (kind == PION) & (e <= p.e_dec_ev * (1.0 + TOL))
        mean = np.where(decays, p.lam_dec * e / p.e_dec_ev, p.lam_i)
        xd = x + _path(rng, p, mean)
        ground = xd >= p.x_ground
        xd = np.minimum(xd, p.x_ground)
        pos_d = pos + u * drift_m(p, x, xd)[:, None]
        idx = tab.add(x0=x, x1=xd, px0=pos[:, 0], py0=pos[:, 1], px1=pos_d[:, 0], py1=pos_d[:, 1], kind=kind, e=e,
                      w=np.ones(n), parent=parent, level=level, ground=ground)
        # --- decays: pi -> mu nu
        dsel = decays & ~ground
        if dsel.any():
            m = int(dsel.sum())
            f = r_mu + (1.0 - r_mu) * rng.random(m)
            # decay kinematics: cos(theta*) = (2 f - 1 - r) / (1 - r); the transverse momentum p* sin(theta*) is shared (opposite signs)
            p_star = (p.m_pion_ev ** 2 - p.m_muon_ev ** 2) / (2.0 * p.m_pion_ev)
            cos_t = np.clip((2.0 * f - 1.0 - r_mu) / (1.0 - r_mu), -1.0, 1.0)
            phi = 2.0 * math.pi * rng.random(m)
            pt_dir = np.column_stack([np.cos(phi), np.sin(phi)]) * (p_star * np.sqrt(1.0 - cos_t ** 2))[:, None]
            for k_, share, sign in ((MUON, f, 1.0), (NEUTRINO, 1.0 - f, -1.0)):
                e_k = e[dsel] * share
                u_k = u[dsel] + sign * pt_dir / e_k[:, None]
                pg = pos_d[dsel] + u_k * drift_m(p, xd[dsel], np.full(m, p.x_ground))[:, None]
                tab.add(x0=xd[dsel], x1=np.full(m, p.x_ground), px0=pos_d[dsel, 0], py0=pos_d[dsel, 1], px1=pg[:, 0], py1=pg[:, 1],
                        kind=k_, e=e_k, w=np.ones(m), parent=idx[dsel], level=level[dsel], ground=np.ones(m, bool))
        # --- interactions: n_ch pi+- and n_ch/2 pi0, equal energies, transverse momenta of sum zero
        isel = ~decays & ~ground
        if not isel.any():
            break
        m = int(isel.sum())
        e_sec = e[isel] / n_sec
        dp = rng.normal(0.0, sigma_c, size=(m, n_sec, 2))
        dp -= dp.mean(axis=1, keepdims=True)
        u_sec = u[isel][:, None, :] + dp / e_sec[:, None, None]
        xs, ps, ids = xd[isel], pos_d[isel], idx[isel]
        lev = level[isel] + 1
        # the photons of the pi0 decays: two with E/2 each, along the pi0 direction
        u_pi0 = np.repeat(u_sec[:, p.n_ch:, :].reshape(-1, 2), 2, axis=0)
        k_pi0 = 2 * n_pi0
        seeds.append(dict(x=np.repeat(xs, k_pi0), e=np.repeat(e_sec / 2.0, k_pi0), pos=np.repeat(ps, k_pi0, axis=0), u=u_pi0,
                          parent=np.repeat(ids, k_pi0), level=np.repeat(lev, k_pi0)))
        pop = dict(x=np.repeat(xs, p.n_ch), e=np.repeat(e_sec, p.n_ch), pos=np.repeat(ps, p.n_ch, axis=0),
                   u=u_sec[:, :p.n_ch, :].reshape(-1, 2), kind=np.full(m * p.n_ch, PION), parent=np.repeat(ids, p.n_ch),
                   level=np.repeat(lev, p.n_ch))
    if not seeds:
        return dict(x=np.zeros(0), e=np.zeros(0), pos=np.zeros((0, 2)), u=np.zeros((0, 2)), parent=np.zeros(0, int), level=np.zeros(0, int))
    return {k: np.concatenate([s[k] for s in seeds]) for k in seeds[0]}


# ------------------------------------------------------------------------------ electromagnetic cascade

def em_cascade(p: Params, rng: np.random.Generator, seeds: dict[str, np.ndarray], tab: Tracks, thin: bool = True,
               keep_second: np.ndarray | None = None, kind: int = GAMMA) -> None:
    """Follow the electromagnetic particles generation by generation (all the seeds together).

    ``thin``: below E_thin only one of the two daughters of a split is followed (weight doubled).
    ``keep_second``: optional per-seed probability to follow the second daughter of a split (for the drawn sample; the
    probability is inherited by the daughters); the first daughter (the electron of a bremsstrahlung, one of the two e+-
    of a pair) is always followed."""
    n0 = len(seeds["x"])
    if n0 == 0:
        return
    cur = dict(x=seeds["x"], e=seeds["e"], pos=seeds["pos"], u=seeds["u"], parent=seeds["parent"], level=seeds["level"],
               kind=np.full(n0, kind), w=seeds.get("w", np.ones(n0)),
               q=(keep_second if keep_second is not None else np.ones(n0)))
    while len(cur["x"]):
        x, e, pos, u, parent, level, kd, w, q = (cur[k] for k in ("x", "e", "pos", "u", "parent", "level", "kind", "w", "q"))
        n = len(x)
        split = e > p.e_c_ev * (1.0 + TOL)
        rng_len = p.x0 * e / p.e_c_ev                                        # ionisation range of a particle below E_c
        s = np.where(split, _path(rng, p, np.full(n, p.x0 * math.log(2.0))), rng_len)
        xd = x + s
        ground = xd >= p.x_ground
        xd = np.minimum(xd, p.x_ground)
        s = xd - x
        # multiple scattering of the electrons over the step
        sig = np.where(kd == ELECTRON, p.e_ms_ev / (math.sqrt(2.0) * e) * np.sqrt(np.maximum(s, 0.0) / p.x0), 0.0)
        kick = rng.normal(size=(n, 2)) * sig[:, None]
        pos_d = pos + (u + 0.5 * kick) * drift_m(p, x, xd)[:, None]
        u_end = u + kick
        idx = tab.add(x0=x, x1=xd, px0=pos[:, 0], py0=pos[:, 1], px1=pos_d[:, 0], py1=pos_d[:, 1], kind=kd, e=e, w=w,
                      parent=parent, level=level, ground=ground)
        sp = split & ~ground
        m = int(sp.sum())
        if m == 0:
            break
        # daughters: e -> (e, gamma), gamma -> (e, e), each with E/2
        k1 = np.full(m, ELECTRON)
        k2 = np.where(kd[sp] == ELECTRON, GAMMA, ELECTRON)
        if keep_second is not None:                       # which of the two daughters is always followed is random
            swap = (kd[sp] == ELECTRON) & (rng.random(m) < 0.5)
            k1 = np.where(swap, GAMMA, ELECTRON)
            k2 = np.where(swap, ELECTRON, k2)
        base = dict(x=xd[sp], e=e[sp] / 2.0, pos=pos_d[sp], u=u_end[sp], parent=idx[sp], level=level[sp])
        w_p, q_p = w[sp], q[sp]
        keep1 = np.ones(m, bool)
        keep2 = np.ones(m, bool)
        w1, w2 = w_p.copy(), w_p.copy()
        if keep_second is not None:
            keep2 = rng.random(m) < q_p
        if thin:
            tsel = e[sp] < p.e_thin_ev
            pick = rng.random(m) < 0.5
            keep1 &= ~tsel | pick
            keep2 &= ~tsel | ~pick
            w1 = np.where(tsel, 2.0 * w_p, w_p)
            w2 = np.where(tsel, 2.0 * w_p, w_p)

        def part(sel: np.ndarray, kk: np.ndarray, ww: np.ndarray) -> dict[str, np.ndarray]:
            d = {k: v[sel] for k, v in base.items()}
            d.update(kind=kk[sel], w=ww[sel], q=q_p[sel])
            return d

        a, b = part(keep1, k1, w1), part(keep2, k2, w2)
        cur = {k: np.concatenate([a[k], b[k]]) for k in a}


# ------------------------------------------------------------------------------ the shower

@dataclass
class Shower:
    p: Params
    tr: dict[str, np.ndarray]
    seeds: dict[str, np.ndarray]
    dx: float
    grid: np.ndarray

    # ---- counts
    def count(self, kinds, grid: np.ndarray | None = None) -> np.ndarray:
        """Number of particles of the given kinds alive at each depth of the grid (weighted)."""
        g = self.grid if grid is None else grid
        tr = self.tr
        mask = np.isin(tr["kind"], list(kinds))
        return count_profile(tr["x0"][mask], tr["x1"][mask], tr["w"][mask], tr["ground"][mask], g)

    def n_ground(self, kinds) -> float:
        tr = self.tr
        mask = np.isin(tr["kind"], list(kinds)) & tr["ground"].astype(bool)
        return float(tr["w"][mask].sum())

    def e_ground(self, kinds) -> float:
        tr = self.tr
        mask = np.isin(tr["kind"], list(kinds)) & tr["ground"].astype(bool)
        return float((tr["w"][mask] * tr["e"][mask]).sum())

    def ionising(self) -> np.ndarray:
        tr = self.tr
        return np.isin(tr["kind"], (GAMMA, ELECTRON)) & (tr["e"] <= self.p.e_c_ev * (1.0 + TOL))

    def deposit_profile(self) -> np.ndarray:
        """dE/dX (eV per g/cm^2) on the grid: every ionising particle deposits E_c / X_0 per g/cm^2."""
        m = self.ionising()
        tr = self.tr
        return self.p.e_c_ev / self.p.x0 * count_profile(tr["x0"][m], tr["x1"][m], tr["w"][m], np.zeros(int(m.sum()), bool), self.grid)

    def deposited_energy(self) -> float:
        m = self.ionising()
        tr = self.tr
        return float(self.p.e_c_ev / self.p.x0 * (tr["w"][m] * (tr["x1"][m] - tr["x0"][m])).sum())

    def em_ground_energy(self) -> float:
        """Energy that the electromagnetic particles bring to the ground (not yet deposited)."""
        tr = self.tr
        m = np.isin(tr["kind"], (GAMMA, ELECTRON)) & tr["ground"].astype(bool)
        lost = np.where(self.ionising()[m], self.p.e_c_ev / self.p.x0 * (tr["x1"][m] - tr["x0"][m]), 0.0)
        return float((tr["w"][m] * (tr["e"][m] - lost)).sum())

    def x_first(self) -> float:
        tr = self.tr
        return float(tr["x1"][tr["kind"] == PROTON][0])

    def peak(self, smooth_bins: int = 3) -> tuple[float, float]:
        """(X_max, N_max) of the electromagnetic profile e+- plus gamma (a running mean over ``smooth_bins`` grid points)."""
        n = self.count((GAMMA, ELECTRON))
        k = np.ones(smooth_bins) / smooth_bins
        ns = np.convolve(np.pad(n, smooth_bins, mode="edge"), k, mode="same")[smooth_bins:-smooth_bins]
        i = int(np.argmax(ns))
        return float(self.grid[i]), float(ns[i])

    def radial_density(self, kinds, r_edges: np.ndarray) -> np.ndarray:
        """Particles per m^2 at the ground in the rings [r_edges[i], r_edges[i+1]) (azimuthal average)."""
        tr = self.tr
        mask = np.isin(tr["kind"], list(kinds)) & tr["ground"].astype(bool)
        r = np.hypot(tr["px1"][mask], tr["py1"][mask])
        h, _ = np.histogram(r, bins=r_edges, weights=tr["w"][mask])
        area = math.pi * (r_edges[1:] ** 2 - r_edges[:-1] ** 2)
        return h / area


def count_profile(x0: np.ndarray, x1: np.ndarray, w: np.ndarray, ground: np.ndarray, grid: np.ndarray) -> np.ndarray:
    """sum of w over the tracks with x0 <= X < x1 at every grid depth X (a track that reaches the ground is alive at the last grid point)."""
    dx = grid[1] - grid[0]
    n = len(grid)
    i0 = np.clip(np.ceil(x0 / dx - TOL).astype(int), 0, n)
    i1 = np.clip(np.ceil(x1 / dx - TOL).astype(int), 0, n)
    i1 = np.where(ground.astype(bool), n, i1)
    arr = np.bincount(i0, weights=w, minlength=n + 1) - np.bincount(i1, weights=w, minlength=n + 1)
    return np.cumsum(arr)[:n]


def simulate(p: Params, seed: int, dx: float, thin: bool = True) -> Shower:
    """The whole cascade of a vertical primary proton with E_0 (every random number comes from ``seed``)."""
    rng = np.random.default_rng(seed)
    tab = Tracks()
    seeds = hadron_cascade(p, rng, tab)
    em_cascade(p, rng, dict(seeds), tab, thin=thin)
    grid = np.arange(0.0, p.x_ground + 0.5 * dx, dx)
    return Shower(p, tab.finish(), seeds, dx, grid)


def simulate_photon(p: Params, seed: int, dx: float, e0_ev: float | None = None, thin: bool = False) -> Shower:
    """A single photon of energy E_0 entering the atmosphere at the depth X = 0 (the pure electromagnetic cascade)."""
    rng = np.random.default_rng(seed)
    tab = Tracks()
    e0 = p.e0_ev if e0_ev is None else e0_ev
    seeds = dict(x=np.zeros(1), e=np.array([e0]), pos=np.zeros((1, 2)), u=np.zeros((1, 2)), parent=np.array([-1]), level=np.zeros(1, int))
    em_cascade(p, rng, seeds, tab, thin=thin)
    grid = np.arange(0.0, p.x_ground + 0.5 * dx, dx)
    return Shower(p, tab.finish(), seeds, dx, grid)


# ------------------------------------------------------------------------------ the drawn sample

@dataclass(frozen=True)
class DrawSpec:
    leaf_pions: int                      # decaying pions of the last level that are drawn (with all their ancestors)
    seed_fraction: tuple[float, ...]     # share of the photon seeds drawn per interaction level (1, 2, ...; the last is repeated)
    tree_tracks: tuple[float, ...]       # target number of tracks of the electromagnetic tree of a seed per interaction level
    seed: int


def _per_level(values: tuple[float, ...], level: np.ndarray) -> np.ndarray:
    v = np.asarray(values, float)
    return v[np.clip(level - 1, 0, len(v) - 1)]


def tree_keep_probability(target: np.ndarray, n_gen: np.ndarray, iterations: int = 40) -> np.ndarray:
    """q such that a tree with the second daughter followed with the probability q has on average ``target`` tracks in n_gen
    generations: sum_{g=0}^{n} (1 + q)^g = ((1 + q)^(n + 1) - 1) / q = target (bisection on q in [0, 1])."""
    n = np.ceil(n_gen)
    lo, hi = np.zeros_like(target, dtype=float), np.ones_like(target, dtype=float)
    for _ in range(iterations):
        q = 0.5 * (lo + hi)
        total = np.where(q > 0, ((1.0 + q) ** (n + 1.0) - 1.0) / np.maximum(q, 1e-12), n + 1.0)
        too_big = total > target
        hi = np.where(too_big, q, hi)
        lo = np.where(too_big, lo, q)
    return 0.5 * (lo + hi)


def draw_sample(sh: Shower, spec: DrawSpec) -> dict[str, np.ndarray]:
    """A random sample of the tracks of ``sh`` for the picture: the primary, the ancestors of ``leaf_pions`` decaying pions
    (with their muons and neutrinos) and pruned electromagnetic trees started by a share of the photon seeds of the drawn
    interactions.  The electromagnetic trees are re-simulated with the same rules (second daughter followed with a probability
    that gives the target size), they are a sample of the same cascade, not the thinned run itself."""
    p, tr = sh.p, sh.tr
    rng = np.random.default_rng(spec.seed)
    kind, parent = tr["kind"], tr["parent"]
    hadron = np.isin(kind, (PROTON, PION))
    decaying_leaf = np.flatnonzero((kind == PION) & (tr["e"] <= p.e_dec_ev * (1.0 + TOL)))
    pick = rng.choice(decaying_leaf, size=min(spec.leaf_pions, len(decaying_leaf)), replace=False) if len(decaying_leaf) else decaying_leaf
    drawn = np.zeros(len(kind), bool)
    cur = pick
    while len(cur):
        drawn[cur] = True
        cur = np.unique(parent[cur][parent[cur] >= 0])
    # muons and neutrinos of the drawn decaying pions
    is_mn = np.isin(kind, (MUON, NEUTRINO))
    drawn |= is_mn & drawn[np.where(parent >= 0, parent, 0)] & (parent >= 0)
    sd = sh.seeds
    par = sd["parent"]
    ok = drawn[par] & hadron[par]
    pair = np.repeat(rng.random(len(par) // 2), 2)                  # the two photons of a pi0 are drawn together
    sel = ok & (pair < _per_level(spec.seed_fraction, sd["level"]))
    seeds = {k: v[sel] for k, v in sd.items()}
    # the electromagnetic tree has n_gen = log2(E / E_c) doublings; q gives (1 + q)^n_gen = the target number of tracks
    n_gen = np.maximum(np.log2(np.maximum(seeds["e"] / p.e_c_ev, 1.0 + TOL)), 1.0)
    target = _per_level(spec.tree_tracks, seeds["level"])
    q = tree_keep_probability(target, n_gen)
    # the new table: the drawn hadronic tracks first (parents re-pointed), then the electromagnetic trees
    hidx = np.flatnonzero(drawn)
    remap = -np.ones(len(kind), int)
    remap[hidx] = np.arange(len(hidx))
    tab = Tracks()
    cols = {f: tr[f][hidx] for f in Tracks.FIELDS}
    cols["parent"] = np.where(cols["parent"] >= 0, remap[np.maximum(cols["parent"], 0)], -1)
    tab.add(**cols)
    seeds["parent"] = remap[seeds["parent"]]
    em_cascade(p, rng, seeds, tab, thin=False, keep_second=q)
    return tab.finish()
