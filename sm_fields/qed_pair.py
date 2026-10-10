"""Two charged wave packets and their electromagnetic field: the scalar-QED model of ../fields/scalar_qed.

The model (see scalar_qed_numerics.py): a narrow Gaussian packet of the charged field phi spreads like a free
relativistic packet; its Coulomb-like potential  A^mu = (Q / m q) p^mu V(r*)  is the field on the photon sheet; the
potential of the other packet adds the eikonal phase  exp(-i q^2 theta)  to phi, with  theta  advected along the
packet and sourced by  p_a . A_b / E_a;  the gradient of this phase deflects the packet (repulsion for equal and
attraction for opposite charges).  The film shows |phi|^2 on the electron sheet and A_0 on the photon sheet.
"""

from __future__ import annotations

from dataclasses import replace
from functools import lru_cache

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dvconfig import load_config  # noqa: E402

import scalar_qed_numerics as nm  # noqa: E402

_CFG = load_config(Path(__file__).resolve().parent)
_Q = _CFG.qed
GRID_STEP = _CFG.heights.grid_step       # every n-th point of the model grid is drawn
LX = _Q.lx                      # the sheet is the square [-LX/2, LX/2]^2 of the plane of the packets
NGRID = _Q.ngrid                # NGRID x NGRID grid, every second point is drawn
T_SIM = _Q.t_sim                # length of the simulation
N_ANIM = _Q.n_anim              # animation samples
SIM_SAMPLES = _Q.sim_samples
MASS, CHARGE_Q, SIGMA, PX, X0, Y0, RATIO = _Q.mass, _Q.charge_q, _Q.sigma, _Q.px, _Q.x0, _Q.y0, _Q.ratio


def make_config(case: str, charge_q: float = CHARGE_Q) -> nm.ScalarQEDConfig:
    """case 'pp': e- e- (equal charges); 'pm': e- e+ (opposite charges)."""
    sign2 = 1.0 if case == "pp" else -1.0

    def pk(idx: int, side: float, ratio: float) -> nm.PacketConfig:
        return nm.PacketConfig(name=f"packet_{idx}", charge_sign=1 if ratio > 0 else -1, charge_ratio=ratio,
                               norm=float(np.sqrt(abs(ratio) / 2.0)), sigma=SIGMA, px=-side * PX, py=0.0,
                               x0=side * X0, y0=-side * Y0)
    return nm.ScalarQEDConfig(
        grid=nm.GridConfig(nx=NGRID, ny=NGRID, lx=LX, ly=LX),
        time=nm.TimeConfig(t_start=0.0, t_end=T_SIM, simulation_num_samples=SIM_SAMPLES, animation_num_samples=N_ANIM),
        physics=nm.PhysicsConfig(mass=MASS, charge_q=charge_q),
        observables=nm.ObservableConfig(lower_surface="phi_abs2", upper_surface="a0"),
        render=nm.RenderConfig(*_Q.render_unused),
        packets=(pk(1, -1.0, RATIO), pk(2, +1.0, sign2 * RATIO)))


@lru_cache(maxsize=None)
def bundle(case: str, charge_q: float = CHARGE_Q) -> dict:
    return nm.build_animation_bundle(make_config(case, charge_q))


def frame(case: str, s: float, charge_q: float = CHARGE_Q, step: int = GRID_STEP) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(|phi|^2, A_0, centres of the two packets) at the simulation time s, linear in time; the grids are every
    `step`-th point."""
    b = bundle(case, charge_q)
    t = b["times"]
    s = float(min(max(s, t[0]), t[-1]))
    i = int(min(np.searchsorted(t, s, side="right") - 1, len(t) - 2))
    w = (s - t[i]) / (t[i + 1] - t[i])
    low = (1 - w) * b["lower_frames"][i] + w * b["lower_frames"][i + 1]
    up = (1 - w) * b["upper_frames"][i] + w * b["upper_frames"][i + 1]
    c = (1 - w) * b["packet_centers"][i] + w * b["packet_centers"][i + 1]
    return low[::step, ::step], up[::step, ::step], c


def axes(step: int = GRID_STEP) -> np.ndarray:
    x = bundle("pp")["x_axis"]
    return x[::step]
