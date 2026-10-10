r"""A small ray-marching renderer of the surfaces |psi|^2 = const of a complex wave function, coloured by the phase of psi.

Orthographic camera, rays marched from the camera side, the first crossing of every level is refined by linear interpolation, the normal
is the gradient of |psi|^2 (central differences), Blinn-Phong shading.  The levels are composited front to back with the given
opacities (the outer one translucent, the inner one opaque).

The colour of a surface point is c(arg psi - delta) with the palette c_ch(phi) = offset + amp cos(phi - 2 pi ch / 3), which is linear in
exp(i phi).  Therefore everything that depends on the common phase factor exp(-i omega t) is kept in two images, S0 = sum w shade and
S1 = sum w shade exp(i arg psi), and the colour at any time is  offset S0 + amp Re[exp(i delta) exp(-2 i pi ch / 3) S1]  (+ specular):
the expensive ray marching is done once per state.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


def camera(elevation_deg: float, azimuth_deg: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(c, u, v): the unit vector from the origin to the camera, the screen right and the screen up (world z is up)."""
    el, az = math.radians(elevation_deg), math.radians(azimuth_deg)
    c = np.array([math.cos(el) * math.cos(az), math.cos(el) * math.sin(az), math.sin(el)])
    u = np.array([-math.sin(az), math.cos(az), 0.0])
    v = np.cross(c, u)
    return c, u, v


def project(points, cam) -> np.ndarray:
    """Screen coordinates (right, up) of world points (..., 3)."""
    _, u, v = cam
    p = np.asarray(points, dtype=float)
    return np.stack([p @ u, p @ v], axis=-1)


@dataclass
class Shading:
    ambient: float
    diffuse: float
    specular: float
    spec_power: float
    light: tuple          # components along (screen right, screen up, towards the camera)


@dataclass
class IsoImage:
    s0: np.ndarray        # sum of w * shade                      (H, W)
    s1: np.ndarray        # sum of w * shade * exp(i arg psi)     (H, W) complex
    spec: np.ndarray      # specular highlights, white           (H, W)
    cover: np.ndarray     # 1 - transmittance                     (H, W)

    def rgb(self, delta: float, offset: float, amp: float, bg: np.ndarray) -> np.ndarray:
        """Colours (H, W, 3) in [0, 1] when the phase is shifted by -delta (the common factor exp(-i omega t)), over the background bg."""
        out = np.empty(self.s0.shape + (3,), dtype=np.float32)
        rot = np.exp(1j * delta) * self.s1
        for ch in range(3):
            out[..., ch] = offset * self.s0 + amp * np.real(np.exp(-2j * math.pi * ch / 3) * rot) + self.spec
        out += (1.0 - self.cover)[..., None] * np.asarray(bg, dtype=np.float32)
        return np.clip(out, 0.0, 1.0)


def _density(psi_fn, P: np.ndarray, r_max: float, edge: float) -> np.ndarray:
    r = np.linalg.norm(P, axis=-1)
    w = np.clip((r_max - r) / edge, 0.0, 1.0)
    w = w * w * (3.0 - 2.0 * w)
    return np.abs(psi_fn(P)) ** 2 * w


def render(psi_fn, cam, levels, alphas, shading: Shading, r_max: float, view: float, size: int, samples: int,
           supersample: int = 1, edge: float = 1.0, grad_step: float = 1e-3) -> IsoImage:
    """Render the surfaces |psi|^2 = levels (ascending) of psi_fn(P) (P: (..., 3), x = p r) into a size x size image of the square
    [-view, view]^2 of the screen plane (row 0 is the top)."""
    c, u, v = cam
    n = size * supersample
    a = (np.arange(n) + 0.5) / n * 2.0 * view - view
    A, B = np.meshgrid(a, a[::-1])
    inside = (A * A + B * B) < r_max * r_max
    ia, ib = A[inside], B[inside]
    base = ia[:, None] * u + ib[:, None] * v                       # points of the screen plane through the origin (m, 3)
    m = base.shape[0]
    ts = r_max * (1.0 - 2.0 * np.arange(samples + 1) / samples)
    t_hit = [np.full(m, np.nan) for _ in levels]
    f_prev = np.zeros(m)
    t_prev = ts[0]
    for t in ts:
        f = _density(psi_fn, base + t * c, r_max, edge)
        if t != ts[0]:
            for k, lev in enumerate(levels):
                new = np.isnan(t_hit[k]) & (f >= lev) & (f_prev < lev)
                if new.any():
                    frac = (lev - f_prev[new]) / np.maximum(f[new] - f_prev[new], 1e-30)
                    t_hit[k][new] = t_prev + frac * (t - t_prev)
        f_prev, t_prev = f, t
    light = shading.light[0] * u + shading.light[1] * v + shading.light[2] * c
    light = light / np.linalg.norm(light)
    half = light + c
    half = half / np.linalg.norm(half)
    h = grad_step * r_max
    s0 = np.zeros(m)
    s1 = np.zeros(m, dtype=complex)
    spec = np.zeros(m)
    transm = np.ones(m)
    for k, alpha in enumerate(alphas):
        hit = ~np.isnan(t_hit[k])
        if not hit.any():
            continue
        P = base[hit] + t_hit[k][hit, None] * c
        grad = np.empty_like(P)
        for i in range(3):
            e = np.zeros(3)
            e[i] = h
            grad[:, i] = _density(psi_fn, P + e, r_max, edge) - _density(psi_fn, P - e, r_max, edge)
        nrm = -grad / np.maximum(np.linalg.norm(grad, axis=1, keepdims=True), 1e-30)
        nrm = np.where(((nrm @ c) < 0.0)[:, None], -nrm, nrm)               # the normal faces the camera
        diff = np.clip(nrm @ light, 0.0, 1.0)
        shade = shading.ambient + shading.diffuse * diff
        hl = shading.specular * np.clip(nrm @ half, 0.0, 1.0) ** shading.spec_power
        w = transm[hit] * alpha
        phase = np.exp(1j * np.angle(psi_fn(P)))
        s0[hit] += w * shade
        s1[hit] += w * shade * phase
        spec[hit] += w * hl
        transm[hit] *= 1.0 - alpha

    def full(vals, dtype=float):
        out = np.zeros((n, n), dtype=dtype)
        out[inside] = vals
        return out

    def down(img):
        if supersample == 1:
            return img
        return img.reshape(size, supersample, size, supersample).mean(axis=(1, 3))

    return IsoImage(down(full(s0)), down(full(s1, complex)), down(full(spec)), down(full(1.0 - transm)))
