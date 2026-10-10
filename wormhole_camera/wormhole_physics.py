r"""Light in the Ellis-Bronnikov (Morris-Thorne) wormhole of the book: ds^2 = c^2dt^2 - dl^2 - (l^2+b^2) dOmega^2, r^2 = l^2 + b^2.

The embedding surface of the equatorial plane is r = b cosh(z/b), so l = b sinh u, z = b u.  A light ray has the conserved impact parameter p = L/C
and lies in a plane through the centre; with u as the parameter its angle round the centre obeys  dphi/du = p / sqrt(b^2 cosh^2 u - p^2).
The integral is elliptic and is evaluated in closed form (the same functions as the Jacobi solution of the book's exercise):

    p < b  (the ray crosses the throat):  swept angle from the camera to the far sky      q (2K(q^2) - F(theta_c | q^2)),  q = p/b, cos theta_c = tanh u_c
    p > b  (the ray turns back):          swept angle from the camera back to our sky     K(k^2) + F(phi_c | k^2),         k^2 = b^2/p^2,     cos phi_c = l_t / l_c
    outward ray, p < b / p > b:           q F(theta_c | q^2)   /   K(k^2) - F(phi_c | k^2)

Every number comes from the arguments (the film passes them from config.toml)."""

from __future__ import annotations

import numpy as np
from scipy.special import ellipk, ellipkinc

TWO_PI = 2.0 * np.pi


def smooth(x: float, a: float, b: float) -> float:
    """0 before a, 1 after b, smoothstep in between."""
    t = np.clip((x - a) / (b - a), 0.0, 1.0)
    return float(t * t * (3.0 - 2.0 * t))


def sweep(psi, l_cam: float, b: float):
    """Rays leaving a camera at the proper radial coordinate l_cam (the side is fixed by sign(l_cam)).

    psi: angle between the ray and the direction from the camera to the throat (0 = straight to the throat, pi = away from it).
    Returns (far, angle): far is True for a ray that ends in the sky on the other side of the throat, angle is the angle swept round the centre."""
    psi = np.asarray(psi, float)
    l_c = abs(float(l_cam))
    r_c = np.hypot(l_c, b)
    p = r_c * np.sin(psi)
    q = p / b
    uc = np.arcsinh(l_c / b)
    theta_c = np.arccos(np.tanh(uc))
    inward = psi <= 0.5 * np.pi
    through = inward & (q < 1.0)
    turn = inward & (q >= 1.0)
    out_thr = (~inward) & (q < 1.0)
    out_turn = (~inward) & (q >= 1.0)
    m_q = np.clip(q * q, 0.0, 1.0 - 1e-15)
    angle = np.zeros_like(psi)
    # k^2 = 1 - 1/q^2 for q >= 1; l_t = b sqrt(q^2-1)
    qq = np.maximum(q, 1.0 + 1e-15)
    k2 = 1.0 / (qq * qq)
    l_t = b * np.sqrt(np.maximum(qq * qq - 1.0, 0.0))
    phi_c = np.arccos(np.clip(l_t / max(l_c, 1e-300), 0.0, 1.0)) if l_c > 0 else np.zeros_like(psi)
    angle = np.where(through, q * (2.0 * ellipk(m_q) - ellipkinc(theta_c, m_q)), angle)
    angle = np.where(turn, ellipk(k2) + ellipkinc(phi_c, k2), angle)
    angle = np.where(out_thr, q * ellipkinc(theta_c, m_q), angle)
    angle = np.where(out_turn, ellipk(k2) - ellipkinc(phi_c, k2), angle)
    return through, angle


def critical_angle(l_cam: float, b: float) -> float:
    """Angular radius (from the throat direction) of the cone of rays that cross the throat: sin psi_c = b / sqrt(l^2 + b^2)."""
    return float(np.arcsin(b / np.hypot(l_cam, b)))


def sweep_quadrature(psi: float, l_cam: float, b: float, n: int = 400, u_max: float = 16.0) -> tuple[bool, float]:
    """The same swept angle by direct Gauss-Legendre integration of dphi/du (an independent check of `sweep`)."""
    xg, wg = np.polynomial.legendre.leggauss(n)
    l_c = abs(l_cam)
    r_c = np.hypot(l_c, b)
    p = r_c * np.sin(psi)
    uc = np.arcsinh(l_c / b)
    tail = 2.0 * p * np.exp(-u_max) / b

    def integ(u0, u1, sing=False):
        if sing:
            s1 = np.sqrt(u1 - u0)
            s = 0.5 * s1 * (xg + 1)
            u = u0 + s * s
            return float(np.sum(0.5 * s1 * wg * 2 * s * p / np.sqrt(np.maximum(b * b * np.cosh(u) ** 2 - p * p, 1e-300))))
        h = 0.5 * (u1 - u0)
        u = h * (xg + 1) + u0
        return float(np.sum(h * wg * p / np.sqrt(b * b * np.cosh(u) ** 2 - p * p)))

    if psi > 0.5 * np.pi:
        if p < b:
            return False, integ(uc, u_max) + tail
        ut = np.arccosh(p / b)
        return False, integ(uc, u_max) + tail
    if p < b:
        # split at u = 0 where the integrand peaks (narrowly when p -> b)
        return True, integ(0.0, uc) + integ(-u_max, 0.0) + tail if uc > 0 else integ(-u_max, 0.0) + tail
    ut = np.arccosh(p / b)
    return False, integ(ut, uc, True) + integ(ut, u_max, True) + tail


# ---------------------------------------------------------------- the surface and the energy of the book (exercises "Wormhole", "Einstein tensor")

def surface_radius(z, b: float):
    """r(z) = b cosh(z/b)."""
    return b * np.cosh(np.asarray(z, float) / b)


def throat_z_of_l(l, b: float):
    return b * np.arcsinh(np.asarray(l, float) / b)


def energy_density_si(r, b: float, c: float, g_newton: float):
    """rho_E = -c^4 b^2 / (8 pi G r^4)  (energy per unit volume, from G_00 = -b^2/r^4)."""
    return -(c ** 4) * b * b / (8.0 * np.pi * g_newton * np.asarray(r, float) ** 4)


def side_energy_si(b: float, c: float, g_newton: float) -> float:
    """E_side = -pi b c^4 / (4 G)."""
    return -np.pi * b * c ** 4 / (4.0 * g_newton)


# ---------------------------------------------------------------- camera frames

def unit(v):
    v = np.asarray(v, float)
    return v / np.linalg.norm(v, axis=-1, keepdims=True)


def pixel_rays(width: int, height: int, fov_h_deg: float):
    """Unit view directions (x right, y up, z forward) of the pixel centres of a pinhole camera."""
    f = 0.5 * width / np.tan(np.radians(fov_h_deg) / 2.0)
    x, y = np.meshgrid(np.arange(width) - width / 2 + 0.5, height / 2 - np.arange(height) - 0.5)
    d = np.stack([x, y, np.full_like(x, f)], -1)
    return d / np.linalg.norm(d, axis=-1, keepdims=True)


def orient(forward, up_hint):
    """Right-handed camera basis (right, up, forward) from the forward direction and an approximate up."""
    f = unit(forward)
    r = unit(np.cross(f, up_hint))
    u = np.cross(r, f)
    return r, u, f


def lensed_directions(dirs_cam, basis, l_cam: float, c_hat, e_l_sign_out, b: float):
    """For the view directions in the camera frame (basis rows = right, up, forward expressed in the local frame of the camera:
    components along (e_out, e_1, e_2), where e_out points to larger |l| on the camera's side and e_1, e_2 are two tangent unit vectors) return
    (far, exit_unit_vector) with the exit direction as a 3D unit vector in the world frame of the angular coordinates.

    c_hat: unit vector of the angular position of the camera; the tangent vectors are given through the arguments of `local_frame`."""
    raise NotImplementedError


def local_frame(c_hat, tangent1):
    """Orthonormal tangent vectors (t1, t2) at the angular position c_hat, t1 given (made orthogonal to c_hat)."""
    c = unit(c_hat)
    t1 = unit(np.asarray(tangent1, float) - np.dot(tangent1, c) * c)
    t2 = np.cross(c, t1)
    return t1, t2


def view_to_sky(dirs_cam, cam_basis, l_cam: float, c_hat, t1, b: float, table_n: int = 6000, crit_pack: float = 0.9995):
    """Map camera pixel directions to the sky.

    dirs_cam: (H, W, 3) directions in the camera frame; cam_basis: 3x3 whose rows are the camera axes (right, up, forward) written in the local
    frame (out, t1, t2) of the camera position, where 'out' is the radial direction pointing away from the throat on the camera's side.
    Returns (far, d_sky) with d_sky the world unit vectors of the exit directions, far = True where the sky is the one on the other side."""
    c_hat = unit(c_hat)
    t1, t2 = local_frame(c_hat, t1)
    local = dirs_cam @ cam_basis                                  # components along (out, t1, t2)
    cos_psi = np.clip(-local[..., 0], -1.0, 1.0)                  # psi measured from the direction to the throat
    psi = np.arccos(cos_psi)
    tr = np.stack([local[..., 1], local[..., 2]], -1)
    chi = np.arctan2(tr[..., 1], tr[..., 0])                      # azimuth of the transverse part of the ray in the (t1, t2) plane
    # a table in psi (dense near the critical angle, log spaced on both sides) interpolated to the pixels
    pc = critical_angle(l_cam, b)
    eps = np.geomspace(1e-9, 1.0, table_n // 3)
    grid = np.unique(np.concatenate([np.linspace(0.0, np.pi, table_n // 3), pc - eps * pc, pc + eps * (np.pi - pc), [pc - 1e-12, pc + 1e-12]]))
    grid = grid[(grid >= 0.0) & (grid <= np.pi)]
    far_g, ang_g = sweep(grid, l_cam, b)
    idx = np.searchsorted(grid, psi).clip(1, len(grid) - 1)
    t = (psi - grid[idx - 1]) / (grid[idx] - grid[idx - 1])
    far = far_g[idx - 1] if True else None
    # the sides differ across the critical angle: take the flag of the nearer grid point
    far = np.where(t < 0.5, far_g[idx - 1], far_g[idx])
    ang = ang_g[idx - 1] * (1 - t) + ang_g[idx] * t
    d_sky = (np.cos(ang)[..., None] * c_hat + np.sin(ang)[..., None] * (np.cos(chi)[..., None] * t1 + np.sin(chi)[..., None] * t2))
    if l_cam < 0.0:
        far = ~far
    return far, d_sky


# ---------------------------------------------------------------- rays drawn on the embedding surface

def ray_path(psi: float, l_cam: float, b: float, u_end: float, n: int = 500):
    """Points (u, phi) along the ray that leaves the camera at angle psi from the direction to the throat, up to |u| = u_end
    (u = asinh(l/b) is z/b on the surface; the camera is at phi = 0 and phi grows along the ray).  Returns (u, phi, goes_through)."""
    l_c = abs(l_cam)
    r_c = np.hypot(l_c, b)
    p = r_c * np.sin(abs(psi))
    uc = np.arcsinh(l_c / b)
    sgn = 1.0 if psi >= 0 else -1.0
    c = lambda u: np.maximum(b * b * np.cosh(u) ** 2 - p * p, 1e-18)

    def cum(u_arr):
        f = p / np.sqrt(c(u_arr))
        return np.concatenate([[0.0], np.cumsum(0.5 * (f[1:] + f[:-1]) * np.abs(np.diff(u_arr)))])

    if abs(psi) > 0.5 * np.pi:                          # away from the throat
        u = np.linspace(uc, u_end, n)
        return u, sgn * cum(u), False
    if p < b:                                            # through the throat
        u = np.linspace(uc, -u_end, n)
        return u, sgn * cum(u), True
    ut = np.arccosh(p / b)                               # turns back at u_t
    s1 = np.sqrt(max(uc - ut, 0.0))
    s_in = np.linspace(s1, 0.0, n // 2)
    s_out = np.linspace(0.0, np.sqrt(max(u_end - ut, 0.0)), n // 2)
    u_in, u_out = ut + s_in ** 2, ut + s_out ** 2
    f_in = 2 * s_in * p / np.sqrt(c(u_in))
    f_out = 2 * s_out * p / np.sqrt(c(u_out))
    ph_in = np.concatenate([[0.0], np.cumsum(0.5 * (f_in[1:] + f_in[:-1]) * np.abs(np.diff(s_in)))])
    ph_out = ph_in[-1] + np.concatenate([[0.0], np.cumsum(0.5 * (f_out[1:] + f_out[:-1]) * np.abs(np.diff(s_out)))])
    return np.concatenate([u_in, u_out[1:]]), sgn * np.concatenate([ph_in, ph_out[1:]]), False
