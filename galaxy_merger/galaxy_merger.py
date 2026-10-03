#!/usr/bin/env python3
"""Merger of two disk galaxies: a restricted N-body model with dynamical friction.

The model follows the idea of Toomre & Toomre (ApJ 178, 623, 1972):

* every galaxy is a massive extended halo (Plummer sphere of mass M and scale a);
  the two halos attract each other and each one is slowed down while it moves
  through the other (Chandrasekhar dynamical friction), so the pair spirals in
  and merges instead of flying apart;
* every galaxy carries a thin disk of massless test stars on initially circular
  orbits; the stars feel the gravity of both halos and not each other.

Units: G = 1, mass of the first galaxy = 1, pericentre of the Kepler start orbit
= 1. All parameters live in ``run.cfg`` and can be overridden on the command line.
Colours only tell which galaxy a star came from.

Requires numpy, matplotlib (only for stills) and ffmpeg.

    python galaxy_merger.py
    python galaxy_merger.py --stars 15000 --pixels 540 --hold 2 --out preview.mp4
"""
from __future__ import annotations

import argparse
import configparser
import math
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
G = 1.0
INT_KEYS = {"stars", "fps", "pixels", "seed"}
COLOURS = {0: np.array([1.00, 0.55, 0.20]),     # first galaxy: warm
           1: np.array([0.25, 0.55, 1.00])}     # second galaxy: cold
RAMP_STOPS = np.array([0.0, 0.18, 0.5, 0.82, 1.0])


# ---------------------------------------------------------------- configuration
def load_cfg(path: Path) -> dict:
    """Flat dictionary of all values from every section of the config file."""
    cfg = configparser.ConfigParser(inline_comment_prefixes=(";",))
    if not cfg.read(path):
        raise FileNotFoundError(path)
    values: dict = {}
    for section in cfg.sections():
        for key, raw in cfg[section].items():
            if key in INT_KEYS:
                values[key] = int(raw)
            else:
                try:
                    values[key] = float(raw)
                except ValueError:
                    values[key] = raw
    return values


def parse_args(argv=None) -> argparse.Namespace:
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--config", type=Path, default=HERE / "run.cfg")
    known, _ = pre.parse_known_args(argv)
    values = load_cfg(known.config)

    ap = argparse.ArgumentParser(description=__doc__, parents=[pre],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    for key, value in values.items():
        ap.add_argument("--" + key.replace("_", "-"), type=type(value), default=value)
    return ap.parse_args(argv)


# --------------------------------------------------------------------- dynamics
def df_accel(r_vec, v_vec, m_self, m_other, a_other, lnl):
    """Chandrasekhar friction on a body of mass ``m_self`` moving with velocity
    ``v_vec`` through a Plummer halo (``m_other``, ``a_other``) at separation ``r_vec``
    from its centre. The halo is isotropic, so sigma^2 = G M / (6 sqrt(r^2 + a^2))."""
    r2 = r_vec @ r_vec
    v = math.sqrt(v_vec @ v_vec) + 1e-12
    rho = 3 * m_other / (4 * math.pi * a_other**3) * (1 + r2 / a_other**2) ** -2.5
    sig2 = G * m_other / (6 * math.sqrt(r2 + a_other**2))
    x = v / math.sqrt(2 * sig2)
    f = math.erf(x) - 2 * x / math.sqrt(math.pi) * math.exp(-x * x)
    return -4 * math.pi * G**2 * m_self * lnl * rho * f * v_vec / v**3


def core_acc(c1, c2, v1, v2, m1, m2, a1, a2, lnl):
    """Accelerations of the two halo centres.

    The mutual attraction of two Plummer spheres is exactly a softened point-mass
    force with eps^2 = a1^2 + a2^2. The friction changes the relative acceleration
    and is shared between the bodies so that the total momentum is conserved."""
    d = c2 - c1
    f = G * d / (d @ d + a1 * a1 + a2 * a2) ** 1.5
    acc1, acc2 = m2 * f, -m1 * f
    if lnl > 0:
        vrel = v2 - v1
        a_rel = df_accel(d, vrel, m2, m1, a1, lnl) - df_accel(d, -vrel, m1, m2, a2, lnl)
        mt = m1 + m2
        acc1 = acc1 - m2 / mt * a_rel
        acc2 = acc2 + m1 / mt * a_rel
    return acc1, acc2


def halo_acc(pos, c, m, a):
    """Acceleration of massless particles in the field of one Plummer halo."""
    d = c - pos
    r2 = (d * d).sum(axis=1) + a * a
    return G * m * d / r2[:, None] ** 1.5


def orbit_start(m1, m2, q, e, theta0):
    """Two centres on a bound Kepler orbit with pericentre ``q`` and eccentricity ``e``,
    at true anomaly ``theta0``; the centre of mass is at rest at the origin."""
    mt = m1 + m2
    p = q * (1 + e)
    r = p / (1 + e * math.cos(theta0))
    k = math.sqrt(G * mt / p)
    vr, vt = k * e * math.sin(theta0), k * (1 + e * math.cos(theta0))
    c, s = math.cos(theta0), math.sin(theta0)
    rel = r * np.array([c, s, 0.0])
    vrel = np.array([vr * c - vt * s, vr * s + vt * c, 0.0])
    return -m2 / mt * rel, m1 / mt * rel, -m2 / mt * vrel, m1 / mt * vrel


def rotation(incl, node):
    """Disk orientation: tilt by ``incl`` about x, then turn by ``node`` about z."""
    ci, si, cn, sn = np.cos(incl), np.sin(incl), np.cos(node), np.sin(node)
    rx = np.array([[1, 0, 0], [0, ci, -si], [0, si, ci]])
    rz = np.array([[cn, -sn, 0], [sn, cn, 0], [0, 0, 1]])
    return rz @ rx


def make_disk(n, m, a, r_max, incl, node, rng, sigma_v=0.04):
    """``n`` stars with surface density ~ exp(-r/h) and a smooth outer edge, on circular
    orbits in the halo (``m``, ``a``) with a relative velocity scatter ``sigma_v``."""
    h = r_max / 3.5
    grid = np.linspace(0.0, r_max, 4000)
    cdf = np.cumsum(grid * np.exp(-grid / h) * (1 - (grid / r_max) ** 2) ** 2)
    cdf /= cdf[-1]
    r = np.interp(rng.random(n), cdf, grid)
    phi = rng.uniform(0, 2 * np.pi, n)
    pos = np.column_stack([r * np.cos(phi), r * np.sin(phi), rng.normal(0, 0.004, n)])
    vc = np.sqrt(G * m * r**2 / (r**2 + a * a) ** 1.5) * (1 + rng.normal(0, sigma_v, n))
    vel = np.column_stack([-vc * np.sin(phi), vc * np.cos(phi), np.zeros(n)])
    rot = rotation(incl, node)
    return pos @ rot.T, vel @ rot.T


def simulate(args):
    """Leapfrog integration of the halo centres and all stars.

    Returns the sky-plane positions of every star in every frame, the halo centres,
    their separation and the galaxy label (0 or 1) of every star."""
    rng = np.random.default_rng(args.seed)
    m1, m2 = 1.0, args.mass_ratio
    a1, a2 = args.halo, args.halo * math.sqrt(m2)
    c1, c2, v1, v2 = orbit_start(m1, m2, args.pericentre, args.ecc, args.theta0)

    n1 = args.stars
    n2 = int(args.stars * m2 / m1)
    p1, w1 = make_disk(n1, m1, a1, args.rdisk, np.radians(args.incl1), np.radians(args.node1), rng)
    p2, w2 = make_disk(n2, m2, a2, args.rdisk * math.sqrt(m2),
                       np.radians(args.incl2), np.radians(args.node2), rng)
    pos = np.vstack([p1 + c1, p2 + c2])
    vel = np.vstack([w1 + v1, w2 + v2])
    lab = np.r_[np.zeros(n1, np.int8), np.ones(n2, np.int8)]

    dt = args.dt
    per_frame = max(1, int(round(args.frame_time / dt)))
    n_steps = int(round(args.duration / dt))
    n_frames = n_steps // per_frame + 1
    xy = np.empty((n_frames, len(lab), 2), np.float32)
    cores = np.empty((n_frames, 2, 2))
    sep = np.empty(n_frames)

    k1, k2 = core_acc(c1, c2, v1, v2, m1, m2, a1, a2, args.lnl)
    acc = halo_acc(pos, c1, m1, a1) + halo_acc(pos, c2, m2, a2)
    t0 = time.time()
    f = 0
    for step in range(n_steps + 1):
        if step % per_frame == 0 and f < n_frames:
            xy[f] = pos[:, :2]
            cores[f, 0], cores[f, 1] = c1[:2], c2[:2]
            sep[f] = np.linalg.norm(c2 - c1)
            f += 1
            if f % 50 == 0:
                print("  simulated %d / %d frames (%.0f s)" % (f, n_frames, time.time() - t0),
                      file=sys.stderr, flush=True)
        vel += 0.5 * dt * acc;  v1 = v1 + 0.5 * dt * k1;  v2 = v2 + 0.5 * dt * k2
        pos += dt * vel;        c1 = c1 + dt * v1;        c2 = c2 + dt * v2
        k1, k2 = core_acc(c1, c2, v1, v2, m1, m2, a1, a2, args.lnl)
        acc = halo_acc(pos, c1, m1, a1) + halo_acc(pos, c2, m2, a2)
        vel += 0.5 * dt * acc;  v1 = v1 + 0.5 * dt * k1;  v2 = v2 + 0.5 * dt * k2
    return xy, cores, sep, lab


# -------------------------------------------------------------------- rendering
def smooth1d(y, sigma):
    """Gaussian smoothing of a 1-d array with edge padding."""
    if sigma <= 0:
        return y
    r = int(4 * sigma)
    k = np.exp(-0.5 * (np.arange(-r, r + 1) / sigma) ** 2)
    k /= k.sum()
    return np.convolve(np.pad(y, r, mode="edge"), k, mode="valid")


def camera(cores, args):
    """Half-width of the view in every frame. It follows the two galaxies rather than
    the far debris and is smoothed in log scale."""
    reach = np.abs(cores).max(axis=(1, 2)) + args.reach * args.rdisk
    width = np.maximum(reach, args.min_width)
    return np.exp(smooth1d(np.log(width), args.cam_smooth))


def blur_kernel(npx, scale=1.0):
    """Fourier multiplier of a sum of three Gaussians: fine detail, halo and glow."""
    ky = np.fft.fftfreq(npx)[:, None]
    kx = np.fft.rfftfreq(npx)[None, :]
    k2 = kx**2 + ky**2
    s = scale * npx / 1080.0
    return sum(w * np.exp(-2 * (np.pi * sg * s) ** 2 * k2)
               for sg, w in ((1.2, 1.0), (4.5, 3.0), (14.0, 8.0)))


def make_backdrop(npx, rng, n=900):
    """A faint fixed field of distant stars behind the galaxies."""
    img = np.zeros((npx, npx))
    xs = rng.integers(0, npx, n)
    ys = rng.integers(0, npx, n)
    np.add.at(img, (ys, xs), 0.10 + 0.9 * rng.random(n) ** 8)
    ky = np.fft.fftfreq(npx)[:, None]
    kx = np.fft.rfftfreq(npx)[None, :]
    smooth = np.exp(-2 * (np.pi * 0.8 * npx / 1080) ** 2 * (kx**2 + ky**2))
    img = np.fft.irfft2(np.fft.rfft2(img) * smooth, s=img.shape)
    img = np.clip(img / img.max() * 2.2, 0, 0.7)
    return img[..., None] * np.array([0.85, 0.9, 1.0])


def surface_density(x, y, sel, half, npx, kernel):
    """Blurred number of selected stars per unit area (rows are y, columns are x)."""
    ix = ((x[sel] + half) / (2 * half) * npx).astype(np.int64)
    iy = ((y[sel] + half) / (2 * half) * npx).astype(np.int64)
    ok = (ix >= 0) & (ix < npx) & (iy >= 0) & (iy < npx)
    h = np.bincount(iy[ok] * npx + ix[ok], minlength=npx * npx).reshape(npx, npx).astype(float)
    area = (2 * half / npx) ** 2
    return np.fft.irfft2(np.fft.rfft2(h) * kernel, s=h.shape) / area


def colourize(dens, dref, args):
    """Tone-map the two density maps (arcsinh stretch, black -> colour -> white) and add them."""
    rgb = np.zeros(dens[0].shape + (3,))
    for g in (0, 1):
        t = np.arcsinh(args.stretch * dens[g] / dref) / np.arcsinh(args.stretch)
        t = np.clip((t - args.black) / (1 - args.black), 0, 1)
        c = COLOURS[g]
        ramp = np.array([0 * c, 0.30 * c, c, 0.55 * c + 0.45, np.ones(3)])
        for k in range(3):
            rgb[..., k] += np.interp(t, RAMP_STOPS, ramp[:, k])
    return rgb


def render_frame(xy, lab, half, angle, dref, kernel, backdrop, vignette, args):
    """One picture: stars of the two galaxies seen from the rotated viewpoint."""
    ca, sa = np.cos(angle), np.sin(angle)
    x = ca * xy[:, 0] - sa * xy[:, 1]
    y = sa * xy[:, 0] + ca * xy[:, 1]
    dens = [surface_density(x, y, lab == g, half, args.pixels, kernel) for g in (0, 1)]
    rgb = colourize(dens, dref, args)
    rgb = 1 - (1 - np.clip(rgb, 0, 1)) * (1 - backdrop)       # screen blend
    rgb = rgb * vignette[..., None]
    return np.clip(rgb, 0, 1)[::-1]                             # y axis points up


def open_ffmpeg(path, npx, fps):
    cmd = ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
           "-s", "%dx%d" % (npx, npx), "-r", str(fps), "-i", "-",
           "-c:v", "libx264", "-preset", "slow", "-crf", "17", "-pix_fmt", "yuv420p",
           "-movflags", "+faststart", str(path)]
    return subprocess.Popen(cmd, stdin=subprocess.PIPE)


def output_path(name: str) -> Path:
    path = Path(name)
    return path if path.is_absolute() else HERE / path


def main(argv=None):
    args = parse_args(argv)
    t0 = time.time()
    print("simulating ...", file=sys.stderr, flush=True)
    xy, cores, sep, lab = simulate(args)
    nf = len(xy)
    print("frames: %d (%.1f s at %d fps), simulation %.0f s"
          % (nf, nf / args.fps, args.fps, time.time() - t0), file=sys.stderr, flush=True)
    print("core separation: start %.2f, minimum %.3f, end %.3f" % (sep[0], sep.min(), sep[-1]),
          file=sys.stderr)

    half = camera(cores, args)
    angles = np.radians(args.view + args.spin * np.arange(nf) / max(nf - 1, 1))
    npx = args.pixels
    kernel = blur_kernel(npx)
    backdrop = make_backdrop(npx, np.random.default_rng(args.seed + 1))
    yy, xx = np.mgrid[0:npx, 0:npx]
    vignette = 1 - 0.30 * ((xx - npx / 2) ** 2 + (yy - npx / 2) ** 2) / (npx / 2) ** 2 / 2

    # One reference brightness for the whole film (no flicker), found on sampled frames.
    dref = 0.0
    for i in list(range(0, nf, max(1, nf // 40))) + [nf - 1]:
        ca, sa = np.cos(angles[i]), np.sin(angles[i])
        x = ca * xy[i][:, 0] - sa * xy[i][:, 1]
        y = sa * xy[i][:, 0] + ca * xy[i][:, 1]
        for g in (0, 1):
            dref = max(dref, np.percentile(surface_density(x, y, lab == g, half[i], npx, kernel), 99.9))
    dref *= args.bright

    def exposure(i):
        """Smooth change of the reference brightness: the compact remnant is much denser."""
        s = np.clip((i / max(nf - 1, 1) - args.exposure_from) / max(1 - args.exposure_from, 1e-9), 0, 1)
        return 1 + (args.end_exposure - 1) * s * s * (3 - 2 * s)

    def to_bytes(rgb):
        return (np.clip(rgb, 0, 1) * 255 + 0.5).astype(np.uint8).tobytes()

    out = output_path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    ff = open_ffmpeg(out, npx, args.fps)
    sheet_idx = set(np.linspace(0, nf - 1, 8).astype(int).tolist())
    poster_idx = int(args.poster_time * (nf - 1))
    sheet = {}
    poster = None

    for i in range(nf):
        rgb = render_frame(xy[i], lab, half[i], angles[i], dref * exposure(i), kernel, backdrop, vignette, args)
        if i in sheet_idx:
            sheet[i] = rgb.copy()
        if i == poster_idx:
            poster = rgb.copy()
        ff.stdin.write(to_bytes(rgb))
        if (i + 1) % 50 == 0:
            print("  rendered %d / %d frames (%.0f s)" % (i + 1, nf, time.time() - t0),
                  file=sys.stderr, flush=True)
    last = rgb

    # The remnant stays as it is while the camera slowly moves in and turns a little.
    n_hold = int(round(args.hold * args.fps))
    for k in range(n_hold):
        u = (k + 1) / n_hold
        e = u * u * (3 - 2 * u)
        rgb_k = render_frame(xy[nf - 1], lab, half[nf - 1] / (1 + (args.hold_zoom - 1) * e),
                             angles[nf - 1] + np.radians(args.hold_spin) * e,
                             dref * exposure(nf - 1), kernel, backdrop, vignette, args)
        ff.stdin.write(to_bytes(rgb_k))
    ff.stdin.close()
    ff.wait()

    if args.poster or args.last_frame or args.sheet:
        import matplotlib.image as mpimg
        if args.poster:
            mpimg.imsave(output_path(args.poster), poster)
        if args.last_frame:
            mpimg.imsave(output_path(args.last_frame), last)
        if args.sheet:
            keys = sorted(sheet)
            rows = [np.concatenate([sheet[k] for k in keys[:4]], axis=1),
                    np.concatenate([sheet[k] for k in keys[4:]], axis=1)]
            mpimg.imsave(output_path(args.sheet), np.concatenate(rows, axis=0))
    print("done in %.0f s -> %s" % (time.time() - t0, out), file=sys.stderr)


if __name__ == "__main__":
    main()
