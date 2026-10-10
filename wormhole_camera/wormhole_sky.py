r"""The two skies of the wormhole film: ours (warm) and the one on the other side (cold), built from procedural galaxies, stars and nebulae.

Each sky is an equirectangular map in linear light (float16).  A map is bad near its poles, so every sky is stored in TWO maps: map A has its pole along the world z axis,
map B along the world x axis (m = (w_y, w_z, w_x)); every object is stamped into both maps with the same orientation of its own axes, and a direction is read
from the map where its latitude is below 45 degrees.  Mip levels give a filtered lookup (the lensed picture compresses the sky strongly near the edge of the throat).

    python3 wormhole_sky.py --side ours --preview ours.png        # a lat/lon preview of one sky
    python3 wormhole_sky.py --build                               # build the cache for both skies (a few minutes)
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dvconfig import load_config  # noqa: E402

HERE = Path(__file__).resolve().parent
CFG = load_config(HERE)
R_B = np.array([[0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [1.0, 0.0, 0.0]])      # world -> map B
ROT = {0: np.eye(3), 1: R_B}


def map_size() -> tuple[int, int]:
    W = int(CFG.sky.width)
    return W, W // 2


def dir_to_map(m):
    """(i, j) pixel coordinates (continuous, centres at .5) of unit vectors m in the frame of a map."""
    W, H = map_size()
    lon = np.arctan2(m[..., 1], m[..., 0])
    lat = np.arcsin(np.clip(m[..., 2], -1, 1))
    return (lon + np.pi) / (2 * np.pi) * W, (np.pi / 2 - lat) / np.pi * H, lat


def tangent_basis_world(n):
    """East and north unit vectors of the world frame at the unit vector n."""
    z = np.array([0.0, 0.0, 1.0])
    east = np.cross(z, n)
    if np.linalg.norm(east) < 1e-6:
        east = np.array([0.0, 1.0, 0.0])
    east /= np.linalg.norm(east)
    return east, np.cross(n, east)


def position_angle_in_map(n, pa, which):
    """Position angle (east -> north) of an axis given in the world frame by the angle pa, measured in the frame of map `which`."""
    e_w, n_w = tangent_basis_world(n)
    a_w = math.cos(pa) * e_w + math.sin(pa) * n_w
    R = ROT[which]
    m = R @ n
    lon, lat = math.atan2(m[1], m[0]), math.asin(max(-1, min(1, m[2])))
    east_m = np.array([-math.sin(lon), math.cos(lon), 0.0])
    north_m = np.array([-math.sin(lat) * math.cos(lon), -math.sin(lat) * math.sin(lon), math.cos(lat)])
    a_m = R @ a_w
    return math.atan2(a_m @ north_m, a_m @ east_m)


def stamp(img, n, which, extent_deg, pa, sprite):
    """Add sprite(X, Y) (X to the east, Y to the north, in degrees, rotated by the position angle pa) into the map `which` around the unit vector n (world frame).
    sprite returns an array (..., 3)."""
    W, H = map_size()
    m = ROT[which] @ n
    ic, jc, lat = dir_to_map(m)
    if abs(math.degrees(lat)) > CFG.sky.pole_margin_deg:
        return
    dpp = 360.0 / W
    ny = int(math.ceil(extent_deg / dpp)) + 1
    cl = max(math.cos(lat), 0.2)
    nx = int(math.ceil(ny / cl))
    ii = np.arange(int(math.floor(ic)) - nx, int(math.floor(ic)) + nx + 1)
    jj = np.arange(int(math.floor(jc)) - ny, int(math.floor(jc)) + ny + 1)
    jj = jj[(jj >= 0) & (jj < H)]
    if len(jj) == 0:
        return
    X = (ii[None, :] + 0.5 - ic) * dpp * cl
    Y = -(jj[:, None] + 0.5 - jc) * dpp
    pa_m = position_angle_in_map(n, pa, which)
    c, s = math.cos(pa_m), math.sin(pa_m)
    val = sprite(X * c + Y * s, -X * s + Y * c)
    edge = np.clip((1.0 - np.maximum(np.abs(X), np.abs(Y)) / extent_deg) / 0.35, 0, 1)      # fade to zero at the border of the patch
    val = val * (edge * edge * (3 - 2 * edge))[..., None]
    img[np.ix_(jj, ii % W)] += val.astype(np.float32)


# ------------------------------------------------------------------------- sprites (in the frame of the object, x along the major axis)

def sprite_spiral(size, incl, arms, pitch, phase, hue, P, rng):
    G = CFG.galaxy
    h = G.disk_scale_over_size * size
    rb = max(G.bulge_over_size * size, 1e-6)
    ci = math.cos(incl)
    kk = G.knots_per_galaxy
    kr = size * rng.uniform(0.08, 0.60, kk)
    kth = (phase + 2 * np.pi * rng.integers(0, arms, kk)) / arms - np.log(kr / h + 0.3) / math.tan(pitch)
    kx, ky = kr * np.cos(kth), kr * np.sin(kth)
    ks = size * 0.018 * rng.uniform(0.6, 1.5, kk)
    kw = rng.uniform(0.3, 1.0, kk)

    def f(x, y):
        yd = y / ci
        R = np.hypot(x, yd)
        th = np.arctan2(yd, x)
        taper = np.clip(1 - (R / size) ** 2, 0, 1) ** 2
        disk = np.exp(-R / h) * taper
        arm = (0.5 * (1 + np.cos(arms * th + arms * np.log(R / h + 0.3) / math.tan(pitch) - phase))) ** G.arm_sharpness
        bulge = np.exp(-2.6 * np.sqrt(R / rb))
        lane = 1 - 0.8 * np.exp(-(y / (G.dust_lane_width * size * 3)) ** 2) * (1 - ci) ** 1.5 * (R < size)
        out = np.zeros(x.shape + (3,))
        out += (disk * (0.30 * (1 - G.arm_amplitude) + 0.55) * lane)[..., None] * np.array(hue)
        out += (disk * arm * G.arm_amplitude * 1.6 * lane)[..., None] * np.array(P["arm"])
        out += (bulge * G.bulge_weight)[..., None] * np.array(P["bulge"])
        for xk, yk, sk, wk, kk_r in zip(kx, ky, ks, kw, kr):
            out += (G.knot_weight * wk * math.exp(-kk_r / (2 * h)) * np.exp(-((x - xk) ** 2 + ((y - yk * ci)) ** 2) / (2 * sk * sk)))[..., None] * np.array(P["arm"])
        return out
    return f


def sprite_elliptical(size, q, hue):
    G = CFG.galaxy
    re = 0.18 * size

    def f(x, y):
        R = np.hypot(x, y / q)
        prof = np.exp(-7.67 * ((R / re + 1e-6) ** 0.25 - 1.0)) * np.clip(1 - (R / size) ** 2, 0, 1) ** 2
        return np.minimum(prof, 40.0)[..., None] * np.array(hue) * 0.12
    return f


def sprite_star(flux, hue, sigma_deg, spikes):
    """A star: a Gaussian core, a soft halo and four diffraction spikes (length scale L, width w in degrees)."""
    L, w = spikes
    x0 = 3.0 * sigma_deg
    S = CFG.sky
    flux = flux * (S.width / S.reference_width) ** 2        # the star keeps its integrated flux when the map has more pixels

    def f(x, y):
        r2 = x * x + y * y
        out = flux * np.exp(-r2 / (2 * sigma_deg ** 2)) + flux * S.star_halo_weight * np.exp(-np.sqrt(r2) / (6 * sigma_deg))
        ax, ay = np.abs(x), np.abs(y)
        hs = (w / (ay + w)) ** 1.5 / (1.0 + ax / x0) ** 1.15
        vs = (w / (ax + w)) ** 1.5 / (1.0 + ay / x0) ** 1.15
        out = out + flux * S.spike_weight * (hs + vs) * np.exp(-np.hypot(ax, ay) / L)
        return out[..., None] * np.array(hue)
    return f


def sprite_texture(tex, size, gain):
    """tex: (h, w, 3) linear image; its width spans `size` degrees."""
    from scipy.ndimage import map_coordinates
    h, w, _ = tex.shape

    def f(x, y):
        u = (x / size + 0.5) * (w - 1)
        v = (0.5 - y / size) * (h - 1)
        ok = (u >= 0) & (u <= w - 1) & (v >= 0) & (v <= h - 1)
        out = np.stack([map_coordinates(tex[..., k], [v, u], order=1, mode="constant") for k in range(3)], -1)
        return out * ok[..., None] * gain
    return f


# ------------------------------------------------------------------------- the sky

def random_dirs(rng, n):
    v = rng.normal(size=(n, 3))
    return v / np.linalg.norm(v, axis=1, keepdims=True)


def nebula_field(rng, n_waves):
    """Random plane waves on the sphere: wave vectors with a power-law spectrum."""
    k = random_dirs(rng, n_waves)
    freq = 1.5 * np.exp(rng.uniform(0, math.log(40.0), n_waves))
    amp = freq ** -0.9
    ph = rng.uniform(0, 2 * np.pi, n_waves)
    return k, freq, amp / amp.sum() * 2.2, ph


def nebula_image(which, fields, colors, strengths, ds=4):
    W, H = map_size()
    w, h = W // ds, H // ds
    lon = (np.arange(w) + 0.5) / w * 2 * np.pi - np.pi
    lat = np.pi / 2 - (np.arange(h) + 0.5) / h * np.pi
    Lo, La = np.meshgrid(lon, lat)
    m = np.stack([np.cos(La) * np.cos(Lo), np.cos(La) * np.sin(Lo), np.sin(La)], -1)
    wdir = m @ ROT[which]                          # R^T m for rotation matrices: (R^T m)_i = sum_j R_ji m_j
    out = np.zeros((h, w, 3), np.float32)
    for (k, freq, amp, ph), col, st in zip(fields, colors, strengths):
        f = np.zeros((h, w))
        for kk, fr, a, p in zip(k, freq, amp, ph):
            f += a * np.cos(fr * (wdir @ kk) + p)
        ridge = np.clip(1.0 - np.abs(f), 0, 1) ** 4
        mask = np.clip(0.5 + 0.5 * f, 0, 1) ** 1.5
        out += (st * (0.55 * ridge + 0.45 * mask))[..., None] * np.array(col, np.float32)
    from scipy.ndimage import zoom
    big = np.stack([zoom(out[..., k], ds, order=1) for k in range(3)], -1)
    return big[:H, :W]


def build_sky(side: str, progress=print) -> list[np.ndarray]:
    """Return the two maps (A, B) of the sky `side` ('ours' or 'theirs') as float32 linear RGB."""
    S = CFG.sky
    P = CFG.sky[side]
    G = CFG.galaxy
    seed = S.seed_ours if side == "ours" else S.seed_theirs
    rng = np.random.default_rng(seed)
    W, H = map_size()
    maps = [np.zeros((H, W, 3), np.float32) for _ in range(2)]
    # nebulae
    nrng = np.random.default_rng(seed + P.nebula_seed_offset)
    fields = [nebula_field(nrng, P.nebula_waves) for _ in P.nebula_colors]
    for w in (0, 1):
        maps[w] += nebula_image(w, fields, P.nebula_colors, P.nebula_strength)
    progress("nebulae")
    # galaxies
    n = int(P.n_galaxies)
    dirs = random_dirs(rng, n)
    u = rng.random(n)
    size = P.size_min_deg * (1 - u * (1 - (P.size_min_deg / P.size_max_deg) ** P.size_slope)) ** (-1 / P.size_slope)
    kind = rng.random(n)
    pal = {"arm": np.array(P.arm_color), "bulge": np.array(P.bulge_color)}
    for i in range(n):
        s = float(size[i])
        pa = rng.uniform(0, math.pi)
        hue = np.array(P.galaxy_colors[rng.integers(len(P.galaxy_colors))])
        bright = G.surface_brightness * (s / P.size_max_deg) ** G.brightness_slope
        if kind[i] < P.spiral_fraction:
            incl = math.radians(rng.uniform(*G.inclination_deg))
            arms = int(rng.choice(G.spiral_arms))
            pitch = math.radians(rng.uniform(*G.spiral_pitch_deg))
            spr = sprite_spiral(s, incl, arms, pitch, rng.uniform(0, 2 * math.pi), hue, pal, rng)
        elif kind[i] < P.spiral_fraction + P.elliptical_fraction:
            spr = sprite_elliptical(s, rng.uniform(*G.elliptical_axis_ratio), hue)
        else:
            spr = sprite_elliptical(s, rng.uniform(0.3, 0.9), hue * np.array([0.8, 1.0, 1.1]))
        for w in (0, 1):
            stamp(maps[w], dirs[i], w, s * 1.05, pa, (lambda spr_, b_: (lambda X, Y: spr_(X, Y) * b_))(spr, bright * 0.12))
        if i % 4000 == 0:
            progress(f"galaxies {i}/{n}")
    # heroes
    for hero in P.heroes if "heroes" in P else []:
        n_h = np.array(hero["dir"], float)
        n_h /= np.linalg.norm(n_h)
        pa = math.radians(hero["pa_deg"])
        if hero["kind"] == "texture":
            tex = np.asarray(__import__("PIL.Image", fromlist=["Image"]).open(HERE / hero["file"]).convert("RGB"), np.float32) / 255.0
            tex = tex ** 2.2
            spr = sprite_texture(tex, hero["size_deg"], hero["gain"])
        elif hero["kind"] == "spiral":
            hrng = np.random.default_rng(hero["seed"])
            spr0 = sprite_spiral(hero["size_deg"] / 2, math.radians(hero["incl_deg"]), hero["arms"], math.radians(hero["pitch_deg"]), hero["phase"],
                                 np.array(P.galaxy_colors[0]), pal, hrng)
            spr = (lambda s_, g_: (lambda X, Y: s_(X, Y) * g_))(spr0, hero["gain"])
        else:
            hrng = np.random.default_rng(hero["seed"])
            spr0 = sprite_elliptical(hero["size_deg"] / 2, hero["axis_ratio"], np.array(P.galaxy_colors[0]))
            spr = (lambda s_, g_: (lambda X, Y: s_(X, Y) * g_))(spr0, hero["gain"])
        for w in (0, 1):
            stamp(maps[w], n_h, w, hero["size_deg"] * 0.75, pa, spr)
    progress("heroes")
    # stars
    ns = int(P.n_stars)
    sd = random_dirs(rng, ns)
    flux = S.star_flux_min * (1.0 - rng.random(ns)) ** (-1 / S.star_flux_slope)          # power law
    flux = np.minimum(flux, S.star_flux_max)
    cols = np.array(P.star_colors)[rng.integers(len(P.star_colors), size=ns)]
    sig = S.star_psf_sigma_px * 360.0 / W
    pa_s = 0.0
    for w in (0, 1):
        m = sd @ ROT[w].T
        ic, jc, lat = dir_to_map(m)
        ok = np.abs(np.degrees(lat)) < S.pole_margin_deg
        i0, j0 = np.floor(ic[ok]).astype(int), np.floor(jc[ok]).astype(int)
        fx, fy = ic[ok] - i0, jc[ok] - j0
        for dj in range(-3, 4):
            for di in range(-3, 4):
                px = di + 0.5 - fx
                py = dj + 0.5 - fy
                g = np.exp(-(px * px + py * py) / (2 * S.star_psf_sigma_px ** 2)) / (2 * math.pi * S.star_psf_sigma_px ** 2)
                jj = j0 + dj
                good = (jj >= 0) & (jj < H)
                np.add.at(maps[w], (jj[good], (i0[good] + di) % W), (g[good] * flux[ok][good])[:, None] * cols[ok][good] * 0.9)
    progress("stars")
    # bright stars with spikes
    nb = int(P.n_bright_stars)
    bd = random_dirs(rng, nb)
    for i in range(nb):
        fl = min(S.bright_flux_min * (1.0 - rng.random()) ** (-1 / S.bright_flux_slope), S.bright_flux_max)
        hue = np.array(P.star_colors[rng.integers(len(P.star_colors))])
        L_deg = S.spike_length_deg * (fl / S.bright_flux_min) ** 0.5
        spr = sprite_star(fl, hue, sig, (L_deg, S.spike_width_px * 360.0 / W))
        pa = rng.uniform(0, math.pi / 2)
        for w in (0, 1):
            stamp(maps[w], bd[i], w, 2.4 * L_deg, pa, spr)
    progress("bright stars")
    return maps


def cache_path(level: int) -> Path:
    return HERE / CFG.sky.cache_dir / f"sky_w{int(CFG.sky.width)}_L{level}.npy"


def build_cache(progress=print) -> None:
    (HERE / CFG.sky.cache_dir).mkdir(exist_ok=True)
    W, H = map_size()
    stack = np.zeros((4, H, W, 3), np.float16)
    for k, side in enumerate(("ours", "theirs")):
        maps = build_sky(side, progress)
        for w in (0, 1):
            stack[2 * k + w] = np.clip(maps[w], 0, 6e4).astype(np.float16)
        del maps
    level = 0
    cur = stack
    while True:
        np.save(cache_path(level), cur)
        if level + 1 >= int(CFG.sky.mip_levels):
            break
        a = cur.astype(np.float32)
        a = 0.25 * (a[:, 0::2, 0::2] + a[:, 1::2, 0::2] + a[:, 0::2, 1::2] + a[:, 1::2, 1::2])
        cur = a.astype(np.float16)
        level += 1
    progress("cache written")


class SkyPyramid:
    """Filtered lookups into the cached skies (mip levels are memory-mapped)."""

    def __init__(self) -> None:
        self.levels = [np.load(cache_path(l), mmap_mode="r") for l in range(int(CFG.sky.mip_levels))]
        self.W, self.H = map_size()

    def sample(self, far, d, lod_extra: float = 0.0):
        """Linear RGB (H, W, 3) of the directions d (H, W, 3, world frame); far selects the sky on the other side."""
        W0, H0 = self.W, self.H
        a_ok = np.abs(d[..., 2]) <= 0.70710678
        m = np.where(a_ok[..., None], d, np.stack([d[..., 1], d[..., 2], d[..., 0]], -1))
        lon = np.arctan2(m[..., 1], m[..., 0])
        lat = np.arcsin(np.clip(m[..., 2], -1, 1))
        # footprint of a pixel in the sky: angular difference of neighbouring directions
        dx = np.zeros(d.shape[:2])
        dy = np.zeros(d.shape[:2])
        dx[:, :-1] = np.linalg.norm(d[:, 1:] - d[:, :-1], axis=-1)
        dx[:, -1] = dx[:, -2]
        dy[:-1] = np.linalg.norm(d[1:] - d[:-1], axis=-1)
        dy[-1] = dy[-2]
        foot = np.maximum(dx, dy) * np.sqrt(np.cos(lat) + 1e-3)
        lod = np.clip(np.log2(np.maximum(foot, 1e-9) / (2 * np.pi / W0)) + lod_extra, 0, len(self.levels) - 1.001)
        l0 = np.floor(lod).astype(int)
        t = (lod - l0)[..., None]
        sid = 2 * far.astype(int) + (~a_ok).astype(int)
        out = np.zeros(d.shape, np.float32)
        for lev in range(len(self.levels)):
            for sel_l, wt in ((l0 == lev, 1.0 - t), (l0 + 1 == lev, t)):
                if not sel_l.any():
                    continue
                arr = self.levels[lev]
                Hl, Wl = arr.shape[1], arr.shape[2]
                u = (lon[sel_l] + np.pi) / (2 * np.pi) * Wl - 0.5
                v = (np.pi / 2 - lat[sel_l]) / np.pi * Hl - 0.5
                u0 = np.floor(u).astype(int)
                v0 = np.floor(v).astype(int)
                fu = (u - u0)[:, None]
                fv = (v - v0)[:, None]
                s = sid[sel_l]
                v0c, v1c = np.clip(v0, 0, Hl - 1), np.clip(v0 + 1, 0, Hl - 1)
                u0c, u1c = u0 % Wl, (u0 + 1) % Wl
                c = ((1 - fu) * (1 - fv) * arr[s, v0c, u0c] + fu * (1 - fv) * arr[s, v0c, u1c]
                     + (1 - fu) * fv * arr[s, v1c, u0c] + fu * fv * arr[s, v1c, u1c])
                out[sel_l] += (c * wt[sel_l]).astype(np.float32)
        return out


def tonemap(x, exposure: float = 1.0):
    """A filmic curve (ACES-like) from linear light to display values."""
    x = x * exposure
    a, b, c, d, e = 2.51, 0.03, 2.43, 0.59, 0.14
    y = np.clip((x * (a * x + b)) / (x * (c * x + d) + e), 0, 1)
    return y ** (1 / 2.2)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--build", action="store_true")
    ap.add_argument("--side", choices=("ours", "theirs"), default="ours")
    ap.add_argument("--preview", type=Path, default=None)
    ap.add_argument("--config", type=Path, default=None)
    ap.add_argument("--set", action="append", default=[])
    args = ap.parse_args()
    if args.build:
        build_cache()
    if args.preview:
        from PIL import Image
        maps = build_sky(args.side)
        img = (tonemap(maps[0], CFG.sky.exposure) * 255).astype(np.uint8)
        Image.fromarray(img).save(args.preview)
        print("wrote", args.preview)


if __name__ == "__main__":
    main()
