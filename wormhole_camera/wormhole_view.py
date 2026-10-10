"""What the camera sees: ray-traced view of the two skies through the wormhole (the rays are the closed-form geodesics of wormhole_physics)."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import wormhole_physics as wp  # noqa: E402
import wormhole_sky as sky  # noqa: E402

CFG = sky.CFG
_PYR: sky.SkyPyramid | None = None
_RAYS: dict = {}


def pyramid() -> sky.SkyPyramid:
    global _PYR
    if _PYR is None:
        _PYR = sky.SkyPyramid()
    return _PYR


def camera_basis(yaw: float, pitch: float, roll: float):
    """Camera axes (right, up, forward) in the physical local frame (e_l, t1, t2) of the camera position.
    yaw = pitch = 0: the camera looks along -e_l (towards decreasing l); yaw turns towards t1, pitch towards t2, roll about the view axis."""
    fwd = np.array([-np.cos(yaw) * np.cos(pitch), np.sin(yaw) * np.cos(pitch), np.sin(pitch)])
    up_hint = np.array([0.0, 0.0, 1.0])
    right = np.cross(fwd, up_hint)
    if np.linalg.norm(right) < 1e-9:
        right = np.array([0.0, 1.0, 0.0])
    right /= np.linalg.norm(right)
    up = np.cross(right, fwd)
    cr, sr = np.cos(roll), np.sin(roll)
    return np.array([cr * right + sr * up, -sr * right + cr * up, fwd])


def blur(img, sigma):
    from scipy.ndimage import gaussian_filter
    return np.stack([gaussian_filter(img[..., k], sigma, mode="nearest") for k in range(3)], -1)


def post(lin, W, H, rng_seed=0):
    V = CFG.view
    x = lin * V.exposure
    glow = np.zeros_like(x)
    bright = np.maximum(x - V.bloom_threshold, 0.0)
    for s_ in V.bloom_sigmas_px:
        glow += blur(bright, s_ * H / 720.0) / len(V.bloom_sigmas_px)
    x = x + V.bloom_strength * glow * 3.0
    out = sky.tonemap(x, 1.0)
    yy, xx = np.mgrid[0:H, 0:W]
    r2 = ((xx - W / 2) / (W / 2)) ** 2 + ((yy - H / 2) / (H / 2)) ** 2
    out = out * (1.0 - V.vignette * np.clip(r2 / 2.0, 0, 1) ** 1.1)[..., None]
    if V.grain > 0:
        out = out + np.random.default_rng(rng_seed).normal(0, V.grain, out.shape[:2])[..., None]
    return np.clip(out, 0, 1)


def render_view(l_cam: float, alpha: float, yaw: float, pitch: float, roll: float, W: int, H: int, frame_seed: int = 0, b: float | None = None, fov: float | None = None):
    """The camera at the proper radial coordinate l_cam (the sign says which side), angular position alpha on the equatorial great circle; returns float RGB in [0, 1].
    Also returns the boolean mask of the pixels that show the sky of the other universe ('theirs', the one behind the throat as seen from our side)."""
    b = CFG.physics.b if b is None else b
    fov = CFG.view.fov_deg if fov is None else fov
    key = (W, H, round(fov, 1))
    if key not in _RAYS:
        _RAYS.clear()
        _RAYS[key] = wp.pixel_rays(W, H, fov)
    dirs = _RAYS[key]
    s = 1.0 if l_cam >= 0 else -1.0
    basis = camera_basis(yaw, pitch, roll) * np.array([s, 1.0, 1.0])       # the 'out' direction flips with the side
    c_hat = np.array([np.cos(alpha), np.sin(alpha), 0.0])
    t1 = np.array([-np.sin(alpha), np.cos(alpha), 0.0])
    far, d_sky = wp.view_to_sky(dirs, basis, l_cam, c_hat, t1, b, table_n=int(CFG.view.table_n))
    lin = pyramid().sample(far, d_sky, CFG.view.lod_bias)
    return post(lin, W, H, frame_seed), far


if __name__ == "__main__":
    import time
    from PIL import Image
    out = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(".")
    for name, (l, al, yw) in {"a_far": (9.0, 0.0, 0.0), "b_mid": (3.0, 0.8, 0.0), "c_near": (1.0, 1.6, 0.0), "d_throat": (0.0, 2.0, 0.0),
                              "e_other": (-2.0, 2.0, 0.0), "f_offcentre": (5.0, 0.0, 0.5)}.items():
        t = time.time()
        img, _ = render_view(l, al, yw, 0.0, 0.0, 1280, 720)
        Image.fromarray((img * 255).astype(np.uint8)).save(out / f"view_{name}.png")
        print(name, round(time.time() - t, 2), "s")
