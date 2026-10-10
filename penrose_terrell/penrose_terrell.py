"""Penrose-Terrell rotation: a football flying past a distant camera at beta = 0.99.

Left panel : the ball at rest, for reference.
Middle     : what a camera records (light that left the ball at different times
             arrives together; ray tracing with retarded times).
Right      : the "snapshot" of the Lorentz-contracted ball at one instant of the
             lab frame, which no camera can take.

Geometry (c = 1, ball radius R = 1).  The camera is far away on the +z axis and
looks along -z (orthographic projection); the ball moves along +x.  The camera
pixel (x, y) at camera time tau receives light emitted at the point
(x, y, z) at the lab time t_e = tau + z.  The ball centre is at
x_c(t) = beta * t, and in the lab frame the ball is a Lorentz-contracted
ellipsoid, so the visible surface point solves

    (gamma * (x - beta*(tau + z)))^2 + y^2 + z^2 = R^2,   take the larger z.

The body-frame (rest-frame) coordinates of that point are
    (x_b, y_b, z_b) = (gamma * (x - beta*(tau + z)), y, z),
which select the football pattern.  The snapshot panel uses t_e = tau for every
point instead.

All the numbers are in config.toml and all the words in texts.toml (see ../dvconfig.py for --config / --set).

Usage:
    python penrose_terrell.py --lang en                # full film -> media/penrose_terrell_en.mp4
    python penrose_terrell.py --lang ru
    python penrose_terrell.py --lang en --preview      # 3 s low-resolution film
    python penrose_terrell.py --lang en --frame 0.0    # one PNG at camera time tau = 0
"""

from __future__ import annotations

import argparse
import functools
import itertools
import math
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.spatial import ConvexHull

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dvconfig import load_config, load_texts  # noqa: E402

HERE = Path(__file__).resolve().parent
CFG = load_config(HERE)
TEXT = {lang: load_texts(HERE, lang) for lang in ("en", "ru")}     # texts.toml
PHI = (1 + 5 ** 0.5) / 2


# --------------------------------------------------------------- the football

def truncated_icosahedron_planes() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Face normals, face plane distances and a pentagon flag (True = pentagon)."""
    base = [(0, 1, 3 * PHI), (1, 2 + PHI, 2 * PHI), (PHI, 2, 2 * PHI + 1)]
    pts = set()
    for a, b, c in base:
        for s in itertools.product([1, -1], repeat=3):
            v = (a * s[0], b * s[1], c * s[2])
            for k in range(3):  # cyclic permutations
                pts.add(tuple(round(x, 9) for x in (v[k], v[(k + 1) % 3], v[(k + 2) % 3])))
    pts_arr = np.array(sorted(pts))
    assert len(pts_arr) == 60, len(pts_arr)
    hull = ConvexHull(pts_arr)
    # merge coplanar triangles into faces
    normals, dists, sizes = [], [], []
    eq = np.round(hull.equations, 6)
    for row in np.unique(eq, axis=0):
        n = row[:3] / np.linalg.norm(row[:3])
        on_plane = np.abs(pts_arr @ n + row[3] / np.linalg.norm(row[:3])) < 1e-5
        normals.append(n)
        dists.append(float(np.mean(pts_arr[on_plane] @ n)))   # exact plane distance
        sizes.append(int(on_plane.sum()))
    normals = np.array(normals)
    dists = np.array(dists)
    is_pentagon = np.array(sizes) == 5
    assert len(normals) == 32 and is_pentagon.sum() == 12
    return normals, dists, is_pentagon


NORMALS, DISTS, IS_PENTAGON = truncated_icosahedron_planes()
# the pentagon that faces the camera when the ball is at rest is painted orange,
# so that the apparent rotation can be followed by eye
HIGHLIGHT = int(np.argmax(np.where(IS_PENTAGON, NORMALS[:, 2], -2.0)))


def football_albedo(xb: np.ndarray, yb: np.ndarray, zb: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Albedo and highlight mask: black pentagons, white hexagons, dark seams (body frame)."""
    d = np.stack([xb, yb, zb], axis=-1)
    d = d / np.linalg.norm(d, axis=-1, keepdims=True)
    cosines = d @ NORMALS.T                       # (..., 32)
    t = DISTS / np.maximum(cosines, 1e-9)         # distance along the ray to each plane
    t = np.where(cosines > 1e-9, t, np.inf)
    order = np.argsort(t, axis=-1)
    first = order[..., 0]
    t_sorted = np.take_along_axis(t, order[..., :2], axis=-1)
    seam = (t_sorted[..., 1] - t_sorted[..., 0]) / t_sorted[..., 0] < CFG.ball.seam_width
    albedo = np.where(IS_PENTAGON[first], CFG.ball.albedo_pentagon, CFG.ball.albedo_hexagon)
    albedo = np.where(seam, CFG.ball.albedo_seam, albedo)
    return albedo, (first == HIGHLIGHT) & ~seam


# ----------------------------------------------------------------- the camera

def camera_image(x: np.ndarray, y: np.ndarray, tau: float, beta: float,
                 snapshot: bool = False) -> tuple[np.ndarray, np.ndarray]:
    """RGB image and coverage mask for pixel coordinates x, y (ball radius 1)."""
    gamma = 1.0 / math.sqrt(1.0 - beta * beta)
    if snapshot:
        # one lab-frame instant t = tau: (gamma*(x - beta*tau))^2 + y^2 + z^2 = 1
        u = gamma * (x - beta * tau)
        r2 = u * u + y * y
        hit = r2 <= 1.0
        z = np.sqrt(np.clip(1.0 - r2, 0.0, None))
        xb = u
    else:
        # gamma^2 (w - beta z)^2 + y^2 + z^2 = 1 with w = x - beta*tau, solve for z
        w = x - beta * tau
        a = gamma ** 2
        b = -2.0 * gamma ** 2 * beta * w
        c = gamma ** 2 * w * w + y * y - 1.0
        disc = b * b - 4 * a * c
        hit = disc >= 0.0
        z = (-b + np.sqrt(np.clip(disc, 0.0, None))) / (2 * a)
        xb = gamma * (w - beta * z)
    # headlight shading from the apparent (circular) outline of the camera image
    floor = CFG.ball.shade_floor
    if snapshot:
        shade = floor + (1.0 - floor) * np.clip(z, 0.0, 1.0)
    else:
        rr = (x - beta * tau) ** 2 + y * y
        shade = floor + (1.0 - floor) * np.sqrt(np.clip(1.0 - rr, 0.0, 1.0))
    albedo, mark = football_albedo(xb, y, z)
    gray = albedo * shade
    rgb = np.stack([gray * tint for tint in CFG.ball.tint], axis=-1)
    orange = np.stack([c * shade for c in CFG.ball.highlight], axis=-1)
    rgb = np.where(mark[..., None], orange, rgb)
    return np.where(hit[..., None], rgb, 0.0), hit


# --------------------------------------------------------------------- frames

def num(x: float, fmt: str, lang: str) -> str:
    """A number for use inside $...$: decimal comma in Russian."""
    s = format(x, fmt)
    return s.replace(".", "{,}") if lang == "ru" else s


@functools.lru_cache(maxsize=None)
def render_header(lang: str, beta: float, size: int) -> Image.Image:
    """The titles above the three panels (matplotlib mathtext; the same for every frame, so it is made once)."""
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure

    tx, LY, ST, dpi = TEXT[lang], CFG.layout, CFG.style, CFG.video.dpi
    head, gap = int(LY.header * size), LY.gap
    width = 3 * size + 2 * gap
    canvas_rgb = tuple(c / 255 for c in ST.canvas)
    fig = Figure(figsize=(width / dpi, head / dpi), dpi=dpi, facecolor=canvas_rgb)
    FigureCanvasAgg(fig)
    gamma = 1 / math.sqrt(1 - beta * beta)
    camera_sub = tx["sub_camera"].replace("{beta}", num(beta, ".2f", lang)).replace("{gamma}", num(gamma, ".1f", lang))
    titles = ((tx["title_rest"], tx["sub_rest"]), (tx["title_camera"], camera_sub), (tx["title_snapshot"], tx["sub_snapshot"]))
    for k, (t1, t2) in enumerate(titles):
        x0 = k * (size + gap) + size * LY.title_pos[0]
        fig.text(x0 / width, 1 - size * LY.title_pos[1] / head, t1, ha="left", va="top",
                 color=tuple(c / 255 for c in ST.title_colours[k]), fontsize=size * LY.title_size * 72 / dpi)
        fig.text(x0 / width, 1 - size * LY.subtitle_pos[1] / head, t2, ha="left", va="top",
                 color=tuple(c / 255 for c in ST.subtitle_colour), fontsize=size * LY.subtitle_size * 72 / dpi)
    fig.canvas.draw()
    return Image.fromarray(np.asarray(fig.canvas.buffer_rgba())[..., :3].copy())


def render_frame(tau: float, beta: float, size: int, span: float, lang: str = "en") -> Image.Image:
    """Three panels (at rest | camera | snapshot), each size x size, field of view +-span."""
    LY, ST = CFG.layout, CFG.style
    xs = np.linspace(-span, span, size)
    ys = np.linspace(span, -span, size)
    X, Y = np.meshgrid(xs, ys)
    bg = np.array(ST.background)
    panels = []
    for b, snap, t in ((0.0, False, 0.0), (beta, False, tau), (beta, True, tau)):
        rgb, hit = camera_image(X, Y, t, b, snapshot=snap)
        img = np.where(hit[..., None], rgb, bg)
        panels.append(Image.fromarray((np.clip(img, 0, 1) * 255).astype(np.uint8)))
    head = int(LY.header * size)
    gap = LY.gap
    canvas = Image.new("RGB", (3 * size + 2 * gap, size + head), tuple(ST.canvas))
    for k, p in enumerate(panels):
        canvas.paste(p, (k * (size + gap), head))
    canvas.paste(render_header(lang, beta, size), (0, 0))
    return canvas


def tau_at(frame: int, frames: int, tau0: float, tau1: float) -> float:
    return tau0 + (tau1 - tau0) * frame / max(frames - 1, 1)


def write_film(out: Path, beta: float, size: int, fps: int, seconds: float, lang: str = "en") -> None:
    if not shutil.which("ffmpeg"):
        sys.exit("ffmpeg is required")
    out.parent.mkdir(parents=True, exist_ok=True)
    frames = int(round(seconds * fps))
    span = CFG.scene.span
    # the ball (centre x = beta*tau) crosses the +-span window
    tau0, tau1 = (-span - CFG.scene.tau_margin) / beta, (span + CFG.scene.tau_margin) / beta
    sample = render_frame(0.0, beta, size, span, lang)
    cmd = ["ffmpeg", "-y", "-v", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
           "-s", f"{sample.width}x{sample.height}", "-r", str(fps), "-i", "-",
           "-c:v", "libx264", "-preset", CFG.video.preset, "-crf", str(CFG.video.crf), "-pix_fmt", "yuv420p",
           "-movflags", "+faststart", str(out)]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    for k in range(frames):
        img = render_frame(tau_at(k, frames, tau0, tau1), beta, size, span, lang)
        proc.stdin.write(img.tobytes())
    proc.stdin.close()
    proc.wait()
    print(f"wrote {out} ({frames} frames)")


def main() -> None:
    V = CFG.video
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--lang", choices=("en", "ru"), default="en")
    ap.add_argument("--beta", type=float, default=CFG.scene.beta)
    ap.add_argument("--size", type=int, default=V.panel_size, help="panel size in pixels")
    ap.add_argument("--fps", type=int, default=V.fps)
    ap.add_argument("--seconds", type=float, default=CFG.scene.film_length)
    ap.add_argument("--preview", action="store_true")
    ap.add_argument("--frame", type=float, default=None, help="write one PNG at camera time tau")
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--config", type=Path, default=None, help="another configuration file instead of config.toml")
    ap.add_argument("--set", action="append", default=[], metavar="SECTION.KEY=VALUE", help="override one configuration value")
    args = ap.parse_args()
    out = args.out or HERE / "media" / f"penrose_terrell_{args.lang}.mp4"
    if args.frame is not None:
        out = out.with_suffix(".png")
        out.parent.mkdir(parents=True, exist_ok=True)
        render_frame(args.frame, args.beta, args.size, CFG.scene.span, args.lang).save(out)
        print(f"wrote {out}")
        return
    if args.preview:
        write_film(out.with_name(out.stem + "_preview.mp4"), args.beta, V.preview_panel_size, V.preview_fps, V.preview_seconds, args.lang)
    else:
        write_film(out, args.beta, args.size, args.fps, args.seconds, args.lang)


if __name__ == "__main__":
    main()
