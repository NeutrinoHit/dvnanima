# A camera at a wormhole

Book: volume 1, chapter "Gravity" (`chapters/Gravity.tex`, the exercises "Wormhole", "A photograph of the Universe at a wormhole" and "Einstein tensor and energy-momentum tensor for a wormhole" in `Gravity_tasks.tex`).
A cinematic film: a camera that flies around a Morris-Thorne wormhole and collects the light of galaxies from our side and from the other side of the throat. 2 min 50 s (170.6 s with the title card), Russian and English separately.

## What the film shows

1. **A solution of Einstein's equations** (34 s). The metric `ds² = c²dt² − dr²/(1 − b²/r²) − r²dΩ²`, the embedding surface `r = b cosh(z/b)` (two sheets, our universe and the other),
   the throat of radius `b`, the exotic matter: `ρc² = −c⁴b²/(8πG r⁴) < 0`, `E_side = −πbc⁴/(4G) ≈ −0.95·10⁴⁷ J` for `b = 1` km, `|E| ≈ 1.1 M☉c²` (the numbers of the book, computed from the constants in `config.toml`).
2. **Rays of light** (30 s). Geodesics of the surface in the equatorial plane; the impact parameter `p = L/C` is conserved; `p < b`: the ray crosses the throat (cyan), `p > b`: it turns back (orange);
   the cone of the rays from the other side, `sin ψ_c = b/√(l² + b²)`, grows as the camera comes closer.
3. **A view from afar** (30 s). The ray-traced picture: the whole other universe in a round window, the sky around it bent into rings; the camera flies around the wormhole.
4. **Through the throat** (34 s). The window grows, the camera crosses the throat (`l = 0`) and goes on in the other universe.
5. **How to measure the throat** (40 s). The camera turns back and sees our universe in a round window; the angular radius of the window `ψ_c(l)` (graph with the current point) depends only on the distance to the throat,
   so a moving camera measures `b` (the idea of the experiment of the exercise); the summary.
An inset with the embedding surface, the camera and its rays is shown during the camera parts.

## The physics (all tests in `test_wormhole_camera.py`)

The spatial geometry is `dl² + (l² + b²)dΩ²`, `r² = l² + b²`, `l = b sinh u`, `z = b u`. A ray has the conserved impact parameter `p`; the angle swept round the centre is `dφ/du = p / sqrt(b² cosh²u − p²)`.
The integral is elliptic and is used in closed form (`wormhole_physics.sweep`): for `p < b` it is `q(2K(q²) − F(θ_c|q²))`, `q = p/b`, for `p > b` it is `K(k²) + F(φ_c|k²)`, `k² = b²/p²`
(the same functions as the Jacobi solution `r(φ)` of the book's exercise). Checks: the closed form against a direct quadrature, the flat-space limit `b → 0`, the cone `sin ψ_c = b/√(l²+b²)`
against the radius of the window in the picture (pixel measurement), the symmetry of the throat, the continuity of the picture across `l = 0`, the energy estimate of the book, `ray_path` against `sweep`.

For every pixel the view direction is turned into the angle `ψ` to the throat direction and the azimuth `χ`; `sweep` gives the exit direction on the sky of this side or of the other side.
There is no redshift or aberration: the wormhole is static and the camera moves slowly.

## The skies

`wormhole_sky.py`: two skies in linear light, built from our own procedural galaxies (spirals with arms and star-forming knots, ellipticals, edge-on disks with dust lanes), stars with diffraction spikes,
nebulae made of random waves on the sphere, and the hero objects: the pair of interacting galaxies from the book's N-body simulation (`assets/hero_merger.png`) on the other side.
The sky on our side is warm (gold and teal), on the other side cold (violet and blue), so that the side is visible at once. Each sky is stored in two maps with different poles (no distortion near the poles),
with mip levels for a filtered lookup. The cache (`.cache/`, about 600 MB) is built by `python wormhole_sky.py --build` (about 15 s). The post-processing: a filmic tone curve, bloom, a vignette, a light grain.

## Run

```bash
python wormhole_sky.py --build                                  # once
python wormhole_camera.py --lang en --workers 3                 # media/wormhole_camera_en.mp4  (about 25 min on three cores)
python wormhole_camera.py --lang ru --workers 3
python wormhole_camera.py --lang en --snapshot 120              # one PNG at the film time 120 s
python wormhole_camera.py --lang en --preview --workers 3       # 640x360, 15 fps (about 4 min)
python test_wormhole_camera.py
./render.sh                                                     # both languages
```

A bigger picture (for example 4K) needs only `--set video.width=3840 --set video.height=2160` (the sky maps are 6144 pixels wide, `--set sky.width=12288` and a new `--build` give more detail).
The render is about 9 times longer.

## Configuration

Every number is in `config.toml` (sections `video`, `physics`, `sky`, `galaxy`, `view`, `timeline`, `style`, `fonts`, `layout`, `surface`, `part1` ... `part5`, `camera`, `inset`, `graph`), every word in `texts.toml`
(tables `[en]` and `[ru]` with the same keys). The camera path is the table `camera.frames` (time, `l/b`, orbit angle, yaw, pitch, roll, field of view; a smooth monotone interpolation).
Override: `--config other.toml` or `--set camera.frames=...`, `--set view.bloom_strength=0.3`.

## Files

`wormhole_camera.py` (the film), `wormhole_physics.py` (rays, surface, energy), `wormhole_sky.py` (the skies), `wormhole_view.py` (the ray-traced picture), `config.toml`, `texts.toml`,
`test_wormhole_camera.py`, `assets/hero_merger.png`, `CARD.md`, `render.sh`.
