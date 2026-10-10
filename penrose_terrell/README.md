# Penrose-Terrell rotation

A football flies past a distant camera at `beta = 0.99` (`gamma = 7.1`). Three panels:

1. the ball at rest, for reference (the orange pentagon faces the camera);
2. what the camera records: light that left different points of the ball at
   different times arrives together. The ball is not flattened: its outline
   is a circle, and the pattern looks rotated by `psi = arcsin(beta)`
   (82 degrees here), so the orange pentagon is seen at the limb;
3. the "snapshot" of the Lorentz-contracted ball at one instant of the lab frame.
   It is a thin ellipsoid, but no camera can take such a picture.

Model: orthographic camera on the `+z` axis, ball moving along `+x`, `c = 1`, `R = 1`.
The camera pixel `(x, y)` at camera time `tau` receives light emitted at lab time
`t_e = tau + z` from the surface point that solves

```text
gamma^2 (x - beta (tau + z))^2 + y^2 + z^2 = 1      (larger root z)
```

Its rest-frame coordinates `(gamma (x - beta (tau + z)), y, z)` select the pattern
(a truncated icosahedron built from its 60 vertices). The snapshot panel uses
`t_e = tau` instead. Shading is a simple headlight; Doppler and aberration of
brightness and colour are not modelled.

## Configuration

Every number of the film (panel size and encoder, field of view, colours and shading of the ball, the layout of the header) is in
`config.toml` (5 sections: `video`, `scene`, `ball`, `layout`, `style`); all the words and formulas are in `texts.toml` (one table per
language, the same keys in both). The script contains only algorithms (the geometry of the football, the ray tracing). The frame is three
square panels with a header, so it is not 16:9 and the `video` section holds the panel size instead of a frame size. A value can be changed
without editing a file:

```bash
python penrose_terrell.py --lang en --frame 0.5 --set scene.beta=0.9 --set video.panel_size=720
python penrose_terrell.py --lang en --config my_config.toml
```

`--set section.key=value` is repeatable (the value is a Python/TOML literal), `--config` takes another file with the same
structure, a misspelt key is an error (see `../dvconfig.py`).

The film exists in two separate versions, English and Russian (`--lang en`, `--lang ru`); the titles are typeset with matplotlib
mathtext (LaTeX), words stay outside the formulas. (The first version of the film showed the titles in both languages at once.)

Run from this directory (numpy, scipy, pillow, matplotlib and ffmpeg are required):

```bash
./render.sh --lang en                 # media/penrose_terrell_en.mp4, 12 s
./render.sh --lang ru                 # media/penrose_terrell_ru.mp4
./render.sh --lang en --preview       # 3 s low-resolution check
./render.sh --lang en --frame 0.0     # one PNG at camera time tau = 0
python -m unittest test_penrose_terrell
```

The tests check the football geometry, the circular outline of the camera image,
the contraction of the snapshot by `gamma`, and `sin(psi) = beta`.

References: R. Penrose, Proc. Camb. Phil. Soc. 55, 137 (1959); J. Terrell, Phys. Rev. 116, 1041 (1959).
