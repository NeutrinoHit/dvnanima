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

Run from this directory (numpy, scipy, pillow and ffmpeg are required):

```bash
./render.sh                 # media/penrose_terrell.mp4, 12 s
./render.sh --preview       # 3 s low-resolution check
./render.sh --frame 0.0     # one PNG at camera time tau = 0
python -m unittest test_penrose_terrell
```

The tests check the football geometry, the circular outline of the camera image,
the contraction of the snapshot by `gamma`, and `sin(psi) = beta`.

References: R. Penrose, Proc. Camb. Phil. Soc. 55, 137 (1959); J. Terrell, Phys. Rev. 116, 1041 (1959).
