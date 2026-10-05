# Running charge

Virtual electron-positron pairs screen an electron. In the vacuum the pairs appear and vanish
with random orientation. Near the electron they are polarized: each positron is pulled in and each
electron pushed out, so the pairs line up radially. A test charge that flies in measures a larger
charge the closer it gets. Beyond about one reduced Compton wavelength `lambda_C = hbar / m_e c`
the charge is the classical `e`.

Model (lengths in `lambda_C`):

- pairs are created uniformly in the plane (plus extra pairs near the charge) at random times and
  live a random time of order one unit;
- the orientation of a new pair relative to the radial direction is drawn from a von Mises
  distribution with concentration `kappa(r) = s K0 / (1 + (r/R0)^2)`, where `s` is the strength
  of the charge (switched on smoothly); the alignment fades with distance;
- the charge seen at distance `r` is

```text
Q(r) / e = 1 / (1 - b L(r)),     L(r) = (1/2) ln(1 + 1/r^2)
```

  which tends to 1 for `r >> lambda_C` and to `1 / (1 - b ln(1/r))` for `r << lambda_C`
  (one-loop running, `b = 2 alpha / 3 pi`). For the electron `b = 0.0015`, the film uses
  `b = 0.25`: the real change is about 1 % at `r = 1e-3 lambda_C`, so the scale is
  exaggerated about 150 times and the film says so.

Dipoles, the creation and annihilation flashes and the field lines are a schematic picture,
not a simulation of the quantum vacuum. The curve is the formula, not a fit.

Run from this directory (numpy, scipy, matplotlib and ffmpeg are required):

```bash
./render.sh                 # media/running_charge.mp4, 26 s
./render.sh --preview       # 6 s low-resolution check
./render.sh --snapshot 18   # one PNG at film time 18 s
python -m unittest test_running_charge
```

The tests check the limits and monotonicity of `Q(r)`, the real-coupling value and the polarization
of the pairs near the charge.
