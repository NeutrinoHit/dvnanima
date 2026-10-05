# Rutherford scattering of alpha particles

Alpha particles cross a thin gold foil (the Geiger-Marsden experiment). Each particle is a
classical point charge that moves in the field of fixed point nuclei; its 3D trajectory is
integrated numerically (DOP853). Nothing is imposed: the angles follow from the integration
and are shown as a histogram against the Rutherford formula.

Model:

- units `m = 1`, `v0 = 1`, `E = 1/2`; `r_min` is the head-on distance of closest approach and
  the potential energy is `U = K exp(-r/a) / r` with `K = r_min E`;
- atoms are neutral: the electrons screen the nucleus beyond `a = 25`. Without screening the
  field of the whole foil would reflect the beam;
- the foil is 4 layers (14 apart) of jittered square lattices, mean spacing of nuclei 50;
- 6000 particles with random impact parameters in a `180 x 180` beam cross section.

Check: for `theta` well above `r_min/a` the histogram follows

```text
dN/dtheta = n (pi r_min^2 / 4) cos(theta/2) / sin^3(theta/2)      (n = layers / spacing^2)
```

At small angles the screening cut-off and multiple scattering make the histogram deviate
from the formula, which is why the film draws the curve only for `theta > 12` degrees.

The picture is schematic: nuclei are drawn much larger than they are and the foil is only a
few layers thick, so that large-angle events appear within a short film. In the real
experiment about one alpha particle in 8000 turned back; Thomson's model predicts none.

Run from this directory (numpy, scipy, matplotlib and ffmpeg are required):

```bash
./render.sh --simulate     # trajectories -> media/rutherford_tracks.npz (about 10 s)
./render.sh                # film -> media/rutherford.mp4 (about 24 s of video)
./render.sh --preview      # 4 s low-resolution check
./render.sh --snapshot 14  # one PNG at film time 14 s
python -m unittest test_rutherford
```

Parameters: `--alphas`, `--r-min`, `--layers`, `--seed`. The tests compare the integrated
deflection with `tan(theta/2) = r_min / 2b`, check that a head-on particle turns back and that
the screening removes the far field.

References: E. Rutherford, Phil. Mag. 21, 669 (1911); H. Geiger, E. Marsden, Proc. R. Soc. A 82, 495 (1909).
