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

## Configuration

Every number of the film (frame size and encoder, the foil and the beam, the integrator, the random seeds, the time scale of the film,
the layout of the frame, fonts, colours, sizes of the dots and lines) is in `config.toml` (sections `video`, `foil`, `simulation`,
`film`, `layout`, `fonts`, `style`); all the words and formulas are in `texts.toml` (one table per language, the same keys in
both; the numbers that appear in the text, such as the number of layers, are filled in by the script). The script contains only
algorithms. A value can be changed without editing a file:

```bash
python rutherford.py --simulate --data media/my_tracks.npz --set foil.screen=40 --set simulation.seed=3     # other tracks
python rutherford.py --lang en --snapshot 14 --data media/my_tracks.npz --set style.gold='"#ffffff"'
python rutherford.py --lang ru --config my_config.toml
```

`--set section.key=value` is repeatable (the value is a Python/TOML literal), `--config` takes another file with the same
structure, a misspelt key is an error (see `../dvconfig.py`, tests in `../test_dvconfig.py`). The parameters of the `foil` and
`simulation` sections change the trajectories: run `--simulate` again after changing them (the stored tracks in
`media/rutherford_tracks.npz` are the same for every language and are reproducible from the seed, `--seed`, `--alphas`, `--r-min`
and `--layers` override the defaults of the config).

The film exists in two separate versions, English and Russian (`--lang en`, `--lang ru`); the formulas and labels are typeset
with matplotlib mathtext (LaTeX), words stay outside the formulas. (The first version had both languages in each label;
it is kept as `media/rutherford.mp4`.)

Run from this directory (numpy, scipy, matplotlib, Pillow and ffmpeg are required):

```bash
./render.sh --simulate         # trajectories -> media/rutherford_tracks.npz (about 15 s)
./render.sh --lang en          # film -> media/rutherford_en.mp4 (about 24 s of video)
./render.sh --lang ru          # film -> media/rutherford_ru.mp4
./render.sh --preview          # 4 s low-resolution check (media/rutherford_en_preview.mp4)
./render.sh --snapshot 14      # one PNG at film time 14 s (media/rutherford_en.png)
python -m unittest test_rutherford
```

Parameters: `--alphas`, `--r-min`, `--layers`, `--seed`. The tests compare the integrated
deflection with `tan(theta/2) = r_min / 2b`, check that a head-on particle turns back, that
the screening removes the far field, and that both languages have the same keys.

References: E. Rutherford, Phil. Mag. 21, 669 (1911); H. Geiger, E. Marsden, Proc. R. Soc. A 82, 495 (1909).
