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
  exaggerated about 160 times and the film says so.

Dipoles, the creation and annihilation flashes and the field lines are a schematic picture,
not a simulation of the quantum vacuum. The curve is the formula, not a fit.

## Configuration

Every number of the film (frame size and encoder, timeline, physics, the number, lifetime and look of the pairs, field lines,
the graph, sizes of the dots, lines and fonts, layout of the frame, colours, opacities, random seeds) is in `config.toml`
(sections `video`, `timeline`, `model`, `pairs`, `flash`, `scene`, `field`, `electron`, `probe`, `glow`, `pair_style`, `panel`,
`panel_text`, `caption`, `fonts`, `style`); all the words and formulas are in `texts.toml` (one table per language, the same
keys in both). The script contains only algorithms. A value can be changed without editing a file:

```bash
python running_charge.py --lang en --snapshot 20 --set model.b_film=0.1 --set pairs.seed=7
python running_charge.py --lang ru --config my_config.toml
```

`--set section.key=value` is repeatable (the value is a Python/TOML literal), `--config` takes another file with the same
structure, a misspelt key is an error (see `../dvconfig.py`, tests in `../test_dvconfig.py`).

The film exists in two separate versions, English and Russian (`--lang en`, `--lang ru`); the formulas and labels are typeset
with matplotlib mathtext (LaTeX), words stay outside the formulas, Russian uses the decimal comma. (The first version of the
film had both languages in one picture, English above and Russian below; it is kept as `media/running_charge.mp4`.)

Run from this directory (numpy, scipy, matplotlib, Pillow and ffmpeg are required):

```bash
./render.sh --lang en       # media/running_charge_en.mp4, 26 s
./render.sh --lang ru       # media/running_charge_ru.mp4, 26 s
./render.sh --preview       # 6 s low-resolution check (media/running_charge_en_preview.mp4)
./render.sh --snapshot 18   # one PNG at film time 18 s (media/running_charge_en.png)
./render.sh --still         # clean 2560x1440 frame without captions and graph, cropped to the left part (media/running_charge_en_still.png)
python -m unittest test_running_charge
```

The tests check the limits and monotonicity of `Q(r)`, the real-coupling value, the polarization of the pairs near the charge,
the reproducibility of the pairs from the seed, the timeline and that both languages have the same keys.
