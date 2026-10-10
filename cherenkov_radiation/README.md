# Cherenkov radiation

One scene with four stages, 40 s. A charge moves along `x` with `v = beta c` in a medium with `n = 1.33`
(water). The wave speed is `c/n`, `c = 1`. The wavelets are circles emitted every 0.1 s; their radii are exact.

* `beta < 1/n` (0-10 s, `beta = 0.55`): the circles are nested, there is no envelope and no light.
* `beta > 1/n` (10-28 s, `beta = 0.85`, then `0.99`): the wavelets have a common envelope, a cone with
  half-angle `psi = arcsin(1/(n beta))`; light travels perpendicular to it at `theta`, `cos(theta) = 1/(n beta)`.
* 28-40 s: `beta` rises slowly from 0.80 to 0.99; the angle and the detector ring `R = L tan(theta)` grow.

Water: threshold `beta = 0.752` (0.26 MeV kinetic energy for an electron, 54.6 MeV for a muon), `theta_max = 41.2 deg`.
The cone line is drawn for the instantaneous `beta`; while `beta` changes it lags the true envelope (the wavelets
remember the past), so it is dimmed during the jumps.

```bash
./render.sh --lang en         # media/cherenkov_radiation_en.mp4
./render.sh --lang ru         # media/cherenkov_radiation_ru.mp4
./render.sh --lang en --preview
./render.sh --lang en --snapshot 24
python -m unittest test_cherenkov_radiation
```

References: P. A. Cherenkov, Dokl. Akad. Nauk SSSR 2, 451 (1934); I. M. Frank and I. E. Tamm, Dokl. Akad.
Nauk SSSR 14, 107 (1937). The older `cherenkov_cone` (3D cylinder) stays in the gallery.

The film exists in two separate versions, English and Russian (`--lang en`, `--lang ru`); all formulas and labels are typeset with matplotlib mathtext (LaTeX), words stay outside the formulas.

## Configuration

Every number of the film (frame size and encoder, timeline, physics, sizes of the dots, lines and fonts, the layout of the frame,
colours, opacities) is in `config.toml` (9 sections: `video`, `physics`, `timeline`, `wavelets`, `scene`, `angle_panel`, `ring_panel`, `layout`, `style`); all the words and formulas are in `texts.toml` (one table per
language, the same keys in both). The script contains only algorithms. A value can be changed without editing a file:

```bash
python cherenkov_radiation.py --lang en --snapshot 20 --set style.background='"#000000"' --set video.crf=18
python cherenkov_radiation.py --lang en --config my_config.toml
```

`--set section.key=value` is repeatable (the value is a Python/TOML literal), `--config` takes another file with the same
structure, a misspelt key is an error (see `../dvconfig.py`, tests in `../test_dvconfig.py`).
