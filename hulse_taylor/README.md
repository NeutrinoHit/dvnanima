# Hulse-Taylor binary pulsar

PSR B1913+16: two neutron stars (1.438 and 1.390 solar masses) on an eccentric Kepler orbit
(`P_b = 7.75 h`, `e = 0.617`). They radiate gravitational waves and lose energy, the orbit shrinks,
the period falls, and the periastron arrives earlier and earlier than a constant-period model
predicts. The accumulated shift after a time `t` is

```text
delta(t) = (1/2) (dP_b/dt / P_b) t^2,        dP_b/dt < 0,
```

about -38.6 s after 30 years (1975 to 2005), as in the figure of the book.

Numbers (Weisberg and Huang, ApJ 829, 55, 2016, as quoted in the book): `P_b = 0.322997448918 d`,
`e = 0.6171334`, `dP_b/dt` = -2.40263e-12 (general relativity), -2.398e-12 (observed, kinematic
contribution subtracted). The semi-major axis follows from Kepler's third law (1.95e9 m). The dashed
curve in the film is the shift computed from the measured derivative, not individual timing points.

Not to scale (and labelled so in the film): the orbit shrinks by `5.4e-8` in 30 years, drawn as 7.5 %;
the phase advance of the stars with respect to the constant-period model (the hollow circles) is
`0.5` degree, drawn 120 times larger; the gravitational-wave pattern is a rotating quadrupole with
retarded phase and a wavelength about a thousand times smaller than the real one, with the amplitude
following the instantaneous separation so that the bursts come at periastron.

## Configuration

Every number of the film (frame size and encoder, timeline, physics of the orbit, the wave field, sizes of the stars, lines and fonts,
the layout of the frame, colours, opacities, the seed of the background stars) is in `config.toml` (8 sections: `video`, `model`, `timeline`,
`layout`, `waves`, `formats`, `fonts`, `style`); all the words and formulas are in `texts.toml` (one table per language, the same keys in both).
The script contains only algorithms. A value can be changed without editing a file:

```bash
python hulse_taylor.py --lang en --snapshot 28 --set model.phase_exaggeration=60 --set video.crf=18
python hulse_taylor.py --lang en --config my_config.toml
```

`--set section.key=value` is repeatable (the value is a Python/TOML literal), `--config` takes another file with the same
structure, a misspelt key is an error (see `../dvconfig.py`, tests in `../test_dvconfig.py`).

The film exists in two separate versions, English and Russian (`--lang en`, `--lang ru`); all formulas and labels are typeset
with matplotlib mathtext (LaTeX), words stay outside the formulas. The first version of the film was bilingual (an English and a Russian
line in the same frame); the two versions now have one language each.

Run from this directory (numpy, matplotlib and ffmpeg are required):

```bash
./render.sh --lang en                # media/hulse_taylor_en.mp4, 36 s
./render.sh --lang ru                # media/hulse_taylor_ru.mp4
./render.sh --lang en --preview      # 8 s low-resolution check
./render.sh --lang en --snapshot 28  # one PNG at film time 28 s
python -m unittest test_hulse_taylor
```

References: R. A. Hulse, J. H. Taylor, Astrophys. J. 195, L51 (1975); J. M. Weisberg, Y. Huang,
Astrophys. J. 829, 55 (2016); P. C. Peters, Phys. Rev. 136, B1224 (1964).
