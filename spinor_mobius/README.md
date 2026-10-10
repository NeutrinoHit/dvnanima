# Spinor and the Moebius strip

A spin-1/2 spinor changes sign under a rotation by 360 degrees and returns to its value only
after 720 degrees. Three linked views of one rotation angle `phi` (0 to 720 degrees):

- a dial with the rotating body;
- a Moebius strip with a vector lying across it, carried once around the core circle per
  360 degrees: it comes back reversed after the first turn and restored after the second;
- the complex plane with the spinor component `psi = exp(-i phi / 2)` (the Dirac equation example
  from the book): `psi = -1` at 360 degrees, `psi = +1` at 720 degrees.

Strip: `P(u, v) = ((1 + v cos(u/2)) cos u, (1 + v cos(u/2)) sin u, v sin(u/2))`, the vector across
it is `w(u) = dP/dv = (cos(u/2) cos u, cos(u/2) sin u, sin(u/2))`, hence `w(u + 2 pi) = -w(u)`.

The film illustrates the sign change of a spinor. It does not show the belt (plate) trick or
the wave function of two fermions.

## Configuration

Every number of the film (frame size and encoder, timeline with the pauses, strip and sampling, camera, layout of the frame,
colours, line widths and font sizes) is in `config.toml` (7 sections: `video`, `timeline`, `model`, `camera`, `layout`, `fonts`, `style`);
all the words and formulas are in `texts.toml` (one table per language, the same keys in both). The script contains only algorithms.
A value can be changed without editing a file:

```bash
python spinor_mobius.py --lang en --snapshot 9.5 --set camera.elevation_deg=30 --set style.accent='"#ffcc00"'
python spinor_mobius.py --lang en --config my_config.toml
```

`--set section.key=value` is repeatable (the value is a Python/TOML literal), `--config` takes another file with the same
structure, a misspelt key is an error (see `../dvconfig.py`, tests in `../test_dvconfig.py`).

The film exists in two separate versions, English and Russian (`--lang en`, `--lang ru`); all formulas are typeset with matplotlib
mathtext (LaTeX), words stay outside the formulas. (The first version of the film showed the captions in both languages at once.)

Run from this directory (numpy, matplotlib and ffmpeg are required):

```bash
./render.sh --lang en                # media/spinor_mobius_en.mp4, 20 s
./render.sh --lang ru                # media/spinor_mobius_ru.mp4
./render.sh --lang en --preview      # 6 s low-resolution check
./render.sh --lang en --snapshot 9.5 # one PNG at film time 9.5 s
python -m unittest test_spinor_mobius
```
