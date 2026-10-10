# A kick to an electron: the front of an electromagnetic wave

A charge at rest is instantaneously given the velocity `v` along `x` at `t = 0` (`c = 1`). The exact
field of this process (Lienard-Wiechert) has three zones:

- `r > ct`: the information about the kick has not arrived; the field is the Coulomb field of the
  charge at rest at the origin, the lines are radial from the origin;
- `r < ct`: the field of a charge moving uniformly; its lines are straight and radial from the present
  position `x_p = v t`, crowded towards the plane perpendicular to `v` (Lorentz contraction);
- `r = ct`: a thin shell, the front of the electromagnetic wave, which carries the field transverse
  to the radius and joins the two fields.

Gauss's law fixes how the lines are joined. The flux inside the cone of half-angle `psi` around the
velocity seen from the present position is `(1/2)(1 - cos(psi)/sqrt(1 - beta^2 sin^2 psi))`; it equals
`(1 - cos(theta))/2`, the flux inside the cone of half-angle `theta` seen from the origin. A line that
leaves the charge at the angle `psi` therefore runs to the shell, goes along it (an arc of `r = ct`) and
continues outwards, radially from the origin, at the angle `theta`:

```text
cos(theta) = cos(psi) / sqrt(1 - beta^2 sin^2 psi),      tan(theta) = tan(psi) / gamma.
```

The lines are drawn with equal flux, so their density is the strength of the field, and the shell shines
where many lines run along it. The film shows the upper half-plane (as in the figure of the book) for
`beta = 0.30` and `beta = 0.85`; the kick is instantaneous, so the front is infinitely thin.

## Configuration

Every number of the film (frame size and encoder, the two runs with `beta = 0.30` and `beta = 0.85`, timeline, number of field lines,
geometry of the frame, sizes of the lines and fonts, colours, opacities) is in `config.toml` (6 sections: `video`, `model`, `timeline`,
`layout`, `fonts`, `style`); all the words and formulas are in `texts.toml` (one table per language, the same keys in both).
The script contains only algorithms. A value can be changed without editing a file:

```bash
python electron_kick.py --lang en --snapshot 6.8 --set model.n_lines=30 --set style.background='"#000000"'
python electron_kick.py --lang en --config my_config.toml
```

`--set section.key=value` is repeatable (the value is a Python/TOML literal), `--config` takes another file with the same
structure, a misspelt key is an error (see `../dvconfig.py`, tests in `../test_dvconfig.py`).

The film exists in two separate versions, English and Russian (`--lang en`, `--lang ru`); all formulas are typeset with matplotlib
mathtext (LaTeX), words stay outside the formulas. (The first version of the film showed every caption in both languages at once.)

Run from this directory (numpy, matplotlib and ffmpeg are required):

```bash
./render.sh --lang en                 # media/electron_kick_en.mp4, 20 s
./render.sh --lang ru                 # media/electron_kick_ru.mp4
./render.sh --lang en --preview       # low-resolution check
./render.sh --lang en --snapshot 6.8  # one PNG at film time 6.8 s
python -m unittest test_electron_kick
```

The tests check the flux continuity, `tan(theta) = tan(psi)/gamma`, that the inner lines end on the shell,
that the arcs vanish for a tiny kick, and the consistency of the configuration and of the two text tables.
