# A 3D lattice of coupled oscillators

119 s (107.5 s of content and seven 1.6 s title cards), two languages (`--lang en`, `--lang ru`;
`media/coupled_lattice_3d_en.mp4`, `media/coupled_lattice_3d_ru.mp4`). The film follows the "field" part of the classical
field theory chapter of the book (`field_as_mattress_3d`) and replaces the older Manim film
`../fields/coupled_oscillators_3d`. It has the same structure as `../coupled_chain` and `../coupled_lattice_2d` (chapter
cards, the same chapters, the same physics).

`NX x NY x NZ = 14 x 8 x 8` masses `m = 1` (896 of them) joined by springs `k = 1`, fixed boundary (the frame of the box),
a scalar displacement of every mass. A mass is a dot whose size and brightness are the displacement (a quiet mass is a
tiny dim dot) and whose colour is the sign; the springs are thin faint lines. Right: the spectrum of the mode energies
(the bars are bins of `omega`, the gold bar is the mode that is shown) and the dispersion `omega` against `|k|`.

| content time, s | part |
|---|---|
| 0-11.5 | the box appears layer by layer; `L = sum m qdot^2/2 - sum_bonds k (q - q')^2/2` and `m qddot = k (sum of the 6 neighbours - 6 q)` |
| 11.5-41.5 | normal modes `(a, b, c) = (1,1,1), (2,1,1), (2,2,1), (5,3,2), (14,8,8)`: `q ~ sin sin sin cos(omega_abc t)`, `omega^2 = 4k/m [sin^2(a pi/(2(NX+1))) + sin^2(b pi/(2(NY+1))) + sin^2(c pi/(2(NZ+1)))]` |
| 41.5-57.5 | a bump is released at the centre: a spherical wave, many modes, constant mode energies |
| 57.5-75.5 | two wave packets collide in the linear lattice and pass through each other |
| 75.5-95.5 | the same collision with the anharmonic bond potential `V(d) = k d^2/2 + alpha d^3/3 + beta d^4/4` (`alpha = 0.45`, `beta = 0.09`, convex): the energy is redistributed among the normal modes, the readout is the share that moved to other modes |
| 95.5-107.5 | `N -> infinity`: the lattice is refined (14x8x8, 28x16x16, 56x32x32 masses, the dots get smaller and smaller) and becomes the field `phi(x, y, z, t)` (the exact continuum solution sampled at the masses), `omega = c |k|` |

The chapter cards add 1.6 s before each part (`film_time(t)` converts the content times; the poster frame at content
53 s is film 59 s). The linear evolution is exact (from the modes); the anharmonic lattice is integrated with the
velocity-Verlet scheme.

Tests: orthonormal modes, the frequencies are the eigenvalues of the lattice matrix, a mode is an eigenvector of the
force, the transform is its own inverse, the period of a mode, the speed of sound from the dispersion, the highest
frequency, the linear solution equals the integrator, constant mode energies, the mode energies sum to the lattice
energy, the packets move towards each other, a convex potential, conservation of the energy (2e-3) and the
redistribution of the energy among the modes, the spectrum adds up to 1, the speed of the spherical wave in the
continuum, the parts and the cards of the film, the texts in both languages.

## Configuration

Every number of the film (frame size and encoder, timeline, physics, sizes of the dots, lines and fonts, the layout of the frame,
colours, opacities) is in `config.toml` (14 sections: `video`, `physics`, `timeline`, `intro`, `modes`, `pluck`, `nonlinear`, `limit`, `spectrum`, `dispersion`, `camera`, `layout`, `fonts`, `style`); all the words and formulas are in `texts.toml` (one table per
language, the same keys in both). The script contains only algorithms. A value can be changed without editing a file:

```bash
python coupled_lattice_3d.py --lang en --snapshot 20 --set style.background='"#000000"' --set video.crf=18
python coupled_lattice_3d.py --lang en --config my_config.toml
```

`--set section.key=value` is repeatable (the value is a Python/TOML literal), `--config` takes another file with the same
structure, a misspelt key is an error (see `../dvconfig.py`, tests in `../test_dvconfig.py`).

Run from this directory (numpy, matplotlib, Pillow and ffmpeg are required):

```bash
./render.sh --lang en               # media/coupled_lattice_3d_en.mp4
./render.sh --lang ru
./render.sh --lang en --snapshot 59 # one PNG at film time 59 s
python -m unittest test_coupled_lattice_3d
```

All formulas and labels are typeset with matplotlib mathtext (LaTeX); words stay outside the formulas.
