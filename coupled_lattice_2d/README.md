# A 2D lattice of coupled oscillators

126 s (112 s of content and seven 2 s title cards), two languages (`--lang en`, `--lang ru`;
`media/coupled_lattice_2d_en.mp4`, `media/coupled_lattice_2d_ru.mp4`). The film follows the "field" part of the
classical field theory chapter of the book (the mattress, `field_as_mattress_2d`) and replaces the older Manim film
`../fields/coupled_oscillators_2d`. It has the same structure as `../coupled_chain` (chapter cards, springs drawn as
coils built in screen pixels).

`NX x NY = 14 x 10` masses `m = 1` joined by springs `k = 1`, fixed edge, displacements perpendicular to the plane.
Right: the map of the mode energies `E_ab` (the column `a`, the row `b`) and the dispersion `omega_ab` against
`|k| = pi sqrt((a/(NX+1))^2 + (b/(NY+1))^2)`.

| content time, s | part |
|---|---|
| 0-16 | the lattice appears; `L = sum m qdot^2/2 - sum_bonds k (q - q')^2/2` and `m qddot_ij = k (q_{i+1,j} + q_{i-1,j} + q_{i,j+1} + q_{i,j-1} - 4 q_ij)` |
| 16-46 | normal modes `(a, b) = (1,1), (2,1), (3,2), (6,5), (14,10)`: `q ~ sin(a pi i/(NX+1)) sin(b pi j/(NY+1)) cos(omega_ab t)`, `omega_ab^2 = 4k/m [sin^2(a pi/(2(NX+1))) + sin^2(b pi/(2(NY+1)))]` |
| 46-62 | a bump is released: circular waves, many modes, constant mode energies |
| 62-80 | two wave packets collide in the linear lattice and pass through each other |
| 80-100 | the same collision with the anharmonic bond potential `V(d) = k d^2/2 + alpha d^3/3 + beta d^4/4` (`alpha = 0.45`, `beta = 0.09`, convex): the energy is redistributed among the normal modes, the readout is the share that moved to other modes |
| 100-112 | `N -> infinity`: the lattice (14x10, 28x20, 56x40) becomes the field `phi(x, y, t)` (a 119x87 sampling of the exact continuum solution), `omega = c |k|` |

The chapter cards add 2 s before each part (`film_time(t)` converts the content times; the poster frame at content
58 s is film 70 s). The linear evolution is exact (from the modes); the anharmonic lattice is integrated with the
velocity-Verlet scheme. The continuum scene uses the exact solution of the wave equation with a fixed edge.

Tests: orthonormal modes, the frequencies are the eigenvalues of the lattice matrix, a mode is an eigenvector of the
force, the period of a mode, the speed of sound from the dispersion, the checkerboard highest mode, the linear solution
equals the integrator, constant mode energies, the mode energies sum to the lattice energy, the packets move towards
each other, a convex potential, conservation of the energy in the anharmonic lattice (2e-3) and the redistribution of
the energy among the modes, the speed of the continuum ring, the parts and the cards of the film, the texts in both
languages.

## Configuration

Every number of the film (frame size and encoder, timeline, physics, sizes of the dots, lines and fonts, the layout of the frame,
colours, opacities) is in `config.toml` (16 sections: `video`, `physics`, `timeline`, `intro`, `modes`, `pluck`, `collision`, `nonlinear`, `limit`, `map`, `dispersion`, `camera`, `grid`, `layout`, `fonts`, `style`); all the words and formulas are in `texts.toml` (one table per
language, the same keys in both). The script contains only algorithms. A value can be changed without editing a file:

```bash
python coupled_lattice_2d.py --lang en --snapshot 20 --set style.background='"#000000"' --set video.crf=18
python coupled_lattice_2d.py --lang en --config my_config.toml
```

`--set section.key=value` is repeatable (the value is a Python/TOML literal), `--config` takes another file with the same
structure, a misspelt key is an error (see `../dvconfig.py`, tests in `../test_dvconfig.py`).

Run from this directory (numpy, matplotlib, Pillow and ffmpeg are required):

```bash
./render.sh --lang en               # media/coupled_lattice_2d_en.mp4
./render.sh --lang ru
./render.sh --lang en --snapshot 70 # one PNG at film time 70 s
python -m unittest test_coupled_lattice_2d
```

All formulas and labels are typeset with matplotlib mathtext (LaTeX); words stay outside the formulas.
