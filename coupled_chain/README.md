# A chain of coupled oscillators

126 s (112 s of content and seven 2 s title cards), two languages (`--lang en`, `--lang ru`; `media/coupled_chain_en.mp4`, `media/coupled_chain_ru.mp4`). The film
follows the section on N material points in the classical field theory chapter of the book and replaces the older
Manim film `../fields/coupled_oscillators_1d`.

N = 24 masses `m = 1` joined by springs `k = 1`, fixed ends. Bottom left: the energy `E_s` of every normal mode; bottom
right: the dispersion curve `omega_s` with the modes that take part marked.

Film times: each part of the table is preceded by a title card (the film title, then the chapters 1-6); the times below are the content times, `film_time(t)` in the script converts them (the poster frame, content 52 s, is film 60 s). The chain is drawn with coil springs built in screen pixels, so a stretched spring keeps its coil radius and gets a longer pitch.

| content time, s | part |
|---|---|
| 0-16 | the chain appears; the Lagrangian `L = sum [m xdot^2/2 - k (x_{i+1} - x_i)^2/2]` and the equations of motion `m xddot_j = k (x_{j+1} - 2 x_j + x_{j-1})`, `xddot + Omega x = 0` |
| 16-46 | normal modes `s = 1, 2, 4, 9, 24`: `x_j ~ sin(s pi j / (N+1)) cos(omega_s t)`, `omega_s = 2 sqrt(k/m) sin(s pi / (2 (N+1)))` |
| 46-62 | one bump is released: two waves, many modes, the energy of every mode is constant |
| 62-80 | two wave packets collide in the linear chain and pass through each other (free particles) |
| 80-100 | the same collision with the anharmonic (cubic and quartic) terms; the energy is redistributed among the normal modes, and this is the interaction, `V = k d^2/2 + alpha d^3/3 + beta d^4/4`, `d = x_{j+1} - x_j` (`alpha = 0.45`, `beta = 0.09`; `beta` keeps the potential convex, `alpha^2 < 3 beta k`); the dashed line is the linear solution, the readout is the share of the energy moved to other modes |
| 100-112 | `N -> infinity`: the masses become dense, the chain becomes the field `phi(x, t)`, the dispersion becomes `omega = c k` |

The modes are exact; the linear evolution is computed from the modes; the nonlinear chain is integrated with the
velocity Verlet scheme (`dt = 0.005`).

Tests: orthonormal modes, the frequencies are the eigenvalues of the matrix `Omega`, the modes diagonalize `Omega`,
a mode oscillates with its frequency, the speed of sound, the alternating highest mode, the linear solution equals the
integrator, constant mode energies in the linear chain, the packets move towards each other, a convex potential,
the nonlinear chain conserves the energy (2e-3) and moves energy between the modes, the bump splits into two waves, fixed
ends of the continuum field, the parts of the film follow each other, the texts of both languages are the same.

## Configuration

Every number of the film (frame size and encoder, timeline, physics, sizes of the dots, lines and fonts, the layout of the frame,
colours, opacities) is in `config.toml` (16 sections: `video`, `physics`, `timeline`, `intro`, `modes`, `pluck`, `collision`, `nonlinear`, `limit`, `spectrum`, `dispersion`, `camera`, `chain`, `layout`, `fonts`, `style`); all the words and formulas are in `texts.toml` (one table per
language, the same keys in both). The script contains only algorithms. A value can be changed without editing a file:

```bash
python coupled_chain.py --lang en --snapshot 20 --set style.background='"#000000"' --set video.crf=18
python coupled_chain.py --lang en --config my_config.toml
```

`--set section.key=value` is repeatable (the value is a Python/TOML literal), `--config` takes another file with the same
structure, a misspelt key is an error (see `../dvconfig.py`, tests in `../test_dvconfig.py`).

Run from this directory (numpy, matplotlib, Pillow and ffmpeg are required):

```bash
./render.sh --lang en               # media/coupled_chain_en.mp4
./render.sh --lang ru
./render.sh --lang en --snapshot 52 # one PNG at film time 52 s
python -m unittest test_coupled_chain
```

All formulas and labels are typeset with matplotlib mathtext (LaTeX); words stay outside the formulas.
