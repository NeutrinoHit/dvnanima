# Thomson's cathode-ray tube: e/m from crossed fields

One scene, 52 s, in two languages (`--lang en`, `--lang ru`; `media/thomson_tube_en.mp4`,
`media/thomson_tube_ru.mp4`). The electrons are accelerated in a gun (speed `v = 8`), fly through a region of
length `ell = 3` with an electric field `E` (between two plates, pointing down) and a magnetic field `B`
(into the page), and hit a fluorescent screen at the distance `D = 3.5`. The trajectories are integrated
exactly (Boris scheme, `e/m = 1`, negative charge: the electric force is up, the magnetic force down); nothing
is drawn by hand. The spot on the screen is the sum of the real hits (with a persistence of 0.16 s) and is
shown again in a front view.

| time, s | what happens | formula |
|---|---|---|
| 0-7 | no fields, the spot is in the centre | `mv^2/2 = eU` |
| 7-18 | `E` only: the spot goes up (towards the positive plate) | `y_E = (e/m) E ell (ell/2 + D) / v^2` |
| 18-29 | `B` only: the spot goes down | `y_B = -(e/m) B ell (ell/2 + D) / v` |
| 29-39 | both: `B` is swept through the balance and fine-tuned | `eE = evB`, `v = E/B` |
| 39-52 | balance, then `B` off: the deflection `y_E` gives `e/m` | `e/m = y_E E / (B^2 ell (ell/2 + D))` |

The measured value shown at the end is computed from the simulated spot with this formula (1.00 in the
units of the film). Thomson (1897) found `e/m` of the order of `10^11` C/kg, about 1800 times that of the
hydrogen ion; the modern value is `1.7588e11` C/kg. The units of the film are arbitrary.

Tests: the speed after the gun, zero deflection without fields, the electric and magnetic deflections against
the small-angle formulas, the balance at `v = E/B` (the sign of the spot changes with `B`), `e/m` recovered
from the measured deflection within 5 %, the beam staying between the plates.

## Configuration

Every number of the film (frame size and encoder, timeline, physics, sizes of the dots, lines and fonts, the layout of the frame,
colours, opacities) is in `config.toml` (7 sections: `video`, `physics`, `schedule`, `scene`, `front_view`, `layout`, `style`); all the words and formulas are in `texts.toml` (one table per
language, the same keys in both). The script contains only algorithms. A value can be changed without editing a file:

```bash
python thomson_tube.py --lang en --snapshot 20 --set style.background='"#000000"' --set video.crf=18
python thomson_tube.py --lang en --config my_config.toml
```

`--set section.key=value` is repeatable (the value is a Python/TOML literal), `--config` takes another file with the same
structure, a misspelt key is an error (see `../dvconfig.py`, tests in `../test_dvconfig.py`).

Run from this directory (numpy, matplotlib, Pillow and ffmpeg are required):

```bash
./render.sh --lang en               # media/thomson_tube_en.mp4
./render.sh --lang ru
./render.sh --lang en --preview     # low-resolution check
./render.sh --lang en --snapshot 14 # one PNG at film time 14 s
python -m unittest test_thomson_tube
```

Reference: J. J. Thomson, Phil. Mag. 44, 293 (1897). This film replaces the Wikimedia GIF used before.

All formulas and labels are typeset with matplotlib mathtext (LaTeX); words stay outside the formulas.
