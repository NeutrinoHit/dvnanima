# The relativity of simultaneity

56 s, two languages (`--lang en`, `--lang ru`; `media/simultaneity_en.mp4`, `media/simultaneity_ru.mp4`).
A Minkowski diagram (`c = 1`) with three flashes A, B, C at `t = 0`, `x = -1, 0, +1` in the frame O. In a frame
O' that moves with the velocity `v`

```text
t'_i = gamma (t_i - v x_i) = -gamma v x_i,        x'_i = gamma x_i,
```

so the order is `C, B, A` for `v > 0` and `A, B, C` for `v < 0`; for `v = 0` the flashes are simultaneous.
The axes of O' are drawn as lines `x = v ct` (the `ct'` axis) and `ct = v x` (the `x'` axis); the lines of
simultaneity of O' are parallel to the `x'` axis, `ct = v x + ct'/gamma`. A dashed line from an event along its
line of simultaneity to the `ct'` axis shows how `t'_i` is read off. Two small panels show the same flashes
in each frame (a view from above, with expanding wave fronts); the clocks of both frames run together from
-0.9 to +0.9, the halos on the diagram appear when the line of simultaneity of O (white) or of O' (amber) passes
an event.

| time, s | part |
|---|---|
| 0-5 | the three events in O |
| 5.5-12.5 | `v = 0`: simultaneous |
| 15-24 | `v = +0.3`: C, B, A |
| 26.5-35.5 | `v = -0.5`: A, B, C |
| 37.5-52 | `v` sweeps from -0.6 to +0.6; the *spacelike* pair P, R (`dt^2 - dx^2 < 0`) changes its order, the *timelike* pair P, Q never does |
| 52-56 | summary: causality is safe, because the order changes only for events that no signal connects |

Tests: the order of the flashes, `t'_i = -gamma v x_i`, the invariance of the interval, the timelike pair
never changes its order for any `|v| < 1`, the spacelike pair does, the axes of O' (`x' = 0` on `x = v ct`,
`t' = 0` on `ct = v x`), the lines of simultaneity and the read-off point on the `ct'` axis.

## Configuration

Every number of the film (frame size and encoder, timeline, physics, sizes of the dots, lines and fonts, the layout of the frame,
colours, opacities) is in `config.toml` (9 sections: `video`, `events`, `timeline`, `velocity`, `diagram`, `panels`, `readouts`, `layout`, `style`); all the words and formulas are in `texts.toml` (one table per
language, the same keys in both). The script contains only algorithms. A value can be changed without editing a file:

```bash
python simultaneity.py --lang en --snapshot 20 --set style.background='"#000000"' --set video.crf=18
python simultaneity.py --lang en --config my_config.toml
```

`--set section.key=value` is repeatable (the value is a Python/TOML literal), `--config` takes another file with the same
structure, a misspelt key is an error (see `../dvconfig.py`, tests in `../test_dvconfig.py`).

Run from this directory (numpy, matplotlib, Pillow and ffmpeg are required):

```bash
./render.sh --lang en               # media/simultaneity_en.mp4
./render.sh --lang ru
./render.sh --lang en --preview     # low-resolution check
./render.sh --lang en --snapshot 20 # one PNG at film time 20 s
python -m unittest test_simultaneity
```

This film replaces the Wikimedia animation used before.

All formulas and labels are typeset with matplotlib mathtext (LaTeX); words stay outside the formulas.
