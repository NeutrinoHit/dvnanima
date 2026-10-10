# Photomultiplier: electron multiplication

Two scenes, 40 s.

1. **One photoelectron.** A photon knocks an electron out of the photocathode; at each of `N = 10` dynodes an
   electron knocks out `Poisson(delta)` secondary electrons, `delta = 4`, so `n_j` is a branching process with
   `<n_j> = n_0 delta^j` and the gain is `delta^N ~ 10^6`. The bars show the random counts against `n_0 delta^j`;
   the counter shows the exact number of electrons (at most 36 per stage are drawn).
2. **Pulse height.** Three scintillation flashes with 2, 6 and 12 photoelectrons: the anode pulses are
   (statistically) proportional to the number of photoelectrons.

```bash
./render.sh --lang en         # media/pmt_multiplication_en.mp4
./render.sh --lang ru         # media/pmt_multiplication_ru.mp4
./render.sh --lang en --preview
./render.sh --lang en --snapshot 12
python -m unittest test_pmt_multiplication
```

The older `pmt` (Manim, field map of the first dynode) stays in the gallery.

The film exists in two separate versions, English and Russian (`--lang en`, `--lang ru`); all formulas and labels are typeset with matplotlib mathtext (LaTeX), words stay outside the formulas.

## Configuration

Every number of the film (frame size and encoder, timeline, physics, sizes of the dots, lines and fonts, the layout of the frame,
colours, opacities) is in `config.toml` (10 sections: `video`, `physics`, `tube`, `events`, `timeline`, `tube_axes`, `bars_panel`, `pulse_panel`, `layout`, `style`); all the words and formulas are in `texts.toml` (one table per
language, the same keys in both). The script contains only algorithms. A value can be changed without editing a file:

```bash
python pmt_multiplication.py --lang en --snapshot 20 --set style.background='"#000000"' --set video.crf=18
python pmt_multiplication.py --lang en --config my_config.toml
```

`--set section.key=value` is repeatable (the value is a Python/TOML literal), `--config` takes another file with the same
structure, a misspelt key is an error (see `../dvconfig.py`, tests in `../test_dvconfig.py`).
