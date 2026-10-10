# Circular polarization

52 s, two languages (`--lang en`, `--lang ru`; `media/circular_polarization_en.mp4`,
`media/circular_polarization_ru.mp4`). The film follows the section "Polarization" of the chapter on classical
fields: the two circular polarizations of the book

```text
eps_R = (1,  i, 0)/sqrt2  ->  E_R = (cos(wt - kz),  sin(wt - kz), 0),   photon spin S_z = +hbar,
eps_L = (1, -i, 0)/sqrt2  ->  E_L = (cos(wt - kz), -sin(wt - kz), 0),   photon spin S_z = -hbar.
```

Naming follows the book and particle physics: the handedness is the sign of the helicity, right = positive helicity =
clockwise rotation of `E` for the source (looking along the wave) = counterclockwise for the receiver (looking
towards the oncoming wave). Optics counts from the receiver and calls the same wave left-circular; the IEEE antenna
convention agrees with particle physics.

1. **0-15 s, right polarization, positive helicity** (`E_R`): a snapshot of `E(z)` in space (a *left-handed* helix: the
   azimuth `wt - kz` decreases with `z`) and, next to it, the two views of the rotation at a fixed point. In time the
   rotation is a right-handed screw about `k`.
2. **15-28 s, left polarization, negative helicity** (`E_L`): the same with the opposite sense.
3. **28-47 s, `E_R + E_L`**: the two rotating vectors are added head to tail; equal amplitudes give a linear
   polarization, the relative phase `delta` turns the axis (`-delta/2`), unequal amplitudes give an ellipse with the
   semi-axes `a_R + a_L` and `|a_R - a_L|`, and `a_L -> 0` a circle.
4. **47-52 s**: the wave as a flux of photons with `S_z = +-hbar`.

Tests: the formulas against the complex amplitudes `Re(eps e^{-i w t + i k z})`, the sense of rotation for each
helicity and view, the snapshot in space is a left-handed helix, the wave moves forward, `E_L + E_R` with equal
amplitudes is linear along `-delta/2`, the semi-axes and the tilt of the ellipse, the circle for `a_R = 0`.

## Configuration

Every number of the film (frame size and encoder, timeline, physics, sizes of the dots, lines and fonts, the layout of the frame,
colours, opacities) is in `config.toml` (12 sections: `video`, `wave`, `timeline`, `mix`, `camera`, `scene3d`, `panels`, `mixing`, `readouts`, `panel_text`, `layout`, `style`); all the words and formulas are in `texts.toml` (one table per
language, the same keys in both). The script contains only algorithms. A value can be changed without editing a file:

```bash
python circular_polarization.py --lang en --snapshot 20 --set style.background='"#000000"' --set video.crf=18
python circular_polarization.py --lang en --config my_config.toml
```

`--set section.key=value` is repeatable (the value is a Python/TOML literal), `--config` takes another file with the same
structure, a misspelt key is an error (see `../dvconfig.py`, tests in `../test_dvconfig.py`).

Run from this directory (numpy, matplotlib, Pillow and ffmpeg are required):

```bash
./render.sh --lang en               # media/circular_polarization_en.mp4
./render.sh --lang ru
./render.sh --lang en --preview     # low-resolution check
./render.sh --lang en --snapshot 10 # one PNG at film time 10 s
python -m unittest test_circular_polarization
```

This film replaces the Wikimedia GIF used before.

All formulas and labels are typeset with matplotlib mathtext (LaTeX); words stay outside the formulas.
