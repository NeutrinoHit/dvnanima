# The path integral

Three scenes, 40 s.

1. **Slits.** The amplitude to reach the detector is the sum over all paths through screens with 3, 7
   and 5 slits (3, 21 and 105 paths). The phase of a path is `2 pi L / lambda`, `L` being its length. The
   phasors are drawn head to tail, the white arrow is the amplitude, the yellow profile along the detector
   is `|A|^2`, computed from the same sum. The detector moves, the interference pattern is read off.
2. **All paths.** With infinitely many screens and slits all paths contribute. For a free particle the
   paths `x_a(t) = a sin(pi t / T)` from `(0, 0)` to `(T, 0)` have the action
   `S(a) = (m pi^2 / 4T) a^2`, so the phase `S/hbar = c a^2` grows quadratically with the deviation `a` from
   the classical path. The phasors `e^{i c a^2}` added head to tail draw the Cornu spiral: near `a = 0` they
   point the same way (stationary action), far from it they curl up and cancel. The sum converges to
   `sqrt(pi / c) e^{i pi / 4}`.
3. **Classical limit.** When `hbar` decreases (`c` grows), the spiral shrinks and only the paths within
   `|a| < sqrt(pi / c)` matter.

Schematic: the family of paths is one-parameter, the action is quadratic in `a`, and the values of `c` are
chosen for clarity (`hbar ~ 1/c`).

## Configuration

Every number of the film (frame size and encoder, timeline, the slits and the wavelength, the action and the values of `c`, sizes of the
dots, lines and fonts, the layout of the frame, colours, opacities, the number formats of the readouts) is in `config.toml`
(7 sections: `video`, `model`, `timeline`, `layout`, `formats`, `fonts`, `style`); all the words and formulas are in `texts.toml`
(one table per language, the same keys in both). The script contains only algorithms. A value can be changed without editing a file:

```bash
python path_integral.py --lang en --snapshot 20 --set model.c_phase=12 --set video.crf=18
python path_integral.py --lang en --config my_config.toml
```

`--set section.key=value` is repeatable (the value is a Python/TOML literal), `--config` takes another file with the same
structure, a misspelt key is an error (see `../dvconfig.py`, tests in `../test_dvconfig.py`).

The film exists in two separate versions, English and Russian (`--lang en`, `--lang ru`); all formulas and labels are typeset
with matplotlib mathtext (LaTeX), words stay outside the formulas. The first version of the film was bilingual (an English and a Russian
line in the same frame); the two versions now have one language each.

Run from this directory (numpy, matplotlib and ffmpeg are required):

```bash
./render.sh --lang en                # media/path_integral_en.mp4, 40 s
./render.sh --lang ru                # media/path_integral_ru.mp4
./render.sh --lang en --preview      # low-resolution check
./render.sh --lang en --snapshot 12  # one PNG at film time 12 s
python -m unittest test_path_integral
```

References: R. P. Feynman, Rev. Mod. Phys. 20, 367 (1948); R. P. Feynman and A. R. Hibbs, Quantum Mechanics
and Path Integrals (1965).
