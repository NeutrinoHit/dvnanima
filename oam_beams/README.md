# Twisted beams: orbital angular momentum

56 s, two languages (`--lang en`, `--lang ru`; `media/oam_beams_en.mp4`, `media/oam_beams_ru.mp4`). The film
follows the section on cylindrical (twisted) waves in the chapter on non-relativistic quantum mechanics.
Colour = phase of `psi` (the colour wheel), brightness = `|psi|`; the time dependence `exp(-i omega t)` makes the
pattern turn.

1. **0-15 s, transverse phase portraits** of the Bessel waves `psi = J_m(kappa rho) exp[i(m phi + k_z z - omega t)]`,
   `m = 0, 1, 2, 3`. The phase winds `m` times around the axis, on the axis (`m != 0`) it is undefined and
   `J_m(0) = 0`: a dark core. `L_z = -i hbar d/dphi` gives `m hbar`.
2. **15-33 s, the wave fronts in 3D.** The surfaces `m phi + k_z z - omega t = 2 pi n` are `m` intertwined
   helicoids (a screw with `m` threads), drawn as translucent ribbons for `m = 1, 2, 3, -2`; the sign of `m`
   reverses the screw. The transverse pattern is shown next to it.
3. **33-50 s, how to make such a beam.** A Gaussian beam passes a spiral phase plate (or a spatial light
   modulator showing `exp(i m phi)`): the field becomes `LG_0^m ~ (sqrt2 rho/w)^|m| exp(-rho^2/w^2) exp(i m phi)`,
   a ring of radius `w sqrt(|m|/2)` with the same winding.
4. **50-56 s**, `m = +2` and `m = -2`: the sign of `m` is the direction of the twist; each photon (or particle)
   carries `L_z = m hbar`.

Tests: the winding number of the phase is `m` (also for the Laguerre-Gauss beam and for the Gaussian beam behind
the plate), `<-i d/dphi> = m`, the dark core, the Bessel beam solves the Helmholtz equation, the first intensity
maximum of `J_m^2` and of the LG beam, the points of the helicoids satisfy `m phi + k_z z - omega t = 2 pi n`,
the `m` threads are separated by `2 pi/|m|`.

## Configuration

Every number of the film (frame size and encoder, timeline, physics, sizes of the dots, lines and fonts, the layout of the frame,
colours, opacities) is in `config.toml` (10 sections: `video`, `beam`, `grid`, `timeline`, `phase_image`, `helicoid`, `inset`, `panels`, `layout`, `style`); all the words and formulas are in `texts.toml` (one table per
language, the same keys in both). The script contains only algorithms. A value can be changed without editing a file:

```bash
python oam_beams.py --lang en --snapshot 20 --set style.background='"#000000"' --set video.crf=18
python oam_beams.py --lang en --config my_config.toml
```

`--set section.key=value` is repeatable (the value is a Python/TOML literal), `--config` takes another file with the same
structure, a misspelt key is an error (see `../dvconfig.py`, tests in `../test_dvconfig.py`).

Run from this directory (numpy, scipy, matplotlib, Pillow and ffmpeg are required):

```bash
./render.sh --lang en               # media/oam_beams_en.mp4
./render.sh --lang ru
./render.sh --lang en --preview     # low-resolution check
./render.sh --lang en --snapshot 20 # one PNG at film time 20 s
python -m unittest test_oam_beams
```

References: L. Allen et al., Phys. Rev. A 45, 8185 (1992); the review by Knyazev and Serbo, Phys. Usp. 61, 449 (2018)
(cited in the book). This film replaces the Wikimedia GIF used before.

All formulas and labels are typeset with matplotlib mathtext (LaTeX); words stay outside the formulas.
