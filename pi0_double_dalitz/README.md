# pi0 -> e+ e- e+ e-: E and B, the planes of the pairs, scalar vs pseudoscalar

A neutral pion decays into two photons and each virtual photon turns into an e+ e- pair (the "double
Dalitz" decay). The pair spans a plane that contains the polarization (the electric field **E**) of its
photon; **B = k x E** is perpendicular to it. The planes have a different orientation in every event; the
measured quantity is the angle `phi` between them. The film has two languages: `--lang en` and `--lang ru`
(`media/pi0_double_dalitz_en.mp4`, `media/pi0_double_dalitz_ru.mp4`), 54 s each.

1. **Pseudoscalar (pion, P = -1), 0-24 s.** The first event is shown slowly: two photons run along the axis
   with their `E` (colour) and `B` (gold) waves, then each converts into a pair and its plane appears; the
   triads `E_i, B_i` are drawn as in the figure of the book (`neutral-pion-two-planes`). The effective
   interaction and the amplitude

   ```text
   L = (alpha / pi f_pi) pi0 E.B,     M ~ E1.B2 + E2.B1 = (m^2/2) khat.(e1 x e2)  ~  sin(phi),
   Gamma = alpha^2 m^3 / (64 pi^3 f_pi^2) = 7.8 eV,
   ```

   so `|M|^2 ~ sin^2 phi`: perpendicular polarizations, perpendicular planes. The events then accelerate and
   the histogram of `phi` builds up, `dN/dphi ~ 1 - a cos 2phi` (humps at `pi/2` and `3 pi/2`).
2. **Scalar (P = +1), 24-47 s.** `L = (g/2) S (E^2 - B^2)`, `M ~ E1.E2 - B1.B2 = 2 E1.E2 ~ cos(phi)`,
   `Gamma = g^2 m^3 / (64 pi)`: the planes prefer to be parallel, `dN/dphi ~ 1 + a cos 2phi` (humps at `0` and
   `pi`). Same sequence: one slow event, then the accumulation.
3. **Summary, 47-54 s.** Both configurations side by side and the two histograms on top of each other.

The amplitude `a = 0.35` is illustrative: it reproduces the height of the humps of the KTeV histogram of the
book (fig. 40.6). The real `e+e-` pair is a *diluted* analyzer of the photon polarization (an ideal analyzer
would give `a = 1`). The picture is schematic: the opening angle of a pair is exaggerated and the events are
accelerated.

Checks in the tests: `E` is perpendicular to `B` and to the photon, `E1.B2 + E2.B1 = 2 sin(phi)` and
`E1.E2 - B1.B2 = 2 cos(phi)` for back-to-back photons, the sampled distributions have the second harmonic
`<cos 2phi> = -+ a/2`, and the width formula gives 7.76 eV.

## Configuration

Every number of the film (frame size and encoder, timeline, physics, sizes of the dots, lines and fonts, the layout of the frame,
colours, opacities) is in `config.toml` (14 sections: `video`, `timeline`, `model`, `events`, `camera`, `photons`, `pairs`, `triad`, `particle`, `dial`, `hist`, `summary`, `layout`, `style`); all the words and formulas are in `texts.toml` (one table per
language, the same keys in both). The script contains only algorithms. A value can be changed without editing a file:

```bash
python pi0_double_dalitz.py --lang en --snapshot 20 --set style.background='"#000000"' --set video.crf=18
python pi0_double_dalitz.py --lang en --config my_config.toml
```

`--set section.key=value` is repeatable (the value is a Python/TOML literal), `--config` takes another file with the same
structure, a misspelt key is an error (see `../dvconfig.py`, tests in `../test_dvconfig.py`).

Run from this directory (numpy, matplotlib, Pillow and ffmpeg are required):

```bash
./render.sh --lang en            # media/pi0_double_dalitz_en.mp4
./render.sh --lang ru
./render.sh --lang en --preview  # low-resolution check
./render.sh --lang en --snapshot 28   # one PNG at film time 28 s
python -m unittest test_pi0_double_dalitz
```

All formulas and labels are typeset with matplotlib mathtext (LaTeX); words stay outside the formulas.
