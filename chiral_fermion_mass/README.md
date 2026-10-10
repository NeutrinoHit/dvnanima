# Chiral waves and the fermion mass

89.6 s (80 s of content and six 1.6 s title cards), two languages (`--lang en`, `--lang ru`; `media/chiral_fermion_mass_en.mp4`,
`media/chiral_fermion_mass_ru.mp4`). The film follows the section "Coupled chiral waves give birth to mass" of the chapter
"Where does a particle get its mass?" (`SymmetryBreaking.tex`, section "Fermion mass, or coupled pendulums") and uses its notation:
`psi = (psi_L, psi_R)`, `gamma^0 = sigma^1`, `gamma^1 = -i sigma^2`, `gamma_5 = sigma^3`, the Dirac equation in the chiral form

    i (d_t + d_x) psi_R = m psi_L,      i (d_t - d_x) psi_L = m psi_R,      H(p) = [[-p, m], [m, p]],      E = +-sqrt(p^2 + m^2),

and, for the last part, the Yukawa term of the Standard Model chapter, `-f (Lbar e_R phi + h.c.)`, `m = f v / sqrt(2)`.

## What is exact, what is schematic

Exact: the (1+1)-dimensional Dirac equation is integrated in Fourier space (`dirac1d.py`; periodic grid of 1024 points, length 64):
for a constant mass the step is the exact unitary `cos(E h) - i H sin(E h)/E`; a time dependent mass `m(t)` is a product of such
steps (midpoint rule, `h <= 0.004`), so the norm is conserved to rounding error. Everything drawn (`Re psi`, `|psi|`, `P_R`, `P_L`,
`<gamma_5> = P_R - P_L`, the centre `<x>`, the weights of the branches `E > 0` and `E < 0`) is computed from that solution.

Schematic: the coil springs between the two lanes are a *symbol* of the coupling `m` (their strength follows `m`; they are not a
mechanical model); the condensate `v` is switched on by hand in part 2 (a thought experiment, shown in slow motion; in the real
Universe `v = 0` only in the hot early Universe); the units are `hbar = c = 1` with arbitrary lengths and masses (toy numbers:
`v = 3 sqrt(2)`, so that `f = 1` gives `m = 3`); the Higgs potential of part 5 is a sketch. The film says so in the captions/labels.

## Parts

| content time, s | part |
|---|---|
| 0-13 | `m = 0`: `psi_R` runs to the right, `psi_L` to the left, both at `c`, `<gamma_5> = +1, -1` are constant; the dispersion `E = +-p`; chirality = helicity = direction of motion in 1+1 |
| 13-30 | the condensate switches on (dial `v`, slow motion x0.15, `m = f v / sqrt(2) = 3`): a pure `psi_R` packet (`p = 3`) feeds `psi_L`; `<gamma_5>` oscillates with `2E = 8.49` and the oscillation dies out because the packet splits into the `E > 0` part (weight 0.88, `v = p/E`) and a small `E < 0` part (weight 0.12) that runs back; the mean `<gamma_5> = 0.53 < p/E`; the centre falls behind the light front |
| 30-48 | a pure `psi_R` wave at rest, `f = 0.5, 1, 2` (`m = 1.5, 3, 6`): `<gamma_5> ~ cos(2 m t)`, the flip frequency `2m` is the gap between `E = +-m`; two coupled pendulums |
| 48-66 | wave packets built from the eigenvectors of `H`: `psi_L/psi_R = m/(E+p)`, `<gamma_5> = p/E = v` is constant (in 1+1 `gamma_5` is the velocity operator), heavier = slower (`m = 0, 3, 6`: `v = 1, 0.70, 0.45`), the slopes of `E(p)` |
| 66-80 | why the spring exists: the mass term couples an `SU(2)` doublet `L` to a singlet `e_R` and is forbidden; the Yukawa term with the Higgs doublet is allowed and `<phi> = (0, v/sqrt(2))` turns it into `m = f v / sqrt(2)` |

## Layout of the picture

Two "sheets" (lanes): `psi_R` (warm, top) and `psi_L` (cold, bottom); the line is `Re psi`, the translucent area is `|psi|`.
Bottom left: the chirality `<gamma_5>(t)` (the area is warm for `> 0`, cold for `< 0`). Right: `E(p)` with the branches coloured by
their chirality (`p/E`), the dots mark the wave (area = probability on the branch), the dial `v`, `f` and `m` above. Part 4 draws
the density of the right chirality up and of the left chirality down on three tracks.

Tests (`python -m unittest test_chiral_fermion_mass`, 41 tests): eigenvalues `+-sqrt(p^2+m^2)`, `H^2 = E^2`, eigenvectors, their
chirality `+-p/E`, the ratio `m/(E+p)`; unitarity (constant and switched-on mass); a massless packet moves with `c`, keeps its
shape and chirality; the group velocity is `p/E` (exactly the mean of `p/E` over the packet; 0.3 % below `p0/E0` because of the
curvature of `v(p)`); the centre moves with `<gamma_5>` (velocity operator); a plane wave flips as `p^2/E^2 + (m^2/E^2) cos 2Et` with a
spectrum peaking at `2E`; at rest `P_R = cos^2 mt`; the branch weights `(1 +- p/E)/2` after a sudden mass, the two parts run with `+-p/E`,
the centre with `p^2/E^2`; the film: parts, cards, slow motion = ramp, dial, `m = f v/sqrt 2`, flips in part 3, race in part 4, no
overlapping or off-frame texts in both languages at 22 times, the decimal comma in Russian, captions fit the frame, texts/keys.

## Configuration

Every number of the film (frame size and encoder, timeline, physics, sizes of the lines, dots and fonts, the layout of the frame,
colours, opacities) is in `config.toml` (sections: `video`, `grid`, `higgs`, `packet`, `timeline`, `part1`-`part5`, `stage`, `springs`,
`dial`, `chirality`, `dispersion`, `potential`, `layout`, `fonts`, `style`); all the words and formulas are in `texts.toml` (one table per
language, the same keys in both; `@name@` marks a number inserted by the script). The script contains only algorithms. A value can be changed without editing a file:

```bash
python chiral_fermion_mass.py --lang en --snapshot 40 --set style.background='"#000000"' --set video.crf=18
python chiral_fermion_mass.py --lang en --config my_config.toml
```

`--set section.key=value` is repeatable (the value is a Python/TOML literal), `--config` takes another file with the same
structure, a misspelt key is an error (see `../dvconfig.py`, tests in `../test_dvconfig.py`).

Run from this directory (numpy, matplotlib, Pillow and ffmpeg are required):

```bash
./render.sh --lang en                 # media/chiral_fermion_mass_en.mp4
./render.sh --lang ru
./render.sh --lang en --preview       # low-resolution check
./render.sh --lang en --snapshot 40   # one PNG at film time 40 s
python -m unittest test_chiral_fermion_mass
```

All formulas and labels are typeset with matplotlib mathtext (LaTeX); words stay outside the formulas.
