# The gauge principle: two friends and a field

97 s (86 s of content and seven title cards), two languages (`--lang en`, `--lang ru`; `media/gauge_principle_en.mp4`, `media/gauge_principle_ru.mp4`).
The film follows the chapter "Gauge Invariance": the box "A Shift of Phase", the sections "QED in a New Way" and "The Covariant Derivative", and the subsections
"Wilson Line" and "Wilson Loop". It uses the book's conventions:

```
psi -> e^{i alpha} psi,        A_mu -> A_mu - (1/q) d_mu alpha,        D_mu = d_mu + i q A_mu,
U(y,x) = exp(-i q int_x^y A_mu dz^mu),        F_mu nu = d_mu A_nu - d_nu A_mu,        U_loop = exp(-i q oint A_mu dz^mu) = exp(-i q int F_12 dx dy).
```

Components of `A` are the covariant (lower-index) ones; for a spatial direction `-(1/2m) D_x^2 = (p - q A^x)^2 / 2m`. Units: `hbar = 1`, `q = 1`.

| content time, s | part |
|---|---|
| 0-15 | **Two friends and a phase.** An illustration (labelled as such): a theorist on the Earth sends two experimenters to the points `x` and `y`; each reads the field as a complex number (a phase dial) and sets his own convention `alpha(x)`, `alpha(y)`. The two numbers `|psi(x) - psi(y)|` (one convention) and `|e^{i alpha(x)} psi(x) - e^{i alpha(y)} psi(y)|` (own conventions) are computed from a plane wave |
| 15-30 | **Phases in an interference experiment.** Two paths (two slits), the exact paraxial field behind the barrier and the pattern on the screen. A common phase of both paths (the dials turn) leaves the pattern unchanged (change 0.000); a phase shifter in one path moves the fringes by `delta / 2 pi` periods, and after `2 pi` the pattern returns; the shift is measured on the computed pattern |
| 30-44 | **A convention at every point.** A wave packet of a non-relativistic particle on a lattice (1364 sites, exact time evolution). The experimenter's state `e^{i alpha(x)} psi` is evolved with the free equation: the packet drifts with `(k_0 + <alpha'>)/m` instead of `k_0/m` and is distorted. Centres and drift velocities are measured and set against the formula |
| 44-59 | **The compensating field.** The links `U = exp(+i q theta_j)`, `theta_j = int A_x dx`, with `A_x = -alpha'/q` make the lattice equation covariant: the packet follows the theorist's prediction (`|psi'|^2 = |psi|^2`, `psi' = e^{i alpha} psi`) to rounding errors; the mismatch of `D psi'` with `e^{i alpha} D psi` is 1e-13 while that of the naive `d psi'` is 0.7 |
| 59-77 | **The field strength and the loop.** A plane with the two-slit apparatus: arrows of `A`, the map of `F_12`, two loops, the dials at the slits and the fringes on the screen. A pure-gauge `A = -grad(alpha)/q` has `F = 0`, loops 0 and the fringes do not move; a thin flux tube gives `F != 0`, the loop around it `q Phi = 0.7 pi` (line integral of `A` and area integral of `F` agree), and the fringes move by `q Phi / 2 pi = 0.35` period; a new `alpha(x, y)` changes every arrow and every dial but not `F`, the loops or the fringes |
| 77-86 | **Summary.** `local choice of phase => field A_mu => interaction`, `D_mu`, the QED Lagrangian, a sentence about `SU(2)` and `SU(3)` |

Chapter cards precede every part (the content times above exclude them; `film_time(t)` converts).

## Physics model (`gauge_physics.py`)

* Two paths: each path is the exact paraxial Gaussian beam of a slit, a solution of `2 i k d_x psi + d_y^2 psi = 0`; the pattern is `|e^{i th_1} psi_1 + e^{i th_2} psi_2|^2`. The colour map of the field is `|psi|^2` with a small modulation by `cos(arg psi)` (the wave crests).
* Particle on a line: lattice with hard walls (the walls are never reached, the view is a part of the box), `H = -(1/2m) D^2` with the covariant lattice Laplacian built from the Wilson-line links; the time evolution is exact (diagonalisation of `H`), the norm is conserved to 1e-10. The gauge transformation acts on the links by `theta_j -> theta_j - (alpha_{j+1} - alpha_j)/q`.
* Plane: `A = (-g(r) y, g(r) x)`, `g = Phi (1 - e^{-r^2/2 s^2}) / (2 pi r^2)`, `F_12 = Phi e^{-r^2/2 s^2} / (2 pi s^2)`; gauge transformations with a smooth `alpha(x, y)` (two sine terms); line integrals along polygons (Gauss-Legendre) and area integrals (tensor Gauss-Legendre). The phase of path `j` at the screen is `-q int_source^slit A + alpha(source) - q int_slit^screen A` along straight lines; its gauge invariance is computed, not assumed. Because the Gaussian tube has tails, the loop around it gives 0.6996 pi instead of 0.7 pi (the part of the flux outside the loop).

What is simplified: the particle is a non-relativistic scalar wave function (the Lagrangian shown in part 3 is the Dirac one, as in the book); the simulation of part 3-4 is one-dimensional, so the field strength appears only in part 5; the paths of part 5 are straight lines from the slits to the screen point (the stationary-phase paths of the paraxial beams); part 1 is an illustration (the numbers on the screen are computed).

## Tests (`test_gauge_principle.py`, 61 tests)

Part 1: the difference of two numbers depends on the conventions, a common phase changes nothing, `psi(y) - U psi(x)` is well defined. Part 2: the beam solves the paraxial equation (finite differences), conservation of the power of a beam, **a common phase leaves the pattern unchanged (1e-12)**, the pattern depends only on `theta_1 - theta_2`, **fringe shift = `delta / 2 pi` periods**, the return after `2 pi`, the fringe period `lambda L / d`. Parts 3-4: Hermiticity, unitarity, energy conservation, the group velocity of the true packet, **the drift `(k_0 + <alpha'>)/m` of the naively rephased packet**, the naive packet is not the rephased true packet, **covariant evolution = rephased true evolution (1e-9)**, **`D psi` is covariant while `d psi` is not**, the links transform like Wilson lines, covariance for a general `A` and a random `alpha`, second-order convergence of the lattice covariant derivative and of `H` to the continuum ones. Part 5: the curl and circulation of the tube, **pure gauge has `F = 0`**, **`F` is gauge invariant**, **Stokes (line integral = flux integral, 1e-6)**, **loop = `q Phi` and the loop that does not enclose the tube = 0**, **loop gauge invariance (1e-9)**, **fringes shift by `q Phi / 2 pi` in any gauge and pattern invariance (1e-9)**, the dials turn with `alpha` at the slits. Film: timeline (60-110 s, parts, events inside parts), configuration (same keys in both languages, balanced `$`, placeholders inside formulas, no Russian words in formulas, video section, `--set` and a misspelt `--set`), texts inside the frame and **no overlapping texts** in 41 frames of both languages, the decimal comma in Russian, the numbers on the screen are the computed ones, a snapshot in both languages.

## Run

```bash
./render.sh --lang en               # media/gauge_principle_en.mp4
./render.sh --lang ru
./render.sh --lang en --preview
./render.sh --lang en --snapshot 40 # one PNG (film time 40 s)
python -m pytest test_gauge_principle.py
```

(numpy, scipy, matplotlib, Pillow and ffmpeg are required; a full render takes several minutes per language.)

## Configuration

Every number of the film (frame size and encoder, timeline, the physics of the three models, the geometry of the panels, line widths, font sizes, colours, opacities, the geometry of the rocket and of the Earth) is in `config.toml`
(sections `video`, `timeline`, `style`, `fonts`, `layout`, `dial`, `interf`, `chain`, `plane`, `rows`, `part1` ... `part6`); all the words and formulas are in `texts.toml` (one table per language, the same keys in both).
The scripts contain only algorithms (`gauge_physics.py`: the physics, `gauge_principle.py`: the film). A value can be changed without editing a file:

```bash
python gauge_principle.py --lang en --snapshot 70 --set plane.flux_over_pi=0.5 --set video.crf=18
python gauge_principle.py --lang en --config my_config.toml
```

`--set section.key=value` is repeatable (the value is a Python/TOML literal), `--config` takes another file with the same structure, a misspelt key is an error (see `../dvconfig.py`).
