# Spin and orbital structure of the f f̄ pair in Higgs decays

105 s (95 s of content and six 1.7 s title cards), two languages (`--lang en`, `--lang ru`; `media/higgs_ff_spin_en.mp4`,
`media/higgs_ff_spin_ru.mp4`). The film follows the subsection on the spin-orbital state of the pair in the chapter on the Higgs decays
of the book and the paper of N. Borodin and D. Naumov (February 2026). The physics is in `ffbar.py`, the surface renderer in `isorender.py`,
the film in `higgs_ff_spin.py`.

## What is shown (content times; add 1.7 s per card)

| content time, s | part |
|---|---|
| 0-16 | **Back-to-back pairs build a wave.** For both spins down on `z` the amplitude to emit the pair along `n` is `M ~ n_x + i n_y` (left: directions `n` as points, colour = phase). Right: the partial sums `(4 pi / N) sum w(n_k) exp(i p n_k.r)` in the plane `z = 0` for `N = 1 ... 2048` directions (a Halton set), then the exact integral `4 pi i j_1(pr) (r_x + i r_y)`: a vortex, dark core `j_1(0) = 0`, one turn of the phase, `L = 1`, `m = +1`. Back-to-back momenta do not forbid `L != 0`. |
| 16-48 | **Scalar `H(0+) -> f f̄`.** The surfaces `|psi|^2 = const` (outer translucent 0.05 N^2, inner 0.10 N^2) coloured by the phase, for one common spin axis `s`, `xi_i = rho_i s`: `rho1 = -rho2`: two lobes, `m = 0`; `rho1 = rho2 = +`: a ring, the phase winds once, `m = -1`; `rho1 = rho2 = -`: `m = +1` (gold arrow: the direction in which the phase grows). `m = -(rho1 + rho2)/2` because `J_s = 0`. Then a probe direction `rhat` at the polar angle `theta` from the axis sweeps `0 -> 90 deg -> 0`; the curves are the angular factor `(1/2)[1 + rho1 rho2 (1 - 2 cos^2 theta)]` of `|psi|^2` with the moving value: `(1 - rho1 rho2)` at `theta = 0` (only `m = 0`), `(1 + rho1 rho2)` at `pi/2` (only `m = +-1`). |
| 48-61 | **Pseudoscalar `A(0-)`.** The same cells morph into `psi ~ j_0(pr) chi^dag eta`: a ball, `L = 0`, singlet, only `rho1 = -rho2`, flat angular curves; `Gamma ~ beta^(2L+1)`: `beta^3` for `0+`, `beta` for `0-`. |
| 61-83 | **Mixed state** `eps1 psi_S + eps2 psi_P` with `xi1 = x`, `xi2 = y`: `|psi|^2 = (1/2) a^2 [...] + (1/2) b^2 [1 - xi1.xi2] + a b xi1.(rhat x xi2)`, `a = eps1 beta j_1`, `b = eps2 j_0`. `eps2/eps1` grows 0 -> 1 and the cloud leans; the cell with the spins exchanged (`C P` image) leans the other way. On the axis `+-(xi2 x xi1)`: `(1/2)(a +- b)^2`, they cancel completely where `a = b` (`pr = 2.04`). |
| 83-95 | **What a detector sees.** The projection on `|p>` keeps `|M(n)|^2` and loses the phase; the surviving spin-angle correlations `d Gamma/d Omega ~ 1 + xi1.xi2 - 2 (xi1.n)(xi2.n)` (0+), `~ 1 - xi1.xi2` (0-); spins of `tau+ tau-` are read from their decay products. |

Colour of all the pictures is the phase of `psi` (a closed colour wheel, legend at the side); the common factor `exp(-i omega t)` makes the
colours turn (`hue_period_s`), the moduli do not change. Everything is computed from the closed forms of the paper (checked against
explicit Dirac spinors, see below); the surfaces are ray-marched in `isorender.py`; the window `pr < 5` shows the first lobe of `j_1`
(the node of `j_1` is at `pr = 4.49`); the mixed state uses the window `pr < 6.4`. The mixing `eps2/eps1 = 1` and `beta = 0.996` (a b quark) are illustrative.

## Physics and conventions

`psi_{rho1 rho2}(r) = i m_f p/(16 pi^2 v) Int dOmega_n e^{i p n.r} ubar(pn, s1) Gamma v(-pn, s2)`, `Gamma = 1` (scalar) or `gamma_5` (pseudoscalar);
`Int dOmega e^{i p n.r} = 4 pi j_0(pr)`, `Int dOmega e^{i p n.r} n = 4 pi i j_1(pr) rhat`. In the unit `N = m_f p E/(2 pi v)`:
`psi_S = beta j_1(pr) chi^dag(s1) (sigma.rhat) eta(s2)`, `psi_P = phase j_0(pr) chi^dag(s1) eta(s2)`. All closed forms are checked numerically in `test_ffbar.py`
(product quadrature on the sphere with explicit Dirac-representation spinors). The orbital projection is `m = -(rho1 + rho2)/2`; the interference term of a scalar-pseudoscalar mixture is
`beta j_0 j_1 [Re(eps1 eps2^*) xi1.(rhat x xi2) - Im(eps1 eps2^*) (xi1 - xi2).rhat]`, the film shows the case of real `eps1 eps2`.

## Tests

`python -m unittest test_ffbar test_higgs_ff_spin` (59 tests): spinors and projectors; Dirac bilinears `ubar v = -2 p chi^dag sigma.n eta`, `ubar gamma5 v = 2 E chi^dag eta`;
spin-summed `|M|^2 = 2 m_H^2 m_f^2 beta^2/v^2`; the spin correlation of the book; `Gamma_S/Gamma_P = beta^2`; the angular integrals; `psi` from the numerical Fourier integral of the
Dirac amplitudes equals the closed forms; (15), (22) and the three-term density for complex couplings; the orbital projection `m = -(rho1+rho2)/2` for arbitrary axes (projection on `Y_1m`);
`theta = 0`, `pi/2`; the pseudoscalar is a pure `Y_00` and independent of `rhat`; norms `8 pi/3`, `4 pi/3`; the CP-odd term is odd under the exchange of the spins, under `r -> -r`, under `rho1 -> -rho1`,
vanishes for a common axis, changes sign with `eps2`, `(1/2)(a +- b)^2` on the axis, complete cancellation at `a = b`; pure `0+` and `0-` are symmetric under the exchange; the plane-wave sums converge
to the vortex; the renderer (hue shift equals phase shift, the donut has a hole, the dumbbell two lobes, the surface lies where `|psi|^2` equals the level); the timeline; the configuration (same text keys in both
languages, balanced `$`, no Cyrillic in formulas, no decimal point in Russian formulas, no numeric literals in the script, `--set` works); every frame at 20 sample times in both languages: no text leaves the frame and no two texts overlap.

## Configuration

Every number of the film (frame size and encoder, timeline, physics, rendering of the surfaces, palette, sizes of lines, dots and fonts, layout, colours, opacities) is in `config.toml`
(sections `video`, `physics`, `iso`, `palette`, `wheel`, `glyph`, `overlay`, `plot`, `partial_sums`, `timeline`, `fonts`, `layout`, `style`, `p1` ... `p5`); all the words and formulas are in `texts.toml`
(one table per language, the same keys in both). The scripts contain only algorithms. A value can be changed without editing a file:

```bash
python higgs_ff_spin.py --lang en --snapshot 40 --set iso.elevation_deg=40.0 --set palette.hue_period_s=8.0
python higgs_ff_spin.py --lang en --config my_config.toml
```

`--set section.key=value` is repeatable, `--config` takes another file with the same structure, a misspelt key is an error (see `../dvconfig.py`). The surfaces of the mixed state are cached in `cache/`
(independent of the language, not to be committed).

Run from this directory (numpy, scipy, matplotlib, Pillow and ffmpeg are required):

```bash
./render.sh --lang en               # media/higgs_ff_spin_en.mp4
./render.sh --lang ru
./render.sh --lang en --preview     # low-resolution check
./render.sh --lang en --snapshot 40 # one PNG (media/higgs_ff_spin_en.png) at film time 40 s
python -m unittest test_ffbar test_higgs_ff_spin
```

All formulas and labels are typeset with matplotlib mathtext (LaTeX); words stay outside the formulas; the Russian film uses the decimal comma.
