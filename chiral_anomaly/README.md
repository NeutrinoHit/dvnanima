# The chiral anomaly: where do the particles come from?

106 s (97 s of content and six 1.5 s title cards), two languages (`--lang en`, `--lang ru`; `media/chiral_anomaly_en.mp4`, `media/chiral_anomaly_ru.mp4`).
The film belongs to the chapter "Quantum anomalies" of volume 2 (the section "The chiral anomaly" and its "Qualitative discussion": the current `j^mu` must be conserved, `j_5^mu` pays).
It shows the anomaly as a flow of the levels of the Dirac sea (spectral flow); every number on the screen is counted from the levels.

Notation of the chapter: `j_5^mu = psibar gamma^mu gamma_5 psi`, `d_mu j_5^mu = (2 alpha / pi) E.B = (e^2 / 2 pi^2) E.B`, `alpha = e^2 / 4 pi`.

## What the film shows

| content time, s | part |
|---|---|
| 0-15 | A massless fermion on a ring of length `L`: right movers `E = +p`, left movers `E = -p`, `p = 2 pi n / L`, the filled Dirac sea, particles (filled levels above `E = 0`) and holes (empty levels below it), `Q = N_R + N_L`, `Q_5 = N_R - N_L`; without a field nothing changes. |
| 15-40 | An electric field shifts `p -> p + eEt`: the levels flow, a counter and a staircase count the levels that cross `E = 0`, `N_R = -N_L = eEL t / 2 pi`, `dQ_5/dt = eEL/pi`, `d_mu j_5^mu = (e/pi) E` per unit length. Ring picture (schematic): particles run one way, holes the other way. |
| 40-63 | The cut-off. (a) A window in the energy `|E| < Lambda` (gauge invariant): filled levels enter at the bottom of the right branch and leave at the bottom of the left one, the total number of filled levels is constant, `N_R - N_L` grows and does not depend on `Lambda`. (b) A fixed set of levels keeps `Q_5`, but the floor of the sea moves with the potential and the vacuum acquires a gauge-dependent current. (c) A crystal, `E = sin ka`: the two branches are the two Fermi points of one band, the levels flow around the zone, the number of electrons is constant; the zoom shows the limit `a -> 0` where the left point goes to infinity. |
| 63-80 | 3+1 dimensions: Landau levels. The lowest level of a massless fermion is a chiral 1+1 branch (`E = +p_z` right, `E = -p_z` left) with `eB/2 pi` states per unit area (a plate with `N_phi = 12` states); higher levels are gapped pairs. An electric field along `B` moves all `N_phi` ladders at once, `dn_5/dt = 2 (eB/2 pi)(eE/2 pi) = (e^2/2 pi^2) E.B = (2 alpha/pi) E.B`. |
| 80-97 | The triangle diagram, `d_mu j_5^mu = (2 alpha/pi) E.B` as the operator that creates two photons; for the neutral pion `N_c sum t_3 Q^2 = 1/2`, `d_mu j_5^{mu 3} = (alpha/pi) E.B`, `Gamma(pi0 -> gamma gamma) = alpha^2 m_pi^3 / (64 pi^3 f_pi^2) = 7.76 eV` against the measured `7.72 +- 0.12 eV` (log axis, with the estimate without the anomaly); the closing answer to the question of the title. |

Film times are shifted by the title cards (`film_time(t)` in the script converts content time to film time; the poster frame, content 31 s, is film 35.5 s).

## Physics model (`spectral_flow.py`)

* Ring of length `L`, charge `e`, field `E`: kinetic momentum `k_n(t) = 2 pi (n + theta)/L + eEt`, energies `E = +k` (right) and `E = -k` (left); the occupation of a level belongs to its label `n` (the electrons stay in their levels), the vacuum has every level with `E < 0` filled. `N_R`, `N_L` are counted level by level (filled levels above `E = 0` minus empty levels below it).
* Regularisations: a window in the kinetic energy `-Lambda_lo <= E <= Lambda_hi` (gauge invariant), a window in fixed labels (moves with the potential), a smooth weight `exp(-(E/Lambda)^2)`, and a tight-binding chain `E = sin k` (finite and exact).
* Landau levels: the Weyl Hamiltonian `chi sigma.(p - eA)`, `A_y = Bx`, `p_y = 2 pi m / L_y`, is diagonalised on a strip (Fourier derivative in `x`); the eigenvalues are `+p_z` (one chiral branch) and `+-sqrt(p_z^2 + 2 n eB)`; the states with the centre in a strip are counted: `eB/2 pi` per unit area.
* Pion: `A = alpha/(pi f_pi)`, `Gamma = A^2 m^3 / (64 pi)`; the measured width is `Br / tau * hbar`.

## Tests

`python -m pytest` (50 tests).

* `test_spectral_flow.py` (physics): the number of levels that cross `E = 0` is `eEL t/2 pi` in each branch (three different rings, whole numbers within one level, and the long-time rate within 0.2 %); the crossings are spaced by `T = 2 pi/(eEL)` and the counter steps exactly at them; `Q = 0`, `dQ_5/dt = eEL/pi` and `d(Q_5/L)/dt = e/pi`; after one period the spectrum repeats and the state has one more particle and one hole; the window in the energy gives the same flow for any cut-off (equal for whole numbers of spacings, within one level otherwise, symmetric and asymmetric windows); a smooth regulator converges to the same number; a fixed-label window keeps `Q_5` while the levels did cross; the ground-state current is bounded for the gauge-invariant cut and grows linearly with the potential for the fixed-label cut; the lattice keeps the number of electrons and shows the same flow; the Landau problem: one chiral branch at `+p_z` (`-p_z` for the left-handed fermion), paired higher levels with the known energies, the centres `p_y/eB`, the degeneracy `eB/2 pi` per unit area for two fields and several strips; the 3+1 coefficient `dn_5/dt = e^2 E.B/(2 pi^2) = (2 alpha/pi) E.B` obtained by multiplying the counted degeneracy by the counted crossing rate; the weight `N_c sum t_3 Q^2 = 1/2`; the pion width 7.76 eV, the measured 7.72 eV, the factor of about a thousand.
* `test_chiral_anomaly.py` (the film): the same keys and placeholders in both languages, balanced `$`, no Russian words inside formulas, numbers only inside formulas (decimal comma), the `video` section, no hard-coded video settings, `--set` works and a misspelt key is an error, the length of the film, the timeline, the counters of the film equal the counted numbers, snapshots in both languages at 27 times: no text leaves the frame or overlaps another text, captions, formulas and the closing answer fit the frame.

## Configuration

Every number of the film (frame size and encoder, timeline, physics, sizes of the dots, lines and fonts, the layout of the frame, colours, opacities, the constants of the pion) is in `config.toml`; all the words and formulas are in `texts.toml` (one table per language, the same keys in both). The script contains only algorithms. A value can be changed without editing a file:

```bash
python chiral_anomaly.py --lang en --snapshot 35.5 --set part2.rate=0.3 --set style.background='"#000000"'
python chiral_anomaly.py --lang en --config my_config.toml
```

`--set section.key=value` is repeatable (the value is a Python/TOML literal), `--config` takes another file with the same structure, a misspelt key is an error (see `../dvconfig.py`).

Run from this directory (numpy, scipy is not needed, matplotlib, Pillow and ffmpeg are required):

```bash
./render.sh --lang en               # media/chiral_anomaly_en.mp4
./render.sh --lang ru
./render.sh --lang en --snapshot 35.5   # one PNG at film time 35.5 s
./render.sh --lang en --preview     # 640x360, 15 fps
python -m pytest
```

All formulas and labels are typeset with matplotlib mathtext (LaTeX); words stay outside the formulas. The ring picture of the part 1-2 and the plate of the part 4 are schematic (they are marked as such on the screen).
