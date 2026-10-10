# Asymmetry in Z-boson production and decay

97 s (89 s of content, the title card and four 1.4 s chapter cards), two languages (`--lang en`, `--lang ru`; `media/z_decay_asymmetry_en.mp4`,
`media/z_decay_asymmetry_ru.mp4`). The film illustrates the section "gamma/Z interference in e+e- -> f fbar" and the decays of the Z in the chapter
"Weak decays of the W and Z bosons" of the book (volume 2, chapter 26) and its problems on `A_LR^f` and `A_FB^f`. It uses the notation of the
chapter: `g_v = t_3 - 2 q sin^2(theta_W)`, `g_a = t_3`, `g_L = g_v + g_a`, `g_R = g_v - g_a`, `rho_ab`, `A_LR^f`, `A_FB^f`, and its numbers:
`sin^2(theta_W)(m_Z) = 0.2316`, `1/alpha_em(m_Z) = 128.95`, `Gamma_Z = 2495.5 MeV` (`m_Z = 91.1876 GeV` is the PDG value, the chapter does not print it).
Everything is tree level, massless fermions.

| content time, s | part |
|---|---|
| 0-13 | the couplings `g_L`, `g_R` of nu, e, u, d (bars of `g^2`, pictograms of the pairs `f_L fbar_R` and `f_R fbar_L`), the polarization asymmetry `A_LR^f = (g_L^2 - g_R^2)/(g_L^2 + g_R^2)` |
| 13-39.5 | `e- e+ -> Z -> f fbar` for the b quark: `e_L- e_R+` gives a Z with `J_z = -1`, the pair `f_L fbar_R` has spin -1 on its axis, so `(1 + cos)^2`; the four channels (2 x 2 matrix, weights `(g_a^e g_b^f)^2`), the sum `1 + cos^2 + (8/3) A_FB cos` against the symmetric dashed curve, `A_FB = (3/4) A_LR^e A_LR^f = 0.103` |
| 39.5-53 | the slider `sin^2(theta_W)` from 1/4 (g_v of the leptons vanishes, no asymmetry) to 0.2316: the distributions of mu and b, their asymmetric parts, `A_LR^e`, `A_FB^mu`, `A_FB^b`; `A_LR^e ~ 8(1/4 - s^2)`, `A_FB^l ~ 48 (1/4 - s^2)^2` |
| 53-77 | Monte Carlo: b events (rays in the detector, the histogram of cos theta, `N_F`, `N_B`, `A_FB = (N_F - N_B)/(N_F + N_B)` with `sigma = sqrt((1 - A^2)/N)`, up to 20 000 events), then muons up to 10^6 events (the estimate with its error band against the model) |
| 77-89 | `A_FB^mu(sqrt s)` between 50 and 150 GeV (the range of the book's problem) from the full `rho_ab` with photon and Z exchange, the cross-section peak, the lobe at the current energy |

Physics (all in `z_decay_asymmetry.py`, checked in `test_z_decay_asymmetry.py`):

- `dsigma/dcos = N_c s/(128 pi) [(|rho_LL|^2 + |rho_RR|^2)(1 + cos)^2 + (|rho_LR|^2 + |rho_RL|^2)(1 - cos)^2]`, the massless limit of the chapter's `|M|^2` and `d sigma`; `theta` is the angle between the e- and the f.
- The chiral channels are verified with explicit Dirac matrices and spinors (sum over all spins of `|vbar gamma^mu P_a u ubar gamma_mu P_b v|^2 = 4 (s+t)^2` for LL, RR and `4 t^2` for LR, RL), and the Z decay rate `~ g_L^2 + g_R^2` with the split `g_L^2 : g_R^2`.
- The summed channels reproduce the chapter's total cross sections (photon only, Z only, `12 pi Gamma_ee Gamma_ff / (m_Z^2 Gamma_Z^2)` at the peak).
- `A_FB = (3/4) A_LR^e A_LR^f` equals the numerical forward-backward integral; limits `g_R = 0 -> A = 1`, `g_v = 0 -> A = 0`, `sin^2 = 1/4 -> A_FB = 0`.
- The Monte-Carlo events are drawn by rejection from the distribution; the sampled asymmetry agrees with the formula within 4 sigma, the histogram has `chi^2/ndf ~ 1`, the error scales as `1/sqrt N`.
- The film's seeds (`sampling.seed_b = 1`, `seed_mu = 2`) are chosen so that the final estimates lie within 0.5 sigma of the model; any seed is statistically valid.

Schematic elements (stated in the film where relevant): events are rotated about the beam axis into the plane of the picture; the spin and momentum pictograms are symbolic;
the tree-level numbers (`A_FB^b = 0.103`, `A_FB^mu = 0.016`) differ slightly from the measured ones because radiative corrections are not included.

Tests: Dirac algebra, couplings and asymmetries, channel sums, cross sections of the chapter, the pole scan, sampling, configuration (same keys in both languages, balanced `$`,
decimal commas, video section, `--set`), timeline, texts inside the frame and not overlapping (measured with the text extents) in 33 sample frames per language, the numbers on the screen
equal the computed ones, a snapshot in both languages.

## Configuration

Every number of the film (frame size and encoder, physics inputs, timeline, all timings, geometry and layout fractions, sizes of dots, arrows and lines, fonts, colours, seeds and event counts)
is in `config.toml` (sections `video`, `physics`, `sampling`, `timeline`, `couplings`, `channels`, `weak_angle`, `events`, `scan`, `layout`, `fonts`, `style`, `icon`); all the words and formulas are in
`texts.toml` (one table per language, the same keys in both; the numbers computed by the script are inserted at the marks `@v@`, `@a@`, ...). The script contains only algorithms.
A value can be changed without editing a file:

```bash
python z_decay_asymmetry.py --lang en --snapshot 38 --content --set physics.sin2_theta_w=0.25
python z_decay_asymmetry.py --lang en --config my_config.toml
```

`--set section.key=value` is repeatable (a Python/TOML literal), `--config` takes another file with the same structure, a misspelt key is an error (see `../dvconfig.py`).

Run from this directory (numpy, scipy, matplotlib, Pillow and ffmpeg are required):

```bash
./render.sh --lang en                       # media/z_decay_asymmetry_en.mp4
./render.sh --lang ru
./render.sh --lang en --snapshot 38 --content   # one PNG at the content time 38 s (without --content: the film time, cards included)
./render.sh --lang en --preview             # low-resolution check
python -m pytest test_z_decay_asymmetry.py
```

All formulas and labels are typeset with matplotlib mathtext (LaTeX); words stay outside the formulas; in Russian every number has a decimal comma.
