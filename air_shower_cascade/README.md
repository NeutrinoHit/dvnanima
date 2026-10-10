# A cosmic-ray air shower (Heitler–Matthews toy model, Monte Carlo)

About 100 s (91 s of content plus a 2.4 s title card and two 1.7 s chapter cards), two separate films (`--lang en`, `--lang ru`;
`media/air_shower_cascade_en.mp4`, `media/air_shower_cascade_ru.mp4`). The film **replaces** the old Manim film `../airshower`
(catalog item 0029 "A cosmic-ray air shower"); nothing of that project is used.

The physics is in `shower_model.py` (no numbers of its own, only `Params`), the film in `air_shower_cascade.py`; every number is
in `config.toml`, every word and formula in `texts.toml`.

## The model

* Depth `X` along the axis in g/cm^2, a vertical shower in an isothermal atmosphere `X = X_sea exp(-h/H)`, `H = 8 km`; the
  observation level is `X_g = 870 g/cm^2` (h = 1.35 km, as the large arrays at about 1.4 km).
* **Electromagnetic part** (Heitler 1944): `e -> e gamma`, `gamma -> e+ e-`; after the depth `d = X_0 ln 2` (`X_0 = 37 g/cm^2`:
  the energy of an electron falls as `exp(-X/X_0)`, one halving per `X_0 ln 2`) the particle splits in two with `E/2` each.
  Part 1 uses the exact fixed step, the Monte Carlo of part 2 draws the depth from an exponential distribution with the same mean.
  Below `E_c = 85 MeV` ionisation wins: the particle loses `E_c/X_0` per g/cm^2 and stops after `X_0 E/E_c`.
  Estimates: `N_max = E_0/E_c`, `X_max = X_0 ln(E_0/E_c)` (the number of doublings is `log2(E_0/E_c)`, each over `X_0 ln 2`).
* **Hadronic part** (Matthews 2005): a proton of `E_0 = 10^15 eV` interacts after a depth with the mean `lambda_I = 120 g/cm^2`;
  every interaction makes `n_ch = 10` charged and `n_ch/2 = 5` neutral pions with equal energy `E/(1.5 n_ch)` (2/3 of the
  secondaries are charged), transverse momenta Rayleigh (mean 0.4 GeV, sum zero). `pi0 -> 2 gamma` at once (starts an
  electromagnetic cascade); a charged pion with `E <= E_dec = 20 GeV` decays, `pi -> mu nu` (flat `E_mu/E_pi` in
  `[m_mu^2/m_pi^2, 1]`, exact for an isotropic decay of a scalar, the transverse momentum `p*` sin theta* is shared), otherwise it interacts again.
  `N_mu = n_ch^m = (E_0/E_dec)^beta`, `beta = ln n_ch / ln(1.5 n_ch) = 0.85`.
* Lateral coordinates: straight tracks with a lateral drift `u H ln(X2/X1)`; hadrons get the transverse-momentum kicks,
  electrons multiple scattering (`<theta^2> = (E_s/E)^2 s/X_0`, `E_s = 21.2 MeV`, so `r_M = E_s X_0/(E_c rho) = 85 m` at the
  observation level), photons go straight. Muon decay and muon ionisation loss are not included.
* Thinning (Hillas): below 10 GeV only one of the two daughters of a split is followed with a doubled weight (unbiased); this keeps
  the 10^7 particles of the shower in a table of about 8*10^5 tracks. `physics.thin_energy_ev = inf` switches it off (the tests do).
* The drawn tracks are a **random sample** of the same cascade (the primary, the ancestors of 40 decaying pions with their muons
  and neutrinos, and electromagnetic trees pruned to a target size, second daughter followed with probability q); the readout
  shows 1 drawn track per K particles. The brightness of an electromagnetic track encodes its energy (log scale).

## What is shown

| content time, s | part |
|---|---|
| 0-19 | Heitler's cascade of one photon `E_0 = 2^5 E_c`: the binary tree drawn exactly, `N = 2^n`, `E = E_0/2^n`; the staircase `N(X)` on the semilog plot, the line `N = 2^{X/d}`, then `N_max = E_0/E_c`, `X_max = X_0 ln(E_0/E_c)` |
| 19-63 | the proton: descent (4.2 s), the first interaction (the rule of the pion interaction), the cascade up to the shower maximum with the live profile `N(X)` of `e+- + gamma`, `mu`, `p, pi+-` and the rules of the electromagnetic cascade, the Heitler markers `E_0/E_c`, `X_max` and the Monte Carlo peak, the fall to the ground, the summary Monte Carlo vs Heitler–Matthews |
| 63-85 | the detector array (top view, 10 m spacing, 1 m^2 stations drawn enlarged): hits per station (Poisson sample of the simulated density) light up from the core outwards, stations hit by a muon in green, the lateral distribution of `e+-` and `mu`, the NKG formula, the Molière radius |
| 85-94 | one card on the light: fluorescence `dE/dX` of the simulated shower (the integral is the electromagnetic energy), the Cherenkov threshold and angle of `e+-` against the altitude (`n - 1` proportional to the density) |

Honest limitations, shown or stated in the film: it is a toy model; the horizontal scale of the picture is exaggerated
(about 15 times at the shower maximum, computed); the tracks are a random sample; the particles of the toy model below `E_c` stop
after `X_0 E/E_c`, so the tail beyond the maximum falls faster than in a real shower; Heitler's `N_max = E_0/E_c` is an upper
estimate (this Monte Carlo: `N_max = 0.17 E_0/E_c` for `e+- + gamma`, because the splitting depths are random; the position of the
maximum is predicted to about 10 %).

Numbers of the shown shower (`mc.seed = 0`, a typical one: first interaction at 82 g/cm^2, the median of the exponential is 83):
`X_max = 540 g/cm^2` (estimate `X_1 + X_0 ln(E_0/(2 n_sec E_c))` with `X_1 = lambda_I`: 597), `N_max = 2.0*10^6`
(`E_0/E_c = 1.2*10^7`), `N_mu = 9.0*10^3` (model `(E_0/E_dec)^beta = 9.9*10^3`; the rest of the pions has not decayed before the ground),
`N(e+-) = 1.6*10^5` at the ground, electromagnetic energy deposit `0.76 E_0`.

## Run

```bash
./render.sh --lang en                 # media/air_shower_cascade_en.mp4
./render.sh --lang ru
./render.sh --lang en --preview       # 640x360, 15 fps
./render.sh --lang en --snapshot 62   # one PNG at the film time 62 s
./render.sh --lang en --content 40    # one PNG at the content time 40 s (cards not counted)
python -m pytest test_air_shower_cascade.py
```

(numpy, scipy, matplotlib, Pillow and ffmpeg are required.)

## Tests

`test_air_shower_cascade.py`: the fixed-step cascade reproduces `N_max = E_0/E_c` and `X_max = X_0 ln(E_0/E_c)` exactly, `N = 2^g` per step,
2/3 of the particles charged, the analytic profile equals the simulation exactly (e+-, gamma, hadrons, muons); with random depths
`X_max` follows `X_0 ln(E_0/E_c)` and grows as `X_0 ln E_0`, `N_max` is 0.12–0.45 of `E_0/E_c`; the track-length integral is proportional
to `E_0`; energy is conserved in every splitting, interaction and pion decay, in the whole shower exactly without thinning and within 3 %
with it; pi0 fraction 1/3; `N_mu = n_ch^m = (E_0/E_dec)^beta` for `E_0 = (1.5 n_ch)^m E_dec`; the Molière radius, NKG normalisation,
Cherenkov threshold (21.1 MeV) and angle (1.39 deg); the drawn sample is a consistent forest; parts and cards of the timeline; the front of the
shower visits the key depths; the ground view; texts of both languages have the same keys and balanced `$`; the script reads the
configuration; `--set` works; snapshots render in both languages; **no text box leaves the frame or overlaps another** (measured with
`get_window_extent` at 16 times covering every part, both languages).

## Configuration

Every number of the film (frame size and encoder, the physics parameters, seeds, sample sizes, geometry of the panels, layout fractions,
fonts, line widths, timings of every stage, colours) is in `config.toml` (sections `video`, `units`, `physics`, `mc`, `atmosphere`, `timeline`, `tree`,
`scene2`, `column`, `profile`, `info`, `ground`, `light`, `layout`, `fonts`, `style`); all the words and formulas are in `texts.toml` (one table per
language, the same keys in both). The scripts contain only algorithms. A value can be changed without editing a file:

```bash
python air_shower_cascade.py --lang en --snapshot 62 --set physics.e0_ev=1e16 --set physics.path='"fixed"'
python air_shower_cascade.py --lang en --config my_config.toml
```

`--set section.key=value` is repeatable, `--config` takes another file with the same structure, a misspelt key is an error
(see `../dvconfig.py`). Note: `../test_dvconfig.py` lists the films by name; add `air_shower_cascade` to its `FILMS` to test this one there too
(it satisfies the rules: `load_config`, complete `video` section, identical keys in both languages).

References: W. Heitler, The Quantum Theory of Radiation (1944); J. Matthews, Astropart. Phys. 22 (2005) 387; K. Kamata, J. Nishimura (1958) and K. Greisen (1960), the NKG lateral distribution; A. M. Hillas, the thinning of showers (as in CORSIKA).
