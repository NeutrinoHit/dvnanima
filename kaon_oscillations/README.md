# Oscillations of neutral kaons

103 s (94 s of content and four title cards: 3 s for the film title and 2 s for each of the parts 2-4), two languages (`--lang en`, `--lang ru`; `media/kaon_oscillations_en.mp4`, `media/kaon_oscillations_ru.mp4`).
The film follows the chapter "The Mysteries of Neutral Kaons" of the book (volume 2): the sections "Strangeness Oscillations" (eqs. 14-19 and the `K_l3` charge asymmetry) and "Regeneration", in the notation of the chapter:
`K^0 = |d sbar>` (`S = +1`), `Kbar^0 = |dbar s>` (`S = -1`), `K_S = K_1 = (K^0 - Kbar^0)/sqrt 2` (`CP = +1`), `K_L = K_2 = (K^0 + Kbar^0)/sqrt 2` (`CP = -1`), `Delta m = m_L - m_S`,
`tau_S = 8.954e-11 s`, `tau_L = 5.116e-8 s`, `Delta m = 3.5e-6 eV = 5.3e9 hbar/s`. CP violation is neglected (the film says so).

| content time, s | part |
|---|---|
| 0-20 | **A K0 and its two states.** The strong reaction `pi- p -> Lambda K0` with the strangeness bookkeeping `0 + 0 -> (-1) + (+1)`; the plane (K0, K0bar) with the state arrow `|K0>`; the axes `K_L`, `K_S` turned by 45 degrees and the two components of `|K0>`; cards with `tau_S`, `tau_L`, `c tau`, `Delta m` |
| 20-29 | **Oscillation, kaons without decays** (`Gamma_S = Gamma_L = 0`): two phasors (the `K_S` part and the `K_L` part of the amplitude, rotating with `+-Delta m/2` in the frame of the mean mass) add to the amplitude of K0 and subtract to the amplitude of K0bar; complete oscillation over the period `2 pi/Delta m = 13.2 tau_S`; the sum stays 1 (unitarity); the beam tube changes its composition |
| 29-43 | **The same with the true widths**: the phasors shrink (`K_S` fast), `P(K0 -> K0)` and `P(K0 -> K0bar)` approach each other; the dashed sum `(e^{-Gamma_S t} + e^{-Gamma_L t})/2`; the curves of the stable kaons stay as ghosts; the panels are enlarged as the amplitudes shrink (the factor is printed) |
| 43-52 | **Zoom out**: the time axis changes from `tau_S` to `tau_L = 571 tau_S` (the ticks cross-fade); the whole oscillation is squeezed against the vertical axis, a pure `K_L` beam is left: the two arrows are equal, `P(K0->K0) = P(K0->K0bar) = e^{-Gamma_L t}/4` |
| 52-66 | **What the detector sees (1)**: nine thin targets along the beam; `Kbar0 N -> Lambda pi` is allowed, `K0 N -> Lambda pi` is forbidden by strangeness; the relative yield of `Lambda` behind every target is the `K0bar` intensity at its position (stems, the values are printed) |
| 66-74 | **What the detector sees (2)**: the lepton charge in `K -> l nu pi` (`Delta S = Delta Q`): the counters `N_l+ ~ I_K0`, `N_l- ~ I_K0bar` and the asymmetry `A_l(t) = cos(Delta m t)/cosh(Delta Gamma t/2)` under the envelope `1/cosh` |
| 74-94 | **Regeneration**: a thin absorber in a `K_L` beam (amplitudes `f = 0.9` for K0, `fbar = 0.1` for K0bar: an illustration); the state arrow leaves the `K_L` axis and acquires a `K_S` component `(f - fbar)/2`; behind the absorber the `K_S` intensity decays within a few `tau_S` (the `K_S -> pi pi` decays reappear) while `K_L` goes on |

Film times: every part from 2 on is preceded by a title card, the times above are content times, `film_time(t)` in the script converts them (the poster frame, content 41 s, is film 46 s).

## Model

Basis (K0, K0bar), `H = M - i Gamma/2`, `M = m0 + (Delta m/2) sigma_x`, `Gamma = (Gamma_S + Gamma_L)/2 + ((Gamma_L - Gamma_S)/2) sigma_x`: `K_L = (1, 1)/sqrt 2` and `K_S = (1, -1)/sqrt 2` are the eigenvectors with
`m_{L,S} - i Gamma_{L,S}/2`; `m0 = 0` (the common phase of the mean mass is removed). The time is in `tau_S`. `i d psi/dt = H psi` is **integrated numerically** (`scipy` DOP853, `rtol = 1e-11`; `kaon_physics.py`),
every picture is drawn from these numbers (the formulas of the book are used only in the tests and in the labels). The amplitudes are decomposed as `a(K0) = s + l`, `a(K0bar) = l - s` with `s = <K_S|psi>/sqrt 2`, `l = <K_L|psi>/sqrt 2`; the phasor panels draw `s` and `l`,
head to tail, and their sum or difference. The beam is a tube along `z = beta gamma c t` (proper time on the axis), the thickness is the total intensity and the colours are the composition (schematic). The absorber multiplies the amplitudes of K0 and K0bar by `f^u` and `fbar^u` (`u` is the depth in units of the thickness).

Tests (`python -m pytest test_kaon_oscillations.py`, 56 tests): numbers of the chapter, `K_S`, `K_L` are eigenvectors with the right eigenvalues and `CP`, `H` commutes with `CP`; **unitarity at `Gamma = 0`**; **the full oscillation with the period `2 pi/Delta m`** (`P(K0 -> K0) = cos^2(Delta m t/2)`);
**numerical integration equals the closed formulas of the book (1e-9)** and an independent `expm`; `P(K0->K0) + P(K0->K0bar) = (e^{-Gamma_S t} + e^{-Gamma_L t})/2`; **trace and determinant of the evolution matrix with decays**; the rate `d|psi|^2/dt = -psi^dagger Gamma psi`;
the oscillation frequency `Delta m` (zero of the interference term); **the late-time composition 50:50** (`|a0|^2 = |abar|^2 = e^{-Gamma_L t}/4`); the projections on `K_S`, `K_L` without cosine (eq. 19 of the book); the phasor decomposition (moduli and the opposite rotation with `Delta m/2`);
the lepton asymmetry `A_l = cos/cosh` from the amplitudes; the yield of hyperons has a maximum above the plateau 1/4; regeneration (the incident beam is `K_L`, the regenerated `K_S` amplitude is `(f - fbar)/2`, no regeneration at `f = fbar`, the decay laws behind the absorber, the beam is 50:50 again, the intensities before the absorber);
the timeline (cards, parts, sub-stages), the numbers inserted into the texts (decimal comma in Russian), **no overlapping texts and no text outside the frame at 23 times of both languages and on the cards**,
the configuration (same keys in both languages, balanced `$`, no Russian words inside formulas, the video section, the script reads the config, `--set` works, no numeric literals in the drawing code, snapshots in both languages).

## Configuration

Every number of the film (frame size and encoder, timeline, physics and the integrator, the absorber, layout, line widths, font sizes, colours, opacities) is in `config.toml`
(sections `video`, `physics`, `integrator`, `timeline`, `part1`-`part4`, `layout`, `fonts`, `style`, `curves`, `tube`, `foils`, `phasor`, `plane`, `reaction`, `reaction2`, `box`, `lepton`, `check`);
all the words and formulas are in `texts.toml` (one table per language, the same keys in both; the numbers computed from the configuration are inserted at the marks `@tau_s@`, `@dm_ev@`, ... and get a decimal comma in Russian).
The scripts contain only algorithms (`kaon_physics.py`: the physics, `kaon_oscillations.py`: the film). A value can be changed without editing a file:

```bash
python kaon_oscillations.py --lang en --snapshot 50 --set style.background='"#000000"' --set video.crf=18
python kaon_oscillations.py --lang en --content 41 --set physics.delta_m_ev=7e-6      # a snapshot at the CONTENT time 41 s
python kaon_oscillations.py --lang en --config my_config.toml
```

`--set section.key=value` is repeatable (the value is a Python/TOML literal), `--config` takes another file with the same structure, a misspelt key is an error (see `../dvconfig.py`).

Run from this directory (numpy, scipy, matplotlib, Pillow and ffmpeg are required):

```bash
./render.sh --lang en               # media/kaon_oscillations_en.mp4
./render.sh --lang ru
./render.sh --lang en --preview
./render.sh --lang en --snapshot 50 # one PNG at film time 50 s
python -m pytest test_kaon_oscillations.py
```

All formulas and labels are typeset with matplotlib mathtext (LaTeX); words stay outside the formulas.
