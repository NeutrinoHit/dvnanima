# The Meissner effect and the photon mass

78.5 s (72 s of content and four title cards), two languages (`--lang en`, `--lang ru`; `media/meissner_photon_mass_en.mp4`, `media/meissner_photon_mass_ru.mp4`).
The film follows the sections "Abelian Higgs model" (`m_A = e v`, `2 + 2 = 1 + 3`) and "Superconductivity and the Meissner effect" of the chapter
"Where does a particle get its mass?" and uses the book's symbols: `v`, `e`, `m_A`, `phi`, `chi`, `A^mu`.

Units `hbar = c = 1`. A fixed condensate `|phi|^2 = v^2/2` (no back-reaction on `v`) gives the gauge field the mass `m_A = e v`
(unitary gauge, `j = -m_A^2 A`, the Proca equation `(d^2 + m_A^2) A^mu = 0`). The penetration depth is called `lambda_L` in the film, so that it is not
confused with the quartic coupling `lambda` of the book (`m_chi = lambda v`).

| content time, s | part |
|---|---|
| 0-20 | **The Meissner effect.** A disc of the condensate in a uniform field `B_0` (2D, `A = A_z e_z`, the field lines are the contours of `A_z`, colour = `|B|`). The condensate grows (`v`, hence `m_A R` from 0.04 to 14) and the lines are pushed out. Each frame is the exact **equilibrium** field for the current `v` (quasi-static; real flux expulsion needs dissipation, which is not simulated). Right: the tangential field along a cut through the top of the disc, its decay `exp(-x/lambda_L)`, `lambda_L = 1/m_A`; then the screening current `j = -m_A^2 A_z` (dots / crosses) |
| 20-48 | **A wave packet at the boundary.** The equation `d_t^2 A - d_x^2 A + m_A^2(x) A = 0`, `m_A(x) = m_A Theta(x)` (one transverse polarization), is integrated by leapfrog with absorbing layers. Packet 1, `omega = 0.7 m_A < m_A`: total reflection, the evanescent tail `exp(-kappa x)`, `kappa = sqrt(m_A^2 - omega^2)` (zoom). Packet 2, `omega = 1.5 m_A`: transmission, `k = sqrt(omega^2 - m_A^2)`, `v_g = k/omega < 1`; the centre of the packet `x_c(t)` bends at the boundary. The table sets the formulas against numbers read off the simulation |
| 48-72 | **Dispersion and polarizations.** `omega^2 = k^2 + m_A^2` against the light cone `omega = k`, the gap, the two packets as horizontal lines, a point sliding on the curve with its tangent (`v_g`); the polarizations (2 transverse, then the third, longitudinal) and the counting `2 + 2 = 1 + 3` with the Goldstone mode eaten by the gauge field |

Chapter cards precede every part (the content times above exclude them; `film_time(t)` converts).

## What is measured in the simulation (shown in the table of part 2)

* penetration depth: straight-line fit of `ln |A_omega(x)|`, the Fourier component of the simulated field at the carrier frequency (a finite packet has a bandwidth and its
  maximum decays more slowly; every frequency separately decays exactly as `exp(-kappa(omega) x)`): 1.402 against 1.400 for `omega = 0.7 m_A`;
* reflected / transmitted energy (97.6 % against 97.9 % transmitted for `omega = 1.5 m_A`, the difference is the bandwidth);
* group velocity: slope of the energy centroid of the transmitted packet, 0.7455 against `k/omega = 0.7454`; light in vacuum 0.9996;
* wave number: slope of the phase of `A_omega(x)`, 1.118 against 1.118 (1.1183 / 1.1180).

## Physics tests (`test_meissner_photon_mass.py`)

Static disc: `A_z` and `B` continuous at the surface, `B = curl A`, the London equation inside and the Laplace equation outside (finite differences), uniform field far away, no effect
without a condensate, monotonic expulsion, the perfect-diamagnet limit (`B_n = 0`, `|B_t| = 2 B_0 |sin theta|`), **penetration depth `1/m_A`** from a numerical solution of the planar problem
(0.5 %) and from the disc (1 % at `m_A R = 200`). Dynamics: **exact energy conservation of the lossless leapfrog scheme** (1e-9), absorbing layers, the dispersion relation
`omega = sqrt(k^2 + m_A^2)` of a standing wave, total reflection below the gap, Fresnel transmission above, **the threshold at `omega = m_A`** (scan of long packets), measured depth, group velocity `k/omega`
(1 %), wave number, `v_g` as the slope of the dispersion curve. Film: timeline, configuration (same keys in both languages, balanced `$`, video section, `--set`), texts inside the frame and
**no overlapping texts** in every checked frame of both languages, the decimal comma in Russian, a snapshot in both languages.

## Run

```bash
./render.sh --lang en               # media/meissner_photon_mass_en.mp4
./render.sh --lang ru
./render.sh --lang en --preview
./render.sh --lang en --snapshot 22 # one PNG (film time 22 s)
python -m pytest test_meissner_photon_mass.py
```

(numpy, scipy, matplotlib, Pillow and ffmpeg are required.) The older films `higgs-honey` etc. stay in the gallery.

## Configuration

Every number of the film (frame size and encoder, timeline, physics, the simulation, the geometry of the panels, line widths, font sizes, colours, opacities) is in `config.toml`
(sections `video`, `physics`, `packets`, `measure`, `timeline`, `part1`, `part2`, `part3`, `layout`, `fonts`, `style`); all the words and formulas are in `texts.toml` (one table per language, the same keys in both).
The scripts contain only algorithms (`meissner_physics.py`: the physics, `meissner_photon_mass.py`: the film). A value can be changed without editing a file:

```bash
python meissner_photon_mass.py --lang en --snapshot 20 --set style.background='"#000000"' --set video.crf=18
python meissner_photon_mass.py --lang en --config my_config.toml
```

`--set section.key=value` is repeatable (the value is a Python/TOML literal), `--config` takes another file with the same structure, a misspelt key is an error (see `../dvconfig.py`).
