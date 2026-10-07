# Cherenkov radiation

One scene with four stages, 40 s. A charge moves along `x` with `v = beta c` in a medium with `n = 1.33`
(water). The wave speed is `c/n`, `c = 1`. The wavelets are circles emitted every 0.1 s; their radii are exact.

* `beta < 1/n` (0-10 s, `beta = 0.55`): the circles are nested, there is no envelope and no light.
* `beta > 1/n` (10-28 s, `beta = 0.85`, then `0.99`): the wavelets have a common envelope, a cone with
  half-angle `psi = arcsin(1/(n beta))`; light travels perpendicular to it at `theta`, `cos(theta) = 1/(n beta)`.
* 28-40 s: `beta` rises slowly from 0.80 to 0.99; the angle and the detector ring `R = L tan(theta)` grow.

Water: threshold `beta = 0.752` (0.26 MeV kinetic energy for an electron, 54.6 MeV for a muon), `theta_max = 41.2 deg`.
The cone line is drawn for the instantaneous `beta`; while `beta` changes it lags the true envelope (the wavelets
remember the past), so it is dimmed during the jumps.

```bash
./render.sh                 # media/cherenkov_radiation.mp4
./render.sh --preview
./render.sh --snapshot 24
python -m unittest test_cherenkov_radiation
```

References: P. A. Cherenkov, Dokl. Akad. Nauk SSSR 2, 451 (1934); I. M. Frank and I. E. Tamm, Dokl. Akad.
Nauk SSSR 14, 107 (1937). The older `cherenkov_cone` (3D cylinder) stays in the gallery.
