# Photomultiplier: electron multiplication

Two scenes, 40 s.

1. **One photoelectron.** A photon knocks an electron out of the photocathode; at each of `N = 10` dynodes an
   electron knocks out `Poisson(delta)` secondary electrons, `delta = 4`, so `n_j` is a branching process with
   `<n_j> = n_0 delta^j` and the gain is `delta^N ~ 10^6`. The bars show the random counts against `n_0 delta^j`;
   the counter shows the exact number of electrons (at most 36 per stage are drawn).
2. **Pulse height.** Three scintillation flashes with 2, 6 and 12 photoelectrons: the anode pulses are
   (statistically) proportional to the number of photoelectrons.

```bash
./render.sh                 # media/pmt_multiplication.mp4
./render.sh --preview
./render.sh --snapshot 12
python -m unittest test_pmt_multiplication
```

The older `pmt` (Manim, field map of the first dynode) stays in the gallery.
