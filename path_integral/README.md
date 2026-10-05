# The path integral

Three scenes, 40 s.

1. **Slits.** The amplitude to reach the detector is the sum over all paths through screens with 3, 7
   and 5 slits (3, 21 and 105 paths). The phase of a path is `2 pi L / lambda`, `L` being its length. The
   phasors are drawn head to tail, the white arrow is the amplitude, the yellow profile along the detector
   is `|A|^2`, computed from the same sum. The detector moves, the interference pattern is read off.
2. **All paths.** With infinitely many screens and slits all paths contribute. For a free particle the
   paths `x_a(t) = a sin(pi t / T)` from `(0, 0)` to `(T, 0)` have the action
   `S(a) = (m pi^2 / 4T) a^2`, so the phase `S/hbar = c a^2` grows quadratically with the deviation `a` from
   the classical path. The phasors `e^{i c a^2}` added head to tail draw the Cornu spiral: near `a = 0` they
   point the same way (stationary action), far from it they curl up and cancel. The sum converges to
   `sqrt(pi / c) e^{i pi / 4}`.
3. **Classical limit.** When `hbar` decreases (`c` grows), the spiral shrinks and only the paths within
   `|a| < sqrt(pi / c)` matter.

Schematic: the family of paths is one-parameter, the action is quadratic in `a`, and the values of `c` are
chosen for clarity (`hbar ~ 1/c`).

Run from this directory (numpy, matplotlib and ffmpeg are required):

```bash
./render.sh                 # media/path_integral.mp4, 40 s
./render.sh --preview       # low-resolution check
./render.sh --snapshot 12   # one PNG at film time 12 s
python -m unittest test_path_integral
```

References: R. P. Feynman, Rev. Mod. Phys. 20, 367 (1948); R. P. Feynman and A. R. Hibbs, Quantum Mechanics
and Path Integrals (1965).
