# A free Gaussian wave packet

Two scenes, 34 s. Units `hbar = m = 1`.

1. **One packet.** `psi(x,0) = (2 pi s0^2)^(-1/4) exp(-x^2/4 s0^2 + i k0 x)` with `s0 = 1`, `k0 = 2.5`. The film
   evolves `psi` exactly (FFT, `psi~(p,t) = psi~(p,0) exp(-i p^2 t / 2)`); it does not use the Gaussian formula.
   Shown: `Re psi`, `|psi|^2`, the momentum distribution `|psi~(p)|^2` (static; the colour is the phase of
   `psi~(p)`, which rotates) and the width `sigma_x(t) = s0 sqrt(1 + (t / 2 s0^2)^2)`.
2. **Narrow and wide.** `s0 = 1` and `s0 = 2.6`: the narrower packet is broader in `p` and spreads faster
   (`sigma_x sigma_p = 1/2` at `t = 0`).

The tests compare the FFT evolution with the analytic width and group velocity and check that the norm and
`|psi~(p)|^2` are conserved.

Run from this directory (numpy, matplotlib, Pillow and ffmpeg are required):

```bash
./render.sh                 # media/free_wavepacket.mp4
./render.sh --preview       # low-resolution check
./render.sh --snapshot 12   # one PNG at film time 12 s
python -m unittest test_free_wavepacket
```

The older `fields/wavepackets_free` (Manim, scalar field) stays in the gallery.
