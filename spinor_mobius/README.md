# Spinor and the Moebius strip

A spin-1/2 spinor changes sign under a rotation by 360 degrees and returns to its value only
after 720 degrees. Three linked views of one rotation angle `phi` (0 to 720 degrees):

- a dial with the rotating body;
- a Moebius strip with a vector lying across it, carried once around the core circle per
  360 degrees: it comes back reversed after the first turn and restored after the second;
- the complex plane with the spinor component `psi = exp(-i phi / 2)` (the Dirac equation example
  from the book): `psi = -1` at 360 degrees, `psi = +1` at 720 degrees.

Strip: `P(u, v) = ((1 + v cos(u/2)) cos u, (1 + v cos(u/2)) sin u, v sin(u/2))`, the vector across
it is `w(u) = dP/dv = (cos(u/2) cos u, cos(u/2) sin u, sin(u/2))`, hence `w(u + 2 pi) = -w(u)`.

The film illustrates the sign change of a spinor. It does not show the belt (plate) trick or
the wave function of two fermions.

Run from this directory (numpy, matplotlib and ffmpeg are required):

```bash
./render.sh                # media/spinor_mobius.mp4, 20 s
./render.sh --preview      # 6 s low-resolution check
./render.sh --snapshot 9.5 # one PNG at film time 9.5 s
python -m unittest test_spinor_mobius
```
