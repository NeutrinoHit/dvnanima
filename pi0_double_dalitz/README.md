# pi0 -> e+ e- e+ e-: planes of the pairs

A neutral pion decays into two photons and each virtual photon turns into an e+ e- pair (the "double
Dalitz" decay). Every pair spans a plane that contains the photon axis. The planes have a different
orientation in every event; the measured quantity is the angle `phi` between them, and its distribution
tells the parity of the pion. For a pseudoscalar the rate is proportional to `|E1 . B2|^2`, largest
for orthogonal planes:

```text
dN/dphi  ~  1 - a cos(2 phi),     phi in [0, 2 pi),
```

two humps at `phi = pi/2` and `3 pi/2` (a scalar would give humps at `0` and `pi`). In the film
`a = 0.35` reproduces the height of the humps of the KTeV histogram of the book (fig. 40.6); the amplitude
is illustrative. The azimuth of the first plane is uniform, the second is rotated by `phi` drawn from the
distribution (rejection sampling). The first event is shown slowly, then the events accelerate and the
histogram builds up.

The picture is schematic: the opening angle of a pair is exaggerated and the events are accelerated.

Run from this directory (numpy, matplotlib and ffmpeg are required):

```bash
./render.sh                 # media/pi0_double_dalitz.mp4, 34 s
./render.sh --preview       # 12 s low-resolution check
./render.sh --snapshot 28   # one PNG at film time 28 s
python -m unittest test_pi0_double_dalitz
```
