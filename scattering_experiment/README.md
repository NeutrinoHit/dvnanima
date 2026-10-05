# A collider scattering experiment: e+ e- -> mu+ mu-

The three stages of a typical scattering experiment, as in the chapter on the S-matrix:

1. **preparation of the initial state**: two counter-propagating beams are accelerated and focused;
2. **interaction**: the particles meet in a localized region. Most pass through each other, rarely an
   e+ e- pair annihilates and a mu+ mu- pair is born (the first event is shown in slow motion);
3. **measurement of the final state**: a detector around the interaction region records where the muons go.

The bunch crossings are then repeated and the muon directions build up the angular distribution

```text
dsigma/dOmega  ~  1 + cos^2(theta),        dN/dcos(theta) = N (3/8) (1 + cos^2 theta),
```

where `theta` is the angle between mu- and the e- beam (high energy, masses neglected). The number of
events is the cross section times the flux and the number of targets, the definition of the cross
section used in the book.

Model: the direction of every event is drawn from the distribution above (rejection sampling,
uniform azimuth); the muons are back to back. The picture is a side view with the beam axis horizontal
and a spherical detector seen from the side, so the muon hits are projected onto the plane. The event
rate is hugely exaggerated and a bunch is a few tens of particles; this is said on the screen.

Run from this directory (numpy, matplotlib and ffmpeg are required):

```bash
./render.sh                 # media/scattering_experiment.mp4, 28 s
./render.sh --preview       # 10 s low-resolution check
./render.sh --snapshot 20   # one PNG at film time 20 s
python -m unittest test_scattering_experiment
```
