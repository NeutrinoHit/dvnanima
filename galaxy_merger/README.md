# Galaxy Merger

Short numerical animation of a merger of two disk galaxies. It follows the idea of
the restricted N-body experiments of Toomre & Toomre (ApJ 178, 623, 1972): tidal
bridges and tails during the encounter and a round remnant after the merger.

Model:

- each galaxy is a massive Plummer halo (mass `M`, scale `a`) carrying a thin disk
  of massless test stars on initially circular orbits;
- the stars feel the gravity of both halos and do not interact with each other;
- the two halos attract each other exactly like two Plummer spheres, that is, like
  point masses softened with `eps^2 = a1^2 + a2^2`;
- each halo is slowed down while it moves through the other one by Chandrasekhar
  dynamical friction. Without it the pair would stay on a closed bound orbit and
  never merge. The friction is shared between the two halos so that the total
  momentum is conserved;
- the start is a bound Kepler orbit with the given pericentre and eccentricity.

Units: `G = 1`, the mass of the first galaxy is 1, the pericentre of the start orbit
is 1. In the default setup the halo centres pass each other three times and merge.
After the merger the camera slowly moves in on the remnant.

Colours only show which galaxy a star came from: warm for the first, cold for the
second. The brightness is a blurred star density with an arcsinh stretch and a
film-wide reference level.

The animation is schematic. It has no gas, star formation or stellar feedback, and
the halo density profile and the Coulomb logarithm are simple fixed choices.

Run from this directory (numpy, matplotlib and ffmpeg are required):

```bash
./render.sh
```

The default output is `media/videos/galaxy_merger/1080p30/GalaxyMerger.mp4`
(about 33 s). All parameters live in `run.cfg`; any of them can be overridden on the
command line, for example a quick preview:

```bash
python galaxy_merger.py --stars 15000 --pixels 540 --hold 2 --out preview.mp4
```

Stills for checking the result can be written with `--poster`, `--last-frame`
and `--sheet` (a contact sheet of eight frames).

Main parameters:

- `stars`, `mass_ratio`, `halo`, `rdisk`: size and mass of the galaxies;
- `pericentre`, `ecc`, `theta0`, `lnl`: orbit and strength of the friction;
- `incl1`, `node1`, `incl2`, `node2`: orientation of the disks;
- `duration`, `frame_time`, `dt`: simulated time, time per video frame, time step;
- `stretch`, `black`, `bright`, `end_exposure`, `exposure_from`: tone mapping;
- `hold`, `hold_zoom`, `hold_spin`: final slow zoom on the remnant.

Check of the numerics (energy conservation without friction, momentum conservation
with friction, circular orbits of the disks, star counting in the renderer):

```bash
python -m unittest test_galaxy_merger -v
```

Generated videos are kept out of git; regenerate them with `render.sh`.
