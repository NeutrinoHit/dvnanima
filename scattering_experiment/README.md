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

## Configuration

Every number of the film (frame size and encoder, the timeline of the stages and of the crossings, the seeds of the random numbers, the
size and speed of the bunches, the number of events per crossing, sizes of the dots, lines and fonts, the layout of the frame, colours,
opacities) is in `config.toml` (6 sections: `video`, `model`, `timeline`, `layout`, `fonts`, `style`; `model` holds the physics and the
random seeds); all the words and formulas are in `texts.toml` (one table per language, the same keys in both). The script contains only
algorithms. A value can be changed without editing a file:

```bash
python scattering_experiment.py --lang en --snapshot 20 --set model.events_seed=7 --set video.crf=18
python scattering_experiment.py --lang en --config my_config.toml
```

`--set section.key=value` is repeatable (the value is a Python/TOML literal), `--config` takes another file with the same
structure, a misspelt key is an error (see `../dvconfig.py`, tests in `../test_dvconfig.py`).

The film exists in two separate versions, English and Russian (`--lang en`, `--lang ru`); all formulas and labels are typeset
with matplotlib mathtext (LaTeX), words stay outside the formulas. The first version of the film was bilingual (an English and a Russian
line in the same frame); the two versions now have one language each.

Run from this directory (numpy, matplotlib and ffmpeg are required):

```bash
./render.sh --lang en                # media/scattering_experiment_en.mp4, 28 s
./render.sh --lang ru                # media/scattering_experiment_ru.mp4
./render.sh --lang en --preview      # 10 s low-resolution check
./render.sh --lang en --snapshot 20  # one PNG at film time 20 s
python -m unittest test_scattering_experiment
```
