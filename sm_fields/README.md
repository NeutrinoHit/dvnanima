# Fields of the Standard Model: a stack of sheets, two fields in the zoom

104 s, two languages (`--lang en`, `--lang ru`; `media/sm_fields_en.mp4`, `media/sm_fields_ru.mp4`). The film follows
the section on quantum fields in the first chapter of the book (the spring mattress).

| time, s | part |
|---|---|
| 0-14.5 | the table of the 17 kinds of fields (6 quark, 3 charged-lepton, 3 neutrino fields, `gamma, g, W, Z`, the Higgs field; a quark tile is marked x3 (colours), the gluon tile x8, one sheet stands for all components) lifts, tile after tile, into a stack of sheets: each field is a separate sheet that fills the whole space |
| 15-20 | a line through the stack marks one point of space: every field is present in every point; all of them fluctuate a little |
| 21-28 | a particle is a wave packet on the sheet of its own field (rings on the electron sheet) |
| 28-40 | two electron waves pass through each other: a free field is linear |
| 40-45 | zoom: the other fifteen sheets fly away up and down, the electron and the electromagnetic sheets become solid |
| 46-63 | `e- e-`: repulsion (scalar QED, see below) |
| 64.5-81.5 | `e- e+`: attraction (scalar QED) |
| 83-100 | `e- e+ -> gamma gamma` (the photons fly away along x), then a separate event: two other photons come along y, meet and create a new `e- e+` pair that flies away along y (schematic) |

## The interaction (parts `e- e-` and `e- e+`)

The numerics are the scalar-QED wave-packet model of `../fields/scalar_qed` (copied to `scalar_qed_numerics.py`,
the configuration is in `qed_pair.py`): `m = 6`, `q = 0.88`, packets of width `sigma = 0.38` that start at
`x = -+4.4`, `y = +-0.54` with `p = -+1.8`. A free packet spreads like a relativistic wave packet; every packet
creates the Coulomb-like field `A^mu = (Q / m q) p^mu V(r*)`; the field of the other packet adds the eikonal phase
`exp(-i q^2 theta)` to the wave (`theta` is advected along the packet and sourced by `p_a . A_b / E_a`), and the
gradient of the phase deflects the packet. The film draws `|phi|^2` on the electron sheet and `A_0` on the photon
sheet (colour = sign of `A_0`; the constant far field of two equal charges is subtracted). The dotted lines join
every charge with its field.

## The annihilation (part `e- e+ -> gamma gamma`)

This part is *schematic*: the packets are analytic, and the energy shares are `E_e = cos^2(theta) E_0`,
`E_gamma = sin^2(theta) E_0` with a smooth step `theta(t)` (annihilation at 5 s, the second event at 11.5 s). It
illustrates the exchange of energy between two fields, but it is not a solution of an equation; scalar QED has no
`e- e+ -> gamma gamma` process.

Tests: 17 fields, the stack order, the cascade lift, the energy shares sum to 1, annihilation and back, equal charges
repel and opposite charges attract (compared with the same packets without the coupling), the free paths do not depend
on the sign of the charge, `A_0` of opposite charges has both signs, the cross-fades of the stages, the shapes of the
heights on the drawn grid.

## Configuration

Every number of the film (frame size and encoder, timeline, physics, sizes of the dots, lines and fonts, the layout of the frame,
colours, opacities) is in `config.toml` (25 sections: `video`, `timeline`, `stack`, `fluctuation`, `stack_waves`, `zoom`, `heights`, `annihilation`, `stages`, `camera`, `wire`, `surface`, `labels`, `legend`, `point_of_space`, `charges`, `panel`, `bars`, `layout`, `fonts`, `style`, `groups`, `table`, `qed`, `fields`); all the words and formulas are in `texts.toml` (one table per
language, the same keys in both). The script contains only algorithms. A value can be changed without editing a file:

```bash
python sm_fields.py --lang en --snapshot 20 --set style.background='"#000000"' --set video.crf=18
python sm_fields.py --lang en --config my_config.toml
```

`--set section.key=value` is repeatable (the value is a Python/TOML literal), `--config` takes another file with the same
structure, a misspelt key is an error (see `../dvconfig.py`, tests in `../test_dvconfig.py`).

Run from this directory (numpy, matplotlib, Pillow and ffmpeg are required):

```bash
./render.sh --lang en               # media/sm_fields_en.mp4
./render.sh --lang ru
./render.sh --lang en --snapshot 56 # one PNG at film time 56 s
python -m unittest test_sm_fields
```

All formulas and labels are typeset with matplotlib mathtext (LaTeX); words stay outside the formulas.
