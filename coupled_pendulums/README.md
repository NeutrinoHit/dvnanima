# Coupled Pendulums

Two identical pendulums linked by a spring.

The scene shows:

- the in-phase normal mode
- the out-of-phase normal mode
- release of one pendulum to show the beat pattern as a superposition of both modes

In the small-angle limit the mode frequencies are

```text
omega_+^2 = g / l
omega_-^2 = g / l + 2k / m
```

Render:

```bash
cd "/Users/dmitrijnaumov/Library/Mobile Documents/com~apple~CloudDocs/Projects/dvnanima/coupled_pendulums"
manim -pqh coupled_pendulums.py CoupledPendulums
```

Main tuning lives in `run.cfg`:

- `length`, `gravity`, `coupling_k`
- `mode_amplitude`
- `initial_left`, `initial_right`
- `mode_periods`, `free_beats`
