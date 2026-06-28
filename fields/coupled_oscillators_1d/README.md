# 1D Coupled Oscillators

Vertical-spring chain along a horizontal axis:

- several normal modes are shown with explicit formulas on screen
- the ending section shows a local displacement of one oscillator and its release

The model is:

```text
m q_j'' = -k0 q_j - kc (2 q_j - q_{j-1} - q_{j+1})
```

with free-end mode shapes

```text
q_j^(s)(t) ~ cos[s pi (j - 1/2) / N] cos(omega_s t)
```

Render:

```bash
cd "/Users/dmitrijnaumov/Library/Mobile Documents/com~apple~CloudDocs/Projects/dvnanima/fields/coupled_oscillators_1d"
manim -pqh coupled_oscillators_1d.py CoupledOscillators1D
```

Main tuning lives in `run.cfg`:

- `mode_list`
- `onsite_k`
- `coupling_k`
- `mode_amplitude`
- `kick_index`
- `kick_amplitude`
