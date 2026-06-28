# 2D Coupled Oscillators

Rectangular 2D lattice of coupled vertical oscillators with fixed boundary.

What the scene shows:

- several normal modes with explicit formulas on screen
- a local displacement of one lattice site and its release
- subsequent spreading of the disturbance across the lattice
- optional discrete coupling springs between neighboring oscillators
- optional vertical onsite springs to a fixed upper support

Model:

```text
m q_ij'' = -k0 q_ij - kc (4 q_ij - q_{i+1,j} - q_{i-1,j} - q_{i,j+1} - q_{i,j-1})
```

Normal modes:

```text
q_ij^(m,n)(t) ~ sin(m pi i / (Nx+1)) sin(n pi j / (Ny+1)) cos(omega_mn t)
```

Render:

```bash
cd "/Users/dmitrijnaumov/Library/Mobile Documents/com~apple~CloudDocs/Projects/dvnanima/fields/coupled_oscillators_2d"
manim -pqh coupled_oscillators_2d.py CoupledOscillators2D
```

Main tuning in `run.cfg`:

- `mode_list`
- `onsite_k`
- `coupling_k`
- `mode_amplitude`
- `kick_i`, `kick_j`
- `kick_amplitude`
- `show_mesh`
- `show_coupling_springs`
- `show_vertical_springs`
- `spring_width`, `spring_turns`, `spring_stroke`
- `support_z`
- `vertical_spring_width`, `vertical_spring_turns`, `vertical_spring_stroke`
- `camera_phi`, `camera_theta`, `camera_zoom`
