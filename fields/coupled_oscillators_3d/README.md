# 3D Coupled Oscillators

Cubic 3D lattice of coupled oscillators with fixed boundary.

What the scene shows:

- several normal modes with explicit formulas on screen
- a local displacement of one lattice site and its release
- subsequent spreading of the disturbance across the volume
- all nearest-neighbor springs of the cubic lattice
- a wireframe box showing the lattice extent

Model:

```text
m q_ijk'' = -k0 q_ijk - kc (6 q_ijk - sum over 6 nearest neighbors)
```

Normal modes:

```text
q_ijk^(a,b,c)(t) ~ sin(a pi i / (Nx+1)) sin(b pi j / (Ny+1)) sin(c pi k / (Nz+1)) cos(omega_abc t)
```

For a cubic lattice, each interior node has 6 nearest neighbors, not 4.

Render:

```bash
cd "/Users/dmitrijnaumov/Library/Mobile Documents/com~apple~CloudDocs/Projects/dvnanima/fields/coupled_oscillators_3d"
manim -pqh coupled_oscillators_3d.py CoupledOscillators3D
```

Main tuning in `run.cfg`:

- `nx`, `ny`, `nz`
- `mode_list`
- `onsite_k`, `coupling_k`
- `mode_amplitude`
- `kick_i`, `kick_j`, `kick_k`
- `kick_amplitude`
- `span_x`, `span_y`, `span_z`
- `disp_axis_x`, `disp_axis_y`, `disp_axis_z`
- `spring_width`, `spring_turns`, `spring_stroke`
- `camera_phi`, `camera_theta`, `camera_zoom`
