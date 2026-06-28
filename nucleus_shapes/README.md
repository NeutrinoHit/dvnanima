# Nucleus Shapes

Short 3D manim scene for lecture inserts about nuclear shapes.

The scene renders:

- a single solid nuclear surface
- smooth morphs between several standard deformation presets

Available shape presets:

- `sphere`
- `prolate`
- `oblate`
- `triaxial`
- `pear`

Main tuning lives in `run.cfg`:

- `shape_sequence`
- `nucleus_radius`
- camera, spin, and lighting parameters
- frame aspect ratio in `[manim]`

Render:

```bash
cd "/Users/dmitrijnaumov/Library/Mobile Documents/com~apple~CloudDocs/Projects/dvnanima/nucleus_shapes"
./render.sh
```

The default setup is intentionally minimal: no labels, no nucleons, no internal scaffolding.
