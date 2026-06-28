# Scalar QED Packet Fields

This project renders analytic scalar-QED wave-packet visualizations.

The current workflow is deliberately simple:

1. Compute a reusable `.npz` dataset.
2. Render it with PyQtGraph.

PyQtGraph is the primary engine here because it is the fastest local renderer
for the 2D and 3D packet-field views.

## Model

- Each packet is a narrow Gaussian packet in momentum space, sampled on a 2D
  spatial slice.
- The free packet uses the dispersive coordinate-space profile
  `Psi_a ~ (1 + i tau_a)^(-3/2) exp[-sigma_a^2 r_{a*}^2 / (1 + i tau_a)]`.
- The analytic electromagnetic potential of each packet is
  `A_a^mu = (Q_a / (m q)) p_a^mu V_{s_a}(r_{a*})`.
- The interaction effect is encoded through an eikonal phase.
- Packet centers are reconstructed from the interaction phase gradient, not
  from an externally imposed Newtonian force.

The default lower surface is `phi_abs2 = |phi|^2`.
The default upper surface is `a0`.

## Files

- `scalar_qed.toml`: default lecture-quality configuration, currently the
  like-charge repulsion case.
- `scalar_qed_pp.toml`: like charges, repulsion.
- `scalar_qed_pm.toml`: opposite charges, attraction.
- `scalar_qed_single.toml`: one moving packet and the field it generates.
- `numerics.py`: analytic packet, field, phase, and trajectory construction.
- `data_io.py`: dataset save/load helpers.
- `export_scalar_qed_dataset.py`: computes a dataset once.
- `render_scalar_qed_pyqtgraph.py`: primary 2D/3D renderer.
- `20_render_pyqtgraph_2d_mp4.sh`: quick 2D movie, useful for tuning.
- `21_render_pyqtgraph_3d_mp4.sh`: final lecture-quality 3D movie.

Generated outputs are intentionally ignored by git:

- `datasets/`: `.npz` datasets
- `media/`: rendered movies

`datasets/scalar_qed_dataset_latest.npz` is a local symlink updated by
`00_export_dataset.sh`. The render scripts read from `datasets/` by default
and write movies to `media/` by default.

## Recommended Commands

Run from this directory:

```bash
cd /Users/dmitrijnaumov/Documents/NeutrinoHit/dvnanima/fields/scalar_qed
```

Create a repulsion dataset:

```bash
./00_export_dataset.sh --config scalar_qed_pp.toml
```

Create an attraction dataset:

```bash
./00_export_dataset.sh --config scalar_qed_pm.toml
```

Create a one-packet dataset:

```bash
./00_export_dataset.sh --config scalar_qed_single.toml
```

Render the latest dataset as the final 3D PyQtGraph movie:

```bash
./21_render_pyqtgraph_3d_mp4.sh
```

Render a specific dataset:

```bash
./21_render_pyqtgraph_3d_mp4.sh \
  --dataset datasets/scalar_qed_dataset_pp_YYYYMMDD_HHMMSS.npz \
  --out media/pp.mp4
```

Render a quick 2D diagnostic movie:

```bash
./20_render_pyqtgraph_2d_mp4.sh \
  --dataset datasets/scalar_qed_dataset_pm_YYYYMMDD_HHMMSS.npz
```

For quick checks without the full grid/time cost:

```bash
./00_export_dataset.sh --config scalar_qed_single.toml --preview --out /tmp/scalar_qed_single_preview.npz
./21_render_pyqtgraph_3d_mp4.sh --dataset /tmp/scalar_qed_single_preview.npz --out /tmp/scalar_qed_single_preview.mp4
```

## Rebuild All Lecture Movies

Use deterministic dataset names when you want simple repeatable render
commands:

```bash
./00_export_dataset.sh --config scalar_qed_single.toml --out datasets/single.npz
./00_export_dataset.sh --config scalar_qed_pm.toml --out datasets/pm.npz
./00_export_dataset.sh --config scalar_qed_pp.toml --out datasets/pp.npz
```

Render the final 3D movies:

```bash
./21_render_pyqtgraph_3d_mp4.sh --dataset datasets/single.npz --out media/single.mp4
./21_render_pyqtgraph_3d_mp4.sh --dataset datasets/pm.npz --out media/pm.mp4
./21_render_pyqtgraph_3d_mp4.sh --dataset datasets/pp.npz --out media/pp.mp4
```

## Final 3D Render Defaults

`21_render_pyqtgraph_3d_mp4.sh` currently uses:

- `--mode 3d`
- `--show-upper`
- `--window-width 1312`
- `--window-height 740`
- `--height-scale 2.10`
- lower camera: elevation `24`, azimuth `-42`, distance `31`
- upper camera: elevation `5`, azimuth `28`, distance `29`
- `--fps 24`

On a Retina display the exported movie may have twice the logical window size.
The current settings were chosen to keep the two panels readable in lecture
slides while keeping rendering fast.

Useful overrides:

```bash
./21_render_pyqtgraph_3d_mp4.sh --show-centers
./21_render_pyqtgraph_3d_mp4.sh --time-stride 2
./21_render_pyqtgraph_3d_mp4.sh --space-stride 2
./21_render_pyqtgraph_3d_mp4.sh --window-width 1600 --window-height 900
```
