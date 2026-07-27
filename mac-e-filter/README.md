# MAC-E Filter Animations

This directory will contain reusable, physics-first animations of

1. relativistic charged-particle motion in a uniform magnetic field;
2. magnetic adiabatic collimation;
3. electrostatic retardation and transmission/reflection in a MAC-E filter;
4. a KATRIN-specific scene once a traceable KATRIN field model or field map is
   selected.

## Tooling decision

The numerical model is independent of every renderer.

- Python/NumPy/SciPy is the single source of truth for fields, trajectories,
  diagnostics, and reusable datasets.
- PyQtGraph is the first renderer: it is fast enough for interactive
  inspection and makes numerical/debug overlays practical.
- HTML/WebGL is a second renderer for interactive Quarto slides. It consumes
  the same precomputed trajectory dataset; it does not reimplement the
  equations of motion in JavaScript.
- Manim is optional for narrated, precisely choreographed video. It also
  consumes the precomputed dataset and never advances the physical state.

This follows the useful compute-once/render-many pattern already used by
`dvnanima/fields/radiating_charge`.

## Physical model

The reference trajectory is obtained from the relativistic Lorentz equations

\[
 \dot{\mathbf x} =
 \frac{\mathbf p}{\gamma m},
 \qquad
 \dot{\mathbf p} =
 q\left(\mathbf E(\mathbf x,t)+
 \frac{\mathbf p}{\gamma m}\times\mathbf B(\mathbf x,t)\right),
 \qquad
 \gamma =
 \sqrt{1+\frac{\lVert\mathbf p\rVert^2}{m^2c^2}}.
\]

For static fields the implementation must monitor

\[
 \mathcal E = \gamma mc^2 + q\Phi(\mathbf x)
\]

as a numerical invariant. In uniform magnetic field it must be compared with
the exact relativistic helical solution.

The guiding-centre trajectory and the adiabatic invariant may be computed as
diagnostics and explanatory overlays, but they must not replace the full
Lorentz trajectory used as the reference solution.

## Accuracy policy

“No unjustified approximations” means:

- every physical model and numerical approximation is explicit;
- uniform-field motion has an analytic reference test;
- time-step and field-grid convergence are tested;
- conservation residuals are stored in the dataset;
- interpolated field maps include interpolation and Maxwell-residual checks;
- an idealized axisymmetric MAC-E model is labelled as idealized;
- a scene is labelled “KATRIN” only when its field/geometry data have
  traceable provenance;
- visual time remapping, helix-radius exaggeration, and geometry deformation
  are disabled by default and clearly labelled if used for explanation.

Radiation reaction, collisions, scattering, and residual-gas effects are
separate model choices. They are not silently omitted or added: their relevance
must be bounded for the scenario and recorded in dataset metadata.

## Planned structure

```text
mac-e-filter/
├── README.md
├── pyproject.toml
├── configs/
│   ├── uniform_b.toml
│   ├── ideal_mac_e.toml
│   └── katrin.toml
├── src/mac_e_filter/
│   ├── dynamics.py
│   ├── fields.py
│   ├── diagnostics.py
│   ├── data_io.py
│   └── scenarios.py
├── renderers/
│   ├── pyqtgraph_viewer.py
│   ├── manim_scene.py
│   └── web/
└── tests/
    ├── test_uniform_field.py
    ├── test_conservation.py
    └── test_convergence.py
```

Generated datasets and movies remain untracked under `datasets/` and `media/`.

## Implemented KATRIN reference field

The first field model reproduces the two published nominal 2013 LFCS
configurations from Glück et al. The finite LFCS winding envelopes use the
published positions, dimensions, turn counts, and signed currents.

The 2013 paper does not publish enough winding geometry to reconstruct the
full superconducting field uniquely. Those remote systems are therefore
represented by explicitly labelled equivalent loops constrained to reproduce
both scalar values given for each source in the paper. This is a documented
central-spectrometer surrogate, not an as-built operational map.

See [SOURCES.md](SOURCES.md) for the complete provenance, equations, numerical
results, and model boundary.

The first full relativistic 18.6 keV electron trajectory is described in
[TRAJECTORY.md](TRAJECTORY.md). The electric field is deliberately zero in
that first scenario. A separate validated transmission/reflection calculation
is described in [ELECTROSTATIC.md](ELECTROSTATIC.md).

The 3D ensemble animation and the reversible nature of magnetic collimation
are explained in [COLLIMATION.md](COLLIMATION.md).

The separate constant-field animation with three exact relativistic helices is
documented in [UNIFORM_B.md](UNIFORM_B.md).

Run the checks and the diagnostic plot from `dvnanima`:

```bash
PYTHONPATH=mac-e-filter/src \
  python -m unittest discover -s mac-e-filter/tests -v

MPLCONFIGDIR=/tmp/mac-e-filter-matplotlib \
  python mac-e-filter/renderers/plot_katrin_2013_field.py

PYTHONPATH=mac-e-filter/src \
  python mac-e-filter/export_katrin_trajectory.py

PYTHONPATH=mac-e-filter/src \
  python mac-e-filter/export_katrin_collimation_ensemble.py

MPLCONFIGDIR=/tmp/mac-e-filter-matplotlib \
PYTHONPATH=mac-e-filter/src \
  python mac-e-filter/renderers/plot_katrin_trajectory.py

bash mac-e-filter/20_render_katrin_trajectory_pyqtgraph.sh

bash mac-e-filter/21_render_katrin_spectrometer_3d.sh

bash mac-e-filter/02_export_katrin_source_to_detector.sh

bash mac-e-filter/22_render_katrin_source_to_detector_4k.sh

bash mac-e-filter/03_export_katrin_mac_e.sh

bash mac-e-filter/23_render_katrin_mac_e_4k.sh

bash mac-e-filter/04_export_uniform_b_ensemble.sh

bash mac-e-filter/24_render_uniform_b_ensemble_4k.sh
```

Commands `02` and `22` build the primary source-to-detector animation. Its
calculation starts in the 3.603 T WGTS field, follows three adiabatic
guiding-centre paths through the analyzing plane to the detector region, and
exports a native 3840×2160 H.264 movie. The older 3D animation is retained as
the full-Lorentz central-region benchmark.

Commands `03` and `23` build the electrostatic MAC-E scene.
The electric potential is a vacuum-Laplace cylindrical surrogate constrained
by the documented spectrometer dimensions and the 18.6 kV retarding scale.
The animation shows two transmitted electrons and one reflected electron;
their classification is independently checked by full relativistic Lorentz
integration in the central region.

Commands `04` and `24` build the standalone uniform-field scene. All three
electrons have the same kinetic energy and evolve for the same physical time,
but start at 15°, 45°, and 75° to the field. Their trajectories use the exact
relativistic solution and are independently checked against DOP853 integration.

## Implementation order

1. Exact uniform-\(\mathbf B\) orbit and invariant/convergence tests.
2. Reusable dataset schema and PyQtGraph viewer.
3. Maxwell-consistent idealized MAC-E fields and full-orbit integration.
4. Transmission/reflection scan against retarding potential.
5. KATRIN field input with provenance and validation.
6. HTML/WebGL and, if useful, Manim renderers consuming the same datasets.
