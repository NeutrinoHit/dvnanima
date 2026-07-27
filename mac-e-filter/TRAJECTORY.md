# Full-orbit KATRIN trajectories

This scenario follows one electron through the published 2013 nominal
magnetic-field reconstruction. It is a full relativistic Lorentz trajectory,
not a guiding-centre trajectory.

## Equations integrated

The numerical momentum variable is

\[
 \mathbf u=\frac{\mathbf p}{mc},\qquad
 \gamma=\sqrt{1+|\mathbf u|^2}.
\]

With no electric field in this first scenario,

\[
 \frac{d\mathbf x}{dt}=\frac{c\mathbf u}{\gamma},
 \qquad
 \frac{d\mathbf u}{dt}
 =\frac{q}{m\gamma}\mathbf u\times\mathbf B(\mathbf x).
\]

The solver is the adaptive eighth-order DOP853 method. Its maximum time step
is independently limited so that the phase advance computed from the declared
upper field bound is at most 0.15 rad. The stored trajectory uses 64-bit
floating-point values.

## Reproducible scenario

The source of truth is
[`configs/katrin_2013_trajectory.toml`](configs/katrin_2013_trajectory.toml):

- electron kinetic energy: 18.6 keV;
- initial momentum angle: 30° to the positive \(z\) flight direction;
- initial position: \((0,0,-6.5\ \mathrm m)\);
- stopping plane: \(z=+6.5\ \mathrm m\);
- magnetic field: published 2013 “one global minimum” configuration;
- electric field: zero.

The last item is essential: this demonstrates full-orbit motion and magnetic
collimation, but it is not yet a MAC-E transmission calculation through a
retarding electrostatic potential.

## Numerical validation

The same integrator is tested against the exact relativistic helix in a
uniform field. In the KATRIN run:

| Quantity | Result |
|---|---:|
| flight time | 179.408605 ns |
| maximum cylindrical radius | 0.820198 m |
| sampled field range | 0.346592–1.008342 mT |
| initial/final instantaneous pitch | 30.0000° / 33.5909° |
| peak-to-peak kinetic-energy span divided by \(K_0\) | \(5.98\times10^{-12}\) |
| peak-to-peak \(\mu\) span divided by \(\mu_0\) | 0.11975 |

Here

\[
 \mu=\frac{p_\perp^2}{2m|\mathbf B|}
\]

is recorded only as an adiabatic diagnostic. It is not imposed on the
dynamics and is not expected to be an exact invariant of the full orbit. The
reported span is instantaneous, not gyro-averaged, and therefore must not be
confused with numerical integration error.

A second calculation reduces the maximum gyro-phase step by two and all
adaptive tolerances by four. Comparing the two trajectories at equal \(z\)
gives:

| Convergence comparison | Difference |
|---|---:|
| flight time | \(3.47\times10^{-18}\) s |
| endpoint position | \(2.88\times10^{-10}\) m |
| maximum path-position difference | \(5.07\times10^{-11}\) m |
| maximum normalized-momentum difference | \(1.63\times10^{-11}\) |

These numbers estimate numerical integration error only. The larger physical
model limitation remains the documented equivalent-loop representation of
the superconducting sources described in [`SOURCES.md`](SOURCES.md).

## Run

From `dvnanima`:

```bash
PYTHONPATH=mac-e-filter/src \
  python mac-e-filter/export_katrin_trajectory.py

MPLCONFIGDIR=/tmp/mac-e-filter-matplotlib \
PYTHONPATH=mac-e-filter/src \
  python mac-e-filter/renderers/plot_katrin_trajectory.py
```

The first command writes a renderer-independent NPZ dataset. The second reads
that dataset without recomputing the trajectory.

## PyQtGraph animation

The animation renderer also reads the NPZ dataset; it never advances the
particle state. It displays:

- the \(z\)-\(y\) projection in physical coordinates;
- the transverse \(x\)-\(y\) projection with equal axis scale;
- the field magnitude and instantaneous pitch sampled by the electron;
- physical time, local relativistic cyclotron frequency, and accumulated
  cyclotron phase.

The default 12 s moving sequence represents 179.409 ns of physical flight
time, a visual slowdown of \(6.689\times10^7\). The slowdown affects playback
only. Coordinates, orbit radius, cyclotron phase, and the final 2.317318 turns
are not exaggerated.

Interactive viewer:

```bash
PYTHONPATH=mac-e-filter/src \
  python mac-e-filter/renderers/pyqtgraph_katrin_trajectory.py
```

MP4 export:

```bash
bash mac-e-filter/20_render_katrin_trajectory_pyqtgraph.sh
```

The checked render settings are 1600 by 900 pixels, H.264, 30 fps, with
0.5 s and 1.0 s presentation holds before and after the physical motion.

## Primary source-to-detector 3D scene

The primary presentation begins in the strong WGTS field rather than at the
edge of the main spectrometer:

```bash
bash mac-e-filter/02_export_katrin_source_to_detector.sh
bash mac-e-filter/22_render_katrin_source_to_detector_4k.sh
```

The first command computes the axisymmetric field lines and relativistic
adiabatic transport from \(z=-38.87\) m to \(z=+13.78\) m. The second exports
the checked native 3840×2160 scene. It shows the logarithmic field profile,
the published LFCS current table, the superconducting-magnet locations,
the analyzing plane, and the segmented detector.

Because a public as-built KATRIN field map was not found, this calculation is
explicitly a guiding-centre result in the documented 2013 scalar surrogate.
The central full-Lorentz solution below is retained as its independent check.

## Legacy central full-orbit 3D scene

The central benchmark animation
shows three electrons moving in three dimensions inside a wireframe vessel
and the LFCS winding packs surrounding them:

```bash
bash mac-e-filter/01_export_katrin_collimation_ensemble.sh
bash mac-e-filter/21_render_katrin_spectrometer_3d.sh
```

The scene distinguishes physical data from presentation geometry:

- the electron coordinates are three unscaled full-Lorentz solutions with
  local starting pitch angles 10°, 30°, and 50°;
- green and magenta vectors show the instantaneous \(v_\parallel\) and
  \(v_\perp\) components; their displayed total length is normalized, while
  their directions and relative lengths are physical;
- LFCS pack centres, inner radius, radial/axial envelope, number of turns, and
  signed currents are the published 2013 inputs;
- because individual conductor coordinates inside each pack were not
  published, the visible turns are uniformly spaced through the published
  19 cm axial envelope at the 6.31 m mean radius; this affects only the drawing,
  not the field calculation;
- blue and red rings denote negative and positive conventional current;
  arrowheads show its direction according to the model sign convention;
- badges above the scene give the signed current of every pack; L14a and L14b
  are the published detector-side counter-coil, not a drawing error; the
  badges are arranged as a compact table outside the 3D viewport;
- a logarithmic axial-field inset distinguishes the 3.6–6 T superconducting
  systems from the highlighted 0.35–1.01 mT interval actually traversed by
  the full-orbit calculation;
- fixed-screen bars show \(v_\perp/v\) and \(v_\parallel/v\) for every
  electron, independently of gyro-phase and perspective;
- a translucent cyan disk marks the \(z=0\) analyzing plane in the middle of
  the spectrometer; it is a coordinate surface, not a material component;
- the detector-side target reproduces the 148-pixel FPD segmentation
  \(4+12\times12\); its documented 90 mm active diameter is enlarged by
  \(35\times\) and its axial placement is explicitly schematic;
- the wireframe vessel is a schematic ellipsoidal envelope constrained only by
  the documented 23.28 m length and 9.80 m maximum diameter; it is not an
  as-built CAD surface;
- line widths and the electron marker are screen-space locators, while the
  orbit and coil coordinates are not enlarged;
- the export uses a fixed camera so projected-vector changes cannot be
  mistaken for changes of the physical component magnitudes.

The scene still has \(E=0\). Adding a retarding potential requires a separately
validated electrostatic field model.

The reversible collimation, the counter-coil, and the distinction between
magnetic collimation and electrostatic retardation are explained in
[COLLIMATION.md](COLLIMATION.md).
