# KATRIN 2013 nominal magnetic-field reconstruction

## Primary source

F. Glück et al., “Electromagnetic design of the KATRIN large-volume air
coil system,” *New Journal of Physics* **15** (2013) 083025:
[arXiv:1304.6569](https://arxiv.org/abs/1304.6569),
[DOI 10.1088/1367-2630/15/8/083025](https://doi.org/10.1088/1367-2630/15/8/083025).

The current model is a reconstruction of the two nominal optimized
configurations in Tables 1 and 2 of that paper. It is not a reconstruction of
a later operational setting.

## Published input

Table 1 gives the axial location \(z_c\), typical maximum field \(B_c\), and
the axial contribution \(B_{z0}\) at the main-spectrometer centre:

| Source | \(z_c\), m | \(B_c\), T | \(B_{z0}\), µT |
|---|---:|---:|---:|
| WGTS coil system | -38.87 | 3.6 | -9.7 |
| DPS coil system | -27.25 | 5.0 | -16.3 |
| CPS coil system | -20.58 | 5.6 | -38.2 |
| PS1 coil | -16.46 | 4.5 | -18.5 |
| PS2 coil | -12.10 | 4.5 | -46.5 |
| PCH coil | +12.18 | 6.0 | -65.2 |
| DET coil | +13.78 | 3.6 | -48.4 |

The same table gives the axial Earth-field contribution \(+20\) µT. Thus at
the origin the published non-LFCS contribution is

\[
  B_{z,\mathrm{SC+Earth}}(0,0)
  = -242.8\ \mu\mathrm T + 20.0\ \mu\mathrm T
  = -222.8\ \mu\mathrm T.
\]

Table 2 gives the LFCS geometry and two sets of currents:

| LFCS | \(z_c\), m | turns | \(I\), A (one minimum) | \(I\), A (two minima) |
|---:|---:|---:|---:|---:|
| 1 | -6.79 | 14 | -11.2 | -0.5 |
| 2 | -4.94 | 14 | -15.3 | 0.0 |
| 3 | -4.04 | 8 | -7.9 | -4.8 |
| 4 | -3.14 | 8 | -13.4 | -7.1 |
| 5 | -2.24 | 8 | -12.2 | -6.6 |
| 6 | -1.34 | 8 | -24.2 | -19.4 |
| 7 | -0.44 | 8 | -17.1 | -57.2 |
| 8 | +0.46 | 8 | -20.3 | -51.2 |
| 9 | +1.35 | 8 | -18.5 | -22.7 |
| 10 | +2.26 | 8 | -23.1 | -12.5 |
| 11 | +3.16 | 8 | -21.9 | -7.7 |
| 12 | +4.06 | 14 | -18.1 | -16.8 |
| 13 | +4.95 | 14 | -13.3 | -15.9 |
| 14a | +6.60 | 14 | +27.3 | +42.1 |
| 14b | +6.90 | 14 | +27.3 | +42.1 |

Every pack has inner radius 6.3 m, radial thickness 0.02 m, and axial length
0.19 m. The paper reports \(|B(0,0)|=0.35\) mT for both configurations.

## Field equations

Each azimuthal current filament uses the exact Biot–Savart solution for a
circular loop. For loop radius \(a\), relative axial coordinate
\(\zeta=z-z_c\), and cylindrical radius \(\rho\), define

\[
 \alpha^2=(a-\rho)^2+\zeta^2,\qquad
 \beta^2=(a+\rho)^2+\zeta^2,\qquad
 k^2=\frac{4a\rho}{\beta^2}.
\]

Then

\[
 B_\rho =
 \frac{\mu_0NI\,\zeta}{2\pi\rho\beta}
 \left[
  -K(k^2)
  +\frac{a^2+\rho^2+\zeta^2}{\alpha^2}E(k^2)
 \right],
\]

\[
 B_z =
 \frac{\mu_0NI}{2\pi\beta}
 \left[
  K(k^2)
  +\frac{a^2-\rho^2-\zeta^2}{\alpha^2}E(k^2)
 \right].
\]

The analytic \(\rho=0\) limit is used on the axis. The LFCS winding-pack
cross-section is integrated by explicit Gauss–Legendre quadrature.

## What is measured and what is reconstructed

- LFCS positions, envelope dimensions, turn counts, and currents are direct
  published inputs.
- Individual conductor positions inside each 2 cm by 19 cm LFCS envelope are
  not tabulated. Uniform turn density in that envelope is therefore an
  explicit source-model choice, not a claim about undocumented winding
  placement. Its quadrature error is convergence-tested separately. Replacing
  every finite pack by a filament at the envelope centre changes the computed
  LFCS centre field by 0.0067 µT and 0.0145 µT in the two configurations. This
  is a sensitivity comparison between two documented representations, not a
  rigorous uncertainty bound on the unknown conductor placement.
- Table 1 does not specify full winding geometry for the seven
  superconducting systems. Each is represented by one equivalent loop that
  exactly matches both its published \(B_c\) and \(B_{z0}\). For a loop,

  \[
   \frac{|B_{z0}|}{B_c}
   =\frac{a^3}{(a^2+z_c^2)^{3/2}},
  \]

  which uniquely fixes \(a\), after which
  \(NI=2a\,\mathrm{sign}(B_{z0})B_c/\mu_0\).
- These equivalent loops are suitable for a traceable visualization of the
  central main-spectrometer field. They are not sufficient for precision
  tracking near the superconducting magnets or for reproducing an as-built
  operational field.
- Transverse Earth field, EMCS imperfections, magnetization, construction
  tolerances, and later SAP settings are outside this axisymmetric 2013 model.

## Reproduction result

With the published rounded currents, the implementation obtains at
\(r=z=0\):

| Configuration | LFCS, µT | SC, µT | Earth, µT | total, µT |
|---|---:|---:|---:|---:|
| one global minimum | -123.879808 | -242.800000 | +20.000000 | -346.679808 |
| two local minima | -128.031660 | -242.800000 | +20.000000 | -350.831660 |

Both agree with the paper’s rounded magnitude \(0.35\) mT. The test suite also
checks the loop solution against an independent direct Biot–Savart integral,
tests vacuum Maxwell residuals away from conductors, and verifies winding-pack
quadrature convergence.

## Focal-plane detector

The detector visualization follows J. F. Amsbaugh et al.,
“Focal-plane detector system for the KATRIN experiment,”
[arXiv:1404.2925](https://arxiv.org/abs/1404.2925). The documented detector is
a monolithic silicon p-i-n diode with 148 pixels and a 90 mm sensitive
diameter. Its topology consists of four central bullseye pixels and twelve
concentric rings of twelve equal-area pixels.

At the scale of the 23.28 m by 9.80 m spectrometer, a 90 mm disk would be
nearly invisible. The renderer therefore enlarges its diameter by \(35\times\)
and labels that display-only scale. The detector is placed schematically in
the published DET-magnet region at positive \(z\); this is not asserted to be
an as-built wafer coordinate.
