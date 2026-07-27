# What the 3D collimation animation shows

## Pitch angle

The pitch angle \(\alpha\) is the acute angle between the electron momentum
\(\mathbf p\) and the local magnetic-field line:

\[
 \alpha=\arctan\frac{p_\perp}{|p_\parallel|}.
\]

Thus \(\alpha=0^\circ\) means motion along a magnetic line, while
\(\alpha=90^\circ\) means purely transverse motion. The animation translates
this explicitly as “угол между импульсом \(p\) и магнитной линией \(B\)”.

## Analyzing plane and detector

For the symmetric 2013 configuration, the analyzing plane is at \(z=0\), in
the middle of the main spectrometer. It is not a material screen and is not
inside the detector. It is the low-field, high-electron-potential-energy
region where the longitudinal kinetic energy reaches its minimum.

Electrons above threshold cross this plane, are reaccelerated, and later reach
the separate focal-plane detector. Electrons below threshold normally turn
around before reaching it. The present animation has \(\mathbf E=0\), so all
three displayed electrons cross the plane; it does not yet calculate the
transmission/reflection boundary.

The displayed detector reproduces its documented 148-pixel topology:
four central bullseye pixels plus twelve rings of twelve pixels. Its physical
active diameter is 90 mm. It is enlarged by a factor of 35 in the scene and
placed schematically in the documented detector-magnet region because an
as-built axial wafer coordinate is not part of the present central-field
model.

## Why L14a and L14b carry the opposite current

This is intentional. In the published 2013 KATRIN LFCS configuration, all
packs L1–L13 carry negative current, while the two 14-turn subcoils L14a and
L14b each carry \(+27.3\ \mathrm A\). The authors explicitly call L14 a
*counter coil*. It compensates the larger detector-side stray field produced
by the pinch and detector superconducting magnets.

The values and geometry come from F. Glück et al.,
“Electromagnetic design of the KATRIN large-volume air coil system,”
[arXiv:1304.6569](https://arxiv.org/abs/1304.6569). The complete table used by
the model is reproduced in [SOURCES.md](SOURCES.md).

## Why the LFCS currents do not fall from L1 to L14

L1–L14 are not the source and detector solenoids. They are 12.6 m diameter
air coils forming the *Low Field Correction System* around the main
spectrometer. Their tens-of-ampere currents fine-shape a millitesla field
created by the superposition of all LFCS coils, the distant superconducting
magnets, and the Earth field.

The strong fields are produced by separate superconducting systems:
WGTS 3.6 T, PS2 4.5 T, PCH 6 T, and DET 3.6 T. In the documented scalar
surrogate their effective ampere-turn values are of order
\(2\text{--}3\times10^6\ \mathrm{A\,turn}\), compared with roughly
\(10^2\ \mathrm{A\,turn}\) for one LFCS pack.

Therefore neither the LFCS index nor its current magnitude is a local field
meter. Field strength also depends on turn count, coil radius, distance, and
the signed superposition of every source. The published LFCS currents were
optimized jointly to obtain the required field shape.

The legacy central full-orbit benchmark samples only

\[
 |B|:\quad
 0.819\ \mathrm{mT}\;(z=-6.5\ \mathrm m)
 \longrightarrow
 0.347\ \mathrm{mT}\;(z=0)
 \longrightarrow
 1.014\ \mathrm{mT}\;(z=+6.5\ \mathrm m).
\]

The new primary animation starts at the WGTS source, where the reconstructed
field is 3.603 T, and follows the guiding centre to the DET region. Its
logarithmic field inset is the relevant comparison: the LFCS currents do not
parameterize the strong-field magnets.

## Magnetic collimation is reversible

For the first animation \(\mathbf E=0\), so the full relativistic Lorentz
equations imply

\[
 \frac{d}{dt}(\gamma mc^2)
 =q\mathbf v\cdot(\mathbf v\times\mathbf B)=0.
\]

The magnetic field does no work: the speed and total kinetic energy stay
constant. In the adiabatic regime the gyro-averaged magnetic moment is
approximately conserved,

\[
 \mu=\frac{p_\perp^2}{2mB}\approx\mathrm{const}.
\]

Consequently, while \(B\) decreases toward the analyzing plane,
\(p_\perp\) decreases and \(p_\parallel\) increases. Beyond the field minimum,
\(B\) rises, and the conversion reverses. Thus the velocity becomes most
parallel at the analyzing plane; it does not remain permanently collimated
after leaving the spectrometer.

The primary source-to-detector animation imposes this adiabatic invariant and
is labelled accordingly. It does not claim to be a full-orbit calculation
through an unpublished as-built field map. The independent full-Lorentz
benchmark on \(-6.5\le z\le+6.5\) m is retained as a numerical check.

## What changes when the electrostatic field is added

For static electric and magnetic fields the conserved quantity is

\[
 \gamma mc^2+q\Phi(\mathbf x).
\]

KATRIN electrons enter an increasingly negative electric potential. Since
\(q=-e\), their electrostatic potential energy increases and their kinetic
energy decreases until the analyzing plane. An electron above threshold
passes it and is reaccelerated on the detector side. An electron below
threshold reaches \(p_\parallel=0\) and is reflected.

Therefore “collimation” and “retardation” must not be conflated:

- the falling magnetic field converts transverse momentum into longitudinal
  momentum;
- the electric field then removes kinetic energy, mainly from the remaining
  longitudinal motion near the analyzing plane;
- after the center, both processes run in reverse for a transmitted electron.

## Source-to-detector trajectories

The three colored paths begin at the same 18 mm guiding-centre radius in the
WGTS field and use source pitch angles of 10°, 30°, and 50°. Different
azimuths keep their guiding-centre paths visually separable. The source flux
tube radius is 45 mm; the computed field line expands to 4.746 m in the
analyzing plane.

| source pitch | pitch near \(z=0\) | pitch in DET region | turns | flight time |
|---:|---:|---:|---:|---:|
| 10° | 0.096° | 10.045° | 6772 | 675.6 ns |
| 30° | 0.278° | 30.149° | 7563 | 684.9 ns |
| 50° | 0.426° | 50.308° | 11396 | 718.9 ns |

The corresponding Larmor radii in the analyzing plane are 2.31, 6.63, and
9.83 mm. They are stored and displayed numerically without enlargement.
The current primary renderer shows the component fractions in fixed-screen
bars and does not draw velocity arrows on the electrons.

## Full-Lorentz central benchmark

The source angles first map adiabatically to the entrance of the directly
integrated interval at \(z=-6.5\) m. From there the code integrates

\[
 \dot{\mathbf x}=\frac{\mathbf p}{\gamma m},\qquad
 \dot{\mathbf p}=q\,\mathbf v\times\mathbf B
\]

without imposing magnetic-moment conservation:

| source pitch | adiabatic pitch at \(z=0\) | full-Lorentz pitch at \(z=0\) | relative difference |
|---:|---:|---:|---:|
| 10° | 0.097591° | 0.095458° | −2.1861% |
| 30° | 0.281003° | 0.274860° | −2.1862% |
| 50° | 0.430524° | 0.421111° | −2.1864% |

Thus the adiabatic approximation is quantitatively checked, not silently
treated as exact. A precision full-orbit result for the whole beamline still
requires the experiment's detailed superconducting-coil configuration or
field map; the public Kassiopeia repository supplies the tracking engine but
does not ship that KATRIN installation data.

## Why the schematic drawing has more oscillations

For an 18.6 keV electron,

\[
 f_c=\frac{|q|B}{2\pi\gamma m}.
\]

At the 3.6 T source the cyclotron frequency is approximately 97 GHz. The
source-to-detector calculation gives 6772–11396 turns. Showing all of them in
14 s at 30 fps would require far more than the available 420 frames and, on
the apparatus scale, their 0.02–0.10 mm source radii would be sub-pixel.

The primary animation therefore draws the computed guiding centres and reports
the physical Larmor radius and accumulated turn count. It does not draw a
false, enlarged helix. Published MAC-E cartoons do enlarge and sparsify the
helix deliberately so the pitch change remains visible.
