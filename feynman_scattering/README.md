# Feynman Scattering Animations

Two Manim scenes with the same visual language as the other repository projects:

- `ElectronElectronScattering`: electron-electron scattering with virtual photon exchange.
- `QuarkQuarkScattering`: quark-quark scattering with virtual gluon exchange.
- `NuMuElectronToNuEMuonScattering`: \(\nu_\mu e^- \to \nu_e \mu^-\) with virtual \(W^-\) exchange.
- `NuMuMuonScatteringViaZ`: \(\nu_\mu \mu^- \to \nu_\mu \mu^-\) with virtual \(Z^0\) exchange.

## Render

From `feynman_scattering/`:

```bash
manim -pqh feynman_scattering.py ElectronElectronScattering
manim -pqh feynman_scattering.py QuarkQuarkScattering
manim -pqh feynman_scattering.py NuMuElectronToNuEMuonScattering
manim -pqh feynman_scattering.py NuMuMuonScatteringViaZ
```

Use `run.cfg` to tweak colors, geometry, and timing.
