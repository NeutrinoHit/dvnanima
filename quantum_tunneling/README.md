# Tunnelling: a point body and an extended body

Book: the chapter on the Klein–Fock–Gordon equation, the paragraph on quantum tunnelling.

## What the film shows (about 106 s: a title card, three chapters with cards, a summary)

1. **A point body and a train.** A Gaussian hill `y = H exp(-x²/2σ²)`, `H = 4 m`, cars of 1 kg joined rigidly 1 m apart along the track. One degree of freedom, the arc length `S` of the head car:
   `L = ½ M Ṡ² − U(S)`, `U(S) = m g Σ h(x(S − i l))`, velocity Verlet (energy conserved to 1e-5). Every car has `E = V/2`. A single car turns back; a train of 20 passes, because the barrier
   of the whole body is `max U(S) < N V`. The scan over `N`: the threshold per car falls from `V` towards the area of the hill per car length (confirmed by bisection on the integrated motion).
2. **A classical wave tunnels.** A transverse chain of masses, `φ̈ₙ = (k/m)(φₙ₊₁ − 2φₙ + φₙ₋₁) − Ωₙ² φₙ`, with springs to the floor only in a segment of thickness `a` (`Ωₙ = Ω` there, 0 elsewhere).
   A wave of the frequency `ω < Ω` (leapfrog, a soft source, damping layers at the ends, `dx = 0.02`) gives a standing wave on the left, the decay `e^{−κx}`, `κ = √(Ω²−ω²)/c`, inside, and a running wave on the right.
   The measured `T` agrees with `T = [1 + sinh²(κa) / (4 (ω/Ω)² (1 − ω²/Ω²))]⁻¹` to 0.1 %, `T + R = 1` to 0.1 %, the decay rate agrees with `κ` to 5 %; three thicknesses are shown against the formula.
3. **The same law for matter waves.** The stationary Schrödinger equation `−ψ'' + (V − E)ψ = 0` is the same equation (`ω² → E`, `Ω² → V`). A wave of definite energy `E = V/2` on the Gaussian hill (exact scattering solution)
   and `log₁₀ T` of the hill against `√(m/m₀)`; the numbers `T ≈ e^{−2κa}` for an electron, a proton and a grain of 1 µg at `V − E = 1 eV`, `a = 0.5 nm`.
4. A summary of four statements.

## Not shown in the film

The code of two further parts is kept in `quantum_tunneling.py` and `tunnel_physics.py` (they are listed under `[timeline_unused]` in `config.toml`): a Gaussian wave packet on the hill with `P_T = ∫|φ(E)|² T(E) dE`,
and a Josephson junction as one collective coordinate (the tilted washboard `U(δ) = −E_J cos δ − (ħI/2e) δ`, the escape rate, `ΔU = 2E_J[√(1−s²) − s arccos s]`). The tests of these parts remain.

## Run

```bash
python quantum_tunneling.py --lang en            # media/quantum_tunneling_en.mp4
python quantum_tunneling.py --lang ru            # media/quantum_tunneling_ru.mp4
python quantum_tunneling.py --lang en --snapshot 52     # one PNG at the content time 52 s
python -m pytest test_quantum_tunneling.py       # physics and configuration tests
```

## Configuration

Every number is in `config.toml` (sections `video`, `timeline`, `classical`, `scan`, `view`, `quantum`, `wave`, `objects`, `mass_plot`, `layout`, `fonts`, `style`; the unused parts: `limits`, `josephson`, `circuit`),
every word in `texts.toml` (tables `[en]` and `[ru]` with the same keys). Override on the command line: `--config other.toml` or `--set classical.cars=12 --set wave.omega=0.7`.
The slow parts are cached in `.cache/`.

## Files

`quantum_tunneling.py` (the film), `tunnel_physics.py` (the physics), `config.toml`, `texts.toml`, `test_quantum_tunneling.py`, `CARD.md`.
