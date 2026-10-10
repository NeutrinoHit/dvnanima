# Attraction and repulsion as the interference of waves

Book: volume 1, chapter "Attraction and Repulsion" (the animations of the figures `interference_1d_attraction` and `interference_1d_repulsion`).
Two films, `--case attraction` and `--case repulsion` (QR 0022 and 0023), each in Russian and English.

## The model (unchanged from `fields/scalar_qed`)

Two Gaussian wave packets in the plane, `scalar_qed_numerics.py` (a copy of `fields/scalar_qed/numerics.py`): the free packet `φ_free` with the dispersive profile,
the potential `A_0` of each packet (the Coulomb potential smoothed on the size of the packet), the eikonal phase `θ` of the interaction, `φ = φ_free · exp(i χ)`, `χ = −s q² θ`
(`s = ±1` the sign of the charge); the centres of the packets follow the gradient of the interaction phase (the transverse projection). The parameters are those of
`scalar_qed_pm.toml` (attraction) and `scalar_qed_pp.toml` (repulsion): `m = 6`, `q = 0.88`, `σ = 0.38`, `p = 1.8`, charges `±4.2`.

## What the films show (98 s of content, three chapters)

1. **Two packets and their fields.** The density of each packet (warm colour: positive charge, blue: negative), the contours of `A_0`, the phase `χ`.
2. **The motion.** The packets fly past each other with a small impact parameter: the trails, the free (straight) motion dashed, the force on each packet,
   the distance between the packets and the transverse position of packet 1 against time.
3. **Interference.** Along the transverse line through packet 1: the phase `χ(s) − χ(0)` with its tangent, whose slope is the shift `ΔP = ⟨∇χ⟩` of the momentum
   (the density-weighted gradient of the phase, in agreement with `m Δv` of the centre to tens of per cent), and the momentum distribution of the free wave and of the free
   wave plus the wave of the interaction, `e^{i g s} φ_free`, `g = ΔP`: the distribution is shifted toward the other packet (attraction) or away from it (repulsion); the interference
   term (the difference of the two) is drawn filled.

## Run

```bash
python attraction_repulsion.py --case attraction --lang en        # media/attraction_en.mp4
python attraction_repulsion.py --case repulsion --lang ru         # media/repulsion_ru.mp4
python attraction_repulsion.py --case attraction --lang en --snapshot 86      # one PNG at the content time 86 s
python -m pytest test_attraction_repulsion.py
```

## Configuration

Every number is in `config.toml` (sections `video`, `scalar_qed`, `cases`, `view`, `timeline`, `style`, `fonts`, `layout`), every word in `texts.toml` (tables `[en]` and `[ru]`
with the same keys; the keys of the cases end with `_attraction` / `_repulsion`). Override: `--config other.toml` or `--set scalar_qed.mass=7 --set timeline.interf_t1=12`.
The simulation (about 30 s) is cached in `.cache/`.

## Files

`attraction_repulsion.py` (the film), `scalar_qed_numerics.py` (the model), `config.toml`, `texts.toml`, `test_attraction_repulsion.py`, `CARD.md`.
