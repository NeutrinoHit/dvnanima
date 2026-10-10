# Card of the film `higgs_ff_spin` (for the site, the catalog and the book)

**Files:** `media/higgs_ff_spin_ru.mp4`, `media/higgs_ff_spin_en.mp4` (1280x720, 30 fps, 105.2 s: 95 s of content and six 1.7 s cards).
**poster_time:** 35.1 s of the film (content 30 s: the three cells m = 0, -1, +1 with the phase arrows; the same frame in both languages).

## Title
- RU: Спиновая и орбитальная структура пары f f̄ в распадах бозона Хиггса
- EN: Spin and orbital structure of the f f̄ pair in Higgs decays

## Caption
- RU: Бозон Хиггса (0⁺) распадается на пару f f̄ в P-волне (³P₀). Фильм строит волновую функцию ψ(r) относительной координаты r = x₁ − x₂ как сумму плоских волн e^{ipn·r} пар, разлетающихся «спиной к спине» во всех направлениях n: получается вихрь ψ ∝ j₁(pr), L = 1. Если зафиксировать проекции обоих спинов на общую ось, пара оказывается в состоянии с определённой проекцией m = −(ρ₁+ρ₂)/2 (J = 0): m = 0 при антипараллельных спинах (две доли вдоль оси) и m = ∓1 при параллельных (кольцо, фаза делает один оборот); на оси остаются только антипараллельные спины, в экваториальной плоскости только параллельные: |ψ|² ∝ j₁²·½[1 + ξ₁·ξ₂ − 2(ξ₁·r̂)(ξ₂·r̂)]. Псевдоскаляр (0⁻) даёт чистую S-волну, синглет, без зависимости от r̂. В смеси скаляра и псевдоскаляра интерференция ∝ ε₁ε₂ j₀j₁ ξ₁·(r̂×ξ₂) наклоняет облако, а обмен спинами f ↔ f̄ меняет наклон (CP-нечётность). Детектор проецирует пару на импульсные состояния, поэтому фаза e^{imφ} не видна: остаются спин-угловые корреляции. Цвет — фаза ψ, поверхности — постоянное |ψ|²; чистая модель, β = 0,996.
- EN: A scalar Higgs boson (0⁺) decays into an f f̄ pair in the P-wave (³P₀). The film builds the wave function ψ(r) of the relative coordinate r = x₁ − x₂ as a sum of the plane waves e^{ipn·r} of pairs flying apart back to back in all directions n: the result is a vortex ψ ∝ j₁(pr), L = 1. Fixing the spin projections of both particles on a common axis puts the pair into a state of definite m = −(ρ₁+ρ₂)/2 (J = 0): m = 0 for antiparallel spins (two lobes along the axis) and m = ∓1 for parallel ones (a ring, the phase winds once); on the axis only antiparallel spins survive, in the equatorial plane only parallel ones: |ψ|² ∝ j₁²·½[1 + ξ₁·ξ₂ − 2(ξ₁·r̂)(ξ₂·r̂)]. A pseudoscalar (0⁻) gives a pure S-wave, the singlet, with no dependence on r̂. In a scalar-pseudoscalar mixture the interference ∝ ε₁ε₂ j₀j₁ ξ₁·(r̂×ξ₂) makes the cloud lean, and exchanging the spins of f and f̄ reverses the lean (CP-odd). A detector projects the pair on momentum states, so the phase e^{imφ} is invisible and only spin-angle correlations survive. Colour is the phase of ψ, surfaces are |ψ|² = const; β = 0.996 (a b quark), the mixing ε₂/ε₁ = 1 is illustrative.

## Credit note
- RU: Модель: волновая функция пары в координатном представлении ψ_{ρ₁ρ₂}(r; s₁, s₂) = i m_f p/(16π²v) ∫dΩ_n e^{ipn·r} ū(pn)Γv(−pn), Γ = 1 (скаляр) или γ₅ (псевдоскаляр); ψ_S ∝ β j₁(pr) χ†(σ·r̂)η, ψ_P ∝ j₀(pr) χ†η; плотность в замкнутом виде (проверена численно, см. README). Код и расчёты: github.com/NeutrinoHit/dvnanima/tree/main/higgs_ff_spin.
- EN: Model: the coordinate-space wave function of the pair ψ_{ρ₁ρ₂}(r; s₁, s₂) = i m_f p/(16π²v) ∫dΩ_n e^{ipn·r} ū(pn)Γv(−pn), Γ = 1 (scalar) or γ₅ (pseudoscalar); ψ_S ∝ β j₁(pr) χ†(σ·r̂)η, ψ_P ∝ j₀(pr) χ†η; the densities in closed form (checked numerically, see the README). Code: github.com/NeutrinoHit/dvnanima/tree/main/higgs_ff_spin.


## Place in the book
- Volume 2, chapter 27 "Распады бозона Хиггса" / "Higgs Boson Decays" (`\begin{mychapter}{Higgs}{27}`).
- File: `chapters/Higgs.tex` (RU) and `en/chapters/Higgs.tex` (EN).
- Section `H -> f + fbar`, subsection "Спин-орбитальное состояние пары f f̄" / "Spin-orbital state of the f f̄ pair": **after the marginfigure `fig:higgs2ff-spin-longitudinal`** (end of the case (ii), just before `\section{\faCoffee \ $H\to Z+Z^*$}`), as a `\QRwithPreview`-style block (or a plain QR block if no preview image is wanted).
- Reason: this is the end of the discussion of the ³P₀ state, the transverse case S_ξ = +1, L_ξ = −1 and the longitudinal case L_n = 0, i.e. exactly the statements that the film shows as orbital states of definite m; the film also explains why the P-wave gives β³ (Eq. higgsdecays_4).
- Suggested block text (RU): "Анимация: пара f f̄ как волна относительной координаты; спины выбирают орбитальную проекцию m, псевдоскаляр — чистая S-волна, интерференция скаляра и псевдоскаляра наклоняет облако (CP-нечётность); детектор видит только спин-угловые корреляции."
