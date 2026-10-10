# CARD: meissner_photon_mass

**Title RU:** Эффект Мейснера и масса фотона
**Title EN:** The Meissner effect and the photon mass

**Caption RU:** Конденсат с |φ|² = v²/2 даёт калибровочному полю массу m_A = ev. Сначала статика: диск конденсата в однородном магнитном поле, по мере роста v силовые линии вытесняются (в каждом кадре точное равновесное поле), вдоль разреза поле спадает как exp(−x/λ_L), λ_L = 1/m_A, а в слое течёт экранирующий ток j = −m_A²A. Затем динамика: уравнение ∂_t²A − ∂_x²A + m_A²(x)A = 0 решается численно. Пакет с ω < m_A отражается и оставляет затухающий хвост exp(−κx), κ = √(m_A² − ω²); пакет с ω > m_A проходит с k = √(ω² − m_A²) и групповой скоростью v_g = k/ω < 1. Наконец, дисперсия ω² = k² + m_A² и поляризации: у безмассового фотона две, у массивного три (продольная — голдстоуновская мода): 2 + 2 = 1 + 3. Конденсат считается заданным фоном.

**Caption EN:** A condensate with |φ|² = v²/2 gives the gauge field the mass m_A = ev. First the static picture: a disc of the condensate in a uniform magnetic field; as v grows the field lines are pushed out (each frame is the exact equilibrium field), along a cut the field decays as exp(−x/λ_L), λ_L = 1/m_A, and a screening current j = −m_A²A flows in the layer. Then the dynamics: the equation ∂_t²A − ∂_x²A + m_A²(x)A = 0 is solved numerically. A packet with ω < m_A is reflected and leaves an evanescent tail exp(−κx), κ = √(m_A² − ω²); a packet with ω > m_A is transmitted with k = √(ω² − m_A²) and group velocity v_g = k/ω < 1. Finally the dispersion ω² = k² + m_A² and the polarizations: a massless photon has two, a massive one three (the longitudinal one is the Goldstone mode): 2 + 2 = 1 + 3. The condensate is a fixed background.

**poster_time:** 22.0 (film time: the disc with the field lines expelled, the screening current and the profile exp(−x/λ_L))

**Credit note RU:** Абелева модель Хиггса / уравнение Прока (m_A = ev) и уравнение Лондонов; статика: точное решение для цилиндра в поперечном поле, динамика: численное решение (leapfrog) уравнения ∂_t²A − ∂_x²A + m_A²(x)A = 0.
**Credit note EN:** Abelian Higgs model / Proca equation (m_A = ev) and the London equation; statics: the exact solution for a cylinder in a transverse field; dynamics: numerical (leapfrog) solution of ∂_t²A − ∂_x²A + m_A²(x)A = 0.

**Author:** Д. В. Наумов / D. V. Naumov; code_url: https://github.com/NeutrinoHit/dvnanima/tree/main/meissner_photon_mass

**Book fields:** volume 2, chapter 24 (chapter_title RU: «Откуда у частицы масса?», EN: "Where Does a Particle Get Its Mass?").

**QR block:** `chapters/SymmetryBreaking.tex` (EN: `en/chapters/SymmetryBreaking.tex`), subsection «Сверхпроводимость и эффект Мейснера» / "Superconductivity and the Meissner Effect": put `\QRwithPreview[0.6]{<preview>}{images/QRcodes/MeissnerPhotonMass.pdf}{...}` right after the paragraph that ends with the solution `A(r) = A(0)e^{-m_γ r}` ("Это решение описывает экспоненциальное затухание поля", RU line ≈ 424) and before the box «Механизм Хиггса и сверхпроводимость». Reason: this is where the book derives the exponential decay and the photon mass in the superconductor; the film shows that decay, the screening current, the packet at the boundary and the 2 + 2 = 1 + 3 counting of the box that follows. A preview image (e.g. a snapshot at film time 22 s) has to be made for `\QRwithPreview`.
