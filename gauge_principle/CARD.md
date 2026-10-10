# CARD: gauge_principle

**Title RU:** Калибровочный принцип: два друга и поле
**Title EN:** The gauge principle: two friends and a field

**Caption RU:** Фаза волновой функции — договорённость. Общий сдвиг фаз двух путей не меняет интерференционную картину, относительный сдвиг двигает полосы на δ/2π периодов (точный расчёт). Если каждый наблюдатель выбирает фазу α(x) в своей точке, свободное уравнение перестаёт работать: градиент ∂α добавляет волне импульс, и пакет экспериментатора дрейфует со скоростью (k₀ + ⟨α′⟩)/m. Ковариантная производная D = ∂ + iqA с A → A − ∂α/q возвращает предсказание теоретика с точностью до ошибок округления (D ψ′ = e^{iα} D ψ). Напряжённость F = ∂A − ∂A от выбора фазы не зависит: у чисто калибровочного поля F = 0, а петля вокруг трубки потока даёт qΦ (формула Стокса) и сдвигает полосы на qΦ/2π периодов в любой калибровке.

**Caption EN:** The phase of the wave function is a convention. A common phase of two paths leaves the interference pattern unchanged, a relative phase moves the fringes by δ/2π periods (exact computation). If every observer chooses the phase α(x) at his own point, the free equation stops working: the gradient ∂α adds momentum to the wave and the experimenter's packet drifts with the velocity (k₀ + ⟨α′⟩)/m. The covariant derivative D = ∂ + iqA with A → A − ∂α/q restores the theorist's prediction to rounding errors (D ψ′ = e^{iα} D ψ). The field strength F = ∂A − ∂A does not depend on the convention: a pure-gauge field has F = 0, and the loop around a flux tube gives qΦ (Stokes) and moves the fringes by qΦ/2π periods in any gauge.

**poster_time:** 29.0 (film time: the two-path pattern with the fringes shifted by the phase shifter, the phase dials and the screen profile)

**Credit note RU:** Двухщелевая интерференция: точное решение параксиального уравнения (гауссовы пучки); частица на решётке с ковариантным лапласианом на связях (линиях Вильсона), точная эволюция диагонализацией гамильтониана; плоскость с трубкой потока, интегралы по контурам и по площади. Соглашения книги: ψ → e^{iα}ψ, A → A − ∂α/q, D = ∂ + iqA.
**Credit note EN:** Two-slit interference: the exact solution of the paraxial equation (Gaussian beams); a particle on a lattice with the covariant Laplacian built from the links (Wilson lines), exact time evolution by diagonalisation of the Hamiltonian; a plane with a flux tube, contour and area integrals. The book's conventions: ψ → e^{iα}ψ, A → A − ∂α/q, D = ∂ + iqA.

**Author:** Д. В. Наумов / D. V. Naumov; code_url: https://github.com/NeutrinoHit/dvnanima/tree/main/gauge_principle

**Book fields:** volume 2, chapter 23 (the chapter file has `\begin{mychapter}{GaugePrinciple}{22}`; chapter_title RU: «Калибровочная инвариантность», EN: "Gauge Invariance"). The film 0036 (electron kick) is already in this chapter.

**QR block:** `chapters/GaugePrinciple.tex` (EN: `en/chapters/GaugePrinciple.tex`): put `\QRwithPreview[0.6]{<preview>}{images/QRcodes/GaugePrinciple.pdf}{...}` right after the box «Сдвиг фазы» / "A Shift of Phase" (RU: after `\end{tcolorbox}` at line 50, before the section «КЭД по-новому»; EN: after `\end{tcolorbox}` at line 47, before `\section{QED in a New Way}`). Reason: the box tells the story of the two experimenters and the theorist; the film shows exactly this story and then the steps of the following sections (the transformation of the Lagrangian, the covariant derivative, the field strength and the Wilson loop). A preview image (a snapshot at film time 29 s) has to be made for `\QRwithPreview`.
