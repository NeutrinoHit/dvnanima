# Card for publication: chiral_fermion_mass

**Files:** `media/chiral_fermion_mass_ru.mp4`, `media/chiral_fermion_mass_en.mp4` (1280x720, 30 fps, 89.6 s). Folder `dvnanima/chiral_fermion_mass`.

## Title
- RU: Киральные волны и масса фермиона
- EN: Chiral waves and the fermion mass

## Caption (3-5 sentences, formulas in plain text)
- RU: Без массы уравнение Дирака в (1+1) измерениях распадается на две независимые волны: psi_R бежит вправо, psi_L влево, обе со скоростью света, и киральность совпадает с направлением движения. Хиггсовский конденсат v связывает их, m = f v / sqrt(2): psi_R становится источником psi_L и наоборот, киральность <gamma_5> колеблется с частотой 2E = 2 sqrt(p^2 + m^2), а пакет замедляется до v = p/E < c. Для покоящейся волны переворот идёт с частотой 2m, то есть равной щели между ветвями E = +-m, как у двух связанных маятников; чем тяжелее фермион, тем быстрее переворот и тем медленнее он движется. Массивной частице нужны обе киральности; массовый член запрещён SU(2)-симметрией, а юкавский член с хиггсовским дублетом даёт m = f v / sqrt(2).
- EN: Without mass the Dirac equation in 1+1 dimensions splits into two independent waves: psi_R runs to the right and psi_L to the left, both at the speed of light, and chirality coincides with the direction of motion. The Higgs condensate v couples them, m = f v / sqrt(2): psi_R becomes a source of psi_L and back, the chirality <gamma_5> oscillates with the frequency 2E = 2 sqrt(p^2 + m^2), and the packet slows down to v = p/E < c. For a wave at rest the flip frequency is 2m, the gap between the branches E = +-m, as for two coupled pendulums; the heavier the fermion, the faster it flips and the slower it moves. A massive particle needs both chiralities; the mass term is forbidden by SU(2) symmetry, and the Yukawa term with the Higgs doublet gives m = f v / sqrt(2).

## poster_time
26.8 s (film time: the oscillating packet with both lanes lit, the springs, the dial at v = 4.24, the splitting into E > 0 and E < 0 parts).

## Credit note
- RU: Модель: уравнение Дирака в (1+1) измерениях в киральном базисе, i(d_t + d_x)psi_R = m psi_L, i(d_t - d_x)psi_L = m psi_R, решённое точно в импульсном представлении (дискретное преобразование Фурье, унитарная эволюция); m = f v / sqrt(2). Включение конденсата, пружинки (символ связи) и потенциал Хиггса схематичны; единицы hbar = c = 1 условные. Анимация создана автором книги.
- EN: Model: the Dirac equation in 1+1 dimensions in the chiral basis, i(d_t + d_x)psi_R = m psi_L, i(d_t - d_x)psi_L = m psi_R, solved exactly in momentum space (discrete Fourier transform, unitary evolution); m = f v / sqrt(2). The switching on of the condensate, the springs (a symbol of the coupling) and the Higgs potential are schematic; the units hbar = c = 1 are arbitrary. Animation made for the book.


## Where the QR block belongs
- **File:** `chapters/SymmetryBreaking.tex` (RU) and `en/chapters/SymmetryBreaking.tex` (EN). Book fields for the catalog: volume 2, chapter 24 (the file has `\begin{mychapter}{SymmetryBreaking}{23}`, and GaugePrinciple = `{22}` is chapter 23 in the catalog), chapter title RU "Откуда у частицы масса?", EN "Where Does a Particle Get Its Mass?".
- **Place:** section "Масса фермионов, или связанные маятники" (EN: the fermion mass section with coupled pendulums), subsection "Связанные киральные волны рождают массу" (EN "Coupled Chiral Waves Give Birth to Mass"): a `\QRwithPreview` figure right before the subsection "Природа массы фермиона. Простыми словами" (RU line ~684, EN line ~652), i.e. after the solution `psi(t,x) = a e^{-iEt+ipx} u(p) + b e^{iEt-ipx} u(-p)` and the sentence "Так у дираковского фермиона возникает масса".
- **Reason:** the film is a direct illustration of exactly this subsection: the same equations (`i(d_t +- d_x)psi = m psi`), the same matrix `H(p)`, the same coupled-pendulum analogy and the same statement that the mass is the coupling of `psi_L` and `psi_R` through the condensate; the last part of the film links to the Yukawa section of `StandardModel.tex` (`m = f v/sqrt 2`), which would be the second candidate place (after eq. `sm_22`).
- The film notation: `m = f v / sqrt(2)` is that of `StandardModel.tex`; the toy model of `SymmetryBreaking.tex` has `m = g v`, the difference is only the normalisation of `v` (the film shows the Standard-Model form).
