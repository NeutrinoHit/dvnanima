# Card of the film 0029 "A cosmic-ray air shower" (new version)

**This film REPLACES the existing book animation 0029** (`dvnanima/airshower`, Manim, catalog id `0029`, slug `air-shower`,
qr_file `AirShower.pdf`, status "proposed"). Nothing of the old project is used. The old poster_time 5.0 and the old site_asset
(`neutrino-particles-air-shower-air-shower.mp4`) must be replaced; the film now exists in two language versions
(`media/air_shower_cascade_ru.mp4`, `media/air_shower_cascade_en.mp4`), so `site_asset` becomes a ru/en dictionary like the other films.

## Where the QR block is in the book now 

I searched `chapters/*.tex` and `en/chapters/*.tex` for `AirShower`, `Air_Shower`, `airshower`, `ливн*`, `shower`: there is **no QR block of the air
shower in the book sources at present** (no `\QRwithPreview` with `AirShower*.pdf`, and `images/QRcodes/` has no `AirShower*.pdf`;
the only hit is the word "shower" in the Gargamelle caption and in the Bhabha biography). The catalog item is only "proposed".
So there is nothing to swap; the block has to be added. Proposed place:

* volume 1, chapter 2 (`\begin{mychapter}{WeakHistory}{1}{Век открытий}`, catalog: volume 1, chapter 2, "Век открытий" / "A Century of Discoveries");
* RU: `chapters/WeakHistory.tex`, directly after the paragraph "1947. Открытие заряженных пионов и первое наблюдение распада
  π+→μ+ν_μ" (it ends with the Nobel prize sentence of Powell, **line 633**), before `\section{Частиц всё больше}` (**line 635**);
* EN: `en/chapters/WeakHistory.tex`, after the same paragraph (ends with "Cecil Powell ... received the Nobel Prize in Physics in 1950", **line 599**),
  before `\section{More and More Particles}` (**line 601**).

Reason: the pions and muons of the preceding paragraphs (1936–1947) were all found in cosmic rays, and the film shows exactly
how a cosmic proton makes pions, how pi0 -> gamma gamma feeds the electromagnetic cascade (the Bhabha–Heitler cascade theory is
mentioned in the biography in the QED chapter), and how pi± -> μ± ν_μ brings muons and neutrinos to the ground. An alternative
is the end of the positron paragraph (RU line 398 / "Хотя Андерсон впервые обнаружил частицы в космических лучах...", EN the same place).

Book fields for the catalog: `"book": {"volume": 1, "chapter": 2, "chapter_title": {"ru": "Век открытий", "en": "A Century of Discoveries"}}`.

## Titles
* RU: Широкий атмосферный ливень
* EN: A cosmic-ray air shower

## Captions (plain-text formulas, like in the catalog)

* RU: Протон космических лучей с энергией E0 = 10^15 эВ входит в атмосферу. Сначала правило Гайтлера для одного фотона: после глубины
  d = X0 ln 2 частица делится на две с половинной энергией, N = 2^n, и размножение останавливается при E = Ec (Nmax = E0/Ec на глубине
  Xmax = X0 ln(E0/Ec)). Затем метод Монте-Карло для протона: пионы, π0 → γγ запускают электромагнитные каскады, заряженные пионы
  с E ≤ 20 ГэВ распадаются на мюоны и нейтрино; на графике растёт профиль N(X), отмечены оценки Гайтлера. В конце — матрица детекторов
  на земле, плотность e± и μ в зависимости от расстояния до оси и свет ливня (флуоресценция и черенковское излучение). Модель игрушечная,
  траектории — случайная выборка, горизонтальный масштаб растянут.
* EN: A cosmic-ray proton with E0 = 10^15 eV enters the atmosphere. First Heitler's rule for a single photon: after the depth d = X0 ln 2 a
  particle splits in two with half the energy, N = 2^n, and the multiplication stops at E = Ec (Nmax = E0/Ec at Xmax = X0 ln(E0/Ec)). Then
  a Monte Carlo of the proton shower: pions, pi0 -> gamma gamma start electromagnetic cascades, charged pions with E <= 20 GeV decay into muons
  and neutrinos; the profile N(X) builds up with the Heitler estimates marked. Finally a detector array on the ground, the density of e± and mu
  against the distance from the axis, and the light of the shower (fluorescence and Cherenkov radiation). A toy model; the tracks are a
  random sample and the horizontal scale is exaggerated.

## poster_time
62.0 (film time, s: the whole cascade is drawn, the profile and the Heitler markers are visible, the summary appears). Check it on the final file;
a second good choice is 40 (the cascade near its maximum).

## Credit note
* RU: Игрушечная модель Гайтлера (электромагнитный каскад, 1944) и Мэтьюза (адронная часть, 2005), метод Монте-Карло с фиксированным зерном,
  разрежение (thinning) по Хиллас; формула НКГ для поперечного распределения; уравнения: N_max = E0/Ec, X_max = X0 ln(E0/Ec), N_μ = (E0/E_dec)^β.
* EN: Toy model of Heitler (electromagnetic cascade, 1944) and Matthews (hadronic part, 2005), Monte Carlo with a fixed seed, Hillas thinning;
  NKG formula for the lateral distribution; equations: N_max = E0/Ec, X_max = X0 ln(E0/Ec), N_mu = (E0/E_dec)^beta.
  code_url: https://github.com/NeutrinoHit/dvnanima/tree/main/air_shower_cascade (the old `airshower` folder is superseded).

## Notes for the integration
* Length about 100 s, 1280x720, 30 fps, two separate files (no mixed-language captions).
* `../test_dvconfig.py` does not list the film yet (`FILMS`); add `air_shower_cascade` there if you want the shared rules checked in that file too.
* New files: `config.toml`, `texts.toml`, `air_shower_cascade.py`, `shower_model.py`, `test_air_shower_cascade.py`, `render.sh`, `README.md`, this card.
