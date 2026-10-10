# Shorts: content plan (draft, to be revisited after the main films are finished)

Decisions of the author (2026-10-08)
- The films for the QR codes stay as they are (about 2 minutes, not cut, one QR code per figure).
- Shorts are separate pieces built from the same material: a hook, an own dramaturgy, up to 60 s, vertical 1080x1920, RU and EN separately.
- Shorts are made one at a time, creatively, not in a batch. This file is only the plan.
- Purpose: promotion of the book. Covers: `TheBook/images/FrontPage` (RU), `TheBook/en/images/FrontPage` (EN: `front_cover_1.jpg`, `front_cover_2.jpg`, back covers).

## Template of a short (proposal)
| time | what | note |
|---|---|---|
| 0-3 s | hook: a question or a paradox, already in motion, one line of large text | no title card, no logo |
| 3-48 s | one idea, one image; large subtitles (many people watch without sound); one formula at most, large | not a cut of the long film: re-framed for 9:16, own pacing |
| 48-52 s | the answer / the surprising consequence in one line | the payoff of the hook |
| 52-60 s | end card: cover of the book, "Volume N, chapter M", the title of the chapter, short address / QR to `qr/<id>/?src=short` | the cover fades in softly and stays 3 s |

Permanent small mark at the bottom of the frame during the whole short: "Book title, vol. N, ch. M" (survives re-posts and works when the viewer leaves before the end).

Technical idea (to be built when the first short is made, not before): `config.short.toml` over `config.toml` (`--profile short`), `texts.short.toml` (hook, subtitles, end card), shared `dvshort.py` for the end card and the mark. The long films and their `config.toml` are not touched.

Open points for the first short: the exact book title for the card; the address/QR policy (the platforms limit links in Shorts descriptions, so the address is in the frame); the series name ("QFT in a minute No. N").

## Candidates (hook is a draft, the author edits)
Chapter numbers are those of `book-animations.catalog.json`.

| id | film | vol./ch. | draft hook (RU / EN idea) | the one idea of the short |
|---|---|---|---|---|
| 0033 | chain of oscillators | 1 / 12 | "Как из пружинок получается поле?" / how do springs become a field | N masses, N modes; N to infinity; the field as the limit |
| 0033 | (second short) | 1 / 12 | "Почему волны проходят друг сквозь друга?" | linear: no scattering; add a cubic term and energy moves between modes = interaction |
| 0026 | 2D lattice | 1 / 12 | "Поверхность, у которой каждая точка думает о соседях" | a ripple on the lattice, modes of a membrane, the continuum |
| 0034 | 3D lattice | 1 / 12 | "Вакуум как кристалл, которого нет" | normal modes in 3D, spectrum, dispersion becomes linear in the limit |
| 0024 | Standard Model fields | 1 / 1 | "Из чего сделана вся материя? Из матраса." | each particle is a wave of its own field; two photons make a pair |
| 0004 | Thomson's tube | 1 / 2 | "Как взвесили электрон, не видя его?" | the beam, E and B balance, e/m |
| 0014 | Rutherford | 1 / 2 | "Почему одна из 8000 альфа-частиц летит назад?" | the nucleus is tiny |
| 0030 | photomultiplier | 1 / 2 | "Как один фотон превращается в ток?" | electron multiplication |
| 0031 | Cherenkov cone | 1 / 13 | "Звуковой удар для света" | wavelets, the cone, the angle |
| 0006 | Penrose-Terrell | 1 / 3 | "Движущийся шар не сплющивается, а поворачивается" | the shape of fast bodies in a photo |
| 0015 | simultaneity | 1 / 3 | "Одновременно для кого?" | two observers disagree |
| 0009 | spinor and Mobius strip | 1 / 10 | "Повернул на 360 градусов, и ничего не вернулось" | the spinor needs 720 degrees |
| 0025 | free wave packet | 1 / 5 | "Почему электрон расплывается?" | spreading of a packet |
| 0007 | OAM beams | 1 / 5 | "Свет, который закручен как штопор" | orbital angular momentum |
| 0008 | circular polarization | 1 / 13 | "Спин фотона, который можно увидеть" | helicity |
| 0032 | Hulse-Taylor | 1 / 4 | "Две звезды, которые теряют энергию на волны" | the orbit decays as predicted |
| 0011 | running charge | 1 / 17 | "Заряд электрона зависит от того, как близко смотришь" | screening by virtual pairs |
| 0036 | kick to an electron | 2 / 23 | "Что даёт волне право ударить электрон?" | the front of an EM wave, gauge |
| 0038 | path integral | 2 / 22 | "Частица пробует все пути сразу" | sum over paths |
| 0035 | e+e- to mu+mu- | 1 / 18 | "Как увидеть то, что живёт 10^-23 с" | a scattering experiment |
| 0037 | pi0 to 2(e+e-) | 2 / 40 | "Распад, в котором спин видно в плоскостях пар" | E, B and planes |
| 0022/23 | attraction / repulsion | 1 / 19 | "Одинаковые заряды отталкиваются из-за интерференции" | interference of wave packets |
| 0029 | air shower | 1 / 2 | "Один космический протон, миллиарды частиц" | cascade |

Planned films not yet made (they will get their short together with the main film): Meissner effect and photon mass; chiral waves and fermion mass; Z-decay asymmetry; Higgs to f f-bar (spin/orbital structure).

## Order proposal
First pilot: 0033 (hook "springs to field"), because the material is fresh and already in the config format. After the pilot the author decides on the template, then one short per decision.
