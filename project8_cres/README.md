# Project 8 CRES

Manim scene for the Project 8 section of the direct neutrino-mass lecture.
It connects three parts of one CRES event:

- a trapped electron moving in a magnetic field;
- microwave cyclotron radiation received by an antenna array;
- a rising frequency track, with a jump after scattering.

The animation is schematic. It emphasizes the measured relation

$$
K\downarrow\quad\Longrightarrow\quad\gamma\downarrow
\quad\Longrightarrow\quad f_c\uparrow.
$$

Render from this directory:

```bash
manim --disable_caching project8_cres.py Project8CRES -o project8_cres
```

Scene dimensions, timing, and colors are configured in `run.cfg`.
