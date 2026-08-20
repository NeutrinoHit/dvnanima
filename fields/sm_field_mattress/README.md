# Standard Model Field Mattress

Manim animation for the particle physics mini-course.

The scene shows a vertically separated stack of Standard Model quantum fields
filling space. All fields constantly fluctuate, different travelling wave
patterns run through the layers, and the final part opens a cutaway so the
internal field layers remain visible while waves pass through them.

The separated layers are a visual metaphor for distinct quantum fields, not a literal spatial stratification of the Standard Model vacuum.

Preview render:

```bash
cd fields/sm_field_mattress
SM_FIELD_MATTRESS_CONFIG=run_preview.cfg manim -ql sm_field_mattress.py StandardModelFieldMattress
```

Lecture render:

```bash
cd fields/sm_field_mattress
manim -qh sm_field_mattress.py StandardModelFieldMattress
```

Main tuning lives in `run.cfg`. Use `run_preview.cfg` for a faster low-quality
check before rendering the lecture version.
