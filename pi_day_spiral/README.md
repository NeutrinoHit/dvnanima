# Pi Day Gift Box Shorts

Vertical short for Pi Day.

Sequence:

- a festive gift box appears at the top
- a visible contour of `pi` is prepared on screen
- digits of `pi` fly out of the box
- each digit takes its place on the contour
- the finished symbol pulses once and holds

Digit source:

- the digits are stored locally in `pi_day_spiral.py` as a fixed string
- this is deterministic and enough for the short
- if needed, this can later be replaced by arbitrary-precision generation

Render:

```bash
cd "/Users/dmitrijnaumov/Library/Mobile Documents/com~apple~CloudDocs/Projects/dvnanima/pi_day_spiral"
manim -pqh pi_day_spiral.py PiDaySpiralShorts
```

Main tuning in `run.cfg`:

- `n_digits`
- `digit_font_size`
- `pi_symbol_scale`, `pi_shift_y`
- `contour_start`, `contour_stroke_width`, `show_outline`
- `box_width`, `box_height`, `box_center_y`
- `stream_time`

The default timing sums to 30 seconds.
