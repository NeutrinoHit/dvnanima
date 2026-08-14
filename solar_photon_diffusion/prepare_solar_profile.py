from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path
from urllib.request import urlopen


MODEL_URL = "https://www.ap.smu.ca/~guenther/evolution/ssm_d3_7d_Grey.txt"
FIELDS = (
    "radius_fraction",
    "radius_cm",
    "temperature_k",
    "density_g_cm3",
    "hydrogen_fraction",
    "metal_fraction",
    "opacity_cm2_g",
    "luminosity_erg_s",
)
SHELL_LINE = re.compile(r"^\s*\d+\s+[+-]?\d\.\d+E[+-]\d+")


def parse_guenther_model(text: str) -> list[dict[str, float]]:
    """Extract the eight columns used by the diffusion calculation."""

    lines = text.splitlines()
    rows: list[dict[str, float]] = []
    for index, line in enumerate(lines):
        if not SHELL_LINE.match(line) or index + 2 >= len(lines):
            continue
        first = line.split()
        second = lines[index + 1].split()
        third = lines[index + 2].split()
        if len(first) != 8 or len(second) != 7 or len(third) != 7:
            continue

        radius_fraction = float(first[1])
        if radius_fraction > 1.0:
            continue
        rows.append(
            {
                "radius_fraction": radius_fraction,
                "radius_cm": float(first[3]),
                "temperature_k": float(first[6]),
                "density_g_cm3": float(first[7]),
                "hydrogen_fraction": float(third[0]),
                "metal_fraction": float(third[1]),
                "opacity_cm2_g": float(third[2]),
                "luminosity_erg_s": float(first[5]),
            }
        )
    if len(rows) < 1000:
        raise ValueError(f"Parsed only {len(rows)} solar shells; input format may have changed.")
    return rows


def write_csv(rows: list[dict[str, float]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: f"{row[key]:.10e}" for key in FIELDS})


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Prepare a compact radiative-diffusion profile from Guenther's SSM."
    )
    parser.add_argument(
        "--source",
        type=Path,
        help="Local ssm_d3_7d_Grey.txt. If omitted, download the cited source.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).with_name("data")
        / "solar_model_2010_grey_profile.csv",
    )
    args = parser.parse_args()

    if args.source is None:
        with urlopen(MODEL_URL, timeout=30) as response:
            text = response.read().decode("utf-8")
    else:
        text = args.source.read_text(encoding="utf-8")
    rows = parse_guenther_model(text)
    write_csv(rows, args.output)
    print(f"Wrote {len(rows)} shells to {args.output}")


if __name__ == "__main__":
    main()
