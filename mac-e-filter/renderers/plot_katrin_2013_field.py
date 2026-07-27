#!/usr/bin/env python3
"""Plot and report the published 2013 KATRIN nominal field reconstruction."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from mac_e_filter.katrin_nominal import (  # noqa: E402
    SUPERCONDUCTING_SOURCES_2013,
    build_katrin_2013_field,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out",
        type=Path,
        default=PROJECT_ROOT / "media" / "katrin_2013_nominal_field.png",
        help="Output image path.",
    )
    parser.add_argument(
        "--z-min",
        type=float,
        default=-7.0,
        help="Left edge of the axial profile in metres.",
    )
    parser.add_argument(
        "--z-max",
        type=float,
        default=7.0,
        help="Right edge of the axial profile in metres.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    import matplotlib.pyplot as plt

    z = np.linspace(args.z_min, args.z_max, 1201)
    models = {
        configuration: build_katrin_2013_field(configuration)
        for configuration in ("one_minimum", "two_minima")
    }

    figure, (axis_total, axis_components) = plt.subplots(
        2,
        1,
        figsize=(10.5, 8.0),
        sharex=True,
        constrained_layout=True,
    )

    labels = {
        "one_minimum": "one global minimum",
        "two_minima": "two local minima",
    }
    colors = {"one_minimum": "#1768AC", "two_minima": "#D1495B"}
    for configuration, model in models.items():
        field_microtesla = 1e6 * np.abs(model.total.axis_field(z))
        axis_total.plot(
            z,
            field_microtesla,
            color=colors[configuration],
            linewidth=2.0,
            label=labels[configuration],
        )
        center_microtesla = 1e6 * abs(model.center_breakdown_t()["total"])
        axis_total.scatter(
            [0.0],
            [center_microtesla],
            color=colors[configuration],
            s=32,
            zorder=3,
        )

    axis_total.axhline(
        350.0,
        color="0.35",
        linestyle=":",
        linewidth=1.2,
        label="published centre value: 350 µT",
    )
    axis_total.set_ylabel(r"$|\mathrm{B}_z(0,z)|$ [µT]")
    axis_total.set_title("KATRIN main spectrometer: published 2013 nominal fields")
    axis_total.grid(alpha=0.22)
    axis_total.legend(ncols=2)

    model = models["one_minimum"]
    components = (
        ("LFCS winding packs", model.lfcs, "#1768AC"),
        (
            "superconducting scalar surrogate",
            model.superconducting_surrogate,
            "#6A4C93",
        ),
        ("axial Earth field", model.earth, "#2A9D8F"),
        ("total", model.total, "#1D1D1D"),
    )
    for label, component, color in components:
        axis_components.plot(
            z,
            1e6 * component.axis_field(z),
            color=color,
            linewidth=2.2 if label == "total" else 1.5,
            label=label,
        )
    axis_components.axvline(0.0, color="0.5", linestyle=":", linewidth=1.0)
    axis_components.set_xlabel("spectrometer coordinate z [m]")
    axis_components.set_ylabel(r"signed $\mathrm{B}_z(0,z)$ [µT]")
    axis_components.grid(alpha=0.22)
    axis_components.legend(ncols=2)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.out, dpi=180)
    plt.close(figure)

    print(f"Saved diagnostic plot: {args.out}")
    print("Centre-field decomposition:")
    for configuration, field_model in models.items():
        values = field_model.center_breakdown_t()
        formatted = ", ".join(
            f"{name}={value * 1e6:+.6f} µT" for name, value in values.items()
        )
        print(f"  {configuration}: {formatted}")

    print("Published-scalar equivalent superconducting loops:")
    for source in SUPERCONDUCTING_SOURCES_2013:
        loop = source.equivalent_loop()
        print(
            f"  {source.name}: radius={loop.radius_m:.6f} m, "
            f"NI={loop.ampere_turns:+.6e} A-turn"
        )


if __name__ == "__main__":
    main()

