from __future__ import annotations

import argparse
import time
from pathlib import Path

from fields.scalar_qed.data_io import build_dataset, save_dataset
from fields.scalar_qed.numerics import load_scalar_qed_config, make_preview_config


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Compute scalar QED observables once and store them for independent renderers.")
    parser.add_argument("--config", type=Path, default=Path(__file__).with_name("scalar_qed.toml"))
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--preview", action="store_true", help="Export a lightweight preview dataset.")
    parser.add_argument("--grid-scale", type=float, default=0.45)
    parser.add_argument("--time-scale", type=float, default=0.35)
    parser.add_argument("--playback-scale", type=float, default=0.35)
    parser.add_argument("--compressed", action="store_true", help="Write a compressed npz. Slower to save, smaller on disk.")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    cfg = load_scalar_qed_config(args.config)
    if args.preview:
        cfg = make_preview_config(
            cfg,
            grid_scale=args.grid_scale,
            time_scale=args.time_scale,
            playback_scale=args.playback_scale,
        )

    t0 = time.time()
    dataset = build_dataset(cfg)
    output_path = save_dataset(args.out, dataset, compressed=args.compressed)
    elapsed = time.time() - t0
    size_mb = output_path.stat().st_size / (1024.0 * 1024.0)
    print(f"Saved scalar QED dataset to {output_path}")
    print(f"elapsed_sec = {elapsed:.3f}")
    print(f"file_size_mb = {size_mb:.2f}")
    print(
        "frames = "
        f"{dataset['times'].shape[0]}, simulation_steps = {dataset['simulation_times'].shape[0]}, "
        f"grid = {dataset['x_axis'].shape[0]} x {dataset['y_axis'].shape[0]}"
    )


if __name__ == "__main__":
    main()
