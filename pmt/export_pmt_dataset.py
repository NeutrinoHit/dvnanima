from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parents[1]))

from pmt.data_io import build_dataset, save_dataset
from pmt.numerics import load_pmt_config, make_preview_config


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Export PMT electrostatic simulation + trajectories to HDF5.")
    parser.add_argument("--config", type=Path, default=Path(__file__).with_name("pmt.toml"))
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--preview", action="store_true", help="Use scaled-down grid/particles/steps for quick checks.")
    parser.add_argument("--grid-scale", type=float, default=0.5)
    parser.add_argument("--particle-scale", type=float, default=0.35)
    parser.add_argument("--step-scale", type=float, default=0.35)
    parser.add_argument("--storage-dtype", choices=("float32", "float64"), default="float32")
    parser.add_argument("--compression", choices=("gzip", "none"), default="gzip")
    parser.add_argument("--compression-level", type=int, default=4)
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()

    cfg = load_pmt_config(args.config)
    if args.preview:
        cfg = make_preview_config(
            cfg,
            grid_scale=args.grid_scale,
            particle_scale=args.particle_scale,
            step_scale=args.step_scale,
        )

    dtype = "float64" if args.storage_dtype == "float64" else "float32"
    compression = None if args.compression == "none" else args.compression

    t0 = time.time()
    dataset = build_dataset(cfg, storage_dtype=dtype)
    output_path = save_dataset(
        args.out,
        dataset,
        compression=compression,
        compression_level=args.compression_level,
    )
    elapsed = time.time() - t0

    emitted = int(dataset["launch_positions"].shape[0])
    collected = int(dataset["particle_impact_mask"].sum())
    efficiency = float(dataset["collection_efficiency"]) if dataset["collection_efficiency"].shape == () else float(dataset["collection_efficiency"].item())

    print(f"Saved PMT dataset to {output_path}")
    print(f"elapsed_sec = {elapsed:.3f}")
    print(f"file_size_mb = {output_path.stat().st_size / (1024.0 * 1024.0):.2f}")
    print(
        "grid = "
        f"{dataset['x_axis'].shape[0]} x {dataset['y_axis'].shape[0]}, "
        f"steps = {dataset['particle_positions'].shape[0]}, "
        f"particles = {dataset['particle_positions'].shape[1]}"
    )
    print(f"solver_iterations = {int(dataset['solver_iterations'])}")
    print(f"solver_residual = {float(dataset['solver_residual']):.3e}")
    print(f"collection = {collected}/{emitted} ({100.0 * efficiency:.2f}%)")


if __name__ == "__main__":
    main()
