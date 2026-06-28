from __future__ import annotations

import argparse
import math
import sys
import time
from pathlib import Path

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parents[2]))

from fields.radiating_charge.data_io import build_dataset, save_dataset
from fields.radiating_charge.numerics import load_radiating_charge_config, make_preview_config


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Compute radiating-charge observables once and store them for renderers.")
    parser.add_argument("--config", type=Path, default=Path(__file__).with_name("radiating_charge.toml"))
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--preview", action="store_true", help="Export a lightweight preview dataset.")
    parser.add_argument("--grid-scale", type=float, default=0.45)
    parser.add_argument("--time-scale", type=float, default=0.35)
    parser.add_argument("--playback-scale", type=float, default=0.35)
    parser.add_argument("--compressed", action="store_true", help="Write a compressed npz archive.")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    cfg = load_radiating_charge_config(args.config)
    if args.preview:
        cfg = make_preview_config(
            cfg,
            grid_scale=args.grid_scale,
            time_scale=args.time_scale,
            playback_scale=args.playback_scale,
        )

    t0 = time.time()
    progress_started = time.time()
    last_progress_print = 0.0

    def on_progress(done: int, total: int, t_value: float, residual: float) -> None:
        nonlocal last_progress_print
        now = time.time()
        is_last = done >= total
        if (now - last_progress_print) < 0.15 and not is_last:
            return
        fraction = done / max(total, 1)
        elapsed = now - progress_started
        eta = (elapsed / fraction - elapsed) if fraction > 1e-9 else float("inf")
        bar_width = 30
        filled = int(round(bar_width * fraction))
        bar = "#" * filled + "-" * (bar_width - filled)
        eta_text = f"{eta:6.1f}s" if math.isfinite(eta) else "   infs"
        end = "\n" if is_last else ""
        print(
            f"\r[{bar}] {done:4d}/{total:<4d} {100.0*fraction:6.2f}%  "
            f"t={t_value:8.3f}  eta={eta_text}  res={residual:.2e}",
            end=end,
            flush=True,
        )
        last_progress_print = now

    dataset = build_dataset(cfg, progress_callback=on_progress)
    output_path = save_dataset(args.out, dataset, compressed=args.compressed)
    elapsed = time.time() - t0
    size_mb = output_path.stat().st_size / (1024.0 * 1024.0)

    frame_keys = dataset["metadata"]["frame_keys"]
    print(f"Saved radiating-charge dataset to {output_path}")
    print(f"elapsed_sec = {elapsed:.3f}")
    print(f"file_size_mb = {size_mb:.2f}")
    print(f"frames = {dataset['times'].shape[0]}, grid = {dataset['x_axis'].shape[0]} x {dataset['y_axis'].shape[0]}")
    print(f"observables = {frame_keys}")
    print(f"max_retarded_residual = {float(dataset['retarded_residual_max'].max()):.3e}")


if __name__ == "__main__":
    main()
