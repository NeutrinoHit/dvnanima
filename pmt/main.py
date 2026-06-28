from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Iterable

if __package__ is None or __package__ == "":
    project_root = Path(__file__).resolve().parent.parent
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

from pmt.config.loader import (
    DEFAULT_BASE_CONFIG,
    discover_scene_files,
    infer_scene_name,
    merge_base_and_scene,
    merged_config_to_pretty_json,
    merged_config_to_runtime,
)
from pmt.io.hdf5_io import save_scene_hdf5
from pmt.io.paths import resolve_batch_output_paths, resolve_single_output_paths
from pmt.physics.simulate import run_scene
from pmt.render.static_scene import render_scene_png


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="PMT static scene generator: config -> field+trajectories -> PNG (+ optional HDF5)."
    )
    parser.add_argument("--base", type=Path, default=DEFAULT_BASE_CONFIG, help="Base TOML config")

    source_group = parser.add_mutually_exclusive_group(required=True)
    source_group.add_argument("--scene", type=Path, help="Single scene TOML config")
    source_group.add_argument("--scene-dir", type=Path, help="Directory with scene TOML files")

    parser.add_argument("--output", type=Path, help="Output PNG path for single-scene mode")
    parser.add_argument("--output-dir", type=Path, help="Output PNG directory for batch mode")

    parser.add_argument("--hdf5", type=Path, help="Optional HDF5 output path for single-scene mode")
    parser.add_argument("--hdf5-dir", type=Path, help="Optional HDF5 output directory for batch mode")
    parser.add_argument("--save-hdf5", action="store_true", help="Save HDF5 alongside PNG when path is not explicitly set")

    parser.add_argument("--dry-run", action="store_true", help="Print merged config and exit without calculations")
    return parser


def _validate_args(args: argparse.Namespace, parser: argparse.ArgumentParser) -> None:
    if args.scene is not None and args.scene_dir is not None:
        parser.error("Use either --scene or --scene-dir, not both")

    if args.scene_dir is not None and args.output is not None:
        parser.error("--output is only valid with --scene")

    if args.scene_dir is not None and args.hdf5 is not None:
        parser.error("--hdf5 is only valid with --scene")


def _iter_scene_files(args: argparse.Namespace) -> Iterable[Path]:
    if args.scene is not None:
        yield args.scene
        return

    scene_files = discover_scene_files(args.scene_dir)
    for path in scene_files:
        yield path


def _run_single_scene(
    *,
    base_path: Path,
    scene_path: Path,
    output_png: Path,
    output_h5: Path | None,
    dry_run: bool,
) -> None:
    merged = merge_base_and_scene(base_path, scene_path)
    scene_name = infer_scene_name(merged, scene_path)

    if dry_run:
        print(f"--- DRY RUN: {scene_name} ({scene_path}) ---")
        print(merged_config_to_pretty_json(merged))
        return

    cfg = merged_config_to_runtime(merged, scene_path=scene_path)
    result = run_scene(cfg)

    png_path = render_scene_png(result, output_png)
    print(f"[{cfg.scene.name}] PNG: {png_path}")

    if output_h5 is not None:
        h5_path = save_scene_hdf5(output_h5, result)
        print(f"[{cfg.scene.name}] HDF5: {h5_path}")

    stats = result.stats
    print(
        f"[{cfg.scene.name}] emitted={stats.electron_count} collected={stats.collected_count} "
        f"eff={100.0 * stats.collection_efficiency:.2f}% misses={stats.miss_count} "
        f"time={stats.compute_seconds:.3f}s"
    )


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()
    _validate_args(args, parser)

    scene_files = list(_iter_scene_files(args))
    if not scene_files:
        parser.error("No scene files found")

    base_path = args.base

    if args.scene is not None:
        merged = merge_base_and_scene(base_path, args.scene)
        scene_name = infer_scene_name(merged, args.scene)
        output_png, output_h5 = resolve_single_output_paths(
            scene_name=scene_name,
            output=args.output,
            output_dir=args.output_dir,
            hdf5_output=args.hdf5,
            hdf5_dir=args.hdf5_dir,
            save_hdf5=args.save_hdf5,
        )

        _run_single_scene(
            base_path=base_path,
            scene_path=args.scene,
            output_png=output_png,
            output_h5=output_h5,
            dry_run=args.dry_run,
        )
        return

    output_dir = args.output_dir or Path("out")
    for scene_path in scene_files:
        merged = merge_base_and_scene(base_path, scene_path)
        scene_name = infer_scene_name(merged, scene_path)
        output_png, output_h5 = resolve_batch_output_paths(
            scene_name=scene_name,
            output_dir=output_dir,
            hdf5_dir=args.hdf5_dir,
            save_hdf5=(args.save_hdf5 or args.hdf5_dir is not None),
        )
        _run_single_scene(
            base_path=base_path,
            scene_path=scene_path,
            output_png=output_png,
            output_h5=output_h5,
            dry_run=args.dry_run,
        )


if __name__ == "__main__":
    main()
