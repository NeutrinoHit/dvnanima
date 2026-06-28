from __future__ import annotations

from pathlib import Path

from pmt.config.loader import slugify_name


def default_png_filename(scene_name: str) -> str:
    return f"{slugify_name(scene_name)}.png"


def default_hdf5_filename(scene_name: str) -> str:
    return f"{slugify_name(scene_name)}.h5"


def resolve_single_output_paths(
    scene_name: str,
    output: str | Path | None,
    output_dir: str | Path | None,
    hdf5_output: str | Path | None,
    hdf5_dir: str | Path | None,
    save_hdf5: bool,
) -> tuple[Path, Path | None]:
    if output is not None:
        png_path = Path(output)
    else:
        base_dir = Path(output_dir) if output_dir is not None else Path("out")
        png_path = base_dir / default_png_filename(scene_name)

    h5_path: Path | None = None
    if hdf5_output is not None:
        h5_path = Path(hdf5_output)
    elif save_hdf5:
        if hdf5_dir is not None:
            h5_path = Path(hdf5_dir) / default_hdf5_filename(scene_name)
        else:
            h5_path = png_path.with_suffix(".h5")

    return png_path, h5_path


def resolve_batch_output_paths(
    scene_name: str,
    output_dir: str | Path,
    hdf5_dir: str | Path | None,
    save_hdf5: bool,
) -> tuple[Path, Path | None]:
    output_root = Path(output_dir)
    png_path = output_root / default_png_filename(scene_name)

    h5_path: Path | None = None
    if save_hdf5:
        if hdf5_dir is not None:
            h5_path = Path(hdf5_dir) / default_hdf5_filename(scene_name)
        else:
            h5_path = output_root / default_hdf5_filename(scene_name)

    return png_path, h5_path
