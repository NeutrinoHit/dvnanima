#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import numpy as np


def get_numpy_data(filenames):
    """Load one or many npz files with keys: hist, xedges, yedges.

    Keeps your original function contract for 1 file
    and extends it to multiple files.
    """
    loaded: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []
    for filename in filenames:
        print(filename)
        data = np.load(filename)
        loaded.append((np.asarray(data["hist"], dtype=float), data["xedges"], data["yedges"]))

    if len(loaded) == 1:
        return loaded[0]
    return loaded


def _gaussian_kernel_1d(sigma_bins: float) -> np.ndarray:
    if sigma_bins <= 0:
        return np.array([1.0], dtype=float)
    radius = max(1, int(np.ceil(4.0 * sigma_bins)))
    x = np.arange(-radius, radius + 1, dtype=float)
    kernel = np.exp(-0.5 * (x / sigma_bins) ** 2)
    kernel /= np.sum(kernel)
    return kernel


def _convolve_axis_constant(values: np.ndarray, kernel: np.ndarray, axis: int) -> np.ndarray:
    pad = kernel.size // 2
    pads = [(0, 0)] * values.ndim
    pads[axis] = (pad, pad)
    padded = np.pad(values, pads, mode="constant", constant_values=0.0)
    return np.apply_along_axis(lambda m: np.convolve(m, kernel, mode="valid"), axis=axis, arr=padded)


def gaussian_smooth(values: np.ndarray, sigma_bins: float) -> np.ndarray:
    if sigma_bins <= 0:
        return np.asarray(values, dtype=float)

    try:
        from scipy.ndimage import gaussian_filter as _gaussian_filter  # type: ignore

        return _gaussian_filter(values, sigma=sigma_bins, mode="constant")
    except Exception:
        kernel = _gaussian_kernel_1d(sigma_bins)
        out = _convolve_axis_constant(np.asarray(values, dtype=float), kernel, axis=0)
        out = _convolve_axis_constant(out, kernel, axis=1)
        return out


def build_roi_mask(hist: np.ndarray, min_count: float = 1.0) -> np.ndarray:
    mask = np.isfinite(hist)
    mask &= hist >= float(min_count)
    return mask


def compute_weights(
    hist: np.ndarray,
    sigma_bins: float,
    eps_frac: float,
    min_count_for_roi: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, float]]:
    hist = np.asarray(hist, dtype=float)
    roi_mask = build_roi_mask(hist, min_count=min_count_for_roi)

    hs = gaussian_smooth(hist, sigma_bins=sigma_bins)

    peak = float(np.max(hs[roi_mask])) if np.any(roi_mask) else float(np.max(hs))
    eps = max(1e-12, eps_frac * peak)

    density_reg = np.maximum(hs, eps)
    weight_map = np.zeros_like(hist, dtype=float)
    weight_map[roi_mask] = 1.0 / density_reg[roi_mask]

    # Normalization: average event weight = 1 within ROI.
    sum_events = float(np.sum(hist[roi_mask]))
    mean_w = float(np.sum(hist[roi_mask] * weight_map[roi_mask]) / sum_events) if sum_events > 0 else 1.0
    if mean_w <= 0 or not np.isfinite(mean_w):
        mean_w = 1.0
    weight_map[roi_mask] /= mean_w

    hist_flat = hist * weight_map
    hist_flat[~roi_mask] = 0.0

    # Keep total yield in ROI same as original.
    sum_flat = float(np.sum(hist_flat[roi_mask]))
    if sum_flat > 0:
        hist_flat *= sum_events / sum_flat

    stats = {
        "eps": float(eps),
        "mean_weight_roi": float(np.mean(weight_map[roi_mask])) if np.any(roi_mask) else 0.0,
        "sum_events_roi": float(sum_events),
        "sum_flat_roi": float(np.sum(hist_flat[roi_mask])) if np.any(roi_mask) else 0.0,
    }
    return hs, weight_map, hist_flat, stats


def uniformity_metrics(hist: np.ndarray, roi_mask: np.ndarray) -> dict[str, float]:
    h = np.asarray(hist, dtype=float)
    roi = h[roi_mask]

    if roi.size == 0:
        return {
            "bins_roi": 0.0,
            "mean": 0.0,
            "std": 0.0,
            "cv": 0.0,
            "chi2_ndf": 0.0,
            "kl_to_uniform": 0.0,
            "p95_over_p05": 0.0,
        }

    mean = float(np.mean(roi))
    std = float(np.std(roi))
    cv = float(std / mean) if mean > 0 else 0.0

    if mean > 0:
        chi2 = float(np.sum((roi - mean) ** 2 / mean))
        ndf = max(1, roi.size - 1)
        chi2_ndf = chi2 / ndf
    else:
        chi2_ndf = 0.0

    total = float(np.sum(roi))
    if total > 0:
        p = roi / total
        q = np.full_like(p, 1.0 / p.size)
        nz = p > 0
        kl = float(np.sum(p[nz] * np.log(p[nz] / q[nz])))
    else:
        kl = 0.0

    p05 = float(np.percentile(roi, 5.0))
    p95 = float(np.percentile(roi, 95.0))
    ratio = float(p95 / max(p05, 1e-12))

    return {
        "bins_roi": float(roi.size),
        "mean": mean,
        "std": std,
        "cv": cv,
        "chi2_ndf": float(chi2_ndf),
        "kl_to_uniform": kl,
        "p95_over_p05": ratio,
    }


def save_diagnostics_plot(
    out_png: Path,
    hist: np.ndarray,
    hs: np.ndarray,
    weight_map: np.ndarray,
    hist_flat: np.ndarray,
    roi_mask: np.ndarray,
    xedges: np.ndarray,
    yedges: np.ndarray,
    title: str,
) -> None:
    if "MPLCONFIGDIR" not in os.environ:
        cache_dir = Path("smooth/.mplconfig")
        cache_dir.mkdir(parents=True, exist_ok=True)
        os.environ["MPLCONFIGDIR"] = str(cache_dir.resolve())

    import matplotlib.pyplot as plt

    x0, x1 = float(xedges[0]), float(xedges[-1])
    y0, y1 = float(yedges[0]), float(yedges[-1])
    extent = [x0, x1, y0, y1]

    eps = 1e-9

    fig, axes = plt.subplots(2, 3, figsize=(16, 9), constrained_layout=True)
    fig.suptitle(title, fontsize=14)

    panels = [
        (np.log1p(hist).T, "log(1 + hist)", "magma"),
        (np.log1p(hs).T, "log(1 + smoothed hist)", "magma"),
        (weight_map.T, "weight map", "viridis"),
        (np.log1p(hist_flat).T, "log(1 + flattened hist)", "magma"),
        ((hist / (np.mean(hist[roi_mask]) + eps)).T, "hist / mean(ROI)", "coolwarm"),
        ((hist_flat / (np.mean(hist_flat[roi_mask]) + eps)).T, "flat / mean(ROI)", "coolwarm"),
    ]

    for ax, (img, label, cmap) in zip(axes.flat, panels):
        im = ax.imshow(img, origin="lower", extent=extent, aspect="equal", cmap=cmap)
        ax.set_title(label)
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        fig.colorbar(im, ax=ax, fraction=0.045, pad=0.02)

    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=160)
    plt.close(fig)


def run_one_file(
    input_path: Path,
    output_dir: Path,
    sigma_bins: float,
    eps_frac: float,
    min_count_for_roi: float,
    output_npz: Path | None = None,
    output_png: Path | None = None,
    output_json: Path | None = None,
) -> dict[str, Any]:
    hist, xedges, yedges = get_numpy_data([str(input_path)])

    hs, weight_map, hist_flat, weight_stats = compute_weights(
        hist=hist,
        sigma_bins=sigma_bins,
        eps_frac=eps_frac,
        min_count_for_roi=min_count_for_roi,
    )
    roi_mask = build_roi_mask(hist, min_count=min_count_for_roi)

    before = uniformity_metrics(hist, roi_mask)
    after = uniformity_metrics(hist_flat, roi_mask)

    output_dir.mkdir(parents=True, exist_ok=True)
    stem = input_path.stem

    out_npz = output_npz if output_npz is not None else (output_dir / f"{stem}_flattened.npz")
    out_npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_npz,
        hist=hist,
        xedges=xedges,
        yedges=yedges,
        smoothed_hist=hs,
        weight_map=weight_map,
        hist_flattened=hist_flat,
        roi_mask=roi_mask,
    )

    out_png = output_png if output_png is not None else (output_dir / f"{stem}_diagnostics.png")
    title = (
        f"{input_path.name} | sigma={sigma_bins:.3g} bins, eps_frac={eps_frac:.3g}, "
        f"ROI min_count={min_count_for_roi:g}"
    )
    save_diagnostics_plot(
        out_png=out_png,
        hist=hist,
        hs=hs,
        weight_map=weight_map,
        hist_flat=hist_flat,
        roi_mask=roi_mask,
        xedges=xedges,
        yedges=yedges,
        title=title,
    )

    out_json = output_json if output_json is not None else (output_dir / f"{stem}_report.json")
    out_json.parent.mkdir(parents=True, exist_ok=True)

    report = {
        "input": str(input_path),
        "output_npz": str(out_npz),
        "output_png": str(out_png),
        "output_json": str(out_json),
        "parameters": {
            "sigma_bins": sigma_bins,
            "eps_frac": eps_frac,
            "min_count_for_roi": min_count_for_roi,
        },
        "weight_stats": weight_stats,
        "before": before,
        "after": after,
        "improvement": {
            "cv_ratio_after_over_before": after["cv"] / max(before["cv"], 1e-12),
            "chi2_ndf_ratio_after_over_before": after["chi2_ndf"] / max(before["chi2_ndf"], 1e-12),
            "kl_ratio_after_over_before": after["kl_to_uniform"] / max(before["kl_to_uniform"], 1e-12),
            "p95p05_ratio_after_over_before": after["p95_over_p05"] / max(before["p95_over_p05"], 1e-12),
        },
    }

    out_json.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    return report


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Flatten 2D nonuniform histogram into near-uniform using inverse-density weights.")
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("smooth/mu_positions_hist.npz"),
        help="Input .npz with keys: hist, xedges, yedges",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("smooth/out"), help="Output directory")
    parser.add_argument("--output-npz", type=Path, default=None, help="Explicit output .npz path")
    parser.add_argument("--output-png", type=Path, default=None, help="Explicit output diagnostics .png path")
    parser.add_argument("--output-json", type=Path, default=None, help="Explicit output report .json path")
    parser.add_argument("--sigma-bins", type=float, default=2.0, help="Gaussian smoothing sigma in bins")
    parser.add_argument(
        "--eps-frac",
        type=float,
        default=0.05,
        help="Regularization epsilon as fraction of max(smoothed_hist)",
    )
    parser.add_argument(
        "--min-count-for-roi",
        type=float,
        default=1.0,
        help="ROI mask threshold in original hist (>= threshold)",
    )
    return parser


def main() -> None:
    args = _build_parser().parse_args()

    report = run_one_file(
        input_path=args.input,
        output_dir=args.output_dir,
        sigma_bins=float(args.sigma_bins),
        eps_frac=float(args.eps_frac),
        min_count_for_roi=float(args.min_count_for_roi),
        output_npz=args.output_npz,
        output_png=args.output_png,
        output_json=args.output_json,
    )

    print("\n=== Uniformity metrics ===")
    print("before:", json.dumps(report["before"], ensure_ascii=False))
    print("after :", json.dumps(report["after"], ensure_ascii=False))
    print("improv:", json.dumps(report["improvement"], ensure_ascii=False))
    print(f"saved: {report['output_png']}")
    print(f"saved: {report['output_npz']}")
    print(f"saved: {report['output_json']}")


if __name__ == "__main__":
    main()
