"""Preprocess the FIVES dataset and generate IW-Net-compatible CSV splits.

The script mirrors the legacy FIVES preprocessing pipeline (green channel, min-pooling,
CLAHE, optional smoothing and shade correction) but stores full-resolution images instead
of extracting patches. The resulting images, masks, and ground-truth labels are written to
``output-root`` and split into ``train.csv``/``val.csv`` so that ``train_cyclical.py`` can
be launched without additional manual steps.
"""

from __future__ import annotations

import argparse
import json
import random
import shutil
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import cv2
import numpy as np
import pandas as pd

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}


def _is_image_file(path: Path) -> bool:
    return path.suffix.lower() in IMAGE_EXTENSIONS


def _build_lookup_with_suffix(folder: Path, suffix: str) -> Dict[str, Path]:
    """Create a dictionary keyed by sample id for quick file matching."""

    lookup: Dict[str, Path] = {}
    if not folder.exists():
        return lookup
    for path in folder.iterdir():
        if not path.is_file() or not _is_image_file(path):
            continue
        stem = path.stem
        if not stem.endswith(suffix):
            continue
        base = stem[: -len(suffix)]
        lookup[base] = path
    return lookup


def _sort_key(path: Path) -> Tuple[str, int, str]:
    stem = path.stem
    prefix, _, remainder = stem.partition("_")
    digits = "".join(ch for ch in remainder if ch.isdigit())
    numeric = int(digits) if digits else 0
    return prefix, numeric, stem


def _extract_green_channel(image: np.ndarray) -> np.ndarray:
    if image is None:
        raise ValueError("Received empty image for green channel extraction")
    if image.ndim == 3 and image.shape[2] >= 2:
        return image[:, :, 1]
    return image


def _min_pooling(image: np.ndarray, alpha: float = 4.0, beta: float = -4.0,
                 sigma: float = 10.0, gamma: float = 128.0) -> np.ndarray:
    img_float = image.astype(np.float32)
    kernel_size = int(6 * sigma + 1)
    if kernel_size % 2 == 0:
        kernel_size += 1
    gaussian_filtered = cv2.GaussianBlur(img_float, (kernel_size, kernel_size), sigma)
    result = alpha * img_float + beta * gaussian_filtered + gamma
    result = np.clip(result, 0, 255).astype(np.uint8)
    return result


def _apply_clahe(image: np.ndarray, clip_limit: float = 0.03,
                 tile_grid_size: Tuple[int, int] = (8, 8)) -> np.ndarray:
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    return clahe.apply(image)


def _median_blur(image: np.ndarray, ksize: int = 5) -> np.ndarray:
    return cv2.medianBlur(image, ksize)


def _shade_correct(enhanced_image: np.ndarray, background_image: np.ndarray) -> np.ndarray:
    float_img = enhanced_image.astype(np.float32)
    background = background_image.astype(np.float32)
    corrected = float_img - background
    corrected[corrected > 0] = 0
    corrected = -corrected
    max_val = float(np.max(corrected))
    if max_val <= 0:
        return enhanced_image
    corrected = (corrected / max_val) * 255.0
    return corrected.astype(np.uint8)


def _apply_fov_mask(image: np.ndarray, mask: Optional[np.ndarray]) -> np.ndarray:
    if mask is None:
        return image
    if mask.shape != image.shape:
        mask = cv2.resize(mask, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST)
    if image.dtype == np.uint8:
        return cv2.bitwise_and(image, image, mask=(mask > 0).astype(np.uint8) * 255)
    return image * ((mask > 0).astype(image.dtype))


def _stack_preview(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    height = max(left.shape[0], right.shape[0])
    width_left, width_right = left.shape[1], right.shape[1]
    left_resized = cv2.resize(left, (width_left, height)) if left.shape[0] != height else left
    right_resized = cv2.resize(right, (width_right, height)) if right.shape[0] != height else right
    if left_resized.ndim == 2:
        left_resized = cv2.cvtColor(left_resized, cv2.COLOR_GRAY2BGR)
    if right_resized.ndim == 2:
        right_resized = cv2.cvtColor(right_resized, cv2.COLOR_GRAY2BGR)
    spacer = np.full((height, 10, 3), 128, dtype=np.uint8)
    return np.concatenate([left_resized, spacer, right_resized], axis=1)


def _ensure_dir(path: Path, clean: bool) -> None:
    if clean and path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def _save_sample_preview(sample_id: str, processed: np.ndarray, gt: np.ndarray, preview_dir: Path) -> None:
    preview_dir.mkdir(parents=True, exist_ok=True)
    stacked = _stack_preview(processed, gt)
    cv2.imwrite(str(preview_dir / f"{sample_id}_preview.png"), stacked)


def _get_fives_file_list(dataset_path: Path) -> List[Tuple[Path, Path, Optional[Path]]]:
    original_dir = dataset_path / "Original/PNG"
    mask_dir = dataset_path / "Mask"
    segmented_dir = dataset_path / "Segmented/PNG"

    if not original_dir.exists() or not segmented_dir.exists():
        raise FileNotFoundError("FIVES dataset must contain Original/PNG and Segmented/PNG directories")

    mask_lookup = _build_lookup_with_suffix(mask_dir, "_mask")
    seg_lookup = _build_lookup_with_suffix(segmented_dir, "_segment")

    file_list: List[Tuple[Path, Path, Optional[Path]]] = []
    for img_path in sorted(
        (p for p in original_dir.iterdir() if p.is_file() and _is_image_file(p)),
        key=_sort_key,
    ):
        stem = img_path.stem
        seg_path = seg_lookup.get(stem)
        if seg_path is None:
            continue
        mask_path = mask_lookup.get(stem)
        file_list.append((img_path, seg_path, mask_path))

    if not file_list:
        raise RuntimeError(f"No usable samples found under {dataset_path}")
    return file_list


def _preprocess_image(image: np.ndarray, mask_image: Optional[np.ndarray],
                      apply_median_blur: bool, shade_correction: bool,
                      median_kernel: int, shade_kernel: int) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    green = _extract_green_channel(image)
    min_pooled = _min_pooling(green, alpha=4.0, beta=-4.0, sigma=10, gamma=128.0)
    enhanced = _apply_clahe(min_pooled, clip_limit=0.03, tile_grid_size=(8, 8))
    processed = _median_blur(enhanced, median_kernel) if apply_median_blur else enhanced

    if shade_correction:
        blurred = _median_blur(processed, median_kernel)
        shade_kernel = max(3, abs(int(shade_kernel)))
        if shade_kernel % 2 == 0:
            shade_kernel += 1
        background = _median_blur(blurred, shade_kernel)
        processed = _shade_correct(blurred, background)

    processed = _apply_fov_mask(processed, mask_image)
    if mask_image is None:
        return processed, None
    if mask_image.shape != processed.shape:
        mask_image = cv2.resize(mask_image, (processed.shape[1], processed.shape[0]), interpolation=cv2.INTER_NEAREST)
    return processed, mask_image


def _load_single_sample(img_path: Path, seg_path: Path, mask_path: Optional[Path],
                        apply_median_blur: bool, shade_correction: bool,
                        median_kernel: int, shade_kernel: int) -> Optional[Dict[str, np.ndarray]]:
    image = cv2.imread(str(img_path), cv2.IMREAD_UNCHANGED)
    if image is None:
        return None

    mask_img = None
    if mask_path is not None:
        mask_img = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)

    processed, resized_mask = _preprocess_image(
        image, mask_img, apply_median_blur, shade_correction, median_kernel, shade_kernel
    )

    gt = cv2.imread(str(seg_path), cv2.IMREAD_GRAYSCALE)
    if gt is None:
        return None
    gt = ((gt > 0).astype(np.uint8)) * 255
    if resized_mask is not None:
        gt[resized_mask == 0] = 0
    else:
        resized_mask = np.ones_like(processed, dtype=np.uint8) * 255

    return {
        "id": img_path.stem,
        "image": processed,
        "gt": gt,
        "mask": resized_mask,
    }


def _save_records(records: Sequence[Dict[str, str]], csv_path: Path) -> None:
    if not records:
        raise RuntimeError("No records available to write CSV")
    df = pd.DataFrame(records)
    df.to_csv(csv_path, index=False)


def _split_records(records: Sequence[Dict[str, str]], val_fraction: float, seed: int) -> Tuple[List[Dict[str, str]], List[Dict[str, str]]]:
    if not 0.0 < val_fraction < 1.0:
        raise ValueError("val_fraction must be between 0 and 1")
    shuffled = list(records)
    random.Random(seed).shuffle(shuffled)
    val_len = max(1, int(len(shuffled) * val_fraction))
    val_records = shuffled[:val_len]
    train_records = shuffled[val_len:]
    if not train_records:
        raise RuntimeError("Validation split consumed all samples; lower val_fraction")
    return train_records, val_records


def _compute_intensity_stats(samples: Iterable[np.ndarray]) -> Dict[str, float]:
    total_pixels = 0
    total_sum = 0.0
    total_sumsq = 0.0
    min_val = float("inf")
    max_val = float("-inf")

    for sample in samples:
        sample_float = sample.astype(np.float64)
        total_pixels += sample_float.size
        total_sum += float(sample_float.sum())
        total_sumsq += float((sample_float ** 2).sum())
        min_val = min(min_val, float(sample_float.min()))
        max_val = max(max_val, float(sample_float.max()))

    if total_pixels == 0:
        raise RuntimeError("Unable to compute statistics for empty dataset")

    mean = total_sum / total_pixels
    variance = max((total_sumsq / total_pixels) - mean ** 2, 1e-8)
    std = variance ** 0.5
    return {
        "mean": mean,
        "std": std,
        "min": min_val,
        "max": max_val,
        "total_pixels": total_pixels,
    }


def _append_metadata(metadata_path: Path, payload: Dict[str, object]) -> None:
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    with metadata_path.open("w", encoding="utf-8") as fp:
        json.dump(payload, fp, indent=2)


def _save_sample_previews_from_files(dataset_path: Path, preview_dir: Path, num_samples: int,
                                     apply_median_blur: bool, shade_correction: bool,
                                     median_kernel: int, shade_kernel: int) -> None:
    file_list = _get_fives_file_list(dataset_path)
    selected = file_list[: min(num_samples, len(file_list))]
    saved = 0
    for img_path, seg_path, mask_path in selected:
        sample = _load_single_sample(
            img_path,
            seg_path,
            mask_path,
            apply_median_blur,
            shade_correction,
            median_kernel,
            shade_kernel,
        )
        if sample is None:
            continue
        _save_sample_preview(sample["id"], sample["image"], sample["gt"], preview_dir)
        saved += 1
    print(f"Saved {saved} sample previews to {preview_dir}")


def process_fives_dataset(dataset_path: Path, output_root: Path, *,
                          apply_median_blur: bool = False,
                          shade_correction: bool = False,
                          median_kernel: int = 5,
                          shade_kernel: int = 25,
                          val_fraction: float = 0.2,
                          seed: int = 42,
                          sample_mode: bool = False,
                          preview_dir: Optional[Path] = None,
                          clean_output: bool = False) -> None:
    dataset_path = dataset_path.expanduser().resolve()
    output_root = output_root.expanduser().resolve()

    if sample_mode:
        preview_dir = preview_dir or (output_root / "sample_previews")
        print("Preview-only mode enabled; no files will be written to output root.")
        _save_sample_previews_from_files(
            dataset_path,
            preview_dir,
            num_samples=10,
            apply_median_blur=apply_median_blur,
            shade_correction=shade_correction,
            median_kernel=median_kernel,
            shade_kernel=shade_kernel,
        )
        return

    img_dir = output_root / "images"
    gt_dir = output_root / "labels"
    mask_dir = output_root / "masks"
    for folder in (img_dir, gt_dir, mask_dir):
        _ensure_dir(folder, clean=clean_output)
    if preview_dir is None:
        preview_dir = output_root / "sample_previews"

    print("Scanning dataset...")
    file_list = _get_fives_file_list(dataset_path)
    print(f"Found {len(file_list)} candidate samples")

    records: List[Dict[str, str]] = []
    processed_images: List[np.ndarray] = []

    for idx, (img_path, seg_path, mask_path) in enumerate(file_list, start=1):
        sample = _load_single_sample(
            img_path,
            seg_path,
            mask_path,
            apply_median_blur,
            shade_correction,
            median_kernel,
            shade_kernel,
        )
        if sample is None:
            print(f"Skipping {img_path.name}: could not load image or ground truth")
            continue

        sample_id = sample["id"]
        img_out = img_dir / f"{sample_id}.png"
        gt_out = gt_dir / f"{sample_id}.png"
        mask_out = mask_dir / f"{sample_id}.png"

        cv2.imwrite(str(img_out), sample["image"])
        cv2.imwrite(str(gt_out), sample["gt"])
        cv2.imwrite(str(mask_out), sample["mask"])

        processed_images.append(sample["image"])
        records.append(
            {
                "id": sample_id,
                "im_paths": img_out.as_posix(),
                "gt_paths": gt_out.as_posix(),
                "mask_paths": mask_out.as_posix(),
            }
        )

        if idx % 25 == 0:
            print(f"Processed {idx}/{len(file_list)} samples")

    if not records:
        raise RuntimeError("Every sample failed to process; nothing to train on")

    stats = _compute_intensity_stats(processed_images)

    train_records, val_records = _split_records(records, val_fraction, seed)
    _save_records(train_records, output_root / "train.csv")
    _save_records(val_records, output_root / "val.csv")

    _append_metadata(
        output_root / "normalization_stats_fives.json",
        {
            "stats": stats,
            "num_train": len(train_records),
            "num_val": len(val_records),
            "median_blur": apply_median_blur,
            "shade_correction": shade_correction,
            "median_kernel": median_kernel,
            "shade_kernel": shade_kernel,
            "val_fraction": val_fraction,
        },
    )

    print(f"Finished preprocessing. Train/val split: {len(train_records)}/{len(val_records)}")
    print(f"CSV files stored in {output_root}")
    print("Sample previews can be generated with --sample-mode if needed.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Preprocess FIVES dataset for IW-Net training")
    parser.add_argument("--dataset-root", "-d", type=Path, required=True,
                        help="Path to the raw FIVES dataset root (Original/Segmented/Mask folders)")
    parser.add_argument("--output-root", "-o", type=Path, default=Path("data/FIVES"),
                        help="Destination directory for processed images and CSV files")
    parser.add_argument("--median-blur", action="store_true",
                        help="Enable median blur after CLAHE")
    parser.add_argument("--shade-correction", action="store_true",
                        help="Enable shade correction (median background subtraction)")
    parser.add_argument("--median-kernel", type=int, default=5,
                        help="Kernel size for median blur (default: 5)")
    parser.add_argument("--shade-kernel", type=int, default=25,
                        help="Kernel size used to estimate background for shade correction")
    parser.add_argument("--val-fraction", type=float, default=0.2,
                        help="Fraction of samples reserved for validation (default: 0.2)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducible splits")
    parser.add_argument("--sample-mode", action="store_true",
                        help="Preview first 10 samples without writing processed outputs")
    parser.add_argument("--preview-dir", type=Path,
                        help="Optional directory to store preview PNGs when --sample-mode is set")
    parser.add_argument("--clean-output", action="store_true",
                        help="Remove existing processed files inside output-root before writing new ones")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    process_fives_dataset(
        dataset_path=args.dataset_root,
        output_root=args.output_root,
        apply_median_blur=args.median_blur,
        shade_correction=args.shade_correction,
        median_kernel=args.median_kernel,
        shade_kernel=args.shade_kernel,
        val_fraction=args.val_fraction,
        seed=args.seed,
        sample_mode=args.sample_mode,
        preview_dir=args.preview_dir,
        clean_output=args.clean_output,
    )


if __name__ == "__main__":
    main()
