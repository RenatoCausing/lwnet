"""RGB-only preprocessing pipeline for the FIVES dataset.

This variant keeps all channels intact and only applies CLAHE per-channel
(without green-channel extraction, min-pooling, shade correction, or blurs).
The outputs mirror IW-Net expectations: processed RGB PNGs plus masks/labels,
with CSV splits consumable by train_cyclical.py.
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
    lookup: Dict[str, Path] = {}
    if not folder.exists():
        return lookup
    for path in folder.iterdir():
        if not path.is_file() or not _is_image_file(path):
            continue
        stem = path.stem
        if not stem.endswith(suffix):
            continue
        lookup[stem[: -len(suffix)]] = path
    return lookup


def _sort_key(path: Path) -> Tuple[str, int, str]:
    stem = path.stem
    prefix, _, remainder = stem.partition("_")
    digits = "".join(ch for ch in remainder if ch.isdigit())
    numeric = int(digits) if digits else 0
    return prefix, numeric, stem


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
        seg_path = seg_lookup.get(img_path.stem)
        if seg_path is None:
            continue
        file_list.append((img_path, seg_path, mask_lookup.get(img_path.stem)))

    if not file_list:
        raise RuntimeError(f"No usable samples found under {dataset_path}")
    return file_list


def _apply_fov_mask(image: np.ndarray, mask: Optional[np.ndarray]) -> np.ndarray:
    if mask is None:
        return image
    if mask.shape[:2] != image.shape[:2]:
        mask = cv2.resize(mask, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST)
    mask_binary = (mask > 0).astype(np.uint8)
    if image.ndim == 3:
        mask_binary = mask_binary[:, :, None]
    return image * mask_binary


def _rgb_clahe(image: np.ndarray, clip_limit: float = 2.0, tile_grid_size: Tuple[int, int] = (8, 8)) -> np.ndarray:
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    channels = cv2.split(image)
    enhanced = [clahe.apply(ch) for ch in channels]
    return cv2.merge(enhanced)


def _preprocess_image(image: np.ndarray, mask_image: Optional[np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
    if image is None:
        raise ValueError("Failed to load image")
    if image.ndim == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    processed = _rgb_clahe(image)
    processed = _apply_fov_mask(processed, mask_image)

    if mask_image is None:
        mask_out = np.ones(processed.shape[:2], dtype=np.uint8) * 255
    else:
        mask_out = mask_image
        if mask_out.shape != processed.shape[:2]:
            mask_out = cv2.resize(mask_out, (processed.shape[1], processed.shape[0]), interpolation=cv2.INTER_NEAREST)
    return processed, mask_out


def _load_single_sample(img_path: Path, seg_path: Path, mask_path: Optional[Path]) -> Optional[Dict[str, np.ndarray]]:
    image = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
    if image is None:
        return None

    mask_img = None
    if mask_path is not None:
        mask_img = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)

    processed, resized_mask = _preprocess_image(image, mask_img)

    gt = cv2.imread(str(seg_path), cv2.IMREAD_GRAYSCALE)
    if gt is None:
        return None
    gt = ((gt > 0).astype(np.uint8)) * 255
    gt[resized_mask == 0] = 0

    return {
        "id": img_path.stem,
        "image": processed,
        "gt": gt,
        "mask": resized_mask,
    }


def _ensure_dir(path: Path, clean: bool) -> None:
    if clean and path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def _save_records(records: Sequence[Dict[str, str]], csv_path: Path) -> None:
    if not records:
        raise RuntimeError("No records available to write CSV")
    pd.DataFrame(records).to_csv(csv_path, index=False)


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


def _compute_stats(samples: Iterable[np.ndarray]) -> Dict[str, float]:
    total_pixels = 0
    total_sum = 0.0
    total_sumsq = 0.0

    for sample in samples:
        sample_float = sample.astype(np.float32)
        total_pixels += sample_float.size
        total_sum += float(sample_float.sum())
        total_sumsq += float((sample_float ** 2).sum())

    if total_pixels == 0:
        raise RuntimeError("Unable to compute statistics for empty dataset")

    mean = total_sum / total_pixels
    variance = max((total_sumsq / total_pixels) - mean ** 2, 1e-8)
    std = variance ** 0.5
    return {
        "mean": mean,
        "std": std,
        "total_pixels": total_pixels,
    }


def process_fives_rgb(dataset_path: Path, output_root: Path, *,
                      val_fraction: float = 0.2,
                      seed: int = 42,
                      clean_output: bool = False) -> None:
    dataset_path = dataset_path.expanduser().resolve()
    output_root = output_root.expanduser().resolve()

    img_dir = output_root / "images_rgb"
    gt_dir = output_root / "labels"
    mask_dir = output_root / "masks"
    for folder in (img_dir, gt_dir, mask_dir):
        _ensure_dir(folder, clean=clean_output)

    print("Scanning dataset...")
    file_list = _get_fives_file_list(dataset_path)
    print(f"Found {len(file_list)} samples")

    records: List[Dict[str, str]] = []
    processed_images: List[np.ndarray] = []

    for idx, (img_path, seg_path, mask_path) in enumerate(file_list, start=1):
        sample = _load_single_sample(img_path, seg_path, mask_path)
        if sample is None:
            print(f"Skipping {img_path.name}: load failure")
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
        raise RuntimeError("Every sample failed; nothing to train on")

    stats = _compute_stats(processed_images)
    train_records, val_records = _split_records(records, val_fraction, seed)
    _save_records(train_records, output_root / "train_rgb.csv")
    _save_records(val_records, output_root / "val_rgb.csv")

    metadata = {
        "stats": stats,
        "num_train": len(train_records),
        "num_val": len(val_records),
        "note": "RGB CLAHE only (no min-pooling, shade correction, or blur)",
    }
    with (output_root / "normalization_stats_fives_rgb.json").open("w", encoding="utf-8") as fp:
        json.dump(metadata, fp, indent=2)

    print(f"Finished preprocessing. Train/val split: {len(train_records)}/{len(val_records)}")
    print(f"CSV files stored in {output_root}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Preprocess FIVES dataset (RGB CLAHE only)")
    parser.add_argument("--dataset-root", "-d", type=Path, required=True,
                        help="Path to raw FIVES dataset")
    parser.add_argument("--output-root", "-o", type=Path, default=Path("data/FIVES_RGB"),
                        help="Destination directory for processed assets")
    parser.add_argument("--val-fraction", type=float, default=0.2,
                        help="Validation fraction (default: 0.2)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for split reproducibility")
    parser.add_argument("--clean-output", action="store_true",
                        help="Remove existing files inside output directory before writing")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    process_fives_rgb(
        dataset_path=args.dataset_root,
        output_root=args.output_root,
        val_fraction=args.val_fraction,
        seed=args.seed,
        clean_output=args.clean_output,
    )


if __name__ == "__main__":
    main()
