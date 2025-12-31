"""Preprocess FIVES dataset: unzip, split 95:5, extract green channel, normalize independently.

Optional preprocessing toggles:
- apply_clahe: Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
- apply_gaussian: Apply Gaussian blur/smoothing
- apply_shade_correction: Apply shade correction
"""

import os
import zipfile
import numpy as np
import cv2
from pathlib import Path
import json
from sklearn.model_selection import train_test_split
import shutil

# =============================================================================
# PREPROCESSING TOGGLES - Set these to True/False as needed
# =============================================================================
APPLY_CLAHE = False              # Apply CLAHE contrast enhancement
APPLY_GAUSSIAN = False           # Apply Gaussian blur/smoothing
APPLY_SHADE_CORRECTION = False   # Apply shade correction
GAUSSIAN_KERNEL_SIZE = 5         # Kernel size for Gaussian blur (must be odd)
CLAHE_CLIP_LIMIT = 2.0           # CLAHE clip limit
CLAHE_TILE_GRID_SIZE = (8, 8)    # CLAHE tile grid size
# =============================================================================

def apply_clahe_to_image(image, clip_limit=CLAHE_CLIP_LIMIT, tile_grid_size=CLAHE_TILE_GRID_SIZE):
    """Apply CLAHE to enhance contrast."""
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    return clahe.apply(image)

def apply_gaussian_blur(image, kernel_size=GAUSSIAN_KERNEL_SIZE):
    """Apply Gaussian blur for smoothing."""
    return cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)

def apply_shade_correction_to_image(image, kernel_size=51):
    """Apply shade correction using morphological operations."""
    # Estimate background using morphological closing
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    background = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel)
    # Subtract background and normalize
    corrected = cv2.subtract(background, image)
    corrected = cv2.normalize(corrected, None, 0, 255, cv2.NORM_MINMAX)
    return corrected

def download_and_unzip(zip_path, extract_to):
    """Unzip the FIVES dataset."""
    print(f"Extracting {zip_path} to {extract_to}")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_to)
    print("Extraction complete!")

def compute_normalization_stats(image_paths, channel_idx=1):
    """Compute mean and std for green channel normalization.
    
    Args:
        image_paths: List of paths to images
        channel_idx: 1 for green channel (BGR format)
    
    Returns:
        Dictionary with mean and std
    """
    print(f"Computing normalization statistics for {len(image_paths)} images...")
    pixel_values = []
    
    for img_path in image_paths:
        img = cv2.imread(str(img_path))
        if img is None:
            print(f"Warning: Could not read {img_path}")
            continue
        green_channel = img[:, :, channel_idx]  # Extract green channel
        pixel_values.extend(green_channel.flatten())
    
    pixel_values = np.array(pixel_values, dtype=np.float32)
    mean = float(np.mean(pixel_values))
    std = float(np.std(pixel_values))
    
    print(f"Mean: {mean:.2f}, Std: {std:.2f}")
    return {"mean": mean, "std": std}

def normalize_and_save_green_channel(image_path, output_path, mean, std,
                                      apply_clahe=APPLY_CLAHE, 
                                      apply_gaussian=APPLY_GAUSSIAN,
                                      apply_shade_correction=APPLY_SHADE_CORRECTION):
    """Load image, extract green channel, apply optional preprocessing, normalize, and save.
    
    Args:
        image_path: Path to input image
        output_path: Path to save normalized green channel image
        mean: Mean for normalization
        std: Standard deviation for normalization
        apply_clahe: Whether to apply CLAHE
        apply_gaussian: Whether to apply Gaussian blur
        apply_shade_correction: Whether to apply shade correction
    """
    img = cv2.imread(str(image_path))
    if img is None:
        raise ValueError(f"Could not read image: {image_path}")
    
    # Extract green channel (BGR format, so index 1)
    green_channel = img[:, :, 1].astype(np.uint8)
    
    # Apply optional preprocessing steps
    if apply_shade_correction:
        green_channel = apply_shade_correction_to_image(green_channel)
    
    if apply_clahe:
        green_channel = apply_clahe_to_image(green_channel)
    
    if apply_gaussian:
        green_channel = apply_gaussian_blur(green_channel)
    
    # Convert to float for normalization
    green_float = green_channel.astype(np.float32)
    
    # Normalize
    normalized = (green_float - mean) / (std + 1e-8)
    
    # Convert back to uint8 for storage (rescale to 0-255)
    # Store as single-channel grayscale
    normalized_uint8 = np.clip((normalized * 50 + 127), 0, 255).astype(np.uint8)
    
    # Save as grayscale image
    cv2.imwrite(str(output_path), normalized_uint8)

def process_fives_dataset(zip_path="fives_preprocessed.zip", output_root="data/FIVES_processed", 
                          train_ratio=0.95, random_seed=42,
                          apply_clahe=APPLY_CLAHE, apply_gaussian=APPLY_GAUSSIAN,
                          apply_shade_correction=APPLY_SHADE_CORRECTION):
    """Main processing pipeline for FIVES dataset.
    
    Args:
        zip_path: Path to fives_preprocessed.zip
        output_root: Root directory for processed data
        train_ratio: Ratio for train split (default 0.95 for 95:5 split)
        random_seed: Random seed for reproducibility
        apply_clahe: Whether to apply CLAHE contrast enhancement
        apply_gaussian: Whether to apply Gaussian blur
        apply_shade_correction: Whether to apply shade correction
    """
    # Print preprocessing configuration
    print("\n" + "="*60)
    print("PREPROCESSING CONFIGURATION")
    print("="*60)
    print(f"  CLAHE:            {'ENABLED' if apply_clahe else 'DISABLED'}")
    print(f"  Gaussian Blur:    {'ENABLED' if apply_gaussian else 'DISABLED'}")
    print(f"  Shade Correction: {'ENABLED' if apply_shade_correction else 'DISABLED'}")
    print("="*60 + "\n")
    # Set random seed
    np.random.seed(random_seed)
    
    # Create output directories
    output_root = Path(output_root)
    extract_dir = output_root / "extracted"
    train_dir = output_root / "train"
    test_dir = output_root / "test"
    
    for dir_path in [output_root, extract_dir, train_dir, test_dir]:
        dir_path.mkdir(parents=True, exist_ok=True)
    
    # Create subdirectories for train and test
    (train_dir / "images").mkdir(exist_ok=True)
    (train_dir / "masks").mkdir(exist_ok=True)
    (test_dir / "images").mkdir(exist_ok=True)
    (test_dir / "masks").mkdir(exist_ok=True)
    
    # Step 1: Unzip dataset
    if not (extract_dir / "Original").exists():
        download_and_unzip(zip_path, extract_dir)
    else:
        print("Dataset already extracted, skipping unzip...")
    
    # Step 2: Get all image paths
    original_dir = extract_dir / "Original"
    segmented_dir = extract_dir / "Segmented"
    
    # Get all original images (0001.png to 3080.png format assumed)
    image_files = sorted(list(original_dir.glob("*.png")) + 
                        list(original_dir.glob("*.jpg")) + 
                        list(original_dir.glob("*.tif")))
    
    print(f"Found {len(image_files)} images")
    
    # Create pairs of (original, segmented) paths
    image_pairs = []
    for img_path in image_files:
        # Extract image number/name
        img_name = img_path.stem
        
        # Find corresponding segmented image
        seg_path = segmented_dir / f"{img_name}_segment{img_path.suffix}"
        if not seg_path.exists():
            # Try alternative naming
            seg_path = segmented_dir / f"{img_name}_segment.png"
        
        if seg_path.exists():
            image_pairs.append((img_path, seg_path))
        else:
            print(f"Warning: No segmented image found for {img_name}")
    
    print(f"Created {len(image_pairs)} image pairs")
    
    # Step 3: Split into train/test (95:5)
    train_pairs, test_pairs = train_test_split(
        image_pairs, 
        train_size=train_ratio, 
        random_state=random_seed,
        shuffle=True
    )
    
    print(f"Train set: {len(train_pairs)} images")
    print(f"Test set: {len(test_pairs)} images")
    
    # Step 4: Compute normalization statistics independently for train and test
    train_original_paths = [pair[0] for pair in train_pairs]
    test_original_paths = [pair[0] for pair in test_pairs]
    
    train_stats = compute_normalization_stats(train_original_paths, channel_idx=1)
    test_stats = compute_normalization_stats(test_original_paths, channel_idx=1)
    
    # Save normalization statistics
    with open(output_root / "train_normalization_stats.json", "w") as f:
        json.dump(train_stats, f, indent=2)
    
    with open(output_root / "test_normalization_stats.json", "w") as f:
        json.dump(test_stats, f, indent=2)
    
    print("Normalization statistics saved")
    
    # Step 5: Process and save train images
    print("\nProcessing training images...")
    for idx, (orig_path, seg_path) in enumerate(train_pairs):
        img_name = f"{idx:04d}.png"
        
        # Normalize and save original image (green channel only)
        normalize_and_save_green_channel(
            orig_path,
            train_dir / "images" / img_name,
            train_stats["mean"],
            train_stats["std"],
            apply_clahe=apply_clahe,
            apply_gaussian=apply_gaussian,
            apply_shade_correction=apply_shade_correction
        )
        
        # Copy segmented mask (no normalization needed)
        shutil.copy(seg_path, train_dir / "masks" / img_name)
        
        if (idx + 1) % 100 == 0:
            print(f"Processed {idx + 1}/{len(train_pairs)} training images")
    
    # Step 6: Process and save test images
    print("\nProcessing test images...")
    for idx, (orig_path, seg_path) in enumerate(test_pairs):
        img_name = f"{idx:04d}.png"
        
        # Normalize and save original image (green channel only)
        normalize_and_save_green_channel(
            orig_path,
            test_dir / "images" / img_name,
            test_stats["mean"],
            test_stats["std"],
            apply_clahe=apply_clahe,
            apply_gaussian=apply_gaussian,
            apply_shade_correction=apply_shade_correction
        )
        
        # Copy segmented mask (no normalization needed)
        shutil.copy(seg_path, test_dir / "masks" / img_name)
        
        if (idx + 1) % 10 == 0:
            print(f"Processed {idx + 1}/{len(test_pairs)} test images")
    
    # Step 7: Create CSV files for training
    print("\nCreating CSV files...")
    
    # Train CSV
    train_csv_data = []
    for idx in range(len(train_pairs)):
        img_name = f"{idx:04d}.png"
        train_csv_data.append({
            "im_paths": str(train_dir / "images" / img_name),
            "gt_paths": str(train_dir / "masks" / img_name),
            "mask_paths": str(train_dir / "masks" / img_name)  # Using same as gt for now
        })
    
    import pandas as pd
    train_df = pd.DataFrame(train_csv_data)
    train_df.to_csv(output_root / "train.csv", index=False)
    
    # Test CSV
    test_csv_data = []
    for idx in range(len(test_pairs)):
        img_name = f"{idx:04d}.png"
        test_csv_data.append({
            "im_paths": str(test_dir / "images" / img_name),
            "gt_paths": str(test_dir / "masks" / img_name),
            "mask_paths": str(test_dir / "masks" / img_name)
        })
    
    test_df = pd.DataFrame(test_csv_data)
    test_df.to_csv(output_root / "test.csv", index=False)
    
    # Save preprocessing config for reference
    preprocess_config = {
        "apply_clahe": apply_clahe,
        "apply_gaussian": apply_gaussian,
        "apply_shade_correction": apply_shade_correction,
        "gaussian_kernel_size": GAUSSIAN_KERNEL_SIZE,
        "clahe_clip_limit": CLAHE_CLIP_LIMIT,
        "clahe_tile_grid_size": CLAHE_TILE_GRID_SIZE,
        "train_ratio": train_ratio,
        "random_seed": random_seed,
        "num_train_images": len(train_pairs),
        "num_test_images": len(test_pairs)
    }
    with open(output_root / "preprocess_config.json", "w") as f:
        json.dump(preprocess_config, f, indent=2)
    
    print(f"\nProcessing complete!")
    print(f"Train CSV: {output_root / 'train.csv'}")
    print(f"Test CSV: {output_root / 'test.csv'}")
    print(f"Normalization stats: {output_root / 'train_normalization_stats.json'} and test version")
    print(f"Preprocessing config: {output_root / 'preprocess_config.json'}")
    
    return output_root

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Preprocess FIVES dataset")
    parser.add_argument("--zip_path", type=str, default="fives_preprocessed.zip",
                       help="Path to fives_preprocessed.zip")
    parser.add_argument("--output_root", type=str, default="data/FIVES_processed",
                       help="Output directory for processed data")
    parser.add_argument("--train_ratio", type=float, default=0.95,
                       help="Train split ratio (default: 0.95 for 95:5)")
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed for reproducibility")
    parser.add_argument("--clahe", action="store_true", default=False,
                       help="Apply CLAHE contrast enhancement")
    parser.add_argument("--gaussian", action="store_true", default=False,
                       help="Apply Gaussian blur/smoothing")
    parser.add_argument("--shade_correction", action="store_true", default=False,
                       help="Apply shade correction")
    
    args = parser.parse_args()
    
    process_fives_dataset(
        zip_path=args.zip_path,
        output_root=args.output_root,
        train_ratio=args.train_ratio,
        random_seed=args.seed,
        apply_clahe=args.clahe,
        apply_gaussian=args.gaussian,
        apply_shade_correction=args.shade_correction
    )
