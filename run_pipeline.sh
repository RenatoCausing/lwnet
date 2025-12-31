#!/bin/bash
# =============================================================================
# Full Pipeline: Preprocess and Train FIVES Dataset
# Usage: bash run_pipeline.sh [zip_path]
# =============================================================================

set -e  # Exit on error

# Configuration - modify these as needed
ZIP_PATH="${1:-fives_preprocessed.zip}"
OUTPUT_DIR="data/FIVES_processed"
EXPERIMENT_NAME="fives_wnet_green"
DEVICE="cuda:0"
BATCH_SIZE=8
CYCLES="10/20"  # 10 cycles of 20 epochs each
IM_SIZE="512"
MODEL="wnet"

# Preprocessing toggles (set to --clahe, --gaussian, --shade_correction to enable)
PREPROCESS_FLAGS=""
# PREPROCESS_FLAGS="--clahe"
# PREPROCESS_FLAGS="--clahe --gaussian"

echo "=============================================="
echo "FIVES Green Channel Training Pipeline"
echo "=============================================="
echo "Zip file:    $ZIP_PATH"
echo "Output dir:  $OUTPUT_DIR"
echo "Experiment:  $EXPERIMENT_NAME"
echo "Device:      $DEVICE"
echo "Model:       $MODEL"
echo "Batch size:  $BATCH_SIZE"
echo "Cycles:      $CYCLES"
echo "Image size:  $IM_SIZE"
echo "=============================================="

# Check if zip file exists
if [ ! -f "$ZIP_PATH" ]; then
    echo "ERROR: Zip file not found: $ZIP_PATH"
    echo "Please upload fives_preprocessed.zip to the working directory"
    exit 1
fi

# Step 1: Preprocess
echo ""
echo "[Step 1/2] Preprocessing FIVES dataset..."
echo "=============================================="
python3 preprocess_fives_green.py \
    --zip_path "$ZIP_PATH" \
    --output_root "$OUTPUT_DIR" \
    --train_ratio 0.95 \
    --seed 42 \
    $PREPROCESS_FLAGS

echo ""
echo "Preprocessing complete!"
echo ""

# Step 2: Train
echo "[Step 2/2] Training WNet model..."
echo "=============================================="
python3 train_fives_green.py \
    --csv_train "$OUTPUT_DIR/train.csv" \
    --csv_val "$OUTPUT_DIR/test.csv" \
    --model_name "$MODEL" \
    --batch_size $BATCH_SIZE \
    --cycle_lens "$CYCLES" \
    --im_size "$IM_SIZE" \
    --in_c 1 \
    --device "$DEVICE" \
    --save_path "$EXPERIMENT_NAME" \
    --save_every_cycle True \
    --metric auc \
    --seed 42

echo ""
echo "=============================================="
echo "Pipeline complete!"
echo "=============================================="
echo "Results saved to: experiments/$EXPERIMENT_NAME"
echo ""
echo "To view results:"
echo "  - Best model: experiments/$EXPERIMENT_NAME/best_model/"
echo "  - All cycles: experiments/$EXPERIMENT_NAME/cycle_XX/"
echo "  - Summary:    experiments/$EXPERIMENT_NAME/all_cycles_summary.json"
echo "=============================================="
