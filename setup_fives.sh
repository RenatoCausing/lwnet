#!/bin/bash
# =============================================================================
# FIVES Dataset Training Setup Script
# Run this script to install all dependencies and set up the environment
# Usage: bash setup_fives.sh
# =============================================================================

set -e  # Exit on error

echo "=============================================="
echo "FIVES Green Channel Training Setup"
echo "=============================================="

# Update system packages
echo ""
echo "[1/6] Updating system packages..."
apt-get update -y

# Install system dependencies
echo ""
echo "[2/6] Installing system dependencies..."
apt-get install -y \
    python3 \
    python3-pip \
    python3-venv \
    unzip \
    wget \
    curl \
    git \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev

# Upgrade pip
echo ""
echo "[3/6] Upgrading pip..."
pip3 install --upgrade pip

# Install Python dependencies
echo ""
echo "[4/6] Installing Python dependencies..."
pip3 install \
    torch \
    torchvision \
    torchaudio \
    numpy \
    pandas \
    opencv-python \
    opencv-python-headless \
    scikit-learn \
    scikit-image \
    Pillow \
    tqdm \
    matplotlib \
    scipy

# Verify installations
echo ""
echo "[5/6] Verifying installations..."
python3 -c "import torch; print(f'PyTorch version: {torch.__version__}')"
python3 -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
python3 -c "import cv2; print(f'OpenCV version: {cv2.__version__}')"
python3 -c "import numpy; print(f'NumPy version: {numpy.__version__}')"
python3 -c "import pandas; print(f'Pandas version: {pandas.__version__}')"
python3 -c "import sklearn; print(f'Scikit-learn version: {sklearn.__version__}')"

# Check if CUDA is available and install CUDA-enabled PyTorch if needed
echo ""
echo "[6/6] Checking CUDA setup..."
if command -v nvidia-smi &> /dev/null; then
    echo "NVIDIA GPU detected:"
    nvidia-smi --query-gpu=name,memory.total --format=csv
    
    # Install CUDA-enabled PyTorch if not already installed with CUDA support
    python3 -c "import torch; exit(0 if torch.cuda.is_available() else 1)" || {
        echo "Reinstalling PyTorch with CUDA support..."
        pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
    }
else
    echo "No NVIDIA GPU detected. Running on CPU."
fi

echo ""
echo "=============================================="
echo "Setup complete!"
echo "=============================================="
echo ""
echo "Next steps:"
echo "1. Upload your fives_preprocessed.zip to the working directory"
echo "2. Run preprocessing:"
echo "   python3 preprocess_fives_green.py --zip_path fives_preprocessed.zip"
echo ""
echo "   Optional preprocessing flags:"
echo "   --clahe              Enable CLAHE contrast enhancement"
echo "   --gaussian           Enable Gaussian blur"
echo "   --shade_correction   Enable shade correction"
echo ""
echo "3. Run training:"
echo "   python3 train_fives_green.py --device cuda:0"
echo ""
echo "Or run the full pipeline:"
echo "   bash run_pipeline.sh"
echo "=============================================="
