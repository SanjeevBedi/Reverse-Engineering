#!/bin/bash
# Run script for Intel Mac

# Check if conda is available
if ! command -v conda &> /dev/null; then
    echo "❌ Conda not found. Please install Miniconda or Anaconda first."
    echo "Download from: https://docs.conda.io/en/latest/miniconda.html"
    exit 1
fi

# Check if pyocc environment exists
if ! conda env list | grep -q "pyocc"; then
    echo "📦 Creating pyocc environment..."
    conda create -n pyocc python=3.9 -y
    
    echo "📦 Installing dependencies..."
    conda activate pyocc
    conda install -c conda-forge pythonocc-core matplotlib numpy shapely -y
else
    echo "✅ pyocc environment found"
fi

# Activate environment and run
echo "🚀 Running application..."
conda run -n pyocc python Base_Solid.py --seed 25

echo "🚀 Running V6_current..."
conda run -n pyocc python V6_current.py --seed 25 --show_combined --quiet