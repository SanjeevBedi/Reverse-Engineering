#!/bin/bash

# OpenCASCADE Engineering Drawings Generator - Setup Script
# This script helps set up the development environment

set -e

echo "=================================================="
echo "OpenCASCADE Engineering Drawings Generator Setup"
echo "=================================================="

# Check if we're in the correct directory
if [ ! -f "pyproject.toml" ] || [ ! -d "src/engineering_drawings" ]; then
    echo "Error: Please run this script from the project root directory"
    echo "Expected files: pyproject.toml, src/engineering_drawings/"
    exit 1
fi

echo "✓ Project structure verified"

# Check Python version
PYTHON_VERSION=$(python3 --version 2>&1 | cut -d" " -f2 | cut -d"." -f1-2)
echo "Python version: $PYTHON_VERSION"

if ! python3 -c "import sys; exit(0 if sys.version_info >= (3, 8) else 1)" 2>/dev/null; then
    echo "Error: Python 3.8 or higher is required"
    echo "Current version: $PYTHON_VERSION"
    exit 1
fi

echo "✓ Python version compatible"

# Check if conda is available
if command -v conda &> /dev/null; then
    echo "Conda detected. Checking for pythonocc-core..."
    
    # Check if pythonocc-core is installed
    if python3 -c "import OCC" 2>/dev/null; then
        echo "✓ pythonocc-core is already installed"
    else
        echo "Installing pythonocc-core via conda..."
        echo "This may take several minutes..."
        
        # Try to install pythonocc-core
        if conda install -c conda-forge pythonocc-core -y; then
            echo "✓ pythonocc-core installed successfully"
        else
            echo "⚠ Warning: Failed to install pythonocc-core automatically"
            echo "Please install manually: conda install -c conda-forge pythonocc-core"
        fi
    fi
else
    echo "⚠ Warning: Conda not detected"
    echo "pythonocc-core installation requires conda or manual setup"
    echo "Please refer to: https://github.com/tpaviot/pythonocc-core"
fi

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
    echo "✓ Virtual environment created"
else
    echo "✓ Virtual environment exists"
fi

# Activate virtual environment and install dependencies
echo "Installing Python dependencies..."
source venv/bin/activate

# Upgrade pip
pip install --upgrade pip

# Install the package in development mode
pip install -e .

# Install development dependencies
pip install -e ".[dev]"

echo "✓ Dependencies installed"

# Run a quick test to verify installation
echo "Running verification test..."
if python3 -c "
import sys
sys.path.insert(0, 'src')
try:
    from engineering_drawings.solid_generator import SolidGenerator
    from engineering_drawings.face_analyzer import FaceAnalyzer
    print('✓ Core modules import successfully')
except ImportError as e:
    print(f'✗ Import error: {e}')
    sys.exit(1)

try:
    import OCC
    print('✓ OpenCASCADE (pythonocc-core) available')
except ImportError:
    print('⚠ Warning: OpenCASCADE not available - some features may not work')
    print('  Install with: conda install -c conda-forge pythonocc-core')

try:
    import numpy, matplotlib, shapely
    print('✓ Required packages (numpy, matplotlib, shapely) available')
except ImportError as e:
    print(f'✗ Missing required package: {e}')
    sys.exit(1)
"; then
    echo "✓ Verification completed successfully"
else
    echo "✗ Verification failed"
    exit 1
fi

deactivate

echo ""
echo "=================================================="
echo "Setup completed successfully! 🎉"
echo "=================================================="
echo ""
echo "Next steps:"
echo "1. Activate the virtual environment:"
echo "   source venv/bin/activate"
echo ""
echo "2. Run the basic example:"
echo "   python examples/basic_examples.py"
echo ""
echo "3. Or run the main application:"
echo "   python run_engineering_drawings.py"
echo ""
echo "4. For development, run tests:"
echo "   python -m pytest tests/"
echo ""
echo "Important notes:"
echo "- Make sure pythonocc-core is installed for OpenCASCADE functionality"
echo "- Use 'conda install -c conda-forge pythonocc-core' if not already installed"
echo "- See README.md for detailed usage instructions"
echo ""
echo "Happy engineering drawing generation!"
