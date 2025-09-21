# Setup Instructions for Intel Mac

## 1. Install Miniconda/Anaconda (if not already installed)
```bash
# Download miniconda for Intel Mac
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-MacOSX-x86_64.sh
bash Miniconda3-latest-MacOSX-x86_64.sh
```

## 2. Create OpenCASCADE Environment
```bash
# Create conda environment with OpenCASCADE
conda create -n pyocc python=3.9
conda activate pyocc

# Install OpenCASCADE and dependencies
conda install -c conda-forge pythonocc-core
conda install -c conda-forge matplotlib numpy shapely

# Alternative if conda-forge doesn't work:
# pip install pythonocc-core matplotlib numpy shapely
```

## 3. Directory Structure Setup
Create the following directory structure on your Intel Mac:

```
your_project_folder/
├── Base_Solid.py
├── V6_current.py
├── Lettering_solid.py (if exists)
└── output files (will be generated)
```

## 4. Path Adjustments Needed
The code currently has hardcoded paths like:
- `/opt/anaconda3/envs/pyocc/bin/python` 
- `/Users/sbedi/Nextcloud/Python/Solid/random_solids`

You'll need to either:
1. Update paths in the code, OR
2. Use relative imports and run from the correct directory

## 5. Test Installation
```bash
# Activate environment
conda activate pyocc

# Test OpenCASCADE import
python -c "from OCC.Core.BRepPrimAPI import BRepPrimAPI_MakeBox; print('OpenCASCADE works!')"

# Test other dependencies
python -c "import matplotlib, numpy, shapely; print('Dependencies work!')"
```

## 6. Running the Code
```bash
# Navigate to your project directory
cd /path/to/your/project

# Run with conda python
conda run -n pyocc python Base_Solid.py --seed 25

# Or activate environment first
conda activate pyocc
python Base_Solid.py --seed 25
python V6_current.py --seed 25 --show_combined
```