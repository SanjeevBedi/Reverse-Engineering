#!/usr/bin/env python3
"""
Portable version of V6_current.py for cross-platform compatibility
"""

import os
import sys
import subprocess
import platform

def get_python_executable():
    """Get the appropriate Python executable for the current environment."""
    
    # First, try to use the current Python interpreter
    current_python = sys.executable
    
    # Check if we're in a conda environment
    conda_env = os.environ.get('CONDA_DEFAULT_ENV')
    if conda_env:
        print(f"Found conda environment: {conda_env}")
        return current_python
    
    # Try to find pyocc environment
    possible_paths = [
        # macOS Intel conda
        "/usr/local/miniconda3/envs/pyocc/bin/python",
        "/usr/local/anaconda3/envs/pyocc/bin/python",
        # macOS Apple Silicon conda  
        "/opt/anaconda3/envs/pyocc/bin/python",
        "/opt/miniconda3/envs/pyocc/bin/python",
        # Linux conda
        "/home/*/miniconda3/envs/pyocc/bin/python",
        "/home/*/anaconda3/envs/pyocc/bin/python",
    ]
    
    for path in possible_paths:
        if os.path.exists(path):
            return path
    
    # Fallback: try conda run
    try:
        result = subprocess.run(['conda', 'run', '-n', 'pyocc', 'which', 'python'], 
                              capture_output=True, text=True)
        if result.returncode == 0:
            return result.stdout.strip()
    except FileNotFoundError:
        pass
    
    # Final fallback
    return current_python

def check_dependencies():
    """Check if required dependencies are available."""
    try:
        import numpy
        import matplotlib
        import shapely
        from OCC.Core.BRepPrimAPI import BRepPrimAPI_MakeBox
        print("✓ All dependencies available")
        return True
    except ImportError as e:
        print(f"✗ Missing dependency: {e}")
        print("Please install missing packages:")
        print("conda activate pyocc")
        print("conda install -c conda-forge pythonocc-core matplotlib numpy shapely")
        return False

if __name__ == "__main__":
    print(f"Platform: {platform.system()} {platform.machine()}")
    print(f"Python: {sys.version}")
    
    if not check_dependencies():
        sys.exit(1)
    
    # Import the main module after dependency check
    try:
        # Add current directory to path for imports
        current_dir = os.path.dirname(os.path.abspath(__file__))
        if current_dir not in sys.path:
            sys.path.insert(0, current_dir)
        
        # Now import and run the main functionality
        # You would import your main V6_current functions here
        print("✓ Ready to run main application")
        
        # Example: import and run main function
        # from V6_current import main
        # main()
        
    except ImportError as e:
        print(f"✗ Import error: {e}")
        sys.exit(1)