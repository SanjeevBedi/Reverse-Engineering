#!/usr/bin/env python3
"""
Environment setup and usage instructions for Polygon Boolean Operations.
This script provides guidance for running the code in the correct environment.
"""

import subprocess
import sys
import os

def check_conda():
    """Check if conda is available."""
    try:
        result = subprocess.run(["conda", "--version"], capture_output=True, text=True)
        print(f"✅ Conda found: {result.stdout.strip()}")
        return True
    except FileNotFoundError:
        print("❌ Conda not found in PATH")
        return False

def list_environments():
    """List all conda environments."""
    try:
        result = subprocess.run(["conda", "env", "list"], capture_output=True, text=True, check=True)
        print("Available conda environments:")
        print(result.stdout)
        return "shapely_env" in result.stdout
    except subprocess.CalledProcessError:
        print("❌ Could not list conda environments")
        return False

def create_shapely_env():
    """Create the shapely_env environment with required packages."""
    print("Creating shapely_env environment...")
    
    commands = [
        ["conda", "create", "-n", "shapely_env", "python=3.9", "-y"],
        ["conda", "run", "-n", "shapely_env", "conda", "install", "shapely", "matplotlib", "numpy", "-y"]
    ]
    
    for cmd in commands:
        print(f"Running: {' '.join(cmd)}")
        try:
            result = subprocess.run(cmd, check=True)
            print("✅ Command completed successfully")
        except subprocess.CalledProcessError as e:
            print(f"❌ Command failed: {e}")
            return False
    
    return True

def main():
    print("="*70)
    print("POLYGON BOOLEAN OPERATIONS - ENVIRONMENT SETUP")
    print("="*70)
    
    # Check conda
    if not check_conda():
        print("\n❌ Conda is required but not found.")
        print("Please install Anaconda or Miniconda first:")
        print("  https://docs.conda.io/en/latest/miniconda.html")
        return
    
    # Check if shapely_env exists
    shapely_env_exists = list_environments()
    
    if not shapely_env_exists:
        print("\n⚠️  shapely_env environment not found.")
        response = input("Would you like to create it now? (y/n): ").lower().strip()
        
        if response == 'y':
            if create_shapely_env():
                print("✅ shapely_env environment created successfully!")
            else:
                print("❌ Failed to create shapely_env environment")
                return
        else:
            print("❌ Cannot proceed without shapely_env environment")
            return
    else:
        print("✅ shapely_env environment found!")
    
    print("\n" + "="*70)
    print("USAGE INSTRUCTIONS")
    print("="*70)
    print("The Polygon Boolean Operations script is ready to run!")
    print("\nOption 1 - Use the wrapper script:")
    print("  python run_polygon_ops.py")
    print("\nOption 2 - Run directly in conda environment:")
    print('  conda run -n shapely_env python "Polgon Boolean Ops from shapely.py"')
    print("\nOption 3 - Activate environment first:")
    print("  conda activate shapely_env")
    print('  python "Polgon Boolean Ops from shapely.py"')
    
    print("\n📊 EXPECTED OUTPUT:")
    print("  • 3D solid creation and analysis")
    print("  • Interactive projection normal input")
    print("  • 4-subplot visualization showing:")
    print("    - Array A: Initial polygons")
    print("    - Array B: Final polygons") 
    print("    - Array C: Intersection polygons")
    print("    - Combined: Arrays B & C together")
    
    print("\n⚙️  REQUIREMENTS:")
    print("  • OpenCASCADE (for 3D operations)")
    print("  • Shapely (for 2D polygon operations)")
    print("  • Matplotlib (for visualization)")
    print("  • NumPy (for numerical operations)")
    
    print("\n🚀 Ready to run!")

if __name__ == "__main__":
    main()
