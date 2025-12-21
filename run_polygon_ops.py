#!/usr/bin/env python3
"""
Simple wrapper script to run the Polygon Boolean Operations in the correct environment.
This script automatically activates the shapely_env and runs the main script.
"""

import subprocess
import sys
import os

def run_in_shapely_env():
    """Run the main script in the shapely_env environment."""
    try:
        # Get the directory where this script is located
        script_dir = os.path.dirname(os.path.abspath(__file__))
        main_script = os.path.join(script_dir, "Polgon Boolean Ops from shapely.py")
        
        if not os.path.exists(main_script):
            print(f"❌ Main script not found: {main_script}")
            return False
        
        print("🚀 Running polygon operations in shapely_env...")
        
        # For interactive input, we need to activate the environment first, then run python
        # This approach preserves stdin for interactive input
        activate_script = os.path.expanduser("~/miniconda3/etc/profile.d/conda.sh")
        if not os.path.exists(activate_script):
            activate_script = os.path.expanduser("~/anaconda3/etc/profile.d/conda.sh")
        
        if os.path.exists(activate_script):
            print("Using conda activation script for interactive mode...")
            # Create a shell command that sources conda, activates environment, then runs script
            cmd = f'source "{activate_script}" && conda activate shapely_env && python "{main_script}"'
            print(f"Command: {cmd}")
            print("-" * 60)
            
            # Use os.system to preserve stdin/stdout interaction
            result = os.system(cmd)
            return result == 0
        else:
            print("⚠️  Conda activation script not found, trying direct conda run...")
            print("Note: This mode may not support interactive input")
            cmd = ["conda", "run", "-n", "shapely_env", "python", main_script]
            print(f"Command: {' '.join(cmd)}")
            print("-" * 60)
            
            result = subprocess.run(cmd)
            return result.returncode == 0
            
    except Exception as e:
        print(f"❌ Error running script: {e}")
        return False

def check_environment():
    """Check if the shapely_env conda environment exists."""
    try:
        result = subprocess.run(
            ["conda", "env", "list"], 
            capture_output=True, 
            text=True, 
            check=True
        )
        
        if "shapely_env" in result.stdout:
            print("✅ shapely_env environment found")
            return True
        else:
            print("❌ shapely_env environment not found")
            print("Available environments:")
            print(result.stdout)
            return False
            
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("❌ Could not check conda environments")
        return False

if __name__ == "__main__":
    print("Checking conda environment...")
    
    if check_environment():
        success = run_in_shapely_env()
        if success:
            print("\n🎉 All done! Check the visualization windows for results.")
        else:
            print("\n❌ There were errors during execution.")
    else:
        print("\n❌ Cannot proceed without shapely_env environment.")
        print("Please create the environment with required packages:")
        print("  conda create -n shapely_env python=3.9")
        print("  conda activate shapely_env") 
        print("  conda install shapely matplotlib numpy")
