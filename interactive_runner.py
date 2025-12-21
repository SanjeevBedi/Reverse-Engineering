#!/usr/bin/env python3
"""
Direct runner for Polygon Boolean Operations with full interactive support.
This script activates shapely_env and runs the main script with proper stdin/stdout.
"""

import os
import sys

def main():
    print("="*60)
    print("POLYGON BOOLEAN OPERATIONS - DIRECT RUNNER")
    print("="*60)
    print("This script will run with full interactive input support.")
    print("You will be prompted to enter projection normal components.")
    print("="*60)
    
    # Get the script directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    main_script = os.path.join(script_dir, "Polgon Boolean Ops from shapely.py")
    
    if not os.path.exists(main_script):
        print(f"❌ Main script not found: {main_script}")
        return
    
    print("🚀 Activating shapely_env and running polygon operations...")
    print("📝 You will be prompted for projection normal components during execution.")
    print("-" * 60)
    
    # Try different conda activation approaches
    conda_paths = [
        os.path.expanduser("~/miniconda3/etc/profile.d/conda.sh"),
        os.path.expanduser("~/anaconda3/etc/profile.d/conda.sh"),
        os.path.expanduser("~/opt/miniconda3/etc/profile.d/conda.sh"),
        os.path.expanduser("~/opt/anaconda3/etc/profile.d/conda.sh"),
        "/opt/conda/etc/profile.d/conda.sh",
        "/usr/local/anaconda3/etc/profile.d/conda.sh"
    ]
    
    conda_script = None
    for path in conda_paths:
        if os.path.exists(path):
            conda_script = path
            break
    
    if conda_script:
        print(f"Found conda activation script: {conda_script}")
        
        # Create a shell command that preserves interactive input
        cmd = f'''
source "{conda_script}"
conda activate shapely_env
echo "Environment activated, starting polygon operations..."
python "{main_script}"
'''
        
        # Use exec to replace the current process, preserving all input/output
        os.system(f'bash -c \'{cmd}\'')
        
    else:
        print("⚠️  No conda activation script found. Trying direct execution...")
        print("Note: This requires shapely_env to be available in PATH")
        
        # Fallback: try to run directly
        import subprocess
        try:
            result = subprocess.run([
                "conda", "run", "-n", "shapely_env", "python", main_script
            ], 
            # Pass through stdin/stdout/stderr for interactive input
            stdin=sys.stdin, 
            stdout=sys.stdout, 
            stderr=sys.stderr
            )
            
        except FileNotFoundError:
            print("❌ Conda not found. Please ensure conda is installed and shapely_env exists.")
        except Exception as e:
            print(f"❌ Error running script: {e}")

if __name__ == "__main__":
    main()
