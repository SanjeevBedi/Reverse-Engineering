#!/usr/bin/env python3
"""
Test the main program with both [1,1,1] and [-1,-1,-1] projection normals
to verify face selection behavior
"""

import sys
import numpy as np
import subprocess
import os

def test_projection_normal_in_main_program():
    """Test both projection normals with the main program"""
    
    print("="*60)
    print("TESTING MAIN PROGRAM WITH DIFFERENT PROJECTION NORMALS")
    print("="*60)
    
    test_cases = [
        {
            'name': 'Positive [1,1,1]',
            'values': ['1', '1', '1']
        },
        {
            'name': 'Negative [-1,-1,-1]',
            'values': ['-1', '-1', '-1']
        }
    ]
    
    os.chdir('/Users/sbedi/Nextcloud/Python/Solid')
    
    for test_case in test_cases:
        print(f"\n" + "="*40)
        print(f"TESTING: {test_case['name']}")
        print("="*40)
        
        # Create input for the program
        input_text = "\n".join(test_case['values']) + "\n"
        
        try:
            # Run the main program with this input
            result = subprocess.run(
                ['conda', 'run', '-n', 'shapely_env', 'python', 'Polgon Boolean Ops from shapely.py'],
                input=input_text,
                text=True,
                capture_output=True,
                timeout=120  # 2 minute timeout
            )
            
            print(f"Return code: {result.returncode}")
            
            # Extract key information from output
            lines = result.stdout.split('\n')
            
            # Look for projection normal info
            projection_info = []
            array_a_info = []
            face_count_info = []
            dot_product_info = []
            
            for line in lines:
                if 'projection normal' in line.lower():
                    projection_info.append(line.strip())
                elif 'array_A contains' in line:
                    array_a_info.append(line.strip())
                elif 'Total valid polygons:' in line:
                    face_count_info.append(line.strip())
                elif 'dot_product' in line and '=' in line:
                    dot_product_info.append(line.strip())
            
            print("\nKey Results:")
            print("Projection Normal Info:")
            for info in projection_info[-2:]:  # Last 2 entries
                print(f"  {info}")
            
            print("\nFace Count Info:")
            for info in face_count_info[-1:]:  # Last entry
                print(f"  {info}")
            
            print("\nArray A Info:")
            for info in array_a_info[-1:]:  # Last entry
                print(f"  {info}")
            
            print("\nDot Product Examples (first 6):")
            for i, info in enumerate(dot_product_info[:6]):
                print(f"  {info}")
            
            # Check for errors
            if result.stderr:
                print(f"\nErrors:")
                error_lines = result.stderr.split('\n')
                for line in error_lines[:5]:  # First 5 error lines
                    if line.strip():
                        print(f"  {line.strip()}")
                        
        except subprocess.TimeoutExpired:
            print(f"  TIMEOUT: Program took longer than 2 minutes")
        except Exception as e:
            print(f"  ERROR: {e}")

if __name__ == "__main__":
    test_projection_normal_in_main_program()
