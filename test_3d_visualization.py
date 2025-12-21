#!/usr/bin/env python3
"""Test script to verify 3D visualization works."""

import sys
import os

# Add the current directory to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import from the main script
from "Polgon Boolean Ops from shapely" import (
    create_opencascade_solid, 
    visualize_3d_solid,
    OPENCASCADE_AVAILABLE
)

def test_visualization():
    """Test the 3D visualization function."""
    print("="*60)
    print("TESTING 3D VISUALIZATION")
    print("="*60)
    
    if not OPENCASCADE_AVAILABLE:
        print("✗ OpenCASCADE not available, cannot test")
        return
    
    print("Step 1: Creating boolean cut solid...")
    solid_shape = create_opencascade_solid()
    
    if solid_shape is None:
        print("✗ Failed to create solid")
        return
    
    print("✓ Solid created successfully")
    
    print("\nStep 2: Testing 3D visualization...")
    try:
        visualize_3d_solid(solid_shape)
        print("✓ Visualization function called successfully")
    except Exception as e:
        print(f"✗ Visualization failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_visualization()
