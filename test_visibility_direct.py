#!/usr/bin/env python3
"""
Direct test of geometric_visibility_test function
"""

import sys
import os
sys.path.append('/Users/sbedi/Nextcloud/Python/Solid')

from Final_HLR_Engineering_Drawings import *

def test_visibility_function():
    """Test the geometric_visibility_test function directly"""
    
    print("Direct Visibility Function Test")
    print("="*50)
    
    # Create generator
    generator = FinalHLREngineeringDrawings()
    generator.base_length = 100
    generator.base_width = 80
    generator.base_height = 50
    generator.features = [
        {
            'type': 'boss',
            'dimensions': (25, 20, 15),
            'position': (15, 25, 50)
        },
        {
            'type': 'cut',
            'dimensions': (20, 15, 25),
            'position': (30, -2, 10)
        }
    ]
    
    print(f"Model setup:")
    print(f"  Base: {generator.base_length}x{generator.base_width}x{generator.base_height}")
    print(f"  Features: {len(generator.features)}")
    
    # Test various points
    test_points = [
        # Base outline points (should be visible)
        (0, 0, 0, "Bottom-front-left corner"),
        (100, 0, 0, "Bottom-front-right corner"),
        (0, 80, 0, "Bottom-back-left corner"),
        (50, 0, 50, "Top-front-center"),
        
        # Interior points (should be hidden)
        (50, 40, 25, "Center interior"),
        (30, 50, 30, "Deep interior"),
        
        # Protrusion points
        (15, 25, 50, "Protrusion base"),
        (27, 35, 65, "Top of protrusion"),
        
        # Cut points
        (35, 0, 20, "Front cut edge"),
        (40, 10, 15, "Inside cut")
    ]
    
    print(f"\nTesting front view visibility:")
    for x, y, z, description in test_points:
        is_visible = generator.geometric_visibility_test((x, y, z), 'front')
        status = "VISIBLE" if is_visible else "HIDDEN"
        print(f"  ({x:3.0f},{y:2.0f},{z:2.0f}) {description:20s} -> {status}")
    
    print(f"\nTesting side view visibility:")
    for x, y, z, description in test_points:
        is_visible = generator.geometric_visibility_test((x, y, z), 'side')
        status = "VISIBLE" if is_visible else "HIDDEN"
        print(f"  ({x:3.0f},{y:2.0f},{z:2.0f}) {description:20s} -> {status}")
    
    print(f"\nTesting top view visibility:")
    for x, y, z, description in test_points:
        is_visible = generator.geometric_visibility_test((x, y, z), 'top')
        status = "VISIBLE" if is_visible else "HIDDEN"
        print(f"  ({x:3.0f},{y:2.0f},{z:2.0f}) {description:20s} -> {status}")

if __name__ == "__main__":
    test_visibility_function()
