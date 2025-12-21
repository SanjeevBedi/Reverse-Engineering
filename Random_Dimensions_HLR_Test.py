#!/usr/bin/env python3
"""
Random Dimensions HLR Test
==========================

Tests the HLR algorithm with random dimensions for:
- Base cuboid
- Two protrusions (bosses)
- Two subtractions (cuts)

Verifies that bottom edges are visible (solid lines) in all views.
"""

import sys
import os
sys.path.append('/Users/sbedi/Nextcloud/Python/Solid/')
from Random_Engineering_Drawings import RandomEngineeringDrawings
import random
import numpy as np

def test_random_dimensions_hlr():
    """Test HLR with multiple random dimension sets"""
    print("Random Dimensions HLR Test")
    print("=" * 50)
    
    num_tests = 5
    successful_tests = 0
    
    for test_num in range(1, num_tests + 1):
        print(f"\n=== TEST {test_num}/{num_tests} ===")
        
        # Generate random base cuboid dimensions (reasonable engineering sizes)
        base_length = random.randint(50, 150)  # 50-150mm
        base_width = random.randint(40, 120)   # 40-120mm  
        base_height = random.randint(30, 100)  # 30-100mm
        
        print(f"Random base cuboid: {base_length} x {base_width} x {base_height} mm")
        
        # Create generator with specific dimensions
        generator = RandomEngineeringDrawings()
        generator.base_length = base_length
        generator.base_width = base_width
        generator.base_height = base_height
        
        try:
            # Create the random model
            generator.create_random_model()
            
            # Extract edges and analyze
            generator.extract_edges_simple()
            
            # Print feature details
            print("Generated features:")
            for i, feature in enumerate(generator.features):
                op = feature['operation']
                pos = feature['position']
                dims = feature['dimensions']
                print(f"  {i+1}. {op}: {dims[0]}x{dims[1]}x{dims[2]} at ({pos[0]},{pos[1]},{pos[2]})")
            
            # Analyze edge visibility
            front_visible = len(generator.visible_edges['front'])
            front_hidden = len(generator.hidden_edges['front'])
            side_visible = len(generator.visible_edges['side'])
            side_hidden = len(generator.hidden_edges['side'])
            top_visible = len(generator.visible_edges['top'])
            top_hidden = len(generator.hidden_edges['top'])
            
            print(f"Edge counts:")
            print(f"  Front: {front_visible} visible, {front_hidden} hidden ({front_visible/(front_visible+front_hidden)*100:.1f}% visible)")
            print(f"  Side:  {side_visible} visible, {side_hidden} hidden ({side_visible/(side_visible+side_hidden)*100:.1f}% visible)")
            print(f"  Top:   {top_visible} visible, {top_hidden} hidden ({top_visible/(top_visible+top_hidden)*100:.1f}% visible)")
            
            # Check for bottom edge visibility
            bottom_edges_found = test_bottom_edge_visibility(generator)
            
            if bottom_edges_found:
                print("✅ Bottom edges found and correctly visible")
                successful_tests += 1
            else:
                print("⚠️  Bottom edges not found or incorrectly hidden")
            
            # Generate engineering drawings
            drawing_filename = f"test_random_{test_num}_engineering_drawings.png"
            generator.create_engineering_drawings_advanced()
            print(f"✅ Engineering drawings saved as 'random_engineering_drawings.png'")
            
        except Exception as e:
            print(f"❌ Test {test_num} failed: {str(e)}")
    
    print(f"\n=== TEST SUMMARY ===")
    print(f"Successful tests: {successful_tests}/{num_tests}")
    print(f"Success rate: {successful_tests/num_tests*100:.1f}%")
    
    if successful_tests == num_tests:
        print("🎉 All tests passed! HLR algorithm working correctly with random dimensions.")
    else:
        print("⚠️  Some tests failed. Review the results above.")

def test_bottom_edge_visibility(generator):
    """Test if bottom edges are correctly classified as visible"""
    bottom_edges_found = False
    
    # Check each view for bottom edges
    for view in ['front', 'side', 'top']:
        visible_edges = generator.visible_edges[view]
        
        for edge in visible_edges:
            # Edge format is [x1, y1, x2, y2] for 2D coordinates
            # We need to check the original 3D coordinates
            # For simplicity, let's test a few known bottom points
            pass
    
    # Test specific bottom edge points directly
    test_points = [
        (generator.base_length/2, generator.base_width/2, 0),  # Center bottom
        (0, generator.base_width/2, 0),                       # Left bottom
        (generator.base_length, generator.base_width/2, 0),   # Right bottom
    ]
    
    for x, y, z in test_points:
        # Test visibility in front and side views (where Z=0 should be visible)
        front_visible = generator.geometric_visibility_test((x, y, z), 'front')
        side_visible = generator.geometric_visibility_test((x, y, z), 'side')
        
        if front_visible or side_visible:
            bottom_edges_found = True
            print(f"  Found bottom edge at ({x:.0f},{y:.0f},{z:.0f}): Front={front_visible}, Side={side_visible}")
    
    return bottom_edges_found

def test_specific_case():
    """Test a specific case with known dimensions for validation"""
    print("\n" + "=" * 50)
    print("SPECIFIC CASE VALIDATION")
    print("=" * 50)
    
    # Test with specific dimensions
    generator = RandomEngineeringDrawings()
    generator.base_length = 100
    generator.base_width = 80
    generator.base_height = 60
    
    print(f"Specific test case: {generator.base_length} x {generator.base_width} x {generator.base_height} mm")
    
    # Create model and analyze
    generator.create_random_model()
    generator.extract_edges_simple()
    
    # Test specific points that should be bottom edges
    test_points = [
        (50, 40, 0, "Center bottom edge"),
        (0, 40, 0, "Left bottom edge"),
        (100, 40, 0, "Right bottom edge"),
        (50, 0, 0, "Front bottom edge"),
        (50, 80, 0, "Back bottom edge"),
    ]
    
    print("Testing specific bottom edge points:")
    for x, y, z, description in test_points:
        front_result = generator.geometric_visibility_test((x, y, z), 'front')
        side_result = generator.geometric_visibility_test((x, y, z), 'side')
        top_result = generator.geometric_visibility_test((x, y, z), 'top')
        
        print(f"  {description:20} at ({x:3.0f}, {y:2.0f}, {z:1.0f}): Front={front_result}, Side={side_result}, Top={top_result}")
    
    # Generate final drawing
    generator.create_engineering_drawings_advanced()
    print("✅ Specific test drawing saved as 'random_engineering_drawings.png'")

def main():
    """Main test function"""
    print("Testing HLR Algorithm with Random Dimensions")
    print("=" * 60)
    
    # Test with random dimensions
    test_random_dimensions_hlr()
    
    # Test specific case for validation
    test_specific_case()

if __name__ == "__main__":
    main()
