#!/usr/bin/env python3
"""
Debug Edge Sampling
Test specific edges to understand why all edges are classified as visible
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from Final_HLR_Engineering_Drawings import FinalHLREngineeringDrawings

def debug_edge_sampling():
    """Debug edge sampling for visibility classification"""
    print("Debugging Edge Sampling")
    print("=" * 40)
    
    # Create a generator
    generator = FinalHLREngineeringDrawings()
    generator.create_random_model()
    
    # Test manual edge points
    print(f"Base dimensions: {generator.base_length} x {generator.base_width} x {generator.base_height}")
    print(f"Features: {len(generator.features)}")
    
    # Test a specific edge that should have hidden portions
    print("\nTesting specific edges:")
    
    # Edge that goes through the interior (should be partially hidden)
    test_edges = [
        # Interior horizontal edge (should be hidden in front view)
        ((generator.base_length/4, generator.base_width/2, generator.base_height/2), 
         (generator.base_length*3/4, generator.base_width/2, generator.base_height/2), 
         "Interior horizontal edge"),
        
        # Back vertical edge (should be hidden in front view)  
        ((generator.base_length/2, generator.base_width*0.9, 0), 
         (generator.base_length/2, generator.base_width*0.9, generator.base_height), 
         "Back vertical edge"),
        
        # Bottom edge (should be hidden in top view)
        ((0, 0, 0), 
         (generator.base_length, 0, 0), 
         "Bottom front edge"),
    ]
    
    for start, end, description in test_edges:
        print(f"\n{description}:")
        print(f"  Start: {start}")
        print(f"  End: {end}")
        
        # Test 7 sample points along this edge
        num_samples = 7
        sample_points = []
        for i in range(num_samples):
            t = i / (num_samples - 1)
            sample_x = start[0] + t * (end[0] - start[0])
            sample_y = start[1] + t * (end[1] - start[1])
            sample_z = start[2] + t * (end[2] - start[2])
            sample_points.append((sample_x, sample_y, sample_z))
        
        for view in ['front', 'top', 'side']:
            print(f"\n  {view.upper()} VIEW:")
            visible_count = 0
            
            for i, point in enumerate(sample_points):
                is_visible = generator.geometric_visibility_test(point, view)
                status = "VISIBLE" if is_visible else "HIDDEN"
                print(f"    Point {i}: {point} -> {status}")
                if is_visible:
                    visible_count += 1
            
            edge_visible = visible_count == len(sample_points)
            print(f"    Edge classification: {'VISIBLE' if edge_visible else 'HIDDEN'} ({visible_count}/{len(sample_points)} visible)")

if __name__ == "__main__":
    debug_edge_sampling()
