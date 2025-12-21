#!/usr/bin/env python3
"""
Test Final HLR System Edge Classification
Tests the actual Final HLR system to verify edge visibility ratios
"""

import sys
import os

# Add the parent directory to the Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from Final_HLR_Engineering_Drawings import FinalHLREngineeringDrawings

def test_final_hlr_edge_classification():
    """Test the Final HLR system's edge classification directly"""
    print("Testing Final HLR System Edge Classification")
    print("=" * 50)
    
    # Create the generator
    generator = FinalHLREngineeringDrawings()
    
    # Create a simple test model with known features
    print("Creating test model with known features...")
    
    # Create base cuboid: 100 x 60 x 40
    generator.base_dimensions = (100, 60, 40)
    generator.features = [
        {
            'type': 'boss',
            'dimensions': (30, 15, 20),
            'position': (25, 35, 40),
            'description': 'Test protrusion at (25,35,40)'
        },
        {
            'type': 'cut',
            'dimensions': (20, 10, 45),
            'position': (50, 20, 0),
            'description': 'Test cut at (50,20,0)'
        }
    ]
    
    print(f"Base cuboid: {generator.base_dimensions}")
    print(f"Features: {len(generator.features)}")
    for i, feature in enumerate(generator.features):
        print(f"  {i+1}. {feature['type']}: {feature['dimensions']} at {feature['position']}")
    
    # Test specific points for visibility
    print("\nTesting specific points for visibility:")
    print("-" * 40)
    
    test_points = [
        # Base cuboid corners (should be visible)
        ((0, 0, 0), "Base corner (0,0,0)"),
        ((100, 60, 40), "Base corner (100,60,40)"),
        ((50, 30, 20), "Base center (50,30,20)"),
        
        # Protrusion points
        ((25, 35, 40), "Protrusion start (25,35,40)"),
        ((40, 42.5, 50), "Protrusion center (40,42.5,50)"),
        ((55, 50, 60), "Protrusion end (55,50,60)"),
        
        # Cut points
        ((50, 20, 0), "Cut start (50,20,0)"),
        ((60, 25, 22.5), "Cut center (60,25,22.5)"),
        ((70, 30, 45), "Cut end (70,30,45)"),
        
        # Interior points (should be hidden)
        ((50, 30, 20), "Deep interior (50,30,20)"),
        ((75, 45, 30), "Interior region (75,45,30)"),
    ]
    
    views = ['front', 'top', 'side']
    
    for point, description in test_points:
        print(f"\n{description}:")
        for view in views:
            is_visible = generator.geometric_visibility_test(point, view)
            status = "VISIBLE" if is_visible else "HIDDEN"
            print(f"  {view} view: {status}")
    
    # Now create edge statistics by sampling edge points
    print("\n" + "=" * 50)
    print("EDGE VISIBILITY ANALYSIS")
    print("=" * 50)
    
    # Generate sample edges and test their visibility
    sample_edges = [
        # Base outline edges
        ((0, 0, 0), (100, 0, 0), "Base bottom X-edge"),
        ((0, 0, 0), (0, 60, 0), "Base bottom Y-edge"),
        ((0, 0, 0), (0, 0, 40), "Base bottom Z-edge"),
        ((100, 60, 40), (0, 60, 40), "Base top X-edge"),
        ((100, 60, 40), (100, 0, 40), "Base top Y-edge"),
        
        # Protrusion edges
        ((25, 35, 40), (55, 35, 40), "Protrusion X-edge"),
        ((25, 35, 40), (25, 50, 40), "Protrusion Y-edge"),
        ((25, 35, 40), (25, 35, 60), "Protrusion Z-edge"),
        
        # Cut edges
        ((50, 20, 0), (70, 20, 0), "Cut X-edge"),
        ((50, 20, 0), (50, 30, 0), "Cut Y-edge"),
        ((50, 20, 0), (50, 20, 45), "Cut Z-edge"),
        
        # Interior edges (should be mostly hidden)
        ((30, 30, 20), (70, 30, 20), "Interior horizontal"),
        ((50, 10, 10), (50, 50, 10), "Interior vertical"),
    ]
    
    for view in views:
        print(f"\n{view.upper()} VIEW ANALYSIS:")
        print("-" * 30)
        visible_count = 0
        hidden_count = 0
        
        for start, end, description in sample_edges:
            # Test midpoint of edge
            mid_x = (start[0] + end[0]) / 2
            mid_y = (start[1] + end[1]) / 2
            mid_z = (start[2] + end[2]) / 2
            midpoint = (mid_x, mid_y, mid_z)
            
            is_visible = generator.geometric_visibility_test(midpoint, view)
            status = "VISIBLE" if is_visible else "HIDDEN"
            
            if is_visible:
                visible_count += 1
            else:
                hidden_count += 1
            
            print(f"  {description}: {status}")
        
        total_edges = visible_count + hidden_count
        if total_edges > 0:
            visible_pct = (visible_count / total_edges) * 100
            hidden_pct = (hidden_count / total_edges) * 100
            print(f"\nSUMMARY for {view} view:")
            print(f"  Visible: {visible_count}/{total_edges} ({visible_pct:.1f}%)")
            print(f"  Hidden:  {hidden_count}/{total_edges} ({hidden_pct:.1f}%)")

if __name__ == "__main__":
    test_final_hlr_edge_classification()
