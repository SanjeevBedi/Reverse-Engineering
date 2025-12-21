#!/usr/bin/env python3
"""
Create a specific model with guaranteed cuts for testing top view logic
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from Final_HLR_Engineering_Drawings import FinalHLREngineeringDrawings
from OCC.Core.BRepPrimAPI import BRepPrimAPI_MakeBox
from OCC.Core.BRepAlgoAPI import BRepAlgoAPI_Cut
from OCC.Core.gp import gp_Pnt

def create_test_model_with_cuts():
    """Create a specific test model with guaranteed cuts"""
    generator = FinalHLREngineeringDrawings()
    
    # Create base box: 100x80x60
    generator.base_length = 100
    generator.base_width = 80 
    generator.base_height = 60
    
    print(f"Creating test model: {generator.base_length}x{generator.base_width}x{generator.base_height}")
    
    # Create base shape
    base_box = BRepPrimAPI_MakeBox(gp_Pnt(0, 0, 0), 
                                   generator.base_length, 
                                   generator.base_width, 
                                   generator.base_height).Shape()
    
    # Create a single shallow cut: 30x20x15 from top surface down
    # This should definitely maintain a single shell
    cut1_box = BRepPrimAPI_MakeBox(gp_Pnt(25, 25, 45), 
                                   gp_Pnt(55, 45, 60)).Shape()
    
    # Apply single cut
    cut_op1 = BRepAlgoAPI_Cut(base_box, cut1_box)
    cut_op1.Build()
    if not cut_op1.IsDone():
        print("✗ Cut failed")
        return None
    final_shape = cut_op1.Shape()
    
    # Validate that we have a single shell
    from OCC.Core.TopExp import TopExp_Explorer
    from OCC.Core.TopAbs import TopAbs_SHELL
    shell_explorer = TopExp_Explorer(final_shape, TopAbs_SHELL)
    shell_count = 0
    while shell_explorer.More():
        shell_count += 1
        shell_explorer.Next()
    
    print(f"  Final shape has {shell_count} shell(s)")
    if shell_count != 1:
        print("✗ Model does not have exactly one shell - HLR will not work correctly")
        return None
    
    # Set up generator with the test model
    generator.main_shape = final_shape
    generator.integrated_shape = final_shape
    
    # Set up features for HLR algorithm
    generator.features = [
        {
            'type': 'cut',
            'dimensions': (30, 20, 15),
            'position': (25, 25, 45)
        }
    ]
    
    print("✓ Test model with 1 cut created successfully")
    print("  Cut 1: 30x20x15 at (25, 25, 45) - shallow cut from top")
    
    return generator

if __name__ == "__main__":
    generator = create_test_model_with_cuts()
    
    # Extract edges and analyze
    print("\nExtracting edges...")
    generator.extract_edges_simple()
    
    # Debug: Look at some actual edge coordinates
    print("\nAnalyzing some actual edges:")
    front_visible = generator.visible_edges['front']
    if len(front_visible) > 0:
        for i, edge in enumerate(front_visible[:5]):  # First 5 edges
            start = edge['start_3d']
            end = edge['end_3d']
            mid_x = (start[0] + end[0]) / 2
            mid_y = (start[1] + end[1]) / 2  
            mid_z = (start[2] + end[2]) / 2
            
            print(f"Edge {i+1}: ({start[0]:.1f},{start[1]:.1f},{start[2]:.1f}) -> ({end[0]:.1f},{end[1]:.1f},{end[2]:.1f})")
            print(f"  Midpoint: ({mid_x:.1f},{mid_y:.1f},{mid_z:.1f})")
            
            # Test this midpoint
            for view in ['front', 'top', 'side']:
                visible = generator.geometric_visibility_test((mid_x, mid_y, mid_z), view)
                print(f"    {view}: {'VISIBLE' if visible else 'HIDDEN'}")
            print()
    
    # Check edge counts
    for view in ['front', 'top', 'side']:
        visible_count = len(generator.visible_edges[view])
        hidden_count = len(generator.hidden_edges[view])
        total_count = visible_count + hidden_count
        
        if total_count > 0:
            visible_pct = (visible_count / total_count) * 100
            hidden_pct = (hidden_count / total_count) * 100
            
            print(f"\n{view.upper()} VIEW:")
            print(f"  Visible edges: {visible_count} ({visible_pct:.1f}%)")
            print(f"  Hidden edges:  {hidden_count} ({hidden_pct:.1f}%)")
            print(f"  Total edges:   {total_count}")
    
    # Debug the specific problematic edge
    print(f"\nDebugging Edge 3 specifically:")
    edge3_point = (0.0, 80.0, 30.0)  # This should be HIDDEN in front view
    print(f"Edge 3 midpoint {edge3_point}:")
    
    # Test this point with detailed debug
    print("  Detailed front view test:")
    result = generator.geometric_visibility_test(edge3_point, 'front')
    print(f"    Result: {'VISIBLE' if result else 'HIDDEN'}")
    
    # Let's check the base bounds
    bounds = {
        'x_max': generator.base_length,
        'y_max': generator.base_width, 
        'z_max': generator.base_height
    }
    print(f"  Base bounds: {bounds}")
    x, y, z = edge3_point
    print(f"  Point coordinates: x={x}, y={y}, z={z}")
    print(f"  Distance from back face: abs(y - y_max) = abs({y} - {bounds['y_max']}) = {abs(y - bounds['y_max'])}")
    print(f"  Should be hidden if distance < 0.5: {abs(y - bounds['y_max']) < 0.5}")

    # Test specific points in cut areas
    print(f"\nTesting points in cut areas:")
    
    # Point in center of the cut
    cut1_center = (40, 35, 52.5)  # Middle of cut 1
    print(f"\nCut center {cut1_center}:")
    for view in ['front', 'top', 'side']:
        visible = generator.geometric_visibility_test(cut1_center, view)
        print(f"  {view}: {'VISIBLE' if visible else 'HIDDEN'}")
    
    # Point on boundary of cut (should be visible in top view)
    cut1_boundary = (25, 35, 52.5)  # Left edge of cut 1
    print(f"\nCut boundary {cut1_boundary}:")
    for view in ['front', 'top', 'side']:
        visible = generator.geometric_visibility_test(cut1_boundary, view)
        print(f"  {view}: {'VISIBLE' if visible else 'HIDDEN'}")
    
    # Point inside the model but not in cut area
    interior_point = (15, 15, 30)  # Interior solid
    print(f"\nInterior point {interior_point}:")
    for view in ['front', 'top', 'side']:
        visible = generator.geometric_visibility_test(interior_point, view)
        print(f"  {view}: {'VISIBLE' if visible else 'HIDDEN'}")
    
    # Generate drawing
    print("\nGenerating drawing...")
    generator.create_engineering_drawings_professional("test_cuts_model.png")
    
    print("✓ Test completed! Check test_cuts_model.png for results")
