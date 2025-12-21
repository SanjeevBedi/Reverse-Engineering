#!/usr/bin/env python3
"""
Debug Hidden Line Rendering
Check if hidden edges are being detected and drawn correctly
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from Final_HLR_Engineering_Drawings import FinalHLREngineeringDrawings

def debug_hidden_lines():
    """Debug hidden line detection and rendering"""
    print("Debugging Hidden Line Rendering")
    print("=" * 50)
    
    # Check if a STEP file was provided as argument
    if len(sys.argv) > 1:
        step_file = sys.argv[1]
        print(f"Loading existing model: {step_file}")
        
        # Load the existing STEP file
        from OCC.Core.STEPControl import STEPControl_Reader
        from OCC.Core.IFSelect import IFSelect_RetDone
        
        generator = FinalHLREngineeringDrawings()
        step_reader = STEPControl_Reader()
        status = step_reader.ReadFile(step_file)
        
        if status == IFSelect_RetDone:
            step_reader.TransferRoots()
            generator.integrated_shape = step_reader.OneShape()
            generator.main_shape = generator.integrated_shape
            print("✓ Existing model loaded successfully")
        else:
            print("✗ Failed to load STEP file")
            return None
    else:
        # Create a generator and run the system
        generator = FinalHLREngineeringDrawings()
        
        # Create a specific test model with guaranteed hidden lines
        print("Creating test model with known hidden geometry...")
        
        # Override the random dimensions with known values for testing
        generator.base_length = 100
        generator.base_width = 80  
        generator.base_height = 60
        
        print(f"Base: {generator.base_length}x{generator.base_width}x{generator.base_height}")
        
        # Use the existing random model creation but it will use our dimensions
        generator.create_random_model()
        generator.integrated_shape = generator.main_shape
        
        print("✓ Test model created successfully")
    
    # Extract edges 
    print("Extracting edges...")
    generator.extract_edges_simple()
    
    # AUTOMATIC EDGE CORRECTION - Validate and fix incorrect classifications
    print("\nValidating and correcting edge classifications...")
    total_corrected = 0
    
    for view in ['front', 'top', 'side']:
        visible_edges = generator.visible_edges[view]
        hidden_edges = generator.hidden_edges[view]
        
        # Check visible edges that should be hidden
        incorrect_visible = []
        for i, edge in enumerate(visible_edges):
            start = edge['start_3d']
            end = edge['end_3d']
            mid_x = (start[0] + end[0]) / 2
            mid_y = (start[1] + end[1]) / 2
            mid_z = (start[2] + end[2]) / 2
            
            should_be_visible = generator.geometric_visibility_test((mid_x, mid_y, mid_z), view)
            if not should_be_visible:
                incorrect_visible.append(i)
        
        # Check hidden edges that should be visible
        incorrect_hidden = []
        for i, edge in enumerate(hidden_edges):
            start = edge['start_3d']
            end = edge['end_3d']
            mid_x = (start[0] + end[0]) / 2
            mid_y = (start[1] + end[1]) / 2
            mid_z = (start[2] + end[2]) / 2
            
            should_be_visible = generator.geometric_visibility_test((mid_x, mid_y, mid_z), view)
            if should_be_visible:
                incorrect_hidden.append(i)
        
        # Move incorrect edges to correct lists
        corrections_made = len(incorrect_visible) + len(incorrect_hidden)
        total_corrected += corrections_made
        
        if corrections_made > 0:
            print(f"  {view.upper()}: Correcting {corrections_made} edges")
            
            # Move incorrectly visible edges to hidden
            for i in reversed(incorrect_visible):  # Reverse to maintain indices
                edge = visible_edges.pop(i)
                hidden_edges.append(edge)
            
            # Move incorrectly hidden edges to visible
            for i in reversed(incorrect_hidden):  # Reverse to maintain indices
                edge = hidden_edges.pop(i)
                visible_edges.append(edge)
    
    if total_corrected > 0:
        print(f"✓ Corrected {total_corrected} incorrectly classified edges")
    else:
        print("✓ All edges correctly classified")
    
    # Debug: Let's examine some actual edge coordinates from visible edges
    print(f"\nAnalyzing corrected edges...")
    total_edges = sum(len(generator.visible_edges[view]) + len(generator.hidden_edges[view]) 
                     for view in ['front', 'top', 'side'])
    print(f"Total edges extracted: {total_edges}")
    
    # Look at a few edge examples from front view
    front_visible = generator.visible_edges['front']
    if len(front_visible) > 0:
        print(f"\nAnalyzing first few front view visible edges:")
        for i, edge in enumerate(front_visible[:3]):  # First 3 edges
            start = edge['start_3d']
            end = edge['end_3d']
            print(f"Edge {i+1}: ({start[0]:.1f},{start[1]:.1f},{start[2]:.1f}) -> ({end[0]:.1f},{end[1]:.1f},{end[2]:.1f})")
            
            # Test midpoint of this edge
            mid_x = (start[0] + end[0]) / 2
            mid_y = (start[1] + end[1]) / 2  
            mid_z = (start[2] + end[2]) / 2
            
            print(f"  Midpoint: ({mid_x:.1f},{mid_y:.1f},{mid_z:.1f})")
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
        else:
            print(f"\n{view.upper()} VIEW: No edges found!")
    
    # Test specific points
    print(f"\nTesting specific points:")
    test_points = [
        ((generator.base_length/2, generator.base_width/2, generator.base_height/2), "Center interior"),
        ((0, 0, 0), "Origin corner"),
        ((generator.base_length, generator.base_width, generator.base_height), "Far corner")
    ]
    
    for point, desc in test_points:
        print(f"\n{desc} {point}:")
        for view in ['front', 'top', 'side']:
            visible = generator.geometric_visibility_test(point, view)
            print(f"  {view}: {'VISIBLE' if visible else 'HIDDEN'}")
    
    return generator

if __name__ == "__main__":
    generator = debug_hidden_lines()
    
    # Generate the drawing to see the result
    print("\nGenerating drawing to verify hidden lines...")
    
    # Before generating, let's output detailed edge information for manual verification
    print("\nDETAILED EDGE ANALYSIS FOR MANUAL VERIFICATION:")
    print("="*60)
    
    for view in ['front', 'top', 'side']:
        print(f"\n{view.upper()} VIEW EDGE DETAILS:")
        print("-"*40)
        
        visible_edges = generator.visible_edges[view]
        hidden_edges = generator.hidden_edges[view]
        
        print(f"\nVISIBLE EDGES ({len(visible_edges)}):")
        for i, edge in enumerate(visible_edges[:10]):  # First 10 edges
            start = edge['start_3d']
            end = edge['end_3d']
            start_2d = edge['start']
            end_2d = edge['end']
            print(f"  V{i+1}: 3D({start[0]:.0f},{start[1]:.0f},{start[2]:.0f})->({end[0]:.0f},{end[1]:.0f},{end[2]:.0f}) | 2D({start_2d[0]:.0f},{start_2d[1]:.0f})->({end_2d[0]:.0f},{end_2d[1]:.0f})")
        
        print(f"\nHIDDEN EDGES ({len(hidden_edges)}):")
        for i, edge in enumerate(hidden_edges[:10]):  # First 10 edges
            start = edge['start_3d']
            end = edge['end_3d']
            start_2d = edge['start']
            end_2d = edge['end']
            print(f"  H{i+1}: 3D({start[0]:.0f},{start[1]:.0f},{start[2]:.0f})->({end[0]:.0f},{end[1]:.0f},{end[2]:.0f}) | 2D({start_2d[0]:.0f},{start_2d[1]:.0f})->({end_2d[0]:.0f},{end_2d[1]:.0f})")
    
    generator.create_engineering_drawings_professional("debug_hidden_lines.png")
