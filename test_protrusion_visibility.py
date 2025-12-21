#!/usr/bin/env python3
"""
Test Protrusion and Subtraction Visibility Issues
Create a specific model to test the exact issues reported by user:
1. Top view: Protrusions showing as dashed when they should be solid
2. Top view: Right face subtraction showing as solid when it should be dashed  
3. Side view: Rear protrusion showing as dashed when it should be solid
4. Side view: Right face subtraction showing as solid when it should be dashed
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from Final_HLR_Engineering_Drawings import FinalHLREngineeringDrawings
from OCC.Core import BRepPrimAPI, gp, TopLoc, BRepAlgoAPI

def create_test_model_with_features():
    """Create a specific test model with protrusions and subtractions"""
    print("Creating test model with specific protrusions and subtractions...")
    
    generator = FinalHLREngineeringDrawings()
    
    # Create base box: 100x80x60
    base_box = BRepPrimAPI.BRepPrimAPI_MakeBox(gp.gp_Pnt(0, 0, 0), 100, 80, 60)
    main_shape = base_box.Shape()
    
    print("✓ Base box created: 100x80x60")
    
    # Create a protrusion on top face (should be visible in top view)
    # Small box on top surface: 20x20x10 at position (40, 30, 60)
    protrusion_box = BRepPrimAPI.BRepPrimAPI_MakeBox(gp.gp_Pnt(40, 30, 60), 20, 20, 10)
    fuse_op = BRepAlgoAPI.BRepAlgoAPI_Fuse(main_shape, protrusion_box.Shape())
    fuse_op.Build()
    
    if fuse_op.IsDone():
        main_shape = fuse_op.Shape()
        print("✓ Top protrusion added: 20x20x10 at (40,30,60)")
    else:
        print("✗ Failed to add top protrusion")
    
    # Create a protrusion on rear face (should be visible in side view)
    # Small box on rear surface: 20x10x20 at position (40, 80, 20)
    rear_protrusion_box = BRepPrimAPI.BRepPrimAPI_MakeBox(gp.gp_Pnt(40, 80, 20), 20, 10, 20)
    fuse_op2 = BRepAlgoAPI.BRepAlgoAPI_Fuse(main_shape, rear_protrusion_box.Shape())
    fuse_op2.Build()
    
    if fuse_op2.IsDone():
        main_shape = fuse_op2.Shape()
        print("✓ Rear protrusion added: 20x10x20 at (40,80,20)")
    else:
        print("✗ Failed to add rear protrusion")
    
    # Create a subtraction on right face (should be hidden in top view, hidden in side view)
    # Small box cutting into right surface: 10x20x20 at position (90, 30, 20)
    subtraction_box = BRepPrimAPI.BRepPrimAPI_MakeBox(gp.gp_Pnt(90, 30, 20), 10, 20, 20)
    cut_op = BRepAlgoAPI.BRepAlgoAPI_Cut(main_shape, subtraction_box.Shape())
    cut_op.Build()
    
    if cut_op.IsDone():
        main_shape = cut_op.Shape()
        print("✓ Right face subtraction added: 10x20x20 at (90,30,20)")
    else:
        print("✗ Failed to add right face subtraction")
    
    generator.main_shape = main_shape
    generator.integrated_shape = main_shape
    
    # Set dimensions for the generator
    generator.base_length = 100
    generator.base_width = 80  
    generator.base_height = 60
    
    return generator

def analyze_specific_features():
    """Analyze the specific features that are causing issues"""
    generator = create_test_model_with_features()
    
    # Get model bounds to determine base height
    # We need to set some reasonable base dimensions since we're loading an existing file
    # Try to get from the current state or use defaults
    if not hasattr(drawer, 'base_height') or drawer.base_height is None:
        drawer.base_length = 100  # Default values
        drawer.base_width = 100
        drawer.base_height = 50
    
    # Extract edge data for side view  
    print('Analyzing side view edge classification...')
    drawer.extract_edges_simple()  # This populates the internal edge data structures
    
    # Get the classified edges for side view
    visible_edges = drawer.visible_edges['side']
    hidden_edges = drawer.hidden_edges['side']
    
    visible_count = len(visible_edges)
    hidden_count = len(hidden_edges)
    protrusion_edges = []
    
    base_height = 0  # z_min is always 0 for base
    max_height = drawer.base_height  # Base cuboid height
    
    print(f"Model bounds: Z from {base_height:.1f} to {max_height:.1f}")
    print(f"Base dimensions: {drawer.base_length} x {drawer.base_width} x {drawer.base_height} mm")
    
    # Analyze all edges for protrusions
    all_edges = visible_edges + hidden_edges
    
    for edge_data in all_edges:
        is_visible = edge_data in visible_edges
        
        # Get 3D coordinates
        start_3d = edge_data['start'] if 'start' in edge_data else [0, 0, 0]
        end_3d = edge_data['end'] if 'end' in edge_data else [0, 0, 0]
        
        # Check if this edge might be from a protrusion (z > base height + some threshold)
        max_z = max(start_3d[2], end_3d[2])
        
        # Consider edges above 50% of max height as potential protrusions
        threshold = base_height + 0.5 * (max_height - base_height)
        
        if max_z > threshold:
            protrusion_edges.append({
                'start_3d': start_3d,
                'end_3d': end_3d,
                'visible': is_visible,
                'max_z': max_z,
                'start_z': start_3d[2],
                'end_z': end_3d[2]
            })
    
    total_edges = visible_count + hidden_count
    print(f'\nSide view edge analysis:')
    print(f'  Total edges: {total_edges}')
    print(f'  Visible: {visible_count} ({100*visible_count/total_edges:.1f}%)')
    print(f'  Hidden: {hidden_count} ({100*hidden_count/total_edges:.1f}%)')
    
    print(f'\nProtrusion edge analysis (Z > {threshold:.1f}):')
    print(f'  Protrusion edges found: {len(protrusion_edges)}')
    
    if len(protrusion_edges) > 0:
        visible_protrusions = sum(1 for p in protrusion_edges if p['visible'])
        hidden_protrusions = len(protrusion_edges) - visible_protrusions
        
        print(f'  Visible protrusion edges: {visible_protrusions}/{len(protrusion_edges)} ({100*visible_protrusions/len(protrusion_edges):.1f}%)')
        print(f'  Hidden protrusion edges: {hidden_protrusions}/{len(protrusion_edges)} ({100*hidden_protrusions/len(protrusion_edges):.1f}%)')
        
        # Show some examples
        print(f'\nFirst 5 protrusion edges:')
        for i, p_edge in enumerate(protrusion_edges[:5]):
            status = 'VISIBLE' if p_edge['visible'] else 'HIDDEN'
            print(f'    Edge {i+1}: Z={p_edge['max_z']:.1f} (start={p_edge['start_z']:.1f}, end={p_edge['end_z']:.1f}) -> {status}')
        
        if visible_protrusions > 0:
            print(f'\n✅ Protrusions are correctly showing as VISIBLE lines in side view')
        else:
            print(f'\n⚠️  WARNING: Protrusions may be incorrectly showing as HIDDEN lines')
    else:
        print(f'  No high-Z protrusion edges found - model may be mostly flat')
    
    print(f'\n{"="*50}')
    print(f'Test completed.')

if __name__ == "__main__":
    test_protrusion_visibility()
