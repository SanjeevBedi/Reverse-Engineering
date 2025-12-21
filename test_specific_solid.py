#!/usr/bin/env python3
"""
Test script to create a specific solid with two protrusions on top face 
and two subtractions on side faces, then test the HLR algorithm
"""

import sys
import os
sys.path.append('/Users/sbedi/Nextcloud/Python/Solid')

from Final_HLR_Engineering_Drawings import *
import random
import OCC.Core.TopExp as TopExp
import OCC.Core.TopAbs as TopAbs

def create_test_solid_with_specific_features():
    """Create a test solid with 2 top protrusions and 2 side subtractions"""
    
    print("Creating Test Solid with Specific Features")
    print("="*60)
    
    # Create the generator instance
    generator = FinalHLREngineeringDrawings()
    
    # Set fixed base dimensions for consistency
    generator.base_length = 100
    generator.base_width = 80  
    generator.base_height = 50
    
    print(f"Base cuboid: {generator.base_length} x {generator.base_width} x {generator.base_height} mm")
    
    # Create the base cuboid using the same approach as create_random_model
    print(f"Creating base cuboid: {generator.base_length} x {generator.base_width} x {generator.base_height}")
    base_box = BRepPrimAPI_MakeBox(generator.base_length, generator.base_width, generator.base_height).Shape()
    generator.main_shape = base_box
    
    # Initialize features list
    generator.features = []
    
    # Create 2 protrusions on TOP FACE (Z = base_height)
    print("\nAdding 2 protrusions on top face...")
    
    # Protrusion 1: Left side of top face
    prot1_width = 25
    prot1_depth = 20  
    prot1_height = 15
    prot1_x = 15  # Left side
    prot1_y = 25  # Middle depth
    prot1_z = generator.base_height  # On top face
    
    protrusion1 = {
        'type': 'boss',
        'dimensions': (prot1_width, prot1_depth, prot1_height),
        'position': (prot1_x, prot1_y, prot1_z)
    }
    generator.features.append(protrusion1)
    
    # Create and add protrusion 1
    prot1_shape = BRepPrimAPI_MakeBox(
        gp_Pnt(prot1_x, prot1_y, prot1_z),
        prot1_width, prot1_depth, prot1_height
    ).Shape()
    generator.main_shape = BRepAlgoAPI_Fuse(generator.main_shape, prot1_shape).Shape()
    print(f"  ✓ Protrusion 1: {prot1_width}x{prot1_depth}x{prot1_height} at ({prot1_x},{prot1_y},{prot1_z})")
    
    # Protrusion 2: Right side of top face  
    prot2_width = 20
    prot2_depth = 25
    prot2_height = 12
    prot2_x = 65  # Right side
    prot2_y = 15  # Front-middle depth
    prot2_z = generator.base_height  # On top face
    
    protrusion2 = {
        'type': 'boss', 
        'dimensions': (prot2_width, prot2_depth, prot2_height),
        'position': (prot2_x, prot2_y, prot2_z)
    }
    generator.features.append(protrusion2)
    
    # Create and add protrusion 2
    prot2_shape = BRepPrimAPI_MakeBox(
        gp_Pnt(prot2_x, prot2_y, prot2_z),
        prot2_width, prot2_depth, prot2_height
    ).Shape()
    generator.main_shape = BRepAlgoAPI_Fuse(generator.main_shape, prot2_shape).Shape()
    print(f"  ✓ Protrusion 2: {prot2_width}x{prot2_depth}x{prot2_height} at ({prot2_x},{prot2_y},{prot2_z})")
    
    # Create 2 subtractions on SIDE FACES
    print("\nAdding 2 subtractions on side faces...")
    
    # Subtraction 1: Front face (Y = 0)
    cut1_width = 20
    cut1_depth = 15  # Goes into the solid
    cut1_height = 25
    cut1_x = 30  # Center-left of front face
    cut1_y = -2   # Slightly before front face to ensure cut
    cut1_z = 10   # Above bottom
    
    subtraction1 = {
        'type': 'cut',
        'dimensions': (cut1_width, cut1_depth, cut1_height),
        'position': (cut1_x, cut1_y, cut1_z)
    }
    generator.features.append(subtraction1)
    
    # Create and subtract cut 1
    cut1_shape = BRepPrimAPI_MakeBox(
        gp_Pnt(cut1_x, cut1_y, cut1_z),
        cut1_width, cut1_depth, cut1_height
    ).Shape()
    generator.main_shape = BRepAlgoAPI_Cut(generator.main_shape, cut1_shape).Shape()
    print(f"  ✓ Subtraction 1: {cut1_width}x{cut1_depth}x{cut1_height} at ({cut1_x},{cut1_y},{cut1_z}) [Front face]")
    
    # Subtraction 2: Right face (X = base_length)
    cut2_width = 12  # Goes into the solid
    cut2_depth = 18
    cut2_height = 20
    cut2_x = generator.base_length - 10  # Into right face
    cut2_y = 45  # Back portion of right face
    cut2_z = 15  # Above bottom
    
    subtraction2 = {
        'type': 'cut',
        'dimensions': (cut2_width, cut2_depth, cut2_height),
        'position': (cut2_x, cut2_y, cut2_z)
    }
    generator.features.append(subtraction2)
    
    # Create and subtract cut 2
    cut2_shape = BRepPrimAPI_MakeBox(
        gp_Pnt(cut2_x, cut2_y, cut2_z),
        cut2_width, cut2_depth, cut2_height
    ).Shape()
    generator.main_shape = BRepAlgoAPI_Cut(generator.main_shape, cut2_shape).Shape()
    print(f"  ✓ Subtraction 2: {cut2_width}x{cut2_depth}x{cut2_height} at ({cut2_x},{cut2_y},{cut2_z}) [Right face]")
    
    # Store the final shape
    generator.shape = generator.main_shape
    
    print(f"\n✅ Test solid created successfully!")
    print(f"   - Base: {generator.base_length}x{generator.base_width}x{generator.base_height} mm")
    print(f"   - 2 protrusions on top face") 
    print(f"   - 2 subtractions on side faces (front + right)")
    print(f"   - Total features: {len(generator.features)}")
    
    return generator

def test_hlr_on_specific_solid():
    """Test HLR algorithm on the specific solid"""
    
    print(f"\n{'='*60}")
    print("Testing HLR Algorithm on Specific Solid")
    print(f"{'='*60}")
    
    # Create the test solid
    generator = create_test_solid_with_specific_features()
    
    # Test edge classification for all views
    edge_explorer = TopExp.TopExp_Explorer(generator.main_shape, TopAbs.TopAbs_EDGE)
    
    view_results = {}
    
    for view_name, view_direction in [('Front', 'front'), ('Top', 'top'), ('Side', 'side')]:
        print(f"\n{view_name} View Analysis:")
        print("-" * 30)
        
        visible_count = 0
        hidden_count = 0
        edge_details = []
        
        # Reset edge explorer
        edge_explorer = TopExp.TopExp_Explorer(generator.main_shape, TopAbs.TopAbs_EDGE)
        
        while edge_explorer.More():
            edge = edge_explorer.Current()
            curve, first, last = BRep_Tool.Curve(edge)
            
            if curve is not None:
                p1 = curve.Value(first)
                p2 = curve.Value(last)
                midpoint = ((p1.X() + p2.X())/2, (p1.Y() + p2.Y())/2, (p1.Z() + p2.Z())/2)
                
                is_visible = generator.geometric_visibility_test(midpoint, view_direction)
                edge_details.append({
                    'midpoint': midpoint,
                    'visible': is_visible,
                    'p1': (p1.X(), p1.Y(), p1.Z()),
                    'p2': (p2.X(), p2.Y(), p2.Z())
                })
                
                if is_visible:
                    visible_count += 1
                else:
                    hidden_count += 1
            
            edge_explorer.Next()
        
        total = visible_count + hidden_count
        visible_pct = (visible_count / total) * 100 if total > 0 else 0
        hidden_pct = (hidden_count / total) * 100 if total > 0 else 0
        
        print(f"  Total edges: {total}")
        print(f"  Visible: {visible_count} ({visible_pct:.1f}%)")
        print(f"  Hidden: {hidden_count} ({hidden_pct:.1f}%)")
        
        view_results[view_name] = {
            'total': total,
            'visible': visible_count,
            'hidden': hidden_count,
            'visible_pct': visible_pct,
            'hidden_pct': hidden_pct,
            'edges': edge_details
        }
        
        # Analyze specific features for this view
        if view_name == 'Top':
            # Check protrusion visibility in top view
            top_protrusion_edges = []
            for edge in edge_details:
                mp = edge['midpoint']
                # Check if edge is at protrusion height (Z > base_height)
                if mp[2] > generator.base_height + 5:
                    top_protrusion_edges.append(edge)
            
            if top_protrusion_edges:
                visible_prot = sum(1 for e in top_protrusion_edges if e['visible'])
                print(f"  Protrusion edges (Z>{generator.base_height+5}): {len(top_protrusion_edges)}")
                print(f"  Visible protrusion edges: {visible_prot}/{len(top_protrusion_edges)} ({100*visible_prot/len(top_protrusion_edges):.1f}%)")
        
        elif view_name == 'Front':
            # Check subtraction visibility in front view
            front_cut_edges = []
            for edge in edge_details:
                mp = edge['midpoint']
                # Check if edge is near front face cut area
                if (25 <= mp[0] <= 55 and mp[1] <= 10 and 5 <= mp[2] <= 40):
                    front_cut_edges.append(edge)
            
            if front_cut_edges:
                visible_cuts = sum(1 for e in front_cut_edges if e['visible'])
                print(f"  Front cut area edges: {len(front_cut_edges)}")
                print(f"  Visible cut edges: {visible_cuts}/{len(front_cut_edges)} ({100*visible_cuts/len(front_cut_edges):.1f}%)")
        
        elif view_name == 'Side':
            # Check right face subtraction in side view
            side_cut_edges = []
            for edge in edge_details:
                mp = edge['midpoint']
                # Check if edge is near right face cut area  
                if (mp[0] >= 85 and 40 <= mp[1] <= 65 and 10 <= mp[2] <= 40):
                    side_cut_edges.append(edge)
            
            if side_cut_edges:
                visible_side_cuts = sum(1 for e in side_cut_edges if e['visible'])
                print(f"  Right cut area edges: {len(side_cut_edges)}")
                print(f"  Visible right cut edges: {visible_side_cuts}/{len(side_cut_edges)} ({100*visible_side_cuts/len(side_cut_edges):.1f}%)")
    
    # Generate engineering drawings
    print(f"\n{'='*60}")
    print("Generating Engineering Drawings")
    print(f"{'='*60}")
    
    # Save the model and generate drawings
    generator.save_step_file('test_specific_model.step')
    print("✓ 3D model saved as 'test_specific_model.step'")
    
    generator.create_engineering_drawings_professional('test_specific_drawings.png')
    print("✓ Engineering drawings saved as 'test_specific_drawings.png'")
    
    # Summary
    print(f"\n{'='*60}")
    print("TEST SUMMARY")
    print(f"{'='*60}")
    print("Model Features:")
    print("  ✓ Base cuboid: 100x80x50 mm")
    print("  ✓ 2 protrusions on top face")
    print("  ✓ 2 subtractions on side faces")
    print()
    print("HLR Results:")
    for view_name, results in view_results.items():
        print(f"  {view_name} view: {results['visible_pct']:.1f}% visible, {results['hidden_pct']:.1f}% hidden")
    print()
    print("Files Generated:")
    print("  - test_specific_model.step (3D model)")
    print("  - test_specific_drawings.png (2D drawings)")

if __name__ == "__main__":
    test_hlr_on_specific_solid()
