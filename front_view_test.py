#!/usr/bin/env python3
"""
Test script to analyze front view HLR algorithm
"""

import sys
import os
sys.path.append('/Users/sbedi/Nextcloud/Python/Solid')

from Final_HLR_Engineering_Drawings import *
import OCC.Core.TopExp as TopExp
import OCC.Core.TopAbs as TopAbs

def test_front_view_hlr():
    """Test front view HLR algorithm in detail"""
    
    print("Front View HLR Algorithm Test")
    print("="*50)
    
    # Create a new model
    generator = FinalHLREngineeringDrawings()
    generator.create_random_model()
    
    print(f"Model bounds:")
    print(f"  Base length (X): 0 to {generator.base_length}")
    print(f"  Base width (Y): 0 to {generator.base_width}")  
    print(f"  Base height (Z): 0 to {generator.base_height}")
    
    # Extract all edges and test front view classification
    edge_explorer = TopExp.TopExp_Explorer(generator.main_shape, TopAbs.TopAbs_EDGE)
    
    visible_count = 0
    hidden_count = 0
    edge_details = []
    
    while edge_explorer.More():
        edge = edge_explorer.Current()
        curve, first, last = BRep_Tool.Curve(edge)
        
        if curve is not None:
            p1 = curve.Value(first)
            p2 = curve.Value(last)
            midpoint = ((p1.X() + p2.X())/2, (p1.Y() + p2.Y())/2, (p1.Z() + p2.Z())/2)
            
            is_visible = generator.geometric_visibility_test(midpoint, 'front')
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
    
    print(f'\nFront view edge classification:')
    print(f'  Total edges: {total}')
    print(f'  Visible: {visible_count} ({visible_pct:.1f}%)')
    print(f'  Hidden: {hidden_count} ({hidden_pct:.1f}%)')
    
    # Analyze edges by Y position (depth from front)
    front_surface_edges = []  # Y ≈ 0 (front face)
    back_surface_edges = []   # Y ≈ max (back face)
    interior_edges = []       # Middle Y values
    
    for edge in edge_details:
        mp = edge['midpoint']
        y_pos = mp[1]
        
        if abs(y_pos) < 2.0:  # Very close to front (Y=0)
            front_surface_edges.append(edge)
        elif y_pos > generator.base_width - 2.0:  # Very close to back
            back_surface_edges.append(edge)
        else:  # Interior/middle
            interior_edges.append(edge)
    
    print(f'\nEdge analysis by Y position (front view perspective):')
    print(f'  Front surface edges (Y≈0): {len(front_surface_edges)}')
    if len(front_surface_edges) > 0:
        front_visible = sum(1 for e in front_surface_edges if e['visible'])
        print(f'    Visible: {front_visible}/{len(front_surface_edges)} ({100*front_visible/len(front_surface_edges):.1f}%)')
    
    print(f'  Back surface edges (Y≈{generator.base_width}): {len(back_surface_edges)}')
    if len(back_surface_edges) > 0:
        back_visible = sum(1 for e in back_surface_edges if e['visible'])
        print(f'    Visible: {back_visible}/{len(back_surface_edges)} ({100*back_visible/len(back_surface_edges):.1f}%)')
    
    print(f'  Interior edges: {len(interior_edges)}')
    if len(interior_edges) > 0:
        interior_visible = sum(1 for e in interior_edges if e['visible'])
        print(f'    Visible: {interior_visible}/{len(interior_edges)} ({100*interior_visible/len(interior_edges):.1f}%)')
    
    # Analyze bottom edges specifically (should be visible)
    bottom_edges = [e for e in edge_details if abs(e['midpoint'][2]) < 1.0]  # Z ≈ 0
    print(f'\nBottom edges (Z≈0): {len(bottom_edges)}')
    if len(bottom_edges) > 0:
        bottom_visible = sum(1 for e in bottom_edges if e['visible'])
        print(f'  Visible: {bottom_visible}/{len(bottom_edges)} ({100*bottom_visible/len(bottom_edges):.1f}%)')
        if bottom_visible < len(bottom_edges):
            print(f'  ⚠️  WARNING: Some bottom edges are hidden - they should be visible!')
    
    # Analyze interior horizontal edges (should mostly be hidden)
    interior_horizontal = []
    for edge in interior_edges:
        mp = edge['midpoint']
        # Check if it's a horizontal edge in the middle
        if (mp[1] > generator.base_width * 0.2 and 
            mp[1] < generator.base_width * 0.8 and
            mp[2] > generator.base_height * 0.2 and
            mp[2] < generator.base_height * 0.8):
            interior_horizontal.append(edge)
    
    print(f'\nInterior horizontal edges: {len(interior_horizontal)}')
    if len(interior_horizontal) > 0:
        int_horiz_visible = sum(1 for e in interior_horizontal if e['visible'])
        print(f'  Visible: {int_horiz_visible}/{len(interior_horizontal)} ({100*int_horiz_visible/len(interior_horizontal):.1f}%)')
        if int_horiz_visible > len(interior_horizontal) * 0.3:
            print(f'  ⚠️  WARNING: Too many interior edges are visible - should be mostly hidden!')
    
    # Show some sample edge coordinates
    print(f'\nSample edge analysis:')
    for i, edge in enumerate(edge_details[:8]):
        status = 'VISIBLE' if edge['visible'] else 'HIDDEN'
        mp = edge['midpoint']
        print(f'  Edge {i+1}: ({mp[0]:.1f}, {mp[1]:.1f}, {mp[2]:.1f}) -> {status}')
    
    print(f'\n{"="*50}')
    print(f'Front view HLR test completed.')

if __name__ == "__main__":
    test_front_view_hlr()
