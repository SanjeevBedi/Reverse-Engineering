#!/usr/bin/env python3
"""
Check for edge duplication and rendering issues
"""

import sys
import os
sys.path.append('/Users/sbedi/Nextcloud/Python/Solid')

from Final_HLR_Engineering_Drawings import FinalHLREngineeringDrawings
from OCC.Core.STEPControl import STEPControl_Reader
from OCC.Core.IFSelect import IFSelect_RetDone

def check_edge_duplication():
    # Load a model with cuts
    generator = FinalHLREngineeringDrawings()
    step_reader = STEPControl_Reader()
    status = step_reader.ReadFile('test_specific_model.step')

    if status != IFSelect_RetDone:
        print("Failed to load model")
        return

    step_reader.TransferRoots()
    generator.integrated_shape = step_reader.OneShape()
    generator.main_shape = generator.integrated_shape
    print('Model loaded successfully')
    
    # Extract edges
    generator.extract_edges_simple()
    
    print('\n' + '='*60)
    print('EDGE DUPLICATION AND RENDERING CHECK')
    print('='*60)
    
    for view in ['front', 'top', 'side']:
        print(f'\n{view.upper()} VIEW:')
        print('-' * 30)
        
        visible_edges = generator.visible_edges[view]
        hidden_edges = generator.hidden_edges[view]
        
        print(f'Visible edges: {len(visible_edges)}')
        print(f'Hidden edges: {len(hidden_edges)}')
        
        # Check for duplicate edges between visible and hidden
        duplicates = 0
        for v_edge in visible_edges:
            v_start = tuple(v_edge['start_3d'])
            v_end = tuple(v_edge['end_3d'])
            
            for h_edge in hidden_edges:
                h_start = tuple(h_edge['start_3d'])
                h_end = tuple(h_edge['end_3d'])
                
                # Check if same edge (either direction)
                if (v_start == h_start and v_end == h_end) or (v_start == h_end and v_end == h_start):
                    duplicates += 1
                    print(f'  DUPLICATE: Edge ({v_start}) -> ({v_end}) appears in both visible and hidden!')
        
        if duplicates == 0:
            print(f'  ✓ No duplicates between visible and hidden edges')
        else:
            print(f'  ✗ Found {duplicates} duplicate edges!')
        
        # Check for actual coordinate issues
        print(f'\nFirst 3 visible edges in {view}:')
        for i, edge in enumerate(visible_edges[:3]):
            start_3d = edge['start_3d']
            end_3d = edge['end_3d']
            start_2d = edge['start']
            end_2d = edge['end']
            
            print(f'  Edge {i+1}:')
            print(f'    3D: ({start_3d[0]:.1f},{start_3d[1]:.1f},{start_3d[2]:.1f}) -> ({end_3d[0]:.1f},{end_3d[1]:.1f},{end_3d[2]:.1f})')
            print(f'    2D: ({start_2d[0]:.1f},{start_2d[1]:.1f}) -> ({end_2d[0]:.1f},{end_2d[1]:.1f})')
            
            # Verify 2D projection is correct
            if view == 'front':
                expected_start = (start_3d[0], start_3d[2])  # X-Z
                expected_end = (end_3d[0], end_3d[2])
            elif view == 'top':
                expected_start = (start_3d[0], start_3d[1])  # X-Y
                expected_end = (end_3d[0], end_3d[1])
            elif view == 'side':
                expected_start = (start_3d[1], start_3d[2])  # Y-Z
                expected_end = (end_3d[1], end_3d[2])
            
            start_match = abs(start_2d[0] - expected_start[0]) < 0.1 and abs(start_2d[1] - expected_start[1]) < 0.1
            end_match = abs(end_2d[0] - expected_end[0]) < 0.1 and abs(end_2d[1] - expected_end[1]) < 0.1
            
            if start_match and end_match:
                print(f'    ✓ 2D projection correct')
            else:
                print(f'    ✗ 2D projection INCORRECT!')
                print(f'      Expected: ({expected_start[0]:.1f},{expected_start[1]:.1f}) -> ({expected_end[0]:.1f},{expected_end[1]:.1f})')
        
        print(f'\nFirst 3 hidden edges in {view}:')
        for i, edge in enumerate(hidden_edges[:3]):
            start_3d = edge['start_3d']
            end_3d = edge['end_3d']
            start_2d = edge['start']
            end_2d = edge['end']
            
            print(f'  Edge {i+1}:')
            print(f'    3D: ({start_3d[0]:.1f},{start_3d[1]:.1f},{start_3d[2]:.1f}) -> ({end_3d[0]:.1f},{end_3d[1]:.1f},{end_3d[2]:.1f})')
            print(f'    2D: ({start_2d[0]:.1f},{start_2d[1]:.1f}) -> ({end_2d[0]:.1f},{end_2d[1]:.1f})')
            
            # Test if this edge should really be hidden
            mid_x = (start_3d[0] + end_3d[0]) / 2
            mid_y = (start_3d[1] + end_3d[1]) / 2
            mid_z = (start_3d[2] + end_3d[2]) / 2
            
            should_be_visible = generator.geometric_visibility_test((mid_x, mid_y, mid_z), view)
            if should_be_visible:
                print(f'    ✗ This hidden edge should actually be VISIBLE!')
            else:
                print(f'    ✓ Correctly classified as hidden')

if __name__ == "__main__":
    check_edge_duplication()
