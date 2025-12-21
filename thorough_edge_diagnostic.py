#!/usr/bin/env python3
"""
Thorough Edge Classification Diagnostic
Identify fundamental issues with geometric visibility testing
"""

import sys
import os
sys.path.append('/Users/sbedi/Nextcloud/Python/Solid')

from Final_HLR_Engineering_Drawings import FinalHLREngineeringDrawings
from OCC.Core.STEPControl import STEPControl_Reader
from OCC.Core.IFSelect import IFSelect_RetDone

def thorough_diagnostic():
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
    print('THOROUGH EDGE CLASSIFICATION ANALYSIS')
    print('='*60)
    
    # Check EVERY edge thoroughly
    for view in ['front', 'top', 'side']:
        print(f'\n{view.upper()} VIEW DETAILED ANALYSIS:')
        print('-' * 40)
        
        visible_edges = generator.visible_edges[view]
        hidden_edges = generator.hidden_edges[view]
        
        total_errors = 0
        
        # Check ALL visible edges
        print(f'\nChecking {len(visible_edges)} VISIBLE edges:')
        for i, edge in enumerate(visible_edges):
            start = edge['start_3d']
            end = edge['end_3d']
            mid_x = (start[0] + end[0]) / 2
            mid_y = (start[1] + end[1]) / 2
            mid_z = (start[2] + end[2]) / 2
            
            should_be_visible = generator.geometric_visibility_test((mid_x, mid_y, mid_z), view)
            
            if not should_be_visible:
                total_errors += 1
                print(f'  ERROR {total_errors}: Visible edge {i+1} should be HIDDEN')
                print(f'    Edge: ({start[0]:.1f},{start[1]:.1f},{start[2]:.1f}) -> ({end[0]:.1f},{end[1]:.1f},{end[2]:.1f})')
                print(f'    Mid:  ({mid_x:.1f},{mid_y:.1f},{mid_z:.1f})')
                
                # Let's debug the visibility test in detail
                print(f'    Debugging geometric_visibility_test for {view}:')
                
                # Get the bounds
                bounds = generator.get_bounds()
                print(f'      Bounds: {bounds}')
                
                # Check each condition manually
                if view == 'front':
                    # Front view looks from -Y direction (negative Y towards positive Y)
                    back_face_distance = abs(mid_y - bounds['y_max'])
                    print(f'      Back face distance: abs({mid_y:.1f} - {bounds["y_max"]}) = {back_face_distance:.1f}')
                    if back_face_distance < 0.5:
                        print(f'      -> Should be HIDDEN (at back face)')
                    else:
                        print(f'      -> Not at back face')
                        
                elif view == 'top':
                    # Top view looks from -Z direction (negative Z towards positive Z)
                    bottom_face_distance = abs(mid_z - 0)
                    print(f'      Bottom face distance: abs({mid_z:.1f} - 0) = {bottom_face_distance:.1f}')
                    if bottom_face_distance < 0.5:
                        print(f'      -> Should be HIDDEN (at bottom face)')
                    else:
                        print(f'      -> Not at bottom face')
                        
                elif view == 'side':
                    # Side view looks from +X direction (positive X towards negative X)
                    left_face_distance = abs(mid_x - 0)
                    print(f'      Left face distance: abs({mid_x:.1f} - 0) = {left_face_distance:.1f}')
                    if left_face_distance < 0.5:
                        print(f'      -> Should be HIDDEN (at left face)')
                    else:
                        print(f'      -> Not at left face')
                print()
        
        # Check ALL hidden edges  
        print(f'\nChecking {len(hidden_edges)} HIDDEN edges:')
        for i, edge in enumerate(hidden_edges):
            start = edge['start_3d']
            end = edge['end_3d']
            mid_x = (start[0] + end[0]) / 2
            mid_y = (start[1] + end[1]) / 2
            mid_z = (start[2] + end[2]) / 2
            
            should_be_visible = generator.geometric_visibility_test((mid_x, mid_y, mid_z), view)
            
            if should_be_visible:
                total_errors += 1
                print(f'  ERROR {total_errors}: Hidden edge {i+1} should be VISIBLE')
                print(f'    Edge: ({start[0]:.1f},{start[1]:.1f},{start[2]:.1f}) -> ({end[0]:.1f},{end[1]:.1f},{end[2]:.1f})')
                print(f'    Mid:  ({mid_x:.1f},{mid_y:.1f},{mid_z:.1f})')
                print()
        
        print(f'TOTAL ERRORS in {view}: {total_errors}')
        error_pct = (total_errors / (len(visible_edges) + len(hidden_edges))) * 100
        print(f'ERROR RATE: {error_pct:.1f}%')
        
        if total_errors == 0:
            print('✓ All edges correctly classified in this view')
        else:
            print(f'✗ {total_errors} incorrectly classified edges found!')
    
    print('\n' + '='*60)
    print('GEOMETRIC VISIBILITY TEST VERIFICATION')
    print('='*60)
    
    # Test some known points manually
    test_points = [
        # Known visible points (corners, front faces)
        ((0, 0, 0), "Origin corner"),
        ((100, 0, 0), "Front right bottom"),
        ((0, 0, 50), "Front left top"),
        
        # Known hidden points (back faces, interior)
        ((50, 80, 25), "Back face center"),
        ((100, 80, 50), "Back right top"),
        ((50, 40, 25), "Interior center"),
    ]
    
    for point, desc in test_points:
        print(f'\nTesting {desc} {point}:')
        for view in ['front', 'top', 'side']:
            result = generator.geometric_visibility_test(point, view)
            print(f'  {view}: {result}')

if __name__ == "__main__":
    thorough_diagnostic()
