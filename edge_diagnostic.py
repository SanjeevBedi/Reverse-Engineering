#!/usr/bin/env python3
"""
Detailed Edge Classification Diagnostic
Check for incorrect edge classifications
"""

import sys
import os
sys.path.append('/Users/sbedi/Nextcloud/Python/Solid')

from Final_HLR_Engineering_Drawings import FinalHLREngineeringDrawings
from OCC.Core.STEPControl import STEPControl_Reader
from OCC.Core.IFSelect import IFSelect_RetDone

def diagnose_edges():
    # Load the test model with cuts
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
    
    print('\nDetailed Edge Analysis:')
    total_incorrect = 0
    
    for view in ['front', 'top', 'side']:
        visible = generator.visible_edges[view]
        hidden = generator.hidden_edges[view]
        
        print(f'\n{view.upper()} VIEW:')
        print(f'  Visible: {len(visible)} edges')
        print(f'  Hidden: {len(hidden)} edges')
        
        # Check for incorrect classifications
        incorrect_count = 0
        
        # Check visible edges
        for i, edge in enumerate(visible):
            start = edge['start_3d']
            end = edge['end_3d']
            mid_x = (start[0] + end[0]) / 2
            mid_y = (start[1] + end[1]) / 2
            mid_z = (start[2] + end[2]) / 2
            
            should_be_visible = generator.geometric_visibility_test((mid_x, mid_y, mid_z), view)
            
            if not should_be_visible:
                incorrect_count += 1
                total_incorrect += 1
                if incorrect_count <= 3:  # Show first 3 incorrect
                    print(f'    INCORRECT: Visible edge {i+1} should be HIDDEN')
                    print(f'      Start: ({start[0]:.1f},{start[1]:.1f},{start[2]:.1f})')
                    print(f'      End: ({end[0]:.1f},{end[1]:.1f},{end[2]:.1f})')
                    print(f'      Mid: ({mid_x:.1f},{mid_y:.1f},{mid_z:.1f})')
        
        # Check hidden edges
        for i, edge in enumerate(hidden):
            start = edge['start_3d']
            end = edge['end_3d']
            mid_x = (start[0] + end[0]) / 2
            mid_y = (start[1] + end[1]) / 2
            mid_z = (start[2] + end[2]) / 2
            
            should_be_visible = generator.geometric_visibility_test((mid_x, mid_y, mid_z), view)
            
            if should_be_visible:
                incorrect_count += 1
                total_incorrect += 1
                if incorrect_count <= 6:  # Show up to 6 total incorrect
                    print(f'    INCORRECT: Hidden edge {i+1} should be VISIBLE')
                    print(f'      Start: ({start[0]:.1f},{start[1]:.1f},{start[2]:.1f})')
                    print(f'      End: ({end[0]:.1f},{end[1]:.1f},{end[2]:.1f})')
                    print(f'      Mid: ({mid_x:.1f},{mid_y:.1f},{mid_z:.1f})')
        
        if incorrect_count > 6:
            print(f'    ... and {incorrect_count - 6} more incorrect edges')
        
        print(f'  Total incorrect in {view}: {incorrect_count}')
    
    print(f'\nTotal incorrect edges across all views: {total_incorrect}')
    
    return generator

if __name__ == "__main__":
    generator = diagnose_edges()
