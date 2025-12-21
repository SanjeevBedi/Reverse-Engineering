#!/usr/bin/env python3
"""
View-Specific HLR Test
======================
Test the corrected HLR for top view and side view specifically
"""

import sys
import os
os.environ["DYLD_LIBRARY_PATH"] = "/Users/sbedi/Anaconda/anaconda3/lib"
sys.path.append('/opt/anaconda3/envs/pyocc/lib/python3.9/site-packages')

from OCC.Core.gp import gp_Pnt
from Random_Engineering_Drawings import RandomEngineeringDrawings

print("🎯 VIEW-SPECIFIC HLR VERIFICATION")
print("="*50)

# Create instance and load the existing model
drawer = RandomEngineeringDrawings()

# Read the generated STEP file
from OCC.Core.STEPControl import STEPControl_Reader
from OCC.Core.IFSelect import IFSelect_RetDone

step_reader = STEPControl_Reader()
status = step_reader.ReadFile("random_engineering_model.step")

if status == IFSelect_RetDone:
    step_reader.TransferRoots()
    shape = step_reader.OneShape()
    drawer.shape = shape
    print("✅ Loaded complex model from STEP file")
    print()

    # Test cases focusing on top view and side view
    test_cases = [
        # TOP VIEW TESTS
        {
            'name': 'TOP VIEW: Cut 1 Edge (Z=39→57)',
            'description': 'Cut edges should be VISIBLE from top view',
            'p1': gp_Pnt(7.0, 6.0, 39.0),
            'p2': gp_Pnt(7.0, 6.0, 57.0),
            'view': 'top',
            'expected': True  # Should be visible - you can see into cuts from above
        },
        {
            'name': 'TOP VIEW: Cut 2 Edge (Z=52→57)',
            'description': 'Cut edges should be VISIBLE from top view',
            'p1': gp_Pnt(30.0, 27.0, 52.0),
            'p2': gp_Pnt(30.0, 27.0, 57.0),
            'view': 'top',
            'expected': True  # Should be visible - you can see into cuts from above
        },
        {
            'name': 'TOP VIEW: Boss 4 Top Face (Z=70)',
            'description': 'Boss top face should be VISIBLE from top view',
            'p1': gp_Pnt(21.0, 17.0, 70.0),
            'p2': gp_Pnt(56.0, 17.0, 70.0),
            'view': 'top',
            'expected': True  # Should be visible - top face
        },
        {
            'name': 'TOP VIEW: Base Bottom Face (Z=0)',
            'description': 'Bottom face should be HIDDEN from top view',
            'p1': gp_Pnt(0.0, 0.0, 0.0),
            'p2': gp_Pnt(94.0, 0.0, 0.0),
            'view': 'top',
            'expected': False  # Should be hidden - bottom face
        },
        
        # SIDE VIEW TESTS
        {
            'name': 'SIDE VIEW: Boss 4 Side Face',
            'description': 'Boss facing viewer should be VISIBLE from side view',
            'p1': gp_Pnt(56.0, 17.0, 57.0),
            'p2': gp_Pnt(56.0, 17.0, 70.0),
            'view': 'side',
            'expected': True  # Should be visible - facing viewer
        },
        {
            'name': 'SIDE VIEW: Boss 3 Above Base',
            'description': 'Boss parts above base should be VISIBLE from side view',
            'p1': gp_Pnt(33.0, -6.0, 13.0),
            'p2': gp_Pnt(63.0, -6.0, 13.0),
            'view': 'side',
            'expected': True  # Should be visible - above base level
        },
        {
            'name': 'SIDE VIEW: Cut Edge',
            'description': 'Cut edges should be VISIBLE from side view',
            'p1': gp_Pnt(7.0, 6.0, 39.0),
            'p2': gp_Pnt(43.0, 6.0, 39.0),
            'view': 'side',
            'expected': True  # Should be visible from side
        },
        {
            'name': 'SIDE VIEW: Far Side Face (X=0)',
            'description': 'Far side face should be HIDDEN from side view',
            'p1': gp_Pnt(0.0, 0.0, 0.0),
            'p2': gp_Pnt(0.0, 61.0, 0.0),
            'view': 'side',
            'expected': False  # Should be hidden - far side
        }
    ]

    print("🔍 TESTING VIEW-SPECIFIC HLR:")
    print("-" * 50)
    
    correct_count = 0
    total_count = len(test_cases)
    
    for i, test in enumerate(test_cases, 1):
        print(f"\n{i}. {test['name']}")
        print(f"   {test['description']}")
        print(f"   From: ({test['p1'].X():.1f}, {test['p1'].Y():.1f}, {test['p1'].Z():.1f})")
        print(f"   To:   ({test['p2'].X():.1f}, {test['p2'].Y():.1f}, {test['p2'].Z():.1f})")
        print(f"   View: {test['view'].upper()}")
        
        # Test the actual HLR function
        result = drawer.is_edge_on_visible_face(test['p1'], test['p2'], test['view'])
        expected = test['expected']
        
        if result == expected:
            status = "✅ CORRECT"
            correct_count += 1
        else:
            status = "❌ WRONG"
        
        visibility = "VISIBLE" if result else "HIDDEN"
        expected_visibility = "VISIBLE" if expected else "HIDDEN"
        
        print(f"   Result: {visibility} (expected: {expected_visibility}) {status}")
    
    print("\n" + "="*50)
    print("🎯 VIEW-SPECIFIC RESULTS:")
    print(f"   ✅ Correct: {correct_count}/{total_count}")
    print(f"   📊 Accuracy: {(correct_count/total_count)*100:.1f}%")
    
    if correct_count == total_count:
        print("   🎉 PERFECT! View-specific HLR is working correctly!")
        print("   🏆 Top view shows cuts as visible, side view shows protrusions correctly")
    else:
        print("   ⚠️  Some view-specific classifications need review")

else:
    print("❌ Failed to read STEP file")
