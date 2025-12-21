#!/usr/bin/env python3
"""
Final HLR Verification - Complex Model
======================================
Test the HLR algorithm on the complex model with:
- 2 protrusions 
- 2 subtractions
"""

import sys
import os
os.environ["DYLD_LIBRARY_PATH"] = "/Users/sbedi/Anaconda/anaconda3/lib"
sys.path.append('/opt/anaconda3/envs/pyocc/lib/python3.9/site-packages')

from OCC.Core.gp import gp_Pnt
from Random_Engineering_Drawings import RandomEngineeringDrawings

print("🎯 FINAL HLR VERIFICATION - COMPLEX MODEL")
print("="*60)
print("Testing model with 2 protrusions + 2 subtractions")
print("="*60)

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

    # Test specific edges based on the decomposition
    test_cases = [
        {
            'name': 'Boss Top Face Edge (Z=70)',
            'description': 'Protrusion top face - should be VISIBLE',
            'p1': gp_Pnt(21.0, 17.0, 70.0),
            'p2': gp_Pnt(56.0, 17.0, 70.0),
            'expected': True
        },
        {
            'name': 'Boss Side Edge (Z=57→70)',
            'description': 'Protrusion vertical edge - should be VISIBLE',
            'p1': gp_Pnt(56.0, 17.0, 57.0),
            'p2': gp_Pnt(56.0, 17.0, 70.0),
            'expected': True
        },
        {
            'name': 'Cut Hole Edge (Z=52→57)',
            'description': 'Internal cut edge - should be HIDDEN',
            'p1': gp_Pnt(30.0, 27.0, 52.0),
            'p2': gp_Pnt(30.0, 27.0, 57.0),
            'expected': False
        },
        {
            'name': 'Cut Bottom Face Edge (Z=39)',
            'description': 'Cut bottom face - should be HIDDEN',
            'p1': gp_Pnt(7.0, 6.0, 39.0),
            'p2': gp_Pnt(43.0, 6.0, 39.0),
            'expected': False
        },
        {
            'name': 'Base Top Face Edge (Z=57)',
            'description': 'Base top face - should be VISIBLE',
            'p1': gp_Pnt(0.0, 0.0, 57.0),
            'p2': gp_Pnt(94.0, 0.0, 57.0),
            'expected': True
        },
        {
            'name': 'Base Bottom Face Edge (Z=0)',
            'description': 'Base bottom face - should be HIDDEN',
            'p1': gp_Pnt(0.0, 0.0, 0.0),
            'p2': gp_Pnt(94.0, 0.0, 0.0),
            'expected': False
        },
        {
            'name': 'Protrusion Internal Edge (Z=34)',
            'description': 'Protrusion extending below base - should be HIDDEN',
            'p1': gp_Pnt(33.0, -6.0, 34.0),
            'p2': gp_Pnt(63.0, -6.0, 34.0),
            'expected': False
        }
    ]

    print("🔍 TESTING HLR CLASSIFICATION:")
    print("-" * 60)
    
    correct_count = 0
    total_count = len(test_cases)
    
    for i, test in enumerate(test_cases, 1):
        print(f"\n{i}. {test['name']}")
        print(f"   {test['description']}")
        print(f"   From: ({test['p1'].X():.1f}, {test['p1'].Y():.1f}, {test['p1'].Z():.1f})")
        print(f"   To:   ({test['p2'].X():.1f}, {test['p2'].Y():.1f}, {test['p2'].Z():.1f})")
        
        # Test the actual HLR function
        result = drawer.is_edge_on_visible_face(test['p1'], test['p2'], 'front')
        expected = test['expected']
        
        if result == expected:
            status = "✅ CORRECT"
            correct_count += 1
        else:
            status = "❌ WRONG"
        
        visibility = "VISIBLE" if result else "HIDDEN"
        expected_visibility = "VISIBLE" if expected else "HIDDEN"
        
        print(f"   Result: {visibility} (expected: {expected_visibility}) {status}")
    
    print("\n" + "="*60)
    print("🎯 FINAL RESULTS:")
    print(f"   ✅ Correct: {correct_count}/{total_count}")
    print(f"   📊 Accuracy: {(correct_count/total_count)*100:.1f}%")
    
    if correct_count == total_count:
        print("   🎉 PERFECT! HLR algorithm is working correctly!")
        print("   🏆 All edge classifications match engineering drawing conventions")
    else:
        print("   ⚠️  Some classifications need review")
    
    print("="*60)
    print("✅ Verification complete for complex model:")
    print("   • 2 protrusions (bosses)")
    print("   • 2 subtractions (cuts)")
    print("   • Multiple visible/hidden edge types tested")

else:
    print("❌ Failed to read STEP file")
