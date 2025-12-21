#!/usr/bin/env python3
"""
Front and Side View HLR Analysis Tool
=====================================

Analyzes the current HLR algorithm for front and side views to identify
specific visibility classification issues.

Current model structure:
- Base cuboid: (0,0,0) to (94,61,57)
- Cut 1: 36.0x17.0x18.0 at (7.0,6.0,39.0) - Z from 39 to 57
- Cut 2: 21.0x22.0x5.0 at (30.0,27.0,52.0) - Z from 52 to 57  
- Boss 3: 30.0x27.0x21.0 at (33.0,-6.0,13.0) - extends below base
- Boss 4: 35.0x10.0x13.0 at (21.0,17.0,57.0) - top at Z=70
"""

import sys
import os
sys.path.append('/Users/sbedi/Nextcloud/Python/Solid/')
from Random_Engineering_Drawings import RandomEngineeringDrawings

def analyze_front_view_issues():
    """Analyze specific front view visibility issues"""
    print("=== FRONT VIEW ANALYSIS ===")
    
    # Create instance for testing
    generator = RandomEngineeringDrawings()
    
    # Test key points that should be visible in front view
    test_points = [
        # Boss 4 top face - should be VISIBLE
        (38.5, 22.0, 70.0, "Boss 4 top face"),
        
        # Boss 4 front-facing edge - should be VISIBLE  
        (38.5, 17.0, 63.5, "Boss 4 front edge"),
        
        # Base top surface (not under boss/cuts) - should be VISIBLE
        (10.0, 40.0, 57.0, "Base top surface"),
        
        # Cut 1 internal edges - should be HIDDEN
        (25.0, 15.0, 48.0, "Cut 1 internal"),
        
        # Cut 2 internal edges - should be HIDDEN
        (40.0, 35.0, 54.5, "Cut 2 internal"),
        
        # Boss 3 below base level - should be HIDDEN
        (50.0, -3.0, 20.0, "Boss 3 below base"),
        
        # Base back face - should be HIDDEN
        (47.0, 61.0, 28.5, "Base back face"),
        
        # Bottom face - should be HIDDEN
        (47.0, 30.5, 0.0, "Bottom face")
    ]
    
    print("Front view visibility test results:")
    for x, y, z, description in test_points:
        result = generator.geometric_visibility_test((x, y, z), 'front')
        status = "VISIBLE" if result else "HIDDEN"
        print(f"  {description:20} at ({x:5.1f}, {y:5.1f}, {z:5.1f}): {status}")
    
    print()

def analyze_side_view_issues():
    """Analyze specific side view visibility issues"""
    print("=== SIDE VIEW ANALYSIS ===")
    
    # Create instance for testing
    generator = RandomEngineeringDrawings()
    
    # Test key points that should be visible in side view
    test_points = [
        # Boss 4 side face (facing viewer) - should be VISIBLE
        (56.0, 22.0, 63.5, "Boss 4 side face"),
        
        # Boss 3 above base level - should be VISIBLE
        (50.0, -3.0, 25.0, "Boss 3 above base"),
        
        # Base right side face - should be VISIBLE
        (94.0, 30.5, 28.5, "Base right side"),
        
        # Cut edges visible from side - should be VISIBLE
        (25.0, 15.0, 48.0, "Cut 1 from side"),
        (40.0, 35.0, 54.5, "Cut 2 from side"),
        
        # Far side face (X=0) - should be HIDDEN
        (0.0, 30.5, 28.5, "Base left side (far)"),
        
        # Bottom face - should be HIDDEN
        (47.0, 30.5, 0.0, "Bottom face"),
        
        # Boss 3 below ground - should be HIDDEN
        (50.0, -3.0, 5.0, "Boss 3 below ground")
    ]
    
    print("Side view visibility test results:")
    for x, y, z, description in test_points:
        result = generator.geometric_visibility_test((x, y, z), 'side')
        status = "VISIBLE" if result else "HIDDEN"
        print(f"  {description:20} at ({x:5.1f}, {y:5.1f}, {z:5.1f}): {status}")
    
    print()

def compare_with_expected():
    """Compare actual results with expected engineering drawing conventions"""
    print("=== EXPECTED vs ACTUAL COMPARISON ===")
    
    generator = RandomEngineeringDrawings()
    
    # Critical test cases with expected results
    critical_tests = [
        # (x, y, z, view, expected_result, description)
        (38.5, 22.0, 70.0, 'front', True, "Boss 4 top - should be visible"),
        (38.5, 17.0, 63.5, 'front', True, "Boss 4 front edge - should be visible"),  
        (25.0, 15.0, 48.0, 'front', False, "Cut 1 internal - should be hidden"),
        (47.0, 61.0, 28.5, 'front', False, "Back face - should be hidden"),
        
        (56.0, 22.0, 63.5, 'side', True, "Boss 4 side - should be visible"),
        (50.0, -3.0, 25.0, 'side', True, "Boss 3 above base - should be visible"),
        (0.0, 30.5, 28.5, 'side', False, "Far side - should be hidden"),
        (47.0, 30.5, 0.0, 'side', False, "Bottom - should be hidden")
    ]
    
    errors = []
    
    for x, y, z, view, expected, description in critical_tests:
        actual = generator.geometric_visibility_test((x, y, z), view)
        
        if actual != expected:
            errors.append({
                'point': (x, y, z),
                'view': view,
                'expected': expected,
                'actual': actual,
                'description': description
            })
            print(f"❌ ERROR: {description}")
            print(f"   Point: ({x}, {y}, {z}) in {view} view")
            print(f"   Expected: {'VISIBLE' if expected else 'HIDDEN'}")
            print(f"   Actual: {'VISIBLE' if actual else 'HIDDEN'}")
            print()
        else:
            print(f"✅ CORRECT: {description}")
    
    if errors:
        print(f"\n🚨 Found {len(errors)} visibility errors!")
        print("\nSummary of issues:")
        for error in errors:
            view = error['view']
            exp = 'VISIBLE' if error['expected'] else 'HIDDEN'
            act = 'VISIBLE' if error['actual'] else 'HIDDEN'
            print(f"  {view.upper()} VIEW: {error['description']} (Expected {exp}, Got {act})")
    else:
        print("\n🎉 All visibility tests passed!")
    
    return errors

def main():
    """Main analysis function"""
    print("Front and Side View HLR Analysis")
    print("=" * 50)
    
    analyze_front_view_issues()
    analyze_side_view_issues()
    errors = compare_with_expected()
    
    if errors:
        print("\n" + "=" * 50)
        print("RECOMMENDATIONS FOR FIXES:")
        print("=" * 50)
        
        front_errors = [e for e in errors if e['view'] == 'front']
        side_errors = [e for e in errors if e['view'] == 'side']
        
        if front_errors:
            print("\nFRONT VIEW FIXES NEEDED:")
            for error in front_errors:
                print(f"  - {error['description']}")
        
        if side_errors:
            print("\nSIDE VIEW FIXES NEEDED:")
            for error in side_errors:
                print(f"  - {error['description']}")

if __name__ == "__main__":
    main()
