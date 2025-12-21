#!/usr/bin/env python3
"""
Side View HLR Diagnostic Tool
=============================

Specifically analyzes the right side view HLR algorithm to identify
why hidden lines are not being classified correctly.
"""

import sys
import os
sys.path.append('/Users/sbedi/Nextcloud/Python/Solid/')
from Random_Engineering_Drawings import RandomEngineeringDrawings

def analyze_side_view_hlr():
    """Analyze side view HLR classification in detail"""
    print("Side View HLR Diagnostic Analysis")
    print("=" * 50)
    
    # Create a test model
    generator = RandomEngineeringDrawings()
    generator.create_random_model()
    
    print(f"Model: {generator.base_length} x {generator.base_width} x {generator.base_height}")
    print("Features:")
    for i, feature in enumerate(generator.features):
        op = feature['operation']
        pos = feature['position']
        dims = feature['dimensions']
        print(f"  {i+1}. {op}: {dims[0]}x{dims[1]}x{dims[2]} at ({pos[0]},{pos[1]},{pos[2]})")
    
    # Extract edges
    generator.extract_edges_simple()
    
    # Analyze side view specifically
    side_visible = len(generator.visible_edges['side'])
    side_hidden = len(generator.hidden_edges['side'])
    side_total = side_visible + side_hidden
    
    print(f"\nSide view edge analysis:")
    print(f"  Visible: {side_visible}")
    print(f"  Hidden:  {side_hidden}")
    print(f"  Total:   {side_total}")
    print(f"  Visible ratio: {side_visible/side_total*100:.1f}%")
    
    if side_visible/side_total > 0.8:
        print("⚠️  WARNING: Too many visible edges! Hidden lines likely missing.")
    
    # Test specific points that should be hidden in side view
    test_hidden_points = [
        # Far left face (X=0) - should be hidden
        (0, generator.base_width/2, generator.base_height/2, "Far left face"),
        
        # Internal cut edges - should be hidden if not facing viewer
        (generator.base_length/4, generator.base_width/2, generator.base_height/2, "Internal region"),
        
        # Back internal edges - should be hidden
        (generator.base_length/2, generator.base_width*0.9, generator.base_height/2, "Back internal"),
        
        # Bottom internal points - should be hidden  
        (generator.base_length/2, generator.base_width/2, 5, "Bottom internal"),
    ]
    
    print(f"\nTesting points that should be HIDDEN in side view:")
    hidden_errors = 0
    
    for x, y, z, description in test_hidden_points:
        result = generator.geometric_visibility_test((x, y, z), 'side')
        status = "VISIBLE" if result else "HIDDEN"
        
        if result:  # Should be hidden but showing as visible
            print(f"  ❌ {description:20} at ({x:5.1f},{y:5.1f},{z:5.1f}): {status} (SHOULD BE HIDDEN)")
            hidden_errors += 1
        else:
            print(f"  ✅ {description:20} at ({x:5.1f},{y:5.1f},{z:5.1f}): {status}")
    
    # Test specific points that should be visible in side view
    test_visible_points = [
        # Right face (X=max) - should be visible
        (generator.base_length, generator.base_width/2, generator.base_height/2, "Right face"),
        
        # Top edges - should be visible
        (generator.base_length/2, generator.base_width/2, generator.base_height, "Top edge"),
        
        # Bottom outline - should be visible
        (generator.base_length/2, generator.base_width/2, 0, "Bottom outline"),
        
        # Front edge - should be visible
        (generator.base_length/2, 0, generator.base_height/2, "Front edge"),
    ]
    
    print(f"\nTesting points that should be VISIBLE in side view:")
    visible_errors = 0
    
    for x, y, z, description in test_visible_points:
        result = generator.geometric_visibility_test((x, y, z), 'side')
        status = "VISIBLE" if result else "HIDDEN"
        
        if not result:  # Should be visible but showing as hidden
            print(f"  ❌ {description:20} at ({x:5.1f},{y:5.1f},{z:5.1f}): {status} (SHOULD BE VISIBLE)")
            visible_errors += 1
        else:
            print(f"  ✅ {description:20} at ({x:5.1f},{y:5.1f},{z:5.1f}): {status}")
    
    print(f"\n=== DIAGNOSTIC SUMMARY ===")
    print(f"Hidden classification errors: {hidden_errors}")
    print(f"Visible classification errors: {visible_errors}")
    
    if hidden_errors > 0:
        print("🚨 ISSUE: Side view is showing edges as visible that should be hidden!")
        print("   This explains why hidden lines are not appearing correctly.")
        return False
    elif visible_errors > 0:
        print("🚨 ISSUE: Side view is hiding edges that should be visible!")
        return False
    else:
        print("✅ Side view HLR classification appears correct.")
        return True

def recommend_fixes():
    """Recommend fixes for side view HLR issues"""
    print("\n" + "=" * 50)
    print("RECOMMENDED FIXES FOR SIDE VIEW HLR")
    print("=" * 50)
    
    print("""
The side view HLR algorithm needs to be more conservative about
what edges are classified as visible. Key issues to address:

1. INTERNAL EDGES: Edges inside cuts/holes should be hidden unless
   they face the viewer directly.

2. FAR SIDE FACES: The left face (X=0) should always be hidden
   when viewing from the right side.

3. DEPTH TESTING: Need better depth analysis to hide edges that
   are behind other geometry.

4. CUT VISIBILITY: Internal cut edges should only be visible if
   the cut opening faces the viewing direction.

Suggested algorithm improvements:
- Make the default fallback more conservative (hidden unless proven visible)
- Add better geometric analysis for internal vs. external edges
- Implement proper depth sorting for overlapping features
- Add specific rules for cut and boss visibility from side view
""")

def main():
    """Main diagnostic function"""
    result = analyze_side_view_hlr()
    
    if not result:
        recommend_fixes()

if __name__ == "__main__":
    main()
