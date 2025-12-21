#!/usr/bin/env python3
"""
Test for polygon crosses with the new orientation-based vertex extraction.

This test checks if the new vertex ordering eliminates the crosses
that were appearing in faces 1, 2, and 5.
"""

import sys
import numpy as np
sys.path.append('/Users/sbedi/Nextcloud/Python/Solid')
sys.path.append('/Users/sbedi/Nextcloud/Python/Solid/OpenCASCADE_Engineering_Drawings/src')

try:
    from engineering_drawings.vertex_extractor import VertexExtractor
    from V5_current import create_opencascade_solid, OPENCASCADE_AVAILABLE
    
    if OPENCASCADE_AVAILABLE:
        from OCC.Core.TopExp import TopExp_Explorer
        from OCC.Core.TopAbs import TopAbs_FACE
except ImportError as e:
    print(f"Import error: {e}")
    OPENCASCADE_AVAILABLE = False


def check_polygon_crosses(vertices):
    """Check if a polygon has self-intersections (crosses)."""
    if len(vertices) < 4:
        return False, []
    
    # Convert to 2D for intersection testing - use XY projection
    vertices_2d = [(v[0], v[1]) for v in vertices]
    
    crosses = []
    n = len(vertices_2d)
    
    for i in range(n):
        edge1_start = vertices_2d[i]
        edge1_end = vertices_2d[(i + 1) % n]
        
        for j in range(i + 2, n):
            # Skip adjacent edges and the last-to-first edge
            if j == (i + n - 1) % n:
                continue
                
            edge2_start = vertices_2d[j]
            edge2_end = vertices_2d[(j + 1) % n]
            
            if edges_intersect_2d(edge1_start, edge1_end, edge2_start, edge2_end):
                crosses.append((i, (i+1) % n, j, (j+1) % n))
    
    return len(crosses) > 0, crosses


def edges_intersect_2d(p1, q1, p2, q2):
    """Check if two 2D line segments intersect."""
    def orientation(p, q, r):
        val = (q[1] - p[1]) * (r[0] - q[0]) - (q[0] - p[0]) * (r[1] - q[1])
        if abs(val) < 1e-10:
            return 0  # Collinear
        return 1 if val > 0 else 2  # Clockwise or Counterclockwise
    
    def on_segment(p, q, r):
        return (q[0] <= max(p[0], r[0]) and q[0] >= min(p[0], r[0]) and
                q[1] <= max(p[1], r[1]) and q[1] >= min(p[1], r[1]))
    
    o1 = orientation(p1, q1, p2)
    o2 = orientation(p1, q1, q2)
    o3 = orientation(p2, q2, p1)
    o4 = orientation(p2, q2, q1)
    
    # General case - segments intersect if orientations are different
    if o1 != o2 and o3 != o4:
        return True
    
    # Special cases - one segment's endpoint lies on the other
    if (o1 == 0 and on_segment(p1, p2, q1) or
        o2 == 0 and on_segment(p1, q2, q1) or
        o3 == 0 and on_segment(p2, p1, q2) or
        o4 == 0 and on_segment(p2, q1, q2)):
        return True
    
    return False


def test_cross_elimination():
    """Test if the new orientation logic eliminates polygon crosses."""
    
    print("="*70)
    print("TESTING POLYGON CROSS ELIMINATION")
    print("="*70)
    
    if not OPENCASCADE_AVAILABLE:
        print("✗ OpenCASCADE not available - cannot run test")
        return
    
    # Create the test solid
    print("Creating OpenCASCADE solid...")
    solid = create_opencascade_solid()
    
    if solid is None:
        print("✗ Failed to create solid")
        return
    
    # Initialize the enhanced vertex extractor
    try:
        extractor = VertexExtractor()
        print("✓ VertexExtractor initialized successfully")
    except Exception as e:
        print(f"✗ Failed to initialize VertexExtractor: {e}")
        return
    
    # Test the problematic faces (1, 2, 5)
    target_faces = [1, 2, 5]
    
    face_explorer = TopExp_Explorer(solid, TopAbs_FACE)
    face_count = 0
    
    print(f"\nTesting cross elimination for faces {target_faces}...")
    
    cross_results = {}
    
    while face_explorer.More() and face_count < 6:
        face = face_explorer.Current()
        face_count += 1
        
        if face_count in target_faces:
            print(f"\n{'-'*50}")
            print(f"FACE {face_count} CROSS ANALYSIS")
            print(f"{'-'*50}")
            
            try:
                vertices = extractor.extract_face_vertices(face)
                
                if vertices and len(vertices) >= 3:
                    print(f"Extracted {len(vertices)} vertices")
                    
                    # Remove duplicates for cross checking
                    unique_vertices = []
                    seen = set()
                    for v in vertices:
                        v_tuple = tuple(np.round(v, 6))
                        if v_tuple not in seen:
                            unique_vertices.append(v)
                            seen.add(v_tuple)
                    
                    print(f"Unique vertices: {len(unique_vertices)}")
                    
                    # Check for crosses
                    has_crosses, crosses = check_polygon_crosses(unique_vertices)
                    
                    cross_results[face_count] = {
                        'vertex_count': len(unique_vertices),
                        'has_crosses': has_crosses,
                        'cross_count': len(crosses),
                        'crosses': crosses
                    }
                    
                    if has_crosses:
                        print(f"✗ CROSSES DETECTED: {len(crosses)} intersections")
                        for cross in crosses:
                            print(f"    Edge {cross[0]}-{cross[1]} × "
                                  f"Edge {cross[2]}-{cross[3]}")
                    else:
                        print("✓ NO CROSSES - Clean polygon!")
                    
                    # Display polygon
                    vertex_display = " → ".join([
                        f"({v[0]:.1f},{v[1]:.1f},{v[2]:.1f})" 
                        for v in unique_vertices
                    ])
                    print(f"Polygon: {vertex_display}")
                    
                else:
                    print("✗ Failed to extract vertices")
                    cross_results[face_count] = {'error': 'extraction_failed'}
                    
            except Exception as e:
                print(f"✗ Exception: {e}")
                cross_results[face_count] = {'error': str(e)}
        
        face_explorer.Next()
    
    # Summary
    print(f"\n{'='*70}")
    print("CROSS ELIMINATION TEST RESULTS")
    print("="*70)
    
    clean_faces = 0
    cross_faces = 0
    
    for face_id in target_faces:
        if face_id in cross_results:
            result = cross_results[face_id]
            if 'error' in result:
                print(f"✗ Face {face_id}: Error - {result['error']}")
            elif result['has_crosses']:
                print(f"✗ Face {face_id}: {result['cross_count']} crosses")
                cross_faces += 1
            else:
                print(f"✓ Face {face_id}: Clean polygon (no crosses)")
                clean_faces += 1
    
    print(f"\nSummary:")
    print(f"  Clean faces (no crosses): {clean_faces}")
    print(f"  Faces with crosses: {cross_faces}")
    print(f"  Total tested: {clean_faces + cross_faces}")
    
    if cross_faces == 0:
        print("🎉 SUCCESS: All faces are clean! No crosses detected!")
        print("   The new orientation logic has eliminated the polygon crosses!")
    else:
        print("⚠️  Some faces still have crosses - may need further refinement")
    
    print("="*70)


if __name__ == "__main__":
    test_cross_elimination()
