#!/usr/bin/env python3
"""
Test the enhanced vertex extractor with real OpenCASCADE data.

This test will validate that the orientation-based logic correctly
handles the mixed edge orientations found in faces 1, 2, and 5.
"""

import sys
import os
sys.path.append('/Users/sbedi/Nextcloud/Python/Solid')
sys.path.append('/Users/sbedi/Nextcloud/Python/Solid/OpenCASCADE_Engineering_Drawings/src')

from engineering_drawings.vertex_extractor import VertexExtractor
from V5_current import create_opencascade_solid, OPENCASCADE_AVAILABLE

if OPENCASCADE_AVAILABLE:
    from OCC.Core.TopExp import TopExp_Explorer
    from OCC.Core.TopAbs import TopAbs_FACE, TopAbs_WIRE
    from OCC.Core.TopoDS import topods


def test_enhanced_vertex_extractor():
    """Test the enhanced vertex extractor with real OpenCASCADE data."""
    
    print("="*70)
    print("TESTING ENHANCED VERTEX EXTRACTOR WITH REAL DATA")
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
    extractor = VertexExtractor()
    
    # Test the problematic faces (1, 2, 5)
    target_faces = [1, 2, 5]
    
    face_explorer = TopExp_Explorer(solid, TopAbs_FACE)
    face_count = 0
    
    print(f"\nTesting faces {target_faces} with enhanced vertex extractor...")
    
    while face_explorer.More() and face_count < 6:
        face = face_explorer.Current()
        face_count += 1
        
        if face_count in target_faces:
            print(f"\n{'-'*60}")
            print(f"TESTING FACE {face_count} (PREVIOUSLY PROBLEMATIC)")
            print(f"{'-'*60}")
            
            # Use the enhanced vertex extractor
            try:
                vertices = extractor.extract_face_vertices(face)
                
                if vertices and len(vertices) >= 3:
                    print(f"✓ Enhanced extractor: {len(vertices)} vertices")
                    
                    # Check for crosses
                    has_crosses = check_for_polygon_crosses(vertices)
                    
                    if not has_crosses:
                        print(f"✓ Face {face_count}: NO CROSSES DETECTED!")
                        print(f"  Enhanced orientation logic SUCCESSFUL")
                    else:
                        print(f"✗ Face {face_count}: Still has crosses")
                        print(f"  May need further orientation refinement")
                    
                    # Display the resulting polygon
                    vertex_coords = " → ".join([
                        f"({v[0]:.1f},{v[1]:.1f},{v[2]:.1f})" 
                        for v in vertices
                    ])
                    print(f"  Final polygon: {vertex_coords}")
                    
                else:
                    print(f"✗ Face {face_count}: Failed to extract vertices")
                    
            except Exception as e:
                print(f"✗ Face {face_count}: Exception - {e}")
        
        face_explorer.Next()
    
    
    print(f"\n{'='*70}")
    print("ENHANCED VERTEX EXTRACTOR TEST COMPLETE")
    print("="*70)


def check_for_polygon_crosses(vertices):
    """Check if a 3D polygon has self-intersections."""
    if len(vertices) < 4:
        return False
    
    # Project to appropriate 2D plane based on vertex distribution
    # For simplicity, use XY projection first
    vertices_2d = [(v[0], v[1]) for v in vertices]
    
    # Check for zero-area projection (all points on a line)
    # If so, try XZ projection
    if is_coplanar_xy(vertices_2d):
        vertices_2d = [(v[0], v[2]) for v in vertices]
        if is_coplanar_xy(vertices_2d):
            # Try YZ projection
            vertices_2d = [(v[1], v[2]) for v in vertices]
    
    # Check edge intersections
    n = len(vertices_2d)
    crosses_found = 0
    
    for i in range(n):
        edge1_start = vertices_2d[i]
        edge1_end = vertices_2d[(i + 1) % n]
        
        for j in range(i + 2, n):
            # Skip adjacent edges and wrap-around
            if j == (i + n - 1) % n:
                continue
                
            edge2_start = vertices_2d[j]
            edge2_end = vertices_2d[(j + 1) % n]
            
            if edges_intersect_2d(edge1_start, edge1_end, edge2_start, edge2_end):
                print(f"    Cross: Edge {i}-{(i+1)%n} × Edge {j}-{(j+1)%n}")
                crosses_found += 1
    
    return crosses_found > 0


def is_coplanar_xy(vertices_2d):
    """Check if all vertices lie on a line in 2D (degenerate case)."""
    if len(vertices_2d) < 3:
        return True
    
    # Check if all points are collinear
    for i in range(2, len(vertices_2d)):
        p1, p2, p3 = vertices_2d[0], vertices_2d[1], vertices_2d[i]
        
        # Calculate cross product (area of triangle)
        cross = (p2[0] - p1[0]) * (p3[1] - p1[1]) - (p2[1] - p1[1]) * (p3[0] - p1[0])
        
        if abs(cross) > 1e-6:  # Non-zero area, not collinear
            return False
    
    return True  # All points are collinear


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
    
    # General case
    if o1 != o2 and o3 != o4:
        return True
    
    # Special cases
    if (o1 == 0 and on_segment(p1, p2, q1) or
        o2 == 0 and on_segment(p1, q2, q1) or
        o3 == 0 and on_segment(p2, p1, q2) or
        o4 == 0 and on_segment(p2, q1, q2)):
        return True
    
    return False


if __name__ == "__main__":
    test_enhanced_vertex_extractor()