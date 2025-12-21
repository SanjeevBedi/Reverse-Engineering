#!/usr/bin/env python3
"""
Test to examine actual edge orientations from OpenCASCADE solid.

This will help us understand if all edges really are FORWARD oriented
and where the crossing issue actually originates.
"""

import sys
import os
sys.path.append('/Users/sbedi/Nextcloud/Python/Solid')

# Import the main V5 functions
from V5_current import (
    create_opencascade_solid, 
    extract_faces_from_solid,
    OPENCASCADE_AVAILABLE,
    TOPEXP_AVAILABLE
)

if OPENCASCADE_AVAILABLE:
    from OCC.Core.TopExp import TopExp_Explorer
    from OCC.Core.TopAbs import TopAbs_FACE, TopAbs_WIRE, TopAbs_EDGE, TopAbs_VERTEX
    from OCC.Core.TopAbs import TopAbs_FORWARD, TopAbs_REVERSED
    from OCC.Core.BRep import BRep_Tool
    from OCC.Core.TopoDS import topods
    
    if TOPEXP_AVAILABLE:
        from OCC.Core.TopExp import topexp


def examine_actual_edge_orientations():
    """Examine actual edge orientations from OpenCASCADE solid."""
    
    print("="*70)
    print("EXAMINING ACTUAL EDGE ORIENTATIONS FROM OPENCASCADE SOLID")
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
    
    print("✓ Solid created successfully")
    
    # Focus on the first few faces to examine edge orientations
    print("\nExamining edge orientations for first 6 faces...")
    
    face_explorer = TopExp_Explorer(solid, TopAbs_FACE)
    face_count = 0
    
    while face_explorer.More() and face_count < 6:
        face = face_explorer.Current()
        face_count += 1
        
        print(f"\n{'-'*50}")
        print(f"FACE {face_count}:")
        print(f"{'-'*50}")
        
        # Get face orientation
        face_orientation = face.Orientation()
        face_ori_str = "FORWARD" if face_orientation == TopAbs_FORWARD else "REVERSED"
        print(f"Face orientation: {face_ori_str}")
        
        # Examine wires in this face
        wire_explorer = TopExp_Explorer(face, TopAbs_WIRE)
        wire_count = 0
        
        while wire_explorer.More():
            wire = wire_explorer.Current()
            wire_count += 1
            
            # Get wire orientation
            wire_orientation = wire.Orientation()
            wire_ori_str = "FORWARD" if wire_orientation == TopAbs_FORWARD else "REVERSED"
            print(f"  Wire {wire_count} orientation: {wire_ori_str}")
            
            # Examine edges in this wire
            edge_explorer = TopExp_Explorer(wire, TopAbs_EDGE)
            edge_count = 0
            edge_orientations = []
            
            print(f"  Edges in Wire {wire_count}:")
            
            while edge_explorer.More():
                edge = edge_explorer.Current()
                edge_count += 1
                
                # Get edge orientation
                edge_orientation = edge.Orientation()
                edge_ori_str = "FORWARD" if edge_orientation == TopAbs_FORWARD else "REVERSED"
                edge_orientations.append(edge_ori_str)
                
                # Get edge vertices if possible
                if TOPEXP_AVAILABLE:
                    try:
                        first_vertex = topods.Vertex()
                        last_vertex = topods.Vertex()
                        topexp.Vertices(edge, first_vertex, last_vertex)
                        
                        # Get coordinates
                        first_pnt = BRep_Tool.Pnt(first_vertex)
                        last_pnt = BRep_Tool.Pnt(last_vertex)
                        
                        first_coord = (first_pnt.X(), first_pnt.Y(), first_pnt.Z())
                        last_coord = (last_pnt.X(), last_pnt.Y(), last_pnt.Z())
                        
                        print(f"    Edge {edge_count}: {edge_ori_str}")
                        print(f"      First vertex: ({first_coord[0]:.1f}, {first_coord[1]:.1f}, {first_coord[2]:.1f})")
                        print(f"      Last vertex:  ({last_coord[0]:.1f}, {last_coord[1]:.1f}, {last_coord[2]:.1f})")
                        
                    except Exception as e:
                        print(f"    Edge {edge_count}: {edge_ori_str} (vertex extraction failed: {e})")
                else:
                    print(f"    Edge {edge_count}: {edge_ori_str} (TopExp not available)")
                
                edge_explorer.Next()
            
            # Summary for this wire
            forward_edges = edge_orientations.count("FORWARD")
            reversed_edges = edge_orientations.count("REVERSED")
            
            print(f"  Wire {wire_count} summary:")
            print(f"    Total edges: {edge_count}")
            print(f"    FORWARD edges: {forward_edges}")
            print(f"    REVERSED edges: {reversed_edges}")
            
            if forward_edges == edge_count:
                print(f"    ✓ ALL EDGES ARE FORWARD - This confirms your observation!")
            elif reversed_edges == edge_count:
                print(f"    ⚠️  ALL EDGES ARE REVERSED")
            else:
                print(f"    ℹ️  Mixed edge orientations")
            
            wire_explorer.Next()
        
        face_explorer.Next()
    
    print(f"\n{'='*70}")
    print("EDGE ORIENTATION EXAMINATION COMPLETE")
    print("="*70)
    
    # If all edges are FORWARD, the issue is elsewhere
    print("\nANALYSIS:")
    print("If all edges are FORWARD oriented, then:")
    print("1. The crossing issue is NOT due to edge orientation logic")
    print("2. The problem may be in:")
    print("   - Incorrect edge vertex definitions")
    print("   - Wrong wire traversal order")
    print("   - Face normal orientation affecting vertex order")
    print("   - Fundamental topology structure")
    print("\nNext step: Examine the actual vertex coordinates and edge connectivity")


def examine_face_topology_details():
    """Examine the detailed topology of problematic faces."""
    
    print("\n" + "="*70)
    print("DETAILED FACE TOPOLOGY ANALYSIS")
    print("="*70)
    
    if not OPENCASCADE_AVAILABLE:
        return
    
    # Create solid and focus on Face 1, 2, and 5 (the problematic ones)
    solid = create_opencascade_solid()
    if solid is None:
        return
    
    face_explorer = TopExp_Explorer(solid, TopAbs_FACE)
    face_count = 0
    target_faces = [1, 2, 5]  # The faces showing crosses
    
    while face_explorer.More() and face_count < 6:
        face = face_explorer.Current()
        face_count += 1
        
        if face_count in target_faces:
            print(f"\n{'-'*60}")
            print(f"DETAILED ANALYSIS - FACE {face_count} (PROBLEMATIC)")
            print(f"{'-'*60}")
            
            # Get all vertices in order and check the resulting polygon
            wire_explorer = TopExp_Explorer(face, TopAbs_WIRE)
            if wire_explorer.More():
                wire = wire_explorer.Current()
                
                # Extract vertices in sequence
                vertices = []
                edge_explorer = TopExp_Explorer(wire, TopAbs_EDGE)
                edge_num = 0
                
                print("Edge-by-edge traversal:")
                while edge_explorer.More():
                    edge = edge_explorer.Current()
                    edge_num += 1
                    
                    if TOPEXP_AVAILABLE:
                        try:
                            first_vertex = topods.Vertex()
                            last_vertex = topods.Vertex()
                            topexp.Vertices(edge, first_vertex, last_vertex)
                            
                            first_pnt = BRep_Tool.Pnt(first_vertex)
                            last_pnt = BRep_Tool.Pnt(last_vertex)
                            
                            first_coord = [first_pnt.X(), first_pnt.Y(), first_pnt.Z()]
                            last_coord = [last_pnt.X(), last_pnt.Y(), last_pnt.Z()]
                            
                            print(f"  Edge {edge_num}: {first_coord} → {last_coord}")
                            
                            # For first edge, add both vertices
                            if edge_num == 1:
                                vertices.append(first_coord)
                                vertices.append(last_coord)
                            else:
                                vertices.append(last_coord)
                                
                        except Exception as e:
                            print(f"  Edge {edge_num}: Error extracting vertices - {e}")
                    
                    edge_explorer.Next()
                
                # Check for crosses in the resulting polygon
                if len(vertices) >= 4:
                    print(f"\nResulting vertex sequence:")
                    for i, v in enumerate(vertices):
                        print(f"  V{i}: ({v[0]:.1f}, {v[1]:.1f}, {v[2]:.1f})")
                    
                    # Check for crosses (simplified 2D cross check)
                    has_crosses = check_polygon_crosses(vertices)
                    
                    if has_crosses:
                        print(f"  ✗ CROSSES DETECTED in Face {face_count}")
                        print(f"  This confirms the issue exists in the edge definitions")
                    else:
                        print(f"  ✓ No crosses detected in Face {face_count}")
        
        face_explorer.Next()


def check_polygon_crosses(vertices):
    """Check if a 3D polygon has self-intersections when projected to 2D."""
    if len(vertices) < 4:
        return False
    
    # Project to XY plane for simplicity (can be made more sophisticated)
    vertices_2d = [(v[0], v[1]) for v in vertices]
    
    # Check edge intersections
    n = len(vertices_2d)
    
    for i in range(n):
        edge1_start = vertices_2d[i]
        edge1_end = vertices_2d[(i + 1) % n]
        
        for j in range(i + 2, n):
            # Skip adjacent edges
            if j == (i + n - 1) % n:
                continue
                
            edge2_start = vertices_2d[j]
            edge2_end = vertices_2d[(j + 1) % n]
            
            if edges_intersect_2d(edge1_start, edge1_end, edge2_start, edge2_end):
                print(f"    Cross: Edge {i}-{(i+1)%n} intersects Edge {j}-{(j+1)%n}")
                return True
    
    return False


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
    examine_actual_edge_orientations()
    examine_face_topology_details()
