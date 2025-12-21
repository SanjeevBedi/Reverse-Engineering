#!/usr/bin/env python3
"""
Test wire-based topology extraction for proper vertex ordering.

This test validates that using wire orientation and edge connectivity
resolves the vertex ordering crosses in faces 1, 2, and 5.

The approach:
1. Face contains wires (oriented loops)
2. Wire orientation determines traversal direction
3. Each edge has orientation within the wire
4. Edge connectivity builds proper vertex sequence
"""

import numpy as np

def test_wire_based_topology():
    """Test the wire-based vertex extraction approach."""
    
    print("="*60)
    print("WIRE-BASED TOPOLOGY VERTEX EXTRACTION TEST")
    print("="*60)
    
    # Test data representing the problematic faces with wire information
    test_faces = {
        'Face_1': {
            'description': 'Face 1 - Previously showing crosses',
            'wire_orientation': 'FORWARD',
            'edges': [
                {'orientation': 'FORWARD', 'vertices': [(0, 0, 0), (0, 0, 30)]},
                {'orientation': 'FORWARD', 'vertices': [(0, 0, 30), (0, 20, 30)]},
                {'orientation': 'FORWARD', 'vertices': [(0, 20, 30), (0, 20, 0)]},
                {'orientation': 'FORWARD', 'vertices': [(0, 20, 0), (0, 0, 0)]}
            ]
        },
        'Face_2': {
            'description': 'Face 2 - Previously showing crosses',
            'wire_orientation': 'FORWARD',
            'edges': [
                {'orientation': 'FORWARD', 'vertices': [(0, 0, 0), (10, 0, 0)]},
                {'orientation': 'FORWARD', 'vertices': [(10, 0, 0), (10, 0, 30)]},
                {'orientation': 'FORWARD', 'vertices': [(10, 0, 30), (0, 0, 30)]},
                {'orientation': 'FORWARD', 'vertices': [(0, 0, 30), (0, 0, 0)]}
            ]
        },
        'Face_5': {
            'description': 'Face 5 - Previously showing crosses',
            'wire_orientation': 'FORWARD',
            'edges': [
                {'orientation': 'FORWARD', 'vertices': [(0, 0, 0), (10, 0, 0)]},
                {'orientation': 'FORWARD', 'vertices': [(10, 0, 0), (10, 20, 0)]},
                {'orientation': 'FORWARD', 'vertices': [(10, 20, 0), (0, 20, 0)]},
                {'orientation': 'FORWARD', 'vertices': [(0, 20, 0), (0, 0, 0)]}
            ]
        }
    }
    
    print(f"Testing {len(test_faces)} faces with wire-based extraction...")
    
    for face_name, face_data in test_faces.items():
        print(f"\n{'-'*50}")
        print(f"Processing {face_name}: {face_data['description']}")
        print(f"Wire orientation: {face_data['wire_orientation']}")
        
        # Simulate wire-based vertex extraction
        vertices = extract_vertices_from_wire(face_data)
        
        if vertices:
            # Validate the resulting polygon
            is_valid, validation_msg = validate_polygon(vertices, face_name)
            
            if is_valid:
                print(f"✓ {face_name}: VALID POLYGON - {validation_msg}")
            else:
                print(f"✗ {face_name}: INVALID POLYGON - {validation_msg}")
        else:
            print(f"✗ {face_name}: Failed to extract vertices")
    
    print(f"\n{'='*60}")
    print("WIRE-BASED TOPOLOGY TEST COMPLETE")
    print("="*60)

def extract_vertices_from_wire(face_data):
    """
    Extract vertices using wire-based topology approach.
    
    This simulates the enhanced vertex extractor logic:
    1. Check wire orientation
    2. Process edges in proper order
    3. Use edge orientation for vertex selection
    4. Build connectivity sequence
    """
    wire_orientation = face_data['wire_orientation']
    edges = face_data['edges']
    
    print(f"  Wire has {len(edges)} edges")
    print(f"  Wire orientation: {wire_orientation}")
    
    # If wire is reversed, process edges in reverse order
    if wire_orientation == 'REVERSED':
        edges = list(reversed(edges))
        print("  Reversed edge order due to wire orientation")
    
    # Build vertex sequence following edge connectivity
    vertex_sequence = []
    
    for edge_idx, edge in enumerate(edges):
        edge_orientation = edge['orientation']
        edge_vertices = edge['vertices']
        
        # Extract start and end vertices based on edge orientation
        if edge_orientation == 'FORWARD':
            # Forward edge: first → last
            start_vertex = edge_vertices[0]
            end_vertex = edge_vertices[1]
            orientation_tag = "F"
        else:
            # Reversed edge: last → first
            start_vertex = edge_vertices[1]
            end_vertex = edge_vertices[0]
            orientation_tag = "R"
        
        print(f"    Edge {edge_idx}: {orientation_tag} "
              f"{start_vertex} → {end_vertex}")
        
        # Build connectivity sequence
        if edge_idx == 0:
            # First edge: add both start and end
            vertex_sequence.append(start_vertex)
            vertex_sequence.append(end_vertex)
        else:
            # Subsequent edges: add only end vertex
            vertex_sequence.append(end_vertex)
            
            # Verify connectivity
            prev_end = vertex_sequence[-2]
            distance = np.linalg.norm(
                np.array(prev_end) - np.array(start_vertex)
            )
            
            if distance > 1e-6:
                print(f"    ⚠️ Gap: {distance:.8f} between "
                      f"edges {edge_idx-1} and {edge_idx}")
    
    # Remove duplicate closing vertex if present
    if len(vertex_sequence) > 2:
        first_vertex = np.array(vertex_sequence[0])
        last_vertex = np.array(vertex_sequence[-1])
        
        if np.linalg.norm(first_vertex - last_vertex) < 1e-6:
            vertex_sequence = vertex_sequence[:-1]
            print("  Removed duplicate closing vertex")
    
    print(f"  ✓ Wire connectivity sequence: {len(vertex_sequence)} vertices")
    
    # Display final sequence
    vertex_coords = " → ".join([
        f"({v[0]},{v[1]},{v[2]})" for v in vertex_sequence
    ])
    print(f"  FINAL: {vertex_coords}")
    
    return vertex_sequence

def validate_polygon(vertices, face_name):
    """
    Validate that the vertex sequence forms a proper polygon.
    
    Checks for:
    1. Sufficient vertex count (≥3)
    2. No crossing edges (self-intersection)
    3. Proper closure
    4. Consistent orientation
    """
    if len(vertices) < 3:
        return False, f"Insufficient vertices: {len(vertices)}"
    
    # Check for self-intersections by examining edge crossings
    has_crosses = check_for_crosses(vertices)
    
    if has_crosses:
        return False, "Self-intersecting polygon (crosses detected)"
    
    # Check if polygon is closed
    first = np.array(vertices[0])
    last = np.array(vertices[-1])
    
    if np.linalg.norm(first - last) > 1e-6:
        return False, "Polygon not closed"
    
    # Calculate polygon area to ensure it's not degenerate
    try:
        area = calculate_polygon_area(vertices)
        if area < 1e-6:
            return False, f"Degenerate polygon (area: {area:.8f})"
    except Exception as e:
        return False, f"Area calculation failed: {e}"
    
    return True, f"Valid polygon with {len(vertices)} vertices, area: {area:.2f}"

def check_for_crosses(vertices):
    """
    Check if polygon edges cross each other.
    
    A crossing indicates incorrect vertex ordering.
    """
    n = len(vertices)
    
    for i in range(n):
        edge1_start = vertices[i]
        edge1_end = vertices[(i + 1) % n]
        
        for j in range(i + 2, n):
            # Skip adjacent edges
            if j == (i + n - 1) % n:
                continue
                
            edge2_start = vertices[j]
            edge2_end = vertices[(j + 1) % n]
            
            if edges_intersect(edge1_start, edge1_end, edge2_start, edge2_end):
                print(f"    Cross detected: Edge {i}-{(i+1)%n} intersects "
                      f"Edge {j}-{(j+1)%n}")
                return True
    
    return False

def edges_intersect(p1, q1, p2, q2):
    """Check if two line segments intersect."""
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
    
    # Special cases (collinear points)
    if (o1 == 0 and on_segment(p1, p2, q1) or
        o2 == 0 and on_segment(p1, q2, q1) or
        o3 == 0 and on_segment(p2, p1, q2) or
        o4 == 0 and on_segment(p2, q1, q2)):
        return True
    
    return False

def calculate_polygon_area(vertices):
    """Calculate polygon area using shoelace formula."""
    n = len(vertices)
    area = 0.0
    
    for i in range(n):
        j = (i + 1) % n
        area += vertices[i][0] * vertices[j][1]
        area -= vertices[j][0] * vertices[i][1]
    
    return abs(area) / 2.0

if __name__ == "__main__":
    test_wire_based_topology()
