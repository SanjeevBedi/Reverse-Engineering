#!/usr/bin/env python3
"""
Advanced wire orientation test to resolve vertex ordering crosses.

This test explores different combinations of wire and edge orientations
to find the correct topology that eliminates crosses.
"""

import numpy as np


def test_wire_orientation_combinations():
    """Test different wire and edge orientation combinations."""
    
    print("="*60)
    print("ADVANCED WIRE ORIENTATION TEST")
    print("="*60)
    
    # Test Face 1 with different orientation combinations
    face_1_base = {
        'description': 'Face 1 - YZ plane rectangle',
        'edges': [
            {'vertices': [(0, 0, 0), (0, 0, 30)]},
            {'vertices': [(0, 0, 30), (0, 20, 30)]},
            {'vertices': [(0, 20, 30), (0, 20, 0)]},
            {'vertices': [(0, 20, 0), (0, 0, 0)]}
        ]
    }
    
    # Test different wire and edge orientation combinations
    test_cases = [
        {
            'name': 'Case 1: All FORWARD',
            'wire_orientation': 'FORWARD',
            'edge_orientations': ['FORWARD', 'FORWARD', 'FORWARD', 'FORWARD']
        },
        {
            'name': 'Case 2: Wire REVERSED, All edges FORWARD',
            'wire_orientation': 'REVERSED',
            'edge_orientations': ['FORWARD', 'FORWARD', 'FORWARD', 'FORWARD']
        },
        {
            'name': 'Case 3: Wire FORWARD, All edges REVERSED',
            'wire_orientation': 'FORWARD',
            'edge_orientations': ['REVERSED', 'REVERSED', 'REVERSED', 'REVERSED']
        },
        {
            'name': 'Case 4: Wire REVERSED, All edges REVERSED',
            'wire_orientation': 'REVERSED',
            'edge_orientations': ['REVERSED', 'REVERSED', 'REVERSED', 'REVERSED']
        },
        {
            'name': 'Case 5: Wire FORWARD, Alternating edges',
            'wire_orientation': 'FORWARD',
            'edge_orientations': ['FORWARD', 'REVERSED', 'FORWARD', 'REVERSED']
        },
        {
            'name': 'Case 6: Wire REVERSED, Alternating edges',
            'wire_orientation': 'REVERSED',
            'edge_orientations': ['FORWARD', 'REVERSED', 'FORWARD', 'REVERSED']
        }
    ]
    
    print(f"Testing {len(test_cases)} orientation combinations for Face 1...")
    
    for case in test_cases:
        print(f"\n{'-'*50}")
        print(f"{case['name']}")
        print(f"Wire: {case['wire_orientation']}, "
              f"Edges: {case['edge_orientations']}")
        
        # Build face data with orientations
        face_data = {
            'description': face_1_base['description'],
            'wire_orientation': case['wire_orientation'],
            'edges': []
        }
        
        for i, edge_orientation in enumerate(case['edge_orientations']):
            edge = {
                'orientation': edge_orientation,
                'vertices': face_1_base['edges'][i]['vertices']
            }
            face_data['edges'].append(edge)
        
        # Extract vertices with this orientation combination
        vertices = extract_vertices_with_orientation(face_data)
        
        if vertices:
            # Check for crosses
            has_crosses = check_for_crosses_detailed(vertices, case['name'])
            
            if not has_crosses:
                print(f"✓ {case['name']}: NO CROSSES DETECTED!")
                print(f"  Solution found: Wire={case['wire_orientation']}, "
                      f"Edges={case['edge_orientations']}")
            else:
                print(f"✗ {case['name']}: Still has crosses")
        else:
            print(f"✗ {case['name']}: Failed to extract vertices")
    
    print(f"\n{'='*60}")
    print("ADVANCED WIRE ORIENTATION TEST COMPLETE")
    print("="*60)


def extract_vertices_with_orientation(face_data):
    """Extract vertices using specific wire and edge orientations."""
    wire_orientation = face_data['wire_orientation']
    edges = face_data['edges'].copy()
    
    print(f"  Wire orientation: {wire_orientation}")
    
    # Apply wire orientation to edge processing order
    if wire_orientation == 'REVERSED':
        edges = list(reversed(edges))
        print("  Reversed edge order due to wire orientation")
    
    # Build vertex sequence
    vertex_sequence = []
    
    for edge_idx, edge in enumerate(edges):
        edge_orientation = edge['orientation']
        edge_vertices = edge['vertices']
        
        # Apply edge orientation to vertex order
        if edge_orientation == 'FORWARD':
            start_vertex = edge_vertices[0]
            end_vertex = edge_vertices[1]
            tag = "F"
        else:  # REVERSED
            start_vertex = edge_vertices[1]
            end_vertex = edge_vertices[0]
            tag = "R"
        
        print(f"    Edge {edge_idx}: {tag} {start_vertex} → {end_vertex}")
        
        # Build sequence
        if edge_idx == 0:
            vertex_sequence.append(start_vertex)
            vertex_sequence.append(end_vertex)
        else:
            vertex_sequence.append(end_vertex)
    
    # Remove closing duplicate
    if len(vertex_sequence) > 2:
        first = np.array(vertex_sequence[0])
        last = np.array(vertex_sequence[-1])
        
        if np.linalg.norm(first - last) < 1e-6:
            vertex_sequence = vertex_sequence[:-1]
    
    print(f"  Final sequence: {vertex_sequence}")
    return vertex_sequence


def check_for_crosses_detailed(vertices, case_name):
    """Check for crosses with detailed output."""
    n = len(vertices)
    crosses_found = 0
    
    for i in range(n):
        edge1_start = vertices[i]
        edge1_end = vertices[(i + 1) % n]
        
        for j in range(i + 2, n):
            # Skip adjacent edges and wrap-around
            if j == (i + n - 1) % n:
                continue
                
            edge2_start = vertices[j]
            edge2_end = vertices[(j + 1) % n]
            
            if edges_intersect(edge1_start, edge1_end, edge2_start, edge2_end):
                print(f"    Cross: Edge {i}-{(i+1)%n} × Edge {j}-{(j+1)%n}")
                crosses_found += 1
    
    if crosses_found > 0:
        print(f"    Total crosses found: {crosses_found}")
        return True
    else:
        print(f"    ✓ No crosses detected!")
        return False


def edges_intersect(p1, q1, p2, q2):
    """Check if two line segments intersect (excluding endpoints)."""
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
    
    # General case - proper intersection
    if o1 != o2 and o3 != o4:
        return True
    
    # Special cases for collinear points
    if (o1 == 0 and on_segment(p1, p2, q1) or
        o2 == 0 and on_segment(p1, q2, q1) or
        o3 == 0 and on_segment(p2, p1, q2) or
        o4 == 0 and on_segment(p2, q1, q2)):
        return True
    
    return False


if __name__ == "__main__":
    test_wire_orientation_combinations()
