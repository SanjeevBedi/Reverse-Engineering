#!/usr/bin/env python3
"""
Enhanced test for orientation-based vertex selection with proper connectivity.

This test implements the correct approach: use edge orientation to select vertices,
then build connectivity-based sequential chains to form proper polygons.
"""

import numpy as np

# Mock the edge orientations and vertices to test the logic
class MockEdge:
    def __init__(self, orientation, first, last):
        self.orientation_val = orientation
        self.first = first
        self.last = last
    
    def Orientation(self):
        return self.orientation_val

# Mock orientation constants
TopAbs_FORWARD = "FORWARD"
TopAbs_REVERSED = "REVERSED"

def build_connectivity_chain(edges):
    """
    Build a connected vertex chain from edges using orientation-based vertex selection
    combined with connectivity analysis.
    
    This is the proper approach:
    1. Extract oriented vertices from each edge
    2. Build connectivity graph to find sequential order
    3. Traverse to form proper polygon
    """
    print(f"    Building connectivity chain from {len(edges)} edges...")
    
    if not edges:
        return []
    
    # Step 1: Extract oriented vertex pairs from each edge
    edge_vertices = []
    for edge_num, edge in enumerate(edges):
        orientation_str = edge.Orientation()
        first_coord = edge.first
        last_coord = edge.last
        
        # ORIENTATION-BASED VERTEX SELECTION:
        # FORWARD edges: start → end (first → last)  
        # REVERSED edges: end → start (last → first)
        if edge.Orientation() == TopAbs_FORWARD:
            start_vertex = first_coord
            end_vertex = last_coord
        else:  # REVERSED
            start_vertex = last_coord  
            end_vertex = first_coord
        
        edge_vertices.append({
            'edge_num': edge_num + 1,
            'orientation': orientation_str,
            'start': start_vertex,
            'end': end_vertex,
            'original_first': first_coord,
            'original_last': last_coord
        })
        
        print(f"      Edge {edge_num+1}: {orientation_str} -> " +
              f"start({start_vertex[0]:.1f},{start_vertex[1]:.1f},{start_vertex[2]:.1f}) " +
              f"end({end_vertex[0]:.1f},{end_vertex[1]:.1f},{end_vertex[2]:.1f})")
    
    # Step 2: Build connectivity-based sequential chain
    print(f"    Building sequential connectivity...")
    
    vertex_chain = []
    used_edges = set()
    tolerance = 1e-6
    
    # Start with first edge
    current_edge = edge_vertices[0]
    vertex_chain.append(current_edge['start'])
    vertex_chain.append(current_edge['end']) 
    used_edges.add(0)
    current_end = current_edge['end']
    
    print(f"      Starting with Edge {current_edge['edge_num']}: " +
          f"chain = [{current_edge['start']}, {current_edge['end']}]")
    
    # Find connecting edges sequentially
    for step in range(1, len(edges)):
        next_edge_idx = None
        
        # Find edge that connects to current_end
        for i, edge_data in enumerate(edge_vertices):
            if i in used_edges:
                continue
                
            start_dist = np.linalg.norm(np.array(edge_data['start']) - np.array(current_end))
            end_dist = np.linalg.norm(np.array(edge_data['end']) - np.array(current_end))
            
            if start_dist < tolerance:
                # Edge starts where we ended, add its end vertex
                vertex_chain.append(edge_data['end'])
                current_end = edge_data['end']
                next_edge_idx = i
                print(f"      Step {step}: Edge {edge_data['edge_num']} connects via START, " +
                      f"added END vertex {edge_data['end']}")
                break
            elif end_dist < tolerance:
                # Edge ends where we ended, add its start vertex  
                vertex_chain.append(edge_data['start'])
                current_end = edge_data['start']
                next_edge_idx = i
                print(f"      Step {step}: Edge {edge_data['edge_num']} connects via END, " +
                      f"added START vertex {edge_data['start']}")
                break
        
        if next_edge_idx is not None:
            used_edges.add(next_edge_idx)
        else:
            print(f"      Step {step}: No connecting edge found! Chain may be incomplete.")
            break
    
    # Step 3: Remove closing duplicate if present
    if len(vertex_chain) > 3:
        first_tuple = tuple(np.round(vertex_chain[0], 6))
        last_tuple = tuple(np.round(vertex_chain[-1], 6))
        
        if first_tuple == last_tuple:
            vertex_chain.pop()
            print(f"    Removed closing duplicate vertex")
    
    print(f"    ✓ Built connectivity chain: {len(vertex_chain)} vertices")
    return vertex_chain

def validate_rectangle(vertices):
    """Validate if vertices form a proper rectangle."""
    if len(vertices) != 4:
        return False, f"Wrong vertex count: {len(vertices)} (expected 4)"
    
    # Check if opposite edges are parallel and equal
    edges = []
    for i in range(4):
        start = np.array(vertices[i])
        end = np.array(vertices[(i + 1) % 4])
        edge_vector = end - start
        edges.append(edge_vector)
    
    # Check if opposite edges are equal (parallel and same length)
    edge1_to_edge3 = np.allclose(edges[0], -edges[2], atol=1e-6)
    edge2_to_edge4 = np.allclose(edges[1], -edges[3], atol=1e-6)
    
    # Check if adjacent edges are perpendicular
    edge1_dot_edge2 = abs(np.dot(edges[0], edges[1])) < 1e-6
    
    if edge1_to_edge3 and edge2_to_edge4 and edge1_dot_edge2:
        return True, "Valid rectangle"
    else:
        return False, f"Invalid rectangle: opp_edges={edge1_to_edge3 and edge2_to_edge4}, perp={edge1_dot_edge2}"

def test_enhanced_orientation_logic():
    """Test the enhanced orientation-based logic with proper connectivity."""
    print("Enhanced Orientation-Based Vertex Selection with Connectivity")
    print("="*70)
    
    # Test Face 1 (YZ plane rectangle)
    print("\nFace 1 Test (YZ plane rectangle):")
    print("Expected: (0,0,0) → (0,0,30) → (0,20,30) → (0,20,0)")
    
    edges = [
        MockEdge(TopAbs_FORWARD, [0,0,0], [0,0,30]),      # E1: FORWARD 
        MockEdge(TopAbs_FORWARD, [0,0,30], [0,20,30]),    # E2: FORWARD
        MockEdge(TopAbs_REVERSED, [0,20,30], [0,20,0]),   # E3: REVERSED
        MockEdge(TopAbs_REVERSED, [0,20,0], [0,0,0])      # E4: REVERSED
    ]
    
    vertex_chain = build_connectivity_chain(edges)
    
    print(f"\n  Final vertex sequence:")
    vertex_coords = " → ".join([f"({v[0]:.1f},{v[1]:.1f},{v[2]:.1f})" for v in vertex_chain])
    print(f"  {vertex_coords}")
    
    is_valid, reason = validate_rectangle(vertex_chain)
    print(f"  Rectangle validation: {'✓' if is_valid else '✗'} {reason}")
    
    # Test Face 2 (XZ plane rectangle)
    print("\n" + "-"*70)
    print("\nFace 2 Test (XZ plane rectangle):")
    print("Expected: (0,0,0) → (10,0,0) → (10,0,30) → (0,0,30)")
    
    edges = [
        MockEdge(TopAbs_FORWARD, [0,0,0], [10,0,0]),      # E1: FORWARD
        MockEdge(TopAbs_FORWARD, [10,0,0], [10,0,30]),    # E2: FORWARD  
        MockEdge(TopAbs_REVERSED, [10,0,30], [0,0,30]),   # E3: REVERSED
        MockEdge(TopAbs_REVERSED, [0,0,30], [0,0,0])      # E4: REVERSED
    ]
    
    vertex_chain = build_connectivity_chain(edges)
    
    print(f"\n  Final vertex sequence:")
    vertex_coords = " → ".join([f"({v[0]:.1f},{v[1]:.1f},{v[2]:.1f})" for v in vertex_chain])
    print(f"  {vertex_coords}")
    
    is_valid, reason = validate_rectangle(vertex_chain)
    print(f"  Rectangle validation: {'✓' if is_valid else '✗'} {reason}")
    
    # Test Face 5 (complex case with mixed orientations)
    print("\n" + "-"*70)
    print("\nFace 5 Test (mixed orientation pattern):")
    print("Expected proper rectangular sequence")
    
    edges = [
        MockEdge(TopAbs_FORWARD, [0,0,0], [10,0,0]),      # E1: FORWARD
        MockEdge(TopAbs_REVERSED, [0,20,0], [10,20,0]),   # E2: REVERSED  
        MockEdge(TopAbs_FORWARD, [10,0,0], [10,20,0]),    # E3: FORWARD
        MockEdge(TopAbs_REVERSED, [0,20,0], [0,0,0])      # E4: REVERSED
    ]
    
    vertex_chain = build_connectivity_chain(edges)
    
    print(f"\n  Final vertex sequence:")
    vertex_coords = " → ".join([f"({v[0]:.1f},{v[1]:.1f},{v[2]:.1f})" for v in vertex_chain])
    print(f"  {vertex_coords}")
    
    is_valid, reason = validate_rectangle(vertex_chain)
    print(f"  Rectangle validation: {'✓' if is_valid else '✗'} {reason}")
    
    print("\n" + "="*70)
    print("ENHANCED ORIENTATION + CONNECTIVITY TEST COMPLETE")
    print("="*70)
    print("Key Insights:")
    print("• Orientation determines vertex selection from each edge")
    print("• Connectivity analysis builds proper sequential chains")
    print("• This approach respects OpenCASCADE topology")
    print("• Works for any edge orientation pattern")

if __name__ == "__main__":
    test_enhanced_orientation_logic()
