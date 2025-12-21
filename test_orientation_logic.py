#!/usr/bin/env python3
"""
Simple test to verify the orientation-based vertex selection logic
"""

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

def test_orientation_logic():
    """Test the orientation-based vertex selection logic"""
    print("Testing Orientation-Based Vertex Selection Logic")
    print("="*60)
    
    # Mock data for Face 1 (based on previous output)
    print("\nFace 1 Test (should form proper rectangle):")
    print("Edges: E1=FORWARD, E2=FORWARD, E3=REVERSED, E4=REVERSED")
    
    edges = [
        MockEdge(TopAbs_FORWARD, [0,0,0], [0,0,30]),      # E1: FORWARD
        MockEdge(TopAbs_FORWARD, [0,0,30], [0,20,30]),    # E2: FORWARD  
        MockEdge(TopAbs_REVERSED, [0,20,30], [0,20,0]),   # E3: REVERSED
        MockEdge(TopAbs_REVERSED, [0,20,0], [0,0,0])      # E4: REVERSED
    ]
    
    vertex_chain = []
    vertex_orientations = []
    
    for edge_num, edge in enumerate(edges):
        orientation_str = edge.Orientation()
        
        first_coord = edge.first
        last_coord = edge.last
        
        # ORIENTATION-BASED VERTEX SELECTION:
        # FORWARD edges: choose START vertex (first)
        # REVERSED edges: choose END vertex (last)
        if edge.Orientation() == TopAbs_FORWARD:
            selected_vertex = first_coord
            vertex_orientations.append(f"E{edge_num+1}F_start")
        else:  # REVERSED
            selected_vertex = last_coord
            vertex_orientations.append(f"E{edge_num+1}R_end")
        
        vertex_chain.append(selected_vertex)
        
        print(f"  Edge {edge_num+1}: {orientation_str} -> " +
              f"Selected {'start' if edge.Orientation() == TopAbs_FORWARD else 'end'} vertex " +
              f"({selected_vertex[0]:.1f},{selected_vertex[1]:.1f},{selected_vertex[2]:.1f})")
    
    print(f"\nResulting vertex sequence:")
    vertex_coords = " → ".join([f"({v[0]:.1f},{v[1]:.1f},{v[2]:.1f})" for v in vertex_chain])
    print(f"  {vertex_coords}")
    
    # Check if this forms a proper rectangle
    print(f"\nRectangle validation:")
    if len(vertex_chain) == 4:
        # Check if edges form proper right angles
        edges_formed = []
        for i in range(4):
            v1 = vertex_chain[i]
            v2 = vertex_chain[(i+1) % 4]
            edge_vec = [v2[j] - v1[j] for j in range(3)]
            edges_formed.append(edge_vec)
            print(f"  Edge {i+1}: {v1} → {v2} = vector {edge_vec}")
        
        # Check if opposite edges are parallel and equal
        edge1_vec = edges_formed[0]
        edge3_vec = edges_formed[2]
        edge2_vec = edges_formed[1]
        edge4_vec = edges_formed[3]
        
        # For a rectangle, edge1 should be opposite to edge3, edge2 opposite to edge4
        edge1_opposite = [-x for x in edge3_vec]
        edge2_opposite = [-x for x in edge4_vec]
        
        print(f"  Edge 1 vector: {edge1_vec}")
        print(f"  Edge 3 opposite: {edge1_opposite}")
        print(f"  Match: {edge1_vec == edge1_opposite}")
        
        print(f"  Edge 2 vector: {edge2_vec}")
        print(f"  Edge 4 opposite: {edge2_opposite}")
        print(f"  Match: {edge2_vec == edge2_opposite}")
        
        if edge1_vec == edge1_opposite and edge2_vec == edge2_opposite:
            print("  ✓ Forms a valid rectangle!")
        else:
            print("  ✗ Does not form a valid rectangle")
    else:
        print(f"  ✗ Wrong number of vertices: {len(vertex_chain)} (expected 4)")
    
    # Test Face 2
    print("\n" + "="*60)
    print("\nFace 2 Test (should form proper rectangle):")
    print("Edges: E1=FORWARD, E2=FORWARD, E3=REVERSED, E4=REVERSED")
    
    edges2 = [
        MockEdge(TopAbs_FORWARD, [0,0,0], [10,0,0]),      # E1: FORWARD
        MockEdge(TopAbs_FORWARD, [10,0,0], [10,0,30]),    # E2: FORWARD
        MockEdge(TopAbs_REVERSED, [10,0,30], [0,0,30]),   # E3: REVERSED  
        MockEdge(TopAbs_REVERSED, [0,0,30], [0,0,0])      # E4: REVERSED
    ]
    
    vertex_chain2 = []
    
    for edge_num, edge in enumerate(edges2):
        orientation_str = edge.Orientation()
        
        first_coord = edge.first
        last_coord = edge.last
        
        if edge.Orientation() == TopAbs_FORWARD:
            selected_vertex = first_coord
        else:
            selected_vertex = last_coord
        
        vertex_chain2.append(selected_vertex)
        
        print(f"  Edge {edge_num+1}: {orientation_str} -> " +
              f"Selected {'start' if edge.Orientation() == TopAbs_FORWARD else 'end'} vertex " +
              f"({selected_vertex[0]:.1f},{selected_vertex[1]:.1f},{selected_vertex[2]:.1f})")
    
    print(f"\nResulting vertex sequence:")
    vertex_coords2 = " → ".join([f"({v[0]:.1f},{v[1]:.1f},{v[2]:.1f})" for v in vertex_chain2])
    print(f"  {vertex_coords2}")

if __name__ == "__main__":
    test_orientation_logic()
