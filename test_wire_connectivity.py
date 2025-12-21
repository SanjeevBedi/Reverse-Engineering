#!/usr/bin/env python3
"""
Test wire connectivity and edge ordering in OpenCASCADE.

This test examines the actual wire structure to understand
why edges appear non-sequential.
"""

import sys
import numpy as np
sys.path.append('/Users/sbedi/Nextcloud/Python/Solid')
sys.path.append('/Users/sbedi/Nextcloud/Python/Solid/OpenCASCADE_Engineering_Drawings/src')

try:
    from V5_current import create_opencascade_solid, OPENCASCADE_AVAILABLE
    
    if OPENCASCADE_AVAILABLE:
        from OCC.Core.TopExp import TopExp_Explorer
        from OCC.Core.TopAbs import TopAbs_FACE, TopAbs_WIRE, TopAbs_EDGE, TopAbs_VERTEX
        from OCC.Core.BRep import BRep_Tool
        from OCC.Core.TopoDS import topods
        from OCC.Core.TopExp import topexp
        from OCC.Core.TopoDS import TopoDS_Vertex
except ImportError as e:
    print(f"Import error: {e}")
    OPENCASCADE_AVAILABLE = False


def analyze_wire_connectivity():
    """Analyze wire connectivity and edge ordering."""
    
    print("="*70)
    print("WIRE CONNECTIVITY ANALYSIS")
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
    
    # Focus on Face 1 (the problematic one)
    face_explorer = TopExp_Explorer(solid, TopAbs_FACE)
    face_count = 0
    
    while face_explorer.More() and face_count < 2:
        face = face_explorer.Current()
        face_count += 1
        
        if face_count == 1:  # Face 1
            print(f"\n{'-'*60}")
            print(f"ANALYZING FACE 1 WIRE CONNECTIVITY")
            print(f"{'-'*60}")
            
            # Get the wire
            wire_explorer = TopExp_Explorer(face, TopAbs_WIRE)
            if wire_explorer.More():
                wire = wire_explorer.Current()
                
                # Method 1: Get all edges in wire
                print("\nMethod 1: TopExp_Explorer edge order")
                edge_explorer = TopExp_Explorer(wire, TopAbs_EDGE)
                edges = []
                
                edge_idx = 0
                while edge_explorer.More():
                    edge = edge_explorer.Current()
                    edges.append(edge)
                    
                    # Get edge vertices
                    first_vertex = TopoDS_Vertex()
                    last_vertex = TopoDS_Vertex()
                    topexp.Vertices(edge, first_vertex, last_vertex, True)
                    
                    # Get coordinates
                    first_pnt = BRep_Tool.Pnt(first_vertex)
                    last_pnt = BRep_Tool.Pnt(last_vertex)
                    
                    first_coord = [first_pnt.X(), first_pnt.Y(), first_pnt.Z()]
                    last_coord = [last_pnt.X(), last_pnt.Y(), last_pnt.Z()]
                    
                    print(f"  Edge {edge_idx}: "
                          f"({first_coord[0]:.1f},{first_coord[1]:.1f},{first_coord[2]:.1f}) → "
                          f"({last_coord[0]:.1f},{last_coord[1]:.1f},{last_coord[2]:.1f})")
                    
                    edge_idx += 1
                    edge_explorer.Next()
                
                print(f"\nTotal edges found: {len(edges)}")
                
                # Method 2: Check connectivity matrix
                print(f"\nMethod 2: Connectivity Analysis")
                print("Checking which edges connect to which...")
                
                # Extract all edge endpoints
                edge_endpoints = []
                for i, edge in enumerate(edges):
                    first_vertex = TopoDS_Vertex()
                    last_vertex = TopoDS_Vertex()
                    topexp.Vertices(edge, first_vertex, last_vertex, True)
                    
                    first_pnt = BRep_Tool.Pnt(first_vertex)
                    last_pnt = BRep_Tool.Pnt(last_vertex)
                    
                    first_coord = np.array([first_pnt.X(), first_pnt.Y(), first_pnt.Z()])
                    last_coord = np.array([last_pnt.X(), last_pnt.Y(), last_pnt.Z()])
                    
                    edge_endpoints.append({
                        'edge_id': i,
                        'start': first_coord,
                        'end': last_coord
                    })
                
                # Build connectivity matrix
                print(f"\nConnectivity Matrix:")
                print("Format: Edge X → Edge Y (distance)")
                
                connectivity = {}
                for i, edge_i in enumerate(edge_endpoints):
                    connectivity[i] = []
                    
                    for j, edge_j in enumerate(edge_endpoints):
                        if i != j:
                            # Check if edge_i's end connects to edge_j's start
                            dist_end_start = np.linalg.norm(edge_i['end'] - edge_j['start'])
                            # Check if edge_i's end connects to edge_j's end  
                            dist_end_end = np.linalg.norm(edge_i['end'] - edge_j['end'])
                            
                            min_dist = min(dist_end_start, dist_end_end)
                            
                            if min_dist < 1e-6:  # Connected
                                connection_type = "start" if dist_end_start < dist_end_end else "end"
                                connectivity[i].append({
                                    'to_edge': j,
                                    'distance': min_dist,
                                    'connects_to': connection_type
                                })
                                print(f"  Edge {i} → Edge {j} ({connection_type}, dist={min_dist:.8f})")
                
                # Method 3: Try to build proper sequence
                print(f"\nMethod 3: Building Proper Edge Sequence")
                
                # Start with edge 0 and try to follow connections
                current_edge = 0
                sequence = [current_edge]
                used_edges = {current_edge}
                
                print(f"Starting with Edge {current_edge}")
                
                while len(sequence) < len(edges):
                    # Find next connected edge
                    next_edge = None
                    
                    if current_edge in connectivity:
                        for connection in connectivity[current_edge]:
                            next_candidate = connection['to_edge']
                            if next_candidate not in used_edges:
                                next_edge = next_candidate
                                break
                    
                    if next_edge is not None:
                        sequence.append(next_edge)
                        used_edges.add(next_edge)
                        print(f"  → Edge {next_edge}")
                        current_edge = next_edge
                    else:
                        print(f"  ✗ No more connections found")
                        break
                
                print(f"\nProper edge sequence: {sequence}")
                
                if len(sequence) == len(edges):
                    print("✓ Found complete connected sequence!")
                else:
                    print(f"✗ Incomplete sequence: {len(sequence)}/{len(edges)} edges")
                
                # Method 4: Show the corrected vertex order
                print(f"\nMethod 4: Corrected Vertex Sequence")
                if len(sequence) == len(edges):
                    corrected_vertices = []
                    
                    for seq_idx, edge_idx in enumerate(sequence):
                        edge_data = edge_endpoints[edge_idx]
                        
                        if seq_idx == 0:
                            # First edge: add both start and end
                            corrected_vertices.append(edge_data['start'])
                            corrected_vertices.append(edge_data['end'])
                        else:
                            # Subsequent edges: determine which vertex to add
                            prev_end = corrected_vertices[-1]
                            
                            start_dist = np.linalg.norm(prev_end - edge_data['start'])
                            end_dist = np.linalg.norm(prev_end - edge_data['end'])
                            
                            if start_dist < 1e-6:
                                # Start connects, add end
                                corrected_vertices.append(edge_data['end'])
                            else:
                                # End connects, add start
                                corrected_vertices.append(edge_data['start'])
                    
                    print("Corrected vertex sequence:")
                    for i, v in enumerate(corrected_vertices):
                        print(f"  V{i}: ({v[0]:.1f}, {v[1]:.1f}, {v[2]:.1f})")
                    
                    # Check if this forms a closed loop
                    if len(corrected_vertices) > 2:
                        first_v = corrected_vertices[0]
                        last_v = corrected_vertices[-1]
                        closing_dist = np.linalg.norm(first_v - last_v)
                        
                        if closing_dist < 1e-6:
                            print(f"✓ Forms closed loop (closing distance: {closing_dist:.8f})")
                        else:
                            print(f"⚠️  Open loop (closing distance: {closing_dist:.8f})")
            
        face_explorer.Next()
    
    print(f"\n{'='*70}")
    print("WIRE CONNECTIVITY ANALYSIS COMPLETE")
    print("="*70)


if __name__ == "__main__":
    analyze_wire_connectivity()
