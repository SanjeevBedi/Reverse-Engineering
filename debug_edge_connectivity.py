#!/usr/bin/env python3
"""
Debug script to analyze edge connectivity and vertex ordering
"""

import sys
sys.path.append('/Users/sbedi/Nextcloud/Python/Solid/OpenCASCADE_Engineering_Drawings/src')

from engineering_drawings.solid_creator import SolidCreator

# OpenCASCADE imports
try:
    from OCC.Core.TopExp import TopExp_Explorer
    from OCC.Core.TopAbs import (TopAbs_FACE, TopAbs_EDGE, TopAbs_WIRE, 
                                 TopAbs_FORWARD, TopAbs_REVERSED)
    from OCC.Core.BRep import BRep_Tool
    from OCC.Core.TopoDS import topods
    from OCC.Core.TopExp import topexp
    from OCC.Core.TopoDS import TopoDS_Vertex
    import numpy as np
    
    OPENCASCADE_AVAILABLE = True
except ImportError:
    print("OpenCASCADE not available")
    sys.exit(1)

def analyze_face_edge_connectivity(solid_shape, target_faces=[1, 2, 5]):
    """Analyze edge connectivity for specific faces"""
    print("="*80)
    print("DETAILED EDGE CONNECTIVITY ANALYSIS")
    print("="*80)
    
    face_explorer = TopExp_Explorer(solid_shape, TopAbs_FACE)
    face_count = 0
    
    while face_explorer.More():
        face_shape = face_explorer.Current()
        face_count += 1
        
        if face_count not in target_faces:
            face_explorer.Next()
            continue
            
        print(f"\nFACE {face_count} DETAILED ANALYSIS:")
        print("-" * 50)
        
        face = topods.Face(face_shape)
        face_orientation = face.Orientation()
        orientation_str = "REVERSED" if face_orientation == TopAbs_REVERSED else "FORWARD"
        print(f"Face orientation: {orientation_str}")
        
        # Get wire from face
        wire_explorer = TopExp_Explorer(face, TopAbs_WIRE)
        if wire_explorer.More():
            wire = wire_explorer.Current()
            
            # Get all edges
            edge_explorer = TopExp_Explorer(wire, TopAbs_EDGE)
            edges = []
            while edge_explorer.More():
                edges.append(edge_explorer.Current())
                edge_explorer.Next()
            
            print(f"Number of edges: {len(edges)}")
            
            # For each edge, get both vertices and show connectivity
            edge_vertex_map = {}
            all_vertices = []
            
            for edge_num, edge in enumerate(edges):
                edge_orientation = edge.Orientation()
                orientation_str = "FORWARD" if edge_orientation == TopAbs_FORWARD else "REVERSED"
                
                # Get vertices
                first_vertex = TopoDS_Vertex()
                last_vertex = TopoDS_Vertex()
                topexp.Vertices(edge, first_vertex, last_vertex)
                
                first_pnt = BRep_Tool.Pnt(first_vertex)
                last_pnt = BRep_Tool.Pnt(last_vertex)
                
                first_coord = [first_pnt.X(), first_pnt.Y(), first_pnt.Z()]
                last_coord = [last_pnt.X(), last_pnt.Y(), last_pnt.Z()]
                
                edge_vertex_map[edge_num] = {
                    'orientation': orientation_str,
                    'first': first_coord,
                    'last': last_coord
                }
                
                print(f"  Edge {edge_num+1} ({orientation_str}):")
                print(f"    First vertex:  ({first_coord[0]:.1f}, {first_coord[1]:.1f}, {first_coord[2]:.1f})")
                print(f"    Last vertex:   ({last_coord[0]:.1f}, {last_coord[1]:.1f}, {last_coord[2]:.1f})")
                
                all_vertices.extend([first_coord, last_coord])
            
            # Find unique vertices
            unique_vertices = []
            for v in all_vertices:
                v_rounded = tuple(np.round(v, 6))
                if v_rounded not in [tuple(np.round(uv, 6)) for uv in unique_vertices]:
                    unique_vertices.append(v)
            
            print(f"\nUnique vertices ({len(unique_vertices)}):")
            for i, v in enumerate(unique_vertices):
                print(f"  V{i+1}: ({v[0]:.1f}, {v[1]:.1f}, {v[2]:.1f})")
            
            # Build connectivity map
            print(f"\nEdge connectivity analysis:")
            connectivity = {}
            
            for edge_num, edge_data in edge_vertex_map.items():
                first = tuple(np.round(edge_data['first'], 6))
                last = tuple(np.round(edge_data['last'], 6))
                orient = edge_data['orientation']
                
                # Find which unique vertices these correspond to
                first_idx = None
                last_idx = None
                
                for i, uv in enumerate(unique_vertices):
                    uv_rounded = tuple(np.round(uv, 6))
                    if uv_rounded == first:
                        first_idx = i + 1
                    if uv_rounded == last:
                        last_idx = i + 1
                
                connectivity[edge_num + 1] = {
                    'orientation': orient,
                    'connects': f"V{first_idx} → V{last_idx}",
                    'first_vertex': first_idx,
                    'last_vertex': last_idx
                }
                
                print(f"  Edge {edge_num+1} ({orient}): V{first_idx} → V{last_idx}")
            
            # Try to build proper rectangular sequence
            print(f"\nAttempting to build rectangular sequence:")
            if len(unique_vertices) == 4:
                # Find the proper ordering for a rectangle
                print("  Perfect! 4 unique vertices for rectangle")
                
                # For a rectangle, we need to find the sequence that forms a proper loop
                print("  Building rectangular loop...")
                
                # Start with vertex 1, then find the next connected vertex
                loop_sequence = []
                used_edges = set()
                current_vertex = 1
                
                for step in range(4):
                    loop_sequence.append(current_vertex)
                    
                    # Find an unused edge that starts from current_vertex
                    next_vertex = None
                    for edge_num, conn in connectivity.items():
                        if edge_num in used_edges:
                            continue
                            
                        if conn['first_vertex'] == current_vertex:
                            next_vertex = conn['last_vertex']
                            used_edges.add(edge_num)
                            print(f"    Step {step+1}: V{current_vertex} → V{next_vertex} (via Edge {edge_num})")
                            break
                        elif conn['last_vertex'] == current_vertex:
                            next_vertex = conn['first_vertex']
                            used_edges.add(edge_num)
                            print(f"    Step {step+1}: V{current_vertex} → V{next_vertex} (via Edge {edge_num} reversed)")
                            break
                    
                    if next_vertex is None:
                        print(f"    ERROR: Could not find next vertex from V{current_vertex}")
                        break
                    
                    current_vertex = next_vertex
                
                print(f"  Final rectangular sequence: {' → '.join([f'V{v}' for v in loop_sequence])}")
                
                # Convert back to coordinates
                final_coords = []
                for vertex_num in loop_sequence:
                    coord = unique_vertices[vertex_num - 1]
                    final_coords.append(coord)
                    print(f"    V{vertex_num}: ({coord[0]:.1f}, {coord[1]:.1f}, {coord[2]:.1f})")
                
            else:
                print(f"  WARNING: Expected 4 vertices but found {len(unique_vertices)}")
        
        face_explorer.Next()

def main():
    print("Creating test solid...")
    creator = SolidCreator()
    solid = creator.create_boolean_solid_with_cut()
    
    if solid:
        print("✓ Solid created successfully")
        analyze_face_edge_connectivity(solid, target_faces=[1, 2, 5])
    else:
        print("✗ Failed to create solid")

if __name__ == "__main__":
    main()
