"""
Diagnostic script to find missing edges between specific vertices
"""
import numpy as np
from OCC.Core.STEPControl import STEPControl_Reader
from OCC.Core.TopExp import TopExp_Explorer
from OCC.Core.TopAbs import TopAbs_EDGE, TopAbs_VERTEX
from OCC.Core.BRep import BRep_Tool
from OCC.Core.TopoDS import topods

def find_vertex_index(coord, all_vertices, tolerance=1e-5):
    """Find the index of a vertex with given coordinates."""
    for i, v in enumerate(all_vertices):
        if np.linalg.norm(v - coord) < tolerance:
            return i
    return None

def check_edge_in_solid(solid, vertex_pairs, all_vertices):
    """
    Check if specific edges exist in the solid.
    
    Args:
        solid: TopoDS_Shape
        vertex_pairs: List of (v1_idx, v2_idx) to check
        all_vertices: Array of all vertex coordinates
    """
    print("=" * 70)
    print("CHECKING EDGES IN ORIGINAL SOLID")
    print("=" * 70)
    
    # Get all edges from solid
    edge_explorer = TopExp_Explorer(solid, TopAbs_EDGE)
    solid_edges = []
    
    print("\nExtracting all edges from solid...")
    edge_count = 0
    while edge_explorer.More():
        edge = topods.Edge(edge_explorer.Current())
        
        # Get vertices of this edge
        vertex_explorer = TopExp_Explorer(edge, TopAbs_VERTEX)
        edge_vertices = []
        while vertex_explorer.More():
            vertex = topods.Vertex(vertex_explorer.Current())
            pnt = BRep_Tool.Pnt(vertex)
            coord = np.array([pnt.X(), pnt.Y(), pnt.Z()])
            edge_vertices.append(coord)
            vertex_explorer.Next()
        
        if len(edge_vertices) == 2:
            # Find indices
            v1_idx = find_vertex_index(edge_vertices[0], all_vertices)
            v2_idx = find_vertex_index(edge_vertices[1], all_vertices)
            
            if v1_idx is not None and v2_idx is not None:
                # Normalize (smaller index first)
                edge_tuple = tuple(sorted([v1_idx, v2_idx]))
                solid_edges.append(edge_tuple)
                edge_count += 1
        
        edge_explorer.Next()
    
    # Remove duplicates (each edge appears twice)
    unique_edges = list(set(solid_edges))
    print(f"Total edges in solid: {len(solid_edges)} (with duplicates)")
    print(f"Unique edges: {len(unique_edges)}")
    
    # Check specific vertex pairs
    print("\n" + "=" * 70)
    print("CHECKING SPECIFIC EDGES")
    print("=" * 70)
    
    for v1_idx, v2_idx in vertex_pairs:
        edge = tuple(sorted([v1_idx, v2_idx]))
        exists = edge in unique_edges
        
        v1_coord = all_vertices[v1_idx]
        v2_coord = all_vertices[v2_idx]
        
        print(f"\nEdge V{v1_idx} -- V{v2_idx}:")
        print(f"  V{v1_idx}: ({v1_coord[0]:.6f}, {v1_coord[1]:.6f}, "
              f"{v1_coord[2]:.6f})")
        print(f"  V{v2_idx}: ({v2_coord[0]:.6f}, {v2_coord[1]:.6f}, "
              f"{v2_coord[2]:.6f})")
        
        # Analyze edge orientation
        same_x = abs(v1_coord[0] - v2_coord[0]) < 1e-6
        same_y = abs(v1_coord[1] - v2_coord[1]) < 1e-6
        same_z = abs(v1_coord[2] - v2_coord[2]) < 1e-6
        
        if same_x and same_y:
            edge_type = "VERTICAL (z-parallel)"
        elif same_x and same_z:
            edge_type = "Y-PARALLEL"
        elif same_y and same_z:
            edge_type = "X-PARALLEL"
        else:
            edge_type = "DIAGONAL"
        
        print(f"  Type: {edge_type}")
        print(f"  Same X: {same_x}, Same Y: {same_y}, Same Z: {same_z}")
        
        if exists:
            print(f"  ✓ EXISTS in original solid")
        else:
            print(f"  ✗ DOES NOT EXIST in original solid")
    
    return unique_edges


if __name__ == "__main__":
    import sys
    
    # Get seed from command line or use default
    seed = int(sys.argv[1]) if len(sys.argv) > 1 else 55
    
    # Read solid
    step_file = f"tagged_solid_seed_{seed}.step"
    print(f"Reading {step_file}...")
    
    reader = STEPControl_Reader()
    status = reader.ReadFile(step_file)
    if status != 1:  # IFSelect_RetDone
        print(f"Error reading file: {step_file}")
        sys.exit(1)
    
    reader.TransferRoots()
    solid = reader.Shape()
    
    # Extract all unique vertices
    vertex_explorer = TopExp_Explorer(solid, TopAbs_VERTEX)
    unique_vertices = []
    seen = set()
    
    while vertex_explorer.More():
        vertex = topods.Vertex(vertex_explorer.Current())
        pnt = BRep_Tool.Pnt(vertex)
        coord = (pnt.X(), pnt.Y(), pnt.Z())
        
        # Round to avoid floating point issues
        coord_key = tuple(round(c, 6) for c in coord)
        if coord_key not in seen:
            seen.add(coord_key)
            unique_vertices.append(np.array(coord))
        
        vertex_explorer.Next()
    
    # Sort vertices
    all_vertices = np.array(sorted(unique_vertices, 
                                   key=lambda v: (v[0], v[1], v[2])))
    
    print(f"Total unique vertices: {len(all_vertices)}\n")
    
    # Check specific edges mentioned by user
    vertex_pairs_to_check = [
        (62, 114),  # First missing edge
        (65, 115) if len(all_vertices) > 115 else (65, 114),  # Second edge
    ]
    
    unique_edges = check_edge_in_solid(solid, vertex_pairs_to_check, 
                                       all_vertices)
    
    print("\n" + "=" * 70)
    print(f"Total unique edges found: {len(unique_edges)}")
    print("=" * 70)
