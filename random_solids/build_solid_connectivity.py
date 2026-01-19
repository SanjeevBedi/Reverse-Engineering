#!/usr/bin/env python3
"""
build_solid_connectivity.py - Build Enhanced Solid Connectivity Matrix

This module:
1. Loads connectivity matrices from three views (top, front, side)
2. Imports the original solid shape from BREP file
3. Scans the solid face-by-face to build enhanced connectivity matrix
4. Enhanced matrix includes: connectivity value, face indices, and plane equations
5. Matrix depth options: 3 (connectivity + 2 face indices) or 11 (+ 8 plane params)
6. Validates connectivity by plotting edges
"""

import numpy as np
import os
import sys
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from OCC.Core.BRepTools import breptools_Read
from OCC.Core.TopoDS import TopoDS_Shape, topods
from OCC.Core.TopExp import TopExp_Explorer
from OCC.Core.TopAbs import TopAbs_FACE, TopAbs_EDGE, TopAbs_VERTEX
from OCC.Core.BRep import BRep_Tool
from OCC.Core.BRepAdaptor import BRepAdaptor_Surface
from OCC.Core.GeomAbs import GeomAbs_Plane
from OCC.Extend.TopologyUtils import TopologyExplorer
from OCC.Core.gp import gp_Pln


def load_connectivity_matrices(seed, input_dir="Output"):
    """Load connectivity matrices from file"""
    filename = os.path.join(input_dir, f"connectivity_matrices_seed_{seed}.npz")
    
    if not os.path.exists(filename):
        raise FileNotFoundError(f"Connectivity matrices not found: {filename}")
    
    print(f"\n[LOAD] Loading connectivity matrices from: {filename}")
    data = np.load(filename, allow_pickle=True)
    
    all_vertices = data['all_vertices']
    top_view_matrix = data['top_view_matrix']
    front_view_matrix = data['front_view_matrix']
    side_view_matrix = data['side_view_matrix']
    
    print(f"  - Loaded {len(all_vertices)} vertices")
    print(f"  - Top view matrix shape: {top_view_matrix.shape}")
    print(f"  - Front view matrix shape: {front_view_matrix.shape}")
    print(f"  - Side view matrix shape: {side_view_matrix.shape}")
    
    # Load unit conversion parameters
    try:
        units = str(data.get('units', 'cm'))
    except:
        units = 'cm'
    
    try:
        drawing_scale_real = float(data.get('drawing_scale_real', 1.0))
    except:
        drawing_scale_real = 1.0
    
    try:
        drawing_scale_drawing = float(data.get('drawing_scale_drawing', 1.0))
    except:
        drawing_scale_drawing = 1.0
    
    print(f"  - Units: {units}")
    print(f"  - Scale: {drawing_scale_real}:{drawing_scale_drawing}")
    
    return (all_vertices, top_view_matrix, front_view_matrix, side_view_matrix,
            units, drawing_scale_real, drawing_scale_drawing)


def load_original_solid(seed, input_dir="Output"):
    """Load the original solid shape from BREP file"""
    filename = os.path.join(input_dir, f"solid_seed_{seed}.brep")
    
    if not os.path.exists(filename):
        raise FileNotFoundError(f"Solid BREP file not found: {filename}")
    
    print(f"\n[LOAD] Loading original solid from: {filename}")
    
    from OCC.Core.BRep import BRep_Builder
    shape = TopoDS_Shape()
    builder = BRep_Builder()
    breptools_Read(shape, filename, builder)
    
    # Analyze solid
    topo = TopologyExplorer(shape)
    n_faces = topo.number_of_faces()
    n_edges = topo.number_of_edges()
    n_vertices = topo.number_of_vertices()
    
    print(f"  - Solid loaded successfully")
    print(f"  - Faces: {n_faces}")
    print(f"  - Edges: {n_edges}")
    print(f"  - Vertices: {n_vertices}")
    
    return shape


def extract_vertex_coordinates(shape, tolerance=1e-6):
    """Load face polygons from saved numpy file"""
    filename = os.path.join(input_dir, f"solid_faces_seed_{seed}.npy")
    
    if not os.path.exists(filename):
        raise FileNotFoundError(f"Face polygons file not found: {filename}")
    
    print(f"\n[LOAD] Loading face polygons from: {filename}")
    
    loaded_data = np.load(filename, allow_pickle=True)
    
    # Debug: Check the type and structure
    print(f"  - Loaded data type: {type(loaded_data)}")
    print(f"  - Loaded data dtype: {loaded_data.dtype}")
    print(f"  - Loaded data shape: {loaded_data.shape}")
    
    # Handle different possible formats
    if loaded_data.dtype == object:
        # It's an object array, could contain dict or list
        if loaded_data.shape == ():
            # Scalar object (likely a dict)
            data_item = loaded_data.item()
            print(f"  - Data is a dict with keys: {list(data_item.keys())}")
            face_data = data_item.get('faces', [])
        else:
            # Array of objects
            face_data = loaded_data.tolist()
    else:
        # Regular array
        face_data = loaded_data
    
    # Ensure face_data is a list
    if not isinstance(face_data, list):
        if isinstance(face_data, np.ndarray):
            face_data = face_data.tolist()
        else:
            face_data = [face_data]
    
    print(f"  - Loaded {len(face_data)} faces")
    
    # Debug: Show structure of first face
    if len(face_data) > 0:
        first_face = face_data[0]
        if isinstance(first_face, dict):
            print(f"  - First face structure: {list(first_face.keys())}")
        else:
            print(f"  - First face type: {type(first_face)}")
    
    return face_data


def extract_vertex_coordinates(shape, tolerance=1e-6):
    """Extract all unique vertices from the solid with their coordinates"""
    print("\n[EXTRACT] Extracting vertices from solid...")
    
    vertex_coords = []
    vertex_map = {}  # Maps OCC vertex to index
    
    explorer = TopExp_Explorer(shape, TopAbs_VERTEX)
    vertex_idx = 0
    
    while explorer.More():
        vertex = topods.Vertex(explorer.Current())
        pnt = BRep_Tool.Pnt(vertex)
        coord = np.array([pnt.X(), pnt.Y(), pnt.Z()])
        
        # Check if this vertex already exists (within tolerance)
        is_duplicate = False
        for existing_idx, existing_coord in enumerate(vertex_coords):
            if np.linalg.norm(coord - existing_coord) < tolerance:
                vertex_map[vertex] = existing_idx
                is_duplicate = True
                break
        
        if not is_duplicate:
            vertex_coords.append(coord)
            vertex_map[vertex] = vertex_idx
            vertex_idx += 1
        
        explorer.Next()
    
    vertex_coords = np.array(vertex_coords)
    print(f"  - Extracted {len(vertex_coords)} unique vertices")
    
    return vertex_coords, vertex_map


def get_face_plane_equation(face):
    """Extract plane equation from a planar face: ax + by + cz + d = 0"""
    try:
        surface = BRepAdaptor_Surface(face)
        
        if surface.GetType() == GeomAbs_Plane:
            plane = surface.Plane()
            
            # Get plane equation coefficients
            a, b, c, d = plane.Axis().Direction().X(), plane.Axis().Direction().Y(), plane.Axis().Direction().Z(), plane.Location().Distance(gp_Pln().Location())
            
            # Normalize
            norm = np.sqrt(a*a + b*b + c*c)
            if norm > 1e-10:
                a, b, c = a/norm, b/norm, c/norm
                d = d/norm
            
            # Calculate d properly: d = -(a*x0 + b*y0 + c*z0) where (x0,y0,z0) is a point on the plane
            loc = plane.Location()
            d = -(a*loc.X() + b*loc.Y() + c*loc.Z())
            
            return np.array([a, b, c, d])
        else:
            return None
    except:
        return None


def extract_faces_from_vertices(face_polygons):
    """Extract all unique vertices from face polygon data"""
    print("\n[EXTRACT] Extracting vertices from face polygons...")
    
    vertex_coords = []
    vertex_map = {}  # Maps vertex tuple to index
    
    for face_idx, face in enumerate(face_data):
        # Extract outer boundary vertices
        outer_boundary = face.get('outer_boundary', [])
        
        for vertex in outer_boundary:
            vertex_array = np.array(vertex)
            vertex_tuple = tuple(np.round(vertex, 6))  # Round for matching
            
            # Check if this vertex already exists
            if vertex_tuple not in vertex_map:
                vertex_map[vertex_tuple] = len(vertex_coords)
                vertex_coords.append(vertex_array)
        
        # Extract hole vertices if present (key is 'holes', not 'cutouts')
        holes = face.get('holes', [])
        if holes is None:
            holes = []
        for hole in holes:
            for vertex in hole:
                vertex_array = np.array(vertex)
                vertex_tuple = tuple(np.round(vertex, 6))
                
                if vertex_tuple not in vertex_map:
                    vertex_map[vertex_tuple] = len(vertex_coords)
                    vertex_coords.append(vertex_array)
    
    vertex_coords = np.array(vertex_coords)
    print(f"  - Extracted {len(vertex_coords)} unique vertices")
    
    return vertex_coords, vertex_map


def match_vertices_to_loaded(solid_vertices, loaded_vertices, tolerance=0.1):
    """
    Match solid vertices to loaded vertices and create index mapping.
    
    Args:
        solid_vertices: Vertices extracted from solid (Nx3)
        loaded_vertices: Vertices from connectivity matrix file (Mx3)
        tolerance: Distance tolerance for matching (mm)
        
    Returns:
        vertex_index_map: Maps solid vertex index to loaded vertex index
    """
    print("\n[MATCH] Matching solid vertices to loaded vertices...")
    print(f"  - Solid vertices: {len(solid_vertices)}")
    print(f"  - Loaded vertices: {len(loaded_vertices)}")
    
    vertex_index_map = {}
    unmatched_solid = []
    unmatched_loaded = set(range(len(loaded_vertices)))
    
    for solid_idx, solid_coord in enumerate(solid_vertices):
        best_match = None
        best_dist = float('inf')
        
        for loaded_idx, loaded_coord in enumerate(loaded_vertices):
            dist = np.linalg.norm(solid_coord - loaded_coord)
            if dist < best_dist:
                best_dist = dist
                best_match = loaded_idx
        
        if best_dist < tolerance:
            vertex_index_map[solid_idx] = best_match
            if best_match in unmatched_loaded:
                unmatched_loaded.remove(best_match)
        else:
            unmatched_solid.append(solid_idx)
    
    print(f"  - Matched vertices: {len(vertex_index_map)}")
    print(f"  - Unmatched solid vertices: {len(unmatched_solid)}")
    print(f"  - Unmatched loaded vertices: {len(unmatched_loaded)}")
    
    if len(unmatched_solid) > 0:
        print(f"\n  [WARNING] Some solid vertices not matched:")
        for idx in unmatched_solid[:5]:  # Show first 5
            coord = solid_vertices[idx]
            print(f"    Vertex {idx}: [{coord[0]:.2f}, {coord[1]:.2f}, {coord[2]:.2f}]")
    
    if len(unmatched_loaded) > 0:
        print(f"\n  [WARNING] Some loaded vertices not found in solid:")
        for idx in list(unmatched_loaded)[:5]:  # Show first 5
            coord = loaded_vertices[idx]
            print(f"    Vertex {idx}: [{coord[0]:.2f}, {coord[1]:.2f}, {coord[2]:.2f}]")
    
    return vertex_index_map


def build_enhanced_connectivity_matrix(shape, vertex_coords, vertex_map, 
                                      solid_to_loaded_map, n_vertices, include_plane_eqs=True):
    """
    Build enhanced connectivity matrix by scanning solid face-by-face.
    
    Matrix structure: (N, N+4, depth) where:
    - Columns 0-3: [vertex_index, x, y, z]
    - Columns 4+: Enhanced connectivity data
    - depth = 3: [connectivity, face1_idx, face2_idx]
    - depth = 11: [connectivity, face1_idx, face2_idx, face1_plane(4), face2_plane(4)]
    
    Args:
        shape: OCC TopoDS_Shape
        vertex_coords: Vertex coordinates from solid
        vertex_map: Maps OCC vertex to solid vertex index
        solid_to_loaded_map: Maps solid vertex index to loaded vertex index
        n_vertices: Number of vertices in loaded matrix
        include_plane_eqs: If True, include plane equations (depth=11), else depth=3
        
    Returns:
        connectivity_matrix: (N, N+4, depth) array with enhanced connectivity
    """
    print("\n[BUILD] Building enhanced connectivity matrix...")
    print(f"  - Include plane equations: {include_plane_eqs}")
    
    depth = 11 if include_plane_eqs else 3
    
    # Initialize matrix: (N, N+4, depth)
    conn_matrix = np.zeros((n_vertices, n_vertices + 4, depth), dtype=float)
    
    # Fill in vertex index and coordinates in first layer (columns 0-3, layer 0)
    loaded_vertices = np.zeros((n_vertices, 3))
    for solid_idx, loaded_idx in solid_to_loaded_map.items():
        loaded_vertices[loaded_idx] = vertex_coords[solid_idx]
    
    for i in range(n_vertices):
        conn_matrix[i, 0, 0] = i  # Vertex index
        conn_matrix[i, 1:4, 0] = loaded_vertices[i]  # x, y, z coordinates
    
    # Extract face plane equations
    topo = TopologyExplorer(shape)
    face_list = list(topo.faces())
    face_planes = {}
    
    print(f"  - Extracting plane equations for {len(face_list)} faces...")
    for face_idx, face in enumerate(face_list):
        plane_eq = get_face_plane_equation(face)
        if plane_eq is not None:
            face_planes[face_idx] = plane_eq
        else:
            face_planes[face_idx] = np.array([0, 0, 0, 0])  # Non-planar or error
    
    # Build edge-to-face mapping
    print(f"  - Building edge-to-face mapping...")
    edge_to_faces = {}  # Maps edge to list of face indices
    
    for face_idx, face in enumerate(face_list):
        edge_explorer = TopExp_Explorer(face, TopAbs_EDGE)
        
        while edge_explorer.More():
            edge = edge_explorer.Current()
            
            # Get edge vertices
            vertex_explorer = TopExp_Explorer(edge, TopAbs_VERTEX)
            edge_vertices = []
            
            while vertex_explorer.More():
                vertex = topods.Vertex(vertex_explorer.Current())
                solid_idx = vertex_map.get(vertex, None)
                if solid_idx is not None:
                    loaded_idx = solid_to_loaded_map.get(solid_idx, None)
                    if loaded_idx is not None:
                        edge_vertices.append(loaded_idx)
                vertex_explorer.Next()
            
            if len(edge_vertices) == 2:
                # Create canonical edge key (sorted)
                edge_key = tuple(sorted(edge_vertices))
                
                if edge_key not in edge_to_faces:
                    edge_to_faces[edge_key] = []
                edge_to_faces[edge_key].append(face_idx)
            
            edge_explorer.Next()
    
    # Fill connectivity matrix with edge data
    print(f"  - Filling connectivity matrix...")
    edge_count = 0
    
    for edge_key, face_indices in edge_to_faces.items():
        i, j = edge_key
        edge_count += 1
        
        connectivity = len(face_indices)
        
        # Layer 0: Connectivity value
        conn_matrix[i, j + 4, 0] = connectivity
        conn_matrix[j, i + 4, 0] = connectivity
        
        # Layer 1: First face index
        if len(face_indices) >= 1:
            conn_matrix[i, j + 4, 1] = face_indices[0]
            conn_matrix[j, i + 4, 1] = face_indices[0]
        
        # Layer 2: Second face index (if exists)
        if len(face_indices) >= 2:
            conn_matrix[i, j + 4, 2] = face_indices[1]
            conn_matrix[j, i + 4, 2] = face_indices[1]
        else:
            conn_matrix[i, j + 4, 2] = -1  # No second face
            conn_matrix[j, i + 4, 2] = -1
        
        # Layers 3-6: First face plane equation (a, b, c, d)
        if include_plane_eqs and len(face_indices) >= 1:
            plane1 = face_planes.get(face_indices[0], np.array([0, 0, 0, 0]))
            conn_matrix[i, j + 4, 3:7] = plane1
            conn_matrix[j, i + 4, 3:7] = plane1
        
        # Layers 7-10: Second face plane equation (a, b, c, d)
        if include_plane_eqs and len(face_indices) >= 2:
            plane2 = face_planes.get(face_indices[1], np.array([0, 0, 0, 0]))
            conn_matrix[i, j + 4, 7:11] = plane2
            conn_matrix[j, i + 4, 7:11] = plane2
    
    print(f"  - Processed {edge_count} unique edges")
    
    # Count edges by connectivity
    conn_layer = conn_matrix[:, 4:, 0]
    n_edges_value_1 = np.sum(conn_layer == 1) // 2
    n_edges_value_2 = np.sum(conn_layer == 2) // 2
    n_edges_value_3plus = np.sum(conn_layer >= 3) // 2
    
    print(f"  - Edges with connectivity = 1: {n_edges_value_1}")
    print(f"  - Edges with connectivity = 2: {n_edges_value_2}")
    print(f"  - Edges with connectivity >= 3: {n_edges_value_3plus}")
    
    return conn_matrix


def extract_edges_from_connectivity(conn_matrix, value=2):
    """
    Extract edges from enhanced connectivity matrix where connectivity == value.
    
    Args:
        conn_matrix: (N, N+4, depth) enhanced connectivity matrix
        value: Connectivity value to filter (default 2)
        
    Returns:
        edges: List of (start_coord, end_coord) tuples
    """
    print(f"\n[EXTRACT] Extracting edges with connectivity = {value}...")
    
    n_vertices = conn_matrix.shape[0]
    edges = []
    
    # Extract connectivity layer (layer 0, columns 4+)
    conn_layer = conn_matrix[:, 4:, 0]
    
    # Find all edges with specified value
    for i in range(n_vertices):
        for j in range(i + 1, n_vertices):  # Only upper triangle to avoid duplicates
            if conn_layer[i, j] == value:
                # Get vertex coordinates (columns 1-3, layer 0)
                start_coord = conn_matrix[i, 1:4, 0]
                end_coord = conn_matrix[j, 1:4, 0]
                edges.append((start_coord, end_coord))
    
    print(f"  - Found {len(edges)} edges with connectivity = {value}")
    
    return edges


def extract_faces_for_visualization(shape):
    """
    Extract face polygons from solid shape for visualization.
    
    Args:
        shape: TopoDS_Shape solid
        
    Returns:
        face_polygons: List of dictionaries with face polygon data
    """
    print(f"\n[EXTRACT] Extracting face polygons from solid...")
    
    from OCC.Core.BRep import BRep_Tool
    from OCC.Core.TopExp import TopExp_Explorer
    from OCC.Core.TopAbs import TopAbs_WIRE, TopAbs_EDGE, TopAbs_VERTEX
    from OCC.Core.BRepAdaptor import BRepAdaptor_Curve
    from OCC.Core.TopoDS import topods
    from OCC.Core.BRepTools import BRepTools_WireExplorer
    
    face_polygons = []
    topo = TopologyExplorer(shape)
    
    for face in topo.faces():
        face_dict = {
            'outer_boundary': [],
            'inner_boundaries': []
        }
        
        # Collect all wires on this face
        wires_data = []
        wire_explorer = TopExp_Explorer(face, TopAbs_WIRE)
        
        while wire_explorer.More():
            wire = topods.Wire(wire_explorer.Current())
            
            # Extract vertices from wire edges in order using BRepTools_WireExplorer
            wire_verts = []
            wire_exp = BRepTools_WireExplorer(wire)
            
            while wire_exp.More():
                edge = wire_exp.Current()
                
                # Get the current vertex (respects edge orientation in wire)
                vertex = wire_exp.CurrentVertex()
                pnt = BRep_Tool.Pnt(vertex)
                wire_verts.append([pnt.X(), pnt.Y(), pnt.Z()])
                
                wire_exp.Next()
            
            if len(wire_verts) > 0:
                wires_data.append(np.array(wire_verts))
            
            wire_explorer.Next()
        
        # The outer wire is typically the first one or the one with largest perimeter
        if len(wires_data) > 0:
            if len(wires_data) == 1:
                face_dict['outer_boundary'] = wires_data[0]
            else:
                # Find outer wire by perimeter (largest is outer)
                perimeters = []
                for wire_verts in wires_data:
                    perimeter = 0
                    for i in range(len(wire_verts)):
                        v1 = wire_verts[i]
                        v2 = wire_verts[(i+1) % len(wire_verts)]
                        perimeter += np.linalg.norm(v2 - v1)
                    perimeters.append(perimeter)
                
                outer_idx = np.argmax(perimeters)
                face_dict['outer_boundary'] = wires_data[outer_idx]
                
                # All others are inner boundaries (holes)
                for i, wire_verts in enumerate(wires_data):
                    if i != outer_idx:
                        face_dict['inner_boundaries'].append(wire_verts)
        
        if len(face_dict['outer_boundary']) > 0:
            face_polygons.append(face_dict)
    
    print(f"  - Extracted {len(face_polygons)} face polygons")
    
    return face_polygons



def plot_comparison(edges_value_2, edges_value_1, face_polygons, conn_matrix, output_file='solid_connectivity_validation.png', show_plot=True):
    """
    Plot comparison between edges from connectivity matrix and original solid.
    
    Args:
        edges_value_2: List of edge tuples with connectivity = 2
        edges_value_1: List of edge tuples with connectivity = 1
        face_polygons: List of face polygon vertices from solid
        conn_matrix: Connectivity matrix for vertex coordinates
        output_file: Output filename for plot
    """
    print(f"\n[PLOT] Creating comparison visualization...")
    
    fig = plt.figure(figsize=(16, 7))
    
    # Plot 1: Edges from connectivity matrix
    ax1 = fig.add_subplot(121, projection='3d')
    ax1.set_title(f'Edges from Connectivity Matrix\nBlue: connectivity=2 ({len(edges_value_2)}), Red: connectivity=1 ({len(edges_value_1)})', 
                  fontsize=12, fontweight='bold')
    
    # Plot edges with connectivity = 2 (manifold edges)
    for start, end in edges_value_2:
        ax1.plot([start[0], end[0]], 
                [start[1], end[1]], 
                [start[2], end[2]], 
                'b-', linewidth=1.5, alpha=0.7, label='conn=2' if start is edges_value_2[0][0] else '')
    
    # Plot edges with connectivity = 1 (boundary/non-manifold edges)
    for start, end in edges_value_1:
        ax1.plot([start[0], end[0]], 
                [start[1], end[1]], 
                [start[2], end[2]], 
                'r-', linewidth=2, alpha=0.8, label='conn=1' if start is edges_value_1[0][0] else '')
    
    # Plot vertices
    vertices = conn_matrix[:, 1:4, 0]  # Layer 0 for coordinates
    ax1.scatter(vertices[:, 0], vertices[:, 1], vertices[:, 2], 
               c='green', s=20, alpha=0.6, label='Vertices')
    
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.set_zlabel('Z')
    
    # Create custom legend
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], color='b', linewidth=2, label=f'Connectivity = 2 ({len(edges_value_2)})'),
        Line2D([0], [0], color='r', linewidth=2, label=f'Connectivity = 1 ({len(edges_value_1)})'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='g', markersize=8, label='Vertices')
    ]
    ax1.legend(handles=legend_elements, loc='upper left')
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Original solid faces with hole edges highlighted
    ax2 = fig.add_subplot(122, projection='3d')
    ax2.set_title(f'Original Solid Faces\n({len(face_polygons)} faces)', 
                  fontsize=12, fontweight='bold')
    
    # Draw faces with transparency and edges
    face_collection = []
    colors = []
    
    for face_dict in face_polygons:
        # Extract outer boundary from dictionary
        outer_boundary = face_dict.get('outer_boundary', [])
        
        if len(outer_boundary) >= 3:
            # Draw outer boundary edges
            for i in range(len(outer_boundary)):
                v1 = outer_boundary[i]
                v2 = outer_boundary[(i + 1) % len(outer_boundary)]
                ax2.plot([v1[0], v2[0]], 
                        [v1[1], v2[1]], 
                        [v1[2], v2[2]], 
                        'darkblue', linewidth=1.5, alpha=0.7)
            
            # Add face to collection for rendering
            face_collection.append(outer_boundary)
            colors.append([0.5, 0.8, 1.0, 0.3])  # Light blue with transparency
        
        # Draw inner boundaries (holes) if present
        inner_boundaries = face_dict.get('inner_boundaries', [])
        for hole in inner_boundaries:
            if len(hole) >= 3:
                for i in range(len(hole)):
                    v1 = hole[i]
                    v2 = hole[(i + 1) % len(hole)]
                    ax2.plot([v1[0], v2[0]], 
                            [v1[1], v2[1]], 
                            [v1[2], v2[2]], 
                            'red', linewidth=2, alpha=0.8)
    
    # Render faces with transparency
    if len(face_collection) > 0:
        poly_collection = Poly3DCollection(face_collection, 
                                           facecolors=colors,
                                           edgecolors='none',
                                           linewidths=0)
        ax2.add_collection3d(poly_collection)
    
    # Set axis limits based on vertices
    if len(vertices) > 0:
        margin = 0.1 * np.ptp(vertices, axis=0)
        ax2.set_xlim([vertices[:, 0].min() - margin[0], vertices[:, 0].max() + margin[0]])
        ax2.set_ylim([vertices[:, 1].min() - margin[1], vertices[:, 1].max() + margin[1]])
        ax2.set_zlim([vertices[:, 2].min() - margin[2], vertices[:, 2].max() + margin[2]])
    
    ax2.set_xlabel('X')
    ax2.set_ylabel('Y')
    ax2.set_zlabel('Z')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"  - Saved visualization to: {output_file}")
    if show_plot:
        plt.show()
    plt.close()


def save_solid_connectivity_matrix(conn_matrix, seed, output_dir="Output"):
    """Save the solid connectivity matrix to file"""
    filename = os.path.join(output_dir, f"solid_connectivity_matrix_seed_{seed}.npy")
    
    print(f"\n[SAVE] Saving solid connectivity matrix to: {filename}")
    np.save(filename, conn_matrix)
    print(f"  - Matrix shape: {conn_matrix.shape}")
    print(f"  - Saved successfully")


def main():
    """Main execution function"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Build enhanced connectivity matrix from BREP solid file.'
    )
    parser.add_argument(
        '--seed', type=int, default=30,
        help='Random seed for solid (must match existing BREP file, default: 30)'
    )
    parser.add_argument(
        '--input-dir', type=str, default='Output',
        help='Directory containing input files (default: Output)'
    )
    parser.add_argument(
        '--output-dir', type=str, default='Output',
        help='Directory to save output files (default: Output)'
    )
    parser.add_argument(
        '--no-plane-equations', action='store_true',
        help='Disable plane equations (depth=3 instead of depth=11)'
    )
    parser.add_argument(
        '--no-graphics', action='store_true',
        help='Disable graphics/plot windows (still saves PNG file)'
    )
    
    args = parser.parse_args()
    
    print("="*70)
    print("ENHANCED SOLID CONNECTIVITY MATRIX BUILDER")
    print("="*70)
    
    # Configuration
    seed = args.seed
    input_dir = args.input_dir
    output_dir = args.output_dir
    include_plane_eqs = not args.no_plane_equations
    
    # Step 1: Load connectivity matrices
    (loaded_vertices, top_matrix, front_matrix, side_matrix,
     units, scale_real, scale_drawing) = load_connectivity_matrices(seed, input_dir)
    
    n_vertices = len(loaded_vertices)
    print(f"\n  Total vertices in loaded matrices: {n_vertices}")
    
    # Step 2: Load original solid from BREP
    solid_shape = load_original_solid(seed, input_dir)
    
    # Step 3: Extract vertices from solid
    solid_vertices, vertex_map = extract_vertex_coordinates(solid_shape, tolerance=1e-6)
    
    # Step 4: Match solid vertices to loaded vertices
    solid_to_loaded_map = match_vertices_to_loaded(solid_vertices, loaded_vertices, tolerance=0.1)
    
    # Step 5: Build enhanced connectivity matrix
    solid_conn_matrix = build_enhanced_connectivity_matrix(
        solid_shape, solid_vertices, vertex_map, solid_to_loaded_map, n_vertices, 
        include_plane_eqs=include_plane_eqs)
    
    # Step 6: Extract edges where connectivity = 2 and 1
    edges_value_2 = extract_edges_from_connectivity(solid_conn_matrix, value=2)
    edges_value_1 = extract_edges_from_connectivity(solid_conn_matrix, value=1)
    
    # Step 7: Prepare faces for visualization
    face_polygons = extract_faces_for_visualization(solid_shape)
    
    # Step 8: Plot comparison
    plot_comparison(edges_value_2, edges_value_1, face_polygons, solid_conn_matrix,
                   output_file=f'solid_connectivity_validation_seed_{seed}.png',
                   show_plot=not args.no_graphics)
    
    # Step 9: Save enhanced connectivity matrix
    save_solid_connectivity_matrix(solid_conn_matrix, seed, output_dir)
    
    # Summary statistics
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print(f"  Vertices: {n_vertices}")
    print(f"  Edges with connectivity = 2: {len(edges_value_2)} (manifold edges)")
    print(f"  Edges with connectivity = 1: {len(edges_value_1)} (boundary/hole edges)")
    print(f"  Faces in original solid: {len(face_polygons)}")
    print(f"  Connectivity matrix shape: {solid_conn_matrix.shape}")
    depth_desc = "11 (conn + 2 face idx + 8 plane params)" if include_plane_eqs else "3 (conn + 2 face idx)"
    print(f"  Matrix structure: (N, N+4, {solid_conn_matrix.shape[2]})")
    print(f"  Matrix depth: {depth_desc}")
    print(f"    Layer 0: Connectivity value")
    print(f"    Layer 1: First face index")
    print(f"    Layer 2: Second face index (-1 if none)")
    if include_plane_eqs:
        print(f"    Layers 3-6: First face plane (a,b,c,d)")
        print(f"    Layers 7-10: Second face plane (a,b,c,d)")
    print("="*70)


if __name__ == "__main__":
    main()
