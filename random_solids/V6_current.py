def print_merged_connectivity_matrix(reconstructed_vertices, front_proj, side_proj, front_conn, side_conn, top_proj=None, top_conn=None):
    """
    Build and print the merged connectivity matrix using available projections and connectivity matrices.
    If top_proj/top_conn are not provided, use zeros.
    """
    import numpy as np
    N = len(reconstructed_vertices)
    if top_proj is None:
        top_proj = np.zeros((N, 2))
    if top_conn is None:
        top_conn = np.zeros((N, N))
    merged_matrix = build_merged_connectivity_matrix(
        reconstructed_vertices, top_proj, front_proj, side_proj,
        top_conn, front_conn, side_conn
    )
    print("\n[MERGED CONNECTIVITY MATRIX]")
    print(f"Shape: {merged_matrix.shape}")
    print("Format: V# | TopProj | FrontProj | SideProj | EdgeConnCounts")
    for i in range(N):
        vinfo = merged_matrix[i, :7]
        edge_counts = merged_matrix[i, 7:]
        print(f"V{i:2d}: Top({vinfo[1]:7.3f},{vinfo[2]:7.3f}) Front({vinfo[3]:7.3f},{vinfo[4]:7.3f}) Side({vinfo[5]:7.3f},{vinfo[6]:7.3f}) | Edges: {edge_counts.astype(int)}")
import traceback
import argparse
# Ensure build_merged_connectivity_matrix is defined
def build_merged_connectivity_matrix(reconstructed_vertices, top_proj, front_proj, side_proj, top_conn, front_conn, side_conn):
    """
    Build merged connectivity matrix from three views.
    For each pair of reconstructed vertices, count in how many views the edge exists.
    Robustly matches reconstructed vertices to projections and view connectivity matrices.
    """
    import numpy as np
    N = len(reconstructed_vertices)
    merged = np.zeros((N, 7 + N))
    # Fill vertex info
    for i, v in enumerate(reconstructed_vertices):
        merged[i, 0] = i
        merged[i, 1:3] = top_proj[i]
        merged[i, 3:5] = front_proj[i]
        merged[i, 5:7] = side_proj[i]

    # Helper to find index in projection list
    def find_proj_idx(proj_list, target):
        for idx, p in enumerate(proj_list):
            if np.allclose(p, target, atol=1e-8):
                return idx
        return None

    # For each pair of reconstructed vertices, check edge existence in each view
    for i in range(N):
        for j in range(i+1, N):
            conn_count = 0
            # Top view
            idx_top_i = find_proj_idx(top_proj, top_proj[i])
            idx_top_j = find_proj_idx(top_proj, top_proj[j])
            if idx_top_i is not None and idx_top_j is not None and idx_top_i < top_conn.shape[0] and idx_top_j < top_conn.shape[1]:
                if top_conn[idx_top_i, idx_top_j] > 0:
                    conn_count += 1
            # Front view
            idx_front_i = find_proj_idx(front_proj, front_proj[i])
            idx_front_j = find_proj_idx(front_proj, front_proj[j])
            if idx_front_i is not None and idx_front_j is not None and idx_front_i < front_conn.shape[0] and idx_front_j < front_conn.shape[1]:
                if front_conn[idx_front_i, idx_front_j] > 0:
                    conn_count += 1
            # Side view
            idx_side_i = find_proj_idx(side_proj, side_proj[i])
            idx_side_j = find_proj_idx(side_proj, side_proj[j])
            if idx_side_i is not None and idx_side_j is not None and idx_side_i < side_conn.shape[0] and idx_side_j < side_conn.shape[1]:
                if side_conn[idx_side_i, idx_side_j] > 0:
                    conn_count += 1
            merged[i, 7 + j] = conn_count
            merged[j, 7 + i] = conn_count
    return merged

import os
import sys

# Example usage after vertex reconstruction:
# reconstructed_vertices = ... # list of [x, y, z]
# top_proj = ... # list of [x, y] for each vertex
# front_proj = ... # list of [y, z] for each vertex
# side_proj = ... # list of [x, z] for each vertex
# top_conn = ... # connectivity matrix (NxN or smaller)
# front_conn = ...
# side_conn = ...
# merged_matrix = build_merged_connectivity_matrix(reconstructed_vertices, top_proj, front_proj, side_proj, top_conn, front_conn, side_conn)
import matplotlib.patches as patches
from shapely.geometry import Polygon, Point
from OCC.Core.gp import gp_Trsf  # noqa: F401
from OCC.Core.TopLoc import TopLoc_Location  # noqa: F401
from OCC.Core.TopExp import TopExp_Explorer
from OCC.Core.TopoDS import topods
from OCC.Core.TopAbs import TopAbs_SHELL, TopAbs_EDGE, TopAbs_VERTEX, TopAbs_WIRE  # noqa: F401, E501
from OCC.Core.BRep import BRep_Tool
from OCC.Core.BRepAdaptor import BRepAdaptor_Curve
from OCC.Core.IFSelect import IFSelect_RetDone
from Reconstruction.config_system import ConfigurationManager, create_default_config, load_config
from Reconstruction.edge_reconstruction import reconstruct_edges_from_views
from Reconstruction.Base_Solid import build_solid_with_polygons
from opencascade import get_face_normal_from_opencascade, extract_and_visualize_faces, extract_wire_vertices_in_sequence, OPENCASCADE_AVAILABLE
from unified_summary import (create_unified_summary, print_summary_info,
                             save_summary_to_file, save_summary_to_numpy,
                             visualize_adjacency_matrix)
from V6_Sept_25 import save_solid_as_step

# ...existing code...

if __name__ == "__main__":
    pass  # Main block intentionally left empty until workflow variables are set
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import CheckButtons

def plot_merged_connectivity(reconstructed_vertices, merged_matrix, original_polygon=None):
    """
    Plot reconstructed vertices and edges from merged connectivity matrix.
    Toggle edges by connectivity (1, 2, 3) with color coding.
    Optionally plot original polygon.
    """
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.set_title('Merged Connectivity Matrix: Reconstructed Vertices and Edges')
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)

    # Plot original polygon if provided
    if original_polygon is not None:
        if hasattr(original_polygon, 'exterior'):
            x, y = original_polygon.exterior.xy
            ax.plot(x, y, color='gray', linewidth=2, label='Original Polygon')
    # Plot reconstructed vertices
    verts = np.array(reconstructed_vertices)
    ax.scatter(verts[:, 0], verts[:, 1], color='red', s=60, label='Reconstructed Vertices', zorder=3)
    for i, v in enumerate(verts):
        ax.text(v[0], v[1], f'V{i}', fontsize=9, color='darkred', ha='right', va='bottom')
    # Prepare edge lines by connectivity
    N = len(reconstructed_vertices)
    edge_lines = {1: [], 2: [], 3: []}
    colors = {1: 'yellow', 2: 'gray', 3: 'black'}
    for i in range(N):
        for j in range(N):
            conn = int(merged_matrix[i, 7 + j])
            if conn in edge_lines and i < j and conn > 0:
                edge_lines[conn].append(((verts[i, 0], verts[i, 1]), (verts[j, 0], verts[j, 1])))
    # Plot all edges, store line objects for toggling
    line_objs = {1: [], 2: [], 3: []}
    for conn in [1, 2, 3]:
        for seg in edge_lines[conn]:
            line, = ax.plot([seg[0][0], seg[1][0]], [seg[0][1], seg[1][1]],
                            color=colors[conn], linewidth=2, alpha=0.8, label=f'Conn {conn}')
            line_objs[conn].append(line)
    # Remove duplicate legend entries
    handles, labels = ax.get_legend_handles_labels()
    unique = dict(zip(labels, handles))
    ax.legend(unique.values(), unique.keys(), loc='upper left')
    # Add toggle buttons for edge classes
    button_ax = plt.axes([0.01, 0.7, 0.15, 0.15], facecolor='lightgoldenrodyellow')
    button_labels = ['Conn 1 (Yellow)', 'Conn 2 (Gray)', 'Conn 3 (Black)']
    initial_states = [True, True, True]
    check = CheckButtons(button_ax, button_labels, initial_states)
    def toggle_visibility(label):
        idx = button_labels.index(label)
        conn = idx + 1
        for line in line_objs[conn]:
            line.set_visible(not line.get_visible())
        plt.draw()
    check.on_clicked(toggle_visibility)
    plt.tight_layout()
    # plt.show()  # Removed to prevent empty plot

    # Build merged matrix and plot
    merged_matrix = build_merged_connectivity_matrix(
        reconstructed_vertices, top_proj, front_proj, side_proj,
        top_conn, front_conn, side_conn
    )
    plot_merged_connectivity(reconstructed_vertices, merged_matrix, original_polygon=original_polygon)
    # Plot original polygon if provided
    if original_polygon is not None:
        if hasattr(original_polygon, 'exterior'):
            x, y = original_polygon.exterior.xy
            ax.plot(x, y, color='gray', linewidth=2, label='Original Polygon')
    # Plot reconstructed vertices
    verts = np.array(reconstructed_vertices)
    ax.scatter(verts[:, 0], verts[:, 1], color='red', s=60, label='Reconstructed Vertices', zorder=3)
    for i, v in enumerate(verts):
        ax.text(v[0], v[1], f'V{i}', fontsize=9, color='darkred', ha='right', va='bottom')
    # Prepare edge lines by connectivity
    N = len(reconstructed_vertices)
    edge_lines = {1: [], 2: [], 3: []}
    colors = {1: 'yellow', 2: 'gray', 3: 'black'}
    for i in range(N):
        for j in range(N):
            conn = int(merged_matrix[i, 7 + j])
            if conn in edge_lines and i < j and conn > 0:
                edge_lines[conn].append(((verts[i, 0], verts[i, 1]), (verts[j, 0], verts[j, 1])))
    # Plot all edges, store line objects for toggling
    line_objs = {1: [], 2: [], 3: []}
    for conn in [1, 2, 3]:
        for seg in edge_lines[conn]:
            line, = ax.plot([seg[0][0], seg[1][0]], [seg[0][1], seg[1][1]],
                            color=colors[conn], linewidth=2, alpha=0.8, label=f'Conn {conn}')
            line_objs[conn].append(line)
    # Remove duplicate legend entries
    handles, labels = ax.get_legend_handles_labels()
    unique = dict(zip(labels, handles))
    ax.legend(unique.values(), unique.keys(), loc='upper left')
    # Add toggle buttons for edge classes
    button_ax = plt.axes([0.01, 0.7, 0.15, 0.15], facecolor='lightgoldenrodyellow')
    button_labels = ['Conn 1 (Yellow)', 'Conn 2 (Gray)', 'Conn 3 (Black)']
    initial_states = [True, True, True]
    check = CheckButtons(button_ax, button_labels, initial_states)
    def toggle_visibility(label):
        idx = button_labels.index(label)
        conn = idx + 1
        for line in line_objs[conn]:
            line.set_visible(not line.get_visible())
        plt.draw()
    check.on_clicked(toggle_visibility)
    plt.tight_layout()
    # plt.show()  # Removed to prevent empty plot


try:
    from OCC.Core.gp import gp_Vec
    from OCC.Core.BRepPrimAPI import BRepPrimAPI_MakeBox
    from OCC.Core.TopExp import TopExp_Explorer
    from OCC.Core.TopAbs import TopAbs_FACE
    from OCC.Core.TopoDS import topods
    from OCC.Core.BRepBuilderAPI import BRepBuilderAPI_Transform
    from OCC.Core.BRepAlgoAPI import BRepAlgoAPI_Cut
    OPENCASCADE_AVAILABLE = True
    # Try to import TopExp for vertex extraction
    try:
        from OCC.Core.TopExp import topexp, topexp_Vertices
        TOPEXP_AVAILABLE = True
    except Exception:
        TOPEXP_AVAILABLE = False
except Exception:
    OPENCASCADE_AVAILABLE = False

def build_solid_with_polygons_test(config, quiet=False):
    from Reconstruction.Base_Solid import build_solid_with_polygons
    seed = config.seed
    print(f"[DEBUG] Calling build_solid_with_polygons(config, seed={seed}, quiet={quiet}) as test...")
    original = build_solid_with_polygons(seed, quiet)
    # (You can add your custom boolean operations here if needed)
    #return original
    # Create box at (0,0,0) of size (60,50,60)
    # box = BRepPrimAPI_MakeBox(55, 25, 35).Shape()
    # # Move box to (10,0,0)
    # trsf = gp_Trsf()
    # trsf.SetTranslation(gp_Vec(10, 35, 0))
    # moved_box = BRepBuilderAPI_Transform(box, trsf, True).Shape()

    # # Subtract box from original
    # cut = BRepAlgoAPI_Cut(original, moved_box)
    # cut_shape = cut.Shape()

    # # Create box at (0,0,0) of size (60,50,60)
    # box = BRepPrimAPI_MakeBox(30, 27, 45).Shape()
    # # Move box to (10,0,0)
    # trsf = gp_Trsf()
    # trsf.SetTranslation(gp_Vec(35, 0, 0))
    # moved_box = BRepBuilderAPI_Transform(box, trsf, True).Shape()

    # # Subtract box from original
    # cut = BRepAlgoAPI_Cut(cut_shape, moved_box)
    # cut_shape1 = cut.Shape()

    # # Create box at (0,0,0) of size (60,50,60)
    # box = BRepPrimAPI_MakeBox(55, 10, 15).Shape()
    # # Move box to (10,25,0)
    # trsf = gp_Trsf()
    # trsf.SetTranslation(gp_Vec(10, 25, 0))
    # moved_box = BRepBuilderAPI_Transform(box, trsf, True).Shape()

    # # Subtract box from original
    # cut = BRepAlgoAPI_Cut(cut_shape1, moved_box)
    # cut_shape2 = cut.Shape()

    # # Create box at (0,0,0) of size (60,50,60)
    # box = BRepPrimAPI_MakeBox(20, 10, 35).Shape()
    # # Move box to (10,25,0)
    # trsf = gp_Trsf()
    # trsf.SetTranslation(gp_Vec(45, 25, 10))
    # moved_box = BRepBuilderAPI_Transform(box, trsf, True).Shape()

    # # Subtract box from original
    # cut = BRepAlgoAPI_Cut(cut_shape2, moved_box)
    # cut_shape1 = cut.Shape()

    # # Create box at (0,0,0) of size (60,50,60)
    # box = BRepPrimAPI_MakeBox(25, 10, 25).Shape()
    # # Move box to (10,25,0)
    # trsf = gp_Trsf()
    # trsf.SetTranslation(gp_Vec(10, 27, 10))
    # moved_box = BRepBuilderAPI_Transform(box, trsf, True).Shape()

    # # Subtract box from original
    # cut = BRepAlgoAPI_Cut(cut_shape1, moved_box)
    # cut_shape2 = cut.Shape()

    # # Create box at (0,0,0) of size (60,50,60)
    # box = BRepPrimAPI_MakeBox(20, 10, 25).Shape()
    # # Move box to (10,25,0)
    # trsf = gp_Trsf()
    # trsf.SetTranslation(gp_Vec(30, 27, 29))
    # moved_box = BRepBuilderAPI_Transform(box, trsf, True).Shape()

    # # Subtract box from original
    # cut = BRepAlgoAPI_Cut(cut_shape2, moved_box)
    # cut_shape1 = cut.Shape()

    # # Create box at (0,0,0) of size (60,50,60)
    # box = BRepPrimAPI_MakeBox(20, 10, 25).Shape()
    # # Move box to (10,25,0)
    # trsf = gp_Trsf()
    # trsf.SetTranslation(gp_Vec(36, 26, 15))
    # moved_box = BRepBuilderAPI_Transform(box, trsf, True).Shape()

    # # Subtract box from original
    # cut = BRepAlgoAPI_Cut(cut_shape1, moved_box)
    # cut_shape2 = cut.Shape()

    return original


# --- OpenCASCADE face extraction helpers ---


def get_face_normal_from_opencascade(face):
    """
    Extract the correct face normal from OpenCASCADE face using multiple robust methods.
    Tries several approaches to get the correct outward-pointing normal:
      1. GeomLProp_SLProps with orientation
      2. Surface derivatives with orientation
      3. BRepGProp_Face method
      4. Geometric analysis fallback
    """
    try:
        from OCC.Core.BRepAdaptor import BRepAdaptor_Surface
        from OCC.Core.TopAbs import TopAbs_FORWARD, TopAbs_REVERSED
        from OCC.Core.GeomLProp import GeomLProp_SLProps
        from OCC.Core.gp import gp_Pnt, gp_Vec
        
        # Get the face orientation from topology
        face_orientation = face.Orientation()
        orientation_str = str(face_orientation).split('.')[-1] if hasattr(face_orientation, '__str__') else str(face_orientation)
        print(f"        Face orientation: {orientation_str}")
        
        # Get the surface adaptor
        surface = BRepAdaptor_Surface(face)
        
        # Get parameter bounds
        u_min = surface.FirstUParameter()
        u_max = surface.LastUParameter()
        v_min = surface.FirstVParameter()
        v_max = surface.LastVParameter()
        
        # Use multiple parameter points to get robust normal
        u_mid = (u_min + u_max) / 2.0
        v_mid = (v_min + v_max) / 2.0
        
        print(f"        Parameter bounds: U[{u_min:.3f}, {u_max:.3f}], V[{v_min:.3f}, {v_max:.3f}]")
        print(f"        Using parameters: U={u_mid:.3f}, V={v_mid:.3f}")
        
        # Method 2: Surface derivatives with proper orientation handling
        try:
            # Get point and derivatives at midpoint
            point = surface.Value(u_mid, v_mid)
            d1u = surface.DN(u_mid, v_mid, 1, 0)  # First derivative in U direction
            d1v = surface.DN(u_mid, v_mid, 0, 1)  # First derivative in V direction
            
            print(f"        Surface point: ({point.X():.3f}, {point.Y():.3f}, {point.Z():.3f})")
            print(f"        dU vector: ({d1u.X():.3f}, {d1u.Y():.3f}, {d1u.Z():.3f})")
            print(f"        dV vector: ({d1v.X():.3f}, {d1v.Y():.3f}, {d1v.Z():.3f})")
            
            # Calculate normal as cross product of derivatives
            normal_vec = d1u.Crossed(d1v)
            
            if normal_vec.Magnitude() > 1e-10:
                normal_vec.Normalize()
                
                # Apply orientation correction based on face topology
                orientation_multiplier = 1.0
                if face_orientation == TopAbs_REVERSED:
                    orientation_multiplier = -1.0
                    print(f"        REVERSED face - flipping derivative normal")
                
                face_normal = np.array([
                    normal_vec.X() * orientation_multiplier,
                    normal_vec.Y() * orientation_multiplier,
                    normal_vec.Z() * orientation_multiplier
                ])
                
                normal_print = f"[{face_normal[0]:.6f}, {face_normal[1]:.6f}, {face_normal[2]:.6f}]"
                print(f"        Derivative normal: {normal_print}")
                
                return face_normal
                
        except Exception as e:
            print(f"        Surface derivative method failed: {e}")
        
        # Method 3: Try BRepGProp_Face as fallback
        try:
            from OCC.Core.BRepGProp import BRepGProp_Face
            
            # This method might work differently
            face_props = BRepGProp_Face(face)
            
            point = gp_Pnt()
            normal_vec = gp_Vec()
            
            # Try to get normal at parameter center
            face_props.Normal(u_mid, v_mid, point, normal_vec)
            
            if normal_vec.Magnitude() > 1e-10:
                face_normal = np.array([normal_vec.X(), normal_vec.Y(), normal_vec.Z()])
                face_normal = face_normal / np.linalg.norm(face_normal)
                
                print(f"        BRepGProp normal: [{face_normal[0]:.6f}, {face_normal[1]:.6f}, {face_normal[2]:.6f}]")
                return face_normal
                
        except Exception as e:
            print(f"        BRepGProp method failed: {e}")
        
        print(f"        ERROR: All normal calculation methods failed!")
        return None
            
    except Exception as e:
        print(f"        CRITICAL ERROR: Could not extract OpenCASCADE normal: {e}")
        traceback.print_exc()
        return None


def extract_wire_vertices_in_sequence(wire, wire_id):
    """Extract vertices from a wire using simplified orientation-based logic.
    
    Uses the corrected approach from the vertex extractor:
    - Forward wire: select start vertex from each edge
    - Reversed wire: select end vertex from each edge
    - Duplicate first vertex at end to close the polygon
    
    Args:
        wire: OpenCASCADE wire object
        wire_id: Wire identifier for debugging
    
    Returns:
        list: Ordered list of [x, y, z] vertex coordinates
    """
    vertices = []
    
    try:
        print(f"          Traversing Wire {wire_id} edges...")
        
        # Import needed constants
        from OCC.Core.TopAbs import TopAbs_FORWARD, TopAbs_REVERSED
        
        # Get wire orientation - this determines vertex selection strategy
        wire_orientation = wire.Orientation()
        is_wire_reversed = (wire_orientation == TopAbs_REVERSED)
        
        wire_orientation_str = "REVERSED" if is_wire_reversed else "FORWARD"
        print(f"            Wire {wire_id} orientation: "
              f"{wire_orientation_str}")
        
        # Method 1: Use BRepTools_WireExplorer for proper wire traversal
        # This is the recommended way to traverse wire edges in correct order
        if TOPEXP_AVAILABLE:
            try:
                from OCC.Core.BRepTools import BRepTools_WireExplorer
                from OCC.Core.TopoDS import TopoDS_Vertex
                
                print("            ✓ Using BRepTools_WireExplorer for proper wire traversal")
                
                # Create wire explorer - this respects wire orientation and edge order
                wire_explorer = BRepTools_WireExplorer(topods.Wire(wire))
                vertex_sequence = []
                
                edge_count = 0
                while wire_explorer.More():
                    edge = wire_explorer.Current()
                    
                    # Get the current vertex from the wire explorer
                    # This gives us vertices in the correct wire traversal order
                    current_vertex_shape = wire_explorer.CurrentVertex()
                    current_vertex = topods.Vertex(current_vertex_shape)
                    
                    # Extract coordinates
                    pnt = BRep_Tool.Pnt(current_vertex)
                    vertex_coords = [pnt.X(), pnt.Y(), pnt.Z()]
                    
                    # Add to sequence
                    vertex_sequence.append(vertex_coords)
                    
                    print(f"              Edge {edge_count}: vertex ({vertex_coords[0]:.1f},"
                          f"{vertex_coords[1]:.1f},{vertex_coords[2]:.1f})")
                    
                    wire_explorer.Next()
                    edge_count += 1
                
                print(f"            ✓ Wire traversal complete: {len(vertex_sequence)} vertices")
                vertices = vertex_sequence
                
            except Exception as e:
                print(f"            ✗ BRepTools_WireExplorer failed: {e}")
                # Fallback to simple TopExp method
                try:
                    print("            → Falling back to simple edge traversal")
                    edge_explorer = TopExp_Explorer(wire, TopAbs_EDGE)
                    vertex_sequence = []
                    
                    while edge_explorer.More():
                        edge = edge_explorer.Current()
                        
                        # Just get all vertices from all edges
                        vertex_explorer = TopExp_Explorer(edge, TopAbs_VERTEX)
                        while vertex_explorer.More():
                            vertex = topods.Vertex(vertex_explorer.Current())
                            pnt = BRep_Tool.Pnt(vertex)
                            v = [pnt.X(), pnt.Y(), pnt.Z()]
                            vertex_sequence.append(v)
                            vertex_explorer.Next()
                        
                        edge_explorer.Next()
                    
                    # Remove duplicates while preserving order
                    seen = set()
                    vertices = []
                    for v in vertex_sequence:
                        v_tuple = tuple(np.round(v, 6))
                        if v_tuple not in seen:
                            vertices.append(v)
                            seen.add(v_tuple)
                    
                    print(f"            ✓ Fallback method: {len(vertices)} vertices")
                    
                except Exception as e2:
                    print(f"            ✗ All methods failed: {e2}")
                    vertices = []
        
        # Ensure vertices list is closed for polygon formation
        if vertices and len(vertices) > 0:
            # Add closing vertex if not already closed
            first_vertex = vertices[0]
            last_vertex = vertices[-1]
            
            if np.linalg.norm(np.array(first_vertex) - np.array(last_vertex)) > 1e-6:
                vertices.append(first_vertex)
                print(f"            Added closing vertex to complete wire loop")
            
            # Display final sequence
            vertex_coords = " → ".join([
                f"({v[0]:.1f},{v[1]:.1f},{v[2]:.1f})"
                for v in vertices
            ])
            print(f"            FINAL: {vertex_coords}")
        
        # Fallback: Basic edge traversal if all methods fail
        if not vertices:
            print("            Using basic fallback edge traversal...")
            
            edge_explorer = TopExp_Explorer(wire, TopAbs_EDGE)
            vertex_list = []
            
            while edge_explorer.More():
                edge = edge_explorer.Current()
                
                # Get vertices from edge
                vertex_explorer = TopExp_Explorer(edge, TopAbs_VERTEX)
                edge_vertices = []
                
                while vertex_explorer.More():
                    vertex = topods.Vertex(vertex_explorer.Current())
                    pnt = BRep_Tool.Pnt(vertex)
                    v = [pnt.X(), pnt.Y(), pnt.Z()]
                    edge_vertices.append(v)
                    vertex_explorer.Next()
                
                vertex_list.extend(edge_vertices)
                edge_explorer.Next()
            
            # Remove duplicates while preserving order
            seen = set()
            for v in vertex_list:
                v_tuple = tuple(np.round(v, 6))
                if v_tuple not in seen:
                    vertices.append(v)
                    seen.add(v_tuple)
            
            print(f"            ✓ Basic fallback: {len(vertices)} vertices")
    
    except Exception as e:
        print(f"          ✗ Error extracting vertices from wire {wire_id}: "
              f"{e}")
        vertices = []
    
    return vertices


def extract_and_visualize_faces(solid, visualize=False):
    """
    Extract face data from an OpenCASCADE solid and optionally visualize in 3D.
    Returns a list of face data dicts. If visualize=True, also plots the solid.
    """
    print(f"[DEBUG] extract_and_visualize_faces called: solid={solid is not None}, visualize={visualize}")
    print(f"[DEBUG] OPENCASCADE_AVAILABLE={OPENCASCADE_AVAILABLE}")
    if not OPENCASCADE_AVAILABLE or solid is None:
        print(f"[DEBUG] Returning empty list - OpenCASCADE not available or solid is None")
        return []
    faces = []
    all_face_data = []
    print("  Traversing BRep topology: Solid -> Shells -> Faces -> Wires -> Edges -> Vertices")
    shell_explorer = TopExp_Explorer(solid, TopAbs_SHELL)
    shell_count = 0
    while shell_explorer.More():
        shell_count += 1
        shell_explorer.Next()
    print(f"  Found {shell_count} shells in solid")
    if shell_count > 2:
        print(f"  ✗ ABORTING: Found {shell_count} shells (expected ≤ 2)")
        print(f"    Complex multi-shell solids not supported")
        return []
    elif shell_count == 2:
        print(f"  ⚠️  WARNING: Found 2 shells - may indicate hollow solid or complex geometry")
    shell_explorer = TopExp_Explorer(solid, TopAbs_SHELL)
    shell_num = 0
    face_count = 0
    while shell_explorer.More():
        shell = shell_explorer.Current()
        shell_num += 1
        print(f"  \nShell {shell_num}:")
        face_explorer = TopExp_Explorer(shell, TopAbs_FACE)
        while face_explorer.More():
            face_shape = face_explorer.Current()
            face_count += 1
            try:
                face = topods.Face(face_shape)
                print(f"    Face {face_count}:")
                face_normal = get_face_normal_from_opencascade(face)
                polygon_data = {}
                wire_explorer = TopExp_Explorer(face, TopAbs_WIRE)
                wires = []
                while wire_explorer.More():
                    wire = wire_explorer.Current()
                    wires.append(wire)
                    wire_explorer.Next()
                print(f"      Found {len(wires)} wires in face {face_count}")
                if wires:
                    outer_boundary = extract_wire_vertices_in_sequence(wires[0], 1)
                    polygon_data['outer_boundary'] = outer_boundary
                    cutouts = []
                    for i, wire in enumerate(wires[1:], 2):
                        cutout_vertices = extract_wire_vertices_in_sequence(wire, i)
                        if cutout_vertices:
                            cutouts.append(cutout_vertices)
                    polygon_data['cutouts'] = cutouts
                else:
                    print(f"      No wires found, using fallback vertex extraction")
                    # Try to extract vertices directly from the face
                    outer_boundary = []
                    try:
                        # Extract vertices from face directly
                        vertex_explorer = TopExp_Explorer(face, TopAbs_VERTEX)
                        vertices_list = []
                        while vertex_explorer.More():
                            vertex = topods.Vertex(vertex_explorer.Current())
                            pnt = BRep_Tool.Pnt(vertex)
                            vertices_list.append([pnt.X(), pnt.Y(), pnt.Z()])
                            vertex_explorer.Next()
                        outer_boundary = vertices_list
                    except Exception as ve:
                        print(f"      Fallback vertex extraction failed: {ve}")
                        outer_boundary = []
                    polygon_data['outer_boundary'] = outer_boundary
                    polygon_data['cutouts'] = []
                if polygon_data['outer_boundary'] and face_normal is not None:
                    polygon_data['normal'] = face_normal
                    polygon_data['face_id'] = face_count
                    faces.append(polygon_data)
                    outer_vertices = len(polygon_data['outer_boundary'])
                    cutout_count = len(polygon_data['cutouts'])
                    total_vertices = outer_vertices + sum(len(cutout) for cutout in polygon_data['cutouts'])
                    print(f"      ✓ Extracted polygon: {outer_vertices} outer vertices, {cutout_count} cutouts, {total_vertices} total vertices")
                    # For visualization, collect face vertices
                    all_face_data.append({'vertices': polygon_data['outer_boundary']})
                else:
                    print(f"      ✗ Failed to extract polygon data")
            except Exception as e:
                print(f"    Face {face_count}: error processing - {e}")
            face_explorer.Next()
        shell_explorer.Next()
    print(f"  \n✓ Successfully extracted {len(faces)} faces from {shell_count} shells")
    print(f"[DEBUG] Extracted faces data:")
    for i, face_data in enumerate(faces):
        outer_verts = len(face_data.get('outer_boundary', []))
        has_normal = face_data.get('normal') is not None
        print(f"  Face {i+1}: face_id={face_data.get('face_id')}, vertices={outer_verts}, normal={has_normal}")
    if visualize:
        try:
            from mpl_toolkits.mplot3d import Axes3D
            import matplotlib.pyplot as plt
            fig = plt.figure(figsize=(15, 12))
            ax = fig.add_subplot(111, projection='3d')
            colors = plt.cm.Set3(np.linspace(0, 1, len(all_face_data)))
            for i, face_data in enumerate(all_face_data):
                vertices = np.array(face_data['vertices'])
                if len(vertices) > 2:
                    vertices_closed = np.vstack([vertices, vertices[0]])
                else:
                    vertices_closed = vertices
                ax.plot(vertices_closed[:, 0], vertices_closed[:, 1], vertices_closed[:, 2], color=colors[i], linewidth=3, alpha=0.9)
                ax.scatter(vertices[:, 0], vertices[:, 1], vertices[:, 2], color=colors[i], s=50, alpha=0.8, edgecolors='black', linewidth=1)
                face_center = np.mean(vertices, axis=0)
                label_text = f'F{i+1} ({len(vertices)}v)'
                ax.text(face_center[0], face_center[1], face_center[2], label_text, fontsize=10, color='black', ha='center', va='center', alpha=0.7)
            ax.set_xlabel('X Coordinate', fontsize=12, weight='bold')
            ax.set_ylabel('Y Coordinate', fontsize=12, weight='bold')
            ax.set_zlabel('Z Coordinate', fontsize=12, weight='bold')
            ax.set_title(f'3D Solid Visualization - POLYGON BOUNDARIES ONLY\n{len(all_face_data)} Faces from Boolean CUT Operation\nNo Triangulation - Pure Polygon Display', fontsize=14, weight='bold')
            all_vertices = np.vstack([face_data['vertices'] for face_data in all_face_data])
            max_range = np.ptp(all_vertices, axis=0).max() / 2.0
            mid_x = np.mean(all_vertices[:, 0])
            mid_y = np.mean(all_vertices[:, 1])
            mid_z = np.mean(all_vertices[:, 2])
            margin = max_range * 0.1
            ax.set_xlim(mid_x - max_range - margin, mid_x + max_range + margin)
            ax.set_ylim(mid_y - max_range - margin, mid_y + max_range + margin)
            ax.set_zlim(mid_z - max_range - margin, mid_z + max_range + margin)
            handles, labels = ax.get_legend_handles_labels()
            if len(handles) > 10:
                ax.legend(handles[:10], labels[:10], loc='upper left', bbox_to_anchor=(0.02, 0.98), fontsize=9)
            else:
                ax.legend(loc='upper left', bbox_to_anchor=(0.02, 0.98), fontsize=9)
            ax.grid(True, alpha=0.3)
            ax.view_init(elev=25, azim=45)
            info_text = f"""PURE POLYGON DISPLAY\n• No triangulation applied\n• All faces shown as true polygons\n• Face 3 should show 5-vertex pentagon\n• Inclined edges clearly visible\n• {len(all_face_data)} faces total"""
            ax.text2D(0.02, 0.02, info_text, transform=ax.transAxes, fontsize=10, verticalalignment='bottom', bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8), fontfamily='monospace')
            plt.tight_layout()
            plt.show()
        except Exception as e:
            print(f"✗ 3D matplotlib visualization failed: {e}")
            print("  → Continuing with array processing...")
            import traceback
            traceback.print_exc()
    print(f"[DEBUG] extract_and_visualize_faces returning {len(faces)} faces")
    return faces


def plot_polygon(polygon, ax, facecolor='none', edgecolor='black', alpha=1.0, linewidth=1.0, linestyle='-', label=None, outline_only=False):
    # Simple polygon plotter using matplotlib
    if hasattr(polygon, 'exterior'):
        x, y = polygon.exterior.xy
    else:
        x, y = zip(*polygon)
    ax.plot(x, y, color=edgecolor, alpha=alpha, linewidth=linewidth, linestyle=linestyle, label=label)
    if not outline_only:
        ax.fill(x, y, facecolor=facecolor, alpha=alpha)


def get_projection_normal_from_user():
    """Get projection normal from user input with [1,1,1] as default."""
    print("\n" + "="*60)
    print("PROJECTION NORMAL INPUT")
    print("="*60)
    
    default_normal = [1, 1, 1]
    
    try:
        print(f"Enter projection normal vector components (default: {default_normal}):")
        print("Format: x y z (space separated) or press Enter for default")
        
        user_input = input("Projection normal: ").strip()
        
        if not user_input:
            # Use default
            projection_normal = np.array(default_normal, dtype=float)
            print(f"Using default projection normal: {projection_normal}")
        else:
            # Parse user input
            components = user_input.split()
            if len(components) != 3:
                print(f"Invalid input format. Using default: {default_normal}")
                projection_normal = np.array(default_normal, dtype=float)
            else:
                try:
                    projection_normal = np.array([float(x) for x in components])
                    print(f"User input projection normal: {projection_normal}")
                except ValueError:
                    print(f"Invalid number format. Using default: {default_normal}")
                    projection_normal = np.array(default_normal, dtype=float)
        
        # Convert to unit vector
        magnitude = np.linalg.norm(projection_normal)
        if magnitude < 1e-10:
            print(f"Zero vector detected. Using default: {default_normal}")
            projection_normal = np.array(default_normal, dtype=float)
            magnitude = np.linalg.norm(projection_normal)
        
        unit_projection_normal = projection_normal / magnitude
        
        print(f"Original projection normal: [{projection_normal[0]:.3f}, {projection_normal[1]:.3f}, {projection_normal[2]:.3f}]")
        print(f"Unit projection normal: [{unit_projection_normal[0]:.6f}, {unit_projection_normal[1]:.6f}, {unit_projection_normal[2]:.6f}]")
        print(f"Magnitude: {magnitude:.6f}")
        print("="*60)
        
        return unit_projection_normal
        
    except KeyboardInterrupt:
        print(f"\nInterrupted. Using default: {default_normal}")
        projection_normal = np.array(default_normal, dtype=float)
        unit_projection_normal = projection_normal / np.linalg.norm(projection_normal)
        return unit_projection_normal
    except Exception as e:
        print(f"Error getting user input: {e}. Using default: {default_normal}")
        projection_normal = np.array(default_normal, dtype=float)
        unit_projection_normal = projection_normal / np.linalg.norm(projection_normal)
        return unit_projection_normal

def find_interior_point(polygon, debug=False):
    """Find an interior point within a polygon. If debug=True, return (point, method_used)."""
    try:
        # Use representative point (guaranteed to be inside)
        interior_point = polygon.representative_point()
        if polygon.contains(interior_point):
            if debug:
                return interior_point, 'representative_point'
            return interior_point
        # Fallback to centroid
        centroid = polygon.centroid
        if polygon.contains(centroid):
            if debug:
                return centroid, 'centroid'
            return centroid
        # Final fallback: use first coordinate of exterior
        coords = list(polygon.exterior.coords)
        if len(coords) > 1:
            if debug:
                return Point(coords[0]), 'first_exterior_coord'
            return Point(coords[0])
    except Exception as e:
        print(f"Error finding interior point: {e}")
    if debug:
        return None, 'failed'
    return None

def intersect_line_with_face(point_2d, projection_normal, face_vertices_3d):
    """Intersect a line with a 3D face to find depth."""
    try:
        if face_vertices_3d is None or len(face_vertices_3d) < 3:
            return None
            
        # Create orthogonal basis vectors for the projection plane
        normal = np.array(projection_normal, dtype=float)
        normal = normal / np.linalg.norm(normal)
        
        # Find a temporary vector not parallel to normal
        if abs(normal[0]) < 0.9:
            temp = np.array([1, 0, 0])
        else:
            temp = np.array([0, 1, 0])
        
        # Create first basis vector (orthogonal to normal)
        u = temp - np.dot(temp, normal) * normal
        u = u / np.linalg.norm(u)
        
        # Create second basis vector (orthogonal to both normal and u)
        v = np.cross(normal, u)
        v = v / np.linalg.norm(v)
        
        # Convert 2D point to 3D point on the projection plane
        plane_origin = np.array([0, 0, 0])  # Simplification
        point_3d_on_plane = plane_origin + point_2d.x * u + point_2d.y * v
        
        # Calculate intersection with face plane
        # Use first three vertices to define the plane
        v0, v1, v2 = face_vertices_3d[0], face_vertices_3d[1], face_vertices_3d[2]
        face_normal = np.cross(v1 - v0, v2 - v0)
        face_normal = face_normal / np.linalg.norm(face_normal)
        
        # Ray-plane intersection
        denominator = np.dot(normal, face_normal)
        if abs(denominator) > 1e-6:
            t = np.dot((v0 - point_3d_on_plane), face_normal) / denominator
            intersection_3d = point_3d_on_plane + t * normal
            return intersection_3d
            
    except Exception as e:
        print(f"Error in line-face intersection: {e}")
    
    return None

def calculate_depth_along_normal(point_3d, projection_normal):
    """Calculate depth of a 3D point along the projection normal."""
    if point_3d is None:
        return 0
    try:
        return np.dot(point_3d, projection_normal)
    except:
        return 0

def create_polygon_from_projection(projected_vertices, allow_invalid=False):
    """Create a Shapely polygon from projected vertices. Optionally allow invalid polygons."""
    if len(projected_vertices) == 0:
        return Polygon()

    projected_vertices = np.array(projected_vertices)
    original_vertex_count = len(projected_vertices)

    if len(projected_vertices) > 0:
        if not np.allclose(projected_vertices[0], projected_vertices[-1], atol=1e-10):
            projected_vertices = np.vstack([projected_vertices, projected_vertices[0]])

    print(f"    → Creating polygon from {original_vertex_count} vertices")

    try:
        polygon = Polygon(projected_vertices)

        # Check if polygon is valid
        if polygon.is_valid and hasattr(polygon, 'area') and polygon.area > 1e-6:
            print(f"    → Valid polygon created with {len(polygon.exterior.coords)-1} vertices")
            return polygon

        # For invalid polygons, always try to fix first
        if not polygon.is_valid:
            from shapely.validation import explain_validity
            reason = explain_validity(polygon)
            print(f"    → Invalid polygon detected: {reason}")
            print(f"    → Original vertices: {original_vertex_count}, coords in polygon: {len(polygon.exterior.coords)-1}")

            # Attempt 1: buffer(0) fix
            try:
                fixed_polygon = polygon.buffer(0)
                if fixed_polygon.is_valid and hasattr(fixed_polygon, 'area') and fixed_polygon.area > 1e-6:
                    if hasattr(fixed_polygon, 'exterior'):
                        fixed_vertex_count = len(fixed_polygon.exterior.coords) - 1
                        print(f"    → ✓ Fixed with buffer(0): {original_vertex_count} → {fixed_vertex_count} vertices")
                    return fixed_polygon
            except Exception as e:
                print(f"    → Buffer(0) fix failed: {e}")

            # Attempt 2: convex_hull fix
            try:
                hull_polygon = Polygon(projected_vertices).convex_hull
                if hull_polygon.is_valid and hasattr(hull_polygon, 'area') and hull_polygon.area > 1e-6:
                    if hasattr(hull_polygon, 'exterior'):
                        hull_vertex_count = len(hull_polygon.exterior.coords) - 1
                        print(f"    → ✓ Fixed with convex_hull: {original_vertex_count} → {hull_vertex_count} vertices")
                    return hull_polygon
            except Exception as e:
                print(f"    → Convex hull fix failed: {e}")

            # Last resort: if allow_invalid=True, store the invalid polygon
            if allow_invalid:
                print(f"    → All fixes failed, but allow_invalid=True: storing invalid polygon")
                print(f"      [WARNING] Invalid polygon stored: {reason}")
                if 'Self-intersection' in reason:
                    print(f"      [INVESTIGATE] Polygon WKT: {polygon.wkt}")
                return polygon

        # If we reach here: polygon is invalid and allow_invalid=False
        print(f"    → All polygon fixes failed, returning empty polygon")
        return Polygon()

    except Exception as e:
        print(f"    → Error creating polygon: {e}")
        return Polygon()

def plot_arrays_visualization(array_A, array_B, array_C, unit_projection_normal):
    """Plot arrays B, C, and B+C with enhanced visualization."""
    print("\n" + "="*60)
    print("PLOTTING ARRAY VISUALIZATION")
    print("="*60)
    
    if not array_B and not array_C:
        print("No polygons to visualize")
        return
    
    # Create figure with subplots
    fig, ((ax1, ax4), (ax3, ax2)) = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle(f'Enhanced Polygon Classification Results\n(Projection Normal: {unit_projection_normal})', 
                fontsize=14, weight='bold')
    
    colors_b = ['lightblue', 'lightcoral', 'lightgreen', 'lightyellow', 'lightpink', 'lavender']
    colors_c = ['orange', 'red', 'purple', 'brown', 'gray', 'cyan']
    
    # Subplot 1: Array B (Visible faces)
    ax1.set_title(f'Array B - Visible Faces ({len(array_B)} polygons)', fontsize=12, weight='bold')
    ax1.set_aspect('equal')
    ax1.grid(True, alpha=0.3)
    ax1.set_xlabel('X coordinate')
    ax1.set_ylabel('Y coordinate')
    
    all_bounds = []
    
    for i, poly_data in enumerate(array_B):
        try:
            polygon = poly_data['polygon']
            name = poly_data['name']
            
            if polygon.geom_type == 'Polygon' and polygon.area > 0:
                color = colors_b[i % len(colors_b)]
                plot_polygon(polygon, ax1, facecolor=color, edgecolor='black', 
                           alpha=0.7, linewidth=1.5, label=f'{name} (area: {polygon.area:.1f})')
                
                # Collect bounds
                bounds = polygon.bounds
                all_bounds.extend([bounds[0], bounds[2], bounds[1], bounds[3]])
                
                # Add face name at centroid
                centroid = polygon.centroid
                ax1.text(centroid.x, centroid.y, name.replace('Face_', 'F'), 
                        ha='center', va='center', fontsize=8, weight='bold')
                        
        except Exception as e:
            print(f"Error plotting {poly_data['name']}: {e}")
    
    if array_B:
        ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
    
    # Subplot 2: Array C (Hidden faces + intersections)
    ax2.set_title(f'Array C - Hidden + Intersections ({len(array_C)} polygons)', fontsize=12, weight='bold')
    ax2.set_aspect('equal')
    ax2.grid(True, alpha=0.3)
    ax2.set_xlabel('X coordinate')
    ax2.set_ylabel('Y coordinate')
    
    for i, poly_data in enumerate(array_C):
        try:
            polygon = poly_data['polygon']
            name = poly_data['name']
            display_name = name.replace('Face_', 'F').replace('Intersection_', 'I_')

            if polygon.geom_type == 'Polygon':
                if polygon.area > 0:
                    # Regular polygon plotting
                    if 'Intersection' in name:
                        color = 'yellow'
                        edge_color = 'red'
                        alpha = 0.8
                        linewidth = 2
                    else:
                        color = colors_c[i % len(colors_c)]
                        edge_color = 'black'
                        alpha = 0.6
                        linewidth = 1
                    plot_polygon(polygon, ax2, facecolor=color, edgecolor=edge_color, 
                               alpha=alpha, linewidth=linewidth, label=f'{name} (area: {polygon.area:.1f})')
                    bounds = polygon.bounds
                    all_bounds.extend([bounds[0], bounds[2], bounds[1], bounds[3]])
                    centroid = polygon.centroid
                    ax2.text(centroid.x, centroid.y, display_name, 
                            ha='center', va='center', fontsize=8, weight='bold')
                else:
                    # Degenerate polygon (zero area): plot as black dashed line
                    coords = list(polygon.exterior.coords)
                    print(f"[DEBUG] Plotting degenerate polygon in Array_C: {name}, coords={coords}")
                    ax2.plot(
                        [c[0] for c in coords],
                        [c[1] for c in coords],
                        color='black', linestyle=(0, (4, 4)), linewidth=2, alpha=0.8,
                        label=f'{name} (degenerate)'
                    )
                    # Mark endpoints
                    ax2.scatter([coords[0][0], coords[-1][0]], [coords[0][1], coords[-1][1]], color='red', s=30)
                    # Add label at midpoint
                    midx = (coords[0][0] + coords[-1][0]) / 2
                    midy = (coords[0][1] + coords[-1][1]) / 2
                    ax2.text(midx, midy, display_name, ha='center', va='center', fontsize=8, color='red', weight='bold')
            else:
                # Other geometry types (e.g., MultiPolygon, LineString)
                try:
                    coords = list(polygon.coords)
                    ax2.plot(
                        [c[0] for c in coords],
                        [c[1] for c in coords],
                        color='black', linestyle='dashed', linewidth=2, alpha=0.8,
                        label=f'{name} (degenerate)'
                    )
                    ax2.scatter([coords[0][0], coords[-1][0]], [coords[0][1], coords[-1][1]], color='red', s=30)
                    midx = (coords[0][0] + coords[-1][0]) / 2
                    midy = (coords[0][1] + coords[-1][1]) / 2
                    ax2.text(midx, midy, display_name, ha='center', va='center', fontsize=8, color='red', weight='bold')
                except Exception:
                    pass
        except Exception as e:
            print(f"Error plotting {poly_data['name']}: {e}")
    
    if array_C:
        ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
    
    # Subplot 3: Combined B + C
    ax3.set_title(f'Combined Arrays B + C ({len(array_B) + len(array_C)} polygons)', fontsize=12, weight='bold')
    ax3.set_aspect('equal')
    ax3.grid(True, alpha=0.3)
    ax3.set_xlabel('X coordinate')
    ax3.set_ylabel('Y coordinate')

    # Plot array_C polygons first as thin dashed light gray lines
    for i, poly_data in enumerate(array_C):
        try:
            polygon = poly_data['polygon']
            name = poly_data['name']
            print(f"[PLOT] Array_C {i+1}/{len(array_C)}: {name}, area={polygon.area:.2f}")
            if polygon.geom_type == 'Polygon':
                if polygon.area > 0:
                    plot_polygon(polygon, ax3, facecolor='none', edgecolor='lightgray', alpha=0.8, linewidth=0.7, linestyle='--', label=f'C: {name}', outline_only=True)
                else:
                    # Degenerate polygon (zero area): plot as black dashed line
                    coords = list(polygon.exterior.coords)
                    ax3.plot(
                        [c[0] for c in coords],
                        [c[1] for c in coords],
                        color='black', linestyle='dashed', linewidth=2, alpha=0.8,
                        label=f'C: {name} (degenerate)'
                    )
                    ax3.scatter([coords[0][0], coords[-1][0]], [coords[0][1], coords[-1][1]], color='red', s=30)
                    midx = (coords[0][0] + coords[-1][0]) / 2
                    midy = (coords[0][1] + coords[-1][1]) / 2
                    ax3.text(midx, midy, name, ha='center', va='center', fontsize=8, color='red', weight='bold')
        except Exception as e:
            print(f"[PLOT] Error plotting array_C polygon in combined subplot: {name}: {e}")

    # Plot array_B polygons afterwards as solid black lines
    for i, poly_data in enumerate(array_B):
        try:
            polygon = poly_data['polygon']
            name = poly_data['name']
            if polygon.geom_type == 'Polygon' and polygon.area > 0:
                plot_polygon(polygon, ax3, facecolor='none', edgecolor='black', alpha=1.0, linewidth=1.2, linestyle='-', label=f'B: {name}', outline_only=True)
        except Exception as e:
            print(f"Error plotting array_B polygon in combined subplot: {e}")

    ax3.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=7)
    
    # Subplot 4: Statistics and algorithm info
    ax4.axis('off')
    stats_text = f"""ENHANCED POLYGON CLASSIFICATION RESULTS

Algorithm: Historic Depth-Based Classification
Projection Normal: [{unit_projection_normal[0]:.3f}, {unit_projection_normal[1]:.3f}, {unit_projection_normal[2]:.3f}]

ARRAY B (VISIBLE FACES):
• Polygons: {len(array_B)}
• Total Area: {sum(p['polygon'].area for p in array_B if hasattr(p['polygon'], 'area')):.2f}
• Type: Depth-processed visible faces

ARRAY C (HIDDEN + INTERSECTIONS):
• Polygons: {len(array_C)}
• Total Area: {sum(p['polygon'].area for p in array_C if hasattr(p['polygon'], 'area')):.2f}
• Type: Hidden faces + intersection regions

ALGORITHM FEATURES:
✓ Historic polygon classification extracted
✓ Depth-based boolean operations
✓ 3D line-face intersection analysis
✓ Multi-point sampling for accuracy
✓ Face association tracking
✓ Enhanced visualization

Total Processed: {len(array_B) + len(array_C)} polygons"""
    
    ax4.text(0.05, 0.95, stats_text, transform=ax4.transAxes, 
             fontsize=9, verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
    
    # Set consistent bounds for all plots
    if all_bounds:
        margin = (max(all_bounds) - min(all_bounds)) * 0.1
        xlim = (min(all_bounds) - margin, max(all_bounds) + margin)
        ylim = (min(all_bounds) - margin, max(all_bounds) + margin)

        # Set y-limits for all views to match top view (ax1)
        for ax in [ax1, ax2, ax3]:
            ax.set_xlim(xlim)
            ax.set_ylim(ylim)
        # If side/front views are plotted elsewhere, set their y-limits here as well
            # Synchronize vertical axis for external side/front views if they exist
            try:
                # If ax2 is front view (Y-Z), set its ylim to match top view's ylim
                ax2.set_ylim(ylim)
            except Exception:
                pass
            try:
                # If ax4 is side view (X-Z), set its ylim to match top view's ylim
                ax4.set_ylim(ylim)
            except Exception:
                pass
    
    plt.tight_layout()
    # plt.show()  # Removed to prevent empty plot
    
    print(f"✓ Array visualization complete")
    print(f"  → Array B: {len(array_B)} visible faces")
    print(f"  → Array C: {len(array_C)} hidden faces + intersections")
    print(f"  → Combined: {len(array_B) + len(array_C)} total polygons")



# old - def visualize_3d_solid(solid_shape, selected_vertices=None, edges=None, edges_with_class=None):
def visualize_3d_solid(face_polygons, selected_vertices=None, edges=None, edges_with_class=None):
    """
    Display the 3D solid using matplotlib 3D plotting.
    Optionally highlight selected vertices and edges with color-coding.
    
    Args:
        solid_shape: The solid shape to visualize
        selected_vertices: Array of selected 3D vertices
        edges: List of edge tuples (v1_idx, v2_idx) - for backward compat
        edges_with_class: List of tuples (v1_idx, v2_idx, classification)
                         Classifications: 1=yellow, 2=gray, 3=black
    """
    # old. if not OPENCASCADE_AVAILABLE or solid_shape is None:
    #     print("✗ Cannot visualize - OpenCASCADE not available or shape is None")
    #     return
    if not OPENCASCADE_AVAILABLE or face_polygons is None:
        print("✗ Cannot visualize - OpenCASCADE not available or shape is None")
        return
    if (selected_vertices is None or len(selected_vertices) == 0) and (edges is None or len(edges) == 0):
        print("✓ Skipping empty 3D plot: no vertices or edges to display.")
        return
    # --- Plot original solid polygons (faces) in a dedicated figure ---
    import matplotlib.pyplot as plt
    import numpy as np
    #Commented by S. Bedi to make program more efficient
    #face_polygons = extract_and_visualize_faces(solid_shape, visualize=True)
    print(f"[DIAG] Number of face polygons extracted: {len(face_polygons)}")
    for idx, poly_data in enumerate(face_polygons):
        verts = np.array(poly_data.get('outer_boundary', []))
        #print(f"[DIAG] Face {idx+1}: {verts.shape[0]} vertices: {verts}")
    if face_polygons:
        first_verts = np.array(face_polygons[0].get('outer_boundary', []))
        print(f"[DEBUG] First face polygon before plotting: {first_verts}")
    # Removed fig1 and ax1: all polygons will be shown in the interactive plot below
    print("\n" + "="*60)
    print("3D SOLID VISUALIZATION WITH MATPLOTLIB")
    print("="*60)
    try:
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D
        from matplotlib.widgets import CheckButtons
        import numpy as np
        import inspect

        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')

        # Plot original solid polygons (faces) using cached face_polygons
        face_handles = []
        for idx, poly_data in enumerate(face_polygons):
            verts = poly_data.get('outer_boundary', [])
            verts = np.array(verts)
            #print(f"[DIAG] Face {idx+1}: verts type={type(verts)}, shape={getattr(verts, 'shape', None)}")
            if idx == 0:
                print(f"[DEBUG] First face polygon in plotting call: {verts}")
            # Only plot if verts is a 2D array with shape (N, 3)
            if isinstance(verts, np.ndarray) and verts.ndim == 2 and verts.shape[1] == 3 and verts.shape[0] >= 3:
                try:
                    handle = ax.plot(verts[:, 0], verts[:, 1], verts[:, 2], color='gray', alpha=0.5, linewidth=2)[0]
                    face_handles.append(handle)
                except Exception as plot_err:
                    print(f"[ERROR] Failed to plot Face {idx+1}: verts={verts}, error={plot_err}")

        # Plot reconstructed vertices
        vertex_handle = None
        if selected_vertices is not None and len(selected_vertices) > 0:
            selected_vertices = np.array(selected_vertices)
            #print(f"[DIAG] selected_vertices type={type(selected_vertices)}, shape={getattr(selected_vertices, 'shape', None)}")
            if selected_vertices.ndim == 2 and selected_vertices.shape[1] == 3:
                vertex_handle = ax.scatter(selected_vertices[:, 0], selected_vertices[:, 1], selected_vertices[:, 2], color='blue', s=60, label='Reconstructed Vertices')

        # Plot edges by connectivity index
        edge_handles_1 = []
        edge_handles_2 = []
        edge_handles_3 = []
        merged_conn = None
        frame = inspect.currentframe().f_back
        merged_conn = frame.f_locals.get('merged_conn', None)
        if edges is not None and len(edges) > 0 and selected_vertices is not None:
            for (i, j) in edges:
                conn_val = None
                if merged_conn is not None:
                    conn_val = merged_conn[i, j]
                if conn_val == 3:
                    color = 'red'
                    lw = 3
                    handle = ax.plot([selected_vertices[i, 0], selected_vertices[j, 0]],
                                    [selected_vertices[i, 1], selected_vertices[j, 1]],
                                    [selected_vertices[i, 2], selected_vertices[j, 2]],
                                    color=color, linewidth=lw)[0]
                    edge_handles_3.append(handle)
                elif conn_val == 2:
                    color = 'orange'
                    lw = 2.5
                    handle = ax.plot([selected_vertices[i, 0], selected_vertices[j, 0]],
                                    [selected_vertices[i, 1], selected_vertices[j, 1]],
                                    [selected_vertices[i, 2], selected_vertices[j, 2]],
                                    color=color, linewidth=lw)[0]
                    edge_handles_2.append(handle)
                elif conn_val == 1:
                    color = 'green'
                    lw = 2
                    handle = ax.plot([selected_vertices[i, 0], selected_vertices[j, 0]],
                                    [selected_vertices[i, 1], selected_vertices[j, 1]],
                                    [selected_vertices[i, 2], selected_vertices[j, 2]],
                                    color=color, linewidth=lw)[0]
                    edge_handles_1.append(handle)

        # Set up interactive check buttons
        rax = plt.axes([0.82, 0.3, 0.15, 0.3])
        labels = ['Polygons', 'Vertices', 'Edges conn=3', 'Edges conn=2', 'Edges conn=1']
        visibility = [True, True, True, True, True]
        check = CheckButtons(rax, labels, visibility)

        def update_visibility(label):
            idx = labels.index(label)
            if idx == 0:
                for h in face_handles:
                    h.set_visible(not h.get_visible())
            elif idx == 1 and vertex_handle is not None:
                vertex_handle.set_visible(not vertex_handle.get_visible())
            elif idx == 2:
                for h in edge_handles_3:
                    h.set_visible(not h.get_visible())
            elif idx == 3:
                for h in edge_handles_2:
                    h.set_visible(not h.get_visible())
            elif idx == 4:
                for h in edge_handles_1:
                    h.set_visible(not h.get_visible())
            fig.canvas.draw_idle()

        check.on_clicked(update_visibility)

        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title('3D Solid Reconstruction')
        plt.show()
    except Exception as e:
        print(f"✗ 3D matplotlib visualization failed: {e}")
        print("  → Continuing with array processing...")
        import traceback
        traceback.print_exc()

def analyze_face_colinearity(face_vertices, face_id):
    """Analyze vertices for co-linearity issues that cause triangle appearance."""
    if not face_vertices or len(face_vertices) < 3:
        return
    
    print(f"        Co-linearity Analysis for Face {face_id}:")
    
    # Check for consecutive co-linear vertices
    colinear_groups = []
    
    for i in range(len(face_vertices)):
        v1 = np.array(face_vertices[i])
        v2 = np.array(face_vertices[(i + 1) % len(face_vertices)])
        v3 = np.array(face_vertices[(i + 2) % len(face_vertices)])
        
        # Calculate vectors
        vec1 = v2 - v1
        vec2 = v3 - v2
        
        # Check if vectors are parallel (cross product near zero)
        cross_product = np.cross(vec1, vec2)
        cross_magnitude = np.linalg.norm(cross_product)
        
        if cross_magnitude < 1e-6:  # Very small cross product = co-linear
            colinear_groups.append([i, (i+1)%len(face_vertices), (i+2)%len(face_vertices)])
    # ...existing code...
   

def classify_faces_by_projection(face_polygons, unit_projection_normal):
    """Enhanced face classification with historic polygon classification algorithm."""
    if face_polygons is None:
        print("Warning: face_polygons is None. Returning empty arrays.")
        return [], [], []
        
    print("\n" + "="*60)
    print("ENHANCED FACE CLASSIFICATION WITH HISTORIC ALGORITHM")
    print("="*60)
    
    array_A_initial = []  # Initial classification for processing
    array_B = []  # Depth-processed polygons (visible)
    array_C = []  # Hidden faces + intersections
    
    print(f"Unit projection normal: [{unit_projection_normal[0]:.6f}, {unit_projection_normal[1]:.6f}, {unit_projection_normal[2]:.6f}]")
    print("\nStep 1: Initial classification and polygon projection...")
    
    # Convert face data to projectable polygons
    valid_polygons = []
    
    for i, polygon_data in enumerate(face_polygons):
        face_id = polygon_data.get('face_id', i+1)
        face_normal = polygon_data.get('normal')
        outer_boundary = polygon_data.get('outer_boundary', [])

        if face_normal is None or len(outer_boundary) < 3:
            print(f"Face F{face_id}: Invalid data - skipping")
            continue

        # Ensure face normal is unit vector
        unit_face_normal = face_normal / np.linalg.norm(face_normal)

        # Calculate dot product
        dot_product = np.dot(unit_face_normal, unit_projection_normal)

        print(f"Face F{face_id}: dot_product={dot_product:.3f}, unit_face_normal=[{unit_face_normal[0]:.3f}, {unit_face_normal[1]:.3f}, {unit_face_normal[2]:.3f}]")

    # Reverted logic: do NOT automatically move perpendicular faces to array_C
    # Instead, treat them like other faces and allow further classification

        # Project face to 2D polygon with holes support
        try:
            projected_outer = project_face_to_projection_plane(outer_boundary, unit_projection_normal)
            cutouts = polygon_data.get('cutouts', [])
            projected_holes = []
            for cutout in cutouts:
                if cutout and len(cutout) >= 3:
                    projected_cutout = project_face_to_projection_plane(cutout, unit_projection_normal)
                    projected_holes.append(projected_cutout)
            if projected_holes:
                polygon = Polygon(projected_outer, holes=projected_holes)
                if not polygon.is_valid:
                    polygon = polygon.buffer(0)
            else:
                polygon = create_polygon_from_projection(projected_outer, allow_invalid=True)
            
            # CRITICAL: Include ALL faces with valid 3D parent_face data,
            # even if 2D projection is degenerate (zero area)
            # The 3D edges are still real and must be captured for vertex reconstruction
            # NOTE: A face may be degenerate in one view but not in others!
            if outer_boundary and len(outer_boundary) >= 3:
                polygon_data_enhanced = {
                    'polygon': polygon,
                    'name': f"Face_{face_id}",
                    'normal': unit_face_normal,
                    'parent_face': np.array(outer_boundary),  # 3D vertices
                    'original_index': i,
                    'dot_product': dot_product,
                    'has_holes': len(projected_holes) > 0,
                    # REMOVED: Don't store is_degenerate flag - check per view!
                }
                valid_polygons.append(polygon_data_enhanced)
                array_A_initial.append(polygon_data_enhanced)
                
                # Report polygon status
                hole_info = f" with {len(projected_holes)} holes" if projected_holes else ""
                if polygon.area < 1e-6:
                    print(f"  → Added Face_{face_id} (DEGENERATE in this view: area={polygon.area:.2e}){hole_info} [3D edges will be extracted]")
                else:
                    print(f"  → Added Face_{face_id} (area: {polygon.area:.2f}){hole_info}")
                print(f"      [DEBUG] Face_{face_id} dot_product: {dot_product:.6f} {'(visible)' if dot_product > 0 else '(hidden)'}")
        except Exception as e:
            print(f"Face F{face_id}: Projection error - {e}")
    
    print(f"\nStep 2: Starting historic polygon classification algorithm...")
    print(f"Initial array_A: {len(array_A_initial)} polygons")
    print(f"[DEBUG] Initial faces in array_A before historic algorithm:")
    for i, poly_data in enumerate(array_A_initial):
        name = poly_data['name']
        dot_product = poly_data['dot_product']
        area = poly_data['polygon'].area
        is_degen = " (DEGENERATE)" if poly_data.get('is_degenerate', False) else ""
        print(f"  A{i+1}: {name} (dot={dot_product:.3f}, area={area:.2f}){is_degen}")

    # Print summary of hidden polygons for top view
    if np.allclose(unit_projection_normal, [0, 0, 1], atol=1e-3):
        print("\nSUMMARY: Hidden polygons (Array_C) for Top View:")
        for poly_data in array_C:
            name = poly_data.get('name', 'Unknown')
            dot_product = poly_data.get('dot_product', 0)
            print(f"  - {name} (dot_product={dot_product:.3f})")
    
    # Display initial array_A contents before sorting
    if array_A_initial:
        print(f"\n" + "="*60)
        print("ARRAY A - INITIAL FACE CLASSIFICATION (BEFORE SORTING)")
        print("="*60)
        
        for i, poly_data in enumerate(array_A_initial):
            polygon = poly_data['polygon']
            name = poly_data['name']
            normal = poly_data['normal']
            dot_product = poly_data['dot_product']
            
            # Check for invalid polygons in Array_A
            if not polygon.is_valid:
                from shapely.validation import explain_validity
                reason = explain_validity(polygon)
                print(f"  [WARNING] {name} is invalid in Array_A: {reason}")
                print(f"    [INVESTIGATE] Polygon WKT: {polygon.wkt}")
            # Handle both Polygon and MultiPolygon cases
            if hasattr(polygon, 'exterior'):
                # Simple polygon
                vertex_count = len(polygon.exterior.coords) - 1  # -1 for closing duplicate
                coords = list(polygon.exterior.coords[:-1])  # Exclude closing duplicate
            elif hasattr(polygon, 'geoms') and len(polygon.geoms) > 0:
                # MultiPolygon - use the largest polygon
                largest_poly = max(polygon.geoms, key=lambda p: p.area)
                vertex_count = len(largest_poly.exterior.coords) - 1
                coords = list(largest_poly.exterior.coords[:-1])
            else:
                # Fallback
                vertex_count = 0
                coords = []
            
            print(f"  Face A{i+1} ({name}):")
            print(f"    • Area: {polygon.area:.2f}")
            print(f"    • Vertices: {vertex_count}")
            print(f"    • Dot product: {dot_product:.6f}")

            if hasattr(normal, '__len__') and len(normal) >= 3 and not isinstance(normal, str):
                try:
                    print(f"    • Face normal: [{normal[0]:.3f}, {normal[1]:.3f}, {normal[2]:.3f}]")
                except (TypeError, IndexError):
                    print(f"    • Face normal: {normal}")
            else:
                print(f"    • Face normal: {normal}")
            
            # Show polygon coordinates for verification
            if coords:
                coords_str = " → ".join([f"({c[0]:.1f},{c[1]:.1f})" for c in coords])
                print(f"    • 2D Polygon: {coords_str}")
            else:
                print(f"    • 2D Polygon: [No coordinates available]")
        
        print("="*60)
        print("ARRAY A DISPLAY COMPLETE - NOW STARTING HISTORIC ALGORITHM")
        print("="*60)
    
    if len(array_A_initial) >= 1:
        # Step 2.1: Move first polygon to array_B as seed
        first_polygon = array_A_initial.pop(0)
        array_B.append(first_polygon)
        print(f"Moved {first_polygon['name']} from array_A to array_B as seed")
        
        # Step 2.2: Process remaining polygons with depth-based classification
        while array_A_initial:
            Pi_data = array_A_initial.pop(0)
            Pi = Pi_data['polygon']
            Pi_name = Pi_data['name']
            Pi_parent_face = Pi_data['parent_face']
            # Check if polygon is degenerate in THIS view
            Pi_is_degenerate = (Pi.area < 1e-6)
            
            # CRITICAL: Degenerate polygons (zero area) cannot intersect
            # with other polygons, so add them directly to array_C
            # to ensure their 3D edges are extracted
            # NOTE: Same face may be non-degenerate in other views!
            if Pi_is_degenerate:
                array_C.append(Pi_data)
                print(f"[DEBUG] {Pi_name} is degenerate in this view (area={Pi.area:.2e}) → array_C")
                continue  # Skip intersection testing for degenerate polygons
            
            # Test intersection with all polygons in array_B
            # Iterate in reverse order to avoid index issues when modifying array_B
            has_intersection = False
            for j in reversed(range(len(array_B))):
                Pj_data = array_B[j]
                Pj = Pj_data['polygon']
                Pj_name = Pj_data['name']
                Pj_parent_face = Pj_data['parent_face']
                try:
                    intersection = Pi.intersection(Pj)
                    print(f"[DEBUG] Checking intersection: {Pi_name} vs {Pj_name}, area={getattr(intersection, 'area', None)}")
                    if intersection.is_empty:
                        print(f"[DEBUG] Intersection is empty: {Pi_name} vs {Pj_name}")
                    elif not hasattr(intersection, 'area') or intersection.area <= 1e-6:
                        print(f"[DEBUG] Intersection too small: {Pi_name} vs {Pj_name}, area={getattr(intersection, 'area', None)}")
                    else:
                        # Find interior point for depth analysis
                        result = find_interior_point(intersection, debug=False)
                        if isinstance(result, tuple):
                            interior_point, method_used = result
                        else:
                            interior_point = result
                        if interior_point is None:
                            print(f"[DEBUG] No interior point found for intersection: {Pi_name} vs {Pj_name}")
                            continue
                        # Calculate 3D depths using line-face intersection
                        try:
                            Pi_intersection_3d = intersect_line_with_face(
                                interior_point, unit_projection_normal, Pi_parent_face)
                            Pj_intersection_3d = intersect_line_with_face(
                                interior_point, unit_projection_normal, Pj_parent_face)
                            Pi_depth = calculate_depth_along_normal(Pi_intersection_3d, unit_projection_normal)
                            Pj_depth = calculate_depth_along_normal(Pj_intersection_3d, unit_projection_normal)
                        except Exception as e:
                            print(f"[DEBUG] Depth calculation failed for {Pi_name} vs {Pj_name}: {e}")
                            continue
                        # Add intersection to array_C
                        intersection_name = f"Intersection_{Pi_name}_{Pj_name}"
                        intersection_data = {
                            'polygon': intersection,
                            'name': intersection_name,
                            'normal': 'intersection',
                            # FIXED: Preserve original face data
                            'parent_face': Pi_parent_face,
                            # FIXED: Use original face name
                            'associated_face': Pi_name,
                            'original_index': -1,
                            'dot_product': 0
                        }
                        array_C.append(intersection_data)
                        print(f"[DEBUG] Added intersection to Array_C: {intersection_name}, area={intersection.area}")
                        # Apply depth-based boolean operations
                        if Pi_depth > Pj_depth:
                            try:
                                new_Pj = Pj.difference(Pi)
                                if not new_Pj.is_empty and new_Pj.area > 1e-6:
                                    array_B[j]['polygon'] = new_Pj
                                    array_B[j]['name'] = f"Modified_{Pj_name}"
                                else:
                                    array_B.pop(j)
                            except Exception as e:
                                print(f"[DEBUG] Exception during Pj.difference(Pi): {e}")
                        else:
                            try:
                                new_Pi = Pi.difference(Pj)
                                if not new_Pi.is_empty and new_Pi.area > 1e-6:
                                    Pi = new_Pi
                                    Pi_data['polygon'] = new_Pi
                                    Pi_data['name'] = f"Modified_{Pi_name}"
                                else:
                                    # Pi consumed
                                    # Update Pi_data to reflect the empty polygon
                                    Pi_data['polygon'] = new_Pi
                                    break
                            except Exception as e:
                                print(f"[DEBUG] Exception during Pi.difference(Pj): {e}")
                except Exception as e:
                    print(f"[DEBUG] Exception in intersection loop: {Pi_name} vs {Pj_name}: {e}")
            
            # Add remaining Pi to array_B if it still has area
            if Pi_data['polygon'].area > 1e-6:
                array_B.append(Pi_data)
        
        # Step 2.3: Apply final dot product classification
        faces_to_move = []
        
        for i, poly_data in enumerate(array_B):
            if poly_data['dot_product'] <= 0:
                faces_to_move.append(i)
        
        # Move faces with negative dot product to array_C
        for i in reversed(faces_to_move):
            moved_face = array_B.pop(i)
            array_C.append(moved_face)
    
    print("\n===== FINAL ARRAY_B =====")
    for poly_data in array_B:
        print(f"  {poly_data['name']}: area={poly_data['polygon'].area:.2f}, dot={poly_data.get('dot_product', 'N/A')}")
    print("\n===== FINAL ARRAY_C =====")
    for poly_data in array_C:
        print(f"  {poly_data['name']}: area={poly_data['polygon'].area:.2f}, dot={poly_data.get('dot_product', 'N/A')}")
    print(f"[DEBUG] classify_faces_by_projection: A={len(array_A_initial)}, B={len(array_B)}, C={len(array_C)}")

    # # Print polygons in array_C
    # print("\nPolygons in array_C:")
    # for poly_data in array_C:
    #     polygon = poly_data.get('polygon')
    #     print(f"  {poly_data.get('name', 'Unknown')}: {polygon}")

    # # Print polygons in array_B
    # print("\nPolygons in array_B:")
    # for poly_data in array_B:
    #     polygon = poly_data.get('polygon')
    #     print(f"  {poly_data.get('name', 'Unknown')}: {polygon}")
    # if np.allclose(unit_projection_normal, [1, 0, 0], atol=1e-3):
    #     print("\nSUMMARY: Plot of Array_B and Array_C for Side View:")
    #     plot_arrays_visualization(
    #         array_A_initial,
    #         array_B,
    #         array_C,
    #         unit_projection_normal
    #     )
    return [], array_B, array_C


def order_rectangular_vertices(vertices):
    """Trust OpenCASCADE's natural vertex ordering for rectangular faces.
    
    OpenCASCADE's edge traversal provides vertices in proper clockwise or counter-clockwise
    order, so we simply return them as-is for correct 3D rendering.
    """
    if len(vertices) != 4:
        return vertices
    
    # OpenCASCADE provides vertices in correct topological order
    # No additional reordering needed
    print(f"      Using OpenCASCADE's natural vertex ordering for rectangular face")
    return vertices

def generate_cuboid_faces(width, height, depth):
    """Generate the 6 faces of a cuboid with given dimensions."""
    w, h, d = width/2, height/2, depth/2
    
    # Define the 8 vertices of the cuboid (centered at origin)
    vertices = np.array([
        [-w, -h, -d], [w, -h, -d], [w, h, -d], [-w, h, -d],  # bottom face
        [-w, -h, d],  [w, -h, d],  [w, h, d],  [-w, h, d]    # top face
    ])
    
    # Define the 6 faces (each face defined by 4 vertex indices)
    faces = [
        ([0, 1, 2, 3], [0, 0, -1]),   # bottom face (z = -d)
        ([4, 7, 6, 5], [0, 0, 1]),    # top face (z = d)
        ([0, 4, 5, 1], [0, -1, 0]),   # front face (y = -h)
        ([2, 6, 7, 3], [0, 1, 0]),    # back face (y = h)
        ([0, 3, 7, 4], [-1, 0, 0]),   # left face (x = -w)
        ([1, 5, 6, 2], [1, 0, 0])     # right face (x = w)
    ]
    
    face_data = []
    for face_indices, normal in faces:
        face_vertices = vertices[face_indices]
        face_data.append((face_vertices, np.array(normal)))
    
    return face_data

def project_face_to_projection_plane(face_vertices, projection_normal):
    """Project 3D face vertices to a 2D plane for engineering drawing display."""
    if face_vertices is None or len(face_vertices) == 0 or projection_normal is None:
        return []
    
    # Ensure we have numpy arrays
    face_vertices = np.array(face_vertices)
    projection_normal = np.array(projection_normal)
    
    # Normalize the projection normal
    normal = projection_normal / np.linalg.norm(projection_normal)
    
    # Create two orthogonal vectors in the projection plane
    # Find a vector that's not parallel to the normal
    if abs(normal[0]) < 0.9:
        temp = np.array([1.0, 0.0, 0.0])
    else:
        temp = np.array([0.0, 1.0, 0.0])
    
    # Create first basis vector (orthogonal to normal)
    u = temp - np.dot(temp, normal) * normal
    u = u / np.linalg.norm(u)
    
    # Create second basis vector (orthogonal to both normal and u)
    v = np.cross(normal, u)
    v = v / np.linalg.norm(v)
    
    # Project each vertex onto the plane using the basis vectors
    projected = []
    for vertex in face_vertices:
        vertex = np.array(vertex)
        # Project vertex onto the plane defined by u and v
        proj_u = np.dot(vertex, u)
        proj_v = np.dot(vertex, v)
        projected.append([proj_u, proj_v])
    
    return np.array(projected)


#def plot_polygons(visible, hidden, show_combined, show_visible, show_hidden):
# old def create_view_connectivity_matrix(face_polygons, projection_normal, view_name, all_vertices_3d=None):
def create_view_connectivity_matrix(visible, hidden, projection_normal, view_name, all_vertices_3d=None):
    """
    Create connectivity matrix for a view with unique projected vertices.
    
    Args:
        face_polygons: List of face polygon data from the solid
        projection_normal: Normal vector for the projection
        view_name: Name of the view for debugging
        all_vertices_3d: List of ALL vertices from solid (for reverse engineering completeness)
    
    Returns:
        numpy array: [vertex_index, proj_x, proj_y, connectivity_matrix...]
        where connectivity_matrix is n×n showing which vertices are connected
    """
    
    def project_vertex_to_plane(vertex, normal):
        """Project a 3D vertex to 2D plane using coordinate dropping for orthogonal views"""
        vertex = np.array(vertex)
        normal = np.array(normal, dtype=float)
        normal = normal / np.linalg.norm(normal)
        
        # Use coordinate dropping for standard orthogonal engineering views
        # This matches the edge reconstruction method for consistency
        if np.allclose(normal, [0, 0, 1], atol=1e-3):  # Top view
            return np.array([vertex[0], vertex[1]])  # Drop Z, keep X,Y
        elif np.allclose(normal, [0, -1, 0], atol=1e-3):  # Front view
            return np.array([vertex[0], vertex[2]])  # Drop Y, keep X,Z
        elif np.allclose(normal, [1, 0, 0], atol=1e-3):  # Side view
            return np.array([vertex[1], vertex[2]])  # Drop X, keep Y,Z
        else:
            # For non-orthogonal views (like isometric), use basis vector method
            # Create orthogonal basis vectors for the projection plane
            if abs(normal[0]) < 0.9:
                temp = np.array([1.0, 0.0, 0.0])
            else:
                temp = np.array([0.0, 1.0, 0.0])
            u = temp - np.dot(temp, normal) * normal
            u = u / np.linalg.norm(u)
            v = np.cross(normal, u)
            v = v / np.linalg.norm(v)
            
            proj_u = np.dot(vertex, u)
            proj_v = np.dot(vertex, v)
            return np.array([proj_u, proj_v])
    
    print(f"\n[DEBUG] Creating connectivity matrix for {view_name}")
    print(f"[DEBUG] Using projection normal: {projection_normal}")
    
    # Step 1: Extract all projected vertices from polygons for all views
    projected_vertices = []
    vertex_to_index = {}
    tolerance = 1e-6
    #_, visible_polygons, hidden_polygons = classify_faces_by_projection(face_polygons, projection_normal)
    all_polygons = visible + hidden
    print(f"[DEBUG] Extracting projected vertices from polygons for {view_name}")
    for poly_data in all_polygons:
        parent_face = poly_data.get('parent_face', None)
        if parent_face is None:
            continue
        if hasattr(parent_face, 'tolist'):
            vertices_3d = parent_face.tolist()
        else:
            vertices_3d = parent_face
        if not isinstance(vertices_3d, (list, tuple)) or len(vertices_3d) < 3:
            continue
        for vertex_3d in vertices_3d:
            proj_2d = project_vertex_to_plane(vertex_3d, projection_normal)
            found_existing = False
            for existing_proj in projected_vertices:
                if np.allclose(proj_2d, existing_proj, atol=tolerance):
                    found_existing = True
                    break
            if not found_existing:
                vertex_index = len(projected_vertices)
                projected_vertices.append(proj_2d)
                vertex_to_index[tuple(proj_2d)] = vertex_index
    print(f"[DEBUG] Found {len(projected_vertices)} unique projected vertices from polygons for {view_name}")
    
    print(f"[DEBUG] Processing {len(all_polygons)} polygons ({len(visible)} visible, {len(hidden)} hidden)")
    
    n_vertices = len(projected_vertices)
    print(f"[DEBUG] Found {n_vertices} unique projected vertices")
    
    if n_vertices == 0:
        print(f"[WARNING] No vertices found for {view_name}")
        return None
    
    # Step 2: Create connectivity matrix structure
    # Format: [vertex_index, proj_x, proj_y, connectivity_matrix...]
    matrix_size = 3 + n_vertices  # index + x + y + n×n connectivity
    result_matrix = np.zeros((n_vertices, matrix_size))
    
    # Fill vertex indices and projected coordinates
    for i, proj_vertex in enumerate(projected_vertices):
        result_matrix[i, 0] = i  # vertex index
        result_matrix[i, 1] = proj_vertex[0]  # projected x
        result_matrix[i, 2] = proj_vertex[1]  # projected y
    
    # Step 3: Populate connectivity matrix from polygon edges
    edges_found = 0
    
    for poly_data in all_polygons:
        parent_face = poly_data.get('parent_face', None)
        if parent_face is None:
            continue
            
        if hasattr(parent_face, 'tolist'):
            vertices_3d = parent_face.tolist()
        else:
            vertices_3d = parent_face
            
        if not isinstance(vertices_3d, (list, tuple)) or len(vertices_3d) < 3:
            continue
        
        #print(f"[SB DEBUG] {view_name}: Processed polygon {poly_data.get('name', 'Unknown')} with {len(vertices_3d)} vertices")       
        # Process edges in this polygon
        for i in range(len(vertices_3d)):
            v1_3d = vertices_3d[i]
            v2_3d = vertices_3d[(i + 1) % len(vertices_3d)]
            
            # Project both vertices
            v1_proj = project_vertex_to_plane(v1_3d, projection_normal)
            v2_proj = project_vertex_to_plane(v2_3d, projection_normal)
            
            # Find indices in our unique vertex list
            v1_idx = None
            v2_idx = None
            
            for idx, existing_proj in enumerate(projected_vertices):
                if np.allclose(v1_proj, existing_proj, atol=tolerance):
                    v1_idx = idx
                if np.allclose(v2_proj, existing_proj, atol=tolerance):
                    v2_idx = idx
            
                # Add edge to connectivity matrix
                if v1_idx is not None and v2_idx is not None and v1_idx != v2_idx:
                    # Determine visibility value (1=hidden, 2=visible)
                    # hacked by SB visibility_value = 2 cahanged to
                    visibility_value = 1 if poly_data in visible else 1
                    
                    # Add connection in both directions
                    result_matrix[v1_idx, 3 + v2_idx] = visibility_value
                    result_matrix[v2_idx, 3 + v1_idx] = visibility_value
                    edges_found += 1
            #print(f"[SB DEBUG] {view_name}: Processed edge from vertex {v1_3d} to {v2_3d} with visibility {visibility_value} ")

    
    print(f"[DEBUG] {view_name}: Added {edges_found} edges to connectivity matrix")
    print(f"[DEBUG] {view_name}: Matrix shape: {result_matrix.shape}")
    # Ensure connectivity part is symmetric (mirrored about the diagonal)
    # n_vertices = result_matrix.shape[0]
    # if result_matrix.shape[1] > 3:
    #     conn = result_matrix[:, 3:]
    #     for i in range(n_vertices):
    #         for j in range(n_vertices):
    #             # Mirror connectivity: if either [i, j] or [j, i] is set, set both
    #             if conn[i, j] > 0 or conn[j, i] > 0:
    #                 conn[i, j] = conn[j, i] = max(conn[i, j], conn[j, i])
    #     result_matrix[:, 3:] = conn

    return result_matrix


# old def plot_four_views(solid, user_normal,
def plot_four_views(face_polygons, user_normal,
    ordered_vertices,
    Vertex_Top_View,
    Vertex_Front_View,
    Vertex_Side_View,
    Vertex_Iso_View,
    pdf_dir="PDFfiles"):
    import matplotlib.pyplot as plt
    import numpy as np

    # Helper to project a 3D vertex to 2D for a given normal
    def project_vertex_to_plane(vertex, normal):
        normal = np.array(normal)
        normal = normal / np.linalg.norm(normal)
        if abs(normal[0]) < 0.9:
            temp = np.array([1.0, 0.0, 0.0])
        else:
            temp = np.array([0.0, 1.0, 0.0])
        u = temp - np.dot(temp, normal) * normal
        u = u / np.linalg.norm(u)
        v = np.cross(normal, u)
        v = v / np.linalg.norm(v)
        vertex = np.array(vertex)
        proj_u = np.dot(vertex, u)
        proj_v = np.dot(vertex, v)
        return np.array([proj_u, proj_v])

    # For each view, fill the corresponding array
    view_configs = [
        (np.array([0, 0, 1]), 'Top View', Vertex_Top_View),
        (user_normal, 'Isometric View', Vertex_Iso_View),
        (np.array([0, -1, 0]), 'Front View', Vertex_Front_View),  # -Y axis
        (np.array([1, 0, 0]), 'Side View', Vertex_Side_View)      # +X axis (right side)
    ]

    # Extract face polygons from the solid only once
    #face_polygons = extract_and_visualize_faces(solid)
    print(f"[DEBUG] plot_four_views: extracted {len(face_polygons) if face_polygons else 0} face polygons")
    print(f"[DEBUG] plot_four_views: face_polygons type = {type(face_polygons)}")
    
    # Create connectivity matrices for each view using new approach
    view_connectivity_matrices = {}
    view_polygons = []
    
    for normal, label, vertex_array in view_configs:
        normal = normal / np.linalg.norm(normal)

        
        # Get visible/hidden polygons for plotting
        _, array_B, array_C = classify_faces_by_projection(face_polygons, normal)
        visible = [data['polygon'] for data in array_B if 'polygon' in data]
        hidden = [data['polygon'] for data in array_C if 'polygon' in data]

        # Create connectivity matrix for this view
        connectivity_matrix = create_view_connectivity_matrix(array_B, array_C, normal, label, ordered_vertices)
        view_connectivity_matrices[label] = connectivity_matrix
        
        # For backwards compatibility, update the old vertex_array if needed
        # (This section can be removed once the downstream code is updated)
        if connectivity_matrix is not None:
            # Clear the old vertex array
            vertex_array.fill(0)
            print(f"[DEBUG] {label}: Updated connectivity matrix with {connectivity_matrix.shape[0]} vertices")
        
        # Store polygons for this view
        view_polygons.append((visible, hidden))
    
    # Print summary of connectivity matrices
    print("\n" + "="*60)
    print("CONNECTIVITY MATRICES SUMMARY")
    print("="*60)
    for view_name, matrix in view_connectivity_matrices.items():
        if matrix is not None:
            print(f"{view_name}:")
            print(f"  - {matrix.shape[0]} unique projected vertices")
            print(f"  - Matrix shape: {matrix.shape}")
            print(f"  - Connectivity entries: {np.count_nonzero(matrix[:, 3:])}")
            # Show first few vertices as sample
            print(f"  - Sample vertices (index, proj_x, proj_y):")
            for i in range(min(3, matrix.shape[0])):
                print(f"    V{int(matrix[i,0])}: ({matrix[i,1]:.3f}, {matrix[i,2]:.3f})")
        else:
            print(f"{view_name}: No connectivity matrix created")
    print("="*60)
    
    # Save connectivity matrices to files for analysis
    for view_name, matrix in view_connectivity_matrices.items():
        if matrix is not None:
            filename = f"{view_name.lower().replace(' ', '_')}_connectivity.npy"
            # Save to current directory since output_dir not available here
            filepath = filename
            try:
                np.save(filepath, matrix)
                print(f"[DEBUG] Saved {view_name} connectivity matrix to {filepath}")
            except Exception as e:
                print(f"[WARNING] Could not save {view_name} matrix: {e}")
    
    # Plotting code continues...
    
    def plot_polygons_on_ax(ax, visible, hidden, label, flip_y=False):
        coords_x = []
        coords_y = []
        polygons_drawn = False
        # Plot hidden polygons first
        for idx, poly in enumerate(hidden):
            plotted = False
            # if label == 'Isometric View':
            #     print(f"[PLOT-ALL] Isometric Hidden {idx+1}/{len(hidden)}: type={getattr(poly, 'geom_type', type(poly))}, area={getattr(poly, 'area', 'N/A'):.4f}")
            # Polygon
            if hasattr(poly, 'exterior') and not poly.is_empty:
                x, y = poly.exterior.xy
                ax.plot(x, y, color='gray', linestyle=(0, (2, 2)), linewidth=1.2, alpha=0.8)
                polygons_drawn = True
                coords_x.append(x)
                coords_y.append(y)
                for interior in poly.interiors:
                    ix, iy = interior.xy
                    ax.plot(ix, iy, color='gray', linestyle=(0, (2, 2)), linewidth=1.2, alpha=0.8)
                plotted = True
            # MultiPolygon
            elif getattr(poly, 'geom_type', None) == 'MultiPolygon':
                for subpoly in poly.geoms:
                    if hasattr(subpoly, 'exterior') and not subpoly.is_empty:
                        x, y = subpoly.exterior.xy
                        ax.plot(x, y, color='gray', linestyle=(0, (2, 2)), linewidth=1.2, alpha=0.8)
                        polygons_drawn = True
                        coords_x.append(x)
                        coords_y.append(y)
                        for interior in subpoly.interiors:
                            ix, iy = interior.xy
                            ax.plot(ix, iy, color='gray', linestyle=(0, (2, 2)), linewidth=1.2, alpha=0.8)
                        plotted = True
            # GeometryCollection
            elif getattr(poly, 'geom_type', None) == 'GeometryCollection':
                for subgeom in poly.geoms:
                    if getattr(subgeom, 'geom_type', None) == 'Polygon' and not subgeom.is_empty:
                        x, y = subgeom.exterior.xy
                        ax.plot(x, y, color='gray', linestyle=(0, (2, 2)), linewidth=1.2, alpha=0.8)
                        polygons_drawn = True
                        coords_x.append(x)
                        coords_y.append(y)
                        for interior in subgeom.interiors:
                            ix, iy = interior.xy
                            ax.plot(ix, iy, color='gray', linestyle=(0, (2, 2)), linewidth=1.2, alpha=0.8)
                        plotted = True
                    elif getattr(subgeom, 'geom_type', None) == 'LineString' and not subgeom.is_empty:
                        x, y = subgeom.xy
                        ax.plot(x, y, color='gray', linestyle='dashed', linewidth=1.2, alpha=0.8)
                        polygons_drawn = True
                        coords_x.append(x)
                        coords_y.append(y)
                        plotted = True
        # Plot visible polygons
        for poly in visible:
            # Polygon
            if hasattr(poly, 'exterior') and not poly.is_empty:
                x, y = poly.exterior.xy
                ax.plot(x, y, color='black', linewidth=1.8, alpha=0.95)
                polygons_drawn = True
                coords_x.append(x)
                coords_y.append(y)
                for interior in poly.interiors:
                    ix, iy = interior.xy
                    ax.plot(ix, iy, color='black', linewidth=1.8, alpha=0.95)
            # MultiPolygon
            elif getattr(poly, 'geom_type', None) == 'MultiPolygon':
                for subpoly in poly.geoms:
                    if hasattr(subpoly, 'exterior') and not subpoly.is_empty:
                        x, y = subpoly.exterior.xy
                        ax.plot(x, y, color='black', linewidth=1.8, alpha=0.95)
                        polygons_drawn = True
                        coords_x.append(x)
                        coords_y.append(y)
                        for interior in subpoly.interiors:
                            ix, iy = interior.xy
                            ax.plot(ix, iy, color='black', linewidth=1.8, alpha=0.95)
            # GeometryCollection
            elif getattr(poly, 'geom_type', None) == 'GeometryCollection':
                for subgeom in poly.geoms:
                    if getattr(subgeom, 'geom_type', None) == 'Polygon' and not subgeom.is_empty:
                        x, y = subgeom.exterior.xy
                        ax.plot(x, y, color='black', linewidth=1.8, alpha=0.95)
                        polygons_drawn = True
                        coords_x.append(x)
                        coords_y.append(y)
                        for interior in subgeom.interiors:
                            ix, iy = interior.xy
                            ax.plot(ix, iy, color='black', linewidth=1.8, alpha=0.95)
                    elif getattr(subgeom, 'geom_type', None) == 'LineString' and not subgeom.is_empty:
                        x, y = subgeom.xy
                        ax.plot(x, y, color='black', linestyle='dashed', linewidth=1.8, alpha=0.95)
                        polygons_drawn = True
                        coords_x.append(x)
                        coords_y.append(y)
        ax.set_title(label, fontsize=14)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_frame_on(False)
        # Compute axis ranges
        if coords_x and coords_y:
            x_range = (np.min(np.concatenate(coords_x)), np.max(np.concatenate(coords_x)))
            y_range = (np.min(np.concatenate(coords_y)), np.max(np.concatenate(coords_y)))
        else:
            x_range = (0, 1)
            y_range = (0, 1)
        x_span = x_range[1] - x_range[0]
        y_span = y_range[1] - y_range[0]
        x_margin = 0.1 * x_span
        y_margin = 0.1 * y_span
        x_lim = (x_range[0] - x_margin, x_range[1] + x_margin)
        y_lim = (y_range[0] - y_margin, y_range[1] + y_margin)
        if polygons_drawn:
            ax.set_xlim(x_lim)
            if flip_y:
                ax.set_ylim(y_lim[::-1])
            else:
                ax.set_ylim(y_lim)
        else:
            ax.plot([0, 1, 1, 0, 0], [0, 0, 1, 1, 0], color='darkgray', linestyle='-', linewidth=2, alpha=0.7)
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
    # Define views and projection normals
    views = [
        (np.array([0, 0, 1]), 'Top View', False),
        (user_normal, 'Isometric View', False),
        (np.array([0, -1, 0]), 'Front View', False),   # No Y-flip
        (np.array([1, 0, 0]), 'Side View', False)      # No Y-flip
    ]
    fig, axes = plt.subplots(2, 2, figsize=(12, 12))
    axes = axes.flatten()
    for i, ax in enumerate(axes):
        normal, label, flip_y = views[i]
        normal = normal / np.linalg.norm(normal)
        visible, hidden = view_polygons[i]
        print(f"[DEBUG] plot_four_views: {label} projection normal: [{normal[0]:.6f}, {normal[1]:.6f}, {normal[2]:.6f}]")
        print(f"[DEBUG] {label}: {len(visible)} visible, {len(hidden)} hidden polygons")
        plot_polygons_on_ax(ax, visible, hidden, label, flip_y)
    plt.tight_layout()
    
    # Create PDFfiles directory if it doesn't exist
    os.makedirs(pdf_dir, exist_ok=True)
    pdf_path = os.path.join(pdf_dir, "four_views.pdf")
    plt.savefig(pdf_path, format="pdf")
    plt.show()  # Removed to prevent empty plot
    
    # Return connectivity matrices for use in main function
    return view_connectivity_matrices


from Vertex_selection import extract_possible_vertices_from_summaries, project_to_view, filter_possible_vertices, make_summary_array

def main():


    parser = argparse.ArgumentParser(
        description=(
            'Solid projection and polygon visibility analysis.\n\n'
            'Displays two plots:\n'
            '1. Original solid faces as extracted polygons in 3D (interactive).\n'
            '2. User-requested polygons (visible, hidden, or combined) in 2D, based on CLI switches.'
        )
    )
    parser.add_argument(
        '--normal', type=str, default='1,1,1',
        help='Projection normal as comma-separated floats, e.g. "0.75,0.5,1"'
    )
    parser.add_argument(
        '--show_combined', action='store_true',
        help='Show combined plot of visible and hidden polygons'
    )
    parser.add_argument(
        '--show_visible', action='store_true',
        help='Show only visible polygons'
    )
    parser.add_argument(
        '--show_hidden', action='store_true',
        help='Show only hidden polygons'
    )
    parser.add_argument(
        '--seed', type=int, default=47315,
        help='Random seed for solid generation (int)'
    )
    parser.add_argument(
        '--rotate', type=str, default='0,0,0',
        help='Rotate solid before processing: angles in degrees as "x,y,z"'
    )
    parser.add_argument(
        '--config-file', type=str,
        help='Load configuration from file instead of generating random values'
    )
    parser.add_argument(
        '--save-config', action='store_true',
        help='Save configuration parameters to file'
    )
    parser.add_argument(
        '--quiet', action='store_true',
        help='Suppress verbose output'
    )
    args = parser.parse_args()

    # Handle configuration loading/creation
    if args.config_file:
        print(f"Loading configuration from: {args.config_file}")
        config = load_config(args.config_file)
        seed = config.seed
    else:
        print(f"Creating default configuration with seed: {args.seed}")
        config = create_default_config(args.seed)
        seed = args.seed

    # Save configuration if requested
    if args.save_config:
        config.save_to_file()

    # Apply seed from configuration
    config.apply_seed()

    print("[DEBUG] Starting main() function.")
    print(f"[DEBUG] CLI args: {sys.argv}")

    # Parse and normalize projection normal from argparse only
    try:
        normal_vals = [float(x) for x in args.normal.split(',')]
        projection_normal = np.array(normal_vals, dtype=float)
        norm = np.linalg.norm(projection_normal)
        if norm == 0:
            raise ValueError("Zero-length normal vector")
        projection_normal = projection_normal / norm
        print(f"[DEBUG] Projection normal: {projection_normal}")
    except Exception as e:
        print(f"[DEBUG] Could not parse projection normal: {args.normal} ({e})")
        projection_normal = np.array([1, 1, 1], dtype=float)
        projection_normal = projection_normal / np.linalg.norm(projection_normal)
    solid = build_solid_with_polygons_test(config=config, quiet=args.quiet)
    print(f"[DEBUG] Solid created: {type(solid)}")
    
    # Parse and apply rotation if requested
    try:
        rotation_angles = [float(x) for x in args.rotate.split(',')]
        if len(rotation_angles) != 3:
            raise ValueError("Need 3 angles")
        rx, ry, rz = rotation_angles
        
        if rx != 0 or ry != 0 or rz != 0:
            print(f"\n[DEBUG] Applying rotation: X={rx}°, Y={ry}°, Z={rz}°")
            from OCC.Core.gp import gp_Trsf, gp_Ax1, gp_Pnt, gp_Dir
            from OCC.Core.BRepBuilderAPI import BRepBuilderAPI_Transform
            import math
            
            # Create transformation
            trsf = gp_Trsf()
            
            # Apply rotations in order: X, then Y, then Z
            if rx != 0:
                axis_x = gp_Ax1(gp_Pnt(0, 0, 0), gp_Dir(1, 0, 0))
                trsf_x = gp_Trsf()
                trsf_x.SetRotation(axis_x, math.radians(rx))
                trsf.Multiply(trsf_x)
            
            if ry != 0:
                axis_y = gp_Ax1(gp_Pnt(0, 0, 0), gp_Dir(0, 1, 0))
                trsf_y = gp_Trsf()
                trsf_y.SetRotation(axis_y, math.radians(ry))
                trsf.Multiply(trsf_y)
            
            if rz != 0:
                axis_z = gp_Ax1(gp_Pnt(0, 0, 0), gp_Dir(0, 0, 1))
                trsf_z = gp_Trsf()
                trsf_z.SetRotation(axis_z, math.radians(rz))
                trsf.Multiply(trsf_z)
            
            # Apply transformation to solid
            transform = BRepBuilderAPI_Transform(solid, trsf, True)
            transform.Build()
            solid = transform.Shape()
            print(f"[DEBUG] Solid rotated successfully")
    except Exception as e:
        print(f"[DEBUG] Could not parse/apply rotation: {args.rotate} ({e})")
    
    save_solid_as_step(solid, "STEPfiles/solid_output.step")

    #Added by S. Bedi t make program more efficient
    face_polygons = extract_and_visualize_faces(solid, visualize=True)

    # Count total edges in the original solid
    print("\n" + "="*70)
    print("ORIGINAL SOLID TOPOLOGY")
    print("="*70)
    edge_explorer = TopExp_Explorer(solid, TopAbs_EDGE)
    total_edges = 0
    while edge_explorer.More():
        total_edges += 1
        edge_explorer.Next()
    expected_unique_edges = total_edges // 2
    print(f"  Total edges in original solid: {total_edges}")
    print("  Note: Each edge appears twice (shared between faces)")
    print(f"  Expected unique edges: {expected_unique_edges}")

    # Robust extraction of all unique vertices from the solid using TopExp_Explorer
    print("\n[DEBUG] Extracting all unique vertices from solid using TopExp_Explorer:")
    vertex_explorer = TopExp_Explorer(solid, TopAbs_VERTEX)
    unique_vertices = []
    seen = set()
    vertex_count = 0
    while vertex_explorer.More():
        vertex = topods.Vertex(vertex_explorer.Current())
        from OCC.Core.BRep import BRep_Tool
        pnt = BRep_Tool.Pnt(vertex)
        v = (round(pnt.X(), 6), round(pnt.Y(), 6), round(pnt.Z(), 6))
        if v not in seen:
            unique_vertices.append(v)
            seen.add(v)
        vertex_explorer.Next()
        vertex_count += 1
    # Order vertices by x, then y, then z
    all_vertices_sorted = sorted(unique_vertices, key=lambda v: (v[0], v[1], v[2]))
    print(f"Total number of unique vertices in the solid: {len(all_vertices_sorted)} (raw count: {vertex_count})")
    print("Ordered unique vertices (x, y, z):")
    for idx, v in enumerate(all_vertices_sorted):
        print(f"  {idx+1}: ({v[0]:.6f}, {v[1]:.6f}, {v[2]:.6f})")

    # Create square arrays for each view
    n_vertices = len(all_vertices_sorted)
    Vertex_Top_View = np.zeros((n_vertices, n_vertices), dtype=int)
    Vertex_Front_View = np.zeros((n_vertices, n_vertices), dtype=int)
    Vertex_Side_View = np.zeros((n_vertices, n_vertices), dtype=int)
    Vertex_Iso_View = np.zeros((n_vertices, n_vertices), dtype=int)

    # Display original polygons in 3D first - old
    #visualize_3d_solid(solid, all_vertices_sorted)
    #new S. Bedi
    visualize_3d_solid(face_polygons, all_vertices_sorted)

    # Create directories for output files (relative to script location)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(script_dir, "Output")
    pdf_dir = os.path.join(script_dir, "PDFfiles")
    
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(pdf_dir, exist_ok=True)

    # Pass arrays and ordered vertices to plot_four_views
    # old view_connectivity_matrices = plot_four_views(solid, projection_normal,
    view_connectivity_matrices = plot_four_views(face_polygons, projection_normal,
                   all_vertices_sorted,
                   Vertex_Top_View,
                   Vertex_Front_View,
                   Vertex_Side_View,
                   Vertex_Iso_View,
                   pdf_dir)

    # === Post-processing: Find z-levels from Front_View and build Possible_Vertices ===
    # This must come after the arrays are created and filled (after plot_four_views)
    print("\n[DEBUG] Using new connectivity matrices for vertex filtering...")

    # Use the new connectivity matrices instead of make_summary_array
    front_view_matrix = view_connectivity_matrices.get('Front View')
    top_view_matrix = view_connectivity_matrices.get('Top View')
    side_view_matrix = view_connectivity_matrices.get('Side View')
    
    # # Save connectivity matrices for debugging
    # if front_view_matrix is not None:
    #     np.save(os.path.join(output_dir, 'front_view_connectivity.npy'), front_view_matrix)
    #     print(f"[DEBUG] Saved front_view_connectivity.npy")
    # if top_view_matrix is not None:
    #     np.save(os.path.join(output_dir, 'top_view_connectivity.npy'), top_view_matrix)
    #     print(f"[DEBUG] Saved top_view_connectivity.npy")
    # if side_view_matrix is not None:
    #     np.save(os.path.join(output_dir, 'side_view_connectivity.npy'), side_view_matrix)
    #     print(f"[DEBUG] Saved side_view_connectivity.npy")

    # Check if we have the necessary connectivity matrices
    if front_view_matrix is not None and top_view_matrix is not None:
        print(f"[DEBUG] Front view matrix shape: {front_view_matrix.shape}")
        print(f"[DEBUG] Top view matrix shape: {top_view_matrix.shape}")
        print(f"\n[DEBUG] CONSERVATIVE APPROACH: Only use vertices that appear in actual edges")
        print(f"[DEBUG] Step 1: Extract vertices that participate in edges from each view")

        # Step 1: Extract unique (x, y) from top view matrix
        top_xy_coords = set()
        for i in range(top_view_matrix.shape[0]):
            x_proj, y_proj = top_view_matrix[i, 1], top_view_matrix[i, 2]
            top_xy_coords.add((x_proj, y_proj))  # Use exact coordinates

        # Step 1: Extract unique z-levels from front view matrix
        z_levels = set()
        for i in range(front_view_matrix.shape[0]):
            z_world = round(front_view_matrix[i, 2], 5)
            z_levels.add(z_world)  # Use rounded coordinates

        # Step 2: Generate candidate vertices (x, y, z)
        candidate_vertices = []

        for x, y in top_xy_coords:
            for z in z_levels:
                candidate_vertices.append([x, y, z])
        candidate_vertices = np.array(candidate_vertices)
        
        print(f"\n" + "="*60)
        print(f"TRUE REVERSE ENGINEERING - STEP-BY-STEP VERTEX RECONSTRUCTION")
        print(f"="*60)
        
        # TRUE REVERSE ENGINEERING APPROACH: Your specified logic
        # Step 1 - Extract coordinates from top and front views
        print("\nStep 1: Extracting coordinates from connectivity matrices...")
        print("Method: (x,y) from top view, z-levels from front view")
        print("Goal: Create all combinations to find real vertices via filtering")
        
        # Extract unique (x,y) coordinates from top view matrix (projected coordinates)
        top_xy_coords = set()
        print(f"DEBUG: Top view matrix has {top_view_matrix.shape[0]} rows")
        for i in range(top_view_matrix.shape[0]):
            x_proj, y_proj = top_view_matrix[i, 1], top_view_matrix[i, 2]
            top_xy_coords.add((x_proj, y_proj))  # Use exact coordinates

        # Extract unique z-levels from front view matrix (world z coordinates from column 2)
        # Note: In reverse engineering, we extract z from front view world coords 
        # since front view projection normal [0,1,0] preserves z-coordinate information
        raw_z = [round(front_view_matrix[i, 2], 6) for i in range(front_view_matrix.shape[0])]
        z_levels = sorted(set(raw_z))
        print(f"DEBUG: Front view matrix has {front_view_matrix.shape[0]} rows")

        print(f"Extracted (x,y) from top view: {len(top_xy_coords)} coordinates")
        print(f"Extracted z-levels from front view: {len(z_levels)} levels")
        print(f"z_levels: {z_levels}")

    # Step 2 - Generate candidate vertices: (x,y) × z-levels
    print("\nStep 2: Generating candidate vertices...")
    print("Method: Every (x,y) from top view at every z-level from front view")
    print("Note: Creates many 'fake' vertices - filtering identifies real ones")

    # Create candidate vertices: every (x,y) at every z-level
    candidate_vertices = []
    candidate_vertices = []
    for x, y in top_xy_coords:
        for z in z_levels:
            candidate_vertices.append([x, y, z])

    candidate_vertices = np.array(candidate_vertices)
    expected_count = len(top_xy_coords) * len(z_levels)
    print(f"Total candidate vertices: {len(candidate_vertices)}")
    print(f"Expected: {len(top_xy_coords)} × {len(z_levels)} = {expected_count}")

    # Show sample candidates
    if len(candidate_vertices) > 0:
        print("Sample candidate vertices:")
        for i in range(min(5, len(candidate_vertices))):
                x, y, z = candidate_vertices[i]
                print(f"  Candidate {i+1}: ({x:8.3f}, {y:8.3f}, {z:8.3f})")
        
        # Step 3: Filter candidates using Front and Side views ONLY
        print("\nStep 3: Filtering candidates by projection matching...")
        print("Method: Project candidates to FRONT and SIDE views for validation")
        print("Logic: Candidates from top view data, validated by other views")
        
        # Use the existing side view connectivity matrix
        if side_view_matrix is not None:
            print(f"\n[DEBUG] SIDE VIEW CONNECTIVITY MATRIX")
            print(f"Side view matrix shape: {side_view_matrix.shape}")
            print(f"Number of vertices in side view: {side_view_matrix.shape[0]}")
            if side_view_matrix.shape[1] > 3:
                print(f"Connectivity matrix columns: {side_view_matrix.shape[1] - 3}")
                # Print ALL vertices in side view
                print(f"\nALL {side_view_matrix.shape[0]} vertices in side view:")
                print(f"Format: V# | Index | 2D proj (y,z) | connections")
                for i in range(side_view_matrix.shape[0]):
                    vertex_idx = int(side_view_matrix[i, 0])
                    # Side view projects (y, z) from 3D (x, y, z)
                    proj_2d = (side_view_matrix[i, 1], side_view_matrix[i, 2])
                    connectivity = side_view_matrix[i, 3:]
                    connected_to = [j for j, val in enumerate(connectivity) if val > 0]
                    print(f"  V{i:2d}: Idx{vertex_idx} -> proj({proj_2d[0]:7.3f}, {proj_2d[1]:7.3f}) -> conn {connected_to}")
        
        if front_view_matrix is not None:
            print(f"\n[DEBUG] FRONT VIEW CONNECTIVITY MATRIX")
            print(f"Front view matrix shape: {front_view_matrix.shape}")
            print(f"Number of vertices in front view: {front_view_matrix.shape[0]}")
            if front_view_matrix.shape[1] > 3:
                print(f"Connectivity matrix columns: {front_view_matrix.shape[1] - 3}")
                # Print ALL vertices in front view
                print(f"\nALL {front_view_matrix.shape[0]} vertices in front view:")
                print(f"Format: V# | Index | 2D proj (x,z) | connections")
                for i in range(min(10, front_view_matrix.shape[0])):
                    vertex_idx = int(front_view_matrix[i, 0])
                    # Front view projects (x, z) from 3D (x, y, z)
                    proj_2d = (front_view_matrix[i, 1], front_view_matrix[i, 2])
                    connectivity = front_view_matrix[i, 3:]
                    connected_to = [j for j, val in enumerate(connectivity) if val > 0]
                    print(f"  V{i:2d}: Idx{vertex_idx} -> proj({proj_2d[0]:7.3f}, {proj_2d[1]:7.3f}) -> conn {connected_to}")
        
        if side_view_matrix is None:
            print("[ERROR] Could not create side view matrix")
            selected_vertices = np.array([])
        else:
            print(f"[DEBUG] Side view matrix shape: {side_view_matrix.shape}")
            
            def project_vertex_to_view_reverse_eng(vertex, normal):
                """Project a 3D vertex to 2D view coordinates using coordinate dropping for orthogonal views"""
                vertex = np.array(vertex)
                normal = np.array(normal)
                normal = normal / np.linalg.norm(normal)
                
                # Use coordinate dropping for standard orthogonal engineering views
                # This matches both connectivity matrix and edge reconstruction methods
                if np.allclose(normal, [0, 0, 1], atol=1e-3):  # Top view
                    return vertex[0], vertex[1]  # Drop Z, keep X,Y
                elif np.allclose(normal, [0, -1, 0], atol=1e-3):  # Front view
                    return vertex[0], vertex[2]  # Drop Y, keep X,Z
                elif np.allclose(normal, [1, 0, 0], atol=1e-3):  # Side view
                    return vertex[1], vertex[2]  # Drop X, keep Y,Z
                else:
                    # For non-orthogonal views, use basis vector method
                    # Create orthogonal basis vectors for the projection plane
                    if abs(normal[0]) < 0.9:
                        temp = np.array([1.0, 0.0, 0.0])
                    else:
                        temp = np.array([0.0, 1.0, 0.0])
                    
                    u = temp - np.dot(temp, normal) * normal
                    u = u / np.linalg.norm(u)
                    v = np.cross(normal, u)
                    v = v / np.linalg.norm(v)
                    
                    proj_u = np.dot(vertex, u)
                    proj_v = np.dot(vertex, v)
                    return proj_u, proj_v
            
            # Extract projected coordinates from view summaries for filtering
            # Front view summary: columns 3,5 are the projected (u,v) coordinates  
            # Side view summary: columns 4,5 are the projected (u,v) coordinates
            tolerance = 1.0e-4
            
            front_view_coords = []
            for i in range(front_view_matrix.shape[0]):
                u_proj, v_proj = front_view_matrix[i, 1], front_view_matrix[i, 2]
                front_view_coords.append((u_proj, v_proj))
            
            side_view_coords = []
            for i in range(side_view_matrix.shape[0]):
                u_proj, v_proj = side_view_matrix[i, 1], side_view_matrix[i, 2]
                side_view_coords.append((u_proj, v_proj))
                
            print(f"Available front view coordinates: {len(front_view_coords)}")
            print(f"Available side view coordinates: {len(side_view_coords)}")
            print("DEBUG: Using front and side views to validate candidates with tolerance")
            
            # Filter candidates: keep those that project to coordinates that match
            # BOTH front view AND side view projected coordinates with tolerance
            selected_vertices = []
            
            # Debug counters
            front_matches = 0
            side_matches = 0
            dual_matches = 0
            for vertex in candidate_vertices:
                front_proj = project_vertex_to_view_reverse_eng(vertex, [0, -1, 0])  # Front view normal
                side_proj = project_vertex_to_view_reverse_eng(vertex, [1, 0, 0])    # Side view normal
                
                front_match = any(np.allclose(front_proj, fc, atol=tolerance) for fc in front_view_coords)
                side_match = any(np.allclose(side_proj, sc, atol=tolerance) for sc in side_view_coords)
                
                if front_match:
                    front_matches += 1
                if side_match:
                    side_matches += 1
                if front_match and side_match:
                    dual_matches += 1
                    selected_vertices.append(vertex)
            
            selected_vertices = np.array(selected_vertices)
            
            print(f"Candidates matching front view: {front_matches}")
            print(f"Candidates matching side view: {side_matches}")
            print(f"Candidates matching BOTH views: {dual_matches}")
            print(f"Selected vertices (before edge validation): {len(selected_vertices)}")
            
            # Store the Step 3 results for visualization (21 vertices)
                # Deduplicate step3_vertices using np.unique with rounding for floating point tolerance
                # (fix indentation)
                # Store the Step 3 results for visualization (21 vertices)
                # Deduplicate step3_vertices using np.unique with rounding for floating point tolerance
                # Deduplicate selected vertices
            # Deduplicate selected vertices
            step3_vertices = np.unique(np.round(selected_vertices, decimals=6), axis=0)
            
            print(f"Original solid vertices: {len(all_vertices_sorted)}")
            
            print(f"\n" + "="*60)
            print(f"TRUE REVERSE ENGINEERING RESULTS")
            print(f"="*60)
            
            print(f"\nReconstructed vertices (x, y, z):")
            
            # Sort for consistent output
            if len(step3_vertices) > 0:
                step3_vertices = step3_vertices[np.lexsort((step3_vertices[:, 2], step3_vertices[:, 1], step3_vertices[:, 0]))]
                for i, vertex in enumerate(step3_vertices):
                    x, y, z = vertex
                    print(f"  Vertex {i+1:2d}: ({x:8.3f}, {y:8.3f}, {z:8.3f})")
                print(f"="*60)

                # === Extract projections and connectivity matrices for merged matrix ===
                print("\n[DEBUG] Extracting projections and connectivity matrices for merged matrix...")
                def find_matching_row(proj, matrix, tol=1e-5):
                    for i in range(matrix.shape[0]):
                        if np.allclose(proj, matrix[i, 1:3], atol=tol):
                            return i
                    return None


                N = step3_vertices.shape[0]
                top_proj = np.zeros((N, 2))
                front_proj = np.zeros((N, 2))
                side_proj = np.zeros((N, 2))
                top_conn = np.zeros((N, N))
                front_conn = np.zeros((N, N))
                side_conn = np.zeros((N, N))

                def project_vertex_to_view_reverse_eng(vertex, normal):
                    vertex = np.array(vertex)
                    normal = np.array(normal)
                    normal = normal / np.linalg.norm(normal)
                    if np.allclose(normal, [0, 0, 1], atol=1e-3):
                        return vertex[0], vertex[1]
                    elif np.allclose(normal, [0, -1, 0], atol=1e-3):
                        return vertex[0], vertex[2]
                    elif np.allclose(normal, [1, 0, 0], atol=1e-3):
                        return vertex[1], vertex[2]
                    else:
                        if abs(normal[0]) < 0.9:
                            temp = np.array([1.0, 0.0, 0.0])
                        else:
                            temp = np.array([0.0, 1.0, 0.0])
                        u = temp - np.dot(temp, normal) * normal
                        u = u / np.linalg.norm(u)
                        v = np.cross(normal, u)
                        v = v / np.linalg.norm(v)
                        proj_u = np.dot(vertex, u)
                        proj_v = np.dot(vertex, v)
                        return proj_u, proj_v

                # Build projection arrays
                for idx, vertex in enumerate(step3_vertices):
                    top_proj[idx] = project_vertex_to_view_reverse_eng(vertex, [0, 0, 1])
                    front_proj[idx] = project_vertex_to_view_reverse_eng(vertex, [0, -1, 0])
                    side_proj[idx] = project_vertex_to_view_reverse_eng(vertex, [1, 0, 0])

                # Build square connectivity matrices
                for i in range(N):
                    # For each reconstructed vertex i
                    tp_i = top_proj[i]
                    fp_i = front_proj[i]
                    sp_i = side_proj[i]
                    top_idx_i = find_matching_row(tp_i, top_view_matrix)
                    front_idx_i = find_matching_row(fp_i, front_view_matrix)
                    side_idx_i = find_matching_row(sp_i, side_view_matrix)
                    for j in range(N):
                        tp_j = top_proj[j]
                        fp_j = front_proj[j]
                        sp_j = side_proj[j]
                        top_idx_j = find_matching_row(tp_j, top_view_matrix)
                        front_idx_j = find_matching_row(fp_j, front_view_matrix)
                        side_idx_j = find_matching_row(sp_j, side_view_matrix)
                        # Top view
                        if top_idx_i is not None and top_idx_j is not None:
                            top_conn[i, j] = top_view_matrix[top_idx_i, 3 + top_idx_j]
                        else:
                            top_conn[i, j] = 0
                        # Front view
                        if front_idx_i is not None and front_idx_j is not None:
                            front_conn[i, j] = front_view_matrix[front_idx_i, 3 + front_idx_j]
                        else:
                            front_conn[i, j] = 0
                        # Side view
                        if side_idx_i is not None and side_idx_j is not None:
                            side_conn[i, j] = side_view_matrix[side_idx_i, 3 + side_idx_j]
                        else:
                            side_conn[i, j] = 0
                        if np.allclose(tp_i, tp_j, atol=1e-6) and front_conn[i, j] == 1 and side_conn[i, j] == 1:
                            top_conn[i, j] = 1
                        if np.allclose(fp_i, fp_j, atol=1e-6) and top_conn[i, j] == 1 and side_conn[i, j] == 1:
                            front_conn[i, j] = 1
                        if np.allclose(sp_i, sp_j, atol=1e-6) and front_conn[i, j] == 1 and top_conn[i, j] == 1:
                            side_conn[i, j] = 1

                print("[DEBUG] TOP VIEW PROJECTIONS:\n", top_proj)
                print("[DEBUG] TOP VIEW CONNECTIVITY MATRIX:\n", top_conn)
                print("[DEBUG] FRONT VIEW PROJECTIONS:\n", front_proj)
                print("[DEBUG] FRONT VIEW CONNECTIVITY MATRIX:\n", front_conn)
                print("[DEBUG] SIDE VIEW PROJECTIONS:\n", side_proj)
                print("[DEBUG] SIDE VIEW CONNECTIVITY MATRIX:\n", side_conn)

                # === Build and print merged connectivity matrix ===
                print("\n[DEBUG] Building merged connectivity matrix (sum of top, front, and side)...")
                N = step3_vertices.shape[0]
                merged_conn = top_conn + front_conn + side_conn
                print("[DEBUG] Merged connectivity matrix (top + front + side):")
                print(merged_conn)

                # Visualize 3D solid with edges from merged_conn
                print("\n[DEBUG] Visualizing 3D solid with reconstructed edges from merged connectivity matrix...")
                edges = []
                for j in range(N):
                    for k in range(j+1, N):
                        if merged_conn[j, k] > 0:
                            edges.append((j, k))
                # old visualize_3d_solid(solid_shape=solid, selected_vertices=step3_vertices, edges=edges)
                visualize_3d_solid(face_polygons, selected_vertices=step3_vertices, edges=edges)
        
    else:
        print("[ERROR] Could not create summary arrays for vertex filtering")
        # Still show the solid even if filtering failed
        # old visualize_3d_solid(solid, None)
        visualize_3d_solid(face_polygons, None)
    #extract_possible_vertices_from_summaries(Vertex_Front_View, Vertex_Top_View, all_vertices_sorted)
    






if __name__ == "__main__":    main()


