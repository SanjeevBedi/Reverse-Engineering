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
        # Offset vertex label further for readability
        ax.text(v[0]+3.0, v[1]+3.0, f'V{i}', fontsize=9, color='darkred', ha='left', va='bottom')
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
        # Offset vertex label further for readability
        ax.text(v[0]+3.0, v[1]+3.0, f'V{i}', fontsize=9, color='darkred', ha='left', va='bottom')
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
    from OCC.Core.BRepAlgoAPI import BRepAlgoAPI_Fuse  # <-- Added import for Fuse
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
    cut_shape2 = original
    # cut_shape1 = original
    #(You can add your custom boolean operations here if needed)
    #return original
    #Create box at (0,0,0) of size (60,50,60)
 

    # # Create box at (0,0,0) of size (60,50,60)
    # box = BRepPrimAPI_MakeBox(60, 55, 25).Shape()
    # # Move box to (10,25,0)
    # trsf = gp_Trsf()
    # trsf.SetTranslation(gp_Vec(0, 0, 0))
    # moved_box = BRepBuilderAPI_Transform(box, trsf, True).Shape()

    # # Subtract box from original
    # cut = BRepAlgoAPI_Cut(cut_shape2, moved_box)
    # cut_shape2 = cut.Shape()

    # # Create box at (0,0,0) of size (60,50,60)
    # box = BRepPrimAPI_MakeBox(50, 20, 25).Shape()
    # # Move box to (10,25,0)
    # trsf = gp_Trsf()
    # trsf.SetTranslation(gp_Vec(0, 0.5, 0))
    # moved_box = BRepBuilderAPI_Transform(box, trsf, True).Shape()

    # # Subtract box from original
    # cut = BRepAlgoAPI_Cut(cut_shape2, moved_box)
    # cut_shape2 = cut.Shape()

    # # Create box at (0,0,0) of size (60,50,60)
    # box = BRepPrimAPI_MakeBox(35, 23, 60).Shape()
    # # Move box to (10,25,0)
    # trsf = gp_Trsf()
    # trsf.SetTranslation(gp_Vec(0, 0, 0))
    # moved_box = BRepBuilderAPI_Transform(box, trsf, True).Shape()

    # # Subtract box from original
    # cut = BRepAlgoAPI_Cut(cut_shape2, moved_box)
    # cut_shape2 = cut.Shape()

    # # Create box at (0,0,0) of size (60,50,60)
    # box = BRepPrimAPI_MakeBox(45, 23, 60).Shape()
    # # Move box to (10,25,0)
    # trsf = gp_Trsf()
    # trsf.SetTranslation(gp_Vec(20, 27, 0))
    # moved_box = BRepBuilderAPI_Transform(box, trsf, True).Shape()

    # # Subtract box from original
    # cut = BRepAlgoAPI_Cut(cut_shape2, moved_box)
    # cut_shape2 = cut.Shape()

    # # Create box at (0,0,0) of size (60,50,60)
    # box = BRepPrimAPI_MakeBox(22, 20, 60).Shape()
    # # Move box to (10,25,0)
    # trsf = gp_Trsf()
    # trsf.SetTranslation(gp_Vec(0, 20, 0))
    # moved_box = BRepBuilderAPI_Transform(box, trsf, True).Shape()

    # # Subtract box from original
    # cut = BRepAlgoAPI_Cut(cut_shape2, moved_box)
    # cut_shape2 = cut.Shape()

    # # Create box at (0,0,0) of size (60,50,60)
    # box = BRepPrimAPI_MakeBox(10, 5, 60).Shape()
    # # Move box to (10,25,0)
    # trsf = gp_Trsf()
    # trsf.SetTranslation(gp_Vec(34, 23, 0))
    # moved_box = BRepBuilderAPI_Transform(box, trsf, True).Shape()

    # # Subtract box from original
    # cut = BRepAlgoAPI_Cut(cut_shape2, moved_box)
    # cut_shape2 = cut.Shape()

    # # Create box at (0,0,0) of size (60,50,60)
    # box = BRepPrimAPI_MakeBox(15, 10, 23).Shape()
    # # Move box to (10,25,0)
    # trsf = gp_Trsf()
    # trsf.SetTranslation(gp_Vec(20, 20, 37))
    # moved_box = BRepBuilderAPI_Transform(box, trsf, True).Shape()

    # # Subtract box from original
    # cut = BRepAlgoAPI_Cut(cut_shape2, moved_box)
    # cut_shape2 = cut.Shape()
    # --- Apply scaling to cut_shape2 ---
    from OCC.Core.gp import gp_GTrsf, gp_Mat
    from OCC.Core.BRepBuilderAPI import BRepBuilderAPI_GTransform
    # Define scaling factors (set these as needed)
    scalex, scaly, scalez = 1.0, 1.0, 1.0  # Example: no scaling
    mat = gp_Mat(
        scalex, 0, 0,
        0, scaly, 0,
        0, 0, scalez
    )
    gtrsf = gp_GTrsf()
    gtrsf.SetVectorialPart(mat)
    scaled_shape = BRepBuilderAPI_GTransform(cut_shape2, gtrsf, True).Shape()
    cut_shape2 = scaled_shape

    return cut_shape2


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
                        pass
                        
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
    # plt.show() removed - controlled by caller (Build_Solid.py)
    
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
    # --- Unified 3D plot: show both original solid polygons and extracted polygons ---
    import matplotlib.pyplot as plt
    import numpy as np
    import inspect

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Plot original solid polygons (faces) using cached face_polygons
    face_handles = []
    for idx, poly_data in enumerate(face_polygons):
        verts = poly_data.get('outer_boundary', [])
        verts = np.array(verts)
        if idx == 0:
            print(f"[DEBUG] First face polygon in plotting call: {verts}")
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

    # Unified check buttons for all elements
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
    ax.set_title('3D Solid Reconstruction: Original + Extracted Polygons')
    plt.show()

   

def classify_faces_by_projection(face_polygons, unit_projection_normal, no_graphics=False):
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
    
    # Only show plot if not suppressed by no_graphics flag
    # (Side view is shown as an example when graphics are enabled)
    if (not no_graphics and 
            np.allclose(unit_projection_normal, [1, 0, 0], atol=1e-3)):
        print("\nSUMMARY: Plot of Array_B and Array_C for Side View:")
        plot_arrays_visualization(
            array_A_initial,
            array_B,
            array_C,
            unit_projection_normal
        )
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
    pdf_dir="PDFfiles",
    units="cm",
    drawing_scale_real=1.0,
    drawing_scale_drawing=1.0,
    no_graphics=False,
    seed=None):
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
        _, array_B, array_C = classify_faces_by_projection(
            face_polygons, normal, no_graphics)
        visible = [data['polygon'] for data in array_B if 'polygon' in data]
        hidden = [data['polygon'] for data in array_C if 'polygon' in data]

        # Create connectivity matrix for this view
        connectivity_matrix = create_view_connectivity_matrix(
            array_B, array_C, normal, label, ordered_vertices)
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
    
    # Add title with scale and unit information
    scale_str = f"1:{drawing_scale_drawing}" if drawing_scale_drawing != 1.0 else "1:1"
    fig.suptitle(f'Engineering Drawing Views - Seed {hash(str(ordered_vertices[0])) % 100000}\n'
                 f'Units: {units} | Scale: {scale_str}',
                 fontsize=14, fontweight='bold')
    
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
    pdf_filename = f"four_views_seed_{seed}.pdf" if seed is not None else "four_views.pdf"
    pdf_path = os.path.join(pdf_dir, pdf_filename)
    plt.savefig(pdf_path, format="pdf")
    print(f"[DEBUG] Saved four-view plot to: {pdf_path}")
    plt.show()  # Removed to prevent empty plot
    
    # Return connectivity matrices for use in main function
    return view_connectivity_matrices


from Vertex_selection import extract_possible_vertices_from_summaries, project_to_view, filter_possible_vertices, make_summary_array


def split_colinear_edges_in_faces(faces, selected_vertices, tolerance=1e-6):
    """
    Split colinear edges in all face polygons (boundaries, holes, and alternates).
    
    For each edge in a polygon, check if any other vertex in the solid lies on that edge.
    If so, split the edge by inserting that vertex into the polygon.
    
    Parameters:
        faces: List of face dicts with 'vertices', 'holes', 'alternates'
        selected_vertices: Nx3 array of all vertex coordinates
        tolerance: Distance tolerance for colinearity check
        
    Returns:
        Updated faces list with split edges
    """
    
    def point_on_segment(p, v1, v2, tol=1e-6):
        """
        Check if point p lies on line segment from v1 to v2.
        Returns True if p is between v1 and v2 (not at endpoints).
        """
        # Vector from v1 to v2
        seg = v2 - v1
        seg_len = np.linalg.norm(seg)
        
        if seg_len < tol:
            return False  # Degenerate segment
        
        # Vector from v1 to p
        v1p = p - v1
        
        # Check if p is colinear with segment
        cross = np.cross(seg, v1p)
        if np.linalg.norm(cross) > tol * seg_len:
            return False  # Not colinear
        
        # Check if p is between v1 and v2 (not at endpoints)
        dot = np.dot(v1p, seg)
        if dot < tol or dot > (seg_len * seg_len - tol):
            return False  # At or beyond endpoints
        
        return True
    
    def split_polygon_edges(vertex_list, all_verts):
        """
        Split edges in a polygon by inserting intermediate vertices.
        
        Parameters:
            vertex_list: List of vertex indices forming the polygon
            all_verts: Nx3 array of all vertex coordinates
            
        Returns:
            Updated vertex list with split edges
        """
        modified = True
        max_iterations = 10
        iteration = 0
        
        while modified and iteration < max_iterations:
            modified = False
            iteration += 1
            new_vertex_list = []
            
            n = len(vertex_list)
            for i in range(n):
                v1_idx = vertex_list[i]
                v2_idx = vertex_list[(i + 1) % n]
                
                v1 = all_verts[v1_idx]
                v2 = all_verts[v2_idx]
                
                # Add the current vertex
                new_vertex_list.append(v1_idx)
                
                # Check all other vertices to see if they lie on this edge
                intermediate_points = []
                
                for test_idx in range(len(all_verts)):
                    # Skip if it's an endpoint or already in polygon
                    if test_idx == v1_idx or test_idx == v2_idx:
                        continue
                    
                    test_pt = all_verts[test_idx]
                    
                    if point_on_segment(test_pt, v1, v2, tolerance):
                        # Calculate distance from v1 to order intermediate points
                        dist = np.linalg.norm(test_pt - v1)
                        intermediate_points.append((dist, test_idx))
                
                # Sort intermediate points by distance from v1
                if intermediate_points:
                    intermediate_points.sort(key=lambda x: x[0])
                    for _, idx in intermediate_points:
                        new_vertex_list.append(idx)
                        modified = True
            
            vertex_list = new_vertex_list
        
        return vertex_list
    
    # Split edges in all face polygons
    total_splits = 0
    
    for face_idx, face in enumerate(faces):
        face_splits = 0
        
        # Split boundary edges
        original_boundary = face['vertices'][:]
        split_boundary = split_polygon_edges(face['vertices'], selected_vertices)
        if len(split_boundary) > len(original_boundary):
            face['vertices'] = split_boundary
            face_splits += len(split_boundary) - len(original_boundary)
        
        # Split hole edges
        for hole_idx, hole in enumerate(face['holes']):
            original_hole = hole[:]
            split_hole = split_polygon_edges(hole, selected_vertices)
            if len(split_hole) > len(original_hole):
                face['holes'][hole_idx] = split_hole
                face_splits += len(split_hole) - len(original_hole)
        
        # Split alternate boundary edges
        if 'alternates' in face:
            for alt_idx, alt in enumerate(face['alternates']):
                original_alt = alt['vertices'][:]
                split_alt = split_polygon_edges(alt['vertices'], selected_vertices)
                if len(split_alt) > len(original_alt):
                    face['alternates'][alt_idx]['vertices'] = split_alt
                    face_splits += len(split_alt) - len(original_alt)
        
        if face_splits > 0:
            total_splits += face_splits
            print(f"[SPLIT EDGES] Face {face_idx+1}: Split {face_splits} edge(s)")
    
    if total_splits == 0:
        print("[SPLIT EDGES] No colinear edges found to split")
    else:
        print(f"[SPLIT EDGES] Total: {total_splits} edge(s) split across all faces")
    
    return faces


def extract_polygon_faces_from_connectivity(selected_vertices, merged_conn, tolerance=1e-6):
    from shapely.geometry import Polygon

    def group_polygons_with_holes(polygons):
        """
        Given a list of dicts with 'shapely_2d' and 'vertices', group into faces with holes.
        
        New logic:
        - One polygon ENCLOSES all others (largest area)
        - One polygon EXCLUDES alternates but shares edges with them (correct boundary, result of merges)
        - Other polygons that touch boundaries are alternates
        - Polygons completely inside with no boundary contact are holes
        
        Returns a list of dicts: {
            'exterior': ..., 
            'holes': [...],
            'alternates': [...]  # Alternate boundary definitions
        }
        """
        from shapely.geometry import Polygon
        
        # Helper to count shared edges
        def count_shared_edges(poly1_verts, poly2_verts):
            edges1 = set()
            for i in range(len(poly1_verts)):
                v1, v2 = poly1_verts[i], poly1_verts[(i+1) % len(poly1_verts)]
                edges1.add((min(v1, v2), max(v1, v2)))
            edges2 = set()
            for i in range(len(poly2_verts)):
                v1, v2 = poly2_verts[i], poly2_verts[(i+1) % len(poly2_verts)]
                edges2.add((min(v1, v2), max(v1, v2)))
            return len(edges1 & edges2)
        
        # Sort polygons by area descending
        sorted_polys = sorted(polygons, key=lambda p: p['shapely_2d'].area, reverse=True)
        used = set()
        faces = []
        
        for i, enclosing in enumerate(sorted_polys):
            if i in used:
                continue
            
            # Find all candidates that are inside or touch this enclosing polygon
            holes = []
            alternates = []
            candidates_indices = []
            
            for j, candidate in enumerate(sorted_polys):
                if i == j or j in used:
                    continue
                
                enclosing_poly = enclosing['shapely_2d']
                cand_poly = candidate['shapely_2d']
                
                # Check spatial relationship
                if enclosing_poly.contains(cand_poly):
                    candidates_indices.append(j)
                    if enclosing_poly.boundary.intersects(cand_poly.boundary):
                        # Touches boundary - potential alternate
                        alternates.append({
                            'vertices': candidate['vertices'],
                            'shapely_2d': cand_poly,
                            'area': cand_poly.area,
                            'index': j
                        })
                        print(f"[POLY FORM]       Polygon {j+1} TOUCHES boundary of Polygon {i+1} -> candidate alternate")
                    else:
                        # Completely inside - true hole
                        holes.append(candidate['vertices'])
                        print(f"[POLY FORM]       Polygon {j+1} is INSIDE Polygon {i+1} -> hole (pocket)")
                        used.add(j)
            
            # If we have alternates, check which polygon shares the most edges with them
            # The one with most shared edges is the "post-merge" boundary (excludes alternates)
            correct_boundary = enclosing
            correct_index = i
            
            if alternates:
                # Check enclosing polygon and all alternates to find which shares most edges
                all_candidates = [{'poly': enclosing, 'index': i}] + \
                                [{'poly': alt, 'index': alt['index']} for alt in alternates]
                
                max_shared = -1
                for cand in all_candidates:
                    shared_count = 0
                    cand_verts = cand['poly']['vertices'] if 'vertices' in cand['poly'] else cand['poly'].get('vertices', [])
                    
                    # Count edges shared with OTHER alternates/candidates
                    for other in all_candidates:
                        if other['index'] == cand['index']:
                            continue
                        other_verts = other['poly']['vertices'] if 'vertices' in other['poly'] else other['poly'].get('vertices', [])
                        shared_count += count_shared_edges(cand_verts, other_verts)
                    
                    if shared_count > max_shared:
                        max_shared = shared_count
                        correct_boundary = cand['poly'] if 'shapely_2d' in cand['poly'] else enclosing
                        correct_index = cand['index']
                
                # If correct boundary is not the enclosing, swap them
                if correct_index != i:
                    print(f"[POLY FORM]       CHOOSING Polygon {correct_index+1} as boundary (shares {max_shared} edges with alternates)")
                    # Remove correct_boundary from alternates list
                    alternates = [alt for alt in alternates if alt['index'] != correct_index]
                    # Add enclosing to alternates
                    alternates.insert(0, {
                        'vertices': enclosing['vertices'],
                        'shapely_2d': enclosing['shapely_2d'],
                        'area': enclosing['shapely_2d'].area
                    })
            
            # Mark all alternates as used
            for alt in alternates:
                if 'index' in alt:
                    used.add(alt['index'])
            
            # Mark the correct boundary as used
            used.add(correct_index)
            
            # Clean up alternates (remove 'index' field)
            cleaned_alternates = [{
                'vertices': alt['vertices'],
                'shapely_2d': alt['shapely_2d'],
                'area': alt['area']
            } for alt in alternates]
            
            faces.append({
                'exterior': correct_boundary['vertices'], 
                'holes': holes,
                'alternates': cleaned_alternates
            })
        
        return faces
    """
    Extract polygon faces from connectivity matrix using planar face detection.
    
    Algorithm:
    1. For each row, find all vectors to connected vertices (conn=3)
    2. Generate face normals from non-collinear vector pairs
    3. Create unique list of face equations
    4. Find all vertices on each face
    5. Build list of possible edges on each face
    6. Join edges into closed polygons
    7. Identify outer boundaries and inner holes
    
    Parameters:
        selected_vertices: Nx3 array of vertex coordinates
        merged_conn: NxN connectivity matrix (values 0-3, where 3=visible in all views)
        tolerance: float, tolerance for geometric comparisons
    
    Returns:
        List of face dictionaries containing vertices, normal, edges, etc.
    """
    print("\n" + "="*70)
    print("[POLY FORM] POLYGON FACE EXTRACTION FROM CONNECTIVITY MATRIX")
    print("="*70)
    
    N = len(selected_vertices)
    faces = []
    face_equations = []  # List of (normal, d) tuples
    
    # Step 1 & 2: Generate face equations from connectivity matrix
    print("\n[POLY FORM] Step 1-2: Generating face ranges from connectivity")
    print("-" * 70)
    
    # Build adjacency list: for each vertex, store connected vertices
    adjacency = {i: [] for i in range(N)}
    edge_count = 0
    for i in range(N):
        for j in range(i+1, N):
            if merged_conn[i, j] == 3:
                adjacency[i].append(j)
                adjacency[j].append(i)
                edge_count += 1
    
    # For each vertex, generate face normals from pairs of connected edges
    pairs_checked = 0
    pairs_added = 0
    for i_row in range(N):
        connected_vertices = adjacency[i_row]
        
        if len(connected_vertices) < 2:
            continue
        
        # Generate face normals from all pairs of edges from this vertex
        for idx1 in range(len(connected_vertices)):
            for idx2 in range(idx1+1, len(connected_vertices)):
                j = connected_vertices[idx1]
                k = connected_vertices[idx2]
                pairs_checked += 1
                
                # Account for vertex rounding errors by computing plane range
                # Perturb common vertex v1 along e1 direction
                eps_base = 0.1  # 0.1mm base error bound
                
                # Vertices
                v1 = selected_vertices[i_row]
                v2 = selected_vertices[j]
                v3 = selected_vertices[k]
                
                # Step 1: Compute cross product of two neighbouring edges
                # tnrm = (v2-v1) X (v3-v1)
                e1 = v2 - v1
                e2 = v3 - v1
                e1_mag = np.linalg.norm(e1)
                e2_mag = np.linalg.norm(e2)
                
                tnrm = np.cross(e1, e2)
                tnrm_mag = np.linalg.norm(tnrm)
                
                # Skip if edges are collinear (degenerate face)
                if tnrm_mag < tolerance:
                    continue
                
                # Normalize tnrm to get unit normal
                tnrm_unit = tnrm / tnrm_mag
                
                # Step 2: Find unit vector perpendicular to edge e1 in the plane
                # derr = tnrm X (v2-v1), then normalize
                derr_v1 = np.cross(tnrm_unit, e1)
                derr_v1_mag = np.linalg.norm(derr_v1)
                if derr_v1_mag < tolerance:
                    continue
                derr_v1_unit = derr_v1 / derr_v1_mag
                
                # Scale error by edge length: eps = eps_base * (edge_mag / 100.0)
                eps_scaled = eps_base * (e1_mag / 100.0)
                
                # Step 3: Perturb v1 (common vertex) along derr_v1_unit
                v1m = v1 - eps_scaled * derr_v1_unit
                v1p = v1 + eps_scaled * derr_v1_unit
                
                # Step 4: Compute the plane range
                # Create edges from perturbed v1 to v2 and v3
                e1_from_v1m = v2 - v1m
                e1_from_v1p = v2 - v1p
                e2_from_v1m = v3 - v1m
                e2_from_v1p = v3 - v1p
                
                # Compute two normals from the perturbed positions
                n_P = np.cross(e1_from_v1p, e2_from_v1p)
                n_M = np.cross(e1_from_v1m, e2_from_v1m)
                
                n_P_mag = np.linalg.norm(n_P)
                n_M_mag = np.linalg.norm(n_M)
                
                # Skip if either normal is degenerate
                if n_P_mag < tolerance or n_M_mag < tolerance:
                    continue
                
                # Normalize both normals
                n_P = n_P / n_P_mag
                n_M = n_M / n_M_mag
                
                # Clean up signed zeros
                n_P = np.where(np.abs(n_P) < 1e-10, 0.0, n_P)
                n_M = np.where(np.abs(n_M) < 1e-10, 0.0, n_M)
                
                # Check if average normal is valid (normals don't cancel out)
                n_avg_test = (n_P + n_M) / 2
                n_avg_test_mag = np.linalg.norm(n_avg_test)
                
                # Skip this edge pair if normals cancel (degenerate face)
                if n_avg_test_mag < 1e-10:
                    continue
                
                # Compute angular separation
                max_angle = np.arccos(np.clip(np.dot(n_P, n_M), -1, 1))
                
                # Compute d values for both planes using vertex v1
                d_P = -np.dot(n_P, v1)
                d_M = -np.dot(n_M, v1)
                
                # If planes are too close (angular separation < 0.001 radians),
                # use d-value range instead
                d_range_fallback = 0.1  # 0.1mm fallback range
                if max_angle < 0.001:  # ~0.057 degrees
                    # Use average normal and create d-range
                    n_avg = n_avg_test / n_avg_test_mag
                    d_avg = -np.dot(n_avg, v1)
                    n_P = n_avg
                    n_M = n_avg
                    d_P = d_avg + d_range_fallback / 2
                    d_M = d_avg - d_range_fallback / 2
                
                # Add plane range to list (will check uniqueness in Step 3)
                face_equations.append({
                    'normal_P': n_P,
                    'd_P': d_P,
                    'normal_M': n_M,
                    'd_M': d_M,
                    'source_row': i_row,
                    'vertices_used': [i_row, j, k]
                })
                pairs_added += 1
    
    print(f"[POLY FORM]   {edge_count} edges, {pairs_checked} pairs checked, "
          f"{pairs_added} valid ranges")
    
    # Step 3: Remove duplicate face ranges
    print("\n[POLY FORM] Step 3: Removing duplicate face ranges")
    
    # Initial tolerances for raw face equations
    # 1 degree: cos(1°) ≈ 0.99985, so 1 - 0.99985 = 0.00015
    initial_normal_tol = 0.00015  # ~1 degree
    initial_d_tol = 1.0  # 1mm
    
    unique_faces = []
    for face_eq in face_equations:
        is_duplicate = False

        # Compute average normal for this face
        n1_avg = (face_eq['normal_P'] + face_eq['normal_M']) / 2
        n1_avg = n1_avg / np.linalg.norm(n1_avg)
        n1_avg = np.where(np.abs(n1_avg) < 1e-10, 0.0, n1_avg)  # Clean signed zeros
        d1_avg = (face_eq['d_P'] + face_eq['d_M']) / 2

        for uf_idx, unique_face in enumerate(unique_faces):
            # Get average normals for comparison
            n2_avg = unique_face['normal']
            d2_avg = unique_face['d']
            
            # Check if normals are parallel (same or opposite direction)
            dot_normals = np.dot(n1_avg, n2_avg)
            normals_parallel = abs(abs(dot_normals) - 1.0) < initial_normal_tol
            
            if normals_parallel:
                # Normals are parallel - check if same plane
                # For opposite normals, d-values have opposite signs
                # Same plane if |d1 + d2| < tolerance (opposite normals)
                # or |d1 - d2| < tolerance (same normals)
                
                if dot_normals > 0:
                    # Same direction: check |d1 - d2|
                    same_plane = abs(d1_avg - d2_avg) < initial_d_tol
                else:
                    # Opposite direction: check |d1 + d2|
                    same_plane = abs(d1_avg + d2_avg) < initial_d_tol
                
                if same_plane:
                    is_duplicate = True
                    break

        if not is_duplicate:
            # Store cleaned average values for compatibility
            face_eq['normal'] = n1_avg
            face_eq['d'] = d1_avg
            
            unique_faces.append(face_eq)
            
            # Show the two bounding planes that define the range
            n_P = face_eq['normal_P']
            d_P = face_eq['d_P']
            n_M = face_eq['normal_M']
            d_M = face_eq['d_M']
            
            print(f"[POLY FORM] F{len(unique_faces):2d}:")
            print(f"            Plane P: n=[{n_P[0]:6.3f},{n_P[1]:6.3f},{n_P[2]:6.3f}] d={d_P:7.2f}")
            print(f"            Plane M: n=[{n_M[0]:6.3f},{n_M[1]:6.3f},{n_M[2]:6.3f}] d={d_M:7.2f}")
            print(f"            Average: n=[{n1_avg[0]:6.3f},{n1_avg[1]:6.3f},{n1_avg[2]:6.3f}] d={d1_avg:7.2f}")
    
    print(f"[POLY FORM] {len(unique_faces)} unique faces found")
    
    # Step 4: Find all vertices on each face using iterative tolerance
    print("\n[POLY FORM] Step 4: Finding vertices on each face")
    print("[POLY FORM] ----------------------------------------------------------------------")
    
    # Initialize tolerance
    current_tolerance = 0.1005  # Start with 0.1005mm
    max_iterations = 10  # Prevent infinite loops
    iteration = 0
    all_assigned = False
    
    while not all_assigned and iteration < max_iterations:
        iteration += 1
        
        # Clear previous assignments
        for face_eq in unique_faces:
            face_eq['vertices_on_face'] = []
        
        # Assign vertices to faces using current tolerance
        for face_idx, face_eq in enumerate(unique_faces):
            vertices_on_face = []
            
            for v_idx in range(N):
                vertex = selected_vertices[v_idx]
                # Compute distance to average plane: eps = v·n + d
                eps = np.dot(face_eq['normal'], vertex) + face_eq['d']
                
                # Assign if within tolerance
                if abs(eps) < current_tolerance:
                    vertices_on_face.append(v_idx)
            
            face_eq['vertices_on_face'] = vertices_on_face
        
        # Count how many faces each vertex belongs to
        vertex_face_count = [0] * N
        for face_eq in unique_faces:
            for v_idx in face_eq['vertices_on_face']:
                vertex_face_count[v_idx] += 1
        
        # Find vertices with < 3 faces
        unassigned_vertices = [v_idx for v_idx in range(N)
                               if vertex_face_count[v_idx] < 3]
        
        if len(unassigned_vertices) == 0:
            all_assigned = True
            print(f"[POLY FORM]   All vertices assigned to ≥3 faces with tolerance {current_tolerance:.4f}mm")
        else:
            # Check if we can increase tolerance before announcing it
            next_tolerance = current_tolerance * 2.0
            if next_tolerance > 1.0:
                print(f"[POLY FORM]   {len(unassigned_vertices)} vertices have < 3 faces")
                print(f"[POLY FORM]   WARNING: Cannot increase tolerance beyond 1.0mm limit (would be {next_tolerance:.2f}mm)")
                break
            
            print(f"[POLY FORM]   {len(unassigned_vertices)} vertices have < 3 faces, increasing tolerance to {next_tolerance:.2f}mm")
            if iteration == 1:
                # On first iteration, list the vertices
                print(f"[POLY FORM]   Vertices with < 3 faces:")
                for v_idx in unassigned_vertices:
                    v = selected_vertices[v_idx]
                    count = vertex_face_count[v_idx]
                    print(f"[POLY FORM]     Vertex {v_idx}: [{v[0]:.1f}, {v[1]:.1f}, {v[2]:.1f}] mm - in {count} face(s)")
            # Double the tolerance for next iteration
            current_tolerance = next_tolerance
    
    if not all_assigned:
        print(f"[POLY FORM]   WARNING: Max iterations reached, some vertices still have < 3 faces")
    
    # Display final vertex assignments
    print(f"\n[POLY FORM]   Final assignments (tolerance={current_tolerance/2 if iteration > 1 else current_tolerance:.4f}mm):")
    for face_idx, face_eq in enumerate(unique_faces):
        vertices_display = [v+1 for v in face_eq['vertices_on_face']]
        print(f"[POLY FORM] F{face_idx+1:2d}: {len(face_eq['vertices_on_face']):2d} vertices: {vertices_display}")
    
    
    for face_idx, face_eq in enumerate(unique_faces):
        edges_on_face = []
        vertices_on_face = face_eq['vertices_on_face']
        
        # Check all pairs of vertices on this face
        for i in range(len(vertices_on_face)):
            v_i = vertices_on_face[i]
            for j in range(i+1, len(vertices_on_face)):
                v_j = vertices_on_face[j]
                
                # Check if there's an edge in connectivity matrix
                if merged_conn[v_i, v_j] == 3:
                    edges_on_face.append((v_i, v_j))
        
        face_eq['edges_on_face'] = edges_on_face
        print(f"[POLY FORM]   Face {face_idx+1}: {len(edges_on_face)} edges with conn=3: {edges_on_face}")
    
    # Step 5.5: Process collinear edges - split edges containing intermediate vertices
    print("\n[POLY FORM] Step 5.5: Processing collinear edges")
    print("-" * 70)
    
    def point_on_ray_segment(start, end, point, tolerance=1e-3):
        """
        Check if point lies on the line segment from start to end.
        
        Args:
            start, end, point: 3D numpy arrays
            tolerance: Distance tolerance for collinearity
            
        Returns:
            (is_on_segment, is_inside, parameter_t)
            - is_on_segment: True if point is collinear with segment
            - is_inside: True if point is strictly between start and end
            - parameter_t: Parameter value (0 at start, 1 at end)
        """
        # Vector from start to end
        seg_vec = end - start
        seg_len = np.linalg.norm(seg_vec)
        
        if seg_len < tolerance:
            # Degenerate segment
            return False, False, 0.0
        
        # Vector from start to point
        point_vec = point - start
        
        # Project point onto segment direction
        t = np.dot(point_vec, seg_vec) / (seg_len * seg_len)
        
        # Find closest point on infinite line
        closest = start + t * seg_vec
        
        # Check distance from point to line
        dist = np.linalg.norm(point - closest)
        
        if dist > tolerance:
            # Point not on line
            return False, False, t
        
        # Point is on line - check if inside segment (excluding endpoints)
        is_inside = (t > tolerance / seg_len) and (t < 1.0 - tolerance / seg_len)
        
        return True, is_inside, t
    
    # Process each face
    for face_idx, face_eq in enumerate(unique_faces):
        edges_on_face = face_eq['edges_on_face']
        vertices_on_face = face_eq['vertices_on_face']
        
        if len(edges_on_face) == 0:
            continue
        
        print(f"\n[POLY FORM]   Processing Face {face_idx+1}: {len(edges_on_face)} edges")
        
        # Store edges to process and new edges to add
        edges_to_remove = []
        edges_to_add = []
        
        # Process each edge (outer loop edge)
        for outer_edge in edges_on_face:
            v_start, v_end = outer_edge
            start_pos = selected_vertices[v_start]
            end_pos = selected_vertices[v_end]
            
            # Find all vertices that lie on this edge
            collinear_vertices = []
            
            # Check all other vertices on this face
            for v_idx in vertices_on_face:
                if v_idx == v_start or v_idx == v_end:
                    continue
                
                v_pos = selected_vertices[v_idx]
                is_on_seg, is_inside, t = point_on_ray_segment(
                    start_pos, end_pos, v_pos, tolerance=0.1
                )
                
                if is_on_seg and is_inside:
                    collinear_vertices.append((v_idx, t))
            
            # If we found collinear vertices on this edge, split it
            if len(collinear_vertices) > 0:
                print(f"      Edge ({v_start}, {v_end}): Found {len(collinear_vertices)} "
                      f"collinear vertices: {[v[0] for v in collinear_vertices]}")
                
                # Sort vertices by parameter t (order along the edge)
                collinear_vertices.sort(key=lambda x: x[1])
                
                # Build sequence: start -> collinear_verts -> end
                vertex_sequence = [v_start] + [v[0] for v in collinear_vertices] + [v_end]
                
                print(f"         Vertex sequence: {vertex_sequence}")
                
                # Mark original edge for removal
                edges_to_remove.append(outer_edge)
                
                # Update merged connectivity matrix:
                # Remove the original edge connection
                merged_conn[v_start, v_end] = 0
                merged_conn[v_end, v_start] = 0
                
                # Add new edges between consecutive vertices in sequence
                for i in range(len(vertex_sequence) - 1):
                    v1 = vertex_sequence[i]
                    v2 = vertex_sequence[i + 1]
                    
                    # Add edge to connectivity matrix (mark as conn=3)
                    merged_conn[v1, v2] = 3
                    merged_conn[v2, v1] = 3
                    
                    # Add to list of new edges
                    new_edge = (min(v1, v2), max(v1, v2))
                    edges_to_add.append(new_edge)
                    print(f"         Added edge: ({v1}, {v2})")
        
        # Update edges_on_face for this face
        if len(edges_to_remove) > 0 or len(edges_to_add) > 0:
            print(f"      Updating edges: removing {len(edges_to_remove)}, "
                  f"adding {len(edges_to_add)}")
            
            # Remove old edges
            for edge in edges_to_remove:
                if edge in edges_on_face:
                    edges_on_face.remove(edge)
            
            # Add new edges
            edges_on_face.extend(edges_to_add)
            
            # Update the face data
            face_eq['edges_on_face'] = edges_on_face
            
            print(f"      Face {face_idx+1} now has {len(edges_on_face)} edges")
    
    print("\n[POLY FORM] Step 5.5 Complete: Collinear edge processing finished")
    
    # Step 6: Join edges to make polygons (connected rings)
    print("\n[POLY FORM] Step 6: Forming polygons from edges (REVISED)")
    print("-" * 70)
    
    def find_all_cycles_from_edges(edges, verts_2d):
        """
        Find ALL cycles/polygons from edges using DFS with backtracking.
        
        This improved version explores all possible paths through the
        edge graph, not just the first valid path found. This ensures
        we discover all valid cycles, including longer ones that might
        be missed by greedy exploration.
        
        Args:
            edges: List of edges [(v1, v2), ...]
            verts_2d: Dict mapping vertex_idx -> (x, y) in 2D
        
        Returns:
            List of polygons (each polygon is a list of vertex indices)
        """
        # Build adjacency list
        adjacency = {}
        edge_set = set()
        for v1, v2 in edges:
            edge = (min(v1, v2), max(v1, v2))
            edge_set.add(edge)
            if v1 not in adjacency:
                adjacency[v1] = []
            if v2 not in adjacency:
                adjacency[v2] = []
            adjacency[v1].append(v2)
            adjacency[v2].append(v1)
        
        all_polygons = []
        found_edge_sets = set()  # Use set of frozensets for faster lookup
        
        def dfs_find_cycles(start_vertex, current, path, path_edges):
            """
            DFS with backtracking to find cycles from start_vertex.
            
            Args:
                start_vertex: The vertex where the cycle must close
                current: Current vertex in the path
                path: List of vertices in current path
                path_edges: Set of edges used in current path
            """
            # Try each neighbor of current vertex
            for neighbor in adjacency.get(current, []):
                edge = (min(current, neighbor), max(current, neighbor))
                
                # Skip if edge not in set or already used
                if edge not in edge_set or edge in path_edges:
                    continue
                
                # Check if we can close the cycle
                if neighbor == start_vertex and len(path) >= 3:
                    # Found a valid cycle
                    cycle_edges = path_edges | {edge}
                    cycle_edges_frozen = frozenset(cycle_edges)
                    
                    # Check if this cycle is new (not found before)
                    if cycle_edges_frozen not in found_edge_sets:
                        found_edge_sets.add(cycle_edges_frozen)
                        
                        # Normalize polygon (start from smallest vertex)
                        min_idx = path.index(min(path))
                        normalized = path[min_idx:] + path[:min_idx]
                        all_polygons.append(normalized)
                    continue
                
                # Continue DFS if neighbor not already in path
                if neighbor not in path:
                    new_path = path + [neighbor]
                    new_path_edges = path_edges | {edge}
                    dfs_find_cycles(start_vertex, neighbor,
                                    new_path, new_path_edges)
        
        # Start DFS from each vertex
        vertices = list(adjacency.keys())
        for start_vertex in vertices:
            for first_neighbor in adjacency.get(start_vertex, []):
                edge = (min(start_vertex, first_neighbor),
                        max(start_vertex, first_neighbor))
                if edge in edge_set:
                    # Start exploring from this edge
                    dfs_find_cycles(start_vertex, first_neighbor,
                                    [start_vertex, first_neighbor],
                                    {edge})
        
        return all_polygons
    
    def polygons_are_same(poly1, poly2):
        """
        Check if two polygons are the same (allowing rotation/reversal).
        
        Args:
            poly1, poly2: Lists of vertex indices
            
        Returns:
            True if polygons have same vertices in same order
        """
        if len(poly1) != len(poly2):
            return False
        
        # Normalize both polygons (start from minimum vertex)
        def normalize(poly):
            if not poly:
                return poly
            min_idx = poly.index(min(poly))
            return poly[min_idx:] + poly[:min_idx]
        
        norm1 = normalize(poly1)
        norm2 = normalize(poly2)
        
        # Check forward direction
        if norm1 == norm2:
            return True
        
        # Check reverse direction
        norm2_rev = [norm2[0]] + norm2[1:][::-1]
        return norm1 == norm2_rev
    
    def build_single_polygon_from_edges(edges, start_vertex, verts_2d):
        """
        Build a single polygon starting from start_vertex using available edges.
        
        Args:
            edges: List of available edges [(v1, v2), ...]
            start_vertex: Vertex to start from
            verts_2d: Dict mapping vertex_idx -> (x, y) in 2D
        
        Returns:
            polygon vertices list if cycle found, None otherwise
        """
        # Build adjacency from edges
        adjacency = {}
        for v1, v2 in edges:
            if v1 not in adjacency:
                adjacency[v1] = []
            if v2 not in adjacency:
                adjacency[v2] = []
            adjacency[v1].append(v2)
            adjacency[v2].append(v1)
        
        if start_vertex not in adjacency:
            return None
        
        # Try to build longest cycle starting from start_vertex
        best_polygon = None
        max_vertices = 0
        
        for first_neighbor in adjacency.get(start_vertex, []):
            poly_vertices = [start_vertex, first_neighbor]
            current = first_neighbor
            prev = start_vertex
            max_iter = len(adjacency) + 1
            
            for _ in range(max_iter):
                neighbors = [n for n in adjacency.get(current, [])
                            if n != prev]
                
                if len(neighbors) == 0:
                    break
                
                next_vertex = None
                for n in neighbors:
                    if n == start_vertex and len(poly_vertices) >= 3:
                        # Can close cycle
                        if len(poly_vertices) > max_vertices:
                            best_polygon = poly_vertices[:]
                            max_vertices = len(poly_vertices)
                        break
                    if n not in poly_vertices:
                        next_vertex = n
                        break
                
                if next_vertex is None:
                    break
                
                poly_vertices.append(next_vertex)
                prev = current
                current = next_vertex
        
        return best_polygon
    
    def get_polygon_edges(polygon):
        """Get list of edges from polygon vertices."""
        edges = []
        for i in range(len(polygon)):
            v1 = polygon[i]
            v2 = polygon[(i + 1) % len(polygon)]
            edges.append((min(v1, v2), max(v1, v2)))
        return edges
    
    def merge_polygons_sharing_edge(poly1, poly2, shared_edge, verts_2d):
        """
        Merge two polygons that share an edge by breaking poly1 at the shared
        edge and inserting the non-shared vertices from poly2.
        
        Algorithm:
        1. Find shared edge (v1, v2) in both polygons
        2. Break poly1 at this edge
        3. Insert poly2's vertices (excluding v1, v2) between v1 and v2
        4. Result: poly1 with poly2's interior vertices inserted
        
        Args:
            poly1: Boundary polygon vertex list
            poly2: Polygon to merge into boundary
            shared_edge: Tuple (v1, v2) of shared edge vertices
            verts_2d: Dict mapping vertex_idx -> (x, y) (not used here)
        
        Returns:
            Merged polygon vertex list, or None if merge fails
        """
        v1, v2 = shared_edge
        
        # Find v1 and v2 in poly1
        if v1 not in poly1 or v2 not in poly1:
            return None
        
        # Find v1 and v2 in poly2
        if v1 not in poly2 or v2 not in poly2:
            return None
        
        idx1_v1 = poly1.index(v1)
        idx1_v2 = poly1.index(v2)
        
        idx2_v1 = poly2.index(v1)
        idx2_v2 = poly2.index(v2)
        
        # Check if v1->v2 are consecutive in poly1 (forward or backward)
        # They should be adjacent since shared_edge comes from connectivity
        forward = (idx1_v2 - idx1_v1) % len(poly1) == 1
        backward = (idx1_v1 - idx1_v2) % len(poly1) == 1
        
        if not (forward or backward):
            # Edge not consecutive in poly1 - can't merge cleanly
            return None
        
        # Rotate poly2 to start at v1
        poly2_rotated = poly2[idx2_v1:] + poly2[:idx2_v1]
        
        # Find v2 in rotated poly2
        idx2_v2_rotated = poly2_rotated.index(v2)
        
        # Extract segment from v1 to v2 in poly2
        # We want the vertices BETWEEN v1 and v2 (excluding both endpoints)
        # Since it's a polygon, there are two paths - choose the shorter one
        # (the one that represents the interior of poly2, not going around)
        
        if idx2_v2_rotated == 1:
            # v1 and v2 are adjacent (direct edge) - no interior vertices
            insert_segment = []
        elif idx2_v2_rotated < len(poly2_rotated) // 2:
            # Short path: v1 -> ... -> v2 (forward)
            insert_segment = poly2_rotated[1:idx2_v2_rotated]
        else:
            # Long path: take the other direction (backward from v1)
            # This is v1 -> ... -> v2 going the other way around
            insert_segment = poly2_rotated[idx2_v2_rotated+1:][::-1]
        
        # Now merge into poly1 by breaking at the shared edge
        if forward:
            # v1 -> v2 in poly1, insert segment between them
            # Result: [..., v1, <segment>, v2, ...]
            merged = poly1[:idx1_v1+1] + insert_segment + poly1[idx1_v2:]
        else:
            # v2 -> v1 in poly1, reverse the segment
            # Result: [..., v2, <reversed segment>, v1, ...]
            merged = poly1[:idx1_v2+1] + insert_segment[::-1] + poly1[idx1_v1:]
        
        return merged if len(merged) >= 3 else None
    
    def expand_colinear_edges_in_polygon(polygon, edges, verts_2d,
                                         deleted_verts=None):
        """
        Expand polygon edges that skip colinear intermediate vertices.
        If polygon has edge (A,C) and there's vertex B colinear with
        edges (A,B) and (B,C) existing, replace (A,C) with (A,B,C).
        
        Args:
            deleted_verts: Set of vertices to exclude (artifacts)
        """
        if deleted_verts is None:
            deleted_verts = set()
            
        expanded = []
        
        for i in range(len(polygon)):
            v1 = polygon[i]
            v2 = polygon[(i + 1) % len(polygon)]
            
            expanded.append(v1)
            
            # Check if there are intermediate vertices between v1 and v2
            p1 = np.array(verts_2d[v1])
            p2 = np.array(verts_2d[v2])
            
            intermediate_verts = []
            for v3 in verts_2d:
                if v3 == v1 or v3 == v2 or v3 in polygon:
                    continue
                
                # Skip deleted/artifact vertices
                if v3 in deleted_verts:
                    continue
                    
                p3 = np.array(verts_2d[v3])
                
                # Check if v3 is colinear with v1-v2
                vec12 = p2 - p1
                vec13 = p3 - p1
                
                cross = vec12[0] * vec13[1] - vec12[1] * vec13[0]
                if abs(cross) < 1e-6:  # Colinear
                    # Check if v3 is between v1 and v2
                    dot = np.dot(vec13, vec12)
                    len_sq = np.dot(vec12, vec12)
                    
                    if 0 < dot < len_sq:  # v3 is between v1 and v2
                        intermediate_verts.append((v3, dot))
            
            if intermediate_verts:
                # Sort by distance from v1
                intermediate_verts.sort(key=lambda x: x[1])
                intermediate_v_list = [v for v, _ in intermediate_verts]
                
                # Check if there's a path v1 → intermediate → v2
                full_path = [v1] + intermediate_v_list + [v2]
                path_exists = True
                
                for j in range(len(full_path) - 1):
                    edge_check = (min(full_path[j], full_path[j+1]),
                                 max(full_path[j], full_path[j+1]))
                    if edge_check not in edges:
                        path_exists = False
                        break
                
                if path_exists and intermediate_v_list:
                    # Insert intermediate vertices
                    expanded.extend(intermediate_v_list)
        
        return expanded if len(expanded) >= 3 else polygon
    
    def build_polygons_from_face_edges(edges_on_face, vertices_on_face,
                                       selected_verts, normal,
                                       merged_conn=None):
        """
        [STEP 6] Build boundary and holes from face edges using DFS cycle detection.
        Returns dict with 'faces' list and 'unused_edges'
        """
        print(f"  [STEP 6.0] Building polygons from {len(edges_on_face)} edges")
        
        # [STEP 6.0] Project vertices to 2D plane (verts_2d: dict {v_idx: (x,y)})
        if abs(normal[2]) < 0.9:
            u = np.cross(normal, [0, 0, 1])
        else:
            u = np.cross(normal, [1, 0, 0])
        u = u / np.linalg.norm(u)
        v = np.cross(normal, u)
        
        verts_2d = {}
        for v_idx in vertices_on_face:
            vert_3d = selected_verts[v_idx]
            verts_2d[v_idx] = (np.dot(vert_3d, u), np.dot(vert_3d, v))
        
        # [STEP 6.0] Build edge set (edge_set: set of (v1, v2) tuples)
        edge_set = set()
        for e in edges_on_face:
            v1, v2 = e[0], e[1]
            edge = (min(v1, v2), max(v1, v2))
            edge_set.add(edge)
        
        # [STEP 6.0] Build adjacency list (adjacency: dict {v_idx: [neighbors]})
        adjacency = {}
        for edge in edge_set:
            v1, v2 = edge
            if v1 not in adjacency:
                adjacency[v1] = []
            if v2 not in adjacency:
                adjacency[v2] = []
            adjacency[v1].append(v2)
            adjacency[v2].append(v1)
        
        # [STEP 6.0] Initialize tracking sets
        used_vertices = set()
        used_edges = set()
        faces = []
        
        iteration = 0
        max_iterations = 10
        
        while len(used_vertices) < len(vertices_on_face) and iteration < max_iterations:
            iteration += 1
            print(f"  [STEP 6.0] Iteration {iteration}: {len(used_vertices)}/"
                  f"{len(vertices_on_face)} vertices used")
            
            # [STEP 6.1] Get remaining vertices and edges
            remaining_verts = set(vertices_on_face) - used_vertices
            if not remaining_verts:
                break
            
            remaining_edges = edge_set - used_edges
            remaining_vert_set = set()
            for e in remaining_edges:
                remaining_vert_set.add(e[0])
                remaining_vert_set.add(e[1])
            
            remaining_verts = remaining_verts & remaining_vert_set
            if not remaining_verts:
                print("  [STEP 6.1] No more vertices with edges")
                break
            
            # [STEP 6.1] Find all cycles using DFS (all_possible_polygons: list of vertex lists)
            print(f"  [STEP 6.1] Finding all polygons from "
                  f"{len(remaining_edges)} edges...")
            remaining_edges_list = list(remaining_edges)
            all_possible_polygons = find_all_cycles_from_edges(
                remaining_edges_list, verts_2d)
            
            if not all_possible_polygons:
                print("  [STEP 6.1] No polygons found, attempting closure...")
                
                # [STEP 6.1] Build adjacency to find chain endpoints
                chain_adj = {}
                for edge in remaining_edges:
                    v1, v2 = edge
                    chain_adj.setdefault(v1, []).append(v2)
                    chain_adj.setdefault(v2, []).append(v1)
                
                endpoints = [v for v, neighbors in chain_adj.items()
                             if len(neighbors) == 1]
                
                if len(endpoints) == 2:
                    ep1, ep2 = endpoints
                    print(f"  [STEP 6.1] Chain endpoints: V{ep1}, V{ep2}")
                    
                    # [STEP 6.1] Check connectivity (require conn == 3)
                    edge_conn_value = 0
                    if merged_conn is not None:
                        if (ep1 < len(merged_conn) and
                                ep2 < len(merged_conn)):
                            edge_conn_value = max(
                                merged_conn[ep1, ep2],
                                merged_conn[ep2, ep1])
                            print(f"  [STEP 6.1] merged_conn[{ep1},{ep2}] = "
                                  f"{edge_conn_value}")
                    else:
                        print("  [STEP 6.1] WARNING: merged_conn not available")
                    
                    if merged_conn is None or edge_conn_value == 3:
                        print(f"  [STEP 6.1] Adding closing edge ({ep1}, {ep2})")
                        
                        remaining_edges_list.append((ep1, ep2))
                        all_possible_polygons = (
                            find_all_cycles_from_edges(
                                remaining_edges_list, verts_2d))
                        
                        if not all_possible_polygons:
                            print("  [STEP 6.1] ERROR: Still no polygons")
                            break
                        else:
                            print(f"  [STEP 6.1] Found "
                                  f"{len(all_possible_polygons)} polygon(s)")
                    else:
                        print(f"  [STEP 6.1] Closing edge invalid "
                              f"(conn={edge_conn_value}, need 3)")
                        break
                else:
                    print(f"  [STEP 6.1] ERROR: Expected 2 endpoints, "
                          f"found {len(endpoints)}")
                    break
                
                if not all_possible_polygons:
                    break
            
            # [STEP 6.2] Expand polygons with colinear intermediate vertices
            print(f"  [STEP 6.2] Expanding {len(all_possible_polygons)} "
                  f"polygon(s)")
            expanded_polygons = []
            for i, poly in enumerate(all_possible_polygons):
                expanded = expand_colinear_edges_in_polygon(
                    poly, edge_set, verts_2d, None)
                if len(expanded) != len(poly):
                    print(f"  [STEP 6.2] Poly {i+1}: {len(poly)} → "
                          f"{len(expanded)} verts")
                expanded_polygons.append(expanded)
            
            all_possible_polygons = expanded_polygons
            
            # [STEP 6.2.5] Deduplicate alternate paths
            print(f"  [STEP 6.2.5] Deduplicating "
                  f"{len(all_possible_polygons)} polygon(s)...")
            
            # [STEP 6.2.5] Calculate areas (polys_with_area: list of dicts)
            polys_with_area = []
            for poly in all_possible_polygons:
                poly_2d = [verts_2d[v] for v in poly]
                try:
                    poly_shapely = Polygon(poly_2d)
                    if poly_shapely.is_valid and poly_shapely.area > 1e-10:
                        polys_with_area.append({
                            'vertices': poly,
                            'shapely': poly_shapely,
                            'area': poly_shapely.area
                        })
                except Exception:
                    pass
            
            if not polys_with_area:
                print("  [STEP 6.2.5] ERROR: No valid polygons")
                break
            
            print(f"  [STEP 6.2.5] Valid polygons: {len(polys_with_area)}")
            
            # [STEP 6.2.5] Remove alternate paths (similar area, shared edges)
            # MODIFIED: Keep alternates instead of removing them
            unique_polygons = []
            alternate_polygons = []  # Store alternates instead of discarding
            removed_count = 0
            
            for i, poly1_data in enumerate(polys_with_area):
                poly1 = poly1_data['vertices']
                area1 = poly1_data['area']
                is_alternate = False
                poly1_edges = set(get_polygon_edges(poly1))
                
                for poly2_data in unique_polygons:
                    poly2 = poly2_data['vertices']
                    area2 = poly2_data['area']
                    
                    area_ratio = min(area1, area2) / max(area1, area2)
                    if area_ratio < 0.90:
                        continue
                    
                    poly2_edges = set(get_polygon_edges(poly2))
                    shared_edges = poly1_edges & poly2_edges
                    
                    if not shared_edges:
                        continue
                    
                    poly1_unique = poly1_edges - shared_edges
                    poly2_unique = poly2_edges - shared_edges
                    
                    poly1_remaining = []
                    if poly1_unique:
                        poly1_remaining = find_all_cycles_from_edges(
                            list(poly1_unique), verts_2d)
                    
                    poly2_remaining = []
                    if poly2_unique:
                        poly2_remaining = find_all_cycles_from_edges(
                            list(poly2_unique), verts_2d)
                    
                    # [STEP 6.2.5] Mark as alternate if both are alternates of each other
                    if not poly1_remaining and not poly2_remaining:
                        if len(poly1) <= len(poly2):
                            is_alternate = True
                            removed_count += 1
                            break
                        else:
                            # Move poly2 to alternates and remove from unique
                            unique_polygons.remove(poly2_data)
                            alternate_polygons.append(poly2_data)
                            removed_count += 1
                            break
                
                if is_alternate:
                    alternate_polygons.append(poly1_data)
                else:
                    unique_polygons.append(poly1_data)
            
            print(f"  [STEP 6.2.5] After deduplication: "
                  f"{len(unique_polygons)} unique, "
                  f"{len(alternate_polygons)} alternates kept")
            
            # Combine unique and alternates for further processing
            # Unique polygons first, then alternates
            polys_with_area = unique_polygons + alternate_polygons
            
            # [STEP 6.3] Sort by vertex count (most vertices first)
            polys_with_area.sort(
                key=lambda p: (len(p['vertices']), p['area']),
                reverse=True
            )
            
            # [STEP 6.3] Select boundary: polygon with most vertices/edges
            boundary_poly = polys_with_area[0]['vertices']
            print(f"  [STEP 6.3] Selected boundary: {len(boundary_poly)} verts, "
                  f"area={polys_with_area[0]['area']:.6f}")
            print(f"  [STEP 6.3]   boundary_poly (vertex list): {boundary_poly}")
            
            # [STEP 6.3] Mark boundary edges as used
            boundary_edges = set(get_polygon_edges(boundary_poly))
            used_edges.update(boundary_edges & edge_set)
            
            # [STEP 6.3] Collect remaining polygons for processing
            all_other_polygons = [p['vertices'] for p in polys_with_area[1:]]
            
            print(f"  [STEP 6.3] Remaining polygons: {len(all_other_polygons)}")
            
            # [STEP 6.4] Merge polygons that share edges with boundary
            alternates_candidates = []  # Will store refined alternates during merge
            
            boundary_2d = [verts_2d[v] for v in boundary_poly]
            boundary_shapely = Polygon(boundary_2d)
            
            boundary_edges = set(get_polygon_edges(boundary_poly))
            
            # [STEP 6.4] Set max iterations based on number of deduplicated polygons
            max_merge_iterations = 2 * len(unique_polygons)
            merged_any = True
            merge_count = 0
            
            while merged_any and merge_count < max_merge_iterations:
                merged_any = False
                merge_count += 1
                print(f"  [STEP 6.4] Merge iteration {merge_count}, "
                      f"{len(all_other_polygons)} polygons remaining")
                boundary_edges = set(get_polygon_edges(boundary_poly))
                
                for poly in all_other_polygons[:]:  # poly: list of vertex indices
                    poly_edges = set(get_polygon_edges(poly))
                    shared_edges = boundary_edges & poly_edges
                    
                    if not shared_edges:
                        continue
                    
                    print(f"  [STEP 6.4] Checking polygon: {len(poly)} verts, "
                          f"{len(shared_edges)} shared edges")
                    print(f"  [STEP 6.4]   Polygon vertices: {poly}")
                    
                    poly_unique_edges = poly_edges - shared_edges
                    
                    if not poly_unique_edges:
                        print(f"  [STEP 6.4]   Fully overlaps boundary")
                        used_edges.update(shared_edges)
                        all_other_polygons.remove(poly)
                        merged_any = True
                        break
                    
                    # [STEP 6.4] Check spatial relationship of unique vertices
                        unique_verts = set()
                        for e in poly_unique_edges:
                            unique_verts.add(e[0])
                            unique_verts.add(e[1])
                        
                        inside_count = 0
                        outside_count = 0
                        boundary_count = 0
                        
                        for v in unique_verts:
                            if v in boundary_poly:
                                boundary_count += 1
                                continue
                            point = Point(verts_2d[v])
                            try:
                                if boundary_shapely.contains(point):
                                    inside_count += 1
                                else:
                                    outside_count += 1
                            except:
                                outside_count += 1
                        
                        print(f"  [STEP 6.4]   Unique verts: inside={inside_count}, "
                              f"outside={outside_count}, on_boundary={boundary_count}")
                        
                        # [STEP 6.4 Case 3] Geometric merge using Shapely operations
                        print(f"  [STEP 6.4] CASE 3: Geometric merge")
                        
                        # Convert to Shapely polygons for geometric operations
                        boundary_shapely_poly = Polygon([verts_2d[v] for v in boundary_poly])
                        poly_shapely_poly = Polygon([verts_2d[v] for v in poly])
                        
                        # Find intersection and differences
                        try:
                            intersection = boundary_shapely_poly.intersection(poly_shapely_poly)
                            diff_boundary = boundary_shapely_poly.difference(poly_shapely_poly)
                            diff_poly = poly_shapely_poly.difference(boundary_shapely_poly)
                            
                            print(f"  [STEP 6.4]   Intersection area: {intersection.area:.2f}")
                            print(f"  [STEP 6.4]   Boundary - Poly area: {diff_boundary.area:.2f}")
                            print(f"  [STEP 6.4]   Poly - Boundary area: {diff_poly.area:.2f}")
                            
                            # Collect all non-empty geometric regions
                            result_regions = []
                            
                            # Add intersection if significant
                            if intersection.area > 1e-6:
                                if intersection.geom_type == 'Polygon':
                                    result_regions.append(('intersection', intersection))
                                elif intersection.geom_type == 'MultiPolygon':
                                    for geom in intersection.geoms:
                                        result_regions.append(('intersection', geom))
                            
                            # Add boundary difference if significant
                            if diff_boundary.area > 1e-6:
                                if diff_boundary.geom_type == 'Polygon':
                                    result_regions.append(('boundary_diff', diff_boundary))
                                elif diff_boundary.geom_type == 'MultiPolygon':
                                    for geom in diff_boundary.geoms:
                                        result_regions.append(('boundary_diff', geom))
                            
                            # Add poly difference if significant
                            if diff_poly.area > 1e-6:
                                if diff_poly.geom_type == 'Polygon':
                                    result_regions.append(('poly_diff', diff_poly))
                                elif diff_poly.geom_type == 'MultiPolygon':
                                    for geom in diff_poly.geoms:
                                        result_regions.append(('poly_diff', geom))
                            
                            print(f"  [STEP 6.4]   Found {len(result_regions)} geometric region(s)")
                            
                            if result_regions:
                                # Convert Shapely polygons back to vertex lists
                                converted_polys = []
                                for region_name, shapely_geom in result_regions:
                                    coords = list(shapely_geom.exterior.coords[:-1])
                                    
                                    # Match to our vertices
                                    matched_verts = []
                                    for coord in coords:
                                        min_dist = float('inf')
                                        best_v = None
                                        for v in vertices_on_face:
                                            v_2d = verts_2d[v]
                                            dist = ((v_2d[0] - coord[0])**2 + 
                                                   (v_2d[1] - coord[1])**2)**0.5
                                            if dist < min_dist:
                                                min_dist = dist
                                                best_v = v
                                        if best_v is not None and min_dist < 0.01:
                                            matched_verts.append(best_v)
                                    
                                    if len(matched_verts) >= 3:
                                        converted_polys.append((region_name, matched_verts, 
                                                              shapely_geom.area))
                                        print(f"  [STEP 6.4]   {region_name}: "
                                              f"{len(matched_verts)} verts, "
                                              f"area={shapely_geom.area:.2f}")
                                
                                if converted_polys:
                                    # Sort by area, largest first
                                    converted_polys.sort(key=lambda x: x[2], reverse=True)
                                    
                                    # Use largest as new boundary
                                    (new_boundary_name, new_boundary_verts, 
                                     new_boundary_area) = converted_polys[0]
                                    
                                    print(f"  [STEP 6.4]   Using {new_boundary_name} "
                                          f"as new boundary")
                                    
                                    boundary_poly = new_boundary_verts
                                    boundary_2d = [verts_2d[v] for v in boundary_poly]
                                    boundary_shapely = Polygon(boundary_2d)
                                    
                                    # Add other regions back to list
                                    for region_name, region_verts, region_area in converted_polys[1:]:
                                        is_duplicate = False
                                        for existing_poly in all_other_polygons:
                                            if polygons_are_same(region_verts, existing_poly):
                                                is_duplicate = True
                                                print(f"  [STEP 6.4]   {region_name} is duplicate")
                                                break
                                        
                                        if is_duplicate:
                                            region_edges = set(get_polygon_edges(region_verts))
                                            if boundary_edges & region_edges:
                                                already_in_alternates = False
                                                for alt in alternates_candidates:
                                                    if polygons_are_same(region_verts, alt):
                                                        already_in_alternates = True
                                                        break
                                                if not already_in_alternates:
                                                    alternates_candidates.append(region_verts)
                                                    print(f"  [STEP 6.4]   Added {region_name} "
                                                          f"to alternates")
                                        else:
                                            all_other_polygons.append(region_verts)
                                            print(f"  [STEP 6.4]   Added {region_name}: "
                                                  f"{len(region_verts)} verts")
                                    
                                    # Original poly is removed; refined regions replace it
                                    used_edges.update(shared_edges)
                                    all_other_polygons.remove(poly)
                                    merged_any = True
                                    break
                        
                        except Exception as e:
                            print(f"  [STEP 6.4]   Geometric operation failed: {e}")
                            # Mark edges as used and remove polygon
                            used_edges.update(shared_edges)
                            all_other_polygons.remove(poly)
                            merged_any = True
                            break
            
            print(f"  [STEP 6.4] Final boundary: {len(boundary_poly)} vertices")
            
            # [STEP 6.4.5] Check for geometric interference among all refined polygons
            print(f"  [STEP 6.4.5] Checking geometric interference among refined polygons")
            
            # Collect all polygons: boundary + alternates + remaining
            all_refined_polys = [boundary_poly]
            all_refined_polys.extend(alternates_candidates)
            all_refined_polys.extend(all_other_polygons)
            
            print(f"  [STEP 6.4.5] Total refined polygons to check: {len(all_refined_polys)}")
            
            # Sequential interference resolution: process each polygon against all others
            # This ensures each polygon is fully separated before moving to the next
            print(f"  [STEP 6.4.5] Processing polygons sequentially to eliminate all overlaps")
            
            # Convert all to Shapely polygons
            shapely_polys = []
            for poly_verts in all_refined_polys:
                poly_2d = [verts_2d[v] for v in poly_verts]
                try:
                    poly_shapely = Polygon(poly_2d)
                    if poly_shapely.is_valid and poly_shapely.area > 1e-6:
                        shapely_polys.append({
                            'vertices': poly_verts,
                            'shapely': poly_shapely,
                            'area': poly_shapely.area
                        })
                except:
                    pass
            
            print(f"  [STEP 6.4.5] Starting with {len(shapely_polys)} valid polygons")
            
            # Process each polygon sequentially
            i = 0
            while i < len(shapely_polys):
                poly_i_data = shapely_polys[i]
                
                # Find all overlaps with subsequent polygons
                j = i + 1
                has_overlap = False
                
                while j < len(shapely_polys):
                    poly_j_data = shapely_polys[j]
                    
                    try:
                        intersection = poly_i_data['shapely'].intersection(poly_j_data['shapely'])
                        
                        if intersection.area > 1e-6:
                            has_overlap = True
                            print(f"  [STEP 6.4.5] Poly {i} ({len(poly_i_data['vertices'])} verts) "
                                  f"overlaps with Poly {j} ({len(poly_j_data['vertices'])} verts), "
                                  f"overlap area={intersection.area:.2f}")
                            
                            # Separate into non-overlapping regions
                            diff_i = poly_i_data['shapely'].difference(poly_j_data['shapely'])
                            diff_j = poly_j_data['shapely'].difference(poly_i_data['shapely'])
                            
                            # Collect all resulting non-overlapping regions
                            result_regions = []
                            
                            # Add intersection if significant
                            if intersection.area > 1e-6:
                                if intersection.geom_type == 'Polygon':
                                    result_regions.append(('intersection', intersection))
                                elif intersection.geom_type == 'MultiPolygon':
                                    for geom in intersection.geoms:
                                        result_regions.append(('intersection', geom))
                            
                            # Add difference from poly i
                            if diff_i.area > 1e-6:
                                if diff_i.geom_type == 'Polygon':
                                    result_regions.append(('diff_i', diff_i))
                                elif diff_i.geom_type == 'MultiPolygon':
                                    for geom in diff_i.geoms:
                                        result_regions.append(('diff_i', geom))
                            
                            # Add difference from poly j
                            if diff_j.area > 1e-6:
                                if diff_j.geom_type == 'Polygon':
                                    result_regions.append(('diff_j', diff_j))
                                elif diff_j.geom_type == 'MultiPolygon':
                                    for geom in diff_j.geoms:
                                        result_regions.append(('diff_j', geom))
                            
                            print(f"  [STEP 6.4.5]   Separated into {len(result_regions)} non-overlapping region(s)")
                            
                            # Convert back to vertex representation using existing vertices only
                            new_polys = []
                            for region_name, shapely_geom in result_regions:
                                coords = list(shapely_geom.exterior.coords[:-1])
                                
                                # Match to existing vertices only
                                matched_verts = []
                                for coord in coords:
                                    min_dist = float('inf')
                                    best_v = None
                                    for v in vertices_on_face:
                                        v_2d = verts_2d[v]
                                        dist = ((v_2d[0] - coord[0])**2 + 
                                               (v_2d[1] - coord[1])**2)**0.5
                                        if dist < min_dist:
                                            min_dist = dist
                                            best_v = v
                                    if best_v is not None and min_dist < 0.01:
                                        matched_verts.append(best_v)
                                
                                if len(matched_verts) >= 3:
                                    new_polys.append({
                                        'vertices': matched_verts,
                                        'shapely': shapely_geom,
                                        'area': shapely_geom.area
                                    })
                                    print(f"  [STEP 6.4.5]   {region_name}: "
                                          f"{len(matched_verts)} verts, area={shapely_geom.area:.2f}")
                            
                            # Remove poly i and poly j, add new regions
                            shapely_polys.pop(j)  # Remove j first (higher index)
                            shapely_polys.pop(i)  # Then remove i
                            shapely_polys.extend(new_polys)
                            
                            print(f"  [STEP 6.4.5]   Updated polygon list: now {len(shapely_polys)} polygons")
                            
                            # Restart from polygon i (which now contains new polygons)
                            break
                        else:
                            # No overlap, check next polygon
                            j += 1
                    
                    except Exception as e:
                        print(f"  [STEP 6.4.5]   Failed to check/separate: {e}")
                        j += 1
                
                # If no overlap found with any subsequent polygon, move to next polygon
                if not has_overlap:
                    print(f"  [STEP 6.4.5] Poly {i} has no overlaps with remaining polygons, moving to next")
                    i += 1
            
            print(f"  [STEP 6.4.5] Sequential processing complete: {len(shapely_polys)} non-overlapping polygons")
            
            # Verify no overlaps remain
            overlaps_found = 0
            for i in range(len(shapely_polys)):
                for j in range(i + 1, len(shapely_polys)):
                    try:
                        intersection = shapely_polys[i]['shapely'].intersection(shapely_polys[j]['shapely'])
                        if intersection.area > 1e-6:
                            overlaps_found += 1
                            print(f"  [STEP 6.4.5] WARNING: Overlap still exists between poly {i} and {j}, area={intersection.area:.2f}")
                    except:
                        pass
            
            if overlaps_found == 0:
                print(f"  [STEP 6.4.5] Verified: No overlaps remain among {len(shapely_polys)} polygons")
            else:
                print(f"  [STEP 6.4.5] WARNING: {overlaps_found} overlaps still present after processing")
            
            # Extract vertex lists from shapely_polys
            all_refined_polys = [p['vertices'] for p in shapely_polys]
            
            # After all separation iterations, deduplicate final polygons
            print(f"  [STEP 6.4.5] Final deduplication of {len(all_refined_polys)} polygons...")
            
            # Convert to shapely for deduplication
            shapely_polys = []
            for poly_verts in all_refined_polys:
                poly_2d = [verts_2d[v] for v in poly_verts]
                try:
                    poly_shapely = Polygon(poly_2d)
                    if poly_shapely.is_valid and poly_shapely.area > 1e-6:
                        shapely_polys.append({
                            'vertices': poly_verts,
                            'shapely': poly_shapely,
                            'area': poly_shapely.area
                        })
                except:
                    pass
            
            deduplicated_polys = []
            for poly_data in shapely_polys:
                is_duplicate = False
                for existing in deduplicated_polys:
                    if polygons_are_same(poly_data['vertices'], existing['vertices']):
                        is_duplicate = True
                        print(f"  [STEP 6.4.5]   Removed duplicate: {len(poly_data['vertices'])} verts, "
                              f"area={poly_data['area']:.2f}")
                        break
                if not is_duplicate:
                    deduplicated_polys.append(poly_data)
            
            print(f"  [STEP 6.4.5] After deduplication: {len(deduplicated_polys)} unique polygons")
            
            # Update boundary, alternates, and all_other_polygons
            if len(deduplicated_polys) > 0:
                # Sort by area, largest first
                deduplicated_polys.sort(key=lambda x: x['area'], reverse=True)
                
                # Largest becomes boundary
                boundary_poly = deduplicated_polys[0]['vertices']
                alternates_candidates = [p['vertices'] for p in deduplicated_polys[1:]]
                all_other_polygons = []
                
                print(f"  [STEP 6.4.5] Final result: "
                      f"boundary={len(boundary_poly)} verts, "
                      f"{len(alternates_candidates)} alternates")
            
            # [STEP 6.5] Compare remaining polygons against each other
            print(f"  [STEP 6.5] Checking {len(all_other_polygons)} "
                  f"remaining polygon(s) against each other")
            
            if len(all_other_polygons) > 1:
                # [STEP 6.5] Set max iterations based on deduplicated polygons
                max_compare_iterations = 2 * len(unique_polygons)
                compared_any = True
                compare_iteration = 0
                
                while compared_any and compare_iteration < max_compare_iterations:
                    compared_any = False
                    compare_iteration += 1
                    
                    for i, poly1 in enumerate(all_other_polygons[:]):
                        if poly1 not in all_other_polygons:
                            continue
                            
                        poly1_edges = set(get_polygon_edges(poly1))
                        
                        for j, poly2 in enumerate(all_other_polygons[:]):
                            if i >= j or poly2 not in all_other_polygons:
                                continue
                            
                            poly2_edges = set(get_polygon_edges(poly2))
                            shared_edges = poly1_edges & poly2_edges
                            
                            if not shared_edges:
                                continue
                            
                            print(f"  [STEP 6.5] Comparing: {len(poly1)}-vert vs "
                                  f"{len(poly2)}-vert, {len(shared_edges)} shared")
                            
                            poly1_unique_edges = poly1_edges - shared_edges
                            poly2_unique_edges = poly2_edges - shared_edges
                            
                            poly1_remaining = []
                            if poly1_unique_edges:
                                poly1_remaining = find_all_cycles_from_edges(
                                    list(poly1_unique_edges), verts_2d)
                            
                            poly2_remaining = []
                            if poly2_unique_edges:
                                poly2_remaining = find_all_cycles_from_edges(
                                    list(poly2_unique_edges), verts_2d)
                            
                            print(f"  [STEP 6.5]   Poly1: {len(poly1_unique_edges)} "
                                  f"edges → {len(poly1_remaining)} polys")
                            print(f"  [STEP 6.5]   Poly2: {len(poly2_unique_edges)} "
                                  f"edges → {len(poly2_remaining)} polys")
                            
                            # [STEP 6.5] Both form polygons → separate touching faces
                            if poly1_remaining and poly2_remaining:
                                print("  [STEP 6.5]   SEPARATE: Both valid, keeping both")
                                used_edges.update(shared_edges)
                                compared_any = True
                            
                            # [STEP 6.5] Only poly1 forms polygon → poly2 is subset
                            elif poly1_remaining and not poly2_remaining:
                                print("  [STEP 6.5]   Poly2 subset, removing")
                                used_edges.update(shared_edges)
                                all_other_polygons.remove(poly2)
                                compared_any = True
                                break
                            
                            # [STEP 6.5] Only poly2 forms polygon → poly1 is subset
                            elif not poly1_remaining and poly2_remaining:
                                print("  [STEP 6.5]   Poly1 subset, removing")
                                used_edges.update(shared_edges)
                                all_other_polygons.remove(poly1)
                                compared_any = True
                                break
                            
                            # [STEP 6.5] Neither forms polygon → alternates
                            else:
                                if len(poly1) >= len(poly2):
                                    to_keep = poly1
                                    to_remove = poly2
                                    keeper_name = "Poly1"
                                else:
                                    to_keep = poly2
                                    to_remove = poly1
                                    keeper_name = "Poly2"
                                
                                print(f"  [STEP 6.5]   ALTERNATE: Keeping {keeper_name} "
                                      f"({len(to_keep)} verts)")
                                
                                to_keep_unique = (poly1_unique_edges 
                                                  if to_keep == poly1 
                                                  else poly2_unique_edges)
                                to_remove_unique = (poly2_unique_edges 
                                                    if to_remove == poly2 
                                                    else poly1_unique_edges)
                                
                                used_edges.update(shared_edges)
                                used_edges.update(to_remove_unique)
                                if to_remove in all_other_polygons:
                                    all_other_polygons.remove(to_remove)
                                compared_any = True
                                break
                        
                        if compared_any:
                            break
            
            print(f"  [STEP 6.5] After comparison: {len(all_other_polygons)} remain")
            
            # [STEP 6.6] Classify remaining polygons relative to boundary
            inside_polygons = []
            outside_polygons = []
            polygons_to_remove = []
            
            for poly in all_other_polygons:
                poly_2d = [verts_2d[v] for v in poly]
                try:
                    poly_shapely = Polygon(poly_2d)
                    if not poly_shapely.is_valid:
                        polygons_to_remove.append(poly)
                        continue
                    
                    # [STEP 6.6] Check spatial relationship with boundary
                    if boundary_shapely.contains(poly_shapely):
                        inside_polygons.append(poly)
                        print(f"  [STEP 6.6] Polygon {len(poly)} verts: INSIDE (hole)")
                        
                    elif boundary_shapely.intersects(poly_shapely):
                        centroid = poly_shapely.centroid
                        
                        if boundary_shapely.contains(centroid):
                            print(f"  [STEP 6.6] Polygon {len(poly)} verts: TOUCHING")
                            
                            try:
                                modified_boundary_shapely = (
                                    boundary_shapely.difference(poly_shapely))
                                
                                boundary_verts_set = set(boundary_poly)
                                poly_verts_set = set(poly)
                                modified_verts_count = 0
                                
                                if (modified_boundary_shapely.is_valid and 
                                    not modified_boundary_shapely.is_empty):
                                    if (modified_boundary_shapely.geom_type == 
                                        'Polygon'):
                                        mod_coords = list(
                                            modified_boundary_shapely.exterior.coords[:-1])
                                        modified_verts_set = set()
                                        for coord in mod_coords:
                                            for v_idx, v_coord in enumerate(verts_2d):
                                                if (abs(v_coord[0] - coord[0]) < 1e-6 and 
                                                    abs(v_coord[1] - coord[1]) < 1e-6):
                                                    modified_verts_set.add(v_idx)
                                                    break
                                        modified_verts_count = len(
                                            modified_verts_set & set(vertices_on_face))
                                
                                orig_boundary_count = len(
                                    boundary_verts_set & set(vertices_on_face))
                                touching_count = len(
                                    poly_verts_set & set(vertices_on_face))
                                
                                max_count = max(orig_boundary_count, 
                                                modified_verts_count, touching_count)
                                
                                if (max_count == modified_verts_count and 
                                    modified_verts_count > 0):
                                    print("  [STEP 6.6]   Using modified boundary")
                                    boundary_poly = list(modified_verts_set)
                                    boundary_shapely = modified_boundary_shapely
                                    polygons_to_remove.append(poly)
                                elif max_count == touching_count:
                                    print("  [STEP 6.6]   Using touching as boundary")
                                    boundary_poly = poly
                                    boundary_shapely = poly_shapely
                                elif max_count == orig_boundary_count:
                                    print("  [STEP 6.6]   Keeping original boundary")
                                    polygons_to_remove.append(poly)
                            except Exception:
                                print("  [STEP 6.6]   Subtraction failed")
                                polygons_to_remove.append(poly)
                        else:
                            print(f"  [STEP 6.6] Polygon {len(poly)} verts: "
                                  f"TOUCHING (outside)")
                            polygons_to_remove.append(poly)
                    else:
                        outside_polygons.append(poly)
                        print(f"  [STEP 6.6] Polygon {len(poly)} verts: OUTSIDE")
                        
                except Exception:
                    polygons_to_remove.append(poly)
            
            print(f"  [STEP 6.6] Classified: {len(inside_polygons)} holes, "
                  f"{len(outside_polygons)} outside, "
                  f"{len(polygons_to_remove)} removed")
            
            # [STEP 6.7] Classify remaining vertices
            boundary_verts = set(boundary_poly)
            used_vertices.update(boundary_verts)
            
            for poly in inside_polygons:
                used_vertices.update(poly)
            
            remaining_verts = set(vertices_on_face) - used_vertices
            inside_verts = []
            outside_verts = []
            
            for v in remaining_verts:
                point = Point(verts_2d[v])
                try:
                    if (boundary_shapely.contains(point) or 
                        boundary_shapely.touches(point)):
                        inside_verts.append(v)
                    else:
                        outside_verts.append(v)
                except Exception:
                    outside_verts.append(v)
            
            # [STEP 6.7] Build holes from inside polygons and vertices
            holes = []
            
            for poly in inside_polygons:
                holes.append(poly)
                poly_edges = set(get_polygon_edges(poly))
                used_edges.update(poly_edges & edge_set)
                print(f"  [STEP 6.7] Hole from polygon: {len(poly)} verts")
            
            inside_set = set(inside_verts)
            
            while inside_set:
                inside_edges = []
                for v1 in inside_set:
                    for v2 in adjacency.get(v1, []):
                        if v2 in inside_set:
                            edge = (min(v1, v2), max(v1, v2))
                            if edge in edge_set and edge not in used_edges:
                                inside_edges.append(edge)
                
                if not inside_edges:
                    break
                
                hole_start = list(inside_set)[0]
                hole = build_single_polygon_from_edges(
                    inside_edges, hole_start, verts_2d)
                
                if hole and len(hole) >= 3:
                    holes.append(hole)
                    hole_edges = set(get_polygon_edges(hole))
                    used_edges.update(hole_edges & edge_set)
                    used_vertices.update(hole)
                    inside_set -= set(hole)
                    print(f"  [STEP 6.7] Hole from vertices: {len(hole)} verts")
                else:
                    break
            
            # [STEP 6.8] Choose best boundary from alternates
            final_boundary = boundary_poly
            final_alternates = []
            
            if alternates_candidates:
                def count_shared_edges_between(poly1_verts, poly2_verts):
                    edges1 = set()
                    for i in range(len(poly1_verts)):
                        v1, v2 = (poly1_verts[i], 
                                  poly1_verts[(i+1) % len(poly1_verts)])
                        edges1.add((min(v1, v2), max(v1, v2)))
                    edges2 = set()
                    for i in range(len(poly2_verts)):
                        v1, v2 = (poly2_verts[i], 
                                  poly2_verts[(i+1) % len(poly2_verts)])
                        edges2.add((min(v1, v2), max(v1, v2)))
                    return len(edges1 & edges2)
                
                all_candidates = [
                    {'vertices': boundary_poly, 'name': 'merged_boundary'}
                ]
                for idx, alt_verts in enumerate(alternates_candidates):
                    all_candidates.append({
                        'vertices': alt_verts,
                        'name': f'alternate_{idx+1}'
                    })
                
                max_shared = -1
                best_candidate = None
                
                for cand in all_candidates:
                    shared_count = 0
                    cand_verts = cand['vertices']
                    
                    for other in all_candidates:
                        if other['name'] == cand['name']:
                            continue
                        other_verts = other['vertices']
                        shared_count += count_shared_edges_between(
                            cand_verts, other_verts)
                    
                    if shared_count > max_shared:
                        max_shared = shared_count
                        best_candidate = cand
                
                if (best_candidate and 
                    best_candidate['name'] != 'merged_boundary'):
                    print(f"  [STEP 6.8] Selecting {best_candidate['name']} "
                          f"as final (shares {max_shared} edges)")
                    final_boundary = best_candidate['vertices']
                    final_alternates = [boundary_poly]
                    for alt in alternates_candidates:
                        if alt != final_boundary:
                            final_alternates.append(alt)
                else:
                    print(f"  [STEP 6.8] Keeping merged boundary "
                          f"(shares {max_shared} edges)")
                    final_alternates = alternates_candidates
            
            # [STEP 6.8] Store face with boundary and holes
            face_data = {
                'boundary': final_boundary,
                'holes': holes
            }
            
            if final_alternates:
                face_data['alternates'] = final_alternates
                print(f"  [STEP 6.8] Stored {len(final_alternates)} alternates")
            
            faces.append(face_data)
            
            # [STEP 6.8] Outside polygons processed in next iteration
            if outside_polygons:
                for poly in outside_polygons:
                    outside_verts.extend(poly)
                outside_verts = list(set(outside_verts))
                print(f"  [STEP 6.8] {len(outside_polygons)} outside polygons, "
                      f"{len(outside_verts)} outside verts → next iteration")
            elif outside_verts:
                print(f"  [STEP 6.8] {len(outside_verts)} outside verts "
                      f"→ next iteration")
        
        unused_edges = edge_set - used_edges
        
        print(f"  [STEP 6] Created {len(faces)} face(s)")
        if unused_edges:
            print(f"  [STEP 6] WARNING: {len(unused_edges)} unused edges")
        
        return {
            'faces': faces,
            'unused_edges': unused_edges,
            'verts_2d': verts_2d,
            'projection': (u, v)
        }
    
    # Process each face and build edge-face associations
    edge_face_map = {}  # edge -> list of (face_idx, polygon_idx) tuples
    all_face_polygons = []  # List of all polygons with metadata
    
    for face_idx, face_eq in enumerate(unique_faces):
        edges = face_eq['edges_on_face']
        
        if len(edges) == 0:
            print(f"[POLY FORM]   Face {face_idx+1}: No edges, skipping")
            continue
        
        print(f"\n[POLY FORM]   Face {face_idx+1}: "
              f"Total edges available: {len(edges)}")
        
        # Build polygons using revised algorithm
        result = build_polygons_from_face_edges(
            edges, face_eq['vertices_on_face'],
            selected_vertices, face_eq['normal'], merged_conn)
        
        # Store result for Step 7
        face_eq['face_results'] = result
        
        # Format result as list of polygon dictionaries
        polygons = []
        polygon_idx_in_face = 0
        
        # Handle multiple faces from grouping
        for face_data in result['faces']:
            boundary_verts = face_data['boundary']
            boundary_2d = [result['verts_2d'][v] for v in boundary_verts]
            try:
                boundary_shapely = Polygon(boundary_2d)
                poly_data = {
                    'vertices': boundary_verts,
                    'shapely_2d': boundary_shapely,
                    'projection': result['projection'],
                    'area': boundary_shapely.area
                }
                polygons.append(poly_data)
                
                # Track this polygon and register its edges
                all_face_polygons.append({
                    'face_idx': face_idx,
                    'polygon_idx': polygon_idx_in_face,
                    'data': poly_data,
                    'face_eq': face_eq
                })
                
                # Register edges in edge_face_map
                for i in range(len(boundary_verts)):
                    v1 = boundary_verts[i]
                    v2 = boundary_verts[(i + 1) % len(boundary_verts)]
                    edge = (min(v1, v2), max(v1, v2))
                    if edge not in edge_face_map:
                        edge_face_map[edge] = []
                    edge_face_map[edge].append((face_idx, polygon_idx_in_face))
                
                polygon_idx_in_face += 1
            except:
                pass
            
            # Store alternates if they exist
            # Treat non-hole alternates as separate face regions
            if 'alternates' in face_data and face_data['alternates']:
                for alt_verts in face_data['alternates']:
                    alt_2d = [result['verts_2d'][v] for v in alt_verts]
                    try:
                        alt_shapely = Polygon(alt_2d)
                        alt_data = {
                            'vertices': alt_verts,
                            'shapely_2d': alt_shapely,
                            'projection': result['projection'],
                            'area': alt_shapely.area,
                            'is_alternate': True
                        }
                        polygons.append(alt_data)
                        
                        # Track this alternate polygon as separate face region
                        all_face_polygons.append({
                            'face_idx': face_idx,
                            'polygon_idx': polygon_idx_in_face,
                            'data': alt_data,
                            'face_eq': face_eq,
                            'is_alternate': True
                        })
                        
                        # Register alternate edges in edge_face_map
                        for i in range(len(alt_verts)):
                            v1 = alt_verts[i]
                            v2 = alt_verts[(i + 1) % len(alt_verts)]
                            edge = (min(v1, v2), max(v1, v2))
                            if edge not in edge_face_map:
                                edge_face_map[edge] = []
                            edge_face_map[edge].append((face_idx, polygon_idx_in_face))
                        
                        polygon_idx_in_face += 1
                    except:
                        pass
            
            for hole in face_data['holes']:
                hole_2d = [result['verts_2d'][v] for v in hole]
                try:
                    hole_shapely = Polygon(hole_2d)
                    poly_data = {
                        'vertices': hole,
                        'shapely_2d': hole_shapely,
                        'projection': result['projection'],
                        'area': hole_shapely.area
                    }
                    polygons.append(poly_data)
                    
                    # Track this hole polygon
                    all_face_polygons.append({
                        'face_idx': face_idx,
                        'polygon_idx': polygon_idx_in_face,
                        'data': poly_data,
                        'face_eq': face_eq,
                        'is_hole': True
                    })
                    
                    # Register hole edges
                    for i in range(len(hole)):
                        v1 = hole[i]
                        v2 = hole[(i + 1) % len(hole)]
                        edge = (min(v1, v2), max(v1, v2))
                        if edge not in edge_face_map:
                            edge_face_map[edge] = []
                        edge_face_map[edge].append((face_idx, polygon_idx_in_face))
                    
                    polygon_idx_in_face += 1
                except:
                    pass
        
        face_eq['polygons'] = polygons
        print(f"[POLY FORM]   Face {face_idx+1}: "
              f"Found {len(polygons)} polygon(s)")
        
        if result['unused_edges']:
            print(f"[POLY FORM]   WARNING: {len(result['unused_edges'])} "
                  f"unused edges remain!")
        
        for poly_idx, poly_data in enumerate(polygons):
            poly_verts = poly_data['vertices']
            poly_area = poly_data['area']
            print(f"[POLY FORM]     Polygon {poly_idx+1}: "
                  f"{len(poly_verts)} vertices, area={poly_area:.6f}")
            print(f"[POLY FORM]       Vertices: {poly_verts}")
    
    # Step 6.4: Validate edge-face topology and filter invalid faces
    print(f"\n[POLY FORM] Step 6.4: Validating edge-face topology")
    print(f"[POLY FORM]   Total edges: {len(edge_face_map)}")
    print("-" * 70)
    
    # ==========================================================================
    # NEW APPROACH: Classify polygons as BOUNDARY, HOLES, or ALT
    # Then split independent boundary polygons into separate faces
    # ==========================================================================
    print(f"\n[POLY FORM]   Classifying polygons as BOUNDARY, HOLES, or ALT...")
    
    for face_idx, face_eq in enumerate(unique_faces):
        polygons = face_eq.get('polygons', [])
        if len(polygons) == 0:
            continue
        
        # If only one polygon, it's the boundary
        if len(polygons) == 1:
            polygons[0]['polygon_type'] = 'BOUNDARY'
            polygons[0]['is_alternate'] = False
            polygons[0]['is_hole'] = False
            continue
        
        # Multiple polygons: find the true outer boundary
        # The boundary should contain all or most other polygons
        
        # Create Shapely polygons for all
        shapely_polygons = []
        for poly in polygons:
            try:
                verts_2d = [face_eq['face_results']['verts_2d'][v] 
                           for v in poly['vertices']]
                from shapely.geometry import Polygon as ShapelyPolygon
                shapely_poly = ShapelyPolygon(verts_2d)
                shapely_polygons.append(shapely_poly)
            except:
                shapely_polygons.append(None)
        
        # For each polygon, count how many others it contains
        containment_counts = []
        for i, shapely_i in enumerate(shapely_polygons):
            count = 0
            if shapely_i is not None:
                for j, shapely_j in enumerate(shapely_polygons):
                    if i != j and shapely_j is not None:
                        try:
                            if shapely_i.contains(shapely_j):
                                count += 1
                        except:
                            pass
            containment_counts.append(count)
        
        # The boundary is the polygon that contains the most others
        # If tied, use largest area as tiebreaker
        max_count = max(containment_counts)
        candidates = [i for i, c in enumerate(containment_counts) if c == max_count]
        
        if len(candidates) == 1:
            largest_idx = candidates[0]
        else:
            # Tiebreaker: largest area among candidates
            largest_idx = max(candidates, key=lambda i: polygons[i].get('area', 0))
        
        largest_poly = polygons[largest_idx]
        
        # Mark largest as boundary (override any pre-existing flags)
        polygons[largest_idx]['polygon_type'] = 'BOUNDARY'
        polygons[largest_idx]['is_alternate'] = False
        polygons[largest_idx]['is_hole'] = False
        
        print(f"[POLY FORM]     Face {face_idx+1}: Selected polygon "
              f"{largest_poly['vertices']} as BOUNDARY (contains "
              f"{containment_counts[largest_idx]} other polygon(s), "
              f"area={largest_poly.get('area', 0):.2f})")
        
        # Get 2D representation of boundary for containment tests
        boundary_shapely = shapely_polygons[largest_idx]
        
        # Classify remaining polygons
        for poly_idx, poly in enumerate(polygons):
            if poly_idx == largest_idx:
                continue
            
            # Always re-classify using Shapely containment test
            # (override any pre-existing flags from earlier processing)
            if boundary_shapely is not None:
                try:
                    poly_verts_2d = [face_eq['face_results']['verts_2d'][v] 
                                     for v in poly['vertices']]
                    poly_shapely = ShapelyPolygon(poly_verts_2d)
                    if boundary_shapely.contains(poly_shapely):
                        # Completely inside -> HOLE
                        poly['polygon_type'] = 'HOLE'
                        poly['is_hole'] = True
                        poly['is_alternate'] = False
                    else:
                        # Not inside -> ALT (alternate boundary candidate)
                        poly['polygon_type'] = 'ALT'
                        poly['is_alternate'] = True
                        poly['is_hole'] = False
                except Exception as e:
                    # Default to ALT if test fails
                    print(f"[POLY FORM]     WARNING: Failed to classify "
                          f"polygon {poly['vertices']}: {e}")
                    poly['polygon_type'] = 'ALT'
                    poly['is_alternate'] = True
                    poly['is_hole'] = False
            else:
                # Default to ALT if no boundary shapely
                poly['polygon_type'] = 'ALT'
                poly['is_alternate'] = True
                poly['is_hole'] = False
    
    # ==========================================================================
    # SPLIT INDEPENDENT BOUNDARY POLYGONS INTO SEPARATE FACES
    # ==========================================================================
    print(f"\n[POLY FORM]   Detecting independent boundary polygons...")
    
    new_faces_to_add = []
    
    for face_idx, face_eq in enumerate(unique_faces):
        polygons = face_eq.get('polygons', [])
        if len(polygons) <= 1:
            continue
        
        # Find the main boundary
        boundary_poly = next((p for p in polygons 
                             if p.get('polygon_type') == 'BOUNDARY'), None)
        if not boundary_poly:
            continue
        
        # Get Shapely polygon for boundary
        try:
            boundary_verts_2d = [face_eq['face_results']['verts_2d'][v] 
                                 for v in boundary_poly['vertices']]
            boundary_shapely = ShapelyPolygon(boundary_verts_2d)
        except:
            continue
        
        # Check ALT polygons to see if they are independent boundaries
        independent_boundaries = []
        
        for poly in polygons:
            if poly.get('polygon_type') != 'ALT':
                continue
            
            try:
                poly_verts_2d = [face_eq['face_results']['verts_2d'][v] 
                                 for v in poly['vertices']]
                poly_shapely = ShapelyPolygon(poly_verts_2d)
                
                # Check if this ALT polygon touches the boundary
                # If they don't intersect and ALT is not contained,
                # it's an independent boundary
                if (not boundary_shapely.intersects(poly_shapely) and
                    not boundary_shapely.contains(poly_shapely)):
                    # This is an independent boundary polygon
                    independent_boundaries.append(poly)
                    poly['is_independent_boundary'] = True
                    print(f"[POLY FORM]     Face {face_idx+1}: Found "
                          f"independent boundary polygon {poly['vertices']}")
            except Exception as e:
                print(f"[POLY FORM]     WARNING: Failed independence test "
                      f"for polygon {poly['vertices']}: {e}")
                continue
        
        # If we found independent boundaries, create new faces for them
        # BUT FIRST: Check if any independent boundaries are holes of others
        if independent_boundaries:
            print(f"[POLY FORM]     Face {face_idx+1}: Found "
                  f"{len(independent_boundaries)} independent "
                  f"boundary polygon(s), checking containment...")
            
            # Create Shapely polygons for all independent boundaries
            indep_with_shapely = []
            for indep_poly in independent_boundaries:
                try:
                    verts_2d = [face_eq['face_results']['verts_2d'][v] 
                                for v in indep_poly['vertices']]
                    shapely_poly = ShapelyPolygon(verts_2d)
                    indep_with_shapely.append({
                        'poly': indep_poly,
                        'shapely': shapely_poly,
                        'is_hole_of': None  # Will store index if this is hole
                    })
                except:
                    # Skip if can't create Shapely polygon
                    continue
            
            # Helper function to check if two vertex lists are the same polygon
            def are_same_polygon(verts1, verts2):
                """Check if two vertex lists represent the same polygon (possibly reversed)."""
                if len(verts1) != len(verts2):
                    return False
                
                # Normalize both lists to sets for comparison
                set1 = set(verts1)
                set2 = set(verts2)
                
                # If they don't have the same vertices, they're different
                if set1 != set2:
                    return False
                
                # Check if verts2 is verts1 in forward or reverse order
                # Find first vertex of verts1 in verts2
                if verts1[0] not in verts2:
                    return False
                
                idx = verts2.index(verts1[0])
                n = len(verts1)
                
                # Check forward direction
                forward_match = all(verts1[i] == verts2[(idx + i) % n] for i in range(n))
                
                # Check reverse direction
                reverse_match = all(verts1[i] == verts2[(idx - i) % n] for i in range(n))
                
                return forward_match or reverse_match
            
            # Check containment between independent boundaries
            # If polygon A contains polygon B, then B is a hole of A
            for i, item_i in enumerate(indep_with_shapely):
                for j, item_j in enumerate(indep_with_shapely):
                    if i == j:
                        continue
                    
                    # Skip if they are the same polygon (reversed vertices)
                    if are_same_polygon(item_i['poly']['vertices'], 
                                       item_j['poly']['vertices']):
                        print(f"[POLY FORM]       Skipping containment check: "
                              f"{item_j['poly']['vertices']} and "
                              f"{item_i['poly']['vertices']} are the same polygon")
                        continue
                    
                    try:
                        if item_i['shapely'].contains(item_j['shapely']):
                            # item_j is contained by item_i, so it's a hole
                            item_j['is_hole_of'] = i
                            print(f"[POLY FORM]       Polygon "
                                  f"{item_j['poly']['vertices']} is HOLE of "
                                  f"{item_i['poly']['vertices']}")
                    except:
                        continue
            
            # Now create faces: only for true boundaries (not holes)
            for i, item in enumerate(indep_with_shapely):
                if item['is_hole_of'] is not None:
                    # This is a hole, skip creating a face for it
                    continue
                
                indep_poly = item['poly']
                
                # Create a new face for this independent boundary
                new_face = {
                    'normal': face_eq['normal'],
                    'd': face_eq['d'],
                    'face_results': face_eq['face_results'],
                    'original_face_idx': face_idx,
                    'polygons': [],
                    'vertices_on_face': face_eq.get('vertices_on_face', []),
                    'edges_on_face': face_eq.get('edges_on_face', [])
                }
                
                # This independent polygon becomes the boundary of new face
                indep_poly['polygon_type'] = 'BOUNDARY'
                indep_poly['is_alternate'] = False
                indep_poly['is_hole'] = False
                new_face['polygons'].append(indep_poly)
                
                # Add any independent boundaries that are holes of this one
                for j, other_item in enumerate(indep_with_shapely):
                    if other_item['is_hole_of'] == i:
                        # This independent boundary is a hole of current one
                        hole_poly = other_item['poly']
                        hole_poly['polygon_type'] = 'HOLE'
                        hole_poly['is_hole'] = True
                        hole_poly['is_alternate'] = False
                        new_face['polygons'].append(hole_poly)
                        hole_poly['moved_to_new_face'] = True
                        print(f"[POLY FORM]       Adding independent polygon "
                              f"{hole_poly['vertices']} as HOLE to boundary "
                              f"{indep_poly['vertices']}")
                
                # Get Shapely polygon for new boundary
                new_boundary_shapely = item['shapely']
                
                # Check if any other polygons belong to this new face
                # This includes both:
                # 1. ALT polygons that are contained by new boundary
                # 2. HOLE polygons from original that actually belong here
                # 3. Independent boundaries already added above as holes
                for other_poly in polygons:
                    # Skip the independent boundary itself
                    if other_poly == indep_poly:
                        continue
                    # Skip already processed independent boundaries
                    if other_poly.get('is_independent_boundary'):
                        continue
                    # Skip if already moved to this face
                    if other_poly.get('moved_to_new_face'):
                        continue
                    
                    poly_type = other_poly.get('polygon_type')
                    
                    # Check both ALT and HOLE polygons
                    if poly_type in ['ALT', 'HOLE']:
                        try:
                            other_verts_2d = [
                                face_eq['face_results']['verts_2d'][v] 
                                for v in other_poly['vertices']
                            ]
                            other_shapely = ShapelyPolygon(other_verts_2d)
                            
                            if new_boundary_shapely.contains(other_shapely):
                                # This polygon belongs to new face as HOLE
                                other_poly['polygon_type'] = 'HOLE'
                                other_poly['is_hole'] = True
                                other_poly['is_alternate'] = False
                                new_face['polygons'].append(other_poly)
                                other_poly['moved_to_new_face'] = True
                                print(f"[POLY FORM]       Moving "
                                      f"{poly_type} polygon "
                                      f"{other_poly['vertices']} as HOLE "
                                      f"to new face")
                        except Exception as e:
                            continue
                
                new_faces_to_add.append(new_face)
            
            # Remove independent boundaries and moved polygons from original
            face_eq['polygons'] = [
                p for p in polygons 
                if not p.get('is_independent_boundary', False) and
                   not p.get('moved_to_new_face', False)
            ]
    
    # Add new faces to the list
    if new_faces_to_add:
        print(f"\n[POLY FORM]   Adding {len(new_faces_to_add)} new face(s) "
              f"from independent boundaries")
        unique_faces.extend(new_faces_to_add)
    
    # Show initial face composition with classifications
    print(f"\n[POLY FORM]   Initial face composition (before validation):")
    for face_idx, face_eq in enumerate(unique_faces):
        poly_count = len(face_eq.get('polygons', []))
        if poly_count == 0:
            print(f"[POLY FORM]     Face {face_idx+1}: NO polygons")
        else:
            print(f"[POLY FORM]     Face {face_idx+1}: {poly_count} polygon(s)")
            for poly_idx, poly_data in enumerate(face_eq.get('polygons', [])):
                poly_verts = poly_data.get('vertices', [])
                poly_type = poly_data.get('polygon_type', 'UNKNOWN')
                print(f"[POLY FORM]       Polygon {poly_idx+1} ({poly_type}): {poly_verts}")
    
    # ==========================================================================
    # MODIFIED APPROACH: Start with ALL polygons, then remove problematic ALTs
    # ==========================================================================
    print(f"\n[POLY FORM]   Building initial edge distribution with ALL polygons (BOUNDARY + HOLES + ALL ALTs)...")
    
    edge_face_map_all = {}  # edges from ALL polygons
    all_polygons_list = []  # List of ALL polygons
    alt_polygons = []  # Track which are ALTs for removal consideration
    
    for face_idx, face_eq in enumerate(unique_faces):
        polygons = face_eq.get('polygons', [])
        if len(polygons) == 0:
            continue
        
        for poly_idx, poly_data in enumerate(polygons):
            poly_type = poly_data.get('polygon_type', 'BOUNDARY')
            poly_verts = poly_data.get('vertices', [])
            
            # Add ALL polygons to initial distribution
            all_polygons_list.append({
                'face_idx': face_idx,
                'polygon_idx': poly_idx,
                'data': poly_data,
                'face_eq': face_eq,
                'poly_type': poly_type
            })
            
            # Track ALT polygons separately for removal testing
            if poly_type == 'ALT':
                alt_polygons.append({
                    'face_idx': face_idx,
                    'polygon_idx': poly_idx,
                    'data': poly_data,
                    'face_eq': face_eq
                })
            
            # Register edges (vertices are 1-based, convert to 0-based for edges)
            for i in range(len(poly_verts)):
                v1 = poly_verts[i] - 1  # Convert to 0-based
                v2 = poly_verts[(i + 1) % len(poly_verts)] - 1
                edge = (min(v1, v2), max(v1, v2))
                if edge not in edge_face_map_all:
                    edge_face_map_all[edge] = []
                edge_face_map_all[edge].append((face_idx, poly_idx, poly_type))
    
    print(f"[POLY FORM]     Registered {len(all_polygons_list)} total polygons")
    print(f"[POLY FORM]     Including {len(alt_polygons)} ALT polygons")
    
    # Check initial edge distribution (with ALL polygons)
    def compute_edge_stats(edge_map):
        """Helper to compute edge distribution statistics"""
        edges_1 = sum(1 for fl in edge_map.values() if len(fl) == 1)
        edges_2 = sum(1 for fl in edge_map.values() if len(fl) == 2)
        edges_3plus = sum(1 for fl in edge_map.values() if len(fl) >= 3)
        return edges_1, edges_2, edges_3plus
    
    initial_boundary, initial_manifold, initial_invalid = compute_edge_stats(edge_face_map_all)
    
    print(f"\n[POLY FORM]   Initial edge distribution (ALL polygons included):")
    print(f"[POLY FORM]     - Boundary edges (1 face): {initial_boundary}")
    print(f"[POLY FORM]     - Manifold edges (2 faces): {initial_manifold}")
    print(f"[POLY FORM]     - Invalid edges (3+ faces): {initial_invalid}")
    
    # ==========================================================================
    # Iteratively remove ALT polygons whose removal improves topology
    # ==========================================================================
    print(f"\n[POLY FORM]   Testing ALT polygon removal (greedy removal approach)...")
    
    removed_alts = []
    remaining_alts = list(alt_polygons)
    current_edge_map = dict(edge_face_map_all)
    current_boundary, current_manifold, current_invalid = initial_boundary, initial_manifold, initial_invalid
    
    iteration = 0
    max_iterations = len(alt_polygons) + 5  # Safety limit
    
    while remaining_alts and iteration < max_iterations:
        iteration += 1
        
        # Test removing each remaining ALT polygon
        best_alt = None
        best_invalid_reduction = -float('inf')
        best_boundary_reduction = -float('inf')
        
        for alt_info in remaining_alts:
            face_idx = alt_info['face_idx']
            poly_idx = alt_info['polygon_idx']
            poly_data = alt_info['data']
            poly_verts = poly_data.get('vertices', [])
            
            # Create edge map without this ALT
            edge_map_without_alt = dict(current_edge_map)
            
            # Remove this ALT's edges
            for i in range(len(poly_verts)):
                v1 = poly_verts[i] - 1
                v2 = poly_verts[(i + 1) % len(poly_verts)] - 1
                edge = (min(v1, v2), max(v1, v2))
                
                if edge in edge_map_without_alt:
                    # Remove this polygon's contribution to the edge
                    edge_map_without_alt[edge] = [
                        entry for entry in edge_map_without_alt[edge]
                        if not (entry[0] == face_idx and entry[1] == poly_idx)
                    ]
                    # Remove edge entirely if no faces use it
                    if len(edge_map_without_alt[edge]) == 0:
                        del edge_map_without_alt[edge]
            
            # Compute stats without this ALT
            boundary_without, manifold_without, invalid_without = compute_edge_stats(edge_map_without_alt)
            
            # Calculate reductions (positive = improvement)
            invalid_reduction = current_invalid - invalid_without
            boundary_reduction = current_boundary - boundary_without
            
            # Update best candidate based on criteria:
            # 1. Most reduction in invalid edges (primary)
            # 2. Most reduction in boundary edges (tiebreaker)
            if (invalid_reduction > best_invalid_reduction or 
                (invalid_reduction == best_invalid_reduction and 
                 boundary_reduction > best_boundary_reduction)):
                best_alt = alt_info
                best_invalid_reduction = invalid_reduction
                best_boundary_reduction = boundary_reduction
                best_edge_map = edge_map_without_alt
                best_stats = (boundary_without, manifold_without, invalid_without)
        
        # Check if we found a beneficial removal
        if best_alt and (best_invalid_reduction > 0 or best_boundary_reduction > 0):
            face_idx = best_alt['face_idx']
            poly_idx = best_alt['polygon_idx']
            poly_verts = best_alt['data'].get('vertices', [])
            
            print(f"\n[POLY FORM]     Removing ALT from Face {face_idx+1}, "
                  f"Polygon {poly_idx+1}: {poly_verts}")
            print(f"[POLY FORM]       Invalid edges: {current_invalid} → {best_stats[2]} "
                  f"(Δ={best_invalid_reduction:+d})")
            print(f"[POLY FORM]       Boundary edges: {current_boundary} → {best_stats[0]} "
                  f"(Δ={best_boundary_reduction:+d})")
            
            # Apply the removal
            current_edge_map = best_edge_map
            current_boundary, current_manifold, current_invalid = best_stats
            removed_alts.append(best_alt)
            remaining_alts.remove(best_alt)
            
            # Mark for removal from face structure
            best_alt['data']['to_remove'] = True
        else:
            # No beneficial removal found, stop
            print(f"\n[POLY FORM]   No more beneficial ALT removals found")
            break
    
    print(f"\n[POLY FORM]   ALT polygon removal complete:")
    print(f"[POLY FORM]     - Removed: {len(removed_alts)} ALT polygons")
    print(f"[POLY FORM]     - Kept: {len(remaining_alts)} ALT polygons")
    
    # Remove rejected ALT polygons from face data
    for alt_info in removed_alts:
        face_idx = alt_info['face_idx']
        face_eq = unique_faces[face_idx]
        
        # Remove from polygons list
        if 'polygons' in face_eq:
            face_eq['polygons'] = [p for p in face_eq['polygons'] 
                                    if not p.get('to_remove', False)]
    
    # Use the final edge map
    edge_face_map = current_edge_map
    
    # Rebuild all_face_polygons list from remaining polygons
    all_face_polygons = []
    for face_idx, face_eq in enumerate(unique_faces):
        polygons = face_eq.get('polygons', [])
        for poly_idx, poly_data in enumerate(polygons):
            all_face_polygons.append({
                'face_idx': face_idx,
                'polygon_idx': poly_idx,
                'data': poly_data,
                'face_eq': face_eq,
                'poly_type': poly_data.get('polygon_type', 'BOUNDARY')
            })
    
    # Final edge distribution
    print(f"\n[POLY FORM]   Final edge distribution after ALT removal:")
    print(f"[POLY FORM]     - Boundary edges (1 face): {current_boundary}")
    print(f"[POLY FORM]     - Manifold edges (2 faces): {current_manifold}")
    print(f"[POLY FORM]     - Invalid edges (3+ faces): {current_invalid}")
    
    # Diagnostic: Check edge distribution
    edges_in_1_face = []
    edges_in_2_faces = []
    edges_in_3plus_faces = []
    
    for edge, face_list in edge_face_map.items():
        if len(face_list) == 1:
            edges_in_1_face.append((edge, face_list))
        elif len(face_list) == 2:
            edges_in_2_faces.append((edge, face_list))
        else:
            edges_in_3plus_faces.append((edge, face_list))
    
    print(f"[POLY FORM]   Edge distribution:")
    print(f"[POLY FORM]     - Edges in 1 face: {len(edges_in_1_face)} (boundary edges)")
    if len(edges_in_1_face) > 0 and len(edges_in_1_face) <= 20:
        print(f"[POLY FORM]       Boundary edges:")
        for edge, face_list in edges_in_1_face:
            face_idx, poly_idx, poly_type = face_list[0]
            poly_letter = chr(ord('a') + poly_idx)
            face_label = f"{face_idx+1}{poly_letter}"
            print(f"[POLY FORM]         Edge {edge} in face {face_label} ({poly_type})")
    print(f"[POLY FORM]     - Edges in 2 faces: {len(edges_in_2_faces)} (manifold edges)")
    print(f"[POLY FORM]     - Edges in 3+ faces: {len(edges_in_3plus_faces)} (invalid!)")
    
    # Show first few edges that appear in only 1 face (might be hole edges)
    if len(edges_in_1_face) > 0 and len(edges_in_1_face) <= 20:
        print(f"[POLY FORM]   Edges appearing in only 1 face:")
        for edge, face_list in edges_in_1_face[:20]:
            face_idx, poly_idx, poly_type = face_list[0]
            is_hole = (poly_type == 'HOLE')
            is_alternate = (poly_type == 'ALT_KEPT')
            poly_vertices = None
            if face_idx < len(unique_faces):
                face_eq = unique_faces[face_idx]
                if 'polygons' in face_eq and poly_idx < len(face_eq['polygons']):
                    poly_data_obj = face_eq['polygons'][poly_idx]
                    poly_vertices = poly_data_obj['vertices']
            hole_marker = " (HOLE EDGE)" if is_hole else ""
            alternate_marker = " (ALTERNATE REGION)" if is_alternate else ""
            # Edge uses 0-based indexing, display as-is
            edge_display = edge
            
            # Verify edge is actually in polygon vertex list
            # Note: poly_vertices are already 1-based, so we need to compare edge+1
            edge_in_polygon = False
            if poly_vertices:
                for i in range(len(poly_vertices)):
                    v1 = poly_vertices[i]
                    v2 = poly_vertices[(i + 1) % len(poly_vertices)]
                    # Convert edge to 1-based for comparison
                    edge_1based = (min(edge[0]+1, edge[1]+1), max(edge[0]+1, edge[1]+1))
                    if (min(v1, v2), max(v1, v2)) == edge_1based:
                        edge_in_polygon = True
                        break
            
            edge_status = "✓" if edge_in_polygon else "✗ MISMATCH"
            # Polygon vertices are already 1-based, display as-is
            poly_info = f" polygon_vertices={poly_vertices}" if poly_vertices else ""
            # Build face label with polygon suffix (a, b, c, ...)
            poly_letter = chr(ord('a') + poly_idx)
            face_label = f"{face_idx+1}{poly_letter}"
            print(f"[POLY FORM]     Edge {edge_display} (connects vertices {edge[0]} and {edge[1]}) "
                  f"{edge_status}: in face {face_label}{hole_marker}{alternate_marker}{poly_info}")
            
            # Check if this edge should appear in other faces
            # by checking if both vertices appear in other face planes
            v1, v2 = edge
            v1_coord = selected_vertices[v1]
            v2_coord = selected_vertices[v2]
            
            # Find other faces that contain both vertices
            for other_face_idx, other_face_eq in enumerate(unique_faces):
                if other_face_idx == face_idx:
                    continue
                
                # Check if both vertices lie on this face's plane
                normal = np.array(other_face_eq['normal'])
                d = other_face_eq['d']
                
                dist1 = abs(np.dot(normal, v1_coord) - d)
                dist2 = abs(np.dot(normal, v2_coord) - d)
                
                if dist1 < 1e-6 and dist2 < 1e-6:
                    # Both vertices on this plane - edge should be in this face too!
                    print(f"[POLY FORM]       → Edge SHOULD also be in face {other_face_idx+1} "
                          f"(both vertices on plane)")
                    
                    # Check if this edge is in the face's edge list
                    if 'edges_on_face' in other_face_eq:
                        if edge in other_face_eq['edges_on_face'] or (edge[1], edge[0]) in other_face_eq['edges_on_face']:
                            print(f"[POLY FORM]          Edge IS in face edges but NOT in polygons!")
                        else:
                            print(f"[POLY FORM]          Edge NOT in face edges list")
    
    # Iteratively remove polygons with invalid edges
    print(f"\n[POLY FORM]   Starting iterative removal of invalid edges...")
    removal_iteration = 0
    max_removal_iterations = 100
    
    while removal_iteration < max_removal_iterations:
        removal_iteration += 1
        print(f"\n[POLY FORM]   === Removal Iteration {removal_iteration} ===")
        
        # Rebuild edge_face_map from current polygons
        edge_face_map = {}
        all_face_polygons = []
        
        for face_idx, face_eq in enumerate(unique_faces):
            if 'polygons' not in face_eq or len(face_eq['polygons']) == 0:
                continue
            
            for poly_idx, poly_data in enumerate(face_eq['polygons']):
                poly_verts = poly_data['vertices']
                
                # Track this polygon
                all_face_polygons.append({
                    'face_idx': face_idx,
                    'polygon_idx': poly_idx,
                    'data': poly_data,
                    'face_eq': face_eq,
                    'is_hole': poly_data.get('is_hole', False),
                    'is_alternate': poly_data.get('is_alternate', False)
                })
                
                # Register edges (vertices are 1-based, convert to 0-based for edges)
                for i in range(len(poly_verts)):
                    v1 = poly_verts[i] - 1  # Convert to 0-based
                    v2 = poly_verts[(i + 1) % len(poly_verts)] - 1  # Convert to 0-based
                    edge = (min(v1, v2), max(v1, v2))
                    if edge not in edge_face_map:
                        edge_face_map[edge] = []
                    edge_face_map[edge].append((face_idx, poly_idx))
        
        # Check edge distribution
        edges_in_1_face = []
        edges_in_2_faces = []
        edges_in_3plus_faces = []
        
        for edge, face_list in edge_face_map.items():
            if len(face_list) == 1:
                edges_in_1_face.append((edge, face_list))
            elif len(face_list) == 2:
                edges_in_2_faces.append((edge, face_list))
            else:
                edges_in_3plus_faces.append((edge, face_list))
        
        print(f"[POLY FORM]     Total edges: {len(edge_face_map)}")
        print(f"[POLY FORM]     - Boundary edges (1 face): {len(edges_in_1_face)}")
        print(f"[POLY FORM]     - Manifold edges (2 faces): {len(edges_in_2_faces)}")
        print(f"[POLY FORM]     - Invalid edges (3+ faces): {len(edges_in_3plus_faces)}")
        
        # If no invalid edges, we're done
        if len(edges_in_3plus_faces) == 0:
            print(f"[POLY FORM]     ✓ No invalid edges remaining!")
            break
        
        # Count invalid edges per polygon
        polygon_invalid_count = {}  # (face_idx, poly_idx) -> count
        
        for edge, face_list in edges_in_3plus_faces:
            for face_idx, poly_idx in face_list:
                key = (face_idx, poly_idx)
                if key not in polygon_invalid_count:
                    polygon_invalid_count[key] = 0
                polygon_invalid_count[key] += 1
        
        # Sort polygons by invalid edge count (descending)
        # If same count, prioritize alternates (2), then holes (1), then boundaries (0)
        def sort_key(item):
            (face_idx, poly_idx), count = item
            # Check if alternate or hole
            is_alternate = False
            is_hole = False
            for fp in all_face_polygons:
                if fp['face_idx'] == face_idx and fp['polygon_idx'] == poly_idx:
                    is_alternate = fp.get('is_alternate', False)
                    is_hole = fp.get('is_hole', False)
                    break
            priority = 2 if is_alternate else 1 if is_hole else 0
            return (-count, -priority)  # Sort by count desc, then priority desc
        
        sorted_polygons = sorted(polygon_invalid_count.items(), key=sort_key)
        
        print(f"\n[POLY FORM]     Polygons ranked by invalid edge count:")
        for (face_idx, poly_idx), count in sorted_polygons[:10]:  # Show top 10
            poly_letter = chr(ord('a') + poly_idx)
            face_label = f"{face_idx+1}{poly_letter}"
            # Check if alternate or hole
            is_alternate = False
            is_hole = False
            for fp in all_face_polygons:
                if fp['face_idx'] == face_idx and fp['polygon_idx'] == poly_idx:
                    is_alternate = fp.get('is_alternate', False)
                    is_hole = fp.get('is_hole', False)
                    break
            poly_type = "(ALTERNATE)" if is_alternate else "(HOLE)" if is_hole else "(BOUNDARY)"
            print(f"[POLY FORM]       Face {face_label} {poly_type}: {count} invalid edges")
        
        # Try removing polygons - for same invalid edge count, choose best reduction
        if len(sorted_polygons) == 0:
            print(f"[POLY FORM]     No polygons to remove")
            break
        
        current_invalid_count = len(edges_in_3plus_faces)
        current_boundary_count = len(edges_in_1_face)
        
        # Group polygons by invalid edge count
        max_invalid_count = sorted_polygons[0][1]
        candidates_with_max = [(poly, count) for poly, count in sorted_polygons if count == max_invalid_count]
        
        # Evaluate ALL candidates with the same invalid edge count
        # regardless of boundary/alternate status to find the best removal
        print(f"\n[POLY FORM]     Evaluating {len(candidates_with_max)} polygon(s) with {max_invalid_count} invalid edges")
        
        # Evaluate each candidate and track the best one
        best_candidate = None
        best_reduction = None
        best_new_invalid = None
        best_new_boundary = None
        best_is_alternate = None
        candidate_results = []
        
        for candidate_polygon, candidate_count in candidates_with_max:
            candidate_face_idx, candidate_poly_idx = candidate_polygon
            
            # Get polygon type
            is_alternate = False
            is_hole = False
            for fp in all_face_polygons:
                if fp['face_idx'] == candidate_face_idx and fp['polygon_idx'] == candidate_poly_idx:
                    is_alternate = fp.get('is_alternate', False)
                    is_hole = fp.get('is_hole', False)
            
            poly_letter = chr(ord('a') + candidate_poly_idx)
            face_label = f"{candidate_face_idx+1}{poly_letter}"
            poly_type = "ALT" if is_alternate else "HOLE" if is_hole else "BOUNDARY"
            
            # Save the polygon before removing
            face_eq = unique_faces[candidate_face_idx]
            if 'polygons' not in face_eq or candidate_poly_idx >= len(face_eq['polygons']):
                continue
            
            removed_poly = face_eq['polygons'][candidate_poly_idx]
            
            # Temporarily remove the polygon
            del face_eq['polygons'][candidate_poly_idx]
            
            # Rebuild edge_face_map to check the effect
            test_edge_face_map = {}
            for test_face_idx, test_face_eq in enumerate(unique_faces):
                if 'polygons' not in test_face_eq or len(test_face_eq['polygons']) == 0:
                    continue
                
                for test_poly_idx, test_poly_data in enumerate(test_face_eq['polygons']):
                    test_poly_verts = test_poly_data['vertices']
                    
                    # Register edges (vertices are 1-based, convert to 0-based for edges)
                    for i in range(len(test_poly_verts)):
                        v1 = test_poly_verts[i] - 1
                        v2 = test_poly_verts[(i + 1) % len(test_poly_verts)] - 1
                        edge = (min(v1, v2), max(v1, v2))
                        if edge not in test_edge_face_map:
                            test_edge_face_map[edge] = []
                        test_edge_face_map[edge].append((test_face_idx, test_poly_idx))
            
            # Count new edge distribution
            new_invalid_count = sum(1 for face_list in test_edge_face_map.values() if len(face_list) > 2)
            new_boundary_count = sum(1 for face_list in test_edge_face_map.values() if len(face_list) == 1)
            
            invalid_reduction = current_invalid_count - new_invalid_count
            boundary_reduction = current_boundary_count - new_boundary_count
            
            # Store results
            candidate_results.append({
                'label': face_label,
                'type': poly_type,
                'polygon': (candidate_face_idx, candidate_poly_idx),
                'removed_poly': removed_poly,
                'new_invalid': new_invalid_count,
                'new_boundary': new_boundary_count,
                'invalid_reduction': invalid_reduction,
                'boundary_reduction': boundary_reduction,
                'is_alternate': is_alternate,
                'is_hole': is_hole
            })
            
            # Restore the polygon for now
            face_eq['polygons'].insert(candidate_poly_idx, removed_poly)
            
            # Track best candidate: prioritize invalid reduction, then boundary reduction
            if best_candidate is None or \
               invalid_reduction > best_reduction[0] or \
               (invalid_reduction == best_reduction[0] and boundary_reduction > best_reduction[1]):
                best_candidate = (candidate_face_idx, candidate_poly_idx)
                best_reduction = (invalid_reduction, boundary_reduction)
                best_new_invalid = new_invalid_count
                best_new_boundary = new_boundary_count
                best_is_alternate = is_alternate
                best_new_boundary = new_boundary_count
        
        # Show evaluation results
        print(f"[POLY FORM]     Candidate evaluation results:")
        for result in candidate_results:
            print(f"[POLY FORM]       {result['label']} ({result['type']}): invalid Δ={result['invalid_reduction']:+d}, "
                  f"boundary Δ={result['boundary_reduction']:+d} "
                  f"(new: invalid={result['new_invalid']}, boundary={result['new_boundary']})")
        
        # Check if best candidate improves things
        if best_candidate is None:
            print(f"[POLY FORM]     No valid candidates found")
            break
        
        # Try candidates in order until we find one that can be removed
        # Sort candidate_results by reduction quality (best first)
        sorted_candidates = sorted(
            candidate_results,
            key=lambda r: (-r['invalid_reduction'], -r['boundary_reduction'])
        )
        
        removal_successful = False
        for candidate_result in sorted_candidates:
            candidate_face_idx = candidate_result['polygon'][0]
            candidate_poly_idx = candidate_result['polygon'][1]
            poly_letter = chr(ord('a') + candidate_poly_idx)
            face_label = f"{candidate_face_idx+1}{poly_letter}"
            
            # Check if removal improves or maintains edge quality
            # IMPORTANT: Don't remove if it would leave the face with NO polygons
            face_eq = unique_faces[candidate_face_idx]
            
            # Count current valid polygons in this face
            current_poly_count = len(face_eq.get('polygons', []))
            
            # ALLOW removal even if it would leave face with 0 polygons
            # The face will be marked for deletion later
            will_delete_face = (current_poly_count <= 1)
            
            # Check if this candidate would improve edge quality
            if candidate_result['new_invalid'] < current_invalid_count or \
               (candidate_result['new_invalid'] == current_invalid_count and candidate_result['new_boundary'] <= current_boundary_count):
                
                print(f"\n[POLY FORM]     ✓ Removing polygon {face_label} from face {candidate_face_idx+1}: "
                      f"invalid Δ={candidate_result['invalid_reduction']:+d}, boundary Δ={candidate_result['boundary_reduction']:+d}")
                print(f"[POLY FORM]       Face {candidate_face_idx+1} will have {current_poly_count-1} polygon(s) remaining")
                
                # Check if we're removing a boundary polygon
                removed_poly = face_eq['polygons'][candidate_poly_idx]
                is_removing_boundary = not removed_poly.get('is_alternate', False) and not removed_poly.get('is_hole', False)
                
                # Mark face for deletion if this will leave it empty
                if will_delete_face:
                    print(f"[POLY FORM]       ⚠ Removing last polygon will mark face {candidate_face_idx+1} for deletion")
                    if 'mark_for_deletion' not in face_eq:
                        face_eq['mark_for_deletion'] = True
                
                print(f"[POLY FORM]       Removed polygon vertices: {removed_poly['vertices']}")
                del face_eq['polygons'][candidate_poly_idx]
                
                # If we removed a boundary polygon, check if face still has a boundary
                if is_removing_boundary:
                    has_boundary = False
                    for poly_data in face_eq.get('polygons', []):
                        if not poly_data.get('is_alternate', False) and not poly_data.get('is_hole', False):
                            has_boundary = True
                            break
                    
                    # If no boundary remains, promote first alternate to boundary
                    if not has_boundary and len(face_eq.get('polygons', [])) > 0:
                        for poly_data in face_eq.get('polygons', []):
                            if poly_data.get('is_alternate', False):
                                poly_data['is_alternate'] = False
                                print(f"[POLY FORM]       ⚠ Promoted alternate {poly_data['vertices']} to boundary polygon")
                                break
                
                removal_successful = True
                break  # Successfully removed, exit candidate loop
            else:
                print(f"\n[POLY FORM]     ✗ Candidate {face_label} would worsen edge quality")
                continue  # Try next candidate
        
        # If no candidate could be removed, stop iterations
        if not removal_successful:
            print(f"[POLY FORM]     No beneficial removal found, stopping iterations")
            print(f"\n[POLY FORM]     ✗ Best candidate {face_label} would worsen edge quality")
            print(f"[POLY FORM]     No beneficial removal found, stopping iterations")
            break
    
    # Final summary after all iterations
    print(f"\n[POLY FORM]   Completed {removal_iteration} removal iteration(s)")
    
    # Verify face integrity and delete faces marked for deletion
    print(f"\n[POLY FORM]   Verifying face integrity...")
    faces_to_delete = []
    for face_idx, face_eq in enumerate(unique_faces):
        poly_count = len(face_eq.get('polygons', []))
        if poly_count == 0 or face_eq.get('mark_for_deletion', False):
            if poly_count == 0:
                print(f"[POLY FORM]     Face {face_idx+1}: NO polygons - marking for deletion")
            else:
                print(f"[POLY FORM]     Face {face_idx+1}: marked for deletion")
            faces_to_delete.append(face_idx)
        else:
            print(f"[POLY FORM]     Face {face_idx+1}: {poly_count} polygon(s) remaining")
            # Show what polygons remain
            for poly_idx, poly_data in enumerate(face_eq.get('polygons', [])):
                poly_verts = poly_data.get('vertices', [])
                is_alt = poly_data.get('is_alternate', False)
                is_hole = poly_data.get('is_hole', False)
                poly_type = "ALT" if is_alt else "HOLE" if is_hole else "BOUNDARY"
                print(f"[POLY FORM]       Polygon {poly_idx+1} ({poly_type}): {poly_verts}")
    
    # Delete faces that were marked for deletion
    if faces_to_delete:
        print(f"\n[POLY FORM]   Deleting {len(faces_to_delete)} face(s) with no polygons...")
        for face_idx in sorted(faces_to_delete, reverse=True):
            print(f"[POLY FORM]     Deleting face {face_idx+1}")
            del unique_faces[face_idx]
        print(f"[POLY FORM]   Remaining faces: {len(unique_faces)}")
    
    # Re-classify multiple boundaries as alternates or holes
    print(f"\n[POLY FORM]   Re-classifying multiple boundaries...")
    for face_idx, face_eq in enumerate(unique_faces):
        if 'polygons' not in face_eq or len(face_eq['polygons']) == 0:
            continue
        
        polygons = face_eq['polygons']
        
        # Find all boundary polygons (not holes, not alternates)
        boundaries = []
        for poly_data in polygons:
            if not poly_data.get('is_hole', False) and not poly_data.get('is_alternate', False):
                boundaries.append(poly_data)
        
        # If multiple boundaries exist, reclassify extras
        if len(boundaries) > 1:
            # Keep the largest boundary, reclassify others
            # Calculate areas for each boundary
            boundary_areas = []
            for boundary in boundaries:
                area = boundary.get('area', 0)
                if area == 0 and boundary.get('shapely_2d') is not None:
                    area = boundary['shapely_2d'].area
                boundary_areas.append(area)
            
            # Find the largest boundary
            largest_idx = boundary_areas.index(max(boundary_areas))
            primary_boundary = boundaries[largest_idx]
            
            # Reclassify others as alternates or holes
            for i, boundary in enumerate(boundaries):
                if i == largest_idx:
                    continue
                
                # Check if this boundary is contained within the primary
                is_hole = False
                try:
                    primary_shapely = primary_boundary.get('shapely_2d')
                    boundary_shapely = boundary.get('shapely_2d')
                    
                    if primary_shapely is not None and boundary_shapely is not None:
                        if primary_shapely.contains(boundary_shapely):
                            is_hole = True
                except Exception:
                    pass
                
                if is_hole:
                    boundary['is_hole'] = True
                    print(f"[POLY FORM]     Face {face_idx+1}: Reclassifying boundary {boundary['vertices']} as HOLE")
                else:
                    boundary['is_alternate'] = True
                    print(f"[POLY FORM]     Face {face_idx+1}: Reclassifying boundary {boundary['vertices']} as ALTERNATE")
    
    # Rebuild final edge statistics
    edge_face_map = {}
    for face_idx, face_eq in enumerate(unique_faces):
        if 'polygons' not in face_eq or len(face_eq['polygons']) == 0:
            continue
        
        for poly_idx, poly_data in enumerate(face_eq['polygons']):
            poly_verts = poly_data['vertices']
            for i in range(len(poly_verts)):
                v1 = poly_verts[i] - 1
                v2 = poly_verts[(i + 1) % len(poly_verts)] - 1
                edge = (min(v1, v2), max(v1, v2))
                if edge not in edge_face_map:
                    edge_face_map[edge] = []
                edge_face_map[edge].append((face_idx, poly_idx))
    
    edges_in_1_face = [e for e in edge_face_map.items() if len(e[1]) == 1]
    edges_in_2_faces = [e for e in edge_face_map.items() if len(e[1]) == 2]
    edges_in_3plus_faces = [e for e in edge_face_map.items() if len(e[1]) > 2]
    
    print(f"[POLY FORM]   Final edge distribution:")
    print(f"[POLY FORM]     - Boundary edges: {len(edges_in_1_face)}")
    print(f"[POLY FORM]     - Manifold edges: {len(edges_in_2_faces)}")
    print(f"[POLY FORM]     - Invalid edges: {len(edges_in_3plus_faces)}")
    
    # Extract alternates as separate faces
    print(f"\n[POLY FORM]   Extracting alternate polygons as separate faces...")
    new_faces = []
    for face_idx, face_eq in enumerate(unique_faces):
        if 'polygons' not in face_eq or len(face_eq['polygons']) == 0:
            continue
        
        # Separate boundary/holes from alternates
        primary_polygons = []
        alternates = []
        boundary_polygon = None
        
        for poly_data in face_eq['polygons']:
            if poly_data.get('is_alternate', False):
                alternates.append(poly_data)
            else:
                primary_polygons.append(poly_data)
                # Find the boundary polygon (not a hole)
                if not poly_data.get('is_hole', False) and boundary_polygon is None:
                    boundary_polygon = poly_data
        
        # Create new face for each alternate that is NOT a hole
        for alt_poly in alternates:
            # Check if alternate is contained within boundary (i.e., it's a hole)
            is_hole = False
            if boundary_polygon is not None:
                try:
                    # Get the shapely polygon if available
                    boundary_shapely = boundary_polygon.get('shapely_2d')
                    alt_shapely = alt_poly.get('shapely_2d')
                    
                    if boundary_shapely is not None and alt_shapely is not None:
                        # Check if alternate is entirely contained within boundary
                        if boundary_shapely.contains(alt_shapely):
                            is_hole = True
                            print(f"[POLY FORM]     Face {face_idx+1} alternate {alt_poly['vertices']} is a HOLE (contained within boundary) - adding to face holes")
                except Exception:
                    pass
            
            if is_hole:
                # Add it back as a hole to the current face
                alt_poly['is_hole'] = True
                alt_poly['is_alternate'] = False
                primary_polygons.append(alt_poly)
            else:
                # Create a new face entry with the same normal and d value
                new_face = {
                    'normal': face_eq['normal'].copy(),
                    'd': face_eq['d'],
                    'vertices': alt_poly['vertices'].copy(),
                    'holes': [],
                    'vertices_on_face': face_eq.get('vertices_on_face', []),
                    'edges_on_face': face_eq.get('edges_on_face', []),
                    'all_vertices_on_face': face_eq.get('all_vertices_on_face', []),
                    'edges': face_eq.get('edges_on_face', []),
                    'polygons': [{
                        'vertices': alt_poly['vertices'].copy(),
                        'is_hole': False,
                        'is_alternate': False  # It's now a primary boundary
                    }]
                }
                new_faces.append(new_face)
                print(f"[POLY FORM]     Created new face from Face {face_idx+1} alternate: {alt_poly['vertices']}")
        
        # Keep primary polygons (boundaries and holes, including alternates that became holes)
        face_eq['polygons'] = primary_polygons
    
    # Add the new faces to unique_faces
    if len(new_faces) > 0:
        unique_faces.extend(new_faces)
        print(f"[POLY FORM]     Added {len(new_faces)} new face(s) from alternates")
        print(f"[POLY FORM]     Total faces now: {len(unique_faces)}")
    
    # Step 6.5: Remove duplicate polygons within each face
    print("\n[POLY FORM] Step 6.5: Removing duplicate polygons")
    print("-" * 70)
    for face_idx, face_eq in enumerate(unique_faces):
        if 'polygons' not in face_eq or len(face_eq['polygons']) == 0:
            continue
        
        polygons = face_eq['polygons']
        unique_polygons = []
        seen_vertex_sets = []
        
        for poly in polygons:
            # Convert vertices list to a frozenset for comparison
            vertex_set = frozenset(poly['vertices'])
            
            # Check if we've seen this exact set of vertices
            if vertex_set not in seen_vertex_sets:
                unique_polygons.append(poly)
                seen_vertex_sets.append(vertex_set)
            else:
                print(f"[POLY FORM]   Face {face_idx+1}: Removed duplicate polygon with vertices {poly['vertices']}")
        
        original_count = len(polygons)
        face_eq['polygons'] = unique_polygons
        removed_count = original_count - len(unique_polygons)
        
        if removed_count > 0:
            print(f"[POLY FORM]   Face {face_idx+1}: Removed {removed_count} duplicate polygon(s), {len(unique_polygons)} remain")
    
    # Step 7: Compile face results (boundary/holes/alternates already determined earlier)
    print("\n[POLY FORM] Step 7: Compiling face results")
    print("-" * 70)
    print("[POLY FORM]   (Using validated polygons from Step 6)")
    
    # Build faces from the validated polygons list (after removal of invalid edges)
    for face_idx, face_eq in enumerate(unique_faces):
        if 'polygons' not in face_eq or len(face_eq['polygons']) == 0:
            continue
        
        # Get validated polygons
        polygons = face_eq['polygons']
        
        # Separate boundaries, alternates, and holes
        boundaries = []
        alternates_list = []
        holes_list = []
        
        for poly_data in polygons:
            if poly_data.get('is_hole', False):
                holes_list.append(poly_data['vertices'])
            elif poly_data.get('is_alternate', False):
                alternates_list.append({
                    'vertices': poly_data['vertices'],
                    'shapely_2d': poly_data.get('shapely_2d'),
                    'area': poly_data.get('area', 0)
                })
            else:
                # This is a boundary polygon
                boundaries.append(poly_data)
        
        # If there are multiple boundaries, treat extras as alternates
        if len(boundaries) > 1:
            # Keep the first boundary, move others to alternates
            primary_boundary = boundaries[0]
            for extra_boundary in boundaries[1:]:
                alternates_list.append({
                    'vertices': extra_boundary['vertices'],
                    'shapely_2d': extra_boundary.get('shapely_2d'),
                    'area': extra_boundary.get('area', 0)
                })
                print(f"[POLY FORM]   Face {face_idx+1}: Moving extra boundary {extra_boundary['vertices']} to alternates")
        elif len(boundaries) == 1:
            primary_boundary = boundaries[0]
        else:
            # No boundaries, skip this face
            continue
        
        # Create one face with the primary boundary
        face_data = {
            'normal': face_eq['normal'],
            'd': face_eq['d'],
            'vertices': primary_boundary['vertices'],
            'holes': holes_list,
            'all_vertices_on_face': face_eq.get('vertices_on_face', []),
            'edges': face_eq.get('edges_on_face', [])
        }
        
        # If there are alternates, store them
        if alternates_list:
            face_data['alternates'] = alternates_list
        
        faces.append(face_data)
    
    print("\n" + "="*70)
    print(f"[POLY FORM] EXTRACTION COMPLETE: {len(faces)} faces found")
    print("="*70)
    
    # Summary statistics
    total_faces_with_holes = sum(1 for f in faces if len(f['holes']) > 0)
    total_holes = sum(len(f['holes']) for f in faces)
    total_faces_with_alternates = sum(1 for f in faces if f.get('alternates'))
    
    print(f"[POLY FORM] Summary:")
    print(f"[POLY FORM]   - Total faces: {len(faces)}")
    print(f"[POLY FORM]   - Faces with holes: {total_faces_with_holes}")
    print(f"[POLY FORM]   - Total holes: {total_holes}")
    print(f"[POLY FORM]   - Faces with alternates: {total_faces_with_alternates}")
    
    for idx, face in enumerate(faces):
        vertices_count = len(face['vertices'])
        holes_count = len(face['holes'])
        alternates_count = len(face.get('alternates', []))
        
        status_str = f"{vertices_count} vertices, {holes_count} hole(s)"
        if alternates_count > 0:
            status_str += f", {alternates_count} alternate(s)"
        print(f"[POLY FORM]   Face {idx+1}: {status_str}")
        
        # Always print the primary boundary vertices
        if alternates_count > 0:
            print(f"[POLY FORM]     Primary boundary: {face['vertices']}")
            for alt_idx, alt in enumerate(face['alternates']):
                print(f"[POLY FORM]     Alternate {alt_idx+1}: {alt['vertices']}")
        elif holes_count > 0:
            print(f"[POLY FORM]     Outer boundary vertices: {face['vertices']}")
        else:
            # Simple face with no alternates or holes - still print vertices
            print(f"[POLY FORM]     Vertices: {face['vertices']}")
        
        # Print detailed hole information if there are holes
        if holes_count > 0:
            for hole_idx, hole in enumerate(face['holes']):
                print(f"[POLY FORM]     Hole {hole_idx+1}: {len(hole)} vertices")
                print(f"[POLY FORM]       Vertices: {hole}")
                # Calculate hole area in 2D projection
                hole_verts_3d = [selected_vertices[v_idx] for v_idx in hole]
                hole_verts_array = np.array(hole_verts_3d)
                if len(hole_verts_array) >= 3:
                    # Project to 2D using face normal
                    normal = np.array(face['normal'])
                    normal = normal / np.linalg.norm(normal)
                    
                    # Simple projection by dropping smallest normal component
                    abs_normal = np.abs(normal)
                    drop_axis = np.argmax(abs_normal)
                    keep_axes = [i for i in range(3) if i != drop_axis]
                    
                    hole_2d = hole_verts_array[:, keep_axes]
                    
                    # Calculate area using shoelace formula
                    x = hole_2d[:, 0]
                    y = hole_2d[:, 1]
                    area = 0.5 * np.abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))
                    print(f"[POLY FORM]       Hole area (2D projection): {area:.6f}")
                    
                    # Calculate hole center
                    center = np.mean(hole_verts_array, axis=0)
                    print(f"[POLY FORM]       Hole center (3D): [{center[0]:.2f}, {center[1]:.2f}, {center[2]:.2f}]")
    
    # =========================================================================
    # Step 7.5: Merge touching holes within each face
    # =========================================================================
    print("\n[POLY FORM] Step 7.5: Checking for touching holes within faces")
    print("-" * 70)
    
    def find_shared_edge(hole1, hole2):
        """Find shared edge between two holes, return (v1, v2) or None"""
        # Check all edges in hole1 against all edges in hole2
        for i in range(len(hole1)):
            v1 = hole1[i]
            v2 = hole1[(i + 1) % len(hole1)]
            edge1 = (min(v1, v2), max(v1, v2))
            
            for j in range(len(hole2)):
                u1 = hole2[j]
                u2 = hole2[(j + 1) % len(hole2)]
                edge2 = (min(u1, u2), max(u1, u2))
                
                if edge1 == edge2:
                    return (v1, v2)  # Return original order from hole1
        return None
    
    def merge_holes_by_removing_shared_edge(hole1, hole2, shared_edge):
        """
        Merge two holes by removing their shared edge.
        
        For example:
        hole1 = [79, 169, 189, 32, 62, 150, 135, 93, 45, 11, 183, 9]
        hole2 = [38, 3, 5, 135, 150, 30]
        shared_edge = (150, 135)
        
        Result: [79, 169, 189, 32, 62, 150, 30, 38, 3, 5, 135, 93,
                 45, 11, 183, 9]
        
        The edge 150→135 is removed:
        - From hole1: take vertices up to 150, then after 135
        - From hole2: take path from 150 to 135 (excluding the edge)
        """
        v1, v2 = shared_edge
        
        # Find v1 and v2 in both holes
        if v1 not in hole1 or v2 not in hole1:
            return None
        if v1 not in hole2 or v2 not in hole2:
            return None
        
        idx1_v1 = hole1.index(v1)
        idx1_v2 = hole1.index(v2)
        idx2_v1 = hole2.index(v1)
        idx2_v2 = hole2.index(v2)
        
        # Verify edges are consecutive in both holes
        if (idx1_v2 - idx1_v1) % len(hole1) != 1 and \
           (idx1_v1 - idx1_v2) % len(hole1) != 1:
            return None
        
        if (idx2_v2 - idx2_v1) % len(hole2) != 1 and \
           (idx2_v1 - idx2_v2) % len(hole2) != 1:
            return None
        
        # Extract segment from hole1 WITHOUT the edge v1->v2
        if (idx1_v2 - idx1_v1) % len(hole1) == 1:
            # v1 -> v2 in hole1, take from v2+1 to v1 (exclusive)
            hole1_segment = []
            i = (idx1_v2 + 1) % len(hole1)
            while i != idx1_v1:
                hole1_segment.append(hole1[i])
                i = (i + 1) % len(hole1)
            start_vertex = v1
            end_vertex = v2
        else:
            # v2 -> v1 in hole1, take from v1+1 to v2 (exclusive)
            hole1_segment = []
            i = (idx1_v1 + 1) % len(hole1)
            while i != idx1_v2:
                hole1_segment.append(hole1[i])
                i = (i + 1) % len(hole1)
            start_vertex = v2
            end_vertex = v1
        
        # Extract segment from hole2 WITHOUT the edge
        if (idx2_v2 - idx2_v1) % len(hole2) == 1:
            # v1 -> v2 in hole2
            hole2_segment = []
            i = (idx2_v2 + 1) % len(hole2)
            while i != idx2_v1:
                hole2_segment.append(hole2[i])
                i = (i + 1) % len(hole2)
            # Need to reverse to connect properly
            hole2_segment = hole2_segment[::-1]
        else:
            # v2 -> v1 in hole2
            hole2_segment = []
            i = (idx2_v1 + 1) % len(hole2)
            while i != idx2_v2:
                hole2_segment.append(hole2[i])
                i = (i + 1) % len(hole2)
        
        # Merge: hole1_part + [start] + hole2_part + [end]
        merged = hole1_segment + [start_vertex] + \
                 hole2_segment + [end_vertex]
        
        return merged if len(merged) >= 3 else None
    
    faces_with_merged_holes = 0
    total_holes_merged = 0
    edges_to_remove = set()  # Track edges removed during merging
    
    for face_idx, face in enumerate(faces):
        holes = face.get('holes', [])
        if len(holes) < 2:
            continue  # Need at least 2 holes to merge
        
        # Build 2D projection for the face
        normal = np.array(face['normal'])
        normal = normal / np.linalg.norm(normal)
        abs_normal = np.abs(normal)
        drop_axis = np.argmax(abs_normal)
        keep_axes = [i for i in range(3) if i != drop_axis]
        
        # Project holes to 2D for visualization
        holes_2d = []
        for hole in holes:
            hole_verts_3d = np.array([selected_vertices[v_idx] 
                                      for v_idx in hole])
            hole_2d = hole_verts_3d[:, keep_axes]
            holes_2d.append(hole_2d)
        
        # Iteratively merge touching holes
        merged_any = True
        iteration = 0
        max_iterations = len(holes) * 2  # Prevent infinite loops
        
        while merged_any and iteration < max_iterations:
            merged_any = False
            iteration += 1
            
            # Check all pairs of holes for shared edges
            for i in range(len(holes)):
                if merged_any:
                    break  # Restart after a merge
                
                for j in range(i + 1, len(holes)):
                    shared_edge = find_shared_edge(holes[i], holes[j])
                    
                    if shared_edge is not None:
                        print(f"[POLY FORM]   Face {face_idx+1}: "
                              f"Holes {i+1} and {j+1} share edge "
                              f"{shared_edge}")
                        
                        # Track this edge for face deletion
                        v1, v2 = shared_edge
                        edge_key = (min(v1, v2), max(v1, v2))
                        edges_to_remove.add(edge_key)
                        
                        # Merge the two holes by removing the shared edge
                        merged_hole = merge_holes_by_removing_shared_edge(
                            holes[i], holes[j], shared_edge)
                        
                        if merged_hole is not None and len(merged_hole) >= 3:
                            # Remove duplicate consecutive vertices
                            deduplicated = []
                            for k, v_idx in enumerate(merged_hole):
                                # Only add if different from previous
                                if k == 0 or v_idx != merged_hole[k-1]:
                                    deduplicated.append(v_idx)
                            
                            # Check if last equals first (polygon closure)
                            if len(deduplicated) > 0 and \
                               deduplicated[-1] == deduplicated[0]:
                                deduplicated = deduplicated[:-1]
                            
                            if len(deduplicated) >= 3:
                                merged_hole = deduplicated
                                print(f"[POLY FORM]     → Merged into "
                                      f"single hole with "
                                      f"{len(merged_hole)} vertices")
                            else:
                                print(f"[POLY FORM]     ✗ Merge failed: "
                                      f"too few vertices after "
                                      f"deduplication")
                                merged_hole = None
                            
                            # Replace hole i with merged, remove hole j
                            new_holes = []
                            for k in range(len(holes)):
                                if k == i:
                                    new_holes.append(merged_hole)
                                elif k != j:
                                    new_holes.append(holes[k])
                            
                            holes = new_holes
                            merged_any = True
                            total_holes_merged += 1
                            break
                        else:
                            print(f"[POLY FORM]     ✗ Merge failed")
            
            if merged_any:
                # Update face holes
                face['holes'] = holes
        
        if iteration > 1:
            faces_with_merged_holes += 1
            print(f"[POLY FORM]   Face {face_idx+1}: Completed hole merging in {iteration-1} iteration(s)")
            print(f"[POLY FORM]     Final: {len(holes)} hole(s)")
    
    if total_holes_merged > 0:
        print(f"\n[POLY FORM] Hole merging summary:")
        print(f"[POLY FORM]   - Faces with merged holes: "
              f"{faces_with_merged_holes}")
        print(f"[POLY FORM]   - Total hole merges: {total_holes_merged}")
        print(f"[POLY FORM]   - Edges removed: {len(edges_to_remove)}")
        
        # Step 7.6: Delete faces containing removed edges
        print(f"\n[POLY FORM] Step 7.6: Deleting faces with removed edges")
        print("-" * 70)
        
        faces_to_delete = []
        for face_idx, face in enumerate(faces):
            outer_boundary = face['vertices']  # Use 'vertices', not 'outer_boundary'
            # Check all edges in this face
            for i in range(len(outer_boundary)):
                v1 = outer_boundary[i]
                v2 = outer_boundary[(i + 1) % len(outer_boundary)]
                edge_key = (min(v1, v2), max(v1, v2))
                
                if edge_key in edges_to_remove:
                    faces_to_delete.append(face_idx)
                    print(f"[POLY FORM]   Marking Face {face_idx+1} for "
                          f"deletion (contains edge {edge_key})")
                    break  # No need to check other edges
        
        # Delete faces in reverse order to maintain indices
        faces_deleted = 0
        for face_idx in sorted(faces_to_delete, reverse=True):
            del faces[face_idx]
            faces_deleted += 1
        
        print(f"[POLY FORM]   Deleted {faces_deleted} faces")
        
        # Step 7.7: Remove duplicate faces
        print(f"\n[POLY FORM] Step 7.7: Removing duplicate faces")
        print("-" * 70)
        
        # Create signature for each face (sorted vertex tuple)
        face_signatures = {}
        duplicates_to_delete = []
        
        for face_idx, face in enumerate(faces):
            outer = face['vertices']  # Use 'vertices' key
            # Create canonical signature: sorted tuple of vertices
            sig = tuple(sorted(outer))
            
            if sig in face_signatures:
                # Duplicate found
                original_idx = face_signatures[sig]
                print(f"[POLY FORM]   Face {face_idx+1} is duplicate of "
                      f"Face {original_idx+1}")
                print(f"[POLY FORM]     Vertices: {outer}")
                duplicates_to_delete.append(face_idx)
            else:
                face_signatures[sig] = face_idx
        
        # Delete duplicates in reverse order
        duplicates_deleted = 0
        for face_idx in sorted(duplicates_to_delete, reverse=True):
            del faces[face_idx]
            duplicates_deleted += 1
        
        print(f"[POLY FORM]   Deleted {duplicates_deleted} duplicate faces")
        print(f"[POLY FORM]   Remaining faces: {len(faces)}")
    else:
        print(f"[POLY FORM]   No touching holes found")
    
    return faces


def plot_extracted_polygon_faces(extracted_faces, selected_vertices,
                                  original_faces,
                                  units="cm",
                                  drawing_scale_real=1.0,
                                  drawing_scale_drawing=1.0):
    """
    Plot extracted polygon faces with controls to toggle visibility.
    Unified view showing both original solid faces and extracted polygons.
    
    Parameters:
        extracted_faces: List of face dicts from extract_polygon_faces
        selected_vertices: Nx3 array of vertex coordinates
        original_faces: Original face polygons for comparison
        units: Drawing units (mm, cm, m, inches, feet)
        drawing_scale_real: Real-world scale (numerator)
        drawing_scale_drawing: Drawing scale (denominator)
    """
    from matplotlib.widgets import CheckButtons
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    
    # Create single unified plot
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Add title with scale and unit information
    scale_str = f"1:{drawing_scale_drawing}" if drawing_scale_drawing != 1.0 else "1:1"
    ax.set_title(f'Original Solid Faces & Extracted Polygon Faces\n'
                 f'Units: {units} | Scale: {scale_str}',
                 fontsize=14, fontweight='bold')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    
    # Plot original faces (semi-transparent)
    colors_orig = plt.cm.rainbow(np.linspace(0, 1, len(original_faces)))
    original_face_collections = []
    
    for idx, face_data in enumerate(original_faces):
        if 'vertices' in face_data:
            verts = face_data['vertices']
            if isinstance(verts, np.ndarray) and verts.ndim == 2:
                poly = Poly3DCollection([verts], alpha=0.2, 
                                       facecolor=colors_orig[idx],
                                       edgecolor='gray', linewidth=1.0)
                ax.add_collection3d(poly)
                original_face_collections.append(poly)
        elif 'outer_boundary' in face_data:
            # Handle face_data from extract_and_visualize_faces
            verts = np.array(face_data['outer_boundary'])
            if verts.ndim == 2 and verts.shape[1] == 3:
                poly = Poly3DCollection([verts], alpha=0.2,
                                       facecolor=colors_orig[idx],
                                       edgecolor='gray', linewidth=1.0)
                ax.add_collection3d(poly)
                original_face_collections.append(poly)
    
    # Plot extracted faces (more prominent)
    colors_ext = plt.cm.viridis(np.linspace(0, 1, len(extracted_faces)))
    extracted_face_collections = []
    
    for idx, face in enumerate(extracted_faces):
        vertices_idx = face['vertices']
        face_verts = selected_vertices[vertices_idx]
        
        poly = Poly3DCollection([face_verts], alpha=0.7,
                               facecolor=colors_ext[idx],
                               edgecolor='black', linewidth=2.0)
        ax.add_collection3d(poly)
        extracted_face_collections.append(poly)
        
        # Plot holes if any
        for hole_idx in face.get('holes', []):
            hole_verts = selected_vertices[hole_idx]
            hole_poly = Poly3DCollection([hole_verts], alpha=0.5,
                                        facecolor='white',
                                        edgecolor='red', linewidth=2.0)
            ax.add_collection3d(hole_poly)
    
    # Plot vertices
    vertex_scatter = ax.scatter(selected_vertices[:, 0], selected_vertices[:, 1], 
                                selected_vertices[:, 2], c='red', s=50, marker='o',
                                label='Vertices')
    
    # Add vertex labels
    for i, vertex in enumerate(selected_vertices):
        # Label shows the vertex index that matches polygon output
        ax.text(vertex[0]+3.0, vertex[1]+3.0, vertex[2]+3.0, f'v{i}', 
                fontsize=8, color='blue', fontweight='bold')
    
    # Set equal aspect ratio
    max_range = np.array([
        selected_vertices[:, 0].max() - selected_vertices[:, 0].min(),
        selected_vertices[:, 1].max() - selected_vertices[:, 1].min(),
        selected_vertices[:, 2].max() - selected_vertices[:, 2].min()
    ]).max() / 2.0
    
    mid_x = (selected_vertices[:, 0].max() + 
            selected_vertices[:, 0].min()) * 0.5
    mid_y = (selected_vertices[:, 1].max() + 
            selected_vertices[:, 1].min()) * 0.5
    mid_z = (selected_vertices[:, 2].max() + 
            selected_vertices[:, 2].min()) * 0.5
    
    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)
    
    # Enable mouse rotation
    ax.mouse_init()
    
    # Add simplified checkboxes with only 3 buttons
    checkbox_ax = plt.axes([0.02, 0.4, 0.20, 0.15])
    
    # Three simple toggle buttons
    labels = ['Vertices', 'Original Solid', 
              f'Extracted Faces ({len(extracted_faces)})']
    visibility = [True, True, True]
    check = CheckButtons(checkbox_ax, labels, visibility)
    
    def toggle_element(label):
        if label == 'Vertices':
            vertex_scatter.set_visible(not vertex_scatter.get_visible())
        elif label == 'Original Solid':
            for poly in original_face_collections:
                poly.set_visible(not poly.get_visible())
        elif 'Extracted Faces' in label:
            # Toggle all extracted faces together
            for poly in extracted_face_collections:
                poly.set_visible(not poly.get_visible())
        fig.canvas.draw_idle()
    
    check.on_clicked(toggle_element)
    
    plt.tight_layout()
    print("[DEBUG] plot_extracted_polygon_faces: About to call plt.show(block=True)...")
    plt.show(block=True)
    print("[DEBUG] plot_extracted_polygon_faces: Returned from plt.show(block=True)")
    plt.close(fig)
    print("[DEBUG] plot_extracted_polygon_faces: Closed figure, returning from function")


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

    # Generate random units and scale for engineering drawing
    np.random.seed(seed)  # Use same seed for reproducibility
    
    # Random unit selection
    unit_options = ["mm", "cm", "m", "inches", "feet"]
    units = np.random.choice(unit_options)
    
    # Random scale selection (1:X format where X is the drawing scale)
    scale_options = [0.1, 0.25, 0.5, 0.75, 1, 2, 3, 4, 5, 6, 7, 8, 9, 
                     10, 25, 50]
    drawing_scale_drawing = np.random.choice(scale_options)
    drawing_scale_real = 1.0  # D_real: actual size (always 1 for 1:X format)
    
    print(f"\n[DRAWING PARAMETERS]")
    print(f"  Units: {units}")
    print(f"  Scale: 1:{drawing_scale_drawing}")

    # Pass arrays and ordered vertices to plot_four_views
    # old view_connectivity_matrices = plot_four_views(solid, projection_normal,
    view_connectivity_matrices = plot_four_views(face_polygons, projection_normal,
                   all_vertices_sorted,
                   Vertex_Top_View,
                   Vertex_Front_View,
                   Vertex_Side_View,
                   Vertex_Iso_View,
                   pdf_dir,
                   units,
                   drawing_scale_real,
                   drawing_scale_drawing)

    # === Post-processing: Find z-levels from Front_View and build Possible_Vertices ===
    # This must come after the arrays are created and filled (after plot_four_views)
    print("\n[DEBUG] Using new connectivity matrices for vertex filtering...")

    # Use the new connectivity matrices instead of make_summary_array
    front_view_matrix = view_connectivity_matrices.get('Front View')
    top_view_matrix = view_connectivity_matrices.get('Top View')
    side_view_matrix = view_connectivity_matrices.get('Side View')
    
    # Save connectivity matrices in .npz format for Reconstruct_Solid.py
    
    if (front_view_matrix is not None and top_view_matrix is not None 
        and side_view_matrix is not None):
        npz_filename = os.path.join(output_dir, 
                                    f'connectivity_matrices_seed_{seed}.npz')
        # Convert numpy types to Python native types for proper saving/loading
        np.savez(npz_filename,
                 all_vertices=all_vertices_sorted,
                 top_view_matrix=top_view_matrix,
                 front_view_matrix=front_view_matrix,
                 side_view_matrix=side_view_matrix,
                 units=str(units),  # Convert to Python string
                 drawing_scale_real=float(drawing_scale_real),
                 drawing_scale_drawing=float(drawing_scale_drawing))
        print(f"\n[SAVE] Saved connectivity matrices to: {npz_filename}")
        print(f"       Units: {units}")
        print(f"       Drawing scale: {drawing_scale_real}:{drawing_scale_drawing}")
    
    # Save face polygons for comparison
    if face_polygons is not None and len(face_polygons) > 0:
        face_filename = os.path.join(output_dir, f'solid_faces_seed_{seed}.npy')
        np.save(face_filename, face_polygons)
        print(f"[SAVE] Saved {len(face_polygons)} face polygons to: {face_filename}")

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
        
        if side_view_matrix is None:
            print("[ERROR] Could not create side view matrix")
            selected_vertices = np.array([])
        else:
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
                
                # === Extract polygon faces using new algorithm ===
                print("\n[DEBUG] Extracting polygon faces from connectivity matrix...")
                extracted_faces = extract_polygon_faces_from_connectivity(
                    step3_vertices, merged_conn, tolerance=0.1005
                )
                
                # Visualize extracted faces
                if len(extracted_faces) > 0:
                    plot_extracted_polygon_faces(
                        extracted_faces, step3_vertices, face_polygons
                    )
        
    else:
        print("[ERROR] Could not create summary arrays for vertex filtering")
        # Still show the solid even if filtering failed
        # old visualize_3d_solid(solid, None)
        visualize_3d_solid(face_polygons, None)
    #extract_possible_vertices_from_summaries(Vertex_Front_View, Vertex_Top_View, all_vertices_sorted)







if __name__ == "__main__":    main()


