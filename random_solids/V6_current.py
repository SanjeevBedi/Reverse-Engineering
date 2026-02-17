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

# Global variable for face visualization tracking
FACE_VIZ_DATA = {}
FACE_VIZ_ENABLED = False
FACE_VIZ_TARGET = None

# Ensure build_merged_connectivity_matrix is defined
def build_merged_connectivity_matrix(reconstructed_vertices, top_proj, front_proj, side_proj, top_conn, front_conn, side_conn):
    """
    Build merged connectivity matrix from three views.
    For each pair of reconstructed vertices, count in how many views the edge exists.
    Robustly matches reconstructed vertices to projections and view connectivity matrices.
    Also checks for edges perpendicular to a view that project to a point.
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
    # Note: edge matrices use value=2 for dashed lines, so check >0 not value itself
    for i in range(N):
        for j in range(i+1, N):
            conn_count = 0
            in_top = False
            in_front = False
            in_side = False
            
            # Top view - count if edge exists (>0) not the value itself
            idx_top_i = find_proj_idx(top_proj, top_proj[i])
            idx_top_j = find_proj_idx(top_proj, top_proj[j])
            if idx_top_i is not None and idx_top_j is not None and idx_top_i < top_conn.shape[0] and idx_top_j < top_conn.shape[1]:
                if top_conn[idx_top_i, idx_top_j] > 0:
                    conn_count += 1
                    in_top = True
            # Front view - count if edge exists (>0) not the value itself
            idx_front_i = find_proj_idx(front_proj, front_proj[i])
            idx_front_j = find_proj_idx(front_proj, front_proj[j])
            if idx_front_i is not None and idx_front_j is not None and idx_front_i < front_conn.shape[0] and idx_front_j < front_conn.shape[1]:
                if front_conn[idx_front_i, idx_front_j] > 0:
                    conn_count += 1
                    in_front = True
            # Side view - count if edge exists (>0) not the value itself
            idx_side_i = find_proj_idx(side_proj, side_proj[i])
            idx_side_j = find_proj_idx(side_proj, side_proj[j])
            if idx_side_i is not None and idx_side_j is not None and idx_side_i < side_conn.shape[0] and idx_side_j < side_conn.shape[1]:
                if side_conn[idx_side_i, idx_side_j] > 0:
                    conn_count += 1
                    in_side = True
            
            # Check for perpendicular edges projecting to points
            # If edge projects as point in third view, elevate conn=2 to conn=3
            if conn_count == 2:
                v1, v2 = reconstructed_vertices[i], reconstructed_vertices[j]
                dx = abs(v2[0] - v1[0])
                dy = abs(v2[1] - v1[1])
                dz = abs(v2[2] - v1[2])
                
                # Perpendicular to top view (vertical edge) - projects as point
                if dx < 1e-6 and dy < 1e-6:
                    conn_count += 1
                # Perpendicular to front view (parallel to X) - projects as point
                elif dy < 1e-6 and dz < 1e-6:
                    conn_count += 1
                # Perpendicular to side view (parallel to Y) - projects as point
                elif dx < 1e-6 and dz < 1e-6:
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
from Reconstruction.polygon_classifier import classify_polygons_by_visibility
from opencascade import get_face_normal_from_opencascade, extract_and_visualize_faces, extract_wire_vertices_in_sequence, OPENCASCADE_AVAILABLE
# from unified_summary import (create_unified_summary, print_summary_info,
#                              save_summary_to_file, save_summary_to_numpy,
#                              visualize_adjacency_matrix)
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

def build_solid_with_polygons_test(config, quiet=False, no_lettering=False):
    from Reconstruction.Base_Solid import build_solid_with_polygons
    seed = config.seed
    print(f"[DEBUG] Calling build_solid_with_polygons(config, seed={seed}, quiet={quiet}) as test...")
    original = build_solid_with_polygons(seed, quiet, no_lettering)
    cut_shape2 = original
    # cut_shape1 = original
    #(You can add your custom boolean operations here if needed)
    #return original
    #Create box at (0,0,0) of size (60,50,60)
 

    # # Create box at (0,0,0) of size (60,50,60)
    # box = BRepPrimAPI_MakeBox(56.0, 39.0, 36.0).Shape()
    # # Move box to (10,25,0)
    # trsf = gp_Trsf()
    # trsf.SetTranslation(gp_Vec(-5.50, -2.00, -1.00))
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
                    edge = topods.Edge(wire_explorer.Current())
                    
                    # Get both vertices of the edge to properly chain them
                    from OCC.Core.TopExp import topexp
                    vertex1 = topexp.FirstVertex(edge, True)  # True = consider orientation
                    vertex2 = topexp.LastVertex(edge, True)
                    
                    pnt1 = BRep_Tool.Pnt(vertex1)
                    pnt2 = BRep_Tool.Pnt(vertex2)
                    
                    v1_coords = [pnt1.X(), pnt1.Y(), pnt1.Z()]
                    v2_coords = [pnt2.X(), pnt2.Y(), pnt2.Z()]
                    
                    # Chain the vertices: add first vertex of first edge, then subsequent end vertices
                    if edge_count == 0:
                        vertex_sequence.append(v1_coords)
                        vertex_sequence.append(v2_coords)
                    else:
                        # Check if the new edge connects to the last vertex
                        last_v = vertex_sequence[-1]
                        if np.linalg.norm(np.array(v1_coords) - np.array(last_v)) < 1e-6:
                            vertex_sequence.append(v2_coords)
                        elif np.linalg.norm(np.array(v2_coords) - np.array(last_v)) < 1e-6:
                            vertex_sequence.append(v1_coords)
                        else:
                            # Edge doesn't connect - might be a gap or different ordering
                            print(f"              WARNING: Edge {edge_count} doesn't connect to previous edge")
                            vertex_sequence.append(v2_coords)
                    
                    print(f"              Edge {edge_count}: ({v1_coords[0]:.1f},{v1_coords[1]:.1f},{v1_coords[2]:.1f}) → "
                          f"({v2_coords[0]:.1f},{v2_coords[1]:.1f},{v2_coords[2]:.1f})")
                    
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


def extract_and_visualize_faces(solid, visualize=False, elev=25, azim=45):
    """
    Extract face data from an OpenCASCADE solid and optionally visualize in 3D.
    Returns a list of face data dicts. If visualize=True, also plots the solid.
    
    Args:
        solid: OpenCASCADE solid shape
        visualize: Whether to create 3D visualization
        elev: Elevation angle in degrees for 3D view (default: 25)
        azim: Azimuth angle in degrees for 3D view (default: 45)
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
            ax.view_init(elev=elev, azim=azim)
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
def visualize_3d_solid(face_polygons, selected_vertices=None, edges=None, edges_with_class=None, seed=None, pdf_dir="PDFfiles", elev=30, azim=-60):
    """
    Display the 3D solid using matplotlib 3D plotting.
    Optionally highlight selected vertices and edges with color-coding.
    
    Args:
        solid_shape: The solid shape to visualize
        selected_vertices: Array of selected 3D vertices
        edges: List of edge tuples (v1_idx, v2_idx) - for backward compat
        edges_with_class: List of tuples (v1_idx, v2_idx, classification)
                         Classifications: 1=yellow, 2=gray, 3=black
        seed: Seed number for filename (optional)
        pdf_dir: Directory to save PDF file (default: "PDFfiles")
        elev: Elevation angle in degrees (default: 30)
        azim: Azimuth angle in degrees (default: -60)
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
            
            # Add vertex number labels offset from vertices
            # Calculate overall data range for offset scaling
            data_range = np.ptp(selected_vertices, axis=0).max()
            offset_distance = 0.06 * data_range  # 6% of data range
            
            # Calculate center of all vertices
            center = np.mean(selected_vertices, axis=0)
            
            for v_idx, vertex in enumerate(selected_vertices):
                # Calculate offset direction (radial from center)
                offset_dir = vertex - center
                offset_norm = np.linalg.norm(offset_dir)
                if offset_norm > 1e-6:
                    offset_dir = offset_dir / offset_norm
                else:
                    offset_dir = np.array([0, 0, 1])  # Default offset if vertex is at center
                
                label_pos = vertex + offset_dir * offset_distance
                ax.text(label_pos[0], label_pos[1], label_pos[2], str(v_idx), 
                       fontsize=7, color='darkblue', ha='center', va='center', 
                       fontweight='bold', bbox=dict(boxstyle='circle,pad=0.1', 
                       facecolor='white', edgecolor='darkblue', alpha=0.8))

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

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('3D Solid Reconstruction: Original + Extracted Polygons')
    
    # Set view angle
    ax.view_init(elev=elev, azim=azim)
    
    # Save clean PDF for LaTeX (before adding interactive elements)
    if seed is not None:
        import os
        os.makedirs(pdf_dir, exist_ok=True)
        
        # Set default visibility for LaTeX version: show only conn=3 edges and vertices
        for h in face_handles:
            h.set_visible(False)  # Hide polygons for cleaner view
        for h in edge_handles_1:
            h.set_visible(False)  # Hide conn=1 edges
        for h in edge_handles_2:
            h.set_visible(False)  # Hide conn=2 edges
        # edge_handles_3 and vertex_handle remain visible (True by default)
        
        # Save clean PDF
        pdf_filename = os.path.join(pdf_dir, f"3d_solid_reconstruction_seed_{seed}.pdf")
        plt.savefig(pdf_filename, format='pdf', bbox_inches='tight', dpi=300)
        print(f"\n[SAVE] Clean PDF saved to: {pdf_filename}")
        print(f"       Format: PDF (vector graphics, ideal for LaTeX)")
        print(f"       Showing: Vertices + conn=3 edges only")
        
        # Also save a version with all edges
        for h in edge_handles_1:
            h.set_visible(True)
        for h in edge_handles_2:
            h.set_visible(True)
        pdf_filename_all = os.path.join(pdf_dir, f"3d_solid_reconstruction_all_edges_seed_{seed}.pdf")
        plt.savefig(pdf_filename_all, format='pdf', bbox_inches='tight', dpi=300)
        print(f"[SAVE] All edges PDF saved to: {pdf_filename_all}")
        
        # Reset visibility for interactive plot
        for h in face_handles:
            h.set_visible(True)
        
        print(f"\n[LaTeX] Include in your document with:")
        print(f"\\begin{{figure}}[htbp]")
        print(f"    \\centering")
        print(f"    \\includegraphics[width=0.8\\textwidth]{{{os.path.basename(pdf_filename)}}}")
        print(f"    \\caption{{3D solid reconstruction showing vertices and conn=3 edges (red) for seed {seed}. ")
        print(f"             All conn=3 edges are visible in the reconstruction, including edges that project ")
        print(f"             as points in one of the orthogonal views.}}")
        print(f"    \\label{{fig:3d_solid_conn3_seed_{seed}}}")
        print(f"\\end{{figure}}")
    
    # Add interactive check buttons for exploration
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
        except Exception as e:
            print(f"Face F{face_id}: Projection error - {e}")
    
    print(f"\nStep 2: Starting historic polygon classification algorithm...")
    print(f"Initial array_A: {len(array_A_initial)} polygons")


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
                            # CRITICAL FIX: Preserve parent_face from Pi (primary face)
                            # The intersection region exists in both faces, but we use Pi's geometry
                            'parent_face': Pi_parent_face,
                            # Also track Pj's parent face for potential edge extraction
                            'parent_face_secondary': Pj_parent_face,
                            # FIXED: Use original face name
                            'associated_face': Pi_name,
                            'original_index': -1,
                            'dot_product': 0
                        }
                        array_C.append(intersection_data)

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
            else:
                # DEBUG: Track faces that are filtered out due to small area
                if Pi_data.get('debug_edges'):
                    print(f"[DEBUG FILTER] {Pi_data['name']} {Pi_data.get('debug_edges')} NOT added to array_B: area={Pi_data['polygon'].area:.2e}, dot={Pi_data['dot_product']:.3f}")
        
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
    
    # Build a set of visible polygon data for quick lookup
    # visible and hidden are arrays of dict objects with 'polygon' key
    visible_set = set(id(poly_data) for poly_data in visible)
    
    print(f"[DEBUG] {view_name}: visible list has {len(visible)} polygons, visible_set has {len(visible_set)} IDs")
    print(f"[DEBUG] {view_name}: hidden list has {len(hidden)} polygons")
    
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
    degenerate_edges_skipped = 0
    
    poly_count = 0
    for poly_data in all_polygons:
        parent_face = poly_data.get('parent_face', None)
        parent_face_secondary = poly_data.get('parent_face_secondary', None)
        poly_name = poly_data.get('name', f'Poly_{poly_count}')
        
        # Process both primary and secondary parent faces (for intersection polygons)
        parent_faces_to_process = []
        if parent_face is not None:
            parent_faces_to_process.append(('primary', parent_face))
        if parent_face_secondary is not None:
            parent_faces_to_process.append(('secondary', parent_face_secondary))
        
        if not parent_faces_to_process:
            continue
        
        poly_count += 1
        
        for face_type, parent_face_data in parent_faces_to_process:
            if hasattr(parent_face_data, 'tolist'):
                vertices_3d = parent_face_data.tolist()
            else:
                vertices_3d = parent_face_data
                
            if not isinstance(vertices_3d, (list, tuple)) or len(vertices_3d) < 3:
                continue
        
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
                
                # Add edge to connectivity matrix (OUTSIDE the vertex search loop)
                if v1_idx is not None and v2_idx is not None:
                    
                    # Check if edge is degenerate (projects to a point)
                    if v1_idx == v2_idx:
                        # Degenerate edge - both vertices project to same 2D point
                        # This is normal for edges perpendicular to view (e.g., vertical edges in Top view)
                        # Skip adding to this view's connectivity matrix
                        degenerate_edges_skipped += 1
                    else:
                        # Mark edge connectivity as 2.0 (edge exists)
                        # Note: We use 2.0 as the standard value for edge existence
                        # The neural network and reconstruction code expect conn > 1.5 for edges
                        
                        # Add connection in both directions (symmetric matrix)
                        result_matrix[v1_idx, 3 + v2_idx] = 2.0
                        result_matrix[v2_idx, 3 + v1_idx] = 2.0
                        edges_found += 1
    
    print(f"{view_name}: Added {edges_found} edges to connectivity matrix")
    print(f"{view_name}: Skipped {degenerate_edges_skipped} degenerate edges (project to points)")
    print(f"{view_name}: Matrix shape: {result_matrix.shape}")
    
    # Check for asymmetric connectivity matrix entries
    n_vertices = result_matrix.shape[0]
    if result_matrix.shape[1] > 3:
        conn = result_matrix[:, 3:]
        asymmetries_found = []
        for i in range(n_vertices):
            for j in range(i+1, n_vertices):  # Only check upper triangle
                if conn[i, j] != conn[j, i]:
                    asymmetries_found.append((i, j, conn[i, j], conn[j, i]))
        
        if asymmetries_found:
            print(f"\n[WARNING] {view_name}: Found {len(asymmetries_found)} asymmetric edges:")
            for i, j, val_ij, val_ji in asymmetries_found[:10]:  # Show first 10
                print(f"  Vertex {i} -> {j}: {val_ij}, but Vertex {j} -> {i}: {val_ji}")
            
            # Fix asymmetries by taking the maximum value
            print(f"[FIXING] Enforcing symmetry by setting both directions to max value...")
            for i in range(n_vertices):
                for j in range(n_vertices):
                    if conn[i, j] > 0 or conn[j, i] > 0:
                        conn[i, j] = conn[j, i] = max(conn[i, j], conn[j, i])
            result_matrix[:, 3:] = conn
            print(f"[FIXED] Symmetry enforced for {view_name}")
        else:
            print(f"[OK] {view_name}: Connectivity matrix is symmetric")

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
    # Generate seed hash from ordered_vertices if available, otherwise use provided seed
    if ordered_vertices and len(ordered_vertices) > 0:
        seed_hash = hash(str(ordered_vertices[0])) % 100000
    elif seed is not None:
        seed_hash = seed
    else:
        seed_hash = 0
    
    fig.suptitle(f'Engineering Drawing Views - Seed {seed_hash}\n'
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


def print_face_summary_debug(unique_faces, title="Face Summary"):
    """
    Print summary of faces with vertices, holes, and alternates.
    
    Args:
        unique_faces: List of face dictionaries with 'polygons', 'normal', etc.
        title: Title string for the summary section
    """
    import numpy as np
    
    # Count faces with holes and alternates
    total_faces_with_holes = 0
    total_holes = 0
    total_faces_with_alternates = 0
    
    for face_eq in unique_faces:
        polygons = face_eq.get('polygons', [])
        if len(polygons) == 0:
            continue
        
        # Count holes and alternates
        holes_count = sum(1 for p in polygons if p.get('polygon_type') == 'HOLE' and not p.get('removed', False))
        alts_count = sum(1 for p in polygons if p.get('polygon_type') == 'ALT' and not p.get('removed', False))
        
        if holes_count > 0:
            total_faces_with_holes += 1
            total_holes += holes_count
        if alts_count > 0:
            total_faces_with_alternates += 1
    
    print(f"\n[POLY FORM] {title}")
    print(f"[POLY FORM]   - Total faces: {len(unique_faces)}")
    print(f"[POLY FORM]   - Faces with holes: {total_faces_with_holes}")
    print(f"[POLY FORM]   - Total holes: {total_holes}")
    print(f"[POLY FORM]   - Faces with alternates: {total_faces_with_alternates}")
    
    for idx, face_eq in enumerate(unique_faces):
        polygons = face_eq.get('polygons', [])
        
        # Check if face has no polygons
        if len(polygons) == 0:
            print(f"[POLY FORM]   Face {idx+1}: NO POLYGONS (empty)")
            continue
        
        # Find boundary (first non-removed boundary or first polygon)
        boundary = None
        for poly in polygons:
            if not poly.get('removed', False):
                if poly.get('polygon_type') == 'BOUNDARY':
                    boundary = poly
                    break
                elif boundary is None:
                    boundary = poly
        
        # If all polygons are removed, show that
        if boundary is None:
            removed_count = sum(1 for p in polygons if p.get('removed', False))
            print(f"[POLY FORM]   Face {idx+1}: ALL POLYGONS REMOVED ({removed_count} removed polygon(s))")
            # Still show what was removed
            for poly_idx, poly in enumerate(polygons):
                if poly.get('removed', False):
                    verts = poly.get('vertices', [])
                    poly_type = poly.get('polygon_type', 'UNKNOWN')
                    print(f"[POLY FORM]     Removed {poly_type} {poly_idx}: {verts}")
            continue
        
        vertices = boundary.get('vertices', [])
        holes_count = sum(1 for p in polygons if p.get('polygon_type') == 'HOLE' and not p.get('removed', False))
        alts_count = sum(1 for p in polygons if p.get('polygon_type') == 'ALT' and not p.get('removed', False))
        
        print(f"[POLY FORM]   Face {idx+1}: {len(vertices)} vertices, {holes_count} hole(s), {alts_count} alternate(s)")
        print(f"[POLY FORM]     Boundary vertices: {vertices}")
        
        # Print holes
        for poly_idx, poly in enumerate(polygons):
            if poly.get('polygon_type') == 'HOLE' and not poly.get('removed', False):
                hole_verts = poly.get('vertices', [])
                print(f"[POLY FORM]     Hole {poly_idx}: {hole_verts}")
        
        # Print alternates
        for poly_idx, poly in enumerate(polygons):
            if poly.get('polygon_type') == 'ALT' and not poly.get('removed', False):
                alt_verts = poly.get('vertices', [])
                print(f"[POLY FORM]     Alternate {poly_idx}: {alt_verts}")


def plot_extraction_debug(selected_vertices, unique_faces, edge_face_map_all, invalid_edge_polygons, 
                          pruned_set=None, title="Extraction Debug Plot"):
    """
    Create an interactive 3D plot showing base set, extraction set, invalid edges, and original solid.
    
    Args:
        selected_vertices: Array of 3D vertex coordinates
        unique_faces: List of face dictionaries
        edge_face_map_all: Edge to face mapping
        invalid_edge_polygons: List of extracted polygon info dicts
        pruned_set: Set of (face_idx, poly_idx) tuples for pruned polygons
        title: Plot title
    """
    import numpy as np
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    from matplotlib.widgets import CheckButtons
    
    if pruned_set is None:
        pruned_set = set()
    
    # Ensure selected_vertices is a numpy array
    selected_vertices = np.array(selected_vertices)
    
    fig = plt.figure(figsize=(14, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Storage for plot elements - using lists to store line objects
    plot_elements = {
        'base_edges': [],
        'extraction_edges': [],
        'invalid_edges_lines': [],
        'boundary_edges_lines': [],
        'vertex_labels': [],
        'original_edges': []
    }
    
    # Colors
    colors = {
        'base': 'blue',
        'extraction': 'red',
        'invalid_edge': 'magenta',
        'boundary_edge': 'yellow',
        'vertex': 'black',
        'original': 'lightgray'
    }
    
    # Build extraction set lookup
    extraction_set = set((p['face_idx'], p['polygon_idx']) for p in invalid_edge_polygons)
    
    # 1. Plot base set polygon edges (not in extraction, not pruned)
    debug_first_polygon = True
    for face_idx, face_eq in enumerate(unique_faces):
        polygons = face_eq.get('polygons', [])
        for poly_idx, poly_data in enumerate(polygons):
            if poly_data.get('removed', False):
                continue
            if (face_idx, poly_idx) in extraction_set or (face_idx, poly_idx) in pruned_set:
                continue
            
            verts = poly_data.get('vertices', [])
            if len(verts) >= 3:
                # Debug first polygon
                if debug_first_polygon:
                    print(f"\n[DEBUG PLOT] First base polygon (Face {face_idx}, Poly {poly_idx}):")
                    print(f"  Raw verts: {verts[:5]}... (type: {type(verts[0]) if verts else 'N/A'})")
                    print(f"  Num selected_vertices: {len(selected_vertices)}")
                    debug_first_polygon = False
                
                # Validate and convert vertices (already 0-based)
                valid_verts = []
                for v in verts:
                    try:
                        v_idx = int(v)  # Ensure it's an integer (already 0-based)
                        if 0 <= v_idx < len(selected_vertices):
                            valid_verts.append(v_idx)
                    except (ValueError, TypeError):
                        continue
                
                if len(valid_verts) >= 3:
                    # Draw edges of this polygon
                    for i in range(len(valid_verts)):
                        v1_idx = valid_verts[i]
                        v2_idx = valid_verts[(i + 1) % len(valid_verts)]
                        
                        p1 = selected_vertices[v1_idx]
                        p2 = selected_vertices[v2_idx]
                        line = ax.plot3D([p1[0], p2[0]], [p1[1], p2[1]], [p1[2], p2[2]], 
                                       color=colors['base'], linewidth=1.5, alpha=0.7)[0]
                        plot_elements['base_edges'].append(line)
    
    # 2. Plot extraction set polygon edges
    debug_first_extraction = True
    for poly_info in invalid_edge_polygons:
        verts = poly_info['data'].get('vertices', [])
        if len(verts) >= 3:
            # Debug first extraction polygon
            if debug_first_extraction:
                print(f"\n[DEBUG PLOT] First extraction polygon:")
                print(f"  Raw verts: {verts[:5]}... (type: {type(verts[0]) if verts else 'N/A'})")
                debug_first_extraction = False
            
            # Validate and convert vertices (already 0-based)
            valid_verts = []
            for v in verts:
                try:
                    v_idx = int(v)  # Ensure it's an integer (already 0-based)
                    if 0 <= v_idx < len(selected_vertices):
                        valid_verts.append(v_idx)
                except (ValueError, TypeError):
                    continue
            
            if len(valid_verts) >= 3:
                # Draw edges of this polygon
                for i in range(len(valid_verts)):
                    v1_idx = valid_verts[i]
                    v2_idx = valid_verts[(i + 1) % len(valid_verts)]
                    
                    p1 = selected_vertices[v1_idx]
                    p2 = selected_vertices[v2_idx]
                    line = ax.plot3D([p1[0], p2[0]], [p1[1], p2[1]], [p1[2], p2[2]], 
                                   color=colors['extraction'], linewidth=2.5, alpha=0.9)[0]
                    plot_elements['extraction_edges'].append(line)
    
    # 3. Plot invalid edges (edges in 3+ faces in the CURRENT state)
    # Need to recompute edge counts based on current remaining polygons
    current_edge_map = {}
    
    # Count edges in base set (not extracted, not pruned, not removed)
    for face_idx, face_eq in enumerate(unique_faces):
        polygons = face_eq.get('polygons', [])
        for poly_idx, poly_data in enumerate(polygons):
            if poly_data.get('removed', False):
                continue
            if (face_idx, poly_idx) in extraction_set or (face_idx, poly_idx) in pruned_set:
                continue
            
            verts = poly_data.get('vertices', [])
            for i in range(len(verts)):
                v1 = verts[i]
                v2 = verts[(i + 1) % len(verts)]
                edge = tuple(sorted([v1, v2]))
                if edge not in current_edge_map:
                    current_edge_map[edge] = []
                current_edge_map[edge].append((face_idx, poly_idx, 'BASE'))
    
    # Count edges in extraction set
    for poly_info in invalid_edge_polygons:
        verts = poly_info['data'].get('vertices', [])
        for i in range(len(verts)):
            v1 = verts[i]
            v2 = verts[(i + 1) % len(verts)]
            edge = tuple(sorted([v1, v2]))
            if edge not in current_edge_map:
                current_edge_map[edge] = []
            current_edge_map[edge].append((poly_info['face_idx'], poly_info['polygon_idx'], 'EXTRACTION'))
    
    # Plot boundary edges (edges with only 1 face) in yellow
    boundary_edge_count = 0
    for edge, face_list in current_edge_map.items():
        if len(face_list) == 1:
            v1, v2 = edge
            if v1 < len(selected_vertices) and v2 < len(selected_vertices):
                p1 = selected_vertices[v1]
                p2 = selected_vertices[v2]
                line = ax.plot3D([p1[0], p2[0]], [p1[1], p2[1]], [p1[2], p2[2]], 
                               color=colors['boundary_edge'], linewidth=3, alpha=1.0, marker='s', markersize=3)[0]
                plot_elements['boundary_edges_lines'].append(line)
                boundary_edge_count += 1
    
    if boundary_edge_count > 0:
        print(f"[DEBUG PLOT] Plotted {boundary_edge_count} boundary edges in current state")
    
    # Plot edges that appear in 3+ faces
    invalid_edge_count = 0
    for edge, face_list in current_edge_map.items():
        if len(face_list) >= 3:
            v1, v2 = edge
            if v1 < len(selected_vertices) and v2 < len(selected_vertices):
                p1 = selected_vertices[v1]
                p2 = selected_vertices[v2]
                line = ax.plot3D([p1[0], p2[0]], [p1[1], p2[1]], [p1[2], p2[2]], 
                               color=colors['invalid_edge'], linewidth=4, alpha=1.0, marker='o', markersize=4)[0]
                plot_elements['invalid_edges_lines'].append(line)
                invalid_edge_count += 1
    
    if invalid_edge_count > 0:
        print(f"[DEBUG PLOT] Plotted {invalid_edge_count} invalid edges in current state")
    
    # 4. Plot all original edges (manifold edges) - initially hidden
    for edge, face_list in edge_face_map_all.items():
        if len(face_list) == 2:  # Manifold edges
            v1, v2 = edge
            if v1 < len(selected_vertices) and v2 < len(selected_vertices):
                p1 = selected_vertices[v1]
                p2 = selected_vertices[v2]
                line = ax.plot3D([p1[0], p2[0]], [p1[1], p2[1]], [p1[2], p2[2]], 
                               color=colors['original'], linewidth=0.5, alpha=0.3)[0]
                line.set_visible(False)  # Start hidden
                plot_elements['original_edges'].append(line)
    
    # 5. Plot vertex labels
    for idx, v in enumerate(selected_vertices):
        txt = ax.text(v[0], v[1], v[2], str(idx), fontsize=6, color=colors['vertex'], alpha=0.8)
        plot_elements['vertex_labels'].append(txt)
    
    # Set axis labels and title
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title(title, fontsize=14, fontweight='bold')
    
    # Create checkboxes for toggling visibility
    rax = plt.axes([0.02, 0.35, 0.18, 0.35], facecolor='lightgray')
    labels = ['Base Set', 'Extraction Set', 'Invalid Edges', 'Boundary Edges', 'Vertex Labels', 'Original Edges']
    visibility = [True, True, True, True, True, False]
    check = CheckButtons(rax, labels, visibility)
    
    # Store check in fig to prevent garbage collection
    fig._check_buttons = check
    
    def toggle_visibility(label):
        if label == 'Base Set':
            for line in plot_elements['base_edges']:
                line.set_visible(not line.get_visible())
        elif label == 'Extraction Set':
            for line in plot_elements['extraction_edges']:
                line.set_visible(not line.get_visible())
        elif label == 'Invalid Edges':
            for line in plot_elements['invalid_edges_lines']:
                line.set_visible(not line.get_visible())
        elif label == 'Boundary Edges':
            for line in plot_elements['boundary_edges_lines']:
                line.set_visible(not line.get_visible())
        elif label == 'Vertex Labels':
            for txt in plot_elements['vertex_labels']:
                txt.set_visible(not txt.get_visible())
        elif label == 'Original Edges':
            for line in plot_elements['original_edges']:
                line.set_visible(not line.get_visible())
        fig.canvas.draw_idle()
    
    check.on_clicked(toggle_visibility)
    
    # Set equal aspect ratio
    if len(selected_vertices) > 0:
        max_range = np.array([selected_vertices[:, 0].max() - selected_vertices[:, 0].min(),
                             selected_vertices[:, 1].max() - selected_vertices[:, 1].min(),
                             selected_vertices[:, 2].max() - selected_vertices[:, 2].min()]).max() / 2.0
        
        mid_x = (selected_vertices[:, 0].max() + selected_vertices[:, 0].min()) * 0.5
        mid_y = (selected_vertices[:, 1].max() + selected_vertices[:, 1].min()) * 0.5
        mid_z = (selected_vertices[:, 2].max() + selected_vertices[:, 2].min()) * 0.5
        ax.set_xlim(mid_x - max_range, mid_x + max_range)
        ax.set_ylim(mid_y - max_range, mid_y + max_range)
        ax.set_zlim(mid_z - max_range, mid_z + max_range)
    
    # Add legend info to title
    info_text = (f"Base: {len(plot_elements['base_edges'])} edges | "
                f"Extraction: {len(plot_elements['extraction_edges'])} edges | "
                f"Boundary: {len(plot_elements['boundary_edges_lines'])} edges | "
                f"Invalid: {len(plot_elements['invalid_edges_lines'])} edges")
    ax.text2D(0.5, 0.95, info_text, transform=ax.transAxes, 
             fontsize=10, ha='center', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # Add color legend
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], color=colors['base'], linewidth=2, label='Base Set (remaining polygons)'),
        Line2D([0], [0], color=colors['extraction'], linewidth=2, label='Extraction Set (to be removed)'),
        Line2D([0], [0], color=colors['boundary_edge'], linewidth=3, marker='s', label='Boundary Edges (1 face)'),
        Line2D([0], [0], color=colors['invalid_edge'], linewidth=3, marker='o', label='Invalid Edges (3+ faces)'),
        Line2D([0], [0], color=colors['original'], linewidth=1, alpha=0.5, label='Original Manifold Edges'),
        Line2D([0], [0], marker='$V$', color=colors['vertex'], linestyle='None', markersize=8, label='Vertex Labels')
    ]
    ax.legend(handles=legend_elements, loc='upper left', fontsize=9, framealpha=0.9)
    
    # Show non-blocking
    plt.ion()  # Turn on interactive mode
    plt.show(block=False)
    plt.pause(0.5)  # Give time for the plot to render


def extract_polygon_faces_from_connectivity(selected_vertices, merged_conn,
                                            top_conn=None, front_conn=None, side_conn=None,
                                            tolerance=1e-6):
    """
    Extract planar polygon faces from vertex connectivity matrix.
    
    Args:
        selected_vertices: Array of 3D vertex coordinates
        merged_conn: Combined connectivity matrix from all views (counts views per edge)
        top_conn: Top view connectivity matrix with visibility (1=solid, 2=dashed)
        front_conn: Front view connectivity matrix with visibility (1=solid, 2=dashed)
        side_conn: Side view connectivity matrix with visibility (1=solid, 2=dashed)
        tolerance: Distance tolerance for plane fitting
    
    When top_conn, front_conn, side_conn are provided, polygon classification uses
    visibility-based logic from engineering drawings. Otherwise falls back to geometric
    containment method.
    """
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
                
                # Debug logging: show which edge pair is being processed
                # print(f"[PLANE GEN] Vertex {i_row}: Processing edge pair ({i_row}-{j}, {i_row}-{k})")
                # print(f"            v1=[{v1[0]:.3f}, {v1[1]:.3f}, {v1[2]:.3f}]")
                # print(f"            v2=[{v2[0]:.3f}, {v2[1]:.3f}, {v2[2]:.3f}]")
                # print(f"            v3=[{v3[0]:.3f}, {v3[1]:.3f}, {v3[2]:.3f}]")
                
                # =========================================================
                # STEP 0: Check if vertices are axis-aligned first
                # Many engineering solids have faces parallel to x, y, z axes
                # This is more accurate than cross-product for axis-aligned faces
                # =========================================================
                axis_aligned_tol = 0.15  # 0.15mm tolerance for axis alignment
                
                # Check if all three vertices share the same X coordinate (Y-Z plane)
                x_coords = [v1[0], v2[0], v3[0]]
                x_min, x_max = min(x_coords), max(x_coords)
                if (x_max - x_min) < axis_aligned_tol:
                    # Y-Z plane: normal = [±1, 0, 0], d = -x_avg
                    x_avg = np.mean(x_coords)
                    n = np.array([1.0, 0.0, 0.0])
                    d = -x_avg
                    
                    # print(f"            → AXIS-ALIGNED X: normal=[{n[0]:.3f}, "
                    #       f"{n[1]:.3f}, {n[2]:.3f}], d={d:.3f}")
                    
                    # Add plane equation directly
                    face_equations.append({
                        'normal': n,
                        'd': d,
                        'source_row': i_row,
                        'vertices_used': [i_row, j, k]
                    })
                    pairs_added += 1
                    continue  # Skip general cross-product method
                
                # Check if all three vertices share the same Y coordinate (X-Z plane)
                y_coords = [v1[1], v2[1], v3[1]]
                y_min, y_max = min(y_coords), max(y_coords)
                if (y_max - y_min) < axis_aligned_tol:
                    # X-Z plane: normal = [0, ±1, 0], d = -y_avg
                    y_avg = np.mean(y_coords)
                    n = np.array([0.0, 1.0, 0.0])
                    d = -y_avg
                    
                    # print(f"            → AXIS-ALIGNED Y: normal=[{n[0]:.3f}, "
                    #       f"{n[1]:.3f}, {n[2]:.3f}], d={d:.3f}")
                    
                    # Add plane equation directly
                    face_equations.append({
                        'normal': n,
                        'd': d,
                        'source_row': i_row,
                        'vertices_used': [i_row, j, k]
                    })
                    pairs_added += 1
                    continue  # Skip general cross-product method
                
                # Check if all three vertices share the same Z coordinate (X-Y plane)
                z_coords = [v1[2], v2[2], v3[2]]
                z_min, z_max = min(z_coords), max(z_coords)
                if (z_max - z_min) < axis_aligned_tol:
                    # X-Y plane: normal = [0, 0, ±1], d = -z_avg
                    z_avg = np.mean(z_coords)
                    n = np.array([0.0, 0.0, 1.0])
                    d = -z_avg
                    
                    # print(f"            → AXIS-ALIGNED Z: normal=[{n[0]:.3f}, "
                    #       f"{n[1]:.3f}, {n[2]:.3f}], d={d:.3f}")
                    
                    # Add plane equation directly
                    face_equations.append({
                        'normal': n,
                        'd': d,
                        'source_row': i_row,
                        'vertices_used': [i_row, j, k]
                    })
                    pairs_added += 1
                    continue  # Skip general cross-product method
                
                # =========================================================
                # NOT AXIS-ALIGNED: Use general cross-product method
                # =========================================================
                
                # Compute cross product of two neighbouring edges
                # n = (v2-v1) X (v3-v1)
                e1 = v2 - v1
                e2 = v3 - v1
                
                n = np.cross(e1, e2)
                n_mag = np.linalg.norm(n)
                
                # Skip if edges are collinear (degenerate face)
                if n_mag < tolerance:
                    continue
                
                # Normalize to get unit normal
                n = n / n_mag
                
                # Clean up signed zeros
                n = np.where(np.abs(n) < 1e-10, 0.0, n)
                
                # Compute d value: n·x + d = 0, so d = -n·v1
                d = -np.dot(n, v1)
                
                # Debug logging
                # print(f"            → CROSS-PRODUCT: normal=[{n[0]:.3f}, "
                #       f"{n[1]:.3f}, {n[2]:.3f}], d={d:.3f}")
                
                # Add plane equation to list (will check uniqueness in Step 3)
                face_equations.append({
                    'normal': n,
                    'd': d,
                    'source_row': i_row,
                    'vertices_used': [i_row, j, k]
                })
                pairs_added += 1
    
    print(f"[POLY FORM]   {edge_count} edges, {pairs_checked} pairs checked, "
          f"{pairs_added} valid ranges")
    
    # Debug: Print all unique normals before merging
    print(f"\n[POLY FORM] Generated {len(face_equations)} plane equations")
    unique_normals_before = set()
    for face_eq in face_equations:
        n = face_eq['normal']
        n_tuple = tuple(np.round(n, 3))
        unique_normals_before.add(n_tuple)
    print(f"[POLY FORM] Unique normal directions (before merge): {len(unique_normals_before)}")
    for n in sorted(unique_normals_before):
        print(f"[POLY FORM]   n=[{n[0]:6.3f}, {n[1]:6.3f}, {n[2]:6.3f}]")
    
    # Step 3: Remove duplicate face ranges
    print("\n[POLY FORM] Step 3: Removing duplicate face ranges")
    
    # Tighter tolerances to avoid merging distinct faces
    # 0.1 degree: cos(0.1°) ≈ 0.9999985, so 1 - 0.9999985 ≈ 0.0000015
    initial_normal_tol = 0.000002  # ~0.1 degree (tightened from 1 degree)
    initial_d_tol = 0.25  # 0.25mm (tightened from 0.5mm to avoid merging nearby parallel faces)
    
    unique_faces = []
    rejected_count = 0
    for face_eq in face_equations:
        is_duplicate = False

        # Get normal and d for this face
        n1 = face_eq['normal']
        d1 = face_eq['d']

        for uf_idx, unique_face in enumerate(unique_faces):
            # Get normal and d for comparison
            n2 = unique_face['normal']
            d2 = unique_face['d']
            
            # Check if normals are parallel (same or opposite direction)
            dot_normals = np.dot(n1, n2)
            normals_parallel = abs(abs(dot_normals) - 1.0) < initial_normal_tol
            
            if normals_parallel:
                # Normals are parallel - check if same plane
                # For opposite normals, d-values have opposite signs
                # Same plane if |d1 + d2| < tolerance (opposite normals)
                # or |d1 - d2| < tolerance (same normals)
                
                if dot_normals > 0:
                    # Same direction: check |d1 - d2|
                    d_diff = abs(d1 - d2)
                    same_plane = d_diff < initial_d_tol
                    if same_plane:
                        is_duplicate = True
                        break
                else:
                    # Opposite direction: check |d1 + d2|
                    d_sum = abs(d1 + d2)
                    same_plane = d_sum < initial_d_tol
                    if same_plane:
                        is_duplicate = True
                        break

        if is_duplicate:
            rejected_count += 1
        
        if not is_duplicate:
            unique_faces.append(face_eq)
            
            # Show the plane equation
            n = face_eq['normal']
            d = face_eq['d']
            
            print(f"[POLY FORM] F{len(unique_faces):2d}: n=[{n[0]:6.3f},{n[1]:6.3f},{n[2]:6.3f}] d={d:7.2f}")
    
    print(f"[POLY FORM] {len(unique_faces)} unique faces found ({rejected_count} duplicates merged)")
    
    # Step 4: Find all vertices on each face using iterative tolerance
    print("\n[POLY FORM] Step 4: Finding vertices on each face")
    print("[POLY FORM] ----------------------------------------------------------------------")
    
    # Initialize tolerance - should be tight enough to distinguish nearby parallel faces
    # but loose enough to handle numerical precision in face equation fitting
    current_tolerance = 0.05  # Start with 0.05mm (tighter than previous 0.25mm)
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
                
                # Debug: Print distances for Face 11 vertices
                if face_idx == 10 and iteration == 1:  # Face 11 (0-indexed), first iteration
                    if v_idx in [6, 7, 14, 15, 20, 21, 30, 31]:
                        print(f"[DEBUG]   Vertex {v_idx}: distance = {abs(eps):.6f} mm")
                
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
            if next_tolerance > 0.15:  # Limit to 0.15mm to avoid merging nearby parallel faces
                print(f"[POLY FORM]   {len(unassigned_vertices)} vertices have < 3 faces")
                print(f"[POLY FORM]   WARNING: Cannot increase tolerance beyond 0.15mm limit (would be {next_tolerance:.2f}mm)")
                print(f"[POLY FORM]   These vertices may be on edges/corners of fewer faces")
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
    
    # Step 4.5: Detect missing faces from under-assigned vertices
    print("\n[POLY FORM] Step 4.5: Detecting missing faces from under-assigned vertices")
    print("[POLY FORM] ----------------------------------------------------------------------")
    
    # Find vertices with < 3 faces
    vertex_face_count = [0] * N
    for face_eq in unique_faces:
        for v_idx in face_eq['vertices_on_face']:
            vertex_face_count[v_idx] += 1
    
    under_assigned = [v_idx for v_idx in range(N) if vertex_face_count[v_idx] < 3]
    
    if len(under_assigned) > 0:
        print(f"[POLY FORM]   Found {len(under_assigned)} vertices with < 3 faces")
        print(f"[POLY FORM]   Attempting to fit planes through these vertices...")
        
        # Try to fit planes through under-assigned vertices
        remaining_vertices = set(under_assigned)
        new_faces_added = 0
        plane_fit_tolerance = 0.1  # 0.1mm tolerance for plane fitting
        
        while len(remaining_vertices) >= 3:
            # Take vertices from remaining set
            candidate_vertices = list(remaining_vertices)
            
            # Start with first 3 vertices to define initial plane
            if len(candidate_vertices) < 3:
                break
                
            v_indices = candidate_vertices[:3]
            v1 = selected_vertices[v_indices[0]]
            v2 = selected_vertices[v_indices[1]]
            v3 = selected_vertices[v_indices[2]]
            
            # Compute normal from first 3 vertices
            e1 = v2 - v1
            e2 = v3 - v1
            n = np.cross(e1, e2)
            n_mag = np.linalg.norm(n)
            
            if n_mag < tolerance:
                # Collinear vertices, skip first vertex and try again
                remaining_vertices.remove(v_indices[0])
                continue
            
            n = n / n_mag
            n = np.where(np.abs(n) < 1e-10, 0.0, n)  # Clean signed zeros
            d = -np.dot(n, v1)
            
            # Find all vertices (including from candidate list) that fit this plane
            vertices_on_plane = []
            for v_idx in candidate_vertices:
                vertex = selected_vertices[v_idx]
                eps = np.dot(n, vertex) + d
                if abs(eps) < plane_fit_tolerance:
                    vertices_on_plane.append(v_idx)
            
            # Only add if we have at least 3 vertices
            if len(vertices_on_plane) >= 3:
                # Check if this plane is a duplicate of existing faces
                is_duplicate = False
                for existing_face in unique_faces:
                    n_existing = existing_face['normal']
                    d_existing = existing_face['d']
                    
                    dot_normals = np.dot(n, n_existing)
                    normals_parallel = abs(abs(dot_normals) - 1.0) < 0.000002
                    
                    if normals_parallel:
                        if dot_normals > 0:
                            d_diff = abs(d - d_existing)
                            if d_diff < 0.5:
                                is_duplicate = True
                                break
                        else:
                            d_sum = abs(d + d_existing)
                            if d_sum < 0.5:
                                is_duplicate = True
                                break
                
                if not is_duplicate:
                    # Add new face to base set (face_equations) and unique set
                    new_face = {
                        'normal': n,
                        'd': d,
                        'vertices_on_face': vertices_on_plane,
                        'source_row': -1,  # Mark as generated from under-assigned vertices
                        'vertices_used': v_indices[:3]
                    }
                    # Add to both base set and unique faces
                    face_equations.append(new_face)
                    unique_faces.append(new_face)
                    new_faces_added += 1
                    
                    print(f"[POLY FORM]   Added missing face {len(unique_faces)}: "
                          f"n=[{n[0]:6.3f},{n[1]:6.3f},{n[2]:6.3f}] d={d:7.2f}")
                    print(f"[POLY FORM]     {len(vertices_on_plane)} vertices on plane: {vertices_on_plane}")
                    
                    # Remove these vertices from remaining set
                    for v_idx in vertices_on_plane:
                        remaining_vertices.discard(v_idx)
                else:
                    # This plane already exists, just remove first vertex and continue
                    remaining_vertices.remove(v_indices[0])
            else:
                # Not enough vertices on this plane, remove first vertex
                remaining_vertices.remove(v_indices[0])
        
        if new_faces_added > 0:
            print(f"[POLY FORM]   Total missing faces added: {new_faces_added}")
            print(f"[POLY FORM]   Total unique faces: {len(unique_faces)}")
        else:
            print(f"[POLY FORM]   No additional faces could be fitted")
    else:
        print(f"[POLY FORM]   All vertices assigned to ≥3 faces, no missing faces detected")
    
    # Display final vertex assignments
    print(f"\n[POLY FORM]   Final assignments (tolerance={current_tolerance/2 if iteration > 1 else current_tolerance:.4f}mm):")
    for face_idx, face_eq in enumerate(unique_faces):
        vertices_display = face_eq['vertices_on_face']  # Keep 0-indexed
        print(f"[POLY FORM] F{face_idx:2d}: {len(face_eq['vertices_on_face']):2d} vertices: {vertices_display}")
    
    
    for face_idx, face_eq in enumerate(unique_faces):
        edges_on_face = []
        vertices_on_face = face_eq['vertices_on_face']
        
        # DIAGNOSTIC: Check indexing consistency for Face 5
        if face_idx == 4:  # Face 5 (0-indexed as 4)
            print(f"\n[DIAGNOSTIC] Face {face_idx} before edge detection:")
            print(f"  vertices_on_face (0-indexed): {sorted(vertices_on_face)}")
        
        # Check all pairs of vertices on this face
        for i in range(len(vertices_on_face)):
            v_i = vertices_on_face[i]
            for j in range(i+1, len(vertices_on_face)):
                v_j = vertices_on_face[j]
                
                # Check if there's an edge in connectivity matrix
                if merged_conn[v_i, v_j] == 3:
                    edges_on_face.append((v_i, v_j))
        
        face_eq['edges_on_face'] = edges_on_face
        
        # DIAGNOSTIC: Check edges for Face 5
        if face_idx == 4:
            print(f"[DIAGNOSTIC] Face {face_idx} edges (0-indexed): {edges_on_face}")
        
        # Display edges with 0-indexed vertices to match plot
        edges_display = edges_on_face  # Keep 0-indexed
        print(f"[POLY FORM]   Face {face_idx}: {len(edges_on_face)} edges with conn=3: {edges_display}")
    
    # DIAGNOSTIC: Check if edges with conn=3 can form closed polygons not in face polygons
    print("\n[POLY FORM] Step 4.5b: Checking for potential missing polygons from conn=3 edges")
    for face_idx, face_eq in enumerate(unique_faces):
        edges_on_face = face_eq.get('edges_on_face', [])
        
        if len(edges_on_face) < 3:
            continue  # Need at least 3 edges to form a polygon
        
        # Try to find closed loops in these edges
        from collections import defaultdict
        edge_graph = defaultdict(list)
        for v1, v2 in edges_on_face:
            edge_graph[v1].append(v2)
            edge_graph[v2].append(v1)
        
        # Find closed loops
        visited_edges = set()
        closed_loops = []
        
        for start_v in edge_graph:
            if len(edge_graph[start_v]) == 0:
                continue
            
            path = [start_v]
            current = start_v
            
            while True:
                # Find next unvisited neighbor
                next_v = None
                for neighbor in edge_graph[current]:
                    edge_key = tuple(sorted([current, neighbor]))
                    if edge_key not in visited_edges:
                        next_v = neighbor
                        visited_edges.add(edge_key)
                        break
                
                if next_v is None:
                    break
                
                if next_v == start_v and len(path) >= 3:
                    # Closed loop found!
                    closed_loops.append(path[:])
                    break
                
                path.append(next_v)
                current = next_v
                
                if len(path) > len(edge_graph):
                    break
        
        if len(closed_loops) > 0:
            print(f"[POLY FORM]   Face {face_idx}: Found {len(closed_loops)} potential closed loop(s) from conn=3 edges")
            
            # Check if these loops are already in the face's polygons
            existing_polygons = face_eq.get('polygons', [])
            
            if len(existing_polygons) == 0:
                print(f"[POLY FORM]     Face {face_idx} has NO existing polygons")
            
            for loop_idx, loop in enumerate(closed_loops):
                loop_set = set(loop)
                
                # Check if this loop matches any existing polygon
                found_match = False
                for poly_idx, poly_data in enumerate(existing_polygons):
                    if poly_data.get('removed', False):
                        continue
                    poly_verts = poly_data.get('vertices', [])
                    poly_set = set(poly_verts)
                    
                    if loop_set == poly_set or loop_set.issubset(poly_set):
                        found_match = True
                        break
                
                if not found_match:
                    # This loop is NOT in any existing polygon - ADD IT!
                    loop_display = loop  # Keep 0-indexed
                    print(f"[POLY FORM]     ⚠️  Loop {loop_idx+1} NOT FOUND in face polygons: {loop_display}")
                    
                    if len(existing_polygons) > 0:
                        print(f"[POLY FORM]         Existing polygons in Face {face_idx}:")
                        for poly_idx, poly_data in enumerate(existing_polygons):
                            if poly_data.get('removed', False):
                                continue
                            poly_verts = poly_data.get('vertices', [])
                            poly_display = poly_verts  # Keep 0-indexed
                            poly_type = poly_data.get('polygon_type', 'UNKNOWN')
                            print(f"[POLY FORM]           Polygon {poly_idx} ({poly_type}): {poly_display}")
                    
                    # Check planarity of this loop
                    if len(loop) >= 3:
                        loop_verts_3d = [selected_vertices[v_idx] for v_idx in loop]
                        
                        # Fit plane using SVD
                        centroid = np.mean(loop_verts_3d, axis=0)
                        centered = np.array(loop_verts_3d) - centroid
                        _, _, Vt = np.linalg.svd(centered)
                        computed_normal = Vt[-1]
                        computed_d = -np.dot(computed_normal, centroid)
                        
                        # Check max distance from fitted plane
                        max_dist = 0.0
                        for v in loop_verts_3d:
                            dist = abs(np.dot(computed_normal, v) + computed_d)
                            max_dist = max(max_dist, dist)
                        
                        if max_dist < 0.001:  # 1 micron tolerance
                            print(f"[POLY FORM]         ✓ Loop is planar (deviation: {max_dist:.6f}mm)")
                            print(f"[POLY FORM]         → Adding loop as new polygon to Face {face_idx}")
                            
                            # Add this loop as a new polygon
                            new_poly_data = {
                                'vertices': loop,
                                'normal': tuple(computed_normal),
                                'd': computed_d,
                                'removed': False,
                                'polygon_type': 'BOUNDARY'
                            }
                            
                            # Ensure face has 'polygons' list
                            if 'polygons' not in face_eq:
                                face_eq['polygons'] = []
                            
                            face_eq['polygons'].append(new_poly_data)
                            print(f"[POLY FORM]         → Added as polygon {len(face_eq['polygons'])-1}")
                        else:
                            print(f"[POLY FORM]         ✗ Loop is NON-PLANAR (deviation: {max_dist:.6f}mm) - skipping")
                else:
                    loop_display = loop  # Keep 0-indexed
                    print(f"[POLY FORM]     Loop {loop_idx+1} exists in face polygons: {loop_display}")
    
    # CRITICAL: Update all face vertex lists from their edges
    # This ensures vertices_on_face includes ALL vertices that participate in edges
    print("\n[POLY FORM] Step 4.6: Synchronizing vertex assignments with edge lists")
    for face_idx, face_eq in enumerate(unique_faces):
        edges_on_face = face_eq['edges_on_face']
        original_verts = set(face_eq['vertices_on_face'])
        
        if len(edges_on_face) > 0:
            vertices_from_edges = set()
            for v1, v2 in edges_on_face:
                vertices_from_edges.add(v1)
                vertices_from_edges.add(v2)
            
            new_verts = vertices_from_edges - original_verts
            missing_verts = original_verts - vertices_from_edges
            
            if len(new_verts) > 0 or len(missing_verts) > 0:
                print(f"[POLY FORM]   Face {face_idx}:")
                print(f"[POLY FORM]     Original {len(original_verts)} verts: {sorted(original_verts)}")
                print(f"[POLY FORM]     From edges {len(vertices_from_edges)} verts: {sorted(vertices_from_edges)}")
                if len(new_verts) > 0:
                    print(f"[POLY FORM]     NEW (in edges, not assigned): {sorted(new_verts)}")
                if len(missing_verts) > 0:
                    print(f"[POLY FORM]     MISSING (assigned, no edges): {sorted(missing_verts)}")
                
                face_eq['vertices_on_face'] = sorted(list(vertices_from_edges))
                print(f"[POLY FORM]     Updated to {len(face_eq['vertices_on_face'])} vertices")

    
    # Step 5.5: Process collinear edges - split edges containing intermediate vertices
    print("\n[POLY FORM] Step 5.5: Processing collinear edges")
    print("-" * 70)
    
    # First, analyze vertex connectivity using merged_conn (only conn==3)
    print("\n[POLY FORM] Step 5.5.1: Analyzing vertex connectivity (conn==3 only)")
    vertex_conn3_count = {}
    vertex_conn3_neighbors = {}
    
    for v_idx in range(len(selected_vertices)):
        neighbors = []
        for other_v in range(len(selected_vertices)):
            if other_v != v_idx and merged_conn[v_idx, other_v] == 3:
                neighbors.append(other_v)
        vertex_conn3_count[v_idx] = len(neighbors)
        vertex_conn3_neighbors[v_idx] = neighbors
    
    print(f"[POLY FORM]   Vertices with conn==3 edges:")
    for v_idx, count in sorted(vertex_conn3_count.items()):
        if count > 0:
            print(f"     v{v_idx}: {count} edges (conn==3)")
    
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
    
    # Track two-edge vertices for later use
    two_edge_vertices = set()
    
    # Track edge splits for cross-face propagation
    global_edge_splits = {}  # {original_edge: [(v_mid, t), ...]} - sorted by t
    
    # Track which edges came from splitting high-connectivity edges
    # These should be preserved even if their individual connectivity is low
    split_edge_from_high_conn = set()  # Set of edges that came from conn>=3 splits
    
    # Process each face
    for face_idx, face_eq in enumerate(unique_faces):
        edges_on_face = face_eq['edges_on_face']
        vertices_on_face = face_eq['vertices_on_face']
        
        if len(edges_on_face) == 0:
            continue
        
        print(f"\n[POLY FORM]   Processing Face {face_idx}: {len(edges_on_face)} edges")
        
        # Store the original vertices assigned to this face before collinear splitting
        # This prevents vertices from other faces being added via collinear detection
        original_face_vertices = set(vertices_on_face)
        
        # DIAGNOSTIC: Check Face 5 for vertex 30
        if face_idx == 4:  # Face 5 is 0-indexed as 4
            print(f"      [DIAGNOSTIC] Face {face_idx} vertices_on_face (0-indexed): {sorted(vertices_on_face)}")
            if 29 in vertices_on_face:
                print(f"      [DIAGNOSTIC] ✓ Vertex 29 IS in Face {face_idx}")
            else:
                print(f"      [DIAGNOSTIC] ✗ Vertex 29 NOT in Face {face_idx} - cannot detect collinearity!")
        
        # Store edges to process and new edges to add
        edges_to_remove = []
        edges_to_add = []
        
        # Process each edge (outer loop edge)
        for outer_edge in edges_on_face:
            v_start, v_end = outer_edge
            start_pos = selected_vertices[v_start]
            end_pos = selected_vertices[v_end]
            
            # DIAGNOSTIC: Check specific edge (5, 72) for vertex 30 (0-indexed: 4, 71 for 29)
            is_diagnostic_edge = (v_start == 4 and v_end == 71) or (v_start == 71 and v_end == 4)
            
            # Find all vertices that lie on this line (including beyond endpoints)
            collinear_vertices = []
            
            # Check all other vertices on this face
            for v_idx in vertices_on_face:
                if v_idx == v_start or v_idx == v_end:
                    continue
                
                # IMPORTANT: Only check vertices that were originally assigned to this face
                # This prevents vertices from other faces being added via collinear detection
                if v_idx not in original_face_vertices:
                    continue
                
                v_pos = selected_vertices[v_idx]
                is_on_seg, is_inside, t = point_on_ray_segment(
                    start_pos, end_pos, v_pos, tolerance=0.1
                )
                
                # DIAGNOSTIC: Special output for vertex 30 (0-indexed: 29)
                if is_diagnostic_edge and v_idx == 29:
                    print(f"      [DIAGNOSTIC] Checking v29 on edge ({v_start}, {v_end}):")
                    print(f"         is_on_seg={is_on_seg}, is_inside={is_inside}, t={t:.6f}")
                    print(f"         v_pos={v_pos}")
                    print(f"         start_pos={start_pos}, end_pos={end_pos}")
                    dist_to_line = np.linalg.norm(v_pos - (start_pos + t * (end_pos - start_pos)))
                    print(f"         distance_to_line={dist_to_line:.6f}")
                
                # Consider ALL vertices on the line, regardless of position
                # We'll sequence them by t and use connectivity to determine edges
                if is_on_seg:
                    collinear_vertices.append((v_idx, t, is_inside))
            
            if len(collinear_vertices) == 0:
                continue
                
            print(f"      Edge ({v_start}, {v_end}): Found {len(collinear_vertices)} collinear vertices")
            
            # Step 5.5.2: Collect ALL vertices on the line (including endpoints)
            # and sequence them by t parameter
            all_vertices_on_line = []
            
            # Add collinear vertices (ONLY if they're inside the segment)
            for v_idx, t, is_inside in collinear_vertices:
                if is_inside:  # Only include vertices actually ON the segment
                    all_vertices_on_line.append((v_idx, t))
            
            # Add the original endpoints
            all_vertices_on_line.append((v_start, 0.0))
            all_vertices_on_line.append((v_end, 1.0))
            
            # Sort all vertices by their t parameter
            all_vertices_on_line.sort(key=lambda x: x[1])
            
            # Remove duplicates while preserving order (keep first occurrence)
            seen = set()
            vertex_sequence = []
            vertex_to_t = {}  # Map vertex to its t parameter
            for v_idx, t in all_vertices_on_line:
                if v_idx not in seen:
                    seen.add(v_idx)
                    vertex_sequence.append(v_idx)
                    vertex_to_t[v_idx] = t
            
            print(f"         Vertex sequence (by t): {vertex_sequence}")
            
            # Step 5.5.3: Check connectivity between consecutive vertices
            # For collinear splits: if original edge had conn>=3, accept ALL sub-edges
            # even if individual connectivity is lower (the path is valid through collinearity)
            original_edge_conn = merged_conn[v_start, v_end]
            is_collinear_split = len(vertex_sequence) > 2
            
            if is_collinear_split:
                print(f"         Original edge ({v_start}, {v_end}): conn={original_edge_conn}")
            
            edges_to_add_for_this_edge = []
            for i in range(len(vertex_sequence) - 1):
                v1 = vertex_sequence[i]
                v2 = vertex_sequence[i + 1]
                
                new_edge = (min(v1, v2), max(v1, v2))
                conn_value = merged_conn[v1, v2]
                
                # For collinear splits of high-connectivity edges (conn>=3):
                # Include ALL sub-edges to maintain geometric sequence, even if conn=0
                # The vertices are geometrically on the line, so they MUST be included
                if is_collinear_split and original_edge_conn >= 3:
                    edges_to_add_for_this_edge.append(new_edge)
                    split_edge_from_high_conn.add(new_edge)  # Track this as a split from high-conn edge
                    if conn_value == 0:
                        print(f"         Edge ({v1}, {v2}): conn={conn_value} - WILL ADD (collinear split of conn={original_edge_conn} edge, geometry overrides connectivity)")
                    else:
                        print(f"         Edge ({v1}, {v2}): conn={conn_value} - WILL ADD (collinear split of conn={original_edge_conn} edge)")
                # For non-collinear edges, require conn>=3
                elif conn_value >= 3:
                    edges_to_add_for_this_edge.append(new_edge)
                    print(f"         Edge ({v1}, {v2}): conn={conn_value} - WILL ADD")
                # Skip low-connectivity edges that aren't part of collinear splits
                else:
                    if conn_value == 0:
                        print(f"         Edge ({v1}, {v2}): conn=0 - SKIPPED (edge does not exist in connectivity matrix)")
                    else:
                        print(f"         Edge ({v1}, {v2}): conn={conn_value} - SKIPPED (need conn>=3 for reverse engineering)")
            
            # If we're adding edges different from the original edge, update
            if len(edges_to_add_for_this_edge) > 0 and len(vertex_sequence) > 2:
                # Mark original edge for removal if we're splitting it
                edges_to_remove.append(outer_edge)
                
                # Add all the validated edges
                edges_to_add.extend(edges_to_add_for_this_edge)
                
                # Record this split for cross-face propagation
                # Store intermediate vertices with their t parameters (exclude endpoints)
                # Only record if not already recorded for this edge
                if outer_edge not in global_edge_splits:
                    global_edge_splits[outer_edge] = []
                    for i in range(1, len(vertex_sequence) - 1):  # Skip endpoints
                        v_mid = vertex_sequence[i]
                        t_val = vertex_to_t[v_mid]
                        global_edge_splits[outer_edge].append((v_mid, t_val))
                
                # Check for two-edge vertices (vertices with exactly 2 neighbors on this line)
                for i in range(1, len(vertex_sequence) - 1):  # Skip endpoints
                    v_mid = vertex_sequence[i]
                    
                    # Count how many neighbors this vertex has on the line
                    neighbors_on_line = 0
                    for other_v in vertex_sequence:
                        if other_v != v_mid and merged_conn[v_mid, other_v] >= 2:
                            neighbors_on_line += 1
                    
                    # Check if this vertex ONLY connects to vertices on this line
                    all_neighbors = set()
                    for other_v in range(len(selected_vertices)):
                        if other_v != v_mid and merged_conn[v_mid, other_v] >= 2:
                            all_neighbors.add(other_v)
                    
                    neighbors_on_line_set = set(v for v in vertex_sequence if v != v_mid and merged_conn[v_mid, v] >= 2)
                    
                    if len(all_neighbors) == 2 and all_neighbors == neighbors_on_line_set:
                        two_edge_vertices.add(v_mid)
                        print(f"         Marked v{v_mid} as two-edge vertex")

        
        # Update edges_on_face for this face
        if len(edges_to_remove) > 0 or len(edges_to_add) > 0:
            print(f"      Updating edges: removing {len(edges_to_remove)}, "
                  f"adding {len(edges_to_add)}")
            print(f"      Edges to remove: {sorted(edges_to_remove)}")
            print(f"      Edges to add: {sorted(edges_to_add)}")
            
            # Remove old edges
            for edge in edges_to_remove:
                if edge in edges_on_face:
                    edges_on_face.remove(edge)
                else:
                    print(f"      WARNING: Edge {edge} not found in edges_on_face for removal")
            
            # Add new edges
            edges_on_face.extend(edges_to_add)
            
            # Update the face data
            face_eq['edges_on_face'] = edges_on_face
            
            # CRITICAL: Update vertices_on_face to include ALL vertices from edges
            # This ensures consistency between vertex assignments and edge lists
            vertices_from_edges = set()
            for v1, v2 in edges_on_face:
                vertices_from_edges.add(v1)
                vertices_from_edges.add(v2)
            face_eq['vertices_on_face'] = sorted(list(vertices_from_edges))
            
            print(f"      Face {face_idx} now has {len(edges_on_face)} edges, {len(face_eq['vertices_on_face'])} vertices")
    
    # Store two-edge vertices for use in Vertex Valency Analysis
    print(f"\n[POLY FORM] Identified {len(two_edge_vertices)} two-edge vertices: {sorted(two_edge_vertices)}")
    
    print("\n[POLY FORM] Step 5.5 Complete: Collinear edge processing finished")
    
    # ==========================================================================
    # Step 5.6: Propagate edge splits across all faces
    # ==========================================================================
    print("\n[POLY FORM] Step 5.6: Propagating edge splits across all faces...")
    print(f"[POLY FORM]   Found {len(global_edge_splits)} edge(s) that were split")
    
    if global_edge_splits:
        # For each split edge, update ALL faces that contain it
        for original_edge, intermediates in global_edge_splits.items():
            v1, v2 = original_edge
            
            # Check if original edge had high connectivity
            original_edge_conn = merged_conn[v1, v2]
            is_high_conn_split = (original_edge_conn >= 3)
            
            # Build complete vertex sequence (sorted by t parameter)
            sorted_intermediates = sorted(intermediates, key=lambda x: x[1])
            vertex_sequence = [v1] + [v_mid for v_mid, t in sorted_intermediates] + [v2]
            
            # Build sub-edges
            sub_edges = []
            for i in range(len(vertex_sequence) - 1):
                sub_edge = (min(vertex_sequence[i], vertex_sequence[i+1]), 
                           max(vertex_sequence[i], vertex_sequence[i+1]))
                sub_edges.append(sub_edge)
                # Track sub-edges from high-connectivity splits
                if is_high_conn_split:
                    split_edge_from_high_conn.add(sub_edge)
            
            print(f"[POLY FORM]   Edge {original_edge} split with vertices {[v for v, t in sorted_intermediates]}")
            print(f"[POLY FORM]     Sub-edges: {sub_edges}")
            
            # Find all faces that have this edge and update them
            faces_updated = 0
            for face_idx, face_eq in enumerate(unique_faces):
                edges_on_face = face_eq['edges_on_face']
                
                # If this face still has the original unsplit edge, replace it
                if original_edge in edges_on_face:
                    print(f"[POLY FORM]     Updating Face {face_idx}: replacing {original_edge} with {sub_edges}")
                    
                    # Remove the original edge
                    edges_on_face.remove(original_edge)
                    
                    # Add sub-edges (avoid duplicates)
                    for sub_edge in sub_edges:
                        if sub_edge not in edges_on_face:
                            edges_on_face.append(sub_edge)
                    
                    # Update vertices_on_face to include intermediate vertices
                    vertices_from_edges = set()
                    for e1, e2 in edges_on_face:
                        vertices_from_edges.add(e1)
                        vertices_from_edges.add(e2)
                    face_eq['vertices_on_face'] = sorted(list(vertices_from_edges))
                    face_eq['edges_on_face'] = edges_on_face
                    
                    faces_updated += 1
            
            if faces_updated > 0:
                print(f"[POLY FORM]     Updated {faces_updated} face(s)")
    
    print(f"\n[POLY FORM] Step 5.6 Complete: Edge splits propagated across all faces")
    
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
                    # Found a valid cycle - verify the closing edge exists
                    closing_edge = (min(current, start_vertex), max(current, start_vertex))
                    if closing_edge not in edge_set:
                        # ERROR: Trying to close cycle with non-existent edge!
                        print(f"[DFS ERROR] Attempted to close cycle {path} with missing edge {closing_edge}")
                        print(f"[DFS ERROR] Current vertex: {current}, Start vertex: {start_vertex}")
                        print(f"[DFS ERROR] Neighbor in adjacency but edge not in edge_set!")
                        continue
                    
                    # Found a valid cycle
                    cycle_edges = path_edges | {closing_edge}
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
    
    def normalize_polygon(poly_verts):
        """
        Normalize a polygon's vertex list for consistent comparison.
        Returns the polygon starting with the smallest vertex index,
        with the second vertex being the smaller of the two neighbors.
        
        Args:
            poly_verts: List of vertex indices (cyclic)
        
        Returns:
            Normalized list of vertex indices
        """
        if len(poly_verts) < 3:
            return poly_verts
        
        # Find the minimum vertex
        min_idx = min(poly_verts)
        min_pos = poly_verts.index(min_idx)
        
        # Get the two neighbors
        n = len(poly_verts)
        prev_vertex = poly_verts[(min_pos - 1) % n]
        next_vertex = poly_verts[(min_pos + 1) % n]
        
        # Choose direction based on which neighbor is smaller
        if next_vertex < prev_vertex:
            # Forward direction
            normalized = poly_verts[min_pos:] + poly_verts[:min_pos]
        else:
            # Reverse direction
            reversed_poly = poly_verts[::-1]
            min_pos_rev = reversed_poly.index(min_idx)
            normalized = reversed_poly[min_pos_rev:] + reversed_poly[:min_pos_rev]
        
        return normalized
    
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
                                       selected_verts, normal, d_value=None,
                                       merged_conn=None, split_edges_from_high_conn=None):
        """
        [STEP 6] Build boundary and holes from face edges using DFS cycle detection.
        Returns dict with 'faces' list and 'unused_edges'
        
        Args:
            split_edges_from_high_conn: Set of edges that came from splitting high-connectivity edges.
                                       These should be preserved even if conn<3.
            d_value: The d coefficient of the plane equation (for visualization)
        """
        DEBUG_STEPS = False  # Set to True to enable STEP debug output
        
        if split_edges_from_high_conn is None:
            split_edges_from_high_conn = set()
        
        if DEBUG_STEPS:
            print(f"  [STEP 6.0] Building polygons from {len(edges_on_face)} edges")
        
        # [STEP 6.0] Project vertices to 2D plane (verts_2d: dict {v_idx: (x,y)})
        if abs(normal[2]) < 0.9:
            basis_u = np.cross(normal, [0, 0, 1])
        else:
            basis_u = np.cross(normal, [1, 0, 0])
        basis_u = basis_u / np.linalg.norm(basis_u)
        basis_v = np.cross(normal, basis_u)
        
        verts_2d = {}
        for v_idx in vertices_on_face:
            vert_3d = selected_verts[v_idx]
            verts_2d[v_idx] = (np.dot(vert_3d, basis_u), np.dot(vert_3d, basis_v))
        
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
            if DEBUG_STEPS:
                print(f"  [STEP 6.0] Iteration {iteration}: {len(used_vertices)}/"
                      f"{len(vertices_on_face)} vertices used")
            
            # [STEP 6.1] Get remaining vertices and edges
            remaining_verts = set(vertices_on_face) - used_vertices
            if not remaining_verts:
                break
            
            remaining_edges = edge_set - used_edges
            
            # [STEP 6.1] Filter remaining edges: only use edges with conn=3
            print(f"  [STEP 6.1] Iteration {iteration}: {len(remaining_edges)} remaining edges before filter")
            if iteration > 1:
                if merged_conn is None:
                    print("  [STEP 6.1] WARNING: merged_conn is None, cannot filter edges")
                else:
                    print(f"  [STEP 6.1] merged_conn available, filtering edges...")
                    filtered_remaining_edges = set()
                    edges_filtered_out = 0
                    for edge in remaining_edges:
                        v1, v2 = edge
                        edge_conn_value = 0
                        if v1 < len(merged_conn) and v2 < len(merged_conn):
                            edge_conn_value = max(merged_conn[v1, v2], merged_conn[v2, v1])
                        
                        # Keep edge if: (1) conn==3, OR (2) it's a split from high-conn edge
                        if edge_conn_value == 3:
                            filtered_remaining_edges.add(edge)
                        elif edge in split_edges_from_high_conn:
                            filtered_remaining_edges.add(edge)
                            print(f"  [STEP 6.1]   Keeping edge {edge}: conn={edge_conn_value} (split from high-conn edge)")
                        else:
                            edges_filtered_out += 1
                            print(f"  [STEP 6.1]   Filtering edge {edge}: conn={edge_conn_value}")
                    
                    print(f"  [STEP 6.1] Filtered out {edges_filtered_out} edges with conn≠3 "
                          f"({len(filtered_remaining_edges)} edges remain)")
                    if len(filtered_remaining_edges) > 0:
                        print(f"  [STEP 6.1] Remaining edges with conn=3: {sorted(filtered_remaining_edges)}")
                    remaining_edges = filtered_remaining_edges
            
            remaining_vert_set = set()
            for e in remaining_edges:
                remaining_vert_set.add(e[0])
                remaining_vert_set.add(e[1])
            
            remaining_verts = remaining_verts & remaining_vert_set
            if not remaining_verts:
                if DEBUG_STEPS:
                    print("  [STEP 6.1] No more vertices with edges")
                break
            
            # [STEP 6.1] Find all cycles using DFS (all_possible_polygons: list of vertex lists)
            print(f"  [STEP 6.1] Finding all polygons from "
                  f"{len(remaining_edges)} edges...")
            remaining_edges_list = list(remaining_edges)
            all_possible_polygons = find_all_cycles_from_edges(
                remaining_edges_list, verts_2d)
            
            # [STEP 6.1] IMPORTANT: In first iteration, ensure we find ALL possible cycles
            # including longer ones that might contain shorter cycles as sub-paths
            if iteration == 1 and len(all_possible_polygons) > 0:
                print(f"  [STEP 6.1] First iteration: Found {len(all_possible_polygons)} polygon(s)")
                print(f"  [STEP 6.1] Polygon sizes: {[len(p) for p in all_possible_polygons]}")
                
                # Check if we might be missing larger cycles
                # A larger cycle exists if there are edges not in any found polygon
                edges_in_polygons = set()
                for poly in all_possible_polygons:
                    for i in range(len(poly)):
                        v1 = poly[i]
                        v2 = poly[(i + 1) % len(poly)]
                        edge = (min(v1, v2), max(v1, v2))
                        edges_in_polygons.add(edge)
                
                unused_edges_set = set(remaining_edges_list) - edges_in_polygons
                if len(unused_edges_set) > 0:
                    print(f"  [STEP 6.1] WARNING: {len(unused_edges_set)} edges not in any polygon: {unused_edges_set}")
                    print(f"  [STEP 6.1] These edges might form a larger cycle that was missed")
                    
                    # Diagnostic: show which edges are used vs unused
                    print(f"  [STEP 6.1] All input edges: {sorted(remaining_edges_list)}")
                    print(f"  [STEP 6.1] Edges in found polygons: {sorted(edges_in_polygons)}")
                    print(f"  [STEP 6.1] Note: DFS may be closing smaller cycles early, missing larger ones")
            
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
                    
                    # [STEP 6.1] Check connectivity (require conn == 3 OR split edge)
                    closing_edge = (min(ep1, ep2), max(ep1, ep2))
                    edge_conn_value = 0
                    is_split_edge = closing_edge in split_edges_from_high_conn
                    
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
                    
                    if merged_conn is None or edge_conn_value == 3 or is_split_edge:
                        if is_split_edge and edge_conn_value != 3:
                            print(f"  [STEP 6.1] Adding closing edge ({ep1}, {ep2}) [split from high-conn edge]")
                        else:
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
                        # Try to find indirect path through used edges
                        print(f"  [STEP 6.1] Closing edge invalid "
                              f"(conn={edge_conn_value}, need 3)")
                        print(f"  [STEP 6.1] Checking for indirect path through used edges...")
                        
                        # Build adjacency from ALL edges in the original face (used + remaining)
                        all_face_edges_adj = {}
                        for edge in edge_set:  # All original edges for this face
                            v1, v2 = edge
                            all_face_edges_adj.setdefault(v1, []).append(v2)
                            all_face_edges_adj.setdefault(v2, []).append(v1)
                        
                        # BFS to find path from ep1 to ep2 using only edges with conn>=2
                        from collections import deque
                        queue = deque([(ep1, [ep1])])
                        visited = {ep1}
                        found_path = False
                        indirect_edges = []
                        
                        while queue and not found_path:
                            current, path = queue.popleft()
                            
                            if current == ep2:
                                found_path = True
                                # Extract edges from path
                                for i in range(len(path) - 1):
                                    v1, v2 = path[i], path[i+1]
                                    edge = (min(v1, v2), max(v1, v2))
                                    if edge not in remaining_edges:
                                        indirect_edges.append(edge)
                                break
                            
                            if current in all_face_edges_adj:
                                for neighbor in all_face_edges_adj[current]:
                                    if neighbor not in visited:
                                        # Check if edge has conn >= 2
                                        edge_tuple = (min(current, neighbor), max(current, neighbor))
                                        conn_val = 0
                                        if merged_conn is not None and current < len(merged_conn) and neighbor < len(merged_conn):
                                            conn_val = max(merged_conn[current, neighbor], merged_conn[neighbor, current])
                                        
                                        if conn_val >= 2 or edge_tuple in split_edges_from_high_conn:
                                            visited.add(neighbor)
                                            queue.append((neighbor, path + [neighbor]))
                        
                        if found_path and indirect_edges:
                            print(f"  [STEP 6.1] Found indirect path through {len(indirect_edges)} used edges")
                            print(f"  [STEP 6.1] Adding used edges to close polygon: {indirect_edges}")
                            
                            # Add the indirect path edges to remaining_edges_list
                            for edge in indirect_edges:
                                remaining_edges_list.append(edge)
                            
                            all_possible_polygons = find_all_cycles_from_edges(
                                remaining_edges_list, verts_2d)
                            
                            if all_possible_polygons:
                                print(f"  [STEP 6.1] Found "
                                      f"{len(all_possible_polygons)} polygon(s) using indirect path")
                            else:
                                print("  [STEP 6.1] ERROR: Still no polygons after adding indirect path")
                                break
                        else:
                            print(f"  [STEP 6.1] No valid indirect path found")
                            break
                elif len(endpoints) > 2:
                    # Multiple endpoints - try to connect pairs with conn=3 edges OR split edges
                    print(f"  [STEP 6.1] Found {len(endpoints)} endpoints: {endpoints}")
                    print("  [STEP 6.1] Attempting to connect endpoint pairs with conn=3 edges or split edges...")
                    
                    # Try all possible pairings of endpoints
                    from itertools import combinations
                    added_edges = []
                    
                    if merged_conn is not None:
                        for ep1, ep2 in combinations(endpoints, 2):
                            if ep1 < len(merged_conn) and ep2 < len(merged_conn):
                                closing_edge = (min(ep1, ep2), max(ep1, ep2))
                                edge_conn_value = max(merged_conn[ep1, ep2], merged_conn[ep2, ep1])
                                is_split_edge = closing_edge in split_edges_from_high_conn
                                
                                if edge_conn_value == 3 or is_split_edge:
                                    if is_split_edge and edge_conn_value != 3:
                                        print(f"  [STEP 6.1]   Connecting V{ep1}-V{ep2} (conn={edge_conn_value}, split from high-conn edge)")
                                    else:
                                        print(f"  [STEP 6.1]   Connecting V{ep1}-V{ep2} (conn={edge_conn_value})")
                                    added_edges.append((ep1, ep2))
                                    remaining_edges_list.append((ep1, ep2))
                        
                        if added_edges:
                            print(f"  [STEP 6.1] Added {len(added_edges)} closing edge(s)")
                            all_possible_polygons = find_all_cycles_from_edges(
                                remaining_edges_list, verts_2d)
                            
                            if all_possible_polygons:
                                print(f"  [STEP 6.1] Found {len(all_possible_polygons)} polygon(s)")
                            else:
                                print("  [STEP 6.1] No polygons formed after adding edges")
                                break
                        else:
                            print("  [STEP 6.1] No conn=3 or split edges found to connect endpoints")
                            break
                    else:
                        print("  [STEP 6.1] WARNING: merged_conn not available")
                        break
                else:
                    print(f"  [STEP 6.1] ERROR: Expected 2+ endpoints, "
                          f"found {len(endpoints)}")
                    break
                
                if not all_possible_polygons:
                    break
            
            # [STEP 6.2] Expand polygons with colinear intermediate vertices
            if DEBUG_STEPS:
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
            if DEBUG_STEPS:
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
            
            if DEBUG_STEPS:
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
            
            # Store all identified polygons for visualization if enabled
            # Use plane equation as key since face numbering changes
            if FACE_VIZ_ENABLED and d_value is not None:
                # Create keys for both normal directions (face normals may get flipped later)
                plane_key = f"plane_{normal[0]:.6f}_{normal[1]:.6f}_{normal[2]:.6f}_{d_value:.6f}"
                plane_key_flipped = f"plane_{-normal[0]:.6f}_{-normal[1]:.6f}_{-normal[2]:.6f}_{-d_value:.6f}"
                
                poly_data = [p['vertices'] for p in polys_with_area]
                
                # Store for original normal direction
                if plane_key not in FACE_VIZ_DATA:
                    FACE_VIZ_DATA[plane_key] = {
                        'normal': normal,
                        'd': d_value,
                        'face_idx': face_idx if 'face_idx' in locals() else -1
                    }
                # Append rather than replace to preserve data from multiple passes
                if 'all_identified_polygons' in FACE_VIZ_DATA[plane_key]:
                    # Keep the version with most polygons
                    if len(poly_data) > len(FACE_VIZ_DATA[plane_key]['all_identified_polygons']):
                        FACE_VIZ_DATA[plane_key]['all_identified_polygons'] = poly_data
                else:
                    FACE_VIZ_DATA[plane_key]['all_identified_polygons'] = poly_data
                
                # Also store for flipped normal direction
                if plane_key_flipped not in FACE_VIZ_DATA:
                    FACE_VIZ_DATA[plane_key_flipped] = {
                        'normal': -normal,
                        'd': -d_value,
                        'face_idx': face_idx if 'face_idx' in locals() else -1
                    }
                if 'all_identified_polygons' in FACE_VIZ_DATA[plane_key_flipped]:
                    if len(poly_data) > len(FACE_VIZ_DATA[plane_key_flipped]['all_identified_polygons']):
                        FACE_VIZ_DATA[plane_key_flipped]['all_identified_polygons'] = poly_data
                else:
                    FACE_VIZ_DATA[plane_key_flipped]['all_identified_polygons'] = poly_data
                
                print(f"[VIZ DEBUG] Stored {len(polys_with_area)} identified polygons for both plane orientations...")
            
            # [STEP 6.2.6] Filter polygons with connected interiors using Shapely
            # A polygon has a "connected interior" if no other polygon's edges cross through it
            if DEBUG_STEPS:
                print(f"  [STEP 6.2.6] Filtering {len(polys_with_area)} polygon(s) for connected interiors...")
            
            from shapely.geometry import LineString, Point
            valid_polygons = []
            invalid_count = 0
            
            # Add detailed debug for first 4 polygons
            ENABLE_DETAILED_DEBUG = len(polys_with_area) > 0 and len(polys_with_area) <= 10
            
            for idx, poly_data in enumerate(polys_with_area):
                target_poly = poly_data['vertices']
                target_shapely = poly_data['shapely']
                
                if ENABLE_DETAILED_DEBUG:
                    print(f"  [STEP 6.2.6]   Checking polygon {idx}: vertices {target_poly}")
                
                # Shrink polygon slightly to get interior
                try:
                    target_interior = target_shapely.buffer(-1e-6)
                    if not target_interior.is_valid or target_interior.is_empty:
                        if ENABLE_DETAILED_DEBUG:
                            print(f"  [STEP 6.2.6]     REJECTED - invalid or empty interior after buffer")
                        invalid_count += 1
                        continue
                except Exception as e:
                    if ENABLE_DETAILED_DEBUG:
                        print(f"  [STEP 6.2.6]     REJECTED - buffer failed: {e}")
                    invalid_count += 1
                    continue
                
                target_edges = set(get_polygon_edges(target_poly))
                has_crossing_edges = False
                crossing_details = []
                
                # Check if any other polygon's edges cross this interior
                for other_idx, other_data in enumerate(polys_with_area):
                    if other_idx == idx:
                        continue
                    
                    other_poly = other_data['vertices']
                    other_edges = set(get_polygon_edges(other_poly))
                    shared_edges = target_edges & other_edges
                    unique_other_edges = other_edges - shared_edges
                    
                    # Check if unique edges from other polygon cross target interior
                    # Check if unique edges from other polygon cross target interior
                    for edge in unique_other_edges:
                        v1_coord = verts_2d[edge[0]]
                        v2_coord = verts_2d[edge[1]]
                        edge_line = LineString([v1_coord, v2_coord])
                        
                        if edge_line.intersects(target_interior):
                            has_crossing_edges = True
                            crossing_details.append(f"edge {edge} from polygon {other_idx}")
                            if not ENABLE_DETAILED_DEBUG and DEBUG_STEPS and idx < 5:  # Debug first few
                                print(f"  [STEP 6.2.6]   Polygon {idx} ({len(target_poly)} verts): "
                                      f"REJECTED - edge {edge} from polygon {other_idx} crosses interior")
                            if not ENABLE_DETAILED_DEBUG:
                                break
                    
                    if has_crossing_edges and not ENABLE_DETAILED_DEBUG:
                        break
                
                if ENABLE_DETAILED_DEBUG:
                    if has_crossing_edges:
                        print(f"  [STEP 6.2.6]     REJECTED - {len(crossing_details)} crossing edge(s):")
                        for detail in crossing_details[:5]:  # Show first 5
                            print(f"  [STEP 6.2.6]       - {detail}")
                        if len(crossing_details) > 5:
                            print(f"  [STEP 6.2.6]       ... and {len(crossing_details)-5} more")
                    else:
                        print(f"  [STEP 6.2.6]     VALID - no edges cross interior")
                
                if not has_crossing_edges:
                    valid_polygons.append(poly_data)
                    if not ENABLE_DETAILED_DEBUG and DEBUG_STEPS and idx < 5:  # Debug first few
                        print(f"  [STEP 6.2.6]   Polygon {idx} ({len(target_poly)} verts): "
                              f"VALID - no edges cross interior")
                else:
                    invalid_count += 1
            
            print(f"  [STEP 6.2.6] Filtered to {len(valid_polygons)} polygon(s) with connected interiors")
            print(f"  [STEP 6.2.6] Removed {invalid_count} polygon(s) with crossing edges")
            
            if not valid_polygons:
                print("  [STEP 6.2.6] ERROR: No polygons with connected interiors found!")
                break
            
            polys_with_area = valid_polygons
            
            # Store selected polygons for visualization if enabled
            if FACE_VIZ_ENABLED and d_value is not None:
                plane_key = f"plane_{normal[0]:.6f}_{normal[1]:.6f}_{normal[2]:.6f}_{d_value:.6f}"
                plane_key_flipped = f"plane_{-normal[0]:.6f}_{-normal[1]:.6f}_{-normal[2]:.6f}_{-d_value:.6f}"
                poly_data = [p['vertices'] for p in polys_with_area]
                
                # Store for both normal directions
                if plane_key in FACE_VIZ_DATA:
                    FACE_VIZ_DATA[plane_key]['selected_polygons'] = poly_data
                if plane_key_flipped in FACE_VIZ_DATA:
                    FACE_VIZ_DATA[plane_key_flipped]['selected_polygons'] = poly_data
            
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
            
            if DEBUG_STEPS:
                print(f"  [STEP 6.3] Remaining polygons: {len(all_other_polygons)}")
            
            # [STEP 6.4] Identify polygons with connected interiors
            # A polygon has a "connected interior" if no other polygon's edges cross through it
            if DEBUG_STEPS:
                print(f"  [STEP 6.4] Identifying polygons with connected interiors...")
            
            # Normalize all polygons for consistent comparison
            boundary_poly = normalize_polygon(boundary_poly)
            all_other_polygons = [normalize_polygon(p) for p in all_other_polygons]
            all_polygons_with_largest = [boundary_poly] + all_other_polygons
            
            # Helper function to check if polygon has connected interior
            def has_connected_interior(poly_idx, all_polys):
                """
                Check if polygon at poly_idx has a connected interior
                (no other polygon's edges cross through its interior)
                """
                target_poly = all_polys[poly_idx]
                target_2d = [verts_2d[v] for v in target_poly]
                
                try:
                    target_shapely = Polygon(target_2d)
                    if not target_shapely.is_valid or target_shapely.area < 1e-6:
                        return False
                    
                    # Shrink polygon slightly to get interior
                    target_interior = target_shapely.buffer(-1e-6)
                    if not target_interior.is_valid:
                        return False
                    
                    target_edges = set(get_polygon_edges(target_poly))
                    
                    # Check if any other polygon's edges cross this interior
                    from shapely.geometry import LineString
                    for other_idx, other_poly in enumerate(all_polys):
                        if other_idx == poly_idx:
                            continue
                        
                        other_edges = set(get_polygon_edges(other_poly))
                        shared_edges = target_edges & other_edges
                        unique_other_edges = other_edges - shared_edges
                        
                        # Check if unique edges from other polygon cross target interior
                        for edge in unique_other_edges:
                            v1_coord = verts_2d[edge[0]]
                            v2_coord = verts_2d[edge[1]]
                            edge_line = LineString([v1_coord, v2_coord])
                            
                            if edge_line.intersects(target_interior):
                                return False  # Another polygon's edge crosses interior
                    
                    return True  # No edges cross interior
                    
                except Exception as e:
                    return False
            
            # Identify all polygons with connected interiors
            connected_interior_indices = []
            for idx in range(len(all_polygons_with_largest)):
                if has_connected_interior(idx, all_polygons_with_largest):
                    connected_interior_indices.append(idx)
            
            if DEBUG_STEPS:
                print(f"  [STEP 6.4] Found {len(connected_interior_indices)} polygon(s) with connected interiors")
                for idx in connected_interior_indices:
                    poly = all_polygons_with_largest[idx]
                    print(f"  [STEP 6.4]   Polygon {idx}: {len(poly)} vertices")
            
            # Now classify polygons based on containment relationships
            # Polygons with connected interiors can be: BOUNDARY, HOLE, or SEPARATE FACE
            # Other polygons are: ALTERNATES or ARTIFACTS
            
            alternates_candidates = []
            invalid_polygons = []
            holes_list = []
            separate_faces = []
            
            # Process polygons with connected interiors first
            boundary_candidates = []
            for idx in connected_interior_indices:
                poly = all_polygons_with_largest[idx]
                poly_2d = [verts_2d[v] for v in poly]
                
                try:
                    poly_shapely = Polygon(poly_2d)
                    boundary_candidates.append((idx, poly, poly_shapely))
                except:
                    continue
            
            # Find the largest polygon as initial boundary
            if boundary_candidates:
                boundary_candidates.sort(key=lambda x: x[2].area, reverse=True)
                boundary_idx, boundary_poly, boundary_shapely = boundary_candidates[0]
                
                print(f"  [STEP 6.4] Selected polygon {boundary_idx} as BOUNDARY ({len(boundary_poly)} verts, area={boundary_shapely.area:.2f})")
                
                # Classify remaining connected-interior polygons relative to boundary
                for idx, poly, poly_shapely in boundary_candidates[1:]:
                    # Validate geometries before containment check
                    if not boundary_shapely.is_valid:
                        boundary_shapely = boundary_shapely.buffer(0)
                    if not poly_shapely.is_valid:
                        poly_shapely = poly_shapely.buffer(0)
                    
                    try:
                        is_contained = boundary_shapely.contains(poly_shapely)
                    except:
                        # If containment check fails, use centroid-based check
                        try:
                            is_contained = boundary_shapely.contains(poly_shapely.centroid)
                        except:
                            is_contained = False
                    
                    if is_contained:
                        # Check if polygon shares edges with boundary
                        poly_edges = set(get_polygon_edges(poly))
                        boundary_edges = set(get_polygon_edges(boundary_poly))
                        shared_edges = poly_edges & boundary_edges
                        
                        if len(shared_edges) > 0:
                            # Polygon shares edges with boundary - merge vertices into boundary
                            print(f"  [STEP 6.4] Polygon {idx} shares {len(shared_edges)} edge(s) with boundary")
                            print(f"  [STEP 6.4]   Shared edges: {shared_edges}")
                            
                            # Insert poly vertices into boundary at the shared edge location
                            for shared_edge in shared_edges:
                                v1, v2 = shared_edge
                                # Find where this edge appears in boundary
                                for i in range(len(boundary_poly)):
                                    b_v1 = boundary_poly[i]
                                    b_v2 = boundary_poly[(i + 1) % len(boundary_poly)]
                                    b_edge = (min(b_v1, b_v2), max(b_v1, b_v2))
                                    
                                    if b_edge == shared_edge:
                                        # Find intermediate vertices in poly between v1 and v2
                                        poly_v1_idx = poly.index(v1) if v1 in poly else (poly.index(v2) if v2 in poly else None)
                                        if poly_v1_idx is not None:
                                            # Get vertices between v1 and v2 in poly
                                            intermediate_verts = []
                                            curr_idx = poly_v1_idx
                                            while True:
                                                next_idx = (curr_idx + 1) % len(poly)
                                                next_v = poly[next_idx]
                                                if next_v == v1 or next_v == v2:
                                                    if next_v == v2 or next_v == v1:
                                                        break
                                                else:
                                                    intermediate_verts.append(next_v)
                                                curr_idx = next_idx
                                                if curr_idx == poly_v1_idx:  # Prevent infinite loop
                                                    break
                                            
                                            # Insert intermediate vertices into boundary
                                            if intermediate_verts:
                                                print(f"  [STEP 6.4]   Inserting vertices {intermediate_verts} into boundary at edge {shared_edge}")
                                                # Insert after position i
                                                for j, v in enumerate(intermediate_verts):
                                                    boundary_poly.insert(i + 1 + j, v)
                                                break
                                break
                            
                            # Update boundary_shapely after modification
                            boundary_2d = [verts_2d[v] for v in boundary_poly]
                            try:
                                boundary_shapely = Polygon(boundary_2d)
                                if not boundary_shapely.is_valid:
                                    boundary_shapely = boundary_shapely.buffer(0)
                            except:
                                pass
                            
                            print(f"  [STEP 6.4]   Updated boundary: {len(boundary_poly)} vertices")
                        
                        # Check if it's contained by another hole (hole-within-hole = separate face)
                        is_hole_within_hole = False
                        for hole in holes_list:
                            hole_2d = [verts_2d[v] for v in hole]
                            try:
                                hole_shapely = Polygon(hole_2d)
                                if not hole_shapely.is_valid:
                                    hole_shapely = hole_shapely.buffer(0)
                                if hole_shapely.contains(poly_shapely):
                                    is_hole_within_hole = True
                                    break
                            except:
                                continue
                        
                        if is_hole_within_hole:
                            print(f"  [STEP 6.4] Polygon {idx}: SEPARATE FACE (hole within hole)")
                            print(f"  [STEP 6.4]   Vertices: {poly}")
                            separate_faces.append(poly)
                        else:
                            print(f"  [STEP 6.4] Polygon {idx}: HOLE (contained within boundary)")
                            print(f"  [STEP 6.4]   Vertices: {poly}")
                            holes_list.append(poly)
                    else:
                        print(f"  [STEP 6.4] Polygon {idx}: SEPARATE FACE (not contained)")
                        print(f"  [STEP 6.4]   Vertices: {poly}")
                        separate_faces.append(poly)
            
            # Process polygons WITHOUT connected interiors
            # These are alternates (share edges with boundary) or artifacts
            for idx, poly in enumerate(all_polygons_with_largest):
                if idx in connected_interior_indices:
                    continue  # Already processed
                
                poly_2d = [verts_2d[v] for v in poly]
                
                try:
                    poly_shapely = Polygon(poly_2d)
                    if not poly_shapely.is_valid or poly_shapely.area < 1e-6:
                        print(f"  [STEP 6.4] Polygon {idx}: INVALID (self-intersecting or zero area) - deleting")
                        invalid_polygons.append(poly)
                        continue
                except Exception as e:
                    print(f"  [STEP 6.4] Polygon {idx}: ERROR creating Shapely polygon - deleting: {e}")
                    invalid_polygons.append(poly)
                    continue
                
                # Check relationship with boundary
                poly_edges = set(get_polygon_edges(poly))
                boundary_edges = set(get_polygon_edges(boundary_poly))
                shared_edges = poly_edges & boundary_edges
                
                try:
                    intersection = boundary_shapely.intersection(poly_shapely)
                    intersection_area = intersection.area if hasattr(intersection, 'area') else 0
                except:
                    intersection_area = 0
                
                if len(shared_edges) > 0:
                    # Shares edges with boundary
                    if intersection_area < 1e-6:
                        # Touches along edges but doesn't overlap (alternate boundary definition)
                        print(f"  [STEP 6.4] Polygon {idx}: ALTERNATE (shares {len(shared_edges)} edges, no interior overlap)")
                        print(f"  [STEP 6.4]   Vertices: {poly}")
                        alternates_candidates.append(poly)
                    else:
                        # Overlaps and shares edges (artifact)
                        print(f"  [STEP 6.4] Polygon {idx}: ARTIFACT (overlaps boundary + shares {len(shared_edges)} edges) - deleting")
                        print(f"  [STEP 6.4]   Vertices: {poly}, intersection area={intersection_area:.6f}")
                        invalid_polygons.append(poly)
                else:
                    # Doesn't share edges
                    if boundary_shapely.contains(poly_shapely):
                        print(f"  [STEP 6.4] Polygon {idx}: ARTIFACT (contained but no connected interior) - deleting")
                        invalid_polygons.append(poly)
                    else:
                        print(f"  [STEP 6.4] Polygon {idx}: ARTIFACT (disconnected, no shared edges) - deleting")
                        invalid_polygons.append(poly)
            
            # Update all_other_polygons to include separate faces AND alternates
            # All connected-interior polygons should be kept for later classification
            all_other_polygons = separate_faces.copy() + alternates_candidates.copy()
            
            if DEBUG_STEPS:
                print(f"  [STEP 6.4] Classification complete:")
                print(f"  [STEP 6.4]   - Boundary: 1 polygon ({len(boundary_poly)} verts)")
                print(f"  [STEP 6.4]   - Holes: {len(holes_list)} polygon(s)")
                print(f"  [STEP 6.4]   - Alternates: {len(alternates_candidates)} polygon(s)")
                print(f"  [STEP 6.4]   - Separate faces: {len(separate_faces)} polygon(s)")
                print(f"  [STEP 6.4]   - Invalid/deleted: {len(invalid_polygons)} polygon(s)")
                print(f"  [STEP 6.4]   - Total to process further: {len(all_other_polygons)} polygon(s)")
            
            # [STEP 6.5] Compare remaining polygons against each other
            # NOTE: Since Step 6.2.6 filtered to connected interiors only,
            # all remaining polygons are valid touching/separate faces
            # Skip comparison logic to keep all of them
            if DEBUG_STEPS:
                print(f"  [STEP 6.5] Checking {len(all_other_polygons)} "
                      f"remaining polygon(s) against each other")
                print(f"  [STEP 6.5] All polygons have connected interiors - keeping all as separate touching faces")
            
            if False and len(all_other_polygons) > 1:  # Disabled - keep all connected-interior polygons
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
            
            if DEBUG_STEPS:
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
                            # Touching polygons are alternates - keep them for later
                            print(f"  [STEP 6.6] Polygon {len(poly)} verts: "
                                  f"TOUCHING (outside) - keeping as alternate")
                            outside_polygons.append(poly)
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
                # [STEP 6.8.0] Check for subset alternates that should replace
                # boundary When an alternate's vertices are a complete subset
                # of the boundary's vertices AND they share edges, the
                # alternate is the true interior boundary and should replace
                # the larger boundary polygon
                boundary_verts_set = set(boundary_poly)
                
                for alt_idx, alt_verts in enumerate(alternates_candidates):
                    alt_verts_set = set(alt_verts)
                    
                    # Check if all alternate vertices are in boundary
                    if alt_verts_set.issubset(boundary_verts_set):
                        # Check if they share edges
                        alt_edges = set(get_polygon_edges(alt_verts))
                        boundary_edges = set(get_polygon_edges(boundary_poly))
                        shared_edges = alt_edges & boundary_edges
                        
                        if len(shared_edges) > 0:
                            # Alternate is subset and shares edges
                            # -> it's the interior, replace boundary
                            print(f"  [STEP 6.8.0] Alternate {alt_idx+1} is "
                                  f"subset of boundary (shares "
                                  f"{len(shared_edges)} edges)")
                            print(f"  [STEP 6.8.0]   Alternate vertices: "
                                  f"{alt_verts}")
                            print(f"  [STEP 6.8.0]   Boundary vertices: "
                                  f"{boundary_poly}")
                            print(f"  [STEP 6.8.0]   Replacing boundary "
                                  f"with alternate {alt_idx+1}")
                            
                            # Move boundary to alternates
                            final_alternates = [boundary_poly]
                            # Add other alternates (excluding this one)
                            for other_idx, other_alt in enumerate(
                                    alternates_candidates):
                                if other_idx != alt_idx:
                                    final_alternates.append(other_alt)
                            
                            # Set this alternate as new boundary
                            final_boundary = alt_verts
                            break
                
                # If no subset replacement occurred, use edge-sharing logic
                if not final_alternates:
                    def count_shared_edges_between(poly1_verts,
                                                   poly2_verts):
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
                        {'vertices': boundary_poly,
                         'name': 'merged_boundary'}
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
                        print(f"  [STEP 6.8] Selecting "
                              f"{best_candidate['name']} "
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
            
            # [STEP 6.8.1] Post-selection check for subset alternates
            if final_alternates:
                final_boundary_verts_set = set(final_boundary)
                
                for alt_idx, alt_verts in enumerate(final_alternates):
                    alt_verts_set = set(alt_verts)
                    
                    # Check if all alternate vertices are in final boundary
                    if alt_verts_set.issubset(final_boundary_verts_set):
                        # Check if they share edges
                        alt_edges = set(get_polygon_edges(alt_verts))
                        boundary_edges = set(get_polygon_edges(
                            final_boundary))
                        shared_edges = alt_edges & boundary_edges
                        
                        if len(shared_edges) > 0:
                            # Alternate is subset and shares edges
                            # -> it's the interior, replace boundary
                            print(f"  [STEP 6.8.1] Alternate {alt_idx+1} "
                                  f"is subset of final boundary "
                                  f"(shares {len(shared_edges)} edges)")
                            print(f"  [STEP 6.8.1]   Alternate: "
                                  f"{alt_verts}")
                            print(f"  [STEP 6.8.1]   Final boundary: "
                                  f"{final_boundary}")
                            print("  [STEP 6.8.1]   Replacing final "
                                  "boundary with subset alternate")
                            
                            # Move current final_boundary to alternates
                            new_alternates = [final_boundary]
                            # Add other alternates (excluding this one)
                            for other_idx, other_alt in enumerate(
                                    final_alternates):
                                if other_idx != alt_idx:
                                    new_alternates.append(other_alt)
                            
                            # Set this alternate as new final boundary
                            final_boundary = alt_verts
                            final_alternates = new_alternates
                            break
            
            # [STEP 6.8.2] Remove polygons cut by other polygon edges
            if final_alternates:
                try:
                    from shapely.geometry import LineString
                    
                    # Build Shapely polygon for final boundary
                    final_boundary_coords = [verts_2d[v]
                                             for v in final_boundary]
                    final_boundary_shapely = Polygon(final_boundary_coords)
                    
                    if final_boundary_shapely.is_valid:
                        valid_alternates = []
                        boundary_replaced = False
                        
                        for alt_idx, alt_verts in enumerate(final_alternates):
                            # Build Shapely polygon for alternate
                            try:
                                alt_coords = [verts_2d[v] for v in alt_verts]
                                alt_shapely = Polygon(alt_coords)
                                
                                if not alt_shapely.is_valid:
                                    print(f"  [STEP 6.8.2] Alternate "
                                          f"{alt_idx+1} is invalid - "
                                          f"removing")
                                    continue
                                
                                # Check if boundary edges cut through alternate
                                boundary_edges = get_polygon_edges(
                                    final_boundary)
                                alt_edges = set(get_polygon_edges(alt_verts))
                                
                                # Get unique boundary edges not shared
                                boundary_edge_set = set(boundary_edges)
                                shared_edges = boundary_edge_set & alt_edges
                                unique_boundary_edges = (
                                    boundary_edge_set - shared_edges
                                )
                                
                                alt_interior = alt_shapely.buffer(-1e-6)
                                boundary_cuts_alternate = False
                                
                                if (unique_boundary_edges and
                                        alt_interior.is_valid):
                                    for edge in unique_boundary_edges:
                                        v1_coord = verts_2d[edge[0]]
                                        v2_coord = verts_2d[edge[1]]
                                        edge_line = LineString(
                                            [v1_coord, v2_coord]
                                        )
                                        
                                        if edge_line.intersects(alt_interior):
                                            boundary_cuts_alternate = True
                                            print(f"  [STEP 6.8.2] Boundary "
                                                  f"edge {edge} cuts through "
                                                  f"alternate {alt_idx+1} - "
                                                  f"removing alternate")
                                            print(f"  [STEP 6.8.2]   "
                                                  f"Alternate: {alt_verts}")
                                            break
                                
                                # Check if alternate edges cut through boundary
                                unique_alt_edges = alt_edges - shared_edges
                                boundary_interior = (
                                    final_boundary_shapely.buffer(-1e-6)
                                )
                                alternate_cuts_boundary = False
                                
                                if (unique_alt_edges and
                                        boundary_interior.is_valid):
                                    for edge in unique_alt_edges:
                                        v1_coord = verts_2d[edge[0]]
                                        v2_coord = verts_2d[edge[1]]
                                        edge_line = LineString(
                                            [v1_coord, v2_coord]
                                        )
                                        
                                        if edge_line.intersects(
                                                boundary_interior):
                                            alternate_cuts_boundary = True
                                            print(f"  [STEP 6.8.2] Alternate "
                                                  f"{alt_idx+1} edge {edge} "
                                                  f"cuts through boundary - "
                                                  f"replacing boundary")
                                            print(f"  [STEP 6.8.2]   "
                                                  f"Alternate: {alt_verts}")
                                            break
                                
                                if boundary_cuts_alternate:
                                    # Boundary cuts alternate - remove it
                                    continue
                                elif alternate_cuts_boundary:
                                    # Alternate cuts boundary - replace
                                    # boundary with alternate
                                    if not boundary_replaced:
                                        valid_alternates.append(
                                            final_boundary
                                        )
                                        final_boundary = alt_verts
                                        boundary_replaced = True
                                    else:
                                        # Already replaced once, keep as alt
                                        valid_alternates.append(alt_verts)
                                else:
                                    # No cutting - keep alternate
                                    valid_alternates.append(alt_verts)
                            
                            except Exception as e:
                                print(f"  [STEP 6.8.2] Error checking "
                                      f"alternate {alt_idx+1}: {e}")
                                valid_alternates.append(alt_verts)
                        
                        if len(valid_alternates) < len(final_alternates):
                            removed_count = (
                                len(final_alternates) - len(valid_alternates)
                            )
                            print(f"  [STEP 6.8.2] Removed "
                                  f"{removed_count} "
                                  f"invalid alternates")
                            final_alternates = valid_alternates
                        
                        if boundary_replaced:
                            print("  [STEP 6.8.2] Boundary was replaced")
                    else:
                        print("  [STEP 6.8.2] Final boundary invalid - "
                              "keeping all alternates")
                
                except Exception as e:
                    print(f"  [STEP 6.8.2] Error during overlap check: {e}")
            
            # [STEP 6.8.3] Validate that all polygon vertices lie on the face plane
            plane_tolerance = 0.2  # 0.2mm tolerance for vertices to be on plane
            normal = face_eq['normal']
            d = face_eq['d']
            
            def validate_polygon_on_plane(poly_verts, poly_name="polygon"):
                """Check if all vertices of polygon lie on face plane"""
                invalid_verts = []
                for v_idx in poly_verts:
                    vertex = selected_verts[v_idx]
                    distance = abs(np.dot(normal, vertex) + d)
                    if distance >= plane_tolerance:
                        invalid_verts.append((v_idx, distance))
                return invalid_verts
            
            # Validate boundary
            invalid_boundary = validate_polygon_on_plane(final_boundary, "boundary")
            if invalid_boundary:
                print(f"  [STEP 6.8.3] WARNING: Boundary polygon has {len(invalid_boundary)} vertices NOT on plane!")
                for v_idx, dist in invalid_boundary:
                    v = selected_verts[v_idx]
                    print(f"    Vertex {v_idx} at [{v[0]:.1f}, {v[1]:.1f}, {v[2]:.1f}]: distance {dist:.3f}mm from plane")
                print(f"  [STEP 6.8.3]   Face plane: n=[{normal[0]:.3f}, {normal[1]:.3f}, {normal[2]:.3f}], d={d:.2f}")
                print(f"  [STEP 6.8.3]   SKIPPING THIS FACE - invalid topology")
                continue  # Skip this face entirely if boundary is invalid
            
            # Validate holes
            valid_holes = []
            for hole_idx, hole in enumerate(holes):
                invalid_hole = validate_polygon_on_plane(hole, f"hole {hole_idx+1}")
                if invalid_hole:
                    print(f"  [STEP 6.8.3] WARNING: Hole {hole_idx+1} has {len(invalid_hole)} vertices NOT on plane - removing hole")
                    for v_idx, dist in invalid_hole:
                        v = selected_verts[v_idx]
                        print(f"    Vertex {v_idx} at [{v[0]:.1f}, {v[1]:.1f}, {v[2]:.1f}]: distance {dist:.3f}mm from plane")
                else:
                    valid_holes.append(hole)
            
            # Validate alternates and add outside polygons as alternates
            valid_alternates = []
            if final_alternates:
                for alt_idx, alt in enumerate(final_alternates):
                    invalid_alt = validate_polygon_on_plane(alt, f"alternate {alt_idx+1}")
                    if invalid_alt:
                        print(f"  [STEP 6.8.3] WARNING: Alternate {alt_idx+1} has {len(invalid_alt)} vertices NOT on plane - removing alternate")
                        for v_idx, dist in invalid_alt:
                            v = selected_verts[v_idx]
                            print(f"    Vertex {v_idx} at [{v[0]:.1f}, {v[1]:.1f}, {v[2]:.1f}]: distance {dist:.3f}mm from plane")
                    else:
                        valid_alternates.append(alt)
            
            # Add outside polygons as alternates (they are touching/separate faces)
            if outside_polygons:
                print(f"  [STEP 6.8.3] Adding {len(outside_polygons)} outside polygons as alternates")
                for poly_idx, poly in enumerate(outside_polygons):
                    invalid_poly = validate_polygon_on_plane(poly, f"outside polygon {poly_idx+1}")
                    if invalid_poly:
                        print(f"  [STEP 6.8.3] WARNING: Outside polygon {poly_idx+1} has {len(invalid_poly)} vertices NOT on plane - skipping")
                        for v_idx, dist in invalid_poly:
                            v = selected_verts[v_idx]
                            print(f"    Vertex {v_idx} at [{v[0]:.1f}, {v[1]:.1f}, {v[2]:.1f}]: distance {dist:.3f}mm from plane")
                    else:
                        valid_alternates.append(poly)
                        print(f"  [STEP 6.8.3]   Added outside polygon {poly_idx+1} as alternate ({len(poly)} verts)")
            
            # [STEP 6.8] Store face with boundary and holes
            face_data = {
                'boundary': final_boundary,
                'holes': valid_holes
            }
            
            if valid_alternates:
                face_data['alternates'] = valid_alternates
                print(f"  [STEP 6.8] Stored {len(valid_alternates)} "
                      f"alternates (validated)")
            
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
        
        if DEBUG_STEPS:
            print(f"  [STEP 6] Created {len(faces)} face(s)")
        if unused_edges:
            print(f"  [STEP 6] WARNING: {len(unused_edges)} unused edges")
        
        return {
            'faces': faces,
            'unused_edges': unused_edges,
            'verts_2d': verts_2d,
            'projection': (basis_u, basis_v)
        }
    
    # Process each face and build edge-face associations
    edge_face_map = {}  # edge -> list of (face_idx, polygon_idx) tuples
    all_face_polygons = []  # List of all polygons with metadata
    
    for face_idx, face_eq in enumerate(unique_faces):
        edges = face_eq['edges_on_face']
        
        if len(edges) == 0:
            print(f"[POLY FORM]   Face {face_idx}: No edges, skipping")
            continue
        
        print(f"\n[POLY FORM]   Face {face_idx}: "
              f"Total edges available: {len(edges)}")
        
        # Build polygons using revised algorithm
        result = build_polygons_from_face_edges(
            edges, face_eq['vertices_on_face'],
            selected_vertices, face_eq['normal'], face_eq.get('d', None), merged_conn,
            split_edges_from_high_conn=split_edge_from_high_conn)
        
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
        
        # Deduplicate polygons before saving to face
        if len(polygons) > 1:
            def normalize_polygon_indices(vertices):
                """Normalize polygon by vertex indices (handles rotation and reversal)"""
                if len(vertices) < 3:
                    return vertices
                verts = list(vertices)
                min_val = min(verts)
                min_idx = verts.index(min_val)
                # Forward rotation
                forward = verts[min_idx:] + verts[:min_idx]
                # Backward rotation
                backward = [verts[min_idx]] + verts[min_idx-1::-1] + verts[:min_idx-1:-1]
                # Return lexicographically smaller
                return forward if forward <= backward else backward
            
            def polygons_equal_indices(verts1, verts2):
                """Check if polygons have same vertices (accounting for rotation/reversal)"""
                if len(verts1) != len(verts2) or set(verts1) != set(verts2):
                    return False
                return normalize_polygon_indices(verts1) == normalize_polygon_indices(verts2)
            
            original_count = len(polygons)
            to_remove = []
            for i in range(len(polygons)):
                if i in to_remove:
                    continue
                for j in range(i + 1, len(polygons)):
                    if j in to_remove:
                        continue
                    if polygons_equal_indices(polygons[i]['vertices'], polygons[j]['vertices']):
                        to_remove.append(j)
            
            # Remove duplicates from both polygons and all_face_polygons
            if to_remove:
                to_remove_set = set(to_remove)
                for idx in sorted(to_remove, reverse=True):
                    polygons.pop(idx)
                    if idx < len(all_face_polygons):
                        all_face_polygons.pop(idx)
                
                print(f"[POLY FORM]   Removed {len(to_remove)} duplicate polygon(s): {original_count} → {len(polygons)}")
        
        # Check for overlapping polygons - process sequentially
        # Each polygon is tested against all previously accepted polygons
        # If overlap > 0.01%, split the NEW polygon (not the existing ones)
        
        # Skip overlap detection if there are no polygons or only one polygon
        if len(polygons) <= 1:
            # No overlap possible with 0 or 1 polygons
            pass
        else:
            from shapely.geometry import Polygon as ShapelyPolygon
            
            # Get plane normal for this face
            plane_normal = np.array(face_eq['normal'])
            
            # Create 2D coordinate system on the plane  
            if abs(plane_normal[2]) > 0.9:  # Z-dominant
                u_axis = np.array([1, 0, 0])
            else:
                u_axis = np.array([0, 0, 1])
            u_axis = u_axis - np.dot(u_axis, plane_normal) * plane_normal
            u_axis = u_axis / np.linalg.norm(u_axis)
            v_axis = np.cross(plane_normal, u_axis)
            
            # Get 3D coordinates from selected_vertices for vertices used in this face
            vertices_in_face = set()
            for poly_data in polygons:
                vertices_in_face.update(poly_data['vertices'])
            
            # Project vertices to 2D
            verts_2d_check = {}
            plane_origin = selected_vertices[min(vertices_in_face)]
            for v_idx in vertices_in_face:
                v_3d = selected_vertices[v_idx]
                rel = v_3d - plane_origin
                u = np.dot(rel, u_axis)
                v_coord = np.dot(rel, v_axis)
                verts_2d_check[v_idx] = (u, v_coord)
            
            # Process polygons sequentially - each new one is tested against accepted ones
            accepted_polygons = []
            accepted_shapely = []
            
            for idx, poly_data in enumerate(polygons):
                poly_verts = poly_data['vertices']
                poly_2d = [verts_2d_check[v] for v in poly_verts]
                
                try:
                    poly_shapely = ShapelyPolygon(poly_2d)
                    if not poly_shapely.is_valid:
                        poly_shapely = poly_shapely.buffer(0)
                except:
                    # Invalid polygon, skip it
                    print(f"[POLY FORM]   Polygon {idx+1} is invalid - skipping")
                    continue
                
                # Test this polygon against all accepted polygons
                current_poly = poly_shapely
                overlaps_found = False
                
                for acc_idx, acc_shapely in enumerate(accepted_shapely):
                    try:
                        intersection = current_poly.intersection(acc_shapely)
                        int_area = intersection.area if hasattr(intersection, 'area') else 0
                        min_area = min(current_poly.area, acc_shapely.area)
                        overlap_ratio = (int_area / min_area) if min_area > 0 else 0
                        
                        # If overlap > 0.01%, split the current (new) polygon
                        if overlap_ratio > 0.0001:  # 0.01%
                            overlaps_found = True
                            print(f"[POLY FORM]   Polygon {idx+1} overlaps with accepted polygon {acc_idx+1} ({overlap_ratio*100:.4f}%)")
                            
                            # Split current polygon - remove intersection with accepted polygon
                            new_poly = current_poly.difference(acc_shapely)
                            if not new_poly.is_empty and new_poly.area > 1e-6:
                                current_poly = new_poly
                                print(f"[POLY FORM]     Split polygon {idx+1}: area now {current_poly.area:.2f}")
                            else:
                                # Current polygon completely contained in accepted polygon - discard it
                                current_poly = None
                                print(f"[POLY FORM]     Polygon {idx+1} completely contained - removing")
                                break
                    except Exception as e:
                        print(f"[POLY FORM]     WARNING: Failed to check overlap: {e}")
                
                # If anything is left of current polygon, accept it
                if current_poly is not None and current_poly.area > 1e-6:
                    if overlaps_found:
                        # Update area since polygon was split
                        poly_data['area'] = current_poly.area
                        
                        # Reconstruct vertex list from split polygon
                        # Map 2D coordinates back to nearest 3D vertex indices
                        split_coords = list(current_poly.exterior.coords[:-1])  # Exclude duplicate closing point
                        new_vertex_list = []
                        
                        for coord_2d in split_coords:
                            # Find closest vertex in original face vertices
                            min_dist = float('inf')
                            closest_v = None
                            for v_idx in vertices_in_face:
                                v_2d = verts_2d_check[v_idx]
                                dist = np.sqrt((v_2d[0] - coord_2d[0])**2 + (v_2d[1] - coord_2d[1])**2)
                                if dist < min_dist:
                                    min_dist = dist
                                    closest_v = v_idx
                            
                            # Accept if distance is small (existing vertex) or add as new vertex
                            if min_dist < 0.1:  # Existing vertex
                                new_vertex_list.append(closest_v)
                            else:
                                # New vertex created by intersection - keep the 2D point
                                # For now, just skip new vertices (only keep existing ones)
                                # This may result in approximate polygons
                                pass
                        
                        # Remove consecutive duplicates
                        cleaned_list = []
                        for v in new_vertex_list:
                            if not cleaned_list or v != cleaned_list[-1]:
                                cleaned_list.append(v)
                        # Check wrap-around duplicate
                        if len(cleaned_list) > 1 and cleaned_list[0] == cleaned_list[-1]:
                            cleaned_list = cleaned_list[:-1]
                        
                        if len(cleaned_list) >= 3:
                            poly_data['vertices'] = cleaned_list
                            print(f"[POLY FORM]   Polygon {idx+1} accepted after splitting (area={current_poly.area:.2f}, new verts={cleaned_list})")
                        else:
                            print(f"[POLY FORM]   Polygon {idx+1} rejected (insufficient vertices after reconstruction)")
                            continue
                    accepted_polygons.append(poly_data)
                    accepted_shapely.append(current_poly)
                elif overlaps_found:
                    print(f"[POLY FORM]   Polygon {idx+1} rejected (nothing left after splits)")
            
            polygons = accepted_polygons
            if len(accepted_polygons) < len(polygons):
                print(f"[POLY FORM]   After overlap filtering: {len(accepted_polygons)} polygons remain (from {original_count})")
        
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
    # TODO: ENHANCED CLASSIFICATION USING VISIBILITY INFORMATION
    # -------------------------------------------------------------------------
    # Future enhancement: Use connectivity matrices from individual views (top, front, side)
    # with edge visibility information (solid/visible=1, dashed/hidden=2) to classify polygons:
    #
    # 1. Determine which view(s) the face is visible in (priority: top > front > side)
    # 2. Use edge visibility from that view's connectivity matrix:
    #    - Mixed (solid + dashed) edges → treat as all dashed
    #    - All solid edges:
    #      * Check if interior is connected and not crossed by other solid polygons
    #      * If yes → BOUNDARY
    #      * Polygons touching boundary but not sharing area → ALT
    #      * Polygons completely inside boundary (not touching) → HOLES
    #      * Check if holes touch each other - if not, they're separate faces
    #    - All dashed edges → treat as solid polygons (follow rules above)
    #    - Mixed edges → First classify using only solid edges, then add dashed edges
    #
    # For now, using geometric containment as implemented below.
    # -------------------------------------------------------------------------
    
    # =========================================================================
    # NEW POLYGON CLASSIFICATION ALGORITHM USING RELATIONSHIP MATRIX
    # =========================================================================
    # Algorithm based on matrix-based relationship analysis between polygons
    # Key change: ALT polygons CANNOT have holes (must have continuous interior)
    #
    # Relationship types (for edges between polygons):
    # - S:  Shares edge(s) with another polygon
    # - ST: Shares edge(s) AND touches at single/multiple points
    # - O:  One polygon completely outside another (no containment)
    # - OT: Outside but touches at point(s) (no shared edges)
    # - I:  One polygon completely inside another (contained)
    # - IT: Inside but touches at point(s) (no shared edges)
    
    for face_idx, face_eq in enumerate(unique_faces):
        polygons = face_eq.get('polygons', [])
        if len(polygons) == 0:
            continue
        
        print(f"\n[POLY FORM]   Face {face_idx+1}: Processing {len(polygons)} polygons with relationship matrix")
        
        # Handle single polygon case
        if len(polygons) == 1:
            polygons[0]['polygon_type'] = 'BOUNDARY'
            polygons[0]['is_alternate'] = False
            polygons[0]['is_hole'] = False
            print(f"[POLY FORM]     Single polygon → BOUNDARY")
            continue
        
        # =====================================================================
        # STEP 0: Build relationship matrix
        # =====================================================================
        n = len(polygons)
        matrix = [['' for _ in range(n)] for _ in range(n)]
        
        # Create Shapely polygons for geometric analysis
        from shapely.geometry import Polygon as ShapelyPolygon
        shapely_polygons = []
        for poly_idx, poly in enumerate(polygons):
            try:
                poly_verts = poly['vertices']
                verts_2d = [face_eq['face_results']['verts_2d'][v] for v in poly_verts]
                shapely_poly = ShapelyPolygon(verts_2d)
                if not shapely_poly.is_valid:
                    shapely_poly = shapely_poly.buffer(0)
                shapely_polygons.append(shapely_poly)
            except Exception as e:
                print(f"[POLY FORM]     WARNING: Failed to create Shapely polygon {poly_idx}: {e}")
                shapely_polygons.append(None)
        
        # Build edge sets for each polygon
        edge_sets = []
        for poly in polygons:
            edges = set()
            verts = poly['vertices']
            for k in range(len(verts)):
                v1, v2 = verts[k], verts[(k+1) % len(verts)]
                # Normalize edge to (min, max) to handle both directions
                edges.add((min(v1, v2), max(v1, v2)))
            edge_sets.append(edges)
        
        # Populate matrix with relationships
        print(f"[POLY FORM]     Building {n}x{n} relationship matrix...")
        
        for i in range(n):
            for j in range(n):
                if i == j:
                    matrix[i][j] = '-'  # Self
                    continue
                
                poly_i_shapely = shapely_polygons[i]
                poly_j_shapely = shapely_polygons[j]
                
                if poly_i_shapely is None or poly_j_shapely is None:
                    matrix[i][j] = 'O'  # Default to outside
                    continue
                
                # Check for shared edges
                shared_edges = edge_sets[i] & edge_sets[j]
                has_shared_edges = len(shared_edges) > 0
                num_shared_edges = len(shared_edges)
                
                # Check geometric relationships
                # Note: In solid modeling, polygons only "touch" if they share edges, not just vertices
                try:
                    i_contains_j = poly_i_shapely.contains(poly_j_shapely)
                    j_contains_i = poly_j_shapely.contains(poly_i_shapely)
                    
                    # Check for area intersection (overlapping geometry)
                    intersection = poly_i_shapely.intersection(poly_j_shapely)
                    intersection_area = intersection.area if hasattr(intersection, 'area') else 0
                    
                    disjoint = poly_i_shapely.disjoint(poly_j_shapely)
                except Exception as e:
                    print(f"[POLY FORM]       WARNING: Shapely comparison failed for ({i},{j}): {e}")
                    i_contains_j = False
                    j_contains_i = False
                    intersection_area = 0
                    disjoint = True
                
                # Determine relationship type
                # S = Separate (disjoint, no shared edges)
                # ST = Separate but Touching (shared edges, zero intersection area)
                # O = Overlapping (area intersection > 0)
                # I = Inside (containment without edge sharing)
                # IT = Inside Touching (containment + shared edges)
                # OT = Outside Touching (from inside polygon's perspective)
                
                if i_contains_j and has_shared_edges:
                    # j is inside i and shares edges (boundary touch)
                    matrix[i][j] = 'IT'
                elif j_contains_i and has_shared_edges:
                    # i is inside j and shares edges (boundary touch from i's perspective)
                    matrix[i][j] = 'OT'
                elif has_shared_edges:
                    # Polygons share edges without containment
                    # ST = Separate but Touching (shared edges, no overlap)
                    matrix[i][j] = 'ST'
                elif i_contains_j:
                    # j is completely inside i without touching boundary
                    matrix[i][j] = 'I'
                elif j_contains_i:
                    # i is completely inside j (from i's perspective, outside)
                    matrix[i][j] = 'O'
                elif disjoint:
                    # Polygons are separate (disjoint, no shared edges, no overlap)
                    matrix[i][j] = 'S'
                else:
                    # Polygons are close/overlapping but not sharing edges
                    matrix[i][j] = 'O'
        
        # Print matrix for debugging
        print(f"[POLY FORM]     Relationship Matrix:")
        print(f"[POLY FORM]       ", end="")
        for j in range(n):
            print(f"  {j:2d}", end="")
        print()
        for i in range(n):
            print(f"[POLY FORM]       {i:2d}", end="")
            for j in range(n):
                print(f"  {matrix[i][j]:>2s}", end="")
            print()
        
        # =====================================================================
        # PREPROCESSING: Normalize and deduplicate polygons
        # =====================================================================
        print(f"[POLY FORM]     Preprocessing: Normalizing and deduplicating polygons...")
        
        def normalize_polygon(vertices):
            """Normalize polygon by comparing vertex indices directly.
            Returns normalized form with smallest vertex first and choosing
            forward/backward direction based on second vertex."""
            if len(vertices) < 3:
                return vertices
            
            # Convert to tuple for comparison
            verts = list(vertices)
            
            # Find minimum vertex index
            min_val = min(verts)
            min_idx = verts.index(min_val)
            
            # Create forward rotation (start from min vertex, go forward)
            forward = verts[min_idx:] + verts[:min_idx]
            
            # Create backward rotation (start from min vertex, go backward)
            backward = [verts[min_idx]] + verts[min_idx-1::-1] + verts[:min_idx-1:-1]
            
            # Choose the lexicographically smaller one
            # (compares element by element: first by second vertex, then third, etc.)
            if forward <= backward:
                return forward
            else:
                return backward
        
        def polygons_equal_by_vertices(verts1, verts2):
            """Check if two polygons have the same vertices (accounting for rotation and reversal)"""
            if len(verts1) != len(verts2):
                return False
            
            # Convert to sets for quick check
            if set(verts1) != set(verts2):
                return False
            
            # Normalize both and compare
            norm1 = normalize_polygon(verts1)
            norm2 = normalize_polygon(verts2)
            return norm1 == norm2
        
        # Find and remove duplicates BEFORE normalization
        # This preserves the original polygon but removes duplicates
        original_count = len(polygons)
        to_remove = []
        
        for i in range(len(polygons)):
            if i in to_remove:
                continue
            for j in range(i + 1, len(polygons)):
                if j in to_remove:
                    continue
                if polygons_equal_by_vertices(polygons[i]['vertices'], polygons[j]['vertices']):
                    print(f"[POLY FORM]       Duplicate found: Polygon {j} ({polygons[j]['vertices']}) == Polygon {i} ({polygons[i]['vertices']}), removing {j}")
                    to_remove.append(j)
        
        # Remove duplicates (in reverse order to maintain indices)
        for idx in sorted(to_remove, reverse=True):
            polygons.pop(idx)
        
        n = len(polygons)
        print(f"[POLY FORM]       Polygons after deduplication: {original_count} → {n}")
        
        # Update edge sets and matrix by removing rows/columns (instead of recomputing)
        if n != original_count:
            print(f"[POLY FORM]       Rebuilding edge sets and matrix...")
            
            # Create list of indices to keep
            to_remove_set = set(to_remove)
            keep_indices = [i for i in range(original_count) if i not in to_remove_set]
            
            # Rebuild edge_sets keeping only non-removed polygons
            edge_sets = [edge_sets[i] for i in keep_indices]
            
            # Rebuild matrix keeping only non-removed rows and columns
            matrix = [[matrix[i][j] for j in keep_indices] for i in keep_indices]
            
            # Print updated matrix
            print(f"[POLY FORM]     Updated Relationship Matrix:")
            print(f"[POLY FORM]       ", end="")
            for j in range(n):
                print(f"  {j:>2d}", end="")
            print()
            for i in range(n):
                print(f"[POLY FORM]       {i:2d}", end="")
                for j in range(n):
                    print(f"  {matrix[i][j]:>2s}", end="")
                print()
        
        # =====================================================================
        # STEP 1: Identify separate faces and boundary polygon
        # =====================================================================
        print(f"[POLY FORM]     Step 1: Identifying separate faces and boundary...")
        
        # Find polygons with all S relationships (separate faces)
        # Note: A boundary with all O from its perspective should NOT be marked as separate
        separate_faces = []
        for i in range(n):
            row = matrix[i]
            # Check if all non-self entries are 'S' (truly separate, no interactions)
            is_separate = all(cell in ['S', '-'] for cell in row)
            if is_separate:
                separate_faces.append(i)
                print(f"[POLY FORM]       Polygon {i} is a separate face (all S)")
        
        # Find boundary polygon
        # Use largest polygon by area as boundary (most conservative choice)
        boundary_candidate = None
        max_score = -1
        
        for i in range(n):
            if i in separate_faces:
                continue
            row = matrix[i]
            # Count I/IT (contains) relationships
            contains_count = sum(1 for cell in row if cell in ['I', 'IT'])
            # Count ST (shares edges) relationships
            shares_count = sum(1 for cell in row if cell == 'ST')
            # Boundary should have mix of I and S relationships
            score = contains_count * 2 + shares_count  # Weight contains higher
            
            if score > max_score:
                max_score = score
                boundary_candidate = i
        
        if boundary_candidate is None:
            # Fallback: largest polygon by area
            boundary_candidate = max(range(n), key=lambda i: polygons[i].get('area', 0))
            print(f"[POLY FORM]       No clear boundary from matrix, using largest polygon {boundary_candidate}")
        else:
            print(f"[POLY FORM]       Polygon {boundary_candidate} identified as BOUNDARY (score: {max_score})")
        
        polygons[boundary_candidate]['polygon_type'] = 'BOUNDARY'
        polygons[boundary_candidate]['is_alternate'] = False
        polygons[boundary_candidate]['is_hole'] = False
        
        # =====================================================================
        # STEP 2: Tag IT polygons for later processing
        # =====================================================================
        print(f"[POLY FORM]     Step 2: Tagging IT polygons...")
        
        it_tagged = set()
        for i in range(n):
            if i == boundary_candidate or i in separate_faces:
                continue
            
            # Check if this polygon has IT in its row
            # IT means the polygon is inside another and touching
            row = matrix[i]
            has_it = 'IT' in row
            
            if has_it:
                it_tagged.add(i)
                it_positions = [j for j, cell in enumerate(row) if cell == 'IT']
                print(f"[POLY FORM]       Polygon {i} tagged as IT (row has IT at positions {it_positions})")
        
        # Helper function to check if polygons form a closed loop
        def check_closed_loop(poly_ids, edge_sets, polygons):
            """Check if the shared edges between polygons form a closed loop matching one of the polygons"""
            if len(poly_ids) < 2:
                return False, None
            
            # Collect all shared edges between these polygons
            shared_edges = set()
            for i in range(len(poly_ids)):
                for j in range(i + 1, len(poly_ids)):
                    edges_ij = edge_sets[poly_ids[i]] & edge_sets[poly_ids[j]]
                    shared_edges.update(edges_ij)
            
            print(f"[DEBUG]         Collected {len(shared_edges)} shared edges: {sorted(shared_edges)}")
            
            if len(shared_edges) < 3:  # Need at least 3 edges to form a loop
                print(f"[DEBUG]         Too few shared edges ({len(shared_edges)} < 3)")
                return False, None
            
            # Check each polygon to see if its edges form a closed loop within the shared edges
            for pid in poly_ids:
                poly_verts = polygons[pid]['vertices']
                if len(poly_verts) < 3:
                    continue
                
                # Build edges for this polygon
                poly_edges = set()
                for i in range(len(poly_verts)):
                    v1, v2 = poly_verts[i], poly_verts[(i + 1) % len(poly_verts)]
                    # Normalize edge (smaller vertex first)
                    edge = (min(v1, v2), max(v1, v2))
                    poly_edges.add(edge)
                
                # Check if all polygon edges are in the shared edges
                if poly_edges.issubset(shared_edges):
                    print(f"[DEBUG]         Polygon {pid} edges are subset of shared edges")
                    print(f"[DEBUG]         Polygon {pid} vertices: {poly_verts}")
                    print(f"[DEBUG]         ✓ Closed loop matches polygon {pid}!")
                    return True, pid
            
            print(f"[DEBUG]         No polygon edges form a subset of shared edges")
            return False, None
        
        # =====================================================================
        # STEP 3: Analyze holes and identify merge candidates
        # =====================================================================
        print(f"[POLY FORM]     Step 3: Analyzing holes with S/ST relationships...")
        
        merge_candidates = {}  # maps polygon_id -> list of polygons to merge with it
        ot_o_tagged = set()  # polygons with OT/O (potential holes within holes)
        independent_holes = set()  # polygons with all S (independent holes)
        closed_loop_groups = []  # groups of polygons forming closed loops
        
        used_polygons = set(separate_faces) | {boundary_candidate} | it_tagged
        
        # Analyze each remaining polygon
        for i in range(n):
            if i in used_polygons:
                continue
            
            row = matrix[i]
            
            # Check if this polygon has ST with the boundary
            # ST with boundary means they share edges - this indicates overlapping regions
            # In solid modeling, polygons that share edges are typically alternate definitions
            # of the same region. We should NOT keep both as separate polygons.
            has_st_with_boundary = (matrix[i][boundary_candidate] == 'ST')
            
            # Check if this polygon has OT with the boundary
            # OT means polygon i is INSIDE boundary and shares edges with it
            # This is also an overlapping region that should be discarded
            has_ot_with_boundary = (matrix[i][boundary_candidate] == 'OT')
            
            if has_ot_with_boundary:
                # OT with boundary means this polygon is inside boundary and shares edges
                # Check for area overlap - if significant, discard it
                poly_i_shapely = shapely_polygons[i]
                boundary_shapely = shapely_polygons[boundary_candidate]
                
                if poly_i_shapely and boundary_shapely:
                    try:
                        intersection = poly_i_shapely.intersection(boundary_shapely)
                        intersection_area = intersection.area if hasattr(intersection, 'area') else 0
                        poly_i_area = poly_i_shapely.area
                        boundary_area = boundary_shapely.area
                        
                        min_area = min(poly_i_area, boundary_area)
                        overlap_ratio = intersection_area / min_area if min_area > 0 else 0
                        
                        print(f"[POLY FORM]       Polygon {i} has OT with boundary: inter_area={intersection_area:.2f}, poly_area={poly_i_area:.2f}, overlap={overlap_ratio*100:.1f}%")
                        
                        if overlap_ratio > 0.5:  # More than 50% overlap (inside and touching)
                            print(f"[POLY FORM]         → DISCARD (overlapping region inside boundary, {overlap_ratio*100:.1f}% overlap)")
                            used_polygons.add(i)
                            continue
                    except Exception as e:
                        print(f"[POLY FORM]       WARNING: Failed to check overlap for polygon {i}: {e}")
                
                # If low overlap, might be a valid alternate
                print(f"[POLY FORM]       Polygon {i} has OT with boundary (low overlap) → ALT")
                polygons[i]['polygon_type'] = 'ALT'
                polygons[i]['is_alternate'] = True
                polygons[i]['is_hole'] = False
                polygons[i]['parent_polygon'] = None
                used_polygons.add(i)
                continue
            
            if has_st_with_boundary:
                # Check the containment relationship
                # If boundary contains this polygon (matrix[i][boundary_candidate] could also show as I/IT),
                # then this is truly an alternate definition
                # But if they just touch without containment, one is likely wrong
                
                # Check if this polygon has area overlap with boundary
                poly_i_shapely = shapely_polygons[i]
                boundary_shapely = shapely_polygons[boundary_candidate]
                
                if poly_i_shapely and boundary_shapely:
                    try:
                        intersection = poly_i_shapely.intersection(boundary_shapely)
                        intersection_area = intersection.area if hasattr(intersection, 'area') else 0
                        poly_i_area = poly_i_shapely.area
                        boundary_area = boundary_shapely.area
                        
                        # If they have significant area overlap (> 10% of smaller polygon),
                        # they represent overlapping regions - discard the smaller one
                        min_area = min(poly_i_area, boundary_area)
                        overlap_ratio = intersection_area / min_area if min_area > 0 else 0
                        
                        print(f"[POLY FORM]       Polygon {i}: inter_area={intersection_area:.2f}, poly_area={poly_i_area:.2f}, boundary_area={boundary_area:.2f}, overlap={overlap_ratio*100:.1f}%")
                        
                        if overlap_ratio > 0.1:  # More than 10% overlap
                            print(f"[POLY FORM]         → DISCARD (overlapping region, {overlap_ratio*100:.1f}% of smaller polygon)")
                            used_polygons.add(i)  # Mark as used but don't assign type
                            continue
                    except Exception as e:
                        print(f"[POLY FORM]       WARNING: Failed to check overlap for polygon {i}: {e}")
                
                # If no significant overlap, treat as alternate (touching boundary)
                print(f"[POLY FORM]       Polygon {i} has ST with boundary (touching, no overlap) → ALT")
                polygons[i]['polygon_type'] = 'ALT'
                polygons[i]['is_alternate'] = True
                polygons[i]['is_hole'] = False
                polygons[i]['parent_polygon'] = None
                used_polygons.add(i)  # Mark as used so it won't be merged
                continue
            
            # Check for OT/O relationships with OTHER polygons (excluding boundary)
            # Having O/OT with boundary is normal (means inside boundary)
            has_ot_o = any(cell in ['OT', 'O'] for j, cell in enumerate(row) 
                          if j != i and j != boundary_candidate and j not in used_polygons)
            if has_ot_o:
                ot_o_tagged.add(i)
                ot_positions = [j for j, cell in enumerate(row) if cell in ['OT', 'O'] and j != boundary_candidate]
                print(f"[POLY FORM]       Polygon {i} has OT/O with other polygons at {ot_positions} → tagged for later")
                continue
            
            # Collect ST neighbors (excluding used polygons and OT/O tagged)
            st_neighbors = [j for j in range(n) 
                           if j != i and j not in used_polygons and j not in ot_o_tagged 
                           and matrix[i][j] == 'ST']
            
            # Check if all relationships are S (independent hole)
            non_used_cells = [(j, cell) for j, cell in enumerate(row) 
                             if j not in used_polygons and j not in ot_o_tagged]
            all_s = all(cell in ['S', '-'] for j, cell in non_used_cells)
            
            if all_s:
                independent_holes.add(i)
                print(f"[POLY FORM]       Polygon {i} has all S → independent hole")
            elif st_neighbors:
                # Check if ST neighbors form a closed loop
                loop_polygons = [i] + st_neighbors
                print(f"[DEBUG]       Checking closed loop for polygon {i} with ST neighbors {st_neighbors}")
                closed_loop, matching_poly = check_closed_loop(loop_polygons, edge_sets, polygons)
                
                if closed_loop and matching_poly is not None:
                    # Only the polygon that matches the closed loop is a "hole within hole"
                    # The other polygons that contributed edges should still be considered for merging
                    print(f"[POLY FORM]       Polygon {matching_poly} forms closed loop from shared edges → hole within hole (separate face)")
                    
                    # Add only the matching polygon to closed loop groups and ot_o_tagged
                    if matching_poly not in closed_loop_groups:
                        closed_loop_groups.append([matching_poly])
                    ot_o_tagged.add(matching_poly)
                    
                    # The other polygons (except the matching one) are still merge candidates
                    remaining_neighbors = [p for p in st_neighbors if p != matching_poly]
                    if remaining_neighbors and i != matching_poly:
                        merge_candidates[i] = remaining_neighbors
                        print(f"[POLY FORM]       Polygon {i} with ST neighbors {remaining_neighbors} (excluding loop polygon {matching_poly}) → merge candidates")
                else:
                    # These are merge candidates (no closed loop)
                    merge_candidates[i] = st_neighbors
                    print(f"[POLY FORM]       Polygon {i} with ST neighbors {st_neighbors} → merge candidates (no closed loop)")

        # =====================================================================
        # STEP 3.5: Cascade ALT classification BEFORE merging
        # =====================================================================
        print(f"[POLY FORM]     Step 3.5: Cascading ALT classification (before merging)...")
        changed = True
        iteration = 0
        while changed and iteration < 10:  # Limit iterations to prevent infinite loops
            changed = False
            iteration += 1
            for i in range(n):
                if polygons[i].get('polygon_type') or i in used_polygons:
                    continue
                # Check if this polygon has ST with any ALT
                has_st_with_alt = any(
                    matrix[i][j] == 'ST' and polygons[j].get('polygon_type') == 'ALT'
                    for j in range(n) if j != i
                )
                if has_st_with_alt:
                    alt_neighbors = [j for j in range(n) if j != i and matrix[i][j] == 'ST' and polygons[j].get('polygon_type') == 'ALT']
                    print(f"[POLY FORM]       Polygon {i} shares edge with ALT {alt_neighbors} → ALT (cascade)")
                    polygons[i]['polygon_type'] = 'ALT'
                    polygons[i]['is_alternate'] = True
                    polygons[i]['is_hole'] = False
                    polygons[i]['parent_polygon'] = None
                    used_polygons.add(i)
                    # Remove from merge candidates if present
                    if i in merge_candidates:
                        del merge_candidates[i]
                    # Remove from others' merge candidate lists
                    for key in list(merge_candidates.keys()):
                        if i in merge_candidates[key]:
                            merge_candidates[key].remove(i)
                            if not merge_candidates[key]:  # Empty list after removal
                                del merge_candidates[key]
                    changed = True

        
        # =====================================================================
        # STEP 4: Validate IT-tagged polygons
        # =====================================================================
        print(f"[POLY FORM]     Step 4: Validating IT-tagged polygons...")
        
        it_to_remove = set()
        for i in it_tagged:
            # Find polygons that have IT relationship with this tagged polygon
            # IT in matrix[i][j] means polygon j is Inside and Touching polygon i
            it_related = []
            for j in range(n):
                if i != j and matrix[i][j] == 'IT':
                    it_related.append(j)
            
            print(f"[POLY FORM]       IT polygon {i} has IT relationship with: {it_related}")
            
            if len(it_related) == 0:
                # No IT relationships, keep as HOLE
                print(f"[POLY FORM]         No IT-related polygons → keep as HOLE")
                polygons[i]['polygon_type'] = 'HOLE'
                polygons[i]['is_hole'] = True
                polygons[i]['is_alternate'] = False
                polygons[i]['parent_polygon'] = 'BOUNDARY'
                continue
            
            # Check if union of IT-related polygons covers the tagged polygon
            try:
                # Project tagged polygon to 2D
                def project_to_2d_local(vertices_3d, normal):
                    """Project 3D vertices to 2D using face normal"""
                    norm = np.array(normal)
                    norm = norm / np.linalg.norm(norm)
                    if abs(norm[0]) < 0.9:
                        temp = np.array([1.0, 0.0, 0.0])
                    else:
                        temp = np.array([0.0, 1.0, 0.0])
                    u = temp - np.dot(temp, norm) * norm
                    u = u / np.linalg.norm(u)
                    v = np.cross(norm, u)
                    v = v / np.linalg.norm(v)
                    result = []
                    for vertex in vertices_3d:
                        vertex = np.array(vertex)
                        proj_u = np.dot(vertex, u)
                        proj_v = np.dot(vertex, v)
                        result.append((proj_u, proj_v))
                    return result
                
                tagged_verts = polygons[i]['vertices']
                tagged_coords_3d = [selected_vertices[v] for v in tagged_verts]
                tagged_coords_2d = project_to_2d_local(tagged_coords_3d, face_eq['normal'])
                tagged_poly_2d = ShapelyPolygon(tagged_coords_2d)
                
                # Union all IT-related polygons
                union_result = None
                for j in it_related:
                    related_verts = polygons[j]['vertices']
                    related_coords_3d = [selected_vertices[v] for v in related_verts]
                    related_coords_2d = project_to_2d_local(related_coords_3d, face_eq['normal'])
                    related_poly_2d = ShapelyPolygon(related_coords_2d)
                    
                    if union_result is None:
                        union_result = related_poly_2d
                    else:
                        union_result = union_result.union(related_poly_2d)
                
                # Check if union covers the tagged polygon (within tolerance)
                coverage = union_result.intersection(tagged_poly_2d).area / tagged_poly_2d.area
                
                print(f"[POLY FORM]         Coverage: {coverage:.2%}")
                
                if coverage > 0.95:  # 95% coverage threshold
                    print(f"[POLY FORM]         IT-related polygons cover tagged polygon → DELETE")
                    it_to_remove.add(i)
                    polygons[i]['_hidden'] = True
                    polygons[i]['_deleted'] = True
                    polygons[i]['removed'] = True
                else:
                    print(f"[POLY FORM]         EXCEPTION: IT-related polygons do NOT fully cover tagged polygon → keep as HOLE")
                    polygons[i]['polygon_type'] = 'HOLE'
                    polygons[i]['is_hole'] = True
                    polygons[i]['is_alternate'] = False
                    polygons[i]['parent_polygon'] = 'BOUNDARY'
                    
            except Exception as e:
                print(f"[POLY FORM]         Warning: Coverage check failed ({e}) → keep as HOLE")
                polygons[i]['polygon_type'] = 'HOLE'
                polygons[i]['is_hole'] = True
                polygons[i]['is_alternate'] = False
                polygons[i]['parent_polygon'] = 'BOUNDARY'
        
        # Remove IT polygons that are covered
        it_tagged -= it_to_remove
        
        # =====================================================================
        # STEP 5: Execute merges
        # =====================================================================
        print(f"[POLY FORM]     Step 5: Executing merges...")
        
        merged_polygons = {}  # maps polygon_id -> list of all merged polygon ids
        processed = set()
        
        # Iteratively merge polygons with most ST relationships
        while True:
            # Find row with most ST labels (excluding processed)
            best_row = -1
            max_st_count = 0
            
            for master, slaves in merge_candidates.items():
                if master in processed:
                    continue
                
                st_count = len(slaves)
                
                if st_count > max_st_count:
                    max_st_count = st_count
                    best_row = master
            
            # If no row found, we're done
            if best_row == -1:
                break
            
            # Execute merge for best_row
            polygons_to_merge = merge_candidates[best_row]
            
            # Initialize merged list
            if best_row not in merged_polygons:
                merged_polygons[best_row] = [best_row]
            
            # Check if any polygon to be merged is already part of another group
            groups_to_merge = set([best_row])
            for j in polygons_to_merge:
                # Find which group j belongs to (if any)
                found_group = None
                if j in merged_polygons:
                    # j is a master of its own group
                    found_group = j
                else:
                    # Check if j is a slave in any existing group
                    for master, slaves in merged_polygons.items():
                        if j in slaves:
                            found_group = master
                            break
                
                if found_group is not None and found_group != best_row:
                    groups_to_merge.add(found_group)
            
            # If multiple groups need to be merged, consolidate them
            if len(groups_to_merge) > 1:
                print(f"[POLY FORM]       Detected shared polygons - merging groups {list(groups_to_merge)}")
                all_polygons = []
                for group_master in groups_to_merge:
                    if group_master in merged_polygons:
                        all_polygons.extend(merged_polygons[group_master])
                        if group_master != best_row:
                            del merged_polygons[group_master]
                # Remove duplicates while preserving order
                seen = set()
                merged_polygons[best_row] = []
                for p in all_polygons:
                    if p not in seen:
                        seen.add(p)
                        merged_polygons[best_row].append(p)
            
            # Merge all polygons into best_row
            for j in polygons_to_merge:
                if j not in merged_polygons[best_row]:
                    if j in merged_polygons:
                        # j was already merged, add all its components
                        merged_polygons[best_row].extend(merged_polygons[j])
                        del merged_polygons[j]
                    else:
                        merged_polygons[best_row].append(j)
                
                # Mark as merged in matrix
                matrix[best_row][j] = 'DST'
                matrix[j][best_row] = 'DST'
                
                processed.add(j)
            
            print(f"[POLY FORM]       Merged polygon {best_row} with {polygons_to_merge} (total: {merged_polygons[best_row]})")
            
            # Use Shapely union to properly merge polygons
            try:
                # Helper function to project 3D vertices to 2D plane
                def project_to_2d(vertices_3d, normal):
                    """Project 3D vertices to 2D using face normal"""
                    norm = np.array(normal)
                    norm = np.array(normal)
                    norm = norm / np.linalg.norm(norm)
                    
                    # Choose a basis vector not parallel to normal
                    if abs(norm[0]) < 0.9:
                        temp = np.array([1.0, 0.0, 0.0])
                    else:
                        temp = np.array([0.0, 1.0, 0.0])
                    
                    # Create orthonormal basis on the plane
                    u = temp - np.dot(temp, norm) * norm
                    u = u / np.linalg.norm(u)
                    v = np.cross(norm, u)
                    v = v / np.linalg.norm(v)
                    
                    # Project each vertex
                    result = []
                    for vertex in vertices_3d:
                        vertex = np.array(vertex)
                        proj_u = np.dot(vertex, u)
                        proj_v = np.dot(vertex, v)
                        result.append((proj_u, proj_v))
                    return result
                
                # Get 3D coordinates for all polygons in the merge group
                merged_shapes = []
                for pid in merged_polygons[best_row]:
                    poly_verts = polygons[pid]['vertices']
                    coords_3d = [selected_vertices[v] for v in poly_verts]
                    
                    # Project to 2D using the face normal
                    coords_2d = project_to_2d(coords_3d, face_eq['normal'])
                    merged_shapes.append(ShapelyPolygon(coords_2d))
                
                # Union all polygons
                union_result = merged_shapes[0]
                for shape in merged_shapes[1:]:
                    union_result = union_result.union(shape)
                
                # Extract boundary vertices from the union
                if union_result.geom_type == 'Polygon':
                    boundary_coords_2d = list(union_result.exterior.coords)[:-1]  # Remove duplicate last point
                elif union_result.geom_type == 'MultiPolygon':
                    # Take the largest polygon if union results in multiple polygons
                    largest = max(union_result.geoms, key=lambda p: p.area)
                    boundary_coords_2d = list(largest.exterior.coords)[:-1]
                else:
                    raise ValueError(f"Union resulted in unexpected geometry type: {union_result.geom_type}")
                
                # Convert 2D coords back to vertex indices by finding closest vertices
                merged_vertex_indices = []
                for coord_2d in boundary_coords_2d:
                    # Find which vertex this 2D point corresponds to
                    min_dist = float('inf')
                    closest_v = None
                    for pid in merged_polygons[best_row]:
                        for v in polygons[pid]['vertices']:
                            v_3d = selected_vertices[v]
                            v_2d = project_to_2d([v_3d], face_eq['normal'])[0]
                            dist = ((v_2d[0] - coord_2d[0])**2 + (v_2d[1] - coord_2d[1])**2)**0.5
                            if dist < min_dist:
                                min_dist = dist
                                closest_v = v
                    if closest_v is not None and closest_v not in merged_vertex_indices:
                        merged_vertex_indices.append(closest_v)
                
                # Update the master polygon's vertices
                polygons[best_row]['vertices'] = merged_vertex_indices
                print(f"[POLY FORM]         Union result: {len(merged_vertex_indices)} vertices")
                
            except Exception as e:
                print(f"[POLY FORM]         Warning: Shapely union failed ({e}), using vertex concatenation")
                # Fallback: concatenate vertices
                all_merged_vertices = []
                for pid in merged_polygons[best_row]:
                    all_merged_vertices.extend(polygons[pid]['vertices'])
                seen = set()
                merged_vertices = []
                for v in all_merged_vertices:
                    if v not in seen:
                        seen.add(v)
                        merged_vertices.append(v)
                polygons[best_row]['vertices'] = merged_vertices
            
            # Check if merged polygon is inside boundary before marking as hole
            # If boundary contains this polygon (I or IT relationship), it's a hole
            # Otherwise, it's an alternate (separate touching face)
            if boundary_candidate is not None:
                boundary_to_merged = matrix[boundary_candidate][best_row]
                if boundary_to_merged in ['I', 'IT']:
                    # Polygon is inside boundary - it's a hole
                    polygons[best_row]['polygon_type'] = 'HOLE'
                    polygons[best_row]['is_hole'] = True
                    polygons[best_row]['is_alternate'] = False
                    polygons[best_row]['parent_polygon'] = 'BOUNDARY'
                    polygons[best_row]['merged_from'] = merged_polygons[best_row]
                else:
                    # Polygon is outside boundary - it's an alternate
                    print(f"[POLY FORM]         Merged polygon is OUTSIDE boundary (relationship: {boundary_to_merged}) → ALT not HOLE")
                    polygons[best_row]['polygon_type'] = 'ALT'
                    polygons[best_row]['is_hole'] = False
                    polygons[best_row]['is_alternate'] = True
                    polygons[best_row]['parent_polygon'] = None
                    polygons[best_row]['merged_from'] = merged_polygons[best_row]
            else:
                # No boundary defined, mark as hole (fallback)
                polygons[best_row]['polygon_type'] = 'HOLE'
                polygons[best_row]['is_hole'] = True
                polygons[best_row]['is_alternate'] = False
                polygons[best_row]['parent_polygon'] = 'BOUNDARY'
                polygons[best_row]['merged_from'] = merged_polygons[best_row]
            
            # Hide the merged slave polygons from final output
            for slave in merged_polygons[best_row]:
                if slave != best_row:
                    polygons[slave]['_merged_into'] = best_row
                    polygons[slave]['_hidden'] = True
                    polygons[slave]['removed'] = True
            
            processed.add(best_row)
        
        # Process independent holes
        for i in independent_holes:
            # Verify the polygon is actually inside the boundary
            if boundary_candidate is not None:
                boundary_to_poly = matrix[boundary_candidate][i]
                if boundary_to_poly in ['I', 'IT']:
                    print(f"[POLY FORM]       Polygon {i} → independent hole (inside boundary)")
                    polygons[i]['polygon_type'] = 'HOLE'
                    polygons[i]['is_hole'] = True
                    polygons[i]['is_alternate'] = False
                    polygons[i]['parent_polygon'] = 'BOUNDARY'
                else:
                    print(f"[POLY FORM]       Polygon {i} → marked as independent hole but OUTSIDE boundary (relationship: {boundary_to_poly}) → ALT")
                    polygons[i]['polygon_type'] = 'ALT'
                    polygons[i]['is_hole'] = False
                    polygons[i]['is_alternate'] = True
                    polygons[i]['parent_polygon'] = None
            else:
                # No boundary, mark as hole (fallback)
                print(f"[POLY FORM]       Polygon {i} → independent hole (all S)")
                polygons[i]['polygon_type'] = 'HOLE'
                polygons[i]['is_hole'] = True
                polygons[i]['is_alternate'] = False
                polygons[i]['parent_polygon'] = 'BOUNDARY'
        
        # Process closed loop polygons (hole within hole - separate faces)
        for i in ot_o_tagged:
            print(f"[POLY FORM]       Polygon {i} → closed loop (hole within hole, separate face)")
            # Mark as ALT so it gets processed as independent boundary later
            polygons[i]['polygon_type'] = 'ALT'
            polygons[i]['is_hole'] = False
            polygons[i]['is_alternate'] = True
            polygons[i]['parent_polygon'] = None
        
        # Process separate faces (all S relationships) - mark as ALT
        # These are completely separate polygons that could be alternate boundaries
        for i in separate_faces:
            if polygons[i].get('polygon_type'):
                continue  # Already classified
            print(f"[POLY FORM]       Polygon {i} → separate face (all S) → ALT")
            polygons[i]['polygon_type'] = 'ALT'
            polygons[i]['is_hole'] = False
            polygons[i]['is_alternate'] = True
            polygons[i]['parent_polygon'] = None
        
        # Process remaining unclassified polygons
        all_processed = used_polygons | it_tagged | processed | independent_holes | ot_o_tagged | set(separate_faces)
        for i in range(n):
            if i in all_processed:
                continue
            
            print(f"[POLY FORM]       Polygon {i} → unclassified, marking as ALT")
            polygons[i]['polygon_type'] = 'ALT'
            polygons[i]['is_alternate'] = True
            polygons[i]['is_hole'] = False
        
        # =====================================================================
        # STEP 6: Validate ALT constraint (no holes allowed)
        # =====================================================================
        print(f"[POLY FORM]     Step 6: Validating ALT constraint (no holes allowed)...")
        
        for i in range(n):
            if polygons[i].get('polygon_type') == 'ALT':
                # Check if any holes are inside this ALT
                for j in range(n):
                    if i == j:
                        continue
                    if polygons[j].get('polygon_type') == 'HOLE':
                        # Check if hole j is inside ALT i
                        if matrix[i][j] in ['I', 'IT']:
                            print(f"[POLY FORM]       ERROR: ALT polygon {i} contains HOLE {j} - violates constraint!")
                            print(f"[POLY FORM]       → Reclassifying ALT {i} as separate BOUNDARY face")
                            # Mark as separate face that needs to be split
                            polygons[i]['polygon_type'] = 'BOUNDARY'
                            polygons[i]['is_alternate'] = False
                            polygons[i]['is_hole'] = False
                            polygons[i]['_needs_split'] = True
                            # Mark the hole as belonging to this new face
                            polygons[j]['_parent_face'] = i
        
        # Print final classification
        print(f"[POLY FORM]     Final classification:")
        boundary_count = sum(1 for p in polygons if p.get('polygon_type') == 'BOUNDARY' and not p.get('_hidden'))
        hole_count = sum(1 for p in polygons if p.get('polygon_type') == 'HOLE' and not p.get('_hidden'))
        alt_count = sum(1 for p in polygons if p.get('polygon_type') == 'ALT' and not p.get('_hidden'))
        print(f"[POLY FORM]       BOUNDARY: {boundary_count}, HOLES: {hole_count}, ALTs: {alt_count}")
        
        for i, poly in enumerate(polygons):
            if poly.get('_hidden'):
                continue  # Skip merged slave polygons
            ptype = poly.get('polygon_type', 'UNKNOWN')
            merged_info = f" (merged from {poly['merged_from']})" if 'merged_from' in poly else ""
            print(f"[POLY FORM]       [{i}] {ptype}: {poly['vertices']}{merged_info}")
        
        # Update face_eq['polygons'] with the processed and classified polygons
        face_eq['polygons'] = polygons
    
    # ==========================================================================
    # SPLIT INDEPENDENT BOUNDARY POLYGONS INTO SEPARATE FACES
    # ==========================================================================
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
                
                # Check if this ALT polygon shares edges with boundary OR other ALT/HOLE polygons
                # If it shares edges with any of them, it's NOT independent
                shares_edge_with_others = False
                
                # Get edges of current polygon
                poly_edges = set()
                for i in range(len(poly['vertices'])):
                    v1 = poly['vertices'][i]
                    v2 = poly['vertices'][(i + 1) % len(poly['vertices'])]
                    edge = tuple(sorted([v1, v2]))
                    poly_edges.add(edge)
                
                # Check if boundary intersects or contains this polygon (geometric check)
                geometric_connection = (boundary_shapely.intersects(poly_shapely) or
                                       boundary_shapely.contains(poly_shapely))
                
                # Check for shared edges with boundary
                if boundary_poly and not shares_edge_with_others:
                    boundary_edges = set()
                    for i in range(len(boundary_poly['vertices'])):
                        v1 = boundary_poly['vertices'][i]
                        v2 = boundary_poly['vertices'][(i + 1) % len(boundary_poly['vertices'])]
                        edge = tuple(sorted([v1, v2]))
                        boundary_edges.add(edge)
                    
                    if poly_edges & boundary_edges:  # Intersection - shared edges exist
                        shares_edge_with_others = True
                
                # Check for shared edges with other ALT and HOLE polygons
                if not shares_edge_with_others:
                    for other_poly in polygons:
                        if other_poly is poly:
                            continue
                        if other_poly.get('polygon_type') not in ['ALT', 'HOLE']:
                            continue
                        
                        other_edges = set()
                        for i in range(len(other_poly['vertices'])):
                            v1 = other_poly['vertices'][i]
                            v2 = other_poly['vertices'][(i + 1) % len(other_poly['vertices'])]
                            edge = tuple(sorted([v1, v2]))
                            other_edges.add(edge)
                        
                        if poly_edges & other_edges:  # Shared edges found
                            shares_edge_with_others = True
                            break
                
                # Polygon is independent if it has NO shared edges and NO geometric connection
                if not shares_edge_with_others and not geometric_connection:
                    # This is an independent boundary polygon
                    independent_boundaries.append(poly)
                    poly['is_independent_boundary'] = True
                    print(f"[POLY FORM]     Face {face_idx+1}: Found "
                          f"independent boundary polygon {poly['vertices']}")
                else:
                    # Check if ALT is inside a HOLE (hole-in-hole case)
                    # If so, it's also an independent boundary (separate face)
                    is_inside_hole = False
                    for other_poly in polygons:
                        if other_poly.get('polygon_type') == 'HOLE' and other_poly is not poly:
                            try:
                                other_verts_2d = [face_eq['face_results']['verts_2d'][v] 
                                                 for v in other_poly['vertices']]
                                other_shapely = ShapelyPolygon(other_verts_2d)
                                if other_shapely.contains(poly_shapely):
                                    is_inside_hole = True
                                    break
                            except:
                                continue
                    
                    if is_inside_hole:
                        independent_boundaries.append(poly)
                        poly['is_independent_boundary'] = True
                        print(f"[POLY FORM]     Face {face_idx+1}: Found "
                              f"independent boundary polygon {poly['vertices']} "
                              f"(hole-in-hole)")
            except Exception as e:
                print(f"[POLY FORM]     WARNING: Failed independence test "
                      f"for polygon {poly['vertices']}: {e}")
                continue
        
        # If we found independent boundaries, create new faces for them
        # BUT FIRST: Check if any independent boundaries are holes of others
        # AND check if independent boundaries touch each other
        if independent_boundaries:
            print(f"[POLY FORM]     Face {face_idx+1}: Found "
                  f"{len(independent_boundaries)} independent "
                  f"boundary polygon(s), checking containment and connectivity...")
            
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
                        'is_hole_of': None,  # Will store index if this is hole
                        'group_id': None  # Will store connectivity group
                    })
                except:
                    # Skip if can't create Shapely polygon
                    continue
            
            # First: Check containment between independent boundaries
            # If polygon A contains polygon B, then B is a hole of A
            for i, item_i in enumerate(indep_with_shapely):
                for j, item_j in enumerate(indep_with_shapely):
                    if i == j:
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
            
            # Second: Group independent boundaries by connectivity
            # Polygons that touch/intersect should be in same group
            next_group_id = 0
            
            for i, item_i in enumerate(indep_with_shapely):
                # Skip if already grouped
                if item_i['group_id'] is not None:
                    continue
                
                # Start a new group
                item_i['group_id'] = next_group_id
                group_members = [i]
                
                # Find all polygons that connect to this group
                changed = True
                while changed:
                    changed = False
                    for j, item_j in enumerate(indep_with_shapely):
                        if item_j['group_id'] is not None:
                            continue
                        
                        # Check if j touches any member of current group
                        for member_idx in group_members:
                            member = indep_with_shapely[member_idx]
                            try:
                                # Two polygons are connected if they touch or intersect
                                if (member['shapely'].intersects(item_j['shapely']) or
                                    member['shapely'].touches(item_j['shapely'])):
                                    item_j['group_id'] = next_group_id
                                    group_members.append(j)
                                    changed = True
                                    print(f"[POLY FORM]       Polygon "
                                          f"{item_j['poly']['vertices']} "
                                          f"CONNECTED to group {next_group_id}")
                                    break
                            except:
                                continue
                
                next_group_id += 1
            
            print(f"[POLY FORM]     Found {next_group_id} "
                  f"connectivity group(s) among independent boundaries")
            
            # Now create faces: one for each connectivity group
            for group_id in range(next_group_id):
                # Get all polygons in this group (not holes of another)
                group_boundaries = [
                    item for item in indep_with_shapely
                    if item['group_id'] == group_id and item['is_hole_of'] is None
                ]
                
                if not group_boundaries:
                    continue
                
                # Create a new face for this group
                # Use the largest polygon as the boundary
                group_boundaries.sort(key=lambda x: x['shapely'].area, reverse=True)
                main_boundary = group_boundaries[0]
                
                new_face = {
                    'normal': face_eq['normal'],
                    'd': face_eq['d'],
                    'face_results': face_eq['face_results'],
                    'original_face_idx': face_idx,
                    'polygons': [],
                    'vertices_on_face': face_eq.get('vertices_on_face', []),
                    'edges_on_face': face_eq.get('edges_on_face', [])
                }
                
                # Add main boundary
                main_boundary['poly']['polygon_type'] = 'BOUNDARY'
                main_boundary['poly']['is_alternate'] = False
                main_boundary['poly']['is_hole'] = False
                new_face['polygons'].append(main_boundary['poly'])
                
                # Add other polygons in group as ALT (alternative boundaries)
                for boundary_item in group_boundaries[1:]:
                    boundary_item['poly']['polygon_type'] = 'ALT'
                    boundary_item['poly']['is_alternate'] = True
                    boundary_item['poly']['is_hole'] = False
                    new_face['polygons'].append(boundary_item['poly'])
                    boundary_item['poly']['moved_to_new_face'] = True
                
                # Add any holes that belong to polygons in this group
                for item in indep_with_shapely:
                    if item['is_hole_of'] is not None:
                        # Check if the parent is in this group
                        parent = indep_with_shapely[item['is_hole_of']]
                        if parent['group_id'] == group_id:
                            hole_poly = item['poly']
                            hole_poly['polygon_type'] = 'HOLE'
                            hole_poly['is_hole'] = True
                            hole_poly['is_alternate'] = False
                            new_face['polygons'].append(hole_poly)
                            hole_poly['moved_to_new_face'] = True
                            print(f"[POLY FORM]       Adding polygon "
                                  f"{hole_poly['vertices']} as HOLE to group {group_id}")
                
                new_faces_to_add.append(new_face)
                print(f"[POLY FORM]     Created new face for group {group_id} "
                      f"with {len(new_face['polygons'])} polygon(s)")
            
            # Remove independent boundaries and moved polygons from original
            if 'polygons' in face_eq:
                original_poly_count = len(face_eq['polygons'])
                face_eq['polygons'] = [
                    p for p in face_eq['polygons']
                    if not p.get('is_independent_boundary', False) and
                       not p.get('moved_to_new_face', False) and
                       not p.get('removed', False) and
                       not p.get('_hidden', False)
                ]
                remaining_count = len(face_eq['polygons'])
                if remaining_count > 0:
                    print(f"[POLY FORM]     Face {face_idx+1}: Keeping {remaining_count} polygon(s) "
                          f"(BOUNDARY and attached HOLES/ALTs)")
                    for p in face_eq['polygons']:
                        poly_type = p.get('polygon_type', 'UNKNOWN')
                        print(f"[POLY FORM]       Kept: {poly_type} {p['vertices']}")
            else:
                print(f"[POLY FORM]     WARNING: Face {face_idx+1} has no 'polygons' key")
    
    # Add new faces to the list
    if new_faces_to_add:
        print(f"\n[POLY FORM]   Adding {len(new_faces_to_add)} new face(s) "
              f"from independent boundaries")
        unique_faces.extend(new_faces_to_add)
    
    # Deduplicate polygons across all faces (remove duplicate faces)
    # print(f"\n[POLY FORM]   Deduplicating polygons across all faces...")
    face_polygon_map = {}  # Map polygon signature to first face that has it
    faces_to_remove = set()
    
    for face_idx, face_eq in enumerate(unique_faces):
        polygons = face_eq.get('polygons', [])
        if len(polygons) != 1:
            continue  # Only check faces with single polygons
        
        poly = polygons[0]
        poly_verts = tuple(sorted(poly['vertices']))  # Normalized signature
        
        if poly_verts in face_polygon_map:
            # Duplicate found!
            original_face_idx = face_polygon_map[poly_verts]
            # print(f"[POLY FORM]     Face {face_idx+1} is duplicate of "
            #       f"Face {original_face_idx+1} (polygon {list(poly['vertices'])})")
            faces_to_remove.add(face_idx)
        else:
            face_polygon_map[poly_verts] = face_idx
    
    # Remove duplicate faces
    if faces_to_remove:
        print(f"[POLY FORM]   Removed {len(faces_to_remove)} duplicate face(s)")
        unique_faces = [face_eq for face_idx, face_eq in enumerate(unique_faces)
                        if face_idx not in faces_to_remove]
    
    # Show initial face composition with classifications
    # print(f"\n[POLY FORM]   Initial face composition (before validation):")
    # for face_idx, face_eq in enumerate(unique_faces):
    #     polygons = face_eq.get('polygons', [])
    #     # Filter out removed polygons for display
    #     active_polygons = [p for p in polygons if not p.get('removed', False)]
    #     poly_count = len(active_polygons)
    #     
    #     if poly_count == 0:
    #         print(f"[POLY FORM]     Face {face_idx+1}: NO polygons")
    #     else:
    #         print(f"[POLY FORM]     Face {face_idx+1}: {poly_count} polygon(s)")
    #         for poly_idx, poly_data in enumerate(active_polygons):
    #             poly_verts = poly_data.get('vertices', [])
    #             poly_type = poly_data.get('polygon_type', 'UNKNOWN')
    #             print(f"[POLY FORM]       Polygon {poly_idx+1} ({poly_type}): {poly_verts}")
    
    # ==========================================================================
    # MODIFIED APPROACH: Start with ALL polygons, then remove problematic ALTs
    # ==========================================================================
    print(f"\n[POLY FORM]   Building initial edge distribution with ALL polygons (BOUNDARY + HOLES + ALL ALTs)...")
    
    edge_face_map_all = {}  # edges from ALL polygons
    all_polygons_list = []  # List of ALL polygons
    alt_polygons = []  # Track which are ALTs for removal consideration
    removed_polygon_count = 0  # Track how many were skipped
    
    for face_idx, face_eq in enumerate(unique_faces):
        polygons = face_eq.get('polygons', [])
        if len(polygons) == 0:
            continue
        
        for poly_idx, poly_data in enumerate(polygons):
            # Skip polygons that were merged into others
            if poly_data.get('removed', False):
                removed_polygon_count += 1
                continue
                
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
    if removed_polygon_count > 0:
        print(f"[POLY FORM]     Skipped {removed_polygon_count} merged/removed polygons")
    print(f"[POLY FORM]     Including {len(alt_polygons)} ALT polygons")
    
    # Check initial edge distribution (with ALL polygons)
    def compute_edge_stats(edge_map):
        """Helper to compute edge distribution statistics"""
        edges_1 = sum(1 for fl in edge_map.values() if len(fl) == 1)
        edges_2 = sum(1 for fl in edge_map.values() if len(fl) == 2)
        edges_3plus = sum(1 for fl in edge_map.values() if len(fl) >= 3)
        return edges_1, edges_2, edges_3plus
    
    def add_polygon_edges(edge_map, face_idx, poly_verts, poly_type):
        """Helper to add polygon edges to edge map"""
        for i in range(len(poly_verts)):
            v1 = poly_verts[i] - 1  # Convert to 0-based
            v2 = poly_verts[(i + 1) % len(poly_verts)] - 1
            edge = (min(v1, v2), max(v1, v2))
            if edge not in edge_map:
                edge_map[edge] = []
            # Store face_idx, polygon_idx (use i as placeholder), and type
            edge_map[edge].append((face_idx, i, poly_type))
    
    def count_ray_intersections(polygon_shapely, projection_data, polygon_normal,
                                all_faces_list, selected_vertices,
                                polygon_verts_3d, debug=False, target_verts=None):
        """
        Cast ray from bounding box through polygon and count intersections.
        Uses same logic as BOUNDARY processing in Reconstruct_Solid.py.
        
        Args:
            polygon_shapely: 2D Shapely polygon
            projection_data: Tuple (u, v) containing original basis vectors
            polygon_normal: Normal vector for the polygon
            all_faces_list: List of all face data for intersection testing
            selected_vertices: Dict of vertex index -> 3D coordinates
            polygon_verts_3d: List of 3D vertex coords for this polygon
            debug: If True, return detailed debug info
            target_verts: List of vertex indices for target polygon (to exclude from detail list)
        
        Returns:
            (count, is_valid, debug_info) where debug_info contains ray info
        """
        try:
            # Get interior point in 2D
            interior_point_2d = polygon_shapely.representative_point()
            
            # Convert 2D point back to 3D using projection basis
            if len(polygon_verts_3d) < 3:
                return (0, False, {})
            
            v0 = np.array(polygon_verts_3d[0])
            
            # Extract original basis vectors from projection data
            if projection_data is None or len(projection_data) != 2:
                return (0, False, {})
            
            u = np.array(projection_data[0])
            v_vec = np.array(projection_data[1])
            
            # Verify basis vectors are valid 3D vectors
            if u.shape != (3,) or v_vec.shape != (3,):
                return (0, False, {})
            
            # Compute normal from cross product of u and v
            basis_normal = np.cross(u, v_vec)
            norm = np.linalg.norm(basis_normal)
            if norm < 1e-9:
                return (0, False, {})
            basis_normal = basis_normal / norm
            
            # Plane constant
            d = np.dot(basis_normal, v0)
            
            # Solve for 3D interior point
            matrix = np.array([u, v_vec, basis_normal])
            rhs = np.array([interior_point_2d.x, interior_point_2d.y, d])
            
            try:
                polygon_interior = np.linalg.solve(matrix, rhs)
            except np.linalg.LinAlgError:
                return (0, False, {})
            
            # Compute bounding box of all vertices
            if isinstance(selected_vertices, dict):
                all_coords = np.array(list(selected_vertices.values()))
            else:
                all_coords = np.array(selected_vertices)
            bbox_min = np.min(all_coords, axis=0)
            bbox_max = np.max(all_coords, axis=0)
            
            # Use polygon normal (ensure it's normalized)
            poly_normal = polygon_normal / np.linalg.norm(polygon_normal)
            
            # Find dominant axis of polygon normal
            abs_normal = np.abs(poly_normal)
            dominant_axis = np.argmax(abs_normal)
            
            # Position bbox point on face perpendicular to dominant normal axis
            bbox_point = polygon_interior.copy()
            if poly_normal[dominant_axis] > 0:
                # Normal points positive, shoot from negative bbox face
                bbox_point[dominant_axis] = bbox_min[dominant_axis]
                offset = (bbox_max[dominant_axis] - bbox_min[dominant_axis]) * 0.01
                bbox_point[dominant_axis] -= offset
            else:
                # Normal points negative, shoot from positive bbox face
                bbox_point[dominant_axis] = bbox_max[dominant_axis]
                offset = (bbox_max[dominant_axis] - bbox_min[dominant_axis]) * 0.01
                bbox_point[dominant_axis] += offset
            
            # Ray from bbox along polygon normal
            ray_origin = bbox_point
            ray_direction = poly_normal / np.linalg.norm(poly_normal)
            
            # Make sure ray points towards polygon
            if np.dot(polygon_interior - bbox_point, ray_direction) < 0:
                ray_direction = -ray_direction
            
            # Collect intersections (including target polygon)
            intersections = []  # (t, face_idx, is_target)
            intersections_detail = []
            
            # First, check target polygon intersection
            target_t = None
            denom = np.dot(ray_direction, poly_normal)
            if abs(denom) > 1e-9:
                t = np.dot(v0 - ray_origin, poly_normal) / denom
                if t > 0:
                    int_point = ray_origin + t * ray_direction
                    int_2d_arr = [int_point]
                    # Simple 2D projection for point-in-polygon test
                    if abs(poly_normal[2]) > 0.5:
                        int_2d = (int_point[0], int_point[1])
                    elif abs(poly_normal[1]) > 0.5:
                        int_2d = (int_point[0], int_point[2])
                    else:
                        int_2d = (int_point[1], int_point[2])
                    
                    from shapely.geometry import Point as ShPoint
                    int_pt = ShPoint(int_2d)
                    
                    # Project polygon to same plane for containment test
                    poly_2d_coords = []
                    for pv in polygon_verts_3d:
                        if abs(poly_normal[2]) > 0.5:
                            poly_2d_coords.append((pv[0], pv[1]))
                        elif abs(poly_normal[1]) > 0.5:
                            poly_2d_coords.append((pv[0], pv[2]))
                        else:
                            poly_2d_coords.append((pv[1], pv[2]))
                    
                    from shapely.geometry import Polygon as ShPoly
                    poly_test = ShPoly(poly_2d_coords)
                    
                    if poly_test.contains(int_pt) or poly_test.touches(int_pt):
                        target_t = t
                        intersections.append((t, -1, True))
                        # Add target to detail list as well
                        intersections_detail.append({
                            'face_idx': -1,
                            't': t,
                            'polygon_verts': target_verts if target_verts else [],
                            'is_target': True
                        })
            
            if target_t is None:
                return (0, False, {})
            
            # Test against all existing faces
            for face_idx, face_data in enumerate(all_faces_list):
                polygons = face_data.get('polygons', [])
                for poly_data in polygons:
                    face_verts_indices = poly_data.get('vertices', [])
                    face_holes = poly_data.get('holes', [])
                    if len(face_verts_indices) < 3:
                        continue
                    # Get 3D coordinates for boundary
                    if isinstance(selected_vertices, dict):
                        face_verts_3d = [selected_vertices[v] for v in face_verts_indices]
                    else:
                        face_verts_3d = [selected_vertices[v] for v in face_verts_indices]
                    # Get 3D coordinates for holes
                    face_holes_3d = []
                    if face_holes:
                        for hole_verts in face_holes:
                            if isinstance(selected_vertices, dict):
                                hole_3d = [selected_vertices[v] for v in hole_verts]
                            else:
                                hole_3d = [selected_vertices[v] for v in hole_verts]
                            face_holes_3d.append(hole_3d)
                    # Compute face plane normal
                    fv0 = np.array(face_verts_3d[0])
                    fv1 = np.array(face_verts_3d[1])
                    fv2 = np.array(face_verts_3d[2])
                    face_normal = np.cross(fv1 - fv0, fv2 - fv0)
                    norm = np.linalg.norm(face_normal)
                    if norm < 1e-9:
                        continue
                    face_normal = face_normal / norm
                    # Ray-plane intersection
                    denominator = np.dot(ray_direction, face_normal)
                    if abs(denominator) < 1e-6:
                        continue  # Ray parallel to plane
                    t = np.dot(fv0 - ray_origin, face_normal) / denominator
                    if t < 1e-6:
                        continue  # Only count positive intersections
                    intersection_point = ray_origin + t * ray_direction
                    # Project to 2D for point-in-polygon test
                    from shapely.geometry import Point as ShPoint
                    from shapely.geometry import Polygon as ShPoly
                    # Use face's dominant plane for projection
                    abs_normal = np.abs(face_normal)
                    if (abs_normal[2] > abs_normal[0] and abs_normal[2] > abs_normal[1]):
                        # Project to XY plane
                        face_2d = [(v[0], v[1]) for v in face_verts_3d]
                        holes_2d = [[(h[0], h[1]) for h in hole_3d] for hole_3d in face_holes_3d]
                        int_2d = (intersection_point[0], intersection_point[1])
                    elif abs_normal[1] > abs_normal[0]:
                        # Project to XZ plane
                        face_2d = [(v[0], v[2]) for v in face_verts_3d]
                        holes_2d = [[(h[0], h[2]) for h in hole_3d] for hole_3d in face_holes_3d]
                        int_2d = (intersection_point[0], intersection_point[2])
                    else:
                        # Project to YZ plane
                        face_2d = [(v[1], v[2]) for v in face_verts_3d]
                        holes_2d = [[(h[1], h[2]) for h in hole_3d] for hole_3d in face_holes_3d]
                        int_2d = (intersection_point[1], intersection_point[2])
                    try:
                        # Create polygon with holes
                        if holes_2d:
                            face_poly = ShPoly(face_2d, holes_2d)
                        else:
                            face_poly = ShPoly(face_2d)
                        int_point = ShPoint(int_2d)
                        # Check if inside polygon (boundary) and outside holes
                        if (face_poly.contains(int_point) or face_poly.intersects(int_point)):
                            is_target = (target_verts is not None and set(face_verts_indices) == set(target_verts))
                            # Skip if it's the target (already added above)
                            if not is_target:
                                intersections.append((t, face_idx, False))
                                intersections_detail.append({
                                    'face_idx': face_idx,
                                    't': t,
                                    'polygon_verts': face_verts_indices,
                                    'is_target': False
                                })
                    except:
                        pass
            
            # Sort intersections by t value
            intersections.sort(key=lambda x: x[0])
            
            # Count unique t-values (treat polygons at same t as single intersection)
            unique_t_count = 0
            if len(intersections) > 0:
                tolerance = 1e-6
                last_t = intersections[0][0]
                unique_t_count = 1
                for i in range(1, len(intersections)):
                    current_t = intersections[i][0]
                    if abs(current_t - last_t) > tolerance:
                        unique_t_count += 1
                        last_t = current_t
            
            # Use unique t-count for parity determination
            total_count = unique_t_count
            
            debug_info = {
                'origin': ray_origin.tolist(),
                'direction': ray_direction.tolist(),
                'intersections': intersections_detail,
                'all_intersections': intersections,  # Include full list for debugging
                'total_count': total_count,
                'raw_count': len(intersections)  # For debugging
            }
            return (total_count, True, debug_info)
            
        except Exception as e:
            print(f"[POLY FORM]     Error in ray casting: {e}")
            return (0, False, {})
    
    initial_boundary, initial_manifold, initial_invalid = compute_edge_stats(edge_face_map_all)
    
    print(f"\n[POLY FORM]   ========== BASE SET STATISTICS ==========")
    print(f"[POLY FORM]   Base set: {len(all_polygons_list)} polygons (before any pruning)")
    print(f"[POLY FORM]   Initial edge distribution (ALL polygons included):")
    print(f"[POLY FORM]     - Boundary edges (1 face): {initial_boundary}")
    print(f"[POLY FORM]     - Manifold edges (2 faces): {initial_manifold}")
    print(f"[POLY FORM]     - Invalid edges (3+ faces): {initial_invalid}")
    print(f"[POLY FORM]   ==========================================")
    
    # ==========================================================================
    # ==========================================================================
    # COMBINATORIAL SEARCH: Greedy Removal Strategy
    # ==========================================================================
    print(f"\n[POLY FORM]   ========== COMBINATORIAL SEARCH (GREEDY REMOVAL) ==========")
    print(f"[POLY FORM]   Strategy: Start with ALL polygons, remove alternates one at a time")
    print(f"[POLY FORM]   Target: ≤10 invalid edges AND ≤10 boundary edges, then exhaustive search")
    
    # Build face structure with boundary, holes, and alternates
    face_polygon_groups = []
    for face_idx, face_eq in enumerate(unique_faces):
        polygons = face_eq.get('polygons', [])
        if len(polygons) == 0:
            continue
        
        boundary_poly = None
        holes = []
        alternates = []
        
        for poly_idx, poly_data in enumerate(polygons):
            if poly_data.get('removed', False):
                continue
            poly_type = poly_data.get('polygon_type', 'BOUNDARY')
            
            if poly_type == 'BOUNDARY':
                boundary_poly = {'idx': poly_idx, 'data': poly_data, 'verts': poly_data.get('vertices', [])}
            elif poly_type == 'HOLE':
                holes.append({'idx': poly_idx, 'data': poly_data, 'verts': poly_data.get('vertices', [])})
            elif poly_type == 'ALT':
                alternates.append({'idx': poly_idx, 'data': poly_data, 'verts': poly_data.get('vertices', [])})
        
        # Only include faces that have at least a boundary or alternates
        if boundary_poly or alternates:
            face_polygon_groups.append({
                'face_idx': face_idx,
                'boundary': boundary_poly,
                'holes': holes,
                'alternates': alternates
            })
    
    print(f"[POLY FORM]   Found {len(face_polygon_groups)} faces with polygons")
    
    # Identify removable alternates (boundaries with holes must be kept together)
    removable_alts = []
    mandatory_polys = []
    
    for group in face_polygon_groups:
        face_idx = group['face_idx']
        
        # Boundaries with holes are MANDATORY (must always be together)
        if group['boundary']:
            mandatory_polys.append((face_idx, group['boundary']['idx'], 'BOUNDARY', group['boundary']['verts']))
            for hole in group['holes']:
                mandatory_polys.append((face_idx, hole['idx'], 'HOLE', hole['verts']))
        
        # Alternates with holes are ALSO mandatory (holes must stay with their boundary)
        for alt in group['alternates']:
            # For now, treat all alternates as removable
            # (If an alternate has holes, we'd need to modify the data structure to track that)
            removable_alts.append((face_idx, alt['idx'], 'ALT', alt['verts']))
    
    print(f"[POLY FORM]   Mandatory polygons (boundaries + holes): {len(mandatory_polys)}")
    print(f"[POLY FORM]   Removable alternates: {len(removable_alts)}")
    
    def compute_edge_stats_for_selection(poly_list):
        """Compute edge statistics for a given polygon selection"""
        edge_map = {}
        for item in poly_list:
            if len(item) == 4:
                face_idx, poly_idx, poly_type, poly_verts = item
            else:
                continue
            
            for i in range(len(poly_verts)):
                v1 = poly_verts[i] - 1
                v2 = poly_verts[(i + 1) % len(poly_verts)] - 1
                edge = (min(v1, v2), max(v1, v2))
                if edge not in edge_map:
                    edge_map[edge] = []
                edge_map[edge].append((face_idx, poly_idx, poly_type))
        
        boundary_edges = sum(1 for fl in edge_map.values() if len(fl) == 1)
        manifold_edges = sum(1 for fl in edge_map.values() if len(fl) == 2)
        invalid_edges = sum(1 for fl in edge_map.values() if len(fl) >= 3)
        return boundary_edges, manifold_edges, invalid_edges
    
    # ==========================================================================
    # PRE-PHASE 0: Edge-based polygon cleanup (iterative)
    # ==========================================================================
    
    # Count faces with alternates
    faces_with_alts = sum(1 for group in face_polygon_groups if len(group['alternates']) > 0)
    
    # Skip Pre-Phase 0 if there are fewer than 1 faces with alternates
    if faces_with_alts < 1:
        print(f"\n[POLY FORM]   PRE-PHASE 0: Skipped (only {faces_with_alts} faces with alternates, threshold: 1)")
        total_removed = 0
    else:
        print(f"\n[POLY FORM]   PRE-PHASE 0: Analyzing edge connectivity for polygon cleanup...")
        print(f"[POLY FORM]     Found {faces_with_alts} faces with alternates")
        
        # Build initial current_selection from face_polygon_groups (all polygons)
        current_selection = []
        for group in face_polygon_groups:
            face_idx = group['face_idx']
            if group['boundary']:
                current_selection.append((face_idx, group['boundary']['idx'], 'BOUNDARY', group['boundary']['verts']))
            for hole in group['holes']:
                current_selection.append((face_idx, hole['idx'], 'HOLE', hole['verts']))
            for alt in group['alternates']:
                current_selection.append((face_idx, alt['idx'], 'ALT', alt['verts']))
        
        print(f"[POLY FORM]     Initial selection: {len(current_selection)} polygons")
        
        # Iterate up to 5 times or until no more polygons can be removed
        max_iterations = 5
        total_removed = 0
    
        for iteration in range(max_iterations):
            print(f"\n[POLY FORM]     === Cleanup Iteration {iteration + 1}/{max_iterations} ===")
            
            # Build edge-to-face mapping for all polygons in current selection
            edge_to_faces = {}  # edge -> [(face_idx, poly_idx, poly_type), ...]
            
            for item in current_selection:
                if len(item) != 4:
                    continue
                face_idx, poly_idx, poly_type, poly_verts = item
                
                # Get edges for this polygon
                for i in range(len(poly_verts)):
                    v1 = poly_verts[i] - 1
                    v2 = poly_verts[(i + 1) % len(poly_verts)] - 1
                    edge = (min(v1, v2), max(v1, v2))
                    
                    if edge not in edge_to_faces:
                        edge_to_faces[edge] = []
                    edge_to_faces[edge].append((face_idx, poly_idx, poly_type))
            
            # Classify edges by face count
            boundary_edge_set = set(edge for edge, faces in edge_to_faces.items() if len(faces) == 1)
            manifold_edge_set = set(edge for edge, faces in edge_to_faces.items() if len(faces) == 2)
            invalid_edge_set = set(edge for edge, faces in edge_to_faces.items() if len(faces) >= 3)
            
            print(f"[POLY FORM]       Edge classification: {len(boundary_edge_set)} boundary, "
                  f"{len(manifold_edge_set)} manifold, {len(invalid_edge_set)} invalid")
            
            # Annotate each polygon with its edge types
            polygon_edge_info = {}  # (face_idx, poly_idx) -> {'boundary': n, 'manifold': n, 'invalid': n, 'edges': [...]}
            
            for item in current_selection:
                if len(item) != 4:
                    continue
                face_idx, poly_idx, poly_type, poly_verts = item
                
                poly_key = (face_idx, poly_idx)
                edge_counts = {'boundary': 0, 'manifold': 0, 'invalid': 0}
                edge_list = []
                
                for i in range(len(poly_verts)):
                    v1 = poly_verts[i] - 1
                    v2 = poly_verts[(i + 1) % len(poly_verts)] - 1
                    edge = (min(v1, v2), max(v1, v2))
                    
                    # Classify this edge
                    if edge in boundary_edge_set:
                        edge_type = 'boundary'
                        edge_counts['boundary'] += 1
                    elif edge in manifold_edge_set:
                        edge_type = 'manifold'
                        edge_counts['manifold'] += 1
                    else:  # invalid
                        edge_type = 'invalid'
                        edge_counts['invalid'] += 1
                    
                    edge_list.append({'edge': edge, 'type': edge_type, 'faces': edge_to_faces[edge]})
                
                polygon_edge_info[poly_key] = {
                    **edge_counts,
                    'edges': edge_list,
                    'total_edges': len(poly_verts),
                    'poly_type': poly_type,
                    'poly_verts': poly_verts
                }
            
            # Only print detailed table on first iteration
            if iteration == 0:
                print(f"\n[POLY FORM]       ========== POLYGON EDGE ANALYSIS ==========")
                print(f"[POLY FORM]       {'Face':>4} {'Poly':>4} {'Type':>8} {'Verts':>5} {'B':>3} {'M':>3} {'I':>3} Vertices")
                print(f"[POLY FORM]       {'-'*4} {'-'*4} {'-'*8} {'-'*5} {'-'*3} {'-'*3} {'-'*3} {'-'*50}")
                
                for item in sorted(current_selection, key=lambda x: (x[0], x[1])):
                    if len(item) != 4:
                        continue
                    face_idx, poly_idx, poly_type, poly_verts = item
                    poly_key = (face_idx, poly_idx)
                    
                    if poly_key not in polygon_edge_info:
                        continue
                    
                    info = polygon_edge_info[poly_key]
                    verts_str = str(poly_verts[:10]) + ('...' if len(poly_verts) > 10 else '')
                    
                    # Use 1-based face indexing for display (face_idx+1) to match Phase 0 output
                    print(f"[POLY FORM]       {face_idx+1:>4} {poly_idx:>4} {poly_type:>8} {len(poly_verts):>5} "
                          f"{info['boundary']:>3} {info['manifold']:>3} {info['invalid']:>3} {verts_str}")
                
                print(f"[POLY FORM]       {'='*86}")
            
            # Find polygons to remove: those with NO manifold edges (only boundary/invalid)
            polygons_to_remove = set()
            
            for item in current_selection:
                if len(item) != 4:
                    continue
                face_idx, poly_idx, poly_type, poly_verts = item
                poly_key = (face_idx, poly_idx)
                
                if poly_key not in polygon_edge_info:
                    continue
                
                info = polygon_edge_info[poly_key]
                
                # Check if polygon has NO manifold edges
                if info['manifold'] == 0:
                    polygons_to_remove.add(poly_key)
                    print(f"[POLY FORM]       Marking Face {face_idx+1} poly {poly_idx} ({poly_type}) for removal: "
                          f"{info['boundary']} boundary, {info['invalid']} invalid, 0 manifold edges")
            
            # Check polygons with manifold edges: if all connected faces are also problematic, remove all
            def get_connected_problematic_faces(start_poly_key, visited=None):
                """Recursively find all polygons connected via manifold edges that have no other manifold connections."""
                if visited is None:
                    visited = set()
                
                if start_poly_key in visited or start_poly_key not in polygon_edge_info:
                    return set()
                
                visited.add(start_poly_key)
                face_idx, poly_idx = start_poly_key
                info = polygon_edge_info[start_poly_key]
                
                # Get manifold edges
                manifold_edges = [e for e in info['edges'] if e['type'] == 'manifold']
                
                if len(manifold_edges) == 0:
                    # No manifold edges, this is isolated
                    return {start_poly_key}
                
                # Check all faces connected via manifold edges
                connected_group = {start_poly_key}
                
                for edge_info in manifold_edges:
                    # Find the other face sharing this edge
                    for other_face_idx, other_poly_idx, other_type in edge_info['faces']:
                        other_key = (other_face_idx, other_poly_idx)
                        if other_key == start_poly_key or other_key in visited:
                            continue
                        
                        # Check if other polygon also has only boundary/invalid edges OR only manifold edges to this group
                        if other_key in polygon_edge_info:
                            # Recursively check connected faces
                            sub_group = get_connected_problematic_faces(other_key, visited)
                            connected_group.update(sub_group)
                
                # Check if entire connected group is isolated (no external manifold connections)
                has_external_manifold = False
                for check_key in connected_group:
                    check_info = polygon_edge_info[check_key]
                    for edge_info in check_info['edges']:
                        if edge_info['type'] == 'manifold':
                            # Check if this manifold edge connects to something outside the group
                            for other_face_idx, other_poly_idx, other_type in edge_info['faces']:
                                other_key = (other_face_idx, other_poly_idx)
                                if other_key not in connected_group:
                                    has_external_manifold = True
                                    break
                        if has_external_manifold:
                            break
                    if has_external_manifold:
                        break
                
                if not has_external_manifold:
                    return connected_group
                return set()
            
            # Find isolated groups
            already_checked = set()
            for item in current_selection:
                if len(item) != 4:
                    continue
                face_idx, poly_idx, poly_type, poly_verts = item
                poly_key = (face_idx, poly_idx)
                
                if poly_key in already_checked or poly_key in polygons_to_remove:
                    continue
                
                isolated_group = get_connected_problematic_faces(poly_key)
                if isolated_group:
                    for isolated_key in isolated_group:
                        if isolated_key not in polygons_to_remove:
                            iso_face_idx, iso_poly_idx = isolated_key
                            print(f"[POLY FORM]       Marking Face {iso_face_idx+1} poly {iso_poly_idx} for removal: "
                                  f"part of isolated group (no external manifold connections)")
                            polygons_to_remove.add(isolated_key)
                    already_checked.update(isolated_group)
            
            # Remove marked polygons from current_selection AND from unique_faces
            if polygons_to_remove:
                original_count = len(current_selection)
                current_selection = [
                    item for item in current_selection
                    if len(item) == 4 and (item[0], item[1]) not in polygons_to_remove
                ]
                removed_count = original_count - len(current_selection)
                total_removed += removed_count
                
                # Also mark polygons as removed in unique_faces
                for face_idx, poly_idx in polygons_to_remove:
                    if face_idx < len(unique_faces):
                        face_eq = unique_faces[face_idx]
                        polygons = face_eq.get('polygons', [])
                        if poly_idx < len(polygons):
                            polygons[poly_idx]['removed'] = True
                            polygons[poly_idx]['removal_reason'] = f'pre_phase0_cleanup_iter{iteration+1}'
                
                print(f"[POLY FORM]       Removed {removed_count} polygon(s) this iteration")
                print(f"[POLY FORM]       Remaining: {len(current_selection)} polygons")
                
                # Recompute edge statistics after removal
                curr_b, curr_m, curr_i = compute_edge_stats_for_selection(current_selection)
                print(f"[POLY FORM]       After iteration {iteration+1}: B={curr_b}, M={curr_m}, I={curr_i}")
            else:
                print(f"[POLY FORM]       No problematic polygons found this iteration")
                break  # Exit early if no more removals
        
        print(f"\n[POLY FORM]     PRE-PHASE 0 Complete: Removed {total_removed} polygon(s) total across {iteration+1} iteration(s)")
        print(f"[POLY FORM]     Final: {len(current_selection)} polygons remaining")
        
        # Rebuild face_polygon_groups to respect polygons removed in Pre-Phase 0
        print(f"[POLY FORM]     Rebuilding face_polygon_groups to exclude removed polygons...")
        face_polygon_groups = []
        for face_idx, face_eq in enumerate(unique_faces):
            polygons = face_eq.get('polygons', [])
            if len(polygons) == 0:
                continue
            
            boundary_poly = None
            holes = []
            alternates = []
            
            for poly_idx, poly_data in enumerate(polygons):
                if poly_data.get('removed', False):
                    continue  # Skip polygons marked for removal in Pre-Phase 0
                poly_type = poly_data.get('polygon_type', 'BOUNDARY')
                
                if poly_type == 'BOUNDARY':
                    boundary_poly = {'idx': poly_idx, 'data': poly_data, 'verts': poly_data.get('vertices', [])}
                elif poly_type == 'HOLE':
                    holes.append({'idx': poly_idx, 'data': poly_data, 'verts': poly_data.get('vertices', [])})
                elif poly_type == 'ALT':
                    alternates.append({'idx': poly_idx, 'data': poly_data, 'verts': poly_data.get('vertices', [])})
            
            # Only include faces that have at least a boundary or alternates
            if boundary_poly or alternates:
                face_polygon_groups.append({
                    'face_idx': face_idx,
                    'boundary': boundary_poly,
                    'holes': holes,
                    'alternates': alternates
                })
        
        print(f"[POLY FORM]     Rebuilt: {len(face_polygon_groups)} faces with non-removed polygons")
    
    # ==========================================================================
    # PHASE 0, 1, 2: Phase 0 skipped, Phase 1 enabled
    # ==========================================================================
    SKIP_PHASE_0 = True  # Phase 0 skipped
    SKIP_PHASES_1_2 = False  # Phase 1 enabled
    
    if SKIP_PHASE_0:
        print(f"\n[POLY FORM]   ⏭️  SKIPPING Phase 0 (per-face optimization)")
        print(f"[POLY FORM]   Using all remaining polygons after Pre-Phase 0 cleanup")
        
        # Build current_selection from all non-removed polygons
        current_selection = []
        for group in face_polygon_groups:
            face_idx = group['face_idx']
            if group['boundary']:
                current_selection.append((face_idx, group['boundary']['idx'], 'BOUNDARY', group['boundary']['verts']))
            for hole in group['holes']:
                current_selection.append((face_idx, hole['idx'], 'HOLE', hole['verts']))
            for alt in group['alternates']:
                current_selection.append((face_idx, alt['idx'], 'ALT', alt['verts']))
        
        curr_b, curr_m, curr_i = compute_edge_stats_for_selection(current_selection)
        print(f"[POLY FORM]   Current state: {len(current_selection)} polygons, B={curr_b}, M={curr_m}, I={curr_i}")
        
        # Jump directly to final selection output (skip all phase 0, 1, 2 code)
        # The code will continue after the "FINAL SELECTION" print statements
    
    if not SKIP_PHASE_0:
        # Original Phase 0, 1, 2 code follows...
    
        # ==========================================================================
        # PHASE 0: Per-Face Optimization (Early Stopping)
        # ==========================================================================
        print(f"\n[POLY FORM]   PHASE 0: Per-face optimization (boundary always kept, minimize invalid edges)...")
    
        from itertools import combinations as iter_combinations
        
        optimized_face_selections = {}
        faces_with_alts = [g for g in face_polygon_groups if len(g['alternates']) > 0]
        
        # Sort by number of alternates (ascending - simplest faces first)
        faces_with_alts_sorted = sorted(faces_with_alts, key=lambda g: len(g['alternates']))
        
        print(f"[POLY FORM]   Found {len(faces_with_alts_sorted)} faces with alternates to optimize")
    
        # Build initial selection with all faces using boundary+holes or all alternates+holes
        for group in face_polygon_groups:
            face_idx = group['face_idx']
            if len(group['alternates']) == 0:
                # No alternates: mark as optimized with boundary+holes (if they exist)
                combo = []
                if group['boundary']:
                    combo.append((face_idx, group['boundary']['idx'], 'BOUNDARY', group['boundary']['verts']))
                for hole in group['holes']:
                    combo.append((face_idx, hole['idx'], 'HOLE', hole['verts']))
                if combo:  # Only add if face has some polygons
                    optimized_face_selections[face_idx] = combo
            else:
                # Has alternates: start with all alternates+holes (will optimize later)
                combo = []
                for alt in group['alternates']:
                    combo.append((face_idx, alt['idx'], 'ALT', alt['verts']))
                # Always include holes in base set
                for hole in group['holes']:
                    combo.append((face_idx, hole['idx'], 'HOLE', hole['verts']))
                optimized_face_selections[face_idx] = combo
    
        # Optimize faces one at a time (smallest alt count first)
        for group in faces_with_alts_sorted:
            face_idx = group['face_idx']
            boundary = group['boundary']
            holes = group['holes']
            alternates = group['alternates']
        
            # Build current global selection (all optimized faces so far)
            other_faces_polys = []
            for other_group in face_polygon_groups:
                if other_group['face_idx'] == face_idx:
                    continue  # Skip current face
            
                # Add optimized selection from other faces
                if other_group['face_idx'] in optimized_face_selections:
                    other_faces_polys.extend(optimized_face_selections[other_group['face_idx']])
            
            if len(other_faces_polys) == 0:
                print(f"[POLY FORM]     WARNING: Face {face_idx+1} has no other faces in selection (processed first?)")
        
            # Holes are always included (mandatory base polygons)
            holes_combo = []
            for hole in holes:
                holes_combo.append((face_idx, hole['idx'], 'HOLE', hole['verts']))
        
            # IMPORTANT: Boundaries are ALWAYS mandatory when they exist
            # Test combinations of alternates WITH the boundary
            best_face_combo = None
            best_face_stats = None
            best_face_score = None
        
            if boundary:
                # Boundary exists: it's MANDATORY, test combinations of alternates WITH boundary
                boundary_combo = [(face_idx, boundary['idx'], 'BOUNDARY', boundary['verts'])] + holes_combo
            
                # Option 1: Boundary + holes only (baseline)
                test_selection = other_faces_polys + boundary_combo
                test_b, test_m, test_i = compute_edge_stats_for_selection(test_selection)
                score = (test_i, test_b, -test_m)
            
                best_face_combo = boundary_combo.copy()
                best_face_stats = (test_b, test_m, test_i)
                best_face_score = score
            
                # Options 2-N: Boundary + holes + various combinations of alternates
                n_alts = len(alternates)
                for r in range(1, n_alts + 1):
                    for alt_subset in iter_combinations(range(n_alts), r):
                        test_combo = boundary_combo.copy()  # Start with boundary + holes
                        for alt_idx in alt_subset:
                            alt = alternates[alt_idx]
                            test_combo.append((face_idx, alt['idx'], 'ALT', alt['verts']))
                    
                        test_selection = other_faces_polys + test_combo
                        test_b, test_m, test_i = compute_edge_stats_for_selection(test_selection)
                        score = (test_i, test_b, -test_m)
                    
                        if score < best_face_score:
                            best_face_combo = test_combo
                            best_face_stats = (test_b, test_m, test_i)
                            best_face_score = score
            else:
                # No boundary: test combinations of alternates + holes only
                n_alts = len(alternates)
                for r in range(n_alts, 0, -1):
                    for alt_subset in iter_combinations(range(n_alts), r):
                        test_combo = holes_combo.copy()  # Always include holes
                        for alt_idx in alt_subset:
                            alt = alternates[alt_idx]
                            test_combo.append((face_idx, alt['idx'], 'ALT', alt['verts']))
                    
                        test_selection = other_faces_polys + test_combo
                        test_b, test_m, test_i = compute_edge_stats_for_selection(test_selection)
                        score = (test_i, test_b, -test_m)
                    
                        if best_face_score is None or score < best_face_score:
                            best_face_combo = test_combo
                            best_face_stats = (test_b, test_m, test_i)
                            best_face_score = score
        
            # Print result
            if best_face_combo:
                # Print result
                poly_types = ', '.join([p[2] for p in best_face_combo])
                print(f"[POLY FORM]     Face {face_idx+1} ({len(alternates)} alts): Best = {len(best_face_combo)} polys ({poly_types}), Score=(B={best_face_stats[0]}, I={best_face_stats[2]})")
            
                # For faces with many alternates, show which specific polygons were selected
                if len(alternates) >= 4:
                    for item in best_face_combo:
                        poly_verts = item[3] if len(item) > 3 else []
                        poly_type = item[2] if len(item) > 2 else 'UNKNOWN'
                        verts_str = str(poly_verts[:12]) + ('...' if len(poly_verts) > 12 else '')
                        print(f"[POLY FORM]       Selected {poly_type}: {verts_str}")
        
            # Store the best combination for this face
            optimized_face_selections[face_idx] = best_face_combo
        
            # Early stopping: if invalid edges < 10, stop Phase 0
            if best_face_stats[2] < 10:
                remaining_faces = len(faces_with_alts_sorted) - len([f for f in faces_with_alts_sorted if f['face_idx'] in optimized_face_selections or f['face_idx'] == face_idx])
                if remaining_faces > 0:
                    print(f"[POLY FORM]   ✓ Invalid edges < 10 (I={best_face_stats[2]}), stopping Phase 0 early")
                    print(f"[POLY FORM]   Skipping {remaining_faces} remaining faces with alternates")
                    # Fill in remaining faces with their current selection (all alternates)
                    break
    
        # Build the initial selection from optimized faces
        current_selection = []
        removed_alts = []
    
        # Track which polygons were selected by Phase 0 (these are mandatory)
        phase0_selected_polys = set()
    
        for group in face_polygon_groups:
            face_idx = group['face_idx']
        
            if face_idx in optimized_face_selections:
                # Use the optimized selection for this face
                for item in optimized_face_selections[face_idx]:
                    current_selection.append(item)
                    # Mark as Phase 0 selected (mandatory)
                    phase0_selected_polys.add((item[0], item[1]))  # (face_idx, poly_idx)
            else:
                # Not in optimized_face_selections: add boundary + holes if they exist
                if group['boundary']:
                    item = (face_idx, group['boundary']['idx'], 'BOUNDARY', group['boundary']['verts'])
                    current_selection.append(item)
                    phase0_selected_polys.add((face_idx, group['boundary']['idx']))
                for hole in group['holes']:
                    item = (face_idx, hole['idx'], 'HOLE', hole['verts'])
                    current_selection.append(item)
                    phase0_selected_polys.add((face_idx, hole['idx']))
    
        # Rebuild removable_alts and mandatory_polys from the optimized selection
        # Phase 1 can now try to improve the solution by replacing boundaries with alternates
        # Only HOLEs are truly mandatory (cannot be removed)
        removable_alts = []
        removable_boundaries = []
        mandatory_polys = []
    
        for item in current_selection:
            face_idx, poly_idx, poly_type, poly_verts = item
            poly_key = (face_idx, poly_idx)
        
            if poly_type == 'HOLE':
                # HOLEs are always mandatory
                mandatory_polys.append(item)
            elif poly_type == 'BOUNDARY':
                # BUONDARYs can be removed/replaced in Phase 1
                removable_boundaries.append(item)
            elif poly_type == 'ALT':
                # ALTs can be removed in Phase 1
                removable_alts.append(item)
            else:
                # Unknown type, treat as mandatory
                mandatory_polys.append(item)
    
        # Combine removable polygons for Phase 1 testing
        removable_polys = removable_boundaries + removable_alts
    
        # Compute initial statistics
        curr_b, curr_m, curr_i = compute_edge_stats_for_selection(current_selection)
        print(f"[POLY FORM]   Initial state after Phase 0: {len(current_selection)} polygons, B={curr_b}, M={curr_m}, I={curr_i}")
        print(f"[POLY FORM]   Removable: {len(removable_boundaries)} boundaries, {len(removable_alts)} alternates")
        
        # Display polygon edge analysis after Phase 0
        print(f"\n[POLY FORM]       ========== POLYGON EDGE ANALYSIS (AFTER PHASE 0) ==========")
        print(f"[POLY FORM]       Face Poly     Type Verts   B   M   I Vertices")
        print(f"[POLY FORM]       ---- ---- -------- ----- --- --- --- " + "-" * 50)
        
        # Rebuild the analysis for selected polygons
        for face_idx, poly_idx, poly_type, poly_verts in current_selection:
            # Count B, M, I edges for this polygon
            poly_edges = []
            for i in range(len(poly_verts)):
                v1 = poly_verts[i]
                v2 = poly_verts[(i + 1) % len(poly_verts)]
                edge = (min(v1, v2), max(v1, v2))
                poly_edges.append(edge)
            
            # Get edge counts from the selection
            edge_counts = {}
            for other_face_idx, other_poly_idx, other_poly_type, other_poly_verts in current_selection:
                for i in range(len(other_poly_verts)):
                    v1 = other_poly_verts[i]
                    v2 = other_poly_verts[(i + 1) % len(other_poly_verts)]
                    edge = (min(v1, v2), max(v1, v2))
                    edge_counts[edge] = edge_counts.get(edge, 0) + 1
            
            b_count = sum(1 for e in poly_edges if edge_counts.get(e, 0) == 1)
            m_count = sum(1 for e in poly_edges if edge_counts.get(e, 0) == 2)
            i_count = sum(1 for e in poly_edges if edge_counts.get(e, 0) >= 3)
            
            verts_str = str(poly_verts[:10]) + ('...' if len(poly_verts) > 10 else '')
            print(f"[POLY FORM]       {face_idx+1:4d} {poly_idx:4d} {poly_type:>8s} {len(poly_verts):5d} {b_count:3d} {m_count:3d} {i_count:3d} {verts_str}")
        
        print(f"[POLY FORM]       " + "=" * 86)
    
        # ==========================================================================
        # PHASE 1: Greedy Removal (SKIPPED)
        # ==========================================================================
        if not SKIP_PHASES_1_2:
            print(f"\n[POLY FORM]   PHASE 1: Greedy removal to reach target (B≤10, I≤10)...")
        else:
            print(f"\n[POLY FORM]   ⏭️  SKIPPING Phase 1 and 2")
            # Skip to end of Phase 0, 1, 2 block
            pass
        
        if not SKIP_PHASES_1_2:
            # Original Phase 1 code
            iteration = 0
    
        while removable_polys and (curr_i > 10 or curr_b > 10):
            iteration += 1
            if iteration > len(removable_polys) + 100:  # Allow more iterations
                print(f"[POLY FORM]   WARNING: Exceeded max iterations, stopping")
                break
        
            print(f"[POLY FORM]   Iteration {iteration}: Testing removal of {len(removable_polys)} polygons...")
        
            best_removal = None
            best_stats = None
            best_score = None
        
            # Try removing each removable polygon
            for poly in removable_polys:
                # Create selection without this polygon
                test_selection = [p for p in current_selection if p != poly]
            
                # Compute statistics
                test_b, test_m, test_i = compute_edge_stats_for_selection(test_selection)
            
                # Score: prioritize reducing invalid edges, then boundary edges
                score = (test_i, test_b, -test_m)  # Lower is better
            
                if best_score is None or score < best_score:
                    best_removal = poly
                    best_stats = (test_b, test_m, test_i)
                    best_score = score
        
            # Apply best removal if it improves (not just maintains)
            curr_score = (curr_i, curr_b, -curr_m)
            if best_removal and best_score < curr_score:  # Must improve the score
                current_selection = [p for p in current_selection if p != best_removal]
                removable_polys.remove(best_removal)
                removed_alts.append(best_removal)
                curr_b, curr_m, curr_i = best_stats
            
                face_idx, poly_idx, poly_type, _ = best_removal
                print(f"[POLY FORM]     Removed Face {face_idx+1} poly {poly_idx} ({poly_type}) → B={curr_b}, M={curr_m}, I={curr_i}")
            else:
                print(f"[POLY FORM]     No improving removal found, stopping greedy phase")
                break
        
            # Check if we've reached target
            if curr_i <= 10 and curr_b <= 10:
                print(f"[POLY FORM]   ✓ Reached target: B={curr_b}, I={curr_i}")
                break
    
            print(f"[POLY FORM]   After greedy removal: {len(current_selection)} polygons, B={curr_b}, M={curr_m}, I={curr_i}")
            print(f"[POLY FORM]   Removed {len(removed_alts)} polygons, {len(removable_polys)} removable remaining")
    
        # ==========================================================================
        # PHASE 2: Per-Face Optimization (SKIPPED)
        # ==========================================================================
        if not SKIP_PHASES_1_2 and len(removable_polys) > 0 and (curr_i > 0 or curr_b > 0):
            print(f"\n[POLY FORM]   PHASE 2: Per-face optimization on faces with alternates...")
        
            from itertools import combinations as iter_combinations
        
            # Build set of faces that have alternates
            faces_with_alternates = set(g['face_idx'] for g in faces_with_alts_sorted)
        
        if not SKIP_PHASES_1_2 and len(removable_polys) > 0 and (curr_i > 0 or curr_b > 0):
            # Group current selection by face
            current_selection_by_face = {}
            for poly in current_selection:
                face_idx = poly[0]
                if face_idx not in current_selection_by_face:
                    current_selection_by_face[face_idx] = []
                current_selection_by_face[face_idx].append(poly)
        
            # Build the base selection (all faces without alternates)
            base_selection = []
            for face_idx, polys in current_selection_by_face.items():
                if face_idx not in faces_with_alternates:
                    base_selection.extend(polys)
        
            print(f"[POLY FORM]   Base selection: {len(base_selection)} polygons from {len(current_selection_by_face) - len(faces_with_alternates)} faces without alternates")
            print(f"[POLY FORM]   Optimizing: {len(faces_with_alternates)} faces with alternates")
        
            # For each face with alternates, find the best combination of its polygons
            optimized_selection = base_selection.copy()
        
            for group in faces_with_alts_sorted:
                face_idx = group['face_idx']
            
                # Get all polygons for this face from current selection
                face_polys = current_selection_by_face.get(face_idx, [])
            
                if len(face_polys) == 0:
                    print(f"[POLY FORM]   WARNING: Face {face_idx+1} has no polygons in current selection!")
                    continue
            
                # Separate by type
                boundaries = [p for p in face_polys if p[2] == 'BOUNDARY']
                holes = [p for p in face_polys if p[2] == 'HOLE']
                alts = [p for p in face_polys if p[2] == 'ALT']
            
                # Build the selection for all OTHER faces
                other_faces_polys = [p for p in optimized_selection if p[0] != face_idx]
            
                # Always include holes (mandatory)
                face_mandatory = holes.copy()
            
                # Test combinations: at least one polygon per face
                best_face_combo = None
                best_face_stats = None
                best_face_score = None
            
                # Option 1: Boundary + holes (if boundary exists)
                if boundaries:
                    for boundary in boundaries:
                        test_combo = [boundary] + face_mandatory
                        test_selection = other_faces_polys + test_combo
                        test_b, test_m, test_i = compute_edge_stats_for_selection(test_selection)
                        score = (test_i, test_b, -test_m)
                    
                        if best_face_score is None or score < best_face_score:
                            best_face_combo = test_combo
                            best_face_stats = (test_b, test_m, test_i)
                            best_face_score = score
            
                # Options 2-N: Test combinations of alternates + holes
                if alts:
                    for r in range(1, len(alts) + 1):
                        for alt_subset in iter_combinations(alts, r):
                            test_combo = list(alt_subset) + face_mandatory
                            test_selection = other_faces_polys + test_combo
                            test_b, test_m, test_i = compute_edge_stats_for_selection(test_selection)
                            score = (test_i, test_b, -test_m)
                        
                            if best_face_score is None or score < best_face_score:
                                best_face_combo = test_combo
                                best_face_stats = (test_b, test_m, test_i)
                                best_face_score = score
            
                # Options 3-N: Test combinations of boundary + alternates + holes
                if boundaries and alts:
                    for boundary in boundaries:
                        for r in range(1, len(alts) + 1):
                            for alt_subset in iter_combinations(alts, r):
                                test_combo = [boundary] + list(alt_subset) + face_mandatory
                                test_selection = other_faces_polys + test_combo
                                test_b, test_m, test_i = compute_edge_stats_for_selection(test_selection)
                                score = (test_i, test_b, -test_m)
                            
                                if best_face_score is None or score < best_face_score:
                                    best_face_combo = test_combo
                                    best_face_stats = (test_b, test_m, test_i)
                                    best_face_score = score
            
                # If no better combination found, use current selection for this face
                if best_face_combo is None:
                    best_face_combo = face_polys
                    test_selection = other_faces_polys + best_face_combo
                    test_b, test_m, test_i = compute_edge_stats_for_selection(test_selection)
                    best_face_stats = (test_b, test_m, test_i)
            
                # Update optimized selection with best for this face
                optimized_selection = other_faces_polys + best_face_combo
            
                poly_types = ', '.join([p[2] for p in best_face_combo])
                print(f"[POLY FORM]   Face {face_idx+1}: Best = {len(best_face_combo)} polys ({poly_types}), Score=(I={best_face_stats[2]}, B={best_face_stats[0]})")
        
            # Final statistics
            final_b, final_m, final_i = compute_edge_stats_for_selection(optimized_selection)
            print(f"[POLY FORM]   Phase 2 result: {len(optimized_selection)} polygons, B={final_b}, M={final_m}, I={final_i}")
        
            current_selection = optimized_selection
            curr_b, curr_m, curr_i = final_b, final_m, final_i
    
    # End of Phase 0 block (Phases 1 & 2 may be skipped)
    
    print(f"\n[POLY FORM]   ==========================================")
    print(f"[POLY FORM]   FINAL SELECTION: {len(current_selection)} polygons")
    print(f"[POLY FORM]     Boundary edges: {curr_b}, Manifold: {curr_m}, Invalid: {curr_i}")
    print(f"[POLY FORM]   ==========================================")

    
    print(f"[POLY FORM]   ")
    print(f"[POLY FORM]   NOTE: Base set edge statistics:")
    print(f"[POLY FORM]   - These are the initial edges BEFORE any pruning")
    print(f"[POLY FORM]   - {initial_invalid} invalid edges need to be resolved")
    print(f"[POLY FORM]   - Loop-based pruning will extract polygons to eliminate invalid edges")
    print(f"[POLY FORM]   ==========================================")
    
    # Print face summary after initial edge distribution (SUPPRESSED)
    # print_face_summary_debug(unique_faces, "Face Summary (After Initial Edge Analysis)")
    
    # DEBUG PLOT: Show current state (SUPPRESSED)
    # print(f"\n[POLY FORM]   Creating debug plot...")
    # plot_extraction_debug(selected_vertices, unique_faces, edge_face_map_all, [], 
    #                      pruned_set=set(), title="Initial State - All Polygons")
    
    # Initialize variables that may be used later
    invalid_edge_polygons = []
    pruned_polygons = []
    boundary_removed_polygons = []
    
    # ==========================================================================
    # NEW APPROACH: Loop-based pruning, then combinatorial search
    # ==========================================================================
    from itertools import combinations
    
    # =======================================================================
    # Rebuild edge map from optimized selection
    # =======================================================================
    print(f"\n[POLY FORM]   Rebuilding edge map from optimized polygon selection...")
    
    # Build edge map from current_selection
    edge_face_map_optimized = {}
    for item in current_selection:
        if len(item) == 4:
            face_idx, poly_idx, poly_type, poly_verts = item
        else:
            continue
        
        for i in range(len(poly_verts)):
            v1 = poly_verts[i] - 1
            v2 = poly_verts[(i + 1) % len(poly_verts)] - 1
            edge = (min(v1, v2), max(v1, v2))
            if edge not in edge_face_map_optimized:
                edge_face_map_optimized[edge] = []
            edge_face_map_optimized[edge].append((face_idx, poly_idx, poly_type))
    
    # Compute statistics from optimized selection
    optimized_boundary = sum(1 for fl in edge_face_map_optimized.values() if len(fl) == 1)
    optimized_manifold = sum(1 for fl in edge_face_map_optimized.values() if len(fl) == 2)
    optimized_invalid = sum(1 for fl in edge_face_map_optimized.values() if len(fl) >= 3)
    
    print(f"[POLY FORM]   Optimized selection edge stats:")
    print(f"[POLY FORM]     - Boundary edges (1 face): {optimized_boundary}")
    print(f"[POLY FORM]     - Manifold edges (2 faces): {optimized_manifold}")
    print(f"[POLY FORM]     - Invalid edges (3+ faces): {optimized_invalid}")
    
    # Use optimized edge map for subsequent phases
    edge_face_map_all = edge_face_map_optimized
    initial_invalid = optimized_invalid
    
    # =======================================================================
    # Update unique_faces structure to reflect Phase 0-2 selections
    # =======================================================================
    print(f"\n[POLY FORM]   Updating unique_faces structure with Phase 0-2 selections...")
    
    # Build set of selected polygons
    selected_poly_set = set()
    for item in current_selection:
        if len(item) == 4:
            face_idx, poly_idx, poly_type, poly_verts = item
            selected_poly_set.add((face_idx, poly_idx))
    
    # Mark all non-selected polygons as removed in unique_faces
    for face_idx, face_data in enumerate(unique_faces):
        # Skip faces that don't have polygons yet (shouldn't happen but defensive)
        if 'polygons' not in face_data:
            print(f"[POLY FORM]   WARNING: Face {face_idx} has no 'polygons' key, skipping")
            continue
            
        for poly_idx, poly_data in enumerate(face_data['polygons']):
            poly_key = (face_idx, poly_idx)
            if poly_key not in selected_poly_set:
                poly_data['removed'] = True
                poly_data['removal_reason'] = 'phase_0_2_optimization'
            else:
                # Ensure selected polygons are marked as NOT removed
                poly_data['removed'] = False
    
    # Rebuild all_polygons_list to reflect Phase 0-2 selections
    print(f"[POLY FORM]   Rebuilding all_polygons_list with only selected polygons...")
    all_polygons_list = []
    alt_polygons = []
    for face_idx, face_eq in enumerate(unique_faces):
        polygons = face_eq.get('polygons', [])
        for poly_idx, poly_data in enumerate(polygons):
            if poly_data.get('removed', False):
                continue  # Skip removed polygons
            
            poly_type = poly_data.get('polygon_type', 'BOUNDARY')
            poly_verts = poly_data.get('vertices', [])
            
            all_polygons_list.append({
                'face_idx': face_idx,
                'polygon_idx': poly_idx,
                'data': poly_data,
                'face_eq': face_eq,
                'poly_type': poly_type
            })
            
            if poly_type == 'ALT':
                alt_polygons.append({
                    'face_idx': face_idx,
                    'polygon_idx': poly_idx,
                    'data': poly_data,
                    'face_eq': face_eq
                })
    
    print(f"[POLY FORM]   Rebuilt with {len(all_polygons_list)} polygons (including {len(alt_polygons)} ALTs)")
    
    # =======================================================================
    # Iterative Refinement: Try to include excluded polygons made from boundary edges
    # =======================================================================
    if optimized_invalid == 0 and optimized_boundary > 0:
        print(f"\n[POLY FORM]   ========== ITERATIVE REFINEMENT ==========")
        print(f"[POLY FORM]   Attempting to reduce boundary edges by including excluded polygons...")
        
        # Build selected_polygon_set from current_selection
        selected_polygon_set = set()
        for item in current_selection:
            if len(item) == 4:
                face_idx, poly_idx, poly_type, poly_verts = item
                selected_polygon_set.add((face_idx, poly_idx))
        
        # Track candidates that have been tried (added but then removed by optimization)
        tried_candidates = set()
        
        # Track best state across iterations
        best_boundary_count = optimized_boundary
        best_selection = current_selection.copy()
        best_edge_map = edge_face_map_optimized.copy()
        best_mandatory = mandatory_polys.copy()
        best_removable = removable_alts.copy()
        
        prev_boundary_count = optimized_boundary
        max_iterations = 10
        for iter_num in range(1, max_iterations + 1):
            print(f"\n[POLY FORM]   Iteration {iter_num}/{max_iterations}:")
            
            # Get boundary edges from current selection
            boundary_edges = set()
            for edge, face_list in edge_face_map_optimized.items():
                if len(face_list) == 1:
                    boundary_edges.add(edge)
            
            print(f"[POLY FORM]     Current boundary edges: {len(boundary_edges)}")
            
            if len(boundary_edges) == 0:
                print(f"[POLY FORM]     ✓ No boundary edges remaining, refinement complete!")
                break
            
            # Find excluded polygons (from all face_polygon_groups)
            excluded_polygons = []
            for group in face_polygon_groups:
                face_idx = group['face_idx']
                
                # Check boundary
                if group['boundary']:
                    poly_key = (face_idx, group['boundary']['idx'])
                    if poly_key not in selected_polygon_set and poly_key not in tried_candidates:
                        excluded_polygons.append((face_idx, group['boundary']['idx'], 'BOUNDARY', group['boundary']['verts']))
                
                # Check alternates
                for alt in group['alternates']:
                    poly_key = (face_idx, alt['idx'])
                    if poly_key not in selected_polygon_set and poly_key not in tried_candidates:
                        excluded_polygons.append((face_idx, alt['idx'], 'ALT', alt['verts']))
            
            print(f"[POLY FORM]     Excluded polygons: {len(excluded_polygons)}")
            
            # For each excluded polygon that shares boundary edges, calculate boundary reduction
            polygon_boundary_reductions = []
            for face_idx, poly_idx, poly_type, poly_verts in excluded_polygons:
                # Get edges of this polygon
                poly_edges = []
                shares_boundary = False
                for i in range(len(poly_verts)):
                    v1 = poly_verts[i] - 1
                    v2 = poly_verts[(i + 1) % len(poly_verts)] - 1
                    edge = (min(v1, v2), max(v1, v2))
                    poly_edges.append(edge)
                    if edge in boundary_edges:
                        shares_boundary = True
                
                # Only consider polygons that share at least one boundary edge
                if not shares_boundary:
                    continue
                
                # Simulate adding this polygon to current selection
                test_selection = current_selection + [(face_idx, poly_idx, poly_type, poly_verts)]
                test_b, test_m, test_i = compute_edge_stats_for_selection(test_selection)
                
                # Calculate reduction in boundary edges
                boundary_reduction = len(boundary_edges) - test_b
                
                polygon_boundary_reductions.append((face_idx, poly_idx, poly_type, poly_verts, poly_edges, boundary_reduction, test_b, test_i))
            
            # Sort by boundary reduction (descending), then by resulting invalid edges (ascending)
            polygon_boundary_reductions.sort(key=lambda x: (-x[5], x[7]))
            
            # Select the polygon with maximum boundary reduction
            candidates = []
            if len(polygon_boundary_reductions) > 0:
                face_idx, poly_idx, poly_type, poly_verts, poly_edges, reduction, test_b, test_i = polygon_boundary_reductions[0]
                if reduction > 0:  # Only include if it reduces boundary edges
                    candidates.append((face_idx, poly_idx, poly_type, poly_verts, poly_edges))
                    print(f"[POLY FORM]       Best candidate: Face {face_idx}, Poly {poly_idx} ({poly_type}): reduction={reduction} (B: {len(boundary_edges)}→{test_b}, I={test_i})")
                else:
                    print(f"[POLY FORM]       Best candidate has no reduction: Face {face_idx}, Poly {poly_idx} ({poly_type}): reduction={reduction}")
            
            print(f"[POLY FORM]     Selected {len(candidates)} candidate polygon(s) with best boundary reduction")
            
            if len(candidates) == 0:
                print(f"[POLY FORM]     No candidates found, stopping refinement")
                break
            
            # Add candidates to current selection as mandatory base polygons
            print(f"[POLY FORM]     Adding {len(candidates)} candidates as base polygons...")
            
            # Add candidates - boundaries are mandatory, ALTs go to current_selection only
            candidate_items = []
            for face_idx, poly_idx, poly_type, poly_verts, poly_edges in candidates:
                item = (face_idx, poly_idx, poly_type, poly_verts)
                current_selection.append(item)
                candidate_items.append(item)
                selected_polygon_set.add((face_idx, poly_idx))
                # BoundARY candidates are treated as mandatory
                if poly_type == 'BOUNDARY':
                    mandatory_polys.append(item)
            
            # Check edge stats before optimization
            curr_b, curr_m, curr_i = compute_edge_stats_for_selection(current_selection)
            print(f"[POLY FORM]     After adding candidates: B={curr_b}, M={curr_m}, I={curr_i}")
            
            # Only run optimization if there are invalid edges
            if curr_i == 0:
                print(f"[POLY FORM]     No invalid edges, keeping all candidates as mandatory")
                # Mark all candidates as mandatory since they don't create conflicts
                for item in candidate_items:
                    if item not in mandatory_polys:
                        mandatory_polys.append(item)
            else:
                print(f"[POLY FORM]     Invalid edges detected, running optimization...")
            
            # Greedy removal
            iteration = 0
            while removable_alts and (curr_i > 10 or curr_b > 10):
                iteration += 1
                if iteration > 20:
                    break
                
                best_removal = None
                best_stats = None
                best_score = None
                
                for alt in removable_alts:
                    test_selection = [p for p in current_selection if p != alt]
                    test_b, test_m, test_i = compute_edge_stats_for_selection(test_selection)
                    score = (test_i, test_b, -test_m)
                    
                    if best_score is None or score < best_score:
                        best_removal = alt
                        best_stats = (test_b, test_m, test_i)
                        best_score = score
                
                curr_score = (curr_i, curr_b, -curr_m)
                if best_removal and best_score < curr_score:
                    current_selection = [p for p in current_selection if p != best_removal]
                    removable_alts.remove(best_removal)
                    removed_alts.append(best_removal)
                    curr_b, curr_m, curr_i = best_stats
                    selected_polygon_set.discard((best_removal[0], best_removal[1]))
                else:
                    break
            
            # Exhaustive search on remaining alternates
            if len(removable_alts) > 0 and len(removable_alts) <= 10:
                total_combos = 2 ** len(removable_alts)
                if total_combos <= 10000:
                    best_combo = current_selection.copy()
                    best_stats = (curr_b, curr_m, curr_i)
                    
                    for r in range(len(removable_alts), -1, -1):
                        for alt_subset in iter_combinations(removable_alts, r):
                            test_selection = mandatory_polys + list(alt_subset)
                            test_b, test_m, test_i = compute_edge_stats_for_selection(test_selection)
                            
                            if test_i == 0 and test_b == 0:
                                best_combo = test_selection
                                best_stats = (test_b, test_m, test_i)
                                break
                            
                            if (test_i, test_b) < (best_stats[2], best_stats[0]):
                                best_combo = test_selection
                                best_stats = (test_b, test_m, test_i)
                        
                        if best_stats[2] == 0 and best_stats[0] == 0:
                            break
                    
                    current_selection = best_combo
                    curr_b, curr_m, curr_i = best_stats
                    
                    # Update selected_polygon_set
                    selected_polygon_set = set()
                    for item in current_selection:
                        if len(item) == 4:
                            selected_polygon_set.add((item[0], item[1]))
            
            print(f"[POLY FORM]     After optimization: {len(current_selection)} polygons, B={curr_b}, M={curr_m}, I={curr_i}")
            
            # Rebuild selected_polygon_set from final current_selection
            selected_polygon_set = set()
            for item in current_selection:
                if len(item) == 4:
                    selected_polygon_set.add((item[0], item[1]))
            
            # Track which candidates were removed by optimization
            for face_idx, poly_idx, poly_type, poly_verts, poly_edges in candidates:
                poly_key = (face_idx, poly_idx)
                if poly_key not in selected_polygon_set:
                    # This candidate was added but then removed by optimization
                    tried_candidates.add(poly_key)
            
            # Rebuild edge map
            edge_face_map_optimized = {}
            for item in current_selection:
                if len(item) == 4:
                    face_idx, poly_idx, poly_type, poly_verts = item
                else:
                    continue
                
                for i in range(len(poly_verts)):
                    v1 = poly_verts[i] - 1
                    v2 = poly_verts[(i + 1) % len(poly_verts)] - 1
                    edge = (min(v1, v2), max(v1, v2))
                    if edge not in edge_face_map_optimized:
                        edge_face_map_optimized[edge] = []
                    edge_face_map_optimized[edge].append((face_idx, poly_idx, poly_type))
            
            optimized_boundary = sum(1 for fl in edge_face_map_optimized.values() if len(fl) == 1)
            optimized_manifold = sum(1 for fl in edge_face_map_optimized.values() if len(fl) == 2)
            optimized_invalid = sum(1 for fl in edge_face_map_optimized.values() if len(fl) >= 3)
            
            print(f"[POLY FORM]     Boundary edges: {prev_boundary_count} → {optimized_boundary}")
            
            if optimized_boundary < best_boundary_count:
                # This iteration improved! Save the best state
                print(f"[POLY FORM]     ✓ Improvement found, saving best state")
                best_boundary_count = optimized_boundary
                best_selection = current_selection.copy()
                best_edge_map = edge_face_map_optimized.copy()
                best_mandatory = mandatory_polys.copy()
                best_removable = removable_alts.copy()
                prev_boundary_count = optimized_boundary
            elif optimized_boundary >= prev_boundary_count:
                print(f"[POLY FORM]     No improvement, restoring best state and stopping refinement")
                # Restore best state
                current_selection = best_selection
                edge_face_map_optimized = best_edge_map
                mandatory_polys = best_mandatory
                removable_alts = best_removable
                optimized_boundary = best_boundary_count
                optimized_manifold = sum(1 for fl in edge_face_map_optimized.values() if len(fl) == 2)
                optimized_invalid = sum(1 for fl in edge_face_map_optimized.values() if len(fl) >= 3)
                break
            else:
                # Update for next iteration
                prev_boundary_count = optimized_boundary
        
        print(f"\n[POLY FORM]   Refinement complete after {iter_num} iteration(s)")
        print(f"[POLY FORM]   Final: {len(current_selection)} polygons, B={optimized_boundary}, M={optimized_manifold}, I={optimized_invalid}")
        
        # =======================================================================
        # Analyze boundary edges to find missing adjacent faces
        # =======================================================================
        if optimized_boundary > 0:
            print(f"\n[POLY FORM]   ========== BOUNDARY EDGE ANALYSIS ==========")
            print(f"[POLY FORM]   Analyzing {optimized_boundary} boundary edges to find missing adjacent faces...")
            
            # Get all boundary edges
            boundary_edges_analysis = []
            for edge, face_list in edge_face_map_optimized.items():
                if len(face_list) == 1:
                    face_idx, poly_idx, poly_type = face_list[0]
                    boundary_edges_analysis.append((edge, face_idx, poly_idx))
            
            print(f"[POLY FORM]   Found {len(boundary_edges_analysis)} boundary edges")
            
            # Build polygon edge sequences for navigation
            polygon_edge_sequences = {}
            for item in current_selection:
                if len(item) == 4:
                    face_idx, poly_idx, poly_type, poly_verts = item
                    edges = []
                    for i in range(len(poly_verts)):
                        v1 = poly_verts[i] - 1
                        v2 = poly_verts[(i + 1) % len(poly_verts)] - 1
                        edge = (min(v1, v2), max(v1, v2))
                        edges.append((edge, v1, v2))  # Store edge with direction
                    polygon_edge_sequences[(face_idx, poly_idx)] = edges
            
            # For each boundary edge, find adjacent missing faces
            missing_face_candidates = set()
            
            for boundary_edge, face_idx, poly_idx in boundary_edges_analysis:
                bs, be = boundary_edge  # boundary start, boundary end
                
                # Find this edge in the polygon sequence
                if (face_idx, poly_idx) not in polygon_edge_sequences:
                    continue
                
                edges = polygon_edge_sequences[(face_idx, poly_idx)]
                edge_index = None
                for i, (edge, v1, v2) in enumerate(edges):
                    if edge == boundary_edge:
                        edge_index = i
                        break
                
                if edge_index is None:
                    continue
                
                # Get previous and next edges in this face
                prev_edge_idx = (edge_index - 1) % len(edges)
                next_edge_idx = (edge_index + 1) % len(edges)
                
                prev_edge, _, bpe = edges[prev_edge_idx]  # (be, bpe)
                next_edge, bne, _ = edges[next_edge_idx]  # (bs, bne)
                
                # Check if previous edge is shared with another face
                if prev_edge in edge_face_map_optimized and len(edge_face_map_optimized[prev_edge]) == 2:
                    # Find the other face (fb1)
                    for f_idx, p_idx, p_type in edge_face_map_optimized[prev_edge]:
                        if f_idx != face_idx or p_idx != poly_idx:
                            # In face fb1, find the next edge after prev_edge (be, bpne)
                            if (f_idx, p_idx) in polygon_edge_sequences:
                                fb1_edges = polygon_edge_sequences[(f_idx, p_idx)]
                                for j, (e, ev1, ev2) in enumerate(fb1_edges):
                                    if e == prev_edge:
                                        next_fb1_idx = (j + 1) % len(fb1_edges)
                                        next_fb1_edge, _, bpne = fb1_edges[next_fb1_idx]  # (be, bpne)
                                        
                                        # This edge might separate fb1 and missing face fb2
                                        if next_fb1_edge in edge_face_map_optimized:
                                            for fb2_idx, fb2_poly_idx, fb2_type in edge_face_map_optimized[next_fb1_edge]:
                                                if fb2_idx != f_idx:
                                                    missing_face_candidates.add(fb2_idx)
                                        break
                            break
                
                # Check if next edge is shared with another face
                if next_edge in edge_face_map_optimized and len(edge_face_map_optimized[next_edge]) == 2:
                    # Find the other face
                    for f_idx, p_idx, p_type in edge_face_map_optimized[next_edge]:
                        if f_idx != face_idx or p_idx != poly_idx:
                            # In this face, find the next edge
                            if (f_idx, p_idx) in polygon_edge_sequences:
                                fb1_edges = polygon_edge_sequences[(f_idx, p_idx)]
                                for j, (e, ev1, ev2) in enumerate(fb1_edges):
                                    if e == next_edge:
                                        next_fb1_idx = (j + 1) % len(fb1_edges)
                                        next_fb1_edge, _, _ = fb1_edges[next_fb1_idx]
                                        
                                        # This edge might separate and missing face
                                        if next_fb1_edge in edge_face_map_optimized:
                                            for fb2_idx, fb2_poly_idx, fb2_type in edge_face_map_optimized[next_fb1_edge]:
                                                if fb2_idx != f_idx:
                                                    missing_face_candidates.add(fb2_idx)
                                        break
                            break
            
            print(f"[POLY FORM]   Found {len(missing_face_candidates)} potential missing faces")
            
            # Check if any excluded polygons belong to these missing faces
            excluded_in_missing_faces = []
            for missing_face_idx in missing_face_candidates:
                if missing_face_idx < len(face_polygon_groups):
                    group = face_polygon_groups[missing_face_idx]
                    
                    # Check boundary
                    if group['boundary']:
                        poly_key = (missing_face_idx, group['boundary']['idx'])
                        if poly_key not in selected_polygon_set:
                            excluded_in_missing_faces.append((missing_face_idx, group['boundary']['idx'], 'BOUNDARY', group['boundary']['verts']))
                    
                    # Check alternates
                    for alt in group['alternates']:
                        poly_key = (missing_face_idx, alt['idx'])
                        if poly_key not in selected_polygon_set:
                            excluded_in_missing_faces.append((missing_face_idx, alt['idx'], 'ALT', alt['verts']))
            
            print(f"[POLY FORM]   Found {len(excluded_in_missing_faces)} excluded polygons in missing faces")
            
            if len(excluded_in_missing_faces) > 0:
                print(f"[POLY FORM]   Adding {len(excluded_in_missing_faces)} missing face polygons and re-optimizing...")
                
                # Add excluded polygons from missing faces
                for face_idx, poly_idx, poly_type, poly_verts in excluded_in_missing_faces:
                    item = (face_idx, poly_idx, poly_type, poly_verts)
                    current_selection.append(item)
                    mandatory_polys.append(item)
                    selected_polygon_set.add((face_idx, poly_idx))
                
                # Re-run optimization
                curr_b, curr_m, curr_i = compute_edge_stats_for_selection(current_selection)
                print(f"[POLY FORM]   After adding missing faces: B={curr_b}, M={curr_m}, I={curr_i}")
                
                if curr_i > 0:
                    print(f"[POLY FORM]   Invalid edges detected, running optimization...")
                    # Run optimization (greedy removal only)
                    for missing_item in excluded_in_missing_faces:
                        item = (missing_item[0], missing_item[1], missing_item[2], missing_item[3])
                        if item in mandatory_polys:
                            mandatory_polys.remove(item)
                        if missing_item[2] == 'ALT':
                            removable_alts.append(item)
                    
                    # Greedy removal
                    iteration = 0
                    while removable_alts and curr_i > 0 and iteration < 20:
                        iteration += 1
                        best_removal = None
                        best_stats = None
                        best_score = None
                        
                        for alt in removable_alts:
                            test_selection = [p for p in current_selection if p != alt]
                            test_b, test_m, test_i = compute_edge_stats_for_selection(test_selection)
                            score = (test_i, test_b, -test_m)
                            
                            if best_score is None or score < best_score:
                                best_removal = alt
                                best_stats = (test_b, test_m, test_i)
                                best_score = score
                        
                        curr_score = (curr_i, curr_b, -curr_m)
                        if best_removal and best_score < curr_score:
                            current_selection = [p for p in current_selection if p != best_removal]
                            removable_alts.remove(best_removal)
                            curr_b, curr_m, curr_i = best_stats
                            selected_polygon_set.discard((best_removal[0], best_removal[1]))
                        else:
                            break
                    
                    print(f"[POLY FORM]   After optimization: {len(current_selection)} polygons, B={curr_b}, M={curr_m}, I={curr_i}")
                
                # Rebuild edge map
                edge_face_map_optimized = {}
                for item in current_selection:
                    if len(item) == 4:
                        face_idx, poly_idx, poly_type, poly_verts = item
                        for i in range(len(poly_verts)):
                            v1 = poly_verts[i] - 1
                            v2 = poly_verts[(i + 1) % len(poly_verts)] - 1
                            edge = (min(v1, v2), max(v1, v2))
                            if edge not in edge_face_map_optimized:
                                edge_face_map_optimized[edge] = []
                            edge_face_map_optimized[edge].append((face_idx, poly_idx, poly_type))
                
                optimized_boundary = sum(1 for fl in edge_face_map_optimized.values() if len(fl) == 1)
                optimized_manifold = sum(1 for fl in edge_face_map_optimized.values() if len(fl) == 2)
                optimized_invalid = sum(1 for fl in edge_face_map_optimized.values() if len(fl) >= 3)
                
                print(f"[POLY FORM]   Final after missing face analysis: B={optimized_boundary}, M={optimized_manifold}, I={optimized_invalid}")
        
        # Update edge map and stats
        edge_face_map_all = edge_face_map_optimized
        initial_invalid = optimized_invalid
    
    # =======================================================================
    # NOTE: Do NOT mark polygons as removed here
    # =======================================================================
    # This function (extract_polygon_faces_from_connectivity) should return ALL
    # extracted polygon faces for visualization purposes. The caller (main workflow)
    # will handle polygon selection/optimization separately.
    # 
    # The current_selection from optimization is for internal statistics only
    # and should not affect the output faces returned by this function.
    
    # =======================================================================
    # PHASE 1: Group-Based Pruning Strategy
    # =======================================================================
    if initial_invalid > 0:
        print(f"\n[POLY FORM]   ========== PHASE 1: Group-Based Pruning ==========")
        
        # Step 1: Extract all invalid edges
        print(f"[POLY FORM]   Step 1: Extracting invalid edges...")
        invalid_edges = []
        for edge, face_list in edge_face_map_all.items():
            if len(face_list) >= 3:  # Invalid edge
                invalid_edges.append(edge)
        
        print(f"[POLY FORM]     Found {len(invalid_edges)} invalid edges")
        
        # Step 2: Group invalid edges into connected components
        print(f"[POLY FORM]   Step 2: Grouping invalid edges by most frequent faces...")
        
        def find_edge_groups_by_frequent_faces(edges, edge_face_map):
            """
            Group edges using greedy algorithm based on most frequent faces:
            1. Count how many edges each face appears in
            2. Select face that appears in most edges
            3. Group all edges containing that face
            4. Remove those edges and repeat
            Each edge belongs to only one group.
            Returns list of groups and face for each group.
            """
            if not edges:
                return [], []
            
            # Map each edge to its set of faces
            edge_to_faces = {}
            for edge in edges:
                if edge in edge_face_map:
                    # Extract just the face indices from (face_idx, poly_idx, poly_type) tuples
                    face_indices = set(face_info[0] for face_info in edge_face_map[edge])
                    edge_to_faces[edge] = face_indices
                else:
                    edge_to_faces[edge] = set()
            
            groups = []
            group_faces = []  # The primary face for each group
            remaining_edges = set(edges)
            
            while remaining_edges:
                # Count how many remaining edges each face appears in
                face_edge_count = {}
                for edge in remaining_edges:
                    for face_idx in edge_to_faces[edge]:
                        if face_idx not in face_edge_count:
                            face_edge_count[face_idx] = 0
                        face_edge_count[face_idx] += 1
                
                if not face_edge_count:
                    # No faces found, group remaining edges together
                    groups.append(list(remaining_edges))
                    group_faces.append(None)
                    break
                
                # Find face that appears in most edges
                most_common_face = max(face_edge_count.keys(), key=lambda f: face_edge_count[f])
                
                # Group all edges containing this face
                current_group = []
                edges_to_remove = set()
                for edge in remaining_edges:
                    if most_common_face in edge_to_faces[edge]:
                        current_group.append(edge)
                        edges_to_remove.add(edge)
                
                groups.append(current_group)
                group_faces.append(most_common_face)
                
                # Remove these edges from remaining
                remaining_edges -= edges_to_remove
            
            return groups, group_faces
        
        edge_loops, group_primary_faces = find_edge_groups_by_frequent_faces(invalid_edges, edge_face_map_all)
        print(f"[POLY FORM]     Found {len(edge_loops)} edge groups (by most frequent faces)")
        
        # Print group details with face information
        for i, (loop, primary_face) in enumerate(zip(edge_loops, group_primary_faces)):
            # Collect unique vertices in this group
            vertices_in_group = set()
            # Collect all faces in this group
            all_faces_in_group = set()
            for edge in loop:
                vertices_in_group.add(edge[0])
                vertices_in_group.add(edge[1])
                if edge in edge_face_map_all:
                    for face_info in edge_face_map_all[edge]:
                        all_faces_in_group.add(face_info[0])
            
            # Convert to sorted list for display
            face_list = sorted(list(all_faces_in_group))
            faces_str = ", ".join([f"F{f+1}" for f in face_list]) if face_list else "no faces"
            primary_str = f"F{primary_face+1}" if primary_face is not None else "none"
            
            print(f"[POLY FORM]       Group {i+1}: {len(loop)} edges, "
                  f"{len(vertices_in_group)} vertices, primary face: {primary_str}, "
                  f"all faces: [{faces_str}]")
        
        # Step 3: For each group, find associated polygons
        print(f"[POLY FORM]   Step 3: Finding polygons for each group...")
        
        loop_polygon_sets = []  # Each entry: {'loop_idx': i, 'edges': [...], 'polygons': [...]}
        
        for loop_idx, loop in enumerate(edge_loops):
            # Find all polygons that contain at least one edge from this group
            loop_polygons = []
            loop_polygon_ids = set()  # Track (face_idx, poly_idx) to avoid duplicates
            
            for edge in loop:
                # Get all polygons associated with this edge
                face_list = edge_face_map_all.get(edge, [])
                for face_idx, poly_idx, poly_type in face_list:
                    poly_id = (face_idx, poly_idx)
                    if poly_id not in loop_polygon_ids:
                        loop_polygon_ids.add(poly_id)
                        # Find the polygon data
                        for poly_info in all_polygons_list:
                            if (poly_info['face_idx'] == face_idx and 
                                poly_info['polygon_idx'] == poly_idx):
                                loop_polygons.append(poly_info)
                                break
            
            loop_polygon_sets.append({
                'loop_idx': loop_idx,
                'edges': loop,
                'polygons': loop_polygons
            })
            
            print(f"[POLY FORM]       Loop {loop_idx+1}: {len(loop_polygons)} polygons")
        
        # Step 4: For each group, find best polygon combination to extract
        print(f"[POLY FORM]   Step 4: Finding optimal combinations for each group...")
        
        from itertools import combinations
        
        best_to_extract = []  # Accumulate best polygons to extract across all groups
        processed_edges = set()  # Track which edges have been resolved
        current_invalid_count = initial_invalid  # Track cumulative invalid count
        
        for loop_data in loop_polygon_sets:
            loop_idx = loop_data['loop_idx']
            loop_edges = loop_data['edges']
            loop_polygons = loop_data['polygons']
            
            print(f"\n[POLY FORM]     Processing Group {loop_idx+1}:")
            print(f"[POLY FORM]       Edges in group: {len(loop_edges)}")
            print(f"[POLY FORM]       Candidate polygons: {len(loop_polygons)}")
            print(f"[POLY FORM]       Starting invalid edges: {current_invalid_count}")
            
            # Skip if no polygons
            if not loop_polygons:
                continue
            
            # Filter out polygons from same face sharing edges in loop
            # For each edge in loop, ensure at most one polygon per face
            print(f"[POLY FORM]       Filtering polygons by face constraints...")
            filtered_polygons = []
            face_edge_polys = {}  # Track which faces have polygons for each edge
            
            for edge in loop_edges:
                edge_polys_by_face = {}
                # Find all polygons containing this edge
                for poly in loop_polygons:
                    poly_verts = poly['data'].get('vertices', [])
                    face_idx = poly['face_idx']
                    
                    # Check if polygon contains this edge
                    contains_edge = False
                    for i in range(len(poly_verts)):
                        v1 = poly_verts[i] - 1
                        v2 = poly_verts[(i + 1) % len(poly_verts)] - 1
                        poly_edge = (min(v1, v2), max(v1, v2))
                        if poly_edge == edge:
                            contains_edge = True
                            break
                    
                    if contains_edge:
                        if face_idx not in edge_polys_by_face:
                            edge_polys_by_face[face_idx] = []
                        edge_polys_by_face[face_idx].append(poly)
                
                face_edge_polys[edge] = edge_polys_by_face
            
            # Create filtered list with constraint that same face can contribute only one polygon per edge
            # Keep all polygons but note the mutual exclusivity
            filtered_polygons = loop_polygons  # Use all for now, constraint will be in combination logic
            
            # Pre-processing: Try single polygon removals to convert invalid to manifold
            print(f"[POLY FORM]       Pre-processing: Testing single polygon removals...")
            single_best = None
            single_best_reduction = 0
            single_best_manifold_conversions = 0
            
            # Build current edge map based on polygons already extracted
            cumulative_excluded = set((p['face_idx'], p['polygon_idx']) for p in best_to_extract)
            
            for poly in filtered_polygons:
                # Test removing just this one polygon (in addition to already extracted)
                temp_edge_map = {}
                test_id = (poly['face_idx'], poly['polygon_idx'])
                excluded_ids = cumulative_excluded | {test_id}
                
                # Rebuild edge map excluding already extracted + this polygon
                for poly_info in all_polygons_list:
                    if (poly_info['face_idx'], poly_info['polygon_idx']) in excluded_ids:
                        continue  # Skip this polygon
                    
                    poly_verts = poly_info['data'].get('vertices', [])
                    face_idx = poly_info['face_idx']
                    poly_idx = poly_info['polygon_idx']
                    poly_type = poly_info['poly_type']
                    
                    for i in range(len(poly_verts)):
                        v1 = poly_verts[i] - 1
                        v2 = poly_verts[(i + 1) % len(poly_verts)] - 1
                        edge = (min(v1, v2), max(v1, v2))
                        if edge not in temp_edge_map:
                            temp_edge_map[edge] = []
                        temp_edge_map[edge].append((face_idx, poly_idx, poly_type))
                
                # Count how many invalid edges are converted to manifold (2 faces)
                manifold_conversions = 0
                invalid_remaining = 0
                for edge, face_list in temp_edge_map.items():
                    if edge in loop_edges:
                        if len(face_list) == 2:
                            manifold_conversions += 1
                        elif len(face_list) >= 3:
                            invalid_remaining += 1
                
                # Calculate total invalid count and boundary count
                temp_invalid_count = sum(1 for e, fl in temp_edge_map.items() if len(fl) >= 3)
                temp_boundary_count = sum(1 for e, fl in temp_edge_map.items() if len(fl) == 1)
                
                reduction = current_invalid_count - temp_invalid_count
                
                if manifold_conversions > single_best_manifold_conversions or \
                   (manifold_conversions == single_best_manifold_conversions and reduction > single_best_reduction):
                    single_best = poly
                    single_best_reduction = reduction
                    single_best_manifold_conversions = manifold_conversions
            
            if single_best and single_best_manifold_conversions > 0:
                print(f"[POLY FORM]         Single polygon test: Face {single_best['face_idx']+1}, "
                      f"Polygon {single_best['polygon_idx']} converts {single_best_manifold_conversions} "
                      f"edge(s) to manifold (testing all combinations...)")
            
            # Limit combination testing to avoid exponential blowup
            from itertools import combinations
            max_test_size = min(len(filtered_polygons), 15)
            
            # Get initial boundary count before any extraction
            initial_boundary_count = sum(1 for e, fl in edge_face_map_all.items() if len(fl) == 1)
            
            best_combination = None
            best_score = float('inf')  # Minimize (B_increase - I_reduction)
            best_reduction = -1
            best_remaining_invalid = float('inf')
            best_boundary_increase = float('inf')
            
            # Test combinations of different sizes
            for combo_size in range(1, max_test_size + 1):
                # Limit number of combinations to test
                max_combos_per_size = 1000
                combo_count = 0
                
                for combo in combinations(filtered_polygons, combo_size):
                    combo_count += 1
                    if combo_count > max_combos_per_size:
                        break
                    
                    # Create temporary edge map without these polygons + already extracted
                    temp_edge_map = {}
                    combo_ids = set((p['face_idx'], p['polygon_idx']) for p in combo)
                    excluded_ids = cumulative_excluded | combo_ids
                    
                    # Rebuild edge map excluding already extracted + combo polygons
                    for poly_info in all_polygons_list:
                        if (poly_info['face_idx'], poly_info['polygon_idx']) in excluded_ids:
                            continue  # Skip this polygon
                        
                        poly_verts = poly_info['data'].get('vertices', [])
                        face_idx = poly_info['face_idx']
                        poly_idx = poly_info['polygon_idx']
                        poly_type = poly_info['poly_type']
                        
                        for i in range(len(poly_verts)):
                            v1 = poly_verts[i] - 1
                            v2 = poly_verts[(i + 1) % len(poly_verts)] - 1
                            edge = (min(v1, v2), max(v1, v2))
                            if edge not in temp_edge_map:
                                temp_edge_map[edge] = []
                            temp_edge_map[edge].append((face_idx, poly_idx, poly_type))
                    
                    # Count invalid and boundary edges in this configuration
                    temp_invalid_count = 0
                    temp_boundary_count = 0
                    for edge, face_list in temp_edge_map.items():
                        if len(face_list) >= 3:
                            temp_invalid_count += 1
                        elif len(face_list) == 1:
                            temp_boundary_count += 1
                    
                    # Calculate reduction in invalid edges and increase in boundary edges
                    invalid_reduction = current_invalid_count - temp_invalid_count
                    boundary_increase = temp_boundary_count - initial_boundary_count
                    
                    # New optimization goal: minimize (B_increase - I_reduction)
                    score = boundary_increase - invalid_reduction
                    
                    # Keep track of best combination
                    # Priority: 1) Lowest score (B_increase - I_reduction), 2) Tiebreaker: fewest polygons
                    is_better = False
                    if score < best_score:
                        is_better = True
                    elif score == best_score:
                        # Tiebreaker: prefer fewer polygons
                        if best_combination is None or len(combo) < len(best_combination):
                            is_better = True
                    
                    if is_better:
                        best_score = score
                        best_reduction = invalid_reduction
                        best_remaining_invalid = temp_invalid_count
                        best_boundary_increase = boundary_increase
                        best_combination = combo
            
            # Report best combination for this loop
            if best_combination:
                print(f"[POLY FORM]       Best combination: {len(best_combination)} polygons")
                print(f"[POLY FORM]       Optimization score (B_increase - I_reduction): {best_score}")
                print(f"[POLY FORM]       Boundary increase: {best_boundary_increase}")
                print(f"[POLY FORM]       Invalid edge reduction: {best_reduction}")
                print(f"[POLY FORM]       Remaining invalid edges: {best_remaining_invalid}")
                
                # Update current invalid count for next group
                current_invalid_count = best_remaining_invalid
                
                # Add to extraction list (avoid duplicates)
                for poly in best_combination:
                    poly_id = (poly['face_idx'], poly['polygon_idx'])
                    # Check if not already in extraction list
                    existing_ids = [(p['face_idx'], p['polygon_idx']) for p in best_to_extract]
                    if poly_id not in existing_ids:
                        best_to_extract.append(poly)
                
                # Mark edges as processed
                for edge in loop_edges:
                    processed_edges.add(edge)
            else:
                print(f"[POLY FORM]       No improvement found for this loop")
        
        # =======================================================================
        # PHASE 1 RESULTS: Detailed reporting
        # =======================================================================
        print(f"\n[POLY FORM]   ========== GROUP-BASED PRUNING RESULTS ==========")
        print(f"[POLY FORM]   Total polygons extracted: {len(best_to_extract)}")
        print(f"[POLY FORM]   Edges processed: {len(processed_edges)} of {len(invalid_edges)} invalid edges")
        
        # Report extracted polygons by group
        if best_to_extract:
            print(f"\n[POLY FORM]   Extracted polygons by group:")
            loop_extraction_map = {}  # loop_idx -> list of unique polygons
            
            # Map each extracted polygon back to its groups (a polygon can appear in multiple groups)
            for poly_info in best_to_extract:
                poly_id = (poly_info['face_idx'], poly_info['polygon_idx'])
                # Find ALL groups this polygon belongs to
                poly_loops = []
                for loop_data in loop_polygon_sets:
                    loop_idx = loop_data['loop_idx']
                    loop_poly_ids = [(p['face_idx'], p['polygon_idx']) for p in loop_data['polygons']]
                    if poly_id in loop_poly_ids:
                        poly_loops.append(loop_idx)
                
                # Add to first group found (primary association)
                if poly_loops:
                    primary_loop = poly_loops[0]
                    if primary_loop not in loop_extraction_map:
                        loop_extraction_map[primary_loop] = []
                    loop_extraction_map[primary_loop].append((poly_info, poly_loops))
            
            # Print by group
            for loop_idx in sorted(loop_extraction_map.keys()):
                polys_with_loops = loop_extraction_map[loop_idx]
                print(f"[POLY FORM]     Group {loop_idx+1}: {len(polys_with_loops)} polygon(s) extracted")
                for poly_info, poly_loops in polys_with_loops:
                    face_idx = poly_info['face_idx']
                    poly_idx = poly_info['polygon_idx']
                    poly_type = poly_info['poly_type']
                    poly_verts = poly_info['data'].get('vertices', [])
                    groups_str = f" (also in groups {[l+1 for l in poly_loops[1:]]}" if len(poly_loops) > 1 else ""
                    print(f"[POLY FORM]       - Face {face_idx+1}, Polygon {poly_idx}, "
                          f"Type: {poly_type}, {len(poly_verts)} vertices{groups_str}")
                    print(f"[POLY FORM]         Vertices: {poly_verts}")
        
        # Compute remaining invalid edges after group-based extraction
        remaining_edge_map = {}
        extracted_ids = set((p['face_idx'], p['polygon_idx']) for p in best_to_extract)
        
        for poly_info in all_polygons_list:
            if (poly_info['face_idx'], poly_info['polygon_idx']) in extracted_ids:
                continue  # Skip extracted polygons
            
            poly_verts = poly_info['data'].get('vertices', [])
            face_idx = poly_info['face_idx']
            poly_idx = poly_info['polygon_idx']
            poly_type = poly_info['poly_type']
            
            for i in range(len(poly_verts)):
                v1 = poly_verts[i] - 1
                v2 = poly_verts[(i + 1) % len(poly_verts)] - 1
                edge = (min(v1, v2), max(v1, v2))
                if edge not in remaining_edge_map:
                    remaining_edge_map[edge] = []
                remaining_edge_map[edge].append((face_idx, poly_idx, poly_type))
        
        # Compute edge statistics for extracted set
        extracted_edge_map = {}
        for poly_info in best_to_extract:
            poly_verts = poly_info['data'].get('vertices', [])
            face_idx = poly_info['face_idx']
            poly_idx = poly_info['polygon_idx']
            poly_type = poly_info['poly_type']
            
            for i in range(len(poly_verts)):
                v1 = poly_verts[i] - 1
                v2 = poly_verts[(i + 1) % len(poly_verts)] - 1
                edge = (min(v1, v2), max(v1, v2))
                if edge not in extracted_edge_map:
                    extracted_edge_map[edge] = []
                extracted_edge_map[edge].append((face_idx, poly_idx, poly_type))
        
        b_extracted, m_extracted, inv_extracted = compute_edge_stats(extracted_edge_map)
        b_remaining, m_remaining, inv_remaining = compute_edge_stats(remaining_edge_map)
        
        print(f"\n[POLY FORM]   Edge statistics after group-based pruning:")
        print(f"[POLY FORM]     Extracted set ({len(best_to_extract)} polygons):")
        print(f"[POLY FORM]       - Boundary edges (1 face): {b_extracted}")
        print(f"[POLY FORM]       - Manifold edges (2 faces): {m_extracted}")
        print(f"[POLY FORM]       - Invalid edges (3+ faces): {inv_extracted}")
        print(f"[POLY FORM]     Remaining set ({len(all_polygons_list) - len(best_to_extract)} polygons):")
        print(f"[POLY FORM]       - Boundary edges (1 face): {b_remaining}")
        print(f"[POLY FORM]       - Manifold edges (2 faces): {m_remaining}")
        print(f"[POLY FORM]       - Invalid edges (3+ faces): {inv_remaining}")
        
        remaining_invalid_edges = [e for e, fl in remaining_edge_map.items() if len(fl) >= 3]
        
        if remaining_invalid_edges:
            print(f"\n[POLY FORM]   WARNING: {len(remaining_invalid_edges)} invalid edges remain after group-based pruning")
            print(f"[POLY FORM]   Remaining invalid edges:")
            for edge in remaining_invalid_edges[:10]:  # Show first 10
                face_list = remaining_edge_map[edge]
                print(f"[POLY FORM]     Edge {edge}: {len(face_list)} faces")
            if len(remaining_invalid_edges) > 10:
                print(f"[POLY FORM]     ... and {len(remaining_invalid_edges) - 10} more")
        else:
            print(f"\n[POLY FORM]   ✓ All invalid edges resolved by group-based pruning!")
        
        print(f"[POLY FORM]   " + "=" * 50)
        
        # Display polygon edge analysis after Phase 1 (remaining polygons only)
        print(f"\n[POLY FORM]       ========== POLYGON EDGE ANALYSIS (AFTER PHASE 1) ==========")
        print(f"[POLY FORM]       Face Poly     Type Verts   B   M   I Vertices")
        print(f"[POLY FORM]       ---- ---- -------- ----- --- --- --- " + "-" * 50)
        
        # Build list of remaining polygons (not extracted)
        extracted_ids = set((p['face_idx'], p['polygon_idx']) for p in best_to_extract)
        remaining_polygons = []
        for poly_info in all_polygons_list:
            poly_id = (poly_info['face_idx'], poly_info['polygon_idx'])
            if poly_id not in extracted_ids:
                remaining_polygons.append(poly_info)
        
        # Analyze each remaining polygon
        for poly_info in remaining_polygons:
            face_idx = poly_info['face_idx']
            poly_idx = poly_info['polygon_idx']
            poly_type = poly_info['poly_type']
            poly_verts = poly_info['data'].get('vertices', [])
            
            # Count B, M, I edges for this polygon
            poly_edges = []
            for i in range(len(poly_verts)):
                v1 = poly_verts[i] - 1
                v2 = poly_verts[(i + 1) % len(poly_verts)] - 1
                edge = (min(v1, v2), max(v1, v2))
                poly_edges.append(edge)
            
            # Get edge counts from remaining edge map
            b_count = sum(1 for e in poly_edges if e in remaining_edge_map and len(remaining_edge_map[e]) == 1)
            m_count = sum(1 for e in poly_edges if e in remaining_edge_map and len(remaining_edge_map[e]) == 2)
            i_count = sum(1 for e in poly_edges if e in remaining_edge_map and len(remaining_edge_map[e]) >= 3)
            
            verts_str = str(poly_verts[:10]) + ('...' if len(poly_verts) > 10 else '')
            print(f"[POLY FORM]       {face_idx+1:4d} {poly_idx:4d} {poly_type:>8s} {len(poly_verts):5d} {b_count:3d} {m_count:3d} {i_count:3d} {verts_str}")
        
        print(f"[POLY FORM]       " + "=" * 86)
        
        # Mark extracted polygons as removed in unique_faces
        # If removing a BOUNDARY, promote an ALT to BOUNDARY
        print(f"\n[POLY FORM]   Marking {len(best_to_extract)} extracted polygons as removed...")
        for poly_info in best_to_extract:
            face_idx = poly_info['face_idx']
            poly_idx = poly_info['polygon_idx']
            poly_type = poly_info['poly_type']
            
            if face_idx < len(unique_faces):
                polygons = unique_faces[face_idx].get('polygons', [])
                if poly_idx < len(polygons):
                    polygons[poly_idx]['removed'] = True
                    print(f"[POLY FORM]     Marked Face {face_idx+1}, Polygon {poly_idx} ({poly_type}) as removed")
                    
                    # If we removed a BOUNDARY, promote an ALT to BOUNDARY
                    if poly_type == 'BOUNDARY':
                        # Find an ALT polygon in this face to promote
                        alt_to_promote = None
                        for p_idx, p in enumerate(polygons):
                            if p.get('polygon_type') == 'ALT' and not p.get('removed', False):
                                alt_to_promote = (p_idx, p)
                                break
                        
                        if alt_to_promote:
                            alt_idx, alt_poly = alt_to_promote
                            alt_poly['polygon_type'] = 'BOUNDARY'
                            alt_verts = alt_poly.get('vertices', [])
                            print(f"[POLY FORM]       → Promoted ALT Polygon {alt_idx} {alt_verts} to BOUNDARY")
                        else:
                            print(f"[POLY FORM]       → WARNING: No ALT available to promote to BOUNDARY")
        
        # =======================================================================
        # PHASE 2: Continue with existing pruning strategy if needed
        # =======================================================================
        if remaining_invalid_edges:
            print(f"\n[POLY FORM]   ========== PHASE 2: Standard Pruning ==========")
        
        print(f"\n[POLY FORM]   Extracting polygons with invalid edges...")
        
        # Find all polygons that contribute to remaining invalid edges
        polygons_with_invalid_edges = set()
        for edge, face_list in remaining_edge_map.items():
            if len(face_list) >= 3:  # Invalid edge
                for face_idx, poly_idx, poly_type in face_list:
                    polygons_with_invalid_edges.add((face_idx, poly_idx))
        
        # Create separate list of these polygons
        invalid_edge_polygons_all = []
        for face_idx, poly_idx in polygons_with_invalid_edges:
            # Find the polygon data
            for poly_info in all_polygons_list:
                if (poly_info['face_idx'] == face_idx and 
                    poly_info['polygon_idx'] == poly_idx):
                    invalid_edge_polygons_all.append(poly_info)
                    break
        
        print(f"[POLY FORM]     Found {len(invalid_edge_polygons_all)} "
              f"polygons with invalid edges (before filtering)")
        
        # Filter to keep only ALT polygons and BOUNDARY polygons (with holes)
        # from faces that have ALT polygons
        print(f"[POLY FORM]     Filtering to keep only ALT and their BOUNDARY "
              f"polygons...")
        
        # First, find all faces that have ALT polygons
        faces_with_alts_set = set()
        for poly_info in invalid_edge_polygons_all:
            if poly_info['poly_type'] == 'ALT':
                faces_with_alts_set.add(poly_info['face_idx'])
        
        # Now keep only ALT polygons and BOUNDARY/HOLE polygons from those faces
        invalid_edge_polygons = []
        returned_to_base = []
        
        for poly_info in invalid_edge_polygons_all:
            face_idx = poly_info['face_idx']
            poly_type = poly_info['poly_type']
            
            # Keep if it's an ALT polygon
            if poly_type == 'ALT':
                invalid_edge_polygons.append(poly_info)
            # Keep if it's a BOUNDARY or HOLE from a face that has ALTs
            elif (poly_type in ['BOUNDARY', 'HOLE'] and 
                  face_idx in faces_with_alts_set):
                invalid_edge_polygons.append(poly_info)
            # Otherwise, return to base polygons
            else:
                returned_to_base.append(poly_info)
        # SKIP FILTERING: For testing, send all polygons with invalid edges directly to pruning
        print(f"[POLY FORM]     SKIPPING FILTERING: All polygons with invalid edges sent to pruning")
        
        # Group polygons by face and polygon type to keep units together
        # Build a map: face_idx -> {BOUNDARY: [poly_info], ALT: [poly_info], HOLE: [poly_info]}
        face_polygon_groups = {}
        for poly_info in invalid_edge_polygons_all:
            face_idx = poly_info['face_idx']
            poly_type = poly_info['poly_type']
            
            if face_idx not in face_polygon_groups:
                face_polygon_groups[face_idx] = {'BOUNDARY': [], 'ALT': [], 'HOLE': []}
            
            face_polygon_groups[face_idx][poly_type].append(poly_info)
        
        # For each face, group BOUNDARY with its HOLES, and each ALT with its HOLES
        # Create polygon units that will move together
        polygon_units = []  # Each unit is a list of poly_info objects that move together
        
        for face_idx, groups in face_polygon_groups.items():
            # Group BOUNDARY with HOLES that don't belong to any ALT
            if groups['BOUNDARY']:
                boundary_unit = list(groups['BOUNDARY'])
                
                # Add HOLES that are children of BOUNDARY (not ALT)
                for hole_info in groups['HOLE']:
                    parent = hole_info['data'].get('parent_polygon', '')
                    if not parent.startswith('ALT_'):
                        boundary_unit.append(hole_info)
                
                if boundary_unit:
                    polygon_units.append(boundary_unit)
            
            # Group each ALT with its HOLES
            for alt_info in groups['ALT']:
                alt_poly_idx = alt_info['polygon_idx']
                alt_unit = [alt_info]
                
                # Add HOLES that belong to this ALT
                for hole_info in groups['HOLE']:
                    parent = hole_info['data'].get('parent_polygon', '')
                    if parent == f'ALT_{alt_poly_idx}':
                        alt_unit.append(hole_info)
                
                polygon_units.append(alt_unit)
        
        print(f"[POLY FORM]     Created {len(polygon_units)} polygon units from {len(invalid_edge_polygons_all)} polygons")
        
        # Flatten units back to list for initial processing
        invalid_edge_polygons = list(invalid_edge_polygons_all)
        returned_to_base = []
        pruned_polygons = []  # Track pruned polygons separately
        print(f"[POLY FORM]     Polygons for combination testing:")
        # print(f"[POLY FORM]     After filtering:")
        # print(f"[POLY FORM]       - Kept for combination testing: "
        #       f"{len(invalid_edge_polygons)}")
        # print(f"[POLY FORM]       - Returned to base (always kept): "
        #       f"{len(returned_to_base)}")
        
        # if returned_to_base:
        #     print(f"[POLY FORM]     Polygons returned to base:")
        #     for poly_info in returned_to_base:
        #         poly_verts = poly_info['data'].get('vertices', [])
        #         poly_type = poly_info['poly_type']
        #         face_idx = poly_info['face_idx']
        #         print(f"[POLY FORM]       - Face {face_idx+1}, Type: "
        #               f"{poly_type}, Vertices: {poly_verts}")
        
        # print(f"[POLY FORM]     Polygons for combination testing:")


        # --- Iterative Pruning: Remove polygons that cause invalid edges ---
        # Run up to 5 times or until extracted set has < 20 polygons
        max_pruning_iterations = 5
        min_polygons_threshold = 20
        
        for pruning_iteration in range(1, max_pruning_iterations + 1):
            print(f"\n[POLY FORM]   ========== Pruning Iteration {pruning_iteration} ==========")
            print(f"[POLY FORM]   Current extracted set size: {len(invalid_edge_polygons)} polygons")
            
            # Check if we should stop
            if len(invalid_edge_polygons) < min_polygons_threshold:
                print(f"[POLY FORM]   Extracted set has < {min_polygons_threshold} polygons - stopping pruning")
                break
            
            print(f"\n[POLY FORM]   Starting pruning analysis...")
            
            # Compute edge distribution for current extracted set
            temp_edge_map = {}
            for poly_info in invalid_edge_polygons:
                poly_verts = poly_info['data'].get('vertices', [])
                face_idx = poly_info['face_idx']
                poly_idx = poly_info['polygon_idx']
                poly_type = poly_info['poly_type']
                for i in range(len(poly_verts)):
                    v1 = poly_verts[i] - 1
                    v2 = poly_verts[(i + 1) % len(poly_verts)] - 1
                    edge = (min(v1, v2), max(v1, v2))
                    if edge not in temp_edge_map:
                        temp_edge_map[edge] = []
                    temp_edge_map[edge].append((face_idx, poly_idx, poly_type))
            b_extracted, m_extracted, inv_extracted = compute_edge_stats(temp_edge_map)
            
            # Compute edge distribution for base set
            base_edge_map = {}
            extracted_set = set((p['face_idx'], p['polygon_idx']) for p in invalid_edge_polygons)
            pruned_set = set((p['face_idx'], p['polygon_idx']) for p in pruned_polygons)
            for poly_info in all_polygons_list:
                if ((poly_info['face_idx'], poly_info['polygon_idx']) not in extracted_set and
                    (poly_info['face_idx'], poly_info['polygon_idx']) not in pruned_set):
                    poly_verts = poly_info['data'].get('vertices', [])
                    face_idx = poly_info['face_idx']
                    poly_idx = poly_info['polygon_idx']
                    poly_type = poly_info['poly_type']
                    for i in range(len(poly_verts)):
                        v1 = poly_verts[i] - 1
                        v2 = poly_verts[(i + 1) % len(poly_verts)] - 1
                        edge = (min(v1, v2), max(v1, v2))
                        if edge not in base_edge_map:
                            base_edge_map[edge] = []
                        base_edge_map[edge].append((face_idx, poly_idx, poly_type))
            b_base, m_base, inv_base = compute_edge_stats(base_edge_map)
            
            print(f"[POLY FORM]   Initial edge distribution before pruning:")
            print(f"[POLY FORM]     Base Set (remaining polygons):")
            print(f"[POLY FORM]       - Boundary edges (1 face): {b_base}")
            print(f"[POLY FORM]       - Manifold edges (2 faces): {m_base}")
            print(f"[POLY FORM]       - Invalid edges (3+ faces): {inv_base}")
            print(f"[POLY FORM]     Extracted Set (invalid edge polygons):")
            print(f"[POLY FORM]       - Boundary edges (1 face): {b_extracted}")
            print(f"[POLY FORM]       - Manifold edges (2 faces): {m_extracted}")
            print(f"[POLY FORM]       - Invalid edges (3+ faces): {inv_extracted}")
            
            # Build edge-to-polygon mapping for invalid edges
            edge_count = {}
            edge_to_poly_indices = {}
            for idx, poly_info in enumerate(invalid_edge_polygons):
                poly_verts = poly_info['data'].get('vertices', [])
                for i in range(len(poly_verts)):
                    v1 = poly_verts[i] - 1
                    v2 = poly_verts[(i + 1) % len(poly_verts)] - 1
                    edge = (min(v1, v2), max(v1, v2))
                    if edge not in edge_count:
                        edge_count[edge] = 0
                        edge_to_poly_indices[edge] = []
                    edge_count[edge] += 1
                    edge_to_poly_indices[edge].append(idx)
            
            # Find all invalid edges (3+ occurrences)
            invalid_edges = {edge: count for edge, count in edge_count.items() if count >= 3}
            
            print(f"\n[POLY FORM]   Invalid edge analysis:")
            print(f"[POLY FORM]     Total invalid edges: {len(invalid_edges)}")
            
            # Show distribution of invalid edge sharing
            edge_sharing_dist = {}
            for edge, count in invalid_edges.items():
                if count not in edge_sharing_dist:
                    edge_sharing_dist[count] = 0
                edge_sharing_dist[count] += 1
            
            for count in sorted(edge_sharing_dist.keys()):
                print(f"[POLY FORM]       Edges shared by {count} polygons: {edge_sharing_dist[count]}")
            
            # Print each invalid edge and the polygons that contain it
            print(f"\n[POLY FORM]   Invalid edges and their polygons:")
            for edge in sorted(invalid_edges.keys()):
                edge_1based = (edge[0]+1, edge[1]+1)
                poly_indices = edge_to_poly_indices[edge]
                poly_labels = []
                for idx in poly_indices:
                    poly_info = invalid_edge_polygons[idx]
                    face_idx = poly_info['face_idx']
                    poly_type = poly_info['poly_type']
                    poly_label = f"[{chr(65+idx)}] F{face_idx+1}:{poly_type}"
                    poly_labels.append(poly_label)
                print(f"[POLY FORM]     Edge {edge_1based}: {', '.join(poly_labels)}")
            
            # Map each polygon to the invalid edges it contains
            poly_to_invalid_edges = {}
            for idx in range(len(invalid_edge_polygons)):
                poly_to_invalid_edges[idx] = []
            
            for edge in invalid_edges:
                for poly_idx in edge_to_poly_indices[edge]:
                    poly_to_invalid_edges[poly_idx].append(edge)
            
            # New strategy: For each invalid edge, identify polygons from same face
            # Keep ALL polygons from same face in extraction set (could be ALT+ALT, ALT+HOLE, BOUNDARY+HOLE, etc.)
            # Return single polygon from different face to base
            
            print(f"\n[POLY FORM]   Analyzing invalid edges for same-face polygon groups:")
            
            polygons_to_return_to_base = set()  # Polygon indices to return
            polygons_to_keep_in_extraction = set()  # Polygon indices to keep
            
            for edge in sorted(invalid_edges.keys()):
                edge_1based = (edge[0]+1, edge[1]+1)
                poly_indices = edge_to_poly_indices[edge]
                count = len(poly_indices)
                
                if count == 3:
                    # Group polygons by face
                    poly_infos = [(idx, invalid_edge_polygons[idx]) for idx in poly_indices]
                    
                    face_groups = {}
                    for idx, poly_info in poly_infos:
                        face_idx = poly_info['face_idx']
                        if face_idx not in face_groups:
                            face_groups[face_idx] = []
                        face_groups[face_idx].append((idx, poly_info))
                    
                    # Find face with 2+ polygons - keep them all in extraction
                    multi_poly_face_found = False
                    
                    for face_idx, polys in face_groups.items():
                        if len(polys) >= 2:
                            # Keep all polygons from this face in extraction
                            poly_labels = []
                            for idx, info in polys:
                                polygons_to_keep_in_extraction.add(idx)
                                poly_labels.append(f"[{chr(65+idx)}] F{face_idx+1}:{info['poly_type']}")
                            
                            print(f"[POLY FORM]     Edge {edge_1based}: Keep {len(polys)} from same face: {', '.join(poly_labels)}")
                            multi_poly_face_found = True
                            break
                    
                    if multi_poly_face_found:
                        # Return the single polygon(s) from other face(s) to base
                        for face_idx, polys in face_groups.items():
                            if len(polys) == 1:
                                single_poly_idx, single_poly_info = polys[0]
                                polygons_to_return_to_base.add(single_poly_idx)
                                print(f"[POLY FORM]       Return to base: [{chr(65+single_poly_idx)}] F{single_poly_info['face_idx']+1}:{single_poly_info['poly_type']}")
                    else:
                        # All 3 polygons from different faces - keep all in extraction
                        print(f"[POLY FORM]     Edge {edge_1based}: All from different faces - keeping all 3 in extraction")
                        print(f"[POLY FORM]     Edge {edge_1based}: No clear BOUNDARY+ALT pair - keeping all 3 in extraction")
                        for idx in poly_indices:
                            polygons_to_keep_in_extraction.add(idx)
                
                elif count > 3:
                    # More than 3 polygons - keep all in extraction set
                    print(f"[POLY FORM]     Edge {edge_1based}: {count} polygons - keeping all in extraction")
                    for idx in poly_indices:
                        polygons_to_keep_in_extraction.add(idx)
            
            # Check for conflicts: polygons marked both to keep AND to return
            conflicting_polygons = polygons_to_return_to_base & polygons_to_keep_in_extraction
            
            if conflicting_polygons:
                print(f"\n[POLY FORM]   WARNING: Found {len(conflicting_polygons)} polygon(s) with conflicting decisions:")
                for idx in sorted(conflicting_polygons):
                    poly_info = invalid_edge_polygons[idx]
                    print(f"[POLY FORM]     [{chr(65+idx)}] F{poly_info['face_idx']+1}:{poly_info['poly_type']} - marked BOTH to keep AND to return")
                    # Show which edges led to each decision
                    for edge in invalid_edges:
                        if idx in edge_to_poly_indices[edge]:
                            edge_1based = (edge[0]+1, edge[1]+1)
                            poly_indices = edge_to_poly_indices[edge]
                            if idx in polygons_to_keep_in_extraction:
                                # Check if this edge led to keep decision
                                face_groups = {}
                                for pi in poly_indices:
                                    pi_info = invalid_edge_polygons[pi]
                                    fi = pi_info['face_idx']
                                    if fi not in face_groups:
                                        face_groups[fi] = []
                                    face_groups[fi].append((pi, pi_info))
                                # Check if idx is in a face group with 2+ polygons
                                for fi, polys in face_groups.items():
                                    if len(polys) >= 2 and any(p[0] == idx for p in polys):
                                        print(f"[POLY FORM]       Edge {edge_1based}: KEEP (same face group with {len(polys)} polygons)")
                                        break
                            if idx in polygons_to_return_to_base:
                                # Check if this edge led to return decision
                                print(f"[POLY FORM]       Edge {edge_1based}: RETURN (single polygon from different face)")
                print(f"[POLY FORM]   Resolution: Keeping conflicting polygons in extraction (prioritizing KEEP over RETURN)")
            
            # Remove polygons from return list if they're in keep list (prioritize KEEP)
            # For conflicting polygons, keeping them in extraction preserves the face structure
            polygons_to_return_to_base -= polygons_to_keep_in_extraction
            
            print(f"\n[POLY FORM]   Pruning decision:")
            print(f"[POLY FORM]     Polygons to return to base: {len(polygons_to_return_to_base)}")
            print(f"[POLY FORM]     Polygons to keep in extraction: {len(polygons_to_keep_in_extraction)}")
            
            # Convert to the format expected by the rest of the code
            polygons_with_single_invalid = list(polygons_to_return_to_base)
            
            # Expand to include entire units
            units_to_remove = set()
            for idx in polygons_with_single_invalid:
                poly_info = invalid_edge_polygons[idx]
                # Find which unit this polygon belongs to
                for unit_idx, unit in enumerate(polygon_units):
                    if any(p['face_idx'] == poly_info['face_idx'] and 
                           p['polygon_idx'] == poly_info['polygon_idx'] for p in unit):
                        units_to_remove.add(unit_idx)
                        break
            
            # Convert unit indices to polygon indices
            expanded_to_remove = set()
            for unit_idx in units_to_remove:
                unit = polygon_units[unit_idx]
                for unit_poly in unit:
                    # Find index in invalid_edge_polygons
                    for idx, poly_info in enumerate(invalid_edge_polygons):
                        if (poly_info['face_idx'] == unit_poly['face_idx'] and
                            poly_info['polygon_idx'] == unit_poly['polygon_idx']):
                            expanded_to_remove.add(idx)
                            break
            
            to_remove = sorted(expanded_to_remove)
            
            if to_remove:
                # For iterations after the first, check if returning polygons would increase invalid edges in base set
                should_return_polygons = True
                
                if pruning_iteration > 1 and to_remove:
                    print(f"\n[POLY FORM]   Checking if returning {len(to_remove)} polygon(s) would increase invalid edges in base set...")
                    
                    # Compute current base set invalid edges (before returning polygons)
                    current_base_edge_map = {}
                    extracted_set = set((p['face_idx'], p['polygon_idx']) for p in invalid_edge_polygons)
                    pruned_set = set((p['face_idx'], p['polygon_idx']) for p in pruned_polygons)
                    for poly_info in all_polygons_list:
                        if ((poly_info['face_idx'], poly_info['polygon_idx']) not in extracted_set and
                            (poly_info['face_idx'], poly_info['polygon_idx']) not in pruned_set):
                            poly_verts = poly_info['data'].get('vertices', [])
                            face_idx = poly_info['face_idx']
                            poly_idx = poly_info['polygon_idx']
                            poly_type = poly_info['poly_type']
                            for i in range(len(poly_verts)):
                                v1 = poly_verts[i] - 1
                                v2 = poly_verts[(i + 1) % len(poly_verts)] - 1
                                edge = (min(v1, v2), max(v1, v2))
                                if edge not in current_base_edge_map:
                                    current_base_edge_map[edge] = []
                                current_base_edge_map[edge].append((face_idx, poly_idx, poly_type))
                    _, _, current_base_invalid = compute_edge_stats(current_base_edge_map)
                    
                    # Simulate adding to_remove polygons to base set
                    simulated_base_edge_map = dict(current_base_edge_map)
                    for idx in to_remove:
                        poly_info = invalid_edge_polygons[idx]
                        poly_verts = poly_info['data'].get('vertices', [])
                        face_idx = poly_info['face_idx']
                        poly_idx = poly_info['polygon_idx']
                        poly_type = poly_info['poly_type']
                        for i in range(len(poly_verts)):
                            v1 = poly_verts[i] - 1
                            v2 = poly_verts[(i + 1) % len(poly_verts)] - 1
                            edge = (min(v1, v2), max(v1, v2))
                            if edge not in simulated_base_edge_map:
                                simulated_base_edge_map[edge] = []
                            simulated_base_edge_map[edge].append((face_idx, poly_idx, poly_type))
                    _, _, simulated_base_invalid = compute_edge_stats(simulated_base_edge_map)
                    
                    print(f"[POLY FORM]     Current base set invalid edges: {current_base_invalid}")
                    print(f"[POLY FORM]     Simulated base set invalid edges (after return): {simulated_base_invalid}")
                    
                    if simulated_base_invalid > current_base_invalid:
                        print(f"[POLY FORM]     WARNING: Returning polygons would increase invalid edges in base set!")
                        print(f"[POLY FORM]     Running combinatorial testing with modified objective...")
                        should_return_polygons = False
                        
                        # ========== NEW STRATEGY: SEPARATE CLEAN AND PROBLEMATIC POLYGONS ==========
                        # Step 1: Identify which polygons contribute to invalid edges
                        print(f"\n[POLY FORM]   Identifying polygons contributing to invalid edges...")
                        
                        # Build edge map for extracted set to find invalid edges
                        extracted_edge_map = {}
                        for idx, poly_info in enumerate(invalid_edge_polygons):
                            poly_verts = poly_info['data'].get('vertices', [])
                            face_idx = poly_info['face_idx']
                            poly_idx = poly_info['polygon_idx']
                            poly_type = poly_info['poly_type']
                            for i in range(len(poly_verts)):
                                v1 = poly_verts[i] - 1
                                v2 = poly_verts[(i + 1) % len(poly_verts)] - 1
                                edge = (min(v1, v2), max(v1, v2))
                                if edge not in extracted_edge_map:
                                    extracted_edge_map[edge] = []
                                extracted_edge_map[edge].append((idx, face_idx, poly_idx, poly_type))
                        
                        # Find polygons that have invalid edges (3+ faces)
                        polygons_with_invalid_edges = set()
                        for edge, polys in extracted_edge_map.items():
                            if len(polys) >= 3:
                                for poly_idx, _, _, _ in polys:
                                    polygons_with_invalid_edges.add(poly_idx)
                        
                        # Separate into two groups
                        clean_polygons_indices = [i for i in range(len(invalid_edge_polygons)) if i not in polygons_with_invalid_edges]
                        problematic_polygons_indices = sorted(polygons_with_invalid_edges)
                        
                        print(f"[POLY FORM]     Group 1 (problematic): {len(problematic_polygons_indices)} polygons with invalid edges")
                        print(f"[POLY FORM]     Group 2 (clean): {len(clean_polygons_indices)} polygons without invalid edges")
                        
                        # Step 2: Try returning clean polygons to base set
                        if len(clean_polygons_indices) > 0:
                            print(f"\n[POLY FORM]   Testing if clean polygons can be returned to base set...")
                            clean_polygons_to_test = [invalid_edge_polygons[i] for i in clean_polygons_indices]
                            
                            # Build edge map with base + clean polygons (to simulate return)
                            test_return_edge_map = {}
                            
                            # Add base polygons
                            for poly_info in all_polygons_list:
                                face_idx = poly_info['face_idx']
                                poly_idx = poly_info['polygon_idx']
                                is_in_extracted = any(p['face_idx'] == face_idx and p['polygon_idx'] == poly_idx for p in invalid_edge_polygons)
                                is_in_pruned = (face_idx, poly_idx) in pruned_set
                                if not is_in_extracted and not is_in_pruned:
                                    poly_verts = poly_info['data'].get('vertices', [])
                                    poly_type = poly_info['poly_type']
                                    for i in range(len(poly_verts)):
                                        v1 = poly_verts[i] - 1
                                        v2 = poly_verts[(i + 1) % len(poly_verts)] - 1
                                        edge = (min(v1, v2), max(v1, v2))
                                        if edge not in test_return_edge_map:
                                            test_return_edge_map[edge] = []
                                        test_return_edge_map[edge].append((face_idx, poly_idx, poly_type))
                            
                            # Add clean polygons being tested for return
                            # NOTE: If clean polygon is ALT, test it as if it were BOUNDARY (post-conversion)
                            for poly_info in clean_polygons_to_test:
                                poly_verts = poly_info['data'].get('vertices', [])
                                face_idx = poly_info['face_idx']
                                poly_idx = poly_info['polygon_idx']
                                poly_type = poly_info['poly_type']
                                # If ALT, simulate as BOUNDARY for testing
                                if poly_type == 'ALT':
                                    poly_type = 'BOUNDARY'
                                for i in range(len(poly_verts)):
                                    v1 = poly_verts[i] - 1
                                    v2 = poly_verts[(i + 1) % len(poly_verts)] - 1
                                    edge = (min(v1, v2), max(v1, v2))
                                    if edge not in test_return_edge_map:
                                        test_return_edge_map[edge] = []
                                    test_return_edge_map[edge].append((face_idx, poly_idx, poly_type))
                            
                            # Check edge statistics
                            _, _, test_return_invalid = compute_edge_stats(test_return_edge_map)
                            
                            print(f"[POLY FORM]     Current base set invalid edges: {inv_base}")
                            print(f"[POLY FORM]     Simulated base set invalid edges (after returning clean polygons): {test_return_invalid}")
                            
                            if test_return_invalid > inv_base:
                                print(f"[POLY FORM]     WARNING: Returning clean polygons would increase invalid edges!")
                                print(f"[POLY FORM]     Keeping clean polygons in extraction set for now.")
                            else:
                                print(f"[POLY FORM]     SUCCESS: Clean polygons can be returned without increasing invalid edges!")
                                print(f"[POLY FORM]     Returning {len(clean_polygons_to_test)} clean polygons to base set.")
                                
                                # Track all polygons to remove (clean polygons + replaced BOUNDARY polygons)
                                polygons_to_remove = set()
                                conversion_actions = []  # Track what conversions happened
                                
                                # Process each clean polygon
                                for clean_idx in clean_polygons_indices:
                                    poly_info = invalid_edge_polygons[clean_idx]
                                    face_idx = poly_info['face_idx']
                                    poly_idx = poly_info['polygon_idx']
                                    poly_type = poly_info['poly_type']
                                    poly_label = chr(65 + clean_idx) if clean_idx < 26 else f"P{clean_idx}"
                                    
                                    # Mark for removal
                                    polygons_to_remove.add(clean_idx)
                                    
                                    # If returning an ALT polygon, convert it to BOUNDARY and find its original BOUNDARY
                                    if poly_type == 'ALT':
                                        conversion_actions.append({
                                            'type': 'convert_alt',
                                            'label': poly_label,
                                            'face_idx': face_idx,
                                            'poly_idx': poly_idx,
                                            'clean_idx': clean_idx
                                        })
                                        
                                        # Find the original BOUNDARY polygon for this face to remove
                                        for idx, p in enumerate(invalid_edge_polygons):
                                            if p['face_idx'] == face_idx and p['poly_type'] == 'BOUNDARY' and idx != clean_idx:
                                                boundary_label = chr(65 + idx) if idx < 26 else f"P{idx}"
                                                polygons_to_remove.add(idx)
                                                conversion_actions.append({
                                                    'type': 'remove_boundary',
                                                    'label': boundary_label,
                                                    'face_idx': face_idx,
                                                    'poly_idx': p['polygon_idx'],
                                                    'idx': idx
                                                })
                                                break
                                    else:
                                        conversion_actions.append({
                                            'type': 'return_as_is',
                                            'label': poly_label,
                                            'face_idx': face_idx,
                                            'poly_type': poly_type,
                                            'clean_idx': clean_idx
                                        })
                                        
                                        # If returning a BOUNDARY, check if there's an associated ALT to remove
                                        if poly_type == 'BOUNDARY':
                                            for idx, p in enumerate(invalid_edge_polygons):
                                                if p['face_idx'] == face_idx and p['poly_type'] == 'ALT' and idx != clean_idx:
                                                    alt_label = chr(65 + idx) if idx < 26 else f"P{idx}"
                                                    polygons_to_remove.add(idx)
                                                    conversion_actions.append({
                                                        'type': 'remove_alt',
                                                        'label': alt_label,
                                                        'face_idx': face_idx,
                                                        'poly_idx': p['polygon_idx'],
                                                        'idx': idx
                                                    })
                                                    break
                                
                                # Print what we're doing
                                for action in conversion_actions:
                                    if action['type'] == 'convert_alt':
                                        print(f"[POLY FORM]       [{action['label']}] F{action['face_idx']+1}:ALT converted to BOUNDARY and returned")
                                    elif action['type'] == 'remove_boundary':
                                        print(f"[POLY FORM]         Removing original BOUNDARY [{action['label']}] F{action['face_idx']+1}:BOUNDARY from extraction")
                                    elif action['type'] == 'remove_alt':
                                        print(f"[POLY FORM]         Removing associated ALT [{action['label']}] F{action['face_idx']+1}:ALT from extraction")
                                    elif action['type'] == 'return_as_is':
                                        print(f"[POLY FORM]       [{action['label']}] F{action['face_idx']+1}:{action['poly_type']} returned to base")
                                        
                                        # If this is a BOUNDARY, check for associated HOLEs
                                        if action['poly_type'] == 'BOUNDARY':
                                            face_idx = action['face_idx']
                                            associated_holes = [idx for idx, p in enumerate(invalid_edge_polygons) 
                                                              if p['face_idx'] == face_idx and p['poly_type'] == 'HOLE' and idx not in polygons_to_remove]
                                            if associated_holes:
                                                hole_labels = [chr(65 + h) if h < 26 else f"P{h}" for h in associated_holes]
                                                print(f"[POLY FORM]         Note: Face has {len(associated_holes)} HOLE(s): [{', '.join(hole_labels)}] - keeping with parent")
                                
                                # Now perform the actual conversions and removals
                                for action in conversion_actions:
                                    if action['type'] == 'convert_alt':
                                        # Convert ALT to BOUNDARY in the face data
                                        face_eq = unique_faces[action['face_idx']]
                                        if 'polygons' in face_eq:
                                            face_eq['polygons'][action['poly_idx']]['polygon_type'] = 'BOUNDARY'
                                            # Mark as not removed (returned to base)
                                            face_eq['polygons'][action['poly_idx']]['removed'] = False
                                    elif action['type'] == 'remove_boundary':
                                        # Mark BOUNDARY as removed in face data
                                        face_eq = unique_faces[action['face_idx']]
                                        if 'polygons' in face_eq:
                                            face_eq['polygons'][action['poly_idx']]['removed'] = True
                                    elif action['type'] == 'remove_alt':
                                        # Mark ALT as removed in face data
                                        face_eq = unique_faces[action['face_idx']]
                                        if 'polygons' in face_eq:
                                            face_eq['polygons'][action['poly_idx']]['removed'] = True
                                    elif action['type'] == 'return_as_is':
                                        # Mark as not removed (returned to base)
                                        face_eq = unique_faces[action['face_idx']]
                                        if 'polygons' in face_eq:
                                            poly_idx = invalid_edge_polygons[action['clean_idx']]['polygon_idx']
                                            if 'removed' not in face_eq['polygons'][poly_idx]:
                                                face_eq['polygons'][poly_idx]['removed'] = False
                                
                                # Remove all marked polygons from extracted set
                                for idx in sorted(polygons_to_remove, reverse=True):
                                    invalid_edge_polygons.pop(idx)
                                
                                print(f"[POLY FORM]     Updated extracted set size: {len(invalid_edge_polygons)} polygons")
                                print(f"[POLY FORM]     Total polygons removed from extraction: {len(polygons_to_remove)}")
                        
                        # Terminate iteration after handling clean polygons
                        print(f"\n[POLY FORM]   Terminating pruning iteration after clean polygon processing.")
                        break  # Exit the pruning loop
                
                if should_return_polygons:
                    print(f"\n[POLY FORM]   Returning {len(to_remove)} polygon(s) in {len(units_to_remove)} unit(s) to base set")
                    
                    for idx in to_remove:
                        poly_info = invalid_edge_polygons[idx]
                        poly_verts = poly_info['data'].get('vertices', [])
                        poly_type = poly_info['poly_type']
                        face_idx = poly_info['face_idx']
                        print(f"[POLY FORM]       - [{chr(65+idx)}] Face {face_idx+1}, Type: {poly_type}, Vertices: {poly_verts}")
                    
                    # Move removed polygons back to base (not to pruned set)
                    returned_to_base = []
                    for idx in sorted(to_remove, reverse=True):
                        returned_to_base.append(invalid_edge_polygons[idx])
                    
                    # Remove these polygons from the extracted set
                    invalid_edge_polygons = [p for i, p in enumerate(invalid_edge_polygons) if i not in to_remove]
                    
                    # Update polygon_units to remove deleted units
                    polygon_units = [unit for unit_idx, unit in enumerate(polygon_units) if unit_idx not in units_to_remove]
                    
                    print(f"[POLY FORM]   Returned {len(returned_to_base)} polygons to base set")
                else:
                    # Don't return polygons - combinatorial testing was done above
                    print(f"[POLY FORM]   Polygons not returned - combinatorial testing completed")
            else:
                print(f"\n[POLY FORM]   No polygons to return to base - all needed for extraction")


            # Print final edge distribution after pruning
            temp_edge_map = {}
            for poly_info in invalid_edge_polygons:
                poly_verts = poly_info['data'].get('vertices', [])
                face_idx = poly_info['face_idx']
                poly_idx = poly_info['polygon_idx']
                poly_type = poly_info['poly_type']
                for i in range(len(poly_verts)):
                    v1 = poly_verts[i] - 1
                    v2 = poly_verts[(i + 1) % len(poly_verts)] - 1
                    edge = (min(v1, v2), max(v1, v2))
                    if edge not in temp_edge_map:
                        temp_edge_map[edge] = []
                    temp_edge_map[edge].append((face_idx, poly_idx, poly_type))
            b_extracted, m_extracted, inv_extracted = compute_edge_stats(temp_edge_map)
            
            # Compute edge distribution for base set after pruning (excluding pruned polygons)
            base_edge_map = {}
            extracted_set = set((p['face_idx'], p['polygon_idx']) for p in invalid_edge_polygons)
            pruned_set = set((p['face_idx'], p['polygon_idx']) for p in pruned_polygons)
            for poly_info in all_polygons_list:
                if ((poly_info['face_idx'], poly_info['polygon_idx']) not in extracted_set and
                    (poly_info['face_idx'], poly_info['polygon_idx']) not in pruned_set):
                    poly_verts = poly_info['data'].get('vertices', [])
                    face_idx = poly_info['face_idx']
                    poly_idx = poly_info['polygon_idx']
                    poly_type = poly_info['poly_type']
                    for i in range(len(poly_verts)):
                        v1 = poly_verts[i] - 1
                        v2 = poly_verts[(i + 1) % len(poly_verts)] - 1
                        edge = (min(v1, v2), max(v1, v2))
                        if edge not in base_edge_map:
                            base_edge_map[edge] = []
                        base_edge_map[edge].append((face_idx, poly_idx, poly_type))
            b_base, m_base, inv_base = compute_edge_stats(base_edge_map)
            
            # Compute edge distribution for pruned set
            pruned_edge_map = {}
            for poly_info in pruned_polygons:
                poly_verts = poly_info['data'].get('vertices', [])
                face_idx = poly_info['face_idx']
                poly_idx = poly_info['polygon_idx']
                poly_type = poly_info['poly_type']
                for i in range(len(poly_verts)):
                    v1 = poly_verts[i] - 1
                    v2 = poly_verts[(i + 1) % len(poly_verts)] - 1
                    edge = (min(v1, v2), max(v1, v2))
                    if edge not in pruned_edge_map:
                        pruned_edge_map[edge] = []
                    pruned_edge_map[edge].append((face_idx, poly_idx, poly_type))
            b_pruned, m_pruned, inv_pruned = compute_edge_stats(pruned_edge_map)
            
            print(f"[POLY FORM]   Iteration {pruning_iteration} edge distribution:")
            print(f"[POLY FORM]     Base Set (original remaining polygons):")
            print(f"[POLY FORM]       - Boundary edges (1 face): {b_base}")
            print(f"[POLY FORM]       - Manifold edges (2 faces): {m_base}")
            print(f"[POLY FORM]       - Invalid edges (3+ faces): {inv_base}")
            print(f"[POLY FORM]     Extracted Set (invalid edge polygons remaining):")
            print(f"[POLY FORM]       - Boundary edges (1 face): {b_extracted}")
            print(f"[POLY FORM]       - Manifold edges (2 faces): {m_extracted}")
            print(f"[POLY FORM]       - Invalid edges (3+ faces): {inv_extracted}")
            print(f"[POLY FORM]     Pruned Set ({len(pruned_polygons)} polygons removed):")
            print(f"[POLY FORM]       - Boundary edges (1 face): {b_pruned}")
            print(f"[POLY FORM]       - Manifold edges (2 faces): {m_pruned}")
            print(f"[POLY FORM]       - Invalid edges (3+ faces): {inv_pruned}")
            
            # Check if we should continue iterating
            if inv_extracted == 0:
                print(f"\n[POLY FORM]   No invalid edges remaining - stopping pruning")
                break
        
        # End of pruning iteration loop
        print(f"\n[POLY FORM]   Pruning complete after {pruning_iteration} iteration(s)")
        
        # Print final results after pruning
        print(f"\n[POLY FORM]   ========== PRUNING RESULTS ==========")
        print(f"[POLY FORM]   Total polygons in extracted set: {len(invalid_edge_polygons)}")
        print(f"[POLY FORM]   Total polygons in pruned set: {len(pruned_polygons)}")
        
        # Compute final edge statistics
        final_extracted_edge_map = {}
        for poly_info in invalid_edge_polygons:
            poly_verts = poly_info['data'].get('vertices', [])
            face_idx = poly_info['face_idx']
            poly_idx = poly_info['polygon_idx']
            poly_type = poly_info['poly_type']
            for i in range(len(poly_verts)):
                v1 = poly_verts[i] - 1
                v2 = poly_verts[(i + 1) % len(poly_verts)] - 1
                edge = (min(v1, v2), max(v1, v2))
                if edge not in final_extracted_edge_map:
                    final_extracted_edge_map[edge] = []
                final_extracted_edge_map[edge].append((face_idx, poly_idx, poly_type))
        final_b_extracted, final_m_extracted, final_inv_extracted = compute_edge_stats(final_extracted_edge_map)
        
        print(f"[POLY FORM]   Final extracted set edge statistics:")
        print(f"[POLY FORM]     - Boundary edges (1 face): {final_b_extracted}")
        print(f"[POLY FORM]     - Manifold edges (2 faces): {final_m_extracted}")
        print(f"[POLY FORM]     - Invalid edges (3+ faces): {final_inv_extracted}")
        print(f"[POLY FORM]   ======================================")
        
        # Print polygons for combination testing as before
        print("\n[POLY FORM]     --- Polygons after pruning ---")
        for idx, poly_info in enumerate(invalid_edge_polygons):
            poly_verts = poly_info['data'].get('vertices', [])
            poly_type = poly_info['poly_type']
            face_idx = poly_info['face_idx']
            print(f"[POLY FORM]       [{chr(65+idx)}] Face {face_idx+1}, Type: {poly_type}, Vertices: {poly_verts}")

        # Print all edges present in the pruned set
        pruned_edges = set()
        for poly_info in invalid_edge_polygons:
            poly_verts = poly_info['data'].get('vertices', [])
            for i in range(len(poly_verts)):
                v1 = poly_verts[i]
                v2 = poly_verts[(i + 1) % len(poly_verts)]
                edge = (min(v1, v2), max(v1, v2))
                pruned_edges.add(edge)
        print(f"[POLY FORM]     --- Edges after pruning ({len(pruned_edges)} total) ---")
        # Don't print individual edges to reduce output clutter
        # for edge in sorted(pruned_edges):
        #     print(f"[POLY FORM]       Edge {edge}")
    
    # ==========================================================================
    # Step 6: Combination Strategy for Remaining Polygons
    # ==========================================================================
    if len(invalid_edge_polygons) > 0:
        print("\n" + "="*70)
        print("[POLY FORM] Step 6: COMBINATION STRATEGY")
        print("="*70)
        print(f"[POLY FORM] Goal: Add polygons back to base set to reduce boundary edges to 0")
        print(f"[POLY FORM] Constraint: Do not increase invalid edges")
        print(f"[POLY FORM] Starting with {len(invalid_edge_polygons)} polygon(s) after pruning")
        
        # Helper function to compute edge statistics
        def compute_edge_statistics(polygon_list, all_polygons_list):
            """Compute edge statistics for a given polygon list"""
            edge_map = {}
            for poly_info in polygon_list:
                poly_verts = poly_info['data'].get('vertices', [])
                face_idx = poly_info['face_idx']
                poly_idx = poly_info['polygon_idx']
                poly_type = poly_info['poly_type']
                for i in range(len(poly_verts)):
                    v1 = poly_verts[i] - 1
                    v2 = poly_verts[(i + 1) % len(poly_verts)] - 1
                    edge = (min(v1, v2), max(v1, v2))
                    if edge not in edge_map:
                        edge_map[edge] = []
                    edge_map[edge].append((face_idx, poly_idx, poly_type))
            
            boundary_edges = sum(1 for faces in edge_map.values() if len(faces) == 1)
            manifold_edges = sum(1 for faces in edge_map.values() if len(faces) == 2)
            invalid_edges = sum(1 for faces in edge_map.values() if len(faces) > 2)
            
            return boundary_edges, manifold_edges, invalid_edges
        
        # Build base set (all polygons not in extraction list)
        extracted_set = set((p['face_idx'], p['polygon_idx']) for p in invalid_edge_polygons)
        pruned_set = set((p['face_idx'], p['polygon_idx']) for p in pruned_polygons)
        base_polygons = [p for p in all_polygons_list 
                        if (p['face_idx'], p['polygon_idx']) not in extracted_set 
                        and (p['face_idx'], p['polygon_idx']) not in pruned_set]
        
        b_base, m_base, inv_base = compute_edge_statistics(base_polygons, all_polygons_list)
        print(f"\n[POLY FORM] Base set statistics:")
        print(f"[POLY FORM]   Boundary: {b_base}, Manifold: {m_base}, Invalid: {inv_base}")
        
        # Group extraction polygons by face
        extraction_by_face = {}
        for poly_info in invalid_edge_polygons:
            face_idx = poly_info['face_idx']
            if face_idx not in extraction_by_face:
                extraction_by_face[face_idx] = []
            extraction_by_face[face_idx].append(poly_info)
        
        print(f"\n[POLY FORM] Extraction set grouped by face:")
        for face_idx in sorted(extraction_by_face.keys()):
            polys = extraction_by_face[face_idx]
            print(f"[POLY FORM]   Face {face_idx+1}: {len(polys)} polygon(s)")
            for poly_info in polys:
                poly_type = poly_info['poly_type']
                poly_verts = poly_info['data'].get('vertices', [])
                print(f"[POLY FORM]     - {poly_type}: {poly_verts}")
        
        # Iteration 1: Try combinations
        print(f"\n[POLY FORM] === Iteration 1: Testing Combinations ===")
        
        from itertools import combinations, product
        
        best_combination = []
        best_boundary = b_base
        best_manifold = m_base
        best_invalid = inv_base
        
        # Strategy: Start with all faces, then reduce to subsets, then individual faces
        # Test combinations from all faces down to single faces
        
        max_tests = 1000
        tests_done = 0
        found_perfect = False
        
        face_indices = sorted(extraction_by_face.keys())
        num_faces = len(face_indices)
        
        # 1. Test combinations from ALL faces (one polygon from each)
        if not found_perfect and tests_done < max_tests and num_faces > 1:
            print(f"\n[POLY FORM] Testing combinations from all {num_faces} faces (one polygon from each)...")
            
            # Limit to avoid combinatorial explosion
            if num_faces <= 5:
                for combo in product(*[extraction_by_face[f] for f in face_indices]):
                    if tests_done >= max_tests or found_perfect:
                        break
                    
                    test_polygons = base_polygons + list(combo)
                    b, m, inv = compute_edge_statistics(test_polygons, all_polygons_list)
                    tests_done += 1
                    
                    if inv <= best_invalid and b < best_boundary:
                        best_combination = list(combo)
                        best_boundary = b
                        best_manifold = m
                        best_invalid = inv
                        face_list = ', '.join([str(f+1) for f in face_indices])
                        print(f"[POLY FORM]   [Test {tests_done}] Faces [{face_list}] - {len(combo)} polygons")
                        print(f"[POLY FORM]     → B:{b}, M:{m}, Inv:{inv} ✓ BETTER")
                        
                        if inv == 0 and b == 0:
                            found_perfect = True
                            print(f"[POLY FORM]     → PERFECT SOLUTION FOUND!")
                            break
        
        # 2. Test combinations from subsets of faces (all combinations of N-1, N-2, ... 2 faces)
        for num_faces_in_combo in range(num_faces - 1, 1, -1):
            if found_perfect or tests_done >= max_tests:
                break
            
            if num_faces_in_combo >= 2:
                print(f"\n[POLY FORM] Testing combinations from {num_faces_in_combo} faces...")
                
                # Get all combinations of faces
                for face_combo in combinations(face_indices, num_faces_in_combo):
                    if tests_done >= max_tests or found_perfect:
                        break
                    
                    # Limit to avoid explosion
                    if num_faces_in_combo <= 5:
                        # Test all product combinations from selected faces
                        for poly_combo in product(*[extraction_by_face[f] for f in face_combo]):
                            if tests_done >= max_tests or found_perfect:
                                break
                            
                            test_polygons = base_polygons + list(poly_combo)
                            b, m, inv = compute_edge_statistics(test_polygons, all_polygons_list)
                            tests_done += 1
                            
                            if inv <= best_invalid and b < best_boundary:
                                best_combination = list(poly_combo)
                                best_boundary = b
                                best_manifold = m
                                best_invalid = inv
                                face_list = ', '.join([str(f+1) for f in face_combo])
                                print(f"[POLY FORM]   [Test {tests_done}] Faces [{face_list}] - {len(poly_combo)} polygons")
                                print(f"[POLY FORM]     → B:{b}, M:{m}, Inv:{inv} ✓ BETTER")
                                
                                if inv == 0 and b == 0:
                                    found_perfect = True
                                    print(f"[POLY FORM]     → PERFECT SOLUTION FOUND!")
                                    break
        
        # 3. Test individual faces (all polygons, then subsets)
        if not found_perfect and tests_done < max_tests:
            print(f"\n[POLY FORM] Testing polygons from individual faces...")
            
            for face_idx in face_indices:
                if tests_done >= max_tests or found_perfect:
                    break
                
                polys = extraction_by_face[face_idx]
                
                # Test all polygons from this face
                test_polygons = base_polygons + polys
                b, m, inv = compute_edge_statistics(test_polygons, all_polygons_list)
                tests_done += 1
                
                if inv <= best_invalid and b < best_boundary:
                    best_combination = polys
                    best_boundary = b
                    best_manifold = m
                    best_invalid = inv
                    print(f"[POLY FORM]   [Test {tests_done}] Face {face_idx+1} - all {len(polys)} polygons")
                    print(f"[POLY FORM]     → B:{b}, M:{m}, Inv:{inv} ✓ BETTER")
                    
                    if inv == 0 and b == 0:
                        found_perfect = True
                        print(f"[POLY FORM]     → PERFECT SOLUTION FOUND!")
                        break
                
                # Test pairs from this face
                if len(polys) >= 2:
                    for pair in combinations(polys, 2):
                        if tests_done >= max_tests or found_perfect:
                            break
                        
                        test_polygons = base_polygons + list(pair)
                        b, m, inv = compute_edge_statistics(test_polygons, all_polygons_list)
                        tests_done += 1
                        
                        if inv <= best_invalid and b < best_boundary:
                            best_combination = list(pair)
                            best_boundary = b
                            best_manifold = m
                            best_invalid = inv
                            print(f"[POLY FORM]   [Test {tests_done}] Face {face_idx+1} - pair")
                            print(f"[POLY FORM]     → B:{b}, M:{m}, Inv:{inv} ✓ BETTER")
                            
                            if inv == 0 and b == 0:
                                found_perfect = True
                                print(f"[POLY FORM]     → PERFECT SOLUTION FOUND!")
                                break
                
                # Test single polygons from this face
                for poly_info in polys:
                    if tests_done >= max_tests or found_perfect:
                        break
                    
                    test_polygons = base_polygons + [poly_info]
                    b, m, inv = compute_edge_statistics(test_polygons, all_polygons_list)
                    tests_done += 1
                    
                    poly_type = poly_info['poly_type']
                    poly_verts = poly_info['data'].get('vertices', [])
                    
                    if inv <= best_invalid and b < best_boundary:
                        best_combination = [poly_info]
                        best_boundary = b
                        best_manifold = m
                        best_invalid = inv
                        print(f"[POLY FORM]   [Test {tests_done}] Face {face_idx+1} - {poly_type} {poly_verts}")
                        print(f"[POLY FORM]     → B:{b}, M:{m}, Inv:{inv} ✓ BETTER")
                        
                        if inv == 0 and b == 0:
                            found_perfect = True
                            print(f"[POLY FORM]     → PERFECT SOLUTION FOUND!")
                            break
        
        print(f"\n[POLY FORM] Iteration 1 complete: {tests_done} combinations tested")
        print(f"[POLY FORM] Best result: B:{best_boundary}, M:{best_manifold}, Inv:{best_invalid}")
        
        # Apply best combination
        if len(best_combination) > 0:
            print(f"\n[POLY FORM] Adding {len(best_combination)} polygon(s) back to base set:")
            for poly_info in best_combination:
                face_idx = poly_info['face_idx']
                poly_type = poly_info['poly_type']
                poly_verts = poly_info['data'].get('vertices', [])
                print(f"[POLY FORM]   Face {face_idx+1} {poly_type}: {poly_verts}")
                
                # Remove from extraction list
                if poly_info in invalid_edge_polygons:
                    invalid_edge_polygons.remove(poly_info)
            
            print(f"[POLY FORM] Remaining in extraction list: {len(invalid_edge_polygons)} polygon(s)")
        else:
            print(f"[POLY FORM] No improvement found, keeping all in extraction list")
        
        # Continue iterating until boundary edges reach 0 or no improvement possible
        iteration_num = 2
        max_iterations = 20  # Safety limit
        previous_boundary = best_boundary
        
        while len(invalid_edge_polygons) > 0 and iteration_num <= max_iterations:
            print(f"\n[POLY FORM] === Iteration {iteration_num}: Processing Remaining Polygons ===")
            
            # Rebuild base with previous iteration results
            extracted_set = set((p['face_idx'], p['polygon_idx']) for p in invalid_edge_polygons)
            base_polygons_iter = [p for p in all_polygons_list 
                                  if (p['face_idx'], p['polygon_idx']) not in extracted_set 
                                  and (p['face_idx'], p['polygon_idx']) not in pruned_set]
            
            b_base_iter, m_base_iter, inv_base_iter = compute_edge_statistics(base_polygons_iter, all_polygons_list)
            print(f"[POLY FORM] Base set after iteration {iteration_num-1}: B:{b_base_iter}, M:{m_base_iter}, Inv:{inv_base_iter}")
            
            # Check if we've reached the goal
            if b_base_iter == 0:
                print(f"[POLY FORM] ✓ GOAL ACHIEVED: Boundary edges = 0!")
                print(f"[POLY FORM] Deleting {len(invalid_edge_polygons)} remaining polygon(s)")
                
                for poly_info in invalid_edge_polygons:
                    face_idx = poly_info['face_idx']
                    poly_idx = poly_info['polygon_idx']
                    if face_idx < len(unique_faces):
                        if poly_idx < len(unique_faces[face_idx]['polygons']):
                            unique_faces[face_idx]['polygons'][poly_idx]['removed'] = True
                            unique_faces[face_idx]['polygons'][poly_idx]['removal_reason'] = 'combination_strategy'
                
                invalid_edge_polygons.clear()
                break
            
            # Regroup remaining polygons
            extraction_by_face_iter = {}
            for poly_info in invalid_edge_polygons:
                face_idx = poly_info['face_idx']
                if face_idx not in extraction_by_face_iter:
                    extraction_by_face_iter[face_idx] = []
                extraction_by_face_iter[face_idx].append(poly_info)
            
            best_combination_iter = []
            best_boundary_iter = b_base_iter
            best_manifold_iter = m_base_iter
            best_invalid_iter = inv_base_iter
            
            tests_done_iter = 0
            found_perfect_iter = False
            
            face_indices_iter = sorted(extraction_by_face_iter.keys())
            num_faces_iter = len(face_indices_iter)
            
            # 1. Test combinations from ALL faces (one polygon from each)
            if not found_perfect_iter and tests_done_iter < max_tests and num_faces_iter > 1:
                # Limit to avoid combinatorial explosion
                if num_faces_iter <= 5:
                    for combo in product(*[extraction_by_face_iter[f] for f in face_indices_iter]):
                        if tests_done_iter >= max_tests or found_perfect_iter:
                            break
                        
                        test_polygons = base_polygons_iter + list(combo)
                        b, m, inv = compute_edge_statistics(test_polygons, all_polygons_list)
                        tests_done_iter += 1
                        
                        if inv <= best_invalid_iter and b < best_boundary_iter:
                            best_combination_iter = list(combo)
                            best_boundary_iter = b
                            best_manifold_iter = m
                            best_invalid_iter = inv
                            
                            if inv == 0 and b == 0:
                                found_perfect_iter = True
                                break
            
            # 2. Test combinations from subsets of faces
            for num_faces_in_combo in range(num_faces_iter - 1, 1, -1):
                if found_perfect_iter or tests_done_iter >= max_tests:
                    break
                
                if num_faces_in_combo >= 2:
                    for face_combo in combinations(face_indices_iter, num_faces_in_combo):
                        if tests_done_iter >= max_tests or found_perfect_iter:
                            break
                        
                        # Limit to avoid explosion
                        if num_faces_in_combo <= 5:
                            for poly_combo in product(*[extraction_by_face_iter[f] for f in face_combo]):
                                if tests_done_iter >= max_tests or found_perfect_iter:
                                    break
                                
                                test_polygons = base_polygons_iter + list(poly_combo)
                                b, m, inv = compute_edge_statistics(test_polygons, all_polygons_list)
                                tests_done_iter += 1
                                
                                if inv <= best_invalid_iter and b < best_boundary_iter:
                                    best_combination_iter = list(poly_combo)
                                    best_boundary_iter = b
                                    best_manifold_iter = m
                                    best_invalid_iter = inv
                                    
                                    if inv == 0 and b == 0:
                                        found_perfect_iter = True
                                        break
            
            # 3. Test individual faces (all polygons, then subsets, then singles)
            if not found_perfect_iter and tests_done_iter < max_tests:
                for face_idx in face_indices_iter:
                    if tests_done_iter >= max_tests or found_perfect_iter:
                        break
                    
                    polys = extraction_by_face_iter[face_idx]
                    
                    # Test all polygons from this face
                    test_polygons = base_polygons_iter + polys
                    b, m, inv = compute_edge_statistics(test_polygons, all_polygons_list)
                    tests_done_iter += 1
                    
                    if inv <= best_invalid_iter and b < best_boundary_iter:
                        best_combination_iter = polys
                        best_boundary_iter = b
                        best_manifold_iter = m
                        best_invalid_iter = inv
                        
                        if inv == 0 and b == 0:
                            found_perfect_iter = True
                            break
                    
                    # Test pairs from this face
                    if len(polys) >= 2:
                        for pair in combinations(polys, 2):
                            if tests_done_iter >= max_tests or found_perfect_iter:
                                break
                            
                            test_polygons = base_polygons_iter + list(pair)
                            b, m, inv = compute_edge_statistics(test_polygons, all_polygons_list)
                            tests_done_iter += 1
                            
                            if inv <= best_invalid_iter and b < best_boundary_iter:
                                best_combination_iter = list(pair)
                                best_boundary_iter = b
                                best_manifold_iter = m
                                best_invalid_iter = inv
                                
                                if inv == 0 and b == 0:
                                    found_perfect_iter = True
                                    break
                    
                    # Test single polygons from this face
                    for poly_info in polys:
                        if tests_done_iter >= max_tests or found_perfect_iter:
                            break
                        
                        test_polygons = base_polygons_iter + [poly_info]
                        b, m, inv = compute_edge_statistics(test_polygons, all_polygons_list)
                        tests_done_iter += 1
                        
                        if inv <= best_invalid_iter and b < best_boundary_iter:
                            best_combination_iter = [poly_info]
                            best_boundary_iter = b
                            best_manifold_iter = m
                            best_invalid_iter = inv
                            
                            if inv == 0 and b == 0:
                                found_perfect_iter = True
                                break
            
            print(f"[POLY FORM] Iteration {iteration_num} complete: {tests_done_iter} combinations tested")
            print(f"[POLY FORM] Best result: B:{best_boundary_iter}, M:{best_manifold_iter}, Inv:{best_invalid_iter}")
            
            # Check for improvement
            if best_boundary_iter >= previous_boundary:
                print(f"[POLY FORM] No improvement in boundary edges, stopping iterations")
                
                # Delete remaining polygons
                if len(invalid_edge_polygons) > 0:
                    print(f"\n[POLY FORM] Deleting {len(invalid_edge_polygons)} remaining polygon(s):")
                    for poly_info in invalid_edge_polygons:
                        face_idx = poly_info['face_idx']
                        poly_idx = poly_info['polygon_idx']
                        poly_type = poly_info['poly_type']
                        poly_verts = poly_info['data'].get('vertices', [])
                        print(f"[POLY FORM]   Face {face_idx+1} {poly_type}: {poly_verts}")
                        
                        if face_idx < len(unique_faces):
                            if poly_idx < len(unique_faces[face_idx]['polygons']):
                                unique_faces[face_idx]['polygons'][poly_idx]['removed'] = True
                                unique_faces[face_idx]['polygons'][poly_idx]['removal_reason'] = 'combination_strategy'
                    
                    invalid_edge_polygons.clear()
                break
            
            # Apply iteration results
            if len(best_combination_iter) > 0:
                print(f"\n[POLY FORM] Adding {len(best_combination_iter)} polygon(s) back to base set:")
                for poly_info in best_combination_iter:
                    face_idx = poly_info['face_idx']
                    poly_type = poly_info['poly_type']
                    poly_verts = poly_info['data'].get('vertices', [])
                    print(f"[POLY FORM]   Face {face_idx+1} {poly_type}: {poly_verts}")
                    
                    if poly_info in invalid_edge_polygons:
                        invalid_edge_polygons.remove(poly_info)
                
                previous_boundary = best_boundary_iter
                
                # Check if we achieved zero boundary edges
                if best_boundary_iter == 0:
                    print(f"[POLY FORM] ✓ GOAL ACHIEVED: Boundary edges = 0!")
                    
                    if len(invalid_edge_polygons) > 0:
                        print(f"[POLY FORM] Deleting {len(invalid_edge_polygons)} remaining polygon(s)")
                        for poly_info in invalid_edge_polygons:
                            face_idx = poly_info['face_idx']
                            poly_idx = poly_info['polygon_idx']
                            if face_idx < len(unique_faces):
                                if poly_idx < len(unique_faces[face_idx]['polygons']):
                                    unique_faces[face_idx]['polygons'][poly_idx]['removed'] = True
                                    unique_faces[face_idx]['polygons'][poly_idx]['removal_reason'] = 'combination_strategy'
                        invalid_edge_polygons.clear()
                    break
            else:
                print(f"[POLY FORM] No improvement found in iteration {iteration_num}")
                
                # Delete remaining polygons
                if len(invalid_edge_polygons) > 0:
                    print(f"\n[POLY FORM] Deleting {len(invalid_edge_polygons)} remaining polygon(s):")
                    for poly_info in invalid_edge_polygons:
                        face_idx = poly_info['face_idx']
                        poly_idx = poly_info['polygon_idx']
                        poly_type = poly_info['poly_type']
                        poly_verts = poly_info['data'].get('vertices', [])
                        print(f"[POLY FORM]   Face {face_idx+1} {poly_type}: {poly_verts}")
                        
                        if face_idx < len(unique_faces):
                            if poly_idx < len(unique_faces[face_idx]['polygons']):
                                unique_faces[face_idx]['polygons'][poly_idx]['removed'] = True
                                unique_faces[face_idx]['polygons'][poly_idx]['removal_reason'] = 'combination_strategy'
                    
                    invalid_edge_polygons.clear()
                break
            
            iteration_num += 1
        
        if iteration_num > max_iterations:
            print(f"[POLY FORM] Reached maximum iteration limit ({max_iterations})")
            
            # Delete any remaining polygons
            if len(invalid_edge_polygons) > 0:
                print(f"\n[POLY FORM] Deleting {len(invalid_edge_polygons)} remaining polygon(s):")
                for poly_info in invalid_edge_polygons:
                    face_idx = poly_info['face_idx']
                    poly_idx = poly_info['polygon_idx']
                    if face_idx < len(unique_faces):
                        if poly_idx < len(unique_faces[face_idx]['polygons']):
                            unique_faces[face_idx]['polygons'][poly_idx]['removed'] = True
                            unique_faces[face_idx]['polygons'][poly_idx]['removal_reason'] = 'combination_strategy'
                
                invalid_edge_polygons.clear()
        
        print(f"\n[POLY FORM] Combination strategy complete")
        print("="*70)
    
    # ==========================================================================
    # Step 6.5: Remove duplicate polygons in each face
    # ==========================================================================
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
    
    # ==========================================================================
    # Step 6.6: Fix faces without boundaries (promote ALT to BOUNDARY)
    # ==========================================================================
    print("\n[POLY FORM] Step 6.6: Checking faces for missing boundaries")
    print("-" * 70)
    
    from shapely.geometry import Polygon as ShapelyPolygon
    from shapely.geometry import Point as ShapelyPoint
    
    faces_fixed = 0
    for face_idx, face_eq in enumerate(unique_faces):
        if 'polygons' not in face_eq or len(face_eq['polygons']) == 0:
            continue
        
        # Check if all polygons in this face are removed
        all_removed = all(poly.get('removed', False) for poly in face_eq['polygons'])
        if all_removed:
            continue  # Skip this face entirely
        
        # Separate current polygons by type
        boundaries = []
        alts = []
        holes = []
        
        for poly in face_eq['polygons']:
            if poly.get('removed', False):
                continue
            
            poly_type = poly.get('polygon_type', 'UNKNOWN')
            if poly_type == 'BOUNDARY':
                boundaries.append(poly)
            elif poly_type == 'ALT':
                alts.append(poly)
            elif poly_type == 'HOLE':
                holes.append(poly)
        
        # Check if face has no boundary
        if len(boundaries) == 0 and len(alts) > 0:
            print(f"[POLY FORM]   Face {face_idx+1}: No BOUNDARY found, has {len(alts)} ALT polygon(s)")
            
            # Promote the first (typically largest) ALT to BOUNDARY
            promoted_alt = alts[0]
            promoted_alt['polygon_type'] = 'BOUNDARY'
            boundaries.append(promoted_alt)
            alts.pop(0)
            
            promoted_verts = promoted_alt['vertices']
            print(f"[POLY FORM]     → Promoted ALT {promoted_verts} to BOUNDARY")
            
            # Check if holes lie within the new boundary
            if len(holes) > 0:
                # Get face normal for 2D projection
                normal = face_eq.get('normal', [0, 0, 1])
                normal = np.array(normal)
                normal = normal / np.linalg.norm(normal)
                
                # Project boundary to 2D
                promoted_verts_3d = np.array([[selected_vertices[v-1] for v in promoted_verts]])
                
                # Create projection basis
                if abs(normal[2]) > 0.9:
                    u = np.array([1.0, 0.0, 0.0])
                else:
                    u = np.array([0.0, 0.0, 1.0])
                u = u - np.dot(u, normal) * normal
                u = u / np.linalg.norm(u)
                v = np.cross(normal, u)
                
                # Project boundary vertices
                boundary_2d = []
                for vert_idx in promoted_verts:
                    vert_3d = selected_vertices[vert_idx - 1]
                    proj_u = np.dot(vert_3d, u)
                    proj_v = np.dot(vert_3d, v)
                    boundary_2d.append((proj_u, proj_v))
                
                try:
                    boundary_poly = ShapelyPolygon(boundary_2d)
                    
                    # Check each hole
                    holes_to_remove = []
                    for hole_idx, hole in enumerate(holes):
                        hole_verts = hole['vertices']
                        
                        # Project hole vertices
                        hole_2d = []
                        for vert_idx in hole_verts:
                            vert_3d = selected_vertices[vert_idx - 1]
                            proj_u = np.dot(vert_3d, u)
                            proj_v = np.dot(vert_3d, v)
                            hole_2d.append((proj_u, proj_v))
                        
                        hole_poly = ShapelyPolygon(hole_2d)
                        
                        # Check if hole is within boundary
                        if not boundary_poly.contains(hole_poly):
                            print(f"[POLY FORM]     → HOLE {hole_verts} is NOT within new boundary, removing")
                            holes_to_remove.append(hole_idx)
                            hole['removed'] = True
                            hole['removal_reason'] = 'outside_promoted_boundary'
                        else:
                            print(f"[POLY FORM]     → HOLE {hole_verts} is within new boundary, keeping")
                    
                    # Remove invalid holes from the list
                    for idx in reversed(holes_to_remove):
                        holes.pop(idx)
                    
                except Exception as e:
                    print(f"[POLY FORM]     → ERROR checking holes: {e}, removing all holes")
                    for hole in holes:
                        hole['removed'] = True
                        hole['removal_reason'] = 'boundary_promotion_error'
                    holes.clear()
            
            faces_fixed += 1
        
        elif len(boundaries) == 0 and len(alts) == 0:
            if len(holes) > 0:
                print(f"[POLY FORM]   Face {face_idx+1}: No BOUNDARY or ALT, only {len(holes)} HOLE(s), removing holes")
                for hole in holes:
                    hole['removed'] = True
                    hole['removal_reason'] = 'no_boundary_available'
    
    if faces_fixed > 0:
        print(f"[POLY FORM]   Fixed {faces_fixed} face(s) by promoting ALT to BOUNDARY")
    else:
        print(f"[POLY FORM]   All faces have valid boundaries")
    
    # ==========================================================================
    # Step 7: Compile face results
    # ==========================================================================
    print("\n[POLY FORM] Step 7: Compiling face results")
    print("-" * 70)
    print(f"[POLY FORM]   Processing {len(unique_faces)} validated face(s)...")
    
    # Build final faces list from unique_faces (skipping any without valid boundaries)
    faces_skipped = 0
    for face_idx, face_eq in enumerate(unique_faces):
        if 'polygons' not in face_eq or len(face_eq['polygons']) == 0:
            faces_skipped += 1
            continue
        
        # Check if all polygons in this face are removed
        all_removed = all(poly.get('removed', False) for poly in face_eq['polygons'])
        if all_removed:
            faces_skipped += 1
            print(f"[POLY FORM]   Face {face_idx+1}: Skipped (all polygons removed)")
            continue
        
        # Get validated polygons (skip those marked for removal)
        polygons = face_eq['polygons']
        
        # Separate boundaries, alternates, and holes
        boundaries = []
        alternates_list = []
        holes_list = []
        
        for poly_data in polygons:
            # Skip polygons marked for removal
            if poly_data.get('removed', False):
                continue
            
            poly_type = poly_data.get('polygon_type', 'UNKNOWN')
            if poly_type == 'BOUNDARY':
                boundaries.append(poly_data)
            elif poly_type == 'ALT':
                alternates_list.append({
                    'vertices': poly_data['vertices'],
                    'shapely_2d': poly_data.get('shapely_2d'),
                    'area': poly_data.get('area', 0)
                })
            elif poly_type == 'HOLE':
                holes_list.append(poly_data['vertices'])
        
        # Select primary boundary (if multiple exist, keep first and move rest to alternates)
        primary_boundary = None
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
            faces_skipped += 1
            print(f"[POLY FORM]   Face {face_idx+1}: Skipped (no boundary polygon)")
            continue
        
        # Create one face with the primary boundary
        face_data = {
            'original_face_idx': face_idx,  # Track original face index
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
    
    if faces_skipped > 0:
        print(f"\n[POLY FORM]   Skipped {faces_skipped} face(s) with no valid boundary polygon")
    
    # ================================================================
    # Merge touching alternate polygons before finalizing
    # ================================================================
    print("\n" + "="*70)
    print("[POLY FORM] FINAL ALTERNATE RESOLUTION")
    print("="*70)
    print("[POLY FORM] Checking for faces with multiple alternates...")
    
    faces_with_alternates = [idx for idx, f in enumerate(faces) if f.get('alternates')]
    print(f"[POLY FORM] Found {len(faces_with_alternates)} face(s) with alternates")
    
    new_faces_from_alternates = []  # Store non-touching alternates as new faces
    
    if len(faces_with_alternates) > 0:
        for face_list_idx in faces_with_alternates:
            face = faces[face_list_idx]
            alternates = face.get('alternates', [])
            original_face_idx = face.get('original_face_idx', face_list_idx)
            
            if len(alternates) == 0:
                continue
            
            print(f"\n[POLY FORM] Face {original_face_idx+1}: {len(alternates)} alternate(s)")
            print(f"[POLY FORM]   Primary boundary: {len(face['vertices'])} vertices")
            
            # Separate touching and non-touching alternates
            touching_alternates = []
            non_touching_alternates = []
            
            for alt in alternates:
                poly_to_check = alt['vertices']
                
                # Check if alternate shares any edge with primary boundary
                primary_edges = set()
                for k in range(len(face['vertices'])):
                    v1, v2 = face['vertices'][k], face['vertices'][(k+1) % len(face['vertices'])]
                    primary_edges.add((min(v1, v2), max(v1, v2)))
                
                alt_edges = set()
                for k in range(len(poly_to_check)):
                    v1, v2 = poly_to_check[k], poly_to_check[(k+1) % len(poly_to_check)]
                    alt_edges.add((min(v1, v2), max(v1, v2)))
                
                if primary_edges & alt_edges:  # Has shared edges
                    touching_alternates.append(alt)
                else:
                    non_touching_alternates.append(alt)
            
            # Move non-touching alternates to new faces
            if non_touching_alternates:
                print(f"[POLY FORM]   Found {len(non_touching_alternates)} non-touching alternate(s), moving to new faces...")
                for alt_idx, alt in enumerate(non_touching_alternates):
                    new_face = {
                        'original_face_idx': original_face_idx,  # Keep same original index for tracking
                        'normal': face['normal'],
                        'd': face['d'],
                        'vertices': alt['vertices'],
                        'holes': [],
                        'all_vertices_on_face': face.get('all_vertices_on_face', []),
                        'edges': face.get('edges', [])
                    }
                    new_faces_from_alternates.append(new_face)
                    print(f"[POLY FORM]     Moved alternate {alt_idx+1} ({len(alt['vertices'])} vertices) to new face")
            
            # Clear alternates list and only keep touching ones for merging
            face['alternates'] = touching_alternates
            
            if not touching_alternates:
                print(f"[POLY FORM]   No touching alternates to merge")
                continue
            
            print(f"[POLY FORM]   Merging {len(touching_alternates)} touching alternate(s) into primary boundary...")
            
            try:
                # Start with primary boundary
                merged_vertices = face['vertices'][:]
                
                # Merge each touching alternate one by one
                for alt_idx, alt in enumerate(touching_alternates):
                    poly_to_merge = alt['vertices']
                    print(f"[POLY FORM]     Merging alternate {alt_idx+1}: {len(poly_to_merge)} vertices")
                    
                    # Find shared edges between merged_vertices and poly_to_merge
                    merged_edges = {}
                    for k in range(len(merged_vertices)):
                        v1, v2 = merged_vertices[k], merged_vertices[(k+1) % len(merged_vertices)]
                        edge_norm = (min(v1, v2), max(v1, v2))
                        is_forward = (v1 < v2)
                        merged_edges[edge_norm] = (k, is_forward)
                    
                    poly_edges = {}
                    for k in range(len(poly_to_merge)):
                        v1, v2 = poly_to_merge[k], poly_to_merge[(k+1) % len(poly_to_merge)]
                        edge_norm = (min(v1, v2), max(v1, v2))
                        is_forward = (v1 < v2)
                        poly_edges[edge_norm] = (k, is_forward)
                    
                    # Find shared edges
                    shared_edges = set(merged_edges.keys()) & set(poly_edges.keys())
                    
                    if len(shared_edges) == 0:
                        print(f"[POLY FORM]       WARNING: No shared edges found with alternate {alt_idx+1}, skipping merge")
                        continue
                    
                    # Remove shared edges from both polygons
                    for edge_norm in shared_edges:
                        del merged_edges[edge_norm]
                        del poly_edges[edge_norm]
                    
                    # Build merged vertex sequence
                    # This is complex - need to walk around both polygons removing shared edges
                    # Use connectivity graph approach
                    vertex_edges = {}
                    for edge_norm, (idx, is_fwd) in merged_edges.items():
                        v1, v2 = edge_norm if is_fwd else (edge_norm[1], edge_norm[0])
                        if v1 not in vertex_edges:
                            vertex_edges[v1] = []
                        if v2 not in vertex_edges:
                            vertex_edges[v2] = []
                        vertex_edges[v1].append(v2)
                        vertex_edges[v2].append(v1)
                    
                    for edge_norm, (idx, is_fwd) in poly_edges.items():
                        v1, v2 = edge_norm if is_fwd else (edge_norm[1], edge_norm[0])
                        if v1 not in vertex_edges:
                            vertex_edges[v1] = []
                        if v2 not in vertex_edges:
                            vertex_edges[v2] = []
                        vertex_edges[v1].append(v2)
                        vertex_edges[v2].append(v1)
                    
                    # Start from any vertex and walk around
                    start_v = list(vertex_edges.keys())[0]
                    new_merged = [start_v]
                    current = start_v
                    visited_edges = set()
                    
                    while True:
                        neighbors = [n for n in vertex_edges[current] if (min(current, n), max(current, n)) not in visited_edges]
                        if not neighbors:
                            break
                        next_v = neighbors[0]
                        visited_edges.add((min(current, next_v), max(current, next_v)))
                        if next_v == start_v:
                            break
                        new_merged.append(next_v)
                        current = next_v
                    
                    merged_vertices = new_merged
                    print(f"[POLY FORM]       After merge: {len(merged_vertices)} vertices")
                
                # Update face with merged boundary
                face['vertices'] = merged_vertices
                
            except Exception as e:
                print(f"[POLY FORM]     ERROR merging alternates: {e}")
                print(f"[POLY FORM]     Keeping original boundary")
            
            # Clear alternates from face since they've been processed
            if 'alternates' in face:
                del face['alternates']
    
    # Add new faces created from non-touching alternates
    if new_faces_from_alternates:
        print(f"\n[POLY FORM] Adding {len(new_faces_from_alternates)} new face(s) from non-touching alternates")
        faces.extend(new_faces_from_alternates)
    
    # Remove duplicates caused by hole-in-hole extraction
    print("\n" + "="*70)
    print("[POLY FORM] CHECKING FOR DUPLICATE FACES")
    print("="*70)
    
    # Build set of face signatures (sorted vertex tuples)
    face_signatures = {}
    for idx, face in enumerate(faces):
        verts = tuple(sorted(face['vertices']))
        if verts not in face_signatures:
            face_signatures[verts] = []
        face_signatures[verts].append(idx)
    
    # Find duplicates
    duplicates = {verts: indices for verts, indices in face_signatures.items() if len(indices) > 1}
    
    if duplicates:
        print(f"[POLY FORM]   Found {len(duplicates)} group(s) of duplicate faces:")
        duplicates_to_delete = set()
        for verts, indices in duplicates.items():
            print(f"[POLY FORM]     Vertices {list(verts)}: {len(indices)} copies (face indices: {indices})")
            # Keep first, mark rest for deletion
            for idx in indices[1:]:
                duplicates_to_delete.add(idx)
        
        # Delete in reverse order to maintain indices
        duplicates_deleted = 0
        for face_idx in sorted(duplicates_to_delete, reverse=True):
            del faces[face_idx]
            duplicates_deleted += 1
        
        print(f"[POLY FORM]   Deleted {duplicates_deleted} duplicate face(s), {len(faces)} remain")
    else:
        print(f"[POLY FORM]   No duplicate faces found")
    
    return faces, two_edge_vertices
    print("[POLY FORM] Checking for faces with multiple alternates...")
    
    faces_with_alternates = [idx for idx, f in enumerate(faces) if f.get('alternates')]
    print(f"[POLY FORM] Found {len(faces_with_alternates)} face(s) with alternates")
    
    if len(faces_with_alternates) > 0:
        for face_idx in faces_with_alternates:
            face = faces[face_idx]
            alternates = face.get('alternates', [])
            
            if len(alternates) == 0:
                continue
            
            print(f"\n[POLY FORM] Face {face_idx+1}: {len(alternates)} alternate(s)")
            print(f"[POLY FORM]   Primary boundary: {len(face['vertices'])} vertices")
            
            # Merge all alternates with the primary boundary by removing shared edges
            # Since non-touching alternates were moved to separate faces earlier,
            # all remaining alternates must be touching
            print(f"[POLY FORM]   Merging {len(alternates)} touching alternate(s) into primary boundary...")
            
            try:
                # Start with primary boundary
                merged_vertices = face['vertices'][:]
                
                # Merge each alternate one by one
                for alt_idx, alt in enumerate(alternates):
                    poly_to_merge = alt['vertices']
                    print(f"[POLY FORM]     Merging alternate {alt_idx+1}: {len(poly_to_merge)} vertices")
                    
                    # Find shared edges between merged_vertices and poly_to_merge
                    merged_edges = {}
                    for k in range(len(merged_vertices)):
                        v1, v2 = merged_vertices[k], merged_vertices[(k+1) % len(merged_vertices)]
                        edge_norm = (min(v1, v2), max(v1, v2))
                        is_forward = (v1 < v2)
                        merged_edges[edge_norm] = (k, is_forward)
                    
                    poly_edges = {}
                    for k in range(len(poly_to_merge)):
                        v1, v2 = poly_to_merge[k], poly_to_merge[(k+1) % len(poly_to_merge)]
                        edge_norm = (min(v1, v2), max(v1, v2))
                        is_forward = (v1 < v2)
                        poly_edges[edge_norm] = (k, is_forward)
                    
                    # Find all shared edges
                    shared_edges = []
                    for edge_norm in merged_edges:
                        if edge_norm in poly_edges:
                            merge_idx, merge_fwd = merged_edges[edge_norm]
                            poly_idx, poly_fwd = poly_edges[edge_norm]
                            same_direction = (merge_fwd == poly_fwd)
                            shared_edges.append((edge_norm, merge_idx, poly_idx, same_direction))
                    
                    if len(shared_edges) == 0:
                        print(f"[POLY FORM]       WARNING: No shared edges - cannot merge")
                        continue
                    
                    print(f"[POLY FORM]       Found {len(shared_edges)} shared edge(s)")
                    
                    # Sort by position in merged polygon
                    shared_edges.sort(key=lambda x: x[1])
                    
                    # Use first/largest consecutive group
                    first_shared = shared_edges[0]
                    last_shared = shared_edges[-1]
                    
                    merge_start_idx = first_shared[1]
                    merge_end_idx = (last_shared[1] + 1) % len(merged_vertices)
                    
                    shared_start_vertex = merged_vertices[merge_start_idx]
                    shared_end_vertex = merged_vertices[merge_end_idx]
                    
                    # Find these vertices in poly_to_merge
                    if shared_start_vertex not in poly_to_merge or shared_end_vertex not in poly_to_merge:
                        print(f"[POLY FORM]       ERROR: Shared vertices not found")
                        continue
                    
                    idx2_v1 = poly_to_merge.index(shared_start_vertex)
                    idx2_v2 = poly_to_merge.index(shared_end_vertex)
                    
                    # Rotate poly_to_merge to start at shared_start_vertex
                    poly2_rotated = poly_to_merge[idx2_v1:] + poly_to_merge[:idx2_v1]
                    idx2_v2_rotated = poly2_rotated.index(shared_end_vertex)
                    
                    # Extract vertices between shared endpoints
                    forward_path = poly2_rotated[1:idx2_v2_rotated]
                    backward_path = poly2_rotated[idx2_v2_rotated+1:]
                    
                    num_shared_vertices = len(shared_edges) + 1
                    expected_shared_path_len = num_shared_vertices - 2
                    
                    if len(forward_path) == 0 and len(backward_path) == 0:
                        insert_vertices = []
                    elif len(backward_path) == 0:
                        insert_vertices = forward_path
                    elif len(forward_path) == 0:
                        insert_vertices = backward_path[::-1]
                    elif abs(len(forward_path) - expected_shared_path_len) <= abs(len(backward_path) - expected_shared_path_len):
                        insert_vertices = backward_path[::-1] if len(backward_path) > 0 else []
                    else:
                        insert_vertices = forward_path
                    
                    print(f"[POLY FORM]       Inserting {len(insert_vertices)} vertices")
                    
                    # Merge by removing shared edges and inserting new vertices
                    if merge_end_idx > merge_start_idx:
                        new_merged = (merged_vertices[:merge_start_idx + 1] + 
                                     insert_vertices + 
                                     merged_vertices[merge_end_idx:])
                    else:
                        new_merged = (merged_vertices[merge_end_idx:merge_start_idx + 1] + 
                                     insert_vertices)
                    
                    merged_vertices = new_merged
                    print(f"[POLY FORM]       Result: {len(merged_vertices)} vertices")
                
                # Update face with merged boundary
                print(f"[POLY FORM]   ✓ Merged boundary: {len(merged_vertices)} vertices (was {len(face['vertices'])})")
                face['vertices'] = merged_vertices
                face['alternates'] = []  # Clear alternates after merging
                
            except Exception as e:
                print(f"[POLY FORM]   ERROR merging alternates: {e}")
                import traceback
                traceback.print_exc()
                print(f"[POLY FORM]   Keeping alternates separate")
                continue
    
    print("\n" + "="*70)
    print(f"[POLY FORM] EXTRACTION COMPLETE: {len(faces)} faces found")
    print("="*70)
    
    # Debug: Check types in faces list
    for idx, f in enumerate(faces):
        if not isinstance(f, dict):
            print(f"[DEBUG] ERROR: faces[{idx}] is {type(f)}, not dict: {f}")
    
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
    extracted_face_collections = []
    
    if len(extracted_faces) > 0:
        colors_ext = plt.cm.viridis(np.linspace(0, 1, len(extracted_faces)))
        
        for idx, face in enumerate(extracted_faces):
            # Handle different face data structures
            if isinstance(face, dict):
                if 'vertices' in face:
                    vertices_idx = face['vertices']
                elif 'exterior' in face:
                    vertices_idx = face['exterior']
                else:
                    print(f"[WARNING] Face {idx} has unexpected structure: {face.keys()}")
                    continue
            else:
                print(f"[WARNING] Face {idx} is not a dictionary: {type(face)}")
                continue
            
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
    else:
        print("[WARNING] No extracted faces to plot")
    
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
                print("\n[DEBUG] Building merged connectivity matrix (counting views with edge > 0)...")
                N = step3_vertices.shape[0]
                # Convert to binary (1 if edge exists, 0 otherwise) before counting
                top_binary = (top_conn > 0).astype(int)
                front_binary = (front_conn > 0).astype(int)
                side_binary = (side_conn > 0).astype(int)
                merged_conn = top_binary + front_binary + side_binary
                
                # Check for perpendicular edges projecting to points
                print("[DEBUG] Checking for perpendicular edges with point projections...")
                
                # Debug specific edges mentioned by user
                debug_edges = [(3, 47), (1, 3), (40, 46), (27, 29), (16, 29)]
                print("[DEBUG] Checking specific edges:")
                for i, j in debug_edges:
                    if i < N and j < N:
                        v1, v2 = step3_vertices[i], step3_vertices[j]
                        dx = abs(v2[0] - v1[0])
                        dy = abs(v2[1] - v1[1])
                        dz = abs(v2[2] - v1[2])
                        print(f"  Edge V{i}-V{j}: conn={merged_conn[i, j]}, dx={dx:.6f}, dy={dy:.6f}, dz={dz:.6f}")
                        print(f"    top_binary={top_binary[i, j]}, front_binary={front_binary[i, j]}, side_binary={side_binary[i, j]}")
                
                elevated_count = 0
                for i in range(N):
                    for j in range(i+1, N):
                        if merged_conn[i, j] == 2:
                            v1, v2 = step3_vertices[i], step3_vertices[j]
                            dx = abs(v2[0] - v1[0])
                            dy = abs(v2[1] - v1[1])
                            dz = abs(v2[2] - v1[2])
                            
                            # Perpendicular to top view (vertical edge) - projects as point
                            if dx < 1e-6 and dy < 1e-6:
                                merged_conn[i, j] += 1
                                merged_conn[j, i] += 1
                                elevated_count += 1
                                print(f"  [ELEVATION] Edge V{i}-V{j}: conn=2→3 (perpendicular to top view, dx={dx:.2e}, dy={dy:.2e})")
                            # Perpendicular to front view (parallel to X) - projects as point
                            elif dy < 1e-6 and dz < 1e-6:
                                merged_conn[i, j] += 1
                                merged_conn[j, i] += 1
                                elevated_count += 1
                                print(f"  [ELEVATION] Edge V{i}-V{j}: conn=2→3 (perpendicular to front view, dy={dy:.2e}, dz={dz:.2e})")
                            # Perpendicular to side view (parallel to Y) - projects as point
                            elif dx < 1e-6 and dz < 1e-6:
                                merged_conn[i, j] += 1
                                merged_conn[j, i] += 1
                                elevated_count += 1
                                print(f"  [ELEVATION] Edge V{i}-V{j}: conn=2→3 (perpendicular to side view, dx={dx:.2e}, dz={dz:.2e})")
                
                print(f"[DEBUG] Elevated {elevated_count} edge(s) from conn=2 to conn=3")
                
                # Check debug edges again after elevation
                print("[DEBUG] Checking specific edges after elevation:")
                for i, j in debug_edges:
                    if i < N and j < N:
                        print(f"  Edge V{i}-V{j}: conn={merged_conn[i, j]}")
                
                print("[DEBUG] Merged connectivity matrix (count of views with edge):")
                print(merged_conn)

                # Visualize 3D solid with edges from merged_conn
                print("\n[DEBUG] Visualizing 3D solid with reconstructed edges from merged connectivity matrix...")
                edges = []
                for j in range(N):
                    for k in range(j+1, N):
                        if merged_conn[j, k] > 0:
                            edges.append((j, k))
                # old visualize_3d_solid(solid_shape=solid, selected_vertices=step3_vertices, edges=edges)
                visualize_3d_solid(face_polygons, selected_vertices=step3_vertices, edges=edges, seed=seed)
                
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


