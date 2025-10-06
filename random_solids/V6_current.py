
import os
import sys
print("Running file:", os.path.abspath(__file__))

# Import configuration system
from config_system import ConfigurationManager, create_default_config, load_config

from OCC.Core.gp import gp_Trsf  # noqa: F401
from OCC.Core.TopLoc import TopLoc_Location  # noqa: F401
from OCC.Core.TopExp import TopExp_Explorer
from OCC.Core.TopoDS import topods
from opencascade import get_face_normal_from_opencascade, extract_and_visualize_faces, extract_wire_vertices_in_sequence, OPENCASCADE_AVAILABLE
from edge_reconstruction import reconstruct_edges_from_views
from unified_summary import (create_unified_summary, print_summary_info,
                             save_summary_to_file, save_summary_to_numpy,
                             visualize_adjacency_matrix)
from OCC.Core.TopAbs import (
    TopAbs_SHELL, TopAbs_EDGE, TopAbs_VERTEX, TopAbs_WIRE
)  # noqa: F401, E501
from OCC.Core.BRep import BRep_Tool
from OCC.Core.BRepAdaptor import BRepAdaptor_Curve
from config_system import ConfigurationManager, create_default_config, load_config
from OCC.Core.IFSelect import IFSelect_RetDone
# V5_current.py
# Saved version of Polgon Boolean Ops from shapely.py as of July 28, 2025
# Includes corrected plotting order: array_C first (dashed light gray), array_B second (solid black)
import argparse

from shapely.geometry import Polygon, Point
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import traceback
from V6_cleaned import build_solid_with_polygons_test
from V6_Sept_25 import save_solid_as_step


# OpenCASCADE imports
OPENCASCADE_AVAILABLE = False
try:
    from OCC.Core.gp import gp_Vec
    from OCC.Core.BRepPrimAPI import BRepPrimAPI_MakeBox
    from OCC.Core.TopExp import TopExp_Explorer
    from OCC.Core.TopAbs import TopAbs_FACE
    from OCC.Core.TopoDS import topods
    OPENCASCADE_AVAILABLE = True
    # Try to import TopExp for vertex extraction
    try:
        from OCC.Core.TopExp import topexp, topexp_Vertices
        TOPEXP_AVAILABLE = True
    except Exception:
        TOPEXP_AVAILABLE = False
except Exception:
    OPENCASCADE_AVAILABLE = False

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


def extract_and_visualize_faces(solid_shape):
    """Extract faces and visualize them from an OpenCASCADE solid shape."""
    if not OPENCASCADE_AVAILABLE or solid_shape is None:
        print("✗ Cannot visualize - OpenCASCADE not available or shape is None")
        return []
    face_explorer = TopExp_Explorer(solid_shape, TopAbs_FACE)
    face_data = []
    face_count = 0
    while face_explorer.More():
        face_shape = face_explorer.Current()
        face_count += 1
        try:
            face = topods.Face(face_shape)
            wire_explorer = TopExp_Explorer(face, TopAbs_WIRE)
            if wire_explorer.More():
                wire = topods.Wire(wire_explorer.Current())
                vertices = extract_wire_vertices_in_sequence(wire, face_count)
                if vertices and len(vertices) >= 3:
                    face_data.append({
                        'face_id': face_count,
                        'vertices': vertices
                    })
        except Exception as e:
            print(f"    Face {face_count}: Error - {e}")
        face_explorer.Next()
    print(f"  → Successfully extracted {len(face_data)} faces for visualization")
    return face_data


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

        if allow_invalid:
            # Return the raw polygon, even if invalid
            if not polygon.is_valid:
                from shapely.validation import explain_validity
                reason = explain_validity(polygon)
                print(f"    → Polygon is invalid, but allow_invalid=True: storing as-is")
                print(f"      Reason: {reason}")
                if 'Self-intersection' in reason:
                    print(f"      [INVESTIGATE] Polygon vertices: {projected_vertices}")
                    print(f"      [INVESTIGATE] Polygon WKT: {polygon.wkt}")
            else:
                print(f"    → Valid polygon created with {len(polygon.exterior.coords)-1} vertices")
            return polygon

        # Default: only return valid polygons
        if polygon.is_valid and hasattr(polygon, 'area') and polygon.area > 1e-6:
            print(f"    → Valid polygon created with {len(polygon.exterior.coords)-1} vertices")
            return polygon

        # For invalid polygons, try to fix
        if not polygon.is_valid:
            from shapely.validation import explain_validity
            reason = explain_validity(polygon)
            print(f"    → Invalid polygon detected: {reason}")
            print(f"    → Original vertices: {original_vertex_count}, coords in polygon: {len(polygon.exterior.coords)-1}")

            try:
                fixed_polygon = polygon.buffer(0)
                if fixed_polygon.is_valid and hasattr(fixed_polygon, 'area') and fixed_polygon.area > 1e-6:
                    if hasattr(fixed_polygon, 'exterior'):
                        fixed_vertex_count = len(fixed_polygon.exterior.coords) - 1
                        print(f"    → Fixed with buffer(0): {original_vertex_count} → {fixed_vertex_count} vertices")
                    return fixed_polygon
            except Exception as e:
                print(f"    → Buffer(0) fix failed: {e}")

            try:
                hull_polygon = Polygon(projected_vertices).convex_hull
                if hull_polygon.is_valid and hasattr(hull_polygon, 'area') and hull_polygon.area > 1e-6:
                    if hasattr(hull_polygon, 'exterior'):
                        hull_vertex_count = len(hull_polygon.exterior.coords) - 1
                        print(f"    → Fixed with convex_hull: {original_vertex_count} → {hull_vertex_count} vertices")
                    return hull_polygon
            except Exception as e:
                print(f"    → Convex hull fix failed: {e}")

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
    plt.show()
    
    print(f"✓ Array visualization complete")
    print(f"  → Array B: {len(array_B)} visible faces")
    print(f"  → Array C: {len(array_C)} hidden faces + intersections")
    print(f"  → Combined: {len(array_B) + len(array_C)} total polygons")



def visualize_3d_solid(solid_shape, selected_vertices=None, edges=None):
    """Display the 3D solid using matplotlib 3D plotting. Optionally highlight selected vertices and edges."""
    if not OPENCASCADE_AVAILABLE or solid_shape is None:
        print("✗ Cannot visualize - OpenCASCADE not available or shape is None")
        return
    print("\n" + "="*60)
    print("3D SOLID VISUALIZATION WITH MATPLOTLIB")
    print("="*60)
    try:
        from mpl_toolkits.mplot3d import Axes3D
        print("  → Extracting face vertices for 3D plot...")
        face_explorer = TopExp_Explorer(solid_shape, TopAbs_FACE)
        face_count = 0
        all_face_data = []
        while face_explorer.More():
            face_shape = face_explorer.Current()
            face_count += 1
            try:
                face = topods.Face(face_shape)
                wire_explorer = TopExp_Explorer(face, TopAbs_WIRE)
                if wire_explorer.More():
                    wire = wire_explorer.Current()
                    vertices = extract_wire_vertices_in_sequence(wire, 1)
                    if vertices and len(vertices) >= 3:
                        all_face_data.append({
                            'face_id': face_count,
                            'vertices': vertices
                        })
            except Exception as e:
                print(f"    Face {face_count}: Error - {e}")
            face_explorer.Next()
        print(f"  → Successfully extracted {len(all_face_data)} faces for visualization")
        if not all_face_data:
            print("  ✗ No face data available for visualization")
            return
        fig = plt.figure(figsize=(16, 12))
        ax = fig.add_subplot(111, projection='3d')
        colors = plt.cm.Set3(np.linspace(0, 1, len(all_face_data)))
        for i, face_data in enumerate(all_face_data):
            vertices = np.array(face_data['vertices'])
            if len(vertices) > 2:
                vertices_closed = np.vstack([vertices, vertices[0]])
                ax.plot(vertices_closed[:, 0], vertices_closed[:, 1], vertices_closed[:, 2], 
                        color=colors[i], linewidth=2.5, alpha=0.8)
                ax.scatter(vertices[:, 0], vertices[:, 1], vertices[:, 2], 
                          color=colors[i], s=30, alpha=0.6, edgecolors='black', linewidth=0.5)
                face_center = np.mean(vertices, axis=0)
                label_text = f'F{i+1}'
                ax.text(face_center[0], face_center[1], face_center[2], 
                       label_text, fontsize=9, color='black', ha='center', va='center', alpha=0.7)
        # Plot selected vertices if provided
        if selected_vertices is not None and len(selected_vertices) > 0:
            selected_vertices = np.array(selected_vertices)
            print(f"  → Plotting {len(selected_vertices)} selected vertices as large red spheres: {selected_vertices}")
            ax.scatter(selected_vertices[:, 0], selected_vertices[:, 1], selected_vertices[:, 2], 
                      color='red', s=200, alpha=1.0, edgecolors='darkred', linewidth=2, 
                      label=f'Selected Vertices ({len(selected_vertices)})', marker='o')
            for i, vertex in enumerate(selected_vertices):
                ax.text(vertex[0], vertex[1], vertex[2], f'V{i+1}', 
                       fontsize=10, color='red', ha='center', va='bottom', weight='bold')
        
        # Plot edges if provided
        if edges is not None and len(edges) > 0 and selected_vertices is not None:
            print(f"  → Plotting {len(edges)} reconstructed edges as black lines")
            for edge_idx, (v1_idx, v2_idx) in enumerate(edges):
                v1 = selected_vertices[v1_idx]
                v2 = selected_vertices[v2_idx]
                ax.plot([v1[0], v2[0]], [v1[1], v2[1]], [v1[2], v2[2]], 
                       color='black', linewidth=3, alpha=0.9, zorder=10)
            # Add label for edges in legend
            ax.plot([], [], color='black', linewidth=3, alpha=0.9, 
                   label=f'Reconstructed Edges ({len(edges)})')
        
        ax.set_xlabel('X Coordinate', fontsize=12, weight='bold')
        ax.set_ylabel('Y Coordinate', fontsize=12, weight='bold')
        ax.set_zlabel('Z Coordinate', fontsize=12, weight='bold')
        title = f'3D Solid Visualization - POLYGON BOUNDARIES ONLY\n{len(all_face_data)} Faces'
        if selected_vertices is not None:
            title += f' + {len(selected_vertices)} Selected Vertices'
        if edges is not None and len(edges) > 0:
            title += f' + {len(edges)} Edges'
        ax.set_title(title, fontsize=14, weight='bold')
        all_vertices = np.vstack([face_data['vertices'] for face_data in all_face_data])
        max_range = np.ptp(all_vertices, axis=0).max() / 2.0
        mid_x = np.mean(all_vertices[:, 0])
        mid_y = np.mean(all_vertices[:, 1])
        mid_z = np.mean(all_vertices[:, 2])
        margin = max_range * 0.15
        ax.set_xlim(mid_x - max_range - margin, mid_x + max_range + margin)
        ax.set_ylim(mid_y - max_range - margin, mid_y + max_range + margin)
        ax.set_zlim(mid_z - max_range - margin, mid_z + max_range + margin)
        if selected_vertices is not None and len(selected_vertices) > 0:
            ax.legend(loc='upper left', bbox_to_anchor=(0.02, 0.98), fontsize=10)
        else:
            handles, labels = ax.get_legend_handles_labels()
            if len(handles) > 10:
                ax.legend(handles[:10], labels[:10], loc='upper left', bbox_to_anchor=(0.02, 0.98), fontsize=9)
            else:
                ax.legend(loc='upper left', bbox_to_anchor=(0.02, 0.98), fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.view_init(elev=25, azim=45)
        info_text = f"""PURE POLYGON DISPLAY\n• No triangulation applied\n• All faces shown as true polygons\n• Face 3 should show 5-vertex pentagon\n• Inclined edges clearly visible\n• {len(all_face_data)} faces total"""
        ax.text2D(0.02, 0.02, info_text, transform=ax.transAxes, 
                 fontsize=10, verticalalignment='bottom', 
                 bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8),
                 fontfamily='monospace')
        plt.tight_layout()
        plt.show()
        print(f"✓ 3D solid visualization complete - POLYGON BOUNDARIES ONLY")
        if selected_vertices is not None:
            print(f"  → Displayed {len(selected_vertices)} selected vertices as red spheres")
        if edges is not None and len(edges) > 0:
            print(f"  → Displayed {len(edges)} reconstructed edges as black lines")
        if selected_vertices is None:
            print(f"  → Displayed {len(all_face_data)} faces as pure polygons")
    except Exception as e:
        print(f"✗ 3D matplotlib visualization failed: {e}")
        print("  → Continuing with array processing...")
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
    # def extract_wire_vertices_in_sequence(wire, face_id=None):
    #     """Extract vertices from a TopoDS_Wire in sequence as a list of [x, y, z] coordinates."""
    #     vertices = []
    #     vertex_explorer = TopExp_Explorer(wire, TopAbs_VERTEX)
    #     while vertex_explorer.More():
    #         vertex = topods.Vertex(vertex_explorer.Current())
    #         pnt = BRep_Tool.Pnt(vertex)
    #         vertices.append([pnt.X(), pnt.Y(), pnt.Z()])
    #         vertex_explorer.Next()
    #     return vertices

    def extract_and_visualize_faces(solid_shape):
        """Extract face polygons and normals from a TopoDS_Shape solid for visualization."""
        face_explorer = TopExp_Explorer(solid_shape, TopAbs_FACE)
        face_polygons = []
        face_count = 0
        while face_explorer.More():
            face_shape = face_explorer.Current()
            face_count += 1
            try:
                face = topods.Face(face_shape)
                wire_explorer = TopExp_Explorer(face, TopAbs_WIRE)
                if wire_explorer.More():
                    wire = topods.Wire(wire_explorer.Current())
                    vertices = extract_wire_vertices_in_sequence(wire, face_count)
                    if vertices and len(vertices) >= 3:
                        # Compute face normal
                        v0, v1, v2 = np.array(vertices[0]), np.array(vertices[1]), np.array(vertices[2])
                        normal = np.cross(v1 - v0, v2 - v0)
                        normal = normal / np.linalg.norm(normal)
                        face_polygons.append({
                            'face_id': face_count,
                            'outer_boundary': vertices,
                            'normal': normal
                        })
            except Exception as e:
                print(f"    Face {face_count}: Error - {e}")
            face_explorer.Next()
        return face_polygons

def classify_faces_by_projection(face_polygons, unit_projection_normal):
    if face_polygons is None:
        print("Warning: face_polygons is None. Returning empty arrays.")
        return [], [], []
    """Enhanced face classification with historic polygon classification algorithm."""
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
            if polygon.area > 1e-6:
                polygon_data_enhanced = {
                    'polygon': polygon,
                    'name': f"Face_{face_id}",
                    'normal': unit_face_normal,
                    'parent_face': np.array(outer_boundary),  # 3D vertices
                    'original_index': i,
                    'dot_product': dot_product,
                    'has_holes': len(projected_holes) > 0
                }
                valid_polygons.append(polygon_data_enhanced)
                array_A_initial.append(polygon_data_enhanced)
                hole_info = f" with {len(projected_holes)} holes" if projected_holes else ""
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
        print(f"  A{i+1}: {name} (dot={dot_product:.3f}, area={area:.2f})")

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
            
            # Test intersection with all polygons in array_B
            # Iterate in reverse order to avoid index issues when modifying array_B
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
def plot_four_views(solid, user_normal,
    ordered_vertices,
    Vertex_Top_View,
    Vertex_Front_View,
    Vertex_Side_View,
    Vertex_Iso_View):
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
    face_polygons = extract_and_visualize_faces(solid)
    print(f"[DEBUG] plot_four_views: extracted {len(face_polygons) if face_polygons else 0} face polygons")
    print(f"[DEBUG] plot_four_views: face_polygons type = {type(face_polygons)}")
    # Store visible/hidden polygons for each view
    view_polygons = []
    for normal, label, vertex_array in view_configs:
        normal = normal / np.linalg.norm(normal)
        print(f"[DEBUG] get_visible_hidden_polygons: Using projection normal: [{normal[0]:.6f}, {normal[1]:.6f}, {normal[2]:.6f}]")
        # Use the already extracted face_polygons, do not re-extract
        _, array_B, array_C = classify_faces_by_projection(face_polygons, normal)
        visible = [data['polygon'] for data in array_B if 'polygon' in data]
        hidden = [data['polygon'] for data in array_C if 'polygon' in data]
        # POPULATE VERTEX ARRAY: Extract edges from face polygons
        print(f"[DEBUG] Populating {label} vertex array...")
        
        def find_3d_vertex_index(vertex_3d, tol=1e-6):
            """Find index of 3D vertex in ordered_vertices list"""
            for idx, v in enumerate(ordered_vertices):
                if np.allclose(vertex_3d, v, atol=tol):
                    return idx
            return None
        
        # Extract edges from face polygons - use original 3D face data
        edges_found = 0
        vertices_used = set()
        
        # Build vertex_array using only visible and hidden polygons
        print(f"[DEBUG] Processing {len(array_B)} visible and {len(array_C)} hidden polygons for edge extraction...")

        def process_polygon_edges(polygon_data, visibility_value):
            # Try to recover original 3D vertices from parent_face
            outer_boundary = polygon_data.get('parent_face', None)
            if outer_boundary is None:
                return
            if hasattr(outer_boundary, 'tolist'):
                outer_boundary = outer_boundary.tolist()
            if not isinstance(outer_boundary, (list, tuple)) or len(outer_boundary) < 3:
                return

            edges_found = 0
            vertices_used = set()
            for i in range(len(outer_boundary)):
                v1_3d = outer_boundary[i]
                v2_3d = outer_boundary[(i + 1) % len(outer_boundary)]
                idx1 = find_3d_vertex_index(v1_3d)
                idx2 = find_3d_vertex_index(v2_3d)
                if (idx1 is not None and idx2 is not None and idx1 != idx2):
                    vertex_array[idx1, idx2] = visibility_value
                    vertex_array[idx2, idx1] = visibility_value
                    edges_found += 1
                    vertices_used.add(idx1)
                    vertices_used.add(idx2)

        # Visible polygons (array_B): visibility_value = 2
        for poly_data in array_B:
            process_polygon_edges(poly_data, 2)

        # Hidden polygons (array_C): visibility_value = 1
        for poly_data in array_C:
            process_polygon_edges(poly_data, 1)
        
        print(f"[DEBUG] {label}: Found {edges_found} edges "
              f"connecting {len(vertices_used)} vertices")
        print(f"[DEBUG] {label}: Vertex array non-zero entries: "
              f"{np.count_nonzero(vertex_array)}")

        print("\n" + "="*60)
        print(f"{label} Vertex Array (1=hidden, 2=visible):")
        print(np.array2string(vertex_array, separator=', '))
        print("="*60)
        # Store polygons for this view
        view_polygons.append((visible, hidden))
        
    # ...existing code...
    # Swap side and isometric views, and fix front view orientation
    
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
    plt.savefig("four_views.pdf", format="pdf")
    plt.show()


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
    save_solid_as_step(solid, "solid_output.step")

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

    # Display original polygons in 3D first
    visualize_3d_solid(solid)

    # Pass arrays and ordered vertices to plot_four_views
    plot_four_views(solid, projection_normal,
                   all_vertices_sorted,
                   Vertex_Top_View,
                   Vertex_Front_View,
                   Vertex_Side_View,
                   Vertex_Iso_View)

    # === Post-processing: Find z-levels from Front_View and build Possible_Vertices ===
    # This must come after the arrays are created and filled (after plot_four_views)
    print("\n[DEBUG] Creating summary arrays for vertex filtering...")

    # Create summary arrays for Front and Top views
    front_view_summary = make_summary_array(Vertex_Front_View, all_vertices_sorted, np.array([0, -1, 0]), 'Front View')
    top_view_summary = make_summary_array(Vertex_Top_View, all_vertices_sorted, np.array([0, 0, 1]), 'Top View')


    if front_view_summary is not None and top_view_summary is not None:
        print(f"[DEBUG] Front view summary shape: {front_view_summary.shape}")
        print(f"[DEBUG] Top view summary shape: {top_view_summary.shape}")
        print(f"\n[DEBUG] CONSERVATIVE APPROACH: Only use vertices that appear in actual edges")
        print(f"[DEBUG] Step 1: Extract vertices that participate in edges from each view")

        # Step 1: Extract unique (x, y) from top view summary
        top_xy_coords = set()
        for i in range(top_view_summary.shape[0]):
            x_proj, y_proj = top_view_summary[i, 3], top_view_summary[i, 4]
            top_xy_coords.add((round(x_proj, 6), round(y_proj, 6)))

        # Step 1: Extract unique z-levels from front view summary
        z_levels = set()
        for i in range(front_view_summary.shape[0]):
            z_world = front_view_summary[i, 2]
            z_levels.add(round(z_world, 6))

        # Step 2: Generate candidate vertices (x, y, z)
        candidate_vertices = []
        for x, y in top_xy_coords:
            for z in z_levels:
                candidate_vertices.append([x, y, z])
        candidate_vertices = np.array(candidate_vertices)
    if front_view_summary is not None and top_view_summary is not None:
        print(f"[DEBUG] Front view summary shape: {front_view_summary.shape}")
        print(f"[DEBUG] Top view summary shape: {top_view_summary.shape}")
        
        print(f"\n[DEBUG] CONSERVATIVE APPROACH: Only use vertices that appear in actual edges")
        print(f"[DEBUG] Step 1: Extract vertices that participate in edges from each view")
        
        # More conservative approach: only consider vertices that are actually connected
        # in the vertex arrays (i.e., have non-zero entries indicating edges)
        
        # Get vertices that actually participate in edges in top view
        top_view_edge_vertices = set()
        n = top_view_summary.shape[0]
        for i in range(n):
            for j in range(n):
                # Check if there's an edge between vertex i and j in the original vertex arrays
                vertex_i_idx = None
                vertex_j_idx = None
                
                # Find vertex indices in the original sorted list
                for idx, orig_vertex in enumerate(all_vertices_sorted):
                    if (abs(orig_vertex[0] - top_view_summary[i, 0]) < 1e-6 and
                        abs(orig_vertex[1] - top_view_summary[i, 1]) < 1e-6 and 
                        abs(orig_vertex[2] - top_view_summary[i, 2]) < 1e-6):
                        vertex_i_idx = idx
                        break
                
                for idx, orig_vertex in enumerate(all_vertices_sorted):
                    if (abs(orig_vertex[0] - top_view_summary[j, 0]) < 1e-6 and
                        abs(orig_vertex[1] - top_view_summary[j, 1]) < 1e-6 and 
                        abs(orig_vertex[2] - top_view_summary[j, 2]) < 1e-6):
                        vertex_j_idx = idx
                        break
                
                # Check if there's an actual edge in the top view vertex array
                if (vertex_i_idx is not None and vertex_j_idx is not None and 
                    vertex_i_idx < Vertex_Top_View.shape[0] and vertex_j_idx < Vertex_Top_View.shape[1] and
                    Vertex_Top_View[vertex_i_idx, vertex_j_idx] > 0):
                    # Add both vertices as they participate in edges
                    top_view_edge_vertices.add((top_view_summary[i, 0], top_view_summary[i, 1], top_view_summary[i, 2]))
                    if j < top_view_summary.shape[0]:
                        top_view_edge_vertices.add((top_view_summary[j, 0], top_view_summary[j, 1], top_view_summary[j, 2]))
        
        # Get vertices that actually participate in edges in front view  
        front_view_edge_vertices = set()
        n = front_view_summary.shape[0]
        for i in range(n):
            for j in range(n):
                # Find vertex indices in the original sorted list
                vertex_i_idx = None
                vertex_j_idx = None
                
                for idx, orig_vertex in enumerate(all_vertices_sorted):
                    if (abs(orig_vertex[0] - front_view_summary[i, 0]) < 1e-6 and
                        abs(orig_vertex[1] - front_view_summary[i, 1]) < 1e-6 and 
                        abs(orig_vertex[2] - front_view_summary[i, 2]) < 1e-6):
                        vertex_i_idx = idx
                        break
                        
                for idx, orig_vertex in enumerate(all_vertices_sorted):
                    if (abs(orig_vertex[0] - front_view_summary[j, 0]) < 1e-6 and
                        abs(orig_vertex[1] - front_view_summary[j, 1]) < 1e-6 and 
                        abs(orig_vertex[2] - front_view_summary[j, 2]) < 1e-6):
                        vertex_j_idx = idx
                        break
                
                # Check if there's an actual edge in the front view vertex array
                if (vertex_i_idx is not None and vertex_j_idx is not None and 
                    vertex_i_idx < Vertex_Front_View.shape[0] and vertex_j_idx < Vertex_Front_View.shape[1] and
                    Vertex_Front_View[vertex_i_idx, vertex_j_idx] > 0):
                    # Add both vertices as they participate in edges
                    front_view_edge_vertices.add((front_view_summary[i, 0], front_view_summary[i, 1], front_view_summary[i, 2]))
                    if j < front_view_summary.shape[0]:
                        front_view_edge_vertices.add((front_view_summary[j, 0], front_view_summary[j, 1], front_view_summary[j, 2]))
        
        print(f"[DEBUG] Top view has {len(top_view_edge_vertices)} vertices in edges")
        print(f"[DEBUG] Front view has {len(front_view_edge_vertices)} vertices in edges")
        
        # Find intersection: vertices that appear in BOTH views' edge sets
        common_vertices = top_view_edge_vertices.intersection(front_view_edge_vertices)
        
        selected_vertices = np.array(list(common_vertices))
        
        print(f"\n" + "="*60)
        print(f"TRUE REVERSE ENGINEERING - STEP-BY-STEP VERTEX RECONSTRUCTION")
        print(f"="*60)
        
        # TRUE REVERSE ENGINEERING APPROACH: Your specified logic
        # Step 1 - Extract coordinates from top and front views
        print("\nStep 1: Extracting coordinates from view summaries...")
        print("Method: (x,y) from top view, z-levels from front view")
        print("Goal: Create all combinations to find real vertices via filtering")
        
        # Extract unique (x,y) coordinates from top view summary (projected coordinates)
        top_xy_coords = set()
        print(f"DEBUG: Top view summary has {top_view_summary.shape[0]} rows")
        for i in range(top_view_summary.shape[0]):
            x_proj, y_proj = top_view_summary[i, 3], top_view_summary[i, 4]
            top_xy_coords.add((round(x_proj, 6), round(y_proj, 6)))
        
        # Extract unique z-levels from front view summary (world z coordinates from column 2)
        # Note: In reverse engineering, we extract z from front view world coords 
        # since front view projection normal [0,1,0] preserves z-coordinate information
        z_levels = set()
        print(f"DEBUG: Front view summary has {front_view_summary.shape[0]} rows")
        for i in range(front_view_summary.shape[0]):
            z_world = front_view_summary[i, 2]  # World z-coordinate
            z_levels.add(round(z_world, 6))
            
        print(f"Extracted (x,y) from top view: {len(top_xy_coords)} coordinates")
        print(f"Extracted z-levels from front view: {len(z_levels)} levels")
        
        # Step 2 - Generate candidate vertices: (x,y) × z-levels
        print("\nStep 2: Generating candidate vertices...")
        print("Method: Every (x,y) from top view at every z-level from front view")
        print("Note: Creates many 'fake' vertices - filtering identifies real ones")
        
        # Create candidate vertices: every (x,y) at every z-level
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
        
        # Create side view summary first
        side_view_summary = make_summary_array(Vertex_Side_View,
                                               all_vertices_sorted,
                                               np.array([1, 0, 0]),
                                               'Side View')
        
        if side_view_summary is None:
            print("[ERROR] Could not create side view summary")
            selected_vertices = np.array([])
        else:
            print(f"[DEBUG] Side view shape: {side_view_summary.shape}")
            
            def project_vertex_to_view_reverse_eng(vertex, normal):
                """Project a 3D vertex to 2D view coordinates for reverse engineering"""
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
                return proj_u, proj_v
            
            # Extract projected coordinates from view summaries for filtering
            # Front view summary: columns 3,5 are the projected (u,v) coordinates  
            # Side view summary: columns 4,5 are the projected (u,v) coordinates
            tolerance = 1.0e-4
            
            front_view_coords = []
            for i in range(front_view_summary.shape[0]):
                u_proj, v_proj = front_view_summary[i, 3], front_view_summary[i, 5]
                front_view_coords.append((u_proj, v_proj))
            
            side_view_coords = []
            for i in range(side_view_summary.shape[0]):
                u_proj, v_proj = side_view_summary[i, 4], side_view_summary[i, 5]
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
            
            # Debug: Print first few coordinates from each view for comparison
            print("DEBUG: First 3 front view coordinates:")
            for i in range(min(3, len(front_view_coords))):
                print(f"  Front {i+1}: ({front_view_coords[i][0]:.6f}, {front_view_coords[i][1]:.6f})")
            
            print("DEBUG: First 3 side view coordinates:")
            for i in range(min(3, len(side_view_coords))):
                print(f"  Side {i+1}: ({side_view_coords[i][0]:.6f}, {side_view_coords[i][1]:.6f})")
            
            for idx, candidate in enumerate(candidate_vertices):
                # Use candidate directly without z-coordinate inversion
                candidate_proj = candidate
                
                # Project candidate to front view (normal = [0,-1,0])
                front_u, front_v = project_vertex_to_view_reverse_eng(candidate_proj, [0, -1, 0])
                
                # Project candidate to side view (normal = [1,0,0])
                side_u, side_v = project_vertex_to_view_reverse_eng(candidate_proj, [1, 0, 0])
                
                # Debug first few projections
                if idx < 3:
                    print(f"DEBUG: Candidate {idx+1}: ({candidate[0]:.3f}, {candidate[1]:.3f}, {candidate[2]:.3f})")
                    print(f"  Front proj: ({front_u:.6f}, {front_v:.6f})")
                    print(f"  Side proj: ({side_u:.6f}, {side_v:.6f})")
                
                # Check if front projection matches any front view coordinate within tolerance
                front_match = False
                for fu, fv in front_view_coords:
                    if abs(front_u - fu) < tolerance and abs(front_v - fv) < tolerance:
                        front_match = True
                        break
                
                # Check if side projection matches any side view coordinate within tolerance
                side_match = False
                for su, sv in side_view_coords:
                    if abs(side_u - su) < tolerance and abs(side_v - sv) < tolerance:
                        side_match = True
                        break
                
                if front_match:
                    front_matches += 1
                if side_match:
                    side_matches += 1
                if front_match and side_match:
                    dual_matches += 1
                    # Store the vertex directly without z-coordinate inversion
                    selected_vertices.append(candidate)
            
            selected_vertices = np.array(selected_vertices)
            
            print(f"Candidates matching front view: {front_matches}")
            print(f"Candidates matching side view: {side_matches}")
            print(f"Candidates matching BOTH views: {dual_matches}")
            print(f"Selected vertices: {len(selected_vertices)}")
            
        print(f"Original solid vertices: {len(all_vertices_sorted)}")
        
        print(f"\n" + "="*60)
        print(f"TRUE REVERSE ENGINEERING RESULTS")
        print(f"="*60)
        
        print(f"\nReconstructed vertices (x, y, z):")
        
        # Sort for consistent output
        if len(selected_vertices) > 0:
            selected_vertices = selected_vertices[np.lexsort((selected_vertices[:, 2], selected_vertices[:, 1], selected_vertices[:, 0]))]
            for i, vertex in enumerate(selected_vertices):
                x, y, z = vertex
                print(f"  Vertex {i+1:2d}: ({x:8.3f}, {y:8.3f}, {z:8.3f})")
        
        print(f"="*60)
        
        # ANALYSIS: Compare reconstructed vs actual vertices
        print("\n" + "="*60)
        print("TRUE REVERSE ENGINEERING COMPARISON")
        print("="*60)
        
        # Convert actual vertices to same format for comparison
        actual_vertices = np.array(all_vertices_sorted)
        print(f"Actual vertices from solid: {len(actual_vertices)}")
        print(f"Reconstructed vertices: {len(selected_vertices)}")
        
        # Find which actual vertices are found in our reconstruction
        found_vertices = []
        tolerance = 1e-3  # Use slightly larger tolerance for comparison
        
        for actual_vertex in actual_vertices:
            found = False
            for reconstructed_vertex in selected_vertices:
                if np.allclose(actual_vertex, reconstructed_vertex, atol=tolerance):
                    found = True
                    found_vertices.append(actual_vertex)
                    break
            if not found:
                pass  # Will count below
        
        missing_count = len(actual_vertices) - len(found_vertices)
        print(f"Missing vertices: {missing_count}")
        
        # Summary statement
        if len(actual_vertices) > 0:
            detection_rate = (len(found_vertices) / len(actual_vertices) * 100)
        else:
            detection_rate = 0
        print(f"\nTRUE REVERSE ENGINEERING DETECTION SUMMARY:")
        print(f"Found {len(found_vertices)} out of {len(actual_vertices)} vertices")
        print(f"Success rate: {detection_rate:.1f}%")
        
        # Report false positives
        false_positives = len(selected_vertices) - len(found_vertices)
        if false_positives > 0:
            print(f"False positives: {false_positives} extra vertices detected")
        
        print(f"Total reconstructed: {len(selected_vertices)} vertices")
        
        print(f"="*60)
        
        # Save to file for further analysis
        if len(selected_vertices) > 0:
            output_filename = "selected_vertices.txt"
            with open(output_filename, 'w') as f:
                f.write("Selected 3D Solid Vertices - TRUE REVERSE ENGINEERING\n")
                f.write("="*50 + "\n")
                f.write(f"Total vertices: {len(selected_vertices)}\n")
                f.write("Format: x, y, z\n\n")
                for i, vertex in enumerate(selected_vertices):
                    x, y, z = vertex
                    f.write(f"{i+1:2d}: {x:8.3f}, {y:8.3f}, {z:8.3f}\n")
            print(f"[DEBUG] Selected vertices saved to {output_filename}")
            
            # === EDGE RECONSTRUCTION ===
            print("\n[DEBUG] Starting edge reconstruction...")
            print(f"[DEBUG] Target: {expected_unique_edges} unique edges")
            valid_edges = reconstruct_edges_from_views(
                selected_vertices,
                top_view_summary,
                front_view_summary,
                side_view_summary,
                Vertex_Top_View,
                Vertex_Front_View,
                Vertex_Side_View,
                all_vertices_sorted,
                expected_unique_edges
            )
            
            # Save edges to file
            if len(valid_edges) > 0:
                edge_filename = "reconstructed_edges.txt"
                with open(edge_filename, 'w') as f:
                    f.write("Reconstructed 3D Solid Edges\n")
                    f.write("="*50 + "\n")
                    f.write(f"Total edges: {len(valid_edges)}\n")
                    f.write("Format: V1_idx -- V2_idx\n\n")
                    for i, (v1_idx, v2_idx) in enumerate(valid_edges):
                        v1 = selected_vertices[v1_idx]
                        v2 = selected_vertices[v2_idx]
                        f.write(f"Edge {i+1:2d}: V{v1_idx:2d} -- V{v2_idx:2d}\n")
                        f.write(f"  Start: ({v1[0]:8.3f}, {v1[1]:8.3f}, {v1[2]:8.3f})\n")
                        f.write(f"  End:   ({v2[0]:8.3f}, {v2[1]:8.3f}, {v2[2]:8.3f})\n")
                print(f"[DEBUG] Reconstructed edges saved to {edge_filename}")
            
            # === CREATE UNIFIED SUMMARY ARRAY ===
            print("\n" + "="*70)
            print("CREATING UNIFIED SUMMARY ARRAY")
            print("="*70)
            
            unified_summary = create_unified_summary(
                selected_vertices,
                valid_edges
            )
            
            # Print summary information
            print_summary_info(unified_summary)
            
            # Save to text file
            summary_txt = "unified_summary.txt"
            save_summary_to_file(unified_summary, summary_txt)
            print(f"[DEBUG] Unified summary saved to {summary_txt}")
            
            # Save to numpy binary file
            summary_npy = "unified_summary.npy"
            save_summary_to_numpy(unified_summary, summary_npy)
            
            # Visualize adjacency matrix
            try:
                visualize_adjacency_matrix(
                    unified_summary,
                    "adjacency_matrix.png"
                )
            except Exception as e:
                print(f"[DEBUG] Could not create adjacency "
                      f"visualization: {e}")
            
            print("="*70)
            
            # Visualize the solid with selected vertices highlighted
            print("\n[DEBUG] Creating 3D visualization...")
            visualize_3d_solid(solid, selected_vertices, valid_edges)
        else:
            print("[WARNING] No vertices selected - showing solid only")
            visualize_3d_solid(solid, None)
        
    else:
        print("[ERROR] Could not create summary arrays for vertex filtering")
        # Still show the solid even if filtering failed
        visualize_3d_solid(solid, None)
    extract_possible_vertices_from_summaries(Vertex_Front_View, Vertex_Top_View, all_vertices_sorted)






if __name__ == "__main__":
    main()


