import os
import sys
print("Running file:", os.path.abspath(__file__))
from OCC.Core.gp import gp_Trsf  # noqa: F401
from OCC.Core.TopLoc import TopLoc_Location  # noqa: F401
from OCC.Core.TopAbs import (
    TopAbs_SHELL, TopAbs_EDGE, TopAbs_VERTEX, TopAbs_WIRE
)  # noqa: F401, E501
from OCC.Core.BRep import BRep_Tool
from OCC.Core.BRepAdaptor import BRepAdaptor_Curve
from OCC.Core.STEPControl import STEPControl_Writer, STEPControl_AsIs
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

# OpenCASCADE imports
try:
    from OCC.Core.gp import gp_Vec
    from OCC.Core.BRepPrimAPI import BRepPrimAPI_MakeBox
    from OCC.Core.TopExp import TopExp_Explorer
    from OCC.Core.TopAbs import TopAbs_FACE
    from OCC.Core.TopoDS import topods
    
    # Try to import TopExp for vertex extraction
    try:
        from OCC.Core.TopExp import topexp, topexp_Vertices
        TOPEXP_AVAILABLE = True
    except Exception:
        TOPEXP_AVAILABLE = False
    
    # Visualization imports
    
    OPENCASCADE_AVAILABLE = True
except ImportError as e:
    print(f"OpenCASCADE not available: {e}")
    OPENCASCADE_AVAILABLE = False
    TOPEXP_AVAILABLE = False
# ============================================================================
# 3D CUBOID FACE PROJECTION AND POLYGON OPERATIONS USING OPENCASCADE
# ============================================================================

# Helper function to plot polygon
def plot_polygon(
    polygon, ax, facecolor='none', edgecolor='black', alpha=0.7, linestyle='-',
    linewidth=2, label=None, outline_only=False
):
    if polygon.geom_type == 'Polygon':
        if outline_only:
            # Only draw the outline (for standalone polygon plots)
            x, y = polygon.exterior.xy
            ax.plot(
                x, y, color=edgecolor, linestyle=linestyle,
                linewidth=linewidth, label=label
            )
        else:
            # Draw filled patch without separate outline (for combined plots)
            if facecolor != 'none':
                patch = patches.Polygon(
                    list(polygon.exterior.coords), closed=True,
                    facecolor=facecolor, alpha=alpha, edgecolor=edgecolor,
                    linewidth=linewidth, linestyle=linestyle
                )
                ax.add_patch(patch)
                # Add invisible line for legend if label is provided
                if label:
                    ax.plot(
                        [], [], color=edgecolor, linestyle=linestyle,
                        linewidth=linewidth, label=label
                    )

    elif polygon.geom_type == 'MultiPolygon':
        for poly in polygon.geoms:
            plot_polygon(
                poly, ax, facecolor, edgecolor, alpha, linestyle, linewidth,
                label=None, outline_only=outline_only
            )

    # Only add legend if there are labeled artists
    handles, labels = ax.get_legend_handles_labels()
    if handles and labels:
        ax.legend(loc='upper left', bbox_to_anchor=(0.02, 0.98), fontsize=9)


def build_solid_with_polygons_test(seed=47315, quiet=False):
    from Base_Solid import build_solid_with_polygons

    print(f"[DEBUG] Calling build_solid_with_polygons(seed={seed}, "
          f"quiet={quiet}) as test...")
    return build_solid_with_polygons(seed, quiet)



def save_solid_as_step(solid_shape, filename="solid_output.step"):
    step_writer = STEPControl_Writer()
    step_writer.Transfer(solid_shape, STEPControl_AsIs)
    status = step_writer.Write(filename)
    if status == IFSelect_RetDone:
        print(f"✓ STEP file saved as '{filename}'")
    else:
        print("✗ Failed to save STEP file.")

# Example usage:
# save_solid_as_step(solid_shape, "solid_output.step")


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

def extract_faces_from_solid(solid):
    """Extract face data from an OpenCASCADE solid using proper BRep traversal.
    
    Follows the OpenCASCADE topology hierarchy:
    Solid -> Shells -> Faces -> Wires(Loops) -> Edges -> Vertices
    
    Each face may have multiple wires:
    - First wire is the outer boundary
    - Additional wires are holes/cutouts
    """
    if not OPENCASCADE_AVAILABLE or solid is None:
        return []
    
    faces = []
    
    print("  Traversing BRep topology: Solid -> Shells -> Faces -> Wires -> Edges -> Vertices")
    
    # Explore shells in the solid
    shell_explorer = TopExp_Explorer(solid, TopAbs_SHELL)
    shell_count = 0
    
    while shell_explorer.More():
        shell_count += 1
        shell_explorer.Next()
    
    print(f"  Found {shell_count} shells in solid")
    
    # Check for multiple shells - abort if more than 2
    if shell_count > 2:
        print(f"  ✗ ABORTING: Found {shell_count} shells (expected ≤ 2)")
        print(f"    Complex multi-shell solids not supported")
        return []
    elif shell_count == 2:
        print(f"  ⚠️  WARNING: Found 2 shells - may indicate hollow solid or complex geometry")
    
    # Reset explorer and process shells
    shell_explorer = TopExp_Explorer(solid, TopAbs_SHELL)
    shell_num = 0
    face_count = 0
    
    while shell_explorer.More():
        shell = shell_explorer.Current()
        shell_num += 1
        print(f"  \nShell {shell_num}:")
        
        # Explore faces in each shell
        face_explorer = TopExp_Explorer(shell, TopAbs_FACE)
        
        while face_explorer.More():
            face_shape = face_explorer.Current()
            face_count += 1
            
            try:
                face = topods.Face(face_shape)
                
                print(f"    Face {face_count}:")
                
                # Extract the actual face normal from OpenCASCADE
                face_normal = get_face_normal_from_opencascade(face)
                
                # Extract polygon with cutouts using proper BRep traversal
                polygon_data = {}
                
                # Extract wires from the face
                wire_explorer = TopExp_Explorer(face, TopAbs_WIRE)
                wires = []
                
                while wire_explorer.More():
                    wire = wire_explorer.Current()
                    wires.append(wire)
                    wire_explorer.Next()
                
                print(f"      Found {len(wires)} wires in face {face_count}")
                
                if wires:
                    # First wire is the outer boundary
                    outer_boundary = extract_wire_vertices_in_sequence(wires[0], 1)
                    polygon_data['outer_boundary'] = outer_boundary
                    
                    # Additional wires are cutouts/holes
                    cutouts = []
                    for i, wire in enumerate(wires[1:], 2):
                        cutout_vertices = extract_wire_vertices_in_sequence(wire, i)
                        if cutout_vertices:
                            cutouts.append(cutout_vertices)
                    
                    polygon_data['cutouts'] = cutouts
                else:
                    print(f"      No wires found, using fallback vertex extraction")
                    outer_boundary = extract_face_vertices_in_sequence(face)
                    polygon_data['outer_boundary'] = outer_boundary
                    polygon_data['cutouts'] = []
                
                if polygon_data['outer_boundary'] and face_normal is not None:
                    polygon_data['normal'] = face_normal
                    polygon_data['face_id'] = face_count
                    
                    # Analyze co-linearity and axis alignment for this face
                    analyze_face_colinearity(polygon_data['outer_boundary'], face_count)
                    
                    faces.append(polygon_data)
                    
                    outer_vertices = len(polygon_data['outer_boundary'])
                    cutout_count = len(polygon_data['cutouts'])
                    total_vertices = outer_vertices + sum(len(cutout) for cutout in polygon_data['cutouts'])
                    
                    print(f"      ✓ Extracted polygon: {outer_vertices} outer vertices, {cutout_count} cutouts, {total_vertices} total vertices")
                else:
                    print(f"      ✗ Failed to extract polygon data")
            
            except Exception as e:
                print(f"    Face {face_count}: error processing - {e}")
            
            face_explorer.Next()
        
        shell_explorer.Next()
    
    print(f"  \n✓ Successfully extracted {len(faces)} faces from {shell_count} shells")
    return faces

def extract_face_vertices_in_sequence(face):
    """Extract vertices from a face in proper sequence by following wires and edges.
    
    Uses OpenCASCADE's natural topology traversal to maintain correct vertex ordering.
    """
    print("        This function has never been checked...")
    vertices = []
    # Tag will be attached later
    
    try:
        # Method 1: Use BRepMesh to triangulate the face and extract vertices
        # This preserves the natural OpenCASCADE ordering
        from OCC.Core.BRepMesh import BRepMesh_IncrementalMesh
        from OCC.Core.TopLoc import TopLoc_Location
        
        # Apply mesh to the face
        mesh = BRepMesh_IncrementalMesh(face, 0.1)
        mesh.Perform()
        
        if mesh.IsDone():
            # Get the triangulation
            location = TopLoc_Location()
            triangulation = BRep_Tool.Triangulation(topods.Face(face), location)
            
            if triangulation:
                # Extract vertices from the boundary
                # Get the outer wire for boundary vertices
                wire_explorer = TopExp_Explorer(face, TopAbs_WIRE)
                
                if wire_explorer.More():
                    wire = wire_explorer.Current()
                    
                    # Method 2: Use sequential edge approach with proper ordering
                    edge_vertices = []
                    edge_explorer = TopExp_Explorer(wire, TopAbs_EDGE)
                    
                    while edge_explorer.More():
                        edge = edge_explorer.Current()
                        
                        if TOPEXP_AVAILABLE:
                            try:
                                # Get edge vertices - topexp_Vertices maintains edge direction
                                from OCC.Core.TopoDS import TopoDS_Vertex
                                vertex1 = TopoDS_Vertex()
                                vertex2 = TopoDS_Vertex()
                                topexp_Vertices(topods.Edge(edge), vertex1, vertex2)
                                
                                pnt1 = BRep_Tool.Pnt(vertex1)
                                pnt2 = BRep_Tool.Pnt(vertex2)
                                
                                v1 = [pnt1.X(), pnt1.Y(), pnt1.Z()]
                                v2 = [pnt2.X(), pnt2.Y(), pnt2.Z()]
                                
                                # Build edge chain - only add vertices that aren't already in the chain
                                if not edge_vertices:
                                    edge_vertices.extend([v1, v2])
                                else:
                                    # Check connectivity - add the vertex that's not already the last vertex
                                    last_vertex = edge_vertices[-1]
                                    last_tuple = tuple(np.round(last_vertex, 6))
                                    v1_tuple = tuple(np.round(v1, 6))
                                    v2_tuple = tuple(np.round(v2, 6))
                                    
                                    if v1_tuple == last_tuple:
                                        # v1 connects to last vertex, add v2
                                        if v2_tuple != tuple(np.round(edge_vertices[0], 6)):  # Don't close loop yet
                                            edge_vertices.append(v2)
                                    elif v2_tuple == last_tuple:
                                        # v2 connects to last vertex, add v1
                                        if v1_tuple != tuple(np.round(edge_vertices[0], 6)):  # Don't close loop yet
                                            edge_vertices.append(v1)
                                    else:
                                        # Edge doesn't connect - this might be a different wire or edge order issue
                                        # For now, just add both vertices
                                        edge_vertices.extend([v1, v2])
                                
                            except Exception as e:
                                print(f"          TopExp.Vertices failed for edge: {e}")
                        
                        edge_explorer.Next()
                    
                    # Remove any duplicate vertices from the end (closing vertex)
                    if edge_vertices and len(edge_vertices) > 1:
                        first_tuple = tuple(np.round(edge_vertices[0], 6))
                        last_tuple = tuple(np.round(edge_vertices[-1], 6))
                        if first_tuple == last_tuple:
                            edge_vertices = edge_vertices[:-1]  # Remove closing duplicate
                    
                    vertices = edge_vertices
                    print(f"        Extracted {len(vertices)} vertices using edge chain method")
                    # Print vertices in one line for debugging
                    if vertices:
                        vertex_coords = " → ".join([f"({v[0]:.1f},{v[1]:.1f},{v[2]:.1f})" for v in vertices])
                        print(f"        VERTEX ORDER: {vertex_coords}")
        
        # Fallback: Basic wire traversal if mesh approach fails
        if not vertices:
            print(f"        Mesh method failed, using basic wire traversal")
            wire_explorer = TopExp_Explorer(face, TopAbs_WIRE)
            
            if wire_explorer.More():
                wire = wire_explorer.Current()
                vertex_explorer = TopExp_Explorer(wire, TopAbs_VERTEX)
                
                vertex_list = []
                while vertex_explorer.More():
                    vertex = topods.Vertex(vertex_explorer.Current())
                    pnt = BRep_Tool.Pnt(vertex)
                    v = [pnt.X(), pnt.Y(), pnt.Z()]
                    vertex_list.append(v)
                    vertex_explorer.Next()
                
                # Remove duplicates while preserving order
                seen = set()
                for v in vertex_list:
                    v_tuple = tuple(np.round(v, 6))
                    if v_tuple not in seen:
                        vertices.append(v)
                        seen.add(v_tuple)
                
                print(f"        Extracted {len(vertices)} vertices using basic traversal")
                # Print vertices in one line for debugging
                if vertices:
                    vertex_coords = " → ".join([f"({v[0]:.1f},{v[1]:.1f},{v[2]:.1f})" for v in vertices])
                    print(f"        VERTEX ORDER: {vertex_coords}")
    
    except Exception as e:
        print(f"      Error extracting vertices: {e}")
        vertices = []
    
    return vertices

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
                print(f"    → Polygon is invalid, but allow_invalid=True: storing as-is")
            else:
                print(f"    → Valid polygon created with {len(polygon.exterior.coords)-1} vertices")
            return polygon

        # Default: only return valid polygons
        if polygon.is_valid and hasattr(polygon, 'area') and polygon.area > 1e-6:
            print(f"    → Valid polygon created with {len(polygon.exterior.coords)-1} vertices")
            return polygon

        # For invalid polygons, try to fix
        if not polygon.is_valid:
            print(f"    → Invalid polygon detected (reason: {polygon.is_valid}), attempting to fix...")
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
            if polygon.geom_type == 'Polygon':
                if polygon.area > 0:
                    plot_polygon(polygon, ax3, facecolor='none', edgecolor='lightgray', alpha=0.8, linewidth=0.7, linestyle='--', label=f'C: {name}', outline_only=True)
                else:
                    # Degenerate polygon (zero area): plot as black dashed line
                    coords = list(polygon.exterior.coords)
                    print(f"[DEBUG] Plotting degenerate polygon in Array_C (combined): {name}, coords={coords}")
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
            print(f"Error plotting array_C polygon in combined subplot: {e}")

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


def visualize_3d_solid(solid_shape):
    """Display the 3D solid using matplotlib 3D plotting - showing only polygon boundaries."""
    if not OPENCASCADE_AVAILABLE or solid_shape is None:
        print("✗ Cannot visualize - OpenCASCADE not available or shape is None")
        return
    
    print("\n" + "="*60)
    print("3D SOLID VISUALIZATION WITH MATPLOTLIB")
    print("="*60)
    
    try:
        from mpl_toolkits.mplot3d import Axes3D
        import matplotlib.pyplot as plt
        
        # Extract all face vertices from the solid for visualization
        print("  → Extracting face vertices for 3D plot...")
        
        face_explorer = TopExp_Explorer(solid_shape, TopAbs_FACE)
        face_count = 0
        all_face_data = []
        
        while face_explorer.More():
            face_shape = face_explorer.Current()
            face_count += 1
            
            try:
                face = topods.Face(face_shape)
                
                # Extract vertices from this face
                wire_explorer = TopExp_Explorer(face, TopAbs_WIRE)
                if wire_explorer.More():
                    wire = wire_explorer.Current()
                    vertices = extract_wire_vertices_in_sequence(wire, 1)
                    
                    if vertices and len(vertices) >= 3:
                        all_face_data.append({
                            'face_id': face_count,
                            'vertices': vertices,
                            'vertex_count': len(vertices)
                        })
                        print(f"    Face {face_count}: {len(vertices)} vertices extracted")
                    else:
                        print(f"    Face {face_count}: Failed to extract enough vertices")
                
            except Exception as e:
                print(f"    Face {face_count}: Error - {e}")
            
            face_explorer.Next()
        
        print(f"  → Successfully extracted {len(all_face_data)} faces for visualization")
        
        if not all_face_data:
            print("  ✗ No face data available for visualization")
            return
        
        # Create 3D plot
        fig = plt.figure(figsize=(15, 12))
        ax = fig.add_subplot(111, projection='3d')
        
        # Plot each face - ONLY POLYGON BOUNDARIES, NO TRIANGULATION
        colors = plt.cm.Set3(np.linspace(0, 1, len(all_face_data)))
        
        for i, face_data in enumerate(all_face_data):
            vertices = np.array(face_data['vertices'])
            if len(vertices) > 2:
                vertices_closed = np.vstack([vertices, vertices[0]])
            ax.plot(vertices_closed[:, 0], vertices_closed[:, 1], vertices_closed[:, 2], 
                color=colors[i], linewidth=3, alpha=0.9)
            ax.scatter(vertices[:, 0], vertices[:, 1], vertices[:, 2], 
                color=colors[i], s=50, alpha=0.8, edgecolors='black', linewidth=1)
            face_center = np.mean(vertices, axis=0)
            label_text = f'F{i+1} ({len(vertices)}v)'
            ax.text(face_center[0], face_center[1], face_center[2], 
                label_text,
                fontsize=10, color='black', ha='center', va='center', alpha=0.7)
            
            # Set labels and title
            ax.set_xlabel('X Coordinate', fontsize=12, weight='bold')
            ax.set_ylabel('Y Coordinate', fontsize=12, weight='bold')
            ax.set_zlabel('Z Coordinate', fontsize=12, weight='bold')
            ax.set_title(f'3D Solid Visualization - POLYGON BOUNDARIES ONLY\n{len(all_face_data)} Faces from Boolean CUT Operation\nNo Triangulation - Pure Polygon Display', 
                        fontsize=14, weight='bold')
            
            # Set equal aspect ratio
            all_vertices = np.vstack([face_data['vertices'] for face_data in all_face_data])
            max_range = np.ptp(all_vertices, axis=0).max() / 2.0
            mid_x = np.mean(all_vertices[:, 0])
            mid_y = np.mean(all_vertices[:, 1])
            mid_z = np.mean(all_vertices[:, 2])
            
            margin = max_range * 0.1  # 10% margin
            ax.set_xlim(mid_x - max_range - margin, mid_x + max_range + margin)
            ax.set_ylim(mid_y - max_range - margin, mid_y + max_range + margin)
            ax.set_zlim(mid_z - max_range - margin, mid_z + max_range + margin)
            
            # Add legend (show only first few faces to avoid clutter)
            handles, labels = ax.get_legend_handles_labels()
            if len(handles) > 10:
                ax.legend(handles[:10], labels[:10], loc='upper left', bbox_to_anchor=(0.02, 0.98), fontsize=9)
            else:
                ax.legend(loc='upper left', bbox_to_anchor=(0.02, 0.98), fontsize=9)
            
            # Add grid for better visualization
            ax.grid(True, alpha=0.3)
            
            # Set viewing angle for better perspective
            ax.view_init(elev=25, azim=45)
            
            # Add information text
            info_text = f"""PURE POLYGON DISPLAY
• No triangulation applied
• All faces shown as true polygons
• Face 3 should show 5-vertex pentagon
• Inclined edges clearly visible
• {len(all_face_data)} faces total"""
        
        ax.text2D(0.02, 0.02, info_text, transform=ax.transAxes, 
                 fontsize=10, verticalalignment='bottom', 
                 bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8),
                 fontfamily='monospace')
        
        plt.tight_layout()
        plt.show()
        
        print(f"✓ 3D solid visualization complete - POLYGON BOUNDARIES ONLY")
        print(f"  → Displayed {len(all_face_data)} faces as pure polygons")
        print(f"  → NO triangulation applied to any face")
        print(f"  → Face 3 should appear as 5-vertex pentagon with inclined edge")
        print(f"  → Total vertices plotted: {sum(len(face_data['vertices']) for face_data in all_face_data)}")
        
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
            print(f"          Co-linear vertices found: {i}-{(i+1)%len(face_vertices)}-{(i+2)%len(face_vertices)}")
            print(f"            V{i}: ({v1[0]:.3f}, {v1[1]:.3f}, {v1[2]:.3f})")
            print(f"            V{(i+1)%len(face_vertices)}: ({v2[0]:.3f}, {v2[1]:.3f}, {v2[2]:.3f})")
            print(f"            V{(i+2)%len(face_vertices)}: ({v3[0]:.3f}, {v3[1]:.3f}, {v3[2]:.3f})")
            print(f"            Cross product magnitude: {cross_magnitude:.8f}")
    
    if not colinear_groups:
        print(f"          ✓ No co-linear vertices detected")
    else:
        print(f"          ⚠️  {len(colinear_groups)} co-linear groups found - may cause triangle appearance")
    
    # Check axis alignment
    print(f"        Axis Alignment Analysis:")
    for i in range(len(face_vertices)):
        v1 = np.array(face_vertices[i])
        v2 = np.array(face_vertices[(i + 1) % len(face_vertices)])
        
        edge_vector = v2 - v1
        edge_length = np.linalg.norm(edge_vector)
        
        if edge_length > 1e-6:
            edge_unit = edge_vector / edge_length
            
            # Check alignment with main axes
            x_alignment = abs(np.dot(edge_unit, [1, 0, 0]))
            y_alignment = abs(np.dot(edge_unit, [0, 1, 0]))
            z_alignment = abs(np.dot(edge_unit, [0, 0, 1]))
            
            max_alignment = max(x_alignment, y_alignment, z_alignment)
            
            if max_alignment > 0.99:  # Very close to axis-aligned
                axis = "X" if x_alignment == max_alignment else "Y" if y_alignment == max_alignment else "Z"
                print(f"          Edge {i}-{(i+1)%len(face_vertices)}: ✓ {axis}-axis aligned ({max_alignment:.6f})")
            else:
                print(f"          Edge {i}-{(i+1)%len(face_vertices)}: ⚠️  Inclined (max alignment: {max_alignment:.6f})")
                print(f"            X: {x_alignment:.6f}, Y: {y_alignment:.6f}, Z: {z_alignment:.6f}")

def classify_faces_by_projection(face_polygons, unit_projection_normal):
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

        print(f"Face F{face_id}: dot_product={dot_product:.3f}")

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
                    
                    if (not intersection.is_empty and
                        hasattr(intersection, 'area') and
                        intersection.area > 1e-6):
                        # Find interior point for depth analysis
                        result = find_interior_point(intersection, debug=False)
                        if isinstance(result, tuple):
                            interior_point, method_used = result
                        else:
                            interior_point = result
                        if interior_point is None:
                            continue
                        # Calculate 3D depths using line-face intersection
                        Pi_intersection_3d = intersect_line_with_face(
                            interior_point, unit_projection_normal, Pi_parent_face)
                        Pj_intersection_3d = intersect_line_with_face(
                            interior_point, unit_projection_normal, Pj_parent_face)
                        Pi_depth = calculate_depth_along_normal(Pi_intersection_3d, unit_projection_normal)
                        Pj_depth = calculate_depth_along_normal(Pj_intersection_3d, unit_projection_normal)
                        
                        # Add intersection to array_C
                        intersection_name = f"Intersection_{Pi_name}_{Pj_name}"
                        intersection_data = {
                            'polygon': intersection,
                            'name': intersection_name,
                            'normal': 'intersection',
                            'parent_face': Pi_parent_face if Pi_depth > Pj_depth else Pj_parent_face,
                            'associated_face': Pi_name if Pi_depth > Pj_depth else Pj_name,
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
                                pass
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
                            except Exception:
                                pass
                
                except Exception:
                    pass
            
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

def get_visible_hidden_polygons(solid, projection_normal):
    """Extract visible and hidden polygons from a solid using projection normal."""
    print(f"[DEBUG] get_visible_hidden_polygons: Using projection normal: [{projection_normal[0]:.6f}, {projection_normal[1]:.6f}, {projection_normal[2]:.6f}]")
    # Extract face polygons from the solid
    face_polygons = extract_faces_from_solid(solid)
    # Classify faces using the projection normal
    _, array_B, array_C = classify_faces_by_projection(face_polygons, projection_normal)
    # array_B: visible polygons, array_C: hidden polygons
    # Return lists of shapely polygons for plotting
    # Tagging logic: assign tag based on face_id and source
    def assign_tag(face_id):
        # Robust tag assignment: if face_id is None or invalid, return empty string
        if face_id is None or face_id < 0:
            return ""
        if face_id < 6:
            return f"B{face_id+1}"
        elif face_id < 12:
            return f"S{face_id-5}"
        else:
            return f"H{face_id-11}"

    visible = []
    for data in array_B:
        if 'polygon' in data:
            face_id = data.get('face_id')
            tag = assign_tag(face_id)
            visible.append((data['polygon'], tag))
    hidden = []
    for data in array_C:
        if 'polygon' in data:
            face_id = data.get('face_id')
            tag = assign_tag(face_id)
            hidden.append((data['polygon'], tag))
    return visible, hidden

#def plot_polygons(visible, hidden, show_combined, show_visible, show_hidden):
def plot_four_views(solid, user_normal):
    import matplotlib.pyplot as plt
    import numpy as np
    # Swap side and isometric views, and fix front view orientation
    def plot_polygons_on_ax(ax, visible, hidden, label, flip_y=False):
        coords_x = []
        coords_y = []
        polygons_drawn = False
        # Plot hidden polygons first
        for poly, _ in hidden:
            if hasattr(poly, 'exterior') and not poly.is_empty:
                x, y = poly.exterior.xy
                # Dashed gray line: small dashes and gaps
                ax.plot(x, y, color='gray', linestyle=(0, (2, 2)), linewidth=1.2, alpha=0.8)
                polygons_drawn = True
                coords_x.append(x)
                coords_y.append(y)
                for interior in poly.interiors:
                    ix, iy = interior.xy
                    ax.plot(ix, iy, color='gray', linestyle=(0, (2, 2)), linewidth=1.2, alpha=0.8)
        # Plot visible polygons
        for poly, _ in visible:
            if hasattr(poly, 'exterior') and not poly.is_empty:
                x, y = poly.exterior.xy
                # Solid black line
                ax.plot(x, y, color='black', linewidth=1.8, alpha=0.95)
                polygons_drawn = True
                coords_x.append(x)
                coords_y.append(y)
                for interior in poly.interiors:
                    ix, iy = interior.xy
                    ax.plot(ix, iy, color='black', linewidth=1.8, alpha=0.95)
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
        (np.array([0,0,1]), 'Top View', False),
        (user_normal, 'Isometric View', False),
        (np.array([0,1,0]), 'Front View', True),
        (np.array([1,0,0]), 'Side View', False)
    ]
    fig, axes = plt.subplots(2, 2, figsize=(12, 12))
    axes = axes.flatten()
    for i, ax in enumerate(axes):
        normal, label, flip_y = views[i]
        normal = normal / np.linalg.norm(normal)
        print(f"[DEBUG] plot_four_views: {label} projection normal: [{normal[0]:.6f}, {normal[1]:.6f}, {normal[2]:.6f}]")
        visible, hidden = get_visible_hidden_polygons(solid, normal)
        print(f"[DEBUG] {label}: {len(visible)} visible, {len(hidden)} hidden polygons")
        plot_polygons_on_ax(ax, visible, hidden, label, flip_y)
    plt.tight_layout()
    plt.savefig("four_views.pdf", format="pdf")
    plt.show()

def main():
    print("Starting solid projection and polygon visibility analysis...")
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
        '--quiet', action='store_true',
        help='Suppress verbose output'
    )
    args = parser.parse_args()

    print("[DEBUG] Starting main() function.")
    print(f"[DEBUG] CLI args: {sys.argv}")

    # Accept comma-separated normal early for debug
    normal_arg = None
    for i, arg in enumerate(sys.argv):
        if arg == '--normal' and i + 1 < len(sys.argv):
            normal_arg = sys.argv[i + 1]
            break
        elif arg.startswith('--normal='):
            normal_arg = arg.split('=', 1)[1]
            break
    if normal_arg is not None:
        try:
            normal_vals = [float(x) for x in normal_arg.split(',')]
            projection_normal = np.array(normal_vals)
            projection_normal = projection_normal / np.linalg.norm(projection_normal)
            print(f"[DEBUG] (early) Projection normal: {projection_normal}")
        except Exception as e:
            print(f"[DEBUG] (early) Could not parse projection normal: {normal_arg} ({e})")
    solid = build_solid_with_polygons_test(seed=args.seed, quiet=args.quiet)
    print(f"[DEBUG] Solid created: {type(solid)}")
    save_solid_as_step(solid, "solid_output.step")
    # Display original polygons in 3D first
    visualize_3d_solid(solid)
    # Accept comma-separated normal
    normal_vals = [float(x) for x in args.normal.split(',')]
    projection_normal = np.array(normal_vals)
    projection_normal = projection_normal / np.linalg.norm(projection_normal)
    print(f"[DEBUG] Projection normal: {projection_normal}")
    visible, hidden = get_visible_hidden_polygons(solid, projection_normal)
    print(f"[DEBUG] Number of visible polygons: {len(visible)}")
    print(f"[DEBUG] Number of hidden polygons: {len(hidden)}")
    # If no switches, default to combined
    if not (args.show_combined or args.show_visible or args.show_hidden):
        args.show_combined = True
    print("[DEBUG] Plotting four views (top, side, front, isometric)")
    plot_four_views(solid, projection_normal)



if __name__ == "__main__":
    print("Running V6_current.py script...")
    main()

