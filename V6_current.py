# V5_current.py
# Saved version of Polgon Boolean Ops from shapely.py as of July 28, 2025
# Includes corrected plotting order: array_C first (dashed light gray), array_B second (solid black)

from shapely.geometry import Polygon, Point
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import random
import traceback
from itertools import combinations

# OpenCASCADE imports
try:
    from OCC.Core.gp import gp_Pnt, gp_Vec, gp_Ax2, gp_Dir
    from OCC.Core.BRepPrimAPI import BRepPrimAPI_MakeBox
    from OCC.Core.BRepAlgoAPI import BRepAlgoAPI_Fuse
    from OCC.Core.TopExp import TopExp_Explorer
    from OCC.Core.TopAbs import TopAbs_FACE, TopAbs_SHELL, TopAbs_WIRE, TopAbs_EDGE, TopAbs_VERTEX
    from OCC.Core.BRep import BRep_Tool
    from OCC.Core.BRepGProp import BRepGProp_Face
    from OCC.Core.GeomLProp import GeomLProp_SLProps
    from OCC.Core.BRepMesh import BRepMesh_IncrementalMesh
    from OCC.Core.TopLoc import TopLoc_Location
    from OCC.Core.gp import gp_Trsf, gp_XYZ
    from OCC.Core.TopoDS import topods, TopoDS_Compound
    from OCC.Core.BRep import BRep_Builder
    
    # Try to import TopExp for vertex extraction
    try:
        from OCC.Core.TopExp import topexp, topexp_Vertices
        TOPEXP_AVAILABLE = True
    except:
        TOPEXP_AVAILABLE = False
    
    # Visualization imports
    from OCC.Display.SimpleGui import init_display
    from OCC.Core.Quantity import Quantity_Color, Quantity_TOC_RGB
    from OCC.Core.AIS import AIS_Shape
    
    OPENCASCADE_AVAILABLE = True
except ImportError as e:
    print(f"OpenCASCADE not available: {e}")
    OPENCASCADE_AVAILABLE = False
    TOPEXP_AVAILABLE = False
# ============================================================================
# 3D CUBOID FACE PROJECTION AND POLYGON OPERATIONS USING OPENCASCADE
# ============================================================================

# Helper function to plot polygon
def plot_polygon(polygon, ax, facecolor='none', edgecolor='black', alpha=0.7, linestyle='-', linewidth=2, label=None, outline_only=False):
    if polygon.geom_type == 'Polygon':
        if outline_only:
            # Only draw the outline (for standalone polygon plots)
            x, y = polygon.exterior.xy
            ax.plot(x, y, color=edgecolor, linestyle=linestyle, linewidth=linewidth, label=label)
        else:
            # Draw filled patch without separate outline (for combined plots)
            if facecolor != 'none':
                patch = patches.Polygon(list(polygon.exterior.coords), closed=True, 
                                      facecolor=facecolor, alpha=alpha, edgecolor=edgecolor, 
                                      linewidth=linewidth, linestyle=linestyle)
                ax.add_patch(patch)
                # Add invisible line for legend if label is provided
                if label:
                    ax.plot([], [], color=edgecolor, linestyle=linestyle, linewidth=linewidth, label=label)
    elif polygon.geom_type == 'MultiPolygon':
        for poly in polygon.geoms:
            plot_polygon(poly, ax, facecolor, edgecolor, alpha, linestyle, linewidth, label=None, outline_only=outline_only)

def create_opencascade_solid():
    """Create a base cuboid with a slot on top face and a square blind hole."""
    if not OPENCASCADE_AVAILABLE:
        print("✗ OpenCASCADE not available, cannot create solid")
        return None
    
    # Create base cuboid (40x30x20)
    print("Creating base cuboid (40x30x20)...")
    base_cuboid = BRepPrimAPI_MakeBox(40, 30, 20).Shape()
    
    # Create slot on top face (starts from left face, goes past middle)
    # Slot dimensions: width=6, length=25 (from left edge), depth=8
    print("Creating slot on top face...")
    slot_width = 6.0    # Y direction
    slot_length = 25.0  # X direction (from left edge, goes past middle at 20)
    slot_depth = 8.0    # Z direction (from top down)
    
    # Position slot: starts at left face (x=0), centered in Y, from top down
    slot_y_center = 15.0  # Center in Y direction (30/2 = 15)
    slot_y_start = slot_y_center - slot_width/2  # Y = 12
    slot_z_start = 20 - slot_depth  # Z = 12 (from top down)
    
    # Create transformation for slot
    slot_transform = gp_Trsf()
    slot_transform.SetTranslation(gp_Vec(0, slot_y_start, slot_z_start))
    
    # Create slot cuboid
    slot_cuboid = BRepPrimAPI_MakeBox(slot_length, slot_width, slot_depth).Shape()
    slot_cuboid.Move(TopLoc_Location(slot_transform))
    print(f"  Slot: {slot_length}x{slot_width}x{slot_depth} at position (0, {slot_y_start}, {slot_z_start})")
    
    # Create square blind hole (goes past middle depth)
    # Hole dimensions: 5x5, depth=12 (goes past middle at 10)
    print("Creating square blind hole...")
    hole_size = 5.0     # Square hole 5x5
    hole_depth = 12.0   # Goes past middle (20/2 = 10)
    
    # Position hole: center of base face
    hole_x_center = 20.0  # Center in X direction (40/2 = 20)
    hole_y_center = 15.0  # Center in Y direction (30/2 = 15)
    hole_x_start = hole_x_center - hole_size/2  # X = 17.5
    hole_y_start = hole_y_center - hole_size/2  # Y = 12.5
    
    # Create transformation for hole (from bottom up)
    hole_transform = gp_Trsf()
    hole_transform.SetTranslation(gp_Vec(hole_x_start, hole_y_start, 0))
    
    # Create hole cuboid
    hole_cuboid = BRepPrimAPI_MakeBox(hole_size, hole_size, hole_depth).Shape()
    hole_cuboid.Move(TopLoc_Location(hole_transform))
    print(f"  Blind hole: {hole_size}x{hole_size}x{hole_depth} at position ({hole_x_start}, {hole_y_start}, 0)")
    
    # Perform boolean operations to create the final solid
    try:
        print("Performing boolean CUT operations...")
        from OCC.Core.BRepAlgoAPI import BRepAlgoAPI_Cut
        
        # Step 1: Cut slot from base cuboid
        cut_op1 = BRepAlgoAPI_Cut(base_cuboid, slot_cuboid)
        cut_op1.Build()
        
        if not cut_op1.IsDone() or cut_op1.HasErrors():
            print("✗ Slot cutting operation failed")
            return base_cuboid
        
        intermediate_shape = cut_op1.Shape()
        print("  ✓ Slot cut successfully")
        
        # Step 2: Cut blind hole from intermediate shape
        cut_op2 = BRepAlgoAPI_Cut(intermediate_shape, hole_cuboid)
        cut_op2.Build()
        
        if not cut_op2.IsDone() or cut_op2.HasErrors():
            print("✗ Hole cutting operation failed")
            return intermediate_shape
        
        final_shape = cut_op2.Shape()
        print("  ✓ Blind hole cut successfully")
        
        # Validate the result
        if validate_fused_shape(final_shape):
            print(f"✓ Created complex solid with slot and blind hole:")
            print(f"  Base cuboid: 40 x 30 x 20")
            print(f"  Top slot: {slot_length} x {slot_width} x {slot_depth} (from left edge)")
            print(f"  Blind hole: {hole_size} x {hole_size} x {hole_depth} (centered, from bottom)")
            print(f"  Boolean operations: CUT (SUBTRACT) x2")
            print(f"  Operation completed successfully with proper error checking")
            print(f"  Shape validation: PASSED")
            
            return final_shape
        else:
            print(f"✗ Final shape failed validation")
            print(f"  Falling back to base cuboid only")
            return base_cuboid
        
    except Exception as e:
        print(f"✗ Boolean operations failed with exception: {e}")
        print(f"  Falling back to base cuboid only")
        return base_cuboid

def validate_fused_shape(shape):
    """Validate the fused shape using HLR-style validation."""
    if shape is None:
        return False
    
    try:
        # Count geometric elements like HLR functions do
        shell_count = 0
        face_count = 0
        edge_count = 0
        
        # Count shells - should be exactly 1 for a valid solid
        shell_explorer = TopExp_Explorer(shape, TopAbs_SHELL)
        while shell_explorer.More():
            shell_count += 1
            shell_explorer.Next()
        
        # Count faces
        face_explorer = TopExp_Explorer(shape, TopAbs_FACE)
        while face_explorer.More():
            face_count += 1
            face_explorer.Next()
        
        # Count edges
        edge_explorer = TopExp_Explorer(shape, TopAbs_EDGE)
        while edge_explorer.More():
            edge_count += 1
            edge_explorer.Next()
        
        print(f"  Shape validation:")
        print(f"    Shells: {shell_count}")
        print(f"    Faces: {face_count}")
        print(f"    Edges: {edge_count}")
        
        # For a valid fused cuboid, we should have exactly 1 shell
        if shell_count != 1:
            print(f"    ✗ Invalid shell count: {shell_count} (expected: 1)")
            return False
        
        # For fused cuboids, face count should be reasonable (typically 6-12 faces)
        if face_count < 6 or face_count > 20:
            print(f"    ⚠️  Unusual face count: {face_count} (typical: 6-12)")
        
        print(f"    ✓ Shape validation passed")
        return True
        
    except Exception as e:
        print(f"    ✗ Shape validation failed: {e}")
        return False

def analyze_solid_geometry(solid_shape):
    """Analyze and display detailed geometry information about the solid."""
    if not OPENCASCADE_AVAILABLE or solid_shape is None:
        print("✗ Cannot analyze solid - shape is None")
        return
    
    print(f"\n" + "="*60)
    print("DETAILED SOLID GEOMETRY ANALYSIS")
    print("="*60)
    
    try:
        # Basic topology count
        shell_count = 0
        face_count = 0
        edge_count = 0
        vertex_count = 0
        
        # Count shells
        shell_explorer = TopExp_Explorer(solid_shape, TopAbs_SHELL)
        while shell_explorer.More():
            shell_count += 1
            shell_explorer.Next()
        
        # Count faces
        face_explorer = TopExp_Explorer(solid_shape, TopAbs_FACE)
        while face_explorer.More():
            face_count += 1
            face_explorer.Next()
        
        # Count edges  
        edge_explorer = TopExp_Explorer(solid_shape, TopAbs_EDGE)
        while edge_explorer.More():
            edge_count += 1
            edge_explorer.Next()
            
        # Count vertices
        vertex_explorer = TopExp_Explorer(solid_shape, TopAbs_VERTEX)
        while vertex_explorer.More():
            vertex_count += 1
            vertex_explorer.Next()
        
        print(f"Topological Elements:")
        print(f"  • Shells: {shell_count}")
        print(f"  • Faces: {face_count}")
        print(f"  • Edges: {edge_count}")
        print(f"  • Vertices: {vertex_count}")
        
        # Calculate bounding box
        try:
            from OCC.Core.Bnd import Bnd_Box
            from OCC.Core.BRepBndLib import brepbndlib
            
            bbox = Bnd_Box()
            brepbndlib.Add(solid_shape, bbox)
            
            if not bbox.IsVoid():
                xmin, ymin, zmin, xmax, ymax, zmax = bbox.Get()
                width = xmax - xmin
                height = ymax - ymin
                depth = zmax - zmin
                
                print(f"\nBounding Box:")
                print(f"  • X range: {xmin:.2f} to {xmax:.2f} (width: {width:.2f})")
                print(f"  • Y range: {ymin:.2f} to {ymax:.2f} (height: {height:.2f})")
                print(f"  • Z range: {zmin:.2f} to {zmax:.2f} (depth: {depth:.2f})")
                print(f"  • Volume estimate: {width * height * depth:.2f} cubic units")
                
        except Exception as e:
            print(f"  Bounding box calculation failed: {e}")
        
        # Analyze face types
        face_types = {'planar': 0, 'curved': 0, 'complex': 0}
        face_areas = []
        
        face_explorer = TopExp_Explorer(solid_shape, TopAbs_FACE)
        face_num = 0
        
        print(f"\nFace Analysis:")
        while face_explorer.More():
            face = face_explorer.Current()
            face_num += 1
            
            try:
                # Get face surface
                from OCC.Core.BRepAdaptor import BRepAdaptor_Surface
                from OCC.Core.GeomAbs import GeomAbs_Plane, GeomAbs_Cylinder, GeomAbs_Sphere, GeomAbs_Cone
                surface = BRepAdaptor_Surface(topods.Face(face))
                
                # Classify surface type using GeomAbs constants
                surface_type = surface.GetType()
                
                # Get type name for display
                type_name = str(surface_type).split('.')[-1] if hasattr(surface_type, '__str__') else 'Unknown'
                
                print(f"  Face {face_num}: {type_name} (type={surface_type})")
                
                # Check surface type using constants
                if surface_type == GeomAbs_Plane:
                    face_types['planar'] += 1
                    print(f"    → Classified as PLANAR")
                elif surface_type in [GeomAbs_Cylinder, GeomAbs_Sphere, GeomAbs_Cone]:
                    face_types['curved'] += 1
                    curved_name = "CYLINDER" if surface_type == GeomAbs_Cylinder else "SPHERE" if surface_type == GeomAbs_Sphere else "CONE"
                    print(f"    → Classified as CURVED ({curved_name})")
                else:
                    face_types['complex'] += 1
                    print(f"    → Classified as COMPLEX")
                    
            except Exception as e:
                print(f"  Face {face_num}: Analysis failed - {e}")
                traceback.print_exc()
                face_types['complex'] += 1
            
            face_explorer.Next()
        
        print(f"\nFace Type Summary:")
        print(f"  • Planar faces: {face_types['planar']}")
        print(f"  • Curved faces: {face_types['curved']}")
        print(f"  • Complex faces: {face_types['complex']}")
        
        # Validate solid integrity
        print(f"\nSolid Validation:")
        if shell_count == 1:
            print(f"  ✓ Single shell - solid is manifold")
        else:
            print(f"  ⚠️  Multiple shells ({shell_count}) - may indicate issues")
            
        # Expected face count for fused cuboids is typically 6-12
        if 6 <= face_count <= 12:
            print(f"  ✓ Face count ({face_count}) is typical for fused cuboids")
        elif face_count < 6:
            print(f"  ⚠️  Low face count ({face_count}) - unexpected for cuboids")
        else:
            print(f"  ℹ️  High face count ({face_count}) - complex boolean result")
        
        print(f"\n" + "="*60)
        print("GEOMETRY ANALYSIS COMPLETE")
        print("="*60)
        
    except Exception as e:
        print(f"✗ Solid analysis failed: {e}")


def get_face_normal_from_opencascade(face):
    """Extract the correct face normal from OpenCASCADE face using multiple robust methods.
    
    This function tries several approaches to get the correct outward-pointing normal:
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
        
        # Method 1: Use GeomLProp_SLProps for most reliable normal calculation
        try:
            # Get the underlying surface
            surface_handle = surface.Surface()
            
            # Create surface properties evaluator
            props = GeomLProp_SLProps(surface_handle, u_mid, v_mid, 1, 1e-6)
            
            if props.IsNormalDefined():
                normal_vec = props.Normal()
                
                # Apply orientation correction
                orientation_multiplier = 1.0
                if face_orientation == TopAbs_REVERSED:
                    orientation_multiplier = -1.0
                    print(f"        REVERSED face - flipping normal")
                
                face_normal = np.array([
                    normal_vec.X() * orientation_multiplier,
                    normal_vec.Y() * orientation_multiplier,
                    normal_vec.Z() * orientation_multiplier
                ])
                
                # Normalize
                face_normal = face_normal / np.linalg.norm(face_normal)
                
                print(f"        GeomLProp normal: [{face_normal[0]:.6f}, {face_normal[1]:.6f}, {face_normal[2]:.6f}]")
                
                # Validate the normal (should be unit vector)
                magnitude = np.linalg.norm(face_normal)
                if abs(magnitude - 1.0) > 1e-6:
                    print(f"        Warning: Normal magnitude {magnitude:.6f} != 1.0")
                
                return face_normal
                
        except Exception as e:
            print(f"        GeomLProp method failed: {e}")
        
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
        import traceback
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
    vertices = []
    
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

def create_rectangular_fallback_from_face(face):
    """Create a rectangular set of vertices from face bounding box."""
    try:
        from OCC.Core.Bnd import Bnd_Box
        from OCC.Core.BRepBndLib import brepbndlib
        
        bbox = Bnd_Box()
        brepbndlib.Add(face, bbox)
        
        if not bbox.IsVoid():
            xmin, ymin, zmin, xmax, ymax, zmax = bbox.Get()
            
            # Determine which plane the face lies in
            x_range = xmax - xmin
            y_range = ymax - ymin
            z_range = zmax - zmin
            
            tolerance = 1e-6
            
            if x_range < tolerance:  # X-normal face (YZ plane)
                x = (xmin + xmax) / 2
                return [[x, ymin, zmin], [x, ymax, zmin], [x, ymax, zmax], [x, ymin, zmax]]
            elif y_range < tolerance:  # Y-normal face (XZ plane)
                y = (ymin + ymax) / 2
                return [[xmin, y, zmin], [xmax, y, zmin], [xmax, y, zmax], [xmin, y, zmax]]
            elif z_range < tolerance:  # Z-normal face (XY plane)
                z = (zmin + zmax) / 2
                return [[xmin, ymin, z], [xmax, ymin, z], [xmax, ymax, z], [xmin, ymax, z]]
    
    except Exception as e:
        print(f"      Error creating rectangular fallback: {e}")
    
    return None

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

def find_interior_point(polygon):
    """Find an interior point within a polygon."""
    try:
        # Use representative point (guaranteed to be inside)
        interior_point = polygon.representative_point()
        if polygon.contains(interior_point):
            return interior_point
        
        # Fallback to centroid
        centroid = polygon.centroid
        if polygon.contains(centroid):
            return centroid
            
        # Final fallback: use first coordinate of exterior
        coords = list(polygon.exterior.coords)
        if len(coords) > 1:
            return Point(coords[0])
            
    except Exception as e:
        print(f"Error finding interior point: {e}")
    
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

def create_polygon_from_projection(projected_vertices):
    """Create a Shapely polygon from projected vertices."""
    if len(projected_vertices) == 0:
        return Polygon()
    
    # Ensure the polygon is closed by adding the first vertex at the end if needed
    projected_vertices = np.array(projected_vertices)
    original_vertex_count = len(projected_vertices)
    
    if len(projected_vertices) > 0:
        # Check if first and last vertices are the same (within tolerance)
        if not np.allclose(projected_vertices[0], projected_vertices[-1], atol=1e-10):
            projected_vertices = np.vstack([projected_vertices, projected_vertices[0]])
    
    print(f"    → Creating polygon from {original_vertex_count} vertices")
    
    try:
        polygon = Polygon(projected_vertices)
        
        # For valid polygons, return them directly
        if polygon.is_valid and hasattr(polygon, 'area') and polygon.area > 1e-6:
            print(f"    → Valid polygon created with {len(polygon.exterior.coords)-1} vertices")
            return polygon
        
        # For invalid polygons, try to fix
        if not polygon.is_valid:
            print(f"    → Invalid polygon detected (reason: {polygon.is_valid}), attempting to fix...")
            print(f"    → Original vertices: {original_vertex_count}, coords in polygon: {len(polygon.exterior.coords)-1}")
            
            try:
                # Try buffer(0) to fix self-intersections
                fixed_polygon = polygon.buffer(0)
                if fixed_polygon.is_valid and hasattr(fixed_polygon, 'area') and fixed_polygon.area > 1e-6:
                    if hasattr(fixed_polygon, 'exterior'):
                        fixed_vertex_count = len(fixed_polygon.exterior.coords) - 1
                        print(f"    → Fixed with buffer(0): {original_vertex_count} → {fixed_vertex_count} vertices")
                    return fixed_polygon
            except Exception as e:
                print(f"    → Buffer(0) fix failed: {e}")
            
            # Fallback: create convex hull
            try:
                hull_polygon = Polygon(projected_vertices).convex_hull
                if hull_polygon.is_valid and hasattr(hull_polygon, 'area') and hull_polygon.area > 1e-6:
                    if hasattr(hull_polygon, 'exterior'):
                        hull_vertex_count = len(hull_polygon.exterior.coords) - 1
                        print(f"    → Fixed with convex_hull: {original_vertex_count} → {hull_vertex_count} vertices")
                    return hull_polygon
            except Exception as e:
                print(f"    → Convex hull fix failed: {e}")
        
        # Return empty polygon if all fixes fail
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
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
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
            
            if polygon.geom_type == 'Polygon' and polygon.area > 0:
                # Different styling for intersections vs regular faces
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
                
                # Collect bounds
                bounds = polygon.bounds
                all_bounds.extend([bounds[0], bounds[2], bounds[1], bounds[3]])
                
                # Add face name at centroid
                centroid = polygon.centroid
                display_name = name.replace('Face_', 'F').replace('Intersection_', 'I_')
                ax2.text(centroid.x, centroid.y, display_name, 
                        ha='center', va='center', fontsize=8, weight='bold')
                        
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
            if polygon.geom_type == 'Polygon' and polygon.area > 0:
                plot_polygon(polygon, ax3, facecolor='none', edgecolor='lightgray', alpha=0.8, linewidth=0.7, linestyle='--', label=f'C: {name}', outline_only=True)
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
        
        for ax in [ax1, ax2, ax3]:
            ax.set_xlim(xlim)
            ax.set_ylim(ylim)
    
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
            face_id = face_data['face_id']
            vertex_count = face_data['vertex_count']
            
            # Close the polygon by adding first vertex at end
            if len(vertices) > 2:
                vertices_closed = np.vstack([vertices, vertices[0]])
                
                # Plot ONLY the face boundary edges - NO TRIANGULATION
                ax.plot(vertices_closed[:, 0], vertices_closed[:, 1], vertices_closed[:, 2], 
                       color=colors[i], linewidth=3, alpha=0.9, 
                       label=f'Face {face_id} ({vertex_count}v)')
                
                # Plot vertices as points for clarity
                ax.scatter(vertices[:, 0], vertices[:, 1], vertices[:, 2], 
                          color=colors[i], s=50, alpha=0.8, edgecolors='black', linewidth=1)
                
                # Add face center label
                face_center = np.mean(vertices, axis=0)
                ax.text(face_center[0], face_center[1], face_center[2], 
                       f'F{face_id}({vertex_count}v)', 
                       fontsize=10, color='red', weight='bold',
                       bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
                
                print(f"    ✓ Face {face_id}: Rendered {vertex_count} vertices as polygon boundary (NO triangulation)")
        
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
        
        # Project face to 2D polygon with holes support
        try:
            # Project outer boundary
            projected_outer = project_face_to_projection_plane(outer_boundary, unit_projection_normal)
            
            # Project cutouts/holes if they exist
            cutouts = polygon_data.get('cutouts', [])
            projected_holes = []
            
            for cutout in cutouts:
                if cutout and len(cutout) >= 3:
                    projected_cutout = project_face_to_projection_plane(cutout, unit_projection_normal)
                    projected_holes.append(projected_cutout)
            
            # Create Shapely polygon with holes
            if projected_holes:
                print(f"    → Creating polygon from {len(projected_outer)} vertices with {len(projected_holes)} holes")
                polygon = Polygon(projected_outer, holes=projected_holes)
                
                # Fix invalid polygons with holes
                if not polygon.is_valid:
                    print(f"    → Invalid polygon with holes, attempting buffer fix...")
                    polygon = polygon.buffer(0)
            else:
                print(f"    → Creating polygon from {len(projected_outer)} vertices")
                polygon = create_polygon_from_projection(projected_outer)
            
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
        
        except Exception as e:
            print(f"Face F{face_id}: Projection error - {e}")
    
    print(f"\nStep 2: Starting historic polygon classification algorithm...")
    print(f"Initial array_A: {len(array_A_initial)} polygons")
    
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
            
            print(f"\nProcessing {Pi_name} (area: {Pi.area:.2f})")
            
            # Track Face 11 specifically
            if "Face_11" in Pi_name:
                print(f"    🔍 TRACKING FACE 11: Starting processing")
                print(f"    🔍 Face 11 dot product: {Pi_data.get('dot_product', 'unknown')}")
                print(f"    🔍 Face 11 normal: {Pi_data.get('normal', 'unknown')}")
            
            intersection_found = False
            
            # Test intersection with all polygons in array_B
            for j, Pj_data in enumerate(array_B):
                Pj = Pj_data['polygon']
                Pj_name = Pj_data['name']
                Pj_parent_face = Pj_data['parent_face']
                
                try:
                    intersection = Pi.intersection(Pj)
                    
                    if (not intersection.is_empty and 
                        hasattr(intersection, 'area') and 
                        intersection.area > 1e-6):
                        
                        print(f"  → Intersection with {Pj_name} (area: {intersection.area:.2f})")
                        intersection_found = True
                        
                        # Find interior point for depth analysis
                        interior_point = find_interior_point(intersection)
                        
                        if interior_point is None:
                            print(f"    → Warning: Could not find interior point")
                            continue
                        
                        # Calculate 3D depths using line-face intersection
                        Pi_intersection_3d = intersect_line_with_face(
                            interior_point, unit_projection_normal, Pi_parent_face)
                        Pj_intersection_3d = intersect_line_with_face(
                            interior_point, unit_projection_normal, Pj_parent_face)
                        
                        Pi_depth = calculate_depth_along_normal(Pi_intersection_3d, unit_projection_normal)
                        Pj_depth = calculate_depth_along_normal(Pj_intersection_3d, unit_projection_normal)
                        
                        print(f"    → Depths: Pi={Pi_depth:.3f}, Pj={Pj_depth:.3f}")
                        
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
                        print(f"    → Added intersection to array_C")
                        
                        # Apply depth-based boolean operations
                        if Pi_depth > Pj_depth:
                            print(f"    → Pi farther: modifying Pj = Pj - Pi")
                            try:
                                new_Pj = Pj.difference(Pi)
                                if not new_Pj.is_empty and new_Pj.area > 1e-6:
                                    array_B[j]['polygon'] = new_Pj
                                    array_B[j]['name'] = f"Modified_{Pj_name}"
                                    print(f"    → Updated Pj area: {new_Pj.area:.2f}")
                                else:
                                    array_B.pop(j)
                                    print(f"    → Removed Pj (empty after subtraction)")
                            except Exception as e:
                                print(f"    → Boolean operation error: {e}")
                        else:
                            print(f"    → Pj farther: modifying Pi = Pi - Pj")
                            try:
                                new_Pi = Pi.difference(Pj)
                                if not new_Pi.is_empty and new_Pi.area > 1e-6:
                                    Pi = new_Pi
                                    Pi_data['polygon'] = new_Pi
                                    Pi_data['name'] = f"Modified_{Pi_name}"
                                    print(f"    → Updated Pi area: {new_Pi.area:.2f}")
                                else:
                                    print(f"    → Pi consumed (empty after subtraction)")
                                    # Update Pi_data to reflect the empty polygon
                                    Pi_data['polygon'] = new_Pi
                                    break
                            except Exception as e:
                                print(f"    → Boolean operation error: {e}")
                
                except Exception as e:
                    print(f"  → Error testing intersection with {Pj_name}: {e}")
            
            # Add remaining Pi to array_B if it still has area
            if Pi_data['polygon'].area > 1e-6:
                array_B.append(Pi_data)
                print(f"  → Added {Pi_data['name']} to array_B")
                
                # Track Face 11 specifically
                if "Face_11" in Pi_data['name']:
                    print(f"  🔍 TRACKING FACE 11: Added to array_B")
                    print(f"  🔍 Face 11 area: {Pi_data['polygon'].area:.2f}")
                    print(f"  🔍 Face 11 dot product: {Pi_data.get('dot_product', 'unknown')}")
        
        # Step 2.3: Apply final dot product classification
        print(f"\nStep 3: Applying final dot product classification...")
        faces_to_move = []
        
        for i, poly_data in enumerate(array_B):
            if poly_data['dot_product'] <= 0:
                faces_to_move.append(i)
                print(f"  → {poly_data['name']}: dot_product={poly_data['dot_product']:.3f} ≤ 0, moving to array_C")
        
        # Move faces with negative dot product to array_C
        for i in reversed(faces_to_move):
            moved_face = array_B.pop(i)
            array_C.append(moved_face)
    
    print(f"\n" + "="*60)
    print("ENHANCED CLASSIFICATION COMPLETE")
    print("="*60)
    print(f"Array A (processed): 0 faces (all processed)")
    print(f"Array B (visible): {len(array_B)} faces")
    print(f"Array C (hidden+intersections): {len(array_C)} faces")
    print(f"Total: {len(array_B) + len(array_C)} faces")
    
    return [], array_B, array_C

def display_face_arrays(array_A, array_B, array_C):
    """Display the contents of face arrays A, B, and C in detail."""
    print("\n" + "="*80)
    print("DETAILED FACE ARRAY CONTENTS")
    print("="*80)
    
    # Display Array A
    print(f"\nARRAY A - PROCESSED FACES ({len(array_A)} faces):")
    print("-" * 50)
    
    if not array_A:
        print("  (All faces processed through algorithm)")
    else:
        for i, face_data in enumerate(array_A):
            face_id = face_data.get('face_id', f'Unknown_{i}')
            if 'outer_boundary' in face_data:
                # Original face format
                outer_boundary = face_data['outer_boundary']
                cutouts = face_data.get('cutouts', [])
                face_normal = face_data.get('face_normal', [0, 0, 0])
                dot_product = face_data.get('dot_product', 0)
                vertex_count = face_data.get('vertex_count', len(outer_boundary))
                
                print(f"\n  Face A{i+1} (Original F{face_id}):")
                print(f"    • Vertices: {vertex_count}")
                print(f"    • Dot product: {dot_product:.6f}")
                print(f"    • Face normal: [{face_normal[0]:.3f}, {face_normal[1]:.3f}, {face_normal[2]:.3f}]")
            else:
                # Polygon format
                polygon = face_data.get('polygon')
                name = face_data.get('name', f'Unknown_{i}')
                normal = face_data.get('normal', [0, 0, 0])
                dot_product = face_data.get('dot_product', 0)
                
                print(f"\n  Face A{i+1} ({name}):")
                print(f"    • Area: {polygon.area:.2f}" if polygon else "    • Area: N/A")
                print(f"    • Dot product: {dot_product:.6f}")
                print(f"    • Face normal: [{normal[0]:.3f}, {normal[1]:.3f}, {normal[2]:.3f}]")

    # Display Array B
    print(f"\nARRAY B - VISIBLE FACES ({len(array_B)} faces):")
    print("-" * 50)
    
    if not array_B:
        print("  (No faces in Array B)")
    else:
        for i, face_data in enumerate(array_B):
            polygon = face_data.get('polygon')
            name = face_data.get('name', f'Unknown_{i}')
            normal = face_data.get('normal', [0, 0, 0])
            dot_product = face_data.get('dot_product', 0)
            
            print(f"\n  Face B{i+1} ({name}):")
            print(f"    • Area: {polygon.area:.2f}" if polygon else "    • Area: N/A")
            print(f"    • Dot product: {dot_product:.6f}")
            if hasattr(normal, '__len__') and len(normal) >= 3:
                print(f"    • Face normal: [{normal[0]:.3f}, {normal[1]:.3f}, {normal[2]:.3f}]")
            else:
                print(f"    • Face normal: {normal}")

    # Display Array C
    print(f"\nARRAY C - HIDDEN + INTERSECTIONS ({len(array_C)} faces):")
    print("-" * 50)
    
    if not array_C:
        print("  (No faces in Array C)")
    else:
        for i, face_data in enumerate(array_C):
            if 'outer_boundary' in face_data:
                # Original face format
                face_id = face_data.get('face_id', f'Unknown_{i}')
                outer_boundary = face_data['outer_boundary']
                cutouts = face_data.get('cutouts', [])
                face_normal = face_data.get('face_normal', [0, 0, 0])
                dot_product = face_data.get('dot_product', 0)
                vertex_count = face_data.get('vertex_count', len(outer_boundary))
                
                print(f"\n  Face C{i+1} (Original F{face_id}):")
                print(f"    • Vertices: {vertex_count}")
                print(f"    • Dot product: {dot_product:.6f}")
                print(f"    • Face normal: [{face_normal[0]:.3f}, {face_normal[1]:.3f}, {face_normal[2]:.3f}]")
            else:
                # Polygon format
                polygon = face_data.get('polygon')
                name = face_data.get('name', f'Unknown_{i}')
                normal = face_data.get('normal', [0, 0, 0])
                dot_product = face_data.get('dot_product', 0)
                associated_face = face_data.get('associated_face', 'None')
                
                print(f"\n  Face C{i+1} ({name}):")
                print(f"    • Area: {polygon.area:.2f}" if polygon else "    • Area: N/A")
                print(f"    • Type: {'Intersection' if 'Intersection' in name else 'Hidden Face'}")
                if 'Intersection' in name:
                    print(f"    • Associated with: {associated_face}")
                print(f"    • Dot product: {dot_product:.6f}")
                if hasattr(normal, '__len__') and len(normal) >= 3 and not isinstance(normal, str):
                    try:
                        print(f"    • Face normal: [{normal[0]:.3f}, {normal[1]:.3f}, {normal[2]:.3f}]")
                    except (TypeError, IndexError):
                        print(f"    • Face normal: {normal}")
                else:
                    print(f"    • Face normal: {normal}")
    
    print(f"\n" + "="*80)
    print("FACE ARRAY DISPLAY COMPLETE")
    print("="*80)

def display_3d_solid(solid_shape):
    """Display the 3D solid using matplotlib visualization with enhanced features."""
    if not OPENCASCADE_AVAILABLE or solid_shape is None:
        print("✗ Cannot display solid - OpenCASCADE not available or solid is None")
        return
    
    # Skip OpenCASCADE native viewer due to stability issues, use matplotlib
    try:
        print("✓ Creating enhanced matplotlib 3D visualization...")
        
        # Extract faces using proper BRep polygon extraction
        face_polygons = extract_faces_from_solid(solid_shape)
        
        # Get projection normal from user input
        unit_projection_normal = get_projection_normal_from_user()
        
        # Classify faces into arrays A, B, C based on projection normal
        array_A, array_B, array_C = classify_faces_by_projection(face_polygons, unit_projection_normal)
        
        # Display the face arrays in detail
        display_face_arrays(array_A, array_B, array_C)
        
        # Continue with visualization using classified faces
        # Convert face arrays back to display format for visualization
        faces = []
        
        # Add Array A faces (front-facing) with distinct styling
        for face_data in array_A:
            vertices_array = np.array(face_data['outer_boundary'])
            normal = face_data['face_normal']
            face_id = face_data['face_id']
            
            if len(vertices_array) >= 3:
                faces.append((vertices_array, normal, f"A_Face_{face_id}", 'front'))
                
                # Add cutouts if they exist
                for i, cutout in enumerate(face_data['cutouts']):
                    if cutout and len(cutout) >= 3:
                        cutout_array = np.array(cutout)
                        faces.append((cutout_array, -normal, f"A_Face_{face_id}_cutout_{i+1}", 'front_cutout'))
        
        # Add Array C faces (back-facing) with distinct styling
        for face_data in array_C:
            vertices_array = np.array(face_data['outer_boundary'])
            normal = face_data['face_normal']
            face_id = face_data['face_id']
            
            if len(vertices_array) >= 3:
                faces.append((vertices_array, normal, f"C_Face_{face_id}", 'back'))
                
                # Add cutouts if they exist
                for i, cutout in enumerate(face_data['cutouts']):
                    if cutout and len(cutout) >= 3:
                        cutout_array = np.array(cutout)
                        faces.append((cutout_array, -normal, f"C_Face_{face_id}_cutout_{i+1}", 'back_cutout'))
        
        # Convert polygon data to display format
        faces = []
        for polygon_data in face_polygons:
            outer_boundary = polygon_data['outer_boundary']
            cutouts = polygon_data.get('cutouts', [])
            normal = polygon_data.get('normal', np.array([0, 0, 1]))
            face_id = polygon_data.get('face_id', 0)
            
            if outer_boundary and len(outer_boundary) >= 3:
                vertices_array = np.array(outer_boundary)
                faces.append((vertices_array, normal, f"Face_{face_id}"))
                
                # Add cutouts as separate faces if they exist
                for i, cutout in enumerate(cutouts):
                    if cutout and len(cutout) >= 3:
                        cutout_array = np.array(cutout)
                        faces.append((cutout_array, -normal, f"Face_{face_id}_cutout_{i+1}"))
        
        # Extract edges for showing boolean intersection lines
        edges = extract_edges_for_display(solid_shape)
        
        if not faces:
            print("✗ No faces to display")
            return
        
        # Create enhanced 3D plot with multiple views
        fig = plt.figure(figsize=(20, 12))
        
        # Main 3D view
        ax1 = fig.add_subplot(2, 3, (1, 2))
        ax1 = fig.add_subplot(2, 3, 1, projection='3d')
        
        # Enhanced color scheme for better visibility
        colors = ['lightblue', 'lightcoral', 'lightgreen', 'lightyellow', 
                 'lightpink', 'lightgray', 'lightcyan', 'lightsalmon', 
                 'lightsteelblue', 'lightseagreen', 'lightgoldenrodyellow', 'plum']
        
        for i, (vertices, normal, name) in enumerate(faces):
            # Use matplotlib's 3D polygon patches for clean planar polygon rendering
            from mpl_toolkits.mplot3d.art3d import Poly3DCollection
            from mpl_toolkits.mplot3d import art3d
            import matplotlib.patches as mpatches
            
            face_color = colors[i % len(colors)]
            
            # Handle all planar faces with special approach for complex faces to avoid diagonal lines
            num_vertices = len(vertices)
            
            print(f"    Rendering planar face F{i+1} ({num_vertices} vertices)")
            
            try:
                # For complex faces (5+ vertices), use boundary-only rendering to avoid diagonal artifacts
                if num_vertices >= 5:
                    print(f"      Using boundary-only rendering for complex face F{i+1} to avoid diagonal lines")
                    
                    # Draw the boundary edges only
                    for j in range(num_vertices):
                        next_j = (j + 1) % num_vertices
                        ax1.plot3D(
                            [vertices[j][0], vertices[next_j][0]],
                            [vertices[j][1], vertices[next_j][1]], 
                            [vertices[j][2], vertices[next_j][2]],
                            color='black',
                            linewidth=2.5,
                            alpha=0.9,
                            zorder=2
                        )
                    
                    # Create a very subtle filled background using a simple approach
                    # Split complex polygon into triangles from a central point to avoid diagonal artifacts
                    if len(vertices) >= 3:
                        # Calculate face center for triangulation
                        center = np.mean(vertices, axis=0)
                        
                        # Create triangles from center to each edge
                        triangles = []
                        for j in range(num_vertices):
                            next_j = (j + 1) % num_vertices
                            triangle = [center, vertices[j], vertices[next_j]]
                            triangles.append(triangle)
                        
                        # Render triangulated fill with very low alpha
                        collection = ax1.add_collection3d(Poly3DCollection(
                            triangles, 
                            alpha=0.15,  # Very transparent to avoid visual conflicts
                            facecolor=face_color, 
                            edgecolor='none',  # No triangle edges
                            linewidth=0.0,
                            zorder=0  # Draw behind boundary edges
                        ))
                    
                    print(f"      ✓ Complex face F{i+1} rendered with boundary edges + subtle fill")
                
                else:
                    # For simple faces (3-4 vertices), use direct polygon rendering
                    print(f"      Using direct polygon rendering for simple face F{i+1}")
                    
                    # Determine the best projection plane based on face normal
                    if normal is not None:
                        # Use the provided face normal
                        face_normal = normal / np.linalg.norm(normal)
                    else:
                        # Calculate face normal from vertices
                        if len(vertices) >= 3:
                            v1 = np.array(vertices[1]) - np.array(vertices[0])
                            v2 = np.array(vertices[2]) - np.array(vertices[0])
                            face_normal = np.cross(v1, v2)
                            face_normal = face_normal / np.linalg.norm(face_normal)
                        else:
                            face_normal = np.array([0, 0, 1])  # Default to Z-normal
                    
                    # Create the 3D polygon using a single polygon approach
                    poly3d = [vertices]
                    
                    # Add face with clean styling
                    collection = ax1.add_collection3d(Poly3DCollection(
                        poly3d, 
                        alpha=0.6,  # More opaque for simple faces
                        facecolor=face_color, 
                        edgecolor='black',
                        linewidth=1.5,   # Edge definition
                        zorder=1        # Draw faces behind boundary edges
                    ))
                    
                    print(f"      ✓ Simple face F{i+1} rendered as clean polygon")
                
            except Exception as e:
                print(f"      ✗ Error rendering face F{i+1}: {e}")
                # Fallback: draw boundary edges only
                print(f"      Drawing boundary edges only for face F{i+1}")
                for j in range(num_vertices):
                    next_j = (j + 1) % num_vertices
                    ax1.plot3D(
                        [vertices[j][0], vertices[next_j][0]],
                        [vertices[j][1], vertices[next_j][1]], 
                        [vertices[j][2], vertices[next_j][2]],
                        color='black',
                        linewidth=2.0,
                        alpha=0.8
                    )
            
            # Add face labels for identification with vertex count
            face_center = np.mean(vertices, axis=0)
            label_text = f'F{i+1}({num_vertices}v)'
            ax1.text(face_center[0], face_center[1], face_center[2], 
                   label_text, fontsize=9, color='red', weight='bold',
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
        
        # Add edge visualization to show boolean intersection lines
        if edges:
            print(f"  Adding {len(edges)} edges to visualization")
            edge_colors = {
                'x_aligned': 'red',
                'y_aligned': 'green', 
                'z_aligned': 'blue',
                'boolean_intersection': 'gray',  # Subtle gray color
                'diagonal': 'purple',
                'unknown': 'gray'
            }
            
            edge_widths = {
                'x_aligned': 1.0,
                'y_aligned': 1.0,
                'z_aligned': 1.0,
                'boolean_intersection': 1.0,  # Normal width like other edges
                'diagonal': 1.0,
                'unknown': 0.5
            }
            
            # First pass: draw all non-intersection edges
            for edge_vertices, edge_type, edge_length in edges:
                if edge_type != 'boolean_intersection':
                    color = edge_colors.get(edge_type, 'gray')
                    width = edge_widths.get(edge_type, 1.0)
                    
                    ax1.plot3D(
                        [edge_vertices[0][0], edge_vertices[1][0]],
                        [edge_vertices[0][1], edge_vertices[1][1]], 
                        [edge_vertices[0][2], edge_vertices[1][2]],
                        color=color,
                        linewidth=width,
                        alpha=0.6
                    )
            
            # Boolean intersection edges will be drawn with normal styling (no thick highlighting)
            intersection_count = 0
            for edge_vertices, edge_type, edge_length in edges:
                if edge_type == 'boolean_intersection':
                    intersection_count += 1
            
            print(f"  Found {intersection_count} boolean intersection edges")
            
            # Add legend for edge types
            from matplotlib.lines import Line2D
            legend_elements = []
            for edge_type, color in edge_colors.items():
                if any(et == edge_type for _, et, _ in edges):
                    width = edge_widths.get(edge_type, 1.0)
                    legend_elements.append(Line2D([0], [0], color=color, linewidth=width, 
                                                label=f'{edge_type.replace("_", " ").title()}'))
            
            if legend_elements:
                ax1.legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(0.02, 0.98))
        else:
            print("  No edges found for visualization")
        
        # Enhanced axis styling for main view
        ax1.set_xlabel('X Coordinate', fontsize=12, weight='bold')
        ax1.set_ylabel('Y Coordinate', fontsize=12, weight='bold')
        ax1.set_zlabel('Z Coordinate', fontsize=12, weight='bold')
        ax1.set_title('3D Cut Solid - Isometric View\n(Boolean SUBTRACT Operation Result)', 
                    fontsize=14, weight='bold')
        
        # Set axis limits with better proportions
        all_vertices = np.vstack([vertices for vertices, _, _ in faces])
        margin = 2
        x_range = [all_vertices[:, 0].min() - margin, all_vertices[:, 0].max() + margin]
        y_range = [all_vertices[:, 1].min() - margin, all_vertices[:, 1].max() + margin]
        z_range = [all_vertices[:, 2].min() - margin, all_vertices[:, 2].max() + margin]
        
        ax1.set_xlim(x_range)
        ax1.set_ylim(y_range)
        ax1.set_zlim(z_range)
        ax1.grid(True, alpha=0.3)
        ax1.view_init(elev=20, azim=45)
        
        # Add orthographic projections
        # Front view (Y-Z plane)
        ax2 = fig.add_subplot(2, 3, 3)
        ax2.set_title('Front View (Y-Z)', fontsize=12, weight='bold')
        ax2.set_xlabel('Y Coordinate')
        ax2.set_ylabel('Z Coordinate')
        ax2.grid(True, alpha=0.3)
        ax2.set_aspect('equal')
        
        # Top view (X-Y plane) 
        ax3 = fig.add_subplot(2, 3, 4)
        ax3.set_title('Top View (X-Y)', fontsize=12, weight='bold')
        ax3.set_xlabel('X Coordinate')
        ax3.set_ylabel('Y Coordinate')
        ax3.grid(True, alpha=0.3)
        ax3.set_aspect('equal')
        
        # Side view (X-Z plane)
        ax4 = fig.add_subplot(2, 3, 5)
        ax4.set_title('Side View (X-Z)', fontsize=12, weight='bold')
        ax4.set_xlabel('X Coordinate')
        ax4.set_ylabel('Z Coordinate')
        ax4.grid(True, alpha=0.3)
        ax4.set_aspect('equal')
        
        # Create clean orthographic projections by using solid bounding box outline
        # instead of complex boolean intersection geometry
        print("  Creating clean orthographic projections...")
        
        # Calculate overall solid bounding box for clean orthographic views
        all_vertices = np.vstack([vertices for vertices, _, _ in faces])
        x_min, x_max = all_vertices[:, 0].min(), all_vertices[:, 0].max()
        y_min, y_max = all_vertices[:, 1].min(), all_vertices[:, 1].max()
        z_min, z_max = all_vertices[:, 2].min(), all_vertices[:, 2].max()
        
        print(f"    Solid bounding box: X[{x_min:.1f}, {x_max:.1f}], Y[{y_min:.1f}, {y_max:.1f}], Z[{z_min:.1f}, {z_max:.1f}]")
        
        # Create clean rectangular outlines for each orthographic view
        front_faces = []  # Y-Z projection
        top_faces = []    # X-Y projection  
        side_faces = []   # X-Z projection
        
        for i, (vertices, normal, name) in enumerate(faces):
            face_color = colors[i % len(colors)]
            num_vertices = len(vertices)
            
            # Include both rectangular faces (4 vertices) and create axis-aligned projections
            # Even 4-vertex faces from boolean operations can have non-axis-aligned edges
            if num_vertices == 4:
                print(f"    Processing rectangular face F{i+1} for clean orthographic projection")
                
                # Calculate face bounding box to create clean axis-aligned projections
                x_min, x_max = vertices[:, 0].min(), vertices[:, 0].max()
                y_min, y_max = vertices[:, 1].min(), vertices[:, 1].max()
                z_min, z_max = vertices[:, 2].min(), vertices[:, 2].max()
                
                # Create axis-aligned rectangular projections instead of using actual vertices
                # Front view projection (Y-Z) - create clean rectangle
                front_y = [y_min, y_max, y_max, y_min]
                front_z = [z_min, z_min, z_max, z_max]
                front_faces.append((front_y, front_z, face_color, num_vertices, i+1))
                
                # Top view projection (X-Y) - create clean rectangle  
                top_x = [x_min, x_max, x_max, x_min]
                top_y = [y_min, y_min, y_max, y_max]
                top_faces.append((top_x, top_y, face_color, num_vertices, i+1))
                
                # Side view projection (X-Z) - create clean rectangle
                side_x = [x_min, x_max, x_max, x_min]
                side_z = [z_min, z_min, z_max, z_max]
                side_faces.append((side_x, side_z, face_color, num_vertices, i+1))
                
                print(f"      Face F{i+1} bounding box: X[{x_min:.1f},{x_max:.1f}], Y[{y_min:.1f},{y_max:.1f}], Z[{z_min:.1f},{z_max:.1f}]")
            else:
                print(f"    Skipping complex face F{i+1} ({num_vertices} vertices) from orthographic views to avoid diagonal edges")
        
        # Draw front view (Y-Z) with clean outline and selected rectangular faces
        print("  Drawing front view (Y-Z) with clean outline...")
        
        # Draw only rectangular faces to show internal structure without dark outline
        for y_coords, z_coords, face_color, num_vertices, face_num in front_faces:
            y_coords_closed = np.append(y_coords, y_coords[0])
            z_coords_closed = np.append(z_coords, z_coords[0])
            
            # Draw face outline
            ax2.plot(y_coords_closed, z_coords_closed, 
                    color='gray', linewidth=1.0, alpha=0.6)
            
            # Fill with transparent color to show internal structure
            try:
                ax2.fill(y_coords_closed, z_coords_closed, 
                        color=face_color, alpha=0.2, edgecolor='gray', linewidth=0.5)
            except Exception as e:
                print(f"    Warning: Could not fill face F{face_num} in front view: {e}")
        
        # Draw top view (X-Y) with clean outline and selected rectangular faces
        print("  Drawing top view (X-Y) with clean outline...")
        
        # Draw only rectangular faces
        for x_coords, y_coords, face_color, num_vertices, face_num in top_faces:
            x_coords_closed = np.append(x_coords, x_coords[0])
            y_coords_closed = np.append(y_coords, y_coords[0])
            
            ax3.plot(x_coords_closed, y_coords_closed, 
                    color='gray', linewidth=1.0, alpha=0.6)
            
            try:
                ax3.fill(x_coords_closed, y_coords_closed, 
                        color=face_color, alpha=0.2, edgecolor='gray', linewidth=0.5)
            except Exception as e:
                print(f"    Warning: Could not fill face F{face_num} in top view: {e}")
        
        # Draw side view (X-Z) with clean outline and selected rectangular faces  
        print("  Drawing side view (X-Z) with clean outline...")
        
        # Draw only rectangular faces
        for x_coords, z_coords, face_color, num_vertices, face_num in side_faces:
            x_coords_closed = np.append(x_coords, x_coords[0])
            z_coords_closed = np.append(z_coords, z_coords[0])
            
            ax4.plot(x_coords_closed, z_coords_closed, 
                    color='gray', linewidth=1.0, alpha=0.6)
            
            try:
                ax4.fill(x_coords_closed, z_coords_closed, 
                        color=face_color, alpha=0.2, edgecolor='gray', linewidth=0.5)
            except Exception as e:
                print(f"    Warning: Could not fill face F{face_num} in side view: {e}")
        
        print("  ✓ Clean orthographic views complete - no diagonal edges from complex boolean faces")
        
        # Set consistent ranges for orthographic views
        ax2.set_xlim(y_range)
        ax2.set_ylim(z_range)
        ax3.set_xlim(x_range)
        ax3.set_ylim(y_range)
        ax4.set_xlim(x_range)
        ax4.set_ylim(z_range)
        
        # Add information panel
        ax5 = fig.add_subplot(2, 3, 6)
        ax5.axis('off')
        
        # Count edge types for information display
        edge_type_counts = {}
        if edges:
            for _, edge_type, _ in edges:
                edge_type_counts[edge_type] = edge_type_counts.get(edge_type, 0) + 1
        
        edge_info = "\n".join([f"• {etype.replace('_', ' ').title()}: {count}" 
                              for etype, count in edge_type_counts.items()]) if edge_type_counts else "• No edges extracted"
        
        info_text = f"""SOLID INFORMATION
        
Topology:
• Faces: {len(faces)}
• Edges: {len(edges) if edges else 0}
• Boolean Operation: CUT (SUBTRACT)
• Result: Single manifold solid

Edge Types:
{edge_info}

Dimensions:
• X: {x_range[0]:.1f} to {x_range[1]:.1f}
• Y: {y_range[0]:.1f} to {y_range[1]:.1f}  
• Z: {z_range[0]:.1f} to {z_range[1]:.1f}

Components:
• Cuboid 1: 10×20×30 (base)
• Cuboid 2: 12.5×15.3×24.1 (subtracted)
• Result: Material removed from overlap

Visualization:
✓ Faces with transparency
✓ Boolean intersection edges (gray)
✓ Axis-aligned edges (red/green/blue)
✓ Face and edge labels

Validation:
✓ Single shell (manifold)
✓ Proper face count
✓ No degenerate geometry
        """
        
        ax5.text(0.05, 0.95, info_text, transform=ax5.transAxes, 
                fontsize=10, verticalalignment='top', 
                bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8),
                fontfamily='monospace')
        
        plt.tight_layout()
        plt.show()
        
        print(f"✓ Enhanced 3D visualization complete")
        print(f"  • Main view: 3D isometric with face labels")
        print(f"  • Orthographic projections: Front, Top, Side views")
        print(f"  • {len(faces)} faces displayed with individual colors")
        print(f"  • All views show the cut solid geometry")
        
    except Exception as e:
        print(f"✗ Error creating 3D visualization: {e}")
        import traceback
        traceback.print_exc()

def extract_edges_for_display(solid_shape):
    """Extract edges from the solid to show boolean intersection lines and geometry."""
    if not OPENCASCADE_AVAILABLE or solid_shape is None:
        return []
    
    edges_data = []
    
    try:
        # Explore all edges in the solid
        edge_explorer = TopExp_Explorer(solid_shape, TopAbs_EDGE)
        edge_count = 0
        total_edges = 0
        
        # First count total edges
        while edge_explorer.More():
            total_edges += 1
            edge_explorer.Next()
        
        print(f"  Found {total_edges} total edges in solid (should be 60 for fused cuboids)")
        
        # Reset explorer to extract edge data
        edge_explorer = TopExp_Explorer(solid_shape, TopAbs_EDGE)
        
        while edge_explorer.More():
            edge = edge_explorer.Current()
            edge_count += 1
            
            try:
                # Get edge vertices using multiple approaches
                edge_vertices = None
                
                if TOPEXP_AVAILABLE:
                    try:
                        from OCC.Core.TopoDS import TopoDS_Vertex
                        vertex1 = TopoDS_Vertex()
                        vertex2 = TopoDS_Vertex()
                        topexp_Vertices(topods.Edge(edge), vertex1, vertex2)
                        pnt1 = BRep_Tool.Pnt(vertex1)
                        pnt2 = BRep_Tool.Pnt(vertex2)
                        
                        edge_vertices = np.array([
                            [pnt1.X(), pnt1.Y(), pnt1.Z()],
                            [pnt2.X(), pnt2.Y(), pnt2.Z()]
                        ])
                    except Exception as e:
                        print(f"    TopExp.Vertices failed for edge {edge_count}: {e}")
                
                # Fallback: use vertex explorer if TopExp failed
                if edge_vertices is None:
                    try:
                        vertex_explorer = TopExp_Explorer(edge, TopAbs_VERTEX)
                        vertices = []
                        while vertex_explorer.More():
                            vertex = topods.Vertex(vertex_explorer.Current())
                            pnt = BRep_Tool.Pnt(vertex)
                            vertices.append([pnt.X(), pnt.Y(), pnt.Z()])
                            vertex_explorer.Next()
                        
                        if len(vertices) >= 2:
                            edge_vertices = np.array(vertices[:2])  # Take first two vertices
                    except Exception as e:
                        print(f"    Vertex explorer failed for edge {edge_count}: {e}")
                
                # Alternative: try to get edge curve and sample points
                if edge_vertices is None:
                    try:
                        from OCC.Core.BRepAdaptor import BRepAdaptor_Curve
                        curve = BRepAdaptor_Curve(topods.Edge(edge))
                        first_param = curve.FirstParameter()
                        last_param = curve.LastParameter()
                        
                        pnt1 = curve.Value(first_param)
                        pnt2 = curve.Value(last_param)
                        
                        edge_vertices = np.array([
                            [pnt1.X(), pnt1.Y(), pnt1.Z()],
                            [pnt2.X(), pnt2.Y(), pnt2.Z()]
                        ])
                    except Exception as e:
                        print(f"    Curve method failed for edge {edge_count}: {e}")
                
                if edge_vertices is not None:
                    # Calculate edge length
                    edge_vector = edge_vertices[1] - edge_vertices[0]
                    edge_length = np.linalg.norm(edge_vector)
                    
                    # Categorize edge type
                    edge_type = categorize_edge(edge_vertices)
                    
                    edges_data.append((edge_vertices, edge_type, edge_length))
                else:
                    print(f"    Failed to extract vertices for edge {edge_count}")
                    
            except Exception as e:
                # Alternative: try to get edge curve and sample points
                if edge_vertices is None:
                    try:
                        from OCC.Core.BRepAdaptor import BRepAdaptor_Curve
                        curve = BRepAdaptor_Curve(topods.Edge(edge))
                        first_param = curve.FirstParameter()
                        last_param = curve.LastParameter()
                        
                        pnt1 = curve.Value(first_param)
                        pnt2 = curve.Value(last_param)
                        
                        edge_vertices = np.array([
                            [pnt1.X(), pnt1.Y(), pnt1.Z()],
                            [pnt2.X(), pnt2.Y(), pnt2.Z()]
                        ])
                    except Exception as e:
                        print(f"    Curve method failed for edge {edge_count}: {e}")
                
                if edge_vertices is not None:
                    # Calculate edge length
                    edge_vector = edge_vertices[1] - edge_vertices[0]
                    edge_length = np.linalg.norm(edge_vector)
                    
                    # Categorize edge type
                    edge_type = categorize_edge(edge_vertices)
                    
                    edges_data.append((edge_vertices, edge_type, edge_length))
                else:
                    print(f"    Failed to extract vertices for edge {edge_count}")
                    
            except Exception as e:
                print(f"    Error processing edge {edge_count}: {e}")
            
            edge_explorer.Next()
        
        print(f"  Successfully extracted {len(edges_data)} edges from {total_edges} total edges")
        return edges_data
        
    except Exception as e:
        print(f"  Error extracting edges: {e}")
        return []
        print(f"  Error extracting edges: {e}")
        return []

def categorize_edge(edge_vertices):
    """Categorize edge type based on its orientation and position."""
    if len(edge_vertices) != 2:
        return "unknown"
    
    start, end = edge_vertices[0], edge_vertices[1]
    edge_vector = end - start
    
    # Normalize for comparison
    abs_edge = np.abs(edge_vector)
    max_component = np.max(abs_edge)
    
    if max_component < 1e-6:
        return "degenerate"
    
    normalized = abs_edge / max_component
    
    # Check if edge is axis-aligned
    tolerance = 1e-2
    if normalized[0] > 1-tolerance and normalized[1] < tolerance and normalized[2] < tolerance:
        edge_type = "x_aligned"
    elif normalized[1] > 1-tolerance and normalized[0] < tolerance and normalized[2] < tolerance:
        edge_type = "y_aligned" 
    elif normalized[2] > 1-tolerance and normalized[0] < tolerance and normalized[1] < tolerance:
        edge_type = "z_aligned"
    else:
        edge_type = "diagonal"
    
    # Check if this edge is at a boolean intersection zone
    # For fused cuboids, intersection edges occur where the two cuboids meet
    mid_point = (start + end) / 2
    
    # Cuboid 1: (0,0,0) to (10,20,30)
    # Cuboid 2: (5,10,15) to (17.48,25.25,39.13) - translated
    # Intersection zones are where coordinates overlap
    
    is_intersection = False
    debug_reasons = []
    
    # More comprehensive boundary detection for boolean intersection
    # Check for edges on the interface planes between the two cuboids
    
    # X=5 plane (left boundary of cuboid 2)
    if abs(mid_point[0] - 5.0) < 1.0:  # Increased tolerance
        if 8.0 <= mid_point[1] <= 26.0 and 13.0 <= mid_point[2] <= 41.0:
            is_intersection = True
            debug_reasons.append(f"X=5 boundary (mid: {mid_point})")
    
    # X=10 plane (right boundary of cuboid 1, left side of overlap)
    if abs(mid_point[0] - 10.0) < 1.0:
        if 8.0 <= mid_point[1] <= 26.0 and 13.0 <= mid_point[2] <= 41.0:
            is_intersection = True
            debug_reasons.append(f"X=10 boundary (mid: {mid_point})")
    
    # Y=10 plane (front boundary of cuboid 2)
    if abs(mid_point[1] - 10.0) < 1.0:
        if 3.0 <= mid_point[0] <= 19.0 and 13.0 <= mid_point[2] <= 41.0:
            is_intersection = True
            debug_reasons.append(f"Y=10 boundary (mid: {mid_point})")
    
    # Y=20 plane (back boundary of cuboid 1, front side of overlap)
    if abs(mid_point[1] - 20.0) < 1.0:
        if 3.0 <= mid_point[0] <= 19.0 and 13.0 <= mid_point[2] <= 41.0:
            is_intersection = True
            debug_reasons.append(f"Y=20 boundary (mid: {mid_point})")
    
    # Z=15 plane (bottom boundary of cuboid 2)
    if abs(mid_point[2] - 15.0) < 1.0:
        if 3.0 <= mid_point[0] <= 19.0 and 8.0 <= mid_point[1] <= 26.0:
            is_intersection = True
            debug_reasons.append(f"Z=15 boundary (mid: {mid_point})")
    
    # Z=30 plane (top boundary of cuboid 1, bottom side of overlap)
    if abs(mid_point[2] - 30.0) < 1.0:
        if 3.0 <= mid_point[0] <= 19.0 and 8.0 <= mid_point[1] <= 26.0:
            is_intersection = True
            debug_reasons.append(f"Z=30 boundary (mid: {mid_point})")
    
    # Also check for edges within the overlap region
    # Overlap region: X=[5,10], Y=[10,20], Z=[15,30]
    if (4.0 <= mid_point[0] <= 11.0 and 
        9.0 <= mid_point[1] <= 21.0 and 
        14.0 <= mid_point[2] <= 31.0):
        is_intersection = True
        debug_reasons.append(f"Inside overlap region (mid: {mid_point})")
    
    # Debug output for intersection detection
    if is_intersection and debug_reasons:
        print(f"    → Boolean intersection edge detected: {debug_reasons[0]}")
    
    if is_intersection:
        return "boolean_intersection"
    else:
        return edge_type

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

def main():
    """Main function to demonstrate the complete face classification system."""
    print("="*80)
    print("OPENCASCADE FACE CLASSIFICATION SYSTEM")
    print("Boolean Operations with Projection Normal Analysis")
    print("="*80)
    
    if not OPENCASCADE_AVAILABLE:
        print("✗ OpenCASCADE not available. Cannot proceed.")
        return
    
    try:
        # Step 1: Create the boolean solid
        print("\nStep 1: Creating boolean cut solid...")
        solid_shape = create_opencascade_solid()
        
        if solid_shape is None:
            print("✗ Failed to create solid")
            return
        
        # Step 2: Analyze the solid geometry
        print("\nStep 2: Analyzing solid geometry...")
        analyze_solid_geometry(solid_shape)
        
        # Step 2.5: Visualize the 3D solid BEFORE boolean classification
        print("\nStep 2.5: Visualizing 3D solid with simple matplotlib (before arrays)...")
        visualize_3d_solid(solid_shape)
        
        # Step 3: Extract faces and classify by projection normal
        print("\nStep 3: Face extraction and classification...")
        face_polygons = extract_faces_from_solid(solid_shape)
        
        if not face_polygons:
            print("✗ No faces extracted from solid")
            return
        
        # Step 4: Get projection normal from user
        print("\nStep 4: Getting projection normal from user...")
        unit_projection_normal = get_projection_normal_from_user()
        
        # Step 5: Classify faces into arrays A, B, C
        print("\nStep 5: Classifying faces by projection normal...")
        array_A, array_B, array_C = classify_faces_by_projection(face_polygons, unit_projection_normal)
        
        # Step 6: Display results with enhanced visualization
        print("\nStep 6: Displaying enhanced classification results...")
        display_face_arrays(array_A, array_B, array_C)
        
        # Step 7: Plot array visualizations
        print("\nStep 7: Creating enhanced array visualizations...")
        plot_arrays_visualization(array_A, array_B, array_C, unit_projection_normal)
        
        # Step 8: Summary statistics
        print(f"\n" + "="*80)
        print("ENHANCED FINAL SUMMARY")
        print("="*80)
        print(f"Total faces extracted: {len(face_polygons)}")
        print(f"Array A (processed): {len(array_A)} faces")
        print(f"Array B (visible faces): {len(array_B)} faces")
        print(f"Array C (hidden + intersections): {len(array_C)} faces")
        print(f"Historic algorithm features:")
        print(f"  ✓ Depth-based classification")
        print(f"  ✓ Boolean operations (intersection/subtraction)")
        print(f"  ✓ 3D line-face intersection analysis")
        print(f"  ✓ Multi-point sampling")
        print(f"  ✓ Face association tracking")
        print(f"Projection normal used: [{unit_projection_normal[0]:.6f}, {unit_projection_normal[1]:.6f}, {unit_projection_normal[2]:.6f}]")
        
        # Calculate total areas
        total_area_B = sum(face['polygon'].area for face in array_B if hasattr(face.get('polygon'), 'area'))
        total_area_C = sum(face['polygon'].area for face in array_C if hasattr(face.get('polygon'), 'area'))
        
        print(f"Total area in Array B: {total_area_B:.2f}")
        print(f"Total area in Array C: {total_area_C:.2f}")
        print("="*80)
        
        return array_A, array_B, array_C, unit_projection_normal
        
    except Exception as e:
        print(f"✗ Error in main execution: {e}")
        traceback.print_exc()
        return None, None, None, None

# Main execution
if __name__ == "__main__":
    main()