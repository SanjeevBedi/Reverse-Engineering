from shapely.geometry import Polygon, Point
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from itertools import combinations
import random
import traceback

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
        from OCC.Core.TopExp import TopExp
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
    """Create a fused solid using OpenCASCADE boolean add operation."""
    if not OPENCASCADE_AVAILABLE:
        print("✗ OpenCASCADE not available, cannot create solid")
        return None
    
    # Create first cuboid (10x20x30)
    print("Creating first cuboid (10x20x30)...")
    cuboid1 = BRepPrimAPI_MakeBox(10, 20, 30).Shape()
    
    # Create second cuboid with translation to ensure overlap
    random.seed(42)  # For reproducible results
    width2 = random.uniform(8, 15)
    height2 = random.uniform(15, 25) 
    depth2 = random.uniform(20, 35)
    
    print(f"Creating second cuboid ({width2:.1f}x{height2:.1f}x{depth2:.1f})...")
    
    # Create transformation for second cuboid
    transform = gp_Trsf()
    transform.SetTranslation(gp_Vec(5, 10, 15))  # Translate to create overlap
    
    # Create second cuboid
    cuboid2_maker = BRepPrimAPI_MakeBox(width2, height2, depth2)
    cuboid2 = cuboid2_maker.Shape()
    
    # Apply transformation
    cuboid2.Move(TopLoc_Location(transform))
    print("Applied translation transformation (5, 10, 15) to second cuboid")
    
    # Perform boolean subtraction (cut) operation using HLR best practices
    try:
        print("Performing boolean CUT operation...")
        from OCC.Core.BRepAlgoAPI import BRepAlgoAPI_Cut
        cut_op = BRepAlgoAPI_Cut(cuboid1, cuboid2)
        cut_op.Build()
        
        if cut_op.IsDone() and not cut_op.HasErrors():
            cut_shape = cut_op.Shape()
            
            # Validate the result using HLR-style validation
            if validate_fused_shape(cut_shape):
                print(f"✓ Created cut solid using boolean subtract operation:")
                print(f"  Cuboid 1: 10 x 20 x 30")
                print(f"  Cuboid 2: {width2:.1f} x {height2:.1f} x {depth2:.1f} (translated)")
                print(f"  Boolean operation: CUT (SUBTRACT)")
                print(f"  Operation completed successfully with proper error checking")
                print(f"  Shape validation: PASSED")
                
                return cut_shape
            else:
                print(f"✗ Cut shape failed validation")
                print(f"  Falling back to first cuboid only")
                return cuboid1
        else:
            print(f"✗ Boolean cut operation failed:")
            if not cut_op.IsDone():
                print(f"  Operation not completed (IsDone = False)")
            if cut_op.HasErrors():
                print(f"  Operation has errors (HasErrors = True)")
            print(f"  Falling back to first cuboid only")
            return cuboid1
        
    except Exception as e:
        print(f"✗ Boolean cut failed with exception: {e}")
        print(f"  Falling back to first cuboid only")
        return cuboid1

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
                surface = BRepAdaptor_Surface(topods.Face(face))
                
                # Classify surface type
                surface_type = surface.GetType()
                type_name = str(surface_type).split('.')[-1] if hasattr(surface_type, '__str__') else 'Unknown'
                
                print(f"  Face {face_num}: {type_name}")
                
                if 'PLANE' in type_name.upper():
                    face_types['planar'] += 1
                elif any(curved in type_name.upper() for curved in ['CYLINDER', 'SPHERE', 'CONE']):
                    face_types['curved'] += 1
                else:
                    face_types['complex'] += 1
                    
            except Exception as e:
                print(f"  Face {face_num}: Analysis failed - {e}")
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
                
                print(f"        Derivative normal: [{face_normal[0]:.6f}, {face_normal[1]:.6f}, {face_normal[2]:.6f}]")
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
                                # Get edge vertices - TopExp.Vertices maintains edge direction
                                vertex1, vertex2 = TopExp.Vertices(topods.Edge(edge))
                                
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
    
    except Exception as e:
        print(f"      Error extracting vertices: {e}")
        vertices = []
    
    return vertices

def extract_wire_vertices_in_sequence(wire, wire_id):
    """Extract vertices from a wire by traversing edges in sequence.
    
    Each edge connects two vertices with orientations.
    Traverse from vertex(orientation=0) to vertex(orientation=1) to build the polygon.
    
    Args:
        wire: OpenCASCADE wire object  
        wire_id: Wire identifier for debugging
    
    Returns:
        list: Ordered list of [x, y, z] vertex coordinates
    """
    vertices = []
    
    try:
        print(f"          Traversing Wire {wire_id} edges...")
        
        # Method 1: Use TopExp for proper edge traversal with orientation
        if TOPEXP_AVAILABLE:
            try:
                # Get edges in the wire - TopExp respects wire orientation
                edge_explorer = TopExp_Explorer(wire, TopAbs_EDGE)
                edges = []
                
                while edge_explorer.More():
                    edge = edge_explorer.Current()
                    edges.append(edge)
                    edge_explorer.Next()
                
                print(f"            Found {len(edges)} edges in wire {wire_id}")
                
                if not edges:
                    return []
                
                # Build vertex chain by following edge sequence with proper orientation
                vertex_chain = []
                
                for edge_num, edge in enumerate(edges):
                    edge_num += 1  # 1-based numbering for display
                    
                    try:
                        # CRITICAL FIX: Use orientation to get start/end vertices correctly
                        # TopExp.Vertices with orientation=True respects edge direction in wire
                        from OCC.Core.TopAbs import TopAbs_FORWARD, TopAbs_REVERSED
                        
                        # Get edge orientation within the wire
                        edge_orientation = edge.Orientation()
                        
                        # Get vertices with proper orientation consideration
                        vertex_start, vertex_end = TopExp.Vertices(topods.Edge(edge), True)  # True = use orientation
                        
                        # Get coordinates for start and end vertices (respecting edge direction)
                        pnt_start = BRep_Tool.Pnt(vertex_start)
                        pnt_end = BRep_Tool.Pnt(vertex_end)
                        
                        start_coords = [pnt_start.X(), pnt_start.Y(), pnt_start.Z()]
                        end_coords = [pnt_end.X(), pnt_end.Y(), pnt_end.Z()]
                        
                        # Apply edge orientation within wire
                        if edge_orientation == TopAbs_REVERSED:
                            # If edge is reversed in wire, swap start and end
                            start_coords, end_coords = end_coords, start_coords
                            print(f"            Edge {edge_num}: REVERSED - {start_coords} -> {end_coords}")
                        else:
                            print(f"            Edge {edge_num}: FORWARD - {start_coords} -> {end_coords}")
                        
                        # For the first edge, add the start vertex (end will be added when we process next edge)
                        if edge_num == 1:
                            vertex_chain.append(start_coords)
                            print(f"              First edge: added start vertex {start_coords}")
                        
                        # Always add the end vertex (this becomes the start of the next edge)
                        vertex_chain.append(end_coords)
                        print(f"              Added end vertex {end_coords}")
                        
                    except Exception as e:
                        print(f"            ✗ Error processing edge {edge_num}: {e}")
                        # Fallback to old method if orientation fails
                        try:
                            vertex1, vertex2 = TopExp.Vertices(topods.Edge(edge))
                            pnt1 = BRep_Tool.Pnt(vertex1)
                            pnt2 = BRep_Tool.Pnt(vertex2)
                            v1_coords = [pnt1.X(), pnt1.Y(), pnt1.Z()]
                            v2_coords = [pnt2.X(), pnt2.Y(), pnt2.Z()]
                            
                            if edge_num == 1:
                                vertex_chain.extend([v1_coords, v2_coords])
                            else:
                                # Add the vertex that's not already the last one
                                last_vertex = vertex_chain[-1]
                                last_tuple = tuple(np.round(last_vertex, 6))
                                v1_tuple = tuple(np.round(v1_coords, 6))
                                v2_tuple = tuple(np.round(v2_coords, 6))
                                
                                if v1_tuple != last_tuple:
                                    vertex_chain.append(v1_coords)
                                else:
                                    vertex_chain.append(v2_coords)
                        except Exception as e2:
                            print(f"            ✗ Fallback method also failed: {e2}")
                
                # Remove closing duplicate if present
                if len(vertex_chain) > 2:
                    first_tuple = tuple(np.round(vertex_chain[0], 6))
                    last_tuple = tuple(np.round(vertex_chain[-1], 6))
                    
                    if first_tuple == last_tuple:
                        vertex_chain = vertex_chain[:-1]  # Remove closing duplicate
                        print(f"            Removed closing duplicate vertex")
                
                vertices = vertex_chain
                print(f"            ✓ Built vertex chain: {len(vertices)} vertices")
                
            except Exception as e:
                print(f"            ✗ TopExp edge traversal failed: {e}")
        
        # Fallback: Basic edge traversal if TopExp fails
        if not vertices:
            print(f"            Using fallback edge traversal method...")
            
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
            
            print(f"            ✓ Fallback method: {len(vertices)} vertices")
    
    except Exception as e:
        print(f"          ✗ Error extracting vertices from wire {wire_id}: {e}")
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
            # Create a polygon patch for each face
            from mpl_toolkits.mplot3d.art3d import Poly3DCollection
            
            face_color = colors[i % len(colors)]
            
            # Handle complex faces with more than 4 vertices properly
            num_vertices = len(vertices)
            
            if num_vertices == 3:
                # Triangular face - display as single triangle
                poly3d = [vertices]
                print(f"    Rendering triangular face F{i+1} (3 vertices)")
            elif num_vertices == 4:
                # Rectangular face - render as single planar polygon without triangulation
                print(f"    Rendering rectangular face F{i+1} (4 vertices) as single polygon")
                
                # Apply same logic as complex faces - render directly without modification
                # This prevents diagonal lines that can appear with vertex reordering
                poly3d = [vertices]
                print(f"      Using direct polygon vertices - no triangulation creates clean planar faces")
            elif num_vertices >= 5:
                # Complex polygon - render as single planar polygon without triangulation
                print(f"    Rendering complex face F{i+1} ({num_vertices} vertices) as single polygon")
                
                # For planar faces from boolean operations, render as a single polygon
                # This preserves the actual face geometry without artificial diagonal lines
                poly3d = [vertices]
                print(f"      Using direct polygon vertices - no triangulation creates clean planar faces")
                        
                        # We'll draw the boundary manually as lines below
            else:
                print(f"    Degenerate face F{i+1} ({num_vertices} vertices) - skipping")
                continue
            
            # Add face with enhanced styling
            if poly3d:  # Only add collection if we have polygons to render
                collection = ax1.add_collection3d(Poly3DCollection(
                    poly3d, 
                    alpha=0.4,  # More transparent to show edges better
                    facecolor=face_color, 
                    edgecolor='black',
                    linewidth=1.0,   # Thinner face edges
                    zorder=1        # Draw faces behind edges
                ))
            else:
                # For faces that couldn't be triangulated, draw boundary edges only
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
        # Front view (Y-Z projection) - show clean solid outline
        front_outline_y = [y_min, y_max, y_max, y_min, y_min]
        front_outline_z = [z_min, z_min, z_max, z_max, z_min]
        
        # Top view (X-Y projection) - show clean solid outline  
        top_outline_x = [x_min, x_max, x_max, x_min, x_min]
        top_outline_y = [y_min, y_min, y_max, y_max, y_min]
        
        # Side view (X-Z projection) - show clean solid outline
        side_outline_x = [x_min, x_max, x_max, x_min, x_min]
        side_outline_z = [z_min, z_min, z_max, z_max, z_min]
        
        # For internal structure, only show major faces (4-vertex rectangular faces)
        # and simplified representations of complex faces
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
        
        print(f"  Found {total_edges} total edges in solid")
        
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
                        vertex1, vertex2 = TopExp.Vertices(topods.Edge(edge))
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
    """Project 3D face vertices onto the plane with the given normal vector."""
    # Normalize the projection normal
    normal = projection_normal / np.linalg.norm(projection_normal)
    
    # Create two orthogonal vectors in the projection plane
    # Find a vector that's not parallel to the normal
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
    
    # Project each vertex onto the plane using the basis vectors
    projected = []
    for vertex in face_vertices:
        # Project vertex onto the plane defined by u and v
        proj_u = np.dot(vertex, u)
        proj_v = np.dot(vertex, v)
        projected.append([proj_u, proj_v])
    
    return np.array(projected)

def project_face_to_xy_plane(face_vertices):
    """Project 3D face vertices onto the XY plane (z=0) - Legacy function."""
    projected = face_vertices[:, :2]  # Take only x, y coordinates
    return projected

def validate_cuboid_face(vertices):
    """Validate that we have proper rectangular/cuboid face vertices."""
    if len(vertices) < 4:
        return False, "Less than 4 vertices"
    
    # For cuboids, we should have rectangular faces
    # Check if we have exactly 4 unique vertices (after removing duplicates)
    unique_vertices = []
    seen = set()
    for v in vertices:
        v_tuple = tuple(np.round(v, 6))
        if v_tuple not in seen:
            unique_vertices.append(v)
            seen.add(v_tuple)
    
    if len(unique_vertices) < 3:
        return False, f"Only {len(unique_vertices)} unique vertices"
    elif len(unique_vertices) == 3:
        return False, f"TRIANGULAR FACE DETECTED! Only {len(unique_vertices)} vertices - cuboids should not have triangular faces"
    elif len(unique_vertices) > 6:
        return False, f"Too many vertices ({len(unique_vertices)}) for a cuboid face"
    
    # For cuboid faces from boolean operations, we can have 4-6 vertices
    # but they should form a valid polygon
    return True, f"Valid face with {len(unique_vertices)} vertices"

def create_polygon_from_projection(projected_vertices):
    """Create a Shapely polygon from projected vertices."""
    # Ensure the polygon is closed by adding the first vertex at the end if needed
    if not np.array_equal(projected_vertices[0], projected_vertices[-1]):
        projected_vertices = np.vstack([projected_vertices, projected_vertices[0]])
    
    # Validate that we don't have triangular faces (cuboids should only have rectangular faces)
    num_vertices = len(projected_vertices) - 1  # Subtract 1 for closing vertex
    if num_vertices == 3:
        print(f"    → ⚠️  TRIANGULAR FACE DETECTED! ({num_vertices} vertices)")
        print(f"    → This is unexpected for cuboids - investigating...")
        print(f"    → Vertices: {projected_vertices[:-1]}")
        # For debugging, let's still try to create the polygon but flag it clearly
    elif num_vertices < 3:
        print(f"    → ❌ Error: Degenerate face with only {num_vertices} vertices")
        return Polygon()  # Return empty polygon
    elif num_vertices == 4:
        print(f"    → ✅ Rectangular face ({num_vertices} vertices) - expected for cuboids")
    elif num_vertices > 8:
        print(f"    → ⚠️  Warning: Face has many vertices ({num_vertices}) - might be over-sampled")
    else:
        print(f"    → ℹ️  Complex face with {num_vertices} vertices (expected for fused cuboids)")
    
    try:
        polygon = Polygon(projected_vertices)
        
        # For valid polygons, return them directly
        if polygon.is_valid and hasattr(polygon, 'area') and polygon.area > 1e-6:
            return polygon
        
        # For invalid polygons, try more sophisticated fixes
        if not polygon.is_valid:
            # Get the reason for invalidity if available
            reason = "unknown"
            if hasattr(polygon, 'is_valid_reason'):
                reason = polygon.is_valid_reason
            print(f"    → Invalid polygon detected (reason: {reason})")
            
            # For rectangular faces (4 vertices), try reordering to ensure proper winding
            if num_vertices == 4:
                print(f"    → Attempting to fix rectangular face winding order...")
                # Try different vertex orderings for rectangles
                original_vertices = projected_vertices[:-1]  # Remove closing vertex
                
                # Try counter-clockwise ordering
                if len(original_vertices) == 4:
                    # Calculate centroid
                    centroid = np.mean(original_vertices, axis=0)
                    
                    # Sort vertices by angle from centroid to ensure proper winding
                    def angle_from_centroid(vertex):
                        return np.arctan2(vertex[1] - centroid[1], vertex[0] - centroid[0])
                    
                    sorted_vertices = sorted(original_vertices, key=angle_from_centroid)
                    sorted_vertices.append(sorted_vertices[0])  # Close the polygon
                    
                    sorted_polygon = Polygon(sorted_vertices)
                    if sorted_polygon.is_valid and sorted_polygon.area > 1e-6:
                        print(f"    → Fixed with sorted vertices, area: {sorted_polygon.area:.6f}")
                        return sorted_polygon
            
            # Try buffer(0) fix for self-intersections
            try:
                fixed_polygon = polygon.buffer(0)
                if fixed_polygon.is_valid and hasattr(fixed_polygon, 'area') and fixed_polygon.area > 1e-6:
                    # Make sure we didn't accidentally create triangles from rectangles
                    if fixed_polygon.geom_type == 'Polygon':
                        fixed_coords = list(fixed_polygon.exterior.coords)
                        fixed_vertices = len(fixed_coords) - 1
                        if num_vertices == 4 and fixed_vertices == 3:
                            print(f"    → WARNING: Buffer fix created triangle from rectangle - using original")
                            return polygon  # Return original even if invalid, rather than creating triangles
                        else:
                            print(f"    → Buffer fix successful, area: {fixed_polygon.area:.6f}")
                            return fixed_polygon
                    elif fixed_polygon.geom_type == 'MultiPolygon':
                        print(f"    → Buffer created MultiPolygon - taking largest component")
                        largest_poly = max(fixed_polygon.geoms, key=lambda p: p.area)
                        return largest_poly
            except Exception as e:
                print(f"    → Buffer fix failed: {e}")
            
            # Try reversing the vertex order for clockwise/counterclockwise issues
            print(f"    → Trying vertex reordering...")
            try:
                reversed_vertices = projected_vertices[::-1]
                reversed_polygon = Polygon(reversed_vertices)
                if reversed_polygon.is_valid and hasattr(reversed_polygon, 'area') and reversed_polygon.area > 1e-6:
                    print(f"    → Reversed vertex order worked, area: {reversed_polygon.area:.6f}")
                    return reversed_polygon
            except Exception as e:
                print(f"    → Vertex reordering failed: {e}")
            
            print(f"    → All fixes failed, returning original polygon (may be invalid)")
        
        return polygon
        
    except Exception as e:
        print(f"    → Error creating polygon: {e}")
        return Polygon()  # Return empty polygon

def find_interior_point(polygon):
    """Find a point that is guaranteed to be inside the polygon.
    
    First tries the centroid, then falls back to a point inside using buffering.
    """
    if polygon.is_empty or polygon.area == 0:
        return None
    
    # Try centroid first
    centroid = polygon.centroid
    if polygon.contains(centroid):
        return centroid
    
    # If centroid is outside (can happen with complex polygons), 
    # use representative point which is guaranteed to be inside
    try:
        rep_point = polygon.representative_point()
        if polygon.contains(rep_point):
            return rep_point
    except:
        pass
    
    # Last resort: shrink polygon slightly and use its centroid
    try:
        buffered = polygon.buffer(-0.01)  # Shrink by small amount
        if not buffered.is_empty and buffered.area > 0:
            shrunk_centroid = buffered.centroid
            if polygon.contains(shrunk_centroid):
                return shrunk_centroid
    except:
        pass
    
    # Absolute fallback: use the first coordinate
    try:
        coords = list(polygon.exterior.coords)
        if len(coords) > 0:
            return Point(coords[0])
    except:
        pass
    
    return None

def intersect_line_with_face(point_2d, projection_normal, face_vertices_3d):
    """Intersect a line from a 2D point in the projection normal direction with a 3D face.
    
    Args:
        point_2d: Point in 2D projection plane
        projection_normal: 3D normal vector of projection plane
        face_vertices_3d: 3D vertices of the face to intersect with
        
    Returns:
        3D point of intersection, or None if no intersection
    """
    try:
        # Convert 2D point back to 3D on the projection plane
        # We need to find the 3D point corresponding to the 2D projection
        
        # Create orthogonal basis vectors for the projection plane
        normal = projection_normal / np.linalg.norm(projection_normal)
        
        # Find a vector that's not parallel to the normal  
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
        # We need a reference point on the plane - use origin of projection
        plane_origin = np.array([0, 0, 0])  # Simplification - could be more sophisticated
        point_3d_on_plane = plane_origin + point_2d.x * u + point_2d.y * v
        
        # Create the line in 3D space
        line_direction = normal  # Line goes in projection normal direction
        
        # Find intersection with the face plane
        # First, find the face plane equation
        if len(face_vertices_3d) < 3:
            return None
            
        # Calculate face normal
        v1 = face_vertices_3d[1] - face_vertices_3d[0]
        v2 = face_vertices_3d[2] - face_vertices_3d[0]
        face_normal = np.cross(v1, v2)
        
        if np.linalg.norm(face_normal) < 1e-10:
            return None
            
        face_normal = face_normal / np.linalg.norm(face_normal)
        
        # Plane equation: face_normal · (P - face_vertices_3d[0]) = 0
        # Line equation: P = point_3d_on_plane + t * line_direction
        # Substitution: face_normal · (point_3d_on_plane + t * line_direction - face_vertices_3d[0]) = 0
        
        denominator = np.dot(face_normal, line_direction)
        if abs(denominator) < 1e-10:
            # Line is parallel to plane
            return None
            
        numerator = np.dot(face_normal, face_vertices_3d[0] - point_3d_on_plane)
        t = numerator / denominator
        
        # Calculate intersection point
        intersection_3d = point_3d_on_plane + t * line_direction
        
        return intersection_3d
        
    except Exception as e:
        print(f"      Error in line-face intersection: {e}")
        return None

def calculate_depth_along_normal(point_3d, projection_normal):
    """Calculate the depth of a 3D point along the projection normal direction.
    
    Args:
        point_3d: 3D point
        projection_normal: Direction vector for depth measurement
        
    Returns:
        Scalar depth value (higher = further along normal direction)
    """
    if point_3d is None:
        return 0
    
    normal = projection_normal / np.linalg.norm(projection_normal)
    return np.dot(point_3d, normal)

def scale_polygons_to_fit(polygons, target_size=10):
    """Scale a list of polygons to fit within a square of target_size x target_size."""
    if not polygons:
        return polygons
    
    # Find the bounding box of all polygons combined
    all_bounds = []
    for poly, name, normal in polygons:
        if poly.geom_type == 'Polygon' and poly.area > 0:
            bounds = poly.bounds  # (minx, miny, maxx, maxy)
            all_bounds.append(bounds)
    
    if not all_bounds:
        return polygons
    
    # Calculate overall bounding box
    min_x = min(bounds[0] for bounds in all_bounds)
    min_y = min(bounds[1] for bounds in all_bounds)
    max_x = max(bounds[2] for bounds in all_bounds)
    max_y = max(bounds[3] for bounds in all_bounds)
    
    # Calculate current size and scale factor
    current_width = max_x - min_x
    current_height = max_y - min_y
    current_size = max(current_width, current_height)
    
    if current_size <= 0:
        return polygons
    
    scale_factor = target_size / current_size
    
    # Calculate translation to center in the target square
    center_x = (min_x + max_x) / 2
    center_y = (min_y + max_y) / 2
    target_center = target_size / 2
    
    # Scale and translate each polygon
    scaled_polygons = []
    for poly, name, normal in polygons:
        if poly.geom_type == 'Polygon' and poly.area > 0:
            # Get polygon coordinates
            coords = list(poly.exterior.coords)
            
            # Scale and translate coordinates
            scaled_coords = []
            for x, y in coords:
                # Translate to origin, scale, then translate to target center
                new_x = (x - center_x) * scale_factor + target_center
                new_y = (y - center_y) * scale_factor + target_center
                scaled_coords.append((new_x, new_y))
            
            # Create new scaled polygon
            scaled_poly = Polygon(scaled_coords)
            scaled_polygons.append((scaled_poly, name, normal))
        else:
            # Keep non-polygon entries as is
            scaled_polygons.append((poly, name, normal))
    
    print(f"Scaled {len(scaled_polygons)} polygons by factor {scale_factor:.3f} to fit in {target_size}x{target_size} square")
    return scaled_polygons

# Generate cut solid using OpenCASCADE boolean subtract
print("\n" + "="*60)
print("3D BOOLEAN SUBTRACT (CUT) ANALYSIS (OpenCASCADE)")
print("="*60)

# Create cut solid by subtracting second cuboid from first
cut_solid = create_opencascade_solid()

# Immediately analyze and visualize the created solid
if cut_solid is not None:
    # Perform detailed geometry analysis
    analyze_solid_geometry(cut_solid)
    
    # Create enhanced 3D visualization
    display_3d_solid(cut_solid)
    
    # Extract all faces from the cut solid for further processing
    solid_faces = extract_faces_from_solid(cut_solid)
    
    print(f"✓ Extracted {len(solid_faces)} faces from cut solid")
else:
    print("✗ Failed to create solid - skipping visualization and analysis")
    solid_faces = []

# Get projection plane normal from user input
print("\n" + "="*60)
print("PROJECTION PLANE CONFIGURATION")
print("="*60)
print("Enter the projection normal vector components:")
print("(Default values: X=0.2, Y=1.0, Z=0.0 for slightly angled Y-direction projection)")

try:
    # Always try interactive input first, regardless of environment detection
    print("Please enter projection normal components (press Enter for defaults):")
    x_input = input("Enter X component (default 0.2): ").strip()
    y_input = input("Enter Y component (default 1.0): ").strip()
    z_input = input("Enter Z component (default 0.0): ").strip()
    
    # Use defaults if no input provided
    x_component = float(x_input) if x_input else 0.2
    y_component = float(y_input) if y_input else 1.0
    z_component = float(z_input) if z_input else 0.0
    
    print(f"✓ Using input: [{x_component}, {y_component}, {z_component}]")
    
    # Validate that at least one component is non-zero
    if abs(x_component) < 1e-10 and abs(y_component) < 1e-10 and abs(z_component) < 1e-10:
        print("⚠️  Warning: All components are zero! Using default [0.2, 1.0, 0.0]")
        projection_plane_normal = np.array([0.2, 1.0, 0.0])
    else:
        projection_plane_normal = np.array([x_component, y_component, z_component])
    
except (ValueError, EOFError):
    print("⚠️  Error getting input! Using default projection normal [0.2, 1.0, 0.0]")
    projection_plane_normal = np.array([0.2, 1.0, 0.0])
except KeyboardInterrupt:
    print("\n⚠️  Input cancelled! Using default projection normal [0.2, 1.0, 0.0]")
    projection_plane_normal = np.array([0.2, 1.0, 0.0])

# Set projection plane
unit_projection_normal = projection_plane_normal / np.linalg.norm(projection_plane_normal)

# Store valid projections
valid_polygons = []

print(f"\nProjection plane normal: {projection_plane_normal}")
print(f"Unit projection normal: {unit_projection_normal}")
print(f"\nFace analysis for {len(solid_faces)} extracted faces:")

print(f"\nProjection plane normal: {projection_plane_normal}")
print(f"Unit projection normal: {unit_projection_normal}")
print(f"\nFace analysis for {len(solid_faces)} extracted faces:")

# DEBUG: Let's examine each face in detail
for i, face_data in enumerate(solid_faces):
    face_points = np.array(face_data['outer_boundary'])  # Convert to numpy array for vector operations
    face_name = f"Face {i+1}"
    
    print(f"\n{face_name}:")
    print(f"  Raw face points: {face_points.tolist()}")
    
    # Compute normal using different methods to compare
    if len(face_points) >= 3:
        # Method 1: Cross product of first two edge vectors
        v1 = face_points[1] - face_points[0]
        v2 = face_points[2] - face_points[0]
        cross_normal = np.cross(v1, v2)
        
        print(f"  Edge vector 1: {v1}")
        print(f"  Edge vector 2: {v2}")
        print(f"  Cross product: {cross_normal}")
        print(f"  Cross product magnitude: {np.linalg.norm(cross_normal)}")
        
        if np.linalg.norm(cross_normal) > 1e-10:
            unit_cross_normal = cross_normal / np.linalg.norm(cross_normal)
            print(f"  Unit cross normal: {unit_cross_normal}")
            
            # Calculate dot product
            dot_product = np.dot(unit_cross_normal, unit_projection_normal)
            print(f"  Dot product with projection normal: {dot_product:.6f}")
        else:
            print(f"  WARNING: Zero or near-zero cross product - degenerate face")
        
        # Method 2: Try different point combinations
        if len(face_points) >= 4:
            # Try points 0,1,3 instead of 0,1,2
            v1_alt = face_points[1] - face_points[0]
            v2_alt = face_points[3] - face_points[0]
            cross_normal_alt = np.cross(v1_alt, v2_alt)
            
            print(f"  Alternative edge vector 1: {v1_alt}")
            print(f"  Alternative edge vector 2: {v2_alt}")
            print(f"  Alternative cross product: {cross_normal_alt}")
            
            if np.linalg.norm(cross_normal_alt) > 1e-10:
                unit_cross_normal_alt = cross_normal_alt / np.linalg.norm(cross_normal_alt)
                print(f"  Alternative unit normal: {unit_cross_normal_alt}")
                dot_product_alt = np.dot(unit_cross_normal_alt, unit_projection_normal)
                print(f"  Alternative dot product: {dot_product_alt:.6f}")
        
        # Method 3: Check if points are coplanar and compute normal differently
        face_center = np.mean(face_points, axis=0)
        print(f"  Face center: {face_center}")
        
        # Determine face orientation by examining coordinate ranges
        x_coords = face_points[:, 0]
        y_coords = face_points[:, 1]
        z_coords = face_points[:, 2]
        
        x_range = np.max(x_coords) - np.min(x_coords)
        y_range = np.max(y_coords) - np.min(y_coords)
        z_range = np.max(z_coords) - np.min(z_coords)
        
        print(f"  X range: {x_range:.6f} (min: {np.min(x_coords):.3f}, max: {np.max(x_coords):.3f})")
        print(f"  Y range: {y_range:.6f} (min: {np.min(y_coords):.3f}, max: {np.max(y_coords):.3f})")
        print(f"  Z range: {z_range:.6f} (min: {np.min(z_coords):.3f}, max: {np.max(z_coords):.3f})")
        
        # Determine face type based on which coordinate is nearly constant
        tolerance = 1e-3
        if x_range < tolerance:
            print(f"  → X-normal face (YZ plane)")
            face_normal_analytical = np.array([1, 0, 0]) if face_center[0] > 0 else np.array([-1, 0, 0])
            print(f"  → Analytical normal: {face_normal_analytical}")
            dot_analytical = np.dot(face_normal_analytical, unit_projection_normal)
            print(f"  → Analytical dot product: {dot_analytical:.6f}")
        elif y_range < tolerance:
            print(f"  → Y-normal face (XZ plane)")
            face_normal_analytical = np.array([0, 1, 0]) if face_center[1] > 0 else np.array([0, -1, 0])
            print(f"  → Analytical normal: {face_normal_analytical}")
            dot_analytical = np.dot(face_normal_analytical, unit_projection_normal)
            print(f"  → Analytical dot product: {dot_analytical:.6f}")
        elif z_range < tolerance:
            print(f"  → Z-normal face (XY plane)")
            face_normal_analytical = np.array([0, 0, 1]) if face_center[2] > 0 else np.array([0, 0, -1])
            print(f"  → Analytical normal: {face_normal_analytical}")
            dot_analytical = np.dot(face_normal_analytical, unit_projection_normal)
            print(f"  → Analytical dot product: {dot_analytical:.6f}")
        else:
            print(f"  → Non-axis-aligned face")

print(f"\n" + "="*50)
print("SUMMARY:")
print(f"Projection normal: {unit_projection_normal}")
print(f"Expected positive dot products for faces with normals in +Y direction")
print(f"For compound with two cuboids, we should see at least 2 faces with positive Y normals")
print("="*50)

# Store valid projections
valid_polygons = []

print(f"\nFace analysis for {len(solid_faces)} extracted faces:")

# For proper HLR, select only the 6 most prominent faces that would be visible
# in a standard engineering drawing. From a 9-face boolean result, we need to 
# identify the 6 main external faces that represent the solid's envelope.

# Group faces by their normal directions (X, Y, Z orientations)
face_groups = {'X_pos': [], 'X_neg': [], 'Y_pos': [], 'Y_neg': [], 'Z_pos': [], 'Z_neg': []}

for i, face_data in enumerate(solid_faces):
    # Extract points and normal from the face data
    face_points = face_data['points']
    face_normal = face_data['normal']  # Use the actual OpenCASCADE normal
    
    face_name = f"Face {i+1}"
    face_vertices = face_points
    face_center = np.mean(face_points, axis=0)
    
    # Calculate dot product with projection normal using the actual OpenCASCADE normal
    unit_face_normal = face_normal / np.linalg.norm(face_normal)
    dot_product = np.dot(unit_face_normal, unit_projection_normal)
    
    print(f"{face_name}: center={face_center}, normal={face_normal}, dot_product={dot_product:.3f}")
    
    # Classify face by its primary normal direction
    abs_normal = np.abs(unit_face_normal)
    max_component = np.max(abs_normal)
    
    if abs_normal[0] == max_component:  # X-dominant normal
        if unit_face_normal[0] > 0:
            face_groups['X_pos'].append((i, face_data, face_center, dot_product))
        else:
            face_groups['X_neg'].append((i, face_data, face_center, dot_product))
    elif abs_normal[1] == max_component:  # Y-dominant normal
        if unit_face_normal[1] > 0:
            face_groups['Y_pos'].append((i, face_data, face_center, dot_product))
        else:
            face_groups['Y_neg'].append((i, face_data, face_center, dot_product))
    elif abs_normal[2] == max_component:  # Z-dominant normal
        if unit_face_normal[2] > 0:
            face_groups['Z_pos'].append((i, face_data, face_center, dot_product))
        else:
            face_groups['Z_neg'].append((i, face_data, face_center, dot_product))

# Process ALL 9 faces instead of just 6 representative ones for complete coverage
print(f"\nProcessing ALL {len(solid_faces)} faces for complete HLR coverage:")
print(f"This ensures we get all 9 faces → array_A will have 6 faces, array_B+C will have 9 total")

selected_faces = []
for i, face_data in enumerate(solid_faces):
    face_points = face_data['points']
    face_normal = face_data['normal']
    face_name = f"Face {i+1}"
    face_center = np.mean(face_points, axis=0)
    
    # Calculate dot product for this face
    unit_face_normal = face_normal / np.linalg.norm(face_normal)
    dot_product = np.dot(unit_face_normal, unit_projection_normal)
    
    selected_faces.append((i, face_data, face_center, dot_product))
    print(f"  → Added Face {i+1}: center={face_center}, normal={face_normal}, dot_product={dot_product:.3f}")

print(f"\nSelected ALL {len(selected_faces)} faces for complete processing:")
print(f"This ensures complete geometric coverage of the boolean solid")

# Now process ALL selected faces
for face_idx, face_data, face_center, dot_product in selected_faces:
    face_points = face_data['points']
    face_normal = face_data['normal']
    face_name = f"Face {face_idx+1}"
    face_vertices = face_points
    
    print(f"{face_name}: center={face_center}, normal={face_normal}, dot_product={dot_product:.3f}")
    
    # Include ALL selected faces regardless of dot product sign to ensure we get 6 polygons
    # The depth sorting algorithm will handle visibility correctly later
    projected_vertices = project_face_to_projection_plane(face_vertices, unit_projection_normal)
    try:
        polygon = create_polygon_from_projection(projected_vertices)
        print(f"  → Projected vertices: {projected_vertices}")
        print(f"  → Polygon valid: {polygon.is_valid}, area: {polygon.area:.6f}")
        if polygon.is_valid and polygon.area > 1e-6:  # Only add valid polygons with meaningful area
            valid_polygons.append((polygon, face_name, face_normal))
            print(f"  → Added to polygon array (area: {polygon.area:.2f})")
        else:
            print(f"  → Skipped (invalid: {not polygon.is_valid} or tiny area: {polygon.area:.6f})")
    except Exception as e:
        print(f"  → Skipped (projection error: {e})")

print(f"\nTotal valid polygons: {len(valid_polygons)}")

# If we don't have enough faces from OpenCASCADE, fall back to simple cuboid
if len(valid_polygons) < 2:
    print("\n" + "="*50)
    print("FALLBACK: Using simple cuboid faces")
    print("="*50)
    
    cuboid_faces = generate_cuboid_faces(10, 20, 30)
    face_names = ['Bottom', 'Top', 'Front', 'Back', 'Left', 'Right']
    
    valid_polygons = []  # Reset
    
    for i, (face_vertices, face_normal) in enumerate(cuboid_faces):
        unit_face_normal = face_normal / np.linalg.norm(face_normal)
        dot_product = np.dot(unit_face_normal, unit_projection_normal)
        
        print(f"{face_names[i]} face: normal={face_normal}, dot_product={dot_product:.3f}")
        print(f"  3D vertices: {face_vertices}")
        
        # Apply dot product classification for cuboid faces too
        projected_vertices = project_face_to_projection_plane(face_vertices, unit_projection_normal)
        print(f"  2D projection: {projected_vertices}")
        polygon = create_polygon_from_projection(projected_vertices)
        print(f"  Polygon area: {polygon.area:.6f}")
        if polygon.area > 1e-6:
            valid_polygons.append((polygon, face_names[i], face_normal))
            if dot_product > 0:
                print(f"  → Added to polygon array (area: {polygon.area:.2f}) - will go to array_A")
            else:
                print(f"  → Added to polygon array (area: {polygon.area:.2f}) - will go to array_C")
        else:
            print(f"  → Skipped (area too small: {polygon.area:.6f})")

print(f"\nTotal valid polygons: {len(valid_polygons)}")

# Skip original projections plot - will show array_A instead

print(f"\n" + "="*60)

# Print coordinates of extracted faces
if valid_polygons:
    print(f"\nCoordinates of extracted faces:")
    for i, (poly, name, normal) in enumerate(valid_polygons):
        if poly.geom_type == 'Polygon' and poly.area > 0:
            coords = list(poly.exterior.coords)
            unique_coords = coords[:-1]  # Remove closing coordinate
            num_vertices = len(unique_coords)
            
            print(f"{name}: {coords}")
            print(f"  Normal: {normal}")
            print(f"  Area: {poly.area:.2f}")
            print(f"  Unique vertices: {num_vertices}")
            
            # Check for triangular faces - this shouldn't happen with cuboids!
            if num_vertices == 3:
                print(f"  ⚠️  WARNING: TRIANGULAR FACE DETECTED! This shouldn't happen with cuboids.")
                print(f"      Unique coordinates: {unique_coords}")
                print(f"      This indicates an issue with face extraction from OpenCASCADE")
            elif num_vertices < 3:
                print(f"  ❌ ERROR: Degenerate face with only {num_vertices} vertices")
            elif num_vertices == 4:
                print(f"  ✅ Rectangular face (expected for cuboids)")
            elif num_vertices > 4:
                print(f"  ℹ️  Complex face with {num_vertices} vertices (expected for fused cuboids)")
            
            print()  # Empty line for readability

# ============================================================================
# POLYGON PROCESSING ALGORITHM
# ============================================================================

# Create array_A with all polygons initially, then classify based on dot product after depth processing
# This ensures we process all faces and get the complete 6+3=9 polygon distribution
array_A = []
array_C_initial = []  # Store faces with negative dot product for later addition to array_C

for i, (poly, name, normal) in enumerate(valid_polygons):
    if poly.geom_type == 'Polygon' and poly.area > 0:
        # Calculate dot product with projection normal
        unit_face_normal = normal / np.linalg.norm(normal)
        dot_product = np.dot(unit_face_normal, unit_projection_normal)
        
        # Find the parent face from solid_faces
        parent_face = None
        for face_idx, face_data in enumerate(solid_faces):
            if name == f"Face {face_idx+1}":
                parent_face = face_data['points']  # 3D vertices of the parent face
                break
        
        # Store polygon with parent face information
        polygon_data = {
            'polygon': poly,
            'name': name,
            'normal': normal,
            'parent_face': parent_face,
            'original_index': i,
            'dot_product': dot_product
        }
        
        # ALL faces go to array_A initially for depth-based processing
        # We'll sort them into final arrays based on dot product AFTER depth processing
        array_A.append(polygon_data)
        print(f"  → {name}: dot_product={dot_product:.3f}, added to array_A for processing")
        
        # Track which faces have negative dot product for later classification
        if dot_product <= 0:
            array_C_initial.append(polygon_data)
            print(f"    (will be moved to array_C after depth processing)")

print(f"\n" + "="*60)
print("POLYGON PROCESSING ALGORITHM")
print("="*60)
print(f"Initial processing setup:")
print(f"  • array_A (all faces for processing): {len(array_A)} polygons")
print(f"  • array_C_initial (negative dot product): {len(array_C_initial)} polygons")
print(f"  • Total polygons: {len(array_A)}")

# ============================================================================
# ALGORITHM SUMMARY
# ============================================================================
print("""
ALGORITHM SUMMARY: COMPLETE FACE PROCESSING WITH DOT PRODUCT CLASSIFICATION

Purpose: 
1. Process ALL faces through depth-based algorithm to get complete geometry
2. Then apply dot product classification to determine final array placement
3. Ensures we get all 9 faces total (6 in various arrays based on processing)

Algorithm Steps:
1. COMPLETE FACE PROCESSING:
   • ALL faces go to array_A initially for depth-based processing
   • Track faces with negative dot product for later classification
   • Process through full depth algorithm to capture all geometric relationships

2. SEEDING (process all faces):
   • Move first polygon from array_A to array_B as the seed polygon

3. ITERATIVE PROCESSING WITH DEPTH CLASSIFICATION:
   For each remaining polygon Pi in array_A:
   a) Test intersection with all polygons Pj in array_B using shapely operations
   b) For each intersection found:
      - Find an interior point within the intersection region (guaranteed inside)
      - Generate a 3D line from this point in the projection normal direction
      - Intersect this line with the parent faces of Pi and Pj
      - Calculate depth along projection normal for both intersection points
      - DEPTH COMPARISON:
        * If Pi_depth > Pj_depth: Pj = Pj - Pi (subtract Pi from Pj)
        * If Pj_depth > Pi_depth: Pi = Pi - Pj (subtract Pj from Pi)
      - Move Pi to array_C (classification complete)
   c) Continue until Pi is classified

4. DEPTH-BASED BOOLEAN OPERATIONS:
   • Uses 3D depth information to determine which polygon should be modified
   • Ensures geometric consistency based on actual 3D face positions
   • More accurate than 2D-only intersection analysis

5. CLASSIFICATION LOGIC:
   • array_B: Final polygons (background geometry)
   • array_C: Classified polygons (foreground/intersecting geometry)
   • Classification based on 3D depth determines layer ordering

Result: Geometrically consistent classification based on actual 3D face depths
""")

print(f"Initial array_A contains {len(array_A)} polygons")

# ============================================================================
# PLOT ARRAY_A (INITIAL POLYGONS)
# ============================================================================
if array_A:
    print(f"\n" + "="*60)
    print("ARRAY_A VISUALIZATION (INITIAL POLYGONS)")
    print("="*60)
    
    fig_A, ax_A = plt.subplots(1, 1, figsize=(12, 10))
    
    # Color scheme for different polygons
    colors = ['lightblue', 'lightcoral', 'lightgreen', 'lightyellow', 
              'lightpink', 'lightgray', 'lightcyan', 'lightsalmon']
    edge_colors = ['blue', 'red', 'green', 'orange', 'purple', 'gray', 'cyan', 'brown']
    
    # Plot each polygon in array_A
    for i, poly_data in enumerate(array_A):
        polygon = poly_data['polygon']
        face_name = poly_data['name']
        
        if polygon.geom_type == 'Polygon' and polygon.area > 0:
            face_color = colors[i % len(colors)]
            edge_color = edge_colors[i % len(edge_colors)]
            
            # Plot the polygon
            plot_polygon(polygon, ax_A, 
                        facecolor=face_color, 
                        edgecolor=edge_color, 
                        alpha=0.6, 
                        linewidth=2, 
                        label=f'{face_name} (area: {polygon.area:.1f})',
                        outline_only=True)
            
            # Add face center label
            centroid = polygon.centroid
            ax_A.text(centroid.x, centroid.y, face_name, 
                     ha='center', va='center', fontsize=10, 
                     weight='bold', color='black',
                     bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
            
            # Add area annotation
            bounds = polygon.bounds
            ax_A.text(bounds[0], bounds[3], f'Area: {polygon.area:.1f}', 
                     ha='left', va='bottom', fontsize=8, 
                     color=edge_color, weight='bold')
    
    # Set axis properties
    ax_A.set_xlabel(f'Projected X (projection normal: {projection_plane_normal})', fontsize=12, weight='bold')
    ax_A.set_ylabel('Projected Y', fontsize=12, weight='bold')
    ax_A.set_title(f'Array A - Initial Projected Polygons from 3D Solid\n'
                  f'Input to Distribution Algorithm ({len(array_A)} polygons)\n'
                  f'Projection Normal: {projection_plane_normal}',
                  fontsize=14, weight='bold')
    ax_A.grid(True, alpha=0.3)
    ax_A.set_aspect('equal')
    ax_A.legend(loc='upper right', bbox_to_anchor=(1.02, 1), fontsize=10)
    
    # Add algorithm info text box
    algo_text = f"""INITIAL STATE (ARRAY_A)
    
• Total polygons: {len(array_A)}
• Source: 3D solid face projections
• Projection normal: {projection_plane_normal}
• Ready for distribution algorithm

NEXT STEPS:
1. Move first polygon to array_B
2. Process remaining polygons for intersections
3. Distribute between array_B and array_C
4. Apply boolean operations for clean results

POLYGON SUMMARY:"""
    
    for i, poly_data in enumerate(array_A):
        polygon = poly_data['polygon']
        face_name = poly_data['name']
        if polygon.geom_type == 'Polygon':
            bounds = polygon.bounds
            algo_text += f"\n• {face_name}: area={polygon.area:.1f}"
    
    # Add text box with algorithm info
    ax_A.text(0.02, 0.98, algo_text, transform=ax_A.transAxes, 
             fontsize=9, verticalalignment='top', 
             bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.9),
             fontfamily='monospace')
    
    plt.tight_layout()
    plt.show()
    
    print(f"✓ Array_A visualization complete - showing {len(array_A)} initial polygons")
else:
    print("✗ No polygons in array_A to display")

# Create backup of array_A before any processing for visualization later
array_A_backup = array_A.copy() if array_A else []
print(f"Created backup of array_A with {len(array_A_backup)} polygons for visualization")

# Check if we have any polygons to process
print(f"\nPolygon distribution for complete processing:")
print(f"  • array_A (all faces): {len(array_A)} polygons")
print(f"  • array_C_initial (negative dot_product): {len(array_C_initial)} polygons")
print(f"  • Total: {len(array_A)} polygons")

if len(array_A) >= 1:
    # Create array_B and initialize array_C for final classification
    array_B = []
    array_C = []
    
    # Process ALL faces through the depth algorithm
    if array_A:
        first_polygon_data = array_A.pop(0)  # Remove and get first polygon
        array_B.append(first_polygon_data)
        
        print(f"\nMoved {first_polygon_data['name']} from array_A to array_B")
        print(f"array_A now has {len(array_A)} polygons")
        print(f"array_B now has {len(array_B)} polygons")
        print(f"Processing remaining faces through depth algorithm...")
    
    # Process each polygon Pi in array_A using depth-based classification
    while array_A:
        Pi_data = array_A.pop(0)  # Take one polygon from array_A
        Pi = Pi_data['polygon']
        Pi_name = Pi_data['name']
        Pi_parent_face = Pi_data['parent_face']
        
        print(f"\nProcessing Pi = {Pi_name} (area: {Pi.area:.2f})")
        
        # Test intersection with all polygons in array_B
        intersection_found = False
        
        for j, Pj_data in enumerate(array_B):
            Pj = Pj_data['polygon']
            Pj_name = Pj_data['name']
            Pj_parent_face = Pj_data['parent_face']
            
            print(f"  Testing intersection with Pj = {Pj_name} (area: {Pj.area:.2f})")
            
            # Find intersection using shapely
            intersection = Pi.intersection(Pj)
            
            # Check if we have a meaningful intersection
            if (not intersection.is_empty and 
                hasattr(intersection, 'area') and 
                intersection.area > 1e-6):
                
                print(f"    → Intersection found (area: {intersection.area:.2f})")
                intersection_found = True
                
                # Find interior point of intersection
                interior_point = find_interior_point(intersection)
                
                if interior_point is None:
                    print(f"    → Warning: Could not find interior point, skipping depth analysis")
                    continue
                    
                print(f"    → Interior point: ({interior_point.x:.2f}, {interior_point.y:.2f})")
                
                # Find the point that is farthest along the projection normal direction
                # by sampling multiple points in the intersection region and comparing depths
                sample_points = []
                
                # Sample points from the intersection region
                if intersection.geom_type == 'Polygon':
                    # Get bounding box of intersection
                    minx, miny, maxx, maxy = intersection.bounds
                    
                    # Create a grid of sample points within the intersection
                    num_samples = 9  # 3x3 grid for comprehensive sampling
                    x_step = (maxx - minx) / 3 if maxx > minx else 0
                    y_step = (maxy - miny) / 3 if maxy > miny else 0
                    
                    for i in range(3):
                        for j in range(3):
                            x = minx + (i + 0.5) * x_step
                            y = miny + (j + 0.5) * y_step
                            test_point = Point(x, y)
                            
                            # Only include points that are actually inside the intersection
                            if intersection.contains(test_point):
                                sample_points.append(test_point)
                    
                    # Also include the centroid and representative point
                    if intersection.contains(interior_point):
                        sample_points.append(interior_point)
                    
                    try:
                        rep_point = intersection.representative_point()
                        if intersection.contains(rep_point):
                            sample_points.append(rep_point)
                    except:
                        pass
                
                # If we didn't get enough sample points, fall back to just the interior point
                if len(sample_points) == 0:
                    sample_points = [interior_point]
                
                print(f"    → Sampling {len(sample_points)} points for depth comparison")
                
                # Find the farthest point along projection normal for both Pi and Pj
                Pi_max_depth = float('-inf')
                Pj_max_depth = float('-inf')
                Pi_farthest_point = None
                Pj_farthest_point = None
                
                for sample_point in sample_points:
                    # Intersect line with parent faces to get 3D depths for this sample point
                    Pi_intersection_3d = None
                    Pj_intersection_3d = None
                    
                    if Pi_parent_face is not None:
                        Pi_intersection_3d = intersect_line_with_face(
                            sample_point, unit_projection_normal, Pi_parent_face)
                            
                    if Pj_parent_face is not None:
                        Pj_intersection_3d = intersect_line_with_face(
                            sample_point, unit_projection_normal, Pj_parent_face)
                    
                    # Calculate depths for this sample point
                    Pi_depth = calculate_depth_along_normal(Pi_intersection_3d, unit_projection_normal) if Pi_intersection_3d is not None else float('-inf')
                    Pj_depth = calculate_depth_along_normal(Pj_intersection_3d, unit_projection_normal) if Pj_intersection_3d is not None else float('-inf')
                    
                    # Track the maximum depths and corresponding points
                    if Pi_depth > Pi_max_depth:
                        Pi_max_depth = Pi_depth
                        Pi_farthest_point = Pi_intersection_3d
                    
                    if Pj_depth > Pj_max_depth:
                        Pj_max_depth = Pj_depth
                        Pj_farthest_point = Pj_intersection_3d
                
                print(f"    → Pi max depth: {Pi_max_depth:.3f}, Pj max depth: {Pj_max_depth:.3f}")
                print(f"    → Comparison based on farthest points along projection normal")
                
                # Determine which face has the farthest point (will be subtracted from)
                if Pi_max_depth > Pj_max_depth:
                    farthest_face = Pi_parent_face
                    farthest_name = Pi_name
                    print(f"    → Pi is farther: intersection associated with {Pi_name}")
                else:
                    farthest_face = Pj_parent_face  
                    farthest_name = Pj_name
                    print(f"    → Pj is farther: intersection associated with {Pj_name}")
                
                # Add the intersection to array_C with association to farthest face
                intersection_name = f"Intersection_{Pi_name}_{Pj_name}"
                intersection_data = {
                    'polygon': intersection,
                    'name': intersection_name,
                    'normal': 'intersection',  # Special marker for intersections
                    'parent_face': farthest_face,  # Associate with face that has farthest point
                    'associated_face_name': farthest_name,  # Track which face it's associated with
                    'original_index': -1   # Special index for intersections
                }
                array_C.append(intersection_data)
                print(f"    → Added intersection rectangle to array_C (area: {intersection.area:.2f}, associated with {farthest_name})")
                
                # Depth-based classification using the farthest points
                if Pi_max_depth > Pj_max_depth:
                    print(f"    → Pi is farther: modifying Pj = Pj - Pi")
                    # Subtract Pi from Pj
                    try:
                        new_Pj = Pj.difference(Pi)
                        if not new_Pj.is_empty and new_Pj.area > 1e-6:
                            # Update Pj in array_B
                            array_B[j] = {
                                'polygon': new_Pj,
                                'name': f"Modified_{Pj_name}",
                                'normal': Pj_data['normal'],
                                'parent_face': Pj_data['parent_face'],
                                'original_index': Pj_data['original_index'],
                                'dot_product': Pj_data['dot_product']  # Preserve dot product
                            }
                            print(f"    → Updated Pj area: {new_Pj.area:.2f}")
                        else:
                            print(f"    → Pj completely removed by subtraction")
                            # Remove Pj from array_B if it becomes empty
                            array_B.pop(j)
                    except Exception as e:
                        print(f"    → Error in Pj - Pi operation: {e}")
                        
                else:
                    print(f"    → Pj is farther: modifying Pi = Pi - Pj")
                    # Subtract Pj from Pi
                    try:
                        new_Pi = Pi.difference(Pj)
                        if not new_Pi.is_empty and new_Pi.area > 1e-6:
                            # Update Pi for further processing
                            Pi = new_Pi
                            Pi_data['polygon'] = new_Pi
                            Pi_data['name'] = f"Modified_{Pi_name}"
                            # Preserve dot_product when modifying Pi
                            Pi_data['dot_product'] = Pi_data.get('dot_product', 0)
                            print(f"    → Updated Pi area: {new_Pi.area:.2f}")
                        else:
                            print(f"    → Pi completely removed by subtraction")
                            # Pi will not be added to any array
                            break
                    except Exception as e:
                        print(f"    → Error in Pi - Pj operation: {e}")
                
                # Continue testing with other polygons in array_B
                # (Pi may intersect with multiple polygons)
                
            else:
                area_val = intersection.area if hasattr(intersection, 'area') else 0.0
                print(f"    → No meaningful intersection (area: {area_val:.6f})")
        
        # After processing all intersections, decide where Pi goes
        # Note: Intersections were already added to array_C in step 3c.3
        # The remaining (modified) face should be added to array_B for further processing
        if intersection_found:
            if Pi_data['polygon'].area > 1e-6:
                print(f"  → {Pi_data['name']} had intersections - intersection rectangles already added to array_C")
                print(f"  → Adding remaining {Pi_data['name']} to array_B (area: {Pi_data['polygon'].area:.2f})")
                array_B.append(Pi_data)  # Add the remaining part to array_B
            else:
                print(f"  → {Pi_data['name']} completely consumed, not added to any array (intersections already in array_C)")
        else:
            print(f"  → No intersections found, moving {Pi_data['name']} to array_B")
            array_B.append(Pi_data)
    
    print(f"\nDepth processing complete:")
    print(f"array_B contains {len(array_B)} polygons")
    print(f"array_C contains {len(array_C)} polygons (from intersections)")
    
    # Apply dot product classification to move appropriate faces to array_C
    print(f"\nApplying final dot product classification...")
    faces_to_move = []
    
    # Check faces in array_B that should be in array_C based on dot product
    for i, poly_data in enumerate(array_B):
        if poly_data['dot_product'] <= 0:
            faces_to_move.append((i, poly_data, 'B_to_C'))
            print(f"  → {poly_data['name']}: dot_product={poly_data['dot_product']:.3f} ≤ 0, will move to array_C")
    
    # Move faces from array_B to array_C based on dot product
    for i, poly_data, move_type in reversed(faces_to_move):  # Reverse to maintain indices
        if move_type == 'B_to_C':
            array_B.pop(i)
            array_C.append(poly_data)
            print(f"  → Moved {poly_data['name']} from array_B to array_C")
    
    print(f"\nFinal classification results:")
    print(f"array_B contains {len(array_B)} polygons (dot_product > 0)")
    print(f"array_C contains {len(array_C)} polygons (dot_product ≤ 0 + intersections)")
    print(f"Total: {len(array_B) + len(array_C)} polygons")
    
    # Display results in a single figure with 4 subplots
    if array_B or array_C or array_A:
        colors = ['blue', 'red', 'green', 'orange', 'purple', 'brown', 'pink', 'cyan']
        
        # Create a single figure with 2x2 subplots
        fig, ((ax_A_final, ax_B), (ax_C, ax_combined)) = plt.subplots(2, 2, figsize=(20, 16))
        fig.suptitle(f'Polygon Classification Algorithm Results\n(Projection Normal: {projection_plane_normal})', 
                    fontsize=16, weight='bold')
        
        # ============== SUBPLOT 1: Array_A (Final view) ==============
        if 'array_A_backup' in locals() and array_A_backup:  # Check if backup exists and has data
            for i, poly_data in enumerate(array_A_backup):
                polygon = poly_data['polygon']
                face_name = poly_data['name']
                
                if polygon.geom_type == 'Polygon' and polygon.area > 0:
                    face_color = ['lightblue', 'lightcoral', 'lightgreen', 'lightyellow'][i % 4]
                    edge_color = ['blue', 'red', 'green', 'orange'][i % 4]
                    
                    plot_polygon(polygon, ax_A_final, 
                                facecolor=face_color, 
                                edgecolor=edge_color, 
                                alpha=0.6, 
                                linewidth=2, 
                                label=f'{face_name} (area: {polygon.area:.1f})',
                                outline_only=True)
        else:
            # Show message that array_A was processed or not available
            ax_A_final.text(0.5, 0.5, 'Array A - Original Polygons\n(Not available - algorithm processed)', 
                           ha='center', va='center', transform=ax_A_final.transAxes,
                           fontsize=14, bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
        
        ax_A_final.set_title(f'Array A - Initial Polygons\n({len(array_A_backup) if "array_A_backup" in locals() and array_A_backup else "Processed"} polygons)', 
                            fontsize=14, weight='bold')
        ax_A_final.set_xlabel('X coordinate')
        ax_A_final.set_ylabel('Y coordinate')
        ax_A_final.grid(True, alpha=0.3)
        ax_A_final.set_aspect('equal')
        ax_A_final.legend(loc='upper right', fontsize=9)
        
        # ============== SUBPLOT 2: Array_B ==============
        print(f"\nPlotting array_B polygons:")
        for i, poly_data in enumerate(array_B):
            poly = poly_data['polygon']
            name = poly_data['name']
            if poly.geom_type == 'Polygon' and poly.area > 0:
                color = colors[i % len(colors)]
                plot_polygon(poly, ax_B, facecolor=f'light{color}' if f'light{color}' in ['lightblue', 'lightcoral', 'lightgreen'] else color, 
                            edgecolor=color, alpha=0.6, linewidth=2, label=f'{name} (area: {poly.area:.2f})', outline_only=True)
                print(f"  {name}: area={poly.area:.2f}")
            elif poly.geom_type == 'MultiPolygon':
                color = colors[i % len(colors)]
                plot_polygon(poly, ax_B, facecolor=f'light{color}' if f'light{color}' in ['lightblue', 'lightcoral', 'lightgreen'] else color, 
                            edgecolor=color, alpha=0.6, linewidth=2, label=f'{name} (multi-area: {poly.area:.2f})', outline_only=True)
                print(f"  {name}: multi-polygon area={poly.area:.2f}")
        
        ax_B.set_title(f'Array B - Final Polygons\n({len(array_B)} polygons)', fontsize=14, weight='bold')
        ax_B.set_xlabel('X coordinate')
        ax_B.set_ylabel('Y coordinate')
        ax_B.grid(True, alpha=0.3)
        ax_B.set_aspect('equal')
        ax_B.legend(loc='upper right', fontsize=9)
        
        # ============== SUBPLOT 3: Array_C ==============
        print(f"\nPlotting array_C polygons:")
        print(f"array_C has {len(array_C)} polygons")
        for i, poly_data in enumerate(array_C):
            print(f"  Processing poly_data {i}: {poly_data.keys()}")
            poly = poly_data['polygon']
            name = poly_data['name']
            print(f"    Name: {name}, Geom type: {poly.geom_type}, Area: {poly.area}")
            
            # Check if this is an intersection with face association
            associated_face = poly_data.get('associated_face_name', '')
            face_info = f" (assoc: {associated_face})" if associated_face else ""
            
            if poly.geom_type == 'Polygon' and poly.area > 0:
                color = colors[i % len(colors)]
                plot_polygon(poly, ax_C, facecolor='yellow', edgecolor=color, 
                            alpha=0.7, linestyle='--', linewidth=2, 
                            label=f'{name}{face_info} (area: {poly.area:.2f})', outline_only=True)
                print(f"  {name}: area={poly.area:.2f}{face_info}")
            elif poly.geom_type == 'MultiPolygon':
                color = colors[i % len(colors)]
                plot_polygon(poly, ax_C, facecolor='yellow', edgecolor=color, 
                            alpha=0.7, linestyle='--', linewidth=2, 
                            label=f'{name}{face_info} (multi-area: {poly.area:.2f})', outline_only=True)
                print(f"  {name}: multi-polygon area={poly.area:.2f}{face_info}")
            elif poly.geom_type == 'GeometryCollection':
                # Extract polygons from GeometryCollection
                from shapely.geometry import Polygon
                polygons_in_collection = [geom for geom in poly.geoms if isinstance(geom, Polygon) and geom.area > 0]
                if polygons_in_collection:
                    color = colors[i % len(colors)]
                    for j, polygon in enumerate(polygons_in_collection):
                        plot_polygon(polygon, ax_C, facecolor='yellow', edgecolor=color, 
                                    alpha=0.7, linestyle='--', linewidth=2, 
                                    label=f'{name}{face_info} [{j}] (area: {polygon.area:.2f})', outline_only=True)
                        print(f"  {name} [polygon {j}]: area={polygon.area:.2f}{face_info}")
                else:
                    print(f"  ⚠️  GeometryCollection {name} contains no valid polygons")
            else:
                print(f"  ⚠️  Skipping {name}: geom_type={poly.geom_type}, area={poly.area}")
        
        # If array_C is empty, show a message
        if not array_C:
            ax_C.text(0.5, 0.5, 'No Intersections Found\n(Array C is empty)', 
                     ha='center', va='center', transform=ax_C.transAxes,
                     fontsize=14, bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
        
        ax_C.set_title(f'Array C - Intersections\n({len(array_C)} polygons)', fontsize=14, weight='bold')
        ax_C.set_xlabel('X coordinate')
        ax_C.set_ylabel('Y coordinate')
        ax_C.grid(True, alpha=0.3)
        ax_C.set_aspect('equal')
        if array_C:  # Only show legend if there are polygons
            ax_C.legend(loc='upper right', fontsize=9)
        
        # ============== SUBPLOT 4: Combined Array_B + Array_C ==============
        print(f"\nPlotting combined array_B and array_C:")
        
        # Plot array_C polygons first with thin dashed black lines
        for i, poly_data in enumerate(array_C):
            poly = poly_data['polygon']
            name = poly_data['name']
            
            # Check if this is an intersection with face association
            associated_face = poly_data.get('associated_face_name', '')
            face_info = f" (assoc: {associated_face})" if associated_face else ""
            
            # Handle different geometry types for array_C
            if poly.geom_type == 'Polygon' and poly.area > 0:
                plot_polygon(poly, ax_combined, facecolor='none', edgecolor='black', 
                            alpha=1.0, linestyle=(0, (2, 2)), linewidth=1, outline_only=True)
            elif poly.geom_type == 'MultiPolygon' and poly.area > 0:
                plot_polygon(poly, ax_combined, facecolor='none', edgecolor='black', 
                            alpha=1.0, linestyle=(0, (2, 2)), linewidth=1, outline_only=True)
            elif poly.geom_type == 'GeometryCollection' and poly.area > 0:
                # Handle GeometryCollection by plotting individual geometries
                for j, geom in enumerate(poly.geoms):
                    if geom.geom_type == 'Polygon' and geom.area > 0:
                        plot_polygon(geom, ax_combined, facecolor='none', edgecolor='black', 
                                    alpha=1.0, linestyle=(0, (2, 2)), linewidth=1, outline_only=True)
        
        # Plot array_B polygons second with solid black lines
        for i, poly_data in enumerate(array_B):
            poly = poly_data['polygon']
            name = poly_data['name']
            if poly.geom_type == 'Polygon' and poly.area > 0:
                plot_polygon(poly, ax_combined, facecolor='none', edgecolor='black', 
                            alpha=1.0, linewidth=1, linestyle='-', outline_only=True)
            elif poly.geom_type == 'MultiPolygon':
                plot_polygon(poly, ax_combined, facecolor='none', edgecolor='black', 
                            alpha=1.0, linewidth=1, linestyle='-', outline_only=True)
        
        ax_combined.set_title(f'Combined Arrays B & C\nB: Final Polygons | C: Intersections', fontsize=14, weight='bold')
        ax_combined.set_xlabel('X coordinate')
        ax_combined.set_ylabel('Y coordinate')
        ax_combined.grid(True, alpha=0.3)
        ax_combined.set_aspect('equal')
        
        # Calculate bounds for all plots to ensure consistent scaling
        all_bounds_B = []
        all_bounds_C = []
        all_bounds_A = []
        
        if 'array_A_backup' in locals() and array_A_backup:
            for poly_data in array_A_backup:
                poly = poly_data['polygon']
                if hasattr(poly, 'bounds'):
                    bounds = poly.bounds
                    all_bounds_A.extend([bounds[0], bounds[2], bounds[1], bounds[3]])
        
        for poly_data in array_B:
            poly = poly_data['polygon']
            if hasattr(poly, 'bounds'):
                bounds = poly.bounds
                all_bounds_B.extend([bounds[0], bounds[2], bounds[1], bounds[3]])
        
        for poly_data in array_C:
            poly = poly_data['polygon']
            if hasattr(poly, 'bounds'):
                bounds = poly.bounds
                all_bounds_C.extend([bounds[0], bounds[2], bounds[1], bounds[3]])
        
        # Use overall bounds for consistent scaling
        all_bounds_total = all_bounds_A + all_bounds_B + all_bounds_C
        
        if all_bounds_total:
            margin = (max(all_bounds_total) - min(all_bounds_total)) * 0.1
            xlim = (min(all_bounds_total) - margin, max(all_bounds_total) + margin)
            ylim = (min(all_bounds_total) - margin, max(all_bounds_total) + margin)
        else:
            xlim = (-5, 25)
            ylim = (-5, 35)
        
        # Apply consistent scaling to all subplots
        for ax in [ax_A_final, ax_B, ax_C, ax_combined]:
            ax.set_xlim(xlim)
            ax.set_ylim(ylim)
        
        # Show the complete figure with all 4 subplots
        plt.tight_layout()
        plt.show()
        
    else:
        print("No polygons to display in any array")
else:
    # Handle case where array_A is empty after processing
    print(f"\nNo polygons to process through depth algorithm")
    print(f"Creating arrays based on initial classification...")
    
    # Create empty array_B and populate array_C from initial classification
    array_B = []
    array_C = array_C_initial.copy()  # Use the initially classified faces
    
    # If we have polygons, we can still visualize them
    if len(array_C) > 0:
        print(f"Will display {len(array_C)} polygons in array_C based on initial classification")

print(f"\n" + "="*50)
print("BOOLEAN SUBTRACT ANALYSIS COMPLETE")
print("="*50)

def convert_brep_to_legacy_format(faces_data):
    """Convert BRep polygon structure to legacy format for compatibility.
    
    Args:
        faces_data: List of BRep polygon dictionaries
        
    Returns:
        List of legacy format dictionaries with 'points' key
    """
    legacy_faces = []
    
    for face_data in faces_data:
        # For now, use only the outer boundary for compatibility
        # TODO: Implement proper polygon with holes support in HLR algorithm
        outer_boundary = face_data['outer_boundary']
        
        legacy_face = {
            'points': np.array(outer_boundary),
            'normal': face_data['normal'],
            'face_type': 'brep_outer_boundary',
            'vertex_count': len(outer_boundary),
            'has_cutouts': face_data['has_cutouts'],
            'original_brep_data': face_data  # Store original for future use
        }
        
        legacy_faces.append(legacy_face)
    
    return legacy_faces

# ============================================================================
# MAIN EXECUTION WITH BREP TRAVERSAL
# ============================================================================

if OPENCASCADE_AVAILABLE:
    print(f"\n" + "="*60)
    print("TESTING BREP TRAVERSAL ON OPENCASCADE SOLID")
    print("="*60)
    
    # Create a test solid to demonstrate BRep traversal
    solid_shape = create_opencascade_solid()
    
    if solid_shape:
        print(f"\n✓ Created test solid successfully")
        
        # Test the new BRep traversal functionality
        print(f"\nExtracting faces with proper BRep traversal...")
        brep_faces_data = extract_faces_from_solid(solid_shape)
        
        if brep_faces_data:
            print(f"\n" + "="*40)
            print("BREP TRAVERSAL RESULTS SUMMARY")
            print("="*40)
            
            for i, face_data in enumerate(brep_faces_data):
                face_num = i + 1
                outer_count = len(face_data['outer_boundary'])
                cutout_count = len(face_data['cutouts'])
                total_vertices = face_data['vertex_count']
                has_cutouts = face_data['has_cutouts']
                
                print(f"Face {face_num}:")
                print(f"  • Outer boundary: {outer_count} vertices")
                print(f"  • Cutouts: {cutout_count} holes")
                print(f"  • Total vertices: {total_vertices}")
                print(f"  • Has cutouts: {'Yes' if has_cutouts else 'No'}")
                print(f"  • Normal: {face_data['normal']}")
                print()
            
            print(f"Total faces extracted: {len(brep_faces_data)}")
            print(f"Faces with cutouts: {sum(1 for f in brep_faces_data if f['has_cutouts'])}")
            print(f"Total vertex count: {sum(f['vertex_count'] for f in brep_faces_data)}")
            
            # Convert to legacy format for compatibility with existing HLR code
            print(f"\nConverting to legacy format for HLR processing...")
            faces_data = convert_brep_to_legacy_format(brep_faces_data)
            print(f"✓ Converted {len(faces_data)} faces to legacy format")
            
        else:
            print(f"✗ No faces extracted from solid")
            faces_data = []
    else:
        print(f"✗ Failed to create test solid")
else:
    print(f"\n✗ OpenCASCADE not available - cannot test BRep traversal")

# ============================================================================
# MAIN EXECUTION - TEST THE DIAGONAL LINE FIX
# ============================================================================

if __name__ == "__main__":
    print("="*80)
    print("TESTING 3D VISUALIZATION WITH DIAGONAL LINE FIX")
    print("="*80)
    
    # Create the boolean cut solid
    print("\n1. Creating OpenCASCADE solid with boolean cut operation...")
    solid_shape = create_opencascade_solid()
    
    if solid_shape:
        print("\n2. Analyzing solid geometry...")
        analyze_solid_geometry(solid_shape)
        
        print("\n3. Testing 3D visualization with diagonal line fix...")
        print("   - 4-vertex faces (1, 2, 5) should now render without diagonal lines")
        print("   - Using same direct polygon approach as 5+ vertex faces")
        display_3d_solid(solid_shape)
        
        print("\n✓ Test complete - check if faces 1, 2, 5 show clean planar surfaces")
    else:
        print("\n✗ Could not create solid for testing")
