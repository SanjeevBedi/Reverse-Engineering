from shapely.geometry import Polygon, Point
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from itertools import combinations
import random

# OpenCASCADE imports
try:
    from OCC.Core.gp import gp_Pnt, gp_Vec, gp_Ax2, gp_Dir
    from OCC.Core.BRepPrimAPI import BRepPrimAPI_MakeBox
    from OCC.Core.BRepAlgoAPI import BRepAlgoAPI_Cut
    from OCC.Core.TopExp import TopExp_Explorer
    from OCC.Core.TopAbs import TopAbs_FACE, TopAbs_SHELL, TopAbs_WIRE, TopAbs_EDGE, TopAbs_VERTEX
    from OCC.Core.BRep import BRep_Tool
    from OCC.Core.TopLoc import TopLoc_Location
    from OCC.Core.gp import gp_Trsf, gp_XYZ
    from OCC.Core.TopoDS import topods
    
    # Try to import TopExp for vertex extraction
    try:
        from OCC.Core.TopExp import TopExp
        TOPEXP_AVAILABLE = True
    except:
        TOPEXP_AVAILABLE = False
    
    OPENCASCADE_AVAILABLE = True
except ImportError as e:
    print(f"OpenCASCADE not available: {e}")
    OPENCASCADE_AVAILABLE = False
    TOPEXP_AVAILABLE = False

# ============================================================================
# SOLID GENERATOR - THREE COMPLEXITY LEVELS
# ============================================================================

def create_simple_solid():
    """
    SIMPLE: Base cuboid with secondary cuboid subtracted.
    Secondary cuboid's top face is above base cuboid and two side faces are outside.
    """
    if not OPENCASCADE_AVAILABLE:
        print("✗ OpenCASCADE not available")
        return None
    
    print("\n" + "="*60)
    print("CREATING SIMPLE SOLID")
    print("="*60)
    
    # Base cuboid: 20x20x10 (larger base)
    print("Creating base cuboid (20x20x10)...")
    base_cuboid = BRepPrimAPI_MakeBox(20, 20, 10).Shape()
    
    # Secondary cuboid: 8x8x15 (taller, narrower)
    # Position it so top face is above base and two sides are outside
    print("Creating secondary cuboid (8x8x15)...")
    secondary_cuboid = BRepPrimAPI_MakeBox(8, 8, 15).Shape()
    
    # Transform to position: center at (16, 16, 0) 
    # This places it so:
    # - X: 16-4=12 to 16+4=20 (right edge aligns with base, left side extends outside)
    # - Y: 16-4=12 to 16+4=20 (back edge aligns with base, front side extends outside)  
    # - Z: 0 to 15 (top face extends above base cuboid height of 10)
    transform = gp_Trsf()
    transform.SetTranslation(gp_Vec(16, 16, 0))
    secondary_cuboid.Move(TopLoc_Location(transform))
    
    print("Applied transformation: translation (16, 16, 0)")
    print("  → Secondary cuboid positioned with top face above base")
    print("  → Two side faces extend outside base cuboid")
    
    # Perform boolean subtraction
    try:
        print("Performing boolean CUT operation...")
        cut_op = BRepAlgoAPI_Cut(base_cuboid, secondary_cuboid)
        cut_op.Build()
        
        if cut_op.IsDone() and not cut_op.HasErrors():
            result = cut_op.Shape()
            print("✓ Simple solid created successfully")
            return result
        else:
            print("✗ Boolean operation failed")
            return base_cuboid
    except Exception as e:
        print(f"✗ Error: {e}")
        return base_cuboid

def create_medium_solid():
    """
    MEDIUM: Base cuboid with one secondary cuboid subtracted to make a hole.
    Secondary cuboid is completely inside the base cuboid.
    """
    if not OPENCASCADE_AVAILABLE:
        print("✗ OpenCASCADE not available")
        return None
    
    print("\n" + "="*60)
    print("CREATING MEDIUM SOLID")
    print("="*60)
    
    # Base cuboid: 25x25x12
    print("Creating base cuboid (25x25x12)...")
    base_cuboid = BRepPrimAPI_MakeBox(25, 25, 12).Shape()
    
    # Secondary cuboid: 10x10x8 (hole in center)
    # Position it completely inside the base
    print("Creating secondary cuboid for hole (10x10x8)...")
    hole_cuboid = BRepPrimAPI_MakeBox(10, 10, 8).Shape()
    
    # Transform to center: (7.5, 7.5, 2)
    # This creates a hole with walls of thickness 7.5 on all sides
    # and 2 units from bottom, 2 units from top
    transform = gp_Trsf()
    transform.SetTranslation(gp_Vec(7.5, 7.5, 2))
    hole_cuboid.Move(TopLoc_Location(transform))
    
    print("Applied transformation: translation (7.5, 7.5, 2)")
    print("  → Creates centered hole with uniform wall thickness")
    
    # Perform boolean subtraction
    try:
        print("Performing boolean CUT operation...")
        cut_op = BRepAlgoAPI_Cut(base_cuboid, hole_cuboid)
        cut_op.Build()
        
        if cut_op.IsDone() and not cut_op.HasErrors():
            result = cut_op.Shape()
            print("✓ Medium solid created successfully")
            return result
        else:
            print("✗ Boolean operation failed")
            return base_cuboid
    except Exception as e:
        print(f"✗ Error: {e}")
        return base_cuboid

def create_complex_solid():
    """
    COMPLEX: Base cuboid with two secondary cuboids subtracted to form two holes.
    """
    if not OPENCASCADE_AVAILABLE:
        print("✗ OpenCASCADE not available")
        return None
    
    print("\n" + "="*60)
    print("CREATING COMPLEX SOLID")
    print("="*60)
    
    # Base cuboid: 30x20x15
    print("Creating base cuboid (30x20x15)...")
    base_cuboid = BRepPrimAPI_MakeBox(30, 20, 15).Shape()
    
    # First hole: 8x6x10 
    print("Creating first hole cuboid (8x6x10)...")
    hole1_cuboid = BRepPrimAPI_MakeBox(8, 6, 10).Shape()
    
    # Position first hole at (5, 3, 2.5)
    transform1 = gp_Trsf()
    transform1.SetTranslation(gp_Vec(5, 3, 2.5))
    hole1_cuboid.Move(TopLoc_Location(transform1))
    
    print("Applied transformation to hole 1: translation (5, 3, 2.5)")
    
    # Second hole: 6x8x12
    print("Creating second hole cuboid (6x8x12)...")
    hole2_cuboid = BRepPrimAPI_MakeBox(6, 8, 12).Shape()
    
    # Position second hole at (17, 6, 1.5)
    transform2 = gp_Trsf()
    transform2.SetTranslation(gp_Vec(17, 6, 1.5))
    hole2_cuboid.Move(TopLoc_Location(transform2))
    
    print("Applied transformation to hole 2: translation (17, 6, 1.5)")
    print("  → Two holes positioned to avoid overlap")
    
    # Perform boolean subtraction - first hole
    try:
        print("Performing first boolean CUT operation...")
        cut_op1 = BRepAlgoAPI_Cut(base_cuboid, hole1_cuboid)
        cut_op1.Build()
        
        if cut_op1.IsDone() and not cut_op1.HasErrors():
            intermediate_result = cut_op1.Shape()
            print("✓ First hole created successfully")
            
            # Perform second boolean subtraction
            print("Performing second boolean CUT operation...")
            cut_op2 = BRepAlgoAPI_Cut(intermediate_result, hole2_cuboid)
            cut_op2.Build()
            
            if cut_op2.IsDone() and not cut_op2.HasErrors():
                final_result = cut_op2.Shape()
                print("✓ Complex solid created successfully")
                return final_result
            else:
                print("✗ Second boolean operation failed")
                return intermediate_result
        else:
            print("✗ First boolean operation failed")
            return base_cuboid
    except Exception as e:
        print(f"✗ Error: {e}")
        return base_cuboid

def analyze_solid_topology(solid, solid_name):
    """Analyze and display topology information for a solid."""
    if not OPENCASCADE_AVAILABLE or solid is None:
        return
    
    print(f"\n--- {solid_name.upper()} TOPOLOGY ANALYSIS ---")
    
    try:
        # Count topological elements
        shell_count = 0
        face_count = 0
        edge_count = 0
        vertex_count = 0
        
        # Count shells
        shell_explorer = TopExp_Explorer(solid, TopAbs_SHELL)
        while shell_explorer.More():
            shell_count += 1
            shell_explorer.Next()
        
        # Count faces
        face_explorer = TopExp_Explorer(solid, TopAbs_FACE)
        while face_explorer.More():
            face_count += 1
            face_explorer.Next()
        
        # Count edges
        edge_explorer = TopExp_Explorer(solid, TopAbs_EDGE)
        while edge_explorer.More():
            edge_count += 1
            edge_explorer.Next()
        
        # Count vertices
        vertex_explorer = TopExp_Explorer(solid, TopAbs_VERTEX)
        while vertex_explorer.More():
            vertex_count += 1
            vertex_explorer.Next()
        
        print(f"  Shells: {shell_count}")
        print(f"  Faces: {face_count}")
        print(f"  Edges: {edge_count}")
        print(f"  Vertices: {vertex_count}")
        
        # Calculate bounding box
        try:
            from OCC.Core.Bnd import Bnd_Box
            from OCC.Core.BRepBndLib import brepbndlib
            
            bbox = Bnd_Box()
            brepbndlib.Add(solid, bbox)
            
            if not bbox.IsVoid():
                xmin, ymin, zmin, xmax, ymax, zmax = bbox.Get()
                width = xmax - xmin
                height = ymax - ymin
                depth = zmax - zmin
                
                print(f"  Bounding Box: {width:.1f} x {height:.1f} x {depth:.1f}")
                print(f"  Volume estimate: {width * height * depth:.1f} cubic units")
        except Exception as e:
            print(f"  Bounding box calculation failed: {e}")
            
    except Exception as e:
        print(f"  Analysis failed: {e}")

def extract_face_data_simple(solid):
    """Extract face data from solid with simplified approach for HLR processing."""
    if not OPENCASCADE_AVAILABLE or solid is None:
        return []
    
    faces = []
    
    # Explore faces in the solid
    face_explorer = TopExp_Explorer(solid, TopAbs_FACE)
    face_count = 0
    
    while face_explorer.More():
        face_shape = face_explorer.Current()
        face_count += 1
        
        try:
            face = topods.Face(face_shape)
            
            # Extract vertices using basic traversal
            vertices = []
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
            
            if len(vertices) >= 3:
                # Calculate face normal using cross product
                if len(vertices) >= 3:
                    v1 = np.array(vertices[1]) - np.array(vertices[0])
                    v2 = np.array(vertices[2]) - np.array(vertices[0])
                    normal = np.cross(v1, v2)
                    if np.linalg.norm(normal) > 1e-10:
                        normal = normal / np.linalg.norm(normal)
                    else:
                        normal = np.array([0, 0, 1])  # Default normal
                
                faces.append({
                    'points': np.array(vertices),
                    'normal': normal,
                    'vertex_count': len(vertices),
                    'face_id': face_count
                })
                
        except Exception as e:
            print(f"    Error processing face {face_count}: {e}")
        
        face_explorer.Next()
    
    return faces

def process_solid_hlr(solid, projection_normal, solid_name="Solid"):
    """
    Process a solid with Hidden Line Removal algorithm to generate array_B and array_C.
    
    Args:
        solid: OpenCASCADE solid shape
        projection_normal: numpy array [x, y, z] for projection direction
        solid_name: name for debugging output
    
    Returns:
        tuple: (array_B, array_C) containing processed polygons
    """
    if not OPENCASCADE_AVAILABLE or solid is None:
        print(f"✗ Cannot process {solid_name} - OpenCASCADE not available or solid is None")
        return [], []
    
    print(f"\n" + "="*60)
    print(f"HLR PROCESSING: {solid_name.upper()}")
    print("="*60)
    
    # Extract faces from solid
    faces = extract_face_data_simple(solid)
    print(f"Extracted {len(faces)} faces from {solid_name}")
    
    if not faces:
        return [], []
    
    # Normalize projection normal
    projection_normal = np.array(projection_normal)
    unit_projection_normal = projection_normal / np.linalg.norm(projection_normal)
    print(f"Projection direction: {unit_projection_normal}")
    
    # Project faces and create polygons
    polygons = []
    for i, face_data in enumerate(faces):
        face_points = face_data['points']
        face_normal = face_data['normal']
        
        # Calculate dot product for face classification
        dot_product = np.dot(face_normal, unit_projection_normal)
        
        # Project face to 2D plane perpendicular to projection normal
        projected_points = project_to_2d_plane(face_points, unit_projection_normal)
        
        if len(projected_points) >= 3:
            try:
                polygon = Polygon(projected_points)
                if polygon.is_valid and polygon.area > 1e-6:
                    polygons.append({
                        'polygon': polygon,
                        'name': f'Face_{i+1}',
                        'normal': face_normal,
                        'dot_product': dot_product,
                        'original_index': i,
                        'face_id': face_data['face_id']
                    })
            except Exception as e:
                print(f"  Error creating polygon for face {i+1}: {e}")
    
    print(f"Created {len(polygons)} valid polygons")
    
    # Classify polygons based on dot product
    array_A = []  # For processing
    array_C_initial = []  # Negative dot product faces
    
    for poly_data in polygons:
        if poly_data['dot_product'] <= 0:
            array_C_initial.append(poly_data)
        array_A.append(poly_data)  # All faces go to array_A for processing
    
    print(f"  array_A (all faces): {len(array_A)} polygons")
    print(f"  array_C_initial (negative dot_product): {len(array_C_initial)} polygons")
    
    # Initialize arrays
    array_B = []
    array_C = []
    
    if not array_A:
        return array_B, array_C
    
    # Move first polygon to array_B as seed
    first_polygon = array_A.pop(0)
    array_B.append(first_polygon)
    print(f"Moved {first_polygon['name']} from array_A to array_B as seed")
    
    # Process remaining polygons
    while array_A:
        Pi_data = array_A.pop(0)
        Pi = Pi_data['polygon']
        Pi_name = Pi_data['name']
        
        print(f"Processing {Pi_name} (area: {Pi.area:.2f})")
        
        had_intersection = False
        
        # Test intersection with all polygons in array_B
        for j, Pj_data in enumerate(array_B):
            Pj = Pj_data['polygon']
            Pj_name = Pj_data['name']
            
            try:
                intersection = Pi.intersection(Pj)
                if hasattr(intersection, 'area') and intersection.area > 1e-6:
                    print(f"  → Intersection found with {Pj_name} (area: {intersection.area:.2f})")
                    
                    # Add intersection to array_C
                    array_C.append({
                        'polygon': intersection,
                        'name': f"Intersection_{Pi_name}_{Pj_name}",
                        'normal': Pi_data['normal'],  # Use Pi's normal
                        'original_index': -1
                    })
                    
                    had_intersection = True
            except Exception as e:
                print(f"  → Error testing intersection with {Pj_name}: {e}")
        
        # Add polygon to array_B regardless of intersections
        array_B.append(Pi_data)
        
        if had_intersection:
            print(f"  → {Pi_name} had intersections - added to array_B")
        else:
            print(f"  → No intersections found, added {Pi_name} to array_B")
    
    # Apply final dot product classification
    faces_to_move = []
    for i, poly_data in enumerate(array_B):
        if poly_data['dot_product'] <= 0:
            faces_to_move.append(i)
    
    # Move faces with negative dot product to array_C
    for i in reversed(faces_to_move):
        moved_face = array_B.pop(i)
        array_C.append(moved_face)
        print(f"  → Moved {moved_face['name']} from array_B to array_C (dot_product ≤ 0)")
    
    print(f"\nFinal classification:")
    print(f"  array_B: {len(array_B)} polygons (visible faces)")
    print(f"  array_C: {len(array_C)} polygons (hidden faces + intersections)")
    
    return array_B, array_C

def project_to_2d_plane(points_3d, projection_normal):
    """Project 3D points to 2D plane perpendicular to projection normal."""
    # Create two orthogonal vectors in the projection plane
    # Choose an arbitrary vector not parallel to projection_normal
    if abs(projection_normal[0]) < 0.9:
        temp_vector = np.array([1, 0, 0])
    else:
        temp_vector = np.array([0, 1, 0])
    
    # Create orthogonal basis vectors
    u = np.cross(projection_normal, temp_vector)
    u = u / np.linalg.norm(u)
    v = np.cross(projection_normal, u)
    v = v / np.linalg.norm(v)
    
    # Project points
    projected_points = []
    for point in points_3d:
        # Project to 2D using dot products with basis vectors
        x = np.dot(point, u)
        y = np.dot(point, v)
        projected_points.append([x, y])
    
    return projected_points

def visualize_results(array_B, array_C, solid_name):
    """Visualize the HLR results."""
    if not array_B and not array_C:
        print(f"No polygons to visualize for {solid_name}")
        return
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    
    # Array B (visible faces)
    ax1.set_title(f'{solid_name} - Array B (Visible Faces)')
    ax1.set_aspect('equal')
    ax1.grid(True, alpha=0.3)
    
    colors_b = ['lightblue', 'lightcoral', 'lightgreen', 'lightyellow', 'lightpink']
    for i, poly_data in enumerate(array_B):
        try:
            polygon = poly_data['polygon']
            color = colors_b[i % len(colors_b)]
            
            if polygon.geom_type == 'Polygon':
                x, y = polygon.exterior.xy
                ax1.fill(x, y, color=color, alpha=0.6, edgecolor='black', linewidth=1)
                ax1.text(polygon.centroid.x, polygon.centroid.y, poly_data['name'], 
                        ha='center', va='center', fontsize=8)
        except Exception as e:
            print(f"Error plotting {poly_data['name']}: {e}")
    
    # Array C (hidden faces + intersections)
    ax2.set_title(f'{solid_name} - Array C (Hidden + Intersections)')
    ax2.set_aspect('equal')
    ax2.grid(True, alpha=0.3)
    
    colors_c = ['lightgray', 'lightsteelblue', 'lightcyan', 'lavender', 'mistyrose']
    for i, poly_data in enumerate(array_C):
        try:
            polygon = poly_data['polygon']
            color = colors_c[i % len(colors_c)]
            
            if polygon.geom_type == 'Polygon':
                x, y = polygon.exterior.xy
                ax2.fill(x, y, color=color, alpha=0.6, edgecolor='red', linewidth=1)
                ax2.text(polygon.centroid.x, polygon.centroid.y, poly_data['name'], 
                        ha='center', va='center', fontsize=8)
        except Exception as e:
            print(f"Error plotting {poly_data['name']}: {e}")
    
    # Combined view
    ax3.set_title(f'{solid_name} - Combined View')
    ax3.set_aspect('equal')
    ax3.grid(True, alpha=0.3)
    
    # Plot array_B
    for i, poly_data in enumerate(array_B):
        try:
            polygon = poly_data['polygon']
            color = colors_b[i % len(colors_b)]
            
            if polygon.geom_type == 'Polygon':
                x, y = polygon.exterior.xy
                ax3.fill(x, y, color=color, alpha=0.4, edgecolor='blue', linewidth=2)
        except:
            pass
    
    # Plot array_C
    for i, poly_data in enumerate(array_C):
        try:
            polygon = poly_data['polygon']
            color = colors_c[i % len(colors_c)]
            
            if polygon.geom_type == 'Polygon':
                x, y = polygon.exterior.xy
                ax3.fill(x, y, color=color, alpha=0.4, edgecolor='red', linewidth=1)
        except:
            pass
    
    # Statistics
    ax4.axis('off')
    stats_text = f"""PROCESSING STATISTICS

Solid: {solid_name}

Array B (Visible):
  • Polygons: {len(array_B)}
  • Total Area: {sum(p['polygon'].area for p in array_B if hasattr(p['polygon'], 'area')):.2f}

Array C (Hidden + Intersections):
  • Polygons: {len(array_C)}
  • Total Area: {sum(p['polygon'].area for p in array_C if hasattr(p['polygon'], 'area')):.2f}

Total Polygons: {len(array_B) + len(array_C)}

Legend:
  Blue outlines = Visible (Array B)
  Red outlines = Hidden/Intersections (Array C)
    """
    
    ax4.text(0.05, 0.95, stats_text, transform=ax4.transAxes, 
            fontsize=11, verticalalignment='top', 
            bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8),
            fontfamily='monospace')
    
    plt.tight_layout()
    plt.show()

def main():
    """Main function to demonstrate the three solid types and HLR processing."""
    print("="*60)
    print("SOLID GENERATOR AND HLR PROCESSOR")
    print("="*60)
    
    if not OPENCASCADE_AVAILABLE:
        print("✗ This demo requires OpenCASCADE to be installed")
        return
    
    # Create the three solids
    simple_solid = create_simple_solid()
    medium_solid = create_medium_solid()
    complex_solid = create_complex_solid()
    
    # Analyze topology of each solid
    if simple_solid:
        analyze_solid_topology(simple_solid, "Simple")
    if medium_solid:
        analyze_solid_topology(medium_solid, "Medium")
    if complex_solid:
        analyze_solid_topology(complex_solid, "Complex")
    
    # Define projection normal (45-degree angle view)
    projection_normal = [1, 1, 1]
    
    print(f"\n" + "="*60)
    print("PROCESSING SOLIDS WITH HLR ALGORITHM")
    print("="*60)
    print(f"Projection normal: {projection_normal}")
    
    # Process each solid with HLR
    if simple_solid:
        array_B_simple, array_C_simple = process_solid_hlr(simple_solid, projection_normal, "Simple")
        visualize_results(array_B_simple, array_C_simple, "Simple")
    
    if medium_solid:
        array_B_medium, array_C_medium = process_solid_hlr(medium_solid, projection_normal, "Medium")
        visualize_results(array_B_medium, array_C_medium, "Medium")
    
    if complex_solid:
        array_B_complex, array_C_complex = process_solid_hlr(complex_solid, projection_normal, "Complex")
        visualize_results(array_B_complex, array_C_complex, "Complex")
    
    print("\n" + "="*60)
    print("PROCESSING COMPLETE")
    print("="*60)
    print("Generated three solids with different complexity levels:")
    print("  1. Simple: Base with partial overlap subtraction")
    print("  2. Medium: Base with centered hole")
    print("  3. Complex: Base with two separate holes")
    print("\nEach processed with HLR algorithm to generate array_B and array_C")

if __name__ == "__main__":
    main()
