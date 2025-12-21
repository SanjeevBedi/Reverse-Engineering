from OCC.Core.gp import gp_Pnt
from OCC.Core.BRepBuilderAPI import (
    BRepBuilderAPI_MakeEdge,
    BRepBuilderAPI_MakeWire,
    BRepBuilderAPI_MakeFace,
    BRepBuilderAPI_Sewing,
    BRepBuilderAPI_MakeSolid
)
from OCC.Core.BRepPrimAPI import BRepPrimAPI_MakeBox
from OCC.Core.BRepAlgoAPI import BRepAlgoAPI_Cut
from OCC.Core.TopAbs import TopAbs_FACE, TopAbs_SHELL, TopAbs_EDGE, TopAbs_VERTEX
from OCC.Core.TopExp import TopExp_Explorer
from OCC.Core.TopoDS import topods_Shell, topods_Face, topods_Edge, topods_Vertex
from OCC.Core.BRep import BRep_Tool
from OCC.Core.BRepCheck import BRepCheck_Analyzer
from OCC.Core.GProp import GProp_GProps
from OCC.Core.BRepGProp import brepgprop_VolumeProperties
from OCC.Core.TopoDS import TopoDS_Compound
from OCC.Display.SimpleGui import init_display
from OCC.Core.BRepBuilderAPI import BRepBuilderAPI_Copy

# Initialize viewer
display, start_display, add_menu, add_function_to_menu = init_display()

def make_face_from_points(points):
    """Create a planar face from a list of 4 points."""
    wire_maker = BRepBuilderAPI_MakeWire()
    for i in range(len(points)):
        p1 = points[i]
        p2 = points[(i + 1) % len(points)]
        edge = BRepBuilderAPI_MakeEdge(p1, p2).Edge()
        wire_maker.Add(edge)
    return BRepBuilderAPI_MakeFace(wire_maker.Wire()).Face()

def make_cuboid_faces(xmin, xmax, ymin, ymax, zmin, zmax):
    """Return 6 faces of a cuboid defined by min/max coordinates."""
    return [
        make_face_from_points([gp_Pnt(xmin, ymin, zmin), gp_Pnt(xmax, ymin, zmin), gp_Pnt(xmax, ymax, zmin), gp_Pnt(xmin, ymax, zmin)]),  # bottom
        make_face_from_points([gp_Pnt(xmin, ymin, zmax), gp_Pnt(xmax, ymin, zmax), gp_Pnt(xmax, ymax, zmax), gp_Pnt(xmin, ymax, zmax)]),  # top
        make_face_from_points([gp_Pnt(xmin, ymin, zmin), gp_Pnt(xmin, ymax, zmin), gp_Pnt(xmin, ymax, zmax), gp_Pnt(xmin, ymin, zmax)]),  # left
        make_face_from_points([gp_Pnt(xmax, ymin, zmin), gp_Pnt(xmax, ymax, zmin), gp_Pnt(xmax, ymax, zmax), gp_Pnt(xmax, ymin, zmax)]),  # right
        make_face_from_points([gp_Pnt(xmin, ymin, zmin), gp_Pnt(xmax, ymin, zmin), gp_Pnt(xmax, ymin, zmax), gp_Pnt(xmin, ymin, zmax)]),  # front
        make_face_from_points([gp_Pnt(xmin, ymax, zmin), gp_Pnt(xmax, ymax, zmin), gp_Pnt(xmax, ymax, zmax), gp_Pnt(xmin, ymax, zmax)])   # back
    ]

# Outer cuboid dimensions - but we'll modify top and bottom to have holes
def make_outer_wire(face_type):
    """Create outer wire for each face - CW winding for outward normals."""
    if face_type == 'bottom':
        points = [gp_Pnt(0, 0, 0), gp_Pnt(100, 0, 0), gp_Pnt(100, 50, 0), gp_Pnt(0, 50, 0)]
    elif face_type == 'top':
        points = [gp_Pnt(0, 0, 50), gp_Pnt(100, 0, 50), gp_Pnt(100, 50, 50), gp_Pnt(0, 50, 50)]
    elif face_type == 'left':
        points = [gp_Pnt(0, 0, 0), gp_Pnt(0, 0, 50), gp_Pnt(0, 50, 50), gp_Pnt(0, 50, 0)]
    elif face_type == 'right':
        points = [gp_Pnt(100, 0, 0), gp_Pnt(100, 50, 0), gp_Pnt(100, 50, 50), gp_Pnt(100, 0, 50)]
    elif face_type == 'front':
        points = [gp_Pnt(0, 0, 0), gp_Pnt(0, 0, 50), gp_Pnt(100, 0, 50), gp_Pnt(100, 0, 0)]
    elif face_type == 'back':
        points = [gp_Pnt(0, 50, 0), gp_Pnt(100, 50, 0), gp_Pnt(100, 50, 50), gp_Pnt(0, 50, 50)]
    
    wire_maker = BRepBuilderAPI_MakeWire()
    for i in range(len(points)):
        p1 = points[i]
        p2 = points[(i + 1) % len(points)]
        edge = BRepBuilderAPI_MakeEdge(p1, p2).Edge()
        wire_maker.Add(edge)
    return wire_maker.Wire()

def make_hole_wire(face_type):
    """Create hole wire for top and bottom faces."""
    if face_type == 'bottom' or face_type == 'top':
        z = 0 if face_type == 'bottom' else 50
        # IMPORTANT: Hole must wind SAME direction as outer wire to create void
        # Both wind CCW when viewed from +Z
        points = [gp_Pnt(30, 15, z), gp_Pnt(30, 35, z), gp_Pnt(70, 35, z), gp_Pnt(70, 15, z)]
        wire_maker = BRepBuilderAPI_MakeWire()
        for i in range(len(points)):
            p1 = points[i]
            p2 = points[(i + 1) % len(points)]
            edge = BRepBuilderAPI_MakeEdge(p1, p2).Edge()
            wire_maker.Add(edge)
        return wire_maker.Wire()
    return None


# Create faces with holes
outer_faces = []
for face_type in ['bottom', 'top', 'left', 'right', 'front', 'back']:
    outer_wire = make_outer_wire(face_type)
    hole_wire = make_hole_wire(face_type)
    
    if hole_wire is not None:
        # Face with hole (top and bottom)
        face_maker = BRepBuilderAPI_MakeFace(outer_wire)
        face_maker.Add(hole_wire)
        outer_faces.append(face_maker.Face())
        print(f"Created {face_type} face with hole")
    else:
        # Simple face (sides)
        outer_faces.append(BRepBuilderAPI_MakeFace(outer_wire).Face())
        print(f"Created {face_type} face")

# Add the 4 wall faces of the hole (connecting top and bottom holes)
hole_wall_faces = []

# Front wall of hole (y=15) - normal points into hole (toward +Y)
points = [gp_Pnt(30, 15, 0), gp_Pnt(70, 15, 0), gp_Pnt(70, 15, 50), gp_Pnt(30, 15, 50)]
hole_wall_faces.append(make_face_from_points(points))
print("Created hole front wall")

# Back wall of hole (y=35) - normal points into hole (toward -Y)
points = [gp_Pnt(70, 35, 0), gp_Pnt(70, 35, 50), gp_Pnt(30, 35, 50), gp_Pnt(30, 35, 0)]
hole_wall_faces.append(make_face_from_points(points))
print("Created hole back wall")

# Left wall of hole (x=30) - normal points into hole (toward +X)
points = [gp_Pnt(30, 15, 0), gp_Pnt(30, 35, 0), gp_Pnt(30, 35, 50), gp_Pnt(30, 15, 50)]
hole_wall_faces.append(make_face_from_points(points))
print("Created hole left wall")

# Right wall of hole (x=70) - normal points into hole (toward -X)
points = [gp_Pnt(70, 15, 0), gp_Pnt(70, 15, 50), gp_Pnt(70, 35, 50), gp_Pnt(70, 35, 0)]
hole_wall_faces.append(make_face_from_points(points))
print("Created hole right wall")

print(f"\nApproach: 6 outer faces (with holes) + 4 hole wall faces = 10 faces total")

# Sew all faces together
sewer = BRepBuilderAPI_Sewing()
for f in outer_faces + hole_wall_faces:
    sewer.Add(f)
sewer.Perform()
sewn_shape = sewer.SewedShape()

# Extract shells from sewn shape
shell_explorer = TopExp_Explorer(sewn_shape, TopAbs_SHELL)
shells = []
while shell_explorer.More():
    shell = topods_Shell(shell_explorer.Current())
    shells.append(shell)
    shell_explorer.Next()

print(f"Found {len(shells)} shell(s)")

# Make solid from shells
solid_maker = BRepBuilderAPI_MakeSolid()
for shell in shells:
    solid_maker.Add(shell)
solid = solid_maker.Solid()

# Print solid structure
print("\n" + "="*80)
print("SOLID STRUCTURE")
print("="*80)

# Collect all unique vertices for numbering
vertex_map = {}
vertex_counter = 0

# Explore the solid
shell_explorer = TopExp_Explorer(solid, TopAbs_SHELL)
shell_num = 0
while shell_explorer.More():
    shell_num += 1
    shell = topods_Shell(shell_explorer.Current())
    print(f"\nShell {shell_num}:")
    
    # Explore faces in this shell
    face_explorer = TopExp_Explorer(shell, TopAbs_FACE)
    face_num = 0
    while face_explorer.More():
        face_num += 1
        face = topods_Face(face_explorer.Current())
        print(f"  Face {face_num}:")
        
        # Explore edges in this face
        edge_explorer = TopExp_Explorer(face, TopAbs_EDGE)
        edge_num = 0
        while edge_explorer.More():
            edge_num += 1
            edge = topods_Edge(edge_explorer.Current())
            
            # Get vertices of this edge
            vertex_explorer = TopExp_Explorer(edge, TopAbs_VERTEX)
            edge_vertices = []
            while vertex_explorer.More():
                vertex = topods_Vertex(vertex_explorer.Current())
                pnt = BRep_Tool.Pnt(vertex)
                
                # Check if vertex already mapped
                coord_key = (round(pnt.X(), 6), round(pnt.Y(), 6), round(pnt.Z(), 6))
                if coord_key not in vertex_map:
                    vertex_map[coord_key] = vertex_counter
                    vertex_counter += 1
                
                v_num = vertex_map[coord_key]
                edge_vertices.append((v_num, pnt.X(), pnt.Y(), pnt.Z()))
                vertex_explorer.Next()
            
            # Print edge info
            if len(edge_vertices) >= 2:
                v1_num, v1_x, v1_y, v1_z = edge_vertices[0]
                v2_num, v2_x, v2_y, v2_z = edge_vertices[1]
                print(f"    Edge {edge_num}: v{v1_num} ({v1_x:.2f}, {v1_y:.2f}, {v1_z:.2f}) -> "
                      f"v{v2_num} ({v2_x:.2f}, {v2_y:.2f}, {v2_z:.2f})")
            
            edge_explorer.Next()
        
        face_explorer.Next()
    
    shell_explorer.Next()

print(f"\nTotal unique vertices: {vertex_counter}")
print("="*80 + "\n")

# Check for free edges
print("CHECKING FOR FREE EDGES...")
print("="*80)

from OCC.Core.TopTools import TopTools_IndexedDataMapOfShapeListOfShape
from OCC.Core.TopExp import topexp

edge_face_map = TopTools_IndexedDataMapOfShapeListOfShape()
topexp.MapShapesAndAncestors(solid, TopAbs_EDGE, TopAbs_FACE, edge_face_map)

free_edge_count = 0
total_edges = edge_face_map.Size()
print(f"\nTotal edges in solid: {total_edges}")

for i in range(1, total_edges + 1):
    edge = topods_Edge(edge_face_map.FindKey(i))
    face_list = edge_face_map.FindFromIndex(i)
    num_faces = face_list.Size()
    
    if num_faces < 2:
        free_edge_count += 1
        # Get edge vertices
        vertex_explorer = TopExp_Explorer(edge, TopAbs_VERTEX)
        edge_vertices = []
        while vertex_explorer.More():
            vertex = topods_Vertex(vertex_explorer.Current())
            pnt = BRep_Tool.Pnt(vertex)
            coord_key = (round(pnt.X(), 6), round(pnt.Y(), 6), round(pnt.Z(), 6))
            v_num = vertex_map.get(coord_key, "?")
            edge_vertices.append((v_num, pnt.X(), pnt.Y(), pnt.Z()))
            vertex_explorer.Next()
        
        if len(edge_vertices) >= 2:
            v1_num, v1_x, v1_y, v1_z = edge_vertices[0]
            v2_num, v2_x, v2_y, v2_z = edge_vertices[1]
            print(f"  Free edge {free_edge_count}: v{v1_num}-v{v2_num} "
                  f"({v1_x:.2f},{v1_y:.2f},{v1_z:.2f}) <-> ({v2_x:.2f},{v2_y:.2f},{v2_z:.2f}) "
                  f"[shared by {num_faces} face(s)]")

if free_edge_count == 0:
    print("\n✓ No free edges found - solid is closed!")
else:
    print(f"\n✗ Found {free_edge_count} free edge(s) - solid is not closed")

print("="*80 + "\n")

# Check solid validity
print("CHECKING SOLID VALIDITY...")
print("="*80)

analyzer = BRepCheck_Analyzer(solid)
is_valid = analyzer.IsValid()

if is_valid:
    print("\n✓ Solid is VALID")
else:
    print("\n✗ Solid is INVALID")

# Compute volume
props = GProp_GProps()
brepgprop_VolumeProperties(solid, props)
volume = props.Mass()
print(f"Solid volume: {volume:.2f} cubic units")

print("="*80 + "\n")

# Create comparison box using MakeBox
print("\n" + "="*80)
print("COMPARISON: MakeBox SOLID WITH CUT HOLE")
print("="*80)

# Make outer box using BRepPrimAPI_MakeBox (same dimensions as outer_faces)
outer_box = BRepPrimAPI_MakeBox(100, 50, 50).Shape()

# Make inner box (hole) - same dimensions as inner_faces
inner_box = BRepPrimAPI_MakeBox(gp_Pnt(30, 15, 0), gp_Pnt(70, 35, 50)).Shape()

# Cut the hole from the outer box
reference_box = BRepAlgoAPI_Cut(outer_box, inner_box).Shape()

print("\nMakeBox with Cut hole structure:")

# Explore shells
shell_explorer = TopExp_Explorer(reference_box, TopAbs_SHELL)
shell_num = 0
while shell_explorer.More():
    shell_num += 1
    shell = topods_Shell(shell_explorer.Current())
    
    # Count faces in shell
    face_explorer = TopExp_Explorer(shell, TopAbs_FACE)
    face_count = 0
    while face_explorer.More():
        face_count += 1
        face_explorer.Next()
    
    print(f"  Shell {shell_num}: {face_count} faces")
    shell_explorer.Next()

print(f"Total shells: {shell_num}")

# Check validity
analyzer = BRepCheck_Analyzer(reference_box)
is_valid = analyzer.IsValid()
print(f"Valid: {is_valid}")

# Compute volume
props = GProp_GProps()
brepgprop_VolumeProperties(reference_box, props)
volume = props.Mass()
print(f"Volume: {volume:.2f} cubic units")

# Check free edges
from OCC.Core.TopTools import TopTools_IndexedDataMapOfShapeListOfShape
from OCC.Core.TopExp import topexp

edge_face_map = TopTools_IndexedDataMapOfShapeListOfShape()
topexp.MapShapesAndAncestors(reference_box, TopAbs_EDGE, TopAbs_FACE, edge_face_map)

free_edge_count = 0
total_edges = edge_face_map.Size()
for i in range(1, total_edges + 1):
    face_list = edge_face_map.FindFromIndex(i)
    if face_list.Size() < 2:
        free_edge_count += 1

print(f"Total edges: {total_edges}")
print(f"Free edges: {free_edge_count}")

print("="*80 + "\n")

# Display result
display.DisplayShape(solid, update=True)
start_display()
