#!/usr/bin/env python3
"""
Visualize the OpenCASCADE solid from STEP file to check if vertical edges are present.
"""

import sys
import os
import numpy as np

from OCC.Display.SimpleGui import init_display
from OCC.Core.Quantity import Quantity_Color, Quantity_TOC_RGB
from OCC.Core.BRepBuilderAPI import BRepBuilderAPI_MakeVertex
from OCC.Core.gp import gp_Pnt
from OCC.Core.TopExp import TopExp_Explorer
from OCC.Core.TopAbs import TopAbs_FACE, TopAbs_EDGE, TopAbs_VERTEX
from OCC.Core.TopoDS import topods, TopoDS_Shape
from OCC.Core.BRep import BRep_Tool
from OCC.Core.STEPControl import STEPControl_Reader
from OCC.Core.IFSelect import IFSelect_RetDone

seed = 250

# Load solid from STEP file
step_file = f"STEPfiles/solid_output.step"  # or use seed-specific file if available

if not os.path.exists(step_file):
    print(f"ERROR: STEP file not found: {step_file}")
    print("Run Build_Solid.py --seed 250 first to generate the STEP file")
    sys.exit(1)

print(f"Loading solid from STEP file: {step_file}")

# Read STEP file
step_reader = STEPControl_Reader()
status = step_reader.ReadFile(step_file)

if status != IFSelect_RetDone:
    print(f"ERROR: Failed to read STEP file (status={status})")
    sys.exit(1)

step_reader.TransferRoots()
solid = step_reader.OneShape()

if solid.IsNull():
    print("ERROR: Loaded shape is null")
    sys.exit(1)

print(f"Solid loaded successfully: {type(solid)}")

# Target vertices to highlight (in cm - based on trace_vertices_in_build.py results)
target_vertices_cm = {
    22: (42.715545, 33.357964, 0.0),
    23: (42.715545, 33.357964, 18.636713),
    37: (37.912317, 38.161192, 0.0),
    38: (37.912317, 38.161192, 18.636713),
    40: (36.5, 33.357964, 18.636713),
    41: (36.5, 33.357964, 30.424143),
    42: (29.875562, 30.124438, 0.0),
    43: (29.875562, 30.124438, 18.636713),
    45: (36.5, 23.5, 0.0),
    46: (36.5, 23.5, 18.636713),
}

# Target edges
target_edges = [
    (22, 23, "V22-V23 (Front MISSING, Side OK)"),
    (37, 38, "V37-V38 (Both OK)"),
    (40, 41, "V40-V41 (Both OK)"),
    (42, 43, "V42-V43 (Front OK, Side MISSING)"),
    (45, 46, "V45-V46 (Front OK, Side MISSING)"),
]

def vertices_match(v1, v2, tol=0.1):
    """Check if two vertices match within tolerance"""
    v1_arr = np.array(v1[:3] if isinstance(v1, (tuple, list)) else [v1.X(), v1.Y(), v1.Z()])
    v2_arr = np.array(v2[:3] if isinstance(v2, (tuple, list)) else [v2.X(), v2.Y(), v2.Z()])
    return np.allclose(v1_arr, v2_arr, atol=tol)

# Initialize display
display, start_display, add_menu, add_function_to_menu = init_display()

# Display the solid with transparency
display.DisplayShape(solid, update=False, transparency=0.6, color=Quantity_Color(0.7, 0.7, 0.7, Quantity_TOC_RGB))

# Check faces in solid
print(f"\n{'='*70}")
print("CHECKING FACES IN SOLID FOR TARGET EDGES")
print(f"{'='*70}")

face_explorer = TopExp_Explorer(solid, TopAbs_FACE)
face_count = 0
faces_with_target_edges = []

while face_explorer.More():
    face = topods.Face(face_explorer.Current())
    face_count += 1
    
    # Extract vertices from this face
    face_vertices = []
    vertex_explorer = TopExp_Explorer(face, TopAbs_VERTEX)
    while vertex_explorer.More():
        vertex = topods.Vertex(vertex_explorer.Current())
        pnt = BRep_Tool.Pnt(vertex)
        face_vertices.append(pnt)
        vertex_explorer.Next()
    
    # Check if any target edges are in this face
    edges_in_face = []
    for v1_num, v2_num, edge_name in target_edges:
        v1_target = target_vertices_cm[v1_num]
        v2_target = target_vertices_cm[v2_num]
        
        # Check if both vertices are in this face
        v1_found = any(vertices_match(fv, v1_target) for fv in face_vertices)
        v2_found = any(vertices_match(fv, v2_target) for fv in face_vertices)
        
        if v1_found and v2_found:
            edges_in_face.append(edge_name)
    
    if edges_in_face:
        faces_with_target_edges.append((face_count, face, edges_in_face))
    
    face_explorer.Next()

print(f"\nTotal faces in solid: {face_count}")
print(f"Faces containing target edges: {len(faces_with_target_edges)}")

if faces_with_target_edges:
    print("\n✓ Faces with target vertical edges:")
    for face_num, face, edge_names in faces_with_target_edges:
        print(f"  Face {face_num}: {', '.join(edge_names)}")
        # Highlight this face in red
        display.DisplayShape(face, color=Quantity_Color(1.0, 0.0, 0.0, Quantity_TOC_RGB), 
                           transparency=0.0, update=False)
else:
    print("\n✗ NO FACES FOUND containing target vertical edges!")

# Check edges in solid
print(f"\n{'='*70}")
print("CHECKING EDGES IN SOLID")
print(f"{'='*70}")

edge_explorer = TopExp_Explorer(solid, TopAbs_EDGE)
edge_count = 0
matching_edges = []

while edge_explorer.More():
    edge = topods.Edge(edge_explorer.Current())
    edge_count += 1
    
    # Get vertices of this edge
    vertex_explorer = TopExp_Explorer(edge, TopAbs_VERTEX)
    edge_vertices = []
    while vertex_explorer.More():
        vertex = topods.Vertex(vertex_explorer.Current())
        pnt = BRep_Tool.Pnt(vertex)
        edge_vertices.append(pnt)
        vertex_explorer.Next()
    
    if len(edge_vertices) >= 2:
        # Check if this matches any target edge
        for v1_num, v2_num, edge_name in target_edges:
            v1_target = target_vertices_cm[v1_num]
            v2_target = target_vertices_cm[v2_num]
            
            # Check if this edge connects the two target vertices
            if ((vertices_match(edge_vertices[0], v1_target) and vertices_match(edge_vertices[1], v2_target)) or
                (vertices_match(edge_vertices[0], v2_target) and vertices_match(edge_vertices[1], v1_target))):
                matching_edges.append((edge_count, edge, edge_name))
                print(f"✓ Found {edge_name} as edge #{edge_count}")
                # Highlight in blue with thicker line
                display.DisplayShape(edge, color=Quantity_Color(0.0, 0.0, 1.0, Quantity_TOC_RGB), 
                                   update=False)
    
    edge_explorer.Next()

print(f"\nTotal edges in solid: {edge_count}")
print(f"Matching target vertical edges: {len(matching_edges)}/5")

if len(matching_edges) < 5:
    missing_count = 5 - len(matching_edges)
    print(f"\n⚠️  MISSING {missing_count} target edge(s) from solid geometry!")
    found_names = {name for _, _, name in matching_edges}
    all_names = {name for _, _, name in target_edges}
    missing = all_names - found_names
    print(f"Missing edges: {', '.join(missing)}")
else:
    print(f"\n✓ ALL 5 target edges found in solid!")

# Display target vertices as green points
print(f"\n{'='*70}")
print("HIGHLIGHTING TARGET VERTICES")
print(f"{'='*70}")

for v_num, coords in target_vertices_cm.items():
    pnt = gp_Pnt(coords[0], coords[1], coords[2])
    vertex_shape = BRepBuilderAPI_MakeVertex(pnt).Vertex()
    # Display in green
    display.DisplayShape(vertex_shape, color=Quantity_Color(0.0, 1.0, 0.0, Quantity_TOC_RGB), 
                        update=False)

print("\nAll target vertices displayed in GREEN")

print(f"\n{'='*70}")
print("LEGEND:")
print("  - Gray (transparent): Original solid")
print("  - RED faces: Contain target vertical edges")
print("  - BLUE lines: Target vertical edges (if found in solid)")
print("  - GREEN points: Target vertices")
print(f"{'='*70}")

display.FitAll()
display.View_Iso()
print("\nOpenCASCADE display window opened.")
print("Close window to exit.")
start_display()
