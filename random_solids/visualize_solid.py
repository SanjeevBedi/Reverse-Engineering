#!/usr/bin/env python3
"""
Visualize the OpenCASCADE solid to check if vertical edge faces are present.
Loads the solid by running Base_Solid.py with seed 250.
"""

import sys
import os
import numpy as np
import subprocess

# Add Reconstruction to path
recon_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'Reconstruction')
if recon_path not in sys.path:
    sys.path.append(recon_path)

from OCC.Display.SimpleGui import init_display
from OCC.Core.Quantity import Quantity_Color, Quantity_TOC_RGB
from OCC.Core.BRepBuilderAPI import BRepBuilderAPI_MakeVertex
from OCC.Core.gp import gp_Pnt
from OCC.Core.TopExp import TopExp_Explorer
from OCC.Core.TopAbs import TopAbs_FACE, TopAbs_EDGE, TopAbs_VERTEX
from OCC.Core.TopoDS import topods
from OCC.Core.BRep import BRep_Tool

seed = 250

# Change to Reconstruction directory and run Base_Solid.py
print(f"Building solid with seed {seed}...")
os.chdir('Reconstruction')

# Run Base_Solid.py which will create the solid and make it available
import Base_Solid
# Set the seed
if hasattr(Base_Solid, '__name__') and Base_Solid.__name__ == "__main__":
    Base_Solid.seed = seed

# Now the solid should be available in Base_Solid module
if hasattr(Base_Solid, 'solid'):
    solid = Base_Solid.solid
    print(f"Solid loaded: {type(solid)}")
else:
    # Try to build it by importing and calling the script's main code
    print("Solid not found, attempting to build...")
    exec(open('Base_Solid.py').read(), {'seed': seed, '__name__': '__main__'})
    
    # Check if solid is now in globals
    if 'solid' in dir():
        print(f"Solid built successfully: {type(solid)}")
    else:
        print("ERROR: Could not build or load solid")
        sys.exit(1)

os.chdir('..')  # Go back to parent directory

# Target vertices to highlight (in cm)
target_vertices_cm = {
    22: np.array([42.72, 33.36, 0.0]),
    23: np.array([42.72, 33.36, 18.64]),
    37: np.array([37.91, 38.16, 0.0]),
    38: np.array([37.91, 38.16, 18.64]),
    40: np.array([36.5, 33.36, 18.64]),
    41: np.array([36.5, 33.36, 30.42]),
    42: np.array([29.88, 30.12, 0.0]),
    43: np.array([29.88, 30.12, 18.64]),
    45: np.array([36.5, 23.5, 0.0]),
    46: np.array([36.5, 23.5, 18.64]),
}

# Target edges
target_edges = [
    (22, 23, "V22-V23"),
    (37, 38, "V37-V38"),
    (40, 41, "V40-V41"),
    (42, 43, "V42-V43"),
    (45, 46, "V45-V46"),
]

def vertices_match(v1, v2, tol=0.1):
    """Check if two vertices match within tolerance"""
    return np.allclose(v1, v2, atol=tol)

# Initialize display
display, start_display, add_menu, add_function_to_menu = init_display()

# Display the solid
display.DisplayShape(solid, update=True, transparency=0.5)

# Extract and check faces
print(f"\n{'='*70}")
print("CHECKING FACES IN SOLID")
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
        face_vertices.append(np.array([pnt.X(), pnt.Y(), pnt.Z()]))
        vertex_explorer.Next()
    
    # Check if any target edges are in this face
    for v1_num, v2_num, edge_name in target_edges:
        v1_target = target_vertices_cm[v1_num]
        v2_target = target_vertices_cm[v2_num]
        
        # Check if both vertices are in this face
        v1_found = any(vertices_match(fv, v1_target) for fv in face_vertices)
        v2_found = any(vertices_match(fv, v2_target) for fv in face_vertices)
        
        if v1_found and v2_found:
            if face_count not in [f[0] for f in faces_with_target_edges]:
                faces_with_target_edges.append((face_count, face, []))
            faces_with_target_edges[-1][2].append(edge_name)
    
    face_explorer.Next()

print(f"\nTotal faces in solid: {face_count}")
print(f"Faces containing target edges: {len(faces_with_target_edges)}")

if faces_with_target_edges:
    print("\nFaces with target vertical edges:")
    for face_num, face, edge_names in faces_with_target_edges:
        print(f"  Face {face_num}: {', '.join(edge_names)}")
        # Highlight this face in red
        display.DisplayShape(face, color=Quantity_Color(1.0, 0.0, 0.0, Quantity_TOC_RGB), 
                           transparency=0.0, update=False)
else:
    print("\n⚠️  NO FACES FOUND containing target vertical edges!")
    print("This confirms the edges are missing from the OpenCASCADE solid geometry.")

# Display target vertices as spheres
print(f"\n{'='*70}")
print("HIGHLIGHTING TARGET VERTICES")
print(f"{'='*70}")

for v_num, coords in target_vertices_cm.items():
    pnt = gp_Pnt(coords[0], coords[1], coords[2])
    vertex_shape = BRepBuilderAPI_MakeVertex(pnt).Vertex()
    # Display in green
    display.DisplayShape(vertex_shape, color=Quantity_Color(0.0, 1.0, 0.0, Quantity_TOC_RGB), 
                        update=False)
    print(f"V{v_num}: {coords}")

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
        edge_vertices.append(np.array([pnt.X(), pnt.Y(), pnt.Z()]))
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
    print(f"\n⚠️  MISSING {5 - len(matching_edges)} target edges from solid geometry!")
    missing = set(name for _, _, name in target_edges) - set(name for _, _, name in matching_edges)
    print(f"Missing edges: {', '.join(missing)}")

print(f"\n{'='*70}")
print("LEGEND:")
print("  - Gray (transparent): Original solid")
print("  - Red: Faces containing target edges")
print("  - Green points: Target vertices")
print("  - Blue lines: Target vertical edges (if found)")
print(f"{'='*70}")

display.FitAll()
print("\nDisplay window opened. Close window to exit.")
start_display()
