

import sys
import os
os.environ["DYLD_LIBRARY_PATH"] = "/Users/sbedi/Anaconda/anaconda3/lib"
sys.path.append('/opt/anaconda3/envs/pyocc/lib/python3.9/site-packages')

from OCC.Core.STEPControl import STEPControl_Reader
from OCC.Core.IFSelect import IFSelect_RetDone
from OCC.Core.TopExp import TopExp_Explorer
from OCC.Core.TopAbs import TopAbs_FACE, TopAbs_EDGE, TopAbs_VERTEX
from OCC.Core.BRep import BRep_Tool
from OCC.Core.BRepGProp import brepgprop_SurfaceProperties
from OCC.Core.GProp import GProp_GProps
from OCC.Core.gp import gp_Pnt

def save_tokens_to_file(tokens, output_file):
    """Save tokens to a text file."""
    with open(output_file, 'w') as f:
        for token in tokens:
            f.write(token + '\n')
    print(f"Tokens saved to {output_file}")

def load_tokens_from_file(input_file):
    """Load tokens from a text file."""
    tokens = []
    try:
        with open(input_file, 'r') as f:
            tokens = [line.strip() for line in f.readlines()]
        print(f"Loaded {len(tokens)} tokens from {input_file}")
    except FileNotFoundError:
        print(f"File {input_file} not found.")
    return tokens

def get_surface_area(face):
    """Calculate the surface area of a face."""
    props = GProp_GProps()
    brepgprop_SurfaceProperties(face, props)
    return props.Mass()

def tokenize_step_file(file_path, include_area=True):
    reader = STEPControl_Reader()
    status = reader.ReadFile(file_path)

    if status != IFSelect_RetDone:
        print("Failed to read STEP file.")
        return []

    reader.TransferRoots()
    shape = reader.OneShape()

    tokens = []

    # Tokenize faces with surface area
    face_explorer = TopExp_Explorer(shape, TopAbs_FACE)
    face_count = 0
    while face_explorer.More():
        face = face_explorer.Current()
        if include_area:
            area = get_surface_area(face)
            tokens.append(f"FACE_{face_count}:TYPE=PLANAR;AREA={area:.4f}")
        else:
            tokens.append(f"FACE_{face_count}:TYPE=PLANAR")
        face_count += 1
        face_explorer.Next()

    # Tokenize edges
    edge_explorer = TopExp_Explorer(shape, TopAbs_EDGE)
    edge_count = 0
    while edge_explorer.More():
        edge = edge_explorer.Current()
        curve, first, last = BRep_Tool.Curve(edge)
        if curve is not None:
            p1 = curve.Value(first)
            p2 = curve.Value(last)
            tokens.append(f"EDGE_{edge_count}:LINE;START=({p1.X():.2f},{p1.Y():.2f},{p1.Z():.2f});END=({p2.X():.2f},{p2.Y():.2f},{p2.Z():.2f})")
        edge_count += 1
        edge_explorer.Next()

    # Tokenize vertices
    vertex_explorer = TopExp_Explorer(shape, TopAbs_VERTEX)
    vertex_count = 0
    while vertex_explorer.More():
        vertex = vertex_explorer.Current()
        point = BRep_Tool.Pnt(vertex)
        tokens.append(f"VERTEX_{vertex_count}:({point.X():.2f},{point.Y():.2f},{point.Z():.2f})")
        vertex_count += 1
        vertex_explorer.Next()

    return tokens

# Example usage
if __name__ == "__main__":
    # Test with the random_cuboid_shape.step file
    print("Testing tokenization with surface area calculation...")
    tokens = tokenize_step_file("random_cuboid_shape.step", include_area=True)
    
    print(f"Total tokens generated: {len(tokens)}")
    print("\nFirst 20 tokens:")
    print("=" * 60)
    
    for i, token in enumerate(tokens[:20]):
        print(f"{i+1:3d}: {token}")
    
    if len(tokens) > 20:
        print("...")
        print(f"(showing first 20 of {len(tokens)} total tokens)")
        
    # Print summary statistics
    face_tokens = [t for t in tokens if t.startswith("FACE_")]
    edge_tokens = [t for t in tokens if t.startswith("EDGE_")]
    vertex_tokens = [t for t in tokens if t.startswith("VERTEX_")]
    
    print("\n" + "=" * 60)
    print("SUMMARY:")
    print(f"Faces: {len(face_tokens)}")
    print(f"Edges: {len(edge_tokens)}")
    print(f"Vertices: {len(vertex_tokens)}")
    print(f"Total: {len(tokens)}")
    
    # Show some example face tokens with areas
    print("\nFace tokens with surface areas:")
    for token in face_tokens[:5]:  # Show first 5 faces
        print(f"  {token}")
    
    # Calculate edge length statistics
    edge_lengths = []
    for token in edge_tokens:
        if "START=" in token and "END=" in token:
            start_str = token.split("START=")[1].split(";")[0]
            end_str = token.split("END=")[1].split(")")[0] + ")"
            
            # Extract coordinates
            start_coords = start_str.strip("()").split(",")
            end_coords = end_str.strip("()").split(",")
            
            if len(start_coords) == 3 and len(end_coords) == 3:
                try:
                    x1, y1, z1 = map(float, start_coords)
                    x2, y2, z2 = map(float, end_coords)
                    length = ((x2-x1)**2 + (y2-y1)**2 + (z2-z1)**2)**0.5
                    edge_lengths.append(length)
                except ValueError:
                    pass
    
    if edge_lengths:
        print(f"\nEdge length statistics:")
        print(f"  Average length: {sum(edge_lengths)/len(edge_lengths):.4f}")
        print(f"  Min length: {min(edge_lengths):.4f}")
        print(f"  Max length: {max(edge_lengths):.4f}")

    print("\nTokenization completed successfully!")
    
    # Save tokens to file
    save_tokens_to_file(tokens, "cuboid_tokens.txt")

# Alternative usage without area calculation (faster)
# tokens = tokenize_step_file("random_cuboid_shape.step", include_area=False)
