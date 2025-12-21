
import sys
import os
os.environ["DYLD_LIBRARY_PATH"] = "/Users/sbedi/Anaconda/anaconda3/lib"
sys.path.append('/opt/anaconda3/envs/pyocc/lib/python3.9/site-packages')

from OCC.Core.STEPControl import STEPControl_Reader
from OCC.Core.IFSelect import IFSelect_RetDone
from OCC.Core.TopExp import TopExp_Explorer
from OCC.Core.TopAbs import TopAbs_FACE, TopAbs_EDGE
from OCC.Core.BRep import BRep_Tool
from OCC.Core.BRepTools import breptools_UVBounds
from OCC.Core.gp import gp_Pnt

# Load the STEP file
step_reader = STEPControl_Reader()
status = step_reader.ReadFile("random_engineering_model.step")

if status == IFSelect_RetDone:
    step_reader.TransferRoots()
    shape = step_reader.OneShape()

    # Traverse faces
    print("Faces:")
    face_explorer = TopExp_Explorer(shape, TopAbs_FACE)
    while face_explorer.More():
        face = face_explorer.Current()
        umin, umax, vmin, vmax = breptools_UVBounds(face)
        print(f"  Face bounds: U({umin:.2f}, {umax:.2f}), V({vmin:.2f}, {vmax:.2f})")
        face_explorer.Next()

    # Traverse edges
    print("\nEdges:")
    edge_explorer = TopExp_Explorer(shape, TopAbs_EDGE)
    while edge_explorer.More():
        edge = edge_explorer.Current()
        curve, first, last = BRep_Tool.Curve(edge)
        if curve is not None:
            p1 = gp_Pnt()
            p2 = gp_Pnt()
            curve.D0(first, p1)
            curve.D0(last, p2)
            print(f"  Edge from ({p1.X():.2f}, {p1.Y():.2f}, {p1.Z():.2f}) to ({p2.X():.2f}, {p2.Y():.2f}, {p2.Z():.2f})")
        edge_explorer.Next()
else:
    print("Failed to read STEP file.")
