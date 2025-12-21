import random
import sys
sys.path.append('/opt/anaconda3/envs/pyocc/lib/python3.9/site-packages')
sys.path.append('/Users/sbedi/Documents/EAGIS/pythonocc-core-master/src/Display')
from OCC.Core.BRepPrimAPI import BRepPrimAPI_MakeBox
from OCC.Core.gp import gp_Pnt, gp_Dir, gp_Ax2
from OCC.Core.HLRAlgo import HLRAlgo_Projector
from OCC.Core.HLRBRep import HLRBRep_Algo, HLRBRep_HLRToShape
from OCC.Core.TopExp import TopExp_Explorer
from OCC.Core.TopAbs import TopAbs_EDGE
from OCC.Core.BRep import BRep_Tool

# 1. Create a simple box
shape = BRepPrimAPI_MakeBox(60, 40, 30).Shape()

# 2. Set up the HLR projector for the top view (Z+)
axis = gp_Ax2(gp_Pnt(0, 0, 0), gp_Dir(0, 0, 1), gp_Dir(0, 1, 0))
projector = HLRAlgo_Projector(axis)

# 3. Run the HLR algorithm
algo = HLRBRep_Algo()
algo.Add(shape)
algo.Projector(projector)
algo.Update()
algo.Hide()

# 4. Extract visible and hidden edges
hlr_shapes = HLRBRep_HLRToShape(algo)
visible = hlr_shapes.VCompound()
hidden = hlr_shapes.Rg1LineVCompound()

def count_edges(shape):
    if shape is None or (hasattr(shape, "IsNull") and shape.IsNull()):
        return 0
    count = 0
    exp = TopExp_Explorer(shape, TopAbs_EDGE)
    while exp.More():
        count += 1
        exp.Next()
    return count

print("Visible edges:", count_edges(visible))
print("Hidden edges:", count_edges(hidden))

# Print endpoints of visible edges
print("\nVisible edge endpoints:")
exp = TopExp_Explorer(visible, TopAbs_EDGE)
while exp.More():
    edge = exp.Current()
    curve_data = BRep_Tool.Curve(edge)
    if curve_data is not None and len(curve_data) >= 2:
        curve_handle = curve_data[0]
        if curve_handle is None or not hasattr(curve_handle, "D0"):
            exp.Next()
            continue
        first = curve_data[1]
        last = curve_data[2] if len(curve_data) == 3 else edge.LastParameter()
        p1 = gp_Pnt()
        p2 = gp_Pnt()
        curve_handle.D0(first, p1)
        curve_handle.D0(last, p2)
        print(f"({p1.X():.2f}, {p1.Y():.2f}, {p1.Z():.2f}) -> ({p2.X():.2f}, {p2.Y():.2f}, {p2.Z():.2f})")
    exp.Next()

# Print endpoints of hidden edges
print("\nHidden edge endpoints:")
if hidden is not None and hasattr(hidden, "IsNull") and not hidden.IsNull():
    exp = TopExp_Explorer(hidden, TopAbs_EDGE)
    while exp.More():
        edge = exp.Current()
        curve_data = BRep_Tool.Curve(edge)
        if curve_data is not None and len(curve_data) >= 2:
            curve_handle = curve_data[0]
            if curve_handle is None or not hasattr(curve_handle, "D0"):
                exp.Next()
                continue
            first = curve_data[1]
            last = curve_data[2] if len(curve_data) == 3 else edge.LastParameter()
            p1 = gp_Pnt()
            p2 = gp_Pnt()
            curve_handle.D0(first, p1)
            curve_handle.D0(last, p2)
            print(f"({p1.X():.2f}, {p1.Y():.2f}, {p1.Z():.2f}) -> ({p2.X():.2f}, {p2.Y():.2f}, {p2.Z():.2f})")
        exp.Next()
else:
    print("No hidden edges found (shape is null or empty).")