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
from OCC.Core.gp import gp_Pnt
from OCC.Core.BRepAlgoAPI import BRepAlgoAPI_Fuse, BRepAlgoAPI_Cut
from OCC.Core.TopAbs import TopAbs_SOLID
from OCC.Core.ShapeFix import ShapeFix_Solid
from OCC.Core.gp import gp_Vec, gp_Trsf
from OCC.Core.BRepBuilderAPI import BRepBuilderAPI_Transform
from OCC.Display.SimpleGui import init_display
from OCC.Core.TopAbs import TopAbs_SHELL, TopAbs_FACE

def print_shape_summary(shape):
    for t, name in [(TopAbs_SOLID, "SOLID"), (TopAbs_SHELL, "SHELL"), (TopAbs_FACE, "FACE"),(TopAbs_EDGE, "EDGE")]:
        count = 0
        print("=========================")
        exp = TopExp_Explorer(shape, t)
        while exp.More():
            count += 1
            exp.Next()
        print(f"{count} {name}(s) in shape")

# 1. Create a simple box
shape1 = BRepPrimAPI_MakeBox(60, 40, 30).Shape()
shape2 = BRepPrimAPI_MakeBox(30, 40, 45).Shape()

# Example: translate by (10, 20, 30)
dx, dy, dz = 10, 20, -30
trsf = gp_Trsf()
trsf.SetTranslation(gp_Vec(dx, dy, dz))
# Apply the transformation to your box
translated_shape = BRepBuilderAPI_Transform(shape2, trsf, True).Shape()

shapea = BRepAlgoAPI_Cut(shape1, translated_shape).Shape()
shapea = shape1
print_shape_summary(shapea)
# shape1 is likely a compound; extract the first solid
exp = TopExp_Explorer(shapea, TopAbs_SOLID)
if exp.More():
    solid = exp.Current()
    # Now you can use ShapeFix_Solid if you want
    fixer = ShapeFix_Solid()
    fixer.Init(solid)
    fixer.Perform()
    fixed_solid = fixer.Solid()
    shape = fixed_solid
else:
    print("No solid found in shape1; using shape1 directly.")
    shape = shape1

print_shape_summary(shapea)

# 4. Display orthographic views
display, start_display, add_menu, add_function_to_menu = init_display()
ais_shapes = display.DisplayShape(shape, update=True)  # Store the returned AIS_InteractiveObject(s)

# Set wireframe mode for all displayed shapes
if not isinstance(ais_shapes, (list, tuple)):
    ais_shapes = [ais_shapes]
for ais_shape in ais_shapes:
    display.Context.SetDisplayMode(ais_shape, 1, True)  # 1 = wireframe

def set_top_view(_=None):
    display.View.SetProj(0, 0, 1)    # Top view (Z+)
    for ais_shape in ais_shapes:
        display.Context.SetDisplayMode(ais_shape, 1, True)
    display.FitAll()

def set_front_view(_=None):
    display.View.SetProj(0, 1, 0)    # Front view (Y+)
    for ais_shape in ais_shapes:
        display.Context.SetDisplayMode(ais_shape, 1, True)
    display.FitAll()

def set_side_view(_=None):
    display.View.SetProj(1, 0, 0)    # Side view (X+)
    for ais_shape in ais_shapes:
        display.Context.SetDisplayMode(ais_shape, 1, True)
    display.FitAll()

add_menu("Views")
add_function_to_menu("Views", set_top_view)
add_function_to_menu("Views", set_front_view)
add_function_to_menu("Views", set_side_view)

if shape is None or (hasattr(shape, "IsNull") and shape.IsNull()):
    print("Shape is null or invalid!")

print("Use the 'Views' menu to switch between Top, Front, and Side views.")
start_display()

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

print_shape_summary(visible)
print_shape_summary(hidden)

# 5. Count and print the number of visible and hidden edges
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


# 6. (Optional) Print the 3D coordinates of the endpoints of visible edges
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
            if len(curve_data) == 3:
                last = curve_data[2]
            else:
                try:
                    last = edge.LastParameter()
                except AttributeError:
                    last = first
            p1 = gp_Pnt()
            p2 = gp_Pnt()
            try:
                curve_handle.D0(first, p1)
                curve_handle.D0(last, p2)
                print(f"Hidden edge: ({p1.X():.2f}, {p1.Y():.2f}, {p1.Z():.2f}) -> ({p2.X():.2f}, {p2.Y():.2f}, {p2.Z():.2f})")
            except Exception as e:
                print("Error evaluating edge:", e)
        exp.Next()
else:
    print("No hidden edges found.")

    # 7. (Optional) Print the 3D coordinates of the endpoints of visible edges
if visible is not None and hasattr(visible, "IsNull") and not visible.IsNull():
    exp = TopExp_Explorer(shapea, TopAbs_EDGE)
    while exp.More():
        edge = exp.Current()
        curve_data = BRep_Tool.Curve(edge)
        if curve_data is not None and len(curve_data) >= 2:
            curve_handle = curve_data[0]
            if curve_handle is None or not hasattr(curve_handle, "D0"):
                exp.Next()
                continue
            first = curve_data[1]
            if len(curve_data) == 3:
                last = curve_data[2]
            else:
                try:
                    last = edge.LastParameter()
                except AttributeError:
                    last = first
            p1 = gp_Pnt()
            p2 = gp_Pnt()
            try:
                curve_handle.D0(first, p1)
                curve_handle.D0(last, p2)
                print(f"visible edge: ({p1.X():.2f}, {p1.Y():.2f}, {p1.Z():.2f}) -> ({p2.X():.2f}, {p2.Y():.2f}, {p2.Z():.2f})")
            except Exception as e:
                print("Error evaluating edge:", e)
        exp.Next()
else:
    print("No visible edges found.")