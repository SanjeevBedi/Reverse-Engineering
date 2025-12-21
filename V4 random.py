
import sys
import os
os.environ["DYLD_LIBRARY_PATH"] = "/Users/sbedi/Anaconda/anaconda3/lib"
sys.path.append('/opt/anaconda3/envs/pyocc/lib/python3.9/site-packages')
#sys.path.append('/Users/sbedi/Documents/EAGIS/pythonocc-core-master/src/Display')
#sys.path.append('/Users/sbedi/Nextcloud/Python/Legendre/pyocc/lib/python3.8/site-packages')  
#sys.path.append('/Users/sbedi/Anaconda/anaconda3/pkgs/pythonocc-core-7.4.0-py38h6efaf97_0/lib/python3.8/site-packages')
from OCC.Core.BRepPrimAPI import BRepPrimAPI_MakeBox
from OCC.Core.BRepAlgoAPI import BRepAlgoAPI_Fuse, BRepAlgoAPI_Cut
from OCC.Core.gp import gp_Pnt
from OCC.Display.SimpleGui import init_display
from OCC.Core.STEPControl import STEPControl_Writer, STEPControl_AsIs
from OCC.Core.IFSelect import IFSelect_RetDone
from OCC.Core.TopExp import TopExp_Explorer
from OCC.Core.TopAbs import TopAbs_EDGE
from OCC.Core.BRep import BRep_Tool
from OCC.Core.HLRBRep import HLRBRep_Algo, HLRBRep_HLRToShape
from OCC.Core.gp import gp_Dir, gp_Pnt, gp_Ax2
from OCC.Core.Geom import Geom_Plane, Geom_Line
from OCC.Core.HLRAlgo import HLRAlgo_Projector
from OCC.Core.ShapeFix import ShapeFix_Solid
from OCC.Core.TopoDS import topods_Solid
from OCC.Core.TopExp import TopExp_Explorer
from OCC.Core.TopAbs import TopAbs_SOLID
from OCC.Core.TopAbs import TopAbs_VERTEX, TopAbs_EDGE, TopAbs_WIRE, TopAbs_FACE, TopAbs_SHELL, TopAbs_SOLID, TopAbs_COMPOUND
import random

# 1. Create main cuboid
main_dims = [random.uniform(50, 100) for _ in range(3)]
main_box = BRepPrimAPI_MakeBox(*main_dims).Shape()
#main_box = BRepPrimAPI_MakeBox(60, 40, 30).Shape()

# 2. Add or subtract up to 10 smaller cuboids that penetrate the main cuboid
shape1 = main_box
for _ in range(random.randint(2, 5)):
    # Random size, smaller than main box
    dims = [random.uniform(10, min(main_dims[i], 40)) for i in range(3)]
    # Position: allow negative start or start beyond main box so it penetrates
    pos = [random.uniform(-dims[i] * 0.7, main_dims[i] - dims[i] * 0.3) for i in range(3)]
    small_box = BRepPrimAPI_MakeBox(gp_Pnt(*pos), *dims).Shape()
    if random.choice([True, False]):
        shape1 = BRepAlgoAPI_Fuse(shape1, small_box).Shape()
    else:
        shape1 = BRepAlgoAPI_Cut(shape1, small_box).Shape()





# shape1 is likely a compound; extract the first solid
exp = TopExp_Explorer(shape1, TopAbs_SOLID)
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

def print_shape_summary(shape):
    for t, name in [(TopAbs_SOLID, "SOLID"), (TopAbs_SHELL, "SHELL"), (TopAbs_FACE, "FACE"),(TopAbs_EDGE, "EDGE")]:
        count = 0
        print("=========================")
        exp = TopExp_Explorer(shape, t)
        while exp.More():
            count += 1
            exp.Next()
        print(f"{count} {name}(s) in shape")

print_shape_summary(shape1)

# # Suppose 'shape' is your TopoDS_Shape (e.g., after Booleans)
# # Convert to TopoDS_Solid if needed
# solid = topods_Solid(shape1)

# # Create the fixer and apply it
# fixer = ShapeFix_Solid()
# fixer.Init(solid)
# fixer.Perform()
# fixed_solid = fixer.Solid()
# shape = fixed_solid

# Now use 'fixed_solid' for HLR or export

# 3. Save as STEP
step_writer = STEPControl_Writer()
step_writer.Transfer(shape, STEPControl_AsIs)
status = step_writer.Write("random_cuboid_shape.step")
if status == IFSelect_RetDone:
    print("STEP file saved as random_cuboid_shape.step")
else:
    print("Failed to save STEP file.")


# Set wireframe mode for all displayed shapes

# 4. Display orthographic views
display, start_display, add_menu, add_function_to_menu = init_display()
ais_shapes = display.DisplayShape(shape, update=True)  # Store the returned AIS_InteractiveObject(s)

# Set wireframe mode for all displayed shapes
if not isinstance(ais_shapes, (list, tuple)):
    ais_shapes = [ais_shapes]
for ais_shape in ais_shapes:
    display.Context.SetDisplayMode(ais_shape, 1, True)  # 1 = wireframe

# def set_top_view(_=None):
#     display.View.SetProj(0, 0, 1)    # Top view (Z+)
#     for ais_shape in ais_shapes:
#         display.Context.SetDisplayMode(ais_shape, 1, True)
#     display.FitAll()

# def set_front_view(_=None):
#     display.View.SetProj(0, 1, 0)    # Front view (Y+)
#     for ais_shape in ais_shapes:
#         display.Context.SetDisplayMode(ais_shape, 1, True)
#     display.FitAll()

# def set_side_view(_=None):
#     display.View.SetProj(1, 0, 0)    # Side view (X+)
#     for ais_shape in ais_shapes:
#         display.Context.SetDisplayMode(ais_shape, 1, True)
#     display.FitAll()
import time

def save_view(filename, set_view_func):
    set_view_func()
    display.FitAll()
    # Give the GUI time to update (important for some systems)
    time.sleep(0.5)
    display.View.Dump(filename)

def set_top_view():
    display.View.SetProj(0, 0, 1)    # Top view (Z+)
    for ais_shape in ais_shapes:
        display.Context.SetDisplayMode(ais_shape, 1, True)

def set_front_view():
    display.View.SetProj(0, 1, 0)    # Front view (Y+)
    for ais_shape in ais_shapes:
        display.Context.SetDisplayMode(ais_shape, 1, True)

def set_side_view():
    display.View.SetProj(1, 0, 0)    # Side view (X+)
    for ais_shape in ais_shapes:
        display.Context.SetDisplayMode(ais_shape, 1, True)


add_menu("Views")
add_function_to_menu("Views", set_top_view)
add_function_to_menu("Views", set_front_view)
add_function_to_menu("Views", set_side_view)


# Save the three views as PNGs
save_view("top_view.png", set_top_view)
save_view("front_view.png", set_front_view)
save_view("side_view.png", set_side_view)

# print("\nLaTeX code to include the views in the requested layout:\n")
# print(r"""\begin{figure}[ht]
#   \centering
#   \begin{tabular}{cc}
#     \includegraphics[width=0.3\textwidth]{top_view.png} & \\
#     \includegraphics[width=0.3\textwidth]{front_view.png} &
#     \includegraphics[width=0.3\textwidth]{side_view.png} \\
#     \textbf{Top view} & \textbf{Side view} \\
#     \textbf{Front view} & \\
#   \end{tabular}
#   \caption{Orthographic projections: top, front, and side views.}
# \end{figure}
# """)

# # Optionally, start the GUI for interactive use
# start_display()
if shape is None or (hasattr(shape, "IsNull") and shape.IsNull()):
    print("Shape is null or invalid!")

print("Use the 'Views' menu to switch between Top, Front, and Side views.")
start_display()

def project_point(pnt, view):
    if view == 'top':
        return (pnt.X(), pnt.Y())
    elif view == 'front':
        return (pnt.X(), pnt.Z())
    elif view == 'side':
        return (pnt.Y(), pnt.Z())
    else:
        return (pnt.X(), pnt.Y())

#Does not work with HLRAlgo_Projector the code is empty
def get_hlr_edges(shape, view):
    if view == 'top':
        axis = gp_Ax2(gp_Pnt(0, 0, 0), gp_Dir(1, 0, 0), gp_Dir(0, 1, 0))
    elif view == 'front':
        axis = gp_Ax2(gp_Pnt(0, 0, 0), gp_Dir(0, 1, 0), gp_Dir(0, 0, 1))
    elif view == 'side':
        axis = gp_Ax2(gp_Pnt(0, 0, 0), gp_Dir(1, 0, 0), gp_Dir(0, 0, 1))
    else:
        raise ValueError("Unknown view")
    projector = HLRAlgo_Projector(axis)
    algo = HLRBRep_Algo()
    algo.Add(shape)
    algo.Projector(projector)
    algo.Update()
    algo.Hide()
    hlr_shapes = HLRBRep_HLRToShape(algo)
    visible = hlr_shapes.VCompound()
    hidden = hlr_shapes.Rg1LineVCompound()
    return visible, hidden

# def get_edges_projection(shape, view):
#     lines = []
#     if shape is None or (hasattr(shape, "IsNull") and shape.IsNull()):
#         return lines
#     exp = TopExp_Explorer(shape, TopAbs_EDGE)
#     while exp.More():
#         edge = exp.Current()
#         curve_data = BRep_Tool.Curve(edge)
#         if curve_data is not None and len(curve_data) >= 2:
#             curve_handle = curve_data[0]
#             if curve_handle is None or not hasattr(curve_handle, "D0"):
#                 exp.Next()
#                 continue
#             first = curve_data[1]
#             if len(curve_data) == 3:
#                 last = curve_data[2]
#             else:
#                 try:
#                     last = edge.LastParameter()
#                 except AttributeError:
#                     last = first
#             if last is None:
#                 exp.Next()
#                 continue
#             p1 = gp_Pnt()
#             p2 = gp_Pnt()
#             try:
#                 curve_handle.D0(first, p1)
#                 curve_handle.D0(last, p2)
#             except Exception:
#                 exp.Next()
#                 continue
#             pt1_2d = project_point(p1, view)
#             pt2_2d = project_point(p2, view)
#             lines.append((pt1_2d, pt2_2d))
#         exp.Next()
#     return lines

# def get_edges_projection(shape, view):
#     """Return a list of 2D line segments for the given view."""
#     lines = []
#     exp = TopExp_Explorer(shape, TopAbs_EDGE)
#     while exp.More():
#         edge = exp.Current()
#         curve_data = BRep_Tool.Curve(edge)
#         if curve_data is not None and len(curve_data) >= 2:
#             #curve_handle = curve_data[0]
#             curve_handle, first, last = BRep_Tool.Curve(edge)
#         # if curve_handle is not None:
#             p1 = gp_Pnt()
#             p2 = gp_Pnt()
#             curve_handle.D0(first, p1)
#             curve_handle.D0(last, p2)
#             pt1_2d = project_point(p1, view)
#             pt2_2d = project_point(p2, view)
#             lines.append((pt1_2d, pt2_2d))
#         exp.Next()

def print_compound_contents(compound):
    shape_types = {
        TopAbs_VERTEX: "VERTEX",
        TopAbs_EDGE: "EDGE",
        TopAbs_WIRE: "WIRE",
        TopAbs_FACE: "FACE",
        TopAbs_SHELL: "SHELL",
        TopAbs_SOLID: "SOLID",
        TopAbs_COMPOUND: "COMPOUND"
    }
    for shape_type, name in shape_types.items():
        count = 0
        exp = TopExp_Explorer(compound, shape_type)
        while exp.More():
            count += 1
            exp.Next()
        if count > 0:
            print(f"{count} {name}(s) found in compound.")

# Usage:
#print_compound_contents(your_compound_shape)#     return lines

def print_compound_details(compound):
    shape_types = {
        TopAbs_VERTEX: "VERTEX",
        TopAbs_EDGE: "EDGE",
        TopAbs_WIRE: "WIRE",
        TopAbs_FACE: "FACE",
        TopAbs_SHELL: "SHELL",
        TopAbs_SOLID: "SOLID",
        TopAbs_COMPOUND: "COMPOUND"
    }
    for shape_type, name in shape_types.items():
        exp = TopExp_Explorer(compound, shape_type)
        idx = 0
        while exp.More():
            shape = exp.Current()
            print(f"{name} #{idx}: {shape}")
            idx += 1
            exp.Next()

def print_edge_geometry(shape):
    exp = TopExp_Explorer(shape, TopAbs_EDGE)
    idx = 0
    while exp.More():
        edge = exp.Current()
        curve_data = BRep_Tool.Curve(edge)
        if curve_data is not None and len(curve_data) >= 2:
            curve_handle = curve_data[0]
            if curve_handle is None or not hasattr(curve_handle, "D0"):
                exp.Next()
                continue
            first = curve_data[1]
            # Get last parameter robustly
            if len(curve_data) == 3:
                last = curve_data[2]
            else:
                try:
                    last = edge.LastParameter()
                except AttributeError:
                    last = first
            if last is None:
                exp.Next()
                continue
            p1 = gp_Pnt()
            p2 = gp_Pnt()
            try:
                curve_handle.D0(first, p1)
                curve_handle.D0(last, p2)
                print(f"Edge #{idx}: Start ({p1.X():.2f}, {p1.Y():.2f}, {p1.Z():.2f}) "
                      f"End ({p2.X():.2f}, {p2.Y():.2f}, {p2.Z():.2f})")
            except Exception as e:
                print(f"Edge #{idx}: Error extracting points: {e}")
            idx += 1
        exp.Next()


def face_equations(shape):
    exp = TopExp_Explorer(shape, TopAbs_FACE)
    while exp.More():
        face = exp.Current()
        surf = BRep_Tool.Surface(face)
        if surf and surf.DynamicType().Name() == "Geom_Plane":
            plane = Geom_Plane.DownCast(surf)
            loc = plane.Location()
            norm = plane.Axis().Direction()
            # Plane equation: norm.X()*(X - loc.X()) + norm.Y()*(Y - loc.Y()) + norm.Z()*(Z - loc.Z()) = 0
            print(f"Plane: point=({loc.X():.2f},{loc.Y():.2f},{loc.Z():.2f}), normal=({norm.X():.2f},{norm.Y():.2f},{norm.Z():.2f})")
        exp.Next()

def line_equation(shape):
    print("inside line")
    exp = TopExp_Explorer(shape, TopAbs_EDGE)
    while exp.More():
        edge = exp.Current()
        curve_data = BRep_Tool.Curve(edge)
        if curve_data:
            curve_handle = curve_data[0]
            if curve_handle and curve_handle.DynamicType().Name() == "Geom_Line":
                line = Geom_Line.DownCast(curve_handle)
                loc = line.Position().Location()
                dir = line.Position().Direction()
                # Line equation: (X, Y, Z) = loc + t * dir
                print(f"Line: point=({loc.X():.2f},{loc.Y():.2f},{loc.Z():.2f}), direction=({dir.X():.2f},{dir.Y():.2f},{dir.Z():.2f})")
        exp.Next()
    print("exiting line")

def get_edges_projection(shape, view):
    """Return a list of 2D line segments for the given view."""
    lines = []
    if shape is None or (hasattr(shape, "IsNull") and shape.IsNull()):
        return lines
    print("==================================")
    print_compound_contents(shape)#     return lines
    face_equations(shape)
    line_equation(shape)
    #print_compound_details(shape)
    #print_edge_geometry(shape)
    print("==================================")
    exp = TopExp_Explorer(shape, TopAbs_EDGE)
    while exp.More():
        edge = exp.Current()
        curve_data = BRep_Tool.Curve(edge)
        if curve_data is not None and len(curve_data) >= 2:
            curve_handle = curve_data[0]
            if curve_handle is None or not hasattr(curve_handle, "D0"):
                exp.Next()
                continue
            first = curve_data[1]
            # Try to get last parameter robustly
            if len(curve_data) == 3:
                last = curve_data[2]
            else:
                try:
                    last = edge.LastParameter()
                except AttributeError:
                    last = first
            if last is None:
                exp.Next()
                continue
            p1 = gp_Pnt()
            p2 = gp_Pnt()
            try:
                curve_handle.D0(first, p1)
                curve_handle.D0(last, p2)
            except Exception:
                exp.Next()
                continue
            pt1_2d = project_point(p1, view)
            pt2_2d = project_point(p2, view)
            lines.append((pt1_2d, pt2_2d))
        exp.Next()
    return lines
# 
# --- MAIN HLR TO TIKZ PIPELINE ---

# top_visible, top_hidden = get_hlr_edges(shape, 'top')
# front_visible, front_hidden = get_hlr_edges(shape, 'front')
# side_visible, side_hidden = get_hlr_edges(shape, 'side')

# top_lines_vis = get_edges_projection(top_visible, 'top')
# top_lines_hid = get_edges_projection(top_hidden, 'top')
# front_lines_vis = get_edges_projection(front_visible, 'front')
# front_lines_hid = get_edges_projection(front_hidden, 'front')
# side_lines_vis = get_edges_projection(side_visible, 'side')
# side_lines_hid = get_edges_projection(side_hidden, 'side')

top_lines_vis = get_edges_projection(shape, 'top')
front_lines_vis = get_edges_projection(shape, 'front')
side_lines_vis = get_edges_projection(shape, 'side')

print("Top visible edge count:", len(top_lines_vis))
print("Front visible edge count:", len(front_lines_vis))
print("Side visible edge count:", len(side_lines_vis))
#print("Top hidden edge count:", len(top_lines_hid))

def tikz_lines(lines, style='solid'):
    tikz = []
    dash = ',dashed' if style == 'dashed' else ''
    for (x1, y1), (x2, y2) in lines:
        tikz.append(f"\\draw[black{dash}] ({x1:.2f},{y1:.2f}) -- ({x2:.2f},{y2:.2f});")
    return "\n".join(tikz)
print("\\begin{tikzpicture}\conda\node at (0,0) {")
print("\n% TikZ for Top View")
print("\\begin{tikzpicture}[scale=0.05]")
print(tikz_lines(top_lines_vis, style='solid'))
#print(tikz_lines(top_lines_hid, style='dashed'))
print("\\end{tikzpicture}};")

print("\n% TikZ for Front View")
print("\\node at (0,-10) {")
print("\\begin{tikzpicture}[scale=0.05]")
print(tikz_lines(front_lines_vis, style='solid'))
#print(tikz_lines(front_lines_hid, style='dashed'))
print("\\end{tikzpicture}};")

print("\n% TikZ for Side View")
print("\\node at (10,-10) {")
print("\\begin{tikzpicture}[scale=0.05]")
print(tikz_lines(side_lines_vis, style='solid'))
#print(tikz_lines(side_lines_hid, style='dashed'))
print("\\end{tikzpicture}};")
print("\\end{tikzpicture}")