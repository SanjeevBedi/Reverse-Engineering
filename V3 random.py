import random
import sys
sys.path.append('/opt/anaconda3/envs/pyocc/lib/python3.9/site-packages')
sys.path.append('/Users/sbedi/Documents/EAGIS/pythonocc-core-master/src/Display')
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
from OCC.Core.HLRAlgo import HLRAlgo_Projector

# 1. Create main cuboid
main_dims = [random.uniform(50, 100) for _ in range(3)]
main_box = BRepPrimAPI_MakeBox(*main_dims).Shape()

# 2. Add or subtract up to 10 smaller cuboids that penetrate the main cuboid
shape = main_box
for _ in range(random.randint(2, 5)):
    # Random size, smaller than main box
    dims = [random.uniform(10, min(main_dims[i], 40)) for i in range(3)]
    # Position: allow negative start or start beyond main box so it penetrates
    pos = [random.uniform(-dims[i] * 0.7, main_dims[i] - dims[i] * 0.3) for i in range(3)]
    small_box = BRepPrimAPI_MakeBox(gp_Pnt(*pos), *dims).Shape()
    if random.choice([True, False]):
        shape = BRepAlgoAPI_Fuse(shape, small_box).Shape()
    else:
        shape = BRepAlgoAPI_Cut(shape, small_box).Shape()

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

# Save the three views as PNGs
save_view("top_view.png", set_top_view)
save_view("front_view.png", set_front_view)
save_view("side_view.png", set_side_view)

print("\nLaTeX code to include the views in the requested layout:\n")
print(r"""\begin{figure}[ht]
  \centering
  \begin{tabular}{cc}
    \includegraphics[width=0.3\textwidth]{top_view.png} & \\
    \includegraphics[width=0.3\textwidth]{front_view.png} &
    \includegraphics[width=0.3\textwidth]{side_view.png} \\
    \textbf{Top view} & \textbf{Side view} \\
    \textbf{Front view} & \\
  \end{tabular}
  \caption{Orthographic projections: top, front, and side views.}
\end{figure}
""")

# Optionally, start the GUI for interactive use
start_display()


print("Use the 'Views' menu to switch between Top, Front, and Side views.")
start_display()


def get_hlr_edges(shape, view):
    # Set up projection direction and up vector for the view
    if view == 'top':
        axis = gp_Ax2(gp_Pnt(0, 0, 0), gp_Dir(0, 0, 1), gp_Dir(0, 1, 0))  # Z up, Y vertical
    elif view == 'front':
        axis = gp_Ax2(gp_Pnt(0, 0, 0), gp_Dir(0, 1, 0), gp_Dir(0, 0, 1))  # Y up, Z vertical
    elif view == 'side':
        axis = gp_Ax2(gp_Pnt(0, 0, 0), gp_Dir(1, 0, 0), gp_Dir(0, 0, 1))  # X up, Z vertical
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

def get_edges_projection_from_shape(shape, view):
    """Return a list of 2D line segments for the given view."""
    lines = []
    if shape is None or (hasattr(shape, "IsNull") and shape.IsNull()):
        return lines
    exp = TopExp_Explorer(shape, TopAbs_EDGE)
    while exp.More():
        edge = exp.Current()
        curve_data = BRep_Tool.Curve(edge)
        # Check curve_data is a tuple/list and first element is a curve object
        if (
            curve_data is not None
            and isinstance(curve_data, (tuple, list))
            and len(curve_data) >= 2
        ):
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

# Get visible and hidden edges for each view
top_visible, top_hidden = get_hlr_edges(shape, 'top')
front_visible, front_hidden = get_hlr_edges(shape, 'front')
side_visible, side_hidden = get_hlr_edges(shape, 'side')

# Project to 2D
top_lines_vis = get_edges_projection_from_shape(top_visible, 'top')
top_lines_hid = get_edges_projection_from_shape(top_hidden, 'top')
front_lines_vis = get_edges_projection_from_shape(front_visible, 'front')
front_lines_hid = get_edges_projection_from_shape(front_hidden, 'front')
side_lines_vis = get_edges_projection_from_shape(side_visible, 'side')
side_lines_hid = get_edges_projection_from_shape(side_hidden, 'side')

# Output TikZ code for each view
def tikz_lines(lines, style='solid'):
    tikz = []
    dash = ',dashed' if style == 'dashed' else ''
    for (x1, y1), (x2, y2) in lines:
        tikz.append(f"\\draw[black{dash}] ({x1:.2f},{y1:.2f}) -- ({x2:.2f},{y2:.2f});")
    return "\n".join(tikz)

print("\n% TikZ for Top View")
print("\\begin{tikzpicture}[scale=0.05]")
print(tikz_lines(top_lines_vis, style='solid'))
print(tikz_lines(top_lines_hid, style='dashed'))
print("\\end{tikzpicture}")

print("\n% TikZ for Front View")
print("\\begin{tikzpicture}[scale=0.05]")
print(tikz_lines(front_lines_vis, style='solid'))
print(tikz_lines(front_lines_hid, style='dashed'))
print("\\end{tikzpicture}")

print("\n% TikZ for Side View")
print("\\begin{tikzpicture}[scale=0.05]")
print(tikz_lines(side_lines_vis, style='solid'))
print(tikz_lines(side_lines_hid, style='dashed'))
print("\\end{tikzpicture}")