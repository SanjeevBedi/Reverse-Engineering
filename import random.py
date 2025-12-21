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

# 1. Create main cuboid
main_dims = [random.uniform(50, 100) for _ in range(3)]
main_dims = [30,40,50]
main_box = BRepPrimAPI_MakeBox(*main_dims).Shape()

# 2. Add or subtract up to 10 smaller cuboids
shape = main_box
for _ in range(random.randint(1, 2)):
    # Random size, smaller than main box
    dims = [random.uniform(10, min(main_dims[i] - 10, 40)) for i in range(3)]
    dims = [10,20,30]
    # Random position inside main box
    pos = [random.uniform(0, main_dims[i] - dims[i]) for i in range(3)]
    pos = [10, 10, 30]
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

# 4. Display orthographic views
display, start_display, add_menu, add_function_to_menu = init_display()
display.DisplayShape(shape, update=True)

# # Set up orthographic views
# def set_top_view():
#     display.View_Top()
#     display.FitAll()

# def set_front_view():
#     display.View_Front()
#     display.FitAll()

# def set_side_view():
#     display.View_Right()
#     display.FitAll()

# add_menu("Views")
# add_function_to_menu("Views", set_top_view)
# add_function_to_menu("Views", set_front_view)
# add_function_to_menu("Views", set_side_view)

# print("Use the 'Views' menu to switch between Top, Front, and Side views.")
# start_display()

# Set up orthographic views using SetProj
# def set_top_view():
#     display.View.SetProj(0, 0, 1)    # Top view (Z+)
#     display.FitAll()

# def set_front_view():
#     display.View.SetProj(0, 1, 0)    # Front view (Y+)
#     display.FitAll()

# def set_side_view():
#     display.View.SetProj(1, 0, 0)    # Side view (X+)
#     display.FitAll()

# add_menu("Views")
# add_function_to_menu("Views", set_top_view)
# add_function_to_menu("Views", set_front_view)
# add_function_to_menu("Views", set_side_view)

# print("Use the 'Views' menu to switch between Top, Front, and Side views.")
# start_display()
# ...existing code...

def set_top_view(_=None):
    display.View.SetProj(0, 0, 1)    # Top view (Z+)
    display.FitAll()

def set_front_view(_=None):
    display.View.SetProj(0, 1, 0)    # Front view (Y+)
    display.FitAll()

def set_side_view(_=None):
    display.View.SetProj(1, 0, 0)    # Side view (X+)
    display.FitAll()

add_menu("Views")
add_function_to_menu("Views", set_top_view)
add_function_to_menu("Views", set_front_view)
add_function_to_menu("Views", set_side_view)

print("Use the 'Views' menu to switch between Top, Front, and Side views.")
start_display()