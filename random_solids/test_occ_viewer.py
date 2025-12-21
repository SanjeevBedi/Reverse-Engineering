from OCC.Core.BRepPrimAPI import BRepPrimAPI_MakeBox
from OCC.Display.SimpleGui import init_display

if __name__ == "__main__":
    print("[OCC] Creating a simple box solid...")
    box = BRepPrimAPI_MakeBox(10, 20, 30).Shape()
    print("[OCC] Initializing OCC viewer...")
    display, start_display, add_menu, add_function_to_menu = init_display()
    display.DisplayShape(box, update=True)
    display.FitAll()
    print("[OCC] OCC viewer event loop starting now...")
    start_display()
    print("[OCC] OCC viewer event loop exited.")
