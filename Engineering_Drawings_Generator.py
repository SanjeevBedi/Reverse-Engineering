import sys
import os
os.environ["DYLD_LIBRARY_PATH"] = "/Users/sbedi/Anaconda/anaconda3/lib"
sys.path.append('/opt/anaconda3/envs/pyocc/lib/python3.9/site-packages')

import random
import numpy as np
from OCC.Core.BRepPrimAPI import BRepPrimAPI_MakeBox
from OCC.Core.BRepAlgoAPI import BRepAlgoAPI_Cut, BRepAlgoAPI_Fuse
from OCC.Core.gp import gp_Pnt, gp_Vec, gp_Trsf, gp_Ax1, gp_Dir
from OCC.Core.BRepBuilderAPI import BRepBuilderAPI_Transform
from OCC.Core.TopExp import TopExp_Explorer
from OCC.Core.TopAbs import TopAbs_FACE, TopAbs_EDGE
from OCC.Core.BRep import BRep_Tool
from OCC.Core.HLRBRep import HLRBRep_Algo, HLRBRep_HLRToShape
from OCC.Core.HLRAlgo import HLRAlgo_Projector
from OCC.Core.gp import gp_Ax2, gp_Dir, gp_Pnt
from OCC.Core.Graphic3d import Graphic3d_Camera
from OCC.Display.SimpleGui import init_display
from OCC.Core.STEPControl import STEPControl_Writer, STEPControl_AsIs
from OCC.Core.IFSelect import IFSelect_RetDone
from OCC.Core.Interface import Interface_Static_SetCVal
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import Rectangle
from matplotlib.lines import Line2D

class EngineeringDrawingGenerator:
    def __init__(self, base_length=100, base_width=80, base_height=60):
        self.base_length = base_length
        self.base_width = base_width  
        self.base_height = base_height
        self.main_shape = None
        self.features = []
        self.feature_count = 0
        
    def create_base_cuboid(self):
        """Create the main base cuboid."""
        print(f"Creating base cuboid: {self.base_length} x {self.base_width} x {self.base_height}")
        self.main_shape = BRepPrimAPI_MakeBox(self.base_length, self.base_width, self.base_height).Shape()
        return self.main_shape
    
    def add_random_feature(self, on_top_face=True, is_addition=True):
        """Add or subtract a random cuboid feature."""
        if self.feature_count >= 10:
            return None
            
        # Random dimensions for the feature (smaller than base)
        feat_length = random.uniform(10, self.base_length * 0.4)
        feat_width = random.uniform(10, self.base_width * 0.4)
        feat_height = random.uniform(5, self.base_height * 0.3)
        
        if on_top_face:
            # Feature on top face
            x_pos = random.uniform(5, self.base_length - feat_length - 5)
            y_pos = random.uniform(5, self.base_width - feat_width - 5)
            z_pos = self.base_height if is_addition else self.base_height - feat_height
        else:
            # Feature on side face (front face)
            x_pos = random.uniform(5, self.base_length - feat_length - 5)
            y_pos = 0 if is_addition else -feat_width
            z_pos = random.uniform(5, self.base_height - feat_height - 5)
            
        # Create the feature
        feature = BRepPrimAPI_MakeBox(gp_Pnt(x_pos, y_pos, z_pos), 
                                     feat_length, feat_width, feat_height).Shape()
        
        # Apply boolean operation
        if is_addition:
            fuse_op = BRepAlgoAPI_Fuse(self.main_shape, feature)
            if fuse_op.IsDone():
                self.main_shape = fuse_op.Shape()
                operation = "Added"
        else:
            cut_op = BRepAlgoAPI_Cut(self.main_shape, feature)
            if cut_op.IsDone():
                self.main_shape = cut_op.Shape()
                operation = "Subtracted"
        
        self.features.append({
            'operation': operation,
            'position': (x_pos, y_pos, z_pos),
            'dimensions': (feat_length, feat_width, feat_height),
            'on_top': on_top_face
        })
        
        self.feature_count += 1
        print(f"{operation} feature {self.feature_count}: {feat_length:.1f}x{feat_width:.1f}x{feat_height:.1f} at ({x_pos:.1f},{y_pos:.1f},{z_pos:.1f})")
        
        return feature
    
    def generate_random_features(self, num_features=8):
        """Generate random features on the model."""
        print(f"\nGenerating {num_features} random features...")
        
        for i in range(num_features):
            # Randomly choose face and operation
            on_top = random.choice([True, False])
            is_addition = random.choice([True, False, False])  # Bias towards subtractions
            
            self.add_random_feature(on_top_face=on_top, is_addition=is_addition)
            
        print(f"Generated {self.feature_count} features total")
    
    def save_step_file(self, filename="engineering_model.step"):
        """Save the model as a STEP file."""
        step_writer = STEPControl_Writer()
        Interface_Static_SetCVal("write.step.schema", "AP203")
        
        step_writer.Transfer(self.main_shape, STEPControl_AsIs)
        status = step_writer.Write(filename)
        
        if status == IFSelect_RetDone:
            print(f"Model saved as {filename}")
        else:
            print("Failed to save STEP file")
    
    def get_bounding_box(self, shape):
        """Get bounding box of a shape."""
        from OCC.Core.Bnd import Bnd_Box
        from OCC.Core.BRepBndLib import brepbndlib_Add
        
        bbox = Bnd_Box()
        brepbndlib_Add(shape, bbox)
        xmin, ymin, zmin, xmax, ymax, zmax = bbox.Get()
        return xmin, ymin, zmin, xmax, ymax, zmax
    
    def extract_edges_for_view(self, view_direction):
        """Extract visible and hidden edges for a specific view direction."""
        visible_edges = []
        hidden_edges = []
        
        # Create HLR algorithm
        hlr_algo = HLRBRep_Algo()
        hlr_algo.Add(self.main_shape)
        
        # Set up projector based on view direction
        if view_direction == "front":
            # Looking from negative Y direction
            projector = HLRAlgo_Projector(gp_Ax2(gp_Pnt(0, -1000, 0), gp_Dir(0, 1, 0)))
        elif view_direction == "top":
            # Looking from positive Z direction  
            projector = HLRAlgo_Projector(gp_Ax2(gp_Pnt(0, 0, 1000), gp_Dir(0, 0, -1)))
        elif view_direction == "side":
            # Looking from positive X direction
            projector = HLRAlgo_Projector(gp_Ax2(gp_Pnt(1000, 0, 0), gp_Dir(-1, 0, 0)))
        
        hlr_algo.Projector(projector)
        hlr_algo.Update()
        hlr_algo.Hide()
        
        # Extract results
        hlr_to_shape = HLRBRep_HLRToShape(hlr_algo)
        
        # Get visible edges
        visible_compound = hlr_to_shape.VCompound()
        if not visible_compound.IsNull():
            edge_explorer = TopExp_Explorer(visible_compound, TopAbs_EDGE)
            while edge_explorer.More():
                edge = edge_explorer.Current()
                visible_edges.append(self.edge_to_2d_coords(edge, view_direction))
                edge_explorer.Next()
        
        # Get hidden edges  
        hidden_compound = hlr_to_shape.HCompound()
        if not hidden_compound.IsNull():
            edge_explorer = TopExp_Explorer(hidden_compound, TopAbs_EDGE)
            while edge_explorer.More():
                edge = edge_explorer.Current()
                hidden_edges.append(self.edge_to_2d_coords(edge, view_direction))
                edge_explorer.Next()
        
        return visible_edges, hidden_edges
    
    def edge_to_2d_coords(self, edge, view_direction):
        """Convert 3D edge to 2D coordinates based on view direction."""
        curve, first, last = BRep_Tool.Curve(edge)
        if curve is None:
            return None
            
        # Get start and end points
        p1 = curve.Value(first)
        p2 = curve.Value(last)
        
        # Project to 2D based on view direction
        if view_direction == "front":
            # X-Z plane (looking along Y)
            return [(p1.X(), p1.Z()), (p2.X(), p2.Z())]
        elif view_direction == "top":
            # X-Y plane (looking along Z)
            return [(p1.X(), p1.Y()), (p2.X(), p2.Y())]
        elif view_direction == "side":
            # Y-Z plane (looking along X)
            return [(p1.Y(), p1.Z()), (p2.X(), p2.Z())]
        
        return None
    
    def create_engineering_drawings(self):
        """Create front, top, and side view engineering drawings."""
        print("\nGenerating engineering drawings...")
        
        # Create figure with three subplots
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6))
        
        views = [
            ("Front View", "front", ax1),
            ("Top View", "top", ax2), 
            ("Side View", "side", ax3)
        ]
        
        for view_name, view_dir, ax in views:
            print(f"Processing {view_name}...")
            
            # Simple projection method (fallback if HLR doesn't work)
            self.draw_simple_projection(ax, view_dir, view_name)
        
        plt.tight_layout()
        plt.savefig('engineering_drawings.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print("Engineering drawings saved as 'engineering_drawings.png'")
    
    def draw_simple_projection(self, ax, view_direction, title):
        """Draw a simple orthographic projection of the model."""
        # Get all edges from the shape
        edges_2d = []
        edge_explorer = TopExp_Explorer(self.main_shape, TopAbs_EDGE)
        
        while edge_explorer.More():
            edge = edge_explorer.Current()
            curve, first, last = BRep_Tool.Curve(edge)
            
            if curve is not None:
                p1 = curve.Value(first)
                p2 = curve.Value(last)
                
                # Project based on view direction
                if view_direction == "front":
                    # X-Z projection (front view)
                    line = [(p1.X(), p1.Z()), (p2.X(), p2.Z())]
                elif view_direction == "top":
                    # X-Y projection (top view)
                    line = [(p1.X(), p1.Y()), (p2.X(), p2.Y())]
                elif view_direction == "side":
                    # Y-Z projection (side view)
                    line = [(p1.Y(), p1.Z()), (p2.Y(), p2.Z())]
                
                edges_2d.append(line)
            
            edge_explorer.Next()
        
        # Draw visible edges (solid lines)
        for edge in edges_2d:
            if len(edge) == 2:
                x_coords = [edge[0][0], edge[1][0]]
                y_coords = [edge[0][1], edge[1][1]]
                ax.plot(x_coords, y_coords, 'b-', linewidth=1.0)
        
        # Add some hidden lines (dashed) - simplified approach
        # For demonstration, we'll add some internal feature lines as dashed
        for feature in self.features:
            if feature['operation'] == 'Subtracted':
                x, y, z = feature['position']
                l, w, h = feature['dimensions']
                
                if view_direction == "front":
                    # Draw rectangle outline as hidden lines
                    rect_x, rect_y = x, z
                    rect_w, rect_h = l, h
                elif view_direction == "top":
                    rect_x, rect_y = x, y
                    rect_w, rect_h = l, w
                elif view_direction == "side":
                    rect_x, rect_y = y, z
                    rect_w, rect_h = w, h
                
                # Draw dashed rectangle for hidden features
                ax.plot([rect_x, rect_x + rect_w], [rect_y, rect_y], 'r--', linewidth=0.8, alpha=0.7)
                ax.plot([rect_x + rect_w, rect_x + rect_w], [rect_y, rect_y + rect_h], 'r--', linewidth=0.8, alpha=0.7)
                ax.plot([rect_x + rect_w, rect_x], [rect_y + rect_h, rect_y + rect_h], 'r--', linewidth=0.8, alpha=0.7)
                ax.plot([rect_x, rect_x], [rect_y + rect_h, rect_y], 'r--', linewidth=0.8, alpha=0.7)
        
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)
        ax.set_title(title, fontsize=14, fontweight='bold')
        
        # Set axis labels based on view
        if view_direction == "front":
            ax.set_xlabel('X (Length)')
            ax.set_ylabel('Z (Height)')
        elif view_direction == "top":
            ax.set_xlabel('X (Length)')
            ax.set_ylabel('Y (Width)')
        elif view_direction == "side":
            ax.set_xlabel('Y (Width)')
            ax.set_ylabel('Z (Height)')
    
    def display_3d_model(self):
        """Display the 3D model using OCC viewer."""
        print("\nDisplaying 3D model...")
        
        try:
            display, start_display, add_menu, add_function_to_menu = init_display()
            display.DisplayShape(self.main_shape, update=True)
            display.FitAll()
            
            print("3D model displayed. Close the window to continue...")
            start_display()
            
        except Exception as e:
            print(f"Could not display 3D model: {e}")
            print("Continuing with 2D drawings generation...")
    
    def print_model_summary(self):
        """Print a summary of the generated model."""
        print("\n" + "="*60)
        print("MODEL SUMMARY")
        print("="*60)
        print(f"Base Cuboid: {self.base_length} x {self.base_width} x {self.base_height}")
        print(f"Total Features: {len(self.features)}")
        
        additions = [f for f in self.features if f['operation'] == 'Added']
        subtractions = [f for f in self.features if f['operation'] == 'Subtracted']
        
        print(f"Additions: {len(additions)}")
        print(f"Subtractions: {len(subtractions)}")
        
        top_features = [f for f in self.features if f['on_top']]
        side_features = [f for f in self.features if not f['on_top']]
        
        print(f"Features on top face: {len(top_features)}")
        print(f"Features on side face: {len(side_features)}")
        
        print("\nFeature Details:")
        for i, feature in enumerate(self.features, 1):
            face = "Top" if feature['on_top'] else "Side"
            print(f"  {i}. {feature['operation']} on {face} face: "
                  f"{feature['dimensions'][0]:.1f}x{feature['dimensions'][1]:.1f}x{feature['dimensions'][2]:.1f} "
                  f"at ({feature['position'][0]:.1f},{feature['position'][1]:.1f},{feature['position'][2]:.1f})")

def main():
    """Main function to generate engineering model and drawings."""
    print("Engineering Drawing Generator")
    print("="*50)
    
    # Create the generator
    generator = EngineeringDrawingGenerator(
        base_length=120,  # mm
        base_width=80,    # mm  
        base_height=60    # mm
    )
    
    # Build the model
    print("Building 3D model...")
    generator.create_base_cuboid()
    generator.generate_random_features(num_features=8)
    
    # Print summary
    generator.print_model_summary()
    
    # Save the model
    generator.save_step_file("engineering_model.step")
    
    # Display 3D model (optional - comment out if display issues)
    try:
        generator.display_3d_model()
    except:
        print("Skipping 3D display (display not available)")
    
    # Generate engineering drawings
    generator.create_engineering_drawings()
    
    print("\n" + "="*50)
    print("Engineering drawings generation completed!")
    print("Files generated:")
    print("  - engineering_model.step (3D model)")
    print("  - engineering_drawings.png (2D drawings)")

if __name__ == "__main__":
    main()
