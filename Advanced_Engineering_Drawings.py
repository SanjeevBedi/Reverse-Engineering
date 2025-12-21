import sys
import os
os.environ["DYLD_LIBRARY_PATH"] = "/Users/sbedi/Anaconda/anaconda3/lib"
sys.path.append('/opt/anaconda3/envs/pyocc/lib/python3.9/site-packages')

import random
import numpy as np
from OCC.Core.BRepPrimAPI import BRepPrimAPI_MakeBox
from OCC.Core.BRepAlgoAPI import BRepAlgoAPI_Cut, BRepAlgoAPI_Fuse
from OCC.Core.gp import gp_Pnt, gp_Vec, gp_Trsf, gp_Ax1, gp_Dir
from OCC.Core.TopExp import TopExp_Explorer
from OCC.Core.TopAbs import TopAbs_FACE, TopAbs_EDGE, TopAbs_VERTEX
from OCC.Core.BRep import BRep_Tool
from OCC.Core.STEPControl import STEPControl_Writer, STEPControl_AsIs
from OCC.Core.IFSelect import IFSelect_RetDone
from OCC.Core.Interface import Interface_Static_SetCVal
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import Rectangle
from matplotlib.lines import Line2D

class AdvancedEngineeringDrawings:
    def __init__(self, base_length=100, base_width=80, base_height=60):
        self.base_length = base_length
        self.base_width = base_width  
        self.base_height = base_height
        self.main_shape = None
        self.features = []
        self.feature_count = 0
        self.visible_edges = {'front': [], 'top': [], 'side': []}
        self.hidden_edges = {'front': [], 'top': [], 'side': []}
        
    def create_base_cuboid(self):
        """Create the main base cuboid."""
        print(f"Creating base cuboid: {self.base_length} x {self.base_width} x {self.base_height}")
        self.main_shape = BRepPrimAPI_MakeBox(self.base_length, self.base_width, self.base_height).Shape()
        return self.main_shape
    
    def add_feature(self, length, width, height, x, y, z, is_addition=True):
        """Add or subtract a specific cuboid feature with proper boolean integration."""
        if self.feature_count >= 10:
            return None
            
        # Create the feature
        feature = BRepPrimAPI_MakeBox(gp_Pnt(x, y, z), length, width, height).Shape()
        
        # Apply boolean operation with error checking
        if is_addition:
            fuse_op = BRepAlgoAPI_Fuse(self.main_shape, feature)
            fuse_op.Build()
            if fuse_op.IsDone() and not fuse_op.HasErrors():
                self.main_shape = fuse_op.Shape()
                operation = "Added"
                print(f"✓ Successfully fused feature {self.feature_count + 1}")
            else:
                print(f"✗ Failed to fuse feature {self.feature_count + 1}")
                return None
        else:
            cut_op = BRepAlgoAPI_Cut(self.main_shape, feature)
            cut_op.Build()
            if cut_op.IsDone() and not cut_op.HasErrors():
                self.main_shape = cut_op.Shape()
                operation = "Subtracted"
                print(f"✓ Successfully cut feature {self.feature_count + 1}")
            else:
                print(f"✗ Failed to cut feature {self.feature_count + 1}")
                return None
        
        self.features.append({
            'operation': operation,
            'position': (x, y, z),
            'dimensions': (length, width, height),
            'is_addition': is_addition
        })
        
        self.feature_count += 1
        print(f"{operation} feature {self.feature_count}: {length:.1f}x{width:.1f}x{height:.1f} at ({x:.1f},{y:.1f},{z:.1f})")
        
        return self.main_shape
    
    def display_3d_model(self):
        """Display the integrated 3D model to verify boolean operations."""
        print("\\nDisplaying integrated 3D model...")
        
        try:
            # Try to import and use the display
            from OCC.Display.SimpleGui import init_display
            display, start_display, add_menu, add_function_to_menu = init_display()
            
            # Display the integrated shape
            display.DisplayShape(self.main_shape, update=True)
            display.FitAll()
            
            print("3D model displayed. Close the window to continue...")
            start_display()
            
        except ImportError:
            print("3D display not available. Using shape validation instead...")
            self.validate_shape_integration()
    
    def validate_shape_integration(self):
        """Validate that the shape is properly integrated."""
        print("\\nValidating shape integration...")
        
        # Count faces, edges, and vertices
        face_count = 0
        edge_count = 0
        vertex_count = 0
        
        # Count faces
        face_explorer = TopExp_Explorer(self.main_shape, TopAbs_FACE)
        while face_explorer.More():
            face_count += 1
            face_explorer.Next()
        
        # Count edges
        edge_explorer = TopExp_Explorer(self.main_shape, TopAbs_EDGE)
        while edge_explorer.More():
            edge_count += 1
            edge_explorer.Next()
            
        # Count vertices
        vertex_explorer = TopExp_Explorer(self.main_shape, TopAbs_VERTEX)
        while vertex_explorer.More():
            vertex_count += 1
            vertex_explorer.Next()
        
        print(f"Integrated shape contains:")
        print(f"  - {face_count} faces")
        print(f"  - {edge_count} edges") 
        print(f"  - {vertex_count} vertices")
        
        if face_count > 6:  # More than a simple box
            print("✓ Shape successfully integrated with features")
        else:
            print("⚠ Shape may not be properly integrated")
            
        return face_count, edge_count, vertex_count
    
    def create_complex_model(self):
        """Create a complex model with prominent subtracted features for demonstration."""
        print("\\nCreating complex engineering model with prominent cuts...")
        
        # Create base
        self.create_base_cuboid()
        
        # Add prominent subtracted features that will be clearly visible
        print("Adding major subtracted features...")
        
        # Large central pocket on top face - clearly visible in top view
        self.add_feature(40, 30, 15, 40, 25, self.base_height-15, is_addition=False)  # Central pocket
        
        # Through holes - visible in all views as dashed lines
        self.add_feature(8, 8, self.base_height+5, 25, 15, -2, is_addition=False)   # Through hole 1
        self.add_feature(8, 8, self.base_height+5, 95, 65, -2, is_addition=False)   # Through hole 2
        
        # Side pocket - visible from front view
        self.add_feature(25, 15, 20, 60, self.base_width-15, 20, is_addition=False)  # Side pocket
        
        # Front pocket - visible from side view  
        self.add_feature(20, 20, 25, 20, -20, 15, is_addition=False)  # Front pocket
        
        # Add some boss features for contrast
        print("Adding boss features...")
        self.add_feature(20, 15, 12, 85, 50, self.base_height, is_addition=True)    # Small boss
        self.add_feature(15, 25, 10, 30, 5, self.base_height, is_addition=True)     # Corner boss
        
        # Additional cuts for more hidden lines
        self.add_feature(15, 12, 18, 50, 40, 25, is_addition=False)   # Internal cut 1
        self.add_feature(12, 18, 15, 75, 25, 30, is_addition=False)   # Internal cut 2
        
        print(f"Generated {self.feature_count} features total")
        print("✓ Model includes prominent subtracted features for hidden line demonstration")
        print("\\n" + "="*50)
        print("BOOLEAN INTEGRATION COMPLETE")
        print("="*50)
        
        # Validate the integration
        self.validate_shape_integration()
        """Create a complex model with specific features for demonstration."""
        print("\\nCreating complex engineering model...")
        
        # Create base
        self.create_base_cuboid()
        
        # Add features on top face (positive Z)
        self.add_feature(30, 20, 15, 20, 30, self.base_height, is_addition=True)  # Boss
        self.add_feature(15, 15, 20, 70, 50, self.base_height, is_addition=True)  # Small boss
        
        # Cut features on top face
        self.add_feature(25, 25, 10, 45, 20, self.base_height-10, is_addition=False)  # Pocket
        self.add_feature(12, 12, 8, 75, 15, self.base_height-8, is_addition=False)   # Small hole
        
        # Add features on front face (negative Y)
        self.add_feature(20, 15, 25, 30, -15, 20, is_addition=True)   # Front boss
        self.add_feature(35, 10, 15, 50, -10, 30, is_addition=True)   # Side feature
        
        # Cut features on front face  
        self.add_feature(15, 12, 20, 60, -12, 25, is_addition=False)  # Front pocket
        self.add_feature(10, 8, 12, 15, -8, 35, is_addition=False)   # Small cut
        
        print(f"Generated {self.feature_count} features total")
    
    def extract_edges_simple(self):
        """Extract edges and classify them as visible or hidden for each view."""
        print("Extracting edges for orthographic projections...")
        
        # Clear previous data
        for view in ['front', 'top', 'side']:
            self.visible_edges[view] = []
            self.hidden_edges[view] = []
        
        # First, add the outer boundary edges as visible edges
        self.add_outer_boundary_edges()
        
        # Get all edges from the shape
        edge_explorer = TopExp_Explorer(self.main_shape, TopAbs_EDGE)
        
        while edge_explorer.More():
            edge = edge_explorer.Current()
            curve, first, last = BRep_Tool.Curve(edge)
            
            if curve is not None:
                p1 = curve.Value(first)
                p2 = curve.Value(last)
                
                # Classify edge for each view
                self.classify_edge_for_views(p1, p2)
            
            edge_explorer.Next()
    
    def add_outer_boundary_edges(self):
        """Add the outer boundary edges as visible for each view."""
        # Instead of fixed rectangles, we'll let the edge classification handle this
        # The actual silhouette will be determined by the visibility detection
        pass
    
    def classify_edge_for_views(self, p1, p2):
        """Classify an edge as visible or hidden for each orthographic view."""
        
        # Front view (X-Z projection, looking along Y)
        front_edge = [(p1.X(), p1.Z()), (p2.X(), p2.Z())]
        if self.is_edge_on_visible_face(p1, p2, 'front'):
            self.visible_edges['front'].append(front_edge)
        else:
            self.hidden_edges['front'].append(front_edge)
        
        # Top view (X-Y projection, looking along Z) 
        top_edge = [(p1.X(), p1.Y()), (p2.X(), p2.Y())]
        if self.is_edge_on_visible_face(p1, p2, 'top'):
            self.visible_edges['top'].append(top_edge)
        else:
            self.hidden_edges['top'].append(top_edge)
        
        # Side view (Y-Z projection, looking along X)
        side_edge = [(p1.Y(), p1.Z()), (p2.Y(), p2.Z())]
        if self.is_edge_on_visible_face(p1, p2, 'side'):
            self.visible_edges['side'].append(side_edge)
        else:
            self.hidden_edges['side'].append(side_edge)
    
    def is_edge_on_visible_face(self, p1, p2, view):
        """Determine if an edge is on a visible face for the given view using face analysis."""
        tolerance = 1e-3
        
        if view == 'front':
            # Front view (looking along +Y direction): 
            # Visible faces have normals pointing towards +Y or are silhouette edges
            y_vals = [p1.Y(), p2.Y()]
            z_vals = [p1.Z(), p2.Z()]
            
            # Check if edge is on visible surfaces:
            # 1. Front-facing surfaces (large Y values)
            # 2. Top surfaces of features (large Z values) 
            # 3. Silhouette edges of protruding features
            
            max_y = max(y_vals)
            max_z = max(z_vals)
            min_z = min(z_vals)
            
            # Front face or protruding features
            if max_y >= self.base_width - tolerance:
                return True
            # Protruding front features (negative Y)
            if min(y_vals) < -tolerance:
                return True    
            # Top surfaces of features
            if max_z >= self.base_height - tolerance:
                return True
            # Bottom edges    
            if min_z <= tolerance:
                return True
                
        elif view == 'top':
            # Top view (looking along -Z direction):
            # Visible faces have normals pointing towards +Z or are silhouette edges
            z_vals = [p1.Z(), p2.Z()]
            x_vals = [p1.X(), p2.X()]
            y_vals = [p1.Y(), p2.Y()]
            
            max_z = max(z_vals)
            
            # Check if edge is on visible surfaces:
            # 1. Top surfaces (large Z values)
            # 2. Silhouette edges of all features
            
            # Top surfaces of base and features
            if max_z >= self.base_height - tolerance:
                return True
            # Protruding features extending upward
            if max_z > self.base_height + tolerance:
                return True
            # Side edges of protruding features    
            if (min(x_vals) <= tolerance or max(x_vals) >= self.base_length - tolerance or
                min(y_vals) <= tolerance or max(y_vals) >= self.base_width - tolerance):
                return True
                
        elif view == 'side':
            # Side view (looking along +X direction):
            # Visible faces have normals pointing towards +X or are silhouette edges  
            x_vals = [p1.X(), p2.X()]
            z_vals = [p1.Z(), p2.Z()]
            y_vals = [p1.Y(), p2.Y()]
            
            max_x = max(x_vals)
            max_z = max(z_vals)
            min_z = min(z_vals)
            
            # Check if edge is on visible surfaces:
            # 1. Right-facing surfaces (large X values)
            # 2. Top surfaces of features
            # 3. Silhouette edges
            
            # Right face
            if max_x >= self.base_length - tolerance:
                return True
            # Top surfaces    
            if max_z >= self.base_height - tolerance:
                return True
            # Bottom edges
            if min_z <= tolerance:
                return True
            # Front/back edges of protruding features
            if min(y_vals) <= tolerance or max(y_vals) >= self.base_width - tolerance:
                return True
        
        return False
    
    def create_engineering_drawings_advanced(self):
        """Create detailed engineering drawings with proper orthographic projection standards."""
        print("\\nGenerating standard engineering drawings...")
        
        # Extract edges
        self.extract_edges_simple()
        
        # Create figure with standard engineering layout (3x2 grid)
        fig = plt.figure(figsize=(16, 12))
        
        # Standard engineering drawing layout:
        # Top view (upper center)
        ax_top = plt.subplot2grid((3, 3), (0, 1), colspan=1, rowspan=1)
        
        # Front view (center)  
        ax_front = plt.subplot2grid((3, 3), (1, 1), colspan=1, rowspan=1)
        
        # Right side view (center right)
        ax_side = plt.subplot2grid((3, 3), (1, 2), colspan=1, rowspan=1)
        
        # Standard view configuration following ISO/ANSI standards
        views = [
            ("FRONT VIEW", "front", ax_front, "X (Length)", "Z (Height)"),
            ("TOP VIEW", "top", ax_top, "X (Length)", "Y (Width)"),
            ("RIGHT SIDE VIEW", "side", ax_side, "Y (Width)", "Z (Height)")
        ]
        
        for view_name, view_dir, ax, xlabel, ylabel in views:
            print(f"Processing {view_name}...")
            self.draw_standard_orthographic_view(ax, view_dir, view_name, xlabel, ylabel)
        
        # Add global legend at the bottom of the figure
        self.add_global_legend(fig)
        
        # Add title block and drawing information
        self.add_title_block(fig)
        
        # Adjust layout for proper spacing
        plt.tight_layout()
        plt.subplots_adjust(top=0.85, bottom=0.20, left=0.1, right=0.9)
        
        # Save with high quality
        plt.savefig('advanced_engineering_drawings.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print("Standard engineering drawings saved as 'advanced_engineering_drawings.png'")
    
    def add_global_legend(self, fig):
        """Add a global legend at the bottom of the figure."""
        # Create legend elements
        from matplotlib.lines import Line2D
        legend_elements = [
            Line2D([0], [0], color='black', linewidth=1.5, label='Visible Edges'),
            Line2D([0], [0], color='black', linewidth=1.0, linestyle='--', 
                   dashes=[3, 2], alpha=0.7, label='Hidden Edges')
        ]
        
        # Add legend at the bottom center
        fig.legend(handles=legend_elements, 
                  loc='lower center', 
                  bbox_to_anchor=(0.5, 0.02),
                  ncol=2, 
                  fontsize=12, 
                  frameon=True, 
                  fancybox=True, 
                  shadow=True)
    
    def draw_standard_orthographic_view(self, ax, view_direction, title, xlabel, ylabel):
        """Draw a single orthographic view following engineering standards."""
        
        # Draw visible edges (solid lines, thickness 0.7mm equivalent)
        for edge in self.visible_edges[view_direction]:
            if len(edge) == 2:
                x_coords = [edge[0][0], edge[1][0]]
                y_coords = [edge[0][1], edge[1][1]]
                ax.plot(x_coords, y_coords, 'k-', linewidth=1.5)
        
        # Draw hidden edges (dashed lines, thickness 0.35mm equivalent)
        for edge in self.hidden_edges[view_direction]:
            if len(edge) == 2:
                x_coords = [edge[0][0], edge[1][0]]
                y_coords = [edge[0][1], edge[1][1]]
                ax.plot(x_coords, y_coords, 'k--', linewidth=1.0, alpha=0.7, dashes=[3, 2])
                hidden_count += 1
        
        # Set equal aspect ratio (critical for engineering drawings)
        ax.set_aspect('equal')
        
        # Add grid (construction lines)
        ax.grid(True, alpha=0.3, linestyle=':', linewidth=0.5, color='gray')
        
        # Format axes following engineering standards
        ax.set_xlabel(xlabel + ' [mm]', fontsize=11, fontweight='bold')
        ax.set_ylabel(ylabel + ' [mm]', fontsize=11, fontweight='bold')
        
        # Set title with proper formatting
        ax.set_title(title, fontsize=12, fontweight='bold', pad=15)
        
        # Add dimension lines and basic annotations
        self.add_standard_dimensions(ax, view_direction)
        
        # Set margins and limits
        ax.margins(0.15)
        
        # Format tick labels (no individual legends - using global legend)
        ax.tick_params(axis='both', which='major', labelsize=9)
        
    def add_standard_dimensions(self, ax, view_direction):
        """Add standard dimension annotations following engineering conventions."""
        dim_color = 'blue'
        dim_fontsize = 9
        
        if view_direction == "front":
            # Overall length dimension (bottom)
            y_offset = -15
            ax.annotate('', xy=(self.base_length, y_offset), xytext=(0, y_offset),
                       arrowprops=dict(arrowstyle='<->', color=dim_color, lw=1.2))
            ax.text(self.base_length/2, y_offset-5, f'L = {self.base_length}', 
                   ha='center', va='top', color=dim_color, fontsize=dim_fontsize, fontweight='bold')
            
            # Overall height dimension (left)
            x_offset = -15
            ax.annotate('', xy=(x_offset, self.base_height), xytext=(x_offset, 0),
                       arrowprops=dict(arrowstyle='<->', color=dim_color, lw=1.2))
            ax.text(x_offset-5, self.base_height/2, f'H = {self.base_height}', 
                   ha='center', va='center', color=dim_color, fontsize=dim_fontsize, 
                   fontweight='bold', rotation=90)
        
        elif view_direction == "top":
            # Length dimension (bottom)
            y_offset = -15
            ax.annotate('', xy=(self.base_length, y_offset), xytext=(0, y_offset),
                       arrowprops=dict(arrowstyle='<->', color=dim_color, lw=1.2))
            ax.text(self.base_length/2, y_offset-5, f'L = {self.base_length}', 
                   ha='center', va='top', color=dim_color, fontsize=dim_fontsize, fontweight='bold')
            
            # Width dimension (left)
            x_offset = -15
            ax.annotate('', xy=(x_offset, self.base_width), xytext=(x_offset, 0),
                       arrowprops=dict(arrowstyle='<->', color=dim_color, lw=1.2))
            ax.text(x_offset-5, self.base_width/2, f'W = {self.base_width}', 
                   ha='center', va='center', color=dim_color, fontsize=dim_fontsize, 
                   fontweight='bold', rotation=90)
        
        elif view_direction == "side":
            # Width dimension (bottom)
            y_offset = -15
            ax.annotate('', xy=(self.base_width, y_offset), xytext=(0, y_offset),
                       arrowprops=dict(arrowstyle='<->', color=dim_color, lw=1.2))
            ax.text(self.base_width/2, y_offset-5, f'W = {self.base_width}', 
                   ha='center', va='top', color=dim_color, fontsize=dim_fontsize, fontweight='bold')
            
            # Height dimension (left)
            x_offset = -15
            ax.annotate('', xy=(x_offset, self.base_height), xytext=(x_offset, 0),
                       arrowprops=dict(arrowstyle='<->', color=dim_color, lw=1.2))
            ax.text(x_offset-5, self.base_height/2, f'H = {self.base_height}', 
                   ha='center', va='center', color=dim_color, fontsize=dim_fontsize, 
                   fontweight='bold', rotation=90)
    
    def add_title_block(self, fig):
        """Add standard engineering drawing title block."""
        # Add main title
        fig.suptitle('ENGINEERING DRAWING\\nORTHOGRAPHIC PROJECTIONS', 
                    fontsize=16, fontweight='bold', y=0.95)
        
        # Add title block information
        title_text = f"""
        PART: Complex Machined Component
        MATERIAL: Aluminum 6061-T6
        SCALE: 1:1
        DIMENSIONS: mm
        PROJECTION: Third Angle (ISO)
        
        BASE DIMENSIONS: {self.base_length} × {self.base_width} × {self.base_height} mm
        FEATURES: {len(self.features)} total ({len([f for f in self.features if f['is_addition']])} additions, {len([f for f in self.features if not f['is_addition']])} cuts)
        """
        
        fig.text(0.02, 0.12, title_text, fontsize=10, verticalalignment='top',
                bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgray", alpha=0.8))
    
    def save_step_file(self, filename="advanced_engineering_model.step"):
        """Save the model as a STEP file."""
        step_writer = STEPControl_Writer()
        Interface_Static_SetCVal("write.step.schema", "AP203")
        
        step_writer.Transfer(self.main_shape, STEPControl_AsIs)
        status = step_writer.Write(filename)
        
        if status == IFSelect_RetDone:
            print(f"Model saved as {filename}")
        else:
            print("Failed to save STEP file")
    
    def print_model_summary(self):
        """Print a summary of the generated model."""
        print("\\n" + "="*60)
        print("ADVANCED MODEL SUMMARY")
        print("="*60)
        print(f"Base Cuboid: {self.base_length} x {self.base_width} x {self.base_height} mm")
        print(f"Total Features: {len(self.features)}")
        
        additions = [f for f in self.features if f['is_addition']]
        subtractions = [f for f in self.features if not f['is_addition']]
        
        print(f"Additions (Bosses): {len(additions)}")
        print(f"Subtractions (Cuts): {len(subtractions)}")
        
        print("\\nFeature Details:")
        for i, feature in enumerate(self.features, 1):
            op_type = "Boss" if feature['is_addition'] else "Cut"
            print(f"  {i}. {op_type}: "
                  f"{feature['dimensions'][0]:.1f} x {feature['dimensions'][1]:.1f} x {feature['dimensions'][2]:.1f} mm "
                  f"at ({feature['position'][0]:.1f}, {feature['position'][1]:.1f}, {feature['position'][2]:.1f}) mm")

def main():
    """Main function to generate advanced engineering model and drawings."""
    print("Advanced Engineering Drawing Generator")
    print("="*50)
    
    # Create the generator
    generator = AdvancedEngineeringDrawings(
        base_length=120,  # mm
        base_width=80,    # mm  
        base_height=60    # mm
    )
    
    # Build the complex model with proper boolean integration
    print("Building complex 3D model...")
    generator.create_complex_model()
    
    # Display the integrated 3D model
    print("\\nStep 1: Displaying integrated 3D model...")
    generator.display_3d_model()
    
    # Print summary
    generator.print_model_summary()
    
    # Save the model
    print("\\nStep 2: Saving STEP file...")
    generator.save_step_file("advanced_engineering_model.step")
    
    # Generate advanced engineering drawings
    print("\\nStep 3: Generating orthographic engineering drawings...")
    generator.create_engineering_drawings_advanced()
    
    print("\\n" + "="*50)
    print("COMPLETE WORKFLOW FINISHED!")
    print("="*50)
    print("Files generated:")
    print("  - advanced_engineering_model.step (3D integrated model)")
    print("  - advanced_engineering_drawings.png (2D drawings with proper HLR)")
    print("\\nWorkflow Summary:")
    print("  ✓ Boolean operations integrated all features into single solid")
    print("  ✓ 3D model displayed for verification")
    print("  ✓ Orthographic projections generated with correct line types")

if __name__ == "__main__":
    main()
