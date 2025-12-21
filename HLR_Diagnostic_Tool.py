#!/usr/bin/env python3
"""
Hidden Line Removal Diagnostic Tool
===================================
This tool analyzes the generated engineering drawings and helps identify
issues with hidden line removal accuracy.
"""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
import numpy as np

class HLRDiagnostic:
    def __init__(self):
        self.drawing_path = "random_engineering_drawings.png"
        
    def analyze_drawing(self):
        """Analyze the generated engineering drawing for HLR issues."""
        print("HLR Diagnostic Tool")
        print("==================")
        
        try:
            # Load the image
            img = Image.open(self.drawing_path)
            img_array = np.array(img)
            
            print(f"✓ Loaded drawing: {self.drawing_path}")
            print(f"  Image size: {img.size}")
            print(f"  Image mode: {img.mode}")
            
            # Display the image with analysis overlay
            self.display_with_analysis(img_array)
            
        except FileNotFoundError:
            print(f"✗ Drawing file not found: {self.drawing_path}")
            print("  Please run Random_Engineering_Drawings.py first")
            
    def display_with_analysis(self, img_array):
        """Display the drawing with analysis overlays."""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Hidden Line Removal Diagnostic Analysis', fontsize=16)
        
        # Original drawing
        axes[0, 0].imshow(img_array)
        axes[0, 0].set_title('Original Drawing')
        axes[0, 0].axis('off')
        
        # Convert to grayscale for analysis
        if len(img_array.shape) == 3:
            gray = np.mean(img_array, axis=2)
        else:
            gray = img_array
            
        # Edge detection to identify line patterns
        from scipy import ndimage
        edges = ndimage.sobel(gray)
        
        axes[0, 1].imshow(edges, cmap='gray')
        axes[0, 1].set_title('Edge Detection (All Lines)')
        axes[0, 1].axis('off')
        
        # Analyze line thickness patterns (solid vs dashed)
        self.analyze_line_patterns(gray, axes[1, 0])
        
        # Generate improvement suggestions
        self.generate_suggestions(axes[1, 1])
        
        plt.tight_layout()
        plt.show()
        
    def analyze_line_patterns(self, gray_img, ax):
        """Analyze line patterns to distinguish solid from dashed lines."""
        # Simple analysis of line continuity
        # This is a basic implementation - could be enhanced
        
        # Threshold to identify lines
        threshold = np.mean(gray_img) - np.std(gray_img)
        binary = gray_img < threshold
        
        ax.imshow(binary, cmap='gray')
        ax.set_title('Line Pattern Analysis\n(Black = Lines)')
        ax.axis('off')
        
        # Count line segments
        line_pixels = np.sum(binary)
        total_pixels = binary.size
        line_density = line_pixels / total_pixels
        
        print(f"  Line density: {line_density:.3f}")
        if line_density > 0.1:
            print("  ⚠️  High line density - may indicate too many visible lines")
        elif line_density < 0.02:
            print("  ⚠️  Low line density - may indicate over-aggressive hidden line removal")
        else:
            print("  ✓ Line density appears reasonable")
            
    def generate_suggestions(self, ax):
        """Generate improvement suggestions based on analysis."""
        suggestions = [
            "HLR Improvement Suggestions:",
            "",
            "1. Ray Tracing Parameters:",
            "   • Increase ray samples (current: 50)",
            "   • Adjust visibility threshold (current: 60%)",
            "   • Refine intersection tolerance",
            "",
            "2. Edge Classification:",
            "   • Review boundary detection logic",
            "   • Check depth-based visibility",
            "   • Validate face normal orientations",
            "",
            "3. Geometric Accuracy:",
            "   • Use BRepClass3d_SolidClassifier",
            "   • Implement proper curve intersection",
            "   • Add surface projection validation",
            "",
            "4. Visual Validation:",
            "   • Compare with 3D model",
            "   • Check against CAD standards",
            "   • Verify orthographic projections"
        ]
        
        ax.text(0.05, 0.95, '\n'.join(suggestions), 
                transform=ax.transAxes, fontsize=10,
                verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue", alpha=0.7))
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis('off')
        ax.set_title('Diagnostic Recommendations')
        
    def run_interactive_analysis(self):
        """Run interactive analysis with user input."""
        print("\nInteractive HLR Analysis")
        print("========================")
        
        while True:
            print("\nOptions:")
            print("1. Analyze current drawing")
            print("2. Suggest ray tracing improvements")
            print("3. Show geometric validation tips")
            print("4. Exit")
            
            choice = input("\nSelect option (1-4): ").strip()
            
            if choice == '1':
                self.analyze_drawing()
            elif choice == '2':
                self.suggest_ray_improvements()
            elif choice == '3':
                self.show_validation_tips()
            elif choice == '4':
                print("Analysis complete.")
                break
            else:
                print("Invalid option. Please select 1-4.")
                
    def suggest_ray_improvements(self):
        """Suggest specific ray tracing improvements."""
        print("\nRay Tracing Improvement Suggestions:")
        print("=====================================")
        print("1. Increase ray sampling resolution:")
        print("   num_samples = 100  # Instead of 50")
        print("")
        print("2. Use adaptive sampling near surfaces:")
        print("   if distance_to_surface < threshold:")
        print("       increase_sample_density()")
        print("")
        print("3. Implement proper intersection testing:")
        print("   from OCC.Core.BRepClass3d import BRepClass3d_SolidClassifier")
        print("   classifier = BRepClass3d_SolidClassifier(solid)")
        print("   state = classifier.State()")
        print("")
        print("4. Add surface normal consideration:")
        print("   if dot_product(view_direction, surface_normal) > 0:")
        print("       surface_facing_viewer = True")
        
    def show_validation_tips(self):
        """Show tips for validating HLR accuracy."""
        print("\nHLR Validation Tips:")
        print("====================")
        print("1. Visual Comparison:")
        print("   • Compare 2D drawing with 3D model")
        print("   • Check edge continuity")
        print("   • Verify hidden edges are dashed")
        print("")
        print("2. Known Test Cases:")
        print("   • Simple cube (6 faces, 12 edges)")
        print("   • Cylinder (curved surfaces)")
        print("   • Boolean operations (intersections)")
        print("")
        print("3. Engineering Standards:")
        print("   • Hidden lines: dashed (- - -)")
        print("   • Visible lines: solid (___)")
        print("   • Construction lines: light weight")
        print("")
        print("4. Debugging Approach:")
        print("   • Test with simple geometries first")
        print("   • Add debug output for edge classification")
        print("   • Validate ray casting with known points")

def main():
    """Main function to run the diagnostic tool."""
    diagnostic = HLRDiagnostic()
    
    print("Hidden Line Removal Diagnostic Tool")
    print("===================================")
    print("This tool helps analyze and improve HLR accuracy.")
    print("")
    
    # Check if drawing exists
    try:
        with open(diagnostic.drawing_path, 'rb'):
            pass
        print(f"✓ Found drawing file: {diagnostic.drawing_path}")
    except FileNotFoundError:
        print(f"✗ Drawing file not found: {diagnostic.drawing_path}")
        print("Please run Random_Engineering_Drawings.py first to generate a drawing.")
        return
    
    # Run analysis
    diagnostic.run_interactive_analysis()

if __name__ == "__main__":
    try:
        from scipy import ndimage
        main()
    except ImportError:
        print("This tool requires scipy for image analysis.")
        print("Install with: pip install scipy")
        
        # Run basic analysis without scipy
        diagnostic = HLRDiagnostic()
        diagnostic.analyze_drawing()
