"""
Visualization module for OpenCASCADE Engineering Drawings Generator.

This module handles 2D and 3D visualization of engineering drawings.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from typing import List, Dict, Any, Optional
from mpl_toolkits.mplot3d import Axes3D

try:
    from OCC.Core.TopExp import TopExp_Explorer
    from OCC.Core.TopAbs import TopAbs_FACE, TopAbs_WIRE
    from OCC.Core.TopoDS import topods
    
    OPENCASCADE_AVAILABLE = True
except ImportError:
    OPENCASCADE_AVAILABLE = False


class Visualizer:
    """
    Handles visualization of 2D engineering drawings and 3D solids.
    
    This class provides methods to:
    - Plot 2D polygon arrays (visible and hidden lines)
    - Visualize 3D solids as wireframes
    - Generate engineering drawing layouts
    - Create statistical summaries
    """
    
    def __init__(self):
        """Initialize the visualizer."""
        pass
    
    def plot_arrays_visualization(self, array_A: List[dict], 
                                  array_B: List[dict], 
                                  array_C: List[dict], 
                                  unit_projection_normal: List[float]):
        """
        Plot arrays B, C, and B+C with enhanced visualization.
        
        Args:
            array_A: Processed polygons (usually empty)
            array_B: Visible polygons
            array_C: Hidden polygons and intersections
            unit_projection_normal: Projection direction for labeling
        """
        print("\n" + "="*60)
        print("PLOTTING ARRAY VISUALIZATION")
        print("="*60)
        
        if not array_B and not array_C:
            print("No polygons to visualize")
            return
        
        # Create figure with subplots
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle(f'Enhanced Polygon Classification Results\n'
                     f'(Projection Normal: {unit_projection_normal})', 
                     fontsize=14, weight='bold')
        
        colors_b = ['lightblue', 'lightcoral', 'lightgreen', 
                    'lightyellow', 'lightpink', 'lavender']
        colors_c = ['orange', 'red', 'purple', 'brown', 'gray', 'cyan']
        
        # Collect bounds for consistent scaling
        all_bounds = []
        
        # Subplot 1: Array B (Visible faces)
        self._plot_array_subplot(ax1, array_B, colors_b, 
                                  "Array B - Visible Faces", all_bounds)
        
        # Subplot 2: Array C (Hidden faces + intersections)
        self._plot_array_subplot(ax2, array_C, colors_c, 
                                  "Array C - Hidden + Intersections", 
                                  all_bounds, highlight_intersections=True)
        
        # Subplot 3: Combined B + C
        self._plot_combined_subplot(ax3, array_B, array_C, all_bounds)
        
        # Subplot 4: Statistics and algorithm info
        self._plot_statistics_subplot(ax4, array_B, array_C, 
                                      unit_projection_normal)
        
        # Set consistent bounds for all plots
        self._set_consistent_bounds([ax1, ax2, ax3], all_bounds)
        
        plt.tight_layout()
        plt.show()
        
        print(f"✓ Array visualization complete")
        print(f"  → Array B: {len(array_B)} visible faces")
        print(f"  → Array C: {len(array_C)} hidden faces + intersections")
        print(f"  → Combined: {len(array_B) + len(array_C)} total polygons")
    
    def _plot_array_subplot(self, ax, array_data: List[dict], colors: List[str], 
                            title: str, all_bounds: List[float], 
                            highlight_intersections: bool = False):
        """Plot a single array subplot."""
        ax.set_title(f'{title} ({len(array_data)} polygons)', 
                     fontsize=12, weight='bold')
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)
        ax.set_xlabel('X coordinate')
        ax.set_ylabel('Y coordinate')
        
        for i, poly_data in enumerate(array_data):
            try:
                polygon = poly_data['polygon']
                name = poly_data['name']
                
                if polygon.geom_type == 'Polygon' and polygon.area > 0:
                    # Different styling for intersections
                    if highlight_intersections and 'Intersection' in name:
                        color = 'yellow'
                        edge_color = 'red'
                        alpha = 0.8
                        linewidth = 2
                    else:
                        color = colors[i % len(colors)]
                        edge_color = 'black'
                        alpha = 0.7
                        linewidth = 1.5
                    
                    self._plot_polygon(polygon, ax, facecolor=color, 
                                       edgecolor=edge_color, alpha=alpha, 
                                       linewidth=linewidth, 
                                       label=f'{name} (area: {polygon.area:.1f})')
                    
                    # Collect bounds
                    bounds = polygon.bounds
                    all_bounds.extend([bounds[0], bounds[2], bounds[1], 
                                       bounds[3]])
                    
                    # Add face name at centroid
                    centroid = polygon.centroid
                    display_name = name.replace('Face_', 'F').replace(
                        'Intersection_', 'I_')
                    ax.text(centroid.x, centroid.y, display_name, 
                            ha='center', va='center', fontsize=8, 
                            weight='bold')
                            
            except Exception as e:
                print(f"Error plotting {poly_data.get('name', 'unknown')}: {e}")
        
        if array_data:
            ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
    
    def _plot_combined_subplot(self, ax, array_B: List[dict], 
                               array_C: List[dict], all_bounds: List[float]):
        """Plot the combined B+C subplot with proper line styles."""
        ax.set_title(f'Combined Arrays B + C '
                     f'({len(array_B) + len(array_C)} polygons)', 
                     fontsize=12, weight='bold')
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)
        ax.set_xlabel('X coordinate')
        ax.set_ylabel('Y coordinate')
        
        # Plot array_C polygons first as thin dashed black lines
        for poly_data in array_C:
            try:
                polygon = poly_data['polygon']
                name = poly_data['name']
                if polygon.geom_type == 'Polygon' and polygon.area > 0:
                    self._plot_polygon(polygon, ax, facecolor='none', 
                                       edgecolor='black', alpha=1.0, 
                                       linewidth=0.7, linestyle='--', 
                                       label=f'C: {name}', outline_only=True)
            except Exception as e:
                print(f"Error plotting array_C polygon: {e}")
        
        # Plot array_B polygons as solid black lines
        for poly_data in array_B:
            try:
                polygon = poly_data['polygon']
                name = poly_data['name']
                if polygon.geom_type == 'Polygon' and polygon.area > 0:
                    self._plot_polygon(polygon, ax, facecolor='none', 
                                       edgecolor='black', alpha=1.0, 
                                       linewidth=1.2, linestyle='-', 
                                       label=f'B: {name}', outline_only=True)
            except Exception as e:
                print(f"Error plotting array_B polygon: {e}")
        
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=7)
    
    def _plot_statistics_subplot(self, ax, array_B: List[dict], 
                                 array_C: List[dict], 
                                 unit_projection_normal: List[float]):
        """Plot the statistics and algorithm information."""
        ax.axis('off')
        
        # Calculate areas
        area_B = sum(p['polygon'].area for p in array_B 
                     if hasattr(p['polygon'], 'area'))
        area_C = sum(p['polygon'].area for p in array_C 
                     if hasattr(p['polygon'], 'area'))
        
        stats_text = f"""ENHANCED POLYGON CLASSIFICATION RESULTS

Algorithm: Historic Depth-Based Classification
Projection Normal: [{unit_projection_normal[0]:.3f}, {unit_projection_normal[1]:.3f}, {unit_projection_normal[2]:.3f}]

ARRAY B (VISIBLE FACES):
• Polygons: {len(array_B)}
• Total Area: {area_B:.2f}
• Type: Depth-processed visible faces

ARRAY C (HIDDEN + INTERSECTIONS):
• Polygons: {len(array_C)}
• Total Area: {area_C:.2f}
• Type: Hidden faces + intersection regions

ALGORITHM FEATURES:
✓ Historic polygon classification extracted
✓ Depth-based boolean operations
✓ 3D line-face intersection analysis
✓ Multi-point sampling for accuracy
✓ Face association tracking
✓ Enhanced visualization

Total Processed: {len(array_B) + len(array_C)} polygons"""
        
        ax.text(0.05, 0.95, stats_text, transform=ax.transAxes, 
                fontsize=9, verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
    
    def _plot_polygon(self, polygon, ax, facecolor='none', edgecolor='black', 
                      alpha=0.7, linestyle='-', linewidth=2, label=None, 
                      outline_only=False):
        """Helper function to plot a polygon."""
        if polygon.geom_type == 'Polygon':
            if outline_only:
                # Only draw the outline
                x, y = polygon.exterior.xy
                ax.plot(x, y, color=edgecolor, linestyle=linestyle, 
                        linewidth=linewidth, label=label)
            else:
                # Draw filled patch
                if facecolor != 'none':
                    patch = patches.Polygon(list(polygon.exterior.coords), 
                                          closed=True, facecolor=facecolor, 
                                          alpha=alpha, edgecolor=edgecolor, 
                                          linewidth=linewidth, 
                                          linestyle=linestyle)
                    ax.add_patch(patch)
                    # Add invisible line for legend if label is provided
                    if label:
                        ax.plot([], [], color=edgecolor, linestyle=linestyle, 
                                linewidth=linewidth, label=label)
        elif polygon.geom_type == 'MultiPolygon':
            for poly in polygon.geoms:
                self._plot_polygon(poly, ax, facecolor, edgecolor, alpha, 
                                   linestyle, linewidth, label=None, 
                                   outline_only=outline_only)
    
    def _set_consistent_bounds(self, axes, all_bounds: List[float]):
        """Set consistent bounds for multiple axes."""
        if all_bounds:
            margin = (max(all_bounds) - min(all_bounds)) * 0.1
            xlim = (min(all_bounds) - margin, max(all_bounds) + margin)
            ylim = (min(all_bounds) - margin, max(all_bounds) + margin)
            
            for ax in axes:
                ax.set_xlim(xlim)
                ax.set_ylim(ylim)
    
    def visualize_3d_solid(self, solid_shape):
        """
        Display the 3D solid using matplotlib 3D plotting.
        
        Args:
            solid_shape: OpenCASCADE solid to visualize
        """
        if not OPENCASCADE_AVAILABLE or solid_shape is None:
            print("✗ Cannot visualize - OpenCASCADE not available or "
                  "shape is None")
            return
        
        print("\n" + "="*60)
        print("3D SOLID VISUALIZATION WITH MATPLOTLIB")
        print("="*60)
        
        try:
            # Extract face vertices for visualization
            print("  → Extracting face vertices for 3D plot...")
            all_face_data = self._extract_3d_face_data(solid_shape)
            
            if not all_face_data:
                print("  ✗ No face data available for visualization")
                return
            
            # Create 3D plot
            fig = plt.figure(figsize=(15, 12))
            ax = fig.add_subplot(111, projection='3d')
            
            self._plot_3d_faces(ax, all_face_data)
            self._setup_3d_plot(ax, all_face_data)
            
            plt.tight_layout()
            plt.show()
            
            print(f"✓ 3D solid visualization complete")
            print(f"  → Displayed {len(all_face_data)} faces as polygons")
            print(f"  → Total vertices plotted: "
                  f"{sum(len(face_data['vertices']) for face_data in all_face_data)}")
            
        except Exception as e:
            print(f"✗ 3D visualization failed: {e}")
            import traceback
            traceback.print_exc()
    
    def _extract_3d_face_data(self, solid_shape) -> List[Dict[str, Any]]:
        """Extract face vertex data for 3D visualization."""
        face_explorer = TopExp_Explorer(solid_shape, TopAbs_FACE)
        face_count = 0
        all_face_data = []
        
        while face_explorer.More():
            face_shape = face_explorer.Current()
            face_count += 1
            
            try:
                face = topods.Face(face_shape)
                
                # Extract vertices from this face
                wire_explorer = TopExp_Explorer(face, TopAbs_WIRE)
                if wire_explorer.More():
                    wire = wire_explorer.Current()
                    
                    # Use a simplified vertex extraction for visualization
                    vertices = self._extract_wire_vertices_simple(wire)
                    
                    if vertices and len(vertices) >= 3:
                        all_face_data.append({
                            'face_id': face_count,
                            'vertices': vertices,
                            'vertex_count': len(vertices)
                        })
                        print(f"    Face {face_count}: {len(vertices)} "
                              f"vertices extracted")
                    else:
                        print(f"    Face {face_count}: Failed to extract "
                              f"enough vertices")
                
            except Exception as e:
                print(f"    Face {face_count}: Error - {e}")
            
            face_explorer.Next()
        
        print(f"  → Successfully extracted {len(all_face_data)} faces "
              f"for visualization")
        return all_face_data
    
    def _extract_wire_vertices_simple(self, wire) -> List[List[float]]:
        """Simple vertex extraction for visualization purposes."""
        try:
            from OCC.Core.TopExp import TopExp_Explorer
            from OCC.Core.TopAbs import TopAbs_VERTEX
            from OCC.Core.BRep import BRep_Tool
            from OCC.Core.TopoDS import topods
            
            vertex_explorer = TopExp_Explorer(wire, TopAbs_VERTEX)
            vertices = []
            seen = set()
            
            while vertex_explorer.More():
                vertex = topods.Vertex(vertex_explorer.Current())
                pnt = BRep_Tool.Pnt(vertex)
                v = [pnt.X(), pnt.Y(), pnt.Z()]
                
                # Remove duplicates
                v_tuple = tuple(np.round(v, 6))
                if v_tuple not in seen:
                    vertices.append(v)
                    seen.add(v_tuple)
                
                vertex_explorer.Next()
            
            return vertices
            
        except Exception as e:
            print(f"        Error in simple vertex extraction: {e}")
            return []
    
    def _plot_3d_faces(self, ax, all_face_data: List[Dict[str, Any]]):
        """Plot 3D faces as polygon boundaries."""
        colors = plt.cm.Set3(np.linspace(0, 1, len(all_face_data)))
        
        for i, face_data in enumerate(all_face_data):
            vertices = np.array(face_data['vertices'])
            face_id = face_data['face_id']
            vertex_count = face_data['vertex_count']
            
            # Close the polygon by adding first vertex at end
            if len(vertices) > 2:
                vertices_closed = np.vstack([vertices, vertices[0]])
                
                # Plot face boundary edges
                ax.plot(vertices_closed[:, 0], vertices_closed[:, 1], 
                        vertices_closed[:, 2], color=colors[i], linewidth=3, 
                        alpha=0.9, label=f'Face {face_id} ({vertex_count}v)')
                
                # Plot vertices as points
                ax.scatter(vertices[:, 0], vertices[:, 1], vertices[:, 2], 
                          color=colors[i], s=50, alpha=0.8, 
                          edgecolors='black', linewidth=1)
                
                # Add face center label
                face_center = np.mean(vertices, axis=0)
                ax.text(face_center[0], face_center[1], face_center[2], 
                       f'F{face_id}({vertex_count}v)', fontsize=10, 
                       color='red', weight='bold',
                       bbox=dict(boxstyle='round,pad=0.3', facecolor='white', 
                                 alpha=0.8))
    
    def _setup_3d_plot(self, ax, all_face_data: List[Dict[str, Any]]):
        """Setup 3D plot appearance and properties."""
        # Set labels and title
        ax.set_xlabel('X Coordinate', fontsize=12, weight='bold')
        ax.set_ylabel('Y Coordinate', fontsize=12, weight='bold')
        ax.set_zlabel('Z Coordinate', fontsize=12, weight='bold')
        ax.set_title(f'3D Solid Visualization - Polygon Boundaries\n'
                     f'{len(all_face_data)} Faces from Boolean Operation\n'
                     f'Pure Polygon Display', fontsize=14, weight='bold')
        
        # Set equal aspect ratio
        all_vertices = np.vstack([face_data['vertices'] 
                                  for face_data in all_face_data])
        max_range = np.ptp(all_vertices, axis=0).max() / 2.0
        mid_x = np.mean(all_vertices[:, 0])
        mid_y = np.mean(all_vertices[:, 1])
        mid_z = np.mean(all_vertices[:, 2])
        
        margin = max_range * 0.1
        ax.set_xlim(mid_x - max_range - margin, mid_x + max_range + margin)
        ax.set_ylim(mid_y - max_range - margin, mid_y + max_range + margin)
        ax.set_zlim(mid_z - max_range - margin, mid_z + max_range + margin)
        
        # Add legend (limited to avoid clutter)
        handles, labels = ax.get_legend_handles_labels()
        if len(handles) > 10:
            ax.legend(handles[:10], labels[:10], loc='upper left', 
                      bbox_to_anchor=(0.02, 0.98), fontsize=9)
        else:
            ax.legend(loc='upper left', bbox_to_anchor=(0.02, 0.98), 
                      fontsize=9)
        
        # Add grid and set viewing angle
        ax.grid(True, alpha=0.3)
        ax.view_init(elev=25, azim=45)
        
        # Add information text
        info_text = f"""POLYGON BOUNDARY DISPLAY
• Pure polygon wireframe
• No surface triangulation
• {len(all_face_data)} faces total
• All edges clearly visible"""
        
        ax.text2D(0.02, 0.02, info_text, transform=ax.transAxes, 
                 fontsize=10, verticalalignment='bottom', 
                 bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8),
                 fontfamily='monospace')
