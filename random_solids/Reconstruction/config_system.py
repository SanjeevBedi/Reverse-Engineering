#!/usr/bin/env python3
"""
Configuration System for Random Solid Generation

This module manages all random parameters and hard-coded values used across
V6_current.py, Base_Solid.py, and Lettering_solid.py files.

Configuration files are saved as JSON with filename format: config_seed_{seed}.json
"""

import json
import os
import random
import numpy as np
from typing import Dict, Any, Optional


class ConfigurationManager:
    """Manages configuration parameters for solid generation"""
    
    def __init__(self, seed: int = 1):
        self.seed = seed
        self.config = {}
        self._initialize_default_config()
    
    def _initialize_default_config(self):
        """Initialize all configuration parameters with default values"""
        # Set random seed for consistent parameter generation
        random.seed(self.seed)
        np.random.seed(self.seed)
        
        self.config = {
            # ===== METADATA =====
            "seed": self.seed,
            "version": "1.0",
            "generated_by": "config_system.py",
            
            # ===== V6_CURRENT.PY PARAMETERS =====
            "v6_current": {
                # Cuboid generation parameters
                "max_cuboids": 10,
                "max_rotated": 3,
                
                # Heights and probabilities
                "heights": [10, 25, 50, 75, 100, 150, 250],
                "height_base_prob": 64 / 127,  # p = 64/127
                
                # Base dimensions
                "base_width": 200,
                "base_depth": 300,
                
                # Polygon generation parameters
                "polygon_regular_probability": 0.7,  # 70% chance for regular polygons
                "polygon_regular_sides": [4, 5, 6, 8, 3, 7, 9, 10],
                "polygon_sides_base_prob": 12800 / 255,  # p = 12800/255
                
                # Polygon sizing parameters
                "polygon_min_area_factor": 0.002,  # min_area = 0.002 * boundary.area
                "polygon_min_radius_factor": 0.15,  # min_radius = 0.15 * max(width, height)
                "polygon_radius_factor": 0.4,  # radius = 0.4 * max_radius
                
                # Extrusion parameters
                "extrude_depth_min": 5,
                "extrude_depth_max": 20,
                
                # Face selection probabilities
                "build_oriented_solid_probability": 0.5,  # 50% chance to use build_oriented_solid
                
                # Maximum attempts for various operations
                "max_solid_attempts": 10,
                "max_polygon_attempts": 100,
                "max_polygon_generation_attempts": 20,
            },
            
            # ===== BASE_SOLID.PY PARAMETERS =====
            "base_solid": {
                # Cuboid generation
                "cuboid_choices": [2, 1, 3, 4, 5, 6, 7, 8, 9, 10],
                "cuboid_base_prob": 514 / 1023,  # p = 514/1023
                
                # Cuboid dimensions (random ranges)
                "cuboid_x_min": 0, "cuboid_x_max": 40,
                "cuboid_y_min": 0, "cuboid_y_max": 40,
                "cuboid_width_min": 10, "cuboid_width_max": 30,
                "cuboid_depth_min": 10, "cuboid_depth_max": 30,
                "cuboid_height_min": 20, "cuboid_height_max": 60,
                
                # Rotation parameters
                "rotation_probability": 0.2,  # 20% chance of being angled
                "rotation_angles": [15, 30, 45, 60, 90],
                
                # Base offset parameters
                "base_offset_min": 5,
                "base_offset_max": 15,
                
                # Bounding box margin factor
                "bounding_margin_factor": 1.0,  # margin = max(xmax-xmin, ymax-ymin, 50)
                "bounding_margin_min": 50,
                
                # Polygon addition parameters (inherited from v6_current for consistency)
                "polygon_attempts": 100,
                "face_normal_threshold": 0.99,
                "position_threshold": 1e-3,
                "position_threshold_x": 1.0,
                
                # Build oriented solid parameters
                "oriented_solid_depth": 50,  # Fixed depth for oriented solids
                "oriented_solid_depth_min": 5,  # For x_faces random depth
                "oriented_solid_depth_max": 20,
            },
            
            # ===== LETTERING_SOLID.PY PARAMETERS =====
            "lettering_solid": {
                # Base expansion
                "base_expansion_margin": 2,  # Expand rectangle by 2 units on all sides
                "base_z_offset": 0.00025,  # Small z-offset for base
                
                # Grid parameters
                "grid_size": 5,  # 5x5 grid
                
                # Horizontal/vertical cuboid counts
                "cuboid_choices": [2, 1, 3, 4],
                "cuboid_base_prob": 8 / 15,  # p = 8/15
                
                # Vertical cuboid parameters
                "vertical_width_min_factor": 0.6,  # min = cell_w * 0.6
                "vertical_width_max_factor": 0.9,  # max = 0.9 * width
                "vertical_length_min_factor": 0.6,  # min = cell_l * 0.6
                "vertical_length_max_factor": 0.9,  # max = 0.9 * length / grid_size
                "vertical_length_divisor": 5,  # length / 5 in the max calculation
                "vertical_depth_min_factor": 0.5,  # min = depth * 0.5
                "vertical_depth_max_factor": 1.0,  # max = depth
                "vertical_embed_factor": 0.3,  # z = -depth * 0.3
                
                # Horizontal cuboid parameters
                "horizontal_width_min_factor": 0.6,  # min = cell_w * 0.6
                "horizontal_width_max_factor": 0.9,  # max = 0.9 * width / grid_size
                "horizontal_width_divisor": 5,  # width / 5 in the max calculation
                "horizontal_length_min_factor": 0.6,  # min = cell_l * 0.6
                "horizontal_length_max_factor": 0.9,  # max = 0.9 * length
                "horizontal_depth_min_factor": 0.5,  # min = depth * 0.5
                "horizontal_depth_max_factor": 1.0,  # max = depth
                "horizontal_embed_factor": 0.3,  # z = -depth * 0.3
            }
        }
    
    def save_to_file(self, filepath: Optional[str] = None) -> str:
        """Save configuration to JSON file"""
        if filepath is None:
            filepath = f"config_seed_{self.seed}.json"
        
        with open(filepath, 'w') as f:
            json.dump(self.config, f, indent=2, sort_keys=True)
        
        print(f"Configuration saved to: {filepath}")
        return filepath
    
    @classmethod
    def load_from_file(cls, filepath: str) -> 'ConfigurationManager':
        """Load configuration from JSON file"""
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Configuration file not found: {filepath}")
        
        with open(filepath, 'r') as f:
            config_data = json.load(f)
        
        # Create instance with the loaded seed
        seed = config_data.get('seed', 1)
        instance = cls(seed=seed)
        instance.config = config_data
        
        print(f"Configuration loaded from: {filepath}")
        return instance
    
    def get(self, section: str, key: str, default=None):
        """Get configuration value"""
        return self.config.get(section, {}).get(key, default)
    
    def get_section(self, section: str) -> Dict[str, Any]:
        """Get entire configuration section"""
        return self.config.get(section, {})
    
    def set(self, section: str, key: str, value: Any):
        """Set configuration value"""
        if section not in self.config:
            self.config[section] = {}
        self.config[section][key] = value
    
    def generate_weighted_probabilities(self, base_prob: float, count: int) -> np.ndarray:
        """Generate weighted probabilities: [p, p/2, p/4, p/8, ...]"""
        probs = np.array([base_prob / (2**i) for i in range(count)])
        return probs / probs.sum()  # Normalize to sum to 1
    
    def apply_seed(self):
        """Apply the configuration seed to random generators"""
        random.seed(self.seed)
        np.random.seed(self.seed)


def create_default_config(seed: int = 1) -> ConfigurationManager:
    """Create a default configuration with the given seed"""
    return ConfigurationManager(seed=seed)


def load_config(filepath: str) -> ConfigurationManager:
    """Load configuration from file"""
    return ConfigurationManager.load_from_file(filepath)


if __name__ == "__main__":
    # Test the configuration system
    import argparse
    
    parser = argparse.ArgumentParser(description="Configuration system test")
    parser.add_argument('--seed', type=int, default=1, help='Seed for configuration')
    parser.add_argument('--load', type=str, help='Load configuration from file')
    parser.add_argument('--save', action='store_true', help='Save configuration to file')
    args = parser.parse_args()
    
    if args.load:
        config = load_config(args.load)
        print(f"Loaded configuration for seed {config.seed}")
    else:
        config = create_default_config(args.seed)
        print(f"Created default configuration for seed {config.seed}")
    
    if args.save:
        filepath = config.save_to_file()
        print(f"Configuration saved to {filepath}")
    
    # Display some sample values
    print(f"Max cuboids: {config.get('v6_current', 'max_cuboids')}")
    print(f"Base width: {config.get('base_solid', 'cuboid_width_min')}-{config.get('base_solid', 'cuboid_width_max')}")
    print(f"Grid size: {config.get('lettering_solid', 'grid_size')}")