#!/usr/bin/env python3
"""
Test script for the create_rotated_grid_from_polygon function
"""

import numpy as np
import torch
import matplotlib.pyplot as plt
from shapely.geometry import Polygon
from src.fw.simulators.py_sim2d import PySimEnv

def test_grid_function():
    """Test the grid creation function with a simple polygon"""
    
    # Create a simple test polygon (a square)
    test_coords = np.array([
        [0, 0],
        [5, 0], 
        [5, 5],
        [0, 5],
        [0, 0]  # Close the polygon
    ])
    
    test_polygon = Polygon(test_coords)
    
    # Create PySimEnv instance (we need this to access the method)
    env = PySimEnv()
    
    # Test parameters
    origin = (2.5, 2.5)  # Center of the square
    angle = 45  # 45 degree rotation
    
    # Create the grid
    grid = env.create_rotated_grid_from_polygon(test_polygon, origin, angle)
    
    print(f"Grid shape: {grid.shape}")
    print(f"Grid dtype: {grid.dtype}")
    print(f"Number of filled cells: {torch.sum(grid).item()}")
    print(f"Total cells: {grid.numel()}")
    
    # Visualize the result
    plt.figure(figsize=(12, 5))
    
    # Plot original polygon
    plt.subplot(1, 2, 1)
    x, y = test_polygon.exterior.xy
    plt.plot(x, y, 'b-', linewidth=2, label='Original Polygon')
    plt.scatter([origin[0]], [origin[1]], color='red', s=100, label='Origin', zorder=5)
    plt.axis('equal')
    plt.grid(True)
    plt.title('Original Polygon')
    plt.legend()
    
    # Plot grid
    plt.subplot(1, 2, 2)
    plt.imshow(grid.numpy(), origin='lower', cmap='Blues', interpolation='nearest')
    plt.title(f'Grid (rotated {angle}°)')
    plt.colorbar()
    
    plt.tight_layout()
    plt.savefig('/home/antoon/Documents/phd/RudoLF/grid_test_visualization.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    return grid

if __name__ == "__main__":
    grid = test_grid_function()
    print("Test completed successfully!")
