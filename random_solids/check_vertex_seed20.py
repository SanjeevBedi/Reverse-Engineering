import numpy as np

# Load connectivity matrices
data = np.load('Output/connectivity_matrices_seed_20.npz', allow_pickle=True)

print('Available views:', data.files)
print()

# Check each view's connectivity matrix
for view_name in ['top_view_matrix', 'front_view_matrix', 'side_view_matrix']:
    if view_name in data:
        matrix = data[view_name]
        view_title = view_name.replace('_matrix', '').replace('_', ' ').title()
        print(f'{view_title}:')
        print(f'  Matrix shape: {matrix.shape}')
        print(f'  Non-zero entries: {np.count_nonzero(matrix)}')
        
        # Extract vertex coordinates (first 2 columns are x,y)
        num_vertices = matrix.shape[0]
        vertices_2d = matrix[:, :2]
        
        print(f'  Vertices: {num_vertices}')
        print(f'  X range: [{vertices_2d[:, 0].min():.3f}, {vertices_2d[:, 0].max():.3f}]')
        print(f'  Y range: [{vertices_2d[:, 1].min():.3f}, {vertices_2d[:, 1].max():.3f}]')
        print()

# Now check for our target vertex
target_3d = np.array([24.56, 23.33, 28.663])
print(f'\nSearching for target vertex: {target_3d}')
print()

# For each view, check if the projected coordinates match
views_info = {
    'top_view_matrix': {'normal': np.array([0, 0, 1]), 'axes': (0, 1)},
    'front_view_matrix': {'normal': np.array([0, -1, 0]), 'axes': (0, 2)},
    'side_view_matrix': {'normal': np.array([1, 0, 0]), 'axes': (1, 2)}
}

for view_name, view_info in views_info.items():
    if view_name in data:
        matrix = data[view_name]
        axes = view_info['axes']
        proj_coords = target_3d[list(axes)]
        
        view_title = view_name.replace('_matrix', '').replace('_', ' ').title()
        print(f'{view_title} - Looking for projection: ({proj_coords[0]:.3f}, {proj_coords[1]:.3f})')
        
        vertices_2d = matrix[:, :2]
        
        # Find close matches
        distances = np.linalg.norm(vertices_2d - proj_coords, axis=1)
        close_idx = np.where(distances < 0.1)[0]
        
        if len(close_idx) > 0:
            print(f'  Found {len(close_idx)} close matches:')
            for idx in close_idx[:5]:  # Show first 5
                v = vertices_2d[idx]
                dist = distances[idx]
                print(f'    V{idx}: ({v[0]:.6f}, {v[1]:.6f}) - Distance: {dist:.6f}')
        else:
            print('  No close matches found')
            # Show closest 3
            closest_idx = np.argsort(distances)[:3]
            print('  Closest vertices:')
            for idx in closest_idx:
                v = vertices_2d[idx]
                dist = distances[idx]
                print(f'    V{idx}: ({v[0]:.6f}, {v[1]:.6f}) - Distance: {dist:.6f}')
        print()
