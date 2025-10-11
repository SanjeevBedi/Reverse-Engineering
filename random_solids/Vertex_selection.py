import numpy as np

def extract_possible_vertices_from_summaries(Vertex_Front_View, Vertex_Top_View, all_vertices_sorted):
    print("[DEBUG] Extracting z-levels from Front View summary array...")
    front_view_summary = make_summary_array(
        Vertex_Front_View, all_vertices_sorted, 
        np.array([0, -1, 0]), 'Front View')
    if front_view_summary is not None and front_view_summary.shape[0] > 0:
        z_coords = np.unique(front_view_summary[:, 2])
        z_coords_sorted = np.sort(z_coords)
        print(f"[DEBUG] Unique z-levels (sorted): {z_coords_sorted}")
        top_view_summary = make_summary_array(Vertex_Top_View, all_vertices_sorted, np.array([0, 0, 1]), 'Top View')
        if top_view_summary is not None and top_view_summary.shape[0] > 0:
            possible_vertices = []
            for row in top_view_summary:
                x, y = row[0], row[1]
                for z in z_coords_sorted:
                    possible_vertices.append([x, y, z])
            possible_vertices = np.array(possible_vertices)
            print("Possible_Vertices array (x, y from Top View, z from z-levels):")
            print(possible_vertices)
        else:
            print("[DEBUG] Top View summary array is empty or None.")
    else:
        print("[DEBUG] Front View summary array is empty or None.")


def project_to_view(vertex, normal):
    normal = np.array(normal)
    normal = normal / np.linalg.norm(normal)
    if np.allclose(normal, [0, 0, 1]):
        u_axis = np.array([1, 0, 0])
        v_axis = np.array([0, 1, 0])
    elif np.allclose(normal, [0, 1, 0]):
        u_axis = np.array([1, 0, 0])
        v_axis = np.array([0, 0, 1])
    elif np.allclose(normal, [1, 0, 0]):
        u_axis = np.array([0, 1, 0])
        v_axis = np.array([0, 0, 1])
    else:
        u_axis = np.cross([0, 0, 1], normal)
        if np.linalg.norm(u_axis) < 1e-8:
            u_axis = np.cross([0, 1, 0], normal)
        u_axis = u_axis / np.linalg.norm(u_axis)
        v_axis = np.cross(normal, u_axis)
        v_axis = v_axis / np.linalg.norm(v_axis)
    vertex = np.array(vertex)
    u = np.dot(vertex, u_axis)
    v = np.dot(vertex, v_axis)
    return u, v


def filter_possible_vertices(possible_vertices, summary_array, view_name, u_col, v_col, normal, tol=1e-6):
    valid_indices = []
    for idx, vert in enumerate(possible_vertices):
        u, v = project_to_view(vert, normal)
        u_matches = np.isclose(u, summary_array[:, u_col], atol=tol)
        v_matches = np.isclose(v, summary_array[:, v_col], atol=tol)
        match_found = np.any(u_matches & v_matches)
        if match_found:
            valid_indices.append(idx)
    return valid_indices


def make_summary_array(vertex_array, all_vertices_sorted, proj_normal, view_name):
    print(f"[DEBUG] Processing {view_name} summary array...")
    try:
        vertex_array = np.asarray(vertex_array)
        n = vertex_array.shape[0]
        print(f"[DEBUG] {view_name}: vertex_array shape = {vertex_array.shape}")
        nonzero_row_indices = [i for i in range(n) if np.any(vertex_array[i, :])]
        num_nonzero = len(nonzero_row_indices)
        print(f"[DEBUG] {view_name}: number of nonzero rows = {num_nonzero}")
        arr = np.zeros((num_nonzero, 6 + num_nonzero), dtype=float)
        def project_vertex(vertex, normal):
            normal = np.array(normal)
            normal = normal / np.linalg.norm(normal)
            if abs(normal[0]) < 0.9:
                temp = np.array([1.0, 0.0, 0.0])
            else:
                temp = np.array([0.0, 1.0, 0.0])
            u = temp - np.dot(temp, normal) * normal
            u = u / np.linalg.norm(u)
            v = np.cross(normal, u)
            v = v / np.linalg.norm(v)
            vertex = np.array(vertex)
            proj_u = np.dot(vertex, u)
            proj_v = np.dot(vertex, v)
            return proj_u, proj_v, 0.0
        for row_idx, v_idx in enumerate(nonzero_row_indices):
            x, y, z = all_vertices_sorted[v_idx]
            arr[row_idx, 0:3] = [x, y, z]
            xp, yp, _ = project_vertex([x, y, z], proj_normal)
            if view_name == 'Top View':
                arr[row_idx, 3] = xp
                arr[row_idx, 4] = yp
            elif view_name == 'Front View':
                arr[row_idx, 3] = xp
                arr[row_idx, 5] = yp
            elif view_name == 'Side View':
                arr[row_idx, 4] = xp
                arr[row_idx, 5] = yp
            else:
                arr[row_idx, 3] = xp
                arr[row_idx, 4] = yp
            arr[row_idx, 6:] = vertex_array[v_idx, nonzero_row_indices]
        print(f"\n[DEBUG] Summary array for {view_name} (shape: {arr.shape}):")
        print(arr)
        print(f"[DEBUG] Finished {view_name} summary array.")
        return arr
    except Exception as e:
        print(f"[ERROR] Exception in make_summary_array for {view_name}: {e}")
        import traceback
        traceback.print_exc()
        return None
