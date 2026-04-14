import io
import numpy as np
from sklearn.decomposition import PCA

def save_vg_plane(points, filepath, planes = None, normals = None):

    from random import random
    index = np.unique(points[:, 3])
    group_num = len(index)
    if planes is None:
        planes = np.zeros((group_num, 4), dtype=np.float32)
        normals = np.zeros((points.shape[0], 3), dtype=np.float32)
        for i in range(group_num):
            group_points = points[points[:, 3] == index[i], :3]
            pca = PCA(n_components=3)
            pca.fit(group_points)
            eig_vec = pca.components_
            normal = eig_vec[2, :]  # (a, b, c) normalized
            centroid = np.mean(group_points, axis=0)

            
            d = -centroid.dot(normal)
            param = np.append(normal, d)
            planes[i, :] = param
           
            normals[points[:, 3] == index[i], :] = normal
    

    out = ''
    out += f'num_points: {points.shape[0]}\n'
    output = io.StringIO()
    np.savetxt(output, points[:, :3], fmt="%.6f %.6f %.6f")
    out += output.getvalue()
    output.close()

    out += f'num_colors: {points.shape[0]}\n'
    colors = np.ones((points.shape[0], 3))
    output = io.StringIO()
    np.savetxt(output, colors, fmt="%d %d %d")
    out += output.getvalue()
    output.close()

    out += f'num_normals: {points.shape[0]}\n'
    output = io.StringIO()
    np.savetxt(output, normals, fmt="%.6f %.6f %.6f")
    out += output.getvalue()
    output.close()

    num_groups = planes.shape[0]
    out += f'num_groups: {num_groups}\n'

    j_base = 0
    for i in range(num_groups):
        group_num = np.sum(points[:, 3] == index[i])
        out += 'group_type: 0\n'
        out += 'num_group_parameters: 4\n'
        out += f'group_parameters: {planes[i][0]} {planes[i][1]} {planes[i][2]} {planes[i][3]}\n'
        out += f'group_label: group_{i}\n'
        out += f'group_color: {random()} {random()} {random()}\n'
        out += f'group_num_point: {group_num}\n'
        out += ' '.join(str(j) for j in range(j_base, j_base + group_num)) + '\n'
        j_base += group_num
        out += 'num_children: 0\n'

    with open(filepath, 'w') as fout:
        fout.writelines(out)


def save_vg(
    points,
    filepath,
    group_info=None,
    group_col: int = 3,
    colors=None,
    normals=None,
):

    from random import random
    # ------------- Collect grouping -------------
    if points.ndim != 2 or points.shape[1] <= group_col:
        raise ValueError("points must have at least group_col+1 columns (xyz + group id)")
    xyz = points[:, :3]
    group_ids = points[:, group_col].astype(int)
    unique_ids = np.unique(group_ids)
    G = len(unique_ids)

    # Minimal handling: expect list of dicts with 'id','type','parameters'.
    if group_info is None:
        raise ValueError("group_info must be provided (list of {id,type,parameters}).")
    normalized_entries = list(group_info)
    lookup = {int(e['id']): e for e in normalized_entries}
    unique_ids_list = unique_ids.tolist()

    if not all(gid in lookup for gid in unique_ids_list):
        if len(normalized_entries) == len(unique_ids_list):
            lookup = {gid: e for gid, e in zip(unique_ids_list, normalized_entries)}
        else:
            missing = [gid for gid in unique_ids_list if gid not in lookup]
            raise ValueError(f"{filepath}: Missing group_info entry for group ids {missing}. Provided ids: {sorted(lookup.keys())}")

    group_param_list = []
    for gid in unique_ids_list:
        e = lookup[gid]
        t = int(e['type'])
        params = np.asarray(e['parameters'], dtype=np.float32).reshape(-1)
        if params.shape[0] != 10:
            raise ValueError(f"group {gid} params length {params.shape[0]} != 10 (params={params})")
        if not np.all(np.isfinite(params)):
            raise ValueError(f"group {gid} has non-finite parameter values: {params}")
        group_param_list.append((gid, t, params))
    G = len(group_param_list)

    # ------------- Colors -------------
    group_colors = {gid: (random(), random(), random()) for gid in unique_ids}
    colors = np.zeros((points.shape[0], 3), dtype=np.float32)
    for gid, (r, g, b) in group_colors.items():
        colors[group_ids == gid] = (r, g, b)

    # ------------- Assemble output -------------
    out = ''
    out += f'num_points: {points.shape[0]}\n'
    sio = io.StringIO(); np.savetxt(sio, xyz, fmt='%.6f %.6f %.6f'); out += sio.getvalue(); sio.close()

    out += f'num_colors: {points.shape[0]}\n'
    sio = io.StringIO(); np.savetxt(sio, colors, fmt='%.6f %.6f %.6f'); out += sio.getvalue(); sio.close()

    out += f'num_normals: {points.shape[0]}\n'
    sio = io.StringIO(); np.savetxt(sio, normals, fmt='%.6f %.6f %.6f'); out += sio.getvalue(); sio.close()

    out += f'num_groups: {G}\n'

    
    running_index = 0
    
    ordered_indices = []
    for gid in unique_ids:
        ordered_indices.extend(np.where(group_ids == gid)[0].tolist())
    
    new_index_map = np.zeros(points.shape[0], dtype=int)
    for new_i, old_i in enumerate(ordered_indices):
        new_index_map[old_i] = new_i

    for order_i, (gid, t, params) in enumerate(group_param_list):
        mask = group_ids == gid
        g_point_ids_new = new_index_map[np.where(mask)[0]]
        
        if t == 0:
            plane_params = params[-4:]  # assuming ordering [A,B,C,D,E,F,G,H,I,J]
            out += f'group_type: {t}\n'
            out += 'num_group_parameters: 4\n'
            out += 'group_parameters: ' + ' '.join(f'{x}' for x in plane_params.tolist()) + '\n'
        else:
            out += f'group_type: {t}\n'
            out += f'num_group_parameters: {len(params)}\n'
            out += 'group_parameters: ' + ' '.join(f'{x}' for x in params.tolist()) + '\n'
        out += f'group_label: group_{order_i}\n'
        rc, gc, bc = group_colors[gid]
        out += f'group_color: {rc} {gc} {bc}\n'
        out += f'group_num_point: {g_point_ids_new.size}\n'
        out += ' '.join(str(i) for i in g_point_ids_new.tolist()) + '\n'
        out += 'num_children: 0\n'

    with open(filepath, 'w') as f:
        f.write(out)
