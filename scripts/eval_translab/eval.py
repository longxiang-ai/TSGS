# adapted from https://github.com/jzhangbs/DTUeval-python
import numpy as np
import open3d as o3d
import sklearn.neighbors as skln
from tqdm import tqdm
from scipy.io import loadmat
import multiprocessing as mp
import argparse
import os

def sample_single_tri(input_):
    n1, n2, v1, v2, tri_vert = input_
    c = np.mgrid[:n1+1, :n2+1]
    c += 0.5
    c[0] /= max(n1, 1e-7)
    c[1] /= max(n2, 1e-7)
    c = np.transpose(c, (1,2,0))
    k = c[c.sum(axis=-1) < 1]  # m2
    q = v1 * k[:,:1] + v2 * k[:,1:] + tri_vert
    return q

def write_vis_pcd(file, points, colors):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colors)
    o3d.io.write_point_cloud(file, pcd)

if __name__ == '__main__':
    mp.freeze_support()

    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default='data_in.ply')
    parser.add_argument('--scan', type=str, default='scene3')
    parser.add_argument('--mode', type=str, default='mesh', choices=['mesh', 'pcd'])
    parser.add_argument('--dataset_dir', type=str, default='.')
    parser.add_argument('--vis_out_dir', type=str, default='.')
    parser.add_argument('--downsample_density', type=float, default=0.002)
    parser.add_argument('--patch_size', type=float, default=60)
    parser.add_argument('--max_dist', type=float, default=20)
    parser.add_argument('--visualize_threshold', type=float, default=10)
    parser.add_argument('--height', type=float, default=-0.1)
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')
    args = parser.parse_args()

    np.random.seed(args.seed)

    thresh = args.downsample_density
    if args.mode == 'mesh':
        pbar = tqdm(total=9)
        pbar.set_description('read data mesh')
        data_mesh = o3d.io.read_triangle_mesh(args.data)

        vertices = np.asarray(data_mesh.vertices)
        triangles = np.asarray(data_mesh.triangles)
        tri_vert = vertices[triangles]

        pbar.update(1)
        pbar.set_description('sample pcd from mesh')
        v1 = tri_vert[:,1] - tri_vert[:,0]
        v2 = tri_vert[:,2] - tri_vert[:,0]
        l1 = np.linalg.norm(v1, axis=-1, keepdims=True)
        l2 = np.linalg.norm(v2, axis=-1, keepdims=True)
        area2 = np.linalg.norm(np.cross(v1, v2), axis=-1, keepdims=True)
        non_zero_area = (area2 > 0)[:,0]
        l1, l2, area2, v1, v2, tri_vert = [
            arr[non_zero_area] for arr in [l1, l2, area2, v1, v2, tri_vert]
        ]
        thr = thresh * np.sqrt(l1 * l2 / area2)
        n1 = np.floor(l1 / thr)
        n2 = np.floor(l2 / thr)

        with mp.Pool() as mp_pool:
            new_pts = mp_pool.map(sample_single_tri, ((n1[i,0], n2[i,0], v1[i:i+1], v2[i:i+1], tri_vert[i:i+1,0]) for i in range(len(n1))), chunksize=1024)

        new_pts = np.concatenate(new_pts, axis=0)
        data_pcd = np.concatenate([vertices, new_pts], axis=0)
    
    elif args.mode == 'pcd':
        pbar = tqdm(total=8)
        pbar.set_description('read data pcd')
        data_pcd_o3d = o3d.io.read_point_cloud(args.data)
        data_pcd = np.asarray(data_pcd_o3d.points)

    pbar.update(1)
    pbar.set_description('random shuffle pcd index')
    shuffle_rng = np.random.default_rng()
    shuffle_rng.shuffle(data_pcd, axis=0)

    pbar.update(1)
    pbar.set_description('downsample pcd')
    nn_engine = skln.NearestNeighbors(n_neighbors=1, radius=thresh, algorithm='kd_tree', n_jobs=-1)
    nn_engine.fit(data_pcd)
    rnn_idxs = nn_engine.radius_neighbors(data_pcd, radius=thresh, return_distance=False)
    mask = np.ones(data_pcd.shape[0], dtype=np.bool_)
    for curr, idxs in enumerate(rnn_idxs):
        if mask[curr]:
            mask[idxs] = 0
            mask[curr] = 1
    data_down = data_pcd[mask]

    
    # 坐标轴转换，从COLMAP坐标系转换到Blender坐标系
    data_down_blender = data_down.copy()
    # COLMAP -> Blender: x->x, y->-z, z->y
    data_down_blender = data_down[:, [0, 2, 1]]  # 先交换y和z
    data_down_blender[:, 2] = -data_down_blender[:, 2]  # 再将新的z轴(原y轴)取反
    
    data_down = data_down_blender
    
    pbar.update(1)
    pbar.set_description('read GT mesh')
    gt_mesh_path = os.path.join(args.dataset_dir, f'{args.scan}/meshes/scene_mesh.obj')
    gt_mesh = o3d.io.read_triangle_mesh(gt_mesh_path)
    
    pbar.set_description('sample points from GT mesh')
    gt_pcd = gt_mesh.sample_points_uniformly(number_of_points=1000000)
    stl = np.asarray(gt_pcd.points)
    
    # 添加高度过滤
    height_threshold = args.height  # 需要在参数中添加这个参数
    # 找出高于阈值的点的索引
    valid_height_mask = stl[:, 1] > height_threshold  # 假设y轴是高度方向，根据实际情况可能需要修改为x或z
    # 过滤点云
    stl = stl[valid_height_mask]
    valid_height_mask = data_down[:, 1] > height_threshold  # 假设y轴是高度方向，根据实际情况可能需要修改为x或z
    # 过滤点云
    data_down = data_down[valid_height_mask]


    # 补充绘制一个点云，包含了data_down和stl
    # 为两种点云添加不同的颜色
    data_down_pcd = o3d.geometry.PointCloud()
    data_down_pcd.points = o3d.utility.Vector3dVector(data_down)
    data_down_pcd.colors = o3d.utility.Vector3dVector(np.tile([1, 0, 0], (len(data_down), 1)))  # 红色表示预测的点云
    
    stl_pcd = o3d.geometry.PointCloud()
    stl_pcd.points = o3d.utility.Vector3dVector(stl)
    stl_pcd.colors = o3d.utility.Vector3dVector(np.tile([0, 1, 0], (len(stl), 1)))  # 绿色表示GT点云
    
    # 合并并保存点云
    merged_pcd = data_down_pcd + stl_pcd
    o3d.io.write_point_cloud(f'{args.vis_out_dir}/vis_{args.scan}_data_down_stl.ply', merged_pcd)
    

    pbar.update(1)
    pbar.set_description('compute data2stl')
    nn_engine.fit(stl)
    dist_d2s, idx_d2s = nn_engine.kneighbors(data_down, n_neighbors=1, return_distance=True)
    max_dist = args.max_dist
    mean_d2s = dist_d2s[dist_d2s < max_dist].mean()

    pbar.update(1)
    pbar.set_description('compute stl2data')
    
    nn_engine.fit(data_down)
    dist_s2d, idx_s2d = nn_engine.kneighbors(stl, n_neighbors=1, return_distance=True)
    mean_s2d = dist_s2d[dist_s2d < max_dist].mean()

    # 计算 F1 分数
    pbar.update(1)
    pbar.set_description('compute F1 score')
    threshold = args.visualize_threshold / 2000 # 使用可视化阈值作为 F1 计算的阈值
    print("f1 score distance threshold:{}".format(threshold))
    
    # 计算精确度和召回率
    precision = float(sum(d < threshold for d in dist_d2s)) / float(len(dist_d2s))
    recall = float(sum(d < threshold for d in dist_s2d)) / float(len(dist_s2d))
    
    # 计算 F1 分数
    if precision + recall > 0:
        f1_score = 2 * precision * recall / (precision + recall)
    else:
        f1_score = 0

    pbar.update(1)
    pbar.set_description('visualize error')
    vis_dist = args.visualize_threshold
    R = np.array([[1,0,0]], dtype=np.float64)
    G = np.array([[0,1,0]], dtype=np.float64)
    B = np.array([[0,0,1]], dtype=np.float64)
    W = np.array([[1,1,1]], dtype=np.float64)
    data_color = np.tile(B, (data_down.shape[0], 1))
    data_alpha = dist_d2s.clip(max=vis_dist) / vis_dist
    data_color = R * data_alpha + W * (1-data_alpha)
    data_color[dist_d2s[:,0] >= max_dist]= G
    write_vis_pcd(f'{args.vis_out_dir}/vis_{args.scan}_d2s.ply', data_down, data_color)
    stl_color = np.tile(B, (stl.shape[0], 1))
    stl_alpha = dist_s2d.clip(max=vis_dist) / vis_dist
    stl_color = R * stl_alpha + W * (1-stl_alpha)
    stl_color[dist_s2d[:,0] >= max_dist]= G
    write_vis_pcd(f'{args.vis_out_dir}/vis_{args.scan}_s2d.ply', stl, stl_color)
    
    pbar.update(1)
    pbar.set_description('done')
    pbar.close()
    over_all = (mean_d2s + mean_s2d) / 2
    print("mean_d2s:{}, mean_s2d:{}, over_all:{}".format(mean_d2s * 1000, mean_s2d * 1000, over_all * 1000))
    print("precision:{}, recall:{}, f1_score:{}".format(precision, recall, f1_score))
    
    import json
    with open(f'{args.vis_out_dir}/results_{os.path.basename(args.data).split("_")[-1].split(".")[0]}.json', 'w') as fp:
        json.dump({
            'mean_d2s': mean_d2s * 1000,
            'mean_s2d': mean_s2d * 1000,
            'overall': over_all * 1000,
            'precision': precision,
            'recall': recall,
            'f1_score': f1_score,
        }, fp, indent=True)