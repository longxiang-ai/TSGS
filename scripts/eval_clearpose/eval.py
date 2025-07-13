# adapted from https://github.com/jzhangbs/DTUeval-python
import numpy as np
import open3d as o3d
import sklearn.neighbors as skln
from tqdm import tqdm
from scipy.io import loadmat
import multiprocessing as mp
import argparse
import os
import trimesh

def write_vis_pcd(file, points, colors):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colors)
    o3d.io.write_point_cloud(file, pcd)

if __name__ == '__main__':
    mp.freeze_support()

    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default='data_in.ply', help='path to the reconstructed mesh ply')
    parser.add_argument('--scan', type=str, default='set1scene1', help='scene id')
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
    pbar = tqdm(total=8)
    pbar.set_description('read data pcd')
    data_trimesh = trimesh.load(args.data)
    data_mesh = o3d.geometry.TriangleMesh()
    data_mesh.vertices = o3d.utility.Vector3dVector(data_trimesh.vertices)
    data_mesh.triangles = o3d.utility.Vector3iVector(data_trimesh.faces)

    data_down = data_mesh.sample_points_uniformly(number_of_points=1000000)
    data_down = np.asarray(data_down.points)

    
    # 坐标轴转换，从COLMAP坐标系转换到Blender坐标系
    data_down_blender = data_down.copy()
    # COLMAP -> Blender: x->x, y->-z, z->y
    data_down_blender = data_down[:, [0, 2, 1]]  # 先交换y和z
    data_down_blender[:, 2] = -data_down_blender[:, 2]  # 再将新的z轴(原y轴)取反
    data_down = data_down_blender
    
    

    pbar.set_description('read GT mesh')
    gt_mesh_path = f'/data0/ClearPose/{args.scan}.obj'
    # gt_mesh = o3d.io.read_triangle_mesh(gt_mesh_path)
    trimesh_obj = trimesh.load(gt_mesh_path)
    if isinstance(trimesh_obj, trimesh.Scene):
        meshes = []
        for geometry in trimesh_obj.geometry.values():
            if hasattr(geometry, 'vertices') and hasattr(geometry, 'faces'):
                meshes.append(geometry)
        
        if len(meshes) == 0:
            raise ValueError("Scene中没有找到有效的mesh几何体")
        elif len(meshes) == 1:
            trimesh_obj = meshes[0]
        else:
            # 合并多个mesh
            trimesh_obj = trimesh.util.concatenate(meshes)
    gt_mesh = o3d.geometry.TriangleMesh()
    gt_mesh.vertices = o3d.utility.Vector3dVector(trimesh_obj.vertices)
    gt_mesh.triangles = o3d.utility.Vector3iVector(trimesh_obj.faces)

    pbar.set_description('sample points from GT mesh')
    gt_pcd = gt_mesh.sample_points_uniformly(number_of_points=1000000)
    stl = np.asarray(gt_pcd.points)
    
    # gt中max height value
    bottom_height_threshold = min(stl[:, 1])
    bottom_valid_height_mask = data_down[:, 1] > bottom_height_threshold
    data_down = data_down[bottom_valid_height_mask]

    top_height_threshold = max(stl[:, 1])
    top_valid_height_mask = data_down[:, 1] < top_height_threshold
    data_down = data_down[top_valid_height_mask]

    left_min_threshold = min(stl[:, 0])
    left_max_threshold = max(stl[:, 0])
    left_valid_mask = (data_down[:, 0] > left_min_threshold) & (data_down[:, 0] < left_max_threshold)
    data_down = data_down[left_valid_mask]

    forward_min_threshold = min(stl[:, 2])
    forward_max_threshold = max(stl[:, 2])
    forward_valid_mask = (data_down[:, 2] > forward_min_threshold) & (data_down[:, 2] < forward_max_threshold)
    data_down = data_down[forward_valid_mask]

    
    
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
    nn_engine = skln.NearestNeighbors(n_neighbors=1, radius=thresh, algorithm='kd_tree', n_jobs=-1)
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
    print(os.path.basename(args.data))
    with open(f'{args.vis_out_dir}/results_{os.path.basename(args.data).split("_")[-1].split(".")[0]}.json', 'w') as fp:
        json.dump({
            'mean_d2s': mean_d2s * 1000,
            'mean_s2d': mean_s2d * 1000,
            'overall': over_all * 1000,
            'precision': precision,
            'recall': recall,
            'f1_score': f1_score,
        }, fp, indent=True)