#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import torch
from scene import Scene
import os

os.environ["OMP_NUM_THREADS"] = "8"
from tqdm import tqdm
from os import makedirs
from gaussian_renderer import render, render_depth
import torchvision
from utils.general_utils import safe_state
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, get_combined_args
from gaussian_renderer import GaussianModel
import numpy as np
import cv2
import open3d as o3d
import copy

clearpose_bounds = {
    "set1scene1": {
        "min": [-0.5290, -0.5420, -0.0670],
        "max": [0.5930, 0.4820, 0.3080]
    },
    "set2scene1": {
        "min": [-0.6280, -0.5520, -0.1320],
        "max": [0.6410, 0.6490, 0.4120]
    },
    "set3scene1": {
        "min": [-0.6890, -0.6410, -0.2310],
        "max": [0.6520, 0.5910, 0.4870]
    },
    "set4scene1": {
        "min": [-0.5820, -0.7040, -0.2250],
        "max": [0.6190, 0.6830, 0.6620]
    }
}
                
def clean_mesh(mesh, min_len=1000):
    with o3d.utility.VerbosityContextManager(o3d.utility.VerbosityLevel.Debug) as cm:
        triangle_clusters, cluster_n_triangles, cluster_area = (mesh.cluster_connected_triangles())
    triangle_clusters = np.asarray(triangle_clusters)
    cluster_n_triangles = np.asarray(cluster_n_triangles)
    cluster_area = np.asarray(cluster_area)
    triangles_to_remove = cluster_n_triangles[triangle_clusters] < min_len
    mesh_0 = copy.deepcopy(mesh)
    mesh_0.remove_triangles_by_mask(triangles_to_remove)
    return mesh_0

def post_process_mesh(mesh, cluster_to_keep=1):
    """
    Post-process a mesh to filter out floaters and disconnected parts
    """
    import copy
    print("post processing the mesh to have {} clusterscluster_to_kep".format(cluster_to_keep))
    mesh_0 = copy.deepcopy(mesh)
    with o3d.utility.VerbosityContextManager(o3d.utility.VerbosityLevel.Debug) as cm:
            triangle_clusters, cluster_n_triangles, cluster_area = (mesh_0.cluster_connected_triangles())

    triangle_clusters = np.asarray(triangle_clusters)
    cluster_n_triangles = np.asarray(cluster_n_triangles)
    cluster_area = np.asarray(cluster_area)
    n_cluster = np.sort(cluster_n_triangles.copy())[-cluster_to_keep]
    n_cluster = max(n_cluster, 50) # filter meshes smaller than 50
    triangles_to_remove = cluster_n_triangles[triangle_clusters] < n_cluster
    mesh_0.remove_triangles_by_mask(triangles_to_remove)
    mesh_0.remove_unreferenced_vertices()
    mesh_0.remove_degenerate_triangles()
    print("num vertices raw {}".format(len(mesh.vertices)))
    print("num vertices post {}".format(len(mesh_0.vertices)))
    return mesh_0

@torch.no_grad()
def render_set(model_path, name, iteration, views, scene, gaussians, pipeline, background, 
               app_model=None, max_depth=5.0, volume=None, use_depth_filter=False, specular=None, window_size: float=0.03, transparency_threshold: float=0.15, start_threshold: float=0.0, end_threshold: float=0.2, use_transparent_depth: bool=False, replace_with_nearest_depth: bool=False, depth_transparency_blended: bool=False):
    gts_path = os.path.join(model_path, name, "ours_{}".format(iteration), "gt")
    render_path = os.path.join(model_path, name, "ours_{}".format(iteration), "renders")
    render_depth_path = os.path.join(model_path, name, "ours_{}".format(iteration), "renders_depth")

    if use_transparent_depth:
        render_depth_transparency_path = os.path.join(model_path, name, "ours_{}".format(iteration), "renders_depth_transparency")
    makedirs(gts_path, exist_ok=True)
    makedirs(render_path, exist_ok=True)
    makedirs(render_depth_path, exist_ok=True)
    if use_transparent_depth:
        makedirs(render_depth_transparency_path, exist_ok=True)

    depths_tsdf_fusion = []
    for idx, view in enumerate(tqdm(views, desc="Rendering progress")):
        gt, _, gt_delight, gt_normal, transparencies_map = view.get_image()
        if specular is not None:
            dir_pp = (gaussians.get_xyz - view.camera_center.repeat(gaussians.get_features.shape[0], 1))
            dir_pp_normalized = dir_pp / dir_pp.norm(dim=1, keepdim=True)
            normal = gaussians.get_normal_axis(dir_pp_normalized=dir_pp_normalized, return_delta=True)
            mlp_color = specular.step(gaussians.get_asg_features, dir_pp_normalized, normal)
        else:
            mlp_color = None
        
        out = render(view, gaussians, pipeline, background, app_model=app_model, mlp_color=mlp_color, transparency_threshold=transparency_threshold)
        if use_transparent_depth:
            transparency_map = out['out_transparency_map'].detach().cpu().numpy()[0]
            cv2.imwrite(os.path.join(render_depth_transparency_path, view.image_name + "_transparency_map.png"), (transparency_map * 255).astype(np.uint8))
        if use_transparent_depth:
            out_depth = render_depth(view, gaussians, pipeline, background, app_model=app_model, mlp_color=mlp_color, transparencies_map=transparencies_map, transparency_threshold=transparency_threshold, window_size=window_size, start_threshold=start_threshold, end_threshold=end_threshold)
        
        rendering = out["render"].clamp(0.0, 1.0)
        _, H, W = rendering.shape
        if not use_transparent_depth:
            print("warning: using plane depth, not our proposed depth")
            """
            This will downgrades to PGSR's unbiased plane depth, not our proposed depth
            """
            depth = out["plane_depth"].squeeze()
        else:
            depth = out_depth['out_transparency_depth'].squeeze()
            
            if depth_transparency_blended:
                # blender out_transparency_depth with plane_depth using transparencies_map
                # default is False, maybe the table will be more flat, however it might cause a performance drop
                depth_transparency_blended = out_depth['out_transparency_depth'].squeeze() * transparencies_map + (1 - transparencies_map) * out["plane_depth"].squeeze()
                depth = depth_transparency_blended.squeeze()
    
        
        depth_tsdf = depth.clone()
        depth = depth.detach().cpu().numpy()
        
        # depth_i, depth_color: for visualization, don't use when tsdf fusion
        depth_i = (depth - depth.min()) / (depth.max() - depth.min() + 1e-20)
        print(depth_i.min(), depth_i.max())
        depth_i = (depth_i * 255).clip(0, 255).astype(np.uint8)
        depth_color = cv2.applyColorMap(depth_i, cv2.COLORMAP_JET)

        if 'out_nearest_depth' in out:
            cpu_out_nearest_depth = out['out_nearest_depth'].squeeze().detach().cpu().numpy()
            
            mask = (cpu_out_nearest_depth >= 1e2) | (cpu_out_nearest_depth <= 0)
            valid_depths = cpu_out_nearest_depth[~mask]
            
            if valid_depths.size > 0:
                normalized = np.zeros_like(cpu_out_nearest_depth)
                normalized[~mask] = valid_depths / 2
                normalized[mask] = 0
            else:
                normalized = np.zeros_like(cpu_out_nearest_depth)
            
            colored_depth = cv2.applyColorMap((normalized * 255).astype(np.uint8), cv2.COLORMAP_JET)
            cv2.imwrite(os.path.join(render_depth_path, view.image_name + "_out_nearest_depth.png"), colored_depth)
            cv2.imwrite(os.path.join(render_depth_path, view.image_name + "_out_nearest_depth_mask.png"), mask.astype(np.uint8) * 255)
                    
            cpu_out_plane_depth = out['plane_depth'].squeeze().detach().cpu().numpy()
            valid_out_plane_depth = cpu_out_plane_depth / 2
            colored_out_plane_depth = cv2.applyColorMap((valid_out_plane_depth * 255).astype(np.uint8), cv2.COLORMAP_JET)
            cv2.imwrite(os.path.join(render_depth_path, view.image_name + "_out_plane_depth.png"), colored_out_plane_depth)
            
            # calculate difference between out_plane_depth and out_nearest_depth
            diff = ((cpu_out_plane_depth - cpu_out_nearest_depth) + 1) / 2. * 5.
            colored_diff = cv2.applyColorMap((diff * 255).astype(np.uint8), cv2.COLORMAP_JET)
            cv2.imwrite(os.path.join(render_depth_path, view.image_name + "_diff_plane_nearest.png"), colored_diff)
            
            # calculate difference between out_plane_depth and out_transparency_depth
            if use_transparent_depth:
                cpu_out_transparency_depth = out_depth['out_transparency_depth'].squeeze().detach().cpu().numpy()
                diff = ((cpu_out_plane_depth - cpu_out_transparency_depth) + 1) / 2. * 5
                colored_diff = cv2.applyColorMap((diff * 255).astype(np.uint8), cv2.COLORMAP_JET)
                cv2.imwrite(os.path.join(render_depth_path, view.image_name + "_diff_plane_transparency.png"), colored_diff)
            
        if replace_with_nearest_depth:
            print("warning: using nearest depth, not our proposed depth")
            """
            This will downgrades to nearest depth, not our proposed depth.
            Nearest depth will introduce artifacts, and unstable floaters.
            """
            depth_tsdf = out['out_nearest_depth'].clone()
        else:
            depth_tsdf = depth_tsdf

        normal = out["rendered_normal"].permute(1,2,0)
        normal = normal/(normal.norm(dim=-1, keepdim=True)+1.0e-8)
        normal = normal.detach().cpu().numpy()
        normal = ((normal+1) * 127.5).astype(np.uint8).clip(0, 255)

        if name == 'test':
            torchvision.utils.save_image(gt.clamp(0.0, 1.0), os.path.join(gts_path, view.image_name + ".png"))
            if gt_delight is not None:
                torchvision.utils.save_image(gt_delight.clamp(0.0, 1.0), os.path.join(gts_path, view.image_name + "_delight.png"))
            if gt_normal is not None:
                torchvision.utils.save_image(gt_normal.clamp(0.0, 1.0), os.path.join(gts_path, view.image_name + "_normal.png"))
            torchvision.utils.save_image(rendering, os.path.join(render_path, view.image_name + ".png"))
        else:
            rendering_np = (rendering.permute(1,2,0).clamp(0,1)[:,:,[2,1,0]]*255).detach().cpu().numpy().astype(np.uint8)
            cv2.imwrite(os.path.join(render_path, view.image_name + ".jpg"), rendering_np)
        cv2.imwrite(os.path.join(render_depth_path, view.image_name + ".jpg"), depth_color)

        if use_depth_filter:
            view_dir = torch.nn.functional.normalize(view.get_rays(), p=2, dim=-1)
            depth_normal = out["depth_normal"].permute(1,2,0)
            depth_normal = torch.nn.functional.normalize(depth_normal, p=2, dim=-1)
            dot = torch.sum(view_dir*depth_normal, dim=-1).abs()
            angle = torch.acos(dot)
            mask = angle > (80.0 / 180 * 3.14159)
            depth_tsdf[mask] = 0
        depths_tsdf_fusion.append(depth_tsdf.squeeze().cpu())
        
    if volume is not None:
        depths_tsdf_fusion = torch.stack(depths_tsdf_fusion, dim=0)
        for idx, view in enumerate(tqdm(views, desc="TSDF Fusion progress")):
            ref_depth = depths_tsdf_fusion[idx].cuda()

            if view.mask is not None:
                ref_depth[view.mask.squeeze() < 0.5] = 0
            ref_depth[ref_depth>max_depth] = 0
            ref_depth = ref_depth.detach().cpu().numpy()
            
            pose = np.identity(4)
            pose[:3,:3] = view.R.transpose(-1,-2)
            pose[:3, 3] = view.T
            color = o3d.io.read_image(os.path.join(render_path, view.image_name + ".jpg"))
            depth = o3d.geometry.Image((ref_depth*1000).astype(np.uint16))
            rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
                color, depth, depth_scale=1000.0, depth_trunc=max_depth, convert_rgb_to_intensity=False)
            volume.integrate(
                rgbd,
                o3d.camera.PinholeCameraIntrinsic(W, H, view.Fx, view.Fy, view.Cx, view.Cy),
                pose)

def render_sets(dataset : ModelParams, iteration : int, pipeline : PipelineParams, skip_train : bool, skip_test : bool,
                 max_depth : float, voxel_size : float, num_cluster: int, use_depth_filter : bool, mesh_expname: str, 
                 window_size: float=0.03, transparency_threshold: float=0.15, start_threshold: float=0.0, 
                 end_threshold: float=0.2, use_transparent_depth: bool=False, skip_mesh: bool=False,
                 train_label: str="train", test_label: str="test", replace_with_nearest_depth: bool=False, depth_transparency_blended: bool=False):
    with torch.no_grad():
        gaussians = GaussianModel(dataset.sh_degree)
        scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False)

        bg_color = [1,1,1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
        volume = o3d.pipelines.integration.ScalableTSDFVolume(
            voxel_length=voxel_size,
            sdf_trunc=4.0*voxel_size,
            color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8)

        if not skip_train:
            print(f"processing train set, with {len(scene.getTrainCameras())} views")
            render_set(dataset.model_path, train_label, scene.loaded_iter, scene.getTrainCameras(), scene, gaussians, pipeline, background, 
                       max_depth=max_depth, volume=volume, use_depth_filter=use_depth_filter, window_size=window_size, transparency_threshold=transparency_threshold, start_threshold=start_threshold, end_threshold=end_threshold, use_transparent_depth=use_transparent_depth, replace_with_nearest_depth=replace_with_nearest_depth, depth_transparency_blended=depth_transparency_blended)
            
            if not skip_mesh:
                print(f"extract_triangle_mesh")
                mesh = volume.extract_triangle_mesh()

                path = os.path.join(dataset.model_path, mesh_expname)
                os.makedirs(path, exist_ok=True)
                # NOTE: folloing codes crop mesh using bounding box for clearpose dataset
                # not worked for translab dataset or other datasets
                # find matching scene in model path
                scene_key = None
                for scene_name in clearpose_bounds.keys():
                    if scene_name in dataset.model_path:
                        scene_key = scene_name
                        break
                
                if scene_key:
                    min_bounds = np.array(clearpose_bounds[scene_key]["min"])
                    max_bounds = np.array(clearpose_bounds[scene_key]["max"])
                else:
                    min_bounds = None
                    max_bounds = None
                    
                if min_bounds is not None and max_bounds is not None:
                    bbox = o3d.geometry.AxisAlignedBoundingBox(min_bounds, max_bounds)
                    mesh_cropped = mesh.crop(bbox)
                    o3d.io.write_triangle_mesh(os.path.join(path, f"tsdf_fusion_cropped_{iteration}.ply"), mesh_cropped, 
                                            write_triangle_uvs=True, write_vertex_colors=True, write_vertex_normals=True)
                    os.system(f"cp -f {os.path.join(path, f'tsdf_fusion_cropped_{iteration}.ply')} {os.path.join(path, f'tsdf_fusion_cropped.ply')}")
                    print("mesh saved at", os.path.join(path, f"tsdf_fusion_cropped_{iteration}.ply"))
                    mesh_cropped = post_process_mesh(mesh_cropped, num_cluster)
                    o3d.io.write_triangle_mesh(os.path.join(path, f"tsdf_fusion_cropped_post_{iteration}.ply"), mesh_cropped, 
                                            write_triangle_uvs=True, write_vertex_colors=True, write_vertex_normals=True)
                    os.system(f"cp -f {os.path.join(path, f'tsdf_fusion_cropped_post_{iteration}.ply')} {os.path.join(path, f'tsdf_fusion_cropped_post.ply')}")
                    print("mesh saved at", os.path.join(path, f"tsdf_fusion_cropped_post_{iteration}.ply"))
                
                # NOTE: above codes are for clearpose dataset, not worked for translab dataset or other datasets
                
                o3d.io.write_triangle_mesh(os.path.join(path, f"tsdf_fusion_{iteration}.ply"), mesh, 
                                        write_triangle_uvs=True, write_vertex_colors=True, write_vertex_normals=True)
                os.system(f"cp -f {os.path.join(path, f'tsdf_fusion_{iteration}.ply')} {os.path.join(path, f'tsdf_fusion.ply')}")
                print("mesh saved at", os.path.join(path, f"tsdf_fusion_{iteration}.ply"))
                mesh = post_process_mesh(mesh, num_cluster)
                o3d.io.write_triangle_mesh(os.path.join(path, f"tsdf_fusion_post_{iteration}.ply"), mesh, 
                                        write_triangle_uvs=True, write_vertex_colors=True, write_vertex_normals=True)
                os.system(f"cp -f {os.path.join(path, f'tsdf_fusion_post_{iteration}.ply')} {os.path.join(path, f'tsdf_fusion_post.ply')}")
                print("mesh saved at", os.path.join(path, f"tsdf_fusion_post_{iteration}.ply"))
        if not skip_test:
            render_set(dataset.model_path, test_label, scene.loaded_iter, scene.getTestCameras(), scene, gaussians, pipeline, background, use_transparent_depth=use_transparent_depth, replace_with_nearest_depth=replace_with_nearest_depth)

if __name__ == "__main__":
    torch.set_num_threads(8)
    # Set up command line argument parser
    parser = ArgumentParser(description="Testing script parameters")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_test", action="store_true")
    parser.add_argument("--skip_mesh", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--max_depth", default=5.0, type=float)
    parser.add_argument("--voxel_size", default=0.002, type=float)
    parser.add_argument("--num_cluster", default=1, type=int)
    parser.add_argument("--use_depth_filter", action="store_true")
    parser.add_argument("--use_asg", default=False, action="store_true")
    parser.add_argument("--mesh_expname", type=str, default='mesh')
    parser.add_argument("--window_size", type=float, default=0.03)
    parser.add_argument("--transparency_threshold", type=float, default=0.15)
    parser.add_argument("--start_threshold", type=float, default=0.0)
    parser.add_argument("--end_threshold", type=float, default=0.1)
    parser.add_argument("--use_transparent_depth", type=bool, default=False)
    parser.add_argument("--train_label", type=str, default="train", help="存储训练集渲染结果的文件夹标签")
    parser.add_argument("--test_label", type=str, default="test", help="存储测试集渲染结果的文件夹标签")
    parser.add_argument("--replace_with_nearest_depth", type=bool, default=False)
    parser.add_argument("--depth_transparency_blended", type=bool, default=False)
    args = get_combined_args(parser)
    print("Rendering " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)
    print(f"multi_view_num {model.multi_view_num}")
    render_sets(model.extract(args), args.iteration, pipeline.extract(args), args.skip_train, args.skip_test, 
                args.max_depth, args.voxel_size, args.num_cluster, args.use_depth_filter, args.mesh_expname, 
                args.window_size, args.transparency_threshold, args.start_threshold, args.end_threshold, 
                args.use_transparent_depth, args.skip_mesh, args.train_label, args.test_label, args.replace_with_nearest_depth, args.depth_transparency_blended)