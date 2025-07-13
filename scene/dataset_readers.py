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

import os
import sys
from PIL import Image
from typing import NamedTuple
from scene.colmap_loader import read_extrinsics_text, read_intrinsics_text, qvec2rotmat, \
    read_extrinsics_binary, read_intrinsics_binary, read_points3D_binary, read_points3D_text
from utils.graphics_utils import getWorld2View2, focal2fov, fov2focal
import numpy as np
import json
from pathlib import Path
from plyfile import PlyData, PlyElement
from utils.sh_utils import SH2RGB
from scene.gaussian_model import BasicPointCloud
import tqdm
from scene.utils import load_img_rgb, load_mask_bool
import torch


class CameraInfo(NamedTuple):
    uid: int
    global_id: int
    R: np.array
    T: np.array
    FovY: np.array
    FovX: np.array
    image_path: str
    image_name: str
    width: int
    height: int
    fx: float
    fy: float
    image : Image.Image = None
    normal: np.array = None
    delight: np.array = None
    transparencies_map: np.array = None
    K: np.array = None
    
class SceneInfo(NamedTuple):
    point_cloud: BasicPointCloud
    train_cameras: list
    test_cameras: list
    nerf_normalization: dict
    ply_path: str

def getNerfppNorm(cam_info):
    def get_center_and_diag(cam_centers):
        cam_centers = np.hstack(cam_centers)
        avg_cam_center = np.mean(cam_centers, axis=1, keepdims=True)
        center = avg_cam_center
        dist = np.linalg.norm(cam_centers - center, axis=0, keepdims=True)
        diagonal = np.max(dist)
        return center.flatten(), diagonal

    cam_centers = []

    for cam in cam_info:
        W2C = getWorld2View2(cam.R, cam.T)
        C2W = np.linalg.inv(W2C)
        cam_centers.append(C2W[:3, 3:4])

    center, diagonal = get_center_and_diag(cam_centers)
    radius = diagonal * 1.1

    translate = -center

    return {"translate": translate, "radius": radius}

def load_poses(pose_path, num):
    poses = []
    with open(pose_path, "r") as f:
        lines = f.readlines()
    for i in range(num):
        line = lines[i]
        c2w = np.array(list(map(float, line.split()))).reshape(4, 4)
        c2w[:3,3] = c2w[:3,3] * 10.0
        w2c = np.linalg.inv(c2w)
        w2c = w2c
        poses.append(w2c)
    poses = np.stack(poses, axis=0)
    return poses

def readColmapCameras(cam_extrinsics, cam_intrinsics, images_folder, use_delight=False, use_normal=False, mask_background=True, use_delighted_normal=False, use_transparencies_map=True, not_delight_only_transparent=False):
    cam_infos = []
    for idx, key in tqdm.tqdm(enumerate(cam_extrinsics)):
        # if idx == 5:
        #     print("warning: debug mode, only use 100 cameras")
        #     break
        sys.stdout.write('\r')
        # the exact output you're looking for:
        sys.stdout.write("Reading camera {}/{}".format(idx+1, len(cam_extrinsics)))
        sys.stdout.flush()

        extr = cam_extrinsics[key]
        intr = cam_intrinsics[extr.camera_id]
        if intr.model == "PINHOLE":
            intr_params = intr.params
            K = np.eye(3).astype(np.float32)
            K[0, 0] = intr_params[0].astype(np.float32)
            K[1, 1] = intr_params[1].astype(np.float32)
            K[0, 2] = intr_params[2].astype(np.float32)
            K[1, 2] = intr_params[3].astype(np.float32)
            K = K.astype(np.float32)
        else:
            K = None
        
        height = intr.height
        width = intr.width

        uid = intr.id
        R = np.transpose(qvec2rotmat(extr.qvec))
        T = np.array(extr.tvec)

        if intr.model=="SIMPLE_PINHOLE":
            focal_length_x = intr.params[0]
            FovY = focal2fov(focal_length_x, height)
            FovX = focal2fov(focal_length_x, width)
            focal_length_y = fov2focal(FovY, height)
        elif intr.model=="PINHOLE":
            focal_length_x = intr.params[0]
            focal_length_y = intr.params[1]
            FovY = focal2fov(focal_length_y, height)
            FovX = focal2fov(focal_length_x, width)
        else:
            assert False, "Colmap camera model not handled: only undistorted datasets (PINHOLE or SIMPLE_PINHOLE cameras) supported!"

        image_path = os.path.join(images_folder, os.path.basename(extr.name))
        path, extension = image_path, ".png"
        image_name = os.path.basename(image_path).split(".")[0]
        try:
            image = Image.open(image_path)
        except:
            image = Image.open(image_path.replace("images", "train"))
            print(f"Error: {image_path} not found, use {image_path.replace('images', 'train')} instead")

        use_nomask = False
        if use_nomask:
            str_nomask = "_nomask"
        else:
            str_nomask = ""

        if use_delight:
            delight_path = image_path[:-4].replace("images", "delights"+str_nomask) + "_delight" + extension
            delight_image = Image.open(delight_path)
            delight_image = np.array(delight_image)
            delight_image = delight_image / 255.0
            delight_image = delight_image[:, :, :3]
            delight_image = delight_image.astype(np.float32)
        else:
            delight_image = None

        if use_normal:
            normal_path = image_path[:-4].replace("images", "normals"+str_nomask) + "_normal" + extension
            if use_delighted_normal:
                normal_path = image_path[:-4].replace("images", "delighted_normals"+str_nomask) + "_delighted_normal" + extension
            normal_image = Image.open(normal_path)
            normal_image = np.array(normal_image)
            normal_image = normal_image / 255.0
            normal_image = normal_image[:, :, :3]
            normal_image = normal_image * 2 - 1 # normalize to [-1, 1]
            normal_image = normal_image.astype(np.float32)
        else:
            normal_image = None
        
        if use_transparencies_map:
            try:
                transparencies_map_path = image_path[:-4].replace("images", "transparent_masks") + extension
                transparencies_map_image = Image.open(transparencies_map_path).convert("L")
                threshold = 0.5
                transparencies_map_image = np.array(transparencies_map_image)
                transparencies_map_image = transparencies_map_image / 255.0
                transparencies_map_image = transparencies_map_image.astype(np.float32)
                transparencies_map_image[transparencies_map_image > threshold] = 1.0
                transparencies_map_image[transparencies_map_image <= threshold] = 0.0
            except:
                transparencies_map_path = image_path[:-4].replace("images", "masks") + extension
                transparencies_map_image = Image.open(transparencies_map_path).convert("L")
                threshold = 0.5
                transparencies_map_image = np.array(transparencies_map_image)
                transparencies_map_image = transparencies_map_image / 255.0
                transparencies_map_image = transparencies_map_image.astype(np.float32)
                transparencies_map_image[transparencies_map_image > threshold] = 1.0
                transparencies_map_image[transparencies_map_image <= threshold] = 0.0
        else:
            transparencies_map_image = None
        # mask background
        if mask_background:
            if image.mode == "RGBA":
                # last channel is alpha, mask out the background
                mask = image.getchannel(3)
                mask = np.array(mask)
                mask = mask / 255.0
                mask = mask.astype(np.float32)
            else:
                mask_path = image_path[:-4].replace("images", "masks") + extension
                mask = Image.open(mask_path).convert("L")
                mask = np.array(mask)
                mask = mask / 255.0
                mask[mask > 0.5] = 1
                mask[mask <= 0.5] = 0
                mask = mask.astype(np.float32)
            delight_image = delight_image * mask[..., None] if delight_image is not None else None
            normal_image = normal_image * mask[..., None] if normal_image is not None else None
            transparencies_map_image = transparencies_map_image * mask if transparencies_map_image is not None else None
            # 将图像转换为numpy数组
            image_array = np.array(image)
            # 只保留RGB通道并应用mask
            image_array = image_array[:, :, :3] * mask[..., None]

            if transparencies_map_image is not None and delight_image is not None and not not_delight_only_transparent:
                # Transparent部分使用delight，其余部分使用image
                delight_image = delight_image * transparencies_map_image[..., None] + image_array / 255.0 * (1.0 - transparencies_map_image[..., None])
                
            # 转回PIL图像,保持RGB格式
            image = Image.fromarray(image_array.astype(np.uint8), "RGB")
            # 转换为RGBA格式,将mask作为alpha通道
            r, g, b = image.split()
            alpha = Image.fromarray((mask * 255).astype(np.uint8), "L")
            image = Image.merge("RGBA", (r, g, b, alpha))

        cam_info = CameraInfo(uid=uid, global_id=idx, R=R, T=T, FovY=FovY, FovX=FovX,
                              image_path=image_path, image_name=image_name, 
                              width=width, height=height, fx=focal_length_x, fy=focal_length_y,
                              image=image, delight=delight_image, normal=normal_image, transparencies_map=transparencies_map_image, K=K)
        cam_infos.append(cam_info)
        # break
    sys.stdout.write('\n')
    return cam_infos

def fetchPly(path):
    plydata = PlyData.read(path)
    vertices = plydata['vertex']
    positions = np.vstack([vertices['x'], vertices['y'], vertices['z']]).T
    colors = np.vstack([vertices['red'], vertices['green'], vertices['blue']]).T / 255.0
    normals = np.vstack([vertices['nx'], vertices['ny'], vertices['nz']]).T
    return BasicPointCloud(points=positions, colors=colors, normals=normals)

def storePly(path, xyz, rgb, normals=None):
    # Define the dtype for the structured array
    dtype = [('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
            ('nx', 'f4'), ('ny', 'f4'), ('nz', 'f4'),
            ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')]
    
    if normals is None:
        normals = np.zeros_like(xyz)

    elements = np.empty(xyz.shape[0], dtype=dtype)
    attributes = np.concatenate((xyz, normals, rgb), axis=1)
    elements[:] = list(map(tuple, attributes))

    # Create the PlyData object and write to file
    vertex_element = PlyElement.describe(elements, 'vertex')
    ply_data = PlyData([vertex_element])
    ply_data.write(path)

def readColmapSceneInfo(path, images, eval, llffhold=8, use_delight=False, use_normal=False, mask_background=True, use_delighted_normal=False, use_transparencies_map=True, not_delight_only_transparent=False):
    try:
        cameras_extrinsic_file = os.path.join(path, "sparse", "images.txt")
        cameras_intrinsic_file = os.path.join(path, "sparse", "cameras.txt")
        cam_extrinsics = read_extrinsics_text(cameras_extrinsic_file)
        cam_intrinsics = read_intrinsics_text(cameras_intrinsic_file)
    except:
        cameras_extrinsic_file = os.path.join(path, "sparse", "images.bin")
        cameras_intrinsic_file = os.path.join(path, "sparse", "cameras.bin")
        cam_extrinsics = read_extrinsics_binary(cameras_extrinsic_file)
        cam_intrinsics = read_intrinsics_binary(cameras_intrinsic_file)
    reading_dir = "images" if images == None else images
    print(f"Reading Colmap Scene Info, use_delight: {use_delight}, use_normal: {use_normal}")
    cam_infos_unsorted = readColmapCameras(cam_extrinsics=cam_extrinsics, cam_intrinsics=cam_intrinsics, images_folder=os.path.join(path, reading_dir), use_delight=use_delight, use_normal=use_normal, mask_background=mask_background, use_delighted_normal=use_delighted_normal, use_transparencies_map=use_transparencies_map, not_delight_only_transparent=not_delight_only_transparent)
    # cam_infos = sorted(cam_infos_unsorted.copy(), key = lambda x : int(x.image_name.split('_')[-1]))
    cam_infos = sorted(cam_infos_unsorted.copy(), key = lambda x : x.image_name)
    
    js_file = f"{path}/split.json"
    train_list = None
    test_list = None
    if os.path.exists(js_file):
        with open(js_file) as file:
            meta = json.load(file)
            train_list = meta["train"]
            test_list = meta["test"]
            print(f"train_list {len(train_list)}, test_list {len(test_list)}")

    if train_list is not None:
        train_cam_infos = [c for idx, c in enumerate(cam_infos) if c.image_name in train_list]
        test_cam_infos = [c for idx, c in enumerate(cam_infos) if c.image_name in test_list]
        print(f"train_cam_infos {len(train_cam_infos)}, test_cam_infos {len(test_cam_infos)}")
    elif eval:
        train_cam_infos = [c for idx, c in enumerate(cam_infos) if idx % llffhold != 0]
        test_cam_infos = [c for idx, c in enumerate(cam_infos) if idx % llffhold == 0]
    else:
        train_cam_infos = cam_infos
        test_cam_infos = []

    nerf_normalization = getNerfppNorm(train_cam_infos)

    ply_path = os.path.join(path, "sparse/points3D.ply")
    bin_path = os.path.join(path, "sparse/points3D.bin")
    txt_path = os.path.join(path, "sparse/points3D.txt")
    if not os.path.exists(ply_path) or True:
        print("Converting point3d.bin to .ply, will happen only the first time you open the scene.")
        try:
            xyz, rgb, _ = read_points3D_binary(bin_path)
            print(f"xyz {xyz.shape}")
        except:
            xyz, rgb, _ = read_points3D_text(txt_path)
        storePly(ply_path, xyz, rgb)
    try:
        pcd = fetchPly(ply_path)
    except:
        pcd = None

    scene_info = SceneInfo(point_cloud=pcd,
                           train_cameras=train_cam_infos,
                           test_cameras=test_cam_infos,
                           nerf_normalization=nerf_normalization,
                           ply_path=ply_path)
    return scene_info
def readCamerasFromTransforms3(path, transformsfile, white_background, extension=".png", debug=False):
    cam_infos = []

    with open(os.path.join(path, transformsfile)) as json_file:
        contents = json.load(json_file)
        fovx = contents["camera_angle_x"]

        frames = contents["frames"]
        for idx, frame in enumerate(tqdm(frames, leave=False)):
            image_path = os.path.join(path, frame["file_path"] + extension)
            mask_path = image_path.replace("_rgb.exr", "_mask.png")
            image_name = Path(image_path).stem

            # NeRF 'transform_matrix' is a camera-to-world transform
            c2w = np.array(frame["transform_matrix"])
            # change from OpenGL/Blender camera axes (Y up, Z back) to COLMAP (Y down, Z forward)
            c2w[:3, 1:3] *= -1

            # get the world-to-camera transform and set R, T
            w2c = np.linalg.inv(c2w)
            R = np.transpose(w2c[:3, :3])  # R is stored transposed due to 'glm' in CUDA code
            T = w2c[:3, 3]

            bg = 1 if white_background else 0
            
            image = load_img_rgb(image_path)
            mask = load_mask_bool(mask_path).astype(np.float32)

            bg = np.array([1, 1, 1]) if white_background else np.array([0, 0, 0])
            image = image[..., :3] * mask[..., None] + bg * (1 - mask[..., None])

            # concatenate the image and the mask
            image = np.concatenate([image, mask[..., None]], axis=-1)


            fovy = focal2fov(fov2focal(fovx, image.shape[0]), image.shape[1])
            
            # using PIL to load the image
            image_PIL = Image.fromarray((image * 255).astype(np.uint8), "RGBA")
            cam_infos.append(CameraInfo(uid=idx, global_id=idx, R=R, T=T, FovY=fovy, FovX=fovx, fx=fov2focal(fovx, image.shape[1]), fy=fov2focal(fovy, image.shape[0]),image=image_PIL,
                                        image_path=image_path, image_name=image_name,
                                        width=image.shape[1], height=image.shape[0]))

            if debug and idx >= 5:
                break

    return cam_infos

def readSynthetic4RelightInfo(path, white_background, eval, debug=False):
    print("Reading Training Transforms")
    train_cam_infos = readCamerasFromTransforms3(path, "transforms_train.json", white_background, "_rgb.exr", debug=debug)
    if eval:
        print("Reading Test Transforms")
        test_cam_infos = readCamerasFromTransforms3(path, "transforms_test.json", white_background, "_rgba.png", debug=debug)
    else:
        test_cam_infos = []

    nerf_normalization = getNerfppNorm(train_cam_infos)

    ply_path = os.path.join(path, "points3d.ply")
    if not os.path.exists(ply_path):
        # Since this data set has no colmap data, we start with random points
        num_pts = 100_000
        print(f"Generating random point cloud ({num_pts})...")

        # We create random points inside the bounds of the synthetic Blender scenes
        xyz = np.random.random((num_pts, 3)) * 2.6 - 1.3
        shs = np.random.random((num_pts, 3)) / 255.0
        normals = np.random.randn(*xyz.shape)
        normals /= np.linalg.norm(normals, axis=-1, keepdims=True)

        storePly(ply_path, xyz, SH2RGB(shs) * 255, normals)

    try:
        pcd = fetchPly(ply_path)
    except:
        pcd = None

    

    scene_info = SceneInfo(point_cloud=pcd,
                           train_cameras=train_cam_infos,
                           test_cameras=test_cam_infos,
                           nerf_normalization=nerf_normalization,
                           ply_path=ply_path)

    return scene_info

def readCamerasFromTransforms(path, transformsfile, white_background, extension=".png"):
    cam_infos = []

    with open(os.path.join(path, transformsfile)) as json_file:
        contents = json.load(json_file)
        fovx = contents["camera_angle_x"]

        frames = contents["frames"]
        for idx, frame in tqdm.tqdm(enumerate(frames)):
            cam_name = os.path.join(path, frame["file_path"] + extension)

            # NeRF 'transform_matrix' is a camera-to-world transform
            c2w = np.array(frame["transform_matrix"])
            # change from OpenGL/Blender camera axes (Y up, Z back) to COLMAP (Y down, Z forward)
            c2w[:3, 1:3] *= -1

            # get the world-to-camera transform and set R, T
            w2c = np.linalg.inv(c2w)
            R = np.transpose(w2c[:3,:3])  # R is stored transposed due to 'glm' in CUDA code
            T = w2c[:3, 3]

            image_path = os.path.join(path, cam_name)
            image_name = Path(cam_name).stem
            image = Image.open(image_path)

            im_data = np.array(image.convert("RGBA"))

            bg = np.array([1,1,1]) if white_background else np.array([0, 0, 0])

            norm_data = im_data / 255.0
            arr = norm_data[:,:,:3] * norm_data[:, :, 3:4] + bg * (1 - norm_data[:, :, 3:4])
            image = Image.fromarray(np.array(arr*255.0, dtype=np.byte), "RGB")

            fovy = focal2fov(fov2focal(fovx, image.size[0]), image.size[1])
            FovY = fovy 
            FovX = fovx

            cam_infos.append(CameraInfo(uid=idx, global_id=idx, R=R, T=T, FovY=FovY, FovX=FovX, image=image,
                            image_path=image_path, image_name=image_name, width=image.size[0], height=image.size[1], fx=fov2focal(fovx, image.size[0]), fy=fov2focal(fovy, image.size[1])))
            
    return cam_infos

def readNerfSyntheticInfo(path, white_background, eval, extension=".png"):
    print("Reading Training Transforms")
    train_cam_infos = readCamerasFromTransforms(path, "transforms_train.json", white_background, extension)
    print("Reading Test Transforms")
    test_cam_infos = readCamerasFromTransforms(path, "transforms_test.json", white_background, extension)
    if not eval:
        train_cam_infos.extend(test_cam_infos)
        test_cam_infos = []

    nerf_normalization = getNerfppNorm(train_cam_infos)

    ply_path = os.path.join(path, "points3d.ply")
    if not os.path.exists(ply_path):
        # Since this data set has no colmap data, we start with random points
        num_pts = 100_000
        print(f"Generating random point cloud ({num_pts})...")
        
        # We create random points inside the bounds of the synthetic Blender scenes
        xyz = np.random.random((num_pts, 3)) * 2.6 - 1.3
        shs = np.random.random((num_pts, 3)) / 255.0
        pcd = BasicPointCloud(points=xyz, colors=SH2RGB(shs), normals=np.zeros((num_pts, 3)))

        storePly(ply_path, xyz, SH2RGB(shs) * 255)
    try:
        pcd = fetchPly(ply_path)
    except:
        pcd = None

    scene_info = SceneInfo(point_cloud=pcd,
                           train_cameras=train_cam_infos,
                           test_cameras=test_cam_infos,
                           nerf_normalization=nerf_normalization,
                           ply_path=ply_path)
    return scene_info

sceneLoadTypeCallbacks = {
    "Colmap": readColmapSceneInfo,
    "Blender" : readNerfSyntheticInfo,
    "Synthetic4Relight" : readSynthetic4RelightInfo
}