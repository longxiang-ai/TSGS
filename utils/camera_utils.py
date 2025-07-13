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

from scene.cameras import Camera, NonCenteredCamera
import numpy as np
from utils.graphics_utils import fov2focal
import sys
import torch
WARNED = False

@torch.no_grad()
def loadCam(args, id, cam_info, resolution_scale):
    orig_w, orig_h = cam_info.width, cam_info.height
    if args.resolution in [1, 2, 4, 8]:
        resolution = round(orig_w/(resolution_scale * args.resolution)), round(orig_h/(resolution_scale * args.resolution))
    else:  # should be a type that converts to float
        if args.resolution == -1:
            if orig_w > 1600:
                global_down = orig_w / 1600
                global WARNED
                if not WARNED:
                    print("[ INFO ] Encountered quite large input images (>1.6K pixels width), rescaling to 1.6K.\n "
                        "If this is not desired, please explicitly specify '--resolution/-r' as 1")
                    print(f"scale {float(global_down) * float(resolution_scale)}")
                    WARNED = True
            else:
                global_down = 1
        else:
            global_down = orig_w / args.resolution

        scale = float(global_down) * float(resolution_scale)
        resolution = (int(orig_w / scale), int(orig_h / scale))

    sys.stdout.write('\r')
    sys.stdout.write("load camera {}".format(id))
    sys.stdout.flush()

    if cam_info.K is not None:
        return NonCenteredCamera(colmap_id=cam_info.uid, R=cam_info.R, T=cam_info.T, 
                  K=cam_info.K,
                  image_width=resolution[0], image_height=resolution[1],
                  image_path=cam_info.image_path,
                  image_PIL=cam_info.image,
                  image_name=cam_info.image_name, uid=cam_info.global_id, 
                  preload_img=args.preload_img, 
                  ncc_scale=args.ncc_scale,
                  data_device=args.data_device,
                  delight=cam_info.delight,
                  normal=cam_info.normal,
                  transparencies_map=cam_info.transparencies_map)
        
    return Camera(colmap_id=cam_info.uid, R=cam_info.R, T=cam_info.T, 
                  FoVx=cam_info.FovX, FoVy=cam_info.FovY,
                  image_width=resolution[0], image_height=resolution[1],
                  image_path=cam_info.image_path,
                  image_PIL=cam_info.image,
                  image_name=cam_info.image_name, uid=cam_info.global_id, 
                  preload_img=args.preload_img, 
                  ncc_scale=args.ncc_scale,
                  data_device=args.data_device,
                  delight=cam_info.delight,
                  normal=cam_info.normal,
                  transparencies_map=cam_info.transparencies_map)

@torch.no_grad()
def cameraList_from_camInfos(cam_infos, resolution_scale, args):
    camera_list = []

    for id, c in enumerate(cam_infos):
        camera_list.append(loadCam(args, id, c, resolution_scale))
    return camera_list

def camera_to_JSON(id, camera : Camera):
    Rt = np.zeros((4, 4))
    Rt[:3, :3] = camera.R.transpose()
    Rt[:3, 3] = camera.T
    Rt[3, 3] = 1.0

    W2C = np.linalg.inv(Rt)
    pos = W2C[:3, 3]
    rot = W2C[:3, :3]
    serializable_array_2d = [x.tolist() for x in rot]
    camera_entry = {
        'id' : id,
        'img_name' : camera.image_name,
        'width' : camera.width,
        'height' : camera.height,
        'position': pos.tolist(),
        'rotation': serializable_array_2d,
        'fy' : fov2focal(camera.FovY, camera.height),
        'fx' : fov2focal(camera.FovX, camera.width)
    }
    return camera_entry

def gen_virtul_cam(cam, trans_noise=1.0, deg_noise=15.0):
    """
    生成虚拟相机，通过对输入相机进行随机平移和旋转扰动
    
    参数:
        cam: 输入相机对象
        trans_noise: 平移扰动的最大范围（米）
        deg_noise: 旋转扰动的最大角度（度）
    """
    # 构建4x4的相机外参矩阵Rt (world to camera)
    Rt = np.zeros((4, 4))
    Rt[:3, :3] = cam.R.transpose()  # 旋转矩阵
    Rt[:3, 3] = cam.T              # 平移向量
    Rt[3, 3] = 1.0                 # 齐次坐标

    # 计算相机到世界坐标系的变换矩阵 (camera to world)
    C2W = np.linalg.inv(Rt)

    # 生成随机平移扰动（在[-trans_noise, trans_noise]范围内）
    translation_perturbation = np.random.uniform(-trans_noise, trans_noise, 3)
    
    # 生成随机旋转扰动（在[-deg_noise, deg_noise]度范围内）
    rotation_perturbation = np.random.uniform(-deg_noise, deg_noise, 3)
    # 将角度转换为弧度
    rx, ry, rz = np.deg2rad(rotation_perturbation)

    # 构建绕X轴旋转矩阵
    Rx = np.array([[1, 0, 0],
                    [0, np.cos(rx), -np.sin(rx)],
                    [0, np.sin(rx), np.cos(rx)]])
    
    # 构建绕Y轴旋转矩阵
    Ry = np.array([[np.cos(ry), 0, np.sin(ry)],
                    [0, 1, 0],
                    [-np.sin(ry), 0, np.cos(ry)]])
    
    # 构建绕Z轴旋转矩阵
    Rz = np.array([[np.cos(rz), -np.sin(rz), 0],
                    [np.sin(rz), np.cos(rz), 0],
                    [0, 0, 1]])
    
    # 合成总的旋转扰动矩阵（按Z->Y->X顺序旋转）
    R_perturbation = Rz @ Ry @ Rx

    # 对相机姿态应用旋转和平移扰动
    C2W[:3, :3] = C2W[:3, :3] @ R_perturbation  # 应用旋转扰动
    C2W[:3, 3] = C2W[:3, 3] + translation_perturbation  # 应用平移扰动
    
    # 转换回world to camera矩阵
    Rt = np.linalg.inv(C2W)

    # 创建新的虚拟相机对象
    virtul_cam = Camera(
        100000,                     # 相机ID
        Rt[:3, :3].transpose(),     # 旋转矩阵
        Rt[:3, 3],                  # 平移向量
        cam.FoVx, cam.FoVy,         # 视场角
        cam.image_width, cam.image_height,  # 图像尺寸
        cam.image_path, cam.image_name,     # 图像路径和名称
        100000,                     # 时间戳
        trans=np.array([0.0, 0.0, 0.0]),   # 额外平移
        scale=1.0,                  # 缩放因子
        preload_img=False,          # 是否预加载图像
        data_device="cuda"          # 数据所在设备
    )
    
    return virtul_cam