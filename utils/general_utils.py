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
import sys
from datetime import datetime
import numpy as np
import random

def inverse_sigmoid(x):
    return torch.log(x/(1-x))

def PILtoTorch(pil_image, resolution=None):
    if resolution is None:
        resized_image_PIL = pil_image
    else:
        resized_image_PIL = pil_image.resize(resolution)
    resized_image = torch.from_numpy(np.array(resized_image_PIL)) / 255.0
    if len(resized_image.shape) == 3:
        return resized_image.permute(2, 0, 1)
    else:
        return resized_image.unsqueeze(dim=-1).permute(2, 0, 1)

def NumpyToTorch(numpy_image, resolution):
    """
    Convert a numpy image to a torch image, and resize it to the given resolution
    H, W, C -> C, h, w
    resolution is (w, h)
    """

    resized_image = torch.tensor(numpy_image, dtype=torch.float32, device="cuda") # [H, W, C]
    # 如果shape不同，则需要interpolate
    if len(resized_image.shape) == 3:
        # 需要先转换为[1, C, H, W]格式才能用interpolate
        resized_image = resized_image.permute(2, 0, 1).unsqueeze(0) # [1, C, H, W]
        resized_image = torch.nn.functional.interpolate(resized_image, size=(resolution[1], resolution[0]), mode='bilinear') # [1, C, h, w]
        return resized_image.squeeze(0) # [C, h, w]
    else:
        # 单通道图像
        resized_image = resized_image.unsqueeze(0).unsqueeze(0) # [1, 1, H, W]
        resized_image = torch.nn.functional.interpolate(resized_image, size=(resolution[1], resolution[0]), mode='bilinear') # [1, 1, h, w]
        return resized_image.squeeze(0) # [1, h, w]
    
def get_expon_lr_func(
    lr_init, lr_final, lr_delay_steps=0, lr_delay_mult=1.0, max_steps=1000000,
    fix_step=None, fix_lr=None
):
    """
    Copied from Plenoxels

    Continuous learning rate decay function. Adapted from JaxNeRF
    The returned rate is lr_init when step=0 and lr_final when step=max_steps, and
    is log-linearly interpolated elsewhere (equivalent to exponential decay).
    If lr_delay_steps>0 then the learning rate will be scaled by some smooth
    function of lr_delay_mult, such that the initial learning rate is
    lr_init*lr_delay_mult at the beginning of optimization but will be eased back
    to the normal learning rate when steps>lr_delay_steps.
    
    If fix_step is provided, the learning rate will be fixed to fix_lr after step > fix_step.
    If fix_lr is not provided, it will use lr_final as the fixed learning rate.
    
    :param conf: config subtree 'lr' or similar
    :param max_steps: int, the number of steps during optimization.
    :param fix_step: int, the step after which learning rate will be fixed
    :param fix_lr: float, the fixed learning rate to use after fix_step
    :return HoF which takes step as input
    """

    def helper(step):
        if step < 0 or (lr_init == 0.0 and lr_final == 0.0):
            # Disable this parameter
            return 0.0
            
        # 如果设置了固定学习率的步数，并且当前步数超过了该值，则返回固定学习率
        if fix_step is not None and step > fix_step:
            return fix_lr if fix_lr is not None else lr_final
            
        if lr_delay_steps > 0:
            # A kind of reverse cosine decay.
            delay_rate = lr_delay_mult + (1 - lr_delay_mult) * np.sin(
                0.5 * np.pi * np.clip(step / lr_delay_steps, 0, 1)
            )
        else:
            delay_rate = 1.0
        t = np.clip(step / max_steps, 0, 1)
        log_lerp = np.exp(np.log(lr_init) * (1 - t) + np.log(lr_final) * t)
        return delay_rate * log_lerp
    return helper

def get_linear_noise_func(
        lr_init, lr_final, lr_delay_steps=0, lr_delay_mult=1.0, max_steps=1000000,
        fix_step=None, fix_lr=None
):
    """
    Copied from Plenoxels

    Continuous learning rate decay function. Adapted from JaxNeRF
    The returned rate is lr_init when step=0 and lr_final when step=max_steps, and
    is log-linearly interpolated elsewhere (equivalent to exponential decay).
    If lr_delay_steps>0 then the learning rate will be scaled by some smooth
    function of lr_delay_mult, such that the initial learning rate is
    lr_init*lr_delay_mult at the beginning of optimization but will be eased back
    to the normal learning rate when steps>lr_delay_steps.
    
    If fix_step is provided, the learning rate will be fixed to fix_lr after step > fix_step.
    If fix_lr is not provided, it will use lr_final as the fixed learning rate.
    
    :param conf: config subtree 'lr' or similar
    :param max_steps: int, the number of steps during optimization.
    :param fix_step: int, the step after which learning rate will be fixed
    :param fix_lr: float, the fixed learning rate to use after fix_step
    :return HoF which takes step as input
    """

    def helper(step):
        if step < 0 or (lr_init == 0.0 and lr_final == 0.0):
            # Disable this parameter
            return 0.0
            
        # 如果设置了固定学习率的步数，并且当前步数超过了该值，则返回固定学习率
        if fix_step is not None and step > fix_step:
            return fix_lr if fix_lr is not None else lr_final
            
        if lr_delay_steps > 0:
            # A kind of reverse cosine decay.
            delay_rate = lr_delay_mult + (1 - lr_delay_mult) * np.sin(
                0.5 * np.pi * np.clip(step / lr_delay_steps, 0, 1)
            )
        else:
            delay_rate = 1.0
        t = np.clip(step / max_steps, 0, 1)
        log_lerp = lr_init * (1 - t) + lr_final * t
        return delay_rate * log_lerp

    return helper

def flip_align_view(normal, viewdir):
    """
    将法向量与视角方向对齐。如果法向量与视角方向夹角大于90度,则翻转法向量方向。
    
    Args:
        normal: 形状为(N, 3)的法向量张量
        viewdir: 形状为(N, 3)的视角方向张量
        
    Returns:
        normal_flipped: 形状为(N, 3)的对齐后的法向量张量
        non_flip: 形状为(N, 1)的布尔张量,表示每个法向量是否需要翻转
    """
    # normal: (N, 3), viewdir: (N, 3)
    dotprod = torch.sum(
        normal * -viewdir, dim=-1, keepdims=True)  # (N, 1)
    non_flip = dotprod >= 0  # (N, 1)
    normal_flipped = normal * torch.where(non_flip, 1, -1)  # (N, 3)
    return normal_flipped, non_flip

def strip_lowerdiag(L):
    uncertainty = torch.zeros((L.shape[0], 6), dtype=torch.float, device="cuda")

    uncertainty[:, 0] = L[:, 0, 0]
    uncertainty[:, 1] = L[:, 0, 1]
    uncertainty[:, 2] = L[:, 0, 2]
    uncertainty[:, 3] = L[:, 1, 1]
    uncertainty[:, 4] = L[:, 1, 2]
    uncertainty[:, 5] = L[:, 2, 2]
    return uncertainty

def strip_symmetric(sym):
    return strip_lowerdiag(sym)

def build_rotation(r):
    norm = torch.sqrt(r[:,0]*r[:,0] + r[:,1]*r[:,1] + r[:,2]*r[:,2] + r[:,3]*r[:,3])

    q = r / norm[:, None]

    R = torch.zeros((q.size(0), 3, 3), device='cuda')

    r = q[:, 0]
    x = q[:, 1]
    y = q[:, 2]
    z = q[:, 3]

    R[:, 0, 0] = 1 - 2 * (y*y + z*z)
    R[:, 0, 1] = 2 * (x*y - r*z)
    R[:, 0, 2] = 2 * (x*z + r*y)
    R[:, 1, 0] = 2 * (x*y + r*z)
    R[:, 1, 1] = 1 - 2 * (x*x + z*z)
    R[:, 1, 2] = 2 * (y*z - r*x)
    R[:, 2, 0] = 2 * (x*z - r*y)
    R[:, 2, 1] = 2 * (y*z + r*x)
    R[:, 2, 2] = 1 - 2 * (x*x + y*y)
    return R

def build_scaling(s):
    L = torch.zeros((s.shape[0], 3, 3), dtype=torch.float, device="cuda")
    L[:,0,0] = s[:,0]
    L[:,1,1] = s[:,1]
    L[:,2,2] = s[:,2]
    return L

def build_scaling_rotation(s, r):
    L = torch.zeros((s.shape[0], 3, 3), dtype=torch.float, device="cuda")
    R = build_rotation(r)

    L[:,0,0] = s[:,0]
    L[:,1,1] = s[:,1]
    L[:,2,2] = s[:,2]

    L = R @ L
    return L

def safe_state(silent):
    old_f = sys.stdout
    class F:
        def __init__(self, silent):
            self.silent = silent

        def write(self, x):
            if not self.silent:
                if x.endswith("\n"):
                    old_f.write(x.replace("\n", " [{}]\n".format(str(datetime.now().strftime("%d/%m %H:%M:%S")))))
                else:
                    old_f.write(x)

        def flush(self):
            old_f.flush()

    sys.stdout = F(silent)

    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    torch.cuda.set_device(torch.device("cuda:0"))
