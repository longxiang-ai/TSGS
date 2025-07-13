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

from argparse import ArgumentParser, Namespace
import sys
import os

class GroupParams:
    pass

class ParamGroup:
    def __init__(self, parser: ArgumentParser, name : str, fill_none = False):
        group = parser.add_argument_group(name)
        for key, value in vars(self).items():
            shorthand = False
            if key.startswith("_"):
                shorthand = True
                key = key[1:]
            t = type(value)
            value = value if not fill_none else None 
            if shorthand:
                if t == bool:
                    group.add_argument("--" + key, ("-" + key[0:1]), default=value, action="store_true")
                else:
                    group.add_argument("--" + key, ("-" + key[0:1]), default=value, type=t)
            else:
                if t == bool:
                    group.add_argument("--" + key, default=value, action="store_true")
                else:
                    group.add_argument("--" + key, default=value, type=t)

    def extract(self, args):
        group = GroupParams()
        for arg in vars(args).items():
            if arg[0] in vars(self) or ("_" + arg[0]) in vars(self):
                setattr(group, arg[0], arg[1])
        return group

class ModelParams(ParamGroup): 
    def __init__(self, parser, sentinel=False):
        self.sh_degree = 3
        self.asg_degree = 24
        self._source_path = ""
        self._model_path = ""
        self._images = "images"
        self._resolution = -1
        self._white_background = False
        self.data_device = "cuda"
        self.eval = False
        self.preload_img = True
        self.ncc_scale = 1.0
        self.multi_view_num = 8
        self.multi_view_max_angle = 30
        self.multi_view_min_dis = 0.01
        self.multi_view_max_dis = 1.5
        self._delight = False
        self._normal = False
        self.mask_background = False
        self.use_delighted_normal = False
        self.use_transparencies_map = False
        self.not_delight_only_transparent = False
        # asg params
        self.load2gpu_on_the_fly = False
        self.is_real = False
        self.is_indoor = False
        self.add_val = False
        super().__init__(parser, "Loading Parameters", sentinel)

    def extract(self, args):
        g = super().extract(args)
        g.source_path = os.path.abspath(g.source_path)
        return g

class PipelineParams(ParamGroup):
    def __init__(self, parser):
        self.convert_SHs_python = False
        self.compute_cov3D_python = False
        self.debug = False
        super().__init__(parser, "Pipeline Parameters")

class OptimizationParams(ParamGroup):
    def __init__(self, parser):
        self.iterations = 30_000
        self.position_lr_init = 0.00016
        self.position_lr_final = 0.0000016
        self.position_lr_delay_mult = 0.01
        self.position_lr_max_steps = 30_000
        self.specular_lr_max_steps = 30_000
        self.feature_lr = 0.0025
        self.opacity_lr = 0.05
        self.scaling_lr = 0.005
        self.rotation_lr = 0.001
        self.percent_dense = 0.001
        self.lambda_dssim = 0.2
        self.densification_interval = 100
        self.opacity_reset_interval = 3000
        self.densify_from_iter = 500
        self.densify_until_iter = 15_000
        self.densify_grad_threshold = 0.0002
        self.scale_loss_weight = 100.0
        
        self.wo_image_weight = False
        self.single_view_weight = 0.015
        self.single_view_weight_from_iter = 7000

        self.use_virtul_cam = False
        self.virtul_cam_prob = 0.5
        self.use_multi_view_trim = True
        self.multi_view_ncc_weight = 0.15
        self.multi_view_geo_weight = 0.03
        self.multi_view_weight_from_iter = 7000
        self.multi_view_patch_size = 3
        self.multi_view_sample_num = 102400
        self.multi_view_pixel_noise_th = 1.0
        self.wo_use_geo_occ_aware = False

        self.opacity_cull_threshold = 0.005
        self.densify_abs_grad_threshold = 0.0008
        self.abs_split_radii2D_threshold = 20
        self.max_abs_split_points = 50_000
        self.max_all_points = 6000_000
        self.exposure_compensation = False
        self.random_background = False

        self.wo_depth_normal_detach = False
        self.use_2dgsnormal_loss = False
        self.use_asg = False
        self.delight_iterations = 15000
        self.sd_normal_until_iter = 30000 # 30000 for easy dataset, where normal is easy to predict, 15000 for hard dataset where loss might be not accurate
        self.lambda_sd_normal = 0.1
        self.normal_cos_threshold_iter = 3000
        self.ncc_loss_from_iter = 7000
        # freeze optimizer
        self.nofix_position = False
        self.nofix_opacity = False
        self.nofix_param = False
        self.nofix_scaling = False
        self.nofix_rotation = False
        
        # clear optimizer momentum
        self.clear_f_dc = False
        self.clear_f_rest = False
        self.clear_opacity = False
        self.clear_scaling = False
        self.clear_rotation = False
        
        super().__init__(parser, "Optimization Parameters")

def get_combined_args(parser : ArgumentParser):
    cmdlne_string = sys.argv[1:]
    cfgfile_string = "Namespace()"
    args_cmdline = parser.parse_args(cmdlne_string)

    try:
        cfgfilepath = os.path.join(args_cmdline.model_path, "cfg_args")
        print("Looking for config file in", cfgfilepath)
        with open(cfgfilepath) as cfg_file:
            print("Config file found: {}".format(cfgfilepath))
            cfgfile_string = cfg_file.read()
    except TypeError:
        print("Config file not found at")
        pass
    args_cfgfile = eval(cfgfile_string)

    merged_dict = vars(args_cfgfile).copy()
    for k,v in vars(args_cmdline).items():
        if v != None:
            merged_dict[k] = v
    return Namespace(**merged_dict)
