import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--skip_remove', action='store_true', default=True)
parser.add_argument("--skip_training", action="store_true")
parser.add_argument("--skip_rendering", action="store_true")
parser.add_argument("--skip_metrics", action="store_true")
parser.add_argument("--delight", action="store_true")
parser.add_argument("--normal", action="store_true")
parser.add_argument("--out_name", type=str, default='test')
parser.add_argument("--gpu_id", type=int, default=0)
parser.add_argument("--wo_depth_normal_detach", action="store_true")
parser.add_argument("--wo_image_weight", action="store_true")
parser.add_argument("--use_2dgsnormal_loss", action="store_true")
parser.add_argument("--use_asg", action="store_true")
parser.add_argument("--is_real", action="store_true")
parser.add_argument("--is_indoor", action="store_true")
parser.add_argument("--load2gpu_on_the_fly", action="store_true")
parser.add_argument("--asg_degree", type=int, default=24)
parser.add_argument("--sd_normal_until_iter", type=int, default=15000)
parser.add_argument("--iterations", type=int, default=30000)
parser.add_argument("--mask_background", action="store_true", default=True)
parser.add_argument("--mesh_expname", type=str, default='mesh')
parser.add_argument("--normal_cos_threshold_iter", type=int, default=3000)
parser.add_argument("--eval", action="store_true")
parser.add_argument("--delight_iterations", type=int, default=15000)
parser.add_argument("--resolution", "-r", type=int, default=-1) # resolution downsample ratio
parser.add_argument("--render_iteration", type=int, default=-1)
parser.add_argument("--num_cluster", type=int, default=5)
parser.add_argument("--use_delighted_normal", action="store_true")
parser.add_argument("--scene_ids", type=list, default=None)
parser.add_argument("--ncc_loss_from_iter", type=int, default=7000)
parser.add_argument("--nofix_position", action="store_true")
parser.add_argument("--nofix_opacity", action="store_true")
parser.add_argument("--nofix_scaling", action="store_true")
parser.add_argument("--nofix_rotation", action="store_true")
parser.add_argument("--nofix_param", action="store_true")
parser.add_argument("--not_delight_only_transparent", action="store_true")
parser.add_argument("--window_size", type=float, default=0.03)
parser.add_argument("--start_threshold", type=float, default=0.0)
parser.add_argument("--end_threshold", type=float, default=0.2)
parser.add_argument("--transparency_threshold", type=float, default=0.15)
parser.add_argument("--use_transparencies_map", type=bool, default=True)
parser.add_argument("--use_transparent_depth", type=bool, default=True)
parser.add_argument("--clear_f_dc", action="store_true")
parser.add_argument("--clear_f_rest", action="store_true")
parser.add_argument("--clear_opacity", action="store_true")
parser.add_argument("--clear_scaling", action="store_true")
parser.add_argument("--clear_rotation", action="store_true")
parser.add_argument("--train_label", type=str, default="train")
parser.add_argument("--test_label", type=str, default="test")

args = parser.parse_args()

if args.scene_ids is None:
  scenes = [
            "scene_01",
            "scene_02",
            "scene_03",
            "scene_04",
            "scene_05",
            "scene_06",
            "scene_07",
            "scene_08",
            ]
else:
  scenes = args.scene_ids

data_base_path='./data/translab'
out_base_path='output_translab'
eval_path='./data/translab'
out_name=args.out_name
gpu_id=args.gpu_id

for scene in scenes:

    if not args.skip_training:
      cmd = f'cp -rf {data_base_path}/{scene}/sparse/0/* {data_base_path}/{scene}/sparse/'
      print(cmd)
      os.system(cmd)

      common_args = ""
      common_args += " -d" if args.delight else ""
      common_args += " -n" if args.normal else ""
      common_args += " --mask_background" if args.mask_background else ""
      common_args += " --wo_depth_normal_detach" if args.wo_depth_normal_detach else ""
      common_args += " --wo_image_weight" if args.wo_image_weight else ""
      common_args += " --use_2dgsnormal_loss" if args.use_2dgsnormal_loss else ""
      common_args += " --use_asg" if args.use_asg else ""
      common_args += " --is_real" if args.is_real else ""
      common_args += " --is_indoor" if args.is_indoor else ""
      common_args += " --load2gpu_on_the_fly" if args.load2gpu_on_the_fly else ""
      common_args += f" --asg_degree {args.asg_degree}" if args.asg_degree else ""
      common_args += f" --sd_normal_until_iter {args.sd_normal_until_iter}" if args.sd_normal_until_iter else ""
      common_args += f" --iterations {args.iterations}" if args.iterations else ""
      common_args += f" --normal_cos_threshold_iter {args.normal_cos_threshold_iter}" if args.normal_cos_threshold_iter else ""
      common_args += " --eval" if args.eval else ""
      common_args += f" --delight_iterations {args.delight_iterations}"
      common_args += f" --resolution {args.resolution}" if args.resolution else ""
      common_args += " --use_delighted_normal" if args.use_delighted_normal else ""
      common_args += f" --ncc_loss_from_iter {args.ncc_loss_from_iter}" if args.ncc_loss_from_iter else ""
      common_args += " --nofix_position" if args.nofix_position else ""
      common_args += " --nofix_opacity" if args.nofix_opacity else ""
      common_args += " --nofix_param" if args.nofix_param else ""
      common_args += " --nofix_scaling" if args.nofix_scaling else ""
      common_args += " --nofix_rotation" if args.nofix_rotation else ""
      common_args += " --not_delight_only_transparent" if args.not_delight_only_transparent else ""
      
      common_args += " --use_transparencies_map" if args.use_transparencies_map else ""
      common_args += " --clear_f_dc" if args.clear_f_dc else ""
      common_args += " --clear_f_rest" if args.clear_f_rest else ""
      common_args += " --clear_opacity" if args.clear_opacity else ""
      common_args += " --clear_scaling" if args.clear_scaling else ""
      common_args += " --clear_rotation" if args.clear_rotation else ""
      cmd = f'CUDA_VISIBLE_DEVICES={gpu_id} python train.py -s {data_base_path}/{scene} -m {out_base_path}/{scene}/{out_name} {common_args}'
      print(cmd)
      os.system(cmd)

    if not args.skip_rendering:
      common_args = f"--quiet --num_cluster {args.num_cluster} --voxel_size 0.002 --max_depth 10.0"
      common_args += " -d" if args.delight else ""
      common_args += " -n" if args.normal else ""
      common_args += " --mask_background" if args.mask_background else ""
      common_args += " --use_asg" if args.use_asg else ""
      common_args += " --is_real" if args.is_real else ""
      common_args += " --is_indoor" if args.is_indoor else ""
      common_args += " --load2gpu_on_the_fly" if args.load2gpu_on_the_fly else ""
      common_args += f" --asg_degree {args.asg_degree}" if args.asg_degree else ""
      common_args += f" --mesh_expname {args.mesh_expname}"
      common_args += f" --iteration {args.render_iteration}"
      common_args += f" --window_size {args.window_size}"
      common_args += f" --start_threshold {args.start_threshold}"
      common_args += f" --end_threshold {args.end_threshold}"
      common_args += f" --transparency_threshold {args.transparency_threshold}"
      common_args += " --use_transparent_depth True" if args.use_transparent_depth else ""
      cmd = f'CUDA_VISIBLE_DEVICES={gpu_id} python render.py -m {out_base_path}/{scene}/{out_name} {common_args}'
      print(cmd)
      os.system(cmd)

    if not args.skip_metrics:
      cmd = f"CUDA_VISIBLE_DEVICES={gpu_id} python scripts/eval_translab/eval.py " + \
            f"--data {out_base_path}/{scene}/{out_name}/{args.mesh_expname}/tsdf_fusion_post_{args.render_iteration}.ply " + \
          f"--scan {scene} --vis_out_dir {out_base_path}/{scene}/{out_name}/{args.mesh_expname} " + \
          f"--dataset_dir {data_base_path} --mode mesh --downsample_density 0.002" 
      print(cmd)
      os.system(cmd)