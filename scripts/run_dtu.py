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
parser.add_argument("--use_asg", action="store_true")
parser.add_argument("--mask_background", action="store_true")
parser.add_argument("--eval", action="store_true")
parser.add_argument("--sd_normal_until_iter", type=int, default=-1)
parser.add_argument("--nofix_param", action="store_true")
parser.add_argument("--nofix_position", action="store_true")
parser.add_argument("--nofix_scaling", action="store_true")
parser.add_argument("--nofix_rotation", action="store_true")
parser.add_argument("--normal_cos_threshold_iter", type=int, default=3000)
parser.add_argument("--ncc_loss_from_iter", type=int, default=7000)
parser.add_argument("--iterations", type=int, default=30000)
parser.add_argument("--resolution", "-r", type=int, default=2)
parser.add_argument("--render_iteration", type=int, default=30000)
parser.add_argument("--num_cluster", type=int, default=1)
parser.add_argument("--mesh_expname", type=str, default='mesh')
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--train_label", type=str, default="train")
parser.add_argument("--test_label", type=str, default="test")
parser.add_argument("--delight_iterations", type=int, default=15000)
parser.add_argument("--normal_folder", type=str, default="normals")
parser.add_argument("--window_size", type=float, default=0.03)
parser.add_argument("--start_threshold", type=float, default=0.0)
parser.add_argument("--end_threshold", type=float, default=0.2)
parser.add_argument("--max_depth", type=float, default=5.0)
parser.add_argument("--voxel_size", type=float, default=0.002)
parser.add_argument("--scene_ids", type=int, nargs='+', default=None)
args = parser.parse_args()

if args.scene_ids is not None:
    scenes = args.scene_ids
else:
    scenes = [24, 37, 40, 55, 63, 65, 69, 83, 97, 105, 106, 110, 114, 118, 122]
data_base_path='./data/dtu_dataset/dtu'
out_base_path='output_dtu'
eval_path='./data/dtu_dataset/dtu_eval'
out_name=args.out_name
gpu_id=args.gpu_id

for scene in scenes:

    if not args.skip_training:
      cmd = f'cp -rf {data_base_path}/scan{scene}/sparse/0/* {data_base_path}/scan{scene}/sparse/'
      print(cmd)
      os.system(cmd)

      common_args = f"--quiet -r {args.resolution} --ncc_scale 0.5"
      common_args += " -d" if args.delight else ""
      common_args += " -n" if args.normal else ""
      common_args += " --use_asg" if args.use_asg else ""
      common_args += " --eval" if args.eval else ""
      common_args += f" --iterations {args.iterations}"
      common_args += f" --sd_normal_until_iter {args.sd_normal_until_iter}" if args.sd_normal_until_iter != -1 else ""
      common_args += f" --normal_cos_threshold_iter {args.normal_cos_threshold_iter}"
      common_args += f" --ncc_loss_from_iter {args.ncc_loss_from_iter}"
      common_args += f" --delight_iterations {args.delight_iterations}"
      common_args += f" --seed {args.seed}"
      common_args += f" --normal_folder {args.normal_folder}"
      common_args += " --nofix_param" if args.nofix_param else ""
      common_args += " --nofix_position" if args.nofix_position else ""
      common_args += " --nofix_scaling" if args.nofix_scaling else ""
      common_args += " --nofix_rotation" if args.nofix_rotation else ""
      common_args += " --mask_background" if args.mask_background else ""
      cmd = f'CUDA_VISIBLE_DEVICES={gpu_id} python train.py -s {data_base_path}/scan{scene} -m {out_base_path}/dtu_scan{scene}/{out_name} {common_args}'
      print(cmd)
      os.system(cmd)

    if not args.skip_rendering:
      common_args = f"--quiet --num_cluster {args.num_cluster} --voxel_size {args.voxel_size} --max_depth {args.max_depth}"
      common_args += " -d" if args.delight else ""
      common_args += " -n" if args.normal else ""
      common_args += " --use_asg" if args.use_asg else ""
      common_args += f" --mesh_expname {args.mesh_expname}"
      common_args += f" --iteration {args.render_iteration}"
      common_args += f" --window_size {args.window_size}"
      common_args += f" --start_threshold {args.start_threshold}"
      common_args += f" --end_threshold {args.end_threshold}"
      common_args += f" --train_label {args.train_label}"
      common_args += f" --test_label {args.test_label}"
      cmd = f'CUDA_VISIBLE_DEVICES={gpu_id} python render.py -m {out_base_path}/dtu_scan{scene}/{out_name} {common_args}'
      print(cmd)
      os.system(cmd)

    if not args.skip_metrics:
      mesh_name = f"tsdf_fusion_post_{args.render_iteration}.ply" if args.render_iteration != -1 else "tsdf_fusion_post.ply"
      cmd = f"CUDA_VISIBLE_DEVICES={gpu_id} python scripts/eval_dtu/evaluate_single_scene.py " + \
            f"--input_mesh {out_base_path}/dtu_scan{scene}/{out_name}/{args.mesh_expname}/{mesh_name} " + \
          f"--scan_id {scene} --output_dir {out_base_path}/dtu_scan{scene}/{out_name}/{args.mesh_expname} " + \
          f"--mask_dir {data_base_path} " + \
          f"--DTU {eval_path}"
      print(cmd)
      os.system(cmd)