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
parser.add_argument("--eval", action="store_true")
parser.add_argument("--sd_normal_until_iter", type=int, default=-1)
parser.add_argument("--nofix_param", action="store_true")
args = parser.parse_args()

scenes = [24, 37, 40, 55, 63, 65, 69, 83, 97, 105, 106, 110, 114, 118, 122]
data_base_path='./data/dtu_dataset/dtu'
out_base_path='output_dtu'
eval_path='./data/dtu_dataset/dtu_eval'
out_name=args.out_name
gpu_id=args.gpu_id

for scene in scenes:
#     if not args.skip_remove:
#       cmd = f'rm -rf {out_base_path}/dtu_scan{scene}/{out_name}/*'
#       print(cmd)
#       os.system(cmd)

    if not args.skip_training:
      cmd = f'cp -rf {data_base_path}/scan{scene}/sparse/0/* {data_base_path}/scan{scene}/sparse/'
      print(cmd)
      os.system(cmd)

      common_args = "--quiet -r 2 --ncc_scale 0.5" + (" -d" if args.delight else "") + (" -n" if args.normal else "") + (" --use_asg" if args.use_asg else "")
      common_args += f" --eval" if args.eval else ""
      common_args += f" --sd_normal_until_iter {args.sd_normal_until_iter}" if args.sd_normal_until_iter != -1 else ""
      common_args += f" --nofix_param" if args.nofix_param else ""
      cmd = f'CUDA_VISIBLE_DEVICES={gpu_id} python train.py -s {data_base_path}/scan{scene} -m {out_base_path}/dtu_scan{scene}/{out_name} {common_args}'
      print(cmd)
      os.system(cmd)

    if not args.skip_rendering:
      common_args = "--quiet --num_cluster 1 --voxel_size 0.002 --max_depth 5.0" + (" -d" if args.delight else "") + (" -n" if args.normal else "")
      common_args += f" --train_label {args.train_label}" if args.train_label else ""
      common_args += f" --test_label {args.test_label}" if args.test_label else ""
      cmd = f'CUDA_VISIBLE_DEVICES={gpu_id} python render.py -m {out_base_path}/dtu_scan{scene}/{out_name} {common_args}'
      print(cmd)
      os.system(cmd)

    if not args.skip_metrics:
      cmd = f"CUDA_VISIBLE_DEVICES={gpu_id} python scripts/eval_dtu/evaluate_single_scene.py " + \
            f"--input_mesh {out_base_path}/dtu_scan{scene}/{out_name}/mesh/tsdf_fusion_post.ply " + \
          f"--scan_id {scene} --output_dir {out_base_path}/dtu_scan{scene}/{out_name}/mesh " + \
          f"--mask_dir {data_base_path} " + \
          f"--DTU {eval_path}"
      print(cmd)
      os.system(cmd)