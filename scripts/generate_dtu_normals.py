"""
Generate normal maps for DTU scans using Lotus-2.
Outputs are saved in the format expected by TSGS: {scan_dir}/normals/{image_name}_normal.png
"""
import argparse
import os
import sys
import shutil
import subprocess

DTU_SCANS = [24, 37, 40, 55, 63, 65, 69, 83, 97, 105, 106, 110, 114, 118, 122]

LOTUS2_DIR = "/home/dataset-local/lmw/codes/Lotus-2"
LOTUS2_WEIGHTS = "/home/dataset-local/lmw/hf_models/lotus-2"
FLUX_MODEL = "/home/dataset-local/lmw/hf_models/black-forest-labs/FLUX.1-dev"
DTU_DATA = "/home/dataset-local/lmw/codes/TSGS/data/dtu_dataset/dtu"


def generate_normals_for_scan(scan_id, gpu_id):
    input_dir = os.path.join(DTU_DATA, f"scan{scan_id}", "images")
    tmp_output = os.path.join(DTU_DATA, f"scan{scan_id}", "normals_tmp")
    final_output = os.path.join(DTU_DATA, f"scan{scan_id}", "normals")

    if os.path.exists(final_output):
        n_files = len([f for f in os.listdir(final_output) if f.endswith("_normal.png")])
        n_images = len([f for f in os.listdir(input_dir) if f.endswith(".png")])
        if n_files == n_images:
            print(f"scan{scan_id}: normals already exist ({n_files} files), skipping")
            return True

    os.makedirs(tmp_output, exist_ok=True)

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    env["HF_HOME"] = "/home/dataset-local/lmw/hf_models"

    cmd = [
        sys.executable, os.path.join(LOTUS2_DIR, "infer.py"),
        "--pretrained_model_name_or_path", FLUX_MODEL,
        "--core_predictor_model_path", os.path.join(LOTUS2_WEIGHTS, "lotus-2_core_predictor_normal.safetensors"),
        "--lcm_model_path", os.path.join(LOTUS2_WEIGHTS, "lotus-2_lcm_normal.safetensors"),
        "--detail_sharpener_model_path", os.path.join(LOTUS2_WEIGHTS, "lotus-2_detail_sharpener_normal.safetensors"),
        "--input_dir", input_dir,
        "--output_dir", tmp_output,
        "--task_name", "normal",
        "--seed", "0",
        "--mixed_precision", "bf16",
    ]

    print(f"scan{scan_id} on GPU {gpu_id}: generating normals...")
    result = subprocess.run(cmd, env=env, capture_output=True, text=True, cwd=LOTUS2_DIR)

    if result.returncode != 0:
        print(f"scan{scan_id} FAILED: {result.stderr[-500:]}")
        return False

    vis_dir = os.path.join(tmp_output, "normal_vis")
    if not os.path.exists(vis_dir):
        print(f"scan{scan_id}: normal_vis dir not found in {tmp_output}")
        return False

    os.makedirs(final_output, exist_ok=True)
    for fname in os.listdir(vis_dir):
        if fname.endswith(".png"):
            stem = os.path.splitext(fname)[0]
            src = os.path.join(vis_dir, fname)
            dst = os.path.join(final_output, f"{stem}_normal.png")
            shutil.copy2(src, dst)

    n_copied = len([f for f in os.listdir(final_output) if f.endswith("_normal.png")])
    print(f"scan{scan_id}: {n_copied} normal maps generated")

    shutil.rmtree(tmp_output, ignore_errors=True)
    return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--scan_id", type=int, required=True)
    parser.add_argument("--gpu_id", type=int, default=0)
    args = parser.parse_args()

    success = generate_normals_for_scan(args.scan_id, args.gpu_id)
    sys.exit(0 if success else 1)
