"""
Predict surface normals for TransLab scenes using TransNormal model.

Saves output to 'transnormals/' subfolder to keep existing StableNormal results intact.
Naming convention: frame_XXXX_normal.png  (same as StableNormal for drop-in replacement)

Usage:
    # Single scene
    python preprocess/process_transnormal.py -s data/translab/scene_01 \
        --transnormal_root /path/to/TransNormal

    # All scenes
    python preprocess/process_transnormal.py --all --data_root data/translab \
        --transnormal_root /path/to/TransNormal
"""

import argparse
import glob
import os
import sys

import torch
from PIL import Image
from tqdm import tqdm


def build_pipeline(transnormal_root, device="cuda", dtype=torch.bfloat16):
    weights_dir = os.path.join(transnormal_root, "weights", "transnormal")
    dinov3_dir = os.path.join(transnormal_root, "weights", "dinov3_vith16plus")
    projector_path = os.path.join(weights_dir, "cross_attention_projector.pt")

    sys.path.insert(0, transnormal_root)
    from transnormal import TransNormalPipeline, create_dino_encoder

    print("[TransNormal] Loading DINOv3 encoder...")
    dino_encoder = create_dino_encoder(
        model_name="dinov3_vith16plus",
        cross_attention_dim=1024,
        weights_path=dinov3_dir,
        projector_path=projector_path,
        device=device,
        dtype=dtype,
        freeze_encoder=True,
    )

    print("[TransNormal] Loading pipeline...")
    pipe = TransNormalPipeline.from_pretrained(
        weights_dir,
        dino_encoder=dino_encoder,
        torch_dtype=dtype,
        safety_checker=None,
    )
    pipe = pipe.to(device)
    print("[TransNormal] Pipeline ready.")
    return pipe


def process_scene(pipe, source_path, output_folder="transnormals", processing_res=768):
    images_dir = os.path.join(source_path, "images")
    save_dir = os.path.join(source_path, output_folder)
    os.makedirs(save_dir, exist_ok=True)

    image_paths = sorted(glob.glob(os.path.join(images_dir, "*.png")))
    if not image_paths:
        image_paths = sorted(glob.glob(os.path.join(images_dir, "*.jpg")))
    if not image_paths:
        print(f"[TransNormal] No images found in {images_dir}")
        return

    print(f"[TransNormal] Processing {len(image_paths)} images from {source_path}")
    print(f"[TransNormal] Saving to {save_dir}")

    for img_path in tqdm(image_paths, desc=os.path.basename(source_path)):
        basename = os.path.splitext(os.path.basename(img_path))[0]
        out_path = os.path.join(save_dir, f"{basename}_normal.png")

        if os.path.exists(out_path):
            continue

        image = Image.open(img_path).convert("RGB")
        with torch.no_grad():
            normal_pil = pipe(
                image=image,
                processing_res=processing_res,
                output_type="pil",
            )
        normal_pil.save(out_path)

    print(f"[TransNormal] Done: {source_path} -> {save_dir}")


def main():
    parser = argparse.ArgumentParser(description="TransNormal batch inference for TSGS")
    parser.add_argument("--source_path", "-s", type=str, help="Single scene path")
    parser.add_argument("--all", action="store_true", help="Process all TransLab scenes")
    parser.add_argument("--data_root", type=str, default="data/translab",
                        help="Root dir of TransLab scenes (used with --all)")
    parser.add_argument("--transnormal_root", type=str, required=True,
                        help="Path to cloned TransNormal repository (contains weights/)")
    parser.add_argument("--output_folder", type=str, default="transnormals",
                        help="Output subfolder name (default: transnormals)")
    parser.add_argument("--processing_res", type=int, default=768)
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    pipe = build_pipeline(args.transnormal_root, device=args.device)

    if args.all:
        scenes = sorted(glob.glob(os.path.join(args.data_root, "scene_*")))
        if not scenes:
            print(f"No scenes found in {args.data_root}")
            return
        for scene in scenes:
            process_scene(pipe, scene, args.output_folder, args.processing_res)
    elif args.source_path:
        process_scene(pipe, args.source_path, args.output_folder, args.processing_res)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
