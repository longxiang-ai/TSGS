import torch
from PIL import Image
import os
import glob
from tqdm import tqdm
import argparse

def process_image(predictor, input_image_path, output_image_path):
    image = Image.open(input_image_path)
    normal_image = predictor(image)
        
    normal_image.save(output_image_path)
    print(f"Processed {input_image_path} to {output_image_path}")   

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source_path", "-s", type=str, help="source path", required=True, default="data/Anisotropic-Synthetic-Dataset/ashtray")
    parser.add_argument("--skip_normal", help="skip normal", action="store_true")
    parser.add_argument("--skip_delight", help="skip delight", action="store_true")

    args = parser.parse_args()
    base_folder = args.source_path

    save_path = os.path.join(base_folder, "normals")
    folder_path = os.path.join(base_folder, "images")

    image_paths = sorted(glob.glob(os.path.join(folder_path, "*.png")))
    print(image_paths)  
    if len(image_paths) == 0:
        print("no image found in ", folder_path)
        exit(0)
    print("results will be saved in ", save_path, "total", len(image_paths), "images")

    if not args.skip_normal:
        # Create predictor instance
        print("loading StableNormal model")
        predictor = torch.hub.load("Stable-X/StableNormal", "StableNormal", trust_repo=True)
        print("StableNormal model loaded")

        if not os.path.exists(save_path):
            os.makedirs(save_path)

        for image_path in tqdm(image_paths):
            process_image(predictor=predictor, 
                        input_image_path=image_path, 
                        output_image_path=os.path.join(save_path, os.path.basename(image_path).replace(".png", "_normal.png")), 
                        mask_path=None)
        print("process normal done, start process delight")

        del predictor

    if not args.skip_delight:
        # Create predictor instance
        print("loading StableDelight model")
        predictor = torch.hub.load("Stable-X/StableDelight", "StableDelight_turbo", trust_repo=True)
        print("StableDelight model loaded")

        save_path = os.path.join(base_folder, "delights")
        if not os.path.exists(save_path):
            os.makedirs(save_path)

        for image_path in tqdm(image_paths):
            process_image(predictor=predictor, 
                        input_image_path=image_path, 
                        output_image_path=os.path.join(save_path, os.path.basename(image_path).replace(".png", "_delight.png")), 
                        mask_path=None)

    print(f"normal and delight done, total {len(image_paths)} images, {base_folder}")