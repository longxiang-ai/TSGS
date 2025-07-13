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

    args = parser.parse_args()
    base_folder = args.source_path

    folder_path = os.path.join(base_folder, "images")
    image_paths = sorted(glob.glob(os.path.join(folder_path, "*.png")))
    print(image_paths)  

    if not os.path.exists(os.path.join(base_folder, "delights")):
        # 首先处理delight
        print("process delight first")
        # Create predictor instance
        print("loading StableDelight model")
        predictor = torch.hub.load("Stable-X/StableDelight", "StableDelight_turbo", trust_repo=True)
        print("StableDelight model loaded")

        delight_save_path = os.path.join(base_folder, "delights")
        if not os.path.exists(delight_save_path):
            os.makedirs(delight_save_path)

        delight_paths = []
        for image_path in tqdm(image_paths):
            delight_output_path = os.path.join(delight_save_path, os.path.basename(image_path).replace(".png", "_delight.png"))
            process_image(predictor=predictor, 
                        input_image_path=image_path, 
                        output_image_path=delight_output_path, 
                        mask_path=None)
            delight_paths.append(delight_output_path)

        print("delight processing done, now process normal using delight results")

        del predictor
    else:
        delight_paths = sorted(glob.glob(os.path.join(os.path.join(base_folder, "delights"), "*.png")))

    # 然后使用delight结果来估计normal
    print("loading StableNormal model")
    predictor = torch.hub.load("Stable-X/StableNormal", "StableNormal", trust_repo=True)
    print("StableNormal model loaded")

    normal_save_path = os.path.join(base_folder, "delighted_normals")
    if not os.path.exists(normal_save_path):
        os.makedirs(normal_save_path)

    for delight_path in tqdm(delight_paths):
        normal_output_path = os.path.join(normal_save_path, os.path.basename(delight_path).replace("_delight.png", "_delighted_normal.png"))
        process_image(predictor=predictor, 
                    input_image_path=delight_path, 
                    output_image_path=normal_output_path, 
                    mask_path=None)

    print("normal processing done")
