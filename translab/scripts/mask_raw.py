import os
import argparse
import cv2
import numpy as np
from tqdm import tqdm

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', '-i', type=str, required=True, help='Path to the input image')
    args = parser.parse_args()
    # Set the path to the raw data
    raw_data_root = os.path.join(args.input, 'raw_images')
    mask_root = os.path.join(args.input, 'masks')
    output_root = os.path.join(args.input, 'images')
    os.makedirs(output_root, exist_ok=True)

    # Loop through each file
    for file in tqdm(os.listdir(raw_data_root)):
        file_path = os.path.join(raw_data_root, file)
        mask_path = os.path.join(mask_root, file)

        # read the raw image
        raw_image = cv2.imread(file_path)
        # read the mask image
        mask_image = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        mask = (mask_image > 0).astype(np.uint8) * 255 # The rendering process cant ensure the mask is 0 or 255 (maybe a little bit small), so we need to convert it to 0 or 255
        # save the mask
        cv2.imwrite(mask_path, mask)
        rbga_image = cv2.merge([raw_image, mask])
        # save the merged image
        output_path = os.path.join(output_root, file)
        cv2.imwrite(output_path, rbga_image)


