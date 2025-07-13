#!/bin/bash

BLENDER_PATH="change to your own blender path" # for example: ../blender-4.3.2-linux-x64/blender
BASE_DIR="../scenes"
HDRI_PATH="../HDRI/lab.exr"

# check parameters
if [ $# -ne 1 ]; then
    echo "usage: $0 <scene name>"
    echo "example: $0 scene_01"
    exit 1
fi

SCENE_DIR="$1"
SCENE_PATH="$BASE_DIR/$SCENE_DIR"

# check if scene directory exists
if [ ! -d "$SCENE_PATH" ]; then
    echo "error: scene directory '$SCENE_PATH' does not exist"
    exit 1
fi


# find .blend files in the scene directory
blend_files=$(find "$SCENE_PATH" -maxdepth 1 -name "*.blend")

# check if .blend file is found
if [ -z "$blend_files" ]; then
    echo "error: no .blend file found in '$SCENE_PATH'"
    exit 1
fi

# process the .blend file
for blend_file in $blend_files; do
    echo "processing file: $blend_file"
    
    # run blender
    # render images, normals, depths and masks
    "$BLENDER_PATH" "$blend_file" -b -P blender_script.py -- --output "$SCENE_PATH" --hdri "$HDRI_PATH"
    # optional: render the mask of transparent objects, if you have already correctly set the transparent objects in the blender file
    # "$BLENDER_PATH" "$blend_file" -b -P transparent_mask_script.py -- --output "$SCENE_PATH" --hdri "$HDRI_PATH"
    
    echo "start merging mask and the raw image"
    python mask_raw.py -i "$SCENE_PATH"

    # run colmap
    echo "start colmap"
    sh colmap_script.sh "$SCENE_PATH" 
    rm "$SCENE_PATH/database.db" # delete the colmap database
    
    echo "finish processing: $blend_file"
    echo "-------------------"
done

echo "finish processing"