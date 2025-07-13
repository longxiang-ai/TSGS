output_path="$1"
database_path="$output_path"/database.db
image_path="$output_path"/images
input_path="$output_path"/sparse/0
# 1. create COLMAP database
colmap database_creator --database_path "$database_path"

# 2. feature extraction, replace the camera parameters with your own camera parameters
colmap feature_extractor --database_path "$database_path" --image_path "$image_path" --ImageReader.camera_model SIMPLE_PINHOLE --ImageReader.camera_params "1111.1110311937682,400.0,400.0"

# 3. feature matching
colmap exhaustive_matcher --database_path "$database_path"

# 4. use the existing parameters for triangulation (We get the extrinsic parameters from the blender script)
colmap point_triangulator --database_path "$database_path" --image_path "$image_path" --input_path "$input_path" --output_path "$input_path" --Mapper.ba_refine_focal_length 0 --Mapper.ba_refine_extra_params 0 

# 5. convert to TXT format for check the result
colmap model_converter --input_path "$input_path" --output_path "$input_path" --output_type TXT

# 6. optional: convert to PLY format to view the result
# colmap model_converter --input_path "$input_path" --output_path "$input_path"/pointcloud.ply --output_type PLY
