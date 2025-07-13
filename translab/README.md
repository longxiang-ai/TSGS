# Translab Dataset Generation Pipeline

This pipeline generates Translab datasets in [TSGS](https://e.gitee.com/zgcai/repos/zgcai/tsgs/sources) from Blender scenes for 3D reconstruction, neural rendering, and computer vision research. It produces multi-view images, depth maps, surface normals, object masks, and camera poses in formats compatible with COLMAP.

## Requirements

- **Blender** (tested with 4.3.2, other versions may work)
- **COLMAP**
- **Python** with the following packages:
  - `opencv-python`
  - `numpy`

## Installation

1. **Install Blender**: Download from [blender.org](https://www.blender.org/download/)
2. **Install COLMAP**: Follow instructions at [colmap.github.io](https://colmap.github.io/install.html)
3. **Install Python dependencies**:

   ```bash
   pip install opencv-python numpy
   ```

## Structure

```
translab/
├── scripts/
│   ├── main.sh                    # Main execution script
│   ├── blender_script.py          # Main Blender rendering script
│   ├── transparent_mask_script.py # Transparent object mask rendering
│   ├── mask_raw.py                # Merge the raw image and mask into PNG
│   ├── colmap_script.sh           # COLMAP reconstruction pipeline
│   └── camera_data.json           # Camera trajectory and parameters
├── HDRI/
│   └── lab.exr                    # Environment map for lighting
├── scenes/
│   └── scene_01/
│       └── XXXXX.blend            # Blender scene file
└── README.md
```

## Usage

### Quick Start

1. **Prepare your Blender scene**:
   - Create or import your 3D models in Blender
   - Save as `.blend` file in the `scenes/scene_name` directory, for example `scenes/scene_01`

2. **Configure the pipeline**:
   - Edit `scripts/main.sh` and set your Blender path:

     ```bash
     BLENDER_PATH="/path/to/your/blender" # for example: ../blender-4.3.2-linux-x64/blender
     ```

   - Adjust HDRI path if needed:

     ```bash
     HDRI_PATH="../HDRI/lab.exr"
     ```

3. **Run the pipeline**:

   ```bash
   cd scripts
   ./main.sh scene_01
   ```

### Advanced Usage

#### Custom Camera Trajectory

The pipeline uses predefined camera trajectories stored in `camera_data.json`. To create custom trajectories:

1. Set up camera keyframes in Blender
2. Enable trajectory extraction in `blender_script.py` (around lines 660-665):

```python
camera_trajectory = get_camera_trajectory(camera)
if camera_trajectory:
    print("success to get camera trajectory")
    # save the original trajectory data (include camera parameters)
    save_camera_trajectory(camera_trajectory, output_path, camera)
else:
    print("no camera trajectory found, please ensure the camera has been set to the keyframe")
```

3. modify the trajectory loading code (around line 669) to use your custom trajectory instead

```python
camera_data_file = r'camera_data.json'
```

#### Transparent Object Masks

For scenes with transparent objects:

1. Assign object indices to transparent objects in Blender
2. Uncomment the transparent mask rendering line in `main.sh`
3. The pipeline will generate additional transparency masks

## Output Structure

After running the pipeline, each scene will contain:

```bash
scenes/scene_01/
├── images/                # Final RGBA images (RGB + mask)
├── raw_images/            # Raw RGB images without masks
├── masks/                 # Object masks (0/255 values)
├── transparent_masks/     # Transparent object masks (optional)
├── normals_blender/       # Surface normal maps (not used in TSGS)
├── depths/                # Depth maps (not used in TSGS)
├── meshes/                # Exported scene geometry
│   ├── scene_mesh.obj
│   └── scene_mesh.mtl
├── sparse/0/              # COLMAP reconstruction
│   ├── cameras.txt
│   ├── images.txt
│   ├── points3D.txt
│   └── [binary files]
└── environment.exr        # Environment map copy
```

## License

This project is provided as-is for research and educational purposes. Please ensure compliance with Blender's license terms when using this pipeline.

## Acknowledgments

- Built with [Blender](https://www.blender.org/) 3D creation suite
- Uses [COLMAP](https://colmap.github.io/) for structure-from-motion
