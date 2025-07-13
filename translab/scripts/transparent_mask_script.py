import bpy
import math
import os
from mathutils import Vector, Matrix, Euler
import json
import time
import numpy as np
import argparse
import sys
import tempfile

def parse_args():
    # get the parameters passed in (skip the parameters of blender itself)
    argv = sys.argv
    if "--" not in argv:
        argv = []  # if no parameters, return an empty list
    else:
        argv = argv[argv.index("--") + 1:]  # get the parameters after --
    
    # set the parameter parser
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=str, required=True,
                       help='output directory path')
    parser.add_argument('--hdri', type=str, required=True,
                       help='HDRI environment map path')
    parser.add_argument('--test', action='store_true',
                       help='whether to use the test mode')
    
    # parse the parameters
    return parser.parse_args(argv)

def setup_scene(clear_scene=False):
    if clear_scene:
        # 清除默认场景中的所有对象
        bpy.ops.object.select_all(action='SELECT')
        bpy.ops.object.delete()

    # set the render engine to Cycles
    scene = bpy.context.scene
    scene.render.engine = 'CYCLES'
    # scene.render.engine = 'BLENDER_EEVEE_NEXT'
    # set the device type to GPU
    scene.cycles.device = 'GPU'
    
    # get the compute device preferences
    prefs = bpy.context.preferences.addons['cycles'].preferences
    
    # enable CUDA/OptiX
    prefs.compute_device_type = 'OPTIX' # or 'OPTIX' if your GPU supports it
    
    # ensure all available GPUs are enabled
    for device in prefs.devices:
        if device.type == 'CUDA' or device.type == 'OPTIX':  # only enable CUDA and OptiX devices
            device.use = True
    
    # set the sampling and performance parameters
    scene.cycles.samples = 4096  # adjust the number of samples according to your needs
    scene.cycles.use_denoising = True  # enable the denoising
    scene.cycles.denoiser = 'OPTIX'  # use the OptiX denoiser
    scene.cycles.use_denoising_passes = True
    scene.cycles.use_fast_gi = False
    scene.cycles.caustics_reflective = True  # enable the reflective caustics
    scene.cycles.caustics_refractive = True  # enable the refractive caustics

    
    scene.cycles.max_bounces = max(12, scene.cycles.max_bounces)  # increase the number of light bounces
    scene.cycles.diffuse_bounces = max(4, scene.cycles.diffuse_bounces)  # increase the number of diffuse light bounces
    scene.cycles.glossy_bounces = max(4, scene.cycles.glossy_bounces)  # increase the number of glossy light bounces
    scene.cycles.transmission_bounces = max(8, scene.cycles.transmission_bounces)   # increase the number of transmission light bounces
    # scene.cycles.adaptive_threshold = 0.005  # use the adaptive sampling
    # scene.cycles.use_adaptive_sampling = True

    # set the render optimization parameters
    scene.render.threads_mode = 'AUTO'  # automatically detect the number of CPU threads
    scene.render.use_persistent_data = True  # keep the data between frames to speed up the rendering
    
    # set the batch rendering
    scene.render.use_file_extension = True
    scene.render.use_placeholder = True  # allow parallel rendering
    scene.render.use_overwrite = True  # avoid rendering the same frame again

def setup_camera():
    # get the existing camera in the scene
    camera = None
    for obj in bpy.context.scene.objects:
        if obj.type == 'CAMERA':
            camera = obj
            print(f'find the camera: {camera.name}')
            break
    
    # if no camera is found, create a new one
    if camera is None:
        bpy.ops.object.camera_add(location=(0, -5, 2))
        camera = bpy.context.active_object
        print(f'create a new camera: {camera.name}')
        # set the camera parameters
        camera.data.lens = 35  # focal length
        camera.data.sensor_width = 32  # sensor width
    
    # set the camera as the scene camera
    bpy.context.scene.camera = camera
    return camera

def create_default_material():
    """create a default material (Diffuse BSDF)"""
    # create a new material
    default_mat = bpy.data.materials.new(name="Default_Material")
    default_mat.use_nodes = True
    
    # clear the nodes of the material
    nodes = default_mat.node_tree.nodes
    nodes.clear()
    
    # create the Diffuse BSDF shader
    diffuse = nodes.new(type='ShaderNodeBsdfDiffuse')
    diffuse.location = (0, 0)
    diffuse.inputs['Color'].default_value = (0.8, 0.8, 0.8, 1.0)  # set to gray (adjustable)
    
    # create the material output node
    mat_output = nodes.new(type='ShaderNodeOutputMaterial')
    mat_output.location = (300, 0)
    
    # connect the Diffuse to the output
    links = default_mat.node_tree.links
    links.new(diffuse.outputs['BSDF'], mat_output.inputs['Surface'])
    
    return default_mat

def set_default_material_to_all_objects():
    """set the default material to all objects"""
    # get or create the default material
    default_mat = bpy.data.materials.get("Default_Material") or create_default_material()
    
    # iterate over all objects in the scene
    for obj in bpy.data.objects:
        if obj.type == 'MESH':  # only process mesh objects
            # clear the existing material
            obj.data.materials.clear()
            # add the default material
            obj.data.materials.append(default_mat)
            
def render_dataset_transparent_mask(output_path, test=False):
    """render the mask of all transparent objects in the scene"""
    # create the output directory
    os.makedirs(os.path.join(output_path, "transparent_masks"), exist_ok=True)
    
    # set the default material to all objects
    set_default_material_to_all_objects()
    
    # set the scene and view layer properties
    scene = bpy.context.scene
    
    scene.render.resolution_x = 1600
    scene.render.resolution_y = 1600
    
    view_layer = bpy.context.view_layer  # fix the view layer

    # enable the object index channel
    view_layer.use_pass_object_index = True

    # set the render engine to Cycles (if needed)
    scene.render.engine = 'CYCLES'
    
    # decrease the number of samples to speed up the rendering (because only the mask is needed)
    scene.cycles.samples = 32  # for mask rendering, low sampling is enough
    
    # close unnecessary rendering features
    scene.cycles.use_denoising = False
    scene.cycles.max_bounces = 1
    scene.cycles.diffuse_bounces = 1
    scene.cycles.glossy_bounces = 1
    scene.cycles.transmission_bounces = 1

    # configure the composite nodes
    # clear the existing nodes
    scene.use_nodes = True
    comp_nodes = scene.node_tree.nodes
    comp_nodes.clear()

    # create the necessary nodes
    render_layers = comp_nodes.new(type='CompositorNodeRLayers')
    id_mask = comp_nodes.new(type='CompositorNodeIDMask')
    output_file = comp_nodes.new(type='CompositorNodeOutputFile')

    # set the node properties
    render_layers.location = (0, 0)
    id_mask.location = (300, 0)
    output_file.location = (600, 0)

    id_mask.index = 1  # set the index of the ID Mask to 1 (corresponding to the transparent objects)

    # connect
    links = scene.node_tree.links
    links.new(render_layers.outputs['IndexOB'], id_mask.inputs['ID value'])
    links.new(id_mask.outputs['Alpha'], output_file.inputs['Image'])

    # set the output path (can be modified according to needs)
    output_file.base_path = os.path.join(output_path, "transparent_masks")  # output to the same directory as the Blender file
    output_file.format.file_format = 'PNG'   # set the output format to PNG
    output_file.file_slots[0].path = "frame_####"

    # set the render settings
    scene.render.image_settings.file_format = 'PNG'
    scene.render.image_settings.color_mode = 'RGBA'  # include the transparent channel
    
    # decrease the render resolution to speed up the rendering (if needed)
    # scene.render.resolution_percentage = 50  # if the original resolution is too high, it can be reduced

    print("transparent object mask rendering settings completed!")
    scene.render.filepath = tempfile.gettempdir() + "/"
    bpy.ops.render.render(animation=True)

def load_camera_trajectory(filepath):
    """load the camera trajectory data and camera parameters from the JSON file"""
    try:
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        # convert the data back to the Blender data type
        trajectory_data = []
        for frame_data in data['trajectory']:
            converted_frame = {
                'frame': frame_data['frame'],
                'location': Vector(frame_data['location']),
                'rotation': Euler(frame_data['rotation'])  # directly create the Euler object
            }
            trajectory_data.append(converted_frame)
        
        return trajectory_data, data['camera_params']
    except Exception as e:
        print(f"error when loading the camera trajectory data: {str(e)}")
        return None, None

def apply_camera_trajectory(camera, trajectory_data, camera_params=None):
    """apply the trajectory data and camera parameters to the camera"""
    # apply the camera parameters
    if camera_params:
        # set the basic parameters
        camera.data.lens = camera_params['lens']
        camera.data.sensor_width = camera_params['sensor_width']
        camera.data.sensor_height = camera_params['sensor_height']
        camera.data.sensor_fit = camera_params['sensor_fit']
        camera.data.clip_start = camera_params['clip_start']
        camera.data.clip_end = camera_params['clip_end']
        
        # set the depth of field parameters
        camera.data.dof.use_dof = camera_params['dof.use_dof']
        camera.data.dof.focus_distance = camera_params['dof.focus_distance']
        camera.data.dof.aperture_fstop = camera_params['dof.aperture_fstop']
        
        # set the shift
        camera.data.shift_x = camera_params['shift_x']
        camera.data.shift_y = camera_params['shift_y']
        
        # set the camera type
        camera.data.type = camera_params['type']
        if camera.data.type == 'ORTHO':
            camera.data.ortho_scale = camera_params['ortho_scale']
            
        print("camera parameters applied")
    
    # clear the existing animation data
    if camera.animation_data:
        camera.animation_data_clear()
    
    # get all frame numbers
    frame_numbers = []
    
    # apply the new keyframes
    for frame_data in trajectory_data:
        frame = frame_data['frame']
        frame_numbers.append(frame)
        # set the location
        camera.location = frame_data['location']
        camera.keyframe_insert(data_path="location", frame=frame)
        
        # set the rotation
        camera.rotation_euler = frame_data['rotation']
        camera.keyframe_insert(data_path="rotation_euler", frame=frame)
    
    print(f"{len(trajectory_data)} keyframes applied to the camera")
    
    min_frame = min(frame_numbers)
    max_frame = max(frame_numbers)
    
    # set the scene frame range
    scene = bpy.context.scene
    scene.frame_start = min_frame
    scene.frame_end = max_frame
    print(f"set the scene frame range: {min_frame} - {max_frame}")

def main():
    args = parse_args()
    # set the output path
    output_path = args.output
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    # initialize the scene (simplified setting, only keep the necessary parts)
    setup_scene(clear_scene=False)
    
    # set the camera
    camera = setup_camera()

    # load and apply the saved trajectory
    camera_data_file = r'camera_data.json'
    camera_trajectory, camera_params = load_camera_trajectory(camera_data_file)
    apply_camera_trajectory(camera, camera_trajectory, camera_params)
    
    if args.test:
        bpy.context.scene.frame_end = 3

    print(f'start rendering the mask of transparent objects, frame number: {bpy.context.scene.frame_end - bpy.context.scene.frame_start + 1}')

    # only render the mask of transparent objects
    render_dataset_transparent_mask(output_path)
    
if __name__ == "__main__":
    main()