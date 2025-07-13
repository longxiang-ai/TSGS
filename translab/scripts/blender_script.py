import bpy
import math
import os
from mathutils import Vector, Matrix, Euler
import json
import time
import numpy as np
import argparse
import sys
import shutil
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
        # clear all objects in the default scene
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
    scene.cycles.samples = 512  # adjust the number of samples according to your needs
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

    # set the resolution
    scene.render.resolution_x = 1600
    scene.render.resolution_y = 1600
    scene.render.resolution_percentage = 100
    scene.render.image_settings.color_depth = '16'
    scene.render.image_settings.color_mode = 'RGB'
    scene.render.image_settings.file_format = 'PNG'

    scene.render.threads_mode = 'AUTO'  # automatically detect the number of CPU threads
    scene.render.use_persistent_data = True  # keep the data between frames to speed up the rendering
    bpy.context.view_layer.use_pass_normal = True # enable the normal pass
    scene.render.film_transparent = False  # disable the transparent background
    scene.render.use_high_quality_normals = True  # improve the normal map quality
    scene.render.use_persistent_data = True  # keep the data, improve the rendering speed
    scene.render.use_simplify = False  # close the simplification mode, ensure the highest quality
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

def get_camera_trajectory(camera):
    """get the existing camera trajectory"""
    trajectory_data = []
    
    # get the scene
    scene = bpy.context.scene
    
    # check if the camera has animation data
    if not camera.animation_data or not camera.animation_data.action:
        print("warning: the camera has no animation data")
        return trajectory_data
    
    # get all keyframes
    fcurves = camera.animation_data.action.fcurves
    
    # find the keyframes of the location and rotation
    frame_numbers = set()
    for fc in fcurves:
        for kf in fc.keyframe_points:
            frame_numbers.add(int(kf.co[0]))
    
    frame_numbers = sorted(list(frame_numbers))
    
    # collect the camera information of each keyframe
    for frame in frame_numbers:
        scene.frame_set(frame)
        trajectory_data.append({
            'frame': frame,
            'location': camera.location.copy(),
            'rotation': camera.rotation_euler.copy()
        })
        
    print(f"find {len(trajectory_data)} keyframes")
    return trajectory_data

def render_dataset(output_path, test=False):
    # set the render output path
    scene = bpy.context.scene
    
    scene.use_nodes = True
    tree = scene.node_tree
    for n in tree.nodes:
        tree.nodes.remove(n)
    links = tree.links

    render_layers = tree.nodes.new('CompositorNodeRLayers')

    # set the normal output node
    scale_normal = tree.nodes.new(type="CompositorNodeMixRGB")
    scale_normal.blend_type = 'MULTIPLY'
    scale_normal.inputs[2].default_value = (0.5, 0.5, 0.5, 1)
    links.new(render_layers.outputs['Normal'], scale_normal.inputs[1])
    bias_normal = tree.nodes.new(type="CompositorNodeMixRGB")
    bias_normal.blend_type = 'ADD'
    bias_normal.inputs[2].default_value = (0.5, 0.5, 0.5, 0)
    links.new(scale_normal.outputs[0], bias_normal.inputs[1])
    normal_file_output = tree.nodes.new(type="CompositorNodeOutputFile")
    normal_file_output.label = 'Normal Output'
    links.new(bias_normal.outputs[0], normal_file_output.inputs[0])

    # set the image output node
    image_file_output = tree.nodes.new(type="CompositorNodeOutputFile")
    image_file_output.label = 'Image'
    links.new(render_layers.outputs['Image'], image_file_output.inputs[0])
    
    scene.render.filepath = tempfile.gettempdir() + "/"

    # enable the Cryptomatte
    scene.view_layers["View Layer"].use_pass_cryptomatte_object = True
    
    # create the output directories
    normal_output_path = os.path.join(output_path, "normals_blender")
    image_output_path = os.path.join(output_path, "raw_images")
    os.makedirs(normal_output_path, exist_ok=True)
    os.makedirs(image_output_path, exist_ok=True)

    normal_file_output.base_path = normal_output_path
    normal_file_output.file_slots[0].path = "frame_####"
    image_file_output.base_path = image_output_path
    image_file_output.file_slots[0].path = "frame_####"
    bpy.ops.render.render(animation=True)

def render_dataset_depth(output_path, test=False):
    # set the render output path
    scene = bpy.context.scene
    scene.render.image_settings.file_format = 'OPEN_EXR'
    scene.render.image_settings.color_mode = 'BW'

    scene.use_nodes = True
    tree = scene.node_tree
    for n in tree.nodes:
        tree.nodes.remove(n)
    links = tree.links

    render_layers = tree.nodes.new('CompositorNodeRLayers')
    depth_file_output = tree.nodes.new(type="CompositorNodeOutputFile")
    depth_file_output.label = 'Depth Output'
    links.new(render_layers.outputs['Depth'], depth_file_output.inputs[0])

    depth_file_output.base_path = os.path.join(output_path, "depths")
    depth_file_output.file_slots[0].path = "frame_####"
    os.makedirs(os.path.join(output_path, "depths"), exist_ok=True)
    bpy.ops.render.render(animation=True)

def render_dataset_mask(output_path, test=False):
    """render the union mask of all objects in the scene"""
    # create the output directory
    os.makedirs(os.path.join(output_path, "masks"), exist_ok=True)
    
    # store the state that needs to be restored
    original_active = bpy.context.view_layer.objects.active
    original_selection = {obj: obj.select_get() for obj in bpy.context.scene.objects}
    original_visibility = {obj: obj.hide_viewport for obj in bpy.context.view_layer.objects}
    original_world = bpy.context.scene.world
    original_engine = bpy.context.scene.render.engine
    original_materials = {}

    try:
        # control the visibility of the objects through the view layer
        for obj in bpy.context.view_layer.objects:
            if obj.type in {'CAMERA', 'LIGHT', 'EMPTY'} or not obj.visible_get():
                obj.hide_viewport = True
                obj.hide_render = True
            else:
                obj.hide_viewport = False
                obj.hide_render = False
                # save and replace the materials
                if obj.type == 'MESH':
                    # save the original materials of all material slots
                    original_materials[obj.name] = []
                    for slot in obj.material_slots:
                        if slot.material:
                            original_materials[obj.name].append(slot.material)
                    
                    # create a new white emission material (if not exist)
                    white_mat_name = "White_Emission_Mask"
                    white_mat = bpy.data.materials.get(white_mat_name)
                    if not white_mat:
                        white_mat = bpy.data.materials.new(name=white_mat_name)
                        white_mat.use_nodes = True
                        nodes = white_mat.node_tree.nodes
                        nodes.clear()
                        emission = nodes.new('ShaderNodeEmission')
                        emission.inputs[0].default_value = (1, 1, 1, 1)
                        emission.inputs[1].default_value = 1
                        output = nodes.new('ShaderNodeOutputMaterial')
                        white_mat.node_tree.links.new(emission.outputs[0], output.inputs[0])
                    
                    # replace the materials of all material slots
                    for slot in obj.material_slots:
                        slot.material = white_mat

        # set the world background to pure black
        world = bpy.context.scene.world
        if world:
            world.use_nodes = True
            nodes = world.node_tree.nodes
            links = world.node_tree.links
            nodes.clear()
            background = nodes.new('ShaderNodeBackground')
            background.inputs['Color'].default_value = (0, 0, 0, 0)  # pure black
            background.inputs['Strength'].default_value = 0  # no strength
            output = nodes.new('ShaderNodeOutputWorld')
            links.new(background.outputs[0], output.inputs[0])

        # set the render engine and output
        scene = bpy.context.scene
        scene.render.engine = 'BLENDER_EEVEE_NEXT'
        scene.render.image_settings.file_format = 'PNG'
        scene.render.film_transparent = True  # enable the transparent background

        # render each frame
        scene.render.filepath = f"{output_path}/masks/frame_"
        bpy.ops.render.render(animation=True)

    finally:
        # restore the original state
        for obj, materials in original_materials.items():
            if obj in bpy.data.objects:
                obj_ref = bpy.data.objects[obj]
                # restore the original materials
                for i, mat in enumerate(materials):
                    if i < len(obj_ref.material_slots):
                        obj_ref.material_slots[i].material = mat
        
        for obj, visibility in original_visibility.items():
            obj.hide_viewport = visibility
        for obj, selected in original_selection.items():
            if obj:
                obj.select_set(selected)
        
        bpy.context.view_layer.objects.active = original_active
        scene.render.engine = original_engine
        scene.world = original_world
        # clear the temporary materials
        for material in bpy.data.materials:
            if material.name.startswith("White_Emission"):
                bpy.data.materials.remove(material)

def setup_world_lighting(output_path, hdri_path=None):
    # get the world node
    world = bpy.context.scene.world
    if world is None:
        world = bpy.data.worlds.new(name="World")
        bpy.context.scene.world = world
        print(f'create a new world: {world.name}')
    
    # enable the nodes
    world.use_nodes = True
    nodes = world.node_tree.nodes
    links = world.node_tree.links
    
    # clear the existing nodes
    nodes.clear()
    
    # create the environment map node
    env_tex = nodes.new('ShaderNodeTexEnvironment')
    env_tex.location = (-300, 0)
    
    # load the environment map
    if hdri_path and os.path.exists(hdri_path):
        try:
            env_image = bpy.data.images.load(hdri_path)
            env_tex.image = env_image
            print(f'load the environment map: {hdri_path}')
            
            # save the environment map copy
            try:
                # copy the original environment map
                env_image_path = os.path.join(output_path, "environment.exr")
                shutil.copy2(hdri_path, env_image_path)
                print(f'environment map has been saved to: {env_image_path}')
                
                # create the PNG format environment map for preview
                preview_path = os.path.join(output_path, "environment_preview.png")
                # create a new image data block for the preview
                preview_image = bpy.data.images.new(
                    name="HDRIPreview",
                    width=1024,
                    height=512
                )
                preview_image.filepath_raw = preview_path
                preview_image.file_format = 'PNG'
                
                # copy the pixel data of the environment map
                preview_image.pixels = env_image.pixels[:]
                preview_image.save()
                
                # clear the preview image
                bpy.data.images.remove(preview_image)
                print(f'environment map preview has been saved to: {preview_path}')
                
            except Exception as e:
                print(f"error when saving the environment map: {str(e)}")
                print(f"continue to execute, but do not save the environment map copy")
                
        except Exception as e:
            print(f"cannot load the specified environment map: {str(e)}")
            return
    else:
        print("no environment map path or path does not exist")
        return
    
    # create the mapping node
    mapping = nodes.new('ShaderNodeMapping')
    mapping.location = (-500, 0)
    
    # create the coordinate node
    tex_coord = nodes.new('ShaderNodeTexCoord')
    tex_coord.location = (-700, 0)
    
    # create the background node
    background = nodes.new('ShaderNodeBackground')
    background.location = (0, 0)
    background.inputs['Strength'].default_value = 1.0  # 光照强度
    
    # create the output node
    output = nodes.new('ShaderNodeOutputWorld')
    output.location = (200, 0)
    
    # link the nodes
    links.new(tex_coord.outputs['Generated'], mapping.inputs['Vector'])
    links.new(mapping.outputs['Vector'], env_tex.inputs['Vector'])
    links.new(env_tex.outputs['Color'], background.inputs['Color'])
    links.new(background.outputs['Background'], output.inputs['Surface'])
    
def save_camera_trajectory(trajectory_data, output_path, camera):
    """save the camera trajectory data and camera parameters to the JSON file"""
    # collect the camera parameters
    camera_params = {
        'lens': camera.data.lens,
        'sensor_width': camera.data.sensor_width,
        'sensor_height': camera.data.sensor_height,
        'sensor_fit': camera.data.sensor_fit,
        'clip_start': camera.data.clip_start,
        'clip_end': camera.data.clip_end,
        'dof.use_dof': camera.data.dof.use_dof,
        'dof.focus_distance': camera.data.dof.focus_distance,
        'dof.aperture_fstop': camera.data.dof.aperture_fstop,
        'shift_x': camera.data.shift_x,
        'shift_y': camera.data.shift_y,
        'type': camera.data.type,  # 'PERSP', 'ORTHO', or 'PANO'
    }
    
    if camera.data.type == 'ORTHO':
        camera_params['ortho_scale'] = camera.data.ortho_scale
    
    # convert the Vector and Euler objects to the serializable format
    serializable_data = []
    for frame_data in trajectory_data:
        serializable_frame = {
            'frame': frame_data['frame'],
            'location': [frame_data['location'].x, 
                        frame_data['location'].y, 
                        frame_data['location'].z],
            'rotation': [frame_data['rotation'].x, 
                        frame_data['rotation'].y, 
                        frame_data['rotation'].z]
        }
        serializable_data.append(serializable_frame)
    
    # create the complete data structure
    save_data = {
        'camera_params': camera_params,
        'trajectory': serializable_data
    }
    
    # save to the JSON file
    camera_data_file = os.path.join(output_path, 'camera_data.json')
    with open(camera_data_file, 'w') as f:
        json.dump(save_data, f, indent=4)
    print(f"camera parameters and trajectory data have been saved to: {camera_data_file}")

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
            
        print("apply the camera parameters")
    
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
    
    print(f"apply {len(trajectory_data)} keyframes to the camera")
    
    min_frame = min(frame_numbers)
    max_frame = max(frame_numbers)
    
    # set the scene frame range
    scene = bpy.context.scene
    scene.frame_start = min_frame
    scene.frame_end = max_frame
    print(f"set the scene frame range: {min_frame} - {max_frame}")

def save_colmap_format(trajectory_data, output_path, camera):
    """save the camera trajectory to the COLMAP format"""
    # create the sparse/0 directory
    colmap_dir = os.path.join(output_path, 'sparse', '0')
    os.makedirs(colmap_dir, exist_ok=True)
    
    w = bpy.context.scene.render.resolution_x
    h = bpy.context.scene.render.resolution_y
    
    # calculate the camera intrinsic parameters
    fx = 0.5 * w / math.tan(0.5 * camera.data.angle_x)  # focal length (pixel)
    cx = 0.5 * w  # principal point x coordinate
    cy = 0.5 * h  # principal point y coordinate
    
    # 1. save cameras.txt
    # COLMAP format: CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]
    with open(os.path.join(colmap_dir, 'cameras.txt'), 'w') as f:
        f.write('# Camera list with one line of data per camera:\n')
        f.write('#   CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]\n')
        f.write('# Number of cameras: 1\n')
        # use the SIMPLE_PINHOLE model, parameters: f, cx, cy
        f.write(f'1 SIMPLE_PINHOLE {w} {h} {fx} {cx} {cy}\n')
    
    # the blender to opencv transformation matrix
    blender2opencv = np.array([
        [1, 0, 0, 0],
        [0, -1, 0, 0],
        [0, 0, -1, 0],
        [0, 0, 0, 1]
    ])
    
    # 2. save images.txt
    with open(os.path.join(colmap_dir, 'images.txt'), 'w') as f:
        f.write('# Image list with one line of data per image:\n')
        f.write('#   IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME\n')
        for idx, frame_data in enumerate(trajectory_data, 1):
            # get the camera location and rotation
            location = np.array(frame_data['location'])
            rotation = np.array(frame_data['rotation'].to_matrix())
            
            # create the transformation matrix
            transform_matrix = np.eye(4)
            transform_matrix[:3, :3] = rotation
            transform_matrix[:3, 3] = location
            
            # apply the blender to opencv transformation
            pose = transform_matrix @ blender2opencv
            
            # COLMAP uses the camera coordinate system to the world coordinate system
            # need to invert to get the correct transformation relationship
            R = np.linalg.inv(pose[:3, :3])
            T = -R @ pose[:3, 3]
            
            # calculate the quaternion
            # use trace method to calculate the quaternion
            trace = R[0, 0] + R[1, 1] + R[2, 2]
            q0 = 0.5 * math.sqrt(1 + trace)
            
            # use numpy matrix indexing
            q1 = (R[2, 1] - R[1, 2]) / (4 * q0)
            q2 = (R[0, 2] - R[2, 0]) / (4 * q0)
            q3 = (R[1, 0] - R[0, 1]) / (4 * q0)
            
            # write the camera pose
            image_name = f'frame_{frame_data["frame"]:04d}.png'
            f.write(f'{idx} {q0} {q1} {q2} {q3} {T[0]} {T[1]} {T[2]} 1 {image_name}\n\n')
    
    # 3. create empty points3D.txt
    with open(os.path.join(colmap_dir, 'points3D.txt'), 'w') as f:
        f.write('# 3D point list with one line of data per point:\n')
        f.write('#   POINT3D_ID, X, Y, Z, R, G, B, ERROR, TRACK[] as (IMAGE_ID, POINT2D_IDX)\n')
    
    print(f"COLMAP format files have been saved to: {colmap_dir}")

def export_merged_mesh(output_path):
    """export the merged mesh of all objects in the scene"""
    # create the mesh output directory
    mesh_dir = os.path.join(output_path, 'meshes')
    os.makedirs(mesh_dir, exist_ok=True)
    
    # store the original selection state
    original_active = bpy.context.view_layer.objects.active
    original_selection = {obj: obj.select_get() for obj in bpy.context.scene.objects}
    original_visibility = {obj: obj.hide_viewport for obj in bpy.context.view_layer.objects}

    try:
        # control the visibility of the objects through the view layer
        for obj in bpy.context.view_layer.objects:
            if obj.type in {'CAMERA', 'LIGHT', 'EMPTY'} or not obj.visible_get():
                obj.hide_viewport = True
            else:
                obj.hide_viewport = False

        # export scene mesh
        scene_mesh_path = os.path.join(mesh_dir, "scene_mesh.obj")
        bpy.ops.wm.obj_export(
            filepath=scene_mesh_path,
            export_selected_objects=False,
            export_materials=True,
            export_triangulated_mesh=True,
            export_normals=True,
            export_uv=True,
            export_colors=True
        )
        
        print(f"scene mesh has been exported to: {scene_mesh_path}")
        return scene_mesh_path
        
    finally:
        # restore the original state
        for obj, visibility in original_visibility.items():
            obj.hide_viewport = visibility
        for obj, selected in original_selection.items():
            if obj:
                obj.select_set(selected)
        bpy.context.view_layer.objects.active = original_active

def main():
    args = parse_args()
    # set output path
    output_path = args.output
    print(f"output path: {output_path}")
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    # initialize the scene
    setup_scene(clear_scene=False)
    
    # set camera
    camera = setup_camera()

    # you can set your own camera trajectory in the blender file, and it will be saved in output_path/camera_data.json
    # camera_trajectory = get_camera_trajectory(camera)
    # if camera_trajectory:
    #     print("success to get camera trajectory")
    #     # save the original trajectory data (include camera parameters)
    #     save_camera_trajectory(camera_trajectory, output_path, camera)
    # else:
    #     print("no camera trajectory found, please ensure the camera has been set to the keyframe")

    # load and apply the saved trajectory
    # replace the file path if you want to use your own camera trajectory
    camera_data_file = r'camera_data.json'
    # apply the camera trajectory read from the json file
    camera_trajectory, camera_params = load_camera_trajectory(camera_data_file)
    apply_camera_trajectory(camera, camera_trajectory, camera_params)

    if args.test:
        bpy.context.scene.frame_end = 3

    # save COLMAP format
    save_colmap_format(camera_trajectory, output_path, camera)
    
    # set environment lighting
    hdri_path = args.hdri
    setup_world_lighting(output_path, hdri_path)
    
    export_merged_mesh(output_path)

    print(f'start rendering, frame number: {bpy.context.scene.frame_end - bpy.context.scene.frame_start + 1}')

    # render raw images and blender normals (not use these normals)
    render_dataset(output_path)
    
    # render depth
    render_dataset_depth(output_path)

    # render mask, the max of mask is not 255, a little bit small, so we need to convert it to 0 or 255 later
    render_dataset_mask(output_path)
    
if __name__ == "__main__":
    main()