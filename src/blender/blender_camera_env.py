import random
import datetime
import os
import time
import shutil
import bpy
import bmesh
from mathutils.bvhtree import BVHTree
from PIL import Image
import tqdm
import gymnasium as gym
from gymnasium.spaces import Box, Dict, Discrete, MultiBinary, MultiDiscrete
import numpy as np
import torch
import torchvision.transforms as T
from transforms3d.quaternions import qinverse, qconjugate, qmult, qnorm, quat2mat, mat2quat, quat2axangle, axangle2quat, nearly_equivalent
from transforms3d.euler import euler2quat, quat2euler, euler2mat, mat2euler
from src.blender.render import cycles_settings, configure_blender, render_blender_image
from src.blender.frames_to_video import frames_to_video, frames_to_gif
from src.data.state_action_conversion import get_states_actions, reverse_states_actions
from src.utils.quaternion_operations import convert_to_global_frame, convert_to_local_frame, interpolate_eulers, interpolate_tvecs

# What is BVHTree?
#     BVHTree: In Blender's Python API, BVHTree is a class that represents a
#     bounding volume hierarchy. You can create a BVHTree from mesh data,
#     and then use it to perform various spatial queries more efficiently
#     than directly using the scene's data structures.
# Benefits of Using BVHTree
#     Efficiency: BVH trees allow for fast culling of objects that are not
#     relevant to a specific query (e.g., those that are not in the path of
#     a ray). This can significantly reduce the number of checks needed,
#     making queries faster, especially in complex scenes.
#     Precomputation: Once a BVH tree is built, it can be reused for
#     multiple queries without having to reprocess the mesh data, further
#     improving performance.


def create_scene_bvhtree(scene):
    # Create a new BMesh
    bm = bmesh.new()
    # Iterate over all mesh objects in the scene
    for obj in scene.objects:
        if obj.type == 'MESH':
            # skip the CameraRigs objects in InifiGen scenes
            if 'camera' in obj.name.lower():
                continue
            # skip the colliders
            if 'collider' in obj.name.lower():
                # continue
                pass
            # skip the atmosphere
            if 'atmosphere' in obj.name.lower():
                continue
            # Duplicate the object's mesh to avoid modifying the original
            mesh = obj.data.copy()
            # Transform the vertices to world space
            mesh.transform(obj.matrix_world)
            # Load the mesh data into the BMesh
            bm.from_mesh(mesh)
            # print(f'add {len(mesh.edges)} edges, total {len(bm.edges)} edges')
            # Free the temporary mesh
            bpy.data.meshes.remove(mesh)
    # Create the BVHTree from the combined BMesh
    bvhtree = BVHTree.FromBMesh(bm)
    # go through all vertices in the bmesh to find the border
    min_coord = np.array([np.inf, np.inf, np.inf])
    max_coord = np.array([-np.inf, -np.inf, -np.inf])
    for v in bm.verts:
        min_coord = np.minimum(min_coord, v.co)
        max_coord = np.maximum(max_coord, v.co)
    # Free the BMesh
    bm.free()
    return bvhtree, (min_coord, max_coord)


def detect_collision(bvh, location, rotation, distance_thres):
    # Define unit vectors for directions in camera's local space
    dir_names = ['Front', 'Right', 'Left', 'Up', 'Down', 'Back']
    # the camera's front is in the negative Z in camera space
    directions = [
        np.array([0, 0, -1]),  # Front
        np.array([1, 0, 0]),  # Right
        np.array([-1, 0, 0]),  # Left
        np.array([0, 1, 0]),  # Up
        np.array([0, -1, 0]),  # Down
        np.array([0, 0, 1]),  # Back
    ]

    # Camera's rotation matrix
    rotation_matrix = euler2mat(*rotation, axes='sxyz')

    # Apply rotation to direction vectors
    directions = [(rotation_matrix @ d) for d in directions]

    # Perform ray casts in each direction
    collisions = {}
    for i in range(len(directions)):
        name, direction = dir_names[i], directions[i]
        hit_location, hit_normal, hit_index, hit_distance = bvh.ray_cast(
            location, direction)
        if hit_distance is None:
            continue
        if hit_distance < distance_thres:
            # if np.dot(direction, hit_normal) > 0:
            #     # back face culling (ignore the collision if the normal is in the same direction as the ray)
            #     continue
            # if scene_extreme_coords is not None:
            #     (min_coord, max_coord) = scene_extreme_coords
            #     hit_location = np.array(hit_location)
            #     # check if the hit location is within the scene
            #     # hit location might be near the border due to potential errors in the BVH tree
            #     if np.any(hit_location < min_coord + 1) or np.any(hit_location > max_coord - 1):
            #         continue
            # print(f"Collision detected: {name} - {hit_distance} meters.")
            collisions[name] = hit_distance

    return collisions

# https://pdfs.semanticscholar.org/c7bc/ddc8b384f0eb0867d2904b45ab8ea9482c1b.pdf
# COLMAP Coordinate System:
# COLMAP uses a right-handed coordinate system where:
#     X-axis is positive to the right.
#     Y-axis is positive downwards.
#     Z-axis is positive into the scene (similar to the typical computer vision and photogrammetry conventions).

# Blender Coordinate System:
# Blender, on the other hand, uses a right-handed coordinate system but with different axis orientations:
#     X-axis is positive to the right.
#     Y-axis is positive forward.
#     Z-axis is positive upwards (common in many 3D modeling and animation software).

# Object Rotation:
# When an object is rotated, a rotation matrix is typically applied directly. This matrix transforms the object's coordinates
# from their original position to a new position according to the defined rotation. For instance, a 30-degree clockwise rotation
# around an axis would involve applying a corresponding rotation matrix to all points constituting the object.

# Coordinate Axes Rotation:
# On the other hand, rotating the axes involves transforming the coordinate system itself, not the object. If the axes are rotated
# 30 degrees clockwise, to express the original coordinates in this new system, they need to be transformed as if they were rotated
# 30 degrees counterclockwise. This is because the transformation needs to counteract the rotation of the axes to maintain the
# original orientation of the object relative to the new axes. This counter rotation is why the inverse of the normal rotation matrix
# is used when rotating the axes. Mathematically, if 'R' is the rotation matrix for rotating an object, the matrix for rotating the
# axes by the same angle in the opposite direction is 'R^-1'.

# blender axes rotate negative 90 degrees around the X-axis to convert to the COLMAP coord system
# so for the objects, the inverse, positive 90 degrees around the X-axis, is needed to convert to the COLMAP coord system


R_colmap_from_blender = euler2mat(np.pi/2, 0, 0, axes='sxyz')
R_blender_from_colmap = euler2mat(-np.pi/2, 0, 0, axes='sxyz')


# COLMAP Camera Direction:
# camera front f is positive along the Z-axis
# camera upwards u is negative along the Y-axis
# convert f, u to camera x (right), y (down), z (in) axes in the COLMAP convention
# x_colmap^camera = X_colmap
# y_colmap^camera = Y_colmap
# z_colmap^camera = Z_colmap
# no rotation needed between the camera default direction and the COLMAP coord system


# Blender Camera Direction:
# camera front f is negative along the Z-axis
# camera upwards u is positive along the Y-axis
# convert f, u to camera x (right), y (down), z (in) axes in the Blender convention
# x_blender^camera = X_blender
# y_blender^camera = -Y_blender
# z_blender^camera = -Z_blender
# rotate 180 degrees around the X-axis to convert the camera default direction to the Blender coord system
# IMPORTANT: the rotation angles in Blender are all in the original Blender world plane, rotating the camera direction as an object

R_blender_cam_dir = euler2mat(np.pi, 0, 0, axes='sxyz')


class BlenderCameraEnv(gym.Env):
    """Custom Environment that follows gym interface"""

    original_fps = 15  # original frame rate used for colmap reconstruction

    def __init__(self, scene_fpath, fps=3, action_fps=15,
                 run_dir='debug/videos',
                 video_duration=10, resolution=(180, 320), motion_option='global',
                 cropped_sensor_width=36, drone_type=1, seed=None):
        super(BlenderCameraEnv, self).__init__()

        t0 = time.time()

        self.scene_fpath = scene_fpath
        self.run_dir = run_dir
        # frame rate for the auto-regressive task tuples (image, state, actions)
        self.fps = fps
        # more action tokens for each tuple (image, state, actions) if original_fps > fps
        assert self.original_fps % fps == 0
        self.fps_downsample = int(self.original_fps / fps)
        self.action_fps = action_fps
        self.action_downsample = self.original_fps // action_fps
        self.n_action_to_predict = self.fps_downsample // self.action_downsample
        self.video_duration = video_duration
        # max number of original frames
        self.max_data_frames = video_duration * action_fps

        self.resolution = resolution  # h, w

        self.motion_option = motion_option
        # state: tvec, qvec (all in global reference frame)
        self.state_dim = 7
        # action: v, omega (global/local, relative to the current frame)
        self.action_dim = 6

        self.t0 = None
        self.run_stepped = False

        # DJI Mavic 3 Pro Specifications
        # Hasselblad Camera
        #     FOV: 84°
        #     Format Equivalent: 24mm
        # Medium Tele Camera
        #     FOV: 35°
        #     Format Equivalent: 70mm
        # Tele Camera
        #     FOV: 15°
        #     Format Equivalent: 166mm
        # Max Ascent Speed
        #     8 m/s
        # Max Descent Speed
        #     6 m/s
        # Max Horizontal Speed (at sea level, no wind)
        #     21 m/s

        # DJI Avata 2 Specifications
        # Camera
        #     FOV: 155°
        #     Format Equivalent: 12 mm
        # Max Ascent Speed
        #     6 m/s (Normal mode)
        #     9 m/s (Sport mode)
        # Max Descent Speed
        #     6 m/s (Normal mode)
        #     9 m/s (Sport mode)
        # Max Horizontal Speed (near sea level, no wind)
        #     8 m/s (Normal mode)
        #     16 m/s (Sport mode)
        #     27 m/s (Manual mode)*

        # scaling on top of drone speed
        # fpv max speed of 16 m/s = 1.0 m/frame at 15 fps
        # non-fpv max speed of 8 m/s = 0.5 m/frame at 15 fps
        self.coord_multipliers = {0: 0.5,  # 0.5 m/frame for non-fpv drones
                                  1: 1.0  # 1 m/frame for fpv drones
                                  }

        # camera focal length in mm
        self.camera_configs = {0: {'value': [24, 70, 166],
                                   'prob': [1, 0, 0]  # [0.85, 0.1, 0.05]
                                   },  # non-fpv drones
                               1: {'value': [12],
                                   'prob': [1]
                                   },  # fpv drones
                               }
        # camera sensor width in mm
        self.cropped_sensor_width = cropped_sensor_width

        # Define action and observation space
        # 6 dof camera movements for v and omega (global/local coordinate system)
        self.action_space = Box(low=-1, high=1, shape=(6,), dtype=np.float64)
        # 7 dof camera state for tvec, qvec (global coordinate system)
        self.observation_space = Dict({
            'image': Box(low=0, high=255, shape=(resolution[0], resolution[1], 3), dtype=np.uint8),
            'state': Box(low=-np.inf, high=np.inf, shape=(7,), dtype=np.float64),
        })

        self.drone_type = drone_type  # 0: non-fpv, 1: fpv

        # seed
        self.seed = seed

        # blender
        bpy.ops.wm.open_mainfile(filepath=self.scene_fpath)
        scene = bpy.context.scene
        # print('creating BVH tree...')
        self.scene_bvh, self.scene_extreme_coords = create_scene_bvhtree(scene)
        # collision detection
        self.collision_thres = 0.5

        # configure blender settings
        self.render_settings = 'online_plus'
        self.is_final_render = False
        np_random_generator = np.random.default_rng(seed)
        self.focal_length = np_random_generator.choice(self.camera_configs[self.drone_type]['value'],
                                                       p=self.camera_configs[self.drone_type]['prob'])
        self.configure_cycles_settings()

        # reset environment state
        # 'raw' blender coordinates
        # in blender convention
        self.raw_locs, self.raw_rots = np.zeros(
            [0, 3]), np.zeros([0, 3])  # in radians
        # in colmap convention
        self.raw_tvec0, self.raw_qvec0 = None, None
        # 'global' coordinates (w.r.t. the initial frame)
        # in colmap convention
        self.tvecs, self.qvecs = np.zeros([0, 3]), np.zeros([0, 4])
        self.vs, self.omegas = np.zeros([0, 3]), np.zeros([0, 3])

        bpy_objects = bpy.data.objects.keys()
        if 'CameraRigs/0' in bpy_objects:
            default_camera_rig = bpy.data.objects['CameraRigs/0']
            loc = np.array(default_camera_rig.location)
            rot = np.array(default_camera_rig.rotation_euler)
        # default camera location and rotation from other Blender scenes
        elif 'Camera' in bpy_objects:
            default_camera = bpy.data.objects['Camera']
            loc = np.array(default_camera.location)
            rot = np.array(default_camera.rotation_euler)
        self.default_loc, self.default_rot = loc, rot

        print(f'env initialized: {time.time() - t0:.2f} seconds')

    def configure_cycles_settings(self, final_render_fps=None):
        custom_settings = cycles_settings[self.render_settings].copy()
        if not self.is_final_render:
            custom_settings['fps'] = self.fps
            custom_settings['resolution'] = self.resolution
            # only resize the sensor width in step() and reset() to simulate the center crop
            custom_settings['sensor_width'] = self.cropped_sensor_width
            # settings to avoid the denoising artifacts under low fps
            # custom_settings['denoiser'] = None
            custom_settings['denoiser'] = 'OPENIMAGEDENOISE'
            # custom_settings['use_persistent_data'] = False
            # custom_settings['use_incremental'] = False
        else:
            if final_render_fps is not None:
                custom_settings['fps'] = final_render_fps
            else:
                custom_settings['fps'] = self.original_fps
            # custom_settings['min_samples'] = 0
            # custom_settings['adaptive_threshold'] = 0
        custom_settings['focal_length'] = self.focal_length
        configure_blender(**custom_settings)
        # camera intrinsic parameters
        camera = bpy.data.objects['Camera']
        # camera parameters in millimeters
        focal_length = camera.data.lens
        sensor_width = camera.data.sensor_width
        # image resolution in pixels
        h, w = bpy.context.scene.render.resolution_y, bpy.context.scene.render.resolution_x
        fx = fy = focal_length / sensor_width * w
        cx, cy = w / 2, h / 2
        self.intrinsics = fx, fy, cx / 2, cy / 2

    def reset(self, seed=None, random_init_pose=False):
        self.t0 = time.time()
        self.run_stepped = False
        if seed is None:
            seed = self.seed
        else:
            self.seed = seed
        np_random_generator = np.random.default_rng(seed)

        # reset environment state
        # 'raw' blender coordinates
        # in blender convention
        self.raw_locs, self.raw_rots = np.zeros(
            [0, 3]), np.zeros([0, 3])  # in radians
        # in colmap convention
        self.raw_tvec0, self.raw_qvec0 = None, None
        # 'global' coordinates (w.r.t. the initial frame)
        # in colmap convention
        self.tvecs, self.qvecs = np.zeros([0, 3]), np.zeros([0, 4])
        self.vs, self.omegas = np.zeros([0, 3]), np.zeros([0, 3])

        scene_x_min, scene_y_min, scene_z_min = self.scene_extreme_coords[0]
        scene_x_max, scene_y_max, scene_z_max = self.scene_extreme_coords[1]
        # default camera location and rotation
        loc, rot = np.copy(self.default_loc), np.copy(self.default_rot)

        # randomize camera location and rotation
        if random_init_pose:
            collision_flag = True
            scene_w, scene_h = scene_x_max - scene_x_min, scene_y_max - scene_y_min
            while collision_flag:
                x = np_random_generator.uniform(
                    scene_x_min + scene_w / 8, scene_x_max - scene_w / 8)
                y = np_random_generator.uniform(
                    scene_y_min + scene_h / 8, scene_y_max - scene_h / 8)
                # location
                if self.drone_type == 0:
                    z = np_random_generator.uniform(
                        loc[2] + 25, loc[2] + max(50, 0.75 * scene_z_max - loc[2]))
                else:
                    z = np_random_generator.uniform(loc[2] - 2, loc[2] + 5)
                loc = np.array([x, y, z])
                # rotation
                # x, y, z -> pitch, roll, yaw
                # default camera direction is downwards, along the negative Z-axis
                if self.drone_type == 0:
                    rot = np.concatenate([
                        np_random_generator.uniform(
                            np.pi * 75 / 180, np.pi * 90 / 180, 1),
                        np_random_generator.normal(0, np.pi * 0.01, 1),
                        np_random_generator.uniform(-np.pi, np.pi, 1)])
                else:
                    rot = np.concatenate([
                        np_random_generator.uniform(
                            np.pi * 75 / 180, np.pi * 105 / 180, 1),
                        np_random_generator.normal(0, np.pi * 0.01, 1),
                        np_random_generator.uniform(-np.pi, np.pi, 1)])
                # rot = np.deg2rad([90, 0, 30])

                # collision detection
                crashes = detect_collision(self.scene_bvh, loc, rot,
                                           1000)
                collision_flag = False
                if 'Front' in crashes:
                    if crashes['Front'] < 100:
                        collision_flag = True
                if 'Left' in crashes:
                    if crashes['Left'] < 10:
                        collision_flag = True
                if 'Right' in crashes:
                    if crashes['Right'] < 10:
                        collision_flag = True
                if 'Up' in crashes:
                    if crashes['Up'] < 10:
                        collision_flag = True
                    # respawn location is beneath the ground
                    if 'Up' in crashes and 'Down' not in crashes:
                        collision_flag = True
                if 'Down' in crashes:
                    if crashes['Down'] < 5:
                        collision_flag = True
                # respawn location is inside some object
                if len(crashes) == 6:
                    collision_flag = True
            pass
        else:
            # for non-fpv drones, the camera is at a higher altitude
            if self.drone_type == 0:
                loc += np.array([0, 0, 20])
                rot[0] = np.pi * 80 / 180

        # Reset the camera to the default position and orientation
        camera = bpy.data.objects['Camera']
        camera.location = loc
        camera.rotation_euler = rot

        # 'raw' blender coordinates
        # in blender convention
        self.raw_locs = np.concatenate([self.raw_locs, loc[None]], axis=0)
        self.raw_rots = np.concatenate([self.raw_rots, rot[None]], axis=0)
        # in colmap convention
        tvec0 = R_colmap_from_blender @ np.array(loc)
        # R_rot is for rotating the blender defualt camera direction in the blender world plane
        R_rot = euler2mat(*rot, axes='sxyz')
        # apply the global (blender world plane) rotation to the blender defualt camera direction
        # R_blender is for rotating the blender world plane to the actual camera direction
        R_blender = R_rot @ R_blender_cam_dir
        # R_colmap is for rotating the colmap world plane to the actual camera direction
        R_colmap = R_colmap_from_blender @ R_blender
        qvec0 = mat2quat(R_colmap)
        self.raw_tvec0, self.raw_qvec0 = tvec0, qvec0

        # 'global' coordinates (w.r.t. the initial frame)
        tvec, qvec, _, _ = convert_to_local_frame(
            self.raw_tvec0, self.raw_qvec0,
            tvec0, qvec0, None, None)
        self.tvecs = np.concatenate([self.tvecs, tvec[None]], axis=0)
        self.qvecs = np.concatenate([self.qvecs, qvec[None]], axis=0)

        # Render the image
        self.render_settings = 'online_plus'
        self.is_final_render = False
        self.focal_length = np_random_generator.choice(self.camera_configs[self.drone_type]['value'],
                                                       p=self.camera_configs[self.drone_type]['prob'])
        self.configure_cycles_settings()

        frame_id = len(self.raw_locs)
        shutil.rmtree(f'{self.run_dir}/frames', ignore_errors=True)
        os.makedirs(f'{self.run_dir}/frames')
        fpath = f'{self.run_dir}/frames/{self.render_settings}_{frame_id:03d}.png'
        image = render_blender_image(fpath, frame_id)
        state = np.concatenate([tvec, qvec])
        observation = {'image': np.array(image), 'state': state, }
        info = {'crash': None,
                'seq_len': (len(self.tvecs) - 1) / self.action_fps, }
        return observation, info

    def step(self, actions, stop=False):
        self.run_stepped = True
        # Apply action
        # reverse state and actions
        # tvec(t+1), qvec(t+1), v(t), omega(t)
        state = np.concatenate([self.tvecs[-1], self.qvecs[-1]])
        # sparse actions was initially shrinked to 1/action_downsample
        actions = actions * self.action_downsample
        next_tvecs, next_qvecs, vs, omegas = [], [], [], []
        for i in range(self.n_action_to_predict):
            next_tvec, next_qvec, v, omega = reverse_states_actions(
                state[None], actions[[i]],
                motion_option=self.motion_option)
            next_tvecs.extend(next_tvec)
            next_qvecs.extend(next_qvec)
            vs.extend(v)
            omegas.extend(omega)
            # Update the state of the environment
            state = np.concatenate([next_tvec[0], next_qvec[0]])

        # 'global' coordinates (w.r.t. the initial frame)
        self.tvecs = np.concatenate([self.tvecs, np.array(next_tvecs)], axis=0)
        self.qvecs = np.concatenate([self.qvecs, np.array(next_qvecs)], axis=0)
        self.vs = np.concatenate([self.vs, np.array(vs)], axis=0)
        self.omegas = np.concatenate([self.omegas, np.array(omegas)], axis=0)

        # 'raw' blender coordinates
        locs = np.zeros([len(next_tvecs) + 1, 3])
        rots = np.zeros([len(next_tvecs) + 1, 3])
        locs[0] = self.raw_locs[-1]
        rots[0] = self.raw_rots[-1]
        for i in range(len(next_tvecs)):
            # in colmap convention
            # include the speed multiplier to the tvec and speed
            raw_tvec_multiplied, raw_qvec, _, _ = convert_to_global_frame(
                self.raw_tvec0, self.raw_qvec0,
                next_tvecs[i] * self.coord_multipliers[self.drone_type],
                next_qvecs[i], None, None)
            # in blender convention
            loc = R_blender_from_colmap @ raw_tvec_multiplied
            # R_colmap is for rotating the colmap world plane to the actual camera direction
            R_colmap = quat2mat(raw_qvec)
            # R_blender is for rotating the blender world plane to the actual camera direction
            R_blender = R_blender_from_colmap @  R_colmap
            # retrieve the global rotation (R_rot) needed for rotating the blender default camera direction
            # R_rot @ R_bcam = R_blender
            R_rot = R_blender @ R_blender_cam_dir.T
            # euler angles conversion
            rot = mat2euler(R_rot, axes='sxyz')
            locs[i + 1] = loc
            rots[i + 1] = rot

        # interpolate the locs and rots to 30 fps
        times = np.arange(0, len(actions) + 1)
        new_times = np.arange(0, len(actions) + 1e-8,
                              self.action_fps / 30)[1:]
        _locs = interpolate_tvecs(locs, times, new_times)
        _rots = interpolate_eulers(rots, times, new_times)
        self.raw_locs = np.concatenate([self.raw_locs, _locs], axis=0)
        self.raw_rots = np.concatenate([self.raw_rots, _rots], axis=0)

        # collision detection
        crash = None
        for i in range(len(_locs)):
            loc = _locs[i]
            rot = _rots[i]
            _crashes = detect_collision(self.scene_bvh, loc, rot,
                                        self.collision_thres)
            # only record the initial crash reason
            if len(_crashes) > 0 and crash is None:
                crash = list(_crashes.keys())[0]
                print(f'crash detected: {crash}')

        # Set the camera location and rotation
        camera = bpy.data.objects['Camera']
        camera.location = loc
        camera.rotation_euler = rot

        # Render the image
        if self.render_settings != 'online_plus' or self.is_final_render:
            print('this should not happen')
            # self.render_settings = 'online_plus'
            # self.is_final_render = False
            # self.configure_cycles_settings()

        frame_id = len(self.raw_locs)
        fpath = f'{self.run_dir}/frames/{self.render_settings}_{frame_id:03d}.png'
        image = render_blender_image(fpath, frame_id)

        # Update the observation
        state = np.concatenate([self.tvecs[-1], self.qvecs[-1]])
        observation = {'image': np.array(image), 'state': state, }

        # Check if the episode is done
        # https://farama.org/Gymnasium-Terminated-Truncated-Step-API
        # To prevent an agent from wandering in circles forever, not doing
        # anything, and for other practical reasons, Gym lets environments have
        # the option to specify a time limit that the agent must complete the
        # environment within. Importantly, this time limit is outside of the
        # agent’s knowledge as it is not contained within their observations.
        # Therefore, when the agent reaches this time limit, the environment
        # should be reset but should not be treated the same as if the agent
        # reaches a goal and the environment ends. We refer to the first type as
        # truncation, when the agent reaches the time limit (maximum number of
        # steps) for the environment, and the second type as termination, when
        # the environment state reaches a specific condition (i.e. the agent
        # reaches the goal).

        # in our case, the specific condition for termination is
        # 1. the maximum number of frames is reached
        # 2. the stop signal is received
        # 3. the collision is detected
        # and the time limit is the maximum number of frames is the forementioned
        # condition 1

        truncated = len(self.tvecs) > self.max_data_frames
        terminated = truncated or stop or (crash is not None)

        # Calculate reward
        if crash is not None:
            reward = -10
        else:
            reward = 1 / self.fps

        # Return step information
        info = {'crash': crash,
                'seq_len': (len(self.tvecs) - 1) / self.action_fps, }

        return observation, reward, terminated, truncated, info

    def final_render(self, run_name, mode='online_plus', save_mp4=True, save_gif=False, re_render=True):
        cur_datetime = f'{datetime.datetime.today():%Y-%m-%d_%H-%M-%S}'
        # save the configs
        # print('saving the configs...')
        os.makedirs(self.run_dir, exist_ok=True)
        np.savetxt(f'{self.run_dir}/{run_name}_{self.seed}_{cur_datetime}_config.txt',
                   np.concatenate([self.raw_locs, self.raw_rots], axis=1),
                   fmt='%f',
                   header=(f'{self.scene_fpath} fps:{self.fps} action-fps:'
                           f'{self.action_fps} focal-length:{self.focal_length}\n'
                           'locs: x, y, z; rots: pitch, roll, yaw'))

        def save_banner():
            images = []
            for image_fname in sorted(os.listdir(f'{self.run_dir}/frames')):
                frame_id = int(image_fname.split('_')[-1].split('.')[0]) - 1
                if frame_id % (30 // self.fps) == 0:
                    images.append(Image.open(
                        f'{self.run_dir}/frames/{image_fname}'))
            # concatenate the images horizontally
            w, h = images[0].size
            banner = Image.new(
                'RGB', (int(w * 1.01) * len(images), h), (255, 255, 255))
            for i in range(len(images)):
                banner.paste(images[i], (i * int(w * 1.01), 0))
            banner.save(
                f'{self.run_dir}/{run_name}_{self.seed}_{cur_datetime}_{mode}_{self.fps}fps.jpg')

        # save a banner image at self.fps
        if os.path.exists(f'{self.run_dir}/frames') and self.run_stepped:
            save_banner()
        else:
            assert re_render, 'no frames found, please re-render'

        # render
        if re_render:
            if mode in ['online_plus', 'low', 'base']:
                render_fps = 30
            else:
                render_fps = 15
            to_render_ids = np.arange(0, len(self.raw_locs), 30 // render_fps)
            # print('configuring blender...')
            self.render_settings = mode
            self.is_final_render = True
            self.configure_cycles_settings(render_fps)
            # print('rendering...')
            t0 = time.time()
            shutil.rmtree(f'{self.run_dir}/frames', ignore_errors=True)
            for i in to_render_ids:
                loc = self.raw_locs[i]
                rot = self.raw_rots[i]
                camera = bpy.data.objects['Camera']
                camera.location = loc
                camera.rotation_euler = rot

                fpath = f'{self.run_dir}/frames/{mode}_{i + 1:03d}.png'
                render_blender_image(fpath, i + 1)
            print(f'{f"run time: {t0 - self.t0:.2f}s, " if self.run_stepped  else ""}'
                  f're-rendering time: {time.time() - t0:.2f}s')
        else:
            render_fps = self.fps
            print(f'run time: {time.time() - self.t0:.2f}s')

        if save_mp4:
            # print(f"converting to video...")
            frames_to_video(f'{self.run_dir}/frames/{mode}*.png',
                            f'{self.run_dir}/{run_name}_{self.seed}_{cur_datetime}_{mode}.mp4',
                            render_fps)
        if save_gif:
            # print(f"converting to gif...")
            frames_to_gif(f'{self.run_dir}/frames/{mode}*.png',
                          f'{self.run_dir}/{run_name}_{self.seed}_{cur_datetime}_{mode}.gif',
                          render_fps)
        save_banner()
        return

    def load(self, fpath):
        self.t0 = time.time()
        self.run_stepped = False
        # load the scene
        with open(fpath, 'r') as f:
            line = f.readline().replace('#', '').strip()
        self.scene_fpath = line.split(' ')[0]
        bpy.ops.wm.open_mainfile(filepath=self.scene_fpath)

        bpy_objects = bpy.data.objects.keys()
        if 'CameraRigs/0' in bpy_objects:
            default_camera_rig = bpy.data.objects['CameraRigs/0']
            loc = np.array(default_camera_rig.location)
            rot = np.array(default_camera_rig.rotation_euler)
        # default camera location and rotation from other Blender scenes
        elif 'Camera' in bpy_objects:
            default_camera = bpy.data.objects['Camera']
            loc = np.array(default_camera.location)
            rot = np.array(default_camera.rotation_euler)
        self.default_loc, self.default_rot = loc, rot

        # load the configs
        try:
            self.fps = int(line.split(' ')[1].replace('fps:', ''))
            self.action_fps = int(line.split(
                ' ')[2].replace('action-fps:', ''))
            self.focal_length = int(line.split(
                ' ')[3].replace('focal-length:', '')) if len(line.split(' ')) > 3 else 12
        except:
            pass
        # only load the raw_locs and raw_rots
        loc_rot = np.loadtxt(fpath)
        self.raw_locs = loc_rot[:, :3]
        self.raw_rots = loc_rot[:, 3:]
        # set everything to the initial state
        self.raw_tvec0, self.raw_qvec0 = None, None
        self.tvecs, self.qvecs = np.zeros([0, 3]), np.zeros([0, 4])
        self.vs, self.omegas = np.zeros([0, 3]), np.zeros([0, 3])


def main():
    import re
    import pandas as pd

    from transforms3d.quaternions import qinverse, qmult, qnorm, quat2mat, mat2quat, quat2axangle, axangle2quat, nearly_equivalent
    from src.utils.flexible_fs import FlexibleFileSystem
    from src.utils.quaternion_operations import add_angular_velocity_to_quaternion

    motion_option = 'local'
    fps = 3
    original_fps = 15
    fps_downsample = original_fps // fps
    action_fps = 15
    action_downsample = original_fps // action_fps
    n_action_to_predict = fps_downsample // action_downsample
    # env
    env = BlenderCameraEnv(scene_fpath='blosm/himeji/scene.blend',
                           action_fps=action_fps,
                           motion_option=motion_option,
                           run_dir='debug/videos',
                           #    resolution=(180, 180), cropped_sensor_width=36/16*9
                           )
    # return

    # Example of using the environment
    obs, info = env.reset(seed=0, random_init_pose=True)

    # Test the state and action conversion functions
    root, filter_results_path = 'youtube_drone_videos', 'dataset_mini.h5'
    fps_downsample = 5

    result_fpaths = []
    h5_fs = FlexibleFileSystem(
        f'{root}/{filter_results_path}')
    for video_id in sorted(h5_fs.listdir(root)):
        for result_fname in sorted(h5_fs.listdir(f'{root}/{video_id}')):
            if '-score' in result_fname and result_fname.endswith('.csv'):
                score = int(re.search(r'-score(\d+)',
                                      result_fname).group(1))
                valid = '_invalid' not in result_fname
                if score and valid:
                    result_fpath = f'{root}/{video_id}/{result_fname}'
                    result_fpaths.append(result_fpath)

    # data_index = np.random.randint(len(result_fpaths))
    data_index = 0
    print(data_index, result_fpaths[data_index])
    with h5_fs.open(result_fpaths[data_index], 'r') as f:
        recons_df = pd.read_csv(f, comment='#')

    recons_array = recons_df.to_numpy()
    # camera path in global coord system (measurements)
    raw_tvecs = recons_array[:, 1:4].astype(float)
    raw_qvecs = recons_array[:, 4:8].astype(float)
    raw_vs = recons_array[:, 8:11].astype(float)
    raw_omegas = recons_array[:, 11:14].astype(float)
    # add the final speed and angular velocity to extend the sequence
    final_tvec = raw_tvecs[-1] + raw_vs[-1]
    final_qvec = add_angular_velocity_to_quaternion(
        raw_qvecs[-1], raw_omegas[-1], 1)
    raw_tvecs = np.concatenate([raw_tvecs, final_tvec[None]], axis=0)
    raw_qvecs = np.concatenate([raw_qvecs, final_qvec[None]], axis=0)
    # change the global coord system to the initial frame
    tvecs = np.zeros_like(raw_tvecs)
    qvecs = np.zeros_like(raw_qvecs)
    vs = np.zeros_like(raw_vs)
    omegas = np.zeros_like(raw_omegas)
    # change the global coord system to the initial frame
    for i in range(len(raw_tvecs)):
        tvecs[i], qvecs[i], _, _ = convert_to_local_frame(
            raw_tvecs[0], raw_qvecs[0],
            raw_tvecs[i], raw_qvecs[i])
    for i in range(len(raw_vs)):
        _, _, vs[i], omegas[i] = convert_to_local_frame(
            raw_tvecs[0], raw_qvecs[0],
            None, None, raw_vs[i], raw_omegas[i])
    # sequence length
    seq_length = len(recons_array) // fps_downsample
    # time steps (same length as images/states)
    time_steps = np.arange(seq_length)

    total_reward = 0
    time_range = (time_steps[:, None] * fps_downsample +
                  np.arange(n_action_to_predict))
    states, actions = get_states_actions(
        tvecs, qvecs,
        motion_option=motion_option,
        action_downsample=action_downsample)
    # include the last state
    states = states[time_range // action_downsample]
    actions = actions[time_range // action_downsample]
    for i in (time_steps).tolist():
        # State and Action
        state = states[i]
        action = actions[i]
        stop = bool(i >= time_steps[-1])
        obs, reward, terminated, truncated, info = env.step(action, stop=stop)
        print(np.abs(env.tvecs[i * fps_downsample // action_downsample:(i + 1) * fps_downsample // action_downsample] -
                     tvecs[i * fps_downsample:i * fps_downsample + fps_downsample // action_downsample]).max(),
              np.abs(env.qvecs[i * fps_downsample // action_downsample:(i + 1) * fps_downsample // action_downsample] -
                     qvecs[i * fps_downsample:i * fps_downsample + fps_downsample // action_downsample]).max(),
              )
        total_reward += reward

    env.final_render('debug', mode='online_plus')
    env.close()


if __name__ == "__main__":
    main()
