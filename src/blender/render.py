import os
import sys
from contextlib import contextmanager
import bpy
import warnings
import logging
import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)


# https://blender.stackexchange.com/a/270201
@contextmanager
def stdout_redirected(to=os.devnull):
    '''
    import os

    with stdout_redirected(to=filename):
        print("from Python")
        os.system("echo non-Python applications are also supported")
    '''
    fd = sys.stdout.fileno()

    # assert that Python and C stdio write using the same file descriptor
    # assert libc.fileno(ctypes.c_void_p.in_dll(libc, "stdout")) == fd == 1

    def _redirect_stdout(to):
        sys.stdout.close()  # + implicit flush()
        os.dup2(to.fileno(), fd)  # fd writes to 'to' file
        sys.stdout = os.fdopen(fd, 'w')  # Python writes to fd

    with os.fdopen(os.dup(fd), 'w') as old_stdout:
        with open(to, 'w') as file:
            _redirect_stdout(to=file)
        try:
            yield  # allow code to be run with the redirected stdout
        finally:
            _redirect_stdout(to=old_stdout)  # restore stdout.
            # buffering and flags such as
            # CLOEXEC may be different


# COLMAP parameters
# h = 1080  # Image height
# w = 1920  # Image width
# f = 850  # Focal length in pixels
# for 35mm sensor width
# f = 850 / 1920 * 36 = 15.9375

# DJI AVATA 2 parameters
# Image Sensor
# ‌1/1.3-inch image sensor
# Effective Pixels: 12 MP
# Lens
# ‌FOV: 155°
# Format Equivalent: 12 mm
# Aperture: f/2.8
# Focus: 0.6 m to ∞

# Go Pro Hero 12 parameters
# 35mm Equivalent Focal Length
# Min = 12mm
# Max = 39mm

# denoiser options
# https://www.reddit.com/r/blenderhelp/comments/ylifkd/comment/iuz04qk/?utm_source=share&utm_medium=web3x&utm_name=web3xcss&utm_term=1&utm_content=share_button
# default_denoiser = 'OPENIMAGEDENOISE'  # on CPUs, slower but more stable
default_denoiser = 'OPTIX'  # on RTX GPUs, faster but more likely to cause glitches

online_settings = dict(
    resolution=(180, 320),
    num_samples=10,
    sensor_width=36,  # 35mm sensor width
    # time_limit=0,
    min_samples=3,
    adaptive_threshold=0.5,
    denoiser=default_denoiser,
    use_persistent_data=True,
    use_incremental=True,
    exposure=1,
    use_dof=False,
    dof_aperture_fstop=3,
    motion_blur=False,
    motion_blur_shutter=0.15
)

online_plus_settings = dict(
    resolution=(225, 400),
    num_samples=64,
    sensor_width=36,  # 35mm sensor width
    # time_limit=0,
    min_samples=16,
    adaptive_threshold=0.5,
    denoiser=default_denoiser,
    use_persistent_data=True,
    use_incremental=True,
    exposure=1,
    use_dof=False,
    dof_aperture_fstop=3,
    motion_blur=False,
    motion_blur_shutter=0.15
)


low_settings = dict(
    resolution=(1080, 1920),
    num_samples=512,
    sensor_width=36,  # 35mm sensor width
    # time_limit=0,
    min_samples=32,
    adaptive_threshold=0.5,
    denoiser=default_denoiser,
    use_persistent_data=True,
    use_incremental=True,
    exposure=1,
    use_dof=False,
    dof_aperture_fstop=3,
    motion_blur=True,
    motion_blur_shutter=0.15
)


base_settings = dict(
    resolution=(1080, 1920),
    num_samples=1024,
    sensor_width=36,  # 35mm sensor width
    # time_limit=0,
    min_samples=32,
    adaptive_threshold=0.5,
    denoiser=default_denoiser,
    use_persistent_data=True,
    use_incremental=True,
    exposure=1,
    use_dof=False,
    dof_aperture_fstop=3,
    motion_blur=True,
    motion_blur_shutter=0.15
)


cycles_settings = {'online': online_settings,
                   'online_plus': online_plus_settings,
                   'low': low_settings,
                   'base': base_settings,
                   }


def enable_gpu(engine_name='CYCLES'):
    # this code is taken from https://github.com/princeton-vl/infinigen
    compute_device_type = None
    prefs = bpy.context.preferences.addons['cycles'].preferences
    # Use cycles
    bpy.context.scene.render.engine = engine_name
    bpy.context.scene.cycles.device = 'GPU'

    preferences = bpy.context.preferences.addons['cycles'].preferences
    for device_type in preferences.get_device_types(bpy.context):
        preferences.get_devices_for_type(device_type[0])

    for gpu_type in ['OPTIX', 'CUDA']:  # , 'METAL']:
        found = False
        for device in preferences.devices:
            if device.type == gpu_type and (compute_device_type is None or compute_device_type == gpu_type):
                bpy.context.preferences.addons['cycles'].preferences.compute_device_type = gpu_type
                logger.info('Device {} of type {} found and used.'.format(
                    device.name, device.type))
                found = True
                break
        if found:
            break

    # make sure that all visible GPUs are used
    for device in prefs.devices:
        device.use = True
    return prefs.devices


def configure_blender(
        resolution,
        fps=None,
        focal_length=None,
        sensor_width=None,
        num_samples=100,
        # time_limit=0,
        min_samples=32,
        adaptive_threshold=0.5,
        denoiser=False,
        use_persistent_data=True,
        use_incremental=True,
        exposure=1,
        use_dof=False,
        dof_aperture_fstop=3,
        motion_blur=False,
        motion_blur_shutter=0.15):

    # image settings
    bpy.context.scene.render.resolution_x = resolution[1]
    bpy.context.scene.render.resolution_y = resolution[0]
    bpy.context.scene.render.resolution_percentage = 100
    bpy.context.scene.render.image_settings.file_format = 'PNG'
    if fps is not None:
        bpy.context.scene.render.fps = fps

    # render settings
    bpy.context.scene.cycles.samples = num_samples  # i.e. infinity
    bpy.context.scene.cycles.adaptive_min_samples = min_samples
    bpy.context.scene.cycles.adaptive_threshold = adaptive_threshold  # i.e. noise threshold
    # bpy.context.scene.cycles.time_limit = time_limit
    bpy.context.scene.cycles.film_exposure = exposure
    # bpy.context.scene.cycles.volume_step_rate = 0.1
    # bpy.context.scene.cycles.volume_preview_step_rate = 0.1
    # bpy.context.scene.cycles.volume_max_steps = 32
    # bpy.context.scene.cycles.volume_bounces = 4

    bpy.context.scene.render.use_motion_blur = motion_blur
    if motion_blur:
        bpy.context.scene.cycles.motion_blur_position = "START"
        bpy.context.scene.render.motion_blur_shutter = motion_blur_shutter

    # camera settings
    # Set the camera if it's not already specified (not the default camera rig in InifiGen)
    if 'Camera' not in bpy.data.objects:
        bpy.ops.object.camera_add()
    camera = bpy.data.objects['Camera']
    if use_dof is not None:
        camera.data.dof.use_dof = use_dof
        camera.data.dof.aperture_fstop = dof_aperture_fstop
    if focal_length is not None:
        camera.data.lens = focal_length
    if sensor_width is not None:
        camera.data.sensor_width = sensor_width
    bpy.context.scene.camera = camera

    # efficient rendering
    enable_gpu()
    bpy.context.scene.cycles.use_denoising = denoiser is not None
    if denoiser is not None:
        try:
            bpy.context.scene.cycles.denoiser = denoiser
        except Exception as e:
            warnings.warn(f"Cannot use {denoiser} denoiser {e}")
    bpy.context.scene.render.use_persistent_data = use_persistent_data
    bpy.context.scene.cycles.use_incremental = use_incremental


def render_blender_image(fpath=None, frame_number=1):
    if fpath is None:
        fpath = '/tmp/render.png'
    bpy.context.scene.render.filepath = fpath

    # Set the current frame (important for motion blur calculation)
    bpy.context.scene.frame_set(frame_number)

    # Render the image
    # slient this process https://blender.stackexchange.com/a/270201
    with stdout_redirected():
        bpy.ops.render.render(write_still=True)

    image = Image.open(fpath).convert('RGB')
    return image


def main():
    from transforms3d.quaternions import qinverse, qconjugate, qmult, qnorm, quat2mat, mat2quat, quat2axangle, axangle2quat, nearly_equivalent
    from transforms3d.euler import euler2quat, quat2euler, euler2mat, mat2euler

    # Create a new scene, set up objects, cameras, etc., or load a .blend file
    # For example, load a .blend file
    bpy.ops.wm.open_mainfile(
        filepath='blosm/sydney/scene.blend')

    configure_blender(**online_settings)
    render_blender_image()


if __name__ == '__main__':
    main()
