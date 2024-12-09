import ffmpeg
import subprocess


def frames_to_video(input_path, output_path, framerate):
    """
    Converts a sequence of images in input_path to a video file at output_path with specified framerate.
    Args:
    - input_path (str): Path to input frames with a wildcard for numbering (e.g., './frames/frame_%04d.png').
    - output_path (str): Path to output video file (e.g., './output/video.mp4').
    - framerate (int): Frame rate of the output video.
    """
    (
        ffmpeg
        .input(input_path, framerate=framerate, pattern_type='glob')
        .output(output_path, pix_fmt='yuv420p', vcodec='libx264', vf='pad=ceil(iw/2)*2:ceil(ih/2)*2')
        .overwrite_output()  # Ensures existing output files are overwritten
        .run(capture_stdout=True, capture_stderr=True)
    )


def frames_to_gif(input_path, output_path, framerate):
    """
    Converts a sequence of images in input_path to a GIF file at output_path with specified framerate.
    Args:
    - input_path (str): Path to input frames with a wildcard for numbering (e.g., './frames/frame_%04d.png').
    - output_path (str): Path to output GIF file (e.g., './output/animation.gif').
    - framerate (int): Frame rate of the output GIF.
    """
    palette_path = f'/tmp/blender/palette.png'
    # Step 1: Generate palette
    (
        ffmpeg
        .input(input_path, framerate=framerate, pattern_type='glob')
        .output(palette_path, vf='palettegen')
        .overwrite_output()  # Ensures existing output files are overwritten
        .run(capture_stdout=True, capture_stderr=True)
    )

    # Step 2: Create GIF using palette without dithering
    command = f'ffmpeg -framerate {framerate} -pattern_type glob -i "{input_path}" -i "{palette_path}" -filter_complex "paletteuse=dither=none" -y "{output_path}"'
    subprocess.run(command, shell=True, stdout=subprocess.DEVNULL,
                   stderr=subprocess.DEVNULL)


def main():
    image_dir = 'debug/videos'
    mode = 'online_plus'
    fps = 15
    print(f"Converting to video...")
    frames_to_video(f'{image_dir}/frames/{mode}*.png',
                    f'{image_dir}/{mode}.mp4', fps)
    print(f"Converting to gif...")
    frames_to_gif(f'{image_dir}/frames/{mode}*.png',
                  f'{image_dir}/{mode}.gif', fps)


if __name__ == '__main__':
    main()
