import os
import ffmpeg
import numpy as np
import re
import pandas as pd
from PIL import Image
from io import BytesIO

# args for using cuda / nvdec
# https://github.com/kkroening/ffmpeg-python/issues/284#issuecomment-551093294
ffmpeg_cuda_input_args = {
    "hwaccel": "nvdec",
    "vcodec": "h264_cuvid",
    "c:v": "h264_cuvid"
}


def run_ffmpeg(video_folder, start_s, end_s, fps, frame_target_size, fix_image_width=True, output_folder=None, use_cuda=False, cuda_device_id=0):
    input_args = {}
    if use_cuda:
        input_args.update(ffmpeg_cuda_input_args)
        input_args['hwaccel_device'] = cuda_device_id
    stream = ffmpeg.input(f'{video_folder}/video.mp4',
                          ss=start_s, to=end_s,
                          **input_args)
    stream = ffmpeg.filter(stream, 'fps', fps=fps)
    # Resize the output frames
    # PIL in w, h format
    if fix_image_width:
        # convert the *width* to frame_target_size while keeping the aspect ratio
        stream = ffmpeg.filter(stream, 'scale',
                               f'min({frame_target_size},iw)', -1)
    else:
        # convert the longer side to frame_target_size while keeping the aspect ratio
        stream = ffmpeg.filter(stream, 'scale',
                               frame_target_size, frame_target_size,
                               force_original_aspect_ratio="decrease")
    if output_folder is not None:
        # save to hard drive
        # https://trac.ffmpeg.org/wiki/Encode/MPEG-4
        os.makedirs(output_folder, exist_ok=True)
        stream = ffmpeg.output(stream,
                               f'{output_folder}/frame%04d.jpg',
                               **{'qscale:v': 2})  # loglevel='quiet'
        out, err = ffmpeg.run(stream,
                              capture_stdout=True,
                              capture_stderr=True)
        return out, err
    else:
        # save to memory
        # https://trac.ffmpeg.org/wiki/Encode/MPEG-4
        stream = ffmpeg.output(
            stream, 'pipe:', format='image2pipe', vcodec='mjpeg', **{'qscale:v': 2})
        out, err = ffmpeg.run(stream,
                              capture_stdout=True,
                              capture_stderr=True)

        frames = {}
        index = 1
        start = 0
        while True:
            # Search for start of image (SOI) marker
            soi = out.find(b'\xff\xd8', start)
            # Search for end of image (EOI) marker
            # Include the EOI marker itself
            eoi = out.find(b'\xff\xd9', soi) + 2
            if soi == -1 or eoi == 1:  # No more images found
                break
            # Create a BytesIO object for this JPEG image and load it
            frame = Image.open(BytesIO(out[soi:eoi]))
            frames[f'frame{index:04d}.jpg'] = frame
            # Prepare for the next iteration
            start = eoi
            index += 1

        return frames


def _recons_images_from_ffmpeg(video_folder, start_s, end_s, recons_array,
                               original_fps, fps, img_resolution,
                               fix_image_width=True,
                               save_folder=None,
                               use_cuda=False, cuda_device_id=None,
                               option='selected'):
    '''
    Read images from memory using ffmpeg

    Args:
        video_folder: path to the video folder
        scene_df: scene dataframe
        recons_array: reconstruction array read from the csv
        original_fps: original frame rate used for reconstruction
        fps: frame rate for extracting images
        img_resolution: image resolution (width or the longer side)
        fix_image_width: resize the image to match the width
        save_fpath: save the images to a directory
        use_cuda: use cuda for ffmpeg
        cuda_device_id: cuda device id
        option: frames to extract, 'all' in scene or 'selected' in reconstruction
    Returns:
        image_dict: dictionary of images (if not saving to disk)
    '''
    # more action tokens for each tuple (image, state, actions) if original_fps > fps
    assert original_fps % fps == 0
    fps_downsample = int(original_fps / fps)

    # sequence length
    seq_length = len(recons_array) // fps_downsample
    # time steps (same length as image/state)
    time_steps = np.arange(seq_length)

    # frame extraction
    # option 1: extract all frames at original fps
    if option == 'all':
        image_dict = run_ffmpeg(
            video_folder,
            start_s, end_s,
            original_fps, img_resolution,
            fix_image_width=fix_image_width,
            output_folder=save_folder,
            use_cuda=use_cuda,
            cuda_device_id=cuda_device_id
        )
        if save_folder is None:
            return dict(sorted(image_dict.items()))
        else:
            return None
    # option 2: only extract frames in reconstruction at dataset fps
    elif option == 'selected':
        recons_fnames = list(
            recons_array[time_steps * fps_downsample, 0])
        start_frame_index = int(re.search(r'frame(\d{4})\.jpg',
                                          recons_array[0, 0]).group(1))
        end_frame_index = int(re.search(r'frame(\d{4})\.jpg',
                                        recons_array[-1, 0]).group(1))
        # compensate for the fps difference in running ffmpeg
        # below show the index for the same image
        # ffmpeg frame index starts at 1
        # ffmpeg_frame_skip=5, offset 3 for image indices 1, 2, 3, 4, 5
        # ffmpeg_frame_skip=3, offset 2 for image indices    1, 2, 3
        # ffmpeg_frame_skip=1, offset 1 for image indices       1
        offset = (fps_downsample + 1) / 2
        recons_start_s = (start_s +
                          (start_frame_index - offset) / original_fps)
        recons_end_s = (start_s +
                        (end_frame_index - offset + 0.5) / original_fps)  # extend by 0.5 frame

        image_dict = run_ffmpeg(
            video_folder,
            recons_start_s, recons_end_s,
            fps, img_resolution,
            fix_image_width=fix_image_width,
            output_folder=save_folder,
            use_cuda=use_cuda,
            cuda_device_id=cuda_device_id
        )
        if save_folder is None:
            ffmpeg_fnames = list(image_dict.keys())
            for i in reversed(time_steps):
                image_dict[recons_fnames[i]] = image_dict.pop(ffmpeg_fnames[i])
            return dict(sorted(image_dict.items()))
        else:
            ffmpeg_fnames = sorted(os.listdir(save_folder))
            for i in reversed(range(len(ffmpeg_fnames))):
                if i in time_steps:
                    os.rename(f'{save_folder}/{ffmpeg_fnames[i]}',
                              f'{save_folder}/{recons_fnames[i]}')
                else:
                    os.remove(f'{save_folder}/{ffmpeg_fnames[i]}')
            return None


def main():
    folder_path = 'demo'
    video_ids = ['8jT9ygmMvMg']
    for video_id in video_ids:
        df_ = pd.read_csv(
            f'{folder_path}/{video_id}/scenes_updated.csv', index_col=0)
        df_ = pd.concat([pd.to_timedelta(df_['start']).dt.total_seconds(),
                        pd.to_timedelta(df_['end']).dt.total_seconds()],
                        axis=1)
        for scene in df_.index:
            start_s, end_s = df_['start'][scene], df_['end'][scene]
            run_ffmpeg(f'{folder_path}/{video_id}/', start_s, end_s,
                       fps=10, frame_target_size=1920,
                       #    fix_image_width=False,
                       output_folder=f'{folder_path}/{video_id}/scene{scene}/images'
                       )
            break


if __name__ == '__main__':
    main()
