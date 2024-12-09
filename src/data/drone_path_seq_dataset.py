import os
import re
import time
import logging
import tqdm
import json
from typing import Dict, Optional, Sequence, List
from PIL import Image
import numpy as np
from transforms3d.quaternions import qinverse, qconjugate, qmult, qnorm, quat2mat, mat2quat, quat2axangle, axangle2quat, nearly_equivalent
from transforms3d.euler import euler2quat, quat2euler, euler2mat, mat2euler
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision.utils import make_grid
import torchvision.transforms as T
import torchvision.transforms.functional as F
from src.data.state_action_conversion import get_states_actions, reverse_states_actions, state_avg, state_std, action_avg, action_std
from src.preparation.youtube_download import get_video_stat
from src.preparation.split_scene_frame import _recons_images_from_ffmpeg
from src.utils.padding import padding, concated_seq_to_instances
from src.utils.quaternion_operations import convert_to_local_frame, horizontal_flip, add_angular_velocity_to_quaternion, quaternions_to_angular_velocity
from src.utils.flexible_fs import FlexibleFileSystem
from src.utils.colmap_official_read_write_model import read_cameras_binary, read_images_binary, read_points3D_binary


def color_jitter(img, brightness=0.0, contrast=0.0, saturation=0.0, hue=0.0, fn_idx=(0, 1, 2, 3)):
    for fn_id in fn_idx:
        if fn_id == 0 and brightness:
            img = F.adjust_brightness(img, 1 + brightness)
        elif fn_id == 1 and contrast:
            img = F.adjust_contrast(img, 1 + contrast)
        elif fn_id == 2 and saturation:
            img = F.adjust_saturation(img, 1 + saturation)
        elif fn_id == 3 and hue:
            img = F.adjust_hue(img, hue)
    return img


def get_noise_vector(index, noise_dim=384):
    random_generator = np.random.RandomState(index)
    return random_generator.randn(noise_dim)


class DronePathSequenceDataset(Dataset):

    original_fps = 15  # original frame rate used for colmap reconstruction
    max_data_frames = 150  # max number of original frames

    def __init__(self, root, hdf5_fname, fps=3, action_fps=15,
                 max_model_frames=150, n_future_frames=15,
                 resolution=(180, 320), motion_option='global', fix_image_width=True, skip_portrait_videos=True, drone_types=[0, 1],
                 use_cuda_ffmpeg=False, noise_dim=384,
                 random_horizontal_flip=False, random_scaling=False, random_temporal_crop=False, random_color_jitter=False,
                 ignore_value=-100, num_quantile_bins=100):
        super().__init__()

        self.root = root
        self.hdf5_fname = hdf5_fname
        # frame rate for the auto-regressive task tuples (image, state, action)
        self.fps = fps

        # max sequence length to consider
        self.max_model_frames = max_model_frames
        # fewer images if original_fps > fps: only one image every fps_downsample frames
        # (image, state, action), (state, action), ..., (state, action)
        assert self.original_fps % fps == 0
        self.fps_downsample = int(self.original_fps / fps)
        self.action_fps = action_fps
        self.action_downsample = self.original_fps // action_fps
        self.n_action_to_predict = self.fps_downsample // self.action_downsample
        # future prediction
        self.n_future_frames = n_future_frames  # at original_fps
        self.n_future_steps = self.n_future_frames // self.fps_downsample  # at fps

        self.resolution = resolution  # h, w

        self.fix_image_width = fix_image_width

        self.noise_dim = noise_dim

        # drone types: 0 for non-fpv, 1 for fpv
        self.drone_types = drone_types

        self.motion_option = motion_option
        # state: tvec, qvec (all in global reference frame)
        self.state_dim = 7
        # action: v, omega (local, relative to the current frame)
        self.action_dim = 6

        # augmentation
        self.random_horizontal_flip = random_horizontal_flip
        self.random_scaling = random_scaling
        self.random_temporal_crop = random_temporal_crop
        self.random_color_jitter = random_color_jitter

        # consider vertical videos (portrait mode)
        self.skip_portrait_videos = skip_portrait_videos

        # labels
        self.ignore_value = ignore_value

        self.num_quantile_bins = num_quantile_bins

        # speed of the camera movement in blender coord (meter/frame) at 15 fps
        # self.drone_speeds = {0: 0.5,  # 0.5 m/frame for non-fpv drones
        #                      1: 1,  # 1 m/frame for fpv drones
        #                      }

        self.transform_img = T.Compose([
            T.CenterCrop(resolution),
            T.ToTensor(),
            T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])
        self.transform_depth = T.Compose([
            T.ToTensor(),
            T.CenterCrop(resolution),
        ])

        # ffmpeg device
        self.use_cuda_ffmpeg = use_cuda_ffmpeg and torch.cuda.is_available()
        if self.use_cuda_ffmpeg:
            print('using cuda for ffmpeg')
        self.num_cuda_devices = torch.cuda.device_count()

        # Update: change from tar to HDF5 for the SWMR mode for multi workers in dataloader
        self.h5_fs = FlexibleFileSystem(
            f'{self.root}/{self.hdf5_fname if self.hdf5_fname else ""}')
        self.video_stats = {}
        self.all_scenes = []
        self.data_list = []
        self.points_info = []
        total_length = 0
        for video_id in tqdm.tqdm(sorted(self.h5_fs.listdir(self.root))):
            try:
                with self.h5_fs.open(f'{self.root}/{video_id}/data.json') as f:
                    metadata = json.load(f)
                video_stat = get_video_stat(metadata)
                video_stat['num_clips'] = 0
                video_stat['quality'] = (video_stat['view_count'] /
                                         video_stat['duration'])
                if not video_stat['is_drone'] or video_stat['has_skip_words']:
                    continue
                if self.skip_portrait_videos and not video_stat['is_landscape']:
                    continue
                if int(video_stat['is_fpv']) not in self.drone_types:
                    continue
                self.video_stats[video_id] = video_stat
            except Exception as e:
                # print(f'Error reading metadata for {video_id}: {e}')
                continue
            current_num_clips = len(self.data_list)
            for result_fname in sorted(self.h5_fs.listdir(f'{self.root}/{video_id}')):
                if not ('-score' in result_fname and result_fname.endswith('.csv')):
                    continue
                scene = os.path.basename(result_fname).split(
                    '-')[0].replace('scene', '')
                video_scene = f'{video_id}/scene{scene}'
                if video_scene not in self.all_scenes:
                    self.all_scenes.append(video_scene)
                recons_index = int(os.path.basename(result_fname).split(
                    '-')[1].replace('recons', ''))
                score = int(re.search(r'-score(\d+)',
                                      result_fname).group(1))
                valid = '_invalid' not in result_fname
                if not score or not valid:
                    continue
                frame_folder = f'{self.root}/{video_id}/scene{scene}-recons{recons_index}-frames/'
                if not self.h5_fs.exists(frame_folder):
                    continue
                result_fpath = f'{self.root}/{video_id}/{result_fname}'

                with self.h5_fs.open(result_fpath, 'r') as f:
                    recons_df = pd.read_csv(f, comment='#')
                recons_array = recons_df.to_numpy()

                total_length += len(recons_array) / self.original_fps
                if self.max_model_frames == self.max_data_frames:
                    self.data_list.append(
                        {'result_fpath': result_fpath,
                         'start_idx': 0,
                         'end_idx': len(recons_array)})
                else:
                    chunk_step = self.max_model_frames // 2 // self.fps_downsample * self.fps_downsample
                    for start_idx in range(0, len(recons_array) //
                                           self.fps_downsample * self.fps_downsample - chunk_step, chunk_step):
                        self.data_list.append(
                            {'result_fpath': result_fpath,
                             'start_idx': start_idx,
                             'end_idx': min(start_idx + self.max_model_frames, len(recons_array))})
            self.video_stats[video_id]['num_clips'] = \
                len(self.data_list) - current_num_clips
        # log the overall stats
        stat_keys = ['view_count', 'like_count', 'comment_count', 'duration',
                     'is_fpv', 'num_clips', 'quality']
        video_stat_array = {}
        for key in stat_keys:
            video_stat_array[key] = np.array([self.video_stats[video_id][key]
                                              for video_id in self.video_stats])
        print(f'Dataset: {len(self.data_list)} sequences, '
              f'{total_length / 3600:.1f} hours')
        is_fpv = video_stat_array["is_fpv"] == 1
        print(f'total videos: {len(self.video_stats)} \t'
              f'fpv: {np.sum(is_fpv)} \tnon-fpv: {np.sum(~is_fpv)}')
        print(f'view count: {np.mean(video_stat_array["view_count"]):.1f} \t'
              f'fpv: {np.mean(video_stat_array["view_count"][is_fpv]):.1f} \t'
              f'non-fpv: {np.mean(video_stat_array["view_count"][~is_fpv]):.1f}')
        # quantize the video stats based ont the number of clips
        clip_quality = []
        for video_id in self.video_stats:
            clip_quality.extend([self.video_stats[video_id]['quality']] *
                                self.video_stats[video_id]['num_clips'])
        quantile_bins = np.quantile(clip_quality,
                                    np.linspace(0, 1, num_quantile_bins))
        for video_id in self.video_stats:
            self.video_stats[video_id]['quality_quantile'] = int(np.digitize(
                self.video_stats[video_id]['quality'], quantile_bins, right=True))
        pass

    def __len__(self) -> int:
        return len(self.data_list)

    def __getitem__(self, index, visualize=False):
        t0 = time.time()

        # info
        result_fpath = self.data_list[index]['result_fpath']
        start_idx = self.data_list[index]['start_idx']
        end_idx = self.data_list[index]['end_idx']
        video_id = result_fpath.split('/')[-2]
        video_stats = self.video_stats[video_id]
        scene = os.path.basename(result_fpath).split(
            '-')[0].replace('scene', '')
        recons_index = int(os.path.basename(result_fpath).split(
            '-')[1].replace('recons', ''))
        is_fpv = int(video_stats['is_fpv'])

        # recons info
        with self.h5_fs.open(result_fpath, 'r') as f:
            coord_multiplier = float(
                f.readline().decode().replace('#', '').strip())
            recons_df = pd.read_csv(f, comment='#')

        # augmentation for entire sequence
        H, W = self.resolution
        # random horizontal flip
        if self.random_horizontal_flip:
            flip = np.random.rand() < 0.5
        else:
            flip = False
        noise_idx = int(flip)
        # random scaling
        if self.random_scaling:
            # random scale up, i.e., no padding, only cropping
            scale = np.random.uniform(1.0, 1.2)
            H = int(H * scale)
            W = int(W * scale)
        else:
            scale = 1.0
        # random termporal crop
        if (self.max_data_frames == self.max_model_frames and
                np.random.rand() < self.random_temporal_crop / 2):
            # reduce the sequence length by start_offset, end_offset
            seq_length = end_idx - start_idx
            start_offset = np.random.randint(
                0, seq_length * 0.2 + 1)
            start_offset = start_offset // self.fps_downsample * self.fps_downsample
            start_idx += start_offset
            # end_offset = np.random.randint(
            #     0, seq_length * 0.2 + 1)
            # end_offset = end_offset // self.fps_downsample * self.fps_downsample
            # end_idx -= end_offset
        else:
            pass
        # random color jitter
        if self.random_color_jitter:
            brightness = np.random.uniform(-0.5, 0.5)
            contrast = np.random.uniform(-0.5, 0.5)
            saturation = np.random.uniform(-0.5, 0.5)
            hue = 0.0
            jitter_fn_idx = np.random.permutation(4)
        else:
            brightness = 0.0
            contrast = 0.0
            saturation = 0.0
            hue = 0.0
            jitter_fn_idx = np.arange(4)

        noise_embed = get_noise_vector(noise_idx + index * 2,
                                       self.noise_dim)

        # camera poses
        # note: the original tvecs and qvecs in read_images_binary() gives camera extrinsic matrix [R=quat2mat(qvec), t=tvec],
        # but the camera pose (location and orientation) in the global coord system is [-R.T @ t, R.T]
        # recons_df include the converted tvecs and qvecs (all in world coord system)
        recons_array = recons_df.to_numpy()[start_idx:]
        # camera path in global coord system (measurements)
        # raw_tvecs = recons_array[:, 1:4].astype(float)
        # raw_qvecs = recons_array[:, 4:8].astype(float)
        # raw_vs = recons_array[:, 8:11].astype(float)
        # raw_omegas = recons_array[:, 11:14].astype(float)
        # camera path the global coord system (estimation)
        raw_tvecs = recons_array[:, 14:17].astype(float)
        raw_qvecs = recons_array[:, 17:21].astype(float)
        raw_vs = recons_array[:, 21:24].astype(float)
        raw_omegas = recons_array[:, 24:27].astype(float)
        # add the final speed and angular velocity to extend the sequence
        final_tvec = raw_tvecs[-1] + raw_vs[-1]
        final_qvec = add_angular_velocity_to_quaternion(
            raw_qvecs[-1], raw_omegas[-1], 1)
        raw_tvecs = np.concatenate([raw_tvecs, final_tvec[None]], axis=0)
        raw_qvecs = np.concatenate([raw_qvecs, final_qvec[None]], axis=0)

        # reference frame
        ref_tvec, ref_qvec = raw_tvecs[0], raw_qvecs[0]

        # change the global coord system to the initial frame
        tvecs = np.zeros_like(raw_tvecs)
        qvecs = np.zeros_like(raw_qvecs)
        for i in range(len(raw_tvecs)):
            tvecs[i], qvecs[i], _, _ = convert_to_local_frame(
                ref_tvec, ref_qvec,
                raw_tvecs[i], raw_qvecs[i])
        # modulate the speed based on the drone type
        # tvecs *= self.drone_speeds[is_fpv]
        # vs *= self.drone_speeds[is_fpv]
        # sequence length based on the start and end index
        seq_length = (end_idx - start_idx) // self.fps_downsample
        # time steps (same length as image/state)
        time_steps = np.arange(seq_length)
        # augmentation
        if flip:
            aug_tvecs = np.zeros_like(tvecs)
            aug_qvecs = np.zeros_like(qvecs)
            for i in range(len(tvecs)):
                aug_tvecs[i], aug_qvecs[i], _, _ = horizontal_flip(
                    tvecs[i], qvecs[i])
        else:
            aug_tvecs, aug_qvecs = tvecs, qvecs

        # states & actions are self.n_action_to_predict times the length
        time_range = (time_steps[:, None] * self.fps_downsample +
                      np.arange(self.n_action_to_predict))
        _states, _actions = get_states_actions(
            aug_tvecs, aug_qvecs,
            motion_option=self.motion_option, action_downsample=self.action_downsample)
        # include the last state
        next_t, next_q, _, _ = reverse_states_actions(
            _states[[-1]], _actions[[-1]], motion_option=self.motion_option)
        _states = np.concatenate(
            [_states, np.concatenate([next_t, next_q], axis=1)], axis=0)
        next_states = _states[time_range // self.action_downsample + 1]
        states = _states[time_range // self.action_downsample]
        # if sparse action, make sure the action is still of the same norm
        actions = (_actions[time_range // self.action_downsample] /
                   self.action_downsample)
        # compute the future waypoints
        n_step_vs = np.ones([seq_length + self.n_future_steps - 1, 3]
                            ) * self.ignore_value
        n_step_omegas = np.ones([seq_length + self.n_future_steps - 1, 3]
                                ) * self.ignore_value
        for i in range(seq_length + self.n_future_steps - 1):
            if (time_steps[0] + i + 1) * self.fps_downsample >= len(aug_tvecs):
                break
            n_step_vs[i] = (aug_tvecs[(time_steps[0] + i + 1) * self.fps_downsample] -
                            aug_tvecs[(time_steps[0] + i) * self.fps_downsample])
            n_step_omegas[i] = quaternions_to_angular_velocity(
                aug_qvecs[(time_steps[0] + i) * self.fps_downsample],
                aug_qvecs[(time_steps[0] + i + 1) * self.fps_downsample], 1)
            if self.motion_option == 'local':
                # change the global coord system to the initial frame
                _, _, n_step_vs[i], n_step_omegas[i] = convert_to_local_frame(
                    aug_tvecs[(time_steps[0] + i) * self.fps_downsample],
                    aug_qvecs[(time_steps[0] + i) * self.fps_downsample],
                    None, None, n_step_vs[i], n_step_omegas[i])
        n_step_actions = np.concatenate(
            [n_step_vs, n_step_omegas], axis=1)[np.arange(seq_length)[:, None] +
                                                np.arange(self.n_future_steps)]
        pass
        t1 = time.time()

        # load images
        frame_folder = f'{self.root}/{video_id}/scene{scene}-recons{recons_index}-frames'
        image_dict = {}
        for i in (time_steps * self.fps_downsample).tolist():
            fname = recons_array[i, 0]
            if self.h5_fs.h5_file is not None:
                with self.h5_fs.open(f'{frame_folder}/{fname}', 'r') as f:
                    img = Image.open(f).convert('RGB')
            else:
                img = Image.open(f'{frame_folder}/{fname}').convert('RGB')
            image_dict[fname] = img

        if visualize:
            imgs = [self.transform_img(img) for img in image_dict.values()]
            T.ToPILImage()(make_grid(torch.stack(imgs), normalize=True)).save('frames.jpg')

        t2 = time.time()

        # camera intrinsics
        cameras_info = read_cameras_binary(
            f'{self.root}/{video_id}/scene{scene}-recons{recons_index}-colmap/cameras.bin',
            self.h5_fs)
        cam = cameras_info[1]
        # original camera parameters
        if cam.model in ("SIMPLE_PINHOLE", "SIMPLE_RADIAL", "RADIAL"):
            fx = fy = cam.params[0]
            cx = cam.params[1]
            cy = cam.params[2]
        elif cam.model in (
            "PINHOLE",
            "OPENCV",
            "OPENCV_FISHEYE",
            "FULL_OPENCV",
        ):
            fx = cam.params[0]
            fy = cam.params[1]
            cx = cam.params[2]
            cy = cam.params[3]
        else:
            raise Exception("Camera model not supported")

        # resize the image to match the resolution
        h, w = cam.height, cam.width
        # assert cx * 2 == w and cy * 2 == h, 'principal point should be at the center'
        if self.fix_image_width and not self.skip_portrait_videos:
            # resize the image to match the width
            img_ratio = W / w
        else:
            # resize the image to match the smaller dimension, i.e., no padding, only cropping
            img_ratio = max(W / w, H / h)
        target_h = int(img_ratio * h)
        target_w = int(img_ratio * w)
        # also update the intrinsics for the resized image
        fx, fy = fx * img_ratio, fy * img_ratio
        # cx, cy = cx * img_ratio, cy * img_ratio
        # will center crop the image so the principal point should always be at the center
        cx, cy = W / 2, H / 2
        K = np.identity(3)
        K[0, 0] = fx
        K[1, 1] = fy
        K[0, 2] = cx
        K[1, 2] = cy

        # images
        images = []
        for i in (time_steps * self.fps_downsample).tolist():
            fname = recons_array[i, 0]
            img = image_dict[fname].convert('RGB')
            # PIL in w, h format
            img = img.resize((target_w, target_h))
            # flip
            if flip:
                img = img.transpose(Image.FLIP_LEFT_RIGHT)
            # color jitter
            if self.random_color_jitter:
                img = color_jitter(img, brightness, contrast, saturation, hue,
                                   jitter_fn_idx)
            img = self.transform_img(img)
            images.append(img)
        images = torch.stack(images)

        t3 = time.time()

        time_steps = torch.tensor(time_steps, dtype=torch.long)
        states = (states - state_avg) / state_std
        states = torch.tensor(states, dtype=torch.float32)
        next_states = (next_states - state_avg) / state_std
        next_states = torch.tensor(next_states, dtype=torch.float32)
        actions = (actions - action_avg) / action_std
        actions = torch.tensor(actions, dtype=torch.float32)
        stop_labels = torch.zeros(seq_length)
        future_mask = (n_step_actions != self.ignore_value).all(axis=-1)
        n_step_actions[future_mask] = (
            n_step_actions[future_mask] - action_avg) / action_std
        n_step_actions = torch.tensor(n_step_actions, dtype=torch.float32)

        scene_part = int(scene.split('_')[1])
        next_scene = f'{scene.split("_")[0]}_{scene_part + 1}'
        is_partial_scene = f'{video_id}/scene{next_scene}' in self.all_scenes
        if (end_idx >= len(recons_df) // self.fps_downsample * self.fps_downsample and
                not is_partial_scene):
            stop_labels[-1] = 1
        else:
            pass
        data_dict = {
            # inputs
            'noise_embed': torch.tensor(noise_embed, dtype=torch.float32),
            'quality': video_stats['quality_quantile'],
            'drone_type': is_fpv,
            'intrinsic': torch.from_numpy(K).float(),
            'time_steps': time_steps,
            'images': images,
            'states': states,
            'actions': actions,
            'seq_length': seq_length,
            # labels
            'next_state_labels': next_states,
            'action_labels': actions.clone(),
            'stop_labels': stop_labels,
            'future_action_labels': n_step_actions,
            'drone_type_labels': is_fpv,
        }

        t4 = time.time()
        logging.debug(f'read csv: {t1 - t0:.3f}s, \tffmpeg: {t2 - t1:.3f}s, \t'
                      f'conversion: {t3 - t2:.3f}s, \toutput: {t4 - t3:.3f}s, \t'
                      f'total: {t4 - t0:.3f}s')
        return data_dict

    def __del__(self):
        if self.h5_fs is not None:
            self.h5_fs.close()


def collate_fn_video_drone_path_dataset(
        instances: Sequence[Dict],
        pad_side='right',
        pad_value=0,
        label_pad_value=-100,
) -> Dict[str, torch.Tensor]:
    _seq_length = [instance['seq_length'] for instance in instances]
    seq_length = torch.tensor(_seq_length)
    # attention mask, 1 for attention, 0 for skip
    _attn_mask = [torch.ones(l) for l in _seq_length]
    attn_mask = padding(_attn_mask, pad_side, 0)
    batch = {}
    default_keys = ['noise_embed', 'quality', 'drone_type', 'intrinsic', 'time_steps', 'images', 'states', 'actions', 'seq_length',
                    'next_state_labels', 'action_labels', 'stop_labels', 'drone_type_labels', 'depth_labels', 'pointcloud_labels', 'future_action_labels']
    keys = set(default_keys) & set(instances[0].keys())
    for key in keys:
        if key == 'seq_length':
            value = seq_length
        else:
            value = [instance[key] for instance in instances]
            if key == 'quality' or 'drone_type' in key:
                value = torch.tensor(value)
            elif key == 'intrinsic' or key == 'noise_embed':
                value = torch.stack(value)
            elif key == 'depth_labels':
                value = padding(value, pad_side, torch.inf)
            elif 'labels' in key:
                value = padding(value, pad_side, label_pad_value)
            else:
                # time_steps, images, states, actions
                value = padding(value, pad_side, pad_value)
        batch.update({key: value, })
    batch['attention_mask'] = attn_mask
    return batch


def main():
    from transformers import set_seed

    set_seed(0)
    # logging.basicConfig(level=logging.DEBUG)
    motion_option = 'global'
    dataset = DronePathSequenceDataset('youtube_drone_videos',
                                       'dataset_mini.h5',
                                       #    resolution=(1080, 1920),
                                       motion_option=motion_option,
                                       #    random_horizontal_flip=True,
                                       random_scaling=True,
                                       random_temporal_crop=True,
                                       #    max_model_frames=30,
                                       )
    print(len(dataset))
    dataset.__getitem__(1, visualize=True)
    dataset.__getitem__(76, visualize=True)
    # indices = [np.random.randint(len(dataset)) for _ in range(20)]
    # for idx in indices:
    #     dataset.__getitem__(idx, visualize=False)
    #     pass
    return

    # dataset statistics
    # tvec_avgs, tvec_stds = [], []
    # qvec_avgs, qvec_stds = [], []
    # v_avgs, v_stds = [], []
    # omega_avgs, omega_stds = [], []
    # for i in tqdm.tqdm(range(2000)):
    #     tvecs, qvecs, vs, omegas = dataset.__getitem__(i, visualize=False)
    #     tvec_avgs.append(tvecs.mean())
    #     tvec_stds.append(tvecs.std())
    #     qvec_avgs.append(qvecs.mean())
    #     qvec_stds.append(qvecs.std())
    #     v_avgs.append(vs.mean())
    #     v_stds.append(vs.std())
    #     omega_avgs.append(omegas.mean())
    #     omega_stds.append(omegas.std())
    # tvec_avgs = np.stack(tvec_avgs)
    # tvec_stds = np.stack(tvec_stds)
    # qvec_avgs = np.stack(qvec_avgs)
    # qvec_stds = np.stack(qvec_stds)
    # v_avgs = np.stack(v_avgs)
    # v_stds = np.stack(v_stds)
    # omega_avgs = np.stack(omega_avgs)
    # omega_stds = np.stack(omega_stds)
    # print(f"{tvec_avgs.mean():.3f}\t{tvec_stds.mean():.3f}\t{tvec_avgs.min():.3f}\t{tvec_avgs.max():.3f}")
    # print(f"{qvec_avgs.mean():.3f}\t{qvec_stds.mean():.3f}\t{qvec_avgs.min():.3f}\t{qvec_avgs.max():.3f}")
    # print(f"{v_avgs.mean():.3f}\t{v_stds.mean():.3f}\t{v_avgs.min():.3f}\t{v_avgs.max():.3f}")
    # print(f"{omega_avgs.mean():.3f}\t{omega_stds.mean():.3f}\t{omega_avgs.min():.3f}\t{omega_avgs.max():.3f}")
    # return

    dataloader = DataLoader(dataset, batch_size=32, shuffle=False,
                            collate_fn=collate_fn_video_drone_path_dataset,
                            num_workers=0, drop_last=True)
    batch = next(iter(dataloader))
    all_states = []
    all_actions = []
    all_stops = []
    all_masks = []
    all_lens = []
    all_points = []
    for i, batch in enumerate(tqdm.tqdm(dataloader)):
        all_states.append(batch['states'].transpose(0, 1))
        all_actions.append(batch['actions'].transpose(0, 1))
        all_masks.append(batch['attention_mask'].transpose(0, 1))
        all_stops.append(batch['stop_labels'].transpose(0, 1))
        all_lens.extend(batch['seq_length'].tolist())
        points = batch.get('pointcloud_labels', None)
        if points is not None:
            points = points[(points != -100).all(dim=-1)]
            all_points.append(points)
        if i == 10:
            break
        pass
    all_states = padding(all_states).transpose(1, 2).flatten(0, 1)
    all_actions = padding(all_actions).transpose(1, 2).flatten(0, 1)
    all_masks = padding(all_masks).transpose(1, 2).flatten(0, 1)
    all_stops = padding(all_stops).transpose(1, 2).flatten(0, 1)
    all_states = all_states[all_masks == 1]
    all_actions = all_actions[all_masks == 1]
    all_stops = all_stops[all_masks == 1]
    print(f'states: mean={all_states.mean(dim=[0, 1])}, '
          f'std={all_states.std(dim=[0, 1])}')
    print(f'actions: mean={all_actions.mean(dim=[0, 1])}, '
          f'std={all_actions.std(dim=[0, 1])}')
    print(f'stops: mean={all_stops.mean()}')
    print(f'lengths: {np.mean(all_lens)}')
    if len(all_points) > 0:
        all_points = torch.cat(all_points)
        all_points = all_points.clamp(-1e4, 1e4)
        print(
            f'points: mean={all_points.mean(dim=0)}, std={all_points.std(dim=0)}')
    tvec_norms = all_states[:, :3].norm(dim=-1)
    v_norms = all_actions[:, :3].norm(dim=-1)
    omega_norms = all_actions[:, 3:].norm(dim=-1)
    print(
        f'norms: tvec={tvec_norms.mean()}, v={v_norms.mean()}, omega={omega_norms.mean()}')
    # for i in tqdm.tqdm(range(len(dataset))):
    #     dataset.__getitem__(i)
    pass


if __name__ == '__main__':
    main()
