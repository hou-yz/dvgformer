from collections import OrderedDict
from typing import Mapping
from packaging import version
import numpy as np
import torch
from transformers import PretrainedConfig, GPT2Config, logging, CONFIG_MAPPING


logger = logging.get_logger(__name__)


class DVGFormerConfig(PretrainedConfig):

    model_type = "dvgformer"

    original_fps = 15
    max_data_frames = 150  # max sequence length in data, 10 seconds at 15 fps
    state_dim = 7
    action_dim = 6

    def __init__(
            self,
            vision_backbone='dinov2_vits14_reg',
            use_depth=True,
            prediction_option='iterative',  # iterative or one-shot
            action_option='dense',  # dense or sparse
            motion_option='local',  # local or global motion
            image_featmap_shape=(5, 9),
            num_quantile_bins=10,
            use_quality_mlps=False,
            drone_types=[1],  # 0: non-fpv, 1: fpv
            n_token_noise=1,
            n_token_quality=0,
            n_token_drone_type=1,
            n_token_state=1,
            n_token_boa=1,
            n_token_action=1,
            pad_side='right',  # left or right
            pad_token_value=0,
            fps=3,
            max_model_frames=150,  # max sequence length to consider
            n_future_frames=15,  # number of future frames to predict
            fix_image_width=True,
            hidden_size=384,
            n_layer=12,
            n_head=6,
            loss_coef_drone_type=0,
            loss_coef_state=0,
            loss_coef_action=1,
            loss_coef_stop=0.1,
            focal_alpha=0.90,
            loss_coef_future=0,
            ignore_value=-100,
            attn_implementation=None,
            # 'allframe': use GT action for all frames for transformer generative pipeline at 15 fps (needs per-action eval),
            # 'keyframe': use GT state in key frame (3 fps) to unroll the actions (15 fps)
            # 'none': execute generated actions (for simulation eval)
            test_gt_forcing='allframe',
            **kwargs,
    ):

        self.vision_backbone = vision_backbone
        if 'efficientnet' in vision_backbone:
            patch_size = 1
            self.backbone_downsample = 32
            self.vision_feat_dim = 1536
            resolution = 240
        elif 'mobilenet' in vision_backbone:
            patch_size = 1
            self.backbone_downsample = 32
            self.vision_feat_dim = 960
            resolution = 240
        elif 'dinov2' in vision_backbone:
            patch_size = 14
            self.backbone_downsample = patch_size
            self.vision_feat_dim = 384
            resolution = 224
        else:
            raise ValueError(
                f'Unsupported architecture "{vision_backbone}".'
            )
        self.use_depth = use_depth

        self.motion_option = motion_option

        self.fix_image_width = fix_image_width
        sensor_width = 36.0  # 35mm sensor width
        if fix_image_width:
            # keep everything at h:w=9:16
            # same area as [resolution, resolution], but with 16:9 aspect ratio
            height = resolution / (16 * 9) ** 0.5 * 9
            height = round(height / patch_size) * patch_size
            width = resolution / (16 * 9) ** 0.5 * 16
            width = round(width / patch_size) * patch_size
            resolution = [height, width]
            cropped_sensor_width = sensor_width  # no cropping
        else:
            # use aspect ratio of h:w=1:1
            resolution = [resolution, resolution]
            # for center crop images, the sensor width should be adjusted
            cropped_sensor_width = sensor_width / 16 * 9
        self.image_resolution = resolution  # h, w
        self.cropped_sensor_width = cropped_sensor_width

        self.fps = fps
        assert self.original_fps % fps == 0
        self.fps_downsample = int(self.original_fps / fps)
        assert max_model_frames % self.fps_downsample == 0, 'max_model_frames should be divisible by fps_downsample'
        # action option: dense or sparse
        self.action_option = action_option
        if self.action_option == 'dense':
            self.action_fps = self.original_fps
        elif self.action_option == 'sparse':
            self.action_fps = self.fps
        else:
            raise ValueError(f'unknown action option: {self.action_option}')
        self.action_downsample = self.original_fps // self.action_fps
        self.n_action_to_predict = self.fps_downsample // self.action_downsample
        self.prediction_option = prediction_option
        if self.prediction_option == 'one-shot' or n_token_action == 0:
            # predict all actions in one token
            self.per_token_preds = self.n_action_to_predict
        elif self.prediction_option == 'iterative':
            # predict one action in one token, and repeat for n_action_to_predict times
            self.per_token_preds = 1
        else:
            raise ValueError(
                f'unknown prediction option: {self.prediction_option}')

        # max sequence length to consider
        self.max_model_frames = max_model_frames
        # number of future frames to predict
        self.n_future_frames = n_future_frames

        # tokens for describing the entire sequence
        # noise token
        self.n_token_noise = n_token_noise
        # quality token
        self.num_quantile_bins = num_quantile_bins
        self.use_quality_mlps = use_quality_mlps
        self.n_token_quality = n_token_quality
        # drone type token
        self.drone_types = drone_types
        self.n_token_drone_type = n_token_drone_type

        # tokens for describing one frame
        # state token
        self.n_token_state = n_token_state
        # image token
        if image_featmap_shape is not None:
            if isinstance(image_featmap_shape, int):
                h, w = self.image_resolution
                # match to the longer side
                if h < w:
                    image_featmap_shape = (int(np.round(image_featmap_shape * h / w)),
                                           image_featmap_shape)
                else:
                    image_featmap_shape = (image_featmap_shape,
                                           int(np.round(image_featmap_shape * w / h)))
            self.image_featmap_shape = image_featmap_shape
        else:
            self.image_featmap_shape = list(
                map(lambda x: int(np.ceil(np.ceil(x / self.backbone_downsample))),
                    self.image_resolution))
        self.n_token_image = int(np.prod(self.image_featmap_shape))
        # begin of action token
        self.n_token_boa = n_token_boa
        # action token
        self.n_token_action = n_token_action

        # number of tokens for overall sequence conditioning
        self.n_token_prepend = (
            self.n_token_noise + self.n_token_quality + self.n_token_drone_type)
        # number of tokens for one frame
        self.n_token_frame = (self.n_token_state + self.n_token_image + self.n_token_boa +
                              self.n_token_action * self.n_action_to_predict // self.per_token_preds)
        # number of tokens to predict (number of action tokens) within one frame
        self.n_token_predict = (
            self.n_token_action * self.n_action_to_predict // self.per_token_preds)

        # padding side
        assert pad_side in ['left', 'right']
        self.pad_side = pad_side
        # different from the padding_token_id in base config since there is no discritized token
        self.pad_token_value = pad_token_value

        # loss weight
        self.loss_coef_drone_type = loss_coef_drone_type
        self.loss_coef_state = loss_coef_state
        self.loss_coef_action = loss_coef_action
        self.loss_coef_stop = loss_coef_stop
        self.focal_alpha = focal_alpha
        self.loss_coef_future = loss_coef_future

        self.ignore_value = ignore_value

        # expand
        self.test_gt_forcing = test_gt_forcing

        # transformer config
        kwargs.pop('gpt2_config', None)
        self.gpt2_config = GPT2Config(
            n_embd=hidden_size,
            n_positions=(max_model_frames // self.fps_downsample * self.n_token_frame +
                         self.n_token_quality + self.n_token_drone_type + self.n_token_noise),  # max length of the sequence
            n_layer=n_layer,
            n_head=n_head,
            attn_implementation=attn_implementation,
            **kwargs)
        self.hidden_size = hidden_size

        super().__init__(**kwargs)
        pass


def main():
    config = DVGFormerConfig()
    pass


if __name__ == '__main__':
    main()
