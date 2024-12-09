import sys
import os
import json
import shutil
import datetime
import argparse
from distutils.util import strtobool
import randomname
from distutils.dir_util import copy_tree
import numpy as np
import torch
from transformers import Trainer, TrainingArguments, set_seed
from src.models import DVGFormerConfig, DVGFormerModel
from src.data.drone_path_seq_dataset import DronePathSequenceDataset, collate_fn_video_drone_path_dataset
from blender_eval import blender_simulation

# torch.inverse multi-threading RuntimeError: lazy wrapper should be called at most once
# https://github.com/pytorch/pytorch/issues/90613#issuecomment-1817307008
torch.inverse(torch.ones((1, 1), device="cuda:0"))


def main(args):
    set_seed(args.seed)

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    # check if in debug mode
    gettrace = getattr(sys, 'gettrace', None)
    if gettrace():
        print('Hmm, Big Debugger is watching me')
        is_debug = True
        torch.autograd.set_detect_anomaly(True)
    else:
        print('No sys.gettrace')
        is_debug = False

    if is_debug:
        args.epochs = 1
        args.num_workers = 0
        args.num_runs = 1

    model_kwargs = dict(vars(args))
    for key in ['random_horizontal_flip', 'random_scaling', 'random_temporal_crop', 'random_color_jitter',
                'seed', 'epochs', 'batch_size', 'learning_rate', 'gradient_accumulation_steps', 'num_workers', 'num_runs']:
        model_kwargs.pop(key)
    if model_kwargs['fix_image_width']:
        model_kwargs['image_featmap_shape'] = (5, 9)
    else:
        model_kwargs['image_featmap_shape'] = (7, 7)
    model_kwargs['drone_types'] = [0, 1] if model_kwargs['drone_types'] == 'both' else [
        1] if model_kwargs['drone_types'] == 'fpv' else [0]

    str_token = (f'{f"n{args.n_token_noise}" if args.n_token_noise else ""}'
                 f'{f"q{args.n_token_quality}"if args.n_token_quality else ""}'
                 f'{f"t{args.n_token_drone_type}"if args.n_token_drone_type else ""}'
                 f's{args.n_token_state}img{np.prod(model_kwargs["image_featmap_shape"])}'
                 f'{f"boa{args.n_token_boa}" if args.n_token_boa else ""}'
                 f'a{args.prediction_option.upper()[0]}{args.action_option.upper()[0]}{args.n_token_action}')
    str_3d = ("depth2d" if args.use_depth else '')
    str_loss = (f'loss{f"s{args.loss_coef_state}" if args.loss_coef_state else ""}'
                f'{f"a{args.loss_coef_action}" if args.loss_coef_action else ""}'
                f'{f"stop{args.loss_coef_stop}" if args.loss_coef_stop else ""}'
                f'{f"fut{args.loss_coef_future}" if args.loss_coef_future else ""}')
    str_aug = (f'{"F" if args.random_horizontal_flip else ""}{"S" if args.random_scaling else ""}'
               f'{"T"if args.random_temporal_crop else ""}{"C" if args.random_color_jitter else ""}')
    logdir = (f'{args.drone_types}-{args.fps}fps-{args.max_model_frames}frames-'
              f'l{args.n_layer}h{args.n_head}-{str_token}-motion{args.motion_option.upper()[0]}-'
              f'{str_3d}-{str_loss}-{str_aug}')

    # setup logdir
    run_name = randomname.generate()
    logdir = f'logs/{"DEBUG_" if is_debug else ""}{logdir}-{run_name}'
    os.makedirs(logdir, exist_ok=True)
    # make a copy of all python scripts
    copy_tree('src', f'{logdir}/scripts/src')
    for script in os.listdir('.'):
        if script.split('.')[-1] == 'py':
            dst_file = f'{logdir}/scripts/{os.path.basename(script)}'
            shutil.copyfile(script, dst_file)
    print(logdir)
    print(args)

    # save args as json
    with open(f'{logdir}/args-{datetime.datetime.today():%Y-%m-%d_%H-%M-%S}.json', 'w') as f:
        json.dump(vars(args), f, indent=4)

    config = DVGFormerConfig(
        attn_implementation='flash_attention_2',
        test_gt_forcing='allframe',
        **model_kwargs)

    train_dataset = DronePathSequenceDataset(
        args.root,
        args.hdf5_fname,
        fps=config.fps,
        action_fps=config.action_fps,
        max_model_frames=config.max_model_frames,
        resolution=config.image_resolution,
        fix_image_width=config.fix_image_width,
        drone_types=config.drone_types,
        motion_option=config.motion_option,
        noise_dim=config.hidden_size,
        random_horizontal_flip=args.random_horizontal_flip,
        random_scaling=args.random_scaling,
        random_temporal_crop=args.random_temporal_crop,
        random_color_jitter=args.random_color_jitter,
        n_future_frames=config.n_future_frames,
        num_quantile_bins=config.num_quantile_bins,
    )

    # set torch default dtype to bfloat16
    cur_dtype = torch.get_default_dtype()
    torch.set_default_dtype(torch.bfloat16)
    model = DVGFormerModel(config)
    # revert torch default dtype
    torch.set_default_dtype(cur_dtype)

    training_args = TrainingArguments(
        output_dir=logdir,
        run_name=run_name,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        dataloader_num_workers=args.num_workers,
        logging_steps=5 if is_debug else 50,
        save_strategy="epoch",
        report_to='none' if is_debug else 'all',
        dataloader_drop_last=True,
        save_safetensors=False,
        # settings from llava
        bf16=True,
        tf32=True,
        save_total_limit=1,
        warmup_ratio=0.03,
        lr_scheduler_type='cosine',
    )

    trainer = Trainer(
        model,
        training_args,
        train_dataset=train_dataset,
        data_collator=collate_fn_video_drone_path_dataset
    )

    trainer.train()
    trainer.save_model()
    del trainer
    print(logdir)

    # clean up cuda memory
    torch.cuda.empty_cache()
    blender_simulation(config, model, logdir, num_runs=args.num_runs)
    print(logdir)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='training script for DVGFormer')
    # data settings
    parser.add_argument('--root', type=str, default='youtube_drone_videos')
    parser.add_argument('--hdf5_fname', type=str,
                        default='dataset_mini.h5')
    # model settings
    parser.add_argument('--fps', type=int, default=3)
    parser.add_argument('--max_model_frames', type=int, default=150)
    parser.add_argument('--n_future_frames', type=int, default=15)
    parser.add_argument('--drone_types', type=str,
                        default='fpv', choices=['fpv', 'non-fpv', 'both'])
    parser.add_argument('--n_layer', type=int, default=12)
    parser.add_argument('--n_head', type=int, default=6)
    parser.add_argument('--n_token_noise', type=int, default=1)
    parser.add_argument('--n_token_quality', type=int, default=0)
    parser.add_argument('--n_token_drone_type', type=int, default=1)
    parser.add_argument('--n_token_state', type=int, default=1)
    parser.add_argument('--n_token_boa', type=int, default=1)
    parser.add_argument('--n_token_action', type=int, default=1)
    parser.add_argument('--fix_image_width',
                        type=lambda x: bool(strtobool(x)), default=True)
    parser.add_argument('--use_depth',
                        type=lambda x: bool(strtobool(x)), default=True)
    parser.add_argument('--prediction_option',
                        type=str, default='iterative', choices=['iterative', 'one-shot'])
    parser.add_argument('--action_option',
                        type=str, default='dense', choices=['dense', 'sparse'])
    parser.add_argument('--motion_option',
                        type=str, default='local', choices=['local', 'global'])
    parser.add_argument('--loss_coef_drone_type', type=float, default=0)
    parser.add_argument('--loss_coef_state', type=float, default=0)
    parser.add_argument('--loss_coef_action', type=float, default=1)
    parser.add_argument('--loss_coef_stop', type=float, default=0)
    parser.add_argument('--loss_coef_future', type=float, default=0)
    # augmentation settings
    parser.add_argument('--random_horizontal_flip',
                        type=lambda x: bool(strtobool(x)), default=True)
    parser.add_argument('--random_scaling',
                        type=lambda x: bool(strtobool(x)), default=True)
    parser.add_argument('--random_temporal_crop',
                        type=lambda x: bool(strtobool(x)), default=True)
    parser.add_argument('--random_color_jitter',
                        type=lambda x: bool(strtobool(x)), default=True)
    # training settings
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--learning_rate', type=float, default=1e-4)
    parser.add_argument('--gradient_accumulation_steps', type=int, default=4)
    parser.add_argument('--num_workers', type=int, default=4)
    # evaluation settings
    parser.add_argument('--num_runs', type=int, default=50)

    args = parser.parse_args()
    main(args)
