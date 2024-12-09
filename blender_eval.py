import os
import threading
import time
import bpy
import tqdm
from transforms3d.quaternions import qinverse, qconjugate, qmult, qnorm, quat2mat, mat2quat, quat2axangle, axangle2quat, nearly_equivalent
from transforms3d.euler import euler2quat, quat2euler, euler2mat, mat2euler
from PIL import Image
import numpy as np
import torch
import torchvision.transforms as T
from src.blender.blender_camera_env import BlenderCameraEnv
from src.models import DVGFormerConfig, DVGFormerModel
from src.utils.quaternion_operations import convert_to_local_frame
from src.data.state_action_conversion import state_avg, state_std, action_avg, action_std


infinigen_root = 'infinigen'
blosm_root = 'blosm'


def expand_episode(env, config, model, run_name, drone_type=1, seed=None, random_init_pose=False, re_render=True):

    # Reset environment
    env.drone_type = drone_type
    observation, info = env.reset(seed=seed, random_init_pose=random_init_pose)
    fx, fy, cx, cy = env.intrinsics
    K = torch.tensor([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=torch.float)

    model.eval()

    transform = T.Compose([
        T.ToTensor(),
        T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])

    # batch size, seq_length
    b, l = 1, 1
    # time step
    t = 0

    random_generator = np.random.RandomState(seed)
    noise_embed = random_generator.randn(b, config.hidden_size)
    # batch data should be normalized
    batch = {'noise_embed': torch.tensor(noise_embed, dtype=torch.float32).cuda(),
             'quality': torch.ones(b, dtype=torch.long).cuda() * (config.num_quantile_bins - 1),
             'drone_type': (torch.ones(b, dtype=torch.long).cuda() * drone_type),
             'intrinsic': K[None].repeat(b, 1, 1).cuda(),
             'time_steps': torch.arange(0, dtype=torch.long).repeat(b, 1).cuda(),
             'images': torch.zeros(b, 0, 3, config.image_resolution[0], config.image_resolution[1]).cuda(),
             'states': torch.zeros(b, 0, model.n_action_to_predict, model.config.state_dim).cuda(),
             'actions': torch.zeros(b, 0, model.n_action_to_predict, model.config.action_dim).cuda(),
             'seq_length': torch.zeros(b, dtype=torch.long).cuda(),
             'past_key_values': None,
             }
    seqence_lvl_keys = ['noise_embed', 'quality', 'drone_type', 'intrinsic']
    batch_pt = {}
    for key in seqence_lvl_keys:
        batch_pt[key] = batch[key]

    done = False
    total_reward = 0
    crash = None
    seq_len = 1
    chunk_offset = 0
    chunk_size = model.max_model_frames // model.fps_downsample
    chunk_step = chunk_size // 2
    t_ref, q_ref = np.zeros(3), np.array([1, 0, 0, 0])
    while not done:
        # Convert observation to tensor & normalize
        img = Image.fromarray(observation['image']).convert('RGB')
        img = transform(img)[None, None]
        state = (observation['state'] - state_avg) / state_std
        states = torch.stack(
            [torch.tensor(state, dtype=torch.float)] +
            [torch.ones([env.state_dim]) * model.config.ignore_value] *
            (model.n_action_to_predict - 1))

        batch['time_steps'] = torch.arange(
            t + 1, dtype=torch.long).repeat(b, 1).cuda()
        batch['images'] = torch.cat(
            [batch['images'], img.cuda()], dim=1)
        batch['states'] = torch.cat(
            [batch['states'], states[None, None].cuda()], dim=1)
        batch['actions'] = torch.cat(
            [batch['actions'], (torch.ones([b, l, model.n_action_to_predict, model.config.action_dim]).cuda() *
                                model.config.ignore_value)], dim=1)
        # length for the sequence, set to 1
        batch['seq_length'] = torch.ones(
            b, dtype=torch.long).cuda() * (t + 1)
        batch['attention_mask'] = torch.ones(
            b, t - chunk_offset + 1, dtype=torch.long).cuda()

        # use batch_pt for expanding actions
        # only include the last frame
        batch_pt.update({key: value[:, t:] for key, value in batch.items()
                         if key != 'seq_length' and key != 'attention_mask' and 'past' not in key and key not in seqence_lvl_keys})
        batch_pt['seq_length'] = torch.ones_like(batch['seq_length']) * 1
        batch_pt['time_steps'] = batch_pt['time_steps'] - chunk_offset
        # include all frames for attention mask and past_key_values
        batch_pt['attention_mask'] = batch['attention_mask']
        batch_pt['past_key_values'] = batch['past_key_values']
        # Get action from policy network
        with torch.no_grad():
            outputs = model.expand_actions(**batch_pt)
        batch['past_key_values'] = outputs.past_key_values
        batch['actions'][:, -1] = outputs.action_preds

        # Revert actions to numpy array and denormalize
        actions = (outputs.action_preds.float()[0, 0].cpu().numpy() *
                   action_std + action_avg)
        if model.motion_option == 'global':
            vs, omegas = actions[:, :3], actions[:, 3:]
            for i in range(len(actions)):
                _, _, vs[i], omegas[i] = convert_to_local_frame(
                    t_ref, q_ref, None, None, vs[i], omegas[i])
            actions = np.concatenate([vs, omegas], axis=1)
        stop = outputs.stop_preds[0].item() > 0

        # Execute action in the environment
        observation, reward, terminated, truncated, info = env.step(
            actions,
            # stop=stop,
        )
        done = terminated or truncated

        tvecs = env.tvecs[-env.n_action_to_predict:-1]
        qvecs = env.qvecs[-env.n_action_to_predict:-1]
        states = np.concatenate(
            [state[None],
             (np.concatenate([tvecs, qvecs], axis=1) - state_avg) / state_std])
        batch['states'][:, -1] = \
            torch.tensor(states, dtype=torch.float32)[None].cuda()

        # Update total reward and observation
        total_reward += reward
        crash = info['crash']
        seq_len = info['seq_len']

        t += 1

        # determine if need chunking
        if (t - chunk_offset) == chunk_size and not done:
            chunk_offset += chunk_step
            # change the reference frame for tvecs and qvecs (also v and omega if they are in global frame)
            states = (batch['states'].view(-1, env.state_dim).cpu().numpy() *
                      state_std + state_avg)
            _tvecs, _qvecs = states[:, :3], states[:, 3:]
            tvecs, qvecs = np.zeros_like(_tvecs), np.zeros_like(_qvecs)
            t_ref = _tvecs[chunk_offset * model.n_action_to_predict]
            q_ref = _qvecs[chunk_offset * model.n_action_to_predict]
            if model.motion_option == 'global':
                actions = (batch['actions'].view(-1, env.action_dim).cpu().numpy() *
                           action_std + action_avg)
                _vs, _omegas = actions[:, :3], actions[:, 3:]
                vs, omegas = np.zeros_like(_vs), np.zeros_like(_omegas)
                for i in range(len(tvecs)):
                    tvecs[i], qvecs[i], vs[i], omegas[i] = convert_to_local_frame(
                        t_ref, q_ref, _tvecs[i], _qvecs[i], _vs[i], _omegas[i])
                actions = np.concatenate([vs, omegas], axis=1)
                batch['actions'] = torch.tensor(
                    (actions - action_avg) / action_std, dtype=torch.float32).view(
                    b, t, model.n_action_to_predict, model.config.action_dim).cuda()
            else:
                for i in range(len(tvecs)):
                    tvecs[i], qvecs[i], _, _ = convert_to_local_frame(
                        t_ref, q_ref, _tvecs[i], _qvecs[i])
            states = np.concatenate([tvecs, qvecs], axis=1)
            batch['states'] = torch.tensor(
                (states - state_avg) / state_std, dtype=torch.float32).view(
                b, t, model.n_action_to_predict, model.config.state_dim).cuda()
            batch['attention_mask'] = \
                batch['attention_mask'][:, -(chunk_size - chunk_step):]
            # use batch_pt for getting the past_key_values
            batch_pt.update({key: value[:, chunk_offset:] for key, value in batch.items()
                            if key != 'seq_length' and key != 'attention_mask' and 'past' not in key and key not in seqence_lvl_keys})
            batch_pt['seq_length'] = torch.ones_like(batch['seq_length']
                                                     ) * (chunk_size - chunk_step)
            batch_pt['time_steps'] = batch_pt['time_steps'] - chunk_offset
            batch_pt['attention_mask'] = batch['attention_mask']
            batch_pt['past_key_values'] = None
            outputs = model(**batch_pt)
            # only include last chunk_offset frames
            batch['past_key_values'] = outputs.past_key_values

    # convert to video
    env.final_render(f'{"fpv" if drone_type else "nonfpv"}_{run_name}_return{total_reward:.2f}_crash{crash}',
                     mode='online_plus', re_render=re_render)

    return total_reward, crash, seq_len


def blender_simulation(config, model, logdir, num_runs=40, video_duration=10, re_render=True):
    # generated scenes
    infinigen_fpaths = {}
    for scene in sorted(os.listdir(infinigen_root)):
        if os.path.isdir(f'{infinigen_root}/{scene}'):
            for random_seed in sorted(os.listdir(f'{infinigen_root}/{scene}')):
                if os.path.isdir(f'{infinigen_root}/{scene}/{random_seed}'):
                    if os.path.exists(f'{infinigen_root}/{scene}/{random_seed}/frames/Image'):
                        if random_seed in ['0', '2b7ab387', '7c17e172', '5658d944']:
                            # skip the extra expensive ones
                            continue
                        if scene not in infinigen_fpaths:
                            infinigen_fpaths[scene] = []
                        infinigen_fpaths[scene].append(
                            f'{infinigen_root}/{scene}/{random_seed}/fine/scene.blend')

    # google map scenes
    blosm_fpaths = {}
    for city in sorted(os.listdir(blosm_root)):
        if os.path.isdir(f'{blosm_root}/{city}'):
            blosm_fpaths[city] = [f'{blosm_root}/{city}/scene.blend']

    # all runs
    scene_fpaths = []
    for scene in blosm_fpaths:
        scene_fpaths.extend(blosm_fpaths[scene])
    for i in range(int(np.ceil((num_runs - len(blosm_fpaths)) / len(infinigen_fpaths)))):
        for scene in infinigen_fpaths:
            if i < len(infinigen_fpaths[scene]):
                scene_fpaths.append(infinigen_fpaths[scene][i])
        pass
    num_repeats = np.ones(len(scene_fpaths), dtype=int) * 3
    num_repeats[:len(blosm_fpaths)] = 10

    results = []
    for i in tqdm.tqdm(range(min(num_runs, len(scene_fpaths)))):
        scene_fpath = scene_fpaths[i]
        if scene_fpath.startswith(blosm_root):
            run_name = scene_fpath.replace(blosm_root, '').split('/')[1]
        elif scene_fpath.startswith(infinigen_root):
            run_name = '_'.join(scene_fpath.replace(
                infinigen_root, '').split('/')[1:3])
        with BlenderCameraEnv(scene_fpath=scene_fpath, fps=config.fps, action_fps=config.action_fps,
                              run_dir=f'{logdir}/videos',
                              resolution=config.image_resolution, video_duration=video_duration,
                              motion_option=config.motion_option,
                              cropped_sensor_width=config.cropped_sensor_width) as env:
            for j in range(num_repeats[i]):
                seed = i * 100 + j + 1
                drone_type = (seed % len(config.drone_types)) if len(
                    config.drone_types) > 1 else config.drone_types[0]
                total_reward, crash, seq_len = expand_episode(
                    env, config, model, run_name=run_name, drone_type=drone_type, seed=seed,
                    random_init_pose=(j > 0), re_render=re_render)
                results.append({'render_fpath': scene_fpath,
                                'seed': seed,
                                'total_reward': total_reward,
                                'crash': crash,
                                'seq_len': seq_len,
                                })

    crash_rate = np.mean([result["crash"] is not None for result in results])
    avg_duration = np.mean([result["seq_len"] for result in results])
    print(f'Average return: {np.mean([result["total_reward"] for result in results])}\n'
          f'Crash rate: {crash_rate}\n'
          f'Sequence length: {avg_duration}\n'
          )

    # save the crash rate as file
    with open(f'{logdir}/crash_{crash_rate}', 'w') as f:
        f.write(f'{crash_rate}')
    # save the average duration as file
    with open(f'{logdir}/duration_{avg_duration:.2f}', 'w') as f:
        f.write(f'{avg_duration}')

    return results


if __name__ == '__main__':
    import argparse
    from transformers import set_seed
    set_seed(0)

    parser = argparse.ArgumentParser(
        description='Blender evaluation')
    # data settings
    parser.add_argument('--logdir', type=str, required=True)
    args = parser.parse_args()

    model = DVGFormerModel.from_pretrained(
        args.logdir, ignore_mismatched_sizes=True).cuda().bfloat16()

    blender_simulation(model.config, model, args.logdir,
                       num_runs=50, video_duration=10)
