import os
import re
from dataclasses import dataclass, field
from typing import Dict, Optional, Sequence, List
import numpy as np
from transforms3d.quaternions import qinverse, qconjugate, qmult, qnorm, quat2mat, mat2quat, quat2axangle, axangle2quat, nearly_equivalent
from transforms3d.euler import euler2quat, quat2euler, euler2mat, mat2euler
import pandas as pd
import torch
from src.utils.quaternion_operations import convert_to_local_frame, convert_to_global_frame, add_angular_velocity_to_quaternion, quaternions_to_angular_velocity
from src.utils.flexible_fs import FlexibleFileSystem
from src.utils.pytorch3d_rotation_conversion import euler_angles_to_matrix, matrix_to_euler_angles, quaternion_to_matrix, matrix_to_quaternion

# non fpv
# states: mean=tensor([ 1.2421e+00,  4.5750e-02,  1.1822e+01,  9.9537e-01,  1.4548e-02,
#         -1.4246e-03, -1.2822e-03]), std=tensor([4.2592e+01, 3.3490e+01, 3.1488e+01, 1.3026e-02, 6.6444e-02, 6.0955e-02,
#         2.6775e-02])
# actions: mean=tensor([ 2.1370e-02,  2.6168e-02,  1.2340e-01,  4.7324e-04, -4.9387e-05,
#         -4.2312e-05]), std=tensor([0.5899, 0.4807, 0.4158, 0.0019, 0.0018, 0.0008])
# stops: mean=0.0428212434053421
# stops: mean=0.003826059401035309 (ignore truncated sequences)
# lengths: 23.352894664466447
# norms: tvec=53.28983688354492, v=0.826926589012146, omega=0.0018915075343102217

# fpv
# states: mean=tensor([-3.2432e-01, -5.3843e+00,  3.2573e+01,  8.2942e-01,  4.8356e-02,
#         -3.5693e-02, -9.3243e-03]), std=tensor([2.4091e+01, 1.9568e+01, 3.4490e+01, 3.2701e-01, 1.6414e-01, 3.7465e-01,
#         1.8474e-01])
# actions: mean=tensor([ 0.0474, -0.0179,  0.5670,  0.0031, -0.0023, -0.0007]), std=tensor([0.4838, 0.3688, 0.5108, 0.0119, 0.0238, 0.0178])
# stops: mean=0.06011144444346428
# stops: mean=0.00889990758150816 (ignore truncated sequences)
# lengths: 16.635766986959506
# norms: tvec=45.921024322509766, v=0.8875867128372192, omega=0.019362641498446465


# state: tvec, qvec (all in global reference frame)
# state_avg = {0: np.array([0, 0, 0,
#                           0.995, 0, 0, 0]),
#              1: np.array([0, 0, 0,
#                           0.8, 0, 0, 0])}
state_avg = np.array([0, 0, 0,
                      0.8, 0, 0, 0])
# state_std = {0: np.array([40, 30, 30,
#                           0.01, 0.07, 0.06, 0.03]),
#              1: np.array([20, 20, 30,
#                           0.3, 0.2, 0.4, 0.2])}
state_std = np.array([30, 30, 30,
                      0.3, 0.3, 0.3, 0.3])

# action: v, omega (both local, relative to the current frame)
# action_avg = {0: np.zeros(6),
#               1: np.zeros(6)}
action_avg = np.zeros(6)
# action_std = {0: np.array([0.6, 0.5, 0.4,
#                            0.002, 0.002, 0.001]),
#               1: np.array([0.5, 0.4, 0.5,
#                            0.01, 0.02, 0.01])}
action_std = np.array([0.5, 0.5, 0.5,
                       0.01, 0.02, 0.01])


def get_states_actions(tvecs, qvecs, motion_option='global', action_downsample=1):
    '''
    Get the states and actions from tvecs, qvecs, vs, and omegas.
        tvec(t)
        qvec(t)
        v(t) = tvec(t+1) - tvec(t)
        omega(t) = quaternions_to_angular_velocity(qvec(t), qvec(t+1), 1)

    state = [tvec(t), qvec(t)] for frames in the global coordinate system
    action = [v(t), omega(t)] for frames in the global/local coordinate system

    Args:
        tvecs (array): (N + 1) x 3 array of translation vectors for all frames.
        qvecs (array): (N + 1) x 4 array of rotation quaternions for all frames.
        motion_option (str): 'global' or 'local' motion.
        action_downsample (int): Downsample rate for the actions.
    Returns:
        states (array): (N // action_downsample) x 7, States (position, rotation).
        actions (tensor): (N // action_downsample) x 6, Actions (velocity, angular velocity).
    '''
    # time steps
    num_frames = (len(tvecs) - 1) // action_downsample
    # States: camera pose (location and rotation) in the global coordinate system
    states = np.concatenate([tvecs, qvecs], axis=1)[
        np.arange(num_frames) * action_downsample]
    # Actions: camera motion (velocities and angular velocities) in the global/local coordinate system (relative to the *current* frame)
    # stack every action_downsample frames
    vs = np.zeros([num_frames, 3])
    omegas = np.zeros([num_frames, 3])
    for i in range(num_frames):
        vs[i] = tvecs[(i + 1) * action_downsample] - \
            tvecs[i * action_downsample]
        omegas[i] = quaternions_to_angular_velocity(
            qvecs[i * action_downsample], qvecs[(i + 1) * action_downsample], 1)
    if motion_option == 'global':
        actions = np.concatenate([vs, omegas], axis=1)
    elif motion_option == 'local':
        actions = []
        for i in range(num_frames):
            _, _, v_local, omega_local = convert_to_local_frame(
                tvecs[i * action_downsample], qvecs[i * action_downsample],
                None, None, vs[i], omegas[i])
            action = np.concatenate([v_local, omega_local])
            actions.append(action)
    else:
        raise ValueError('Invalid motion_option')
    return states, np.stack(actions)


def reverse_states_actions(states, actions, motion_option='global'):
    '''
    Reconstruct tvecs, qvecs, vs, and omegas using the states and actions.
        tvec(t)
        qvec(t)
        v(t) = tvec(t+1) - tvec(t)
        omega(t) = quaternions_to_angular_velocity(qvec(t), qvec(t+1), 1)

    for t  = [0,1,2,3,...],
    state  = [tvec(t), qvec(t)] in the global coordinate system
    action = [v(t), omega(t)] in the global/local coordinate system
    return   next_tvec/next_qvec for t+1=[1,2,3,4,...] and v/omega for t=[0,1,2,3,...] in global coord system

    Args:
        states (array): N x 7 array of states (camera pose in global coord system).
        actions (array): N x 6 array of actions (camera motion in global/local coord system).
        motion_option (str): 'global' or 'local' motion.
    Returns:
        next_tvecs (array): N x 3 array of translation vector.
        next_qvecs (array): N x 4 array of rotation quaternions.
        vs (array): N x 3 array of velocities.
        omegas (array): N x 3 array of angular.
    '''
    N = len(states)
    # global coord system (w.r.t. the initial frame)
    next_tvecs = np.zeros([N, 3])
    next_qvecs = np.zeros([N, 4])
    vs = np.zeros([N, 3])
    omegas = np.zeros([N, 3])

    # for the initial frame, the global position and orientation are specified by state
    last_tvec, last_qvec = states[0, :3], states[0, 3:]

    # Apply actions to reconstruct subsequent frames
    for i in range(N):
        if motion_option == 'global':
            vs[i], omegas[i] = actions[i][:3], actions[i][3:]
        elif motion_option == 'local':
            # convert to the global coordinate system
            _, _, vs[i], omegas[i] = convert_to_global_frame(
                states[i, :3], states[i, 3:],
                None, None, actions[i][:3], actions[i][3:])
        else:
            raise ValueError('Invalid motion_option')
        # location and rotation for the next frame
        next_tvecs[i] = last_tvec + vs[i]
        next_qvecs[i] = add_angular_velocity_to_quaternion(
            last_qvec, omegas[i], 1)
        # update the last location and rotation
        last_tvec, last_qvec = next_tvecs[i], next_qvecs[i]

    return next_tvecs, next_qvecs, vs, omegas


def reverse_states_actions_tensor(states, actions, motion_option='global'):
    '''
    Differentiable version of reverse_states_actions.
    Args:
        states (tensor): [N, 7] tensor of states (camera pose in global coord system).
        actions (tensor): [N, 6] tensor of  of actions (camera motion in global/local coord system).
    Returns:
        next_tvecs (tensor): [N, 3] tensor of translation vector.
        next_qvecs (tensor): [N, 4] tensor of rotation quaternions.
        vs (tensor): [N, 3] tensor of velocities.
        omegas (tensor): [N, 3] tensor of angular velocities.
    '''
    N = len(states)
    # global coord system (w.r.t. the initial frame)
    next_tvecs = []
    next_qvecs = []
    vs = []
    omegas = []

    # Apply actions to reconstruct subsequent frames
    for i in range(N):
        R1 = quaternion_to_matrix(states[i, 3:])
        if motion_option == 'global':
            v, omega = actions[i, :3], actions[i, 3:]
        elif motion_option == 'local':
            # convert to the global coordinate system
            v = R1 @ actions[i, :3]
            omega = R1 @ actions[i, 3:]
        else:
            raise ValueError('Invalid motion_option')
        vs.append(v)
        omegas.append(omega)
        delta_R = euler_angles_to_matrix(omega, 'XYZ')
        # location and rotation for the next frame
        next_tvec = states[i, :3] + v
        next_tvecs.append(next_tvec)
        next_R = delta_R @ R1
        next_qvec = matrix_to_quaternion(next_R)
        next_qvecs.append(next_qvec)
    next_tvecs = torch.stack(next_tvecs)
    next_qvecs = torch.stack(next_qvecs)
    vs = torch.stack(vs)
    omegas = torch.stack(omegas)

    return next_tvecs, next_qvecs, vs, omegas


def main():
    import time
    from transforms3d.quaternions import qinverse, qmult, qnorm, quat2mat, mat2quat, quat2axangle, axangle2quat, nearly_equivalent
    from src.utils.quaternion_operations import add_angular_velocity_to_quaternion, quaternions_to_angular_velocity

    root, filter_results_path = 'youtube_drone_videos', 'dataset_mini.h5'
    fps_downsample = 5
    motion_option = 'global'

    result_fpaths = []
    h5_fs = FlexibleFileSystem(
        f'{root}/{filter_results_path}')
    for video_id in sorted(h5_fs.listdir(root)):
        for result_fname in sorted(h5_fs.listdir(f'{root}/{video_id}')):
            if 'score' in result_fname and result_fname.endswith('.csv'):
                score = int(re.search(r'-score(\d+)',
                                      result_fname).group(1))
                valid = '_invalid' not in result_fname
                if score and valid:
                    result_fpath = f'{root}/{video_id}/{result_fname}'
                    result_fpaths.append(result_fpath)

    # data_index = np.random.randint(len(result_fpaths))
    data_index = 21
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

    t0 = time.time()

    # State and Action
    # global coord system (w.r.t. the initial frame)
    states, actions = get_states_actions(
        tvecs, qvecs, motion_option=motion_option)
    _next_tvecs, _next_qvecs, _vs, _omegas = reverse_states_actions(
        states, actions, motion_option=motion_option)
    __next_tvecs, __next_qvecs, __vs, __omegas = reverse_states_actions_tensor(
        torch.tensor(states, dtype=torch.float32),
        torch.tensor(actions, dtype=torch.float32),
        motion_option=motion_option)
    print(np.abs(_next_tvecs[:-1] - tvecs[1:-1]).max(),
          np.abs(_next_qvecs[:-1] - qvecs[1:-1]).max(),
          np.abs(_vs - vs).max(),
          np.abs(_omegas - omegas).max())
    print(np.abs(__next_tvecs.numpy()[:-1] - tvecs[1:-1]).max(),
          np.abs(__next_qvecs.numpy()[:-1] - qvecs[1:-1]).max(),
          np.abs(__vs.numpy() - vs).max(),
          np.abs(__omegas.numpy() - omegas).max())
    print(time.time() - t0)

    # action_downsample  = 5
    t0 = time.time()
    states_, actions_ = get_states_actions(
        tvecs, qvecs, motion_option=motion_option, action_downsample=5)
    next_tvecs_, next_qvecs_, _, _ = reverse_states_actions(
        states_, actions_, motion_option=motion_option)
    print(np.abs(next_tvecs_ - tvecs[5:-1:5]).max(),
          np.abs(next_qvecs_ - qvecs[5:-1:5]).max())
    print(time.time() - t0)
    pass


if __name__ == '__main__':
    main()
