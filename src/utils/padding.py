from typing import Optional, Tuple, Union, Dict, Sequence, List
from einops import pack, unpack, repeat, reduce, rearrange, einsum
import torch
from torch import nn


def concated_seq_to_instances(concated_seq: torch.Tensor, seq_lengths: torch.LongTensor) -> Sequence[torch.Tensor]:
    '''
    Split a concatenated sequence into instances.

    Args:
        concated_seq (torch.Tensor): [batch_size, seq_length, ...]
        seq_lengths (torch.LongTensor): [batch_size]
    Returns:
        instances (Sequence[torch.Tensor]): list of tensors, each with shape [seq_length, ...]
    '''
    seq_lens = seq_lengths.tolist()
    instances = []
    # Calculate the start and end indices for each instance in the batch
    start_idx = 0
    for seq_len in seq_lens:
        # Use sequence length to determine the end index for each type of data
        end_idx = start_idx + seq_len

        # Slice each tensor to get the data for the current instance
        # Append the data for the current instance to the instances list
        instances.append(concated_seq[start_idx:end_idx])

        # Update the start index for the next instance
        start_idx = end_idx

    return instances


def padded_seq_to_instances(padded_seq: torch.Tensor, seq_lengths: torch.LongTensor) -> Sequence[torch.Tensor]:
    '''
    Slice a padded sequence into instances.
    Args:
        padded_seq (torch.Tensor): [batch_size, max_seq_length, ...]
        seq_lengths (torch.LongTensor): [batch_size]
    Returns:
        instances (Sequence[torch.Tensor]): list of tensors, each with shape [seq_length, ...]
    '''
    seq_lens = seq_lengths.tolist()
    instances = []
    for i, seq_len in enumerate(seq_lens):
        # Slice each tensor to get the data for the current instance
        # Append the data for the current instance to the instances list
        instances.append(padded_seq[i, :seq_len])
    return instances


def padding(instances, pad_side='right', pad_value=0, pad_length=None):
    '''
    Pad a list of tensors to the same length.
    Args:
        instances (Sequence[torch.Tensor]): list of tensors, each with shape [seq_length, ...]
        pad_side (str): 'left' or 'right'
        pad_value (int): the value to pad with
        pad_length (int): the length to pad to
    Returns:
        padded_seq (torch.Tensor): [batch_size, pad_length, ...]
    '''
    if pad_side == 'left':
        padded_seq = nn.utils.rnn.pad_sequence(
            # reverse the list and create tensors
            [instance[::-1] for instance in instances],
            # reverse/flip the padded tensor in first dimension
            batch_first=True, padding_value=pad_value
        ).flip(dims=[1])
    else:
        padded_seq = nn.utils.rnn.pad_sequence(
            instances, batch_first=True, padding_value=pad_value)
    if pad_length is not None:
        # batch_size, seq_length, ...
        pad_shape = list(padded_seq.shape)
        pad_shape[1] = pad_length - pad_shape[1]
        to_fix_size_seq = torch.ones(pad_shape, dtype=padded_seq.dtype,
                                     device=padded_seq.device) * pad_value
        if pad_side == 'left':
            pad_list = [to_fix_size_seq, padded_seq]
        else:
            pad_list = [padded_seq, to_fix_size_seq]
        padded_seq = torch.cat(pad_list, dim=1)
    return padded_seq
