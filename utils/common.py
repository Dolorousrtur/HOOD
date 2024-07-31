from copy import deepcopy
import enum
import math
import os
import random

import einops
import numpy as np
import torch
from omegaconf import OmegaConf
from pytorch3d.structures import Pointclouds, packed_to_list
from scipy.spatial.transform import Rotation as R


class NodeType(enum.IntEnum):
    NORMAL = 0
    OBSTACLE = 1
    OBSTACLE_OMIT = 2
    HANDLE = 3
    SIZE = 9


# TODO: add CCraft edge types ???
class EdgeType(enum.IntEnum):
    NORMAL = 0
    BUTTON = 1
    LONG_RANGE = 2
    SIZE = 3


class EdgeTypeCC(enum.IntEnum):
    NORMAL = 0
    FUTUREPINNED = 1
    SIZE = 2


def move2device(data, device):
    """
    Move the given data to the given device (e.g. `cuda:0`).
    """
    if hasattr(data, 'to'):
        return data.to(device)
    elif type(data) == dict:
        out = {}
        for k, v in data.items():
            out[k] = move2device(v, device)
    elif type(data) == list:
        out = [move2device(x, device) for x in data]
    else:
        out = data

    return out


def detach_dict(data):
    """
    Detach all items in the given object (dict, list, etc.) from the graph.
    """
    if hasattr(data, 'detach'):
        return data.detach()
    elif type(data) == dict:
        out = {}
        for k, v in data.items():
            out[k] = detach_dict(v)
    elif type(data) == list:
        out = [detach_dict(x) for x in data]
    else:
        out = data

    return out


def set_manual_seed(seed: int):
    """
    Set the random seed for all possible random number generators.
    """
    torch.random.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


def triangles_to_edges(faces: torch.LongTensor, links: torch.LongTensor = None, two_way=True):
    """Computes mesh edges from triangles."""

    # collect edges from triangles
    edges_list = [faces[..., 0:2],
                  faces[..., 1:3],
                  torch.stack([faces[..., 2], faces[..., 0]], dim=-1)]
    edges = torch.cat(edges_list, dim=1)

    # those edges are sometimes duplicated (within the mesh) and sometimes
    # single (at the mesh boundary).
    # sort & pack edges as single tf.int64
    receivers = edges.min(dim=-1)[0]
    senders = edges.max(dim=-1)[0]

    if links is not None:
        senders_links, receivers_links = torch.unbind(links, dim=-1)
        senders = torch.cat([senders, senders_links], dim=1)
        receivers = torch.cat([receivers, receivers_links], dim=1)

    packed_edges = torch.stack([senders, receivers], dim=-1)
    unique_edges = torch.unique(packed_edges, dim=1)

    id_offset = torch.max(unique_edges) + 1

    sortvals = unique_edges[..., 0] * id_offset + unique_edges[..., 1]
    sort_idx = torch.sort(sortvals, dim=-1).indices[0]
    unique_edges = unique_edges[:, sort_idx]
    senders, receivers = torch.unbind(unique_edges, dim=-1)

    if two_way:
        # create two-way connectivity
        all_senders = torch.cat([senders, receivers], dim=1)
        all_receivers = torch.cat([receivers, senders], dim=1)
    else:
        all_senders = senders
        all_receivers = receivers

    edges = torch.cat([all_receivers, all_senders], dim=0)
    return edges



def make_einops_str(ndims, insert_k=None):
    linds = ['l', 'm', 'n', 'o', 'p']

    if insert_k is None:
        symbols = linds[:ndims]
    else:
        symbols = linds[:insert_k]
        symbols.append('k')
        symbols += linds[insert_k:ndims]

    out_str = ' '.join(symbols)
    return out_str


def make_repeat_str(tensor, dim):
    ndims = len(tensor.shape)

    out_str = []
    out_str.append(make_einops_str(ndims))
    out_str.append('->')
    out_str.append(make_einops_str(ndims, insert_k=dim))

    out_str = ' '.join(out_str)

    return out_str


def gather(data: torch.Tensor, index: torch.LongTensor, dim_gather: int, dim_data: int, dim_index: int):
    input_repeat_str = make_repeat_str(data, dim_data)
    index_repeat_str = make_repeat_str(index, dim_index + 1)

    data_repeat = einops.repeat(data, input_repeat_str, k=index.shape[dim_index])
    index_repeat = einops.repeat(index, index_repeat_str, k=data.shape[dim_data])

    out = torch.gather(data_repeat, dim_gather, index_repeat)

    return out


def unsorted_segment_sum(data, index, dim_sum: int, dim_input: int, dim_index: int, n_verts=None):
    input_repeat_str = make_repeat_str(data, dim_input)
    index_repeat_str = make_repeat_str(index, dim_index + 1)

    data_repeat = einops.repeat(data, input_repeat_str, k=index.shape[dim_index])
    index_repeat = einops.repeat(index, index_repeat_str, k=data.shape[dim_input])

    B = data.shape[:dim_sum]
    n_verts = n_verts or index.max().item() + 1

    out = torch.zeros(*B, n_verts, index.shape[dim_index], data.shape[dim_input]).to(data.device)
    out = out.scatter_add(dim_sum, index_repeat, data_repeat)
    out = out.sum(dim=-2)

    return out


def save_checkpoint(runner, aux_modules, config, file):
    """
    Save a checkpoint of the training state.
    :param runner: Runner object
    :param aux_modules: a dictionary of auxiliary modules (optimizer, scheduler) to save
    :param config: OmegaConf config object
    :param file: path to save the checkpoint
    """

    os.makedirs(os.path.dirname(file), exist_ok=True)
    out_dict = dict()
    out_dict['training_module'] = runner.state_dict()
    out_dict['config'] = OmegaConf.to_container(config)
    for k, v in aux_modules.items():
        if hasattr(v, 'state_dict'):
            out_dict[k] = v.state_dict()

    torch.save(out_dict, file)


def make_pervertex_tensor_from_lens(lens, val_tensor):
    val_list = []
    for i, n in enumerate(lens):
        val_list.append(val_tensor[i].repeat(n).unsqueeze(-1))
    val_stack = torch.cat(val_list)
    return val_stack


def add_field_to_pyg_batch(batch, new_key, value, node_key, reference_key=None, one_per_sample=False, zero_inc=False, new_dim=False):
    batch[node_key]._mapping[new_key] = value
    B = batch.num_graphs

    if new_dim:
        device = value.device
        N = value.shape[0]
        slice = torch.LongTensor([0, N])
        inc = torch.zeros(B).long().to(device)

        batch._slice_dict[node_key][new_key] = slice
        batch._inc_dict[node_key][new_key] = inc
    elif reference_key is None:
        if one_per_sample:
            device = value.device
            slice = torch.arange(B + 1).to(device)
            inc = torch.zeros(B).long().to(device)
        else:
            slice = []
            inc = []

        batch._slice_dict[node_key][new_key] = slice
        batch._inc_dict[node_key][new_key] = inc
    else:
        batch._slice_dict[node_key][new_key] = batch._slice_dict[node_key][reference_key]

        if zero_inc:
            device = value.device
            inc = torch.zeros(B).long().to(device)
        else:
            inc = batch._inc_dict[node_key][reference_key]
        batch._inc_dict[node_key][new_key] = inc
    return batch


def random_between(fr, to, shape, return_norm=False, device=None):
    """
    Generate a random tensor with values between `fr` and `to`.
    :param fr: minimum value
    :param to: maximum value
    :param shape: shape of the output tensor
    :param return_norm: if True, return the normalized tensor (with values in [0,1]) as well
    :param device: torch device
    :return: a random tensor
    """
    if fr == to:
        rand_norm = torch.zeros(*shape)
    else:
        rand_norm = torch.rand(*shape)
    if device is not None:
        rand_norm = rand_norm.to(device)

    rand = rand_norm * (to - fr)
    rand += fr

    if return_norm:
        return rand, rand_norm
    return rand


def relative_between(fr, to, value: torch.Tensor):
    """
    Normalize a tensor with values between `fr` and `to` to the range [0,1].
    :param fr: minimum value
    :param to: maximum value
    :param value: tensor to normalize
    :return: normalized tensor
    """
    if fr == to:
        return torch.zeros_like(value)

    value_norm = value - fr
    value_norm = value_norm / (to - fr)
    return value_norm


def random_between_log(fr, to, shape, return_norm=False, device=None):
    """
    Generate a random tensor with values between `fr` and `to` sampled from a log scale.
    :param fr: minimum value
    :param to: maximum value
    :param shape: shape of the output tensor
    :param return_norm: if True, return the normalized tensor (with values in [0,1]) as well
    :param device: torch device
    :return: a random tensor
    """
    if fr == to:
        rand_norm = torch.zeros(*shape)
    else:
        rand_norm = torch.rand(*shape)
    if device is not None:
        rand_norm = rand_norm.to(device)

    fr_log = math.log(fr)
    to_log = math.log(to)

    rand_log = rand_norm * (to_log - fr_log)
    rand_log += fr_log

    rand = torch.exp(rand_log)

    if return_norm:
        return rand, rand_norm
    return rand


def relative_between_log(fr, to, value: torch.Tensor):
    """
    Normalize a tensor with values between `fr` and `to` to the range [0,1] using a log scale.
    :param fr: minimum value
    :param to: maximum value
    :param value: tensor to normalize
    :return: normalized tensor
    """
    if fr == to:
        return torch.zeros_like(value)

    fr_log = math.log(fr)
    to_log = math.log(to)

    if type(value) == torch.Tensor:
        value_log = torch.log(value)
    else:
        value_log = math.log(value)

    value_norm = value_log - fr_log
    value_norm = value_norm / (to_log - fr_log)

    return value_norm

def separate_arms(poses: np.ndarray, angle=20, left_arm=17, right_arm=16):
    """
    Modify the SMPL poses to avoid self-intersections of the arms and the body.
    Adapted from the code of SNUG (https://github.com/isantesteban/snug/blob/main/snug_utils.py#L93)

    :param poses: [Nx72] SMPL poses
    :param angle: angle to rotate the arms
    :param left_arm: index of the left arm in the SMPL model
    :param right_arm: index of the right arm in the SMPL model
    :return:
    """
    num_joints = poses.shape[-1] // 3

    poses = poses.reshape((-1, num_joints, 3))
    rot = R.from_euler('z', -angle, degrees=True)
    poses[:, left_arm] = (rot * R.from_rotvec(poses[:, left_arm])).as_rotvec()
    rot = R.from_euler('z', angle, degrees=True)
    poses[:, right_arm] = (rot * R.from_rotvec(poses[:, right_arm])).as_rotvec()

    # poses[:, 23] *= 0.1
    # poses[:, 22] *= 0.1

    return poses.reshape((poses.shape[0], -1))


def create_zero_pointclouds(pc_list):
    zero_list = []

    for pc in pc_list:
        zshape = list(pc.shape)
        zshape[-1] = 3
        zero_tensor = torch.zeros(*zshape).to(pc.device)
        zero_list.append(zero_tensor)

        # print(pc.shape)

    pointclouds = Pointclouds(zero_list, features=pc_list)
    return pointclouds

def pcd_replace_features_packed(pcd, features_packed):
    npppc = pcd.num_points_per_cloud().detach().cpu().numpy().tolist()
    features_list = list(packed_to_list(features_packed, npppc))
    updated_pcd = create_zero_pointclouds(features_list)
    return updated_pcd


def copy_pyg_batch(batch):
    batch_copy = batch.detach().clone()
    batch_copy['cloth']._mapping = deepcopy(batch_copy['cloth']._mapping)
    batch_copy._slice_dict = deepcopy(batch_copy._slice_dict)
    batch_copy._inc_dict = deepcopy(batch_copy._inc_dict)
    return batch_copy

class TorchTimer(object):

    def __init__(self, time_dict, label, start=None, end=None, to_print=False):
        if start is None:
            start = torch.cuda.Event(enable_timing=True)
        if end is None:
            end = torch.cuda.Event(enable_timing=True)

        self.time_dict = time_dict
        self.label = label
        self.to_print = to_print
        self.start = start
        self.end = end

        # self.args = args

    def __enter__(self):
        self.start.record()
        return self

    def __call__(self, *args):
        return self

    # def __exit__(self, _type=None, value=None, traceback=None):
    def __exit__(self, *args):
        self.end.record()
        torch.cuda.synchronize()
        t = self.start.elapsed_time(self.end)

        if self.time_dict is not None:
            if type(self.time_dict[self.label]) == list:
                self.time_dict[self.label].append(t)
            else:
                self.time_dict[self.label] = t

        if self.to_print:
            print(f"{self.label}: {t:.2f} ms.")