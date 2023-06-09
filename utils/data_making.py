import os

import numpy as np
from scipy.spatial.transform import Rotation
from scipy.spatial.transform import Rotation as R
from scipy.spatial.transform import Slerp

from utils.common import pickle_dump, pickle_load


def make_slerp(pA: np.ndarray, pB: np.ndarray, n_steps: int, omit_last: bool = True) -> np.ndarray:
    """
    Create a slerp path between two rotations

    :param pA: [3,] first rotation vector
    :param pB: [3,] second rotation vector
    :param n_steps: number of steps between the two rotations
    :param omit_last: whether to omit the last rotation vector
    :return: [n_steps+1, 3] or [n_steps+2, 3] interpolated rotation vectors
    """

    times = np.linspace(0.0, 1.0, n_steps + 2)
    if omit_last:
        times = times[:n_steps + 1]

    p = np.stack([pA, pB])
    p = R.from_rotvec(p)
    slerp = Slerp([0., 1.], p)
    interp = slerp(times).as_rotvec()

    return interp


def make_slerp_batch(pA: np.ndarray, pB: np.ndarray, n_steps: int, omit_last: bool = True):
    """
    Create a slerp path between two batches of rotation vectors

    :param pA: [Bx3] first batch of rotation vectors
    :param pB: [Bx3] second batch of rotation vectors
    :param n_steps: number of steps between the two rotations
    :param omit_last: whether to omit the last rotation vector

    :return: [n_steps+1, B, 3] or [n_steps+2, B, 3] interpolated rotation vectors
    """
    slerped = []
    B = pA.shape[0]

    for i in range(B):
        s = make_slerp(pA[i], pB[i], n_steps, omit_last=omit_last)
        slerped.append(s)
    slerped = np.stack(slerped, axis=1)

    return slerped


def make_interpolated_arrays(seq_dict, ind_from, n_inter_steps):
    pose_from = seq_dict['pose'][ind_from:ind_from + 1].reshape(-1, 3)
    translation_from = seq_dict['transl'][ind_from:ind_from + 1]

    pose_to = seq_dict['pose'][ind_from + 1:ind_from + 2].reshape(-1, 3)
    translation_to = seq_dict['transl'][ind_from + 1:ind_from + 2]
    linspace = np.linspace(0, 1., n_inter_steps + 2)[:n_inter_steps + 1, None]

    pose_interpolated = make_slerp_batch(pose_from, pose_to, n_inter_steps)
    pose_interpolated = pose_interpolated.reshape(n_inter_steps + 1, -1)
    translation_interpolated = (translation_to * linspace) + (translation_from * (1 - linspace))

    return pose_interpolated, translation_interpolated


def make_interpolated_dict(seq_dict, n_inter_steps, n_zeropose_interpolation_steps=0,
                           append_end_steps=0):
    n_steps = seq_dict['body_pose'].shape[0]
    betas = seq_dict['betas']
    seq_dict['pose'] = np.concatenate([seq_dict['global_orient'], seq_dict['body_pose']], axis=-1)
    pose_list = []
    translation_list = []

    start = 0
    if n_zeropose_interpolation_steps > 0:
        start = 1
        temp_dict = dict()
        temp_dict.update(seq_dict)

        temp_dict['pose'] = np.concatenate([temp_dict['pose'][:1], temp_dict['pose']], axis=0)
        temp_dict['transl'] = np.concatenate([temp_dict['transl'][:1], temp_dict['transl']], axis=0)
        temp_dict['pose'][0, 3:] = 0

        pose, translation = make_interpolated_arrays(temp_dict, 0, n_zeropose_interpolation_steps)
        pose_list.append(pose)
        translation_list.append(translation)

    if n_inter_steps > 0:
        for i in range(start, n_steps - 1):
            pose, translation = make_interpolated_arrays(seq_dict, i, n_inter_steps)

            pose_list.append(pose)
            translation_list.append(translation)
    else:
        pose_list.append(seq_dict['pose'])
        translation_list.append(seq_dict['transl'])

    pose_stack = np.concatenate(pose_list, 0)
    translation_stack = np.concatenate(translation_list, 0)

    if append_end_steps > 0:
        pose_append = np.concatenate([pose_stack[-1:, :]] * append_end_steps, axis=0)
        translation_append = np.concatenate([translation_stack[-1:, :]] * append_end_steps, axis=0)

        pose_stack = np.concatenate([pose_stack, pose_append])
        translation_stack = np.concatenate([translation_stack, translation_append])

    out_dict = dict(body_pose=pose_stack[:, 3:], global_orient=pose_stack[:, :3], transl=translation_stack, betas=betas)
    return out_dict


def adjust_global_orient(global_orient):
    theta = -np.pi / 2
    sin = np.sin(theta)
    cos = np.cos(theta)

    rot = Rotation.from_rotvec(global_orient)
    global_orient_matrix = rot.as_matrix()

    Rot = np.array([[1, 0, 0], [0, cos, -sin], [0, sin, cos]])  # Rx
    global_orient_matrix = Rot @ global_orient_matrix
    rot = Rotation.from_matrix(global_orient_matrix)
    global_orient = rot.as_rotvec()
    global_orient = global_orient

    return global_orient


def adjust_transl(transl):
    transl = transl[:, [0, 2, 1]]
    transl[:, 2] *= -1
    return transl


def convert_amass_to_pkl(amass_seq_path, out_path, start=0, n_frames=None, n_inter_steps=0, target_fps=30,
                         n_zeropose_interpolation_steps=0):
    """
    Converts a sequence from the AMASS dataset to a pkl file that can be used to generate sequences with HOOD.

    :param amass_seq_path: path to the AMASS npz file
    :param out_path: path to the output pkl file
    :param start: first frame to use (default: 0)
    :param n_frames: number of frames to use (default: None, use whole sequence)
    :param n_inter_steps: number of interpolation steps between each frame (default: 0)
    :param target_fps: target fps of the output sequence (default: 30)
    :param n_zeropose_interpolation_steps: number of interpolation steps between zeropose and the first frame
    :return:
    """
    seq_dict = np.load(amass_seq_path)
    mocap_framerate = seq_dict['mocap_framerate']

    if target_fps > mocap_framerate:
        raise ValueError(f'target_fps ({target_fps}) must be larger than the original framerate ({mocap_framerate})')

    if mocap_framerate % target_fps != 0:
        raise ValueError(f'target_fps ({target_fps}) must be a divisor of the original framerate ({mocap_framerate})')

    sparcity = int(mocap_framerate / target_fps)

    transl = seq_dict['trans'][start::sparcity]
    betas = seq_dict['betas'][:10]
    poses = seq_dict['poses'][start::sparcity]

    if n_frames is not None:
        transl = transl[:n_frames]
        poses = poses[:n_frames]

    global_orient = poses[:, :3]
    body_pose = poses[:, 3:72]

    global_orient = adjust_global_orient(global_orient)
    transl = adjust_transl(transl)

    out_dict = dict()
    out_dict['body_pose'] = body_pose
    out_dict['global_orient'] = global_orient
    out_dict['transl'] = transl
    out_dict['betas'] = betas

    out_dict = make_interpolated_dict(out_dict, n_inter_steps=n_inter_steps,
                                      n_zeropose_interpolation_steps=n_zeropose_interpolation_steps)

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    pickle_dump(out_dict, out_path)


def convert_vto_to_pkl(vto_seq_path, out_path, start=0, n_frames=None, n_inter_steps=0,
                       n_zeropose_interpolation_steps=0):
    """
    Converts a sequence from the VTO dataset to a pkl file that can be used to generate sequences with HOOD.
    :param vto_seq_path: Path to the VTO .pkl sequence file
    :param out_path: path to the output pkl file
    :param start: first frame to use (default: 0)
    :param n_frames: number of frames to use (default: None, use whole sequence)
    :param n_inter_steps: umber of interpolation steps between each frame (default: 0)
    :param n_zeropose_interpolation_steps: number of interpolation steps between zeropose and the first frame
    :return:
    """
    vto_dict = pickle_load(vto_seq_path)

    out_dict = dict()
    out_dict['transl'] = vto_dict['translation'][start:]
    out_dict['body_pose'] = vto_dict['pose'][start:, 3:72]
    out_dict['global_orient'] = vto_dict['pose'][start:, :3]
    out_dict['betas'] = vto_dict['shape'][0]

    if n_frames is not None:
        out_dict['transl'] = out_dict['transl'][:n_frames]
        out_dict['body_pose'] = out_dict['body_pose'][:n_frames]
        out_dict['global_orient'] = out_dict['global_orient'][:n_frames]

    out_dict = make_interpolated_dict(out_dict, n_inter_steps=n_inter_steps,
                                      n_zeropose_interpolation_steps=n_zeropose_interpolation_steps)

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    pickle_dump(out_dict, out_path)
