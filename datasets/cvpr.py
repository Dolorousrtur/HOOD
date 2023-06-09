import os
import pickle
from dataclasses import dataclass, MISSING
from typing import Optional

import numpy as np
import pandas as pd
import smplx
import torch
from torch_geometric.data import HeteroData

from utils.coarse import make_coarse_edges
from utils.common import NodeType, triangles_to_edges, separate_arms, pickle_load
from utils.datasets import convert_lbs_dict
from utils.defaults import DEFAULTS
from utils.garment_smpl import GarmentSMPL


@dataclass
class Config:
    data_root: str = MISSING
    smpl_model: str = MISSING
    garment_dict_file: str = MISSING
    split_path: Optional[str] = None
    obstacle_dict_file: Optional[str] = None
    noise_scale: float = 3e-3
    lookup_steps: int = 5
    pinned_verts: bool = False
    wholeseq: bool = False
    random_betas: bool = False
    use_betas_for_restpos: bool = False
    betas_scale: float = 0.
    restpos_scale_min: float = 1.
    restpos_scale_max: float = 1.
    n_coarse_levels: int = 1
    separate_arms: bool = False
    zero_betas: bool = False
    button_edges: bool = False
    single_sequence_file: Optional[str] = None
    single_sequence_garment: Optional[str] = None


def create_loader(mcfg):
    garment_dict_path = os.path.join(DEFAULTS.aux_data, mcfg.garment_dict_file)
    garments_dict = pickle_load(garment_dict_path)

    for garment, g_dict in garments_dict.items():
        g_dict['lbs'] = convert_lbs_dict(g_dict['lbs'])

    smpl_model_path = os.path.join(DEFAULTS.aux_data, mcfg.smpl_model)
    smpl_model = smplx.SMPL(smpl_model_path)
    garment_smpl_model_dict = dict()
    for garment, g_dict in garments_dict.items():
        g_smpl_model = GarmentSMPL(smpl_model, g_dict['lbs'])
        garment_smpl_model_dict[garment] = g_smpl_model

    if mcfg.obstacle_dict_file is not None:
        obstacle_dict_path = os.path.join(DEFAULTS.aux_data, mcfg.obstacle_dict_file)
        with open(obstacle_dict_path, 'rb') as f:
            obstacle_dict = pickle.load(f)
    else:
        obstacle_dict = {}

    if mcfg.single_sequence_file is None:
        mcfg.data_root = os.path.join(DEFAULTS.vto_root, mcfg.data_root)

    loader = Loader(mcfg, garments_dict,
                    smpl_model, garment_smpl_model_dict, obstacle_dict=obstacle_dict)
    return loader


def create(mcfg: Config):
    loader = create_loader(mcfg)

    if mcfg.single_sequence_file is not None:
        datasplit = pd.DataFrame()
        datasplit['id'] = [mcfg.single_sequence_file]
        datasplit['garment'] = [mcfg.single_sequence_garment]
    else:
        split_path = os.path.join(DEFAULTS.aux_data, mcfg.split_path)
        datasplit = pd.read_csv(split_path, dtype='str')

    dataset = Dataset(loader, datasplit, wholeseq=mcfg.wholeseq)
    return dataset


class Loader:
    def __init__(self, mcfg, garments_dict, smpl_model,
                 garment_smpl_model_dict, obstacle_dict):
        self.data_path = mcfg.data_root
        self.garments_dict = garments_dict
        self.smpl_model = smpl_model
        self.garment_smpl_model_dict = garment_smpl_model_dict
        self.obstacle_dict = obstacle_dict
        self.mcfg = mcfg

    def _add_noise(self, sample):

        if self.mcfg.noise_scale == 0:
            return sample

        world_pos = sample['pos']
        node_type = sample['node_type']
        if len(node_type.shape) == 1:
            node_type = node_type[..., None]

        noise = np.random.normal(scale=self.mcfg.noise_scale, size=world_pos.shape).astype(np.float32)
        noise_prev = np.random.normal(scale=self.mcfg.noise_scale, size=world_pos.shape).astype(np.float32)
        mask = node_type == NodeType.NORMAL

        noise = noise * mask
        sample['pos'] = sample['pos'] + noise
        sample['prev_pos'] = sample['prev_pos'] + noise_prev

        return sample

    def pad_lookup(self, sample):
        if 'lookup' not in sample:
            return sample
        lookup = sample['lookup']
        n_lookup = lookup.shape[0]
        n_topad = self.mcfg.lookup_steps - n_lookup

        if n_topad == 0:
            return sample

        padlist = [lookup] + [lookup[-1:]] * n_topad
        sample['lookup'] = np.concatenate(padlist, axis=0)
        return sample

    def make_smpl_vertices(self, body_pose, global_orient, transl, betas, **kwargs):
        body_pose = torch.FloatTensor(body_pose)
        global_orient = torch.FloatTensor(global_orient)
        transl = torch.FloatTensor(transl)
        betas = torch.FloatTensor(betas)
        if len(body_pose.shape) == 1:
            body_pose = body_pose.unsqueeze(0)
            global_orient = global_orient.unsqueeze(0)
            transl = transl.unsqueeze(0)
        if len(betas.shape) == 1:
            betas = betas.unsqueeze(0)
        wholeseq = self.mcfg.wholeseq or body_pose.shape[0] > 1

        with torch.no_grad():
            smpl_output = self.smpl_model(betas=betas, body_pose=body_pose, transl=transl, global_orient=global_orient)
        vertices = smpl_output.vertices.numpy().astype(np.float32)

        if not wholeseq:
            vertices = vertices[0]

        return vertices

    def make_pyg_batch(self, sequence_dict, idx_start, idx_end=None, single=True, garment=False,
                       garment_name=None):
        make_vertices = self.make_cloth_verts if garment else self.make_smpl_vertices
        if idx_end is None and single:
            idx_end = idx_start + 1

        betas = sequence_dict['betas']
        if len(betas.shape) == 2 and betas.shape[0] != 1:
            betas = betas[idx_start: idx_end]
        verts = make_vertices(sequence_dict['body_pose'][idx_start: idx_end],
                              sequence_dict['global_orient'][idx_start: idx_end],
                              sequence_dict['transl'][idx_start: idx_end],
                              betas, garment_name=garment_name)
        return verts

    def make_obstacle_dict(self, sequence_dict, idx):
        obstacle_dict = {}

        N_steps = sequence_dict['body_pose'].shape[0]

        if self.mcfg.wholeseq:
            obstacle_dict['prev_pos'] = self.make_pyg_batch(sequence_dict, 0, -2)
            obstacle_dict['pos'] = self.make_pyg_batch(sequence_dict, 1, -1)
            obstacle_dict['target_pos'] = self.make_pyg_batch(sequence_dict, 2, None, single=False)
        else:
            obstacle_dict['prev_pos'] = self.make_pyg_batch(sequence_dict, idx)
            obstacle_dict['pos'] = self.make_pyg_batch(sequence_dict, idx + 1)
            obstacle_dict['target_pos'] = self.make_pyg_batch(sequence_dict, idx + 2)
            if self.mcfg.lookup_steps > 0:
                n_lookup = min(self.mcfg.lookup_steps, N_steps - idx - 2)
                obstacle_dict['lookup'] = self.make_pyg_batch(sequence_dict, idx + 2, idx + 2 + n_lookup)

        obstacle_dict['faces'] = self.smpl_model.faces.astype(np.int64)
        obstacle_dict = self.pad_lookup(obstacle_dict)

        N = obstacle_dict['prev_pos'].shape[1] if self.mcfg.wholeseq else obstacle_dict['prev_pos'].shape[0]

        if 'vertex_type' in self.obstacle_dict:
            obstacle_dict['node_type'] = self.obstacle_dict['vertex_type']
        else:
            obstacle_dict['node_type'] = np.ones((N, 1)).astype(np.int64)

        obstacle_dict['vertex_level'] = np.zeros((N, 1)).astype(np.int64)

        return obstacle_dict

    def make_cloth_verts(self, body_pose, global_orient, transl, betas, garment_name):
        body_pose = torch.FloatTensor(body_pose)
        global_orient = torch.FloatTensor(global_orient)
        transl = torch.FloatTensor(transl)
        betas = torch.FloatTensor(betas)

        garment_smpl_model = self.garment_smpl_model_dict[garment_name]

        if len(body_pose.shape) == 1:
            body_pose = body_pose.unsqueeze(0)
            global_orient = global_orient.unsqueeze(0)
            transl = transl.unsqueeze(0)
        if len(betas.shape) == 1:
            betas = betas.unsqueeze(0)

        wholeseq = self.mcfg.wholeseq or body_pose.shape[0] > 1
        full_pose = torch.cat([global_orient, body_pose], dim=1)

        if wholeseq and betas.shape[0] == 1:
            betas = betas.repeat(body_pose.shape[0], 1)

        with torch.no_grad():
            vertices = garment_smpl_model.make_vertices(betas=betas, full_pose=full_pose, transl=transl).numpy()

        if not wholeseq:
            vertices = vertices[0]

        return vertices

    def resize_restpos(self, restpos):
        if self.mcfg.restpos_scale_min == self.mcfg.restpos_scale_max == 1.:
            return restpos

        scale = np.random.rand()
        scale *= (self.mcfg.restpos_scale_max - self.mcfg.restpos_scale_min)
        scale += self.mcfg.restpos_scale_min

        mean = restpos.mean(axis=0, keepdims=True)
        restpos -= mean
        restpos *= scale
        restpos += mean

        return restpos

    def make_shaped_restpos(self, sequence_dict, garment_name):
        body_pose = np.zeros_like(sequence_dict['body_pose'][:1])
        global_orient = np.zeros_like(sequence_dict['global_orient'][:1])
        transl = np.zeros_like(sequence_dict['transl'][:1])
        verts = self.make_cloth_verts(body_pose,
                                      global_orient,
                                      transl,
                                      sequence_dict['betas'], garment_name=garment_name)
        return verts

    def make_vertex_level(self, coarse_edges_dict, N_nodes):
        vertex_level = np.zeros((N_nodes, 1)).astype(np.int64)
        for i in range(self.mcfg.n_coarse_levels):
            edges_coarse = coarse_edges_dict[i].astype(np.int64)
            nodes_unique = np.unique(edges_coarse.reshape(-1))
            vertex_level[nodes_unique] = i + 1
        return vertex_level

    def make_cloth_dict(self, sequence_dict, idx, garment_name):
        cloth_dict = {}
        N_steps = sequence_dict['body_pose'].shape[0]
        if self.mcfg.wholeseq:
            cloth_dict['prev_pos'] = self.make_pyg_batch(sequence_dict, 0, -2, garment=True,
                                                         garment_name=garment_name)
            cloth_dict['pos'] = self.make_pyg_batch(sequence_dict, 1, -1, garment=True, garment_name=garment_name)
            cloth_dict['target_pos'] = self.make_pyg_batch(sequence_dict, 2, None, single=False, garment=True,
                                                           garment_name=garment_name)
        else:
            cloth_dict['prev_pos'] = self.make_pyg_batch(sequence_dict, idx, garment=True,
                                                         garment_name=garment_name)
            cloth_dict['pos'] = self.make_pyg_batch(sequence_dict, idx + 1, garment=True,
                                                    garment_name=garment_name)
            cloth_dict['target_pos'] = self.make_pyg_batch(sequence_dict, idx + 2, garment=True,
                                                           garment_name=garment_name)

            if self.mcfg.lookup_steps > 0:
                n_lookup = min(self.mcfg.lookup_steps, N_steps - idx - 2)
                cloth_dict['lookup'] = self.make_pyg_batch(sequence_dict, idx + 2, idx + 2 + n_lookup,
                                                           garment=True, garment_name=garment_name)

        if self.mcfg.pinned_verts:
            cloth_dict['node_type'] = self.garments_dict[garment_name]['node_type'].astype(np.int64)
        else:
            N = cloth_dict['prev_pos'].shape[1] if self.mcfg.wholeseq else cloth_dict['prev_pos'].shape[0]
            cloth_dict['node_type'] = np.zeros((N, 1)).astype(np.int64)

        garment_dict = self.garments_dict[garment_name]
        cloth_dict['faces'] = garment_dict['faces'].astype(np.int64)

        if self.mcfg.use_betas_for_restpos:
            cloth_dict['rest_pos'] = self.make_shaped_restpos(sequence_dict, garment_name)[0]
        else:
            cloth_dict['rest_pos'] = self.resize_restpos(garment_dict['rest_pos'])

        center_nodes = garment_dict['center']
        center = np.random.choice(center_nodes)
        if 'coarse_edges' not in garment_dict:
            garment_dict['coarse_edges'] = dict()
        if center in garment_dict['coarse_edges']:
            coarse_edges_dict = garment_dict['coarse_edges'][center]
        else:
            coarse_edges_dict = make_coarse_edges(cloth_dict['faces'], center, n_levels=self.mcfg.n_coarse_levels)
            garment_dict['coarse_edges'][center] = coarse_edges_dict

        for i in range(self.mcfg.n_coarse_levels):
            edges_coarse = coarse_edges_dict[i].astype(np.int64)
            edges_coarse = np.concatenate([edges_coarse, edges_coarse[:, [1, 0]]], axis=0)
            cloth_dict[f'coarse_edge{i}'] = edges_coarse

        N = cloth_dict['prev_pos'].shape[1] if self.mcfg.wholeseq else cloth_dict['prev_pos'].shape[0]
        cloth_dict['vertex_level'] = self.make_vertex_level(coarse_edges_dict, N)

        if 'button_edges' in garment_dict:
            cloth_dict['button_edges'] = garment_dict['button_edges']

        return cloth_dict

    def convert_cloth_to_pygdata(self, cloth_dict, hetero_sample):
        faces = torch.tensor(cloth_dict['faces'])
        edges = triangles_to_edges(faces.unsqueeze(0))
        hetero_sample['cloth', 'mesh_edge', 'cloth'].edge_index = edges

        for i in range(self.mcfg.n_coarse_levels):
            key = f'coarse_edge{i}'
            coarse_edges = torch.tensor(cloth_dict[key].T)
            hetero_sample['cloth', key, 'cloth'].edge_index = coarse_edges

        hetero_sample['cloth'].faces_batch = faces.T

        if 'button_edges' in cloth_dict:
            button_edges = torch.LongTensor(cloth_dict['button_edges'])
            hetero_sample['cloth'].button_edges_batch = button_edges.T

        for k in ['pos', 'prev_pos', 'target_pos', 'rest_pos', 'node_type', 'vertex_level']:
            v = torch.tensor(cloth_dict[k])
            if len(v.shape) == 3:
                v = v.permute(1, 0, 2)
            if k == 'node_type':
                k = 'vertex_type'
            setattr(hetero_sample['cloth'], k, v)

        if 'lookup' in cloth_dict:
            lookup = torch.tensor(cloth_dict['lookup']).permute(1, 0, 2)
            hetero_sample['cloth'].lookup = lookup

        return hetero_sample

    def convert_obstacle_to_pygdata(self, obstacle_dict, hetero_sample):
        faces = torch.tensor(obstacle_dict['faces']).T
        hetero_sample['obstacle'].faces_batch = faces

        for k in ['pos', 'prev_pos', 'target_pos', 'node_type', 'vertex_level']:
            v = torch.tensor(obstacle_dict[k])
            if len(v.shape) == 3:
                v = v.permute(1, 0, 2)
            if k == 'node_type':
                k = 'vertex_type'

            setattr(hetero_sample['obstacle'], k, v)

        if 'lookup' in obstacle_dict:
            lookup = torch.tensor(obstacle_dict['lookup']).permute(1, 0, 2)
            hetero_sample['obstacle'].lookup = lookup

        return hetero_sample

    def load_sample(self, fname, idx, garment_name):
        filepath = os.path.join(self.data_path, fname + '.pkl')
        with open(filepath, 'rb') as f:
            sequence = pickle.load(f)

        if self.mcfg.separate_arms:
            body_pose = sequence['body_pose']
            global_orient = sequence['global_orient']
            full_pos = np.concatenate([global_orient, body_pose], axis=1)
            full_pos = separate_arms(full_pos)
            sequence['global_orient'] = full_pos[:, :3]
            sequence['body_pose'] = full_pos[:, 3:]

        if self.mcfg.random_betas:
            betas = sequence['betas']
            random_betas = np.random.rand(*betas.shape)
            random_betas = random_betas * self.mcfg.betas_scale * 2
            random_betas -= self.mcfg.betas_scale
            sequence['betas'] = random_betas

        sequence['body_pose'][:, -6:] *= 0

        if self.mcfg.zero_betas:
            sequence['betas'] *= 0
        sample = HeteroData()

        cloth_dict = self.make_cloth_dict(sequence, idx, garment_name)
        cloth_dict = self._add_noise(cloth_dict)
        sample = self.convert_cloth_to_pygdata(cloth_dict, sample)

        obstacle_dict = self.make_obstacle_dict(sequence, idx)
        sample = self.convert_obstacle_to_pygdata(obstacle_dict, sample)

        return sample


class Dataset:
    def __init__(self, loader: Loader, datasplit: pd.DataFrame, wholeseq=False):
        self.loader = loader
        self.datasplit = datasplit
        self.wholeseq = wholeseq

        if self.wholeseq:
            self._len = self.datasplit.shape[0]
        else:
            all_lens = datasplit.length.tolist()
            self.all_lens = [int(x) - 7 for x in all_lens]
            self._len = sum(self.all_lens)

    def _find_idx(self, index):
        fi = 0

        while self.all_lens[fi] <= index:
            index -= self.all_lens[fi]
            fi += 1

        return self.datasplit.id[fi], index, self.datasplit.garment[fi]

    def __getitem__(self, item):
        if self.wholeseq:
            fname = self.datasplit.id[item]
            garment_name = self.datasplit.garment[item]
            idx = 0
        else:
            fname, idx, garment_name = self._find_idx(item)

        sample = self.loader.load_sample(fname, idx, garment_name)
        sample['sequence_name'] = fname
        sample['garment_name'] = garment_name

        return sample

    def __len__(self):
        return self._len
