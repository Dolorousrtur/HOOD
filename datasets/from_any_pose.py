import os
import pickle
from dataclasses import dataclass, MISSING
from typing import Optional, Dict, Tuple

import numpy as np
import pandas as pd
import smplx
import torch
from smplx import SMPL
from torch_geometric.data import HeteroData

from utils.coarse import make_coarse_edges
from utils.common import NodeType, triangles_to_edges, separate_arms, pickle_load
from utils.defaults import DEFAULTS
from utils.mesh_creation import add_coarse_edges, obj2template
import warnings

@dataclass
class Config:
    pose_sequence_path: str = MISSING  #  Path to the pose sequence relative to $HOOD_DATA. Can be either sequence of SMLP parameters of a sequence of meshes depending on the value of pose_sequence_type
    garment_template_path: str = MISSING  # Path to the garment template relative to $HOOD_DATA. Can  be either .obj file or preprocessed or .pkl file (see utils/mesh_creation::obj2template)
    pose_sequence_type: str = "smpl"  # "smpl" | "mesh" if "smpl" the pose_sequence_path is a sequence of SMPL parameters, if "mesh" the pose_sequence_path is a sequence of meshes
    smpl_model: Optional[str] = None  # Path to the SMPL model relative to $HOOD_DATA/aux_data/
    obstacle_dict_file: Optional[str] = None  # Path to the file with auxiliary data for obstacles relative to $HOOD_DATA/aux_data/
    n_coarse_levels: int = 4  # Number of coarse levels with long-range edges
    separate_arms: bool = False  # Whether to separate the arms from the rest of the body (to avoid body self-intersections)



def make_obstacle_dict(mcfg: Config) -> dict:
    if mcfg.obstacle_dict_file is None:
        return {}

    obstacle_dict_path = os.path.join(DEFAULTS.aux_data, mcfg.obstacle_dict_file)
    with open(obstacle_dict_path, 'rb') as f:
        obstacle_dict = pickle.load(f)
    return obstacle_dict


def create_loader(mcfg: Config):

    garment_template_path = os.path.join(DEFAULTS.data_root, mcfg.garment_template_path)

    if garment_template_path.endswith('.obj'):

        garment_dict = obj2template(garment_template_path)
        warnings.warn("""Loading from garment geometry from .obj. \n
        It may take a while to build coarse edges. \n
        Consider converting the garment to .pkl using utils/mesh_creation::obj2template())""")

    elif garment_template_path.endswith('.pkl'):
        garment_dict = pickle_load(garment_template_path)
    else:
        raise ValueError(f'Unknown garment template format: {mcfg.garment_template_path}, has to be .obj or .pkl')

    if mcfg.smpl_model is None:
        smpl_model = None
    else:
        smpl_model_path = os.path.join(DEFAULTS.aux_data, mcfg.smpl_model)
        smpl_model = smplx.SMPL(smpl_model_path)

    obstacle_dict = make_obstacle_dict(mcfg)

    loader = Loader(mcfg, garment_dict, obstacle_dict, smpl_model)
    return loader


def create(mcfg: Config):
    loader = create_loader(mcfg)

    pose_sequence_path = os.path.join(DEFAULTS.data_root, mcfg.pose_sequence_path)
    dataset = Dataset(loader, pose_sequence_path)
    return dataset


class VertexBuilder:
    """
    Helper class to build garment and body vertices from a sequence of SMPL poses.
    """

    def __init__(self, mcfg):
        self.mcfg = mcfg

    @staticmethod
    def build(sequence_dict: dict, f_make, idx_start: int, idx_end: int = None, garment_name: str = None) -> np.ndarray:
        """
        Build vertices from a sequence of SMPL poses using the given `f_make` function.
        :param sequence_dict: a dictionary of SMPL parameters
        :param f_make: a function that takes SMPL parameters and returns vertices
        :param idx_start: first frame index
        :param idx_end: last frame index
        :param garment_name: name of the garment (None for body)
        :return: [Nx3] mesh vertices
        """
        for k in ['body_pose', 'global_orient', 'transl']:
            sequence_dict[k] = sequence_dict[k][idx_start: idx_end]

        N = sequence_dict['body_pose'].shape[0]
        sequence_dict['betas'] = np.tile(sequence_dict['betas'], (N, 1))

        verts = f_make(sequence_dict, garment_name=garment_name)

        return verts

    def pos2tensor(self, pos: np.ndarray) -> torch.Tensor:
        """
        Convert a numpy array of vertices to a tensor and permute the axes into [VxNx3] (torch geometric format)
        """
        pos = torch.tensor(pos).permute(1, 0, 2)
        return pos

    def add_verts(self, sample: HeteroData, sequence_dict: dict, f_make, object_key: str,
                  **kwargs) -> HeteroData:
        """
        Builds the vertices from the given SMPL pose sequence and adds them to the HeteroData sample.
        :param sample: HetereoData object
        :param sequence_dict: sequence of SMPL parameters
        :param f_make: function that takes SMPL parameters and returns vertices
        :param object_key: name of the object to build vertices for ('cloth' or 'obstacle')
        :return: updated HeteroData object
        """

        pos_dict = {}

        # Build the vertices for the whole sequence
        all_vertices = VertexBuilder.build(sequence_dict, f_make, 0, None,
                                           **kwargs)
        pos_dict['prev_pos'] = all_vertices
        pos_dict['pos'] = all_vertices
        pos_dict['target_pos'] = all_vertices


        for k, v in pos_dict.items():
            v = self.pos2tensor(v)
            setattr(sample[object_key], k, v)

        return sample



class GarmentBuilder:
    """
    Class to build the garment meshes from SMPL parameters.
    """

    def __init__(self, mcfg: Config, garment_dict: dict):
        """
        :param mcfg: config
        :param garments_dict: dictionary with data for all garments
        """
        self.mcfg = mcfg
        self.garment_dict = garment_dict
        self.vertex_builder = VertexBuilder(mcfg)

    def add_verts(self, sample: HeteroData, garment_dict: dict) -> HeteroData:
        pos = garment_dict['vertices']
        pos = torch.FloatTensor(pos)[None,].permute(1, 0, 2)

        sample['cloth'].prev_pos = pos
        sample['cloth'].pos = pos
        sample['cloth'].target_pos = pos
        sample['cloth'].rest_pos = pos[:, 0]

        return sample

    def add_vertex_type(self, sample: HeteroData, garment_dict: dict) -> HeteroData:
        """
        Add `vertex_type` tensor to `sample['cloth']`

        utils.common.NodeType.NORMAL (0) for normal vertices
        utils.common.NodeType.HANDLE (3) for pinned vertices

        if `self.mcfg.pinned_verts` == True, take `vertex_type` from `self.garments_dict`
        else: fill all with utils.common.NodeType.NORMAL (0)

        :param sample: HeteroData sample
        :param garment_name: name of the garment in `self.garments_dict`

        :return: sample['cloth'].vertex_type: torch.LongTensor [Vx1]
        """

        V = sample['cloth'].pos.shape[0]
        vertex_type = np.zeros((V, 1)).astype(np.int64)

        sample['cloth'].vertex_type = torch.tensor(vertex_type)
        return sample


    def add_faces_and_edges(self, sample: HeteroData, garment_dict: dict) -> HeteroData:
        """
        Add garment faces to `sample['cloth']`
        Add bi-directional edges to `sample['cloth', 'mesh_edge', 'cloth']`

        :param sample: HeteroData
        :param garment_name: name of the garment in `self.garment_smpl_model_dict` and `self.garments_dict`
        :return:
            sample['cloth'].faces_batch: torch.LongTensor [3xF]
            ample['cloth', 'mesh_edge', 'cloth'].edge_index: torch.LongTensor [2xE]
        """

        faces = torch.LongTensor(garment_dict['faces'])
        edges = triangles_to_edges(faces.unsqueeze(0))
        sample['cloth', 'mesh_edge', 'cloth'].edge_index = edges

        sample['cloth'].faces_batch = faces.T

        return sample

    def make_vertex_level(self, sample: HeteroData, coarse_edges_dict: Dict[int, np.array]) -> HeteroData:
        """
        Add `vertex_level` labels to `sample['cloth']`
        for each garment vertex, `vertex_level` is the number of the deepest level the vertex is in
        starting from `0` for the most shallow level

        :param sample: HeteroData
        :param coarse_edges_dict: dictionary with list of edges for each coarse level
        :return: sample['cloth'].vertex_level: torch.LongTensor [Vx1]
        """
        N = sample['cloth'].pos.shape[0]
        vertex_level = np.zeros((N, 1)).astype(np.int64)
        for i in range(self.mcfg.n_coarse_levels):
            edges_coarse = coarse_edges_dict[i].astype(np.int64)
            nodes_unique = np.unique(edges_coarse.reshape(-1))
            vertex_level[nodes_unique] = i + 1
        sample['cloth'].vertex_level = torch.tensor(vertex_level)
        return sample

    def add_coarse(self, sample: HeteroData, garment_dict: dict) -> HeteroData:
        """
        Add coarse edges to `sample` as `sample['cloth', f'coarse_edge{i}', 'cloth'].edge_index`.
        where `i` is the number of the coarse level (starting from `0`)

        :param sample: HeteroData
        :param garment_name:
        :return: sample['cloth', f'coarse_edge{i}', 'cloth'].edge_index: torch.LongTensor [2, E_i]
        """
        if self.mcfg.n_coarse_levels == 0:
            return sample

        faces = garment_dict['faces']

        # Randomly choose center of the mesh
        # center of a graph is a node with minimal eccentricity (distance to the farthest node)
        if 'center' not in garment_dict:
            garment_dict = add_coarse_edges(garment_dict, self.mcfg.n_coarse_levels)

        center_nodes = garment_dict['center']
        center = np.random.choice(center_nodes)
        if 'coarse_edges' not in garment_dict:
            garment_dict['coarse_edges'] = dict()

        # if coarse edges are already precomputed for the given `center`,
        # take them from `garment_dict['coarse_edges'][center]`
        # else compute them with `make_coarse_edges` and stash in `garment_dict['coarse_edges'][center]`
        if center in garment_dict['coarse_edges']:
            coarse_edges_dict = garment_dict['coarse_edges'][center]
        else:
            coarse_edges_dict = make_coarse_edges(faces, center, n_levels=self.mcfg.n_coarse_levels)
            garment_dict['coarse_edges'][center] = coarse_edges_dict

        # for each level `i` add edges to sample as  `sample['cloth', f'coarse_edge{i}', 'cloth'].edge_index`
        for i in range(self.mcfg.n_coarse_levels):
            key = f'coarse_edge{i}'
            edges_coarse = coarse_edges_dict[i].astype(np.int64)
            edges_coarse = np.concatenate([edges_coarse, edges_coarse[:, [1, 0]]], axis=0)
            coarse_edges = torch.tensor(edges_coarse.T)
            sample['cloth', key, 'cloth'].edge_index = coarse_edges

        # add `vertex_level` labels to sample
        sample = self.make_vertex_level(sample, coarse_edges_dict)

        return sample

    def build(self, sample: HeteroData) -> HeteroData:
        """
        Add all data for the garment to the sample

        :param sample: HeteroData
        :return:
            sample['cloth'].pos torch.FloatTensor [VxNx3]: vertex positions at the current frame
            sample['cloth'].rest_pos torch.FloatTensor [Vx3]: vertex positions in the canonical pose
            sample['cloth'].faces_batch torch.LongTensor [3xF]: garment faces
            sample['cloth'].vertex_type torch.LongTensor [Vx1]: vertex type (0 - regular, 3 - pinned)
            sample['cloth'].vertex_level torch.LongTensor [Vx1]: level of the vertex in the hierarchy (always 0 for the body)

            sample['cloth', 'mesh_edge', 'cloth'].edge_index: torch.LongTensor [2xE]: mesh edges

            for each coarse level `i` in [0, self.mcfg.n_coarse_levels]:
                sample['cloth', f'coarse_edge{i}', 'cloth'].edge_index: torch.LongTensor [2, E_i]: coarse edges at level `i`

        """
        sample = self.add_verts(sample, self.garment_dict)
        sample = self.add_coarse(sample, self.garment_dict)
        sample = self.add_vertex_type(sample, self.garment_dict)
        sample = self.add_faces_and_edges(sample, self.garment_dict)

        return sample


class SMPLBodyBuilder:
    """
    Class for building body meshed from SMPL parameters
    """

    def __init__(self, mcfg: Config, smpl_model: SMPL, obstacle_dict: dict):
        """
        :param mcfg: Config
        :param smpl_model:
        :param obstacle_dict: auxiliary data for the obstacle
                obstacle_dict['vertex_type']: vertex type (1 - regular obstacle node, 2 - hand node (omitted during inference to avoid body self-penetrations))
        """
        self.smpl_model = smpl_model
        self.obstacle_dict = obstacle_dict
        self.mcfg = mcfg
        self.vertex_builder = VertexBuilder(mcfg)

    def make_smpl_vertices(self, sequence_dict, **kwargs) -> np.ndarray:
        """
        Create body vertices from SMPL parameters (used in VertexBuilder.add_verts)

        :param sequence_dict: dict with SMPL parameters:
            body_pose: SMPL pose parameters [Nx69]
            global_orient: SMPL global_orient [Nx3]
            transl: SMPL translation [Nx3]
            betas: SMPL betas [Nx10]

        :return: vertices [NxVx3]
        """

        input_dict = {k: torch.FloatTensor(v) for k, v in sequence_dict.items()}

        for k, v in input_dict.items():
            print(k, v.shape)

        with torch.no_grad():
            smpl_output = self.smpl_model(**input_dict)
        vertices = smpl_output.vertices.numpy().astype(np.float32)

        return vertices

    def add_vertex_type(self, sample: HeteroData) -> HeteroData:
        """
        Add vertex type field to the obstacle object in the sample
        """
        N = sample['obstacle'].pos.shape[0]
        if 'vertex_type' in self.obstacle_dict:
            vertex_type = self.obstacle_dict['vertex_type']
        else:
            vertex_type = np.ones((N, 1)).astype(np.int64)
        sample['obstacle'].vertex_type = torch.tensor(vertex_type)
        return sample

    def add_faces(self, sample: HeteroData) -> HeteroData:
        """
        Add body faces to the obstacle object in the sample
        """
        faces = torch.LongTensor(self.smpl_model.faces.astype(np.int64))
        sample['obstacle'].faces_batch = faces.T
        return sample

    def add_vertex_level(self, sample: HeteroData) -> HeteroData:
        """
        Add vertex level field to the obstacle object in the sample (always 0 for the body)
        """
        N = sample['obstacle'].pos.shape[0]
        vertex_level = torch.zeros(N, 1).long()
        sample['obstacle'].vertex_level = vertex_level
        return sample

    def build(self, sample: HeteroData, sequence_dict: dict) -> HeteroData:
        """
        Add all data for the body (obstacle) to the sample
        :param sample: HeteroData object to add data to
        :param sequence_dict: dict with SMPL parameters

        :return:
            sample['obstacle'].prev_pos torch.FloatTensor [VxNx3]: vertex positions at the previous frame
            sample['obstacle'].pos torch.FloatTensor [VxNx3]: vertex positions at the current frame
            sample['obstacle'].target_pos torch.FloatTensor [VxNx3]: vertex positions at the next frame

            sample['obstacle'].faces_batch torch.LongTensor [3xF]: garment faces
            sample['obstacle'].vertex_type torch.LongTensor [Vx1]: vertex type (1 - regular obstacle, 2 - omitted)
            sample['obstacle'].vertex_level torch.LongTensor [Vx1]: level of the vertex in the hierarchy (always 0 for the body)

        """
        sample = self.vertex_builder.add_verts(sample, sequence_dict, self.make_smpl_vertices, "obstacle")
        sample = self.add_vertex_type(sample)
        sample = self.add_faces(sample)
        sample = self.add_vertex_level(sample)
        return sample


class BareMeshBodyBuilder:
    """
    Class for building body meshed from SMPL parameters
    """

    def __init__(self, mcfg: Config, obstacle_dict: dict):
        """
        :param mcfg: Config
        :param obstacle_dict: auxiliary data for the obstacle
                obstacle_dict['vertex_type']: vertex type (1 - regular obstacle node, 2 - hand node (omitted during inference to avoid body self-penetrations))
        """
        self.obstacle_dict = obstacle_dict
        self.mcfg = mcfg
        self.vertex_builder = VertexBuilder(mcfg)

    def add_vertex_type(self, sample: HeteroData) -> HeteroData:
        """
        Add vertex type field to the obstacle object in the sample
        """
        N = sample['obstacle'].pos.shape[0]
        if 'vertex_type' in self.obstacle_dict:
            vertex_type = self.obstacle_dict['vertex_type']
        else:
            vertex_type = np.ones((N, 1)).astype(np.int64)
        sample['obstacle'].vertex_type = torch.tensor(vertex_type)
        return sample

    def add_faces(self, sample: HeteroData, sequence_dict: dict) -> HeteroData:
        """
        Add body faces to the obstacle object in the sample
        """
        faces = torch.LongTensor(sequence_dict['faces'].astype(np.int64))
        sample['obstacle'].faces_batch = faces.T
        return sample

    def add_verts(self, sample: HeteroData, sequence_dict: dict) -> HeteroData:
        """
        Add body vertices to the obstacle object in the sample
        """

        pos = torch.FloatTensor(sequence_dict["verts"]).permute(1, 0, 2)

        sample['obstacle'].prev_pos = pos
        sample['obstacle'].pos = pos
        sample['obstacle'].target_pos = pos
        return sample

    def add_vertex_level(self, sample: HeteroData) -> HeteroData:
        """
        Add vertex level field to the obstacle object in the sample (always 0 for the body)
        """
        N = sample['obstacle'].pos.shape[0]
        vertex_level = torch.zeros(N, 1).long()
        sample['obstacle'].vertex_level = vertex_level
        return sample

    def build(self, sample: HeteroData, sequence_dict: dict) -> HeteroData:
        """
        Add all data for the body (obstacle) to the sample
        :param sample: HeteroData object to add data to
        :param sequence_dict: dict with SMPL parameters

        :return:
            sample['obstacle'].prev_pos torch.FloatTensor [VxNx3]: vertex positions at the previous frame
            sample['obstacle'].pos torch.FloatTensor [VxNx3]: vertex positions at the current frame
            sample['obstacle'].target_pos torch.FloatTensor [VxNx3]: vertex positions at the next frame

            sample['obstacle'].faces_batch torch.LongTensor [3xF]: garment faces
            sample['obstacle'].vertex_type torch.LongTensor [Vx1]: vertex type (1 - regular obstacle, 2 - omitted)
            sample['obstacle'].vertex_level torch.LongTensor [Vx1]: level of the vertex in the hierarchy (always 0 for the body)

        """
        sample = self.add_verts(sample, sequence_dict)
        sample = self.add_vertex_type(sample)
        sample = self.add_faces(sample, sequence_dict)
        sample = self.add_vertex_level(sample)
        return sample

class SequenceLoader:
    def __init__(self, mcfg):
        self.mcfg = mcfg

    def process_sequence(self, sequence: dict) -> dict:
        """
        Apply transformations to the SMPL sequence
        :param sequence: dict with SMPL parameters
        :return: processed dict with SMPL parameters
        """
        # from SNUG, eliminates hand-body penetrations
        if self.mcfg.separate_arms:
            body_pose = sequence['body_pose']
            global_orient = sequence['global_orient']
            full_pos = np.concatenate([global_orient, body_pose], axis=1)
            full_pos = separate_arms(full_pos)
            sequence['global_orient'] = full_pos[:, :3]
            sequence['body_pose'] = full_pos[:, 3:]

        # zero-out hand pose (eliminates unrealistic hand poses)
        sequence['body_pose'][:, -6:] *= 0

        return sequence

    def load_sequence(self, fname: str) -> dict:
        """
        Load sequence of SMPL parameters from disc
        and process it

        :param fname: file name of the sequence
        :return: dict with SMPL parameters:
            sequence['body_pose'] np.array [Nx69]
            sequence['global_orient'] np.array [Nx3]
            sequence['transl'] np.array [Nx3]
            sequence['betas'] np.array [10]
        """
        with open(fname, 'rb') as f:
            sequence = pickle.load(f)

        if self.mcfg.pose_sequence_type == 'smpl':
            sequence = self.process_sequence(sequence)

        return sequence


class Loader:
    """
    Class for building HeteroData objects containing all data for a single sample
    """

    def __init__(self, mcfg: Config, garment_dict: dict, obstacle_dict: dict, smpl_model: SMPL = None):
        self.sequence_loader = SequenceLoader(mcfg)
        self.garment_builder = GarmentBuilder(mcfg, garment_dict)

        if mcfg.pose_sequence_type == 'smpl':
            self.body_builder = SMPLBodyBuilder(mcfg, smpl_model, obstacle_dict)
        elif mcfg.pose_sequence_type == 'mesh':
            self.body_builder = BareMeshBodyBuilder(mcfg, obstacle_dict)
        else:
            raise ValueError(f'Unknown pose sequence type {mcfg.pose_sequence_type}. Should be "smpl" or "mesh"')


    def load_sample(self, fname: str) -> HeteroData:
        """
        Build HeteroData object for a single sample
        :param fname: path to the pose sequence file
        :return: HelteroData object (see BodyBuilder.build and GarmentBuilder.build for details)
        """
        sequence = self.sequence_loader.load_sequence(fname)
        sample = HeteroData()
        sample = self.garment_builder.build(sample)
        sample = self.body_builder.build(sample, sequence)
        return sample


class Dataset:
    def __init__(self, loader: Loader, pose_sequence_path: str):
        """
        Dataset class for building training and validation samples
        :param loader: Loader object
        """

        self.loader = loader
        self.pose_sequence_path = pose_sequence_path

        self._len = 1


    def __getitem__(self, item: int) -> HeteroData:
        """
        Load a sample given a global index
        """

        sample = self.loader.load_sample(self.pose_sequence_path)
        sample['sequence_name'] = self.pose_sequence_path
        sample['garment_name'] = 'stub'

        return sample

    def __len__(self) -> int:
        return self._len
