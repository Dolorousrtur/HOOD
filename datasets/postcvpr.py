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
from utils.datasets import load_garments_dict, make_garment_smpl_dict
from utils.defaults import DEFAULTS
from utils.garment_smpl import GarmentSMPL


@dataclass
class Config:
    data_root: str = MISSING  # Path to the data root relative to $HOOD_DATA/
    smpl_model: str = MISSING  # Path to the SMPL model relative to $HOOD_DATA/aux_data/
    garment_dict_file: str = MISSING  # Path to the garment dict file with data for all garments relative to $HOOD_DATA/aux_data/
    split_path: Optional[str] = None  # Path to the .csv split file relative to $HOOD_DATA/aux_data/
    obstacle_dict_file: Optional[
        str] = None  # Path to the file with auxiliary data for obstacles relative to $HOOD_DATA/aux_data/
    noise_scale: float = 3e-3  # Noise scale for the garment vertices (not used in validation)
    lookup_steps: int = 5  # Number of steps to look up in the future (not used in validation)
    pinned_verts: bool = False  # Whether to use pinned vertices
    wholeseq: bool = False  # whether to load the whole sequence (always True in validation)
    random_betas: bool = False  # Whether to use random beta parameters for the SMPL model
    use_betas_for_restpos: bool = False  # Whether to use beta parameters to get canonical garment geometry
    betas_scale: float = 0.1  # Scale for the beta parameters (not used if random_betas is False)
    restpos_scale_min: float = 1.  # Minimum scale for randomly sampling the canonical garment geometry
    restpos_scale_max: float = 1.  # Maximum scale for randomly sampling the canonical garment geometry
    n_coarse_levels: int = 1  # Number of coarse levels with long-range edges
    separate_arms: bool = False  # Whether to separate the arms from the rest of the body (to avoid body self-intersections)
    zero_betas: bool = False  # Whether to set the beta parameters to zero
    button_edges: bool = False  # Whether to load the button edges
    single_sequence_file: Optional[str] = None  # Path to the single sequence to load (used in Inference.ipynb)
    single_sequence_garment: Optional[
        str] = None  # Garment name for the single sequence to load (used in Inference.ipynb)

    betas_file: Optional[
        str] = None  # Path to the file with the table of beta parameters (used in validation to generate sequences with specific body shapes)


def make_obstacle_dict(mcfg: Config) -> dict:
    if mcfg.obstacle_dict_file is None:
        return {}

    obstacle_dict_path = os.path.join(DEFAULTS.aux_data, mcfg.obstacle_dict_file)
    with open(obstacle_dict_path, 'rb') as f:
        obstacle_dict = pickle.load(f)
    return obstacle_dict


def create_loader(mcfg: Config):
    garment_dict_path = os.path.join(DEFAULTS.aux_data, mcfg.garment_dict_file)
    garments_dict = load_garments_dict(garment_dict_path)

    smpl_model_path = os.path.join(DEFAULTS.aux_data, mcfg.smpl_model)
    smpl_model = smplx.SMPL(smpl_model_path)

    garment_smpl_model_dict = make_garment_smpl_dict(garments_dict, smpl_model)
    obstacle_dict = make_obstacle_dict(mcfg)

    if mcfg.single_sequence_file is None:
        mcfg.data_root = os.path.join(DEFAULTS.data_root, mcfg.data_root)

    if mcfg.betas_file is not None:
        betas_table = pickle_load(os.path.join(DEFAULTS.aux_data, mcfg.betas_file))['betas']
    else:
        betas_table = None

    loader = Loader(mcfg, garments_dict,
                    smpl_model, garment_smpl_model_dict, obstacle_dict=obstacle_dict, betas_table=betas_table)
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

        betas = sequence_dict['betas']
        if len(betas.shape) == 2 and betas.shape[0] != 1:
            betas = betas[idx_start: idx_end]

        verts = f_make(sequence_dict['body_pose'][idx_start: idx_end],
                       sequence_dict['global_orient'][idx_start: idx_end],
                       sequence_dict['transl'][idx_start: idx_end],
                       betas, garment_name=garment_name)

        return verts

    def pad_lookup(self, lookup: np.ndarray) -> np.ndarray:
        """
        Pad the lookup sequence to the required number of steps.
        """
        n_lookup = lookup.shape[0]
        n_topad = self.mcfg.lookup_steps - n_lookup

        if n_topad == 0:
            return lookup

        padlist = [lookup] + [lookup[-1:]] * n_topad
        lookup = np.concatenate(padlist, axis=0)
        return lookup

    def pos2tensor(self, pos: np.ndarray) -> torch.Tensor:
        """
        Convert a numpy array of vertices to a tensor and permute the axes into [VxNx3] (torch geometric format)
        """
        pos = torch.tensor(pos).permute(1, 0, 2)
        if not self.mcfg.wholeseq and pos.shape[1] == 1:
            pos = pos[:, 0]
        return pos

    def add_verts(self, sample: HeteroData, sequence_dict: dict, idx: int, f_make, object_key: str,
                  **kwargs) -> HeteroData:
        """
        Builds the vertices from the given SMPL pose sequence and adds them to the HeteroData sample.
        :param sample: HetereoData object
        :param sequence_dict: sequence of SMPL parameters
        :param idx: frame index (not used if self.mcfg.wholeseq is True)
        :param f_make: function that takes SMPL parameters and returns vertices
        :param object_key: name of the object to build vertices for ('cloth' or 'obstacle')
        :return: updated HeteroData object
        """

        N_steps = sequence_dict['body_pose'].shape[0]
        pos_dict = {}

        # Build the vertices for the whole sequence
        if self.mcfg.wholeseq:
            all_vertices = VertexBuilder.build(sequence_dict, f_make, 0, None,
                                               **kwargs)
            pos_dict['prev_pos'] = all_vertices[:-2]
            pos_dict['pos'] = all_vertices[1:-1]
            pos_dict['target_pos'] = all_vertices[2:]

        # Build the vertices for several frames starting from `idx`
        else:
            n_lookup = 1
            if self.mcfg.lookup_steps > 0:
                n_lookup = min(self.mcfg.lookup_steps, N_steps - idx - 2)
            all_vertices = VertexBuilder.build(sequence_dict, f_make, idx, idx + 2 + n_lookup,
                                               **kwargs)
            pos_dict["prev_pos"] = all_vertices[:1]
            pos_dict["pos"] = all_vertices[1:2]
            pos_dict["target_pos"] = all_vertices[2:3]

            lookup = all_vertices[2:]
            lookup = self.pad_lookup(lookup)

            pos_dict["lookup"] = lookup

        for k, v in pos_dict.items():
            v = self.pos2tensor(v)
            setattr(sample[object_key], k, v)

        return sample


class NoiseMaker:
    """
    Helper class to add noise to the garment vertices.
    """

    def __init__(self, mcfg: Config):
        self.mcfg = mcfg

    def add_noise(self, sample: HeteroData) -> HeteroData:
        """
        Add gaussian noise with std == self.mcfg.noise_scale to `pos` and `prev_pos`
        tensors in `sample['cloth']`
        :param sample: HeteroData
        :return: sample: HeteroData
        """
        if self.mcfg.noise_scale == 0:
            return sample

        world_pos = sample['cloth'].pos
        vertex_type = sample['cloth'].vertex_type
        if len(vertex_type.shape) == 1:
            vertex_type = vertex_type[..., None]

        noise = np.random.normal(scale=self.mcfg.noise_scale, size=world_pos.shape).astype(np.float32)
        noise_prev = np.random.normal(scale=self.mcfg.noise_scale, size=world_pos.shape).astype(np.float32)

        noise = torch.tensor(noise)
        noise_prev = torch.tensor(noise_prev)

        mask = vertex_type == NodeType.NORMAL
        if len(mask.shape) == 2 and len(noise.shape) == 3:
            mask = mask.unsqueeze(-1)
        noise = noise * mask

        sample['cloth'].pos = sample['cloth'].pos + noise
        sample['cloth'].prev_pos = sample['cloth'].prev_pos + noise_prev

        return sample


class GarmentBuilder:
    """
    Class to build the garment meshes from SMPL parameters.
    """

    def __init__(self, mcfg: Config, garments_dict: dict, garment_smpl_model_dict: Dict[str, GarmentSMPL]):
        """
        :param mcfg: config
        :param garments_dict: dictionary with data for all garments
        :param garment_smpl_model_dict: dictionary with SMPL models for all garments
        """
        self.mcfg = mcfg
        self.garments_dict = garments_dict
        self.garment_smpl_model_dict = garment_smpl_model_dict

        self.vertex_builder = VertexBuilder(mcfg)
        self.noise_maker = NoiseMaker(mcfg)

    def make_cloth_verts(self, body_pose: np.ndarray, global_orient: np.ndarray, transl: np.ndarray, betas: np.ndarray,
                         garment_name: str) -> np.ndarray:
        """
        Make vertices of a garment `garment_name` in a given pose

        :param body_pose: SMPL pose parameters [Nx69] OR [69]
        :param global_orient: SMPL global_orient [Nx3] OR [3]
        :param transl: SMPL translation [Nx3] OR [3]
        :param betas: SMPL betas [Nx10] OR [10]
        :param garment_name: name of the garment in `self.garment_smpl_model_dict`

        :return: vertices [NxVx3]
        """
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

    def add_vertex_type(self, sample: HeteroData, garment_name: str) -> HeteroData:
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
        garment_dict = self.garments_dict[garment_name]

        if self.mcfg.pinned_verts:
            vertex_type = garment_dict['node_type'].astype(np.int64)
        else:
            V = sample['cloth'].pos.shape[0]
            vertex_type = np.zeros((V, 1)).astype(np.int64)

        sample['cloth'].vertex_type = torch.tensor(vertex_type)
        return sample

    def resize_restpos(self, restpos: np.array) -> np.array:
        """
        Randomly resize resting geometry of a garment
        with scale from `self.mcfg.restpos_scale_min` to `self.mcfg.restpos_scale_max`

        :param restpos: Vx3
        :return: resized restpos: Vx3
        """
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

    def make_shaped_restpos(self, sequence_dict: dict, garment_name: str) -> np.ndarray:
        """
        Create resting pose geometry for a garment in SMPL zero pose with given SMPL betas

        :param sequence_dict: dict with
            sequence_dict['body_pose'] np.array SMPL body pose [Nx69]
            sequence_dict['global_orient'] np.array SMPL global_orient [Nx3]
            sequence_dict['transl'] np.array SMPL translation [Nx3]
            sequence_dict['betas'] np.array SMPL betas [10]
        :param garment_name: name of the garment in `self.garment_smpl_model_dict`
        :return: zeroposed garment with given shape [Vx3]
        """
        body_pose = np.zeros_like(sequence_dict['body_pose'][:1])
        global_orient = np.zeros_like(sequence_dict['global_orient'][:1])
        transl = np.zeros_like(sequence_dict['transl'][:1])
        verts = self.make_cloth_verts(body_pose,
                                      global_orient,
                                      transl,
                                      sequence_dict['betas'], garment_name=garment_name)
        return verts

    def add_restpos(self, sample: HeteroData, sequence_dict: dict, garment_name: str) -> HeteroData:
        """
        Add resting pose geometry to `sample['cloth']`

        :param sample: HeteroData
        :param sequence_dict: dict with SMPL parameters
        :param garment_name: name of the garment in `self.garment_smpl_model_dict` and `self.garments_dict`
        :return: sample['cloth'].rest_pos: torch.FloatTensor [Vx3]
        """
        garment_dict = self.garments_dict[garment_name]
        if self.mcfg.use_betas_for_restpos:
            rest_pos = self.make_shaped_restpos(sequence_dict, garment_name)[0]
        else:
            rest_pos = self.resize_restpos(garment_dict['rest_pos'])

        sample['cloth'].rest_pos = torch.tensor(rest_pos)
        return sample

    def add_faces_and_edges(self, sample: HeteroData, garment_name: str) -> HeteroData:
        """
        Add garment faces to `sample['cloth']`
        Add bi-directional edges to `sample['cloth', 'mesh_edge', 'cloth']`

        :param sample: HeteroData
        :param garment_name: name of the garment in `self.garment_smpl_model_dict` and `self.garments_dict`
        :return:
            sample['cloth'].faces_batch: torch.LongTensor [3xF]
            ample['cloth', 'mesh_edge', 'cloth'].edge_index: torch.LongTensor [2xE]
        """

        garment_dict = self.garments_dict[garment_name]

        faces = torch.tensor(garment_dict['faces'])
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

    def add_coarse(self, sample: HeteroData, garment_name: str) -> HeteroData:
        """
        Add coarse edges to `sample` as `sample['cloth', f'coarse_edge{i}', 'cloth'].edge_index`.
        where `i` is the number of the coarse level (starting from `0`)

        :param sample: HeteroData
        :param garment_name:
        :return: sample['cloth', f'coarse_edge{i}', 'cloth'].edge_index: torch.LongTensor [2, E_i]
        """
        if self.mcfg.n_coarse_levels == 0:
            return sample

        garment_dict = self.garments_dict[garment_name]
        faces = garment_dict['faces']

        # Randomly choose center of the mesh
        # center of a graph is a node with minimal eccentricity (distance to the farthest node)
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

    def add_button_edges(self, sample: HeteroData, garment_name: str) -> HeteroData:
        """
        Add set of node pairs that should serve as buttons (needed for unzipping/unbuttoning demonstration)
        :param sample: HeteroData
        :param garment_name: name of the garment in `self.garment_smpl_model_dict` and `self.garments_dict`
        :return: if `button_edges` are on,
            sample['cloth'].button_edges_batch: torch.LongTensor [2xBE]
        """

        # if button_edges flag is off, do nothing
        if not hasattr(self.mcfg, 'button_edges') or not self.mcfg.button_edges:
            return sample

        garment_dict = self.garments_dict[garment_name]

        # if there are no buttons for the given garment, do nothing
        if 'button_edges' not in garment_dict:
            return sample

        button_edges = garment_dict['button_edges']

        button_edges = torch.LongTensor(button_edges)
        sample['cloth'].button_edges_batch = button_edges.T

        return sample

    def build(self, sample: HeteroData, sequence_dict: dict, idx: int, garment_name: str) -> HeteroData:
        """
        Add all data for the garment to the sample

        :param sample: HeteroData
        :param sequence_dict: dictionary with SMPL parameters
        :param idx: starting index in a sequence (not used if  `self.mcfg.wholeseq`)
        :param garment_name: name of the garment in `self.garment_smpl_model_dict` and `self.garments_dict`
        :return:
            if self.mcfg.wholeseq:
                sample['cloth'].prev_pos torch.FloatTensor [VxNx3]: vertex positions at the previous frame
                sample['cloth'].pos torch.FloatTensor [VxNx3]: vertex positions at the current frame
                sample['cloth'].target_pos torch.FloatTensor [VxNx3]: vertex positions at the next frame
            else:
                sample['cloth'].prev_pos torch.FloatTensor [Vx3]: vertex positions at the previous frame
                sample['cloth'].pos torch.FloatTensor [Vx3]: vertex positions at the current frame
                sample['cloth'].target_pos torch.FloatTensor [Vx3]: vertex positions at the next frame
                sample['cloth'].lookup torch.FloatTensor [VxLx3] (L == self.mcfg.lookup_steps): vertex positions at several future frames

            sample['cloth'].rest_pos torch.FloatTensor [Vx3]: vertex positions in the canonical pose
            sample['cloth'].faces_batch torch.LongTensor [3xF]: garment faces
            sample['cloth'].vertex_type torch.LongTensor [Vx1]: vertex type (0 - regular, 3 - pinned)
            sample['obstacle'].vertex_level torch.LongTensor [Vx1]: level of the vertex in the hierarchy (always 0 for the body)

            sample['cloth', 'mesh_edge', 'cloth'].edge_index: torch.LongTensor [2xE]: mesh edges

            for each coarse level `i` in [0, self.mcfg.n_coarse_levels]:
                sample['cloth', f'coarse_edge{i}', 'cloth'].edge_index: torch.LongTensor [2, E_i]: coarse edges at level `i`

            if self.mcfg.button edges and the garment has buttons:
                sample['cloth'].button_edges_batch: torch.LongTensor [2xBE]: button edges

        """
        sample = self.vertex_builder.add_verts(sample, sequence_dict, idx, self.make_cloth_verts, "cloth",
                                               garment_name=garment_name)

        sample = self.add_vertex_type(sample, garment_name)
        sample = self.noise_maker.add_noise(sample)
        sample = self.add_restpos(sample, sequence_dict, garment_name)
        sample = self.add_faces_and_edges(sample, garment_name)
        sample = self.add_coarse(sample, garment_name)
        sample = self.add_button_edges(sample, garment_name)

        return sample


class BodyBuilder:
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

    def make_smpl_vertices(self, body_pose: np.ndarray, global_orient: np.ndarray, transl: np.ndarray,
                           betas: np.ndarray, **kwargs) -> np.ndarray:
        """
        Create body vertices from SMPL parameters (used in VertexBuilder.add_verts)

        :param body_pose: SMPL pose parameters [Nx69] OR [69]
        :param global_orient: SMPL global_orient [Nx3] OR [3]
        :param transl: SMPL translation [Nx3] OR [3]
        :param betas: SMPL betas [Nx10] OR [10]

        :return: vertices [NxVx3]
        """
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
        faces = torch.tensor(self.smpl_model.faces.astype(np.int64))
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

    def build(self, sample: HeteroData, sequence_dict: dict, idx: int) -> HeteroData:
        """
        Add all data for the body (obstacle) to the sample
        :param sample: HeteroData object to add data to
        :param sequence_dict: dict with SMPL parameters
        :param idx: index of the current frame in the sequence
        
        :return:
            if self.mcfg.wholeseq:
                sample['obstacle'].prev_pos torch.FloatTensor [VxNx3]: vertex positions at the previous frame
                sample['obstacle'].pos torch.FloatTensor [VxNx3]: vertex positions at the current frame
                sample['obstacle'].target_pos torch.FloatTensor [VxNx3]: vertex positions at the next frame
            else:
                sample['obstacle'].prev_pos torch.FloatTensor [Vx3]: vertex positions at the previous frame
                sample['obstacle'].pos torch.FloatTensor [Vx3]: vertex positions at the current frame
                sample['obstacle'].target_pos torch.FloatTensor [Vx3]: vertex positions at the next frame
                sample['obstacle'].lookup torch.FloatTensor [VxLx3] (L == self.mcfg.lookup_steps): vertex positions at several future frames

            sample['obstacle'].rest_pos torch.FloatTensor [Vx3]: vertex positions in the canonical pose
            sample['obstacle'].faces_batch torch.LongTensor [3xF]: garment faces
            sample['obstacle'].vertex_type torch.LongTensor [Vx1]: vertex type (1 - regular obstacle, 2 - omitted)
            sample['obstacle'].vertex_level torch.LongTensor [Vx1]: level of the vertex in the hierarchy (always 0 for the body)

            sample['cloth', 'mesh_edge', 'cloth'].edge_index: torch.LongTensor [2xE]: mesh edges

        
        """

        sample = self.vertex_builder.add_verts(sample, sequence_dict, idx, self.make_smpl_vertices, "obstacle")
        sample = self.add_vertex_type(sample)
        sample = self.add_faces(sample)
        sample = self.add_vertex_level(sample)
        return sample


class SequenceLoader:
    def __init__(self, mcfg, data_path, betas_table=None):
        self.mcfg = mcfg
        self.data_path = data_path
        self.betas_table = betas_table

    def process_sequence(self, sequence: dict) -> dict:
        """
        Apply transformations to the SMPL sequence
        :param sequence: dict with SMPL parameters
        :return: processed dict with SMPL parameters
        """
        #
        # from SNUG, eliminates hand-body penetrations
        if self.mcfg.separate_arms:
            body_pose = sequence['body_pose']
            global_orient = sequence['global_orient']
            full_pos = np.concatenate([global_orient, body_pose], axis=1)
            full_pos = separate_arms(full_pos)
            sequence['global_orient'] = full_pos[:, :3]
            sequence['body_pose'] = full_pos[:, 3:]

        # sample random SMPLX beta parameters
        if self.mcfg.random_betas:
            betas = sequence['betas']
            random_betas = np.random.rand(*betas.shape)
            random_betas = random_betas * self.mcfg.betas_scale * 2
            random_betas -= self.mcfg.betas_scale
            sequence['betas'] = random_betas

        # zero-out hand pose (eliminates unrealistic hand poses)
        sequence['body_pose'][:, -6:] *= 0

        # zero-out all SMPL beta parameters
        if self.mcfg.zero_betas:
            sequence['betas'] *= 0

        return sequence

    def load_sequence(self, fname: str, betas_id: int=None) -> dict:
        """
        Load sequence of SMPL parameters from disc
        and process it

        :param fname: file name of the sequence
        :param betas_id: index of the beta parameters in self.betas_table
                        (used only in validation to generate sequences for metrics calculation
        :return: dict with SMPL parameters:
            sequence['body_pose'] np.array [Nx69]
            sequence['global_orient'] np.array [Nx3]
            sequence['transl'] np.array [Nx3]
            sequence['betas'] np.array [10]
        """
        filepath = os.path.join(self.data_path, fname + '.pkl')
        with open(filepath, 'rb') as f:
            sequence = pickle.load(f)

        assert betas_id is None or self.betas_table is not None, "betas_id should be specified only in validation mode with valid betas_table"

        if self.betas_table is not None:
            sequence['betas'] = self.betas_table[betas_id]

        sequence = self.process_sequence(sequence)

        return sequence


class Loader:
    """
    Class for building HeteroData objects containing all data for a single sample
    """

    def __init__(self, mcfg: Config, garments_dict: dict, smpl_model: SMPL,
                 garment_smpl_model_dict: Dict[str, GarmentSMPL], obstacle_dict: dict, betas_table=None):
        self.sequence_loader = SequenceLoader(mcfg, mcfg.data_root, betas_table=betas_table)
        self.garment_builder = GarmentBuilder(mcfg, garments_dict, garment_smpl_model_dict)
        self.body_builder = BodyBuilder(mcfg, smpl_model, obstacle_dict)

        self.data_path = mcfg.data_root

    def load_sample(self, fname: str, idx: int, garment_name: str, betas_id: int) -> HeteroData:
        """
        Build HeteroData object for a single sample
        :param fname: name of the pose sequence relative to self.data_path
        :param idx: index of the frame to load (not used if self.mcfg.wholeseq == True)
        :param garment_name: name of the garment to load
        :param betas_id: index of the beta parameters in self.betas_table (only used to generate validation sequences when comparing to snug/ssch)
        :return: HelteroData object (see BodyBuilder.build and GarmentBuilder.build for details)
        """
        sequence = self.sequence_loader.load_sequence(fname, betas_id=betas_id)
        sample = HeteroData()
        sample = self.garment_builder.build(sample, sequence, idx, garment_name)
        sample = self.body_builder.build(sample, sequence, idx)
        return sample


class Dataset:
    def __init__(self, loader: Loader, datasplit: pd.DataFrame, wholeseq=False):
        """
        Dataset class for building training and validation samples
        :param loader: Loader object
        :param datasplit: pandas DataFrame with the following columns:
            id: sequence name relative to loader.data_path
            length: number of frames in the sequence
            garment: name of the garment
        :param wholeseq: if True, load the whole sequence, otherwise load a single frame
        """

        self.loader = loader
        self.datasplit = datasplit
        self.wholeseq = wholeseq

        if self.wholeseq:
            self._len = self.datasplit.shape[0]
        else:
            all_lens = datasplit.length.tolist()
            self.all_lens = [int(x) - 7 for x in all_lens]
            self._len = sum(self.all_lens)

    def _find_idx(self, index: int) -> Tuple[str, int, str]:
        """
        Takes a global index and returns the sequence name, frame index and garment name for it
        """
        fi = 0
        while self.all_lens[fi] <= index:
            index -= self.all_lens[fi]
            fi += 1
        return self.datasplit.id[fi], index, self.datasplit.garment[fi]

    def __getitem__(self, item: int) -> HeteroData:
        """
        Load a sample given a global index
        """

        betas_id = None
        if self.wholeseq:
            fname = self.datasplit.id[item]
            garment_name = self.datasplit.garment[item]
            idx = 0

            if 'betas_id' in self.datasplit:
                betas_id = int(self.datasplit.betas_id[item])
        else:
            fname, idx, garment_name = self._find_idx(item)

        sample = self.loader.load_sample(fname, idx, garment_name, betas_id=betas_id)
        sample['sequence_name'] = fname
        sample['garment_name'] = garment_name

        return sample

    def __len__(self) -> int:
        return self._len
