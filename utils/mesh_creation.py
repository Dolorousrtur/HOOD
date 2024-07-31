import itertools
import os
import pickle

import networkx as nx
import numpy as np
import smplx
import trimesh
from sklearn import neighbors
from tqdm import tqdm

from utils.io import load_obj, pickle_dump, pickle_load
from utils.coarse import make_graph_from_faces, make_coarse_edges
from utils.common import NodeType
from utils.defaults import DEFAULTS


def add_pinned_verts(file, garment_name, pinned_indices):
    """
    Modify `node_type` field in the pickle file to mark pinned vertices with NodeType.HANDLE
    :param file: path top the garments dict file
    :param garment_name: name of the garment to add pinned vertices to
    :param pinned_indices: list of pinned vertex indices
    """
    with open(file, 'rb') as f:
        pkl = pickle.load(f)
    node_type = np.zeros_like(pkl[garment_name]['rest_pos'][:, :1])
    node_type[pinned_indices] = NodeType.HANDLE
    node_type = node_type.astype(np.int64)
    pkl[garment_name]['node_type'] = node_type

    with open(file, 'wb') as f:
        pickle.dump(pkl, f)


def add_pinned_verts_single_template(file, pinned_indices):
    """
    Modify `node_type` field in the pickle file to mark pinned vertices with NodeType.HANDLE
    :param file: path top the garments dict file
    :param garment_name: name of the garment to add pinned vertices to
    :param pinned_indices: list of pinned vertex indices
    """
    with open(file, 'rb') as f:
        pkl = pickle.load(f)
    node_type = np.zeros_like(pkl['rest_pos'][:, :1])
    node_type[pinned_indices] = NodeType.HANDLE
    node_type = node_type.astype(np.int64)
    pkl['node_type'] = node_type

    with open(file, 'wb') as f:
        pickle.dump(pkl, f)


def add_buttons(file, button_edges):
    with open(file, 'rb') as f:
        pkl = pickle.load(f)

    pkl['button_edges'] = button_edges

    with open(file, 'wb') as f:
        pickle.dump(pkl, f)


def sample_skinningweights(points, smpl_tree, sigmas, smpl_model):
    """
    For each point in the garment samples a random point for a Gaussian distribution around it,
    finds the nearest SMPL vertex and returns the corresponding skinning weights

    :param points: garment vertices
    :param smpl_tree: sklearn.neighbors.KDTree with SMPL vertex positions
    :param sigmas: standard deviation of the Gaussian distributions
    :param smpl_model: SMPL model
    :return: garment_shapedirs: shape blend shapes for the garment vertices
    :return: garment_posedirs: pose blend shapes for the garment vertices
    :return: garment_lbs_weights: skinning weights for the garment vertices
    """
    noise = np.random.randn(*points.shape)
    points_sampled = noise * sigmas + points

    _, nn_list = smpl_tree.query(points_sampled)
    nn_inds = nn_list[..., 0]

    garment_shapedirs = smpl_model.shapedirs[nn_inds].numpy()

    N = smpl_model.posedirs.shape[0]
    garment_posedirs = smpl_model.posedirs.reshape(N, -1, 3)[:, nn_inds].reshape(N, -1).numpy()
    garment_lbs_weights = smpl_model.lbs_weights[nn_inds].numpy()

    return garment_shapedirs, garment_posedirs, garment_lbs_weights

class GarmentCreator:
    def __init__(self, garments_dict_path, body_models_root, model_type, gender, 
                 collect_lbs=True, n_samples_lbs=0, coarse=True, n_coarse_levels=4, verbose=False):
        self.garments_dict_path = garments_dict_path
        self.body_models_root = body_models_root
        self.model_type = model_type
        self.gender = gender


        self.collect_lbs = collect_lbs
        self.n_lbs_samples = n_samples_lbs
        self.coarse = coarse
        self.n_coarse_levels = n_coarse_levels
        self.verbose = verbose

        if body_models_root is not None:
            self.body_model = smplx.create(body_models_root, model_type, gender=gender)


    def _load_garments_dict(self):
        if os.path.exists(self.garments_dict_path):
            garments_dict = pickle_load(self.garments_dict_path)
        else:
            garments_dict = {}

        return garments_dict
    
    def _save_garments_dict(self, garments_dict):
        pickle_dump(garments_dict, self.garments_dict_path)   


    def add_coarse_edges(self, garment_dict):
        n_levels = self.n_coarse_levels

        faces = garment_dict['faces']
        G = make_graph_from_faces(faces)

        components = list(nx.connected_components(G))

        cGd_list = []
        for component in components:
            cg_dict = dict()

            cG = G.subgraph(component)
            component_ids = np.array(list(component))
            faces_mask = np.isin(faces, component_ids).all(axis=1)

            faces_component = faces[faces_mask]

            center_nodes = nx.center(cG)

            cg_dict['center'] = center_nodes
            cg_dict['coarse_edges'] = dict()

            for center in center_nodes[:3]:
                coarse_edges_dict = make_coarse_edges(faces_component, center, n_levels=n_levels)
                cg_dict['coarse_edges'][center] = coarse_edges_dict
            cGd_list.append(cg_dict)

        cGdk_list = [d['coarse_edges'].keys() for d in cGd_list]
        ctuples = list(itertools.product(*cGdk_list))


        center_list = []
        coarse_edges_dict = dict()
        for ci, ctuple in enumerate(ctuples):
            center_list.append(ci)
            coarse_edges_dict[ci] = dict()

            for l in range(n_levels):
                ce_list = []
                for i, d in enumerate(cGd_list):
                    ce_list.append(d['coarse_edges'][ctuple[i]][l])

                ce_list = np.concatenate(ce_list, axis=0)

                coarse_edges_dict[ci][l] = ce_list


        garment_dict['center'] = np.array(center_list)
        garment_dict['coarse_edges'] = coarse_edges_dict

        return garment_dict



    def make_lbs_dict(self, garment_template_verts, garment_faces):
        """
        Collect linear blend skinning weights for a garment mesh
        :param obj_file:
        :param smpl_file:
        :param n_samples:
        :return:
        """

        body_verts_rest_pose = self.body_model().vertices[0].detach().cpu().numpy()
        n_samples = self.n_lbs_samples


        body_verts_tree = neighbors.KDTree(body_verts_rest_pose)
        distances, nn_list = body_verts_tree.query(garment_template_verts)

        nn_inds = nn_list[..., 0]

        n_posedirs = self.body_model.posedirs.shape[0]

        if n_samples == 0:
            # Take weights of the closest SMPL vertex
            garment_shapedirs = self.body_model.shapedirs[nn_inds].numpy()
            garment_posedirs = self.body_model.posedirs.reshape(n_posedirs, -1, 3)[:, nn_inds].reshape(n_posedirs, -1).numpy()
            garment_lbs_weights = self.body_model.lbs_weights[nn_inds].numpy()
        else:
            garment_shapedirs = 0
            garment_posedirs = 0
            garment_lbs_weights = 0

            # Randomly sample n_samples from a normal distribution with std = distance to the closest SMPL vertex
            # around the garment node and take the average of the weights for the closest SMPL nodes
            # Following "Self-Supervised Collision Handling via Generative 3D Garment Models for Virtual Try-On" [Santesteban et al. 2021]
            for _ in tqdm(range(n_samples)):
                garment_shapedirs_sampled, garment_posedirs_sampled, garment_lbs_weights_sampled = sample_skinningweights(
                    garment_template_verts, body_verts_tree, distances ** 0.5, self.body_model)
                garment_shapedirs += garment_shapedirs_sampled
                garment_posedirs += garment_posedirs_sampled
                garment_lbs_weights += garment_lbs_weights_sampled

            garment_shapedirs = garment_shapedirs / n_samples
            garment_posedirs = garment_posedirs / n_samples
            garment_lbs_weights = garment_lbs_weights / n_samples

        out_dict = dict(v=garment_template_verts, f=garment_faces, shapedirs=garment_shapedirs,
                        posedirs=garment_posedirs, lbs_weights=garment_lbs_weights)

        return out_dict

    def make_garment_dict(self, obj_file):
        """
        Create a dictionary for a garment from an obj file
        """
        vertices_full, faces_full = load_obj(obj_file, tex_coords=False)
        outer_trimesh = trimesh.Trimesh(vertices=vertices_full,
                                        faces=faces_full, process=True)

        vertices_full = outer_trimesh.vertices
        faces_full = outer_trimesh.faces

        garment_dict = make_restpos_dict(vertices_full, faces_full)

        if self.collect_lbs:
            if self.verbose:
                print('Sampling LBS weights...')
            lbs = self.make_lbs_dict(vertices_full, faces_full)
            garment_dict['lbs'] = lbs
            if self.verbose:
                print('Done.')

        if self.coarse:
            if self.verbose:
                print('Adding coarse edges... (may take a while)')
            garment_dict = self.add_coarse_edges(garment_dict)
            if self.verbose:
                print('Done.')

        garment_dict['gender'] = self.gender
        garment_dict['model_type'] = self.model_type

        return garment_dict


    def add_garment(self, objfile, garment_name):
        """
        Add a new garment from a given obj file to the garments_dict_file

        :param objfile: path to the obj file with the new garment
        :param garments_dict_file: path to the garments dict file storing all garments
        :param garment_name: name of the new garment
        :param smpl_file: path to the smpl model file
        :param coarse: whether to add coarse edges
        :param n_coarse_levels: number of coarse levels to add
        :param training: whether to add the lbs weights (needed to initialize from arbitrary pose)
        :param n_samples_lbs: number of samples to use for the 'diffused' lbs weights [Santesteban et al. 2021]
            use 0 for tight-fitting garments, and 1000 for loose-fitting garments
        """

        garment_dict = self.make_garment_dict(objfile)

        garments_dict = self._load_garments_dict()

        garments_dict[garment_name] = garment_dict
        self._save_garments_dict(garments_dict)

        if self.verbose:
            print(f"Garment '{garment_name}' added to {self.garments_dict_path}")


def make_restpos_dict(vertices_full, faces_full):
    """
    Create a dictionary for a garment from an obj file
    """

    restpos_dict = dict()
    restpos_dict['rest_pos'] = vertices_full
    restpos_dict['faces'] = faces_full.astype(np.int64)
    restpos_dict['node_type'] = np.zeros_like(vertices_full[:, :1]).astype(np.int64)

    return restpos_dict


def obj2template(obj_path, verbose=False):

    gc = GarmentCreator(None, None, None, None, collect_lbs=False, coarse=True, verbose=verbose)    
    out_dict = gc.make_garment_dict(obj_path)

    return out_dict


