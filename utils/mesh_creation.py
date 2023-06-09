import os
import pickle

import networkx as nx
import numpy as np
import smplx
from sklearn import neighbors
from tqdm import tqdm

from utils.cloth_and_material import load_obj
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


def make_lbs_dict(obj_file, smpl_file, n_samples=0):
    """
    Collect linear blend skinning weights for a garment mesh
    :param obj_file:
    :param smpl_file:
    :param n_samples:
    :return:
    """

    smpl_model = smplx.SMPL(smpl_file)
    smplx_v_rest_pose = smpl_model().vertices[0].detach().cpu().numpy()

    garment_template, garment_faces = load_obj(obj_file, tex_coords=False)

    smpl_tree = neighbors.KDTree(smplx_v_rest_pose)
    distances, nn_list = smpl_tree.query(garment_template)

    nn_inds = nn_list[..., 0]

    if n_samples == 0:
        # Take weights of the closest SMPL vertex
        garment_shapedirs = smpl_model.shapedirs[nn_inds].numpy()
        garment_posedirs = smpl_model.posedirs.reshape(207, -1, 3)[:, nn_inds].reshape(207, -1).numpy()
        garment_lbs_weights = smpl_model.lbs_weights[nn_inds].numpy()
    else:
        garment_shapedirs = 0
        garment_posedirs = 0
        garment_lbs_weights = 0

        # Randomly sample n_samples from a normal distribution with std = distance to the closest SMPL vertex
        # around the garment node and take the average of the weights for the closest SMPL nodes
        # Following "Self-Supervised Collision Handling via Generative 3D Garment Models for Virtual Try-On" [Santesteban et al. 2021]
        for i in tqdm(range(n_samples)):
            garment_shapedirs_sampled, garment_posedirs_sampled, garment_lbs_weights_sampled = sample_skinningweights(
                garment_template, smpl_tree, distances ** 0.5, smpl_model)
            garment_shapedirs += garment_shapedirs_sampled
            garment_posedirs += garment_posedirs_sampled
            garment_lbs_weights += garment_lbs_weights_sampled

        garment_shapedirs = garment_shapedirs / n_samples
        garment_posedirs = garment_posedirs / n_samples
        garment_lbs_weights = garment_lbs_weights / n_samples

    # out_dict = dict(v=garment_template, f=garment_faces, uv=uvs, f_uv=faces_uv, shapedirs=garment_shapedirs,
    #                 posedirs=garment_posedirs, lbs_weights=garment_lbs_weights)

    out_dict = dict(v=garment_template, f=garment_faces, shapedirs=garment_shapedirs,
                    posedirs=garment_posedirs, lbs_weights=garment_lbs_weights)
    return out_dict


def make_restpos_dict(objfile):
    """
    Create a dictionary for a garment from an obj file
    """
    vertices_full, faces_full = load_obj(objfile, tex_coords=False)

    restpos_dict = dict()
    restpos_dict['rest_pos'] = vertices_full
    restpos_dict['faces'] = faces_full.astype(np.int64)
    restpos_dict['node_type'] = np.zeros_like(vertices_full[:, :1]).astype(np.int64)

    return restpos_dict


def add_coarse_edges(garment_dict, n_levels=4):
    """
    Add coarse edges to a garment dictionary
    """

    faces = garment_dict['faces']
    G = make_graph_from_faces(faces)

    center_nodes = nx.center(G)
    garment_dict['center'] = center_nodes
    garment_dict['coarse_edges'] = dict()
    for center in center_nodes:
        coarse_edges_dict = make_coarse_edges(faces, center, n_levels=n_levels)
        garment_dict['coarse_edges'][center] = coarse_edges_dict

    return garment_dict


def make_garment_dict(obj_file, smpl_file, coarse=True, n_coarse_levels=4, training=True, n_samples_lbs=0):
    """
    Create a dictionary for a garment from an obj file
    """

    garment_dict = make_restpos_dict(obj_file)

    if training:
        lbs = make_lbs_dict(obj_file, smpl_file, n_samples=n_samples_lbs)
        garment_dict['lbs'] = lbs

    if coarse:
        garment_dict = add_coarse_edges(garment_dict, n_levels=n_coarse_levels)

    return garment_dict


def add_garment_to_garments_dict(objfile, garments_dict_file, garment_name, smpl_file=None, coarse=True,
                                 n_coarse_levels=4, training=True, n_samples_lbs=0):
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

    if smpl_file is None:
        smpl_file = os.path.join(DEFAULTS.aux_data, 'smpl', 'SMPL_FEMALE.pkl')

    garment_dict = make_garment_dict(objfile, smpl_file, coarse=coarse, n_coarse_levels=n_coarse_levels,
                                     training=training, n_samples_lbs=n_samples_lbs)

    if os.path.exists(garments_dict_file):
        with open(garments_dict_file, 'rb') as f:
            garments_dict = pickle.load(f)
    else:
        garments_dict = {}

    garments_dict[garment_name] = garment_dict
    with open(garments_dict_file, 'wb') as f:
        pickle.dump(garments_dict, f)
