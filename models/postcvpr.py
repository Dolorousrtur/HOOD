import importlib
from dataclasses import dataclass
from typing import Optional

import torch
from omegaconf import II
from torch import nn
from torch_geometric.data import Batch

from utils import normalization, common
from utils.cloth_and_material import VertexNormalsPYG
from utils.common import NodeType, gather, add_field_to_pyg_batch, \
    make_pervertex_tensor_from_lens
from utils.connectivity import compute_connectivity_pt


@dataclass
class Config:
    core_model: str = 'postcvpr'        # core_model to use
    output_size: int = 3                # output dimensions of the model (3 for 3D accelerations)
    latent_size: int = 128              # number of latent dimensions
    num_layers: int = 2                 # number of hidden layers in MLPs in the model
    n_nodefeatures: int = 24            # number of dimensions in node feature vectors (don't chenge)
    n_edgefeatures_mesh: int = 12       # number of dimensions in edge feature vectors (don't chenge)
    n_edgefeatures_coarse: int = 12     # number of dimensions in edge feature vectors for long-range edges (don't chenge)
    n_edgefeatures_world: int = 9       # number of dimensions in edge feature vectors for world (body) edges (don't chenge)
    message_passing_steps: int = 15     # total number of message passing steps
    n_coarse_levels: int = 3            # number of coarse levels in the input graph
    collision_radius: float = 3e-2      # maximal distance between garment and body nodes to add a body edge between them
    k_world_edges: Optional[int] = 1    # maximal number of body edges a single node edge may have (if None, unlimited)

    # a string that describes which edge levels of the graph should be used for each message passing step
    # MP-steps are separated by '|'
    # f - fine, cX - coarse level X
    # levels where we move from finer levels to coarse levels are preceded by 'd:'
    # levels where we move from coarse levels to finer levels are preceded by 'u:'
    architecture: str = "f,c0|f,c0|f,c0|d:c0,c1|c0,c1|c0,c1|d:c1|c1|c1|u:c0,c1|c0,c1|c0,c1|u:f,c0|f,c0|f,c0"

    device: str = II('device')


def create(mcfg: Config):
    core_model = importlib.import_module(f'models.core.{mcfg.core_model}')
    learned_model = core_model.EncodeProcessDecode(mcfg)
    model = Model(learned_model, mcfg).to(mcfg.device)

    return model


class Model(nn.Module):
    def __init__(self, learned_model, mcfg: Config):
        super().__init__()

        self._learned_model = learned_model
        self._output_normalizer = normalization.Normalizer(
            size=3)
        self._node_normalizer = normalization.Normalizer(
            size=mcfg.n_nodefeatures - 3)
        self._mesh_edge_normalizer = normalization.Normalizer(
            size=mcfg.n_edgefeatures_mesh - 3)
        self._world_edge_normalizer = normalization.Normalizer(
            size=mcfg.n_edgefeatures_world)

        self.collision_radius = mcfg.collision_radius
        self.k_world_edges = mcfg.k_world_edges

        self.nodetype_embedding = nn.Embedding(common.NodeType.SIZE, common.NodeType.SIZE, max_norm=1.)
        self.n_coarse_levels = mcfg.n_coarse_levels
        self.vertexlevel_embedding = nn.Embedding(self.n_coarse_levels + 1, 4, max_norm=1.)

        self.normals_f = VertexNormalsPYG()
        self.i = 0

    def embed(self, labels, embedding_layer):
        """
        Helper function to use nn.Embedding layer
        Required to avoid errors when using max_norm with nn.Embedding and invoking its forward pass several times before .backward()
        """
        emb_matrix = embedding_layer.weight.clone().t()
        N = emb_matrix.shape[1]

        if len(labels.shape) == 2 and labels.shape[1] == 1:
            labels = labels[:, 0]
        labels_onehot = torch.nn.functional.one_hot(labels, N).t().float()
        embedding = emb_matrix @ labels_onehot
        embedding = embedding.t()
        return embedding

    def add_positional_edges(self, sample):
        """
        Constructs body edges between garment nodes and body nodes if distance between them is less than self.collision_radius
        """
        B = sample.num_graphs

        examples_updated = []

        # Instead of operating over whole Batch object, it s easier to handle each sample in the batch separately
        # and then merge them back together
        # note, that we only use batch size of 1
        for i in range(B):
            example = sample.get_example(i)

            vertices_cloth = example['cloth'].pos
            vertices_obstacle = example['obstacle'].pos
            obstacle_vertex_type = example['obstacle'].vertex_type

            # find `self.k_world_edges` closest body nodes for each garment node that are closer than `self.collision_radius`
            indices_from, indices_to = compute_connectivity_pt(vertices_cloth, vertices_obstacle,
                                                               self.collision_radius, k=self.k_world_edges)

            # remove edges to obstacle nodes that are marked as omitted
            #  (we omit hand nodes to avoid body self-penetrations)
            indices_to_vertex_type = obstacle_vertex_type[indices_to][..., 0]
            vertex_type_mask = indices_to_vertex_type != NodeType.OBSTACLE_OMIT

            indices_from = indices_from[vertex_type_mask]
            indices_to = indices_to[vertex_type_mask]

            edges_direct = torch.stack([indices_from, indices_to], dim=0)
            edges_inverse = torch.stack([indices_to, indices_from], dim=0)

            example['cloth', 'world_edge', 'obstacle'].edge_index = edges_direct
            example['obstacle', 'world_edge', 'cloth'].edge_index = edges_inverse

            obstacle_active_mask = torch.zeros_like(vertices_obstacle[:, :1])
            active_obstacle_nodes = torch.unique(indices_to)
            obstacle_active_mask[active_obstacle_nodes] = 1
            obstacle_active_mask = obstacle_active_mask > 0
            example['obstacle'].active_mask = obstacle_active_mask

            examples_updated.append(example)

        sample_updated = Batch.from_data_list(examples_updated)

        return sample_updated

    def get_relative_pos(self, pos, edges):
        """Compute relative positions of nodes in each edge"""
        edges_pos = gather(pos, edges, 0, 1, 1).permute(0, 2, 1)
        pos_senders, pos_receivers = edges_pos.unbind(-1)
        relative_pos = pos_senders - pos_receivers
        return relative_pos

    def create_mesh_edge_set(self, sample, is_training, edge_label, normalizer):
        """
        Constructs feature vector for each edge of type `edge_label` (either mesh_edge or coarse_edgeX) in the graph (See Supplementary Material for details)
        """
        pos = sample['cloth'].pos  # (v, 3)
        rest_pos = sample['cloth'].rest_pos  # (v, 3)
        edges = sample['cloth', edge_label, 'cloth'].edge_index.T  # (e, 2)

        relative_pos = self.get_relative_pos(pos, edges)  # (e, 3)
        relative_pos_norm = torch.norm(relative_pos, dim=-1, keepdim=True)  # (e, 1)

        relative_rest_pos = self.get_relative_pos(rest_pos, edges)  # (e, 3)
        relative_rest_pos_norm = torch.norm(relative_rest_pos, dim=-1, keepdim=True)  # (e, 1)

        edge_slice = sample._slice_dict['cloth', edge_label, 'cloth']['edge_index']
        lens = edge_slice[1:] - edge_slice[:-1]

        timestep = make_pervertex_tensor_from_lens(lens, sample['cloth'].timestep)  # (e, 1)
        bending_coeff = make_pervertex_tensor_from_lens(lens, sample['cloth'].bending_coeff_input)  # (e, 1)
        lame_mu = make_pervertex_tensor_from_lens(lens, sample['cloth'].lame_mu_input)  # (e, 1)
        lame_lambda = make_pervertex_tensor_from_lens(lens, sample['cloth'].lame_lambda_input)  # (e, 1)

        # (e, 9)
        edge_features_to_norm = torch.cat([
            relative_pos,
            relative_pos_norm,
            relative_rest_pos,
            relative_rest_pos_norm,
            timestep], dim=-1
        )

        # (e, 3)
        edge_features_nonorm = torch.cat([
            bending_coeff,
            lame_mu,
            lame_lambda], dim=-1
        )

        edge_features_normalized = normalizer(edge_features_to_norm, is_training)  # (e, 9)
        edge_features_final = torch.cat([edge_features_normalized, edge_features_nonorm], dim=-1)  # (e, 12)

        sample = add_field_to_pyg_batch(sample, 'features', edge_features_final, ('cloth', edge_label, 'cloth'),
                                        'edge_index', zero_inc=True)

        return sample

    def create_world_edge_set(self, sample, is_training):
        """
        Constructs feature vector for each body edge in the graph (See Supplementary Material for details)
        """

        cloth_pos = sample['cloth'].pos  # (v, 3)
        obstacle_next_pos = sample['obstacle'].target_pos  # (w, 3)
        obstacle_pos = sample['obstacle'].pos  # (w, 3)

        edges_direct = sample['cloth', 'world_edge', 'obstacle'].edge_index  # (2, e)

        senders, receivers = edges_direct.unbind(0)  # (e,), (e,)
        senders_pos = cloth_pos[senders]  # (e, 3)
        receivers_next_pos = obstacle_next_pos[receivers]  # (e, 3)
        receivers_pos = obstacle_pos[receivers]  # (e, 3)

        relative_next_pos = senders_pos - receivers_next_pos  # (e, 3)
        relative_pos = senders_pos - receivers_pos  # (e, 3)

        relative_next_pos_norm = torch.norm(relative_next_pos, dim=-1, keepdim=True)  # (e, 1)
        relative_pos_norm = torch.norm(relative_pos, dim=-1, keepdim=True)  # (e, 1)

        edge_slice = sample._slice_dict['cloth', 'world_edge', 'obstacle']['edge_index']
        lens = edge_slice[1:] - edge_slice[:-1]
        timestep = make_pervertex_tensor_from_lens(lens, sample['cloth'].timestep)  # (e, 1)

        # (e, 9)
        features_direct = torch.cat([
            relative_pos,
            relative_pos_norm,
            relative_next_pos,
            relative_next_pos_norm,
            timestep
        ], dim=-1)

        # (e, 9)
        features_inverse = torch.cat([
            -relative_pos,
            relative_pos_norm,
            -relative_next_pos,
            relative_next_pos_norm,
            timestep
        ], dim=-1)

        normalizer = self._world_edge_normalizer

        # (2e, 9)
        features_combined = torch.cat([features_direct, features_inverse])
        N_direct = features_direct.shape[0]
        features_combined_normalized = normalizer(features_combined, is_training)

        features_direct_normalized = features_combined_normalized[:N_direct]  # (e, 9)
        features_inverse_normalized = features_combined_normalized[N_direct:]  # (e, 9)

        sample = add_field_to_pyg_batch(sample, 'features', features_direct_normalized,
                                        ('cloth', 'world_edge', 'obstacle'),
                                        'edge_index', zero_inc=True)
        sample = add_field_to_pyg_batch(sample, 'features', features_inverse_normalized,
                                        ('obstacle', 'world_edge', 'cloth'),
                                        'edge_index', zero_inc=True)

        return sample

    def add_vertex_type_embedding(self, sample):
        for k in ['cloth', 'obstacle']:
            vertex_type = sample[k].vertex_type

            vertex_type_emb = self.embed(vertex_type[..., 0], self.nodetype_embedding)
            sample = add_field_to_pyg_batch(sample, 'vertex_type_embedding', vertex_type_emb, k, 'pos')
        return sample

    def add_vertex_level_embedding(self, sample):
        for k in ['cloth', 'obstacle']:
            vertex_level = sample[k].vertex_level

            vertex_level_emb = self.embed(vertex_level[..., 0], self.vertexlevel_embedding)
            sample = add_field_to_pyg_batch(sample, 'vertex_level_embedding', vertex_level_emb, k, 'pos')
        return sample

    def add_node_features(self, sample):
        """
        Adds feature vectors for all nodes in the graph (see Supplementary Material for details)
        """
        for k in ['cloth', 'obstacle']:
            velocity = sample[k].velocity
            vertex_type_emb = sample[k].vertex_type_embedding  # (v, 9)
            vertex_level_emb = sample[k].vertex_level_embedding  # (v, 4)
            normals = sample[k].normals  # (v, 3)

            # turn time step from (1,1) tensor to (v, 1)
            slice = sample._slice_dict[k]['pos']
            lens = slice[1:] - slice[:-1]
            timestep = make_pervertex_tensor_from_lens(lens, sample['cloth'].timestep)

            # v_mass: (v, 1)
            if 'v_mass' in sample[k]:
                v_mass = torch.log(sample[k].v_mass)
            else:
                # vass is -1 for body nodes
                v_mass = torch.ones_like(velocity[:, :1]) * -1

            if 'bending_coeff' in sample[k]:
                bending_coeff = make_pervertex_tensor_from_lens(lens, sample[k].bending_coeff_input)
                lame_mu = make_pervertex_tensor_from_lens(lens, sample[k].lame_mu_input)
                lame_lambda = make_pervertex_tensor_from_lens(lens, sample[k].lame_lambda_input)
            else:
                # all material parameters are set to -1 for body nodes
                device = velocity.device
                bending_coeff = torch.ones_like(velocity[:, :1]).to(device) * -1
                lame_mu = torch.ones_like(velocity[:, :1]).to(device) * -1
                lame_lambda = torch.ones_like(velocity[:, :1]).to(device) * -1

            node_features = torch.cat(
                [velocity, vertex_type_emb, vertex_level_emb, normals, timestep, v_mass, bending_coeff, lame_mu,
                 lame_lambda], dim=-1)

            sample = add_field_to_pyg_batch(sample, 'node_features', node_features, k, 'pos')
        return sample

    def normalize_node_features(self, sample, is_training):
        cloth_node_features = sample['cloth'].node_features
        obstacle_node_features = sample['obstacle'].node_features
        obstacle_active_mask = sample['obstacle'].active_mask[:, 0]

        active_obstacle_node_features = obstacle_node_features[obstacle_active_mask]

        N_cloth = cloth_node_features.shape[0]

        all_features = torch.cat([cloth_node_features, active_obstacle_node_features])

        # material parameters are already normalized, so we don't normalize them
        all_features_to_norm = all_features[:, :-3]
        all_features_nonorm = all_features[:, -3:]

        all_features_normalized = self._node_normalizer(all_features_to_norm, is_training)
        all_features_normalized_final = torch.cat([all_features_normalized, all_features_nonorm], dim=-1)

        cloth_node_features_normalized = all_features_normalized_final[:N_cloth]
        obstacle_node_features_normalized = all_features_normalized_final[N_cloth:]
        obstacle_node_features[obstacle_active_mask] = obstacle_node_features_normalized
        sample['cloth'].node_features = cloth_node_features_normalized
        sample['obstacle'].node_features = obstacle_node_features

        return sample

    def make_nodefeatures(self, sample):
        """
        Construct feature vectors for each node in the graph
        """
        sample = self.add_vertex_type_embedding(sample)
        sample = self.add_vertex_level_embedding(sample)

        # Compute normal vectors for each vertex
        sample = self.normals_f(sample, 'cloth', 'pos')
        sample = self.normals_f(sample, 'obstacle', 'pos')
        sample = self.add_node_features(sample)

        return sample

    def replace_pinned_verts(self, sample):
        """
        Replaces pinned vertices with their positions in the following step
        (computed with linear blend-skinning).
        """
        cloth_sample = sample['cloth']
        prev_pos = cloth_sample.prev_pos
        pos = cloth_sample.pos
        target_pos = cloth_sample.target_pos

        nodetype = cloth_sample.vertex_type
        pinned_mask = nodetype == NodeType.HANDLE

        prev_pos = prev_pos * torch.logical_not(pinned_mask) + pos * pinned_mask
        pos = pos * torch.logical_not(pinned_mask) + target_pos * pinned_mask

        cloth_sample.prev_pos = prev_pos
        cloth_sample.pos = pos

        return sample

    def prepare_inputs(self, sample, is_training):
        """Builds input graph with input feature vector for each node and edge"""

        # replace pinned vertices with their target positions
        sample = self.replace_pinned_verts(sample)

        # construct body edges
        sample = self.add_positional_edges(sample)

        # make feature vectors for all nodes in the graph
        sample = self.make_nodefeatures(sample)

        # make feature vectors for mesh edges and normalize them
        sample = self.create_mesh_edge_set(sample, is_training, 'mesh_edge', self._mesh_edge_normalizer)

        # make feature vectors for each set of coarse edges and normalize them
        for i in range(self.n_coarse_levels):
            key = f'coarse_edge{i}'
            sample = self.create_mesh_edge_set(sample, is_training, key, self._mesh_edge_normalizer)

        # make feature vectors for body edges and normalize them
        sample = self.create_world_edge_set(sample, is_training)

        # normalize nodes' feature vectors
        sample = self.normalize_node_features(sample, is_training)

        return sample

    def get_position(self, sample, is_training):
        """
        Unnormalize model's outputs to get accelerations for each garment node.
        Then, integrate the garment geometry forward in time to get the next position.
        """

        # get mask for pinned vertices, for them we don;t use the predicted acceleration,
        # but set their positions to ones generated by linear blend-skinning
        vertex_type = sample['cloth'].vertex_type
        pinned_mask = vertex_type == NodeType.HANDLE

        # get predicted accelerations
        cloth_features = sample['cloth'].node_features
        acceleration = self._output_normalizer.inverse(cloth_features)

        # add predicted accelerations to the sample
        sample = add_field_to_pyg_batch(sample, 'pred_acceleration', acceleration, 'cloth', 'pos')

        # integrate forward
        cur_position = sample['cloth'].pos
        prev_position = sample['cloth'].prev_pos
        target_position = sample['cloth'].target_pos
        velocity = sample['cloth'].velocity

        pred_velocity = velocity + acceleration
        target_velocity = target_position - cur_position
        pred_velocity = pred_velocity * torch.logical_not(pinned_mask) + target_velocity * pinned_mask

        position = cur_position + velocity + acceleration
        position = position * torch.logical_not(pinned_mask) + target_position * pinned_mask

        # add predicted velocities and positions to the sample
        sample = add_field_to_pyg_batch(sample, 'pred_pos', position, 'cloth', 'pos')
        sample = add_field_to_pyg_batch(sample, 'pred_velocity', pred_velocity, 'cloth', 'pos')

        # Pass lbs-based accelerations through the normalizer to gather statistics
        target_acceleration = target_position - 2 * cur_position + prev_position
        target_acceleration_norm = self._output_normalizer(target_acceleration, is_training)
        sample = add_field_to_pyg_batch(sample, 'target_acceleration', target_acceleration_norm, 'cloth', 'pos')
        return sample

    def forward(self, sample, is_training=True):
        """
        Forward pass. Predicts axxelerations for each garment node and computes their posiions in the next frame.

        :param sample: torch_geometric Batch with following fields (fields not used by the model are omitted):

                (note that v, f, e are different for each object (cloth, obstacle) and each edge sets)
                sample['cloth']:
                    pos: (v, 3) tensor with current positions of cloth vertices
                    prev_pos: (v, 3) tensor with previous positions of cloth vertices
                    target_pos: (v, 3) tensor with target positions of cloth vertices (used to get positions of pinned vertices)
                    rest_pos: (v, 3) tensor with positions of cloth vertices in canonical pose
                    velocity: (v, 3) tensor with current velocities of cloth vertices

                    v_mass: (v, 1) vertex masses
                    vertex_type: (v, 1) tensor with vertex types (0 - regular, 3 - pinned)
                    vertex_level: (v, 1) tensor with the deepest level in graph hierarchy the node is present in levels (0 - finest, 3 - coarsest)

                    faces_batch: (3, f) tensor with vertex indices for each face

                    timestep: (1,1) time between frames
                    lame_mu_input, lame_lambda_input, bending_coeff_input: (1,1) normalized material parameters

                sample['obstacle']:
                    pos: (v, 3) tensor with current positions of body vertices
                    prev_pos: (v, 3) tensor with previous positions of body vertices
                    target_pos: (v, 3) tensor with next positions of body vertices
                    velocity: (v, 3) tensor with current velocities of body vertices
                    next_velocity: (v, 3) tensor with next velocities of body vertices

                    vertex_type: (v, 1) tensor with vertex types (1 - regular body, 2 - omitted)
                    vertex_level: (v, 1) tensor with the level in graph hierarchy (always 0 for body)

                    faces_batch: (3, f) tensor with vertex indices for each face

                sample['cloth', 'mesh_edge', 'cloth'].edge_index: (2, e) tensor with node index pairs for mesh edges
                sample['cloth', 'coarse_edgeX', 'cloth'].edge_index: (2, e) tensor with node index pairs for X's level of coarse edges

        :param is_training: whether to update statistics used for normalization

        :return:
            sample: updated torch_geometric Batch with following fields added:
                sample['cloth']:
                    pred_acceleration: (v, 3) tensor with predicted accelerations of cloth vertices
                    pred_velocity: (v, 3) tensor with predicted velocities of cloth vertices (sample['cloth'].velocity + sample['cloth'].pred_acceleration)
                    pred_pos: (v, 3) tensor with predicted positions of cloth vertices (sample['cloth'].pos + sample['cloth'].pred_velocity)

        """
        sample = self.prepare_inputs(sample, is_training=is_training)
        sample = self._learned_model(sample)
        sample = self.get_position(sample, is_training=is_training)
        return sample
