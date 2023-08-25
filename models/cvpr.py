from dataclasses import dataclass
from typing import Optional

import torch
from omegaconf import II
from torch import nn
from torch_geometric.data import Batch

from models.core import cvpr as core_model
from utils import normalization, common
from utils.cloth_and_material import VertexNormalsPYG
from utils.common import NodeType, gather, add_field_to_pyg_batch, \
    make_pervertex_tensor_from_lens
from utils.connectivity import compute_connectivity_pt


@dataclass
class Config:
    output_size: int = 3
    latent_size: int = 128
    num_layers: int = 2
    n_nodefeatures: int = 24
    n_edgefeatures_mesh: int = 12
    n_edgefeatures_coarse: int = 12
    n_edgefeatures_world: int = 9
    message_passing_steps: int = 15
    n_coarse_levels: int = 1
    collision_radius: float = 5e-2
    k_world_edges: Optional[int] = None

    device: str = II('device')


def create(mcfg):
    learned_model = core_model.EncodeProcessDecode(mcfg)
    model = Model(learned_model, collision_radius=mcfg.collision_radius,
                  n_nodefeatures=mcfg.n_nodefeatures,
                  n_edgefeatures_mesh=mcfg.n_edgefeatures_mesh,
                  n_edgefeatures_world=mcfg.n_edgefeatures_world,
                  k_world_edges=mcfg.k_world_edges,
                  n_coarse_levels=mcfg.n_coarse_levels).to(mcfg.device)

    return model


class Model(nn.Module):
    def __init__(self, learned_model, collision_radius, n_nodefeatures, n_edgefeatures_mesh, n_edgefeatures_world, k_world_edges, n_coarse_levels):
        super().__init__()

        self._learned_model = learned_model
        self._output_normalizer = normalization.Normalizer(
            size=3)
        self._node_normalizer = normalization.Normalizer(
            size=n_nodefeatures - 3)
        self._mesh_edge_normalizer = normalization.Normalizer(
            size=n_edgefeatures_mesh - 3)
        self._world_edge_normalizer = normalization.Normalizer(
            size=n_edgefeatures_world)

        self.collision_radius = collision_radius
        self.k_world_edges = k_world_edges

        self.nodetype_embedding = nn.Embedding(common.NodeType.SIZE, common.NodeType.SIZE, max_norm=1.)
        # self.edgetype_embedding = nn.Embedding(common.EdgeType.SIZE, 4, max_norm=1.)
        self.n_coarse_levels = n_coarse_levels

        self.vertexlevel_embedding = nn.Embedding(self.n_coarse_levels + 1, 4, max_norm=1.)

        self.normals_f = VertexNormalsPYG()
        self.i = 0

    def embed(self, labels, embedding_layer):
        emb_matrix = embedding_layer.weight.clone().t()
        N = emb_matrix.shape[1]

        if len(labels.shape) == 2 and labels.shape[1] == 1:
            labels = labels[:, 0]
        labels_onehot = torch.nn.functional.one_hot(labels, N).t().float()
        embedding = emb_matrix @ labels_onehot
        embedding = embedding.t()
        return embedding

    def _add_positional_edges(self, sample):
        B = sample.num_graphs

        examples_updated = []
        for i in range(B):
            example = sample.get_example(i)

            vertices_cloth = example['cloth'].pos
            vertices_obstacle_prev = example['obstacle'].pos
            obstacle_vertex_type = example['obstacle'].vertex_type

            indices_from, indices_to = compute_connectivity_pt(vertices_cloth, vertices_obstacle_prev,
                                                               self.collision_radius, k=self.k_world_edges)

            indices_to_vertex_type = obstacle_vertex_type[indices_to][..., 0]
            vertex_type_mask = indices_to_vertex_type != NodeType.OBSTACLE_OMIT

            indices_from = indices_from[vertex_type_mask]
            indices_to = indices_to[vertex_type_mask]

            edges_direct = torch.stack([indices_from, indices_to], dim=0)
            edges_inverse = torch.stack([indices_to, indices_from], dim=0)

            example['cloth', 'world_edge', 'obstacle'].edge_index = edges_direct
            example['obstacle', 'world_edge', 'cloth'].edge_index = edges_inverse

            obstacle_active_mask = torch.zeros_like(vertices_obstacle_prev[:, :1])
            active_obstacle_nodes = torch.unique(indices_to)
            obstacle_active_mask[active_obstacle_nodes] = 1
            obstacle_active_mask = obstacle_active_mask > 0
            example['obstacle'].active_mask = obstacle_active_mask

            examples_updated.append(example)

        sample_updated = Batch.from_data_list(examples_updated)

        return sample_updated

    def get_relative_pos(self, pos, edges):
        edges_pos = gather(pos, edges, 0, 1, 1).permute(0, 2, 1)
        pos_senders, pos_receivers = edges_pos.unbind(-1)
        relative_pos = pos_senders - pos_receivers
        return relative_pos

    def _create_mesh_edge_set(self, sample, is_training, edge_label, normalizer):
        pos = sample['cloth'].pos
        rest_pos = sample['cloth'].rest_pos
        edges = sample['cloth', edge_label, 'cloth'].edge_index.T

        relative_pos = self.get_relative_pos(pos, edges)
        relative_pos_norm = torch.norm(relative_pos, dim=-1, keepdim=True)

        relative_rest_pos = self.get_relative_pos(rest_pos, edges)
        relative_rest_pos_norm = torch.norm(relative_rest_pos, dim=-1, keepdim=True)

        edge_slice = sample._slice_dict['cloth', edge_label, 'cloth']['edge_index']
        lens = edge_slice[1:] - edge_slice[:-1]

        timestep = make_pervertex_tensor_from_lens(lens, sample['cloth'].timestep)
        bending_coeff = make_pervertex_tensor_from_lens(lens, sample['cloth'].bending_coeff_input)
        lame_mu = make_pervertex_tensor_from_lens(lens, sample['cloth'].lame_mu_input)
        lame_lambda = make_pervertex_tensor_from_lens(lens, sample['cloth'].lame_lambda_input)

        edge_features_to_norm = torch.cat([
            relative_pos,
            relative_pos_norm,
            relative_rest_pos,
            relative_rest_pos_norm,
            timestep], dim=-1
        )

        edge_features_nonorm = torch.cat([
            bending_coeff,
            lame_mu,
            lame_lambda], dim=-1
        )

        edge_features_normalized = normalizer(edge_features_to_norm, is_training)
        edge_features_final = torch.cat([edge_features_normalized, edge_features_nonorm], dim=-1)

        sample = add_field_to_pyg_batch(sample, 'features', edge_features_final, ('cloth', edge_label, 'cloth'),
                                        'edge_index', zero_inc=True)

        return sample

    def _create_world_edge_set(self, sample, is_training):
        cloth_pos = sample['cloth'].pos
        obstacle_pos = sample['obstacle'].target_pos
        obstacle_prev_pos = sample['obstacle'].pos

        edges_direct = sample['cloth', 'world_edge', 'obstacle'].edge_index
        edges_inverse = sample['obstacle', 'world_edge', 'cloth'].edge_index

        senders, receivers = edges_direct.unbind(0)
        senders_pos = cloth_pos[senders]
        receivers_pos = obstacle_pos[receivers]
        receivers_prev_pos = obstacle_prev_pos[receivers]

        relative_pos = senders_pos - receivers_pos
        relative_prev_pos = senders_pos - receivers_prev_pos

        relative_pos_norm = torch.norm(relative_pos, dim=-1, keepdim=True)
        relative_prev_pos_norm = torch.norm(relative_prev_pos, dim=-1, keepdim=True)

        edge_slice = sample._slice_dict['cloth', 'world_edge', 'obstacle']['edge_index']
        lens = edge_slice[1:] - edge_slice[:-1]
        timestep = make_pervertex_tensor_from_lens(lens, sample['cloth'].timestep)

        features_direct = torch.cat([
            relative_prev_pos,
            relative_prev_pos_norm,
            relative_pos,
            relative_pos_norm,
            timestep
        ], dim=-1)

        features_inverse = torch.cat([
            -relative_prev_pos,
            relative_prev_pos_norm,
            -relative_pos,
            relative_pos_norm,
            timestep
        ], dim=-1)

        normalizer = self._world_edge_normalizer

        features_combined = torch.cat([features_direct, features_inverse])
        N_direct = features_direct.shape[0]
        features_combined_normalized = normalizer(features_combined, is_training)

        features_direct_normalized = features_combined_normalized[:N_direct]
        features_inverse_normalized = features_combined_normalized[N_direct:]

        sample = add_field_to_pyg_batch(sample, 'features', features_direct_normalized,
                                        ('cloth', 'world_edge', 'obstacle'),
                                        'edge_index', zero_inc=True)
        sample = add_field_to_pyg_batch(sample, 'features', features_inverse_normalized,
                                        ('obstacle', 'world_edge', 'cloth'),
                                        'edge_index', zero_inc=True)

        return sample

    def add_velocities(self, sample):
        for k in ['cloth', 'obstacle']:
            if k == 'cloth':
                velocity = sample[k].pos - sample[k].prev_pos
            else:
                velocity = sample[k].target_pos - sample[k].pos
            sample = add_field_to_pyg_batch(sample, 'velocity', velocity, k, 'pos')
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
        for k in ['cloth', 'obstacle']:
            velocity = sample[k].velocity
            vertex_type_emb = sample[k].vertex_type_embedding
            vertex_level_emb = sample[k].vertex_level_embedding
            normals = sample[k].normals

            slice = sample._slice_dict[k]['pos']
            lens = slice[1:] - slice[:-1]
            timestep = make_pervertex_tensor_from_lens(lens, sample['cloth'].timestep)

            if 'v_mass' in sample[k]:
                v_mass = torch.log(sample[k].v_mass)
            else:
                v_mass = torch.ones_like(velocity[:, :1]) * -1

            if 'bending_coeff' in sample[k]:
                bending_coeff = make_pervertex_tensor_from_lens(lens, sample[k].bending_coeff_input)
                lame_mu = make_pervertex_tensor_from_lens(lens, sample[k].lame_mu_input)
                lame_lambda = make_pervertex_tensor_from_lens(lens, sample[k].lame_lambda_input)
            else:
                device = velocity.device
                bending_coeff = torch.ones_like(velocity[:, :1]).to(device) * -1
                lame_mu = torch.ones_like(velocity[:, :1]).to(device) * -1
                lame_lambda = torch.ones_like(velocity[:, :1]).to(device) * -1

            node_features = torch.cat(
                [velocity, vertex_type_emb, vertex_level_emb, normals, timestep, v_mass, bending_coeff, lame_mu,
                 lame_lambda], dim=-1)
            # print('node_features', node_features.shape)

            sample = add_field_to_pyg_batch(sample, 'node_features', node_features, k, 'pos')
        return sample

    def normalize_node_features(self, sample, is_training):
        cloth_node_features = sample['cloth'].node_features
        obstacle_node_features = sample['obstacle'].node_features
        obstacle_active_mask = sample['obstacle'].active_mask[:, 0]

        active_obstacle_node_features = obstacle_node_features[obstacle_active_mask]

        N_cloth = cloth_node_features.shape[0]

        all_features = torch.cat([cloth_node_features, active_obstacle_node_features])
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

    def _make_nodefeatures(self, sample):

        sample = self.add_velocities(sample)
        sample = self.add_vertex_type_embedding(sample)
        sample = self.add_vertex_level_embedding(sample)
        sample = self.normals_f(sample, 'cloth', 'pos')
        sample = self.normals_f(sample, 'obstacle', 'pos')
        sample = self.add_node_features(sample)

        return sample

    def replace_pinned_verts(self, sample):
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

    def _normalize(self, sample, is_training):
        """Builds input graph."""

        sample = self.replace_pinned_verts(sample)

        # construct graph edges
        sample = self._add_positional_edges(sample)
        sample = self._make_nodefeatures(sample)

        sample = self._create_mesh_edge_set(sample, is_training, 'mesh_edge', self._mesh_edge_normalizer)
        for i in range(self.n_coarse_levels):
            key = f'coarse_edge{i}'
            sample = self._create_mesh_edge_set(sample, is_training, key, self._mesh_edge_normalizer)
        sample = self._create_world_edge_set(sample, is_training)

        sample = self.normalize_node_features(sample, is_training)

        return sample

    def _get_position(self, sample, is_training):
        """Integrate model outputs."""
        vertex_type = sample['cloth'].vertex_type
        pinned_mask = vertex_type == NodeType.HANDLE

        cloth_features = sample['cloth'].node_features
        acceleration = self._output_normalizer.inverse(cloth_features)

        sample = add_field_to_pyg_batch(sample, 'pred_acceleration', acceleration, 'cloth', 'pos')

        # integrate forward
        cur_position = sample['cloth'].pos
        prev_position = sample['cloth'].prev_pos
        target_position = sample['cloth'].target_pos

        velocity = cur_position - prev_position
        pred_velocity = velocity + acceleration
        target_velocity = target_position - cur_position
        pred_velocity = pred_velocity * torch.logical_not(pinned_mask) + target_velocity * pinned_mask

        position = cur_position + pred_velocity
        position = position * torch.logical_not(pinned_mask) + target_position * pinned_mask

        sample = add_field_to_pyg_batch(sample, 'pred_pos', position, 'cloth', 'pos')
        sample = add_field_to_pyg_batch(sample, 'pred_velocity', pred_velocity, 'cloth', 'pos')

        target_acceleration = target_position - 2 * cur_position + prev_position
        target_acceleration_norm = self._output_normalizer(target_acceleration, is_training)
        sample = add_field_to_pyg_batch(sample, 'target_acceleration', target_acceleration_norm, 'cloth', 'pos')
        return sample

    def forward(self, inputs, is_training=True):
        sample = self._normalize(inputs, is_training=is_training)
        sample = self._learned_model(sample)
        sample = self._get_position(sample, is_training=is_training)
        return sample
