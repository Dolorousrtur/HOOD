import functools
from collections import defaultdict

import torch
from torch import nn
from torch_geometric.data import Batch
from torch_geometric.typing import Size

from models import networks
from models.core.base import BaseBlock


class GraphNetBlock(BaseBlock):
    def __init__(self, node_processor_fn, edge_processor_fn, edge_sets, latent_size):

        self.edge_sets = edge_sets
        self.latent_size = latent_size
        self.edge_keys = sorted(list(set([v['edge_key'] for v in edge_sets.values()])))
        edge_processor_dict = {v['edge_key']: edge_processor_fn for v in self.edge_sets.values()}
        node_processor_dict = dict(node=node_processor_fn)
        super().__init__(edge_processor_dict, node_processor_dict)

    def get_updated_edge_features(self, sample):
        updated_features = {}

        for edgeset_key, edgeset in self.edge_sets.items():
            edge_features_updated = self.update_edge_features(sample, edgeset['edge_key'], edgeset['source'],
                                                              edgeset['edge_key'], edgeset['target'])

            updated_features[edgeset_key] = edge_features_updated

        return updated_features

    def aggregate_node_features(self, sample, updated_edge_features):
        nodeset_input_features = defaultdict(dict)

        for edgeset_key, edgeset in self.edge_sets.items():
            source_nodes = sample[edgeset['source']]
            target_nodes = sample[edgeset['target']]

            N_source = source_nodes.node_features.shape[0]
            N_target = target_nodes.node_features.shape[0]
            size = (N_source, N_target)

            edge_index = sample[edgeset['source'], edgeset['edge_key'], edgeset['target']].edge_index
            edge_features = updated_edge_features[edgeset_key]
            aggregated_features = self.aggregate_nodes(edge_features, edge_index, size)

            nodeset_input_features[edgeset['target']][edgeset['edge_key']] = aggregated_features

        return nodeset_input_features

    def get_updated_node_features(self, sample, nodeset_input_features):
        updated_node_features = dict()
        for node_key, features_dict in nodeset_input_features.items():
            nodes = sample[node_key]

            node_features = nodes.node_features
            device = node_features.device
            N_nodes = node_features.shape[0]

            input_features_list = [node_features]

            for edge_key in self.edge_keys:
                if edge_key in features_dict:
                    input_features_list.append(features_dict[edge_key])
                else:
                    dummy_features = torch.zeros(N_nodes, self.latent_size, device=device)
                    input_features_list.append(dummy_features)

            updated_features = self.update(input_features_list, 'node')
            updated_node_features[node_key] = updated_features

        return updated_node_features

    def update_node_features_sample(self, sample, updated_node_features):
        for node_key, updated_features in updated_node_features.items():
            nodes = sample[node_key]
            prev_features = nodes.node_features
            sample[node_key].node_features = prev_features + updated_features

        return sample

    def update_edge_features_sample(self, sample, updated_edge_features):
        for edgeset_key, edgeset in self.edge_sets.items():
            # print(edgeset_key, edgeset['source'], edgeset['edge_key'], edgeset['target'])
            prev_features = sample[edgeset['source'], edgeset['edge_key'], edgeset['target']].features
            update_features = updated_edge_features[edgeset_key]

            sample[edgeset['source'], edgeset['edge_key'], edgeset['target']].features = prev_features + update_features

        return sample

    def propagate(self, sample, size: Size = None, **kwargs):

        # FOR EACH EDGE SET GET FEATURES
        updated_edge_features = self.get_updated_edge_features(sample)

        # FOR EACH NODE TYPE
        # FOR EACH EDGESET
        # AGGREGATE FEATURES
        nodeset_input_features = self.aggregate_node_features(sample, updated_edge_features)

        # FOR EACH NODE TYPE
        # PASS AGGREGATED FEATURES THROUGH MLP TO GET UPDATED
        updated_node_features = self.get_updated_node_features(sample, nodeset_input_features)

        # FOR EACH NODE TYPE
        # UPDATE FEATURES
        sample = self.update_node_features_sample(sample, updated_node_features)

        # FOR EACH EDGESET
        # UPDATE FEATURES
        sample = self.update_edge_features_sample(sample, updated_edge_features)

        return sample


class DownSample(nn.Module):
    def __init__(self, coarse_edgeset, edgesets_to_filter):
        super().__init__()

        self.coarse_edgeset = coarse_edgeset
        self.edgesets_to_filter = edgesets_to_filter

    def get_remaining_node_mask(self, sample):
        coarse_edgeset = sample[
            self.coarse_edgeset['source'], self.coarse_edgeset['edge_key'], self.coarse_edgeset['target']]
        coarse_edge_index = coarse_edgeset.edge_index
        unique_nodes = torch.unique(coarse_edge_index.view(-1))

        nodes = sample[self.coarse_edgeset['source']]
        N_nodes = nodes.node_features.shape[0]

        mask = torch.zeros(N_nodes).bool().to(coarse_edge_index.device)
        mask[unique_nodes] = 1
        return mask

    def filter_edgeset(self, sample, edgeset_to_filter, nodes_mask):
        e2f = edgeset_to_filter

        edgeset = sample[e2f['source'], e2f['edge_key'], e2f['target']]
        edge_index = edgeset.edge_index
        features = edgeset.features

        # print('edge_index', edge_index.shape)
        edge_mask = torch.ones_like(edge_index[0]).bool()
        # print('edge_mask', edge_mask.shape, edge_mask.sum())
        if self.coarse_edgeset['source'] == e2f['source']:
            edge_mask *= nodes_mask[edge_index[0]]
        # print('edge_mask', edge_mask.shape, edge_mask.sum())
        if self.coarse_edgeset['source'] == e2f['target']:
            edge_mask *= nodes_mask[edge_index[1]]
        # print('edge_mask', edge_mask.shape, edge_mask.sum())

        edge_index_new = edge_index[:, edge_mask]
        # print('edge_index_new', edge_index_new.shape)
        features_new = features[edge_mask]

        out_dict = dict()
        out_dict['old'] = dict(edge_index=edge_index, features=features)
        out_dict['mask'] = edge_mask

        edgeset.features = features_new
        edgeset.edge_index = edge_index_new

        return sample, out_dict

    def forward(self, sample):
        remaining_nodes_mask = self.get_remaining_node_mask(sample)

        stashed_edge_data = defaultdict(dict)

        for edge_label, edgeset in self.edgesets_to_filter.items():
            sample, stashed_edge_data[edge_label] = self.filter_edgeset(sample, edgeset, remaining_nodes_mask)

        return sample, stashed_edge_data


class UpSample(nn.Module):
    def __init__(self, edgesets):
        super().__init__()
        self.edgesets = edgesets

    def restore_edgeset(self, sample, edgeset_dict, stashed_data):
        edgeset = sample[edgeset_dict['source'], edgeset_dict['edge_key'], edgeset_dict['target']]

        mask = stashed_data['mask']

        edgeset.edge_index = stashed_data['old']['edge_index']
        features_restored = stashed_data['old']['features']
        features_restored[mask] = edgeset.features

        edgeset.features = features_restored

        return sample

    def forward(self, sample, stashed_data):
        for edge_label, stashed_dict in stashed_data.items():
            edgeset_dict = self.edgesets[edge_label]
            sample = self.restore_edgeset(sample, edgeset_dict, stashed_dict)

        return sample


class EncodeProcessDecode(nn.Module):
    def __init__(self, mcfg):
        """Encode-Process-Decode GraphNet model."""
        super().__init__()
        self._latent_size = mcfg.latent_size
        self._output_size = mcfg.output_size
        self._num_layers = mcfg.num_layers
        self.n_nodefeatures = mcfg.n_nodefeatures
        self.n_edgefeatures_mesh = mcfg.n_edgefeatures_mesh
        self.n_edgefeatures_world = mcfg.n_edgefeatures_world
        self.n_edgefeatures_coarse = mcfg.n_edgefeatures_coarse
        self._message_passing_steps = mcfg.message_passing_steps

        self.node_encoder = self._make_mlp(self.n_nodefeatures, self._latent_size)
        self.decoder = self._make_mlp(self._latent_size, self._output_size, layer_norm=False)

        edgeset_encoders = {}
        edgeset_encoders['mesh'] = self._make_mlp(self.n_edgefeatures_mesh, self._latent_size)
        edgeset_encoders['world'] = self._make_mlp(self.n_edgefeatures_world, self._latent_size)
        edgeset_encoders['coarse'] = self._make_mlp(self.n_edgefeatures_coarse, self._latent_size)
        self.edgeset_encoders = nn.ModuleDict(edgeset_encoders)

        node_proc_model = functools.partial(self._make_mlp, input_size=self._latent_size * (1 + 3),
                                            output_size=self._latent_size)
        node_proc_model_coarse = functools.partial(self._make_mlp, input_size=self._latent_size * (1 + 2),
                                                   output_size=self._latent_size)
        edge_proc_model = functools.partial(self._make_mlp, input_size=self._latent_size * 3,
                                            output_size=self._latent_size)

        edge_sets_full = defaultdict(dict)
        edge_sets_full['mesh']['source'] = 'cloth'
        edge_sets_full['mesh']['edge_key'] = 'mesh_edge'
        edge_sets_full['mesh']['target'] = 'cloth'

        edge_sets_full['coarse']['source'] = 'cloth'
        edge_sets_full['coarse']['edge_key'] = 'coarse_edge'
        edge_sets_full['coarse']['target'] = 'cloth'

        for i in range(3):
            edge_sets_full[f'coarse{i}']['source'] = 'cloth'
            edge_sets_full[f'coarse{i}']['edge_key'] = f'coarse_edge{i}'
            edge_sets_full[f'coarse{i}']['target'] = 'cloth'

        edge_sets_full['world_direct']['source'] = 'obstacle'
        edge_sets_full['world_direct']['edge_key'] = 'world_edge'
        edge_sets_full['world_direct']['target'] = 'cloth'

        edge_sets_full['world_inverse']['source'] = 'cloth'
        edge_sets_full['world_inverse']['edge_key'] = 'world_edge'
        edge_sets_full['world_inverse']['target'] = 'obstacle'

        edgeset_level0 = {k: edge_sets_full[k] for k in ['mesh', 'coarse0', 'world_direct', 'world_inverse']}
        edgeset_level1 = {k: edge_sets_full[k] for k in ['coarse0', 'coarse1', 'world_direct', 'world_inverse']}
        edgeset_level2 = {k: edge_sets_full[k] for k in ['coarse1', 'world_direct', 'world_inverse']}

        edge_sets_2filt = {k: edge_sets_full[k] for k in ['world_direct', 'world_inverse']}
        self.downsample1 = DownSample(edge_sets_full['coarse0'], edge_sets_2filt)
        self.downsample2 = DownSample(edge_sets_full['coarse1'], edge_sets_2filt)

        self.upsample = UpSample(edge_sets_2filt)

        processor_steps1 = []
        for i in range(3):
            processor_steps1.append(GraphNetBlock(node_proc_model, edge_proc_model, edgeset_level0, self._latent_size))
        self.processor_steps1 = nn.ModuleList(processor_steps1)

        processor_steps2 = []
        for i in range(3):
            processor_steps2.append(GraphNetBlock(node_proc_model, edge_proc_model, edgeset_level1, self._latent_size))
        self.processor_steps2 = nn.ModuleList(processor_steps2)

        processor_steps3 = []
        for i in range(3):
            processor_steps3.append(
                GraphNetBlock(node_proc_model_coarse, edge_proc_model, edgeset_level2, self._latent_size))
        self.processor_steps3 = nn.ModuleList(processor_steps3)

        processor_steps4 = []
        for i in range(3):
            processor_steps4.append(GraphNetBlock(node_proc_model, edge_proc_model, edgeset_level1, self._latent_size))
        self.processor_steps4 = nn.ModuleList(processor_steps4)

        processor_steps5 = []
        for i in range(3):
            processor_steps5.append(GraphNetBlock(node_proc_model, edge_proc_model, edgeset_level0, self._latent_size))
        self.processor_steps5 = nn.ModuleList(processor_steps5)
        self.i = 0

    def _make_mlp(self, input_size: int, output_size: int, layer_norm: bool = True) -> nn.Module:
        """Builds an MLP."""
        widths = [input_size] + [self._latent_size] * self._num_layers + [output_size]
        network = networks.MLP(widths, activate_final=None)
        if layer_norm:
            network = nn.Sequential(network, nn.LayerNorm(output_size))
        return network

    def _encode_nodes(self, sample):
        cloth_features = sample['cloth'].node_features
        obstacle_features = sample['obstacle'].node_features
        obstacle_active_mask = sample['obstacle'].active_mask[:, 0]
        obstacle_features_active = obstacle_features[obstacle_active_mask]

        N_cloth = cloth_features.shape[0]
        N_obstacle = obstacle_features.shape[0]

        combined_features = torch.cat([cloth_features, obstacle_features_active], dim=0)
        combined_latents = self.node_encoder(combined_features)

        cloth_latents = combined_latents[:N_cloth]
        obstacle_active_latents = combined_latents[N_cloth:]
        latent_features = obstacle_active_latents.shape[1]

        obstacle_latents = torch.zeros(N_obstacle, latent_features).to(obstacle_active_latents.device)
        obstacle_latents[obstacle_active_mask] = obstacle_active_latents

        sample['cloth'].node_features = cloth_latents
        sample['obstacle'].node_features = obstacle_latents

        return sample

    def _encode_edges(self, sample):
        mesh_edge_features = sample['cloth', 'mesh_edge', 'cloth'].features
        mesh_edge_latents = self.edgeset_encoders['mesh'](mesh_edge_features)
        sample['cloth', 'mesh_edge', 'cloth'].features = mesh_edge_latents

        for i in range(3):
            key = f'coarse_edge{i}'
            coarse_edge_features = sample['cloth', key, 'cloth'].features
            coarse_edge_latents = self.edgeset_encoders['coarse'](coarse_edge_features)
            sample['cloth', key, 'cloth'].features = coarse_edge_latents

        cloth_edge_features_direct = sample['cloth', 'world_edge', 'obstacle'].features
        cloth_edge_features_inverse = sample['obstacle', 'world_edge', 'cloth'].features
        N_world_edges = cloth_edge_features_direct.shape[0]

        cloth_edge_features_cat = torch.cat([cloth_edge_features_direct, cloth_edge_features_inverse], dim=0)
        cloth_edge_latents_cat = self.edgeset_encoders['world'](cloth_edge_features_cat)
        cloth_edge_latents_direct = cloth_edge_latents_cat[:N_world_edges]
        cloth_edge_latents_inverse = cloth_edge_latents_cat[N_world_edges:]
        sample['cloth', 'world_edge', 'obstacle'].features = cloth_edge_latents_direct
        sample['obstacle', 'world_edge', 'cloth'].features = cloth_edge_latents_inverse
        return sample

    def _encode(self, sample: Batch) -> Batch:
        """Encodes node and edge features into latent features."""
        sample = self._encode_nodes(sample)
        sample = self._encode_edges(sample)
        return sample

    def _decode(self, sample):
        """Decodes node features from graph."""
        cloth_features = sample['cloth'].node_features
        out_features = self.decoder(cloth_features)
        sample['cloth'].node_features = out_features
        return sample

    def forward(self, sample) -> torch.Tensor:
        """Encodes and processes a multigraph, and returns node features."""
        sample = self._encode(sample)
        # print('\nprocessor_steps1')
        for ps in self.processor_steps1:
            sample = ps(sample)
        sample, stashed_edge_data1 = self.downsample1(sample)
        # print('\nprocessor_steps2')
        for ps in self.processor_steps2:
            sample = ps(sample)
        sample, stashed_edge_data2 = self.downsample2(sample)
        # print('\nprocessor_steps3')
        for ps in self.processor_steps3:
            sample = ps(sample)
        sample = self.upsample(sample, stashed_edge_data2)
        # print('\nprocessor_steps4')
        for ps in self.processor_steps4:
            sample = ps(sample)
        sample = self.upsample(sample, stashed_edge_data1)
        # print('\nprocessor_steps5')
        for ps in self.processor_steps5:
            sample = ps(sample)

        # assert False

        return self._decode(sample)
