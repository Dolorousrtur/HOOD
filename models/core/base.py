from collections import defaultdict

import torch
from torch import nn
from torch_geometric.nn import MessagePassing


class BaseBlock(MessagePassing):
    def __init__(self, edge_processor_dict, node_processor_dict):
        super().__init__(aggr='add')

        edge_processor_dict = {k: v() for k, v in edge_processor_dict.items()}
        self.edge_processor_dict = nn.ModuleDict(edge_processor_dict)

        node_processor_dict = {k: v() for k, v in node_processor_dict.items()}
        self.node_processor_dict = nn.ModuleDict(node_processor_dict)

        self.inspector.inspect(self.message)

        self.__user_args__ = self.inspector.keys(
            ['message', 'aggregate', 'update']).difference(
            self.special_args)

    def forward(self, sample):
        sample = self.propagate(sample)
        return sample

    def message(self, edge_processor_key, target_features_i=None, source_features_j=None, edge_features=None):
        in_features = []
        for features in [target_features_i, source_features_j, edge_features]:
            if features is not None:
                in_features.append(features)

        assert (len(in_features) > 0)

        in_features = torch.cat(in_features, dim=-1)
        out_features = self.edge_processor_dict[edge_processor_key](in_features)
        return out_features

    def aggregate_nodes(self, edge_features, edge_index, size, **kwargs):
        user_args = self.inspector.keys(['aggregate']).difference(self.special_args)
        coll_dict = self._collect(user_args, edge_index,
                                     size, kwargs)
        aggr_kwargs = self.inspector.distribute('aggregate', coll_dict)
        node_features = self.aggregate(edge_features, **aggr_kwargs)
        return node_features

    def update_edge_features(self, sample, edge_processor_key, source_key, edge_key, target_key):

        mesh_edges = sample[source_key, edge_key, target_key]
        source = sample[source_key]
        target = sample[target_key]

        edge_index = mesh_edges.edge_index


        source_features = source.node_features
        target_features = target.node_features
        N_source = source_features.shape[0]
        N_target = target_features.shape[0]
        size = (N_source, N_target)
        size = self._check_input(edge_index, size)
        coll_dict = self._collect(self.__user_args__, edge_index,
                                     size, dict(source_features=source_features,
                                                target_features=target_features))
        coll_dict['edge_features'] = mesh_edges.features
        coll_dict['edge_processor_key'] = edge_processor_key
        msg_kwargs = self.inspector.distribute('message', coll_dict)
        out = self.message(**msg_kwargs)
        return out

    def update(self, features_list, node_processor_key):
        input_features = torch.cat(features_list, dim=1)
        out_features = self.node_processor_dict[node_processor_key](input_features)
        # print(f'update\tfrom {input_features.shape} to {out_features.shape}')
        return out_features


def make_edgesets_dict(n_coarse_levels, body=True, selfcoll=False):
    edge_sets_full = defaultdict(dict)
    edge_sets_full['mesh']['source'] = 'cloth'
    edge_sets_full['mesh']['edge_key'] = 'mesh_edge'
    edge_sets_full['mesh']['target'] = 'cloth'

    for i in range(n_coarse_levels):
        edge_sets_full[f'coarse{i}']['source'] = 'cloth'
        edge_sets_full[f'coarse{i}']['edge_key'] = f'coarse_edge{i}'
        edge_sets_full[f'coarse{i}']['target'] = 'cloth'

    if body:
        edge_sets_full['world_direct']['source'] = 'obstacle'
        edge_sets_full['world_direct']['edge_key'] = 'world_edge'
        edge_sets_full['world_direct']['target'] = 'cloth'

        edge_sets_full['world_inverse']['source'] = 'cloth'
        edge_sets_full['world_inverse']['edge_key'] = 'world_edge'
        edge_sets_full['world_inverse']['target'] = 'obstacle'

    if selfcoll:
        edge_sets_full['world_cloth']['source'] = 'cloth'
        edge_sets_full['world_cloth']['edge_key'] = 'world_edge'
        edge_sets_full['world_cloth']['target'] = 'cloth'


    return edge_sets_full
