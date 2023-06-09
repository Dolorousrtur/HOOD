import functools

import torch
from torch import nn
from torch_geometric.data import Batch
from torch_geometric.nn import MessagePassing
from torch_geometric.typing import Size

from models import networks


class GraphNetBlock(MessagePassing):
    def __init__(self, node_processor_fn, edge_processor_fn):
        super().__init__(aggr='add')
        self.node_processor = node_processor_fn()
        self.mesh_edge_processor = edge_processor_fn()
        self.world_edge_processor = edge_processor_fn()

        self.inspector.inspect(self.message_mesh)
        self.inspector.inspect(self.message_world_direct)
        self.inspector.inspect(self.message_world_inverse)

        self.__user_args__ = self.inspector.keys(
            ['message_mesh', 'message_world_direct', 'message_world_inverse', 'aggregate', 'update']).difference(
            self.special_args)

    def forward(self, sample):
        sample = self.propagate(sample)
        return sample

    def message_mesh(self, node_features_i, node_features_j, edge_features):
        in_features = torch.cat([node_features_i, node_features_j, edge_features], dim=-1)
        out_features = self.mesh_edge_processor(in_features)
        return out_features

    def message_world_direct(self, cloth_features_j, obstacle_features_i, edge_features):
        in_features = torch.cat([obstacle_features_i, cloth_features_j, edge_features], dim=-1)
        out_features = self.world_edge_processor(in_features)
        return out_features

    def message_world_inverse(self, cloth_features_i, obstacle_features_j, edge_features):
        in_features = torch.cat([cloth_features_i, obstacle_features_j, edge_features], dim=-1)
        out_features = self.world_edge_processor(in_features)
        return out_features

    def update_mesh_edge_features(self, sample):
        edge_index = sample['mesh_edge'].edge_index
        node_features = sample['cloth'].node_features
        size = self._check_input(edge_index, None)
        coll_dict = self._collect(self.__user_args__, edge_index,
                                     size, dict(node_features=node_features))
        coll_dict['edge_features'] = sample['mesh_edge'].features
        msg_kwargs = self.inspector.distribute('message_mesh', coll_dict)
        out = self.message_mesh(**msg_kwargs)
        return out

    def update_world_edge_direct_features(self, sample):
        edge_index = sample['cloth', 'world_edge', 'obstacle'].edge_index
        edge_features = sample['cloth', 'world_edge', 'obstacle'].features
        cloth_features = sample['cloth'].node_features
        obstacle_features = sample['obstacle'].node_features
        N_cloth = cloth_features.shape[0]
        N_obstacle = obstacle_features.shape[0]
        size = (N_cloth, N_obstacle)

        size = self._check_input(edge_index, size)
        __user_args__ = self.inspector.keys(['message_world_direct']).difference(self.special_args)
        coll_dict = self._collect(__user_args__, edge_index,
                                     size, dict(cloth_features=cloth_features, obstacle_features=obstacle_features))
        coll_dict['edge_features'] = edge_features
        msg_kwargs = self.inspector.distribute('message_world_direct', coll_dict)
        out = self.message_world_direct(**msg_kwargs)
        return out

    def update_world_edge_inverse_features(self, sample):
        edge_index = sample['obstacle', 'world_edge', 'cloth'].edge_index
        edge_features = sample['obstacle', 'world_edge', 'cloth'].features
        cloth_features = sample['cloth'].node_features
        obstacle_features = sample['obstacle'].node_features
        N_cloth = cloth_features.shape[0]
        N_obstacle = obstacle_features.shape[0]
        size = (N_obstacle, N_cloth)

        size = self._check_input(edge_index, size)
        __user_args__ = self.inspector.keys(['message_world_inverse']).difference(self.special_args)
        coll_dict = self._collect(__user_args__, edge_index,
                                     size, dict(cloth_features=cloth_features, obstacle_features=obstacle_features))
        coll_dict['edge_features'] = edge_features
        msg_kwargs = self.inspector.distribute('message_world_inverse', coll_dict)
        out = self.message_world_inverse(**msg_kwargs)
        return out

    def aggregate_nodes(self, edge_features, edge_index, user_args, size, **kwargs):
        coll_dict = self._collect(user_args, edge_index,
                                     size, kwargs)
        aggr_kwargs = self.inspector.distribute('aggregate', coll_dict)
        node_features = self.aggregate(edge_features, **aggr_kwargs)
        return node_features

    def update(self, aggregated_features_mesh, aggregated_features_world, features):
        input_features = torch.cat([aggregated_features_mesh, aggregated_features_world, features], dim=1)
        out_features = self.node_processor(input_features)
        return out_features

    def propagate(self, sample, size: Size = None, **kwargs):
        N_cloth = sample['cloth'].node_features.shape[0]
        N_obstacle = sample['obstacle'].node_features.shape[0]

        mesh_edge_features_updated = self.update_mesh_edge_features(sample)
        world_direct_features_updated = self.update_world_edge_direct_features(sample)
        world_inverse_features_updated = self.update_world_edge_inverse_features(sample)

        aggr_args = self.inspector.keys(['aggregate']).difference(self.special_args)
        mesh_edge_index = sample['mesh_edge'].edge_index
        mesh_size = (N_cloth, N_cloth)
        cloth_features_from_mesh = self.aggregate_nodes(mesh_edge_features_updated, mesh_edge_index, aggr_args,
                                                        mesh_size)

        world_inverse_edge_index = sample['obstacle', 'world_edge', 'cloth'].edge_index
        world_inverse_size = (N_obstacle, N_cloth)
        cloth_features_from_world = self.aggregate_nodes(world_inverse_features_updated, world_inverse_edge_index,
                                                         aggr_args, world_inverse_size)

        world_direct_edge_index = sample['cloth', 'world_edge', 'obstacle'].edge_index
        world_direct_size = (N_cloth, N_obstacle)
        obstacle_features_aggr = self.aggregate_nodes(world_direct_features_updated, world_direct_edge_index, aggr_args,
                                                      world_direct_size)
        obstacle_features_dummy = torch.zeros_like(obstacle_features_aggr)

        cloth_features = sample['cloth'].node_features
        obstacle_features = sample['obstacle'].node_features

        cloth_features_new = self.update(cloth_features_from_mesh, cloth_features_from_world, cloth_features)
        obstacle_features_new = self.update(obstacle_features_dummy, obstacle_features_aggr, obstacle_features)

        cloth_features_new = cloth_features + cloth_features_new
        obstacle_features_new = obstacle_features + obstacle_features_new
        sample['cloth'].node_features = cloth_features_new
        sample['obstacle'].node_features = obstacle_features_new

        mesh_edge_features_new = mesh_edge_features_updated + sample['mesh_edge'].features
        world_direct_features_new = world_direct_features_updated + sample['cloth', 'world_edge', 'obstacle'].features
        world_inverse_features_new = world_inverse_features_updated + sample['obstacle', 'world_edge', 'cloth'].features

        sample['mesh_edge'].features = mesh_edge_features_new
        sample['cloth', 'world_edge', 'obstacle'].features = world_direct_features_new
        sample['obstacle', 'world_edge', 'cloth'].features = world_inverse_features_new

        return sample


class EncodeProcessDecode(nn.Module):
    def __init__(self, output_size: int, latent_size: int, num_layers: int, n_nodefeatures: int,
                 n_edgefeatures_mesh: int,
                 n_edgefeatures_world: int,
                 message_passing_steps: int):
        """Encode-Process-Decode GraphNet model."""
        super().__init__()
        self._latent_size = latent_size
        self._output_size = output_size
        self._num_layers = num_layers
        self.n_nodefeatures = n_nodefeatures
        self.n_edgefeatures_mesh = n_edgefeatures_mesh
        self.n_edgefeatures_world = n_edgefeatures_world
        self._message_passing_steps = message_passing_steps

        self.node_encoder = self._make_mlp(self.n_nodefeatures, self._latent_size)
        self.decoder = self._make_mlp(self._latent_size, self._output_size, layer_norm=False)

        edgeset_encoders = {}
        edgeset_encoders['mesh'] = self._make_mlp(self.n_edgefeatures_mesh, self._latent_size)
        edgeset_encoders['world'] = self._make_mlp(self.n_edgefeatures_world, self._latent_size)
        self.edgeset_encoders = nn.ModuleDict(edgeset_encoders)

        node_proc_model = functools.partial(self._make_mlp, input_size=self._latent_size * (1 + 2),
                                            output_size=self._latent_size)
        edge_proc_model = functools.partial(self._make_mlp, input_size=self._latent_size * 3,
                                            output_size=self._latent_size)

        processor_steps = []
        for i in range(message_passing_steps):
            processor_steps.append(GraphNetBlock(node_proc_model, edge_proc_model))
        self.processor_steps = nn.ModuleList(processor_steps)
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

        for i in range(self._message_passing_steps):
            sample = self.processor_steps[i](sample)

        return self._decode(sample)
