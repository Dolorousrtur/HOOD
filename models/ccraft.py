# ccoo == cut_coarse_optbody_ombp
import importlib
from dataclasses import dataclass

import torch
from omegaconf import II
from torch import nn
from torch_geometric.data import Batch

from utils import normalization, common
from utils.cloth_and_material import VertexNormalsPYG, FaceNormals
from utils.common import NodeType, gather, pcd_replace_features_packed, add_field_to_pyg_batch, \
    make_pervertex_tensor_from_lens
from utils.icontour import compute_icontour_grad, make_icontour_masks
from utils.warp_u.proximity import get_closest_nodes_and_faces_pt_dummmy, get_proximity_self_pt_dummmy


@dataclass
class Config:
    core_model: str = 'ccraft'
    output_size: int = 3
    latent_size: int = 128
    num_layers: int = 2
    n_nodefeatures: int = 24
    n_edgefeatures_mesh: int = 9
    n_edgefeatures_coarse: int = 9
    n_edgefeatures_world: int = 20
    n_edgefeatures_body: int = 13
    message_passing_steps: int = 15
    collision_radius: float = 5e-3
    body_collision_radius: float = 5e-3
    architecture: str = "f|f|f|f|f|f|f|f|f|f|f|f|f|f|f"
    selfcoll: bool = True
    n_coarse_levels: int = 2
    regular_ts: float = II("experiment.regular_ts")
    allrep: bool = False

    device: str = II('device')


def create(mcfg):
    core_model = importlib.import_module(f'models.core.{mcfg.core_model}')
    learned_model = core_model.EncodeProcessDecode(mcfg)
    model = Model(learned_model, mcfg).to(mcfg.device)
    return model


class Model(nn.Module):
    def __init__(self, learned_model, mcfg):
        super().__init__()
        self.mcfg = mcfg

        self._learned_model = learned_model
        self._output_normalizer = normalization.Normalizer(
            size=3)
        self._node_normalizer = normalization.Normalizer(
            size=self.mcfg.n_nodefeatures - 4)
        self._mesh_edge_normalizer = normalization.Normalizer(
            size=self.mcfg.n_edgefeatures_mesh - 4)
        self._world_edge_normalizer = normalization.Normalizer(
            size=self.mcfg.n_edgefeatures_world - 4)
        self._coarse_edge_normalizer = normalization.Normalizer(
            size=self.mcfg.n_edgefeatures_coarse - 4)
        self._body_edge_normalizer = normalization.Normalizer(
            size=self.mcfg.n_edgefeatures_body-1)

        self.collision_radius = self.mcfg.collision_radius
        self.nodetype_embedding = nn.Embedding(4, common.NodeType.SIZE, max_norm=1.)
        self.edgetype_embedding = nn.Embedding(4, 4, max_norm=1.)
        self.vertexlevel_embedding = nn.Embedding(self.mcfg.n_coarse_levels + 1, 4, max_norm=1.)

        self.normals_f = VertexNormalsPYG()
        self.f_normals_f = FaceNormals()
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

    def make_future_pinned_edges(self, example):
        vertex_type = example['cloth'].vertex_type
        pinned_mask = vertex_type == common.NodeType.HANDLE

        if pinned_mask.sum() == 0:
            device = pinned_mask.device
            return torch.zeros((0, 2), dtype=torch.long, device=device), \
                torch.zeros((0, 1), dtype=torch.long, device=device), \
                torch.zeros((0, 1), dtype=torch.long, device=device)

        pinned_ids = torch.nonzero(pinned_mask)[:, 0]
        input_pos = example['cloth'].input_pos
        velocity = example['cloth'].velocity
        faces = example['cloth'].faces_batch.T

        velocity_dx = velocity

        next_pos = input_pos + velocity_dx
        next_pos_pinned = next_pos[pinned_mask[:, 0]]

        face_to, edges, _, _ = get_proximity_self_pt_dummmy(next_pos_pinned, next_pos,
                                                           faces,
                                                           self.mcfg.body_collision_radius,
                                                            query_ids=pinned_ids)

        ids_from, ids_to = edges.unbind(-1)
        # ids_from = pinned_ids[pinned_ids_from]


        return edges, ids_from[:, None], face_to[:, None]


    def add_body_edges(self, example):

        if 'obstacle' not in example.node_types:
            return example


        vertices_cloth = example['cloth'].pos
        vertices_obstacle = example['obstacle'].pos
        obstacle_faces = example['obstacle'].faces_batch.T
        obstacle_vertex_type = example['obstacle'].vertex_type

        vertices_cloth = vertices_cloth.contiguous()
        vertices_obstacle = vertices_obstacle.contiguous()

        indices_from, indices_to, _, _ = get_closest_nodes_and_faces_pt_dummmy(vertices_cloth, vertices_obstacle, obstacle_faces, self.mcfg.body_collision_radius)


        indices_to_vertex_type = obstacle_vertex_type[indices_to][..., 0]
        vertex_type_mask = indices_to_vertex_type != NodeType.OBSTACLE_OMIT

        if 'cutout_mask' in example['obstacle']:
            obstacle_cutout_mask = example['obstacle'].cutout_mask
            obstacle_cutout_mask = obstacle_cutout_mask[indices_to]
            vertex_type_mask = torch.logical_and(vertex_type_mask, obstacle_cutout_mask)

        if 'cutout_mask' in example['cloth']:
            node_mask = example['cloth'].cutout_mask
            cutout_mask = node_mask[indices_from]
            vertex_type_mask = torch.logical_and(vertex_type_mask, cutout_mask)

        indices_from = indices_from[vertex_type_mask]
        indices_to = indices_to[vertex_type_mask]

        edges_direct = torch.stack([indices_from, indices_to], dim=0)
        edges_inverse = torch.stack([indices_to, indices_from], dim=0)

        example['cloth', 'body_edge', 'obstacle'].edge_index = edges_direct
        example['obstacle', 'body_edge', 'cloth'].edge_index = edges_inverse

        obstacle_active_mask = torch.zeros_like(vertices_obstacle[:, :1])
        active_obstacle_nodes = torch.unique(indices_to)

        obstacle_active_mask[active_obstacle_nodes] = 1
        obstacle_active_mask = obstacle_active_mask > 0
        example['obstacle'].active_mask = obstacle_active_mask


        return example

    def make_face2node(self, example, nodes_from, faces_to, mask_is_contour, nodes_enclosed_or_nenc_mask):


        verts = example['cloth'].pos
        faces = example['cloth'].faces_batch.T

        nodes_from = nodes_from[:, 0]
        faces_to = faces_to[:, 0]
        mask_is_contour = mask_is_contour[:, 0]

        icontour_grad = example['cloth'].icontour_grad
        icontour_grad_from = icontour_grad[nodes_from]
        repulsion_sign = torch.ones_like(nodes_from).float()

        nodes_to = faces[faces_to, 0]
        normals_to = self.f_normals_f(verts.unsqueeze(0), faces[faces_to].unsqueeze(0))[0]

        pos_from = verts[nodes_from]
        pos_to = verts[nodes_to]

        relative_pos = pos_from - pos_to
        dists_normal = (relative_pos * normals_to).sum(-1, keepdim=True)
        node2face = dists_normal * normals_to

        icontour_grad_proj = (icontour_grad_from * normals_to).sum(-1)
        icontour_sign = torch.sign(icontour_grad_proj)
        prev_sign = torch.sign(dists_normal)[:, 0]

        if nodes_enclosed_or_nenc_mask is None:
            mask_encompassed = torch.zeros_like(icontour_sign).bool()
        else:
            mask_encompassed = nodes_enclosed_or_nenc_mask[nodes_from]

        mask_icgrad = icontour_sign[mask_is_contour] * prev_sign[mask_is_contour]
        mask_icgrad = mask_icgrad == -1

        repulsion_sign[mask_encompassed] = -1

        node2face[dists_normal[:,0] < 0] *= -1
        n2f_distance = dists_normal.abs()
        repulsion_sign = repulsion_sign[:, None]

        return node2face, n2f_distance, repulsion_sign

    def add_world_edge_data(self, example, tensor_to_add, key, repulsion_sign, transpose=False):
        repulsion_mask = repulsion_sign == 1
        attraction_mask = repulsion_sign == -1

        tensor_repulsion = tensor_to_add[repulsion_mask]
        tensor_attraction = tensor_to_add[attraction_mask]

        if transpose:
            tensor_repulsion = tensor_repulsion.T
            tensor_attraction = tensor_attraction.T

        example['cloth', 'repulsion_edge', 'cloth'][key] = tensor_repulsion
        example['cloth', 'attraction_edge', 'cloth'][key] = tensor_attraction

        return example

    def add_world_edges(self, example): # TODO: pass fake_icontour
        cloth_pos = example['cloth'].pos
        faces_cloth = example['cloth'].faces_batch.T
        icontour_grad = example['cloth'].icontour_grad
        device = cloth_pos.device


        obstacle_pos = None
        faces_to, world_edges_curr, _, _ = get_proximity_self_pt_dummmy(cloth_pos, cloth_pos, faces_cloth, self.mcfg.collision_radius)
        nodes_from = world_edges_curr[:, :1]
        faces_to = faces_to[:, None]


        world_edges_curr_label = torch.ones_like(world_edges_curr[:, :1]).to(device).long() * common.EdgeTypeCC.NORMAL

        edges_fpin, nodes_from_fpin, faces_to_fpin = self.make_future_pinned_edges(example)
        edges_fpin_label = torch.ones_like(edges_fpin[:, :1]).to(
            device).long() * common.EdgeTypeCC.FUTUREPINNED


        world_edges_all = torch.cat([world_edges_curr, edges_fpin])
        labels_all = torch.cat([world_edges_curr_label, edges_fpin_label], dim=0)
        nodes_from_all = torch.cat([nodes_from, nodes_from_fpin], dim=0)
        faces_to_all = torch.cat([faces_to, faces_to_fpin], dim=0)

        enclosed_nodes_mask = None
        nodes_enclosed_or_nenc_mask = None
        if "enclosed_nodes_mask" in example['cloth']:
            enclosed_nodes_mask = example['cloth'].enclosed_nodes_mask
            nodes_enclosed_or_nenc_mask = example['cloth'].nodes_enclosed_or_nenc_mask[:, 0]
            node2contour = example['cloth'].node2contour
            face2contour = example['cloth'].face2contour
            icontour_grad_from = icontour_grad[nodes_from_all]

            mask_omit, mask_is_contour = make_icontour_masks(nodes_from_all, faces_to_all,
                                                             node2contour, face2contour, enclosed_nodes_mask,
                                                             icontour_grad_from)

            mask_omit = mask_omit[:, 0]
            world_edges_all = world_edges_all[~mask_omit]
            labels_all = labels_all[~mask_omit]
            nodes_from_all = nodes_from_all[~mask_omit]
            faces_to_all = faces_to_all[~mask_omit]
            mask_is_contour = mask_is_contour[~mask_omit]
        else:
            mask_is_contour = torch.zeros_like(world_edges_all[:, :1]).bool()


        if 'faces_cutout_mask_batch' in example['cloth']:
            face_mask = example['cloth'].faces_cutout_mask_batch[0]
            nodes_mask = example['cloth'].cutout_mask

            face_to_mask = face_mask[faces_to_all]
            nodes_from_mask = nodes_mask[nodes_from_all]
            mask = face_to_mask & nodes_from_mask
            mask = mask[:, 0]

            world_edges_all = world_edges_all[mask]
            labels_all = labels_all[mask]
            nodes_from_all = nodes_from_all[mask]
            faces_to_all = faces_to_all[mask]
            mask_is_contour = mask_is_contour[mask]

        node2face, n2f_distance, repulsion_sign = self.make_face2node(example, nodes_from_all, faces_to_all, mask_is_contour, nodes_enclosed_or_nenc_mask)
        repulsion_sign = repulsion_sign[:, 0]

        if self.mcfg.allrep:
            repulsion_sign = torch.ones_like(repulsion_sign)

        example = self.add_world_edge_data(example, world_edges_all, 'edge_index', repulsion_sign, transpose=True)
        example = self.add_world_edge_data(example, labels_all, 'labels', repulsion_sign)
        example = self.add_world_edge_data(example, nodes_from_all, 'nodes_from', repulsion_sign)
        example = self.add_world_edge_data(example, faces_to_all, 'faces_to', repulsion_sign)
        example = self.add_world_edge_data(example, node2face, 'node2face', repulsion_sign)
        example = self.add_world_edge_data(example, n2f_distance, 'signed_distance', repulsion_sign)


        return example

    def filter_mesh_edges(self, example, key):
        edges = example['cloth', key, 'cloth'].edge_index

        if 'cutout_mask' in example['cloth']:
            node_mask = example['cloth'].cutout_mask
            edges_mask = node_mask[edges].all(dim=0)
            edges = edges[:, edges_mask]

        example['cloth', key, 'cloth'].edge_index = edges
        return example

    def add_edges(self, sample):
        B = sample.num_graphs

        examples_updated = []
        for i in range(B):
            example = sample.get_example(i)

            example = self.add_world_edges(example)
            example = self.add_body_edges(example)

            example = self.filter_mesh_edges(example, 'mesh_edge')

            for i in range(self.mcfg.n_coarse_levels):
                key = f'coarse_edge{i}'
                example = self.filter_mesh_edges(example, key)

            examples_updated.append(example)
        sample_updated = Batch.from_data_list(examples_updated)

        return sample_updated

    def get_relative_pos(self, pos, edges):
        edges_pos = gather(pos, edges, 0, 1, 1).permute(0, 2, 1)
        pos_senders, pos_receivers = edges_pos.unbind(-1)
        relative_pos = pos_senders - pos_receivers
        return relative_pos

    def make_is_init(self, sample, lens):
        is_init = make_pervertex_tensor_from_lens(lens, sample['cloth'].is_init)
        return is_init

    def _create_mesh_edge_set(self, sample, is_training, edge_label, normalizer):
        pos = sample['cloth'].input_pos
        rest_pos = sample['cloth'].rest_pos
        edges = sample['cloth', edge_label, 'cloth'].edge_index.T

        # has_buttons = ('cloth', 'button_edge', 'cloth') in sample.edge_types
        # if has_buttons:
        #     button_edges = sample['cloth', 'button_edge', 'cloth'].edge_index.T
        #     edges = torch.cat([edges, button_edges], dim=0)


        relative_pos = self.get_relative_pos(pos, edges)
        relative_pos_norm = torch.norm(relative_pos, dim=-1, keepdim=True)

        relative_rest_pos = self.get_relative_pos(rest_pos, edges)
        relative_rest_pos_norm = torch.norm(relative_rest_pos, dim=-1, keepdim=True)

        edge_slice = sample._slice_dict['cloth', edge_label, 'cloth']['edge_index']
        lens = edge_slice[1:] - edge_slice[:-1]

        is_init = self.make_is_init(sample, lens)
        # bending_coeff = make_pervertex_tensor_from_lens(lens, sample['cloth'].bending_coeff_input)
        # lame_mu = make_pervertex_tensor_from_lens(lens, sample['cloth'].lame_mu_input)
        # lame_lambda = make_pervertex_tensor_from_lens(lens, sample['cloth'].lame_lambda_input)


        bending_coeff = sample['cloth', edge_label, 'cloth'].bending_coeff_input
        lame_mu = sample['cloth', edge_label, 'cloth'].lame_mu_input
        lame_lambda = sample['cloth', edge_label, 'cloth'].lame_lambda_input


        edge_features_to_norm = torch.cat([
            relative_pos,
            relative_pos_norm,
            relative_rest_pos_norm], dim=-1
        )

        edge_features_nonorm = torch.cat([
            is_init,
            bending_coeff,
            lame_mu,
            lame_lambda], dim=-1
        )

        edge_features_normalized = normalizer(edge_features_to_norm, is_training)
        edge_features_final = torch.cat([edge_features_normalized, edge_features_nonorm], dim=-1)

        sample = add_field_to_pyg_batch(sample, 'features', edge_features_final, ('cloth', edge_label, 'cloth'),
                                        'edge_index', zero_inc=True)

        return sample

    def _create_world_edge_set(self, sample, is_training, edge_label):

        pos = sample['cloth'].pos
        prev_pos = sample['cloth'].prev_pos
        velocity = sample['cloth'].velocity

        edges = sample['cloth', edge_label, 'cloth'].edge_index.T

        relative_pos = self.get_relative_pos(pos, edges)
        relative_pos_norm = torch.norm(relative_pos, dim=-1, keepdim=True)

        relative_prev_pos = self.get_relative_pos(prev_pos, edges)
        relative_prev_pos_norm = torch.norm(relative_prev_pos, dim=-1, keepdim=True)

        relative_pos_normalized = torch.nn.functional.normalize(relative_pos, dim=-1, p=2)

        relative_velocity = self.get_relative_pos(velocity, edges)
        relative_velocity_norm = torch.norm(relative_velocity, dim=-1, keepdim=True)

        edge_slice = sample._slice_dict['cloth', edge_label, 'cloth']['edge_index']
        lens = edge_slice[1:] - edge_slice[:-1]

        is_init = self.make_is_init(sample, lens)
        # bending_coeff = make_pervertex_tensor_from_lens(lens, sample['cloth'].bending_coeff_input)
        # lame_mu = make_pervertex_tensor_from_lens(lens, sample['cloth'].lame_mu_input)
        # lame_lambda = make_pervertex_tensor_from_lens(lens, sample['cloth'].lame_lambda_input)



        bending_coeff = sample['cloth', edge_label, 'cloth'].bending_coeff_input
        lame_mu = sample['cloth', edge_label, 'cloth'].lame_mu_input
        lame_lambda = sample['cloth', edge_label, 'cloth'].lame_lambda_input


        labels = sample['cloth', edge_label, 'cloth'].labels

        if labels.shape[0] == 0:
            embeddings = torch.zeros(labels.shape[0], 4).to(pos.device)
        else:
            embeddings = self.embed(labels, self.edgetype_embedding)
            # embeddings = self.edgetype_embedding(labels[:, 0])

        edge_features_to_norm = torch.cat([
            relative_pos,
            relative_pos_norm,
            relative_prev_pos,
            relative_prev_pos_norm,
            relative_velocity,
            relative_velocity_norm,
            embeddings], dim=-1
        )

        edge_features_nonorm = torch.cat([
            is_init,
            bending_coeff,
            lame_mu,
            lame_lambda], dim=-1
        )

        edge_features_normalized = self._world_edge_normalizer(edge_features_to_norm, is_training)

        edge_features_final = torch.cat([edge_features_normalized, edge_features_nonorm], dim=-1)
        sample = add_field_to_pyg_batch(sample, 'features', edge_features_final, ('cloth', edge_label, 'cloth'),
                                        'edge_index', zero_inc=True)

        return sample

    def _create_body_edge_set(self, sample, is_training):
        cloth_pos = sample['cloth'].pos
        cloth_velocity = sample['cloth'].velocity

        obstacle_next_pos = sample['obstacle'].target_pos
        obstacle_pos = sample['obstacle'].pos
        obstacle_velocity = sample['obstacle'].next_velocity

        edges_direct = sample['cloth', 'body_edge', 'obstacle'].edge_index
        edges_inverse = sample['obstacle', 'body_edge', 'cloth'].edge_index

        senders, receivers = edges_direct.unbind(0)
        senders_pos = cloth_pos[senders]
        receivers_next_pos = obstacle_next_pos[receivers]
        receivers_pos = obstacle_pos[receivers]

        senders_velocity = cloth_velocity[senders]
        receivers_velocity = obstacle_velocity[receivers]

        relative_next_pos = senders_pos - receivers_next_pos
        relative_pos = senders_pos - receivers_pos

        relative_velocity = senders_velocity - receivers_velocity

        relative_next_pos_norm = torch.norm(relative_next_pos, dim=-1, keepdim=True)
        relative_pos_norm = torch.norm(relative_pos, dim=-1, keepdim=True)
        relative_velocity_norm = torch.norm(relative_velocity, dim=-1, keepdim=True)

        edge_slice = sample._slice_dict['cloth', 'body_edge', 'obstacle']['edge_index']
        lens = edge_slice[1:] - edge_slice[:-1]
        is_init = self.make_is_init(sample, lens)

        features_direct = torch.cat([
            relative_pos,
            relative_pos_norm,
            relative_next_pos,
            relative_next_pos_norm,
            relative_velocity,
            relative_velocity_norm,
        ], dim=-1)

        features_inverse = torch.cat([
            -relative_pos,
            relative_pos_norm,
            -relative_next_pos,
            relative_next_pos_norm,
            -relative_velocity,
            relative_velocity_norm,
        ], dim=-1)

        normalizer = self._body_edge_normalizer

        features_combined = torch.cat([features_direct, features_inverse])
        N_direct = features_direct.shape[0]
        features_combined_normalized = normalizer(features_combined, is_training)

        features_direct_normalized = features_combined_normalized[:N_direct]
        features_inverse_normalized = features_combined_normalized[N_direct:]

        features_direct_normalized = torch.cat([features_direct_normalized, is_init], dim=-1)
        features_inverse_normalized = torch.cat([features_inverse_normalized, is_init], dim=-1)

        sample = add_field_to_pyg_batch(sample, 'features', features_direct_normalized,
                                        ('cloth', 'body_edge', 'obstacle'),
                                        'edge_index', zero_inc=True)
        sample = add_field_to_pyg_batch(sample, 'features', features_inverse_normalized,
                                        ('obstacle', 'body_edge', 'cloth'),
                                        'edge_index', zero_inc=True)

        return sample

    def _make_ts_pointcloud(self, reference_pcd, ts_tensor):
        npppc = reference_pcd.num_points_per_cloud().detach().cpu().numpy().tolist()
        ts_list = []
        for i, n in enumerate(npppc):
            ts_list.append(ts_tensor[i].repeat(n))
        ts_list = torch.cat(ts_list).unsqueeze(-1)
        ts_pcd = pcd_replace_features_packed(reference_pcd, ts_list)
        return ts_pcd

    def add_vertex_type_embedding(self, sample):
        for k in ['cloth', 'obstacle']:
            if k not in sample.node_types:
                continue
            vertex_type = sample[k].vertex_type
            vertex_type_emb = self.embed(vertex_type, self.nodetype_embedding)
            sample = add_field_to_pyg_batch(sample, 'vertex_type_embedding', vertex_type_emb, k, 'pos')
        return sample

    def add_vertex_level_embedding(self, sample):
        for k in ['cloth', 'obstacle']:
            if k not in sample.node_types:
                continue
            vertex_level = sample[k].vertex_level

            vertex_level = torch.clamp(vertex_level, 0, self.mcfg.n_coarse_levels)

            vertex_level_emb = self.embed(vertex_level, self.vertexlevel_embedding)
            sample = add_field_to_pyg_batch(sample, 'vertex_level_embedding', vertex_level_emb, k, 'pos')
        return sample
    
    def add_materials_to_nodes(self, sample):

        # print(sample)

        for k in ['cloth', 'obstacle']:
            if k not in sample.node_types:
                continue
            # print(k)
            velocity = sample[k].velocity
            device = velocity.device
            if k == 'obstacle':
                for m in ['lame_mu_input', 'lame_lambda_input', 'bending_coeff_input']:
                    mvec = torch.ones_like(velocity[:, :1]).to(device) * -1
                    sample = add_field_to_pyg_batch(sample, m, mvec, k, 'pos')
            else:
                slice = sample._slice_dict[k]['pos']
                lens = slice[1:] - slice[:-1]
                for m in ['lame_mu_input', 'lame_lambda_input', 'bending_coeff_input']:
                    mvec = sample['cloth'][m]
                    if mvec.shape[0] == 1:
                        mvec = make_pervertex_tensor_from_lens(lens, mvec)
                    else:
                        mvec = mvec[:, None]
                    sample = add_field_to_pyg_batch(sample, m, mvec, k, 'pos')

        return sample



    def add_materials_to_mesh_edges(self, sample, edge_label):
        edge_index = sample['cloth', edge_label, 'cloth'].edge_index.T

        for m in ['lame_mu_input', 'lame_lambda_input', 'bending_coeff_input']:
            mvec = sample['cloth'][m]
            mvec_edge = mvec[edge_index].mean(-2)

            sample = add_field_to_pyg_batch(sample, m, mvec_edge, ('cloth', edge_label, 'cloth'),
                                'edge_index', zero_inc=True)

        return sample


    
    def add_materials(self, sample):
        sample = self.add_materials_to_nodes(sample)

        sample = self.add_materials_to_mesh_edges(sample, 'mesh_edge')
        for i in range(self.mcfg.n_coarse_levels):
            key = f'coarse_edge{i}'
            sample = self.add_materials_to_mesh_edges(sample, key)

            
        for edge_label in ['repulsion_edge', 'attraction_edge']:
            sample = self.add_materials_to_mesh_edges(sample, edge_label)
        

        # edge_slice = sample._slice_dict['cloth', edge_label, 'cloth']['edge_index']
        # lens = edge_slice[1:] - edge_slice[:-1]

        # bending_coeff = make_pervertex_tensor_from_lens(lens, sample['cloth'].bending_coeff_input)
        # lame_mu = make_pervertex_tensor_from_lens(lens, sample['cloth'].lame_mu_input)
        # lame_lambda = make_pervertex_tensor_from_lens(lens, sample['cloth'].lame_lambda_input)

        return sample

        pass    

    def add_node_features(self, sample):
        for k in ['cloth', 'obstacle']:
            if k not in sample.node_types:
                continue

            if k == 'cloth':
                velocity = sample[k].velocity
            else:
                velocity = sample[k].next_velocity

            vertex_type_emb = sample[k].vertex_type_embedding
            vertex_level_emb = sample[k].vertex_level_embedding
            normals = sample[k].normals

            slice = sample._slice_dict[k]['pos']
            lens = slice[1:] - slice[:-1]
            is_init = self.make_is_init(sample, lens)

            if 'v_mass' in sample[k]:
                v_mass = sample[k].v_mass
            else:
                v_mass = torch.ones_like(velocity[:, :1]) * -1

            # if 'bending_coeff' in sample[k]:
            #     print('-----------add_node_features')
            #     print('bending_coeff_input', sample[k].bending_coeff_input.shape)
            #     bending_coeff = make_pervertex_tensor_from_lens(lens, sample[k].bending_coeff_input)
            #     lame_mu = make_pervertex_tensor_from_lens(lens, sample[k].lame_mu_input)
            #     lame_lambda = make_pervertex_tensor_from_lens(lens, sample[k].lame_lambda_input)
            #     print('bending_coeff', bending_coeff.shape)
            # else:
            #     device = velocity.device
            #     bending_coeff = torch.ones_like(velocity[:, :1]).to(device) * -1
            #     lame_mu = torch.ones_like(velocity[:, :1]).to(device) * -1
            #     lame_lambda = torch.ones_like(velocity[:, :1]).to(device) * -1


            bending_coeff = sample[k].bending_coeff_input
            lame_mu = sample[k].lame_mu_input
            lame_lambda = sample[k].lame_lambda_input

            node_features = torch.cat(
                [velocity, vertex_type_emb, vertex_level_emb, normals, v_mass, is_init, bending_coeff, lame_mu,
                 lame_lambda], dim=-1)

            sample = add_field_to_pyg_batch(sample, 'node_features', node_features, k, 'pos')
        return sample

    def normalize_node_features(self, sample, is_training):
        cloth_node_features = sample['cloth'].node_features
        N_cloth = cloth_node_features.shape[0]
        is_obstacle = 'obstacle' in sample.node_types

        if is_obstacle:
            obstacle_node_features = sample['obstacle'].node_features
            obstacle_active_mask = sample['obstacle'].active_mask[:, 0]
            active_obstacle_node_features = obstacle_node_features[obstacle_active_mask]
            all_features = torch.cat([cloth_node_features, active_obstacle_node_features])
        else:
            all_features = cloth_node_features

        all_features_to_norm = all_features[:, :-4]
        all_features_nonorm = all_features[:, -4:]

        all_features_normalized = self._node_normalizer(all_features_to_norm, is_training)
        all_features_normalized_final = torch.cat([all_features_normalized, all_features_nonorm], dim=-1)

        cloth_node_features_normalized = all_features_normalized_final[:N_cloth]
        sample['cloth'].node_features = cloth_node_features_normalized

        if is_obstacle:
            obstacle_node_features_normalized = all_features_normalized_final[N_cloth:]
            obstacle_node_features[obstacle_active_mask] = obstacle_node_features_normalized
            sample['obstacle'].node_features = obstacle_node_features

        return sample

    def _make_nodefeatures(self, sample):

        sample = self.add_vertex_type_embedding(sample)
        sample = self.add_vertex_level_embedding(sample)
        sample = self.normals_f(sample, 'cloth', 'pos')
        if 'obstacle' in sample.node_types:
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

        add_field_to_pyg_batch(sample, 'input_pos', pos, 'cloth', reference_key='pos')
        add_field_to_pyg_batch(sample, 'input_prev_pos', prev_pos, 'cloth', reference_key='pos')

        return sample

    def prepare_inputs(self, sample, is_training):
        """Builds input graph."""

        sample = self.replace_pinned_verts(sample)

        # construct graph edges
        sample = self.add_edges(sample)


        sample = self.add_materials(sample)

        sample = self._make_nodefeatures(sample)

        sample = self._create_mesh_edge_set(sample, is_training, 'mesh_edge', self._mesh_edge_normalizer)
        for i in range(self.mcfg.n_coarse_levels):
            key = f'coarse_edge{i}'
            sample = self._create_mesh_edge_set(sample, is_training, key, self._coarse_edge_normalizer)


        for edge_label in ['repulsion_edge', 'attraction_edge']:
            sample = self._create_world_edge_set(sample, is_training, edge_label)

        if 'obstacle' in sample.node_types:
            sample = self._create_body_edge_set(sample, is_training)

        sample = self.normalize_node_features(sample, is_training)

        return sample

    def _get_position(self, sample, is_training):
        """Integrate model outputs."""
        vertex_type = sample['cloth'].vertex_type
        pinned_mask = vertex_type == NodeType.HANDLE

        if 'cutout_mask' in sample['cloth']:
            node_mask = sample['cloth'].cutout_mask[:, None]
            pinned_mask = torch.logical_or(pinned_mask, torch.logical_not(node_mask))

        cloth_features = sample['cloth'].node_features


        acceleration = self._output_normalizer.inverse(cloth_features[:, :3])
        sample = add_field_to_pyg_batch(sample, 'pred_acceleration', acceleration, 'cloth', 'pos')


        ts = sample['cloth'].timestep[0]
        acceleration_dv = acceleration


        # integrate forward
        cur_position = sample['cloth'].pos
        prev_position = sample['cloth'].prev_pos
        target_position = sample['cloth'].target_pos
        target_velocity = target_position - cur_position

        velocity = sample['cloth'].velocity
        pred_velocity = velocity + acceleration_dv
        pred_velocity = pred_velocity * torch.logical_not(pinned_mask) + target_velocity * pinned_mask

        pred_velocity_dx = pred_velocity


        position = cur_position + pred_velocity_dx
        position = position * torch.logical_not(pinned_mask) + target_position * pinned_mask

        sample = add_field_to_pyg_batch(sample, 'pred_pos', position, 'cloth', 'pos')
        sample = add_field_to_pyg_batch(sample, 'pred_velocity', pred_velocity, 'cloth', 'pos')

        target_acceleration = target_position - 2 * cur_position + prev_position

        target_acceleration_norm = self._output_normalizer(target_acceleration, is_training)
        sample = add_field_to_pyg_batch(sample, 'target_acceleration', target_acceleration, 'cloth', 'pos')
        return sample

    def add_icontour(self, sample):
        cloth_pos = sample['cloth'].pos
        cloth_faces = sample['cloth'].faces_batch.T

        is_cutout = 'faces_cutout_mask_batch' in sample['cloth']

        if is_cutout:
            face_mask = sample['cloth'].faces_cutout_mask_batch[0]
            cloth_faces_masked = cloth_faces[face_mask]
        else:
            cloth_faces_masked = cloth_faces

        icontour_grad, ic_dict = compute_icontour_grad(cloth_pos, cloth_faces_masked, cl_weighting=True)


        add_field_to_pyg_batch(sample, 'icontour_grad', icontour_grad, 'cloth', 'pos')

        if 'enclosed_nodes_mask' in ic_dict:
            face2contour = ic_dict['face2contour']


            if is_cutout:
                face2contour_all = torch.zeros((cloth_faces.shape[0], face2contour.shape[1]), dtype=torch.bool, device=cloth_faces.device)

                face2contour_all[face_mask] = face2contour

                face2contour = face2contour_all


            add_field_to_pyg_batch(sample, 'enclosed_nodes_mask', ic_dict['enclosed_nodes_mask'], 'cloth', 'pos')
            add_field_to_pyg_batch(sample, 'nodes_enclosed_or_nenc_mask', ic_dict['nodes_enclosed_or_nenc_mask'], 'cloth', 'pos')
            add_field_to_pyg_batch(sample, 'node2contour', ic_dict['node2contour'], 'cloth', 'pos')
            add_field_to_pyg_batch(sample, 'face2contour', face2contour, 'cloth', 'faces_batch', new_dim=True)

        return sample

    def forward(self, inputs, is_training=True):        

        inputs = self.add_icontour(inputs)
        sample = self.prepare_inputs(inputs, is_training=is_training)

        sample = self._learned_model(sample)
        sample = self._get_position(sample, is_training=is_training)

        return sample
