from collections import defaultdict
from dataclasses import dataclass

import torch
from pytorch3d.ops import knn_points
from torch import nn

from utils.cloth_and_material import FaceNormals
from utils.common import gather


@dataclass
class Config:
    weight: float = 1.
    g: float = 9.81
    mu: float = 1.
    friction_radius: float = 5e-3


def create(mcfg):
    return Criterion(weight=mcfg.weight, g=mcfg.g, mu=mcfg.mu,
                     friction_radius=mcfg.friction_radius)


class Criterion(nn.Module):
    def __init__(self, weight, g=9.81, mu=1, friction_radius=3e-3):
        super().__init__()
        self.weight = weight
        self.g = g
        self.mu = mu
        self.friction_radius = friction_radius
        self.f_normals_f = FaceNormals()
        self.floor_normal = torch.tensor([0., 0., 1.])[None]
        self.eps = 1e-4

        self.name = 'friction_energy'

    def calc_cos_to_floor(self, vecs):
        self.floor_normal = self.floor_normal.to(vecs.device)
        cos_normal = torch.nn.functional.cosine_similarity(self.floor_normal, vecs, dim=-1).abs()
        cos_plane = 1. - cos_normal.pow(2)
        cos_plane = cos_plane.abs()
        cos_plane = (cos_plane + self.eps).sqrt()
        return cos_plane

    def calc_sample_dict(self, example):
        obstacle_prev_pos = example['obstacle'].prev_pos.unsqueeze(0)
        obstacle_curr_pos = example['obstacle'].pos.unsqueeze(0)
        obstacle_faces = example['obstacle'].faces_batch.T.unsqueeze(0)

        prev_pos = example['cloth'].pos.unsqueeze(0)
        curr_pos = example['cloth'].pred_pos.unsqueeze(0)

        obstacle_face_prev_pos = gather(obstacle_prev_pos, obstacle_faces, 1, 2, 2).mean(dim=2)
        obstacle_face_curr_pos = gather(obstacle_curr_pos, obstacle_faces, 1, 2, 2).mean(dim=2)

        obstacle_face_prev_normals = self.f_normals_f(obstacle_prev_pos, obstacle_faces)
        obstacle_face_curr_normals = self.f_normals_f(obstacle_curr_pos, obstacle_faces)

        nn_dist_prev, nn_idx_prev, nn_points_prev = knn_points(prev_pos, obstacle_face_prev_pos, return_nn=True)
        nn_dist_curr, nn_idx_curr, nn_points_curr = knn_points(curr_pos, obstacle_face_curr_pos, return_nn=True)

        nn_points_prev_curr = gather(obstacle_face_curr_pos, nn_idx_prev, 1, 2, 2)[:, :, 0]
        nn_points_prev = nn_points_prev[:, :, 0]

        nn_normals_prev = gather(obstacle_face_prev_normals, nn_idx_prev, 1, 2, 2)[:, :, 0]
        nn_normals_curr = gather(obstacle_face_curr_normals, nn_idx_curr, 1, 2, 2)[:, :, 0]

        out_dict = dict(nn_points_prev=nn_points_prev[0], nn_points_prev_curr=nn_points_prev_curr[0],
                        nn_normals_prev=nn_normals_prev[0], nn_normals_curr=nn_normals_curr[0],
                        nn_dist_prev=nn_dist_prev[0], nn_dist_curr=nn_dist_curr[0])

        return out_dict

    def forward(self, sample):
        B = sample.num_graphs
        curr_pos = sample['cloth'].pred_pos
        prev_pos = sample['cloth'].pos
        v_mass = sample['cloth'].v_mass[:, 0]

        bysample_datalists = defaultdict(list)

        for i in range(B):
            bysample_data = self.calc_sample_dict(sample.get_example(i))
            for k, v in bysample_data.items():
                bysample_datalists[k].append(v)

        for k, v in bysample_datalists.items():
            bysample_datalists[k] = torch.cat(v, dim=0)

        nn_points_prev = bysample_datalists['nn_points_prev']
        nn_points_prev_curr = bysample_datalists['nn_points_prev_curr']
        nn_normals_prev = bysample_datalists['nn_normals_prev']
        nn_normals_curr = bysample_datalists['nn_normals_curr']
        nn_dist_prev = bysample_datalists['nn_dist_prev']
        nn_dist_curr = bysample_datalists['nn_dist_curr']

        nn_normals_mean = (nn_normals_prev + nn_normals_curr) / 2

        distance_mask = torch.logical_and(nn_dist_prev < self.friction_radius, nn_dist_curr < self.friction_radius)[
            ..., 0]

        deltas_obstacle = nn_points_prev_curr - nn_points_prev
        deltas = curr_pos - (prev_pos + deltas_obstacle)
        normals_proj = (deltas * nn_normals_mean).sum(dim=-1, keepdims=True) * nn_normals_mean
        deltas_proj = deltas - normals_proj
        cos2floor = self.calc_cos_to_floor(deltas_proj)

        distances = torch.norm(deltas_proj, dim=-1)

        friction = v_mass * cos2floor * self.g * distances * distance_mask * self.mu
        loss = friction.sum(-1).mean() * self.weight
        loss = loss / B

        return dict(loss=loss)
