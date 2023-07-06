from dataclasses import dataclass

import torch
from pytorch3d.ops import knn_points
from torch import nn

from utils.cloth_and_material import VertexNormals, FaceNormals
from utils.common import gather


@dataclass
class Config:
    weight: float = 1.
    start_rampup_iteration: int = 50000
    n_rampup_iterations: int = 100000
    eps: float = 1e-3
    exp_min: float = 1.


def create(mcfg):
    return Criterion(weight=mcfg.weight,
                     start_rampup_iteration=mcfg.start_rampup_iteration, n_rampup_iterations=mcfg.n_rampup_iterations,
                     eps=mcfg.eps, exp_min=mcfg.exp_min)


class Criterion(nn.Module):
    def __init__(self, weight, start_rampup_iteration, n_rampup_iterations, eps=1e-3, exp_min=1.):
        super().__init__()
        self.weight = weight
        self.eps = eps
        self.start_rampup_iteration = start_rampup_iteration
        self.n_rampup_iterations = n_rampup_iterations
        self.v_normals_f = VertexNormals()
        self.f_normals_f = FaceNormals()
        self.name = 'collision_penalty'
        self.exp_min = exp_min

    def get_pow(self, iter_num):
        iter_num = iter_num - self.start_rampup_iteration
        iter_num = max(iter_num, 0)
        progress = iter_num / self.n_rampup_iterations
        progress = min(progress, 1.)
        pow = 3 - 2 * progress

        pow = max(pow, self.exp_min)
        return pow

    def calc_loss_prev_obstacle(self, example):
        obstacle_pos = example['obstacle'].pos
        obstacle_faces = example['obstacle'].faces_batch.T
        pred_pos = example['cloth'].pred_pos

        obstacle_face_pos = gather(obstacle_pos, obstacle_faces, 0, 1, 1).mean(dim=-2)

        obstacle_fn = self.f_normals_f(obstacle_pos.unsqueeze(0), obstacle_faces.unsqueeze(0))[0]

        _, nn_idx, nn_points_prev = knn_points(pred_pos.unsqueeze(0), obstacle_face_pos.unsqueeze(0), return_nn=True)
        nn_idx = nn_idx[0]

        nn_points = gather(obstacle_face_pos, nn_idx, 0, 1, 1)
        nn_normals = gather(obstacle_fn, nn_idx, 0, 1, 1)

        nn_points = nn_points[:, 0]
        nn_normals = nn_normals[:, 0]
        device = pred_pos.device

        distance = ((pred_pos - nn_points) * nn_normals).sum(dim=-1)
        interpenetration = torch.maximum(self.eps - distance, torch.FloatTensor([0]).to(device))

        perc = (interpenetration > 0).float().mean()

        interpenetration = interpenetration.pow(3)
        loss = interpenetration.sum(-1)

        return loss, perc

    def forward(self, sample):
        B = sample.num_graphs

        loss_list = []
        perc_list = []
        for i in range(B):
            loss, perc = self.calc_loss_prev_obstacle(sample.get_example(i))
            loss_list.append(loss)
            perc_list.append(perc)

        loss = sum(loss_list) / B * self.weight
        perc = sum(perc_list) / B

        return dict(loss=loss, perc=perc)
