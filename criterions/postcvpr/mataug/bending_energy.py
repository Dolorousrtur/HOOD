from dataclasses import dataclass

import torch
from torch import nn

from utils.cloth_and_material import FaceNormals
from utils.common import gather


@dataclass
class Config:
    weight: float = 1.
    return_average: bool = True


def create(mcfg):
    return Criterion(weight=mcfg.weight, return_average=mcfg.return_average)


class Criterion(nn.Module):
    def __init__(self, weight, return_average=True):
        super().__init__()
        self.weight = weight
        self.return_average = return_average
        self.face_normals_f = FaceNormals()
        self.name = 'bending_energy'

    def calc_single(self, example):
        pred_pos = example['cloth'].pred_pos
        faces = example['cloth'].faces_batch.T
        f_connectivity = example['cloth'].f_connectivity
        f_connectivity_edges = example['cloth'].f_connectivity_edges
        f_area = example['cloth'].f_area
        bending_coeff = example['cloth'].bending_coeff

        fn = self.face_normals_f(pred_pos.unsqueeze(0), faces.unsqueeze(0))[0]

        n = gather(fn, f_connectivity, 0, 1, 1)
        n0, n1 = torch.unbind(n, dim=-2)

        v = gather(pred_pos, f_connectivity_edges, 0, 1, 1)
        v0, v1 = torch.unbind(v, dim=-2)
        e = v1 - v0
        l = torch.norm(e, dim=-1, keepdim=True)
        e_norm = e / l

        f_area_repeat = f_area.repeat(1, f_connectivity.shape[-1])
        a = torch.gather(f_area_repeat, 0, f_connectivity).sum(dim=-1)

        cos = (n0 * n1).sum(dim=-1)
        sin = (e_norm * torch.linalg.cross(n0, n1)).sum(dim=-1)
        theta = torch.atan2(sin, cos)

        scale = l[..., 0] ** 2 / (4 * a)

        energy = bending_coeff * scale * (theta ** 2) / 2
        loss = energy.sum()
        return loss

    def forward(self, sample):
        loss_list = []
        B = sample.num_graphs
        for i in range(B):
            loss_sample = self.calc_single(sample.get_example(i))
            loss_list.append(loss_sample)

        loss = sum(loss_list) / B

        return dict(loss=loss)
