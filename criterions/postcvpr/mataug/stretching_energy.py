from dataclasses import dataclass

import einops
import torch
from torch import nn

from utils.cloth_and_material import gather_triangles, get_shape_matrix
from utils.common import make_pervertex_tensor_from_lens


@dataclass
class Config:
    weight: float = 1.
    thickness: float = 4.7e-4


def create(mcfg):
    return Criterion(weight=mcfg.weight, thickness=mcfg.thickness)


def deformation_gradient(triangles, Dm_inv):
    Ds = get_shape_matrix(triangles)

    return Ds @ Dm_inv


def green_strain_tensor(F):
    device = F.device
    I = torch.eye(2, dtype=F.dtype).to(device)

    Ft = F.permute(0, 2, 1)
    return 0.5 * (Ft @ F - I)


class Criterion(nn.Module):
    def __init__(self, weight, thickness):
        super().__init__()
        self.weight = weight
        self.thickness = thickness
        self.name = 'stretching_energy'

    def create_stack(self, triangles_list, param):
        lens = [x.shape[0] for x in triangles_list]
        stack = make_pervertex_tensor_from_lens(lens, param)[:, 0]
        return stack

    def forward(self, sample):
        Dm_inv = sample['cloth'].Dm_inv

        f_area = sample['cloth'].f_area[None, ..., 0]
        device = Dm_inv.device

        B = sample.num_graphs

        triangles_list = []
        for i in range(B):
            example = sample.get_example(i)
            v = example['cloth'].pred_pos
            f = example['cloth'].faces_batch.T

            triangles = gather_triangles(v.unsqueeze(0), f)[0]
            triangles_list.append(triangles)

        lame_mu_stack = self.create_stack(triangles_list, sample['cloth'].lame_mu)
        lame_lambda_stack = self.create_stack(triangles_list, sample['cloth'].lame_lambda)
        triangles = torch.cat(triangles_list, dim=0)

        F = deformation_gradient(triangles, Dm_inv)
        G = green_strain_tensor(F)

        I = torch.eye(2).to(device)
        I = einops.repeat(I, 'm n -> k m n', k=G.shape[0])

        G_trace = G.diagonal(dim1=-1, dim2=-2).sum(-1)  # trace

        S = lame_mu_stack[:, None, None] * G + 0.5 * lame_lambda_stack[:, None, None] * G_trace[:, None, None] * I
        energy_density_matrix = S.permute(0, 2, 1) @ G
        energy_density = energy_density_matrix.diagonal(dim1=-1, dim2=-2).sum(-1)  # trace
        f_area = f_area[0]

        energy = f_area * self.thickness * energy_density
        loss = energy.sum() / B

        return dict(loss=loss)
