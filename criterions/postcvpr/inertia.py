from dataclasses import dataclass

from torch import nn


@dataclass
class Config:
    weight: float = 1.


def create(mcfg):
    return Criterion(weight=mcfg.weight)


class Criterion(nn.Module):
    def __init__(self, weight):
        super().__init__()
        self.weight = weight
        self.name = 'inertia'

    def forward(self, sample):
        cloth_sample = sample['cloth']
        timestep = cloth_sample.timestep

        pred_pos = cloth_sample.pred_pos
        pos = cloth_sample.pos
        velocity = cloth_sample.velocity

        mass = cloth_sample.v_mass

        B = sample.num_graphs

        x_hat = pos + velocity
        x_diff = pred_pos - x_hat
        num = (x_diff * mass * x_diff).sum(dim=-1).unsqueeze(1)
        den = 2 * timestep[None, ..., 0] ** 2
        loss = num / den

        loss = loss.sum() / B

        # print('loss', loss)
        return dict(loss=loss)
