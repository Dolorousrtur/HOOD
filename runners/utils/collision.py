import torch
from pytorch3d.ops import knn_points
from torch_geometric.data import Batch

from utils.cloth_and_material import FaceNormals
from utils.common import gather


class CollisionPreprocessor:
    """
    Resolves garment-body collisions.
    """

    def __init__(self, mcfg):
        self.mcfg = mcfg
        self.normals_f = FaceNormals()

    def calc_direction(self, cloth_pos, obstacle_pos, obstacle_faces):
        pred_pos = cloth_pos
        device = pred_pos.device

        obstacle_face_pos = gather(obstacle_pos, obstacle_faces, 1, 2, 2).mean(dim=2)

        obstacle_fn = self.normals_f(obstacle_pos, obstacle_faces)
        nn_distances, nn_idx, nn_points = knn_points(pred_pos, obstacle_face_pos, return_nn=True)
        nn_points = nn_points[:, :, 0]

        nn_normals = gather(obstacle_fn, nn_idx, 1, 2, 2)
        nn_normals = nn_normals[:, :, 0]

        direction = pred_pos - nn_points
        distance = (direction * nn_normals).sum(dim=-1)
        interpenetration = torch.minimum(distance - self.mcfg.push_eps, torch.FloatTensor([0]).to(device))

        distance_mask = distance < 0
        direction_upd = distance[..., None] * nn_normals
        direction_upd *= distance_mask[..., None]

        direction_upd = interpenetration[..., None] * nn_normals

        return direction_upd

    def solve(self, sample):
        B = sample.num_graphs
        new_example_list = []
        for i in range(B):
            example = sample.get_example(i)
            cloth_pos = example['cloth'].pos.unsqueeze(0)
            obstacle_pos = example['obstacle'].pos.unsqueeze(0)
            obstacle_faces = example['obstacle'].faces_batch.T.unsqueeze(0)
            cloth_prev_pos = example['cloth'].prev_pos.unsqueeze(0)

            pos_shift = self.calc_direction(cloth_pos, obstacle_pos, obstacle_faces)
            prev_pos_shift = self.calc_direction(cloth_prev_pos, obstacle_pos, obstacle_faces)

            new_pos = cloth_pos - pos_shift
            new_prev_pos = cloth_prev_pos - prev_pos_shift

            example['cloth'].pos = new_pos[0]
            example['cloth'].prev_pos = new_prev_pos[0]
            new_example_list.append(example)

        sample = Batch.from_data_list(new_example_list)
        return sample
