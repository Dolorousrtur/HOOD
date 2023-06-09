import torch

from utils.common import add_field_to_pyg_batch


class SampleCollector:
    """
    Helper class to collect data from a sequence for a particular rollout step
    """
    def __init__(self, mcfg, obstacle=True):
        self.mcfg = mcfg
        self.obstacle = obstacle
        self.objects = ['cloth', 'obstacle'] if self.obstacle else ['cloth']

    def copy_from_prev(self, sample, prev_sample):
        if prev_sample is None:
            return sample
        sample['cloth'].prev_pos = prev_sample['cloth'].pos.detach()
        sample['cloth'].pos = prev_sample['cloth'].pred_pos.detach()

        if self.obstacle:
            sample['obstacle'].prev_pos = prev_sample['obstacle'].pos
            sample['obstacle'].pos = prev_sample['obstacle'].target_pos

        return sample

    @staticmethod
    def make_ts_tensor(ts, B, device):
        ts_tensor = torch.ones(B, 1).to(device)
        ts_tensor = ts_tensor * ts
        return ts_tensor

    @staticmethod
    def update_sample_with_timestep(sample, timestep_tensor):
        add_field_to_pyg_batch(sample, 'timestep', timestep_tensor, 'cloth', reference_key=None,
                               one_per_sample=True)
        return sample

    def add_velocity(self, sample, prev_sample):
        if prev_sample is not None:
            velocity_cloth = prev_sample['cloth'].pred_velocity
        else:
            velocity_cloth = sample['cloth'].pos - sample['cloth'].prev_pos
        add_field_to_pyg_batch(sample, 'velocity', velocity_cloth, 'cloth', 'pos')

        if self.obstacle:
            velocity_obstacle_curr = sample['obstacle'].pos - sample['obstacle'].prev_pos
            velocity_obstacle_next = sample['obstacle'].target_pos - sample['obstacle'].pos
            add_field_to_pyg_batch(sample, 'velocity', velocity_obstacle_curr, 'obstacle', 'pos')
            add_field_to_pyg_batch(sample, 'next_velocity', velocity_obstacle_next, 'obstacle', 'pos')
        return sample

    def lookup2target(self, sample, idx):
        for obj in self.objects:
            sample[obj].target_pos = sample[obj].lookup[:, idx]
        return sample

    def pos2prev(self, sample):
        for obj in self.objects:
            sample[obj].prev_pos = sample[obj].pos
        return sample

    def pos2target(self, sample):
        for obj in self.objects:
            sample[obj].target_pos = sample[obj].pos
        return sample

    def target2pos(self, sample):
        for obj in self.objects:
            sample[obj].pos = sample[obj].target_pos
        return sample

    def add_timestep(self, sample, ts):
        B = sample.num_graphs
        device = sample['cloth'].pos.device
        timestep = self.make_ts_tensor(ts, B, device)
        sample = self.update_sample_with_timestep(sample, timestep)
        return sample

    def sequence2sample(self, sample, idx):
        for obj in self.objects:
            for k in ['pos', 'prev_pos', 'target_pos']:
                v = sample[obj][k]
                v = v[:, idx]
                sample[obj][k] = v
        return sample
