import torch

from utils.common import add_field_to_pyg_batch
from utils.io import pickle_load


class SampleCollector:
    def __init__(self, mcfg, obstacle=True, changing_rest=False, body_pred=False, cloth_pred=True):
        self.mcfg = mcfg
        self.obstacle = obstacle
        self.changing_rest = changing_rest
        self.objects = ['cloth', 'obstacle'] if self.obstacle else ['cloth']
        self.body_pred = body_pred
        self.cloth_pred= cloth_pred

    def copy_from_prev(self, sample, prev_sample):
        if prev_sample is None:
            if self.body_pred:
                dn = torch.zeros_like(sample['obstacle'].pos[:, :1])
                add_field_to_pyg_batch(sample, 'dn', dn, 'obstacle', 'pos')
            else:
                'NO BODY PREV'
            return sample

        sample['cloth'].prev_pos = prev_sample['cloth'].pos.detach()

        if self.cloth_pred:
            sample['cloth'].pos = prev_sample['cloth'].pred_pos.detach()
        else:
            sample['cloth'].pos = prev_sample['cloth'].target_pos

        if self.obstacle and 'obstacle' in sample.node_types:
            sample['obstacle'].prev_pos = prev_sample['obstacle'].pos

            if not self.body_pred:
                sample['obstacle'].pos = prev_sample['obstacle'].target_pos
            else:
                sample['obstacle'].dn = prev_sample['obstacle'].pred_dn.detach()
                sample['obstacle'].pos = prev_sample['obstacle'].pred_pos.detach()


            if 'curr_pn' in sample['obstacle']:
                sample['obstacle'].prev_pn = prev_sample['obstacle'].curr_pn
                sample['obstacle'].pn = prev_sample['obstacle'].target_pn

        return sample

    @staticmethod
    def make_ts_tensor(ts, B, device):
        ts_tensor = torch.ones(B, 1).to(device)
        ts_tensor = ts_tensor * ts
        return ts_tensor

    @staticmethod
    def update_sample_with_timestep(sample, timestep_tensor, key='timestep'):
        add_field_to_pyg_batch(sample, key, timestep_tensor, 'cloth', reference_key=None,
                               one_per_sample=True)
        return sample

    def add_velocity(self, sample, prev_sample):
        ts = sample['cloth'].timestep[0]
        if prev_sample is not None and self.cloth_pred:
            velocity_cloth = prev_sample['cloth'].pred_velocity
        else:
            velocity_cloth = sample['cloth'].pos - sample['cloth'].prev_pos

        add_field_to_pyg_batch(sample, 'velocity', velocity_cloth, 'cloth', 'pos')

        if self.obstacle and 'obstacle' in sample.node_types:


            velocity_obstacle_curr = sample['obstacle'].pos - sample['obstacle'].prev_pos
            velocity_obstacle_next = sample['obstacle'].target_pos - sample['obstacle'].pos

            add_field_to_pyg_batch(sample, 'velocity', velocity_obstacle_curr, 'obstacle', 'pos')
            add_field_to_pyg_batch(sample, 'next_velocity', velocity_obstacle_next, 'obstacle', 'pos')

        return sample

    def lookup2target(self, sample, idx):
        for obj in self.objects:
            if obj not in sample.node_types:
                continue
            sample[obj].target_pos = sample[obj].lookup[:, idx]
        return sample

    def pos2prev(self, sample):
        for obj in self.objects:
            if obj not in sample.node_types:
                continue
            sample[obj].prev_pos = sample[obj].pos
        return sample

    def pos2target(self, sample):
        for obj in self.objects:
            if obj not in sample.node_types:
                continue
            sample[obj].target_pos = sample[obj].pos
        return sample

    def target2pos(self, sample):
        for obj in self.objects:
            if obj not in sample.node_types:
                continue
            sample[obj].pos = sample[obj].target_pos
        return sample

    def add_timestep(self, sample, ts):
        B = sample.num_graphs
        device = sample['cloth'].pos.device
        timestep = self.make_ts_tensor(ts, B, device)
        sample = self.update_sample_with_timestep(sample, timestep)
        return sample

    def add_is_init(self, sample, is_init):
        is_init_val = 1 if is_init else 0

        B = sample.num_graphs
        device = sample['cloth'].pos.device
        timestep = self.make_ts_tensor(is_init_val, B, device)
        sample = self.update_sample_with_timestep(sample, timestep, key='is_init')
        return sample

    def sequence2sample(self, sample, idx, only_target=False):
        to_copy_list = ['target_pos'] if only_target else ['pos', 'prev_pos', 'target_pos']
        if self.changing_rest:
            to_copy_list.append('rest_pos')

        for obj in self.objects:
            if obj not in sample.node_types:
                continue
            for k in to_copy_list:
                if k not in sample[obj]:
                    continue
                v = sample[obj][k]
                v = v[:, idx]
                sample[obj][k] = v


        if 'obstacle' in sample.node_types and 'curr_pn' in sample['obstacle']:
            for k in ['curr_pn', 'prev_pn', 'target_pn']:
                sample['obstacle'][k] = sample['obstacle'][k][:, idx]
        return sample




class SampleCollectorFromFile(SampleCollector):
    def __init__(self, mcfg, obstacle=True, changing_rest=False, body_pred=False, cloth_pred=True,
                 file_path=None, ind=0):
        super().__init__(mcfg, obstacle, changing_rest, body_pred, cloth_pred)

        self.traj_dict = pickle_load(file_path)
        self.traj_dict['cloth'] = torch.FloatTensor(self.traj_dict['pred']).to(self.mcfg.device)

        if 'obstacle' in self.traj_dict:
            self.traj_dict['obstacle'] = torch.FloatTensor(self.traj_dict['obstacle']).to(self.mcfg.device)

        self.start_ind = ind


    def add_velocity(self, sample, prev_sample):
        ts = sample['cloth'].timestep[0]
        if prev_sample is not None and self.cloth_pred:
            velocity_cloth = prev_sample['cloth'].pred_velocity
        else:
            velocity_cloth = sample['cloth'].pos - sample['cloth'].prev_pos

        add_field_to_pyg_batch(sample, 'velocity', velocity_cloth, 'cloth', 'pos')

        if self.obstacle and 'obstacle' in sample.node_types:
            velocity_obstacle_curr = sample['obstacle'].pos - sample['obstacle'].prev_pos
            velocity_obstacle_next = sample['obstacle'].target_pos - sample['obstacle'].pos

            if self.mcfg.ts_agnostic:
                velocity_obstacle_curr = velocity_obstacle_curr / ts
                velocity_obstacle_next = velocity_obstacle_next / ts
            add_field_to_pyg_batch(sample, 'velocity', velocity_obstacle_curr, 'obstacle', 'pos')
            add_field_to_pyg_batch(sample, 'next_velocity', velocity_obstacle_next, 'obstacle', 'pos')

        return sample

    def sequence2sample(self, sample, idx, only_target=False):
        for obj in self.objects:
            if obj not in sample.node_types:
                continue

            sample[obj].target_pos = self.traj_dict[obj][idx + self.start_ind + 1]
            if not only_target:
                sample[obj].pos = self.traj_dict[obj][idx + self.start_ind]
                sample[obj].prev_pos = self.traj_dict[obj][idx + self.start_ind - 1]


        return sample