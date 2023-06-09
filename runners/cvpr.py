import os
import time
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, Optional

import torch
from omegaconf import II
from omegaconf.dictconfig import DictConfig
from pytorch3d.ops import knn_points
from torch import nn
from torch.optim import Adam
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader
from torch_geometric.data import Batch
from tqdm import tqdm

from utils.cloth_and_material import FaceNormals, ClothMatAug, Material
from utils.common import move2device, gather, save_checkpoint, add_field_to_pyg_batch, \
    random_between, relative_between, relative_between_log, random_between_log
from utils.defaults import DEFAULTS
from huepy import yellow


@dataclass
class MaterialConfig:
    density_min: float = 0.20022
    density_max: float = 0.20022
    thickness_min: float = 4.7e-4
    thickness_max: float = 4.7e-4
    lame_mu_min: float = 23600.0
    lame_mu_max: float = 23600.0
    lame_lambda_min: float = 44400
    lame_lambda_max: float = 44400
    bending_coeff_min: float = 3.9625778333333325e-05
    bending_coeff_max: float = 3.9625778333333325e-05
    bending_multiplier: float = 1.

    density_override: Optional[float] = None
    lame_mu_override: Optional[float] = None
    lame_lambda_override: Optional[float] = None
    bending_coeff_override: Optional[float] = None


@dataclass
class OptimConfig:
    lr: float = 1e-4
    decay_rate: float = 1e-1
    decay_min: float = 0
    decay_steps: int = 5_000_000
    step_start: int = 0


@dataclass
class Config:
    optimizer: OptimConfig = OptimConfig()
    material: MaterialConfig = MaterialConfig()
    warmup_steps: int = 100
    n_opt_steps: int = 1
    increase_roll_every: int = -1
    roll_max: int = 1
    push_eps: float = 0.
    grad_clip: Optional[float] = 1.
    overwrite_pos_every_step: bool = False

    initial_ts: float = 1 / 3
    regular_ts: float = 1 / 3e2
    device: str = II('device')


class Runner(nn.Module):
    def __init__(self, model: nn.Module, criterion_dict: Dict[str, nn.Module], mcfg: DictConfig):
        super().__init__()

        self.model = model
        self.criterion_dict = criterion_dict
        self.mcfg = mcfg

        self.cloth_obj = ClothMatAug(None, always_overwrite_mass=True)
        self.normals_f = FaceNormals()

    def valid_rollout(self, sequence, n_steps=-1, bare=False, record_time=False):

        cloth_faces = sequence['cloth'].faces_batch.T.cpu().numpy()
        obstacle_faces = sequence['obstacle'].faces_batch.T.cpu().numpy()

        n_samples = sequence['obstacle'].pos.shape[1]

        sequence = add_field_to_pyg_batch(sequence, 'iter', [0], 'cloth', reference_key=None)

        sequence = self._add_cloth_obj(sequence)
        sequence['cloth']['cloth_obj'] = self.cloth_obj

        if n_steps > 0:
            n_samples = min(n_samples, n_steps)

        trajectories_dicts = defaultdict(lambda: defaultdict(list))

        if record_time:
            st_time = time.time()

        st = 0
        trajectory, gt_trajectory, obstacle_trajectory, metrics_dict = self._rollout(sequence, st, n_samples - st,
                                                                                     progressbar=True, bare=bare)
        trajectories_dicts['pred'] = trajectory
        trajectories_dicts['obstacle'] = obstacle_trajectory
        trajectories_dicts['metrics'] = dict(metrics_dict)
        trajectories_dicts['cloth_faces'] = cloth_faces
        trajectories_dicts['obstacle_faces'] = obstacle_faces

        for s in ['pred', 'obstacle']:
            trajectories_dicts[s] = torch.stack(trajectories_dicts[s], dim=0).cpu().numpy()

        if record_time:
            total_time = time.time() - st_time
            trajectories_dicts['metrics']['time'] = total_time

        return trajectories_dicts

    def _rollout(self, sequence, start_idx, n_steps, progressbar=False, bare=False) -> (
            list, list, list, dict):
        trajectory = []
        gt_trajectory = []
        obstacle_trajectory = []

        metrics_dict = defaultdict(list)

        pbar = range(start_idx, start_idx + n_steps)
        if progressbar:
            pbar = tqdm(pbar)

        cloth_state_next = None
        for i in pbar:
            state = self.collect_sample_wholeseq(sequence, i, i - start_idx, cloth_state_next)

            if i == start_idx:
                state = self._remove_collisions(state)
            with torch.no_grad():
                state = self.model(state, is_training=False)
            next_pos = state['cloth'].pred_pos
            trajectory.append(next_pos)
            gt_trajectory.append(state['cloth'].target_pos)
            obstacle_trajectory.append(state['obstacle'].pos)

            if not bare:
                state.cloth_obj = sequence['cloth'].cloth_obj

                loss_dict = self._criterion_pass(state)
                for k, v in loss_dict.items():
                    metrics_dict[k].append(v.item())

            cloth_state_next = state.clone()
            cloth_state_next['cloth'].prev_pos = state['cloth'].pos
            cloth_state_next['cloth'].pos = state['cloth'].pred_pos


        return trajectory, gt_trajectory, obstacle_trajectory, metrics_dict

    def collect_sample_wholeseq(self, sequence, index, relative_index, cloth_state_next):
        state = sequence.clone()

        for obj in ['cloth', 'obstacle']:

            for k, v in sequence[obj].items():

                if k in ['pos', 'prev_pos', 'target_pos']:
                    v = v[:, index]
                    state[obj][k] = v
                else:
                    state[obj][k] = v

        cloth_data = state['cloth']
        B = sequence.num_graphs
        device = cloth_data.pos.device

        if cloth_state_next is not None:
            state['cloth'].pos = cloth_state_next['cloth'].pos
            state['cloth'].prev_pos = cloth_state_next['cloth'].prev_pos

        ts = self.mcfg.regular_ts
        if relative_index == 0:
            state['cloth'].prev_pos = sequence['cloth'].pos[:, index]
            ts = self.mcfg.initial_ts
        elif relative_index == 1:
            state['cloth'].prev_pos = cloth_state_next['cloth'].pos

        timestep = self.make_ts_tensor(ts, B, device)

        state = self.update_sample_with_timestep(state, timestep)
        state['cloth'].cloth_obj = sequence['cloth'].cloth_obj
        return state

    def set_random_material(self, sample):

        B = sample.num_graphs
        device = sample['cloth'].pos.device
        size = [B]

        if self.mcfg.material.density_override is None:
            density = random_between(self.mcfg.material.density_min, self.mcfg.material.density_max, shape=size).to(
                device)
        else:
            density = torch.ones(B).to(device) * self.mcfg.material.density_override

        if self.mcfg.material.lame_mu_override is None:
            lame_mu, lame_mu_input = random_between_log(self.mcfg.material.lame_mu_min, self.mcfg.material.lame_mu_max,
                                                        shape=size, return_norm=True, device=device)
        else:
            lame_mu = torch.ones(B).to(device) * self.mcfg.material.lame_mu_override
            lame_mu_input = relative_between_log(self.mcfg.material.lame_mu_min, self.mcfg.material.lame_mu_max,
                                                 lame_mu)

        if self.mcfg.material.lame_lambda_override is None:
            lame_lambda, lame_lambda_input = random_between(self.mcfg.material.lame_lambda_min,
                                                            self.mcfg.material.lame_lambda_max,
                                                            shape=size, return_norm=True, device=device)
        else:
            lame_lambda = torch.ones(B).to(device) * self.mcfg.material.lame_lambda_override
            lame_lambda_input = relative_between(self.mcfg.material.lame_lambda_min, self.mcfg.material.lame_lambda_max,
                                                 lame_lambda)

        if self.mcfg.material.bending_coeff_override is None:
            bending_coeff, bending_coeff_input = random_between_log(self.mcfg.material.bending_coeff_min,
                                                                    self.mcfg.material.bending_coeff_max,
                                                                    shape=size, return_norm=True, device=device)
        else:
            bending_coeff = torch.ones(B).to(device) * self.mcfg.material.bending_coeff_override
            bending_coeff_input = relative_between_log(self.mcfg.material.bending_coeff_min,
                                                       self.mcfg.material.bending_coeff_max, bending_coeff)

        bending_multiplier = self.mcfg.material.bending_multiplier
        material = Material(density, lame_mu, lame_lambda,
                            bending_coeff, bending_multiplier)

        add_field_to_pyg_batch(sample, 'lame_mu', lame_mu, 'cloth', reference_key=None, one_per_sample=True)
        add_field_to_pyg_batch(sample, 'lame_lambda', lame_lambda, 'cloth', reference_key=None,
                               one_per_sample=True)
        add_field_to_pyg_batch(sample, 'bending_coeff', bending_coeff, 'cloth', reference_key=None,
                               one_per_sample=True)
        add_field_to_pyg_batch(sample, 'lame_mu_input', lame_mu_input, 'cloth', reference_key=None, one_per_sample=True)
        add_field_to_pyg_batch(sample, 'lame_lambda_input', lame_lambda_input, 'cloth', reference_key=None,
                               one_per_sample=True)
        add_field_to_pyg_batch(sample, 'bending_coeff_input', bending_coeff_input, 'cloth', reference_key=None,
                               one_per_sample=True)

        self.cloth_obj.set_material(material)

    def _add_cloth_obj(self, sample):
        self.set_random_material(sample)
        sample = self.cloth_obj.set_batch(sample, overwrite_pos=self.mcfg.overwrite_pos_every_step)
        return sample

    def _criterion_pass(self, criterion_in: dict) -> dict:
        loss_dict = dict()
        for criterion_name, criterion in self.criterion_dict.items():
            ld = criterion(criterion_in)
            for k, v in ld.items():
                loss_dict[f"{criterion_name}_{k}"] = v

        return loss_dict

    def make_ts_tensor(self, ts, B, device):
        ts_tensor = torch.ones(B, 1).to(device)
        ts_tensor = ts_tensor * ts
        return ts_tensor

    def make_random_ts_tensor(self, B, device):
        rand = torch.randint(2, size=(B, 1)).to(device)
        ts_tensor = self.mcfg.initial_ts * rand + self.mcfg.regular_ts * (1. - rand)
        return ts_tensor

    def update_sample_with_timestep(self, sample, timestep_tensor):

        add_field_to_pyg_batch(sample, 'timestep', timestep_tensor, 'cloth', reference_key=None,
                               one_per_sample=True)
        return sample

    def collect_sample(self, sample, idx, prev_out_dict=None, random_ts=False):
        sample_step = sample.clone()

        cloth_data = sample_step['cloth']
        B = sample.num_graphs
        device = cloth_data.pos.device

        if idx == 0:
            sample_step['cloth'].prev_pos = sample_step['cloth'].pos
            sample_step['cloth']['target_pos'] = sample_step['cloth'].lookup[:, 0]

            sample_step['obstacle'].target_pos = sample_step['obstacle'].lookup[:, 0]

            if random_ts:
                timestep = self.make_random_ts_tensor(B, device)
            else:
                timestep = self.make_ts_tensor(self.mcfg.initial_ts, B, device)
        elif idx == 1:
            sample_step['cloth'].prev_pos = prev_out_dict['cloth'].pred_pos.detach()
            sample_step['cloth'].pos = prev_out_dict['cloth'].pred_pos.detach()
            timestep = self.make_ts_tensor(self.mcfg.regular_ts, B, device)
            sample_step['cloth']['target_pos'] = sample_step['cloth'].lookup[:, 1]

            sample_step['obstacle']['pos'] = sample_step['obstacle'].lookup[:, 0]
            sample_step['obstacle']['target_pos'] = sample_step['obstacle'].lookup[:, 1]
        else:
            sample_step['cloth'].prev_pos = prev_out_dict['cloth'].pos
            sample_step['cloth'].pos = prev_out_dict['cloth'].pred_pos.detach()
            timestep = self.make_ts_tensor(self.mcfg.regular_ts, B, device)
            sample_step['cloth'].target_pos = sample_step['cloth'].lookup[:, idx]

            sample_step['obstacle'].pos = sample_step['obstacle'].lookup[:, idx - 1]
            sample_step['obstacle'].target_pos = sample_step['obstacle'].lookup[:, idx]

        sample_step = self.update_sample_with_timestep(sample_step, timestep)
        sample_step['cloth'].cloth_obj = sample['cloth'].cloth_obj
        return sample_step

    def _calc_collision_direction(self, cloth_pos, obstacle_pos, obstacle_faces):
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

    def _remove_collisions(self, sample):
        B = sample.num_graphs
        new_example_list = []
        for i in range(B):
            example = sample.get_example(i)
            cloth_pos = example['cloth'].pos.unsqueeze(0)
            obstacle_pos = example['obstacle'].pos.unsqueeze(0)
            obstacle_faces = example['obstacle'].faces_batch.T.unsqueeze(0)
            cloth_prev_pos = example['cloth'].prev_pos.unsqueeze(0)

            pos_shift = self._calc_collision_direction(cloth_pos, obstacle_pos, obstacle_faces)
            prev_pos_shift = self._calc_collision_direction(cloth_prev_pos, obstacle_pos, obstacle_faces)

            new_pos = cloth_pos - pos_shift
            new_prev_pos = cloth_prev_pos - prev_pos_shift

            example['cloth'].pos = new_pos[0]
            example['cloth'].prev_pos = new_prev_pos[0]
            new_example_list.append(example)

        sample = Batch.from_data_list(new_example_list)
        return sample

    def forward(self, sample: dict, roll_steps=1, optimizer=None, scheduler=None) -> dict:
        random_ts = (roll_steps == 1)
        sample = self._add_cloth_obj(sample)
        sample['cloth']['cloth_obj'] = self.cloth_obj

        prev_out_sample = None
        for i in range(roll_steps):
            sample_step = self.collect_sample(sample, i, prev_out_sample, random_ts=random_ts)

            if i == 0:
                sample_step = self._remove_collisions(sample_step)

            sample_step = self.model(sample_step)
            sample_step.cloth_obj = self.cloth_obj
            loss_dict = self._criterion_pass(sample_step)
            prev_out_sample = sample_step.detach()

            if optimizer is not None:
                loss = sum(loss_dict.values())
                optimizer.zero_grad()
                loss.backward()
                if self.mcfg.grad_clip is not None:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.mcfg.grad_clip)

                optimizer.step()
                if scheduler is not None:
                    scheduler.step()

        ld_to_write = {k: v.item() for k, v in loss_dict.items()}
        return ld_to_write


def create_optimizer(training_module: Runner, mcfg: DictConfig):
    optimizer = Adam(training_module.parameters(), lr=mcfg.lr)

    def sched_fun(step):
        decay = mcfg.decay_rate ** (step // mcfg.decay_steps) + 1e-2
        decay = max(decay, mcfg.decay_min)
        return decay

    scheduler = LambdaLR(optimizer, sched_fun)
    scheduler.last_epoch = mcfg.step_start

    return optimizer, scheduler


def run_epoch(training_module: Runner, aux_modules: dict, dataloader: DataLoader,
              n_epoch: int, cfg: DictConfig, global_step=None):
    global_step = global_step or len(dataloader) * n_epoch

    optimizer = aux_modules['optimizer']
    scheduler = aux_modules['scheduler']

    prbar = tqdm(dataloader, desc=cfg.config)

    if hasattr(cfg, 'run_dir'):
        checkpoints_dir = os.path.join(cfg.run_dir, 'checkpoints')
    else:
        now = datetime.now()
        dt_string = now.strftime("%Y%m%d_%H%M%S")
        cfg.run_dir = os.path.join(DEFAULTS.experiment_root, dt_string)
        checkpoints_dir = os.path.join(cfg.run_dir, 'checkpoints')

    print(yellow(f'run_epoch started, checkpoints will be saved in {checkpoints_dir}'))

    for sample in prbar:
        global_step += 1


        if cfg.experiment.max_iter is not None and global_step > cfg.experiment.max_iter:
            break
        sample = move2device(sample, cfg.device)
        B = sample.num_graphs
        sample = add_field_to_pyg_batch(sample, 'iter', [global_step] * B, 'cloth', reference_key=None)

        if training_module.mcfg.increase_roll_every < 0:
            roll_steps = training_module.mcfg.n_opt_steps
        else:
            roll_steps = training_module.mcfg.n_opt_steps + (global_step // training_module.mcfg.increase_roll_every)
            roll_steps = min(roll_steps, training_module.mcfg.roll_max)

        if global_step >= training_module.mcfg.warmup_steps:
            ld_to_write = training_module(sample, roll_steps=roll_steps, optimizer=optimizer, scheduler=scheduler)
        else:
            ld_to_write = training_module(sample, roll_steps=roll_steps, optimizer=None, scheduler=None)

        if global_step % cfg.experiment.save_checkpoint_every == 0:
            os.makedirs(checkpoints_dir, exist_ok=True)
            checkpoint_path = os.path.join(checkpoints_dir, f"step_{global_step:010d}.pth")
            save_checkpoint(training_module, aux_modules, cfg, checkpoint_path)

    return global_step
