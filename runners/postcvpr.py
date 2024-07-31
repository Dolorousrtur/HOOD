import os
import time
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, Optional

import numpy as np
import torch
from huepy import yellow
from omegaconf import II
from omegaconf.dictconfig import DictConfig
from torch import nn
from torch.optim import Adam
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader
from tqdm import tqdm

from runners.utils.collector import SampleCollector
from runners.utils.collision import CollisionPreprocessor
from runners.utils.material import RandomMaterial
from utils.cloth_and_material import FaceNormals, ClothMatAug
from utils.common import move2device, save_checkpoint, add_field_to_pyg_batch
from utils.defaults import DEFAULTS


@dataclass
class MaterialConfig:
    density_min: float = 0.20022            # minimal density to sample from (used to compute nodal masses)
    density_max: float = 0.20022            # maximal density to sample from (used to compute nodal masses)
    lame_mu_min: float = 23600.0            # minimal shear modulus to sample from
    lame_mu_max: float = 23600.0            # maximal shear modulus to sample from
    lame_lambda_min: float = 44400          # minimal Lame's lambda to sample from
    lame_lambda_max: float = 44400          # maximal Lame's lambda to sample from
    bending_coeff_min: float = 3.96e-05     # minimal bending coefficient to sample from
    bending_coeff_max: float = 3.96e-05     # maximal bending coefficient to sample from
    bending_multiplier: float = 1.          # multiplier for bending coefficient

    density_override: Optional[float] = None        # if set, overrides the sampled density (used in validation)
    lame_mu_override: Optional[float] = None        # if set, overrides the sampled shear modulus (used in validation)
    lame_lambda_override: Optional[float] = None    # if set, overrides the sampled Lame's lambda (used in validation)
    bending_coeff_override: Optional[float] = None  # if set, overrides the sampled bending coefficient (used in validation)


@dataclass
class OptimConfig:
    lr: float = 1e-4                # initial learning rate
    decay_rate: float = 1e-1        # decay multiplier for the scheduler
    decay_min: float = 0            # minimal decay
    decay_steps: int = 5_000_000    # number of steps for one decay step
    step_start: int = 0             # step to start from (used to resume training)


@dataclass
class Config:
    optimizer: OptimConfig = OptimConfig()
    material: MaterialConfig = MaterialConfig()
    warmup_steps: int = 100                 # number of steps to warm up the normalization statistics
    increase_roll_every: int = 5000         # we start from predicting only one step, then increase the number of steps each `increase_roll_every` steps
    roll_max: int = 5                       # maximum number of steps to predict
    push_eps: float = 2e-3                  # threshold for collision solver, we apply it once before the first step
    grad_clip: Optional[float] = 1.         # if set, clips the gradient norm to this value
    overwrite_pos_every_step: bool = False  # if true, the canonical poses of each garment are not cached

    # In the paper, the difference between the initial and regular time steps is explained with alpha coeffitient in the inertia loss term
    initial_ts: float = 1 / 3   # time between the first two steps in training, used to allow the model to faster reach static equiliblium
    regular_ts: float = 1 / 30  # time between the regular steps in training and validation

    device: str = II('device')


class Runner(nn.Module):
    def __init__(self, model: nn.Module, criterion_dict: Dict[str, nn.Module], mcfg: DictConfig):
        super().__init__()

        self.model = model
        self.criterion_dict = criterion_dict
        self.mcfg = mcfg

        self.cloth_obj = ClothMatAug(None, always_overwrite_mass=True)
        self.normals_f = FaceNormals()

        self.sample_collector = SampleCollector(mcfg)
        self.collision_solver = CollisionPreprocessor(mcfg)
        self.random_material = RandomMaterial(mcfg.material)

    def valid_rollout(self, sequence, n_steps=-1, bare=False, record_time=False):
        """
        Generates a trajectory for the full sequence, use this function for inference

        :param sequence: torch geometric batch with the sequence
        :param n_steps: number of steps to predict, if -1, predicts for the full sequence
        :param bare: if true, the loss terms are not computed, use for faster validation
        :param record_time: if true, records the time spent on the forward pass and adds it to trajectories_dicts['metrics']
        :return: trajectories_dicts: dict with the following fields:
            'pred': np.ndarray (NxVx3) predicted cloth trajectory
            'obstacle': np.ndarray (NxWx3) body trajectory
            'cloth_faces': np.ndarray (Fx3) cloth faces
            'obstacle_faces': np.ndarray (Fox3) body faces
            'metrics': dictionary with by-frame loss values
        """
        sequence = add_field_to_pyg_batch(sequence, 'iter', [0], 'cloth', reference_key=None)
        sequence = self.add_cloth_obj(sequence)

        n_samples = sequence['obstacle'].pos.shape[1]
        if n_steps > 0:
            n_samples = min(n_samples, n_steps)

        trajectories_dicts = defaultdict(lambda: defaultdict(list))

        if record_time:
            st_time = time.time()

        trajectory, obstacle_trajectory, metrics_dict = self._rollout(sequence, n_samples,
                                                                      progressbar=True, bare=bare)

        if record_time:
            total_time = time.time() - st_time
            metrics_dict['time'] = total_time

        trajectories_dicts['pred'] = trajectory
        trajectories_dicts['obstacle'] = obstacle_trajectory
        trajectories_dicts['metrics'] = dict(metrics_dict)
        trajectories_dicts['cloth_faces'] = sequence['cloth'].faces_batch.T.cpu().numpy()
        trajectories_dicts['obstacle_faces'] = sequence['obstacle'].faces_batch.T.cpu().numpy()

        for s in ['pred', 'obstacle']:
            trajectories_dicts[s] = np.stack(trajectories_dicts[s], axis=0)
        return trajectories_dicts

    def _rollout(self, sequence, n_steps, progressbar=False, bare=False):
        trajectory = []
        obstacle_trajectory = []

        metrics_dict = defaultdict(list)

        pbar = range(0, n_steps)
        if progressbar:
            pbar = tqdm(pbar)

        prev_out_dict = None
        for i in pbar:
            state = self.collect_sample_wholeseq(sequence, i, prev_out_dict)

            if i == 0:
                state = self.collision_solver.solve(state)

            state = self.model(state, is_training=False)

            trajectory.append(state['cloth'].pred_pos.detach().cpu().numpy())
            obstacle_trajectory.append(state['obstacle'].target_pos.detach().cpu().numpy())

            if not bare:
                loss_dict = self.criterion_pass(state)
                for k, v in loss_dict.items():
                    metrics_dict[k].append(v.item())
            prev_out_dict = state.clone()

        
        return trajectory, obstacle_trajectory, metrics_dict

    def collect_sample_wholeseq(self, sequence, index, prev_out_dict):

        """
        Collects a sample from the sequence, given the previous output and the index of the current step
        This function is only used in validation
        For training, see `collect_sample`

        :param sequence: torch geometric batch with the sequence
        :param index: index of the current step
        :param prev_out_dict: previous output of the model

        """
        sample_step = sequence.clone()

        # gather infor for the current step
        sample_step = self.sample_collector.sequence2sample(sample_step, index)

        # move to device
        sample_step = move2device(sample_step, self.mcfg.device)

        # coly fields from the previous step (pred_pos -> pos, pos->prev_pos)
        sample_step = self.sample_collector.copy_from_prev(sample_step, prev_out_dict)
        ts = self.mcfg.regular_ts

        # in the first step, the obstacle and positions of the pinned vertices are static
        if index == 0:
            sample_step = self.sample_collector.target2pos(sample_step)
            sample_step = self.sample_collector.pos2prev(sample_step)
            ts = self.mcfg.initial_ts
        # in the second step, we set velocities for both the cloth and the obstacle to zero
        elif index == 1:
            sample_step = self.sample_collector.pos2prev(sample_step)

        sample_step = self.sample_collector.add_velocity(sample_step, prev_out_dict)
        sample_step = self.sample_collector.add_timestep(sample_step, ts)
        return sample_step

    def set_random_material(self, sample):
        """
        Add material parameters to the cloth object and the sample
        :param sample:
        :return:
        """
        sample, self.cloth_obj = self.random_material.add_material(sample, self.cloth_obj)
        return sample

    def add_cloth_obj(self, sample):
        """
        - Updates self.cloth_obj with the cloth object in the sample
        - Samples the material properties of the cloth object and adds them to the sample
        - Adds info about the garment to the sample, which is later used by the the GNN and to compute objective terms (see utils.cloth_and_material.ClothMatAug for details)
        """
        sample = self.set_random_material(sample)
        sample = self.cloth_obj.set_batch(sample, overwrite_pos=self.mcfg.overwrite_pos_every_step)
        sample['cloth'].cloth_obj = self.cloth_obj
        return sample

    def criterion_pass(self, sample_step):
        """
        Pass the sample through all the loss terms in self.criterion_dict
        and gathers their values in a dictionary
        """
        sample_step.cloth_obj = self.cloth_obj
        loss_dict = dict()
        for criterion_name, criterion in self.criterion_dict.items():
            ld = criterion(sample_step)
            for k, v in ld.items():
                loss_dict[f"{criterion_name}_{k}"] = v

        return loss_dict

    def collect_sample(self, sample, idx, prev_out_dict=None, random_ts=False):
        """
        Collects a sample from the sequence, given the previous output and the index of the current step
        This function is only used in training
        For validation, see `collect_sample_wholeseq`

        :param sample: pytroch geometric batch from the dataloader
        :param idx: index of the current step
        :param prev_out_dict: previous output of the model
        :param random_ts: if True, the time step is randomly chosen between the initial and the regular time step
        :return: the sample for the current step
        """

        sample_step = sample.clone()

        # coly fields from the previous step (pred_pos -> pos, pos->prev_pos)
        sample_step = self.sample_collector.copy_from_prev(sample_step, prev_out_dict)
        ts = self.mcfg.regular_ts

        # copy positions from the lookup steps
        if idx != 0:
            sample_step = self.sample_collector.lookup2target(sample_step, idx - 1)

        # in the first step, the obstacle and positions of the pinned vertices are static
        if idx == 0:
            is_init = np.random.rand() > 0.5
            sample_step = self.sample_collector.pos2target(sample_step)
            if is_init or not random_ts:
                sample_step = self.sample_collector.pos2prev(sample_step)
                ts = self.mcfg.initial_ts
        # for the second frame, we set velocity to zero
        elif idx == 1:
            sample_step = self.sample_collector.pos2prev(sample_step)

        sample_step = self.sample_collector.add_velocity(sample_step, prev_out_dict)
        sample_step = self.sample_collector.add_timestep(sample_step, ts)
        return sample_step

    def optimizer_step(self, loss_dict, optimizer=None, scheduler=None):
        if optimizer is not None:
            loss = sum(loss_dict.values())
            optimizer.zero_grad()
            loss.backward()
            if self.mcfg.grad_clip is not None:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.mcfg.grad_clip)

            optimizer.step()
            if scheduler is not None:
                scheduler.step()

    def forward(self, sample, roll_steps=1, optimizer=None, scheduler=None) -> dict:

        # for the first 5000 steps, we randomly chose between initial and regular timesteps so that model does not overfit
        # Then, we always use initial timestep for the first frame and regular timestep for the rest of the frames
        random_ts = (roll_steps == 1)

        # add
        sample = self.add_cloth_obj(sample)

        prev_out_sample = None
        for i in range(roll_steps):
            sample_step = self.collect_sample(sample, i, prev_out_sample, random_ts=random_ts)

            if i == 0:
                sample_step = self.collision_solver.solve(sample_step)

            sample_step = self.model(sample_step)
            loss_dict = self.criterion_pass(sample_step)
            prev_out_sample = sample_step.detach()

            self.optimizer_step(loss_dict, optimizer, scheduler)

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

        # add `iter` field to the sample (used in some criterions with dynamically changing weight)
        B = sample.num_graphs
        sample = add_field_to_pyg_batch(sample, 'iter', [global_step] * B, 'cloth', reference_key=None)

        # number of autoregressive steps to simulate for the training sample
        roll_steps = 1 + (global_step // training_module.mcfg.increase_roll_every)
        roll_steps = min(roll_steps, training_module.mcfg.roll_max)

        # ld_to_write is a dictionary of loss values averaged across a training sample
        # you can feed it to tensorboard or wandb writer
        optimizer_to_pass = optimizer if global_step >= training_module.mcfg.warmup_steps else None
        scheduler_to_pass = scheduler if global_step >= training_module.mcfg.warmup_steps else None
        ld_to_write = training_module(sample, roll_steps=roll_steps, optimizer=optimizer_to_pass,
                                      scheduler=scheduler_to_pass)

        # save checkpoint every `save_checkpoint_every` steps
        if global_step % cfg.experiment.save_checkpoint_every == 0:
            checkpoint_path = os.path.join(checkpoints_dir, f"step_{global_step:010d}.pth")
            save_checkpoint(training_module, aux_modules, cfg, checkpoint_path)

    return global_step
