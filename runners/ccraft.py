# wbody2 == mimp_cut_wbody_omw_ombp
import os
import random
import time
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, Optional

import numpy as np
import torch
from omegaconf import II
from omegaconf.dictconfig import DictConfig
from torch import nn
from torch.optim import Adam
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader
from torch_geometric.data import HeteroData, Batch
from tqdm import tqdm

from runners.utils.collector import SampleCollector
from runners.utils.collision import CollisionPreprocessor
from runners.utils.impulses import CollisionSolver
from runners.utils.material import RandomMaterial
from utils.cloth_and_material import FaceNormals, ClothMatAug
from utils.common import move2device, save_checkpoint, add_field_to_pyg_batch, copy_pyg_batch, TorchTimer, NodeType
from utils.defaults import DEFAULTS


@dataclass
class MaterialConfig:
    density_min: float = 0.20022
    density_max: float = 0.20022
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
class SafecheckConfig:
    n_impulse_iters: int = 10
    max_riz_size: int = 30
    n_rest2pos_steps: int = 10
    riz_max_mp_steps: int = 10
    riz_max_steps_total: int = 100
    pinned_mass: float = 1e9
    riz_epsilon: float = 1e-10
    max_ncoll: int = -1
    max_impulse_norm: Optional[float] = None
    double_precision_impulse: bool = True
    double_precision_riz: bool = True
    device: str = II('device')


@dataclass
class Config:
    optimizer: OptimConfig = OptimConfig()
    material: MaterialConfig = MaterialConfig()
    safecheck: SafecheckConfig = SafecheckConfig()
    warmup_steps: int = 100
    increase_roll_every: int = 5000
    roll_max: int = 5
    push_eps: float = 2e-3
    grad_clip: Optional[float] = 1.
    overwrite_pos_every_step: bool = False
    nobody_freq: float = .5
    nocollect_after: int = 1
    random_pin_nobody: bool = False
    n_random_pin_nobody: int = 2
    nopin_freq: float = 0.
    long_is_training: bool = True
    short_is_training: bool = True
    no_world_edges_every: int = -1
    fake_icontour: bool = True
    cutout_with_attractions: bool = False

    enable_repulsions: bool = II("experiment.enable_repulsions")
    enable_attractions: bool = II("experiment.enable_attractions")
    enable_attractions_from: Optional[int] = II("experiment.enable_attractions_from")

    initial_ts: float = II("experiment.initial_ts")
    regular_ts: float = II("experiment.regular_ts")


class Runner(nn.Module):
    def __init__(self, model: nn.Module, criterion_dict: Dict[str, nn.Module],
                 mcfg: DictConfig):
        super().__init__()

        self.model = model
        self.criterion_dict_hood = criterion_dict
        self.mcfg = mcfg

        self.cloth_obj = ClothMatAug(None, always_overwrite_mass=True)
        self.normals_f = FaceNormals()

        self.sample_collector = SampleCollector(mcfg, obstacle=True)
        self.random_material = RandomMaterial(mcfg.material)
        self.safecheck_solver = CollisionSolver(mcfg.safecheck)
        self.collision_solver = CollisionPreprocessor(mcfg)


        if self.mcfg.enable_attractions_from is not None:
            self.safecheck_until = self.mcfg.enable_attractions_from
        elif self.mcfg.enable_attractions:
            self.safecheck_until = -1
        else:
            self.safecheck_until = np.inf

        self.short_steps = 0
        self.long_steps = 0

    def valid_rollout(self, sequence, n_steps=-1, bare=False, record_time=False, safecheck=True):

        record_time = True

        sequence = add_field_to_pyg_batch(sequence, 'iter', [0], 'cloth', reference_key=None)
        sequence = self._add_cloth_obj(sequence)

        is_obstacle = 'obstacle' in sequence.node_types
        if is_obstacle:
            n_samples = sequence['obstacle'].target_pos.shape[1]
        else:
            n_samples = sequence['cloth'].target_pos.shape[1]

        if n_steps >= 0:
            n_samples = min(n_samples, n_steps)

        trajectories_dicts = defaultdict(list)

        if record_time:
            st_time = time.time()

        st = 0
        trajectories, metrics_dict = self._rollout(sequence, st, n_samples - st,
                                                    progressbar=True, bare=bare, safecheck=safecheck)

        if record_time:
            total_time = time.time() - st_time
            metrics_dict['time'] = total_time

        # trajectories_dicts['pred'] = trajectory
            
        cloth_faces = sequence['cloth'].faces_batch.T.cpu().numpy()
        garment_name = sequence.garment_name[0]
        print('garment_name', garment_name)
        if 'celina' in garment_name:
            faces_mask = (cloth_faces == 8504).any(axis=-1)
            cloth_faces = cloth_faces[~faces_mask]
            print('!!!!!!!!!!faces_mask', faces_mask.sum())
        else:
            faces_mask = None
        

        trajectories_dicts.update(trajectories)
        trajectories_dicts['metrics'] = dict(metrics_dict)
        trajectories_dicts['cloth_faces'] = cloth_faces

        if 'garment_id' in sequence['cloth']:
            trajectories_dicts['garment_id'] = sequence['cloth'].garment_id.cpu().numpy()
            
        trajectories_dicts['vertex_type'] = sequence['cloth'].vertex_type.cpu().numpy()

        if 'uv_coords' in sequence['cloth']:
            uv_faces = sequence['cloth'].uv_faces_batch.T.cpu().numpy()
            if faces_mask is not None:
                uv_faces = uv_faces[~faces_mask]

            trajectories_dicts['uv_coords'] = sequence['cloth'].uv_coords.cpu().numpy()
            trajectories_dicts['uv_faces'] = uv_faces

        if is_obstacle:
            trajectories_dicts['obstacle_faces'] = sequence['obstacle'].faces_batch.T.cpu().numpy()

        for s in ['pred', 'obstacle']:
            if s in trajectories_dicts:
                trajectories_dicts[s] = torch.stack(trajectories_dicts[s], dim=0).cpu().numpy()
        return trajectories_dicts

    def _rollout(self, sample, start_idx, n_steps, progressbar=False, bare=False, safecheck=True):
        trajectories = defaultdict(list)
        is_obstacle = 'obstacle' in sample.node_types

        metrics_dict = defaultdict(list)

        if n_steps == 0:
            pbar = range(start_idx, start_idx + 1)
        else:
            pbar = range(start_idx, start_idx + n_steps)
        if progressbar:
            pbar = tqdm(pbar)

        prev_out_sample = None
        fail_reason = 'SUCCESS'

        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)

        sample = self.prepare_sample(sample)

        for i in pbar:
            # print('\n\n')
            sample_step = self.collect_sample(sample, i, prev_out_sample, wholeseq=True)
            if i == 0:
                # print('NOSOLVE')

                # print('SOLVE')
                # sample_step = self.collision_solver.solve(sample_step)
                sample_step, sample = self.update_sample_1st_step(sample_step, sample)

            if i == 0:
                trajectories['pred'].append(sample_step['cloth'].prev_pos)
                trajectories['pred'].append(sample_step['cloth'].pos)

                ncoll = self.safecheck_solver.calc_tritri_collisions2(sample_step, verts_key='pos')
                print('!!!!!!!!!!!!!!!!!!!!! start ncoll', ncoll)
                metrics_dict['ncoll'].append(ncoll)

                if is_obstacle:
                    trajectories['obstacle'].append(sample_step['obstacle'].prev_pos)
                    trajectories['obstacle'].append(sample_step['obstacle'].pos)


            if n_steps == 0:
                break


            with TorchTimer(metrics_dict, 'hood_time', start=start, end=end):
                sample_step = self.model(sample_step, is_training=False)


            ncoll = self.safecheck_solver.calc_tritri_collisions2(sample_step, verts_key='pred_pos')
            metrics_dict['ncoll'].append(ncoll)
            # print('ncoll', ncoll)

            prev_out_sample = sample_step.detach()

            trajectories['pred'].append(prev_out_sample['cloth'].pred_pos)
            if is_obstacle:
                trajectories['obstacle'].append(prev_out_sample['obstacle'].target_pos)

            if not bare:
                loss_dict_hood, loss_weight_dict_hood, _, _ = self.criterion_pass(sample_step, self.criterion_dict_hood)
                metrics_dict = self.add_metrics(loss_dict_hood, loss_weight_dict_hood, None, None,
                                                metrics_dict)

        metrics_dict['fail_reason'] = fail_reason

        return trajectories, metrics_dict



    def set_random_material(self, sample):
        sample, self.cloth_obj = self.random_material.add_material(sample, self.cloth_obj)
        return sample

    def _add_cloth_obj(self, sample):
        sample = self.set_random_material(sample)
        sample = self.cloth_obj.set_batch(sample, overwrite_pos=self.mcfg.overwrite_pos_every_step)
        sample['cloth'].cloth_obj = self.cloth_obj
        return sample

    def criterion_pass(self, sample_step, criterion_dict):
        sample_step.cloth_obj = self.cloth_obj
        loss_dict = dict()
        gradient_dict = dict()
        loss_weight_dict = dict()
        metrics_dict = dict()
        for criterion_name, criterion in criterion_dict.items():
            ld = criterion(sample_step)
            for k, v in ld.items():
                if 'loss' in k:
                    loss_dict[f"{criterion_name}_{k}"] = v
                elif 'gradient' in k:
                    gradient_dict[f"{criterion_name}_{k}"] = v
                elif 'weight' in k:
                    loss_weight_dict[f"{criterion_name}_{k}"] = v
                else:
                    metrics_dict[f"{criterion_name}_{k}"] = v




        return loss_dict, loss_weight_dict, gradient_dict, metrics_dict

    def check_nan(self, tensor):
        is_nan = (tensor != tensor).any()
        return is_nan

    def check_nan_pred(self, sample):
        return self.check_nan(sample['cloth'].pred_pos) or self.check_nan(sample['cloth'].pred_velocity)

    def check_nan_input(self, sample):
        return self.check_nan(sample['cloth'].pos) or self.check_nan(sample['cloth'].velocity)

    def collect_sample(self, sample, index, prev_out_dict=None, wholeseq=False, random_ts=False):
        sample_step = copy_pyg_batch(sample)

        sample_step = self.sample_collector.sequence2sample(sample_step, index)
        sample_step = sample_step = move2device(sample_step, 'cuda:0')
        sample_step = self.sample_collector.copy_from_prev(sample_step, prev_out_dict)
        ts = self.mcfg.regular_ts

        is_init = False
        if not wholeseq and index == 0:
            if random_ts:
                is_init = np.random.rand() > 0.5
                if is_init:
                    ts = self.mcfg.initial_ts
            else:
                is_init = True
                ts = self.mcfg.initial_ts


        sample_step = self.sample_collector.add_timestep(sample_step, ts)
        sample_step = self.sample_collector.add_is_init(sample_step, is_init)
        sample_step = self.sample_collector.add_velocity(sample_step, prev_out_dict)

        return sample_step

    def add_metrics_from_dict(self, loss_dict, loss_weight_dict, metrics_dict_step, prefix):
        for k, v in loss_dict.items():
            k_weight = k.replace('loss', 'weight')
            v = v.item()


            weight = loss_weight_dict[k_weight] if k_weight in loss_weight_dict else 1
            v = v / weight if weight != 0 else 0

            k_pref = prefix + k
            metrics_dict_step[k_pref] = v

        return metrics_dict_step

    def add_metrics_ratios(self, metrics_dict_step, prefix_from, prefix_to, remove_to=False):
        keys_to = [k for k in metrics_dict_step if k.startswith(prefix_to)]

        for key_to in keys_to:
            v_to = metrics_dict_step[key_to]
            key_from = key_to.replace(prefix_to, prefix_from)

            if key_from not in metrics_dict_step:
                continue

            v_from = metrics_dict_step[key_from]
            ratio = v_to / v_from

            key_ratio = key_to.replace(prefix_to, 'ratio/')
            metrics_dict_step[key_ratio] = ratio

            if remove_to:
                del metrics_dict_step[key_to]

        return metrics_dict_step

    def add_metrics(self, loss_dict_hood, loss_weight_dict_hood, loss_dict_impulse, loss_weight_dict_impulse,
                    metrics_dict):
        prefix_hood = 'hood/'
        prefix_impulse = 'impulse/'

        metrics_dict_step = {}
        metrics_dict_step = self.add_metrics_from_dict(loss_dict_hood, loss_weight_dict_hood, metrics_dict_step,
                                                       prefix=prefix_hood)

        if loss_dict_impulse is not None:
            metrics_dict_step = self.add_metrics_from_dict(loss_dict_impulse, loss_weight_dict_impulse,
                                                           metrics_dict_step,
                                                           prefix=prefix_impulse)

            metrics_dict_step = self.add_metrics_ratios(metrics_dict_step, prefix_hood, prefix_impulse, remove_to=True)

        for k, v in metrics_dict_step.items():
            metrics_dict[k].append(v)

        return metrics_dict

    def collect_penetrating_mask(self, sample_step, sample):
        penetrating_mask = torch.zeros_like(sample_step['cloth'].pos[:, :1]).bool()
        sample_step = add_field_to_pyg_batch(sample_step, 'penetrating_mask', penetrating_mask,
                                             'cloth',
                                             reference_key='pos')
        sample = add_field_to_pyg_batch(sample, 'penetrating_mask', penetrating_mask,
                                        'cloth',
                                        reference_key='pos')
        return sample_step, sample

    def check_step(self, sample_step, prev=True, threshold=0, to_print=False):
        if self.check_nan_input(sample_step):
            print(f'NAN PRED')
            return False

        ncoll_prev = self.safecheck_solver.calc_tritri_collisions(sample_step, prev=prev, threshold=threshold)

        if to_print:
            print('ncoll_prev', ncoll_prev, ncoll_prev == 0)
        return ncoll_prev == 0
    
    def add_gradient(self, sample_step, optimizer, gradient, weight=1.):
        params = optimizer.param_groups[0]['params']
        params = [param for param in params if param.requires_grad]

        model_icontour_grad = torch.autograd.grad(sample_step['cloth'].pred_pos, params, grad_outputs=gradient, allow_unused=True, retain_graph=True)

        i = 0
        for param in optimizer.param_groups[0]['params']:
            if param.requires_grad:
                if model_icontour_grad[i] is not None:
                    if param.grad is None:
                        param.grad = torch.zeros_like(param)
                    param.grad += model_icontour_grad[i]*weight
                i += 1

    def optimizer_step(self, optimizer, scheduler, loss_dict, gradient_dict, sample_step):
        if optimizer is not None:
            # for group in optimizer.param_groups:
            #     print(group['lr'])
            loss = sum(loss_dict.values())
            optimizer.zero_grad()


            for gradient_name, gradient in gradient_dict.items():
                # print(gradient_name, gradient.abs().sum())
                self.add_gradient(sample_step, optimizer, gradient, weight=1.)

            loss.backward()

                

            if self.mcfg.grad_clip is not None:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.mcfg.grad_clip)

            optimizer.step()

            if scheduler is not None:
                scheduler.step()

    def safecheck_impulse_pass(self, sample_step, metrics_dict):
        sample_step = self.safecheck_solver.safecheck_impulses(sample_step, metrics_dict)

        if self.mcfg.safecheck.max_impulse_norm is not None and self.mcfg.safecheck.max_impulse_norm > 0:
            hc_impulse_dx = sample_step['cloth'].hc_impulse_dx
            hc_impulse_norm = torch.norm(hc_impulse_dx, dim=-1)
            hc_impulse_norm_max = hc_impulse_norm.max()
            if hc_impulse_norm_max > self.mcfg.safecheck.max_impulse_norm:
                return sample_step, metrics_dict, False

        if self.mcfg.safecheck.max_ncoll > 0 and metrics_dict['impulse_stencil_ncoll'][-1] > self.mcfg.safecheck.max_ncoll:
                return sample_step, metrics_dict, False


        if self.check_nan_pred(sample_step):
            return sample_step, metrics_dict, False
        return sample_step, metrics_dict, True

    def safecheck_riz_pass(self, sample_step, metrics_dict):
        sample_step = self.safecheck_solver.safecheck_RIZ(sample_step, metrics_dict)
        if self.check_nan_pred(sample_step):
            return sample_step, metrics_dict, False
        return sample_step, metrics_dict, True

    def prepare_sample(self, sample):
        sample = self._add_cloth_obj(sample)
        sample = self.safecheck_solver.mark_penetrating_faces(sample, dummy=True)

        if random.random() < self.mcfg.nopin_freq:
            sample['cloth'].vertex_type *= 0
        return sample

    def update_sample_1st_step(self, sample_step, sample):
        sample_step, sample = self.collect_penetrating_mask(sample_step, sample)

        iter_num = sample['cloth'].iter[0].item()
        if self.safecheck_until < 0:
            is_safecheck = False
        else:
            is_safecheck = iter_num < self.safecheck_until

        if is_safecheck or self.mcfg.cutout_with_attractions:
            sample_step = self.safecheck_solver.mark_penetrating_faces(sample_step)

            sample = add_field_to_pyg_batch(sample, 'cutout_mask', sample_step['cloth'].cutout_mask,
                                            'cloth',
                                            reference_key='cutout_mask')
            sample = add_field_to_pyg_batch(sample, 'faces_cutout_mask_batch', sample_step['cloth'].faces_cutout_mask_batch,
                                            'cloth',
                                            reference_key='faces_cutout_mask_batch')
        return sample_step, sample

    def aggregate_metrics_dict(self, metrics_dict):
        metrics_dict_agg = {}
        for k, v in metrics_dict.items():
            if k == 'riz_itersmax_riz_size':
                v = np.max(v)
                v = min(v, self.mcfg.safecheck.max_riz_size)
            else:
                v = np.mean(v)

            metrics_dict_agg[k] = v
        return metrics_dict_agg

    def forward_short(self, sample, roll_steps=1, optimizer=None, scheduler=None) -> dict:
        random_ts = (roll_steps == 1)

        sample = self.prepare_sample(sample)

        metrics_dict = defaultdict(list)

        prev_out_sample = None
        _i = 0

        self.short_steps += 1

        use_wedges_seq = True
        if self.mcfg.no_world_edges_every > 0 and self.short_steps % self.mcfg.no_world_edges_every == 0:
            use_wedges_seq = False

        iter_num = sample['cloth'].iter[0].item()
        if self.safecheck_until < 0:
            is_safecheck_global = False
        else:
            is_safecheck_global = iter_num < self.safecheck_until

        for i in range(roll_steps):
            is_safecheck = is_safecheck_global

            # print('roll_steps', roll_steps)
            sample = add_field_to_pyg_batch(sample, 'step', [i], 'cloth', reference_key=None)

            is_first_step = i == 0
            is_last_step = i == roll_steps - 1
            use_wedges = (roll_steps == 1) or (i > 0)
            use_wedges = use_wedges and use_wedges_seq

            if not use_wedges:
                is_safecheck = False
            use_ocontour = i > 0

            sample = self._add_cloth_obj(sample)
            sample_step = self.collect_sample(sample, i, prev_out_sample,
                                              random_ts=random_ts)
            sample_step = self.safecheck_solver.mark_penetrating_faces_obstacle(sample_step)

            if i == 0:
                sample_step = self.collision_solver.solve(sample_step)
            if i == 1:
                sample_step, sample = self.update_sample_1st_step(sample_step, sample)
                # sample_step, sample = self.collect_penetrating_mask(sample_step, sample)

            if is_safecheck and i > 0 and not self.check_step(sample_step):
                break

            if self.mcfg.nocollect_after > 0:
                is_training = self.mcfg.short_is_training and (i < self.mcfg.nocollect_after)
            else:
                is_training = self.mcfg.short_is_training
            if i == 0 and roll_steps > 1:
                is_training = False


            fake_icontour = is_safecheck_global or not use_wedges

            sample_step = self.model(sample_step, world_edges=use_wedges, is_training=is_training, fake_icontour=fake_icontour)
            loss_dict_hood, loss_weight_dict_hood, gradient_dict, loss_metrics_dict = self.criterion_pass(sample_step, self.criterion_dict_hood)

            if use_wedges_seq:
                ncoll = self.safecheck_solver.calc_tritri_collisions2(sample_step)
                metrics_dict['ncoll'].append(ncoll)

            loss_dict_impulse = None
            loss_weight_dict_impulse = None


            self.optimizer_step(optimizer, scheduler, loss_dict_hood, gradient_dict, sample_step)

            prev_out_sample = sample_step.detach()
            if is_safecheck and not is_first_step and not is_last_step:
                prev_out_sample, metrics_dict, is_valid = self.safecheck_impulse_pass(prev_out_sample, metrics_dict)
                if not is_valid:
                    # print('not valid impulse pass')
                    break

                prev_out_sample, metrics_dict, is_valid = self.safecheck_riz_pass(prev_out_sample, metrics_dict)
                if not is_valid:
                    # print('not valid riz pass')
                    break

            _i = i + 1



            loss_dict_hood.update(loss_metrics_dict)
            metrics_dict = self.add_metrics(loss_dict_hood, loss_weight_dict_hood, loss_dict_impulse,
                                            loss_weight_dict_impulse,
                                            metrics_dict)

        metrics_dict = self.aggregate_metrics_dict(metrics_dict)


        if use_wedges_seq:
            metrics_dict['n_steps'] = _i
            metrics_dict['finished'] = int(_i == roll_steps)
        metrics_dict = {'short/' + k: v for k, v in metrics_dict.items()}

        return metrics_dict

    def forward_long(self, sample, optimizer=None, scheduler=None) -> dict:
        sample = self.prepare_sample(sample)
        roll_steps = sample['cloth'].pos.shape[1]

        self.long_steps += 1
        use_wedges_seq = True
        if self.mcfg.no_world_edges_every > 0 and self.short_steps % self.mcfg.no_world_edges_every == 0:
            use_wedges_seq = False

        metrics_dict = defaultdict(list)
        prev_out_sample = None
        _i = 0

        iter_num = sample['cloth'].iter[0].item()
        if self.safecheck_until < 0:
            is_safecheck_global = False
        else:
            is_safecheck_global = iter_num < self.safecheck_until

        for i in range(roll_steps):
            is_safecheck = is_safecheck_global
            sample = add_field_to_pyg_batch(sample, 'step', [i], 'cloth', reference_key=None)
            is_last_step = i == roll_steps - 1

            use_wedges = True
            use_wedges = use_wedges and use_wedges_seq
            if not use_wedges:
                is_safecheck = False

            sample = self._add_cloth_obj(sample)
            sample_step = self.collect_sample(sample, i, prev_out_sample, wholeseq=True)
            sample_step = self.safecheck_solver.mark_penetrating_faces_obstacle(sample_step)

            if i == 0:
                sample_step = self.collision_solver.solve(sample_step)
                sample_step, sample = self.update_sample_1st_step(sample_step, sample)
                # sample_step, sample = self.collect_penetrating_mask(sample_step, sample)

            if is_safecheck and i > 0 and not self.check_step(sample_step):
                break

            if self.mcfg.nocollect_after > 0:
                is_training = self.mcfg.long_is_training and (i < self.mcfg.nocollect_after)
            else:
                is_training = self.mcfg.long_is_training

            fake_icontour = is_safecheck_global or not use_wedges


            sample_step = self.model(sample_step, world_edges=use_wedges, is_training=is_training, fake_icontour=fake_icontour)
            loss_dict_hood, loss_weight_dict_hood, gradient_dict, _ = self.criterion_pass(sample_step, self.criterion_dict_hood)


            if use_wedges_seq:
                ncoll = self.safecheck_solver.calc_tritri_collisions2(sample_step)
                metrics_dict['ncoll'].append(ncoll)



            loss_dict_impulse, loss_weight_dict_impulse = None, None
            self.optimizer_step(optimizer, scheduler, loss_dict_hood, gradient_dict, sample_step)

            prev_out_sample = sample_step.detach()
            if is_safecheck and not is_last_step:
                prev_out_sample, metrics_dict, is_valid = self.safecheck_impulse_pass(prev_out_sample, metrics_dict)
                if not is_valid:
                    break

                prev_out_sample, metrics_dict, is_valid = self.safecheck_riz_pass(prev_out_sample, metrics_dict)
                if not is_valid:
                    break

            _i = i + 1

            metrics_dict = self.add_metrics(loss_dict_hood, loss_weight_dict_hood, loss_dict_impulse,
                                            loss_weight_dict_impulse,
                                            metrics_dict)

        metrics_dict = self.aggregate_metrics_dict(metrics_dict)
        if use_wedges_seq:
            metrics_dict['n_steps'] = _i
            metrics_dict['finished'] = int(_i == roll_steps)
        metrics_dict = {'long/' + k: v for k, v in metrics_dict.items()}
        return metrics_dict


def create_optimizer(training_module: Runner, mcfg: DictConfig):
    optimizer = Adam(training_module.model.parameters(), lr=mcfg.lr)

    def sched_fun(step):
        decay = mcfg.decay_rate ** (step // mcfg.decay_steps)
        decay = max(decay, mcfg.decay_min)

        # print(step, decay)
        return decay

    scheduler = LambdaLR(optimizer, sched_fun)
    scheduler.last_epoch = mcfg.step_start

    return optimizer, scheduler


def compute_epoch_size(dataloader_short, dataloader_long, cfg, global_step):
    len_short = len(dataloader_short)
    len_long = len(dataloader_long)
    n_steps_only_short = cfg.experiment.n_steps_only_short
    n_steps_only_short_togo = max(n_steps_only_short - global_step, 0)

    if len_short < n_steps_only_short_togo:
        return len_short

    short_steps_alternating = len_short - n_steps_only_short_togo
    long_steps_alternating = len_long

    total_steps_alternating = min(long_steps_alternating, short_steps_alternating) * 2
    total_steps = n_steps_only_short_togo + total_steps_alternating

    return total_steps


def step_short(training_module, global_step, sample, optimizer, scheduler):
    roll_steps = 1 + (global_step // training_module.mcfg.increase_roll_every)
    roll_steps = min(roll_steps, training_module.mcfg.roll_max)

    if global_step >= training_module.mcfg.warmup_steps:
        ld_to_write = training_module.forward_short(sample, roll_steps=roll_steps, optimizer=optimizer,
                                                    scheduler=scheduler)
    else:
        ld_to_write = training_module.forward_short(sample, roll_steps=roll_steps, optimizer=None, scheduler=None)

    return ld_to_write


def step_long(training_module, sample, optimizer, scheduler):
    ld_to_write = training_module.forward_long(sample, optimizer=optimizer,
                                               scheduler=scheduler)
    return ld_to_write


def make_checkpoint(training_module, aux_modules, cfg, global_step):
    if global_step < cfg.experiment.n_steps_only_short:
        to_save = global_step % cfg.experiment.save_checkpoint_every == 0
    else:
        to_save = global_step % cfg.experiment.save_checkpoint_every_wlong == 0

    if to_save:

        if hasattr(cfg, 'run_dir'):
            checkpoints_dir = os.path.join(cfg.run_dir, 'checkpoints')
        else:
            now = datetime.now()
            dt_string = now.strftime("%Y%m%d_%H%M%S")
            cfg.run_dir = os.path.join(DEFAULTS.experiment_root, dt_string)
            checkpoints_dir = os.path.join(cfg.run_dir, 'checkpoints')

        os.makedirs(checkpoints_dir, exist_ok=True)
        checkpoint_path = os.path.join(checkpoints_dir, f"step_{global_step:010d}.pth")
        save_checkpoint(training_module, aux_modules, cfg, checkpoint_path)


def make_random_pin_nobody(sample, mcfg):
    if 'obstacle' in sample.node_types:
        return sample
    if not mcfg.random_pin_nobody:
        return sample

    vertex_type = sample['cloth'].vertex_type
    N = vertex_type.shape[0]
    vertex_type = vertex_type * 0

    pinned_ids = np.random.choice(N, size=mcfg.n_random_pin_nobody, replace=False)
    vertex_type[pinned_ids] = NodeType.HANDLE
    sample['cloth'].vertex_type = vertex_type
    return sample


def run_epoch(training_module: Runner, aux_modules: dict, dataloader_short: DataLoader,
              dataloader_long: DataLoader, n_epoch: int, cfg: DictConfig, writer=None, global_step=None):
    global_step = global_step or 0

    optimizer = aux_modules['optimizer']
    scheduler = aux_modules['scheduler']

    n_steps = compute_epoch_size(dataloader_short, dataloader_long, cfg, global_step)

    prbar = tqdm(range(n_steps), desc=cfg.config)

    short_iter = dataloader_short.__iter__()
    long_iter = dataloader_long.__iter__()
    # assert False

    last_step = 'short'
    for i in prbar:
        global_step += 1

        if cfg.experiment.max_iter is not None and global_step > cfg.experiment.max_iter:
            break

        # try:
        if last_step == 'short' and global_step > cfg.experiment.n_steps_only_short:
            curr_step = 'long'
            sample = next(long_iter)
        else:
            curr_step = 'short'
            sample = next(short_iter)
        # except:
        #     break

        # print('STEP:', curr_step)

        last_step = curr_step
        sample = move2device(sample, cfg.device)

        sample = make_random_pin_nobody(sample, training_module.mcfg)

        B = sample.num_graphs
        sample = add_field_to_pyg_batch(sample, 'iter', [global_step] * B, 'cloth', reference_key=None)

        # print('ITER: ', global_step)

        if curr_step == 'short':
            ld_to_write = step_short(training_module, global_step, sample, optimizer, scheduler)
        elif curr_step == 'long':
            ld_to_write = step_long(training_module, sample, optimizer, scheduler)
        else:
            raise Exception(f'Wrong step label {curr_step}')

        if writer is not None:
            writer.write_dict(ld_to_write)

        make_checkpoint(training_module, aux_modules, cfg, global_step)

    return global_step
