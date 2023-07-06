import os
import pickle
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Batch
from tqdm import tqdm
from utils.arguments import load_params, create_modules
from utils.common import move2device, pickle_load


def set_garment(sample, rest_pos, faces_batch_cloth, faces_batch_obstacle):
    example = sample.get_example(0)
    example['cloth'].rest_pos = rest_pos
    example['cloth'].faces_batch = faces_batch_cloth

    example['obstacle'].faces_batch = faces_batch_obstacle

    sample = Batch.from_data_list([example])
    return sample


def set_cloth_obj(runner, sample, seq_dict, rest_pos, device):
    rest_pos = torch.FloatTensor(rest_pos).to(device)
    faces_batch_cloth = torch.LongTensor(seq_dict['cloth_faces']).to(device).T
    faces_batch_obstacle = torch.LongTensor(seq_dict['obstacle_faces'].astype(np.int32)).to(device).T

    sample = set_garment(sample, rest_pos, faces_batch_cloth, faces_batch_obstacle)
    runner.cloth_obj.cache = defaultdict(dict)
    sample = runner.add_cloth_obj(sample)

    sample = runner.sample_collector.add_timestep(sample, runner.mcfg.regular_ts)

    return sample


def set_step(sample, seq_dict, index, device):
    obstacle_pos = torch.FloatTensor(seq_dict['obstacle'][index]).to(device)
    obstacle_prev_pos = torch.FloatTensor(seq_dict['obstacle'][index - 1]).to(device)
    cloth_pos = torch.FloatTensor(seq_dict['pred'][index - 1]).to(device)
    cloth_prev_pos = torch.FloatTensor(seq_dict['pred'][index - 2]).to(device)
    cloth_pred_pos = torch.FloatTensor(seq_dict['pred'][index]).to(device)

    example = sample.get_example(0)
    example['obstacle'].pos = obstacle_pos
    example['obstacle'].prev_pos = obstacle_prev_pos
    example['cloth'].pred_pos = cloth_pred_pos
    example['cloth'].pos = cloth_pos
    example['cloth'].prev_pos = cloth_prev_pos

    sample = Batch.from_data_list([example])
    return sample


def calc_metrics_by_seq(seq_path, sample, runner, rest_pos, device='cuda:0', to_mean=True):
    with open(seq_path, 'rb') as f:
        seq_dict = pickle.load(f)

    metric_lists = defaultdict(list)

    sample = set_cloth_obj(runner, sample, seq_dict, rest_pos, device)
    N = seq_dict['pred'].shape[0]

    it = range(2, N)
    for i in it:
        sample = set_step(sample, seq_dict, i, device)
        sample['cloth'].pred_pos.requires_grad = True

        # compute metrics
        loss_dict = runner.criterion_pass(sample)

        # compute total loss
        total_val = sum(v for k, v in loss_dict.items() if 'loss' in k)

        # compute gradient norm
        grad = torch.autograd.grad(total_val, sample['cloth'].pred_pos)[0]
        grad_norm = torch.norm(grad, p=2, dim=1).mean()

        metric_lists['grad_norm'].append(grad_norm.item())
        metric_lists['total'].append(total_val.item())

        for k, v in loss_dict.items():
            if to_mean:
                metric_lists[k].append(v.item())
            else:
                metric_lists[k].append(v.detach().cpu().numpy())

    if to_mean:
        metrics_mean = {k: np.mean(v) for k, v in metric_lists.items()}
    else:
        metrics_mean = metric_lists
    metrics_mean['n_steps'] = N - 2

    return metrics_mean


def get_runner_and_sample_for_metrics(config_name, device='cuda:0'):
    modules, config = load_params(config_name)
    config.dataloader.batch_size = 1
    config.dataloader.num_workers = 0

    dataloader_m, runner, training_module, aux_modules = create_modules(modules, config)
    dataloader = dataloader_m.create_dataloader(is_eval=False)

    sample = next(iter(dataloader))
    sample = move2device(sample, device)

    return training_module, sample


def make_canonicalpos_dict(canonicalpos_by_beta, datasplit):
    out_dict = defaultdict(dict)
    for i in range(datasplit.shape[0]):
        row = datasplit.iloc[i]

        seq = row.id
        garment = row.garment
        betas_id = row.betas_id

        out_dict[garment][seq] = canonicalpos_by_beta[betas_id]

    return out_dict


def _compute_and_store_metrics_vs_sota(seqs_root, out_file, canonicalpos_dict_path, datasplit_path,
                                       verbose=True):

    # build runner object and exemplar sample
    runner, sample = get_runner_and_sample_for_metrics('aux/metrics')

    all_metrics_dict = defaultdict(lambda: defaultdict(dict))

    with open(canonicalpos_dict_path, 'rb') as f:
        canonicalpos_by_beta = pickle.load(f)
    datasplit = pd.read_csv(datasplit_path)
    canonicalpos_dict = make_canonicalpos_dict(canonicalpos_by_beta, datasplit)

    garments = os.listdir(seqs_root)
    for garment in tqdm(garments):
        garment_dir = os.path.join(seqs_root, garment)
        seqs = os.listdir(garment_dir)

        for seq in tqdm(seqs):
            seq_path = os.path.join(garment_dir, seq)
            rest_pos = canonicalpos_dict[garment][seq.split('.')[0]][0]

            # compute metrics over the sequence given the canonical garment geometry
            metrics_dict = calc_metrics_by_seq(seq_path, sample, runner, rest_pos)
            all_metrics_dict[garment][seq.split('.')[0]] = metrics_dict

    os.makedirs(os.path.dirname(out_file), exist_ok=True)
    all_metrics_dict = dict(all_metrics_dict)

    with open(out_file, 'wb') as f:
        pickle.dump(all_metrics_dict, f)

    if verbose:
        print(f'Saved metrics to {out_file}')


def compute_and_store_metrics(out_root, seqs_root, canonicalpos_root, datasplit_root, verbose=True):
    """
    Computes metrics for all sequences in seqs_root and stores them in out_root.

    :param out_root: Path to directory where metrics will be stored
    :param seqs_root: Path to directory with sequences generated by HOOD
    :param canonicalpos_root: Path to directory with canonical geometries used by HOOD
    :param datasplit_root: Path to directory with .csv datasplits used to generate sequences
    """

    seqs_root = Path(seqs_root)
    out_root = Path(out_root)
    canonicalpos_root = Path(canonicalpos_root)
    datasplit_root = Path(datasplit_root)

    for sota in ['snug', 'ssch']:
        seqs_root_sota = seqs_root / f'vs_{sota}'
        out_file = out_root / f'vs_{sota}.pkl'
        canonicalpos_dict_path = canonicalpos_root / f'{sota}.pkl'
        datasplit_path = datasplit_root / f'comparison_seqs_to_{sota}.csv'

        if verbose:
            print(f'\nComputing metrics for sequences with {sota} restposes')
        _compute_and_store_metrics_vs_sota(seqs_root_sota, out_file, canonicalpos_dict_path, datasplit_path, verbose=True)

        if verbose:
            print('Done')


def _print_dict_vsb(metrics_sum_dict):
    n_steps = metrics_sum_dict['n_steps']
    metrics_mean_dict = {k: v / n_steps for k, v in metrics_sum_dict.items()}

    key_mapping = dict()

    key_mapping['total'] = 'total loss'
    key_mapping['grad_norm'] = 'gradient norm'
    key_mapping['stretching_energy_loss'] = 'stretching energy'
    key_mapping['bending_energy_loss'] = 'bending_energy'
    key_mapping['inertia_loss'] = 'inertia'
    key_mapping['gravitational_energy_loss'] = 'gravitational_energy'
    key_mapping['collision_penalty_loss'] = 'collision loss'
    key_mapping['friction_energy_loss'] = 'friction_energy'

    for k, k_print in key_mapping.items():
        v = metrics_mean_dict[k]
        print(f'{k_print}:\t\t\t {v:.3e}')


def print_metrics_vs_baselines(metrics_root, match_paper=False):
    """
    Prints metrics used in Table 1 of the paper and Table 2 of the supplementary material

    :param metrics_root: path to the folder with metrics .pkl files (computed by compute_and_store_metrics)
    """
    metrics_sum_dict_allgarments = defaultdict(lambda: 0)
    metrics_sum_dict_dress = defaultdict(lambda: 0)

    metrics_root = Path(metrics_root)
    vs_snug_path = metrics_root / 'vs_snug.pkl'
    vs_ssch_path = metrics_root / 'vs_ssch.pkl'

    metrics_file_list = [vs_snug_path, vs_ssch_path]

    for metrics_file in metrics_file_list:
        metrics_dict = pickle_load(metrics_file)

        for garment in metrics_dict:

            # For these tables we use all garments with SNUG resting poses + dress with SSCH restposes
            # tshirt with SSCH restposes is not included
            if 'vs_ssch' in str(metrics_file) and garment == 'tshirt':
                continue

            garment_dict = metrics_dict[garment]
            for seq in garment_dict:

                # omit sequence where Fine15 baseline diverged
                if match_paper and 'vs_ssch' in str(metrics_file) and seq == '55_27':
                    continue

                seq_dict = garment_dict[seq]

                n_steps = seq_dict['n_steps']

                for k, v in seq_dict.items():
                    if k != 'n_steps':
                        v *= n_steps
                    metrics_sum_dict_allgarments[k] += v

                    if garment == 'dress':
                        metrics_sum_dict_dress[k] += v

    print('All garments:')
    _print_dict_vsb(metrics_sum_dict_allgarments)

    print('\nOnly dress:')
    _print_dict_vsb(metrics_sum_dict_dress)


def _print_dict_vss(metrics_sum_dict):
    n_steps = metrics_sum_dict['n_steps']
    metrics_mean_dict = {k: v / n_steps for k, v in metrics_sum_dict.items()}

    key_mapping = dict()

    key_mapping['collision_penalty_loss'] = 'collision loss'
    key_mapping['collision_penalty_perc'] = '% penetrating vertices'

    for k, k_print in key_mapping.items():
        v = metrics_mean_dict[k]
        print(f'{k_print}:\t\t\t {v:.3e}')


def _print_metrics_vs_snug(vs_snug_path):
    metrics_sum_dict = defaultdict(lambda: 0)

    metrics_dict = pickle_load(vs_snug_path)

    for garment in metrics_dict:
        garment_dict = metrics_dict[garment]
        for seq in garment_dict:

            seq_dict = garment_dict[seq]

            n_steps = seq_dict['n_steps']

            for k, v in seq_dict.items():
                if k != 'n_steps':
                    v *= n_steps
                metrics_sum_dict[k] += v

    _print_dict_vss(metrics_sum_dict)


def _print_metrics_vs_ssch(vs_ssch_path):
    metrics_sum_dict = defaultdict(lambda: 0)

    metrics_dict = pickle_load(vs_ssch_path)

    for garment in metrics_dict:

        garment_dict = metrics_dict[garment]
        for seq in garment_dict:

            seq_dict = garment_dict[seq]

            n_steps = seq_dict['n_steps']

            for k, v in seq_dict.items():
                if k != 'n_steps':
                    v *= n_steps
                metrics_sum_dict[k] += v

    _print_dict_vss(metrics_sum_dict)


def print_metrics_vs_sota(metrics_root):
    """
    Prints metrics used in Table 1 of the supplementary material

    :param metrics_root: path to the folder with metrics .pkl files (computed by compute_and_store_metrics)
    """

    metrics_root = Path(metrics_root)
    vs_snug_path = metrics_root / 'vs_snug.pkl'
    vs_ssch_path = metrics_root / 'vs_ssch.pkl'

    print('vs SNUG:')
    _print_metrics_vs_snug(vs_snug_path)

    print('\nvs SSCH:')
    _print_metrics_vs_ssch(vs_ssch_path)
