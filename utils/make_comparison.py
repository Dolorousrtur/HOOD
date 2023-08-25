import os
import sys

# sys.path.append(os.environ['HOOD_PROJECT'])
sys.path.insert(0, os.environ['HOOD_PROJECT'])
# print(sys.path)

from utils.arguments import load_params, create_dataloader_module
from utils.defaults import DEFAULTS
from utils.validation import update_config_for_validation, load_runner_from_checkpoint, replace_model

import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from omegaconf import MISSING, OmegaConf
from tqdm import tqdm

from utils.common import move2device, pickle_dump


def create_sequences(out_root, model_name, competitor, conf):
    if competitor == 'snug':
        conf.split_path = 'datasplits/comparison_seqs_to_snug.csv'
        conf.restpos_file = 'garf/rest_geometries_snug_allseqs.pkl'
    elif competitor == 'vto':
        conf.split_path = 'datasplits/comparison_seqs_to_ssch.csv'
        conf.restpos_file = 'garf/rest_geometries_vto.pkl'
    elif competitor == 'arcsim-snug':
        conf.split_path = 'datasplits/comparison_seqs_to_snug.csv'
        conf.zero_betas = True
    elif competitor == 'arcsim-vto':
        conf.split_path = 'datasplits/comparison_seqs_to_ssch.csv'
        conf.zero_betas = True
    else:
        raise Exception(f'Invalid competitor name: {competitor}')
    experiment_name = get_experiment_name_by_id(conf.experiment_id, conf.server)
    checkpoint = conf.checkpoint

    dataloader_m, runner, training_module, aux_modules = load_training_module(experiment_name, checkpoint, conf)
    dataloader = dataloader_m.create_dataloader(is_eval=True)
    for i, sequence in tqdm(enumerate(dataloader)):
        gname = sequence['garment_name'][0]
        seq_name = sequence['sequence_name'][0].split('/')[-1]

        out_path = out_root / gname / (seq_name + '.pkl')
        out_path.parent.mkdir(parents=True, exist_ok=True)

        print(out_path)

        sequence = move2device(sequence, 'cuda:0')
        trajectories_dict = training_module.valid_rollout(sequence, n_rollout_steps=-1,
                                                          n_steps=-1, bare=True, record_time=True)
        with open(out_path, 'wb') as f:
            pickle.dump(trajectories_dict['rollout'], f)


@dataclass
class Config:
    config_name: str = 'aux/comparisons'
    rollouts_dir: str = 'validation_sequences/my_hood_rollouts'
    obstacle_dict_file: str = 'smpl_aux.pkl'
    garment_dict_file: str = 'garments_dict.pkl'
    smpl_model: str = 'smpl/SMPL_FEMALE.pkl'

    density: Optional[float] = 0.20022
    lame_mu: Optional[float] = 23600.0
    lame_lambda: Optional[float] = 44400
    bending_coeff: Optional[float] = 3.9625778333333325e-05

    # do not set
    split_path: Optional[str] = None
    restpos_file: Optional[str] = None
    zero_betas: bool = False

    model: str = MISSING  # 'cvpr' or 'postcvpr'
    comparison: str = MISSING  # 'snug' | 'ssch'

    render_videos: bool = False
    separate_arms: bool = True


if __name__ == '__main__':
    conf = OmegaConf.structured(Config)
    conf_cli = OmegaConf.from_cli()
    conf = OmegaConf.merge(conf, conf_cli)

    if conf.comparison == 'snug':
        conf.split_path = 'validation_sequences/datasplits/comparison_seqs_to_snug.csv'
        conf.restpos_file = 'validation_sequences/rest_geometries/snug.pkl'
        rollouts_dir = Path(DEFAULTS.aux_data) / conf.rollouts_dir / 'vs_snug'
    elif conf.comparison == 'ssch':
        conf.split_path = 'validation_sequences/datasplits/comparison_seqs_to_ssch.csv'
        conf.restpos_file = 'validation_sequences/rest_geometries/ssch.pkl'
        rollouts_dir = Path(DEFAULTS.aux_data) / conf.rollouts_dir / 'vs_ssch'

    if conf.model == 'postcvpr':
        checkpoint_path = Path(DEFAULTS.data_root) / 'trained_models' / 'postcvpr.pth'
        model_config = 'postcvpr'
    elif conf.model == 'cvpr':
        checkpoint_path = Path(DEFAULTS.data_root) / 'trained_models' / 'cvpr_submission.pth'
        model_config = 'cvpr'
    elif conf.model == 'fine15':
        checkpoint_path = Path(DEFAULTS.data_root) / 'trained_models' / 'fine15.pth'
        model_config = 'cvpr_baselines/fine15'

        print('***********************FINE15***********************')
    elif conf.model == 'fine48':
        checkpoint_path = Path(DEFAULTS.data_root) / 'trained_models' / 'fine48.pth'
        model_config = 'cvpr_baselines/fine48'



    modules, experiment_config = load_params(conf.config_name)
    experiment_config = update_config_for_validation(experiment_config, conf)
    replace_model(modules, experiment_config, model_config)
    runner_module, runner = load_runner_from_checkpoint(checkpoint_path, modules, experiment_config)
    dataloader_m = create_dataloader_module(modules, experiment_config)
    dataloader = dataloader_m.create_dataloader(is_eval=True)

    for i, sequence in tqdm(enumerate(dataloader)):
        gname = sequence['garment_name'][0]
        seq_name = sequence['sequence_name'][0].split('/')[-1]

        out_path = rollouts_dir / gname / (seq_name + '.pkl')
        out_path.parent.mkdir(parents=True, exist_ok=True)


        sequence = move2device(sequence, 'cuda:0')
        trajectories_dict = runner.valid_rollout(sequence, bare=True, record_time=True)
        pickle_dump(dict(trajectories_dict), out_path)


