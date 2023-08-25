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

from omegaconf import MISSING, OmegaConf, DictConfig
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
    model: str = MISSING  # 'cvpr' or 'postcvpr'
    comparison: str = MISSING  # 'snug' | 'ssch'

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

    render_videos: bool = False
    separate_arms: bool = True

def create_sequences(model: str, comparison: str, rollouts_dir: str, verbose=True, **kwargs):
    """
    Creates and saves to disc HOOD rollouts used for comparison with SNUG and SSCH in the paper

    :param model: pretrained model to use ('postcvpr'|'cvpr'|'fine15'|'fine48')
    :param comparison: 'snug' or 'ssch'
    :param rollouts_dir: directory to save rollouts to, relative to DEFAULTS.aux_data
    :param kwargs: additional arguments, see Config
    """
    conf = Config(model=model, comparison=comparison, rollouts_dir=rollouts_dir, **kwargs)
    _create_sequences_from_config(conf, verbose=verbose)

def _create_sequences_from_config(validation_conf: DictConfig, verbose=True):

    if validation_conf.comparison == 'snug':
        validation_conf.split_path = 'validation_sequences/datasplits/comparison_seqs_to_snug.csv'
        validation_conf.restpos_file = 'validation_sequences/rest_geometries/snug.pkl'
        rollouts_dir = Path(DEFAULTS.aux_data) / validation_conf.rollouts_dir / 'vs_snug'
    elif validation_conf.comparison == 'ssch':
        validation_conf.split_path = 'validation_sequences/datasplits/comparison_seqs_to_ssch.csv'
        validation_conf.restpos_file = 'validation_sequences/rest_geometries/ssch.pkl'
        rollouts_dir = Path(DEFAULTS.aux_data) / validation_conf.rollouts_dir / 'vs_ssch'

    if validation_conf.model == 'postcvpr':
        checkpoint_path = Path(DEFAULTS.data_root) / 'trained_models' / 'postcvpr.pth'
        model_config = 'postcvpr'
    elif validation_conf.model == 'cvpr':
        checkpoint_path = Path(DEFAULTS.data_root) / 'trained_models' / 'cvpr_submission.pth'
        model_config = 'cvpr'
    elif validation_conf.model == 'fine15':
        checkpoint_path = Path(DEFAULTS.data_root) / 'trained_models' / 'fine15.pth'
        model_config = 'cvpr_baselines/fine15'

    elif validation_conf.model == 'fine48':
        checkpoint_path = Path(DEFAULTS.data_root) / 'trained_models' / 'fine48.pth'
        model_config = 'cvpr_baselines/fine48'

    if verbose:
        print(f'Creating sequences for {validation_conf.comparison} using {validation_conf.model} model')
        print(f'Rollouts will be saved to {rollouts_dir}')


    modules, experiment_config = load_params(validation_conf.config_name)
    experiment_config = update_config_for_validation(experiment_config, validation_conf)
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

if __name__ == '__main__':
    conf = OmegaConf.structured(Config)
    conf_cli = OmegaConf.from_cli()
    conf = OmegaConf.merge(conf, conf_cli)

    _create_sequences_from_config(conf)
