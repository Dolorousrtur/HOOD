import importlib
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import torch
from omegaconf import OmegaConf, MISSING, DictConfig
from torch.utils.data import DataLoader

from utils.arguments import create_runner, create_dataloader_module, load_module
from utils.defaults import DEFAULTS

from utils.dataloader import DataloaderModule
from utils.arguments import DataConfig as DataloaderConfig

@dataclass
class Config:
    garment_dict_file: str = MISSING                    # Path to the garment dict file with data for all garments relative to $HOOD_DATA/aux_data/
    smpl_model: str = MISSING                           # Path to the .csv split file relative to $HOOD_DATA/aux_data/
    data_root: Optional[str] = None                     # Path to the data root
    obstacle_dict_file: Optional[str] = 'smpl_aux.pkl'  # Path to the file with auxiliary data for obstacles relative to $HOOD_DATA/aux_data/

    random_betas: bool = False          # Whether to use random beta parameters for the SMPL model
    separate_arms: bool = False         # Whether to separate the arms from the rest of the body (to avoid body self-intersections)
    split_path: Optional[str] = None    # Path to the .csv split file relative to $HOOD_DATA/aux_data/

    # material parameters to use in the simulation
    density: Optional[float] = 0.20022
    lame_mu: Optional[float] = 23600.0
    lame_lambda: Optional[float] = 44400
    bending_coeff: Optional[float] = 3.9625778333333325e-05


    restpos_scale: Optional[float] = None   # if set, scales canonical geometries of garments by that value
    zero_betas: bool = False                # Whether to set the beta parameters to zero





def update_config_single_sequence(experiment_config, sequence_path, garment_name):
    """
    Update the experiment config loaded from .yaml file to use it for single sequence inference.
    :param experiment_config: OmegaConf config loaded from .yaml file
    :param sequence_path: path to the SMPL pose sequence file
    :param garment_name: name of the garment to be used for inference
    :return: updated experiment config
    """
    dataset_name = list(experiment_config.dataloader.dataset.keys())[0]

    data_root, file_name = os.path.split(sequence_path)
    file_name, ext = os.path.splitext(file_name)

    experiment_config.dataloader.dataset[dataset_name].data_root = data_root
    experiment_config.dataloader.dataset[dataset_name].single_sequence_file = file_name
    experiment_config.dataloader.dataset[dataset_name].single_sequence_garment = garment_name

    return experiment_config


def update_config_for_validation(experiment_config: DictConfig, validation_config: Config):
    """
    Update the experiment config loaded from .yaml file to use it for validation.
    :param experiment_config: OmegaConf config loaded from .yaml file
    :param validation_config: validation config
    :return: updated experiment config
    """
    dataset_name = list(experiment_config.dataloader.dataset.keys())[0]

    if hasattr(validation_config, "data_root") and validation_config.data_root is not None:
        experiment_config.dataloader.dataset[dataset_name].data_root = validation_config.data_root

    # TODO replace with model_type and gender
    if hasattr(validation_config, "smpl_model") and validation_config.smpl_model is not None:
        experiment_config.dataloader.dataset[dataset_name].smpl_model = validation_config.smpl_model
    if hasattr(validation_config, "split_path") and validation_config.split_path is not None:
        experiment_config.dataloader.dataset[dataset_name].split_path = validation_config.split_path
    if hasattr(validation_config, "garment_dict_file") and validation_config.garment_dict_file is not None:
        experiment_config.dataloader.dataset[dataset_name].garment_dict_file = validation_config.garment_dict_file
    if hasattr(validation_config, "restpos_scale") and validation_config.restpos_scale is not None:
        experiment_config.dataloader.dataset[dataset_name].restpos_scale_min = validation_config.restpos_scale
        experiment_config.dataloader.dataset[dataset_name].restpos_scale_max = validation_config.restpos_scale
    if hasattr(validation_config, "separate_arms") and validation_config.separate_arms is not None:
        experiment_config.dataloader.dataset[dataset_name].separate_arms = validation_config.separate_arms

    if hasattr(validation_config, "obstacle_dict_file") and validation_config.obstacle_dict_file is not None:
        experiment_config.dataloader.dataset[dataset_name].obstacle_dict_file = validation_config.obstacle_dict_file
    if hasattr(validation_config, "random_betas") and validation_config.random_betas is not None:
        experiment_config.dataloader.dataset[dataset_name].random_betas = validation_config.random_betas
    if hasattr(validation_config, "zero_betas") and validation_config.zero_betas is not None:
        experiment_config.dataloader.dataset[dataset_name].zero_betas = validation_config.zero_betas

    experiment_config.dataloader.batch_size = 1
    experiment_config.dataloader.num_workers = 0
    experiment_config.dataloader.dataset[dataset_name].wholeseq = True
    experiment_config.dataloader.dataset[dataset_name].noise_scale = 0

    runner_name = list(experiment_config.runner.keys())[0]
    experiment_config.runner[runner_name].material.density_override = validation_config.density
    experiment_config.runner[runner_name].material.lame_mu_override = validation_config.lame_mu
    experiment_config.runner[runner_name].material.lame_lambda_override = validation_config.lame_lambda
    experiment_config.runner[runner_name].material.bending_coeff_override = validation_config.bending_coeff

    experiment_config = OmegaConf.to_container(experiment_config)
    experiment_config = OmegaConf.create(experiment_config)
    return experiment_config


def load_runner_from_checkpoint(checkpoint_path: str, modules: dict, experiment_config: DictConfig):
    """
    Builds a Runned objcect
    :param checkpoint_path: path to the checkpoint to load
    :param modules: dictionary  of .py modules (from utils.arguments.load_params())
    :param experiment_config: OmegaConf config for the experiment
    :return: runner_module: .py module containing the Runner class
                runner: Runner object
    """
    runner_module, runner, _ = create_runner(modules, experiment_config,
                                             create_aux_modules=False)

    sd = torch.load(checkpoint_path)

    sd = sd['training_module']
    runner.load_state_dict(sd)

    return runner_module, runner



def create_one_sequence_dataloader_old(sequence_path: str, garment_name: str, modules: dict,
                                   experiment_config: DictConfig) -> DataLoader:
    """
    Create a dataloader for a single pose sequence and a given garment
    :param sequence_path: path to the SMPL pose sequence
    :param garment_name: garment to load
    :param modules: dictionary  of .py modules (from utils.arguments.load_params())
    :param experiment_config: OmegaConf config for the experiment
    :return:
    """
    experiment_config = update_config_single_sequence(experiment_config, sequence_path, garment_name)

    dataloader_m = create_dataloader_module(modules, experiment_config)
    dataloader = dataloader_m.create_dataloader(is_eval=True)
    return dataloader


def create_one_sequence_dataloader(sequence_path: str, garment_name: str, garment_dict_file: str, 
                                   use_config=None, dataset_name=None, **kwargs) -> DataLoader:


    if use_config is not None:
        config_dir = Path(DEFAULTS.project_dir) / 'configs'
        config_path = os.path.join(config_dir, use_config + '.yaml')
        conf_file = OmegaConf.load(config_path)

        dataset_name = list(conf_file.dataloader.dataset.keys())[0]
        dataset_config_dict = conf_file.dataloader.dataset[dataset_name]
    else:
        dataset_config_dict = {}

    dataset_module = importlib.import_module(f'datasets.{dataset_name}')
    DatasetConfig = dataset_module.Config
    create_dataset = dataset_module.create

    data_root, file_name = os.path.split(sequence_path)
    file_name, ext = os.path.splitext(file_name)

    dataset_config_dict['data_root'] = data_root
    dataset_config_dict['single_sequence_file'] = file_name
    dataset_config_dict['single_sequence_garment'] = garment_name
    dataset_config_dict['garment_dict_file'] = garment_dict_file
    dataset_config_dict.update(kwargs)


    config = DatasetConfig(**dataset_config_dict)


    dataset = create_dataset(config)
    dataloader_config = DataloaderConfig(num_workers=0)
    dataloader = DataloaderModule(dataset, dataloader_config).create_dataloader()
    return dataloader


def replace_model(modules: dict, current_config: DictConfig, model_config_name: str, config_dir: str=None):


    if config_dir is None:
        config_dir = Path(DEFAULTS.project_dir) / 'configs'

    model_config_path = os.path.join(config_dir, model_config_name + '.yaml')
    model_config = OmegaConf.load(model_config_path)

    current_config.model = OmegaConf.merge(model_config.model, current_config.model)
    modules['model'] = load_module('models', current_config.model)

    return modules, current_config
