import importlib
import os
import typing
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from omegaconf import OmegaConf, DictConfig

from utils.dataloader import DataloaderModule
from utils.defaults import DEFAULTS


@dataclass
class ExperimentConfig:
    name: Optional[str] = None              # name for the experiment
    save_checkpoint_every: int = 100000     # save checkpoint every n iterations
    n_epochs: int = 200                     # number of epochs
    checkpoint_path: Optional[str] = None   # path to checkpoint to load
    max_iter: Optional[int] = None          # max number of iterations


@dataclass
class DataConfig:
    num_workers: int = 0                   # number of workers for dataloader
    batch_size: int = 1                    # batch size (only 1 is supported)


@dataclass
class MainConfig:
    config: Optional[str] = None           # name of the config file relative to $HOOD_PROJECT/configs (without .yaml)

    device: str = 'cuda:0'                 # device to use
    dataloader: DataConfig = DataConfig()
    experiment: ExperimentConfig = ExperimentConfig()
    detect_anomaly: bool = False           # torch.autograd.detect_anomaly
    step_start: int = 0                    # start iteration


def struct_fix(config):
    OmegaConf.set_struct(config, False)
    for k, v in config.items():
        if type(v) == DictConfig:
            struct_fix(v)


def load_module(module_type: str, module_config: DictConfig, module_name: str = None):
    """
    This function loads a singular module from OmegaConf config.
    It also merges the default config of the module with the config from the config file.

    :param module_type: type of module to load (e.g. models, runners, etc.)
    :param module_config: OmegaConf config
    :param module_name: module name
    :return: loaded python module
    """
    # if module_name is not specified, take the first OmegaConf key
    if module_name is None:
        module_name = list(module_config.keys())[0]

    # load python module
    module = importlib.import_module(f'{module_type}.{module_name}')

    # load default config from module.Config and merge with config from config file
    default_module_config = OmegaConf.create(module.Config)
    OmegaConf.set_struct(default_module_config, False)

    if module_config[module_name] is None:
        module_config[module_name] = default_module_config
    else:
        module_config[module_name] = OmegaConf.merge(default_module_config, module_config[module_name])
    return module


def load_params(config_name: str=None, config_dir: str=None):
    """
    Build OmegaConf config and the modules from the config file.
    :param config_name: name of the config file (without .yaml)
    :param config_dir: root directory of the config files
    :return:
        modules: dict of loaded modules
        conf: OmegaConf config
    """

    # Set default config directory
    if config_dir is None:
        config_dir = Path(DEFAULTS.project_dir) / 'configs'

    # Load default config from MainConfig and merge in cli parameters
    conf = OmegaConf.structured(MainConfig)
    struct_fix(conf)
    if config_name is None:
        conf_cli = OmegaConf.from_cli()
        config_name = conf_cli.config
        conf = OmegaConf.merge(conf, conf_cli)

    # Load config file and merge it in
    conf['config'] = config_name
    config_path = os.path.join(config_dir, config_name + '.yaml')
    conf_file = OmegaConf.load(config_path)
    OmegaConf.set_struct(conf, False)
    OmegaConf.set_struct(conf_file, False)
    conf = OmegaConf.merge(conf, conf_file)

    # Load modules from config
    modules = {}

    modules['model'] = load_module('models', conf.model)
    modules['runner'] = load_module('runners', conf.runner)

    # Can have arbitrary number of criterions
    modules['criterions'] = {}
    conf_criterions = conf.criterions
    for criterion_name in conf_criterions:
        criterion_module = load_module('criterions', conf.criterions, criterion_name)
        modules['criterions'][criterion_name] = criterion_module

    dataset_module = load_module('datasets', conf.dataloader.dataset)
    modules['dataset'] = dataset_module

    return modules, conf


def create_module(module, module_config: DictConfig, module_name: str=None):
    """
    Create a module object from the python module and the config file.
    :param module: python module (should have `create` method)
    :param module_config: OmegaConf config for the module
    :param module_name: name of the module
    :return: module object: loaded module object
    """
    if module_name is None:
        module_name = list(module_config.keys())[0]
    module_config = module_config[module_name]
    module_object = module.create(module_config)
    return module_object


def create_runner(modules: dict, config: DictConfig, create_aux_modules=True):
    """
    Create a runner object from the specified runner module.
    :param modules: dict of loaded .py modules
    :param config: OmegaConf config
    :param create_aux_modules: whether to create optimizer and scheduler
    :return: runner_module: .py runner module
    :return: runner: Runner object
    :return aux_modules: dict of auxiliary modules (optimizer, scheduler, etc.)
    """
    runner_module = modules['runner']
    runner_name = list(config['runner'].keys())[0]
    runner_config = config['runner'][runner_name]

    # create model object
    model = create_module(modules['model'], config['model'])

    # fill criterion dict with criterion objects
    criterions = {}
    for criterion_name, criterion_module in modules['criterions'].items():
        criterion = create_module(criterion_module, config['criterions'],
                                  module_name=criterion_name)
        if hasattr(criterion, 'name'):
            criterion_name = criterion.name
        criterions[criterion_name] = criterion

    # create Runner object from the specified runner module
    runner = runner_module.Runner(model, criterions, runner_config)

    # create optimizer and scheduler from the specified runner module
    aux_modules = dict()

    if create_aux_modules:
        optimizer, scheduler = runner_module.create_optimizer(runner, runner_config.optimizer)
        aux_modules['optimizer'] = optimizer
        aux_modules['scheduler'] = scheduler

    return runner_module, runner, aux_modules


def create_dataloader_module(modules: dict, config: DictConfig):
    """
    Create a dataloader module.
    :param modules:
    :param config: OmegaConf config
    :return: DataloaderModule
    """

    # create dataset object and pass it to the dataloader module
    dataset = create_module(modules['dataset'], config['dataloader']['dataset'])
    dataloader_m = DataloaderModule(dataset, config['dataloader'])
    return dataloader_m


def create_modules(modules: dict, config: DictConfig, create_aux_modules: bool=True):
    """
    Create all the modules from the config file.
    :param modules: dict of loaded python modules
    :param config: full OmegaConf config
    :param create_aux_modules: whether to create optimizer and scheduler
    :return:
        dataloader_m: dataloader module (should have `create_dataloader` method)
        runner: runner module
        training_module: Runner object from the selected runner
        aux_modules: dict of auxiliary modules (e.g. optimizer, scheduler)
    """

    runner_module, runner, aux_modules = create_runner(modules, config, create_aux_modules=create_aux_modules)
    dataloader_m = create_dataloader_module(modules, config)
    return dataloader_m, runner_module, runner, aux_modules
