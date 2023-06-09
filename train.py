import os

import numpy as np
import torch

from utils.arguments import load_params, create_modules


def main():
    os.environ['OMP_NUM_THREADS'] = '1'
    os.environ['MKL_NUM_THREADS'] = '1'
    torch.set_num_threads(1)
    modules, config = load_params()
    dataloader_m, runner, training_module, aux_modules = create_modules(modules, config)

    if config.experiment.checkpoint_path is not None and os.path.exists(config.experiment.checkpoint_path):
        sd = torch.load(config.experiment.checkpoint_path)

        if 'training_module' in sd:
            training_module.load_state_dict(sd['training_module'])

            for k, v in aux_modules.items():
                if k in sd:
                    print(f'{k} LOADED!')
                    v.load_state_dict(sd[k])
        else:
            training_module.load_state_dict(sd)
        print('LOADED:', config.experiment.checkpoint_path)

    if config.detect_anomaly:
        torch.autograd.set_detect_anomaly(True)


    global_step = config.step_start

    torch.manual_seed(57)
    np.random.seed(57)
    for i in range(config.experiment.n_epochs):
        dataloader = dataloader_m.create_dataloader()
        global_step = runner.run_epoch(training_module, aux_modules, dataloader, i, config,
                                       global_step=global_step)

        if config.experiment.max_iter is not None and global_step > config.experiment.max_iter:
            break


if __name__ == '__main__':
    main()
