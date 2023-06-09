import torch_geometric as pyg


class DataloaderModule:
    def __init__(self, dataset, mcfg):
        self.dataset = dataset
        self.mcfg = mcfg

    def create_dataloader(self, is_eval=False):
        mcfg = self.mcfg

        dl_class = pyg.loader.DataLoader
        dataloader = dl_class(self.dataset, batch_size=mcfg.batch_size, num_workers=mcfg.num_workers,
                              shuffle=(not is_eval))
        return dataloader
