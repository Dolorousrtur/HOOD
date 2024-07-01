

import os
import pickle
import numpy as np

from utils.common import separate_arms


class SequenceLoader:
    def __init__(self, mcfg, data_path, betas_table=None):
        self.mcfg = mcfg
        self.data_path = data_path
        self.betas_table = betas_table

        assert mcfg.model_type == 'smpl', f"Only SMPL model is supported for datasets.sequence_loaders.hood_pkl"

    def process_sequence(self, sequence: dict) -> dict:
        """
        Apply transformations to the SMPL sequence
        :param sequence: dict with SMPL parameters
        :return: processed dict with SMPL parameters
        """

        # from SNUG, eliminates hand-body penetrations
        if self.mcfg.separate_arms:
            body_pose = sequence['body_pose']
            global_orient = sequence['global_orient']
            full_pos = np.concatenate([global_orient, body_pose], axis=1)
            full_pos = separate_arms(full_pos)
            sequence['global_orient'] = full_pos[:, :3]
            sequence['body_pose'] = full_pos[:, 3:]

        # sample random SMPL beta parameters
        if self.mcfg.random_betas:
            betas = sequence['betas']
            random_betas = np.random.rand(*betas.shape)
            random_betas = random_betas * self.mcfg.betas_scale * 2
            random_betas -= self.mcfg.betas_scale
            sequence['betas'] = random_betas

        # zero-out hand pose (eliminates unrealistic hand poses)
        sequence['body_pose'][:, -6:] *= 0

        # zero-out all SMPL beta parameters
        if self.mcfg.zero_betas:
            sequence['betas'] *= 0

        return sequence

    def load_sequence(self, fname: str, betas_id: int=None) -> dict:
        """
        Load sequence of SMPL parameters from disc
        and process it

        :param fname: file name of the sequence
        :param betas_id: index of the beta parameters in self.betas_table
                        (used only in validation to generate sequences for metrics calculation
        :return: dict with SMPL parameters:
            sequence['body_pose'] np.array [Nx69]
            sequence['global_orient'] np.array [Nx3]
            sequence['transl'] np.array [Nx3]
            sequence['betas'] np.array [10]
        """
        filepath = os.path.join(self.data_path, fname + '.pkl')
        with open(filepath, 'rb') as f:
            sequence = pickle.load(f)

        assert betas_id is None or self.betas_table is not None, "betas_id should be specified only in validation mode with valid betas_table"

        if self.betas_table is not None:
            sequence['betas'] = self.betas_table[betas_id]

        sequence = self.process_sequence(sequence)

        return sequence