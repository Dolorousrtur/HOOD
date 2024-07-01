

import os
import pickle
import numpy as np

from utils.common import separate_arms


class SequenceLoader:
    def __init__(self, mcfg, data_path, betas_table=None):
        self.mcfg = mcfg
        self.data_path = data_path
        self.betas_table = betas_table

        if mcfg.model_type != 'smplx':
            raise AttributeError("Only SMPLX model is supported for datasets.sequence_loaders.cmu_npz_smpl, would you like to use datasets.sequence_loaders.cmu_npz_smpl instead?")
        
        
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
    


    def convert_seq_to_hood_format(self, sequence_raw: dict) -> dict:
        sequence = dict()
        sequence['betas'] = sequence_raw['betas'][:10]
        sequence['global_orient'] = sequence_raw['root_orient']
        sequence['body_pose'] = sequence_raw['pose_body']
        sequence['left_hand_pose'] = sequence_raw['pose_hand'][:, :45]
        sequence['right_hand_pose'] = sequence_raw['pose_hand'][:, 45:]
        sequence['jaw_pose'] = sequence_raw['pose_jaw']
        sequence['leye_pose'] = sequence_raw['pose_eye'][:, :3]
        sequence['reye_pose'] = sequence_raw['pose_eye'][:, 3:6]
        sequence['expression'] = np.zeros_like(sequence['betas'])

        sequence['transl'] = sequence_raw['trans']

        mocap_fps = sequence_raw['mocap_frame_rate'].item()
        target_fps = self.mcfg.fps
        skip_every = int(round(mocap_fps / target_fps))

        for k in ['global_orient', 'body_pose', 'transl', 
                  'left_hand_pose', 'right_hand_pose', 
                  'jaw_pose', 'leye_pose', 'reye_pose']:            
            sequence[k] = sequence[k][::skip_every]

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
        filepath = os.path.join(self.data_path, fname + '.npz')
        sequence_raw = dict(np.load(filepath, allow_pickle=True))
        sequence = self.convert_seq_to_hood_format(sequence_raw)

        assert betas_id is None or self.betas_table is not None, "betas_id should be specified only in validation mode with valid betas_table"

        if self.betas_table is not None:
            sequence['betas'] = self.betas_table[betas_id]

        sequence = self.process_sequence(sequence)

        return sequence
    

