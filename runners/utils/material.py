import torch

from utils.cloth_and_material import Material
from utils.common import random_between, random_between_log, relative_between_log, relative_between, \
    add_field_to_pyg_batch


class RandomMaterial:
    """
    Helper class to sample random material parameters
    """
    def __init__(self, mcfg):
        self.mcfg = mcfg
        
    def get_density(self, device, B):
        if self.mcfg.density_override is None:
            density = random_between(self.mcfg.density_min, self.mcfg.density_max, shape=[B]).to(
                device)
        else:
            density = torch.ones(B).to(device) * self.mcfg.density_override
            
        return density
    
    def get_lame_mu(self, device, B):
        if self.mcfg.lame_mu_override is None:
            lame_mu, lame_mu_input = random_between_log(self.mcfg.lame_mu_min, self.mcfg.lame_mu_max,
                                                        shape=[B], return_norm=True, device=device)
        else:
            lame_mu = torch.ones(B).to(device) * self.mcfg.lame_mu_override
            lame_mu_input = relative_between_log(self.mcfg.lame_mu_min, self.mcfg.lame_mu_max,
                                                 lame_mu)
            
        return lame_mu, lame_mu_input
    
    def get_lame_lambda(self, device, B):
        if self.mcfg.lame_lambda_override is None:
            lame_lambda, lame_lambda_input = random_between(self.mcfg.lame_lambda_min,
                                                            self.mcfg.lame_lambda_max,
                                                            shape=[B], return_norm=True, device=device)
        else:
            lame_lambda = torch.ones(B).to(device) * self.mcfg.lame_lambda_override
            lame_lambda_input = relative_between(self.mcfg.lame_lambda_min, self.mcfg.lame_lambda_max,
                                                 lame_lambda)
            
        return lame_lambda, lame_lambda_input
    
    
    def get_bending_coeff(self, device, B):
        if self.mcfg.bending_coeff_override is None:
            bending_coeff, bending_coeff_input = random_between_log(self.mcfg.bending_coeff_min,
                                                                    self.mcfg.bending_coeff_max,
                                                                    shape=[B], return_norm=True, device=device)
        else:
            bending_coeff = torch.ones(B).to(device) * self.mcfg.bending_coeff_override
            bending_coeff_input = relative_between_log(self.mcfg.bending_coeff_min,
                                                       self.mcfg.bending_coeff_max, bending_coeff)
            
        return bending_coeff, bending_coeff_input
    
    def add_material(self, sample, cloth_obj):

        B = sample.num_graphs
        device = sample['cloth'].pos.device
        
        density = self.get_density(device, B)
        lame_mu, lame_mu_input = self.get_lame_mu(device, B)
        lame_lambda, lame_lambda_input = self.get_lame_lambda(device, B)
        bending_coeff, bending_coeff_input = self.get_bending_coeff(device, B)
        bending_multiplier = self.mcfg.bending_multiplier
        
        
        add_field_to_pyg_batch(sample, 'lame_mu', lame_mu, 'cloth', reference_key=None, one_per_sample=True)
        add_field_to_pyg_batch(sample, 'lame_lambda', lame_lambda, 'cloth', reference_key=None,
                               one_per_sample=True)
        add_field_to_pyg_batch(sample, 'bending_coeff', bending_coeff, 'cloth', reference_key=None,
                               one_per_sample=True)
        add_field_to_pyg_batch(sample, 'lame_mu_input', lame_mu_input, 'cloth', reference_key=None, one_per_sample=True)
        add_field_to_pyg_batch(sample, 'lame_lambda_input', lame_lambda_input, 'cloth', reference_key=None,
                               one_per_sample=True)
        add_field_to_pyg_batch(sample, 'bending_coeff_input', bending_coeff_input, 'cloth', reference_key=None,
                               one_per_sample=True)
        
        
        material = Material(density, lame_mu, lame_lambda,
                            bending_coeff, bending_multiplier)
        cloth_obj.set_material(material)
        
        return sample, cloth_obj
    