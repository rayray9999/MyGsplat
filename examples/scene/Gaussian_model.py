import torch
import numpy as np
from utils.helper import get_expon_lr_func
from torch import nn
import os
import math
class GaussianModel:  
    def __init__(self, num_points, device):  
        self.num_points = num_points  
        self.device = device  
        self._init_gaussians() 
        self.optimizer = None  
        self.xyz_scheduler_args = None   
  
    def _init_gaussians(self):  
        """Random gaussians"""  
        bd = 2  
        self.means = bd * (torch.rand(self.num_points, 3, device=self.device) - 0.5)  
        self.scales = torch.rand(self.num_points, 3, device=self.device)  
        self.rgbs = torch.rand(self.num_points, 3, device=self.device)  
  
        u = torch.rand(self.num_points, 1, device=self.device)  
        v = torch.rand(self.num_points, 1, device=self.device)  
        w = torch.rand(self.num_points, 1, device=self.device)  
  
        self.quats = torch.cat(  
            [  
                torch.sqrt(1.0 - u) * torch.sin(2.0 * math.pi * v),  
                torch.sqrt(1.0 - u) * torch.cos(2.0 * math.pi * v),  
                torch.sqrt(u) * torch.sin(2.0 * math.pi * w),  
                torch.sqrt(u) * torch.cos(2.0 * math.pi * w),  
            ],  
            -1,  
        )  
        self.opacities = torch.ones((self.num_points, 1), device=self.device)  
        # Set requires_grad = True  
        self.means.requires_grad = True  
        self.scales.requires_grad = True  
        self.quats.requires_grad = True  
        self.rgbs.requires_grad = True  
        self.opacities.requires_grad = True 

        #add some parameter (might be some redundent)
        self.iterations = 4_000
        self.position_lr_init = 0.00016
        self.position_lr_final = 0.0000016
        self.position_lr_delay_mult = 0.01
        self.position_lr_max_steps = 4_000
        self.feature_lr = 0.0025
        self.opacity_lr = 0.05
        self.scaling_lr = 0.005
        self.rotation_lr = 0.001
        self.percent_dense = 0.01
        self.densification_interval = 100
        self.opacity_reset_interval = 3000
        self.densify_from_iter = 500
        self.densify_until_iter = 15_000
        self.densify_grad_threshold = 0.0002
        self.random_background = False

    def training_setup(self):
        self.xyz_gradient_accum = torch.zeros((self.means.shape[0], 1), device=self.device)  
        self.denom = torch.zeros((self.means.shape[0], 1), device=self.device)  
  
        l = [  
            {'params': [self.means], 'lr': self.position_lr_init , "name": "xyz"},  
            {'params': [self.rgbs], 'lr': self.feature_lr, "name": "f_dc"},  
            {'params': [self.quats], 'lr': self.rotation_lr, "name": "rotation"},  
            {'params': [self.opacities], 'lr': self.opacity_lr, "name": "opacity"},  
            {'params': [self.scales], 'lr': self.scaling_lr, "name": "scaling"}  
        ]

        self.optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15)  
        self.xyz_scheduler_args = get_expon_lr_func(lr_init=self.position_lr_init,  
                                                    lr_final=self.position_lr_final,  
                                                    lr_delay_mult=self.position_lr_delay_mult,  
                                                    max_steps=self.position_lr_max_steps)
        
    def update_learning_rate(self, iteration):  
        ''' Learning rate scheduling per step '''  
        if self.optimizer is not None:  
            for param_group in self.optimizer.param_groups:  
                if param_group["name"] == "xyz":  
                    lr = self.xyz_scheduler_args(iteration)  
                    param_group['lr'] = lr  
                    return lr  
        else:  
            print("Optimizer is not defined.") 