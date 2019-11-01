# Define settings of YOPO\

import torch
import numpy as np

Parameter_setting ={
    'batch_size': 256,
    'weight_decay': 5e-4,
    'momentum': 0.9,
    'lr' :0.2,
    'max_epoch': 105,
    'print_step': 20,  # define the print frequency of the training phase 
    'eval_print_step': 10,
    'device': torch.device('cuda:0') # define the gpu used
}

YOPO_setting = {
    'm' : 3,
    'n' : 4,
    'epsilon': 0.031,  # define boundary of eta
    'sigma' : 0.007  # define the learning rate(step) of eta 
}

Attack_setting = {
    'attack_iter': 20,
    'mean': torch.tensor(np.array([0]).astype(np.float32)[np.newaxis, :, np.newaxis, np.newaxis]),
    'std': torch.tensor(np.array([1]).astype(np.float32)[np.newaxis, :, np.newaxis, np.newaxis]),
    'sigma': 0.0088889, # 2/225
    'epsilon': 0.03137, # 8/255
    'norm': np.inf
}

class training_setting(object):
    def __init__(self, criterion=None, optimizer=None, lr_scheduler=None):
        self.criterion = criterion
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler




