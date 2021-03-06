import torch
import torch.nn as nn
from torch.nn.modules.loss import _Loss
import torch.nn.functional as F

# Define Loss function here
class Hamiltonian_Loss(nn.Module):
    def __init__(self, layer, reg_cof = 1e-4):
        super(Hamiltonian_Loss, self).__init__()
        self.layer = layer
        self.reg_cof = 0

    def forward(self, x, p):
        return torch.sum(self.layer(x) * p)
 

class CrossEntropyWithWeightPenlty(_Loss):
    def __init__(self, module, reg_cof = 1e-4):
        super(CrossEntropyWithWeightPenlty, self).__init__()

        self.reg_cof = reg_cof
        self.criterion = nn.CrossEntropyLoss()
        self.module = module

    def __call__(self, pred, label):
        cross_loss = self.criterion(pred, label)
        weight_loss = cal_l2_norm(self.module)

        loss = cross_loss + self.reg_cof * weight_loss
        return loss

def cal_l2_norm(layer: torch.nn.Module):
 loss = 0.
 for name, param in layer.named_parameters():
     if name == 'weight':
         loss = loss + 0.5 * torch.norm(param,) ** 2

 return loss        
