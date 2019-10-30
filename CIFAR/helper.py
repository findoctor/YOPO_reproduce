
# helper.py
# define Hanmilton function and CrossEntropyWeighted Loss class
import torch
import torch.nn as nn
from torch.nn.modules.loss import _Loss
import torch.nn.functional as F

'''
def Hamilton(X, p, layers):
    ft = layers(X)
    return torch.sum(ft*p)
'''

# revise 1: use Hamilton as son class of _Loss
class Hamiltonian(_Loss):

    def __init__(self, layer, reg_cof = 1e-4):
        super(Hamiltonian, self).__init__()
        self.layer = layer
        self.reg_cof = 0


    def forward(self, x, p):

        y = self.layer(x)
        H = torch.sum(y * p)
        return H

def get_acc(yopo_pred, y):
    # yopo_pred 128,10
    # y 128,
    num_samples = yopo_pred.size()[0]
    _, pred = torch.max(yopo_pred.data, 1)
    hit = (pred == y).sum()
    return float(hit) / num_samples

# define CrossEntropyLoss with weight decay
class CrossEntropyLossWeighted(_Loss):
    def __init__(self, net, coef=1e-4):
        super(CrossEntropyLossWeighted, self).__init__()
        self.net=net
        self.coef=coef
    def __call__(self, pred, target):
        rule = nn.CrossEntropyLoss()
        loss1 = rule(pred, target)
        loss2 = self.weightedLoss(self.net)
        return loss1 + self.coef * loss2
    def weightedLoss(self, layers):
        loss = 0.0
        for name, param in layers.named_parameters():
            if name == 'weight':
                loss += 0.5 * torch.norm(param,) ** 2
        return loss