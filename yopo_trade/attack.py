import torch
import config
import numpy as np

def attacking(x, label, model):
    
    mean = config.Attack_setting['mean'].to(config.Parameter_setting['device'])
    std = config.Attack_setting['std'].to(config.Parameter_setting['device'])
    criterion = torch.nn.CrossEntropyLoss().to(config.Parameter_setting['device'])

    # Change to evaluation mode 
    model.eval()

    # Initialize eta
    eta = torch.Tensor(x.shape).uniform_(-config.YOPO_setting['epsilon'], config.YOPO_setting['epsilon'])
    eta = eta.to(config.Parameter_setting['device'])
    eta = (eta - mean) / std
    eta.requires_grad_()
    x.requires_grad_()
    
    
    # Compute a new perturbation
    for i in range(config.Attack_setting['attack_iter']):
        
        # Generate adverserial samples 
        adv_inp = x + eta

        # Forward pass
        y_pred = model(adv_inp)
        loss = criterion(y_pred, label)

        # Calculate the grad
        grad_sign = torch.autograd.grad(loss, adv_inp)[0].sign()

        adv_inp = adv_inp + grad_sign * (config.Attack_setting['sigma'] / std)
        adv_inp = torch.clamp(adv_inp * std +  mean, 0, 1)

        tmp_x = x * std + mean
        
        
        tmp_eta = adv_inp - tmp_x
        tmp_eta = clip_eta(tmp_eta, norm=config.Attack_setting['norm'], eps=config.Attack_setting['epsilon'])

        eta = tmp_eta / std
        
        
    # generate the adverserial samples
    final_inp = (torch.clamp((x + eta) * std + mean, 0, 1)- mean) / std

    return final_inp


def clip_eta(eta, norm, eps, DEVICE = torch.device('cuda:0')):
    '''
    helper functions to project eta into epsilon norm ball
    :param eta: Perturbation tensor (should be of size(N, C, H, W))
    :param norm: which norm. should be in [1, 2, np.inf]
    :param eps: epsilon, bound of the perturbation
    :return: Projected perturbation
    '''

    assert norm in [1, 2, np.inf], "norm should be in [1, 2, np.inf]"

    with torch.no_grad():
        avoid_zero_div = torch.tensor(1e-12).to(DEVICE)
        eps = torch.tensor(eps).to(DEVICE)
        one = torch.tensor(1.0).to(DEVICE)

        if norm == np.inf:
            eta = torch.clamp(eta, -eps, eps)
        else:
            normalize = torch.norm(eta.reshape(eta.size(0), -1), p = norm, dim = -1, keepdim = False)
            normalize = torch.max(normalize, avoid_zero_div)

            normalize.unsqueeze_(dim = -1)
            normalize.unsqueeze_(dim=-1)
            normalize.unsqueeze_(dim=-1)

            factor = torch.min(one, eps / normalize)
            eta = eta * factor
    return eta