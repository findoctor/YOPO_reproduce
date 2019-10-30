from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
import time
import argparse
from torchvision import datasets, transforms
from collections import OrderedDict
from torch.utils.data import DataLoader
from helper import Hamiltonian
from helper import get_acc
'''
Functions:
* update_eta
  return new_eta
* update_theta
  Update params of first and rest layers 
  return single step loss
'''
DEVICE = torch.device('cuda:0')
torch.backends.cudnn.benchmark = True

class Yopo_train(object):
    def __init__(self, Hamilton, layer_1, layer_rest, eps, lr, m, n):
        self.layer_1 = layer_1
        self.layer_rest = layer_rest
        self.eps = eps
        self.lr = lr
        self.m = m
        self.n = n
        self.Hamilton = Hamilton
    
    def update_eta(self, x, eta, p):
        p = p.detach()
        for iter in range(self.n):
            noise_x = x + eta
            noise_x = torch.clamp(noise_x, 0, 1)
            ham = self.Hamilton(noise_x, p)
            #print(eta_grad)
            eta_grad_sign = torch.autograd.grad(ham, eta, only_inputs=True, retain_graph=False)[0].sign()
            #print(eta_grad_sign[0])
            eta = eta-(eta_grad_sign*self.lr)
            
            eta = torch.clamp(eta, -1.0 * self.eps, self.eps)
            eta = torch.clamp(x + eta, 0.0, 1.0) - x
            eta = eta.detach()  # The result will never require gradient.
            eta.requires_grad_()
            eta.retain_grad()
        newX = eta + x
        newX = torch.clamp(newX, 0, 1)
        return newX, eta.detach()
    
    def update_theta(self, diff, net, criterion, optimizer, data_generator, first_layer_optimizer, current_epoch, DEVICE=torch.device('cuda:0')):
        pbar = data_generator
        net.train()
        start_time = time.time()
        start_time -= diff
        time_arr = []
        clean_err = []
        robust_err = []
        for i, (x, y) in enumerate(pbar):
            x = x.to(DEVICE)
            y = y.to(DEVICE)
            optimizer.zero_grad()
            first_layer_optimizer.zero_grad()
            eta = torch.FloatTensor(*x.shape).uniform_(-self.eps, self.eps)
            eta = eta.to(DEVICE)
            eta.requires_grad=True
            optimizer.zero_grad()
            first_layer_optimizer.zero_grad()
            batchLoss = 0.0

            for iter in range(self.m):
                eta.requires_grad=True
                
                pred = net(x+eta)
                loss = criterion(pred, y)
                #layer_one_grad = net.conv1.weight.grad
                loss.backward()
                #net.conv1.weight.grad = layer_one_grad
                p = -1.0 * net.layer_one_out.grad 

                noise_input, eta = self.update_eta(x, eta, p)
                # Update first and rest layers weight
                # first fix first layer weights
                with torch.no_grad():
                    if iter == 0:
                        batchLoss = loss
                        # **************************************************************** Modify ****************************************************************
                        # clean_pred = net(x)
                        clean_acc = get_acc(pred, y)
                
                    if iter == self.m - 1:
                        # compute accuracy with noise data
                        yopo_pred = net(noise_input)
                        yopo_train_acc = get_acc(yopo_pred, y)
                
                #net.layer_one_out.grad.zero_()                                           # REVISION ************

            # net.conv1.weight.grad = layer_one_grad
            end_time = time.time()
            end_time -= diff
            time_arr.append(end_time)
            clean_err.append(1.0-clean_acc)
            robust_err.append(1.0-yopo_train_acc)
            
            optimizer.step()
            first_layer_optimizer.step()
            optimizer.zero_grad()
            first_layer_optimizer.zero_grad()
            print("Epoch: "+ str(current_epoch) + ", Loss: " + str(batchLoss.item() ) + ", Yopo Acc: " + str(yopo_train_acc)\
            + ", Clean acc: "+ str(clean_acc) )

            #if end_time - start_time > 10:
                # break    
        return batchLoss, yopo_train_acc, time_arr, clean_err, robust_err
