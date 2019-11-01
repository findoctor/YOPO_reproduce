# Implement of the YOPO training phase

import torch
import numpy as np
import torch.nn.functional as F
import torch.nn as nn
from loss import cal_l2_norm

import copy
import config
import evaluate
from typing import List
import time
import json


def torch_accuracy(output, target, topk=(1,)):
    '''
    Calculate the batch accuracy here

    '''
    
    topn = max(topk)
    batch_size = output.size(0)

    _, pred = output.topk(topn, 1, True, True)
    pred = pred.t()

    is_correct = pred.eq(target.view(1, -1).expand_as(pred))

    ans = []
    for i in topk:
        is_correct_i = is_correct[:i].view(-1).float().sum(0, keepdim=True)
        ans.append(is_correct_i.mul_(100.0 / batch_size))

    return ans

def inner_loop(x, eta, p_s1, inner):
    '''
    Implement outer loop of YOPO algorithm here
    x: single sample data
    eta: adversary parameter
    p_s1: gradient of the loss(first layer output)
    
    '''
    # Update eta for n times
    for i in range(config.YOPO_setting['n']):

        # Get adversial x
        adv_x = torch.clamp(x + eta, 0, 1)

        # Forward pass of the first layer <need to define Hamiltonian loss later>
        y_first = inner.criterion(adv_x, p_s1)

        # Update eta
        eta = eta - config.YOPO_setting['sigma'] * torch.autograd.grad(y_first, eta)[0].sign()
        

        # control the input in a reasonable range 
        eta = torch.clamp(eta,-config.YOPO_setting['epsilon'], config.YOPO_setting['epsilon'])
        eta = torch.clamp(x + eta, 0, 1) - x
           

    # Get final adversial x
    final_x = x + eta


    # Calculate the loss <need to define Hamiltonian loss later>
    inner_loss = - (inner.criterion(final_x,  p_s1) -  config.Parameter_setting['weight_decay'] * cal_l2_norm(inner.criterion.layer) )

    # Backward pass
    inner_loss.backward(retain_graph=True)

    return eta    


def outer_loop(model, x, eta, outer, inner, label, soft_label, trades_criterion):
    '''
    Implement outer loop of YOPO algorithm here

    model: training model
    x: single sample data
    eta: adversary parameter
    label: ground truth of data x
    '''

    # For each sample, pass m times
    for j in range(config.YOPO_setting['m']):
        # Forward Pass, we don't need to calculate gradient of eta here
        y_pred = model(x + eta.detach())
        
        # Get Loss
        with torch.enable_grad():
            loss =  trades_criterion(F.log_softmax(y_pred, dim = 1), soft_label)

        # Calculate p_s^1, the gradient of the output of the first layer
        p_s1 =  -torch.autograd.grad(loss, [model.layer_one_out, ])[0]
        # Inner loop
        eta = inner_loop(x, eta, p_s1, inner)

        # Calculate the  final accuracy
        if j == config.YOPO_setting['m'] - 1:
            final_pred = model(x + eta)
            final_acc = torch_accuracy(final_pred, label, (1,))[0].item()

    # print("Total Loss: "+ str(total_loss) + " Clean_acc: "+ str(clean_acc) + " Final_acc: " + str(final_acc))        

    return eta, final_acc
      

# Need to add checkpoint/evalation function
def train(model, train_loader, test_loader, outer, inner):
    '''
    Implementing the trainning processes
    '''

    # Save the total training time
    total_time = 0
    
    # Save the initial accuracy
    '''
    print("Begin Evaluating...")
    test_clean_acc, test_final_acc = evaluate.eval(model, test_loader)
    record = {'epoch': 0, 'clean_acc': test_clean_acc, 'final_acc': test_final_acc, 'time': total_time}
    with open('result.txt', 'a') as f:
        f.write(json.dumps(record) + '\n')
    '''

    # begin training
    for epoch in range(config.Parameter_setting['max_epoch']):
        print("Now Begin Epoch: " + str(epoch))

        # change to training phase
        model.train()
        
        trades_criterion = torch.nn.KLDivLoss(size_average=False)

        # train one batch
        for step, (x, label) in enumerate(train_loader):
            
            # save the start time
            start_time  = time.time()

            # To GPU device
            x = x.to(config.Parameter_setting['device'])
            label = label.to(config.Parameter_setting['device'])

            model.eval()
            
            # define eta and soft_label
            eta = 0.001 * torch.randn(x.shape).detach().to(config.Parameter_setting['device'])
            eta.requires_grad_()
            soft_label = F.softmax(model(x), dim=1).detach()
            
            # Outer loop
            eta, final_acc = outer_loop(model, x, eta, outer, inner, label, soft_label, trades_criterion)         

            model.train()

            # Clear previous gradients
            outer.optimizer.zero_grad()
            inner.optimizer.zero_grad()

            y_pred = model(x)
            clean_acc = torch_accuracy(y_pred, label, (1,))[0].item()
            clean_loss = outer.criterion(y_pred, label)


            adv_pred = model(torch.clamp(x + eta.detach(), 0.0, 1.0))
            kl_loss = trades_criterion(F.log_softmax(adv_pred, dim=1), F.softmax(y_pred, dim=1)) / x.shape[0]

            loss = clean_loss + kl_loss
            loss.backward()

            # Update the weights
            outer.optimizer.step()
            inner.optimizer.step()

            # calculate the total time used
            end_time = time.time()
            total_time += end_time -start_time

            # print information
            if step % config.Parameter_setting['print_step'] == 0:
                print("Epoch: "+str(epoch)+" Batch step "+str(step)+" LR: "+str(outer.lr_scheduler.get_lr()[0])+" Total Loss: "+str(loss.item())
                + " Clean_acc: "+str(clean_acc)+" Final_acc: "+str(final_acc)+" Total Time: "+str(total_time))

            # Save information to txt file
            train_record = {'epoch': epoch, 'clean_acc': clean_acc, 'final_acc': final_acc, 'time': total_time}
            with open('train_log.txt', 'a') as f:
                f.write(json.dumps(train_record) + '\n')    
        
    
        # Evaluate after each epoch
        print("Begin Evaluating...")
        test_clean_acc, test_final_acc = evaluate.eval(model, test_loader)

        # Save test set information to txt file
        test_record = {'epoch': epoch, 'clean_acc': test_clean_acc, 'final_acc': test_final_acc}
        with open('test_log.txt', 'a') as f:
            f.write(json.dumps(test_record) + '\n')


        # Update learning rate
        outer.lr_scheduler.step()
        inner.lr_scheduler.step()

        # Save checkpoint<Fill in>
        


        

         


    
    




    
