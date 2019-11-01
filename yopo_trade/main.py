# Main file

import argparse
import torch
from torchvision import datasets, transforms
import torch.optim as optim
import torch.nn as nn

import pre_res18
import config
import loss
import train

def main():
    # Program Config <Fill in, some of them can be set in config.py if needed>
    # parser = argparse.ArgumentParser(description='MNIST Experiment')
    # parser.add_argument('--batch-size', type=int, default=256, help='training batch size')
    

    # Load Data (CIFAR)

    # Define transformer first
    transform_train = transforms.Compose([transforms.RandomCrop(32, padding=4),transforms.RandomHorizontalFlip(),transforms.ToTensor(),])
    train_loader = torch.utils.data.DataLoader(datasets.CIFAR10('../data', train=True, download=True, transform=transform_train), batch_size=config.Parameter_setting['batch_size'], shuffle=True)
    test_loader = torch.utils.data.DataLoader(datasets.CIFAR10('../data', train=False, download=True, transform=transforms.Compose([transforms.ToTensor()])), batch_size=config.Parameter_setting['batch_size'], shuffle=True)

    # Load model
    model = pre_res18.PreActResNet18()
    model.to(config.Parameter_setting['device'])


    # Define other layer criterion/optimizer/lr_scheduler
    outer_criterion = nn.CrossEntropyLoss()
    outer_optimizer = optim.SGD(model.other_layers.parameters(), lr = config.Parameter_setting['lr'],
     momentum = config.Parameter_setting['momentum'], weight_decay = config.Parameter_setting['weight_decay'])
    outer_lr_scheduler = optim.lr_scheduler.MultiStepLR(outer_optimizer, milestones = [69, 70, 89, 90, 100, 101], gamma = 0.1)
    outer = config.training_setting(outer_criterion, outer_optimizer, outer_lr_scheduler)

    
    # Define first layer criterion/optimizer (inner loop)
    inner_criterion = loss.Hamiltonian_Loss(model.layer_one, config.Parameter_setting['weight_decay'])
    inner_optimizer = optim.SGD(model.layer_one.parameters(), lr = config.Parameter_setting['lr'],
     momentum = config.Parameter_setting['momentum'], weight_decay = config.Parameter_setting['weight_decay'])
    inner_lr_scheduler = optim.lr_scheduler.MultiStepLR(inner_optimizer, milestones =  [69, 70, 89, 90, 100, 101], gamma = 0.1)
    inner = config.training_setting(inner_criterion, inner_optimizer, inner_lr_scheduler)


    # Start training
    train.train(model, train_loader, test_loader, outer, inner)


if __name__ == "__main__":
    main()