from dataset import create_test_dataset
from dataset import create_train_dataset
from network import create_network
from train_one_epoch import Yopo_train
from helper import CrossEntropyLossWeighted
from helper import Hamiltonian
from helper import get_acc
from attack import IPGD
from attack import cw_l2_attack
from attack import train_step
from attack import pgd

import torch
import torch.nn as nn
import torch.optim as optim
import time
import matplotlib.pyplot as plt
import pickle

batch_size = 128
num_epoch = 100
eps = 0.3

DEVICE = torch.device('cuda:0')
torch.backends.cudnn.benchmark = True

def visualize2(time_arr1, time_arr2,yopo_clean_err, yopo_robust_err, pgd_clean_err, pgd_robust_err):
    fig = plt.figure()
    ax1 = fig.add_subplot(111)

    ax1.plot(time_arr1, pgd_clean_err, color = 'red', label='PGD Clean Error')
    ax1.plot(time_arr1, pgd_robust_err, color = 'red', linestyle='-.', label='PGD Robust Error')

    ax1.plot(time_arr2, yopo_clean_err, color = 'g', label='YOPO Clean Error')
    ax1.plot(time_arr2, yopo_robust_err, color = 'g', linestyle='-.', label='YOPO Robust Error')
    plt.legend(loc='upper left');
    plt.show()

net = create_network()
net.to(DEVICE)
criterion = nn.CrossEntropyLoss().to(DEVICE)
optimizer = optim.SGD(net.parameters(), lr=5e-2, momentum=0.9, weight_decay=5e-4)

# prepare dataset
ds_train = create_train_dataset(batch_size)
ds_val = create_test_dataset(batch_size)

def visualize(time_arr, pgd_clean_err, pgd_robust_err):
    fig = plt.figure()
    ax1 = fig.add_subplot(111)

    ax1.plot(time_arr, pgd_clean_err, color = 'red', label='PGD Clean Error')
    ax1.plot(time_arr, pgd_robust_err, color = 'red', linestyle='-.', label='PGD Robust Error')
    plt.legend(loc='upper left');
    plt.show()

all_time_arr = []
pgd_clean_err = []
pgd_robust_err = []

start_time = time.time()
for i in range(num_epoch):
    print("Start Training...")

    time_arr, clean_err, robust_err = train_step(start_time, net, criterion, optimizer, ds_train, i+1)

    pgd_clean_err += clean_err
    pgd_robust_err += robust_err
    all_time_arr += time_arr


with open('pgd_clean_err.pkl', 'wb') as f:
   pickle.dump(pgd_clean_err, f)
with open('pgd_robust_err.pkl', 'wb') as f:
   pickle.dump(pgd_robust_err, f)
with open('pgd_time_arr.pkl', 'wb') as f:
   pickle.dump(all_time_arr, f)    

visualize(all_time_arr, pgd_clean_err, pgd_robust_err)