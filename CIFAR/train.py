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
num_epoch = 40
eps = 8/255.0
eta_lr = 0.008
m = 5
n = 3
weight_decay = 5e-4

DEVICE = torch.device('cuda:0')
torch.backends.cudnn.benchmark = True

# Evaluate function 
def eval_one_epoch(net, batch_generator, AttackMethod, DEVICE=torch.device('cpu')):
    net.eval()
    #pbar = tqdm(batch_generator)
    #clean_accuracy = AvgMeter()
    #adv_accuracy = AvgMeter()
    clean_accuracy = 0.0
    adv_accuracy = 0.0
    #pbar.set_description('Evaluating')
    for (data, label) in batch_generator:
        pred = net(data)
        acc = get_acc(pred, label)
        clean_accuracy = acc

        
        adv_inp = AttackMethod.attack(net, data, label)
        pred = net(adv_inp)
        acc = get_acc(pred, label)
        adv_accuracy = acc

    return clean_accuracy, adv_accuracy

net = create_network()
net.to(DEVICE)
criterion = nn.CrossEntropyLoss().to(DEVICE)
# criterion = CrossEntropyLossWeighted(net.other_layers, weight_decay)
optimizer = optim.SGD(net.other_layers.parameters(), lr =1e-2 * 1 / 5, momentum=0.9, weight_decay=5e-4 ) 
first_layer_optimizer = optim.SGD(net.layer_one.parameters(), lr = 1e-1 * 2 / 5, momentum=0.9, weight_decay=5e-4)
# prepare dataset
ds_train = create_train_dataset(batch_size)
ds_val = create_test_dataset(batch_size)

'''
# Attack and evaluate
print("Start computing cw input...")
# use the first batch as the sample
for i, (x, y) in enumerate(ds_val):
    cw_test_input = x
    cw_test_labels = y
    break

cw_noise_input = cw_l2_attack(net, cw_test_input, cw_test_labels)
print("Finish computing cw input!")
'''

def visualize(time_arr, yopo_clean_err, yopo_robust_err):
    fig = plt.figure()
    ax1 = fig.add_subplot(111)

    ax1.plot(time_arr, yopo_clean_err, 'g', label='YOPO-5-3 Clean Error')
    ax1.plot(time_arr, yopo_robust_err, color = 'g', linestyle='-.', label='YOPO-5-3 Robust Error')
    plt.legend(loc='upper right');
    plt.show()

yopo_clean_err = []
yopo_robust_err = []
all_time_arr = []

'''
file1 = open('yopo_clean_err.pkl', 'rb')
yopo_clean_err = pickle.load(file1)

file2 = open('yopo_robust_err.pkl', 'rb')
yopo_robust_err = pickle.load(file2)

file3 = open('yopo_time_arr.pkl', 'rb')
all_time_arr = pickle.load(file3)

yopo_clean_err = yopo_clean_err[0::10]
yopo_robust_err = yopo_robust_err[0::10]
all_time_arr = all_time_arr[0::10]

visualize(all_time_arr, yopo_clean_err, yopo_robust_err)
'''



start_time = time.time()
for i in range(num_epoch):
        print("Start Training...")
        
        Ham_func = Hamiltonian(net.layer_one)
        trainer = Yopo_train(Ham_func ,net.layer_one, net.other_layers, eps , eta_lr, m, n)

        
        train_loss, train_acc, time_arr, clean_err, robust_err = trainer.update_theta(start_time, net, criterion, optimizer, ds_train, first_layer_optimizer, i+1)
        
        # Test on CW attack
        #cw_output = net(cw_noise_input)
        #cw_acc = get_acc(cw_output, cw_test_labels)
        #print("CW Test ACC: "+ str(cw_acc))

        yopo_clean_err += clean_err
        yopo_robust_err += robust_err
        all_time_arr += time_arr

        
        # we can save the net for future eval function
        #clean_test_acc, adv_test_acc = eval_one_epoch(net, ds_val, pgd)
        #print("Epoch: "+ str(i+1) + ", Loss: " + str(train_loss.item() ) + ", Yopo Acc: " + str(train_acc) + "test acc"\
            # + str(adv_test_acc) )
        # save to checkpoint

        # retrive from checkpoint

with open('yopo_clean_err.pkl', 'wb') as f:
   pickle.dump(yopo_clean_err, f)
with open('yopo_robust_err.pkl', 'wb') as f:
   pickle.dump(yopo_robust_err, f)
with open('yopo_time_arr.pkl', 'wb') as f:
   pickle.dump(all_time_arr, f)

visualize(all_time_arr, yopo_clean_err, yopo_robust_err)
