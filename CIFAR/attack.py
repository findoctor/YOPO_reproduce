# attack.py
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import time

from helper import get_acc

DEVICE = torch.device('cuda:0')
torch.backends.cudnn.benchmark = True

def pgd(delta, model, X, y, epsilon=8/255.0, alpha=2/255.0, num_iter=10):
    
    for t in range(num_iter):
        loss = nn.CrossEntropyLoss()(model(X+delta), y)
        loss.backward()
        #print(delta.grad.detach())
        #print(delta.grad.detach().sign())
        delta.data = (delta + alpha*delta.grad.detach().sign()).clamp(-epsilon,epsilon)
        delta.grad.zero_()
    return delta.detach()

# train the whole net with PGD adv
def train_step(diff, net, criterion, optimizer, data_generator, current_epoch):
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
        eta = torch.rand_like(x, requires_grad=True)
        eta.to(DEVICE)
        # Clean acc
        clean_pred = net(x+eta)
        clean_acc = get_acc(clean_pred, y)

        eta = pgd(eta, net, x, y) 
        
        # PGD acc
        pgd_pred = net(x+eta)
        pgd_train_acc = get_acc(pgd_pred, y)

        loss = criterion(pgd_pred, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        
        
        
        end_time = time.time()
        end_time -= diff
        time_arr.append(end_time)
        clean_err.append(1.0-clean_acc)
        robust_err.append(1.0-pgd_train_acc)
        
        
        
        print("Epoch: "+ str(current_epoch) + ", Loss: " + str(loss.item()) + ", PGD Acc: " + str(pgd_train_acc)\
        + ", Clean acc: "+ str(clean_acc) )

        # if end_time - start_time > 100:
            # break    
    return time_arr, clean_err, robust_err

def cw_l2_attack(net, input, labels, targeted=False, c=1e-4, kappa=0, max_iter=100, learning_rate=0.01, device=torch.device('cpu')):

    # Define f-function
    def f(x):
        outputs = net(x)
        one_hot_labels = torch.eye(len(outputs[0]))[labels].to(device)

        i, _ = torch.max((1-one_hot_labels)*outputs, dim=1)
        j = torch.masked_select(outputs, one_hot_labels.byte())
    
        # If targeted, optimize for making the other class most likely 
        if targeted :
            return torch.clamp(i-j, min=-kappa)
        
        # If untargeted, optimize for making the other class most likely 
        else :
            return torch.clamp(j-i, min=-kappa)

    
    input = input.to(device)     
    labels = labels.to(device)
    w = torch.zeros_like(input, requires_grad=True).to(device)
    optimizer = optim.Adam([w], lr=learning_rate)
    prev = 1e10

    for step in range(max_iter) :
        a = 1/2*(nn.Tanh()(w) + 1)

        loss1 = nn.MSELoss(reduction='sum')(a, input)
        loss2 = torch.sum(c*f(a))

        cost = loss1 + loss2

        optimizer.zero_grad()
        cost.backward()
        optimizer.step()
        # Early Stop when loss does not converge.
        if step % (max_iter//10) == 0 :
            if cost > prev :
                print('Attack Stopped due to CONVERGENCE....')
                return a
            prev = cost
        
        #print('- Learning Progress : %2.2f %%        ' %((step+1)/max_iter*100), end='\r')

    attack_input = 1/2*(nn.Tanh()(w) + 1)

    return attack_input

class IPGD(object):
    def __init__(self, eps = 6 / 255.0, sigma = 3 / 255.0, nb_iter = 20,
                 norm = np.inf, DEVICE = torch.device('cpu'),
                 mean = torch.tensor(np.array([0]).astype(np.float32)[np.newaxis, :, np.newaxis, np.newaxis]),
                 std = torch.tensor(np.array([1.0]).astype(np.float32)[np.newaxis, :, np.newaxis, np.newaxis]), random_start = True):
        '''
        :param eps: maximum distortion of adversarial examples
        :param sigma: single step size
        :param nb_iter: number of attack iterations
        :param norm: which norm to bound the perturbations
        '''
        self.eps = eps
        self.sigma = sigma
        self.nb_iter = nb_iter
        self.norm = norm
        self.criterion = torch.nn.CrossEntropyLoss().to(DEVICE)
        self.DEVICE = DEVICE
        self._mean = mean.to(DEVICE)
        self._std = std.to(DEVICE)
        self.random_start = random_start
    
    def clip_eta(self, eta, norm, eps, DEVICE = torch.device('cuda:0')):
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

    def single_attack(self, net, inp, label, eta, target = None):
        '''
        Given the original image and the perturbation computed so far, computes
        a new perturbation.
        :param net:
        :param inp: original image
        :param label:
        :param eta: perturbation computed so far
        :return: a new perturbation
        '''

        adv_inp = inp + eta

        #net.zero_grad()

        pred = net(adv_inp)
        if target is not None:
            targets = torch.sum(pred[:, target])
            grad_sign = torch.autograd.grad(targets, adv_inp, only_inputs=True, retain_graph = False)[0].sign()

        else:
            loss = self.criterion(pred, label)
            grad_sign = torch.autograd.grad(loss, adv_inp,
                                            only_inputs=True, retain_graph = False)[0].sign()

        adv_inp = adv_inp + grad_sign * (self.sigma / self._std)
        tmp_adv_inp = adv_inp * self._std +  self._mean

        tmp_inp = inp * self._std + self._mean
        tmp_adv_inp = torch.clamp(tmp_adv_inp, 0, 1) ## clip into 0-1
        #tmp_adv_inp = (tmp_adv_inp - self._mean) / self._std
        tmp_eta = tmp_adv_inp - tmp_inp
        tmp_eta = self.clip_eta(tmp_eta, norm=self.norm, eps=self.eps, DEVICE=self.DEVICE)

        eta = tmp_eta/ self._std

        return eta

    def attack(self, net, inp, label, target = None):

        if self.random_start:
            eta = torch.FloatTensor(*inp.shape).uniform_(-self.eps, self.eps)
        else:
            eta = torch.zeros_like(inp)
        eta = eta.to(self.DEVICE)
        eta = (eta - self._mean) / self._std
        net.eval()

        inp.requires_grad = True
        eta.requires_grad = True
        for i in range(self.nb_iter):
            eta = self.single_attack(net, inp, label, eta, target)
            #print(i)

        #print(eta.max())
        adv_inp = inp + eta
        tmp_adv_inp = adv_inp * self._std +  self._mean
        tmp_adv_inp = torch.clamp(tmp_adv_inp, 0, 1)
        adv_inp = (tmp_adv_inp - self._mean) / self._std

        return adv_inp

    def to(self, device):
        self.DEVICE = device
        self._mean = self._mean.to(device)
        self._std = self._std.to(device)
        self.criterion = self.criterion.to(device)