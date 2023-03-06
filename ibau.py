import torch
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
import random
import copy
from itertools import repeat
from typing import List, Callable
from torch import Tensor
from torch.autograd import grad as torch_grad

'''
Based on the paper 'On the Iteration Complexity of Hypergradient Computation,' this code was created.
Source: https://github.com/prolearner/hypertorch/blob/master/hypergrad/hypergradients.py
Original Author: Riccardo Grazzi
'''

class DifferentiableOptimizer:
    def __init__(self, loss_f, dim_mult, data_or_iter=None):
        """
        Args:
            loss_f: callable with signature (params, hparams, [data optional]) -> loss tensor
            data_or_iter: (x, y) or iterator over the data needed for loss_f
        """
        self.data_iterator = None
        if data_or_iter:
            self.data_iterator = data_or_iter if hasattr(data_or_iter, '__next__') else repeat(data_or_iter)

        self.loss_f = loss_f
        self.dim_mult = dim_mult
        self.curr_loss = None

    def get_opt_params(self, params):
        opt_params = [p for p in params]
        opt_params.extend([torch.zeros_like(p) for p in params for _ in range(self.dim_mult-1) ])
        return opt_params

    def step(self, params, hparams, create_graph):
        raise NotImplementedError

    def __call__(self, params, hparams, create_graph=True):
        with torch.enable_grad():
            return self.step(params, hparams, create_graph)

    def get_loss(self, params, hparams):
        if self.data_iterator:
            data = next(self.data_iterator)
            self.curr_loss = self.loss_f(params, hparams, data)
        else:
            self.curr_loss = self.loss_f(params, hparams)
        return self.curr_loss

class GradientDescent(DifferentiableOptimizer):
    def __init__(self, loss_f, step_size, data_or_iter=None):
        super(GradientDescent, self).__init__(loss_f, dim_mult=1, data_or_iter=data_or_iter)
        self.step_size_f = step_size if callable(step_size) else lambda x: step_size

    def step(self, params, hparams, create_graph):
        loss = self.get_loss(params, hparams)
        sz = self.step_size_f(hparams)
        return gd_step(params, loss, sz, create_graph=create_graph)


def gd_step(params, loss, step_size, create_graph=True):
    grads = torch.autograd.grad(loss, params, create_graph=create_graph)
    return [w - step_size * g for w, g in zip(params, grads)]


def grad_unused_zero(output, inputs, grad_outputs=None, retain_graph=False, create_graph=False):
    grads = torch.autograd.grad(output, inputs, grad_outputs=grad_outputs, allow_unused=True,
                                retain_graph=retain_graph, create_graph=create_graph)

    def grad_or_zeros(grad, var):
        return torch.zeros_like(var) if grad is None else grad

    return tuple(grad_or_zeros(g, v) for g, v in zip(grads, inputs))

def get_outer_gradients(outer_loss, params, hparams, retain_graph=True):
    grad_outer_w = grad_unused_zero(outer_loss, params, retain_graph=retain_graph)
    grad_outer_hparams = grad_unused_zero(outer_loss, hparams, retain_graph=retain_graph)

    return grad_outer_w, grad_outer_hparams

def update_tensor_grads(hparams, grads):
    for l, g in zip(hparams, grads):
        if l.grad is None:
            l.grad = torch.zeros_like(l)
        if g is not None:
            l.grad += g


def fixed_point(params: List[Tensor],
                hparams: List[Tensor],
                K: int ,
                fp_map: Callable[[List[Tensor], List[Tensor]], List[Tensor]],
                outer_loss: Callable[[List[Tensor], List[Tensor]], Tensor],
                tol=1e-10,
                set_grad=True,
                stochastic=False) -> List[Tensor]:
    """
    Computes the hypergradient by applying K steps of the fixed point method (it can end earlier when tol is reached).
    Args:
        params: the output of the inner solver procedure.
        hparams: the outer variables (or hyperparameters), each element needs requires_grad=True
        K: the maximum number of fixed point iterations
        fp_map: the fixed point map which defines the inner problem
        outer_loss: computes the outer objective taking parameters and hyperparameters as inputs
        tol: end the method earlier when  the normed difference between two iterates is less than tol
        set_grad: if True set t.grad to the hypergradient for every t in hparams
        stochastic: set this to True when fp_map is not a deterministic function of its inputs
    Returns:
        the list of hypergradients for each element in hparams
    """

    params = [w.detach().requires_grad_(True) for w in params]
    o_loss = outer_loss(params, hparams)
    grad_outer_w, grad_outer_hparams = get_outer_gradients(o_loss, params, hparams)

    if not stochastic:
        w_mapped = fp_map(params, hparams)

    vs = [torch.zeros_like(w) for w in params]
    vs_vec = cat_list_to_tensor(vs)
    for k in range(K):
        vs_prev_vec = vs_vec

        if stochastic:
            w_mapped = fp_map(params, hparams)
            vs = torch_grad(w_mapped, params, grad_outputs=vs, retain_graph=False)
        else:
            vs = torch_grad(w_mapped, params, grad_outputs=vs, retain_graph=True)

        vs = [v + gow for v, gow in zip(vs, grad_outer_w)]
        vs_vec = cat_list_to_tensor(vs)
        if float(torch.norm(vs_vec - vs_prev_vec)) < tol:
            break

    if stochastic:
        w_mapped = fp_map(params, hparams)

    grads = torch_grad(w_mapped, hparams, grad_outputs=vs, allow_unused=True)
    grads = [g + v if g is not None else v for g, v in zip(grads, grad_outer_hparams)]

    if set_grad:
        update_tensor_grads(hparams, grads)

    return grads

def cat_list_to_tensor(list_tx):
    return torch.cat([xx.reshape([-1]) for xx in list_tx])

# I-BAU Defense
def IBAU(net, val_dataset, device="cuda"):
    '''Code from https://github.com/YiZeng623/I-BAU'''
    val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=128, num_workers=4,  shuffle=True)

    images_list, labels_list = [], []
    for index, (images, labels) in enumerate(val_dataloader):
        images_list.append(images)
        labels_list.append(labels)

    def loss_inner(perturb, model_params):
        images = images_list[0].to(device)
        labels = labels_list[0].long().to(device)
        per_img = images+perturb[0]
        per_logits = net.forward(per_img)
        loss = F.cross_entropy(per_logits, labels, reduction='none')
        loss_regu = torch.mean(-loss) +0.001*torch.pow(torch.norm(perturb[0]),2)
        return loss_regu
    
    def loss_outer(perturb, model_params):
        portion = 0.02
        images, labels = images_list[batchnum].to(device), labels_list[batchnum].long().to(device)
        patching = torch.zeros_like(images, device='cuda')
        number = images.shape[0]
        rand_idx = random.sample(list(np.arange(number)),int(number*portion))
        patching[rand_idx] = perturb[0]
        unlearn_imgs = images+patching
        logits = net(unlearn_imgs)
        criterion = nn.CrossEntropyLoss()
        loss = criterion(logits, labels)
        return loss
    
    # curr_lr = get_lr(net, val_dataloader)
    net = net.cuda()
    outer_opt = torch.optim.Adam(net.parameters(), lr=0.002)
    inner_opt = GradientDescent(loss_inner, 0.1)


    net.train()
    batch_pert = torch.zeros_like(val_dataset[0][0].unsqueeze(0), requires_grad=True, device='cuda')
    batch_lr = 0.01
    batch_opt = torch.optim.Adam(params=[batch_pert],lr=0.05)
    for round in range(5):
        
        for index, (images, labels) in enumerate(val_dataloader):
            images = images.to(device)
            ori_lab = torch.argmax(net.forward(images),axis = 1).long()
            per_logits = net.forward(images+batch_pert)
            loss = -F.cross_entropy(per_logits, ori_lab) + 0.001*torch.pow(torch.norm(batch_pert),2)
            batch_opt.zero_grad()
            loss.backward(retain_graph = True)
            batch_opt.step()
        
        pert = batch_pert * min(1, 10 / torch.norm(batch_pert))
        
        #unlearn step
        for batchnum in range(len(images_list)):
            outer_opt.zero_grad()
            fixed_point(pert, list(net.parameters()), 5, inner_opt, loss_outer) 
            outer_opt.step()
            
        print('Round:',round)
    return net