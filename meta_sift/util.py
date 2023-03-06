import torch.nn.functional as F
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
import random 

seed = 0

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

def get_results(model, data_set):
    data_loader = torch.utils.data.DataLoader(data_set, batch_size=128, num_workers=4, shuffle=False)
    model = model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (data) in enumerate(data_loader):
            inputs, targets = data[0].cuda(), data[1].cuda()
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
    return correct / total * 100

def get_NCR(dataset, poi_idx, result_idx):
    return len(set(poi_idx) & set(result_idx)) / len(result_idx) / (len(poi_idx) / len(dataset)) * 100

def norm_weight(weights):
    norm = torch.sum(weights)
    if norm >= 0.0001:
        normed_weights = weights / norm
    else:
        normed_weights = weights
    return normed_weights

class nnVent(nn.Module):
    def __init__(self, input, hidden1, hidden2, output):
        super(nnVent, self).__init__() 
        self.linear1 = nn.Linear(input, hidden1)
        self.relu1 = nn.ReLU(inplace=True)
        self.linear2 = nn.Linear(hidden1, hidden2)
        self.linear3 = nn.Linear(hidden2, output)
        
    def forward(self, x):
        x = self.linear1(x)
        x = self.relu1(x)
        x = self.linear2(x)
        x = self.relu1(x)
        out = self.linear3(x)
        return torch.sigmoid(out)

def warmup(model, optimizer, data_loader, args):
    for w_i in range(args.warmup_epochs):
        for iters, (input_train, target_train) in enumerate(data_loader):
            model.train()
            input_var,target_var = input_train.cuda(), target_train.cuda()
            optimizer.zero_grad()
            outputs = model(input_var)
            loss = F.cross_entropy(outputs, target_var)
            loss.backward()  
            optimizer.step() 
        print('Warmup Epoch {} '.format(w_i)) 
    return model, optimizer



def grad_function(grad, grad_model):
    grad_size = grad.size()
    if len(grad_size) == 4:
        reduced_grad = torch.sum(grad, dim=[1, 2, 3]).view(-1, grad_size[0])
        grad_act = grad_model(reduced_grad.detach())
        grad_act = grad_act[:, 1].view(-1)
    elif len(grad_size) == 2:
        reduced_grad = torch.sum(grad, dim=[1]).view(-1, grad_size[0])
        grad_act = grad_model(reduced_grad.detach())
        grad_act = grad_act[:, 1].view(-1)
    else:
        reduced_grad = grad.view(-1, grad_size[0])
        grad_act = grad_model(reduced_grad.detach())
        grad_act = grad_act[:, 1].view(-1)
    return grad_act

def compute_gated_grad(grads, grad_models, num_opt, num_act):
    new_grads = []
    acts = []
    gates = []
    for grad in grads[0:-num_opt]:
        new_grads.append(grad.detach())
    for g_id, grad in enumerate(grads[-num_opt:-2]):
        grad_act = grad_function(grad, grad_models[g_id])
        if grad_act > 0.5:
            new_grads.append(grad_act * grad)
        else:
            new_grads.append((1-grad_act) * grad.detach())
    acts.append(grad_act)
    for grad in grads[-2::]:
        new_grads.append(grad)
    act_loss = (torch.sum(torch.cat(acts)) - num_act)**2
    return new_grads, act_loss

def to_var(x, requires_grad=True):
    if torch.cuda.is_available():
        x = x.cuda()
    return Variable(x, requires_grad=requires_grad)

class nnGradGumbelSoftmax(nn.Module):
    def __init__(self, input, hidden, input_norm=False):
        super(nnGradGumbelSoftmax, self).__init__()
        #self.bn = MetaBatchNorm1d(input)
        self.linear1 = nn.Linear(input, hidden)
        self.relu1 = nn.PReLU()
        self.linear2 = nn.Linear(hidden, hidden)
        self.relu2 = nn.PReLU()

        self.act = nn.Linear(hidden, 2)
        self.register_buffer('weight_act', to_var(self.act.weight.data, requires_grad=True))
        self.register_buffer('bias_act', to_var(self.act.bias.data, requires_grad=True))
        self.input_norm = input_norm
        
    def forward(self, x):
        if self.input_norm:
            x_mean, x_std = x.mean(), x.std()
            x = (x-x_mean)/(x_std+1e-9)
        x = self.relu1(self.linear1(x))
        x = self.relu2(self.linear2(x))
        x = F.linear(x, self.weight_act, self.bias_act)
        y = F.gumbel_softmax(x,tau=5)
        shape = y.size()
        _, ind = y.max(dim=-1)
        y_hard = torch.zeros_like(y).view(-1, shape[-1])
        y_hard.scatter_(1, ind.view(-1, 1), 1)
        y_hard = y_hard.view(*shape)
        return y_hard

def sample_gumbel(shape, eps=1e-20):
    U = torch.cuda.FloatTensor(shape).uniform_()
    return -Variable(torch.log(-torch.log(U + eps) + eps))

def gumbel_softmax_sample(logits, temperature):
    y = logits + sample_gumbel(logits.size())
    return F.softmax(y / temperature, dim=-1)

def gumbel_softmax(logits, temperature = 5):
    """
    input: [*, n_class]
    return: [*, n_class] an one-hot vector
    """
    y = gumbel_softmax_sample(logits, temperature)
    shape = y.size()
    _, ind = y.max(dim=-1)
    y_hard = torch.zeros_like(y).view(-1, shape[-1])
    y_hard.scatter_(1, ind.view(-1, 1), 1)
    y_hard = y_hard.view(*shape)
    return (y_hard - y).detach() + y

def build_thres_model(args, weight_shape):
    hidden_dim = 128
    model = nnGradGumbelSoftmax(weight_shape[0], hidden_dim, input_norm=True)
    model.cuda()
    return model

def build_grad_models(args, model):
    grad_models = []
    grad_optimizers = []
    for param in list(model.parameters())[-args.top_k:-2]:
        param_shape = param.size()
        _grad_model = build_thres_model(args, param_shape)
        _optimizer = torch.optim.SGD(_grad_model.parameters(), args.go_lr,
            momentum=args.momentum, nesterov=args.nesterov,
            weight_decay=0)
        grad_models.append(_grad_model)
        grad_optimizers.append(_optimizer)
    return grad_models, grad_optimizers

from torch.optim.sgd import SGD


class MetaSGD(SGD):
    def __init__(self, net, *args, **kwargs):
        super(MetaSGD, self).__init__(*args, **kwargs)
        self.net = net

    def set_parameter(self, current_module, name, parameters):
        if '.' in name:
            name_split = name.split('.')
            module_name = name_split[0]
            rest_name = '.'.join(name_split[1:])
            for children_name, children in current_module.named_children():
                if module_name == children_name:
                    self.set_parameter(children, rest_name, parameters)
                    break
        else:
            current_module._parameters[name] = parameters
            # current_module._parameters.__setitem__(name, parameters)

    def meta_step(self, grads):
        group = self.param_groups[0]
        lr = group['lr']

        for (name, parameter), grad in zip(self.net.named_parameters(), grads):
            parameter.detach_()
            self.set_parameter(self.net, name, parameter.add(grad, alpha=-lr))


def named_params(model, curr_module=None, memo=None, prefix=''):
    if memo is None:
        memo = set()

    if hasattr(curr_module, 'named_leaves'):
        for name, p in curr_module.named_leaves():
            if p is not None and p not in memo:
                memo.add(p)
                yield prefix + ('.' if prefix else '') + name, p
    else:
        for name, p in curr_module._parameters.items():
            if p is not None and p not in memo:
                memo.add(p)
                yield prefix + ('.' if prefix else '') + name, p

    for mname, module in curr_module.named_children():
        submodule_prefix = prefix + ('.' if prefix else '') + mname
        for name, p in named_params(model, module, memo, submodule_prefix):
            yield name, p

def update_params(model, lr_inner, first_order=False, source_params=None, detach=False):
    '''
        official implementation
    '''
    if source_params is not None:
        for tgt, src in zip(model.named_parameters(), source_params):
            name_t, param_t = tgt
            # name_s, param_s = src
            # grad = param_s.grad
            # name_s, param_s = src
            if src is None:
                print('skip param')
                continue
            #grad = src
            tmp = param_t - lr_inner * src
            set_param(model, model, name_t, tmp)
    return model

def set_param(model, curr_mod, name, param):
    if '.' in name:
        n = name.split('.')
        module_name = n[0]
        rest = '.'.join(n[1:])
        for name, mod in curr_mod.named_children():
            if module_name == name:
                set_param(model, mod, rest, param)
                break
    else:
        setattr(getattr(curr_mod, name), 'data', param)