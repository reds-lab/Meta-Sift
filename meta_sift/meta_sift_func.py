import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.optim import Optimizer
import torch.backends.cudnn as cudnn
from torch.utils.data import TensorDataset, DataLoader, Dataset
import numpy as np
from tqdm import tqdm
import imageio
import torchvision.transforms as transforms
import copy
from .poi_util import *
import math
from models import ResNet18, PreActResNet18
from .util import *

device = 'cuda'

def get_dataset(args):
    train_transform = transforms.Compose([
        transforms.Resize(32),
        transforms.ToTensor(),])
    trainset = h5_dataset(args.dataset_root, True, None)
    args.num_classes = len(np.unique(np.array(trainset.targets)))
    train_poi_set, poi_idx = poi_dataset(trainset, poi_methond=args.corruption_type, transform=train_transform, poi_rates=args.corruption_ratio,random_seed=args.random_seed, tar_lab=args.tar_lab)

    # test_trans = transforms.Compose([
    #     transforms.Resize(32),
    #     transforms.RandomCrop(32, padding=4),
    #     transforms.RandomHorizontalFlip(),
    #     transforms.ToTensor(),
    # ])
    # test_poi_set, poi_idx = poi_dataset(trainset, poi_methond=args.corruption_type, transform=test_trans, poi_rates=args.corruption_ratio,random_seed=args.random_seed, tar_lab=args.tar_lab)
    return train_poi_set, poi_idx
    

def build_training(args):
    model = ResNet18(num_classes = args.num_classes).cuda()
    optimizer_a = torch.optim.AdamW(model.parameters(), lr=2.5e-5 * args.batch_size / 32, betas=(0.9, 0.95), weight_decay=0.05)

    vnet = nnVent(1, 100, 150, 1).cuda()
    optimizer_c = torch.optim.SGD(vnet.parameters(), args.v_lr)
    return model, optimizer_a, vnet, optimizer_c

def train_sifter(args, dataset):
    train_dataloader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, num_workers=args.num_workers, pin_memory=True, shuffle=True)
    criterion = nn.CrossEntropyLoss(reduction='none').cuda()
    mnet_list = []
    vnet_list = []
    for i in range(args.repeat_rounds):
        print("-----------Training sifter number: " + str(i) + "-----------")
        model, optimizer_a, vnet, optimizer_c = build_training(args)
        grad_models, grad_optimizers = build_grad_models(args, model)
        model, optimizer_a = warmup(model, optimizer_a, train_dataloader, args)
        raw_meta_model = ResNet18(num_classes = args.num_classes).cuda()
        for i in range(args.res_epochs):
            train_iter = tqdm(enumerate(train_dataloader), total=int(len(dataset)/args.batch_size)+1)
            for iteration, (input_train, target_train) in train_iter:
                input_var,target_var = input_train.cuda(), target_train.cuda()

                # virtual training
                meta_model = copy.deepcopy(raw_meta_model)
                meta_model.load_state_dict(model.state_dict())
                y_f_hat = meta_model(input_var)
                cost = criterion(y_f_hat, target_var)
                cost_v = torch.reshape(cost, (len(cost), 1))
                v_lambda = vnet(cost_v.data)
                v_lambda = v_lambda.view(-1)
                v_lambda = norm_weight(v_lambda)
                l_f_meta = torch.sum(v_lambda * cost)

                # virtual backward & update
                meta_model.zero_grad()
                grads = torch.autograd.grad(l_f_meta,(meta_model.parameters()),create_graph=True, allow_unused=True)

                # compute gradient gates and update the model
                new_grads,_ = compute_gated_grad(grads, grad_models, args.top_k, args.num_act)
                pseudo_optimizer = MetaSGD(meta_model, meta_model.parameters(), lr=args.meta_lr)
                pseudo_optimizer.load_state_dict(optimizer_a.state_dict())
                pseudo_optimizer.meta_step(new_grads)

                res_y_f_hat = meta_model(input_var)
                res_cost = criterion(res_y_f_hat, target_var)
                res_cost_v = torch.reshape(res_cost, (len(res_cost), 1))
                res_v_bf_lambda = vnet(res_cost_v.data)
                res_v_bf_lambda = res_v_bf_lambda.view(-1)
                res_v_lambda = 1-res_v_bf_lambda
                res_v_lambda = norm_weight(res_v_lambda)

                valid_loss = -torch.sum((res_v_lambda) * res_cost)
                
                optimizer_c.zero_grad()
                for go in grad_optimizers:
                    go.zero_grad()
                valid_loss.backward()
                optimizer_c.step()
                for go in grad_optimizers:
                    go.step()
                del grads, new_grads

                #actuall update
                y_f = model(input_var)
                cost_w = criterion(y_f, target_var)
                cost_v = torch.reshape(cost_w, (len(cost_w), 1))

                with torch.no_grad():
                    w_new = vnet(cost_v)

                w_new = w_new.view(-1)#
                w_new = norm_weight(w_new)
                l_f = torch.sum(w_new * cost_w)

                optimizer_a.zero_grad()
                l_f.backward()
                optimizer_a.step()
        vnet_list.append(copy.deepcopy(vnet))
        mnet_list.append(copy.deepcopy(model))
    return vnet_list, mnet_list

def test_sifter(args, dataset, vnet_list, mnet_list):
    test_dataloader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)
    criterion = nn.CrossEntropyLoss(reduction='none').cuda()
    v_res = np.zeros((args.repeat_rounds, len(dataset)), dtype=np.float32)
    for i in range(args.repeat_rounds):
        v = np.zeros((len(dataset)), dtype=np.float32)
        meta_model = mnet_list[i]
        meta_model.eval()
        vnet = vnet_list[i]
        # meta_model.train()
        for b, (images, labels) in tqdm(enumerate(test_dataloader),total=int(len(dataset) / args.batch_size)):
            input_var, target_var = images.cuda(), labels.cuda()
            y_f_hat = meta_model(input_var)
            cost = criterion(y_f_hat, target_var)
            cost_v = torch.reshape(cost, (len(cost), 1))
            
            v_lambda = vnet(cost_v.data)
            batch_size = v_lambda.size()[0]
            v_lambda = v_lambda.view(-1)
            
            zero_idx = b*batch_size
            v[zero_idx:zero_idx+batch_size] = v_lambda.detach().cpu().numpy()
        
        v_res[i,:] = copy.deepcopy(v)
    return v_res

def get_sifter_result(args, dataset, v_res):
    total_pick = 1000

    class_per = []
    for i in np.unique(dataset.targets):
        percent = len(np.where(np.array(dataset.targets) == i)[0])/len(dataset)
        class_per.append(math.ceil(total_pick*percent))

    new_mat = np.mean(v_res,axis=0)
    new_idx = []
    for i in range(args.num_classes):
        pick_p = class_per[i]
        tar_idx = np.where(np.array(dataset.targets) == i)[0]
        p_tail = (len(tar_idx) - pick_p)/len(tar_idx)*100
        cutting = np.percentile(new_mat[tar_idx],p_tail)
        tar_new_idx = np.where(new_mat[tar_idx]>=cutting)[0]
        if tar_new_idx.shape[0] > pick_p:
            tar_new_idx = tar_new_idx[:pick_p]
        new_idx.append(tar_idx[tar_new_idx])
    new_idx = [i for item in new_idx for i in item]
    new_idx = np.array(new_idx)
    return new_idx



def meta_sift(args, dataset):
    set_seed(args.random_seed)
    test_poi_set = copy.deepcopy(dataset)
    test_poi_set.transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomRotation(25),
        transforms.ToTensor(),
    ])
    vnet_list, mnet_list = train_sifter(args, dataset)
    v_res = test_sifter(args, test_poi_set, vnet_list, mnet_list)
    return get_sifter_result(args, test_poi_set, v_res)