from ast import Not
import logging
import os
from models import ResNet18
import torch.nn.functional as F
import numpy as np
import torch
from torch.utils.data import Dataset, Subset
import cv2 as cv
import torch.nn as nn
from collections import OrderedDict
import copy
from PIL import Image
from tqdm import tqdm
import random
from torch.autograd import Variable
import imageio

seed = 0

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


class my_subset(Dataset):
    r"""
    Subset of a dataset at specified indices.

    Arguments:
        dataset (Dataset): The whole Dataset
        indices (sequence): Indices in the whole set selected for subset
        labels(sequence) : targets as required for the indices. will be the same length as indices
    """
    def __init__(self, dataset, indices,  transform):
        self.indices = indices
        if not isinstance(self.indices, list):
            self.indices = list(self.indices)
        self.dataset = Subset(dataset, self.indices)
        self.transform = transform

    def __getitem__(self, idx):
        image = self.dataset[idx][0]
        label = self.dataset[idx][1]
        if self.transform != None:
            # image = Image.fromarray(image.astype(np.uint8))
            image = self.transform(image)
        return (image, label)

    def __len__(self):
        return len(self.indices)

class delete_subset(Dataset):
    r"""
    Subset of a dataset at specified indices.

    Arguments:
        dataset (Dataset): The whole Dataset
        indices (sequence): Indices in the whole set selected for subset
        labels(sequence) : targets as required for the indices. will be the same length as indices
    """
    def __init__(self, dataset, indices):
        self.indices = indices
        if not isinstance(self.indices, list):
            self.indices = list(self.indices)
        self.data = copy.deepcopy(self.dataset.data) 
        self.targets = copy.deepcopy(self.dataset.targets)
        self.data = np.delete(self.data,indices,0)
        self.targets = np.delete(self.targets,indices)

    def __getitem__(self, idx):
        image = self.data[idx]
        label = self.targets[idx]
        return (image, label)

    def __len__(self):
        return len(self.targets)
        

class data_prefetcher():
    def __init__(self, loader):
        self.loader = iter(loader)
        self.stream = torch.cuda.Stream()
        self.mean = torch.tensor([0.485 * 255, 0.456 * 255, 0.406 * 255]).cuda().view(1,3,1,1)
        self.std = torch.tensor([0.229 * 255, 0.224 * 255, 0.225 * 255]).cuda().view(1,3,1,1)
        self.preload()

    def preload(self):
        try:
            self.next_input, self.next_target = next(self.loader)
        except StopIteration:
            self.next_input = None
            self.next_target = None
            return
        with torch.cuda.stream(self.stream):
            self.next_input = self.next_input.cuda(non_blocking=True)
            self.next_target = self.next_target.cuda(non_blocking=True)
            self.next_input = self.next_input.float()
            self.next_input = self.next_input.sub_(self.mean).div_(self.std)
            
    def next(self):
        torch.cuda.current_stream().wait_stream(self.stream)
        input = self.next_input
        target = self.next_target
        self.preload()
        return input, target
    
def apply_noise_patch(noise,images,offset_x=0,offset_y=0,mode='change',padding=20,position='fixed'):
    '''
    noise: torch.Tensor(1, 3, pat_size, pat_size)
    images: torch.Tensor(N, 3, 512, 512)
    outputs: torch.Tensor(N, 3, 512, 512)
    '''
    length = images.shape[2] - noise.shape[2]
    if position == 'fixed':
        wl = offset_x
        ht = offset_y
    else:
        wl = np.random.randint(padding,length-padding)
        ht = np.random.randint(padding,length-padding)
    if len(images.shape) == 3:
        noise_now = np.copy(noise[0,:,:,:])
        wr = length-wl
        hb = length-ht
        m = nn.ZeroPad2d((wl, wr, ht, hb))
        if(mode == 'change'):
            images[:,ht:ht+noise.shape[2],wl:wl+noise.shape[3]] = 0
            images[:,ht:ht+noise.shape[2],wl:wl+noise.shape[3]] = noise_now
        else:
            images += noise_now
    else:
        for i in range(images.shape[0]):
            noise_now = np.copy(noise)
            wr = length-wl
            hb = length-ht
            m = nn.ZeroPad2d((wl, wr, ht, hb))
            if(mode == 'change'):
                images[i:i+1,:,ht:ht+noise.shape[2],wl:wl+noise.shape[3]] = 0
                images[:,ht:ht+noise.shape[2],wl:wl+noise.shape[3]] = noise_now
            else:
                images[i:i+1] += noise_now
    return images

class noisy_label(Dataset):
    def __init__(self, dataset, indices, num_classes, transform, seed):
        set_seed(seed)
        print('Random seed is: ', seed)
        self.dataset = dataset
        self.indices = indices
        self.data = copy.deepcopy(self.dataset.data) 
        self.targets = copy.deepcopy(self.dataset.targets)
        self.num_classes = num_classes
        self.transform = transform

        allos_idx = []
        for i in range(num_classes):
            allowed_values = list(range(num_classes))
            allowed_values.remove(i)
            allos_idx.append(allowed_values)
        for i in range(len(indices)):
            tar_lab = self.targets[indices[i]]
            self.targets[indices[i]] = random.choice(allos_idx[tar_lab])

    def __getitem__(self, idx):
        image = self.data[idx]
        target = self.targets[idx]
        if self.transform is not None:
            image = Image.fromarray(image.astype(np.uint8))
            image = self.transform(image)
        return (image, target, 0)

    def __len__(self):
        return len(self.dataset)

class flipping_label(Dataset):
    def __init__(self, dataset, indices, tar_lab, transform, seed):
        set_seed(seed)
        self.dataset = dataset
        self.indices = indices
        self.data = copy.deepcopy(self.dataset.data) 
        self.targets = copy.deepcopy(self.dataset.targets)
        self.tar_lab = tar_lab
        for i in self.indices:
            self.targets[i] = self.tar_lab
        self.transform = transform

    def __getitem__(self, idx):
        image = self.data[idx]
        target = self.targets[idx]
        if self.transform is not None:
            image = Image.fromarray(image.astype(np.uint8))
            image = self.transform(image)
        return (image, target)

    def __len__(self):
        return len(self.dataset)

class change_label(Dataset):
    def __init__(self, dataset, tar_lab):
        set_seed(seed)
        self.dataset = dataset
        self.data = copy.deepcopy(self.dataset.data) 
        self.targets = copy.deepcopy(self.dataset.targets)
        self.indices = np.where(np.array(self.targets)==tar_lab[0])[0]
        self.tar_lab = tar_lab

    def __getitem__(self, idx):
        image = self.dataset[self.indices[idx]][0]
        return (image, self.tar_lab[1])

    def __len__(self):
        return self.indices.shape[0]

class posion_image_nottar_label(Dataset):
    def __init__(self, dataset,indices,noise,lab):
        self.dataset = dataset
        self.indices = indices
        self.noise = noise
        self.lab = lab

    def __getitem__(self, idx):
        image = self.dataset[idx][0]
        label = self.dataset[idx][1]
        if idx in self.indices:
            image = torch.clamp(apply_noise_patch(self.noise,image,mode='add'),-1,1)
            label = self.lab
        return (image, label)

    def __len__(self):
        return len(self.dataset)

class posion_image_all2all(Dataset):
    def __init__(self, dataset,noise,poi_list,num_classes, transform):
        self.dataset = dataset
        self.data = dataset.data
        self.targets = self.dataset.targets
        self.poi_list = poi_list
        self.noise = noise
        self.num_classes = num_classes
        self.transform = transform
        for i in self.poi_list:
            self.targets[i] = self.targets[i] + 1
            if self.targets[i] == self.num_classes:
                self.targets[i] = 0

    def __getitem__(self, idx):
        image = self.data[idx]
        label = self.targets[idx]
        if idx in self.poi_list:
            pat_size = 4
            image[32 - pat_size:32, 32 - pat_size:32, :] = 255
        
        if self.transform is not None:
            image = Image.fromarray(image.astype(np.uint8))
            image = self.transform(image)
        return (image, label, 0)

    def __len__(self):
        return len(self.dataset)

class posion_noisy_all2all(Dataset):
    def __init__(self, dataset,noise,poi_list,num_classes, transform):
        self.dataset = dataset
        self.data = copy.deepcopy(self.dataset.data) 
        self.targets = copy.deepcopy(self.dataset.targets)
        self.poi_list = poi_list
        self.noise = noise
        self.num_classes = num_classes
        self.transform = transform
        for i in self.poi_list:
            self.targets[i] = self.targets[i] + 1
            if self.targets[i] == self.num_classes:
                self.targets[i] = 0

    def __getitem__(self, idx):
        image = self.data[idx]
        label = self.targets[idx]
        poi = 0
        if idx in self.poi_list:
            poi = 1
            image = image.astype(int)
            image += self.noise
            image = np.clip(image,0,255)
        
        if self.transform is not None:
            image = Image.fromarray(image.astype(np.uint8))
            image = self.transform(image)
        return (image, label, poi)

    def __len__(self):
        return len(self.dataset)

class posion_image_all2one(Dataset):
    def __init__(self, dataset,poi_list,tar_lab, transform):
        self.dataset = dataset
        self.targets = copy.deepcopy(self.dataset.targets)
        self.poi_list = poi_list
        self.tar_lab = tar_lab
        self.transform = transform
        for i in self.poi_list:
            self.targets[i] = self.tar_lab

    def __getitem__(self, idx):
        poi = 0
        image = self.dataset[idx][0]
        label = self.targets[idx]
        if idx in self.poi_list:
            poi = 1
            pat_size = 4
            image[32 - pat_size:32, 32 - pat_size:32, :] = 255
        if self.transform is not None:
            image = Image.fromarray(image.astype(np.uint8))
            image = self.transform(image)
        return (image, label)

    def __len__(self):
        return len(self.dataset)

class posion_noisy_all2one(Dataset):
    def __init__(self, dataset,poi_list,tar_lab, transform, noisy):
        self.dataset = dataset
        self.data = copy.deepcopy(self.dataset.data) 
        self.targets = copy.deepcopy(self.dataset.targets)
        self.poi_list = poi_list
        self.tar_lab = tar_lab
        self.transform = transform
        self.noisy = noisy
        for i in self.poi_list:
            self.targets[i] = self.tar_lab

    def __getitem__(self, idx):
        image = self.data[idx]
        label = self.targets[idx]
        poi = 0
        if idx in self.poi_list:
            poi = 1
            image = image.astype(int)
            image += self.noisy
            image = np.clip(image,0,255)
        if self.transform is not None:
            image = Image.fromarray(image.astype(np.uint8))
            image = self.transform(image)
        return (image, label, poi)

    def __len__(self):
        return len(self.dataset)

class posion_image(Dataset):
    def __init__(self, dataset,indices,noise, transform):
        self.dataset = dataset
        self.indices = indices
        self.noise = noise
        self.data = copy.deepcopy(self.dataset.data) 
        self.targets = copy.deepcopy(self.dataset.targets)
        self.transform = transform

    def __getitem__(self, idx):
        image = self.data[idx]
        poi = 0
        if self.transform is not None:
            image = Image.fromarray(image.astype(np.uint8))
            image = self.transform(image)
        if idx in self.indices:
            poi = 1
            image += self.noise
            # image = torch.clip(image,0,255)
        label = self.targets[idx]
        return (image, label,poi)

    def __len__(self):
        return len(self.dataset)
    
class posion_image_label(Dataset):
    def __init__(self, dataset,indices,noise,target,transform):
        self.dataset = dataset
        self.indices = indices
        self.noise = noise
        self.target = target
        self.targets = copy.deepcopy(self.dataset.targets)
        self.data = copy.deepcopy(self.dataset.data) 
        self.transform = transform

    def __getitem__(self, idx):
        image = self.data[idx]
        if idx in self.indices:
            image = image.astype(int)
            image += self.noise
            image = np.clip(image,0,255)
        if self.transform is not None:
            image = Image.fromarray(image.astype(np.uint8))
            image = self.transform(image)
        #label = self.dataset[idx][1]
        return (image, self.target)

    def __len__(self):
        return len(self.indices)
    
class get_labels(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset

    def __getitem__(self, idx):
        return self.dataset[idx][1]

    def __len__(self):
        return len(self.dataset)

class concoct_dataset(torch.utils.data.Dataset):
    def __init__(self, target_dataset,outter_dataset):
        self.idataset = target_dataset
        self.odataset = outter_dataset

    def __getitem__(self, idx):
        if idx < len(self.odataset):
            img = self.odataset[idx][0]
            labels = self.odataset[idx][1]
        else:
            img = self.idataset[idx-len(self.odataset)][0]
            #labels = torch.tensor(len(self.odataset.classes),dtype=torch.long)
            labels = len(self.odataset.classes)
        #label = self.dataset[idx][1]
        return (img,labels)

    def __len__(self):
        return len(self.idataset)+len(self.odataset)

def inverse_normalize(img, mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)):
    for i in range(len(mean)):
        img[:,:,i] = img[:,:,i]*std[i]+mean[i]
    return img

def poi_dataset(Dataset, poi_methond='badnets', transform=None, tar_lab = 0, poi_rates = 0.2, random_seed = 0, noisy = None):
    set_seed(random_seed)
    label = Dataset.targets
    num_classes = len(np.unique(label))
    if poi_methond == 'targeted_label_filpping':
        poi_idx = []
        current_label = np.where(np.array(label)==tar_lab[0])[0]
        samples_idx = np.random.choice(current_label, size=int(current_label.shape[0] * poi_rates), replace=False)
        poi_idx.extend(samples_idx)
        posion_dataset = flipping_label(Dataset, poi_idx, tar_lab[1], transform, random_seed)
        return posion_dataset, poi_idx
    elif poi_methond == 'badnets':
        current_label = np.where(np.array(label)!=tar_lab)[0]
        poi_idx = np.random.choice(current_label, size=int(len(Dataset) * poi_rates), replace=False)
        posion_dataset = posion_image_all2one(Dataset, poi_idx, tar_lab, transform)
        return posion_dataset, poi_idx
    elif poi_methond == 'narcissus':
        current_label = np.where(np.array(label)==tar_lab)[0]
        poi_idx = np.random.choice(current_label, size=int(current_label.shape[0] * poi_rates), replace=False)
        if noisy is None:
            noisy = np.load('/home/minzhou/public_html/dataeval/github_repo/tar5_narcissus.npy')
        posion_dataset = posion_image(Dataset, poi_idx, noisy, transform)
        return posion_dataset, poi_idx
    else:
        raise Exception("Not a valid poison method!")

import h5py
class h5_dataset(Dataset):
    def __init__(self, path, train, transform):
        f = h5py.File(path,'r') 
        if train:
            self.data = np.vstack((np.asarray(f['X_train']),np.asarray(f['X_val']))).astype(np.uint8)
            self.targets = list(np.argmax(np.vstack((np.asarray(f['Y_train']),np.asarray(f['Y_val']))),axis=1))
        else:
            self.data = np.asarray(f['X_test']).astype(np.uint8)
            self.targets = list(np.argmax(np.asarray(f['Y_test']),axis=1))
        self.transform = transform
        self.num_classes = len(np.unique(self.targets))

    def __getitem__(self, idx):
        image = self.data[idx]
        label = self.targets[idx]
        if self.transform is not None:
            image = Image.fromarray(image.astype(np.uint8))
            image = self.transform(image)
        return (image, label)

    def __len__(self):
        return len(self.targets)

cfg = {'small_VGG16': [32, 32, 'M', 64, 64, 'M', 128, 128, 'M'],}
drop_rate = [0.3,0.4,0.4]

class VGG(nn.Module):
    def __init__(self, vgg_name):
        super(VGG, self).__init__()
        self.features = self._make_layers(cfg[vgg_name])
        self.classifier = nn.Linear(2048, 43)

    def forward(self, x):
        out = self.features(x)
        out = out.view(out.size(0), -1)
        out = self.classifier(out)
        return out

    def _make_layers(self, cfg):
        layers = []
        in_channels = 3
        key = 0
        for x in cfg:
            if x == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2),
                           nn.Dropout(drop_rate[key])]
                key += 1
            else:
                layers += [nn.Conv2d(in_channels, x, kernel_size=3, padding=1),
                           nn.BatchNorm2d(x),
                           nn.ELU(inplace=True)]
#                            nn.ReLU(inplace=True)]
                in_channels = x
#         layers += [nn.AvgPool2d(kernel_size=1, stride=1)]
        return nn.Sequential(*layers)

def get_validset(dataset, poi_idx, total_num = 1000, poi_num = 0):
    clean_idx = list(set(np.arange(len(dataset))) - set(poi_idx))
    subset_indices1 = np.random.choice(clean_idx, size=total_num - poi_num, replace=False)
    subset_indices2 = np.random.choice(poi_idx, size=poi_num, replace=False)
    subset_indices = np.concatenate([subset_indices1, subset_indices2])
    subset = torch.utils.data.Subset(dataset, subset_indices)
    return subset