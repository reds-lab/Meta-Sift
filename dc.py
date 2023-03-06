import math
import numpy as np
import torch
import torch.nn.functional as F


def DC(dataset, total_pick = 1000):

    # Calculate the average image of each class
    dataset.num_classes = len(np.unique(dataset.targets))
    class_avg = torch.zeros((dataset.num_classes, dataset[0][0].shape[0], dataset[0][0].shape[1], dataset[0][0].shape[2])) # 10 classes
    class_count = torch.zeros(dataset.num_classes)
    for x, y in dataset:
        class_avg[y] += x
        class_count[y] += 1
    for i in range(dataset.num_classes):
        class_avg[i] /= class_count[i]

    # Find the y/class number images idx that are closest to the average image
    dist = []
    for i, (x, y) in enumerate(dataset):
        dist.append(F.l1_loss(x, class_avg[y]))
    total_pick = 1000

    class_per = []
    for i in range(dataset.num_classes):
        percent = len(np.where(np.array(dataset.targets) == i)[0])/len(dataset)
        class_per.append(math.ceil(total_pick*percent))

    # create a dictionary to hold the indices for each class
    class_indices = {}
    for i in range(len(dataset)):
        if dataset.targets[i] not in class_indices:
            class_indices[dataset.targets[i]] = [i]
        else:
            class_indices[dataset.targets[i]].append(i)

    # create a list to hold the 100 lowest indices for each class
    lowest_indices = []
    for class_key in class_indices:
        indices = class_indices[class_key]
        sorted_indices = sorted(indices, key=lambda x: dist[x])
        lowest_indices.extend(sorted_indices[:class_per[class_key]])

    return lowest_indices