from __future__ import absolute_import

import numpy as np
from copy import deepcopy

from problems.cnn_cifar10.cnn import train_nn
from problems.arxiv.generate_training_data import TrainingData
from stratified_bayesian_optimization.util.json_file import JSONFile
from stratified_bayesian_optimization.lib.parallel import Parallel
from stratified_bayesian_optimization.lib.util import (
    convert_dictionary_to_list,
)

import torch
import torchvision
import torchvision.transforms as transforms
import math
from torch.autograd import Variable
import torch.nn as nn
import random
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data

class PartialDataset(torch.utils.data.Dataset):
    def __init__(self, parent_ds, indexes):
        self.parent_ds = parent_ds
        self.length = len(indexes)
        self.indexes = indexes
        assert len(parent_ds)>=max(indexes), Exception("Parent Dataset not long enough")
        super(PartialDataset, self).__init__()

    def __len__(self):
        return self.length
    def __getitem__(self, i):
        return self.parent_ds[self.indexes[i]]

torch.manual_seed(1)

transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transform)

random.seed(1)
indexes_data = range(len(trainset))
random.shuffle(indexes_data)

n_folds = 5

n_batch = len(indexes_data) / n_folds
random_indexes = [indexes_data[i * n_batch: n_batch + i * n_batch] for i in xrange(n_folds)]

extra = 0
for j in xrange(len(indexes_data) % n_folds):
    random_indexes[j].append(indexes_data[n_batch + extra + (n_folds - 1) * n_batch])
    extra += 1

def get_training_test(fold):
    i = fold
    validation = PartialDataset(trainset, random_indexes[i])

    training_indexes = []
    for j in xrange(n_folds):
        if j != i:
            training_indexes += random_indexes[j]

    training = PartialDataset(trainset, training_indexes)

    return training, validation

def toy_example(x, cuda=False):
    """

    :param x: [int, float, int, int, int]
    :return: [float]
    """
    n_epochs = x[0]
    batch_size = x[1]
    lr = x[2]
    weight_decay = x[3]
    number_chanels_second = x[4]
    number_hidden_units = x[5]
    size_kernel = x[6]
    task = x[7]

    training, validation = get_training_test(task)

    val = train_nn(
        random_seed=1, n_epochs=n_epochs, batch_size=batch_size, lr=lr, weight_decay=weight_decay,
        number_chanels_first=6, number_chanels_second=number_chanels_second,
        number_hidden_units=number_hidden_units, size_kernel=size_kernel, cuda=cuda,
        trainset=training, testset=validation)

    return [val]


def integrate_toy_example(x):
    """

    :param x: [float, float, int, int]
    :return: [float]
    """

    points = {}
    for task in xrange(n_folds):
        point = deepcopy(x)
        point.append(task)
        points[task] = point
        # val = toy_example(point)
        # values.append(val[0])

    errors = Parallel.run_function_different_arguments_parallel(
        toy_example, points)

    values = convert_dictionary_to_list(errors)

    return [np.mean(np.array(values))]

def main(*params):
#    print 'Anything printed here will end up in the output directory for job #:', str(2)
    return toy_example(*params)

def main_objective(*params):
    # Integrate out the task parameter
    return integrate_toy_example(*params)
