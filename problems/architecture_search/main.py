import os
import sys
import time
import glob
import numpy as np
import torch
import utils
import logging
import argparse
import torch.nn as nn
import torch.utils
import torch.nn.functional as F
import torchvision.datasets as dset
import torch.backends.cudnn as cudnn

from torch.autograd import Variable
from model_search import Network
from architect import Architect

import torch
import torch.nn as nn
import torch.nn.functional as F
from operations import *
from torch.autograd import Variable
from genotypes import PRIMITIVES
from genotypes import Genotype




CIFAR_CLASSES = 10

criterion = nn.CrossEntropyLoss()
criterion = criterion.cuda()


init_channels = 16
layers = 8
model = Network(init_channels, CIFAR_CLASSES, layers, criterion)
model = model.cuda()

seed=2
gpu = 0
np.random.seed(seed)
torch.cuda.set_device(gpu)
cudnn.benchmark = True
torch.manual_seed(seed)
cudnn.enabled=True
torch.cuda.manual_seed(seed)

optimizer = torch.optim.SGD(
  model.parameters(),
  0.025,
  momentum=0.9,
  weight_decay=3e-4)



import torchvision.transforms as transforms
def _data_transforms_cifar10():
  CIFAR_MEAN = [0.49139968, 0.48215827, 0.44653124]
  CIFAR_STD = [0.24703233, 0.24348505, 0.26158768]

  train_transform = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
  ])
  valid_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
    ])
  return train_transform, valid_transform

train_transform, valid_transform = _data_transforms_cifar10()
train_data = dset.CIFAR10(root='data/', train=True, download=True, transform=train_transform)
valid_data = dset.CIFAR10(root='data/', train=False, download=True, transform=valid_transform)

num_train = len(train_data)
indices = list(range(num_train))
split = int(np.floor(0.5 * num_train))

train_queue = torch.utils.data.DataLoader(
  train_data, batch_size=64,
  sampler=torch.utils.data.sampler.SubsetRandomSampler(indices[:split]),
  pin_memory=True, num_workers=2)

valid_queue = torch.utils.data.DataLoader(
  valid_data, batch_size=10000, shuffle=False, pin_memory=True, num_workers=2)


epochs = 50

scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer, float(epochs), eta_min=0.001)

class Args(object):
    def __init__(self, momentum, weight_decay,arch_learning_rate,arch_weight_decay):
        self.momentum=momentum
        self.weight_decay = weight_decay
        self.arch_learning_rate = arch_learning_rate
        self.arch_weight_decay = arch_weight_decay

momentum=0.9
weight_decay=3e-4
arch_learning_rate=3e-4
arch_weight_decay=1e-3

architect = Architect(model, Args(momentum,weight_decay,arch_learning_rate,arch_weight_decay))

num_ops = len(PRIMITIVES)
k = sum(1 for i in range(model._steps) for n in range(2+i))




def train(alphas_normal, alphas_reduce):
    """

    :param alphas_normal: np.array(k,num_ops)
    :param alphas_reduce: np.array(k,num_ops)
    :return: (float) loss,
        [(np.array) gradient_respect_alphas_normal, (np.array) gradient_respect_alphas_reduce]
    """

    alphas_normal = torch.from_numpy(alphas_normal).cuda()
    alphas_reduce = torch.from_numpy(alphas_reduce).cuda()

    alphas_normal = alphas_normal.float()
    alphas_reduce = alphas_reduce.float()


    model.alphas_normal = Variable(alphas_normal, requires_grad=True)
    model.alphas_reduce = Variable(alphas_reduce, requires_grad=True)
    model._arch_parameters = [
      model.alphas_normal,
      model.alphas_reduce,
    ]


    for epoch in range(epochs):
        # scheduler.step()
        # lr = scheduler.get_lr()[0]
        for step, (input, target) in enumerate(train_queue):
            objs = utils.AvgrageMeter()
            top1 = utils.AvgrageMeter()
            top5 = utils.AvgrageMeter()
            grad = utils.AvgrageMeter()
            model.train()
            n = input.size(0)

            input = Variable(input, requires_grad=False).cuda()
            target = Variable(target, requires_grad=False).cuda(async=True)

            optimizer.zero_grad()
            logits = model(input)
            loss = criterion(logits, target)

            loss.backward()
            nn.utils.clip_grad_norm(model.parameters(), 5)
            optimizer.step()

            prec1, prec5 = utils.accuracy(logits, target, topk=(1, 5))
            objs.update(loss.data[0], n)
            top1.update(prec1.data[0], n)
            top5.update(prec5.data[0], n)

            if step %50 == 0:
              logging.info('train %03d %e %f %f', step, objs.avg, top1.avg, top5.avg)

    input_search, target_search = next(iter(valid_queue))
    input_search = Variable(input_search, requires_grad=False, volatile=True).cuda()
    target_search = Variable(target_search, requires_grad=False, volatile=True).cuda(async=True)

    architect.optimizer.zero_grad()
    loss = architect.model._loss(input_search, target_search)


    for v in architect.model.arch_parameters():
      if v.grad is not None:
        v.grad.data.zero_()

    loss.backward()

    gradient = []
    for v in architect.model.arch_parameters():
        gr = v.grad
        if gr is not None:
            gradient.append(gr.numpy())
        else:
            gradient.append(gr)

    return loss, gradient


if __name__ == '__main__':
    alphas_normal = 1e-3 * np.random.randn(k, num_ops)
    alphas_reduce = 1e-3 * np.random.randn(k, num_ops)

    loss, gradient = train(alphas_normal, alphas_reduce)

    logging.info('loss is %f', loss)

    print ("gradient is: ")
    print (gradient)
