from __future__ import print_function

import argparse
import os
import random
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import torchvision.utils as vutils
from torch.autograd import Variable
import torch.nn.functional as F
import math

import numpy as np
from stratified_bayesian_optimization.initializers.log import SBOLog
from stratified_bayesian_optimization.util.json_file import JSONFile

logger = SBOLog(__name__)


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
       # self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2(x), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
      #  x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)




args_opt = {}
args_opt['test_batch_size'] = 1000
args_opt['train_batch_size'] = 1000
args_opt['manualSeed'] = 5
args_opt['momentum'] = 0.0

random.seed(args_opt['manualSeed'])
torch.manual_seed(args_opt['manualSeed'])

kwargs = {}
train_test = torch.utils.data.DataLoader(
        datasets.MNIST('../data', train=True, download=True,
                       transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                       ])),
        batch_size=args_opt['test_batch_size'], shuffle=False, **kwargs)

training_loaders = torch.utils.data.DataLoader(
    datasets.MNIST('../data', train=True, download=True,
                   transform=transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0.1307,), (0.3081,))
                   ])),
    batch_size=args_opt['train_batch_size'], shuffle=False, **kwargs)


train_dict = {}
for batch_idx, (data, target) in enumerate(training_loaders):
    train_dict[batch_idx] = (data, target)


def train_nn(model, n_epochs=20, name_model='a.json', random_seed=1):
    np.random.seed(1)
    values = {}
    for epoch in range(1, n_epochs + 1):
        logger.info('epoch is %d' % epoch)
        values[epoch] = []
        optimizer = optim.SGD(model.parameters(), lr=(0.1 / np.sqrt(epoch)),
                              momentum=args_opt['momentum'])
        shuffled_order = np.arange(len(train_dict))
        np.random.shuffle(shuffled_order)
        for i in shuffled_order:
            total = 0
            correct = 0
            for data in train_test:
                images, labels = data
                outputs = model(Variable(images))
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum()
            values[epoch].append(100. * correct / float(total))

            logger.info('Error in epoch %d is:' % epoch)
            logger.info(100. * correct / float(total))

            data, target = train_dict[i]
            data, target = Variable(data), Variable(target)
            optimizer.zero_grad()
            output = model(data)
            loss = F.nll_loss(output, target)
            loss.backward()
            optimizer.step()

        f_name = 'data/multi_start/neural_networks/training_results/'
        f_name += name_model
        JSONFile.write(values, f_name)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('nn', help='e.g. neural network: an intenger from 0 to 10')
    parser.add_argument('n_epochs', help='e.g. 20')
    args = parser.parse_args()

    nn_model = int(args.nn)
    n_epochs = int(args.n_epochs)
    model = Net()

    f_name = 'data/multi_start/neural_networks/nn_' + str(nn_model)
    model.load_state_dict(torch.load(f_name))
    logger.info('nn loaded successfully')
    train_nn(model, n_epochs=n_epochs)
