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




def _calculate_fan_in_and_fan_out(tensor):
    dimensions = tensor.ndimension()
    if dimensions < 2:
        raise ValueError("Fan in and fan out can not be computed for tensor with less than 2 dimensions")

    if dimensions == 2:  # Linear
        fan_in = tensor.size(1)
        fan_out = tensor.size(0)
    else:
        num_input_fmaps = tensor.size(1)
        num_output_fmaps = tensor.size(0)
        receptive_field_size = 1
        if tensor.dim() > 2:
            receptive_field_size = tensor[0][0].numel()
        fan_in = num_input_fmaps * receptive_field_size
        fan_out = num_output_fmaps * receptive_field_size

    return fan_in, fan_out


def weights_init(m, xavier=True, std_given=10., random_seed=1):
    classname = m.__class__.__name__
    random.seed(random_seed)
    torch.manual_seed(random_seed)
    print(classname)
    if classname.find('Conv') != -1:
        # m.weight.data.normal_(0.0, 0.02)

        if xavier:
            z = _calculate_fan_in_and_fan_out(m.weight.data)
            std = math.sqrt(2.0 / (z[0] + z[1]))
            m.weight.data.uniform_(-std, std)
        else:
            m.weight.data.uniform_(-std_given, std_given)
    if classname.find('Linear') != -1:
        if xavier:
            z = _calculate_fan_in_and_fan_out(m.weight.data)
            std = math.sqrt(2.0 / (z[0] + z[1]))
            m.weight.data.uniform_(-std, std)

            m.bias.data.fill_(0.0)
        else:
            m.weight.data.uniform_(-std_given, std_given)
            m.bias.data.uniform_(-std_given, std_given)


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


args = {}
args['test_batch_size'] = 1000
args['train_batch_size'] = 1000
args['manualSeed'] = 5

random.seed(args['manualSeed'])
torch.manual_seed(args['manualSeed'])


n_networks = 10
neural_networks = {}
for i in range(n_networks):
    neural_networks[i] = Net()

for i in range(1, n_networks - 1):
    magnitudes = 10 ** (- i + 4)
    g = lambda x: weights_init(x, xavier=False, std_given=magnitudes, random_seed=i)
    neural_networks[i].apply(g)

neural_networks[0].apply(weights_init)

for i in range(n_networks):
    torch.save(neural_networks[i].state_dict(), 'data/multi_start/neural_networks/nn_%d' % i)





