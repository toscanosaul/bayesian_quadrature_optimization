from __future__ import absolute_import

import torch
import torchvision
import torchvision.transforms as transforms
import math
from torch.autograd import Variable
import torch.nn as nn
import random
import torch.nn.functional as F
import torch.optim as optim


torch.manual_seed(1)

transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

# trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
#                                         download=True, transform=transform)
#
# testset = torchvision.datasets.CIFAR10(root='./data', train=False,
#                                        download=True, transform=transform)
# testloader = torch.utils.data.DataLoader(testset, batch_size=4,
#                                          shuffle=False, num_workers=2)

classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')


number_chanels_first = 6
size_kernel = 5
number_hidden_units = 120

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        self.conv1 = nn.Conv2d(3, number_chanels_first, size_kernel)
        dim_1 = (32 - size_kernel + 1) / 2
        self.pool = nn.MaxPool2d(2, 2)
        number_chanels_second = 2 * number_chanels_first
        self.conv2 = nn.Conv2d(number_chanels_first, number_chanels_second, size_kernel)
        dim_2 = (dim_1 - size_kernel + 1) / 2
        self.total_dim = number_chanels_second * dim_2 * dim_2
        self.fc1 = \
            nn.Linear(self.total_dim, number_hidden_units)
        self.fc2 = nn.Linear(number_hidden_units, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, self.total_dim)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features

trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transform)
batch_size = 4
net = Net()
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                          shuffle=True, num_workers=2)

net.cuda()

