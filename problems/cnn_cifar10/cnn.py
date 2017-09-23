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


def res_parameters_linear(self):
    m = self.weight.size(0)
    n = self.weight.size(1)
    stdv = math.sqrt(6. / (n + m))
    self.weight.data.uniform_(-stdv, stdv)
    if self.bias is not None:
        stdv = math.sqrt(1. / n)
        self.bias.data.uniform_(-stdv, stdv)


def res_parameters_conv(self):
    n = self.in_channels
    m = self.out_channels
    for k in self.kernel_size:
        n *= k
    stdv = math.sqrt(6. / (n + m))
    self.weight.data.uniform_(-stdv, stdv)
    if self.bias is not None:
        stdv = math.sqrt(1. / n)
        self.bias.data.uniform_(-stdv, stdv)


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        res_parameters_conv(m)
    elif classname.find('Linear') != -1:
        res_parameters_linear(m)




# random.seed(1)
# indexes_data = range(len(trainset))
# random.shuffle(indexes_data)



def train_nn(random_seed, n_epochs=2, batch_size=4, lr=0.001, weight_decay=0,
             number_chanels_first=6, number_chanels_second=16, number_hidden_units=120,
             size_kernel=5, cuda=False, trainset=None, testset=None):

    torch.manual_seed(1)

    if trainset is None:
        trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                                download=True, transform=transform)
    if testset is None:
        testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                            download=True, transform=transform)

    class Net(nn.Module):
        def __init__(self):
            super(Net, self).__init__()

            self.conv1 = nn.Conv2d(3, number_chanels_first, size_kernel)
            dim_1 = (32 - size_kernel + 1) / 2
            self.pool = nn.MaxPool2d(2, 2)
            number_chanels_second = number_chanels_first + 10
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

    net = Net()
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                              shuffle=True, num_workers=2)

    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                             shuffle=False, num_workers=2)
    if random_seed is not None:
        if not cuda:
            torch.manual_seed(random_seed)
        else:
            torch.manual_seed(random_seed)
            torch.cuda.manual_seed(random_seed)

    if cuda:
        net.cuda()

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=lr, momentum=0.9, weight_decay=weight_decay)

    for epoch in range(n_epochs):  # loop over the dataset multiple times

        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            # get the inputs
            inputs, labels = data

            if cuda:
                inputs = inputs.cuda()
                labels = labels.cuda()

            # wrap them in Variable
            inputs, labels = Variable(inputs), Variable(labels)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # # print statistics
            # running_loss += loss.data[0]

    # running_loss = 0.0
    # total_data = 0
    # for i, data in enumerate(trainloader, 0):
    #     # get the inputs
    #     inputs, labels = data
    #
    #     total_data += len(labels)
    #
    #     # wrap them in Variable
    #     inputs, labels = Variable(inputs), Variable(labels)
    #
    #     # zero the parameter gradients
    #     optimizer.zero_grad()
    #
    #     # forward + backward + optimize
    #     outputs = net(inputs)
    #     loss = criterion(outputs, labels)
    #
    #     running_loss += loss.data[0]
    #
    # error = running_loss / float(total_data)

    correct = 0
    total = 0
    for data in testloader:
        images, labels = data

        if cuda:
            images = images.cuda()
            labels = labels.cuda()

        outputs = net(Variable(images))
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum()

    error = 1.0 - float(correct) / float(total)

    return error
