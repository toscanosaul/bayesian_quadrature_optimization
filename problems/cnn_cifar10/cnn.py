from __future__ import absolute_import

import torch
import torchvision
import torchvision.transforms as transforms
import math
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


torch.manual_seed(1)

transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=4,
                                          shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=4,
                                         shuffle=False, num_workers=2)

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

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

torch.manual_seed(1)
net = Net()


def train_nn(random_seed, n_epochs=2):
    if random_seed is not None:
        torch.manual_seed(random_seed)
    net.apply(weights_init)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

    for epoch in range(n_epochs):  # loop over the dataset multiple times

        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            # get the inputs
            inputs, labels = data

            # wrap them in Variable
            inputs, labels = Variable(inputs), Variable(labels)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.data[0]

    error = running_loss / (4.0 * 12500)


    correct = 0
    total = 0
    for data in testloader:
        images, labels = data
        outputs = net(Variable(images))
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum()

    error_images = 100.0 * correct / total

    return {'loss_objective': error, 'test_error_images': error_images}
