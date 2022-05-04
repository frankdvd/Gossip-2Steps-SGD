'''Download CIFAR10 with PyTorch.'''

import pathlib
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms

import os
import argparse

from network.resent import ResNet18
from dataset.cifar10 import get_cifar10_dataloader
import time
import time
import torchvision.models as models



if __name__ == '__main__':

    # Data
    print('==> Preparing data..')

    current_path = pathlib.Path(__file__).parent.resolve()

    trainloader, testloader = get_cifar10_dataloader(current_path.parent.resolve() / 'data')

    classes = ('plane', 'car', 'bird', 'cat', 'deer',
            'dog', 'frog', 'horse', 'ship', 'truck')
