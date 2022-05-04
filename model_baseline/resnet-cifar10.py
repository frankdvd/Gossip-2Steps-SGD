'''Train CIFAR10 with PyTorch.'''

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
import torchvision.models as models


def evaluate(model, device, test_loader):

    model.eval()

    correct = 0
    total = 0
    with torch.no_grad():
        for data in test_loader:
            images, labels = data[0].to(device), data[1].to(device)
            outputs = model(images)
            predicted = outputs.argmax(dim=1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
    accuracy = correct / total

    return accuracy


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
    parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
    parser.add_argument('--resume', '-r', action='store_true',
                        help='resume from checkpoint')
    parser.add_argument('--evaluate', '-e', action='store_true',
                        help='evaluate epoch')                        
    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    best_acc = 0  # best test accuracy
    num_epochs = 100
    model_dir = "checkpoint"
    model_filename = "resnet_single.pth"

    model_filepath = os.path.join(model_dir, model_filename)


    if not os.path.isdir(model_dir):
        os.mkdir(model_dir)

    # Data
    print('==> Preparing data..')


    current_path = pathlib.Path(__file__).parent.resolve()

    trainloader, testloader = get_cifar10_dataloader(current_path.parent.resolve() / 'data')

    classes = ('plane', 'car', 'bird', 'cat', 'deer',
            'dog', 'frog', 'horse', 'ship', 'truck')

    # Model
    print('==> Building model..') 
    net = ResNet18()
    #net = models.resnet18(pretrained=False)
    net = net.to(device)

    if device == 'cuda':
        net = torch.nn.DataParallel(net)
        cudnn.benchmark = True

    if args.resume:
        # Load checkpoint.
        print('==> Resuming from checkpoint..')
        assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
        checkpoint = torch.load('./checkpoint/ckpt.pth')
        net.load_state_dict(checkpoint['net'])
        best_acc = checkpoint['acc']
        start_epoch = checkpoint['epoch']

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=args.lr,
                        momentum=0.9, weight_decay=5e-4)
    #scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[40, 75], gamma=0.1)
    

    T1 = time.perf_counter()

    for epoch in range(num_epochs):
        ## training ------------------------------
        print("Epoch: {}, Training ...".format(epoch))

        net.train()

        for data in trainloader:
            inputs, labels = data[0].to(device), data[1].to(device)
            optimizer.zero_grad()
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

        ## testing -----------------------------
        if args.evaluate or epoch == num_epochs - 1:
            # evaluate model
            accuracy = evaluate(model=net, device=device, test_loader=testloader)
            #torch.save(net.state_dict(), model_filepath)
            print("-" * 75)
            print("Epoch: {}, Accuracy: {}".format(epoch, accuracy))
            print("-" * 75)
            scheduler.step()

    T2 = time.perf_counter()

    print("Total training time in second", T2-T1)

