

import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data.distributed import DistributedSampler

# Image preprocessing modules
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

def get_cifar10_dataloader(root_folder, parallel = False, batch_size = 256):
    # CIFAR-10 dataset
    trainset = torchvision.datasets.CIFAR10(
        root=root_folder, train=True, download=(not parallel), transform=transform_train)
    testset = torchvision.datasets.CIFAR10(
        root=root_folder, train=False, download=(not parallel), transform=transform_test)

    train_sampler = None
    if parallel:
        train_sampler = DistributedSampler(dataset=trainset)

    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=batch_size, sampler=train_sampler, shuffle=not parallel, num_workers=4)
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=128, shuffle=False, num_workers=4)

    return trainloader, testloader
