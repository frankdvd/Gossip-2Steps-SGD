import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim

import torchvision
import torchvision.transforms as transforms
from dataset.cifar10 import get_cifar10_dataloader
from network.resent import ResNet18, ResNet50

import torch.distributed as dist

import argparse
import os
import random
import numpy as np
import pathlib
import time

def set_random_seeds(random_seed=0):

    torch.manual_seed(random_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(random_seed)
    random.seed(random_seed)

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

def average_gradients(model):
    """ Gradient averaging. """
    size = float(dist.get_world_size())
    for param in model.parameters():
        dist.all_reduce(param.grad.data, op=dist.ReduceOp.SUM)
        param.grad.data /= size

def main():

    num_epochs_default = 100
    learning_rate_default = 0.2
    random_seed_default = 0
    model_dir_default = "checkpoint"
    model_filename_default = "resnet_distributed_gossip.pth"

    # Each process runs on 1 GPU device specified by the local_rank argument.
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--local_rank", type=int, help="Local rank. Necessary for using the torch.distributed.launch utility.")
    parser.add_argument("--num_epochs", type=int, help="Number of training epochs.", default=num_epochs_default)
    parser.add_argument("--learning_rate", type=float, help="Learning rate.", default=learning_rate_default)
    parser.add_argument("--random_seed", type=int, help="Random seed.", default=random_seed_default)
    parser.add_argument("--model_dir", type=str, help="Directory for saving models.", default=model_dir_default)
    parser.add_argument("--model_filename", type=str, help="Model filename.", default=model_filename_default)
    parser.add_argument("--resume", action="store_true", help="Resume training from saved checkpoint.")
    parser.add_argument("--resnet50", action="store_true", help="using resnet50 model")
    parser.add_argument('--evaluate', '-e', action='store_true',
                            help='evaluate epoch')    
    argv = parser.parse_args()

    local_rank = argv.local_rank
    num_epochs = argv.num_epochs
    learning_rate = argv.learning_rate
    random_seed = argv.random_seed
    model_dir = argv.model_dir
    model_filename = argv.model_filename
    resume = argv.resume
    use_resnet50 = argv.resnet50
    total_batch_size = 512


    # Create directories outside the PyTorch program
    # Do not create directory here because it is not multiprocess safe
    '''
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    '''

    model_filepath = os.path.join(model_dir, model_filename)

    # We need to use seeds to make sure that the models initialized in different processes are the same
    set_random_seeds(random_seed=random_seed)

    # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
    torch.distributed.init_process_group(backend="nccl")
    # torch.distributed.init_process_group(backend="gloo")

    global_rank = dist.get_rank()
    print("Global rank is", global_rank)

    world_size = dist.get_world_size()
    print("World size is", world_size)

    batch_size = total_batch_size // world_size

    # Encapsulate the model on the GPU assigned to the current process
    
    ## can change to other model
    model = ResNet18()
    if use_resnet50:
        model = ResNet50()
        print("Using resnet50 model")
    #model = torchvision.models.resnet18(pretrained=False)
    device = torch.device("cuda:{}".format(local_rank))
    model = model.to(device)

    # We only save the model who uses device "cuda:0"
    # To resume, the device for the saved model would also be "cuda:0"
    if resume == True:
        map_location = {"cuda:0": "cuda:{}".format(local_rank)}
        model.load_state_dict(torch.load(model_filepath, map_location=map_location))

    current_path = pathlib.Path(__file__).parent.resolve()
    trainloader, testloader = get_cifar10_dataloader(current_path / 'data', parallel = True, batch_size = batch_size)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=learning_rate,
                        momentum=0.9, weight_decay=5e-4)
    #scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[40, 75], gamma=0.1)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=0)

    T1 = time.perf_counter()

    # Loop over the dataset multiple times
    for epoch in range(num_epochs):

        print("Global Rank: {}, Local Rank: {}, Epoch: {}, Training ...".format(global_rank, local_rank, epoch))
        
        model.train()

        for data in trainloader:
            inputs, labels = data[0].to(device), data[1].to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            average_gradients(model)
            optimizer.step()

        # Save and evaluate model routinely
        if argv.evaluate or epoch == num_epochs - 1:
            if global_rank == 0:
                accuracy = evaluate(model=model, device=device, test_loader=testloader)
                #torch.save(model.state_dict(), model_filepath)
                print("-" * 75)
                print("Local Rank: {}, Epoch: {}, Accuracy: {}".format(local_rank, epoch, accuracy))
                print("-" * 75)
        
        scheduler.step()


    T2 = time.perf_counter()
    print("Total training time in second", T2-T1)

    # sleep for nccl finish clean up
    time.sleep(3)


if __name__ == "__main__":
    
    main()