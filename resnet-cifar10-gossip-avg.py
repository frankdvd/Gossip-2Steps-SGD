import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim

import torchvision
import torchvision.transforms as transforms
from dataset.cifar10 import get_cifar10_dataloader
from network.resent import ResNet18, ResNet50
from utils.central_model import Central_Model
from utils.tensor_utils import all_reduce_avg

import torch.distributed as dist

import argparse
import os
import random
import numpy as np
import pathlib
import time
import threading

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


def in_node_average_gradients(model, nproc_per_node, in_node_group):
    """ in node Gradient averaging. """
    tensors = []
    for idx, param in enumerate(model.parameters()):
        tensors.append(param.grad.data)
    all_reduce_avg(tensors, in_node_group, nproc_per_node)


def main():

    # default values
    num_epochs_default = 100
    alpha_default = 0.35
    communication_period_default = 40
    learning_rate_default = 0.2
    random_seed_default = 0
    model_dir_default = "checkpoint"
    model_filename_default = "resnet_distributed_gossip.pth"


    # Each process runs on 1 GPU device specified by the local_rank argument.
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--local_rank", type=int, help="Local rank. Necessary for using the torch.distributed.launch utility.")
    parser.add_argument("--num_epochs", type=int, help="Number of training epochs.", default=num_epochs_default)
    parser.add_argument("--alpha", type=int, help="Model sync rate", default=alpha_default)
    parser.add_argument("--communication_period", type=int, help="central model commnication period", default=communication_period_default)
    parser.add_argument("--learning_rate", type=float, help="Learning rate.", default=learning_rate_default)
    parser.add_argument("--random_seed", type=int, help="Random seed.", default=random_seed_default)
    parser.add_argument("--model_dir", type=str, help="Directory for saving models.", default=model_dir_default)
    parser.add_argument("--model_filename", type=str, help="Model filename.", default=model_filename_default)
    parser.add_argument("--resnet50", action="store_true", help="using resnet50 model")
    parser.add_argument('--evaluate', '-e', action='store_true',
                            help='evaluate epoch')    
    parser.add_argument("--node_rank", type=int, help="which node is this process in")
    parser.add_argument("--nproc_per_node", type=int, help="number of proc per node")
    argv = parser.parse_args()

    local_rank = argv.local_rank
    num_epochs = argv.num_epochs
    alpha = argv.alpha
    communication_period = argv.communication_period
    learning_rate = argv.learning_rate
    random_seed = argv.random_seed
    model_dir = argv.model_dir
    model_filename = argv.model_filename

    node_rank = argv.node_rank
    nproc_per_node = argv.nproc_per_node
    print('Node rank', node_rank)
    print('nproc_per_node', nproc_per_node)

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
    dist.init_process_group(backend="nccl")
    #torch.distributed.init_process_group(backend="gloo")


    # setup process infos
    global_rank = dist.get_rank()
    print("Global rank is", global_rank)

    world_size = dist.get_world_size()
    print("World size is", world_size)

    nnodes = int(world_size/nproc_per_node)
    print("number of  nnodes", nnodes)

    batch_size = total_batch_size // world_size
    
    # setup in_node gpu and out_node gpu group
    in_node_ranks = [[j * nproc_per_node + i for i in range(nproc_per_node)]  for j in range(nnodes)]
    out_node_rank = [ i * nproc_per_node for i in range(nnodes)]
    print('in_node_ranks', in_node_ranks)
    print('out_node_rank', out_node_rank)
    in_node_groups = [dist.new_group(ranks=in_node_rank) for in_node_rank in in_node_ranks]
    out_node_group = dist.new_group(ranks=out_node_rank)



    # Encapsulate the model on the GPU assigned to the current process
    ## can change to other model
    model = ResNet18()
    #model = torchvision.models.resnet18(pretrained=False)
    if use_resnet50:
        model = ResNet50()
        print("Using resnet50 model")
    # init model
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda:{}".format(local_rank))
    model = model.to(device)

    # central model for async gossip model params
    # code is in utils.central_model.py
    central_model = Central_Model(
        model, 
        in_node_groups[node_rank], 
        out_node_group, 
        nnodes, 
        node_rank * nproc_per_node, 
        alpha = alpha, 
        communication_period = communication_period)

    # load tada
    current_path = pathlib.Path(__file__).parent.resolve()
    trainloader, testloader = get_cifar10_dataloader(current_path / 'data', parallel = True, batch_size = batch_size)

    # loss function
    criterion = nn.CrossEntropyLoss()
    # optimizer & scheduler
    optimizer = optim.SGD(model.parameters(), lr=learning_rate,
                        momentum=0.9, weight_decay=5e-4)

    #scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[40, 75], gamma=0.1)
    #scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=num_epochs//3, T_mult=2, eta_min=0)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=0)

    T1 = time.perf_counter()
    # increase accumulation step could reduce communication but it will reduce accuracy 
    # set it to 1 to disable it for now
    accumulation_steps = 1

    # Loop over the dataset multiple times
    for epoch in range(num_epochs):

        print("Global Rank: {}, Local Rank: {}, Epoch: {}, Training ...".format(global_rank, local_rank, epoch))
        
        model.train()
        for i, data in enumerate(trainloader):
            # 1. input output
            inputs, labels = data[0].to(device), data[1].to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            # 2.1 loss regularization
            loss = loss/accumulation_steps
            # 2.2 back propagation
            loss.backward()
            # 3. update parameters of net
            if((i+1) % accumulation_steps ) == 0:
                ## we need to make sure nccl sync 
                central_model.comm_done.wait()
                # 3.1 average_gradients
                in_node_average_gradients(model, nproc_per_node, in_node_groups[node_rank])
                # 3.2 optimizer the net
                optimizer.step()        # update parameters of net
                optimizer.zero_grad()   # reset gradient
                # 3.3 async transfer model param
                central_model.avg_model(model)

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