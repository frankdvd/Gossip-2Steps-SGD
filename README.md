# cs267-final-project

Decentralized ML training using gossip 2 steps communication

# 1. Setup Envionment(local laptop only)

*You have to skip this step if you run the code on bridge-2*

1. Create a new conda environment and activate it
    ```
    conda create --name cs267
    conda activate cs267
    conda install pip
    ```
2. Install requirement packages
    ```
    pip install -r requirements.txt
    ```
    You might want to install the cuda version of pytorch again using the cmd [here](https://pytorch.org/get-started/locally/) if you want to use gpu for pytorch


# 2. Run 1 GPU baseline
1. run on your own laptop
    ```
    python -m model_baseline.resnet-cifar10
    ```
2. run on bridges-2
    ```
    # login in to interactive node
    salloc -N 1 -p GPU-shared --gres=gpu:1 -q interactive -t 01:00:00

    # load pytorch environment
    singularity shell --network=host --nv  /ocean/containers/ngc/pytorch/pytorch_21.08-py3.sif

    # run code
    python -m model_baseline.resnet-cifar10
    ```


# 3. Run DDP baseline
1. run on bridge2, as an example, you can start 2 nodes in different terminal using each of the cmd below 
    ```
    interact -p GPU-small --gres=gpu:4 -t 04:00:00
    ```
    or
    ```
    interact -p GPU-shared --gres=gpu:4 -t 04:00:00
    ```
    

1. load pytorch environment
    ```
    singularity shell --nv  /ocean/containers/ngc/pytorch/pytorch_21.08-py3.sif
    ``` 

1. make required folder
    ```
    mkdir -p checkpoint 
    ```

1. check first node host ip, and replace the --master_addr value in the following cmd by the second value returned
    ```
    hostname -I
    ```

1. change cmd based on the gpu node and run code.

    *Before running the parallel code, make sure you run the following code to download the dataset*

    ```
    python -m dataset.download-datasets
    ```
    
    *Before you launch the training, you have to make sure that 2 nodes are not the same shared node(they have different ip)*

    For example, on the first node run
    ```
    python -m torch.distributed.launch --nproc_per_node=4 --nnodes=2 --node_rank=0 --master_addr="10.8.10.250" --master_port=1233 ./resnet-cifar10-all-reduce.py
    ```
    on the second node run
    ```
    python -m torch.distributed.launch --nproc_per_node=4 --nnodes=2 --node_rank=1 --master_addr="10.8.10.250" --master_port=1233 ./resnet-cifar10-all-reduce.py
    ```
    The only difference is the node_rank.

# 4. Run gossip version of DDP

1. run steps 1-5 from `Run DDP baseline` section

1. change cmd based on the gpu node and run code.
    For example, on the first node run
    ```
    python -m torch.distributed.launch --nproc_per_node=4 --nnodes=2 --node_rank=0 --master_addr="10.8.10.250" --master_port=1235 ./resnet-cifar10-gossip-avg.py --nproc_per_node=4 --node_rank=0
    ```
    on the second node run
    ```
    python -m torch.distributed.launch --nproc_per_node=4 --nnodes=2 --node_rank=1 --master_addr="10.8.10.250" --master_port=1235 ./resnet-cifar10-gossip-avg.py --nproc_per_node=4 --node_rank=1
    ```
    


1. To debug nccl run 
    ```
    export NCCL_DEBUG=INFO
    export NCCL_DEBUG_SUBSYS=ALL
    # check nvlink
    nvidia-smi topo -m    
    nvidia-smi nvlink --status
    ```

# Other Notes

The default model used in our scripts is ResNet18. However, it is possible for users to use resnet50 as well.
To use resnet50, just add argument `--resnet50`. for example:
On the first node run
    
```
python -m torch.distributed.launch --nproc_per_node=4 --nnodes=2 --node_rank=0 --master_addr="10.8.10.250" --master_port=1235 ./resnet-cifar10-gossip-avg.py --nproc_per_node=4 --node_rank=0 --resnet50
```
    
On the second node run
    
```
python -m torch.distributed.launch --nproc_per_node=4 --nnodes=2 --node_rank=1 --master_addr="10.8.10.250" --master_port=1235 ./resnet-cifar10-gossip-avg.py --nproc_per_node=4 --node_rank=1 --resnet50
```
