import torch
import torch.distributed as dist
import collections


def all_reduce_avg_boardcast(tensors, in_node_group, out_node_group, out_node_group_size, comm_node_rank):
    for t in tensors:
        dist.all_reduce(t, op=dist.ReduceOp.SUM, group = out_node_group)
        t /= out_node_group_size
        dist.broadcast(t, comm_node_rank, group=in_node_group)

def all_reduce_avg(tensors, group, group_size):
    for t in tensors:
        dist.all_reduce(t, op=dist.ReduceOp.SUM, group = group)
        t /= group_size
