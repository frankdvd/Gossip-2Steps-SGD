from .tensor_utils import all_reduce_avg_boardcast
import torch

import threading

class Central_Model:
    def __init__(self, model, in_node_group, out_node_group, out_node_group_size, comm_node_rank, alpha, communication_period):
        # input
        self.out_node_group_size = out_node_group_size
        self.in_node_group = in_node_group
        self.out_node_group = out_node_group
        self.comm_node_rank = comm_node_rank

        # update parameter
        self.alpha = alpha

        # counter
        self.step = 0
        self.communication_period = communication_period

        # data
        self.tensors = []
        for idx, param in enumerate(model.parameters()):
            self.tensors.append(param.data.detach().clone().cuda())

        # thread control
        self.comm_done = threading.Event()
        self.tensors_ready = threading.Event()
        self.comm_done.set()
        self.tensors_ready.clear()

        # daemon thread for communication
        self.t = threading.Thread(
                target=Central_Model._async_all_reduce_,
                args=(
                    self.tensors, 
                    self.comm_done, 
                    self.tensors_ready, 
                    self.in_node_group, 
                    self.out_node_group, 
                    self.out_node_group_size,
                    self.comm_node_rank),
                daemon = True)
        self.t.start()


    def avg_model(self, model):
        step_flag = (self.step != 0 and self.step % self.communication_period == 0)
        self.step += 1
        if step_flag:
            
            # if communication is not done we have to wait to avoid nccl error
            self.comm_done.wait()
            # update central_param
            for idx, param in enumerate(model.parameters()):
                central_param = self.tensors[idx]
                # pull local model back to central model to avoid model diverge 
                param.data.mul_(1-self.alpha).add_(central_param, alpha = self.alpha)
                # copy new data to tensors to update central model local value
                self.tensors[idx] = param.data.detach().clone().cuda()
            
            # start average central model data across comm nodes
            self.comm_done.clear()
            self.tensors_ready.set()


    @staticmethod
    def _async_all_reduce_(tensors, comm_done, tensors_ready, in_node_group, out_node_group, out_node_group_size, comm_node_rank):
        while True:
            # Wait for the local update of the central model to complete
            tensors_ready.wait()
            # gossip all reduce accross communication node, then boardcast value to all the in node rank
            all_reduce_avg_boardcast(tensors, in_node_group, out_node_group, out_node_group_size, comm_node_rank)
            # set flag start to update the local central model again
            tensors_ready.clear()
            comm_done.set()