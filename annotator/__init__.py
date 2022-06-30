import nvtx
import json
import torch

dtype_map = {torch.float32:'float32', torch.float16:'float16', torch.int32:'int32'}

# Model Parameters
def input_begin(tensor, name='input'):
    event = {
        'event':'input',
        'name':'input',
        'tensor': (tuple(tensor.shape), dtype_map[tensor.dtype])
    }
    nvtx.push_range(json.dumps(event))

def input_end():
    nvtx.pop_range()

def output_begin(tensor, name='output'):
    event = {
        'event':'output',
        'name':'output',
        'tensor': (tuple(tensor.shape), dtype_map[tensor.dtype])
    }
    nvtx.push_range(json.dumps(event))

def output_end():
    nvtx.pop_range()

def compute_begin(name, model = None):
    weights = []
    if model:
        for param in model.parameters():
            weights.append((tuple(param.shape), dtype_map[param.dtype]))
    event = {
        'event':'compute',
        'name': name,
        'weights': weights,
    }
    nvtx.push_range(json.dumps(event))

def compute_end():
    nvtx.pop_range()

def communication_begin(name, recv_tensor_list=[], send_tensor_list=[]):
    send_tensors=[]
    recv_tensors=[]
    for tensor in send_tensor_list:
        send_tensors.append((tuple(tensor.shape), dtype_map[tensor.dtype]))
    for tensor in recv_tensor_list:
        recv_tensors.append((tuple(tensor.shape), dtype_map[tensor.dtype]))
    event = {
        'event':'communication',
        'name': name,
        'send_tensors': send_tensors,
        'recv_tensors': recv_tensors
    }
    nvtx.push_range(json.dumps(event))

def communication_end():
    nvtx.pop_range()

def inference_begin(rank, local_rank, index, batch_size, name='inference'):
    event = {
        'event':'inference',
        'name':'inference',
        'rank': rank,
        'local_rank': local_rank,
        'step':index,
        'batch_size':batch_size
    }
    nvtx.push_range(json.dumps(event))

def inference_end():
    nvtx.pop_range()
