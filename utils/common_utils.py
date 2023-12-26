import random
import torch
import numpy as np
from copy import deepcopy
import json
import constants

def set_seed(seed:int=42):
    """Sets the seed for torch, numpy and random
    Args:
        seed (int): [description]
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

def get_device():
    return "cuda:0"

def set_cuda_device(gpu_num: int):
    import os
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_num)

def insert_kwargs(kwargs:dict, new_args:dict):
    assert type(new_args) == type(kwargs), "Please pass two dictionaries"
    merged_args = kwargs.copy()
    merged_args.update(new_args)
    return merged_args

def dict_print(d:dict):
    d_new = deepcopy(d)

    def cast_str(d_new:dict):
        for k, v in d_new.items():
            if isinstance(v, dict):
                d_new[k] = cast_str(v)
            d_new[k] = str(v)
        return d_new
    d_new = cast_str(d_new)

    pretty_str = json.dumps(d_new, sort_keys=False, indent=4)
    print(pretty_str)
    return pretty_str

def get_do_cls_phi(epoch, local_step, inter_epochs:dict, inter_iters:dict):
    """This is a utility funbctio to determine if we should minimize theta (or) phi (or) psi
    The code may create unnecessary clutter thus factoring it out to common utils

    Args:
        epoch ([type]): [description]
        inter_epochs ([type]): [description]
        inter_iters ([type]): [description]
    """
    if inter_epochs is None and inter_iters is None:
        return 1, 1
    if inter_iters is not None and inter_iters is not None:
        raise ValueError("We can only interleavce epochs or iterations but not both")
    if inter_iters is not None:
        mod_div = sum(inter_iters.values())
        bin = local_step % mod_div
        if bin <= inter_iters[constants.THETA]:
            return 1, 0
        else:
            return 0, 1
    elif inter_epochs is not None:
        mod_div = sum(inter_epochs.values())
        bin = epoch % mod_div
        if bin < inter_epochs[constants.THETA]:
            return 1, 0
        else:
            return 0, 1
