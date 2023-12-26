import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

def init_weights(m:nn.Module):

    def set_params(w):
        if isinstance(w, nn.Linear):
            torch.nn.init.xavier_uniform(w.weight)
            w.bias.data.fill_(0.01)
    m.apply(set_params)

def get_lr_scheduler(optimizer, scheduler_name, n_rounds=None):
    """
    Gets torch.optim.lr_scheduler given an lr_scheduler name and an optimizer
    :param optimizer:
    :type optimizer: torch.optim.Optimizer
    :param scheduler_name: possible are
    :type scheduler_name: str
    :param n_rounds: number of training rounds, only used if `scheduler_name == multi_step`
    :type n_rounds: int
    :return: torch.optim.lr_scheduler
    """

    if scheduler_name == "sqrt":
        return optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda x: 1 / np.sqrt(x) if x > 0 else 1)

    elif scheduler_name == "linear":
        return optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda x: 1 / x if x > 0 else 1)

    elif scheduler_name == "constant":
        return optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda x: 1)

    elif scheduler_name == "cosine_annealing":
        return optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200, eta_min=0)

    elif scheduler_name == "multi_step":
        assert n_rounds is not None, "Number of rounds is needed for \"multi_step\" scheduler!"
        milestones = [n_rounds//2, 3*(n_rounds//4)]
        return optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=0.1)

    else:
        raise NotImplementedError("Other learning rate schedulers are not implemented")



def _sel_nzro(self, t, sij):
    sel_nonzero = lambda t, sij : torch.squeeze(t[torch.nonzero(sij)])
    res = sel_nonzero(t, sij)
    if res.dim() == t.dim()-1:
        res = torch.unsqueeze(res, 0)        
    return res
    
def _sel_zro(self, t, sij):
    sel_zero = lambda t, sij : torch.squeeze(1-t[torch.nonzero(sij)])
    res = sel_zero(t, sij)
    if res.dim() == t.dim()-1:
        res = torch.unsqueeze(res, 0)        
    return res