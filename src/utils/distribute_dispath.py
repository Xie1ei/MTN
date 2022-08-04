import torch
import torch.nn as nn

from typing import Dict
from torch.optim.lr_scheduler import _LRScheduler

import os
import pathlib

import torch.distributed as dist

def resume_training(checkpoint: Dict,
                    model: nn.Module,
                    optimizer: torch.optim.Optimizer,
                    scheduler: _LRScheduler):
    # Load checkpoint
    sd = load_checkpoint(checkpoint, model, optimizer)
    # TODO: Fix warning
    scheduler.step(sd['epoch'])
    return sd['epoch']



def load_checkpoint(checkpoint, net, optimizer=None, map_loc="cuda"):
    sd = torch.load(checkpoint, map_location=map_loc)
    net.load_state_dict(sd['model'])
    if optimizer and sd['optimizer']:
        optimizer.load_state_dict(sd['optimizer'])
    return sd


def save_checkpoint(net, optimizer, save_path, name, epoch, **kwargs):
    if isinstance(net, torch.nn.parallel.DistributedDataParallel):
        net = net.module

    if (dist.is_initialized() and dist.get_rank() == 0) or (dist.is_initialized() is False):
        model_state_dict = net.state_dict()
        state = {
            'model': model_state_dict,
            'optimizer': optimizer.state_dict() if optimizer else None,
            'epoch': epoch
        }
        state.update(kwargs)

        # state = {'model': model_state_dict}
        if not os.path.exists(save_path):
            pathlib.Path(save_path).mkdir(parents=True, exist_ok=True)
            # os.mkdir(save_path)
        model_path = os.path.join(save_path, name)
        torch.save(state, model_path)