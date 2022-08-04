import importlib
import typing
from typing import Callable

from attrdict import AttrDict
from torch.optim import Optimizer
from torch.optim.lr_scheduler import CosineAnnealingLR, _LRScheduler

class DistParams(object):
    def __init__(self, py_module_configs):
        DeprecationWarning("DistParams will be deprecated.")
        self.backend = getattr(py_module_configs, "backend", "nccl")
        self.dist_url = getattr(py_module_configs, "dist_url", "tcp://127.0.0.1:7777")
        self.use_sync_bn = getattr(py_module_configs, "use_sync_bn", True)


class DistributeParams(object):
    def __init__(self, py_module_configs):
        self.backend = getattr(py_module_configs, "backend", "nccl")
        self.dist_url = getattr(py_module_configs, "dist_url", "tcp://")
        self.world_size = getattr(py_module_configs, "world_size", 1)
        self.rank = getattr(py_module_configs, "rank", 0)


def load_config_module(config_file):
    spec = importlib.util.spec_from_file_location("tuly.config", config_file)
    configs = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(configs)
    return configs


def try_get_attr(obj, attr, default=None, required=True):
    try:
        return getattr(obj, attr)
    except AttributeError:
        if required:
            raise AttributeError(f"Attribute {attr} is required in config")
        return default


def safe_get_attr(obj, attr, attr_type, default=None, required=True):
    aobject = try_get_attr(obj, attr, default, required)
    if isinstance(aobject, attr_type) or issubclass(aobject.__class__, attr_type):
        return aobject
    else:
        raise TypeError(f"{attr} is not type or subtype of {attr_type}")
    return aobject


class AttributeInfo(object):
    def __init__(self, type, default=None, required=True):
        self.type = type
        self.default = default
        self.required = required


config_attribute_mapping = {
    "model": AttributeInfo(type=Callable, required=True),
    "save_path": AttributeInfo(type=str, required=True),
    "optimizer": AttributeInfo(type=type(Optimizer), required=True),
    "optimizer_params": AttributeInfo(type=dict, required=True),
    "scheuler_param" : AttributeInfo(type=dict, required=True),
    "epochs": AttributeInfo(type=int, required=True),
    "scheduler": AttributeInfo(type=type(_LRScheduler), required=True, default=CosineAnnealingLR),
    "dataset": AttributeInfo(type=Callable, required=True),
    "train_trans": AttributeInfo(type=Callable, required=True),
    "val_trans": AttributeInfo(type=Callable, required=True),
    # "training_dataset_args": AttributeInfo(type=dict, required=True),
    # "val_dataset_args": AttributeInfo(type=dict, required=True),
    "val_loader_args": AttributeInfo(type=dict, required=True),
    "loader_args": AttributeInfo(type=dict, required=True),
    "loss": AttributeInfo(type=Callable, required=True),
    "val_loss": AttributeInfo(type=Callable, required=True),

    "tensorboard_log": AttributeInfo(type=str, required=False, default=""),
    "use_sync_bn": AttributeInfo(type=bool, required=False, default=True),
    "with_validation": AttributeInfo(type=bool, required=False, default=True),
    "load_optimizer": AttributeInfo(type=bool, required=False, default=True),
    "find_unused_parameters": AttributeInfo(type=bool, required=False, default=False),
    "to_device": AttributeInfo(type=Callable, required=False, default=lambda: None),
    "infer_settings": AttributeInfo(type=dict, required=False),
    "print_model": AttributeInfo(type=bool, required=False, default=False),
    "print_summary": AttributeInfo(type=bool, required=False, default=False),
    "input_size": AttributeInfo(type=object, required=False, default=(1, 3, 224, 224)),
}


def parse_from_module(py_module_configs):
    dp = DistributeParams(py_module_configs)

    attrs = AttrDict()
    for aname, ainfo in config_attribute_mapping.items():
        attrs[aname] = safe_get_attr(
            py_module_configs,
            aname, ainfo.type, ainfo.default, ainfo.required)
    attrs.distribute_params = dp
    return attrs


if __name__ == '__main__':
    
    print(parse_from_module(load_config_module('/home/xielei/multi_view_transformer/configs/multi_cam.py')))