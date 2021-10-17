import sys
import torch
import math

import numpy as np
import copy
import torch.nn as nn
from torch import Tensor

from .pytorch_example_imagenet_main import *

# https://github.com/Lyken17/pytorch-OpCounter
import thop

INPUT_IMAGE_HEIGHT = 224
INPUT_IMAGE_WIDTH = 224

def calculate_macs_params(model, input, turn_on_warnings, verbose=True):
    # MACs and Parameters data
    macs, params = thop.profile(model, inputs=(input, ), verbose=turn_on_warnings)
    format_macs, format_params = thop.clever_format([macs, params], "%.3f")
    if verbose:
        print("MACs:", format_macs, "Params:", format_params)
    return macs, params

def update_feature_map_size(name, module, current_feature_map_size):
    if isinstance(module, nn.MaxPool2d):
        return (current_feature_map_size[0] / module.stride, current_feature_map_size[1] / module.stride)
    elif isinstance(module, nn.Conv2d) and "downsample" not in name:    
        return (current_feature_map_size[0] / module.stride[0], current_feature_map_size[1] / module.stride[1])
    else:
        return current_feature_map_size