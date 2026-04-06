import torch
import torch.nn as nn
import numpy as np

def get_weights_as_1d(pytorch_module):
    """
    Extracts all trainable parameters from a PyTorch module and flattens them into a single 1D PyTorch tensor.
    """
    with torch.no_grad():
        parameters = [param.view(-1) for param in pytorch_module.parameters()]
        return torch.cat(parameters)

def set_weights_from_1d(pytorch_module, weights_1d):
    """
    Takes a 1D PyTorch tensor of weights and injects it back into the PyTorch module's parameters.
    """
    with torch.no_grad():
        current_idx = 0
        for param in pytorch_module.parameters():
            param_length = param.numel()
            # Extract the slice of weights for this parameter
            new_weights = weights_1d[current_idx : current_idx + param_length]
            # Reshape it to match the parameter's shape and copy it over
            param.copy_(new_weights.view(param.size()))
            current_idx += param_length
