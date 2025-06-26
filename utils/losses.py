import torch
import torch.nn as nn

def get_loss_function(loss_name,**kwargs):
    if loss_name.lower() == 'mse':
        return nn.MSELoss(**kwargs)
    elif loss_name.lower() == 'mae':
        # Mean Absolute Error Loss
        return nn.L1Loss(**kwargs)
    else:
        raise ValueError('Loss name must be either "mse" or "mae"')
