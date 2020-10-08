from abc import ABC

import torch.nn as nn


class WI_Module(nn.Module, ABC):
    def __init__(self):
        super().__init__()

    def init_weights(self):
        init_weights_functional(self, self.activation)


def init_weights_functional(module, activation='default'):
    if isinstance(activation, nn.ReLU):
        for param in module.parameters():
            if param.dim() > 1:
                nn.init.kaiming_normal_(param, nonlinearity='relu')

    elif activation == 'relu':
        for param in module.parameters():
            if param.dim() > 1:
                nn.init.kaiming_normal_(param, nonlinearity='relu')

    elif isinstance(activation, nn.LeakyReLU):
        for param in module.parameters():
            if param.dim() > 1:
                nn.init.kaiming_normal_(param, nonlinearity='leaky_relu')

    elif activation == 'leaky_relu':
        for param in module.parameters():
            if param.dim() > 1:
                nn.init.kaiming_normal_(param, nonlinearity='leaky_relu')

    elif isinstance(activation, nn.Sigmoid):
        for param in module.parameters():
            if param.dim() > 1:
                nn.init.xavier_normal_(param)

    elif activation == 'sigmoid':
        for param in module.parameters():
            if param.dim() > 1:
                nn.init.xavier_normal_(param)

    elif isinstance(activation, nn.Tanh):
        for param in module.parameters():
            if param.dim() > 1:
                nn.init.xavier_normal_(param)

    elif activation == 'tanh':
        for param in module.parameters():
            if param.dim() > 1:
                nn.init.xavier_normal_(param)

    else:
        for param in module.parameters():
            if param.dim() > 1:
                nn.init.xavier_normal_(param)
