"""
Inspiration was taken from https://medium.com/the-owl/simclr-in-pytorch-5f290cb11dd7
"""

from collections import OrderedDict
import torch.nn as nn


class BatchNorm1dNoBias(nn.BatchNorm1d):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.bias.requires_grad = False


class ProjectionHead(nn.Module):
    def __init__(
            self, 
            in_features,
            hidden_features,
            out_features
        ):
        super().__init__()
        projecton_layers = [
            # From original to hidden features
            ("fc1", nn.Linear(in_features, hidden_features, bias=False)),
            ("bn1", nn.BatchNorm1d(hidden_features)),
            # Non-linearity
            ("relu1", nn.ReLU()),
            # From hidden to output features
            ("fc2", nn.Linear(hidden_features, out_features, bias=False)),
            ("bn2", BatchNorm1dNoBias(out_features))
        ]
        self.projection = nn.Sequential(OrderedDict(projecton_layers))
    
    def forward(self, h_0, h_1):
        return self.projection(h_0), self.projection(h_1)