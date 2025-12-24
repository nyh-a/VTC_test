import torch
from torch import nn
from torch import Tensor
from torch.nn import functional as F
from transformers import AutoModel
from typing import Union, List, Tuple, Dict # For type hints
import math
# from d2l import torch as d2l


class MLP(nn.Module):
    """ Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers, dropout=0.0, act="relu"):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))
        self.dropout = dropout
        self.dropouts = nn.ModuleList(nn.Dropout(dropout) for _ in range(self.num_layers - 1))
        if act == "relu":
            self.act_fn = F.relu
        elif act == "gelu":
            self.act_fn = F.gelu

    def forward(self, x):
        if not hasattr(self, 'act_fn'):
            self.act_fn = F.relu
        for i, layer in enumerate(self.layers):
            x = self.act_fn(layer(x)) if i < self.num_layers - 1 else layer(x)
            if hasattr(self, 'dropouts') and i < self.num_layers - 1:
                x = self.dropouts[i](x)
        return x