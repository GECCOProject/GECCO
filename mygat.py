from typing import Optional, Tuple, Union

import torch
import torch.nn.functional as F
from torch import Tensor
from torch.nn import Parameter
import torch.nn as nn
from torch.nn import MultiheadAttention
from torch import Tensor
from torch.nn import Parameter, Linear, Dropout, LayerNorm, ReLU

        
class Att(nn.Module):
    def __init__(
        self,
        in_channels: Union[int, Tuple[int, int]],
        out_channels: int,
        heads: int = 4,
        concat: bool = True,
        dropout: float = 0.0,
        fill_value: Union[float, Tensor, str] = 'mean',
        bias: bool = True,
        share_weights: bool = False,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.heads = heads
        self.concat = concat
        self.dropout = dropout
        self.fill_value = fill_value
        self.share_weights = share_weights
        self.bias = bias
        
        self.linear = Linear(in_channels, out_channels, bias=bias)
        self.dropout = Dropout(dropout)
        self.norm = LayerNorm(out_channels)
        
        self.reset_parameters()
        
    def reset_parameters(self):
        nn.init.xavier_uniform_(self.linear.weight)
        if self.bias:
            nn.init.zeros_(self.linear.bias)
            
    def forward(self, x: Tensor) -> Tensor:
        x = self.linear(x)
        x = self.norm(x)
        x = self.dropout(x)
        
        # x = x.permute(1, 0, 2)
        # x, _ = self.att(x, x, x)
        # x = x.permute(1, 0, 2)
        
        return x