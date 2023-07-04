from typing import Optional
import torch
import torch.nn as nn

from torchvision import ops

import numpy as np

from einops import rearrange, reduce, repeat

from mmcv.runner import BaseModule
from mmcv.cnn import build_activation_layer, build_norm_layer

import math

class ScoreEmbed(BaseModule):
    """embed confidence score to a vector using sine embed and MLP"""
    def __init__(self, d_embed, temperature=32):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(d_embed, d_embed),
            nn.GELU(),
            nn.LayerNorm(d_embed),
        )
        self.temperature = temperature
        self.d_embed = d_embed
        
    def posembed_1d(self, position: torch.Tensor):
        device = position.device
        pe = torch.zeros(*position.shape[:-1], self.d_embed, device=device)
        div_term = torch.exp((torch.arange(0, self.d_embed, 2, dtype=torch.float) *
                            -(math.log(self.temperature) / self.d_embed))).to(device)
        pe[..., 0::2] = torch.sin(position * div_term)
        pe[..., 1::2] = torch.cos(position * div_term)
        return pe
    
    def forward(self, conf_score):
        pe = self.posembed_1d(conf_score)
        return self.proj(pe)