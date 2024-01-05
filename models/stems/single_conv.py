import torch.nn as nn
from .stem import stem
from ..utils import *


class LayerNormGeneral(nn.Module):
    
    def __init__(self, affine_shape=None, normalized_dim=(-1,), scale=True, bias=True, eps=1e-5):
        super().__init__()
        self.normalized_dim = normalized_dim
        self.use_scale = scale
        self.use_bias = bias
        self.weight = nn.Parameter(torch.ones(affine_shape)) if scale else None
        self.bias = nn.Parameter(torch.zeros(affine_shape)) if bias else None
        self.eps = eps

    def forward(self, x):
        c = x - x.mean(self.normalized_dim, keepdim=True)
        s = c.pow(2).mean(self.normalized_dim, keepdim=True)
        x = c / torch.sqrt(s + self.eps)
        if self.use_scale:
            x = x * self.weight
        if self.use_bias:
            x = x + self.bias
        return x


@stem
class SingleConv(nn.Module):

    def __init__(self, in_channels, patch_size, emb_dim):
        super().__init__()
        kernel_size = tuple(2 * s - 1 for s in patch_size)
        padding = tuple((k - 1) // 2 for k in kernel_size)
        self.conv = nn.Conv2d(in_channels, emb_dim, kernel_size=kernel_size, stride=patch_size, padding=padding)
        self.norm = LayerNormGeneral(emb_dim, bias=False, eps=1e-6)
    
    def forward(self, x):
        """shape: [b c (h ph) (w pw)] -> [b h w d]"""
        x = self.conv(x)
        x = torch.movedim(x, 1, -1) # channel first -> last
        x = self.norm(x)
        return x