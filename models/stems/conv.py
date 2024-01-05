import torch.nn as nn
from .stem import stem
from ..utils import *


@stem
class Conv(nn.Module):

    def __init__(self, in_channels, patch_size, emb_dim):
        super().__init__()
        n_axis = len(patch_size)
        n_layers, channels, kernel_sizes, strides, paddings = self.decompose_patching_into_convs(in_channels, emb_dim, patch_size)
        self.layers = nn.Sequential(
            *[ConvBNReLU(n_axis, channels[i], channels[i + 1], kernel_sizes[i], strides[i], paddings[i]) for i in range(n_layers)],
            get_Conv(n_axis)(channels[-1], emb_dim, kernel_size=1, stride=1), # last projection to emb_dim
        )
        self._initialize_weights()
    
    @staticmethod
    def decompose_patching_into_convs(in_channels, emb_dim, patch_size):
        """Decompose patching in patch_size into a succession of small convolutions and return its parameters."""

        # Strides
        strides = decompose_patching(patch_size)
        n_layers = len(strides)

        # Kernel sizes & paddings
        kernel_sizes = [tuple(2 * s - 1 for s in stride) for stride in strides]
        paddings = [tuple((k - 1) // 2 for k in kernel_size) for kernel_size in kernel_sizes]

        # Channels
        expansion_factors = [geometric_mean(stride) for stride in strides]
        # With fi = expansion_factors[i], choose f0 such that out_channels = (in_channels x f0 x f1 x ... x fn-1) equals emb_dim,
        # but with f0 greater than or equal to the original f0 (the latter constraint has higher priority).
        d0 = flat_dim(in_channels, expansion_factors[1:])
        if len(expansion_factors) > 0:
            expansion_factors[0] = max(expansion_factors[0], emb_dim / d0)
        channels = [in_channels]
        for expansion_factor in expansion_factors:
            channels.append(channels[-1] * expansion_factor)
        channels = [round(c) for c in channels]

        return n_layers, channels, kernel_sizes, strides, paddings
    
    def _initialize_weights(self):
        proj = self.layers[-1]
        # Initialize conv like nn.Linear.
        nn.init.xavier_uniform_(proj.weight.flatten(1))
        nn.init.zeros_(proj.bias)
    
    def forward(self, x):
        """shape: [b c (n1 p1) ... (nd pd)] -> [b n1 ... nd d]"""
        x = self.layers(x)
        x = torch.movedim(x, 1, -1) # channel first -> last
        return x