import torch.nn as nn
from .stem import stem
from ..utils import *


@stem
class Linear(nn.Module):

    def __init__(self, in_channels, patch_size, emb_dim):
        super().__init__()
        n_axis = len(patch_size)
        self.conv = get_Conv(n_axis)
        if self.conv is not None:
            self.proj = nn.Sequential(
                self.conv(in_channels, emb_dim, kernel_size=patch_size, stride=patch_size),
                MoveDim(1, -1),
            )
        else:
            in_dim = flat_dim(in_channels, patch_size)
            self.proj = nn.Sequential(
                RearrangeNd('b c [(n0 p0)] -> b [n0] ([p0] c)', {'p': patch_size}),
                nn.Linear(in_dim, emb_dim),
            )
        self._initialize_weights()
    
    def _initialize_weights(self):
        if self.conv is not None:
            m = self.proj[0]
            # Initialize conv like nn.Linear.
            nn.init.xavier_uniform_(m.weight.flatten(1))
            nn.init.zeros_(m.bias)
        else:
            m = self.proj[1]
            nn.init.xavier_uniform_(m.weight)
            nn.init.zeros_(m.bias)
    
    def forward(self, x):
        """shape: [b c (n1 p1) ... (nd pd)] -> [b n1 ... nd d]"""
        x = self.proj(x)
        return x