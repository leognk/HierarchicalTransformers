import torch.nn as nn
from .head import head
from ...utils import *


@head
class Unpatch(nn.Module):

    def __init__(self, in_dim, out_dim, patch_size):
        super().__init__()
        self.unpatch = RearrangeNd('b [n0] ([p0] c) -> b c [(n0 p0)]', {'p': patch_size})
    
    def forward(self, x):
        """shape: [b n1 ... nd d] -> [b c (n1 p1) ... (nd pd)]"""
        x = self.unpatch(x)
        return x