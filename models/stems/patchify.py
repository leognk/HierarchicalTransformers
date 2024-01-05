import torch.nn as nn
from .stem import stem
from ..utils import *


@stem
class Patchify(nn.Module):

    def __init__(self, in_channels, patch_size, emb_dim):
        super().__init__()
        self.patchify = RearrangeNd('b c [(n0 p0)] -> b [n0] ([p0] c)', {'p': patch_size})
    
    def forward(self, x):
        """shape: [b c (n1 p1) ... (nd pd)] -> [b n1 ... nd d]"""
        x = self.patchify(x)
        return x