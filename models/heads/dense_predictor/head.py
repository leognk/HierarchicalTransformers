import torch.nn as nn
from ...utils import do_pad


def head(cls):

    class Head(nn.Module):
        """
        Wrapper around a head to handle:
            - pre-norm
            - cropping back
        """
        
        def __init__(self, in_dim, out_dim, patch_size, pre_norm, *args, **kwargs):
            super().__init__()
            self.pre_norm = pre_norm() if pre_norm else None
            self.head = cls(in_dim, out_dim, patch_size, *args, **kwargs)
        
        def forward(self, x, cp=None):
            """shape: [b n1 ... nd d] -> [b c N1 ... Nd]"""
            if self.pre_norm: x = self.pre_norm(x)
            x = self.head(x)
            if cp: x = do_pad(x, cp)
            return x
    
    return Head