import torch
import torch.nn as nn


def head(cls):

    class Head(nn.Module):
        """
        Wrapper around a head to handle:
            - global average pooling (if necessary)
            - pre-norm
        """
        
        def __init__(self, in_dim, out_dim, pre_norm, *args, **kwargs):
            super().__init__()
            self.pre_norm = pre_norm() if pre_norm else None
            self.head = cls(in_dim, out_dim, *args, **kwargs)
        
        def forward(self, x):
            """shape: [b * d] -> [b c]"""
            # Global average pooling: [b n1 ... nd d] -> [b d]
            if x.dim() > 2:
                x = torch.movedim(x, -1, 1)
                x = torch.mean(torch.flatten(x, start_dim=2), dim=-1)
            if self.pre_norm: x = self.pre_norm(x)
            x = self.head(x) # [b d] -> [b c]
            return x
    
    return Head