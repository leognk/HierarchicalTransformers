import torch.nn as nn
from ..utils import pad_missing, get_nd_sin_posemb
import utils


def stem(cls):
    """
    stem shape: [b c N1 ... Nd] -> [b n1 ... nd d]
        where c=in_channels, N=(n p), p=patch_size, d=emb_dim
    """

    class Stem(nn.Module):
        """
        Wrapper around a patchify stem to handle:
            - padding
            - sinusoidal positional embedding
        """

        def __init__(self, in_channels, in_size, patch_size, emb_dim, add_sin_posemb, *args, **kwargs):
            super().__init__()
            self.stem = cls(in_channels, patch_size, emb_dim, *args, **kwargs)
            self.patch_size = patch_size
            if add_sin_posemb:
                psize = utils.ceil_div_it(in_size, patch_size)
                self.register_buffer(
                    "posemb",
                    get_nd_sin_posemb(psize, emb_dim),
                )
            else:
                self.posemb = None
        
        def forward(self, x):
            """shape: [b c (n1 p1) ... (nd pd)] -> [b n1 ... nd d]"""
            x, cp = pad_missing(x, self.patch_size, emb_dim_last=False)
            x = self.stem(x) # [b c (n1 p1) ... (nd pd)] -> [b n1 ... nd d]
            if self.posemb is not None:
                x = x + self.posemb
            return x, cp
    
    return Stem