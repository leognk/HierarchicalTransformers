import torch.nn as nn
from .head import head
from ...utils import *
from utils import ArgumentParser, get_args
from jsonargparse import Namespace


@head
class Linear(nn.Module):

    def __init__(self, in_dim, out_dim, patch_size, head_init_std=None):
        super().__init__()
        flat_out_dim = flat_dim(out_dim, patch_size)
        self.fc = nn.Linear(in_dim, flat_out_dim)
        self.unpatch = RearrangeNd('b [n0] ([p0] c) -> b c [(n0 p0)]', {'p': patch_size})
        self._initialize_weights(head_init_std)
    
    def _initialize_weights(self, head_init_std):
        if head_init_std is not None:
            nn.init.trunc_normal_(self.fc.weight, mean=0, std=head_init_std)
        else:
            nn.init.xavier_uniform_(self.fc.weight)
        nn.init.zeros_(self.fc.bias)
    
    def forward(self, x):
        """shape: [b n1 ... nd d] -> [b c (n1 p1) ... (nd pd)]"""
        x = self.fc(x)
        x = self.unpatch(x)
        return x


def get_args_parser():
    parser = ArgumentParser("linear", add_help=False)
    parser.add_argument("--head_init_std", type=float)
    return parser


def create_head(config_root, config, in_dim, out_dim, patch_size, pre_norm):
    args = get_args(config_root, (get_args_parser(), config))
    model = Linear(in_dim, out_dim, patch_size, pre_norm, args.head_init_std)
    model.args = Namespace()
    return model