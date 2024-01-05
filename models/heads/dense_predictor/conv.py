import torch.nn as nn
from .head import head
from ...utils import *
from utils import ArgumentParser, get_args


@head
class Conv(nn.Module):

    def __init__(self, in_dim, out_dim, patch_size, head_init_std=None):
        super().__init__()
        n_axis = len(patch_size)
        patch_sizes = decompose_patching(patch_size)
        self.layers = nn.Sequential(
            *[ConvTLNGELU(n_axis, in_dim, in_dim, s, s) for s in patch_sizes],
            get_Conv(n_axis)(in_dim, out_dim, kernel_size=1, stride=1), # last projection to out_dim
        )
        self._initialize_weights(head_init_std)
    
    @staticmethod
    def _get_head_init(head_init_std):
        if head_init_std is not None:
            return lambda x: nn.init.trunc_normal_(x, mean=0, std=head_init_std)
        return nn.init.xavier_uniform_
    
    def _initialize_weights(self, head_init_std):
        self.apply(self._init_weights)
        head_init = self._get_head_init(head_init_std)
        proj = self.layers[-1]
        # Initialize convt like nn.Linear.
        head_init(proj.weight.flatten(1))
        nn.init.zeros_(proj.bias)
    
    def _init_weights(self, m):
        if isinstance(m, nn.LayerNorm):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)
    
    def forward(self, x):
        """shape: [b n1 ... nd d] -> [b c (n1 p1) ... (nd pd)]"""
        x = x.movedim(-1, 1)
        x = self.layers(x)
        return x


def get_args_parser():
    parser = ArgumentParser("conv", add_help=False)
    parser.add_argument("--head_init_std", type=float)
    return parser


def create_head(config_root, config, in_dim, out_dim, patch_size, pre_norm):
    args = get_args(config_root, (get_args_parser(), config))
    model = Conv(in_dim, out_dim, patch_size, pre_norm, args.head_init_std)
    model.args = args
    return model