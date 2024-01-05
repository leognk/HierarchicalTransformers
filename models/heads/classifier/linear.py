import torch.nn as nn
from .head import head
from utils import ArgumentParser, get_args
from jsonargparse import Namespace


@head
class Linear(nn.Module):

    def __init__(self, in_dim, out_dim, head_init_std=None):
        super().__init__()
        self.fc = nn.Linear(in_dim, out_dim)
        self._initialize_weights(head_init_std)
    
    def _initialize_weights(self, head_init_std):
        if head_init_std is not None:
            nn.init.trunc_normal_(self.fc.weight, mean=0, std=head_init_std)
        else:
            nn.init.xavier_uniform_(self.fc.weight)
        nn.init.zeros_(self.fc.bias)
    
    def forward(self, x):
        """shape: [b d] -> [b c]"""
        x = self.fc(x)
        return x


def get_args_parser():
    parser = ArgumentParser("linear", add_help=False)
    parser.add_argument("--head_init_std", type=float)
    return parser


def create_head(config_root, config, in_dim, out_dim, pre_norm):
    args = get_args(config_root, (get_args_parser(), config))
    model = Linear(in_dim, out_dim, pre_norm, args.head_init_std)
    model.args = args
    return model