import torch.nn as nn
from .head import head
from jsonargparse import Namespace


@head
class Identity(nn.Identity):

    def __init__(self, in_dim, out_dim, patch_size):
        super().__init__()


def create_head(config_root, config, in_dim, out_dim, patch_size, pre_norm):
    assert in_dim == out_dim
    assert all(p == 1 for p in patch_size)
    model = Identity(in_dim, out_dim, patch_size, pre_norm)
    model.args = Namespace()
    return model