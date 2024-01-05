import torch.nn as nn
from .head import head
from jsonargparse import Namespace


@head
class Identity(nn.Identity):

    def __init__(self, in_dim, out_dim):
        super().__init__()


def create_head(config_root, config, in_dim, out_dim, pre_norm):
    assert in_dim == out_dim
    model = Identity(in_dim, out_dim, pre_norm=None)
    model.args = Namespace()
    return model