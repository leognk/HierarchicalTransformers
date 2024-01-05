import torch
import torch.nn as nn
from .head import head
from utils import ArgumentParser, get_args


class SquaredReLU(nn.Module):
    
    def __init__(self, inplace=False):
        super().__init__()
        self.relu = nn.ReLU(inplace=inplace)
    
    def forward(self, x):
        return torch.square(self.relu(x))


norm_layers = {None: None, "LayerNorm": nn.LayerNorm}
activations = {"ReLU": nn.ReLU, "SquaredReLU": SquaredReLU}


@head
class MLP(nn.Module):
    """shape: [b d] -> [b c]"""

    def __init__(self, in_dim, out_dim, hidden_ratio, dropout, norm, activation, head_init_std=None):
        super().__init__()
        hidden_dim = int(hidden_ratio * in_dim)
        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.act = activations[activation]()
        self.norm = norm_layers[norm](hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, out_dim)
        self.dropout = nn.Dropout(dropout)
        self._initialize_weights(head_init_std)
    
    def _initialize_weights(self, head_init_std):
        self.apply(self._init_weights)
        if head_init_std is not None:
            head = [m for m in self.modules() if isinstance(m, nn.Linear)][-1]
            nn.init.trunc_normal_(head.weight, mean=0, std=head_init_std)
    
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            nn.init.zeros_(m.bias)
        elif isinstance(m, nn.LayerNorm):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)
    
    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.norm(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x


def get_args_parser():
    parser = ArgumentParser("mlp", add_help=False)
    parser.add_argument("--hidden_ratio", type=int)
    parser.add_argument("--dropout", type=float)
    parser.add_argument("--norm")
    parser.add_argument("--activation", type=str)
    parser.add_argument("--head_init_std", type=float)
    return parser


def create_head(config_root, config, in_dim, out_dim, pre_norm):
    args = get_args(config_root, (get_args_parser(), config))
    model = MLP(
        in_dim,
        out_dim,
        pre_norm,
        args.hidden_ratio,
        args.dropout,
        args.norm,
        args.activation,
        args.head_init_std,
    )
    model.args = args
    return model