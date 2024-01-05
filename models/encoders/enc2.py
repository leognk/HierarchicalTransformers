import torch.nn as nn
from ..utils import flat_dim
from utils import ArgumentParser, get_args


class TwoLinearClassifier(nn.Module):

    def __init__(self, in_channels, in_size, n_classes, hidden_dim):
        super().__init__()
        in_features = flat_dim(in_channels, in_size)
        self.layers = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, n_classes),
        )
        self._initialize_weights()
        self.out_dim = n_classes
        self.head_pre_norm = None
    
    def _initialize_weights(self):
        self.apply(self._init_weights)
        head = self.layers[-1]
        nn.init.trunc_normal_(head.weight, mean=0, std=2e-5)
        nn.init.zeros_(head.bias)
    
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            nn.init.zeros_(m.bias)
    
    def forward(self, x):
        """shape: [b c N1 ... Nd] -> [b c']"""
        x = x.flatten(1)
        x = self.layers(x)
        return x
    
    @property
    def num_lr_scale_groups(self):
        return 1
    
    def lr_scale_group_id(self, name):
        return 0


def get_args_parser():
    parser = ArgumentParser("enc2", add_help=False)
    parser.add_argument("--hidden_dim", type=int)
    return parser


def create_encoder(config_root, config, in_channels, in_size, n_classes):
    args = get_args(config_root, (get_args_parser(), config))
    model = TwoLinearClassifier(in_channels, in_size, n_classes, args.hidden_dim)
    model.args = args
    return model