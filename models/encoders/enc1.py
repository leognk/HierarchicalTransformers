import torch.nn as nn
from ..utils import flat_dim
from jsonargparse import Namespace


class LinearClassifier(nn.Module):

    def __init__(self, in_channels, in_size, n_classes):
        super().__init__()
        in_features = flat_dim(in_channels, in_size)
        self.fc = nn.Linear(in_features, n_classes)
        self._initialize_weights()
        self.out_dim = n_classes
        self.head_pre_norm = None
    
    def _initialize_weights(self):
        nn.init.trunc_normal_(self.fc.weight, mean=0, std=2e-5)
        nn.init.zeros_(self.fc.bias)
    
    def forward(self, x):
        """shape: [b c N1 ... Nd] -> [b c']"""
        x = x.flatten(1)
        x = self.fc(x)
        return x
    
    @property
    def num_lr_scale_groups(self):
        return 1
    
    def lr_scale_group_id(self, name):
        return 0


def create_encoder(config_root, config, in_channels, in_size, n_classes):
    model = LinearClassifier(in_channels, in_size, n_classes)
    model.args = Namespace()
    return model