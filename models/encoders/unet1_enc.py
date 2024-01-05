import torch
import torch.nn as nn
from jsonargparse import Namespace


class DoubleConv(nn.Module):
    """(Conv -> BatchNorm -> ReLU) * 2"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
        )

    def forward(self, x):
        return self.layers(x)


class EncoderBlock(nn.Module):
    """Downscaling with maxpool then double conv."""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.layers = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.layers(x)


class UNet1_Encoder(nn.Module):
    """Only for 2D data."""

    def __init__(self, in_channels, in_size):
        super().__init__()
        assert len(in_size) == 2
        assert in_size[0] >= 16 and in_size[1] >= 16
        
        self.stem = DoubleConv(in_channels, 64)
        self.encoder = nn.ModuleList([
            EncoderBlock(64, 128),
            EncoderBlock(128, 256),
            EncoderBlock(256, 512),
            EncoderBlock(512, 1024),
        ])

        self._initialize_weights()
        self.out_dim = 1024
        self.head_pre_norm = None
    
    def _initialize_weights(self):
        self.apply(self._init_weights)
    
    def _init_weights(self, m):
        if hasattr(m, "bias") and m.bias is not None:
            nn.init.zeros_(m.bias)
    
    def forward(self, x):
        """shape: [b c N1 ... Nd] -> [b n1 ... nd d]"""
        x = self.stem(x)
        for block in self.encoder:
            x = block(x)
        x = torch.movedim(x, 1, -1) # channel first -> last
        return x
    
    @property
    def num_lr_scale_groups(self):
        return 1 + len(self.encoder)
    
    def lr_scale_group_id(self, name):
        if name.startswith("stem"):
            return 0
        elif name.startswith("encoder"):
            layer_id = int(name.split('.')[1])
            return 1 + layer_id
        else:
            raise ValueError(f"Invalid parameter name: {name}")


def create_encoder(config_root, config, in_channels, in_size, n_classes):
    model = UNet1_Encoder(in_channels, in_size)
    model.args = Namespace()
    return model