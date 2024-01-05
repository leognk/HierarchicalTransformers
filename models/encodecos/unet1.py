import torch
import torch.nn as nn
from ..encoders.unet1_enc import *
from jsonargparse import Namespace


class DecoderBlock(nn.Module):
    """Upscaling then double conv."""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.convt = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
        self.convs = DoubleConv(in_channels, out_channels)

    def forward(self, x, xp):
        x = self.convt(x)

        # padding
        dh = xp.shape[2] - x.shape[2]
        dw = xp.shape[3] - x.shape[3]
        x = nn.functional.pad(x, [dw // 2, dw - dw // 2, dh // 2, dh - dh // 2])

        x = torch.cat([x, xp], dim=1) # [b c/2 h w] x [b c/2 h w] -> [b c h w]
        x = self.convs(x)
        return x


class UNet1(nn.Module):
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
        self.decoder = nn.ModuleList([
            DecoderBlock(1024, 512),
            DecoderBlock(512, 256),
            DecoderBlock(256, 128),
            DecoderBlock(128, 64),
        ])

        self._initialize_weights()
        self.out_dim = 64
        self.out_patch_size = (1, 1)
        self.head_pre_norm = None
    
    def _initialize_weights(self):
        self.apply(self._init_weights)
    
    def _init_weights(self, m):
        if hasattr(m, "bias") and m.bias is not None:
            nn.init.zeros_(m.bias)
    
    def forward(self, x):
        """shape: [b c N1 ... Nd] -> [b N1 ... Nd d]"""
        x = self.stem(x)
        x0s = []
        for block in self.encoder:
            x0s.append(x)
            x = block(x)
        for block, x0 in zip(self.decoder, x0s[::-1]):
            x = block(x, x0)
        x = torch.movedim(x, 1, -1) # channel first -> last
        return x, None
    
    @property
    def num_lr_scale_groups(self):
        return 1 + len(self.encoder) + len(self.decoder)
    
    def lr_scale_group_id(self, name):
        if name.startswith("stem"):
            return 0
        elif name.startswith("encoder"):
            layer_id = int(name.split('.')[1])
            return 1 + layer_id
        elif name.startswith("decoder"):
            layer_id = int(name.split('.')[1])
            return 1 + len(self.encoder) + layer_id
        else:
            raise ValueError(f"Invalid parameter name: {name}")


def create_encodeco(config_root, config, in_channels, in_size):
    model = UNet1(in_channels, in_size)
    model.args = Namespace()
    return model