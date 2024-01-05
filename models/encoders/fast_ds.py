import torch
import torch.nn as nn
from timm.models.layers import trunc_normal_, DropPath
from ..utils import *
from utils import ArgumentParser, get_args


class Scale(nn.Module):
    """
    Scale vector by element multiplications.
    """
    def __init__(self, dim, init_value=1.0, trainable=True):
        super().__init__()
        self.scale = nn.Parameter(init_value * torch.ones(dim), requires_grad=trainable)

    def forward(self, x):
        return x * self.scale


class StarReLU(nn.Module):
    """
    StarReLU: s * relu(x) ** 2 + b
    """
    def __init__(self, scale_value=1.0, bias_value=0.0, scale_learnable=True, bias_learnable=True, inplace=False):
        super().__init__()
        self.inplace = inplace
        self.relu = nn.ReLU(inplace=inplace)
        self.scale = nn.Parameter(scale_value * torch.ones(1), requires_grad=scale_learnable)
        self.bias = nn.Parameter(bias_value * torch.ones(1), requires_grad=bias_learnable)
        
    def forward(self, x):
        return self.scale * self.relu(x) ** 2 + self.bias


class MLP(nn.Module):
    
    def __init__(self, dim, mlp_ratio=4, out_features=None, act_layer=StarReLU, drop=0., bias=False):
        super().__init__()
        in_features = dim
        out_features = out_features or in_features
        hidden_features = int(mlp_ratio * in_features)
        drop_probs = drop if isinstance(drop, (list, tuple)) else (drop, drop)

        self.fc1 = nn.Linear(in_features, hidden_features, bias=bias)
        self.act = act_layer()
        self.drop1 = nn.Dropout(drop_probs[0])
        self.fc2 = nn.Linear(hidden_features, out_features, bias=bias)
        self.drop2 = nn.Dropout(drop_probs[1])

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x


class Block(nn.Module):
    
    def __init__(self, emb_dim, drop=0., drop_path=0., res_scale_init_value=None):
        super().__init__()
        self.norm = nn.LayerNorm(emb_dim, eps=1e-6, bias=False)
        self.mlp = MLP(dim=emb_dim, drop=drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.res_scale = Scale(dim=emb_dim, init_value=res_scale_init_value) \
            if res_scale_init_value else nn.Identity()
    
    def forward(self, x):
        """x: [b d] -> [b d]"""
        x = self.res_scale(x) + self.drop_path(self.mlp(self.norm(x)))
        return x


class FastDS(nn.Module):
    
    def __init__(self, in_channels, in_size, emb_dim, depth, drop_path):
        super().__init__()

        # Stem
        n_axis = len(in_size)
        n_layers, channels, kernel_sizes, strides, paddings = self.decompose_patching_into_convs(in_channels, emb_dim, in_size)
        self.stem = nn.Sequential(
            *[ConvBNReLU(n_axis, channels[i], channels[i + 1], kernel_sizes[i], strides[i], paddings[i]) for i in range(n_layers - 1)],
            ConvBNReLU(n_axis, channels[-2], channels[-1], strides[-1], (1, 1), (0, 0)),
            get_Conv(n_axis)(channels[-1], emb_dim, kernel_size=1, stride=1), # last projection to emb_dim
        )

        dp_rates = [x.item() for x in torch.linspace(0, drop_path, depth)]
        self.blocks = nn.Sequential(*[Block(emb_dim, drop_path=dp_rates[i], res_scale_init_value=1.0) for i in range(depth)])

        self.out_dim = emb_dim
        self.head_pre_norm = lambda: nn.LayerNorm(self.out_dim, eps=1e-6)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
    
    @staticmethod
    def decompose_patching_into_convs(in_channels, emb_dim, patch_size):
        """Decompose patching in patch_size into a succession of small convolutions and return its parameters."""

        # Strides
        strides = decompose_patching(patch_size)
        n_layers = len(strides)

        # Kernel sizes & paddings
        kernel_sizes = [tuple(2 * s - 1 for s in stride) for stride in strides]
        paddings = [tuple((k - 1) // 2 for k in kernel_size) for kernel_size in kernel_sizes]

        # Channels
        expansion_factors = [geometric_mean(stride) for stride in strides]
        # With fi = expansion_factors[i], choose f0 such that out_channels = (in_channels x f0 x f1 x ... x fn-1) equals emb_dim,
        # but with f0 greater than or equal to the original f0 (the latter constraint has higher priority).
        d0 = flat_dim(in_channels, expansion_factors[1:])
        if len(expansion_factors) > 0:
            expansion_factors[0] = max(expansion_factors[0], emb_dim / d0)
        channels = [in_channels]
        for expansion_factor in expansion_factors:
            channels.append(channels[-1] * expansion_factor)
        channels = [round(c) for c in channels]

        return n_layers, channels, kernel_sizes, strides, paddings
    
    def forward(self, x):
        """x: [b c H W] -> [b d]"""
        x = self.stem(x).flatten(1) # [b c H W] -> [b d]
        x = self.blocks(x) # [b d] -> [b d]
        return x


def get_args_parser():
    parser = ArgumentParser("fast_ds", add_help=False)
    parser.add_argument("--emb_dim", type=int)
    parser.add_argument("--depth", type=int)
    parser.add_argument("--drop_path", type=float)
    return parser


def create_encoder(config_root, config, in_channels, in_size, n_classes):
    args = get_args(config_root, (get_args_parser(), config))
    model = FastDS(in_channels, in_size, args.emb_dim, args.depth, args.drop_path)
    model.args = args
    return model