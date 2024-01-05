import torch
import torch.nn as nn
from ..stems.linear import Linear as LinearStem
from .transformer import Transformer
from einops import pack
from utils import ArgumentParser, get_args


class ViTBackbone(nn.Module):
    """ViT backbone without the stem and head."""

    def __init__(self, emb_dim, mlp_hidden_ratio, n_heads, dropout, drop_path, use_flash_attn, n_layers):
        super().__init__()
        self.n_layers = n_layers
        if drop_path is None: drop_path = 0
        dpr = torch.linspace(0, drop_path, n_layers).tolist()
        self.transformer = Transformer(
            n_layers=n_layers,
            self_attend=True,
            emb_dim=emb_dim,
            mlp_hidden_dim=int(emb_dim * mlp_hidden_ratio),
            n_heads=n_heads,
            dropout=dropout,
            drop_path=dpr,
            use_flash_attn=use_flash_attn,
        )
        self._initialize_weights()
    
    def _initialize_weights(self):
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            nn.init.zeros_(m.bias)
        elif isinstance(m, nn.LayerNorm):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)
    
    def forward(self, x):
        x = self.transformer(x)
        return x
    
    @property
    def num_lr_scale_groups(self):
        return self.n_layers
    
    def lr_scale_group_id(self, name):
        layer_id = int(name.split('.')[2])
        return layer_id


class ViT(nn.Module):

    def __init__(self, in_channels, in_size, patch_size, emb_dim, mlp_hidden_ratio, n_heads, dropout, drop_path, use_flash_attn, n_layers):
        super().__init__()
        self.stem = LinearStem(in_channels, in_size, patch_size, emb_dim, add_sin_posemb=True)
        self.backbone = ViTBackbone(emb_dim, mlp_hidden_ratio, n_heads, dropout, drop_path, use_flash_attn, n_layers)
        self.out_dim = emb_dim
        self.head_pre_norm = lambda: nn.LayerNorm(self.out_dim)
    
    def forward(self, x):
        x, _ = self.stem(x) # [b c N1 ... Nd] -> [b n1 ... nd d]
        x, _ = pack([x], 'b * d') # -> [b n d]
        x = self.backbone(x) # -> [b n d]
        return x
    
    @property
    def num_lr_scale_groups(self):
        return 1 + self.backbone.num_lr_scale_groups
    
    def lr_scale_group_id(self, name):
        if name.startswith("stem"):
            return 0
        elif name.startswith("backbone"):
            _name = name.split('.', 1)[1]
            return 1 + self.backbone.lr_scale_group_id(_name)
        else:
            raise ValueError(f"Invalid parameter name: {name}")


def get_args_parser():
    parser = ArgumentParser("vit", add_help=False)
    parser.add_argument("--patch_size") # list[int]
    parser.add_argument("--emb_dim", type=int)
    parser.add_argument("--mlp_hidden_ratio", type=float)
    parser.add_argument("--n_heads", type=int)
    parser.add_argument("--dropout", type=float)
    parser.add_argument("--drop_path", type=float)
    parser.add_argument("--use_flash_attn", type=bool)
    parser.add_argument("--n_layers", type=int)
    return parser


def create_encoder(config_root, config, in_channels, in_size, n_classes):
    args = get_args(config_root, (get_args_parser(), config))
    model = ViT(
        in_channels,
        in_size,
        args.patch_size,
        args.emb_dim,
        args.mlp_hidden_ratio,
        args.n_heads,
        args.dropout,
        args.drop_path,
        args.use_flash_attn,
        args.n_layers,
    )
    model.args = args
    return model