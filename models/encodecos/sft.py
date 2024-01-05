import torch.nn as nn
from ..stems.linear import Linear as LinearStem
from ..encoders.sft.size_dynamics import PipelineDynamics
from ..encoders.sft.sft import SFTBackbone
from utils import ArgumentParser, get_args


class SFT(nn.Module):

    def __init__(
        self,
        in_channels,
        in_size,
        patch_size,
        ctx_sizes,
        qry_sizes,
        emb_dims,
        mlp_hidden_ratios,
        n_heads,
        n_layers_codec,
        attend_to_query,
        qry_init_method,
        dropout,
        use_flash_attn,
        n_passes,
        use_aux,
    ):
        super().__init__()
        assert n_passes % 2 == 0

        self.pdy = PipelineDynamics(in_size, patch_size, ctx_sizes, qry_sizes)
        self.n_layers = self.pdy.get_bottleneck_position() if len(ctx_sizes) == 1 else len(ctx_sizes)
        
        self.stem = LinearStem(in_channels, in_size, patch_size, emb_dims[0], add_sin_posemb=True)
        self.backbone = SFTBackbone(
            ctx_sizes,
            qry_sizes,
            emb_dims,
            mlp_hidden_ratios,
            n_heads,
            n_layers_codec,
            attend_to_query,
            qry_init_method,
            dropout,
            use_flash_attn,
            self.n_layers,
            n_passes,
            use_aux,
        )
        self.out_dim = emb_dims[0]
        self.out_patch_size = patch_size
        self.head_pre_norm = lambda: nn.LayerNorm(self.out_dim)
    
    def size_dynamics(self):
        return self.pdy.get_sizes_str(0, self.n_layers)
    
    def forward(self, x):
        x, cp = self.stem(x)
        x = self.backbone(x)
        return x, cp
    
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
    parser = ArgumentParser("sft", add_help=False)
    parser.add_argument("--patch_size") # list[int]

    parser.add_argument("--ctx_sizes") # list[list[int]]
    parser.add_argument("--qry_sizes") # list[list[int]]

    parser.add_argument("--emb_dims") # list[int]
    parser.add_argument("--mlp_hidden_ratios") # list[int]
    parser.add_argument("--n_heads") # list[int]

    parser.add_argument("--n_layers_codec.forward") # list[list[int]]
    parser.add_argument("--n_layers_codec.backward") # list[list[int]]
    parser.add_argument("--attend_to_query.forward") # list[bool]
    parser.add_argument("--attend_to_query.backward") # list[bool]

    parser.add_argument("--qry_init_method", type=str)
    parser.add_argument("--dropout", type=float)
    parser.add_argument("--use_flash_attn", type=bool)

    parser.add_argument("--n_passes", type=int)
    parser.add_argument("--use_aux", type=bool)
    return parser


def create_encodeco(config_root, config, in_channels, in_size):
    args = get_args(config_root, (get_args_parser(), config))
    model = SFT(
        in_channels,
        in_size,
        args.patch_size,
        args.ctx_sizes,
        args.qry_sizes,
        args.emb_dims,
        args.mlp_hidden_ratios,
        args.n_heads,
        [args.n_layers_codec.forward, args.n_layers_codec.backward],
        [args.attend_to_query.forward, args.attend_to_query.backward],
        args.qry_init_method,
        args.dropout,
        args.use_flash_attn,
        args.n_passes,
        args.use_aux,
    )
    model.args = args
    return model