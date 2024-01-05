import torch
import torch.nn as nn

from ..encoders.sft.size_dynamics import PipelineDynamics
from ..encoders.sft.sft import SFTBackbone, SFT as SFT_Encoder
from ..encodecos.sft import SFT as SFT_Encodeco
from ..stems.linear import Linear as LinearStem
from ..heads.dense_predictor.linear import Linear as LinearHead
from .utils import RandomMasking, MAELoss

from ..utils import get_nd_sin_posemb
import utils


class MAE_SFT(nn.Module):

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
        mask_ratio,
        mask_size,
        norm_tokens,
    ):
        super().__init__()
        assert n_passes % 2 == 0

        self.in_channels = in_channels
        self.in_size = in_size
        self.patch_size = patch_size
        self.ctx_sizes = ctx_sizes
        self.qry_sizes = qry_sizes
        self.emb_dims = emb_dims
        self.mlp_hidden_ratios = mlp_hidden_ratios
        self.n_heads = n_heads
        self.n_layers_codec = n_layers_codec
        self.attend_to_query = attend_to_query
        self.qry_init_method = qry_init_method
        self.dropout = dropout
        self.use_flash_attn = use_flash_attn
        self.n_passes = n_passes
        self.use_aux = use_aux
        self.norm_tokens = norm_tokens

        self.pdy = PipelineDynamics(in_size, patch_size, ctx_sizes, qry_sizes)
        self.n_layers = self.pdy.get_bottleneck_position() if len(ctx_sizes) == 1 else len(ctx_sizes)

        self.stem = LinearStem(in_channels, in_size, patch_size, emb_dims[0], add_sin_posemb=False)
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
        head_pre_norm = lambda: nn.LayerNorm(emb_dims[0])
        self.head = LinearHead(emb_dims[0], in_channels, patch_size, head_pre_norm)
        self.mae_loss = MAELoss(patch_size)
        
        psize = utils.ceil_div_it(in_size, patch_size)
        self.masking = RandomMasking(psize, mask_ratio, mask_size=mask_size)
        self.mask_token = nn.Parameter(torch.zeros(emb_dims[0]))

        self.register_buffer(
            "posemb",
            get_nd_sin_posemb(psize, emb_dims[0]),
        )

        self._initialize_weights()
    
    def _initialize_weights(self):
        nn.init.normal_(self.mask_token, mean=0, std=0.02)
    
    def size_dynamics(self):
        return self.pdy.get_sizes_str(0, self.n_layers)
    
    def forward(self, x):
        """shape: [b c N1 ... Nd] -> [1]"""
        tgts = x.clone()
        x, cp = self.stem(x) # [b c N1 ... Nd] -> [b n1 ... nd d]
        x, masks = self.masking.replace_with_mask_tokens(x, self.mask_token)
        x = x + self.posemb
        x = self.backbone(x)
        x = self.head(x, cp) # [b n1 ... nd d] -> [b c N1 ... Nd]

        n_aux = x.shape[0] // tgts.shape[0]
        loss = self.mae_loss(
            x,
            utils.repeat_tensor(tgts, n_aux),
            utils.repeat_tensor(masks, n_aux),
            self.norm_tokens,
            patchify=True,
        )
        x = x[:tgts.shape[0]]

        return loss, "loss", x, masks
    
    def load_encoder(self, encoder):
        msg = encoder.stem.load_state_dict(self.stem.state_dict(), strict=False)
        assert set(msg.missing_keys) == {'posemb'}
        # If n_passes = 1, backward stage is not defined in the encoder, hence strict=False.
        encoder.backbone.load_state_dict(self.backbone.state_dict(), strict=False)
    
    def load_encodeco(self, encodeco):
        msg = encodeco.stem.load_state_dict(self.stem.state_dict(), strict=False)
        assert set(msg.missing_keys) == {'posemb'}
        encodeco.backbone.load_state_dict(self.backbone.state_dict())
    
    def create_encoder(self):
        encoder = SFT_Encoder(
            self.in_channels,
            self.in_size,
            self.patch_size,
            self.ctx_sizes,
            self.qry_sizes,
            self.emb_dims,
            self.mlp_hidden_ratios,
            self.n_heads,
            self.n_layers_codec,
            self.attend_to_query,
            self.qry_init_method,
            self.dropout,
            self.use_flash_attn,
            self.n_passes - 1,
            self.use_aux,
        )
        self.load_encoder(encoder)
        return encoder
    
    def create_encodeco(self):
        encodeco = SFT_Encodeco(
            self.in_channels,
            self.in_size,
            self.patch_size,
            self.ctx_sizes,
            self.qry_sizes,
            self.emb_dims,
            self.mlp_hidden_ratios,
            self.n_heads,
            self.n_layers_codec,
            self.attend_to_query,
            self.qry_init_method,
            self.dropout,
            self.use_flash_attn,
            self.n_passes,
            self.use_aux,
        )
        self.load_encodeco(encodeco)
        return encodeco


def get_args_parser():
    parser = utils.ArgumentParser("mae sft", add_help=False)
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

    parser.add_argument("--mask_ratio", type=float)
    parser.add_argument("--mask_size") # list[int]
    parser.add_argument("--norm_tokens", type=bool)
    return parser


def create_ssl(config_root, config, in_channels, in_size):
    args = utils.get_args(config_root, (get_args_parser(), config))
    model = MAE_SFT(
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
        args.mask_ratio,
        args.mask_size,
        args.norm_tokens,
    )
    model.args = args
    return model