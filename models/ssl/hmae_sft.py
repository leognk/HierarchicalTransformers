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


class HMAE_SFT(nn.Module):

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
        local_mae,
        mae_steps_first_lvl,
        mae_steps_factor,
        mask_ratios,
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
        self.local_mae = local_mae
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
        )
        head_pre_norm = lambda: nn.LayerNorm(emb_dims[0])
        self.head = LinearHead(emb_dims[0], in_channels, patch_size, head_pre_norm)
        self.mae_loss = MAELoss(patch_size)

        self.mae_lvls = []
        mae_steps = mae_steps_first_lvl
        for i in range(self.n_layers):
            self.mae_lvls.extend([i] * int(mae_steps))
            mae_steps *= mae_steps_factor
        self.register_buffer("mae_step", torch.tensor([0], dtype=int))

        lvls_sizes = self.pdy.get_sizes(0, self.n_layers - 1)
        mask_sizes = [[1] * len(in_size)] + qry_sizes[:-1]
        self.masking = nn.ModuleList(RandomMasking(
            lvls_sizes[i],
            mask_ratios[i],
            group_size=ctx_sizes[i] if local_mae else None,
            mask_size=mask_sizes[i],
        ) for i in range(self.n_layers))
        self.mask_tokens = nn.ParameterList([nn.Parameter(torch.zeros(d)) for d in emb_dims[:-1]])

        for i in range(self.n_layers):
            self.register_buffer(f"posemb{i}", get_nd_sin_posemb(lvls_sizes[i], emb_dims[i]))

        self._initialize_weights()
    
    def _initialize_weights(self):
        for mask_token in self.mask_tokens:
            nn.init.normal_(mask_token, mean=0, std=0.02)
    
    def size_dynamics(self):
        return self.pdy.get_sizes_str(0, self.n_layers)
    
    def forward(self, x):
        """shape: [b c N1 ... Nd] -> [1]"""

        lvl = self.mae_lvls[self.mae_step] if self.training else 0

        # Patchify.
        if lvl == 0:
            tgts = x.clone()
            x, cp = self.stem(x) # [b c N1 ... Nd] -> [b n1 ... nd d]
            a = self.backbone.create_activations(x)
        # Patchify & get abstract representation.
        else:
            with torch.no_grad():
                x, cp = self.stem(x) # [b c N1 ... Nd] -> [b n1 ... nd d]
                a = self.backbone.create_activations(x)
                x = self.backbone.update_activations(a, lvl, n_passes=1)
                tgts = x.clone()

        # Mask some tokens.
        x, masks = self.masking[lvl].replace_with_mask_tokens(x, self.mask_tokens[lvl])
        posemb = getattr(self, f"posemb{lvl}")
        x = x + posemb
        
        # Predict masked tokens.
        a.ctx = x
        stop_position = a.position + 1 if self.local_mae else self.n_layers
        x = self.backbone.update_activations(a, stop_position, self.n_passes, self.use_aux)
        if lvl == 0:
            x = self.head(x, cp) # [b n1 ... nd d] -> [b c N1 ... Nd]
        
        # Compute loss.
        n_aux = x.shape[0] // tgts.shape[0]
        loss = self.mae_loss(
            x,
            utils.repeat_tensor(tgts, n_aux),
            utils.repeat_tensor(masks, n_aux),
            self.norm_tokens[lvl],
            patchify=lvl == 0,
        )
        loss_name = "loss" if lvl == 0 else f"loss_{lvl}"
        x = x[:tgts.shape[0]]

        if self.training:
            self.mae_step = (self.mae_step + 1) % len(self.mae_lvls)

        return loss, loss_name, x, masks
    
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
    parser = utils.ArgumentParser("hmae sft", add_help=False)
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

    parser.add_argument("--local_mae", type=bool)
    parser.add_argument("--mae_steps.first_lvl", type=int)
    parser.add_argument("--mae_steps.factor", type=float)
    parser.add_argument("--mask_ratios") # list[float]
    parser.add_argument("--norm_tokens") # list[bool]
    return parser


def create_ssl(config_root, config, in_channels, in_size):
    args = utils.get_args(config_root, (get_args_parser(), config))
    model = HMAE_SFT(
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
        args.local_mae,
        args.mae_steps.first_lvl,
        args.mae_steps.factor,
        args.mask_ratios,
        args.norm_tokens,
    )
    model.args = args
    return model