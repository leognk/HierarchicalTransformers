import torch
import torch.nn as nn

from ..encoders.sft.size_dynamics import SFTDynamics, PipelineDynamics
from ..encoders.sft.sft import SFTBackbone
from ..encoders.sft_s2s import SFT_S2S as SFT_S2S_Encoder
from ..encodecos.sft_s2s import SFT_S2S as SFT_S2S_Encodeco
from ..stems.linear import Linear as LinearStem
from ..heads.dense_predictor.linear import Linear as LinearHead
from .utils import RandomMasking, MAELoss

from einops import pack, unpack
from ..utils import get_nd_sin_posemb
import utils


class MAE_SFT_S2S(nn.Module):

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
        n_passes,

        dec_ctx_sizes,
        dec_qry_sizes,
        dec_emb_dims,
        dec_mlp_hidden_ratios,
        dec_n_heads,
        dec_n_layers_codec,
        dec_attend_to_query,
        dec_n_passes,
        dec_use_aux,

        qry_init_method,
        dropout,
        use_flash_attn,

        mask_ratio,
        mask_size,
        norm_tokens,
    ):
        super().__init__()
        assert n_passes % 2 == 0
        assert dec_n_passes % 2 == 0
        
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
        self.norm_tokens = norm_tokens
        
        psize = utils.ceil_div_it(in_size, patch_size)
        self.masking = RandomMasking(psize, mask_ratio, mask_size=mask_size)
        self.mask_token = nn.Parameter(torch.zeros(dec_emb_dims[0]))

        in_size_visible = [self.masking.n_visible]
        ctx_sizes_flat = [[utils.product(s)] for s in ctx_sizes]
        qry_sizes_flat = [[utils.product(s)] for s in qry_sizes]

        self.stem = LinearStem(in_channels, in_size, patch_size, emb_dims[0], add_sin_posemb=True)
        self.sdy = SFTDynamics(in_size_visible, ctx_sizes_flat, qry_sizes_flat)
        self.n_layers = self.sdy.get_bottleneck_position() if len(ctx_sizes_flat) == 1 else len(ctx_sizes_flat)
        self.encoder = SFTBackbone(
            ctx_sizes_flat,
            qry_sizes_flat,
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
            output_aux=False,
        )
        self.post_norm = nn.LayerNorm(emb_dims[0])

        self.dec_stem = nn.Linear(emb_dims[0], dec_emb_dims[0])
        self.dec_pdy = PipelineDynamics(in_size, patch_size, dec_ctx_sizes, dec_qry_sizes)
        self.dec_n_layers = self.dec_pdy.get_bottleneck_position() if len(dec_ctx_sizes) == 1 else len(dec_ctx_sizes)
        self.decoder = SFTBackbone(
            dec_ctx_sizes,
            dec_qry_sizes,
            dec_emb_dims,
            dec_mlp_hidden_ratios,
            dec_n_heads,
            dec_n_layers_codec,
            dec_attend_to_query,
            qry_init_method,
            dropout,
            use_flash_attn,
            self.dec_n_layers,
            dec_n_passes,
            dec_use_aux,
        )
        self.dec_post_norm = nn.LayerNorm(dec_emb_dims[0])

        self.head = LinearHead(dec_emb_dims[0], in_channels, patch_size, None)
        self.mae_loss = MAELoss(patch_size)

        self.register_buffer(
            "dec_posemb",
            get_nd_sin_posemb(psize, dec_emb_dims[0]),
        )

        self._initialize_weights()
    
    def _initialize_weights(self):
        nn.init.normal_(self.mask_token, mean=0, std=0.02)
    
    def enc_size_dynamics(self):
        return self.sdy.get_levels_sizes_str(0, self.n_layers)
    
    def dec_size_dynamics(self):
        return self.dec_pdy.get_sizes_str(0, self.dec_n_layers)
    
    def forward(self, x):
        """shape: [b c N1 ... Nd] -> [1]"""
        tgts = x.clone()

        # Patchify.
        x, cp = self.stem(x) # [b c N1 ... Nd] -> [b n1 ... nd d]
        x, ps = pack([x], 'b * d') # -> [b n d]

        # Remove some tokens & encode.
        x, masks = self.masking.remove_tokens(x) # [b n d] -> [b n' d]
        x = self.post_norm(self.encoder(x))

        # Add mask tokens & decode.
        x = self.dec_stem(x) # [b n' d] -> [b n' d']
        x = self.masking.add_mask_tokens(x, self.mask_token) # [b n' d'] -> [b n d']
        [x] = unpack(x, ps, 'b * d') # -> [b n1 ... nd d']
        x = x + self.dec_posemb
        x = self.dec_post_norm(self.decoder(x))

        # Unpatch.
        x = self.head(x, cp) # [b n1 ... nd d'] -> [b c N1 ... Nd]

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
        encoder.stem.load_state_dict(self.stem.state_dict())
        encoder.backbone.load_state_dict(self.encoder.state_dict())
    
    def load_encodeco(self, encodeco):
        encodeco.stem.load_state_dict(self.stem.state_dict())
        encodeco.backbone.load_state_dict(self.encoder.state_dict())
    
    def create_encoder(self):
        encoder = SFT_S2S_Encoder(
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
            use_aux=False,
        )
        self.load_encoder(encoder)
        return encoder
    
    def create_encodeco(self):
        encodeco = SFT_S2S_Encodeco(
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
            use_aux=False,
        )
        self.load_encodeco(encodeco)
        return encodeco


def get_args_parser():
    parser = utils.ArgumentParser("mae sft-s2s", add_help=False)
    parser.add_argument("--patch_size") # list[int]

    # Encoder
    parser.add_argument("--encoder.ctx_sizes") # list[list[int]]
    parser.add_argument("--encoder.qry_sizes") # list[list[int]]

    parser.add_argument("--encoder.emb_dims") # list[int]
    parser.add_argument("--encoder.mlp_hidden_ratios") # list[int]
    parser.add_argument("--encoder.n_heads") # list[int]

    parser.add_argument("--encoder.n_layers_codec.forward") # list[list[int]]
    parser.add_argument("--encoder.n_layers_codec.backward") # list[list[int]]
    parser.add_argument("--encoder.attend_to_query.forward") # list[bool]
    parser.add_argument("--encoder.attend_to_query.backward") # list[bool]

    parser.add_argument("--encoder.n_passes", type=int)

    # Decoder
    parser.add_argument("--decoder.ctx_sizes") # list[list[int]]
    parser.add_argument("--decoder.qry_sizes") # list[list[int]]

    parser.add_argument("--decoder.emb_dims") # list[int]
    parser.add_argument("--decoder.mlp_hidden_ratios") # list[int]
    parser.add_argument("--decoder.n_heads") # list[int]

    parser.add_argument("--decoder.n_layers_codec.forward") # list[list[int]]
    parser.add_argument("--decoder.n_layers_codec.backward") # list[list[int]]
    parser.add_argument("--decoder.attend_to_query.forward") # list[bool]
    parser.add_argument("--decoder.attend_to_query.backward") # list[bool]

    parser.add_argument("--decoder.n_passes", type=int)
    parser.add_argument("--decoder.use_aux", type=bool)

    parser.add_argument("--qry_init_method", type=str)
    parser.add_argument("--dropout", type=float)
    parser.add_argument("--use_flash_attn", type=bool)

    parser.add_argument("--mask_ratio", type=float)
    parser.add_argument("--mask_size") # list[int]
    parser.add_argument("--norm_tokens", type=bool)
    return parser


def create_ssl(config_root, config, in_channels, in_size):
    args = utils.get_args(config_root, (get_args_parser(), config))
    model = MAE_SFT_S2S(
        in_channels,
        in_size,
        args.patch_size,
        
        args.encoder.ctx_sizes,
        args.encoder.qry_sizes,
        args.encoder.emb_dims,
        args.encoder.mlp_hidden_ratios,
        args.encoder.n_heads,
        [args.encoder.n_layers_codec.forward, args.encoder.n_layers_codec.backward],
        [args.encoder.attend_to_query.forward, args.encoder.attend_to_query.backward],
        args.encoder.n_passes,
        
        args.decoder.ctx_sizes,
        args.decoder.qry_sizes,
        args.decoder.emb_dims,
        args.decoder.mlp_hidden_ratios,
        args.decoder.n_heads,
        [args.decoder.n_layers_codec.forward, args.decoder.n_layers_codec.backward],
        [args.decoder.attend_to_query.forward, args.decoder.attend_to_query.backward],
        args.decoder.n_passes,
        args.decoder.use_aux,

        args.qry_init_method,
        args.dropout,
        args.use_flash_attn,

        args.mask_ratio,
        args.mask_size,
        args.norm_tokens,
    )
    model.args = args
    return model